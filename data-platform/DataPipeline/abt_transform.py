import os
import json
import io
import numpy as np
import pandas as pd
from airflow.providers.postgres.hooks.postgres import PostgresHook


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features derivadas da aplicação (razões de comprometimento de renda).

    A renda já foi sanitizada/winsorizada em application_clean (sem zeros/nulos),
    mas mantemos o guarda contra divisão por zero por robustez.
    """
    df_features = df.copy()

    if "amt_income_total" in df_features.columns:
        income = df_features["amt_income_total"].replace(0, np.nan)

        if "amt_credit" in df_features.columns:
            df_features["fe_credit_income_percent"] = df_features["amt_credit"] / income

        if "amt_annuity" in df_features.columns:
            df_features["fe_annuity_income_percent"] = df_features["amt_annuity"] / income

    return df_features


def aggregate_previous_application(conn, config: dict):
    """Agrega previous_application_clean por cliente (Parte 5 do exp_analysis).

    Das features testadas na EDA, `prev_refused_rate` foi a de maior sinal.
    Retorna também a média global da taxa (imputação de quem não tem histórico).
    """
    tbl = config["database"]["output_prev_table"]
    prev = pd.read_sql(
        f'SELECT sk_id_curr, sk_id_prev, name_contract_status FROM "{tbl}"', conn
    )

    agg = prev.groupby("sk_id_curr").agg(
        prev_contract_count=("sk_id_prev", "count"),
        prev_refused_count=("name_contract_status", lambda x: (x == "Refused").sum()),
    )
    agg["prev_refused_rate"] = agg["prev_refused_count"] / agg["prev_contract_count"]
    media_refused = agg["prev_refused_rate"].mean()

    return agg[["prev_refused_rate"]].reset_index(), media_refused


def aggregate_bureau(conn, config: dict) -> pd.DataFrame:
    """Agrega bureau_clean por cliente (Parte 6 do exp_analysis).

    Mantém as features de maior sinal: recência/atividade do crédito externo,
    razão dívida/crédito e contagem de atrasos. `bureau_credit_count` é retornada
    apenas para derivar a flag has_bureau no merge (descartada depois).
    """
    tbl = config["database"]["output_bureau_table"]
    cols = ("sk_id_curr, sk_id_bureau, credit_active, amt_credit_sum, "
            "amt_credit_sum_debt, amt_credit_sum_overdue, credit_day_overdue, days_credit")
    b = pd.read_sql(f'SELECT {cols} FROM "{tbl}"', conn)

    agg = b.groupby("sk_id_curr").agg(
        bureau_credit_count=("sk_id_bureau", "count"),
        bureau_active_count=("credit_active", lambda x: (x == "Active").sum()),
        bureau_closed_count=("credit_active", lambda x: (x == "Closed").sum()),
        bureau_total_credit=("amt_credit_sum", "sum"),
        bureau_total_debt=("amt_credit_sum_debt", "sum"),
        bureau_overdue_count=("credit_day_overdue", lambda x: (x > 0).sum()),
        bureau_avg_days_credit=("days_credit", "mean"),
        bureau_last_days_credit=("days_credit", "max"),
    ).reset_index()

    agg["bureau_active_rate"] = agg["bureau_active_count"] / agg["bureau_credit_count"]
    agg["bureau_closed_rate"] = agg["bureau_closed_count"] / agg["bureau_credit_count"]
    ratio = agg["bureau_total_debt"] / agg["bureau_total_credit"].replace(0, np.nan)
    agg["bureau_debt_credit_ratio"] = ratio.fillna(0).clip(lower=-1, upper=1)

    return agg[[
        "sk_id_curr", "bureau_credit_count",
        "bureau_avg_days_credit", "bureau_last_days_credit",
        "bureau_active_rate", "bureau_active_count", "bureau_closed_rate",
        "bureau_debt_credit_ratio", "bureau_overdue_count",
    ]]


def create_abt_table_schema(cursor, sample_df: pd.DataFrame, target_table: str):
    """Cria a estrutura da tabela ABT dinamicamente baseada nas colunas do DataFrame."""
    colunas = []
    for col, dtype in zip(sample_df.columns, sample_df.dtypes):
        t = str(dtype).lower()
        if "int" in t:
            pg_type = "BIGINT"
        elif "float" in t:
            pg_type = "DOUBLE PRECISION"
        elif "bool" in t:
            pg_type = "BOOLEAN"
        else:
            pg_type = "TEXT"
        colunas.append(f'"{col}" {pg_type}')

    cursor.execute(f'DROP TABLE IF EXISTS "{target_table}" CASCADE;')
    cursor.execute(f'CREATE TABLE "{target_table}" ({", ".join(colunas)});')


def run_abt_generation(conn_id: str):
    """Monta a ABT unindo application_clean + previous_application + bureau (Parte 8).

    Cada fonte histórica é agregada para 1 linha por cliente e juntada por
    `sk_id_curr` via left join a partir do application_clean, garantindo 1 linha
    por cliente. Clientes sem histórico recebem imputação + flag de presença.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_config(os.path.join(base_dir, "config_pipeline.json"))

    pg_hook = PostgresHook(postgres_conn_id=conn_id)
    conn = pg_hook.get_conn()
    cursor = conn.cursor()

    input_table = config["database"]["output_table"]   # application_clean
    output_table = config["database"]["abt_table"]
    chunk_size = config["cleaning_parameters"]["chunk_size"]

    print("Agregando previous_application e bureau por cliente...")
    prev_agg, media_refused = aggregate_previous_application(conn, config)
    bureau_agg = aggregate_bureau(conn, config)

    bureau_feature_cols = [
        "bureau_avg_days_credit", "bureau_last_days_credit",
        "bureau_active_rate", "bureau_active_count", "bureau_closed_rate",
        "bureau_debt_credit_ratio", "bureau_overdue_count",
    ]

    offset = 0
    is_first_chunk = True
    print(f"Construindo a ABT '{output_table}' em lotes...")

    while True:
        query = f'SELECT * FROM "{input_table}" LIMIT {chunk_size} OFFSET {offset};'
        chunk_df = pd.read_sql(query, conn)

        if chunk_df.empty:
            break

        print(f"Processando lote (Offset: {offset}, Linhas: {len(chunk_df)}) para a ABT...")
        abt = build_features(chunk_df)

        # --- previous_application: prev_refused_rate + flag de presença ---
        abt = abt.merge(prev_agg, on="sk_id_curr", how="left")
        abt["has_prev_app"] = abt["prev_refused_rate"].notna().astype(int)
        abt["prev_refused_rate"] = abt["prev_refused_rate"].fillna(media_refused)

        # --- bureau: features + flag de presença (NaN -> 0 para quem não tem histórico) ---
        abt = abt.merge(bureau_agg, on="sk_id_curr", how="left")
        abt["has_bureau"] = abt["bureau_credit_count"].notna().astype(int)
        abt = abt.drop(columns=["bureau_credit_count"])
        abt[bureau_feature_cols] = abt[bureau_feature_cols].fillna(0)

        if is_first_chunk:
            create_abt_table_schema(cursor, abt, output_table)
            conn.commit()
            is_first_chunk = False

        output = io.StringIO()
        abt.to_csv(output, sep="\t", header=False, index=False)
        output.seek(0)
        cursor.copy_expert(
            f'COPY "{output_table}" FROM STDIN WITH CSV DELIMITER \'\t\' NULL \'\'', output
        )
        conn.commit()
        offset += chunk_size

    cursor.close()
    conn.close()
    print(f"--- ABT Enriquecida Construída com Sucesso! Tabela: '{output_table}' ---")


if __name__ == "__main__":
    run_abt_generation("postgres_data_db")
