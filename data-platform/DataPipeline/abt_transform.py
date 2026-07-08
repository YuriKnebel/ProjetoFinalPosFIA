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


def table_exists(conn, table_name: str) -> bool:
    """Verifica se uma tabela existe no banco (to_regclass retorna NULL se não existir)."""
    cur = conn.cursor()
    cur.execute("SELECT to_regclass(%s)", (table_name,))
    existe = cur.fetchone()[0] is not None
    cur.close()
    return existe


def aggregate_previous_application(conn, config: dict):
    """Agrega previous_application_clean por cliente no Postgres (Parte 5 do exp_analysis).

    Das features testadas na EDA, `prev_refused_rate` foi a de maior sinal. O GROUP BY
    roda no banco e devolve 1 linha por cliente; a média global (imputação de quem não
    tem histórico) é calculada em pandas sobre esse resultado pequeno.
    """
    tbl = config["database"]["output_prev_table"]
    q = f"""
        SELECT
            sk_id_curr,
            SUM(CASE WHEN name_contract_status = 'Refused' THEN 1 ELSE 0 END)::float
                / COUNT(sk_id_prev) AS prev_refused_rate
        FROM "{tbl}"
        GROUP BY sk_id_curr
    """
    agg = pd.read_sql(q, conn)
    media_refused = agg["prev_refused_rate"].mean()
    return agg, media_refused


def aggregate_bureau(conn, config: dict) -> pd.DataFrame:
    """Agrega bureau_clean por cliente no Postgres (Parte 6 do exp_analysis).

    O GROUP BY (recência/atividade, dívida, atrasos) roda no banco; as taxas e o
    ratio dívida/crédito são derivados em pandas sobre o resultado (1 linha/cliente).
    `bureau_credit_count` é retornada só para derivar a flag has_bureau (descartada depois).
    """
    tbl = config["database"]["output_bureau_table"]
    q = f"""
        SELECT
            sk_id_curr,
            COUNT(sk_id_bureau)                                                   AS bureau_credit_count,
            SUM(CASE WHEN credit_active = 'Active' THEN 1 ELSE 0 END)             AS bureau_active_count,
            SUM(CASE WHEN credit_active = 'Closed' THEN 1 ELSE 0 END)             AS bureau_closed_count,
            SUM(COALESCE(amt_credit_sum, 0))                                      AS bureau_total_credit,
            SUM(COALESCE(amt_credit_sum_debt, 0))                                 AS bureau_total_debt,
            SUM(CASE WHEN COALESCE(credit_day_overdue, 0) > 0 THEN 1 ELSE 0 END)  AS bureau_overdue_count,
            AVG(days_credit)                                                      AS bureau_avg_days_credit,
            MAX(days_credit)                                                      AS bureau_last_days_credit
        FROM "{tbl}"
        GROUP BY sk_id_curr
    """
    agg = pd.read_sql(q, conn)

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


def aggregate_installments(conn, config: dict):
    """Agrega installments_clean por cliente no Postgres (Parte 7 do exp_analysis).

    Da análise, `inst_late_payment_rate` foi a feature selecionada (a `inst_underpayment_rate`
    saiu por redundância — corr. 0,81). A tabela já vem sanitizada (linhas válidas filtradas
    em data_sanitization.run_installments_sanitization), então aqui é só o GROUP BY. Se a
    tabela não existir, retorna None (degrada suave: clientes ficam como 'sem histórico').
    """
    tbl = config["database"].get("output_installments_table", "installments_clean")
    if not table_exists(conn, tbl):
        print(f"Aviso: tabela '{tbl}' não encontrada — features de installments ficarão como 'sem histórico'.")
        return None

    q = f"""
        SELECT
            sk_id_curr,
            AVG(CASE WHEN (days_entry_payment - days_instalment) > 0 THEN 1.0 ELSE 0.0 END)
                AS inst_late_payment_rate
        FROM "{tbl}"
        GROUP BY sk_id_curr
    """
    return pd.read_sql(q, conn)


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
    """Monta a ABT unindo application_clean + previous_application + bureau + installments (Parte 8).

    Cada fonte histórica é agregada no Postgres (GROUP BY) para 1 linha por cliente e
    juntada por `sk_id_curr` via left join a partir do application_clean. Clientes sem
    histórico recebem imputação + flag de presença (has_prev_app / has_bureau /
    has_installments_history).
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_config(os.path.join(base_dir, "config_pipeline.json"))

    pg_hook = PostgresHook(postgres_conn_id=conn_id)
    conn = pg_hook.get_conn()
    cursor = conn.cursor()

    input_table = config["database"]["output_table"]   # application_clean
    output_table = config["database"]["abt_table"]
    chunk_size = config["cleaning_parameters"]["chunk_size"]
    # application_clean compartilha a chave de application_train (mesmo sk_id_curr)
    key_col = config["indexes"][config["database"]["input_table"]]

    print("Agregando previous_application, bureau e installments no Postgres...")
    prev_agg, media_refused = aggregate_previous_application(conn, config)
    bureau_agg = aggregate_bureau(conn, config)
    inst_agg = aggregate_installments(conn, config)   # pode ser None se a tabela não existir

    bureau_feature_cols = [
        "bureau_avg_days_credit", "bureau_last_days_credit",
        "bureau_active_rate", "bureau_active_count", "bureau_closed_rate",
        "bureau_debt_credit_ratio", "bureau_overdue_count",
    ]

    last_id = None
    is_first_chunk = True
    print(f"Construindo a ABT '{output_table}' em lotes (keyset por {key_col})...")

    while True:
        # Paginação por keyset (WHERE key > último visto): determinística, não duplica
        # clientes. application_clean é pequena (~307k), rápida mesmo sem índice.
        if last_id is None:
            query = f'SELECT * FROM "{input_table}" ORDER BY "{key_col}" LIMIT {chunk_size};'
        else:
            query = (f'SELECT * FROM "{input_table}" WHERE "{key_col}" > {last_id} '
                     f'ORDER BY "{key_col}" LIMIT {chunk_size};')
        chunk_df = pd.read_sql(query, conn)

        if chunk_df.empty:
            break

        print(f"Processando lote de {len(chunk_df)} linhas para a ABT...")
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

        # --- installments: inst_late_payment_rate + flag de presença ---
        if inst_agg is not None:
            abt = abt.merge(inst_agg, on="sk_id_curr", how="left")
        else:
            abt["inst_late_payment_rate"] = np.nan
        abt["has_installments_history"] = abt["inst_late_payment_rate"].notna().astype(int)
        abt["inst_late_payment_rate"] = abt["inst_late_payment_rate"].fillna(0)

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
        last_id = int(chunk_df[key_col].iloc[-1])

    cursor.close()
    conn.close()
    print(f"--- ABT Enriquecida Construída com Sucesso! Tabela: '{output_table}' ---")


if __name__ == "__main__":
    run_abt_generation("postgres_data_db")