import os
import json
import io
import numpy as np
import pandas as pd
from airflow.providers.postgres.hooks.postgres import PostgresHook


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def iter_chunks_keyset(conn, source_table: str, key_col: str, chunk_size: int):
    """Percorre a tabela em lotes por paginação de chave (keyset): 1 passada, determinística.

    Em vez de LIMIT/OFFSET (que re-varre as linhas puladas e, sem ORDER BY estável,
    pode duplicar/pular registros), usamos WHERE key > último_visto ORDER BY key.
    Com o índice na chave (criado na ingestão), cada lote "salta" direto para o
    próximo trecho — O(n) total. Exige que `key_col` seja único e ordenável.
    """
    last_id = None
    while True:
        if last_id is None:
            q = f'SELECT * FROM "{source_table}" ORDER BY "{key_col}" LIMIT {chunk_size};'
        else:
            q = (f'SELECT * FROM "{source_table}" WHERE "{key_col}" > {last_id} '
                 f'ORDER BY "{key_col}" LIMIT {chunk_size};')
        chunk_df = pd.read_sql(q, conn)
        if chunk_df.empty:
            break
        yield chunk_df
        last_id = int(chunk_df[key_col].iloc[-1])


def create_table_from_dataframe(cursor, sample_df: pd.DataFrame, target_table: str):
    """Cria a tabela de destino dinamicamente a partir do schema do DataFrame limpo."""
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


def copy_dataframe_to_table(cursor, df: pd.DataFrame, target_table: str):
    """Carga rápida do DataFrame na tabela via COPY (STDIN)."""
    output = io.StringIO()
    df.to_csv(output, sep="\t", header=False, index=False)
    output.seek(0)
    cursor.copy_expert(
        f'COPY "{target_table}" FROM STDIN WITH CSV DELIMITER \'\t\' NULL \'\'', output
    )


# ---------------------------------------------------------------------------
# application_train  (equivalente às Partes 3 e 4 do exp_analysis.ipynb)
# ---------------------------------------------------------------------------
def compute_app_stats(conn, config: dict) -> dict:
    """Estatísticas globais de imputação/winsorização (medianas, p99, cardinalidade).

    Precisam ser calculadas sobre a base completa — nunca por lote, senão cada
    chunk usaria uma mediana diferente. Lemos apenas as colunas necessárias.
    """
    params = config["cleaning_parameters"]
    minf = params["cardinality_min_freq"]
    input_table = config["database"]["input_table"]

    cols = [
        "ext_source_1", "ext_source_2", "ext_source_3",
        "days_last_phone_change", "cnt_fam_members", "amt_annuity",
        "amt_income_total", "own_car_age", "flag_own_car",
        "organization_type", "name_income_type",
    ]
    df = pd.read_sql(f'SELECT {", ".join(cols)} FROM "{input_table}"', conn)

    def num(col):
        return pd.to_numeric(df[col], errors="coerce")

    es1, es2, es3 = num("ext_source_1"), num("ext_source_2"), num("ext_source_3")
    ext_mean = pd.concat([es1, es2, es3], axis=1).mean(axis=1)

    # renda: zero é tratado como ausente antes de imputar / winsorizar
    income = num("amt_income_total").replace(0, np.nan)
    median_income = income.median()
    income_filled = income.fillna(median_income)

    has_car = df["flag_own_car"].astype(str).str.strip() == "Y"
    own_car_age = num("own_car_age")

    org_freq = df["organization_type"].value_counts()
    inc_freq = df["name_income_type"].value_counts()

    stats = {
        "median_ext_source_1": es1.median(),
        "median_ext_source_2": es2.median(),
        "median_ext_source_3": es3.median(),
        "median_ext_source_mean": ext_mean.median(),
        "median_days_last_phone_change": num("days_last_phone_change").median(),
        "median_cnt_fam_members": num("cnt_fam_members").median(),
        "median_amt_annuity": num("amt_annuity").median(),
        "median_amt_income_total": median_income,
        "p99_amt_income_total": income_filled.quantile(params["income_winsor_quantile"]),
        "median_own_car_age": own_car_age[has_car].median(),
        "org_valid": org_freq[org_freq >= minf].index.tolist(),
        "inc_valid": inc_freq[inc_freq >= minf].index.tolist(),
    }
    return stats


def sanitize_data(df: pd.DataFrame, config: dict, stats: dict) -> pd.DataFrame:
    """Seleção + limpeza do application_train alinhada ao exp_analysis (Partes 3 e 4).

    Decisões incorporadas da análise exploratória:
      * scores externos 1/2/3 imputados por mediana + média combinada (ext_source_mean);
      * redundâncias removidas (region_rating_client, def_30_cnt_social_circle, amt_goods_price);
      * flag has_car separando 'sem carro' de 'carro novo (idade 0)';
      * renda: zero tratado como ausente, imputado por mediana e winsorizado no p99;
      * categóricas: nulos -> 'Unknown' e categorias raras (<min_freq) -> 'Other_low_freq';
      * features derivadas: age, years_employed e flag da anomalia days_employed (365243).
    """
    params = config["cleaning_parameters"]
    anom = params["days_employed_anomaly"]
    dpy = params["days_per_year"]

    def num(col):
        return pd.to_numeric(df[col], errors="coerce")

    c = pd.DataFrame()
    c["sk_id_curr"] = num("sk_id_curr").astype("Int64")
    c["target"] = num("target").astype("Int64")

    # --- Scores externos: incluir 1/2/3 (imputados) + média combinada (preditor mais forte) ---
    es1, es2, es3 = num("ext_source_1"), num("ext_source_2"), num("ext_source_3")
    ext_mean = pd.concat([es1, es2, es3], axis=1).mean(axis=1)
    c["ext_source_1"] = es1.fillna(stats["median_ext_source_1"])
    c["ext_source_2"] = es2.fillna(stats["median_ext_source_2"])
    c["ext_source_3"] = es3.fillna(stats["median_ext_source_3"])
    c["ext_source_mean"] = ext_mean.fillna(stats["median_ext_source_mean"])

    # --- region_rating_client removido (redundante com _w_city, corr 0.95) ---
    c["region_rating_client_w_city"] = num("region_rating_client_w_city").astype("Int64")

    c["days_last_phone_change"] = num("days_last_phone_change").fillna(stats["median_days_last_phone_change"])
    c["days_id_publish"] = num("days_id_publish")
    c["days_registration"] = num("days_registration")

    for col in ["reg_city_not_work_city", "reg_city_not_live_city", "live_city_not_work_city"]:
        c[col] = num(col).fillna(0).astype(int)

    # --- own_car_age: flag has_car separa 'não tem carro' (66% nulo) de 'carro novo (idade 0)' ---
    own_car_age = num("own_car_age")
    c["has_car"] = (df["flag_own_car"].astype(str).str.strip() == "Y").astype(int)
    c["own_car_age"] = own_car_age.where(c["has_car"] == 1, 0).fillna(stats["median_own_car_age"])

    # --- def_30_cnt_social_circle removido (redundante com def_60, corr 0.86) ---
    c["def_60_cnt_social_circle"] = num("def_60_cnt_social_circle").fillna(0)
    c["amt_req_credit_bureau_year"] = num("amt_req_credit_bureau_year").fillna(0)
    c["cnt_children"] = num("cnt_children").fillna(0).astype(int)
    c["cnt_fam_members"] = num("cnt_fam_members").fillna(stats["median_cnt_fam_members"])

    # --- monetárias: amt_goods_price removido (redundante com amt_credit, corr 0.99) ---
    income = num("amt_income_total").replace(0, np.nan).fillna(stats["median_amt_income_total"])
    c["amt_income_total"] = income.clip(upper=stats["p99_amt_income_total"])
    c["amt_credit"] = num("amt_credit")
    c["amt_annuity"] = num("amt_annuity").fillna(stats["median_amt_annuity"])

    # --- qualitativas: imputação 'Unknown' + redução de cardinalidade ---
    c["occupation_type"] = df["occupation_type"].fillna("Unknown").astype(str).str.strip()
    c["organization_type"] = (df["organization_type"]
                              .where(df["organization_type"].isin(stats["org_valid"]), "Other_low_freq")
                              .fillna("Unknown").astype(str).str.strip())
    c["name_income_type"] = (df["name_income_type"]
                             .where(df["name_income_type"].isin(stats["inc_valid"]), "Other_low_freq")
                             .fillna("Unknown").astype(str).str.strip())
    c["name_education_type"] = df["name_education_type"].fillna("Unknown").astype(str).str.strip()
    c["code_gender"] = df["code_gender"].replace("XNA", "Unknown").fillna("Unknown").astype(str).str.strip()

    # --- features de demografia derivadas (idade, tempo de emprego e flag da anomalia) ---
    days_emp_raw = num("days_employed")
    c["age"] = np.abs(num("days_birth")) / dpy
    c["years_employed"] = np.abs(days_emp_raw.replace(anom, np.nan)).fillna(0) / dpy
    c["days_employed_anom"] = (days_emp_raw == anom).astype(int)

    return c


def run_sanitization(conn_id: str):
    """Função mestre chamada pela Task do Airflow: sanitiza o application_train."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_config(os.path.join(base_dir, "config_pipeline.json"))

    pg_hook = PostgresHook(postgres_conn_id=conn_id)
    conn = pg_hook.get_conn()
    cursor = conn.cursor()

    input_table = config["database"]["input_table"]
    output_table = config["database"]["output_table"]
    chunk_size = config["cleaning_parameters"]["chunk_size"]
    key_col = config["indexes"][input_table]

    print("Calculando estatísticas globais (medianas, p99, cardinalidade)...")
    stats = compute_app_stats(conn, config)

    is_first_chunk = True
    print(f"Preparando '{output_table}' e processando em lotes (keyset por {key_col})...")

    for chunk_df in iter_chunks_keyset(conn, input_table, key_col, chunk_size):
        print(f"Processando lote de {len(chunk_df)} linhas...")
        cleaned_df = sanitize_data(chunk_df, config, stats)

        if is_first_chunk:
            create_table_from_dataframe(cursor, cleaned_df, output_table)
            conn.commit()
            is_first_chunk = False

        copy_dataframe_to_table(cursor, cleaned_df, output_table)
        conn.commit()

    cursor.close()
    conn.close()
    print(f"--- Sanitização do application concluída! Tabela: '{output_table}' ---")


# ---------------------------------------------------------------------------
# previous_application  (usada na agregação da ABT)
# ---------------------------------------------------------------------------
def sanitize_prev_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()

    # 1. Padroniza colunas de texto cruciais
    if "name_contract_status" in df_clean.columns:
        df_clean["name_contract_status"] = df_clean["name_contract_status"].astype(str).str.strip().str.title()

    # 2. Trata valores nulos ou negativos no valor pedido (amt_application)
    if "amt_application" in df_clean.columns:
        df_clean["amt_application"] = df_clean["amt_application"].fillna(0)
        df_clean["amt_application"] = np.where(df_clean["amt_application"] < 0, 0, df_clean["amt_application"])

    return df_clean


def run_prev_sanitization(conn_id: str):
    """Função mestre para limpar a tabela previous_application em lotes (keyset por sk_id_prev)."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_config(os.path.join(base_dir, "config_pipeline.json"))

    pg_hook = PostgresHook(postgres_conn_id=conn_id)
    conn = pg_hook.get_conn()
    cursor = conn.cursor()

    input_table = config["database"]["input_prev_table"]
    output_table = config["database"]["output_prev_table"]
    chunk_size = config["cleaning_parameters"]["chunk_size"]
    key_col = config["indexes"][input_table]

    print(f"Preparando tabela de destino histórica '{output_table}'...")
    cursor.execute(f'DROP TABLE IF EXISTS "{output_table}" CASCADE;')
    cursor.execute(f'CREATE TABLE "{output_table}" (LIKE "{input_table}" INCLUDING ALL);')
    conn.commit()

    print("Iniciando sanitização do histórico em lotes (keyset)...")
    for chunk_df in iter_chunks_keyset(conn, input_table, key_col, chunk_size):
        cleaned_df = sanitize_prev_data(chunk_df)
        copy_dataframe_to_table(cursor, cleaned_df, output_table)
        conn.commit()

    cursor.close()
    conn.close()
    print(f"--- Histórico limpo com sucesso na tabela '{output_table}' ---")


# ---------------------------------------------------------------------------
# bureau  (equivalente à sanitização da Parte 6 do exp_analysis.ipynb)
# ---------------------------------------------------------------------------
def sanitize_bureau_data(df: pd.DataFrame) -> pd.DataFrame:
    """Tipagem/imputação das colunas do bureau usadas nas agregações da ABT."""
    df_clean = df.copy()

    for col in ["credit_active", "credit_type"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()

    # valores monetários/contagens ausentes -> 0 (aceitável para 'sem valor em atraso/dívida')
    for col in ["amt_credit_sum", "amt_credit_sum_debt", "amt_credit_sum_overdue",
                "credit_day_overdue", "cnt_credit_prolong"]:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0)

    for col in ["days_credit", "days_credit_update"]:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    return df_clean


def run_bureau_sanitization(conn_id: str):
    """Função mestre para limpar a tabela bureau em lotes (keyset por sk_id_bureau)."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_config(os.path.join(base_dir, "config_pipeline.json"))

    pg_hook = PostgresHook(postgres_conn_id=conn_id)
    conn = pg_hook.get_conn()
    cursor = conn.cursor()

    input_table = config["database"]["input_bureau_table"]
    output_table = config["database"]["output_bureau_table"]
    chunk_size = config["cleaning_parameters"]["chunk_size"]
    key_col = config["indexes"][input_table]

    print(f"Preparando tabela de destino do bureau '{output_table}'...")
    cursor.execute(f'DROP TABLE IF EXISTS "{output_table}" CASCADE;')
    cursor.execute(f'CREATE TABLE "{output_table}" (LIKE "{input_table}" INCLUDING ALL);')
    conn.commit()

    print("Iniciando sanitização do bureau em lotes (keyset)...")
    for chunk_df in iter_chunks_keyset(conn, input_table, key_col, chunk_size):
        cleaned_df = sanitize_bureau_data(chunk_df)
        copy_dataframe_to_table(cursor, cleaned_df, output_table)
        conn.commit()

    cursor.close()
    conn.close()
    print(f"--- Bureau limpo com sucesso na tabela '{output_table}' ---")


# ---------------------------------------------------------------------------
# installments_payments  (Parte 7 do exp_analysis.ipynb)
# ---------------------------------------------------------------------------
def run_installments_sanitization(conn_id: str):
    """Sanitiza installments_payments -> installments_clean.

    A tabela não tem chave única de linha (um cliente tem várias parcelas), então a
    paginação por keyset não se aplica. Como a 'sanitização' aqui é apenas um filtro
    de linhas válidas (vencimento e valor previsto conhecidos), fazemos em SQL puro:
    uma passada no Postgres, sem trazer as ~13,6M linhas para o pandas.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_config(os.path.join(base_dir, "config_pipeline.json"))

    pg_hook = PostgresHook(postgres_conn_id=conn_id)
    conn = pg_hook.get_conn()
    cursor = conn.cursor()

    input_table = config["database"]["input_installments_table"]
    output_table = config["database"]["output_installments_table"]

    print(f"Sanitizando '{input_table}' -> '{output_table}' (SQL, filtro de linhas válidas)...")
    cursor.execute(f'DROP TABLE IF EXISTS "{output_table}" CASCADE;')
    cursor.execute(f"""
        CREATE TABLE "{output_table}" AS
        SELECT
            sk_id_curr, sk_id_prev,
            num_instalment_version, num_instalment_number,
            days_instalment, days_entry_payment,
            amt_instalment, amt_payment
        FROM "{input_table}"
        WHERE sk_id_curr IS NOT NULL
          AND sk_id_prev IS NOT NULL
          AND days_instalment IS NOT NULL
          AND amt_instalment  IS NOT NULL;
    """)
    conn.commit()

    cursor.close()
    conn.close()
    print(f"--- Installments limpo com sucesso na tabela '{output_table}' ---")


if __name__ == "__main__":
    run_sanitization("postgres_data_db")
