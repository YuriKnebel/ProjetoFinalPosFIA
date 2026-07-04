# -*- coding: utf-8 -*-
"""
data_sanitization_v2.py — Limpeza e padronizacao (Home Credit), no padrao da pipeline.

Segue o mesmo estilo dos scripts do Airflow (PostgresHook + config_pipeline.json +
funcao mestre run_*(conn_id)), para poder ser plugado como task no DAG
`pipeline_orchestration`. Contem a logica VALIDADA do notebook `exp_analysis_v2`:
scores externos combinados (ext_source_1/2/3 + media), flag has_car, winsorizacao
da renda e reducao de cardinalidade.

Grava na tabela `output_table_v2` (application_clean_v2), SEM sobrescrever as tabelas
da pipeline existente (application_clean/application_abt).

Diferente do data_sanitization.py (que processa em lotes), aqui a tabela e carregada
INTEIRA (application_train ~307k linhas) porque a limpeza usa estatisticas GLOBAIS
(mediana, p99, frequencia de categorias) — incompativeis com chunking.
"""
import os
import io
import json

import numpy as np
import pandas as pd
from airflow.providers.postgres.hooks.postgres import PostgresHook


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Helpers de banco (mesmo padrao COPY dos scripts existentes)
# --------------------------------------------------------------------------
def create_table_from_df(cursor, df: pd.DataFrame, table: str) -> None:
    """Cria a tabela destino dinamicamente a partir dos dtypes do DataFrame."""
    colunas = []
    for col, dtype in zip(df.columns, df.dtypes):
        d = str(dtype).lower()
        if "int" in d:
            pg_type = "BIGINT"
        elif "float" in d:
            pg_type = "DOUBLE PRECISION"
        elif "bool" in d:
            pg_type = "BOOLEAN"
        else:
            pg_type = "TEXT"
        colunas.append(f'"{col}" {pg_type}')
    cursor.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE;')
    cursor.execute(f'CREATE TABLE "{table}" ({", ".join(colunas)});')


def copy_df_to_table(cursor, df: pd.DataFrame, table: str) -> None:
    """Carga rapida do DataFrame na tabela via COPY (mesmo padrao dos scripts existentes)."""
    output = io.StringIO()
    df.to_csv(output, sep="\t", header=False, index=False)
    output.seek(0)
    cursor.copy_expert(f'COPY "{table}" FROM STDIN WITH CSV DELIMITER \'\t\' NULL \'\'', output)


# --------------------------------------------------------------------------
# Logica de sanitizacao (pura, validada no notebook)
# --------------------------------------------------------------------------
def _reduz_cardinalidade(serie: pd.Series, min_freq: int) -> pd.Series:
    """Categorias com frequencia < min_freq viram 'Other_low_freq'; nulos viram 'Unknown'."""
    freq = serie.value_counts()
    validas = freq[freq >= min_freq].index
    return (serie.where(serie.isin(validas), "Other_low_freq")
                 .fillna("Unknown").astype(str).str.strip())


def sanitize_application_train(df: pd.DataFrame, min_freq: int = 500,
                               income_winsor_q: float = 0.99) -> pd.DataFrame:
    """Sanitiza `application_train` -> base limpa (1 linha por cliente, sem nulos)."""
    c = pd.DataFrame()
    c["sk_id_curr"] = pd.to_numeric(df["sk_id_curr"], errors="coerce").astype("Int64")
    c["target"] = pd.to_numeric(df["target"], errors="coerce").astype("Int64")

    # Scores externos: ext_source_2 + (1 e 3 imputados) + media combinada (preditor mais forte)
    c["ext_source_2"] = pd.to_numeric(df["ext_source_2"], errors="coerce").fillna(df["ext_source_2"].median())
    _es = pd.concat([
        pd.to_numeric(df["ext_source_1"], errors="coerce"),
        pd.to_numeric(df["ext_source_2"], errors="coerce"),
        pd.to_numeric(df["ext_source_3"], errors="coerce"),
    ], axis=1)
    c["ext_source_mean"] = _es.mean(axis=1)
    c["ext_source_mean"] = c["ext_source_mean"].fillna(c["ext_source_mean"].median())
    c["ext_source_1"] = _es.iloc[:, 0].fillna(_es.iloc[:, 0].median())
    c["ext_source_3"] = _es.iloc[:, 2].fillna(_es.iloc[:, 2].median())

    # region_rating: mantem apenas _w_city (redundante com region_rating_client, corr 0.95)
    c["region_rating_client_w_city"] = pd.to_numeric(df["region_rating_client_w_city"], errors="coerce").astype("Int64")

    c["days_last_phone_change"] = pd.to_numeric(df["days_last_phone_change"], errors="coerce").fillna(df["days_last_phone_change"].median())
    c["days_id_publish"] = pd.to_numeric(df["days_id_publish"], errors="coerce")
    c["days_registration"] = pd.to_numeric(df["days_registration"], errors="coerce")

    for col in ["reg_city_not_work_city", "reg_city_not_live_city", "live_city_not_work_city"]:
        c[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # own_car_age + flag has_car (separa 'sem carro' de 'carro novo')
    _own_car_age = pd.to_numeric(df["own_car_age"], errors="coerce")
    c["has_car"] = (df["flag_own_car"].astype(str).str.strip() == "Y").astype(int)
    _median_car_age = _own_car_age[c["has_car"] == 1].median()
    c["own_car_age"] = _own_car_age.where(c["has_car"] == 1, 0).fillna(_median_car_age)

    # def_60 mantido (redundante com def_30, corr 0.86)
    c["def_60_cnt_social_circle"] = pd.to_numeric(df["def_60_cnt_social_circle"], errors="coerce").fillna(0)
    c["amt_req_credit_bureau_year"] = pd.to_numeric(df["amt_req_credit_bureau_year"], errors="coerce").fillna(0)
    c["cnt_children"] = pd.to_numeric(df["cnt_children"], errors="coerce").fillna(0).astype(int)
    c["cnt_fam_members"] = pd.to_numeric(df["cnt_fam_members"], errors="coerce").fillna(df["cnt_fam_members"].median())

    # Renda: zero -> NaN -> mediana, e winsorizacao no p99 (outlier extremo de ~117M)
    income = pd.to_numeric(df["amt_income_total"], errors="coerce").replace(0, np.nan)
    income = income.fillna(income.median())
    p99 = income.quantile(income_winsor_q)
    c["amt_income_total"] = income.clip(upper=p99)

    c["amt_credit"] = pd.to_numeric(df["amt_credit"], errors="coerce")
    c["amt_annuity"] = pd.to_numeric(df["amt_annuity"], errors="coerce").fillna(df["amt_annuity"].median())
    # amt_goods_price removido (redundante com amt_credit, corr 0.99)

    # Categoricas: imputacao 'Unknown' + reducao de cardinalidade das de alta cardinalidade
    c["occupation_type"] = df["occupation_type"].fillna("Unknown").astype(str).str.strip()
    c["organization_type"] = _reduz_cardinalidade(df["organization_type"], min_freq)
    c["name_income_type"] = _reduz_cardinalidade(df["name_income_type"], min_freq)
    c["name_education_type"] = df["name_education_type"].fillna("Unknown").astype(str).str.strip()
    c["code_gender"] = df["code_gender"].replace("XNA", "Unknown").fillna("Unknown").astype(str).str.strip()

    return c


def sanitize_bureau(df_bureau: pd.DataFrame) -> pd.DataFrame:
    """Tipagem + imputacao das colunas de `bureau` usadas nas agregacoes (usada pelo abt_transform_v2)."""
    b = df_bureau.copy()
    b["credit_active"] = b["credit_active"].astype(str).str.strip()
    b["credit_type"] = b["credit_type"].astype(str).str.strip()
    for col in ["amt_credit_sum", "amt_credit_sum_debt", "amt_credit_sum_overdue",
                "credit_day_overdue", "cnt_credit_prolong"]:
        b[col] = pd.to_numeric(b[col], errors="coerce").fillna(0)
    for col in ["days_credit", "days_credit_update"]:
        b[col] = pd.to_numeric(b[col], errors="coerce")
    return b


# --------------------------------------------------------------------------
# Funcao mestre chamada pelo DAG
# --------------------------------------------------------------------------
def run_sanitization_v2(conn_id: str) -> None:
    """Sanitiza application_train e grava a base limpa em `output_table_v2` (application_clean_v2)."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_config(os.path.join(base_dir, "config_pipeline.json"))
    db = config["database"]
    params = config.get("sanitization_v2", {})
    min_freq = params.get("cardinalidade_min_freq", 500)
    winsor_q = params.get("income_winsor_q", 0.99)

    input_table = db["input_table"]
    output_table = db["output_table_v2"]

    pg_hook = PostgresHook(postgres_conn_id=conn_id)
    conn = pg_hook.get_conn()
    cursor = conn.cursor()

    print(f"[v2] Carregando '{input_table}' inteiro (estatisticas globais)...")
    df = pd.read_sql(f'SELECT * FROM "{input_table}"', conn)

    clean = sanitize_application_train(df, min_freq=min_freq, income_winsor_q=winsor_q)
    nulos = int(clean.isna().sum().sum())
    print(f"[v2] Base limpa: {clean.shape[0]:,} linhas x {clean.shape[1]} colunas | nulos={nulos}")

    print(f"[v2] Gravando na tabela '{output_table}'...")
    create_table_from_df(cursor, clean, output_table)
    conn.commit()
    copy_df_to_table(cursor, clean, output_table)
    conn.commit()

    cursor.close()
    conn.close()
    print(f"--- [v2] Sanitizacao concluida! Tabela '{output_table}' criada. ---")


if __name__ == "__main__":
    run_sanitization_v2("postgres_data_db")
