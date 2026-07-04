# -*- coding: utf-8 -*-
"""
abt_transform_v2.py — Transformacao em ABT (Home Credit), no padrao da pipeline.

Segue o estilo dos scripts do Airflow (PostgresHook + config_pipeline.json + funcao
mestre run_*(conn_id)), para ser plugado como task no DAG `pipeline_orchestration`.
Contem a logica VALIDADA do notebook `exp_analysis_v2`: features derivadas,
agregacoes de previous_application e bureau, flag has_prev_app/has_bureau e
checagens de integridade.

Le a base limpa de `output_table_v2` (application_clean_v2, gerada pelo
data_sanitization_v2) + as tabelas brutas necessarias, e grava a ABT em
`abt_table_v2` (application_abt_v2), SEM sobrescrever as tabelas da pipeline atual.
"""
import os

import numpy as np
import pandas as pd
from airflow.providers.postgres.hooks.postgres import PostgresHook

from data_sanitization_v2 import (
    load_config,
    create_table_from_df,
    copy_df_to_table,
    sanitize_bureau,
)

# Features de bureau que entram na ABT (as demais agregacoes sao descartadas por baixo sinal)
BUREAU_FEATURE_COLS = [
    "bureau_avg_days_credit",
    "bureau_last_days_credit",
    "bureau_active_rate",
    "bureau_active_count",
    "bureau_closed_rate",
    "bureau_debt_credit_ratio",
    "bureau_overdue_count",
]


# --------------------------------------------------------------------------
# application_train -> features derivadas
# --------------------------------------------------------------------------
def build_features_application(df_app: pd.DataFrame) -> pd.DataFrame:
    """Features derivadas do application_train (idade, tempo de emprego, razoes)."""
    f = pd.DataFrame()
    f["sk_id_curr"] = pd.to_numeric(df_app["sk_id_curr"], errors="coerce").astype("Int64")
    f["age"] = np.abs(pd.to_numeric(df_app["days_birth"], errors="coerce")) / 365.25

    _days_emp = pd.to_numeric(df_app["days_employed"], errors="coerce")
    f["years_employed"] = np.abs(_days_emp.replace(365243, np.nan)).fillna(0) / 365.25
    f["days_employed_anom"] = (_days_emp == 365243).astype(int)

    credit = pd.to_numeric(df_app["amt_credit"], errors="coerce")
    income = pd.to_numeric(df_app["amt_income_total"], errors="coerce").replace(0, np.nan)
    annuity = pd.to_numeric(df_app["amt_annuity"], errors="coerce")
    f["fe_credit_income_percent"] = credit / income
    f["fe_annuity_income_percent"] = annuity / income
    ratios = ["fe_credit_income_percent", "fe_annuity_income_percent"]
    f[ratios] = f[ratios].fillna(f[ratios].median())
    return f


# --------------------------------------------------------------------------
# previous_application -> agregacao + feature
# --------------------------------------------------------------------------
def aggregate_previous_application(df_prev: pd.DataFrame) -> pd.DataFrame:
    """Agrega previous_application para 1 linha por cliente."""
    # Normalizacao defensiva do status (robustez a caixa/espacos: ' refused ' -> 'Refused')
    df_prev = df_prev.copy()
    df_prev["name_contract_status"] = df_prev["name_contract_status"].astype(str).str.strip().str.title()
    agg = (
        df_prev.groupby("sk_id_curr").agg(
            prev_contract_count=("sk_id_prev", "count"),
            prev_refused_count=("name_contract_status", lambda x: (x == "Refused").sum()),
            prev_approved_count=("name_contract_status", lambda x: (x == "Approved").sum()),
            prev_avg_amt_application=("amt_application", "mean"),
            prev_avg_amt_credit=("amt_credit", "mean"),
            prev_avg_annuity=("amt_annuity", "mean"),
            prev_max_annuity=("amt_annuity", "max"),
            prev_avg_down_payment=("amt_down_payment", "mean"),
            prev_avg_cnt_payment=("cnt_payment", "mean"),
            prev_last_days_decision=("days_decision", "max"),
        ).reset_index()
    )
    agg["prev_refused_rate"] = agg["prev_refused_count"] / agg["prev_contract_count"]
    return agg


def build_prev_features(df_prev_agg: pd.DataFrame, df_app_ids: pd.DataFrame) -> pd.DataFrame:
    """Seleciona prev_refused_rate + flag has_prev_app, imputando quem nao tem historico."""
    media = df_prev_agg["prev_refused_rate"].mean()
    feats = df_app_ids[["sk_id_curr"]].merge(
        df_prev_agg[["sk_id_curr", "prev_refused_rate"]], on="sk_id_curr", how="left"
    )
    feats["has_prev_app"] = feats["prev_refused_rate"].notna().astype(int)
    feats["prev_refused_rate"] = feats["prev_refused_rate"].fillna(media)
    return feats


# --------------------------------------------------------------------------
# bureau -> agregacao + selecao
# --------------------------------------------------------------------------
def aggregate_bureau(df_bureau_clean: pd.DataFrame) -> pd.DataFrame:
    """Agrega bureau (creditos em outras instituicoes) para 1 linha por cliente."""
    agg = (
        df_bureau_clean.groupby("sk_id_curr").agg(
            bureau_credit_count=("sk_id_bureau", "count"),
            bureau_active_count=("credit_active", lambda x: (x == "Active").sum()),
            bureau_closed_count=("credit_active", lambda x: (x == "Closed").sum()),
            bureau_bad_debt_count=("credit_active", lambda x: (x == "Bad debt").sum()),
            bureau_sold_count=("credit_active", lambda x: (x == "Sold").sum()),
            bureau_total_credit=("amt_credit_sum", "sum"),
            bureau_avg_credit=("amt_credit_sum", "mean"),
            bureau_total_debt=("amt_credit_sum_debt", "sum"),
            bureau_avg_debt=("amt_credit_sum_debt", "mean"),
            bureau_total_overdue=("amt_credit_sum_overdue", "sum"),
            bureau_max_overdue=("amt_credit_sum_overdue", "max"),
            bureau_max_days_overdue=("credit_day_overdue", "max"),
            bureau_overdue_count=("credit_day_overdue", lambda x: (x > 0).sum()),
            bureau_total_prolong=("cnt_credit_prolong", "sum"),
            bureau_avg_days_credit=("days_credit", "mean"),
            bureau_last_days_credit=("days_credit", "max"),
            bureau_avg_days_credit_update=("days_credit_update", "mean"),
            bureau_last_days_credit_update=("days_credit_update", "max"),
        ).reset_index()
    )
    agg["bureau_active_rate"] = agg["bureau_active_count"] / agg["bureau_credit_count"]
    agg["bureau_closed_rate"] = agg["bureau_closed_count"] / agg["bureau_credit_count"]
    agg["bureau_debt_credit_ratio"] = agg["bureau_total_debt"] / agg["bureau_total_credit"].replace(0, np.nan)
    agg["bureau_overdue_credit_ratio"] = agg["bureau_total_overdue"] / agg["bureau_total_credit"].replace(0, np.nan)
    agg[["bureau_debt_credit_ratio", "bureau_overdue_credit_ratio"]] = (
        agg[["bureau_debt_credit_ratio", "bureau_overdue_credit_ratio"]].fillna(0)
    )
    # outlier no ratio de divida/credito (visto na EDA)
    agg["bureau_debt_credit_ratio"] = agg["bureau_debt_credit_ratio"].clip(lower=-1, upper=1)
    return agg


def select_bureau_features(df_bureau_agg: pd.DataFrame) -> pd.DataFrame:
    """Mantem apenas as features de bureau selecionadas para a ABT."""
    return df_bureau_agg[["sk_id_curr"] + BUREAU_FEATURE_COLS].copy()


# --------------------------------------------------------------------------
# Montagem da ABT
# --------------------------------------------------------------------------
def build_abt(df_app_clean: pd.DataFrame, feats_app: pd.DataFrame,
              feats_prev: pd.DataFrame, feats_bureau: pd.DataFrame) -> pd.DataFrame:
    """Junta as 4 fontes (1 linha por cliente) + flag has_bureau + imputacao das features de bureau."""
    abt = (
        df_app_clean
        .merge(feats_app, on="sk_id_curr", how="left")
        .merge(feats_prev, on="sk_id_curr", how="left")
        .merge(feats_bureau, on="sk_id_curr", how="left")
    )
    abt["has_bureau"] = abt["bureau_active_count"].notna().astype(int)
    abt[BUREAU_FEATURE_COLS] = abt[BUREAU_FEATURE_COLS].fillna(0)
    return abt


def check_abt_integrity(abt: pd.DataFrame, df_app_clean: pd.DataFrame) -> None:
    """Gate de qualidade da ABT (1 linha/cliente, sem perda de linhas, sem nulos)."""
    assert bool(abt["sk_id_curr"].is_unique), "sk_id_curr duplicado na ABT"
    assert abt.shape[0] == df_app_clean["sk_id_curr"].nunique(), "n de linhas != n de clientes"
    nulos = int(abt.isna().sum().sum())
    dist = (abt["target"].value_counts(normalize=True) * 100).round(2).to_dict()
    print(f"[v2] Integridade ABT: 1 linha/cliente OK | linhas={abt.shape[0]:,} | "
          f"colunas={abt.shape[1]} | nulos={nulos} | target%={dist}")
    assert nulos == 0, "ha nulos na ABT"


# --------------------------------------------------------------------------
# Funcao mestre chamada pelo DAG
# --------------------------------------------------------------------------
def run_abt_generation_v2(conn_id: str) -> None:
    """Monta a ABT rica (application + previous_application + bureau) em `abt_table_v2`."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_config(os.path.join(base_dir, "config_pipeline.json"))
    db = config["database"]

    clean_table = db["output_table_v2"]      # application_clean_v2 (base limpa)
    input_table = db["input_table"]          # application_train (bruto, p/ features derivadas)
    prev_table = db["input_prev_table"]      # previous_application (bruto)
    bureau_table = db["input_bureau_table"]  # bureau (bruto)
    abt_table = db["abt_table_v2"]           # application_abt_v2 (saida)

    pg_hook = PostgresHook(postgres_conn_id=conn_id)
    conn = pg_hook.get_conn()
    cursor = conn.cursor()

    print("[v2] Carregando tabelas...")
    df_clean = pd.read_sql(f'SELECT * FROM "{clean_table}"', conn)
    df_app = pd.read_sql(
        f'SELECT sk_id_curr, days_birth, days_employed, amt_credit, amt_income_total, amt_annuity '
        f'FROM "{input_table}"', conn)
    df_prev = pd.read_sql(
        f'SELECT sk_id_curr, sk_id_prev, name_contract_status, amt_application, amt_credit, '
        f'amt_annuity, amt_down_payment, cnt_payment, days_decision FROM "{prev_table}"', conn)
    df_bureau = pd.read_sql(f'SELECT * FROM "{bureau_table}"', conn)

    print("[v2] Gerando features e agregacoes...")
    feats_app = build_features_application(df_app)
    feats_prev = build_prev_features(aggregate_previous_application(df_prev), df_app[["sk_id_curr"]])
    feats_bureau = select_bureau_features(aggregate_bureau(sanitize_bureau(df_bureau)))

    print("[v2] Montando ABT...")
    abt = build_abt(df_clean, feats_app, feats_prev, feats_bureau)
    check_abt_integrity(abt, df_clean)

    print(f"[v2] Gravando na tabela '{abt_table}'...")
    create_table_from_df(cursor, abt, abt_table)
    conn.commit()
    copy_df_to_table(cursor, abt, abt_table)
    conn.commit()

    cursor.close()
    conn.close()
    print(f"--- [v2] ABT construida com sucesso! Tabela '{abt_table}' ({abt.shape[1]} colunas). ---")


if __name__ == "__main__":
    run_abt_generation_v2("postgres_data_db")
