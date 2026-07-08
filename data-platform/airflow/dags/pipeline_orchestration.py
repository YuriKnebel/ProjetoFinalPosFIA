from datetime import datetime
import sys

# Mapeamento dos caminhos do projeto
sys.path.append("/opt/airflow/DataPipeline")

from ingestion import run_csv_ingestion
from data_sanitization import (
    run_sanitization,
    run_prev_sanitization,
    run_bureau_sanitization,
    run_installments_sanitization,
)
from abt_transform import run_abt_generation

from airflow import DAG
from airflow.decorators import task

# Parâmetros de infraestrutura imutáveis
CONN_ID = "postgres_data_db"
PASTA_DATA = "/opt/airflow/data/csv"

with DAG(
    dag_id="pipeline_orchestration",
    start_date=datetime(2026, 6, 24),
    schedule=None,
    catchup=False,
    tags=["pipeline", "ingestion", "sanitization", "abt"],
) as dag:

    @task(task_id="ingest_csv_source")
    def task_ingest(conn_id: str, pasta_origem: str):
        # Ingestão dos CSVs brutos para o Postgres
        run_csv_ingestion(conn_id, pasta_origem)

    @task(task_id="sanitize_application_train")
    def task_sanitize_app(conn_id: str):
        # application_train -> application_clean
        run_sanitization(conn_id)

    @task(task_id="sanitize_previous_application")
    def task_sanitize_prev(conn_id: str):
        # previous_application -> previous_application_clean
        run_prev_sanitization(conn_id)

    @task(task_id="run_bureau_sanitization")
    def task_sanitize_bureau(conn_id: str):
        # bureau -> bureau_clean
        run_bureau_sanitization(conn_id)

    @task(task_id="clean_installments")
    def task_sanitize_installments(conn_id: str):
        # installments_payments -> installments_clean
        run_installments_sanitization(conn_id)

    @task(task_id="generate_analytical_base_table")
    def task_abt(conn_id: str):
        # Une as quatro fontes -> application_abt
        run_abt_generation(conn_id)

    # Instanciando as tasks
    carga_inicial = task_ingest(conn_id=CONN_ID, pasta_origem=PASTA_DATA)
    limpeza_app = task_sanitize_app(CONN_ID)
    limpeza_prev = task_sanitize_prev(CONN_ID)
    limpeza_bureau = task_sanitize_bureau(CONN_ID)
    limpeza_installments = task_sanitize_installments(CONN_ID)
    construcao_abt = task_abt(CONN_ID)

    # --- ORQUESTRAÇÃO DAS TASKS ---
    # Sanitizações em pares (no máximo 2 simultâneas, para não saturar o Postgres):
    #   1º par: application + previous_application
    #   2º par: bureau + installments
    # As duas mais pesadas em pandas (prev e bureau) ficam em pares distintos, então
    # nunca disputam CPU/RAM do worker ao mesmo tempo.
    carga_inicial >> [limpeza_app, limpeza_prev]
    limpeza_app >> [limpeza_bureau, limpeza_installments]
    limpeza_prev >> [limpeza_bureau, limpeza_installments]
    [limpeza_bureau, limpeza_installments] >> construcao_abt
