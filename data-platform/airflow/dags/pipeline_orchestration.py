from datetime import datetime
import sys
from airflow import DAG
from airflow.decorators import task

sys.path.append("/opt/airflow/DataPipeline")

from data_sanitization import (
    run_sanitization,
    run_prev_sanitization,
    run_bureau_sanitization,
)
from abt_transform import run_abt_generation

# Constantes centralizadas
CONN_ID = "postgres_data_db"

default_args = {
    "depends_on_past": False,
    "start_date": datetime(2026, 6, 28),
    "retries": 0,
}

with DAG(
    "pipeline_orchestration",
    default_args=default_args,
    description="Orquestrador de Sanitização e ABT (application + previous_application + bureau)",
    schedule_interval=None,
    catchup=False,
    tags=["pipeline", "sanitization", "abt"],
) as dag:

    @task(task_id="data_sanitization")
    def task_sanitize(conn_id: str):
        # Sanitiza o application_train -> application_clean
        run_sanitization(conn_id)

    @task(task_id="clean_previous_application")
    def task_sanitize_prev(conn_id: str):
        # Sanitiza o previous_application -> previous_application_clean
        run_prev_sanitization(conn_id)

    @task(task_id="clean_bureau")
    def task_sanitize_bureau(conn_id: str):
        # Sanitiza o bureau -> bureau_clean
        run_bureau_sanitization(conn_id)

    @task(task_id="abt_transform")
    def task_abt(conn_id: str):
        # Monta a ABT unindo as três fontes -> application_abt
        run_abt_generation(conn_id)

    # Fluxo de execução: sanitizações das três fontes e, por fim, a montagem da ABT
    (
        task_sanitize(CONN_ID)
        >> task_sanitize_prev(CONN_ID)
        >> task_sanitize_bureau(CONN_ID)
        >> task_abt(CONN_ID)
    )
