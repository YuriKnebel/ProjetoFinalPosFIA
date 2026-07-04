from datetime import datetime
import sys
import os
from airflow import DAG
from airflow.decorators import task

sys.path.append("/opt/airflow/DataPipeline")
# sys.path.append("/opt/airflow/modelos")

# from data_sanitization import run_sanitization, run_prev_sanitization
# from abt_transform import run_abt_generation
from data_sanitization_v2 import run_sanitization_v2
from abt_transform_v2 import run_abt_generation_v2
# from train import train_model
# from train_logistic import train_logistic_model

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
    description="Orquestrador de Sanitização e ABT Nativo TaskFlow",
    schedule_interval=None,
    catchup=False,
    tags=["pipeline", "sanitization", "abt"],
) as dag:

    # @task(task_id="data_sanitization")
    # def task_sanitize(conn_id: str):
    #     # Chama a função mestre do script passando o ID da conexão nativa
    #     run_sanitization(conn_id)

    # @task(task_id="clean_previous_application")
    # def task_sanitize_prev(conn_id: str):
    #     run_prev_sanitization(conn_id)
    
    # @task(task_id="abt_transform")
    # def task_abt(conn_id: str):
    #     # Chama a função mestre do script da ABT passando o ID da conexão nativa
    #     run_abt_generation(conn_id)

    # @task(task_id="train_machine_learning_model")
    # def task_train(conn_id: str):
    #     train_model(conn_id)

    # @task(task_id="train_logistic_model")
    # def task_train_logistic_model(conn_id: str):
    #     train_logistic_model(conn_id)

    # Fluxo de execução nativo e limpo
    # task_sanitize(CONN_ID) >> task_sanitize_prev(CONN_ID) >> task_abt(CONN_ID) >> task_train(CONN_ID) >> task_train_logistic_model(CONN_ID)

    @task(task_id="data_sanitization_v2")
    def task_sanitize_v2(conn_id: str):
        run_sanitization_v2(conn_id)

    @task(task_id="abt_transform_v2")
    def task_abt_v2(conn_id: str):
        run_abt_generation_v2(conn_id)

    # fluxo v2 — independente da pipeline atual
    task_sanitize_v2(CONN_ID) >> task_abt_v2(CONN_ID)

