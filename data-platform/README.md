###############################################
UPLOAD DOS ARQUIVOS:
###############################################
insira os arquivos (datasets) na pasta data-platform/data/csv

###############################################
PARA INICIAR O CONTAINER:
###############################################
cd data-platform
docker compose up -d --build

###############################################
PARA ACESSAR O CONTAINER DO AIRFLOW:
###############################################

docker compose exec airflow-webserver bash

##################################################
ACESSE O AIRFLOW 
##################################################

http://localhost:8080/

LOGIN: admin
SENHA: admin

Habilite a DAG pipeline_orchestration e a execute clicando no botão com símbolo de play no canto superior direito para trigar a execução da DAG (ela faz a ingestão dos CSVs, as sanitizações e a montagem da ABT)

O processo leva alguns minutos. Aguarde sua conclusão para seguir para a análise exploratória com o Jupyter notebook

##################################################
ACESSE O JUPYTER
##################################################

http://localhost:8888/

TOKEN: analytics

