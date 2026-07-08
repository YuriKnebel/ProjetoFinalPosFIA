FROM jupyter/datascience-notebook:latest

# Driver do Postgres (psycopg2) usado pelo SQLAlchemy no exp_analysis.ipynb.
# A imagem base não traz o psycopg2; instalamos aqui para a conexão funcionar.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt