import os
import io
import pandas as pd
from airflow.providers.postgres.hooks.postgres import PostgresHook


def _map_pg_type(dtype) -> str:
    d = str(dtype)
    if "int" in d:
        return "BIGINT"
    if "float" in d:
        return "DOUBLE PRECISION"
    if "bool" in d:
        return "BOOLEAN"
    if "datetime" in d:
        return "TIMESTAMP"
    return "TEXT"


def run_csv_ingestion(conn_id: str, pasta_origem: str):
    """Carrega para o Postgres todos os arquivos (CSV/JSON/Excel) da pasta de origem.

    Cada arquivo vira uma tabela (drop & create) nomeada pelo próprio arquivo,
    normalizado. Falhas em um arquivo não interrompem os demais.
    """
    pg_hook = PostgresHook(postgres_conn_id=conn_id)
    conn = pg_hook.get_conn()
    cursor = conn.cursor()

    if not os.path.exists(pasta_origem):
        raise FileNotFoundError(f"A pasta {pasta_origem} não existe no container.")

    arquivos = os.listdir(pasta_origem)
    print(f"Arquivos detectados para processamento: {arquivos}")

    if not arquivos:
        print("Nenhum arquivo encontrado para processar.")
        cursor.close()
        conn.close()
        return

    for arquivo in arquivos:
        caminho_completo = os.path.join(pasta_origem, arquivo)

        if os.path.isdir(caminho_completo):
            continue

        nome_tabela, extensao = os.path.splitext(arquivo)
        nome_tabela = nome_tabela.lower().replace("-", "_").replace(" ", "_")

        try:
            # 1. Leitura resiliente do arquivo (fallback de encoding para CSV)
            if extensao.lower() == ".csv":
                try:
                    df = pd.read_csv(caminho_completo, encoding="utf-8")
                except UnicodeDecodeError:
                    print(f"Aviso: UTF-8 falhou para {arquivo}. Tentando latin-1...")
                    df = pd.read_csv(caminho_completo, encoding="latin-1")
            elif extensao.lower() == ".json":
                df = pd.read_json(caminho_completo)
            elif extensao.lower() in [".xlsx", ".xls"]:
                df = pd.read_excel(caminho_completo)
            else:
                print(f"Formato '{extensao}' ignorado para o arquivo: {arquivo}")
                continue

            print(f"Iniciando carga de {arquivo} para tabela '{nome_tabela}'...")

            # 2. Mapeamento de colunas -> tipos Postgres
            colunas = []
            for col, dtype in zip(df.columns, df.dtypes):
                col_nome = str(col).lower().replace("-", "_").replace(" ", "_").replace(".", "_")
                colunas.append(f'"{col_nome}" {_map_pg_type(dtype)}')

            # 3. Recria a tabela do zero antes da carga
            cursor.execute(f'DROP TABLE IF EXISTS "{nome_tabela}" CASCADE;')
            cursor.execute(f'CREATE TABLE "{nome_tabela}" ({", ".join(colunas)});')

            # 4. Carga de alta performance via COPY em memória
            output = io.StringIO()
            df.to_csv(output, sep="\t", header=False, index=False)
            output.seek(0)
            cursor.copy_expert(
                f'COPY "{nome_tabela}" FROM STDIN WITH CSV DELIMITER \'\t\' NULL \'\'', output
            )
            conn.commit()

            print(f"Sucesso! Tabela '{nome_tabela}' criada e populada com {len(df)} linhas.")

        except Exception as e:
            conn.rollback()
            print(f"Falha ao processar o arquivo {arquivo}. Erro: {str(e)}")
            print("Aviso: Pulando para o próximo arquivo para não travar o pipeline...")
            continue

    cursor.close()
    conn.close()
    print("--- Ingestão dos arquivos concluída! ---")
