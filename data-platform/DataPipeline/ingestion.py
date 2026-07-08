import os
import io
import pandas as pd
from airflow.providers.postgres.hooks.postgres import PostgresHook

# Lotes na leitura/gravação de CSV — mantém a memória baixa mesmo em arquivos
# grandes (ex.: installments_payments.csv ~723MB). Ler o arquivo inteiro de uma
# vez causava OOM (o processo era morto com return code -9).
INGESTION_CHUNK_SIZE = 200_000


def _map_pg_type(dtype) -> str:
    d = str(dtype).lower()
    if "int" in d:
        return "BIGINT"
    if "float" in d:
        return "DOUBLE PRECISION"
    if "bool" in d:
        return "BOOLEAN"
    if "datetime" in d:
        return "TIMESTAMP"
    return "TEXT"


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [
        str(c).lower().replace("-", "_").replace(" ", "_").replace(".", "_")
        for c in df.columns
    ]
    return df


def _create_table(cursor, nome_tabela: str, sample_df: pd.DataFrame) -> list:
    """Recria a tabela a partir do schema do DataFrame. Retorna as colunas int."""
    colunas, int_cols = [], []
    for col, dtype in zip(sample_df.columns, sample_df.dtypes):
        pg_type = _map_pg_type(dtype)
        colunas.append(f'"{col}" {pg_type}')
        if pg_type == "BIGINT":
            int_cols.append(col)

    cursor.execute(f'DROP TABLE IF EXISTS "{nome_tabela}" CASCADE;')
    cursor.execute(f'CREATE TABLE "{nome_tabela}" ({", ".join(colunas)});')
    return int_cols


def _copy_chunk(cursor, nome_tabela: str, chunk: pd.DataFrame, int_cols: list):
    # Colunas inteiras (BIGINT no schema) são normalizadas para Int64 anulável, de
    # forma que valores nulos que só aparecem em lotes posteriores não virem "1.0"
    # (o que quebraria o COPY em uma coluna BIGINT).
    for col in int_cols:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("Int64")

    output = io.StringIO()
    chunk.to_csv(output, sep="\t", header=False, index=False)
    output.seek(0)
    cursor.copy_expert(
        f'COPY "{nome_tabela}" FROM STDIN WITH CSV DELIMITER \'\t\' NULL \'\'', output
    )


def _detect_encoding(caminho: str) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            pd.read_csv(caminho, nrows=5, encoding=enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def _ingest_csv(cursor, conn, caminho: str, nome_tabela: str, arquivo: str):
    """Ingestão de CSV em lotes (memória constante, independente do tamanho)."""
    encoding = _detect_encoding(caminho)
    if encoding != "utf-8":
        print(f"Aviso: UTF-8 falhou para {arquivo}. Lendo com {encoding}...")

    total = 0
    int_cols = []
    is_first = True
    for chunk in pd.read_csv(caminho, chunksize=INGESTION_CHUNK_SIZE, encoding=encoding):
        chunk = _normalize_cols(chunk)
        if is_first:
            int_cols = _create_table(cursor, nome_tabela, chunk)
            conn.commit()
            is_first = False
        _copy_chunk(cursor, nome_tabela, chunk, int_cols)
        conn.commit()
        total += len(chunk)
        print(f"  ... {nome_tabela}: {total} linhas carregadas")

    if is_first:
        # arquivo sem linhas de dados
        _create_table(cursor, nome_tabela, pd.read_csv(caminho, nrows=0, encoding=encoding))
        conn.commit()
    return total


def _ingest_dataframe(cursor, conn, df: pd.DataFrame, nome_tabela: str):
    """Ingestão de arquivos pequenos (JSON/Excel) carregados por inteiro."""
    df = _normalize_cols(df)
    int_cols = _create_table(cursor, nome_tabela, df)
    conn.commit()
    _copy_chunk(cursor, nome_tabela, df, int_cols)
    conn.commit()
    return len(df)


def run_csv_ingestion(conn_id: str, pasta_origem: str):
    """Carrega para o Postgres todos os arquivos (CSV/JSON/Excel) da pasta de origem.

    Cada arquivo vira uma tabela (drop & create) nomeada pelo próprio arquivo,
    normalizado. CSVs são carregados em lotes para não estourar a memória.
    Falhas em um arquivo não interrompem os demais.
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
        ext = extensao.lower()

        try:
            print(f"Iniciando carga de {arquivo} para tabela '{nome_tabela}'...")

            if ext == ".csv":
                total = _ingest_csv(cursor, conn, caminho_completo, nome_tabela, arquivo)
            elif ext == ".json":
                total = _ingest_dataframe(cursor, conn, pd.read_json(caminho_completo), nome_tabela)
            elif ext in [".xlsx", ".xls"]:
                total = _ingest_dataframe(cursor, conn, pd.read_excel(caminho_completo), nome_tabela)
            else:
                print(f"Formato '{extensao}' ignorado para o arquivo: {arquivo}")
                continue

            print(f"Sucesso! Tabela '{nome_tabela}' criada e populada com {total} linhas.")

        except Exception as e:
            conn.rollback()
            # Evita deixar uma tabela meio carregada em caso de falha no meio do arquivo
            try:
                cursor.execute(f'DROP TABLE IF EXISTS "{nome_tabela}" CASCADE;')
                conn.commit()
            except Exception:
                conn.rollback()
            print(f"Falha ao processar o arquivo {arquivo}. Erro: {str(e)}")
            print("Aviso: Pulando para o próximo arquivo para não travar o pipeline...")
            continue

    cursor.close()
    conn.close()
    print("--- Ingestão dos arquivos concluída! ---")
