"""Servico de predicao local: aplica o modelo treinado a novos clientes.

Corresponde a "Etapa 10" do fluxo de modelagem (usar o modelo em dados novos).
Le o artefato produzido pelo ``train.py`` e devolve o score de risco + a decisao.

Tres formas de entrada (a partir de ``data-platform``, ex.: no container jupyter)::

    # 1) um cliente da propria ABT no Postgres, pelo sk_id_curr
    python Model/predict.py --sk-id 100002

    # 2) um JSON com as features do cliente
    python Model/predict.py --input cliente.json

    # 3) uma linha de um CSV (sk_id_curr/target sao ignorados)
    python Model/predict.py --input Dados/abt.csv --row 0
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


MODEL_DIR = Path(__file__).resolve().parent
CONFIG_PATH = MODEL_DIR / "config_model.json"


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Carrega variaveis, parametros e metadados documentados do modelo."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    missing = {"metadata", "variables", "parameters"}.difference(config)
    if missing:
        raise ValueError(f"Configuracao invalida; secoes ausentes: {sorted(missing)}")
    return config


def default_model_path() -> Path:
    return MODEL_DIR.parent / load_config()["metadata"]["artifact"]


def load_artifact(model_path: Path | None = None) -> dict[str, Any]:
    """Carrega e valida o pacote produzido pelo train.py."""
    model_path = model_path or default_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Modelo nao encontrado: {model_path}. Rode antes: python Model/train.py")
    with model_path.open("rb") as file:
        artifact = pickle.load(file)
    missing = {"model", "decision_threshold", "input_features"}.difference(artifact)
    if missing:
        raise ValueError(f"Artefato invalido; chaves ausentes: {sorted(missing)}")
    return artifact


def non_feature_columns(config: dict[str, Any]) -> set[str]:
    v = config["variables"]
    return {v["identifier"], v["target"]}


def load_features_from_file(input_path: Path, row: int, config: dict[str, Any]) -> dict[str, Any]:
    """Le as features de um JSON (objeto) ou de uma linha de CSV."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Entrada nao encontrada: {input_path}")

    if input_path.suffix.lower() == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        features = payload.get("features", payload)
        if not isinstance(features, dict):
            raise ValueError("O JSON deve ser um objeto de features (ou conter a chave 'features').")
        return features

    if input_path.suffix.lower() == ".csv":
        frame = pd.read_csv(input_path, skiprows=range(1, row + 1), nrows=1)
        if frame.empty:
            raise IndexError(f"A linha {row} nao existe no CSV.")
        ignore = non_feature_columns(config)
        return {k: v for k, v in frame.iloc[0].to_dict().items() if k not in ignore}

    raise ValueError("Formato nao suportado. Use um arquivo .json ou .csv.")


def load_features_from_abt(sk_id: int, config: dict[str, Any]) -> dict[str, Any]:
    """Busca um cliente da ABT no Postgres pelo identificador (sk_id_curr)."""
    from sqlalchemy import create_engine  # import tardio: so precisa se usar --sk-id

    db = config["database"]
    host = db["host_docker"] if os.path.exists("/.dockerenv") else db["host_local"]
    engine = create_engine(f"postgresql://{db['user']}:{db['password']}@{host}:{db['port']}/{db['dbname']}")
    table = config["metadata"]["abt_table"]
    idcol = config["variables"]["identifier"]
    frame = pd.read_sql(f"SELECT * FROM {table} WHERE {idcol} = {int(sk_id)}", engine)
    if frame.empty:
        raise ValueError(f"Cliente {idcol}={sk_id} nao encontrado em {table}.")
    ignore = non_feature_columns(config)
    return {k: v for k, v in frame.iloc[0].to_dict().items() if k not in ignore}


def prepare_features(features: dict[str, Any], artifact: dict[str, Any]) -> pd.DataFrame:
    """Monta o DataFrame na ordem do treino e reconstroi as categoricas nativas.

    As categoricas precisam ser marcadas como 'category' com as MESMAS categorias
    vistas no treino (guardadas no artefato) — categoria desconhecida vira NaN, que
    o LightGBM trata como ausente.
    """
    expected = list(artifact["input_features"])
    faltando = sorted(set(expected).difference(features))
    if faltando:
        raise ValueError(f"Features ausentes na entrada: {', '.join(faltando)}")

    df = pd.DataFrame([features]).reindex(columns=expected)
    categoricals = artifact.get("categorical_features", [])
    categories = artifact.get("categories", {})
    for col in expected:
        if col in categoricals:
            df[col] = pd.Categorical(df[col].astype("object"), categories=categories.get(col))
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def predict(features: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    """Calcula o score de risco e a decisao para um cliente."""
    customer = prepare_features(features, artifact)
    score = float(artifact["model"].predict_proba(customer)[0, 1])
    threshold = float(artifact["decision_threshold"])
    return {
        "risk_score": round(score, 6),
        "decision_threshold": threshold,
        "predicted_class": int(score >= threshold),
        "decision": "NEGAR" if score >= threshold else "APROVAR",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predicao local de risco de credito")
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--input", type=Path, help="Arquivo .json ou .csv com as features")
    grupo.add_argument("--sk-id", type=int, help="Busca o cliente na ABT do Postgres pelo sk_id_curr")
    parser.add_argument("--row", type=int, default=0, help="Linha do CSV (inicia em zero)")
    parser.add_argument("--model", type=Path, default=None, help="Sobrescreve o caminho do artefato")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.row < 0:
        raise ValueError("--row deve ser maior ou igual a zero.")
    config = load_config()
    artifact = load_artifact(args.model)

    if args.sk_id is not None:
        features = load_features_from_abt(args.sk_id, config)
    else:
        features = load_features_from_file(args.input, args.row, config)

    result = predict(features, artifact)
    if args.sk_id is not None:
        result = {"sk_id_curr": args.sk_id, **result}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()