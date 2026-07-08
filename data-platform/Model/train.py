"""Treina o modelo de risco de credito (LightGBM) usando ``config_model.json``.

Le a ABT ja limpa direto do Postgres (tabela ``application_abt``, saida da pipeline),
treina o LightGBM com **categoricas nativas** (sem one-hot) usando os hiperparametros
escolhidos na validacao (Modelo 25 do ``validacao_modelos.ipynb``), avalia num holdout
e retreina o modelo final na base completa. O artefato final e um pacote (pickle) com
o modelo + metadados, pronto para o ``predict.py``.

Uso, a partir de ``data-platform`` (ex.: dentro do container jupyter)::

    python Model/train.py                 # treino completo
    python Model/train.py --sample-size 20000   # smoke test rapido
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             classification_report, roc_auc_score, roc_curve)
from lightgbm import LGBMClassifier


MODEL_DIR = Path(__file__).resolve().parent
DATA_PLATFORM_DIR = MODEL_DIR.parent
DEFAULT_CONFIG_PATH = MODEL_DIR / "config_model.json"


def load_config(path: Path) -> dict[str, Any]:
    """Carrega e valida as secoes obrigatorias do config."""
    config = json.loads(path.read_text(encoding="utf-8"))
    missing = {"metadata", "variables", "parameters"}.difference(config)
    if missing:
        raise ValueError(f"Configuracao incompleta; secoes ausentes: {sorted(missing)}")
    return config


def project_path(configured_path: str) -> Path:
    """Resolve um caminho do config relativo a pasta data-platform."""
    return DATA_PLATFORM_DIR / configured_path


def get_engine(config: dict[str, Any]):
    """Cria a engine do Postgres, detectando o host (docker x local)."""
    db = config["database"]
    host = db["host_docker"] if os.path.exists("/.dockerenv") else db["host_local"]
    url = f"postgresql://{db['user']}:{db['password']}@{host}:{db['port']}/{db['dbname']}"
    print(f"[dados] Conectando ao Postgres em {host}:{db['port']}/{db['dbname']}")
    return create_engine(url)


def load_training_data(config: dict[str, Any], sample_size: int | None = None):
    """Le a ABT do Postgres e devolve X, y com as categoricas marcadas como 'category'.

    Marcar as categoricas como ``category`` faz o LightGBM usar o split otimo nativo
    (agrupa categorias numa unica divisao), sem one-hot e sem impor ordem falsa.
    """
    engine = get_engine(config)
    table = config["metadata"]["abt_table"]
    query = f"SELECT * FROM {table}"
    if sample_size:
        query += f" LIMIT {int(sample_size)}"
    frame = pd.read_sql(query, engine)
    print(f"[dados] ABT carregada: {frame.shape[0]:,} linhas x {frame.shape[1]} colunas")

    variables = config["variables"]
    features = variables["input_features"]
    target = variables["target"]
    categoricals = variables["categorical_features"]

    required = set(features) | {target}
    faltando = sorted(required.difference(frame.columns))
    if faltando:
        raise ValueError(f"A ABT nao contem as colunas configuradas: {faltando}")

    X = frame[features].replace([np.inf, -np.inf], np.nan).copy()
    y = frame[target].astype(int)
    for col in categoricals:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X, y


def build_model(config: dict[str, Any]) -> LGBMClassifier:
    """Instancia o LightGBM com os hiperparametros fixos do config (Modelo 25)."""
    hp = dict(config["parameters"]["classifier"]["hyperparameters"])
    return LGBMClassifier(
        random_state=config["parameters"]["random_state"],
        n_jobs=-1,
        verbosity=-1,
        **hp,
    )


def credit_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Metricas de risco de credito a partir do score previsto."""
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)
    return {
        "roc_auc": round(float(auc), 4),
        "gini": round(float(2 * auc - 1), 4),
        "ks": round(float((tpr - fpr).max()), 4),
        "average_precision": round(float(average_precision_score(y_true, proba)), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
    }


def train(config: dict[str, Any], sample_size: int | None = None) -> dict[str, Any]:
    """Treina, avalia no holdout e retreina o modelo final na base completa."""
    X, y = load_training_data(config, sample_size)
    params = config["parameters"]
    seed = params["random_state"]
    threshold = params["inference"]["decision_threshold"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["split"]["test_size"],
        stratify=y if params["split"]["stratify"] else None,
        random_state=seed,
    )

    # 1) modelo de avaliacao: treina no treino, mede no holdout (metricas honestas)
    print("[treino] Ajustando modelo de avaliacao (holdout)...")
    eval_model = build_model(config).fit(X_train, y_train)
    score = eval_model.predict_proba(X_test)[:, 1]
    metrics = credit_metrics(y_test.to_numpy(), score)
    print(f"[avaliacao] Metricas no teste externo: {json.dumps(metrics, ensure_ascii=False)}")
    print(classification_report(y_test, (score >= threshold).astype(int),
                                target_names=["Adimplente (0)", "Inadimplente (1)"], digits=4))

    # 2) modelo final: retreina em TODA a base (usa 100% dos dados para o deploy)
    print("[treino] Retreinando o modelo final na base completa...")
    final_model = build_model(config).fit(X, y)

    categoricals = [c for c in config["variables"]["categorical_features"] if c in X.columns]
    return {
        "model": final_model,
        "decision_threshold": threshold,
        "input_features": list(X.columns),
        "categorical_features": categoricals,
        "categories": {c: [str(v) for v in X[c].cat.categories] for c in categoricals},
        "metrics": metrics,
        "algorithm": config["parameters"]["classifier"]["algorithm"],
        "hyperparameters": config["parameters"]["classifier"]["hyperparameters"],
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_version": config["metadata"]["version"],
    }


def save_artifact(artifact: dict[str, Any], output: Path) -> None:
    """Salva o pacote do modelo (pickle) e um metrics.json legivel ao lado."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        pickle.dump(artifact, file)
    metrics_path = output.parent / "metrics.json"
    resumo = {
        "algorithm": artifact["algorithm"],
        "hyperparameters": artifact["hyperparameters"],
        "test_metrics": artifact["metrics"],
        "decision_threshold": artifact["decision_threshold"],
        "trained_at_utc": artifact["trained_at_utc"],
    }
    metrics_path.write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[artefato] Modelo salvo em: {output}")
    print(f"[artefato] Metricas salvas em: {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="Caminho do config_model.json")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Le apenas N linhas da ABT (smoke test rapido)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Sobrescreve o caminho do artefato")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    artifact = train(config, args.sample_size)
    output = args.output or project_path(config["metadata"]["artifact"])
    save_artifact(artifact, output)


if __name__ == "__main__":
    main()