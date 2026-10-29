"""Shared paths, seeding, metrics and result logging for CallShield."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(os.environ.get("CALLSHIELD_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
for _p in (RAW, PROCESSED, MODELS, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def binary_metrics(y_true, y_score, threshold: float = 0.5, sample_weight=None) -> dict:
    """Metrics for an imbalanced binary problem.

    PR-AUC (average precision) is the headline metric: with few positives,
    ROC-AUC and accuracy look good even for weak models, PR-AUC does not.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    out = {
        "n": int(len(y_true)),
        "positive_rate": float(np.average(y_true, weights=w)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0, sample_weight=w)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0, sample_weight=w)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0, sample_weight=w)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0, sample_weight=w)),
        "accuracy": float(np.average(y_true == y_pred, weights=w)),
        "threshold": float(threshold),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score, sample_weight=w))
        out["pr_auc"] = float(average_precision_score(y_true, y_score, sample_weight=w))
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}


def save_results(name: str, payload: dict) -> Path:
    """Write a results file to reports/ - the single source of truth for README/CV numbers."""
    payload = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), **payload}
    path = REPORTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"[results] wrote {path}")
    return path


def mlflow_run(experiment: str, run_name: str):
    """Start an MLflow run tracked in a local SQLite DB (mlflow.db; view with `mlflow ui --backend-store-uri sqlite:///mlflow.db`)."""
    try:
        import mlflow
    except ImportError:  # allows quick runs without MLflow installed
        import contextlib

        return contextlib.nullcontext()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
    mlflow.set_experiment(experiment)
    return mlflow.start_run(run_name=run_name)


def hardware_info() -> dict:
    import platform

    info = {"machine": platform.machine(), "system": platform.system(), "cpus": os.cpu_count()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["device"] = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    except ImportError:
        pass
    return info


class _NoMlflow:
    def __getattr__(self, _):
        return lambda *a, **k: None


def mlflow_or_stub():
    try:
        import mlflow

        return mlflow
    except ImportError:
        return _NoMlflow()
