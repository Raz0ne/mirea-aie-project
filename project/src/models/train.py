"""End-to-end training: load data, train baseline + improved model, evaluate, save artifact.

Usage:
    python -m src.models.train
    python -m src.models.train --config configs/model.yaml --out artifacts/model.joblib

The script saves a single joblib bundle that the service consumes:

    {
        "pipeline":  sklearn.pipeline.Pipeline,
        "feature_columns":  list[str],
        "threshold":  float,
        "metrics":  {...},
        "model_name":  str,
    }
"""
from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import joblib

from src.data.dataset import FEATURE_COLUMNS, TARGET, load_dataset, split_dataset
from src.models.baseline import build_baseline_pipeline
from src.models.improved import build_improved_pipeline
from src.models.metrics import evaluate
from src.utils.config import PROJECT_ROOT, TrainConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _fit_and_score(name, pipeline, X_train, y_train, X_val, y_val, threshold=0.5):
    logger.info("Training %s ...", name)
    pipeline.fit(X_train, y_train)
    val_proba = pipeline.predict_proba(X_val)[:, 1]
    metrics = evaluate(y_val, val_proba, threshold=threshold)
    logger.info(
        "%-12s val: ROC-AUC=%.4f PR-AUC=%.4f KS=%.4f F1@%.2f=%.4f",
        name,
        metrics.roc_auc,
        metrics.pr_auc,
        metrics.ks_statistic,
        threshold,
        metrics.f1_at_threshold,
    )
    return pipeline, metrics


def train(config_path: str | None = None, output_path: str | None = None) -> dict:
    cfg = TrainConfig.load(config_path) if config_path else TrainConfig.load()

    df = load_dataset(cfg)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, cfg)

    threshold = 0.5

    baseline_pipeline, baseline_val = _fit_and_score(
        "baseline",
        build_baseline_pipeline(
            **cfg.baseline.params,
            random_state=cfg.data.random_state,
        ),
        X_train,
        y_train,
        X_val,
        y_val,
        threshold=threshold,
    )

    improved_pipeline, improved_val = _fit_and_score(
        "improved",
        build_improved_pipeline(
            **cfg.improved.params,
            random_state=cfg.data.random_state,
        ),
        X_train,
        y_train,
        X_val,
        y_val,
        threshold=threshold,
    )

    # Choose the better of the two by ROC-AUC on the val set.
    if improved_val.roc_auc >= baseline_val.roc_auc:
        chosen_name = "improved (lightgbm)"
        chosen_pipeline = improved_pipeline
        chosen_val = improved_val
    else:
        chosen_name = "baseline (logreg)"
        chosen_pipeline = baseline_pipeline
        chosen_val = baseline_val

    logger.info("Selected final model: %s (val ROC-AUC=%.4f)", chosen_name, chosen_val.roc_auc)

    # Final unbiased estimate on the held-out test set.
    test_proba = chosen_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_proba, threshold=threshold)
    logger.info(
        "Final test metrics: ROC-AUC=%.4f PR-AUC=%.4f KS=%.4f F1@%.2f=%.4f",
        test_metrics.roc_auc,
        test_metrics.pr_auc,
        test_metrics.ks_statistic,
        threshold,
        test_metrics.f1_at_threshold,
    )

    bundle = {
        "pipeline": chosen_pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET,
        "threshold": threshold,
        "metrics": {
            "baseline_val": baseline_val.as_dict(),
            "improved_val": improved_val.as_dict(),
            "final_test": test_metrics.as_dict(),
        },
        "model_name": chosen_name,
        "trained_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "python_version": platform.python_version(),
    }

    out_path = Path(output_path) if output_path else PROJECT_ROOT / "artifacts" / "model.joblib"
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle, out_path)
    logger.info("Saved model artifact to %s (%.2f MB)", out_path, out_path.stat().st_size / 1e6)

    # Also dump a side-by-side metrics JSON for quick reading.
    metrics_json = out_path.with_suffix(".metrics.json")
    metrics_json.write_text(json.dumps(bundle["metrics"], indent=2), encoding="utf-8")
    logger.info("Saved metrics JSON to %s", metrics_json)

    return bundle


def main():
    parser = argparse.ArgumentParser(description="Train credit-scoring model.")
    parser.add_argument("--config", default=None, help="Path to YAML training config.")
    parser.add_argument("--out", default=None, help="Output path for the joblib model artifact.")
    args = parser.parse_args()
    train(config_path=args.config, output_path=args.out)


if __name__ == "__main__":
    main()
