"""Baseline-модель кредитного скоринга — Logistic Regression на предобработанных признаках."""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.preprocessing import build_preprocessor


def build_baseline_pipeline(
    C: float = 1.0,
    max_iter: int = 1000,
    class_weight: str | None = "balanced",
    random_state: int = 42,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    class_weight=class_weight,
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )
