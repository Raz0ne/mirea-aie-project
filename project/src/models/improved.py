"""Improved credit-scoring model — LightGBM.

LightGBM handles missing values natively, so we don't impute. We still one-hot
encode categoricals (cheap subset, ~9 columns) to keep the pipeline pure-sklearn
and avoid bundling lightgbm-specific categorical encoders into the artifact.
"""
from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data.dataset import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=20),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_improved_pipeline(
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    max_depth: int = -1,
    min_child_samples: int = 50,
    reg_alpha: float = 0.1,
    reg_lambda: float = 0.1,
    class_weight: str | None = "balanced",
    n_jobs: int = -1,
    random_state: int = 42,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor()),
            (
                "classifier",
                LGBMClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    num_leaves=num_leaves,
                    max_depth=max_depth,
                    min_child_samples=min_child_samples,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    class_weight=class_weight,
                    n_jobs=n_jobs,
                    random_state=random_state,
                    verbose=-1,
                ),
            ),
        ]
    )
