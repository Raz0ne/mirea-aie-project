"""Dataset loading and splitting for credit-scoring project."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import TrainConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


NUMERIC_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "REGION_POPULATION_RELATIVE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]

CATEGORICAL_FEATURES = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "TARGET"


def load_dataset(cfg: TrainConfig) -> pd.DataFrame:
    """Load the raw dataset. Falls back to synthetic data if the file is missing
    and `use_synthetic_if_missing` is enabled in the config.
    """
    raw_path = Path(cfg.data.raw_path)
    if not raw_path.is_absolute():
        # Resolve relative to project root.
        from src.utils.config import PROJECT_ROOT

        raw_path = PROJECT_ROOT / raw_path

    if raw_path.exists():
        logger.info("Loading dataset from %s", raw_path)
        df = pd.read_csv(raw_path)
    elif cfg.data.use_synthetic_if_missing:
        logger.warning(
            "Dataset not found at %s; generating synthetic Home Credit-like data (%d rows).",
            raw_path,
            cfg.data.synthetic_n_rows,
        )
        from src.data.synthetic import write_synthetic_csv

        write_synthetic_csv(
            raw_path,
            n_rows=cfg.data.synthetic_n_rows,
            random_state=cfg.data.random_state,
        )
        df = pd.read_csv(raw_path)
    else:
        raise FileNotFoundError(
            f"Dataset not found at {raw_path}. Set data.use_synthetic_if_missing=true "
            f"or place application_train.csv there."
        )

    _validate_columns(df)
    df = _clean(df)
    logger.info("Loaded %d rows; positive rate = %.4f", len(df), df[TARGET].mean())
    return df


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in FEATURE_COLUMNS + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    # Real Home Credit uses 365243 as a sentinel for "no employment" days. Replace with NaN.
    df = df.copy()
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    # CODE_GENDER has rare 'XNA' values.
    if "CODE_GENDER" in df.columns:
        df["CODE_GENDER"] = df["CODE_GENDER"].replace("XNA", np.nan)
    return df


def split_dataset(
    df: pd.DataFrame, cfg: TrainConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified split into train / val / test."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].astype(int)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=cfg.data.test_size,
        stratify=y,
        random_state=cfg.data.random_state,
    )
    val_relative = cfg.data.val_size / (1.0 - cfg.data.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_relative,
        stratify=y_trainval,
        random_state=cfg.data.random_state,
    )
    logger.info(
        "Split: train=%d, val=%d, test=%d (positive rates: %.3f / %.3f / %.3f)",
        len(X_train),
        len(X_val),
        len(X_test),
        y_train.mean(),
        y_val.mean(),
        y_test.mean(),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
