"""Predictor — обёртка над обученным бандлом модели для FastAPI-сервиса."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import joblib
import numpy as np
import pandas as pd

from src.data.dataset import NUMERIC_FEATURES

from src.utils.logging import get_logger

logger = get_logger(__name__)


class Predictor:
    """Загружает обученный бандл модели и предоставляет batched predict_proba."""

    def __init__(self, bundle: dict):
        self.pipeline = bundle["pipeline"]
        self.feature_columns: List[str] = bundle["feature_columns"]
        self.threshold: float = float(bundle.get("threshold", 0.5))
        self.model_name: str = bundle.get("model_name", "unknown")
        self.trained_at: str | None = bundle.get("trained_at")
        self.metrics = bundle.get("metrics", {})

    @classmethod
    def load(cls, path: str | Path) -> "Predictor":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Артефакт модели не найден по пути {path}. Сначала запустите `python -m src.models.train`."
            )
        logger.info("Загружаем бандл модели из %s", path)
        bundle = joblib.load(path)
        return cls(bundle)

    def _to_frame(self, items: Iterable[dict]) -> pd.DataFrame:
        df = pd.DataFrame(list(items))
        # Гарантируем, что все ожидаемые колонки признаков существуют и имеют нужный dtype.
        # Для числовых нужен именно numpy.nan — pd.NA ломает импутеры sklearn.
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = np.nan if col in NUMERIC_FEATURES else None
        df = df[self.feature_columns].copy()
        for col in NUMERIC_FEATURES:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def predict_proba(self, items: Iterable[dict]) -> List[float]:
        X = self._to_frame(items)
        proba = self.pipeline.predict_proba(X)[:, 1]
        return [float(p) for p in proba]

    def decide(self, proba: float, threshold: float | None = None) -> str:
        t = self.threshold if threshold is None else threshold
        return "reject" if proba >= t else "approve"
