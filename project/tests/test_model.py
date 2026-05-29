"""Sanity-проверки артефакта модели."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.service.predictor import Predictor
from src.utils.config import ServiceSettings


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    settings = ServiceSettings()
    path = settings.resolved_model_path()
    if not Path(path).exists():
        pytest.skip(f"Артефакт модели не найден по пути {path}. Сначала запустите `python -m src.models.train`.")
    return Predictor.load(path)


def test_predictor_loads(predictor: Predictor):
    assert predictor.pipeline is not None
    assert len(predictor.feature_columns) > 0
    assert 0.0 < predictor.threshold < 1.0


def test_predict_proba_in_range(predictor: Predictor):
    item = {"AMT_INCOME_TOTAL": 200_000, "AMT_CREDIT": 500_000}
    proba = predictor.predict_proba([item])
    assert len(proba) == 1
    assert 0.0 <= proba[0] <= 1.0


def test_risk_ordering(predictor: Predictor):
    """Рискованный заявитель должен получать более высокий скор, чем явно безопасный."""
    safe = {
        "AMT_INCOME_TOTAL": 300_000,
        "AMT_CREDIT": 400_000,
        "AMT_ANNUITY": 15_000,
        "DAYS_BIRTH": -16_000,
        "DAYS_EMPLOYED": -5_000,
        "EXT_SOURCE_1": 0.80,
        "EXT_SOURCE_2": 0.80,
        "EXT_SOURCE_3": 0.80,
        "NAME_EDUCATION_TYPE": "Higher education",
    }
    risky = {
        "AMT_INCOME_TOTAL": 40_000,
        "AMT_CREDIT": 2_000_000,
        "AMT_ANNUITY": 90_000,
        "DAYS_BIRTH": -8_500,
        "DAYS_EMPLOYED": -90,
        "EXT_SOURCE_1": 0.10,
        "EXT_SOURCE_2": 0.10,
        "EXT_SOURCE_3": 0.10,
        "NAME_EDUCATION_TYPE": "Lower secondary",
    }
    safe_p, risky_p = predictor.predict_proba([safe, risky])
    assert risky_p > safe_p, f"скор рискованного ({risky_p:.3f}) должен быть выше безопасного ({safe_p:.3f})"
