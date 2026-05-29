"""Общие фикстуры pytest.

Делает корень проекта импортируемым как `src.*` и предоставляет FastAPI
TestClient на основе обученного артефакта модели.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from src.service.app import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_payload():
    return {
        "AMT_INCOME_TOTAL": 180000,
        "AMT_CREDIT": 500000,
        "AMT_ANNUITY": 25000,
        "AMT_GOODS_PRICE": 480000,
        "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": -3000,
        "EXT_SOURCE_1": 0.45,
        "EXT_SOURCE_2": 0.55,
        "EXT_SOURCE_3": 0.50,
        "CNT_CHILDREN": 1,
        "CNT_FAM_MEMBERS": 3,
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "F",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y",
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "OCCUPATION_TYPE": "Managers",
    }
