"""Генератор синтетического датасета, похожего на Home Credit.

Используется как fallback, когда реальный CSV из Kaggle недоступен.
Схема повторяет `application_train.csv` из соревнования Home Credit Default Risk:
те же названия колонок, типы данных и примерно те же диапазоны значений.
Целевая переменная TARGET генерируется из логистической модели над небольшим
набором признаков (EXT_SOURCE_*, DAYS_BIRTH, отношение стажа, credit/income),
плюс шум — так что любая разумная модель уловит реальный сигнал, но в датасете
нет настоящих персональных данных.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


CONTRACT_TYPES = ["Cash loans", "Revolving loans"]
GENDERS = ["F", "M", "XNA"]
INCOME_TYPES = [
    "Working",
    "Commercial associate",
    "Pensioner",
    "State servant",
    "Unemployed",
    "Student",
]
EDUCATION_TYPES = [
    "Secondary / secondary special",
    "Higher education",
    "Incomplete higher",
    "Lower secondary",
    "Academic degree",
]
FAMILY_STATUS = ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"]
HOUSING_TYPES = [
    "House / apartment",
    "With parents",
    "Municipal apartment",
    "Rented apartment",
    "Office apartment",
    "Co-op apartment",
]
OCCUPATION_TYPES = [
    "Laborers",
    "Sales staff",
    "Core staff",
    "Managers",
    "Drivers",
    "High skill tech staff",
    "Accountants",
    "Medicine staff",
    "Security staff",
    "Cooking staff",
    "Cleaning staff",
    "Private service staff",
    "Low-skill Laborers",
    "Waiters/barmen staff",
    "Secretaries",
    "Realty agents",
    "HR staff",
    "IT staff",
]


def _sample_choice(rng: np.random.Generator, options, size, p=None):
    return rng.choice(options, size=size, p=p)


def generate(n_rows: int = 50_000, random_state: int = 42) -> pd.DataFrame:
    """Сгенерировать синтетический датасет, похожий на Home Credit.

    Колонки совпадают с теми, что используются в модели, плюс несколько типичных
    дополнительных. Примерный баланс классов — ~8% положительных, как в реальном датасете.
    """
    rng = np.random.default_rng(random_state)
    n = int(n_rows)

    df = pd.DataFrame(index=np.arange(n))
    df["SK_ID_CURR"] = 100000 + np.arange(n)

    df["NAME_CONTRACT_TYPE"] = _sample_choice(rng, CONTRACT_TYPES, n, p=[0.9, 0.1])
    df["CODE_GENDER"] = _sample_choice(rng, GENDERS, n, p=[0.66, 0.339, 0.001])
    df["FLAG_OWN_CAR"] = _sample_choice(rng, ["N", "Y"], n, p=[0.66, 0.34])
    df["FLAG_OWN_REALTY"] = _sample_choice(rng, ["Y", "N"], n, p=[0.69, 0.31])

    df["CNT_CHILDREN"] = rng.choice(
        [0, 1, 2, 3, 4], n, p=[0.70, 0.20, 0.08, 0.015, 0.005]
    )

    # Доход — лог-нормальное распределение с обрезкой по границам.
    income = rng.lognormal(mean=12.0, sigma=0.55, size=n)
    df["AMT_INCOME_TOTAL"] = np.clip(income, 25_000, 5_000_000).round(-2)

    # Сумма кредита — коррелирует с доходом.
    credit_mult = rng.lognormal(mean=1.2, sigma=0.5, size=n)
    df["AMT_CREDIT"] = np.clip(df["AMT_INCOME_TOTAL"] * credit_mult, 45_000, 4_000_000).round(-2)

    # Цена товара — чуть ниже суммы кредита.
    df["AMT_GOODS_PRICE"] = (df["AMT_CREDIT"] * rng.uniform(0.85, 1.0, n)).round(-2)

    # Аннуитет ~ кредит / срок, срок 12..60 месяцев.
    term = rng.integers(12, 60, size=n)
    df["AMT_ANNUITY"] = (df["AMT_CREDIT"] / term * rng.uniform(1.05, 1.25, n)).round(-1)

    df["NAME_INCOME_TYPE"] = _sample_choice(
        rng, INCOME_TYPES, n, p=[0.52, 0.23, 0.18, 0.06, 0.005, 0.005]
    )
    df["NAME_EDUCATION_TYPE"] = _sample_choice(
        rng, EDUCATION_TYPES, n, p=[0.71, 0.24, 0.034, 0.015, 0.001]
    )
    df["NAME_FAMILY_STATUS"] = _sample_choice(
        rng, FAMILY_STATUS, n, p=[0.64, 0.15, 0.10, 0.06, 0.05]
    )
    df["NAME_HOUSING_TYPE"] = _sample_choice(
        rng, HOUSING_TYPES, n, p=[0.89, 0.05, 0.03, 0.02, 0.005, 0.005]
    )

    # DAYS_* — отрицательные целые (кол-во дней до подачи заявки).
    df["DAYS_BIRTH"] = -rng.integers(21 * 365, 70 * 365, size=n)
    # Большинство клиентов работают; у пенсионеров — большой положительный sentinel (365243), как в реальных данных.
    employed = rng.random(n) > 0.18
    days_employed = -rng.integers(30, 40 * 365, size=n)
    df["DAYS_EMPLOYED"] = np.where(employed, days_employed, 365243)
    df["DAYS_REGISTRATION"] = -rng.integers(30, 30 * 365, size=n).astype(float)
    df["DAYS_ID_PUBLISH"] = -rng.integers(30, 20 * 365, size=n)

    df["CNT_FAM_MEMBERS"] = (df["CNT_CHILDREN"] + rng.choice([1, 2], n, p=[0.3, 0.7])).astype(float)

    df["REGION_POPULATION_RELATIVE"] = rng.uniform(0.0003, 0.073, size=n).round(6)

    occ = pd.Series(_sample_choice(rng, OCCUPATION_TYPES, n), dtype=object)
    occ_mask = rng.random(n) > 0.30  # ~30% пропусков, как в реальных данных
    occ[~occ_mask] = np.nan
    df["OCCUPATION_TYPE"] = occ.values

    # Внешние скоры — главный сигнал. Сильная отрицательная корреляция с дефолтом.
    ext1 = rng.beta(2, 3, size=n)
    ext2 = rng.beta(2.2, 2.8, size=n)
    ext3 = rng.beta(2.5, 2.5, size=n)
    # Добавляем MAR-подобные пропуски, похожие на реальные.
    df["EXT_SOURCE_1"] = np.where(rng.random(n) > 0.56, ext1, np.nan)
    df["EXT_SOURCE_2"] = np.where(rng.random(n) > 0.005, ext2, np.nan)
    df["EXT_SOURCE_3"] = np.where(rng.random(n) > 0.20, ext3, np.nan)

    # Целевая переменная — логистическая модель над ключевыми факторами риска.
    age_years = -df["DAYS_BIRTH"] / 365.0
    emp_years = np.where(df["DAYS_EMPLOYED"] == 365243, 0.0, -df["DAYS_EMPLOYED"] / 365.0)
    credit_to_income = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].clip(lower=1.0)
    annuity_to_income = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].clip(lower=1.0)

    e1 = df["EXT_SOURCE_1"].fillna(ext1.mean()).to_numpy()
    e2 = df["EXT_SOURCE_2"].fillna(ext2.mean()).to_numpy()
    e3 = df["EXT_SOURCE_3"].fillna(ext3.mean()).to_numpy()

    logit = (
        2.6  # intercept подобран так, чтобы positive rate ~ 8% (как в реальном Home Credit).
        - 3.4 * e1
        - 3.4 * e2
        - 3.4 * e3
        - 0.03 * age_years.to_numpy()
        - 0.04 * emp_years
        + 0.35 * np.log1p(credit_to_income.to_numpy())
        + 0.40 * annuity_to_income.to_numpy()
    )
    # Эффект уровня образования.
    edu_bonus = df["NAME_EDUCATION_TYPE"].map(
        {
            "Higher education": -0.4,
            "Academic degree": -0.5,
            "Incomplete higher": -0.1,
            "Secondary / secondary special": 0.0,
            "Lower secondary": 0.3,
        }
    ).fillna(0.0).to_numpy()
    logit = logit + edu_bonus
    # Случайный шум.
    logit += rng.normal(0, 0.6, n)

    proba = 1.0 / (1.0 + np.exp(-logit))
    df["TARGET"] = (rng.random(n) < proba).astype(int)

    return df


def write_synthetic_csv(path, n_rows: int = 50_000, random_state: int = 42) -> str:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = generate(n_rows=n_rows, random_state=random_state)
    df.to_csv(p, index=False)
    return str(p)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Сгенерировать синтетический датасет Home Credit.")
    parser.add_argument("--out", default="data/raw/application_train.csv")
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    path = write_synthetic_csv(args.out, n_rows=args.rows, random_state=args.seed)
    print(f"Синтетический датасет сохранён в {path}")
