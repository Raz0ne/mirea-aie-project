from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


def test_quality_flags_constant_columns():
    """Тест для проверки эвристики константных колонок."""
    df = pd.DataFrame(
        {
            "constant_col": [1, 1, 1, 1],
            "normal_col": [1, 2, 3, 4],
            "another_constant": ["A", "A", "A", "A"],
        }
    )
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)

    assert flags["has_constant_columns"] is True
    assert "constant_col" in flags["constant_columns"]
    assert "another_constant" in flags["constant_columns"]
    assert "normal_col" not in flags["constant_columns"]


def test_quality_flags_high_cardinality():
    """Тест для проверки эвристики высокой кардинальности категориальных признаков."""
    n_rows = 100
    high_cardinality_col = [f"value_{i}" for i in range(n_rows)]
    df = pd.DataFrame(
        {
            "high_card_col": high_cardinality_col,
            "low_card_col": ["A", "B"] * (n_rows // 2),
            "numeric_col": list(range(n_rows)),
        }
    )
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)

    assert flags["has_high_cardinality_categoricals"] is True
    assert "high_card_col" in flags["high_cardinality_columns"]
    assert "low_card_col" not in flags["high_cardinality_columns"]
    assert "numeric_col" not in flags["high_cardinality_columns"]


def test_quality_flags_many_zero_values():
    """Тест для проверки эвристики большого количества нулевых значений."""
    df = pd.DataFrame(
        {
            "many_zeros": [0, 0, 0, 0, 0, 1, 2, 3, 4, 5],
            "few_zeros": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "no_zeros": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "string_col": ["A", "B", "C"] * 3 + ["D"],
        }
    )
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)

    assert flags["has_many_zero_values"] is True
    assert "many_zeros" in flags["zero_columns"]
    assert "few_zeros" not in flags["zero_columns"]
    assert "no_zeros" not in flags["zero_columns"]
    assert "string_col" not in flags["zero_columns"]


def test_quality_flags_all_new_heuristics():
    """Комплексный тест для всех новых эвристик одновременно."""
    n_rows = 120
    df = pd.DataFrame(
        {
            "constant": [5] * n_rows,
            "high_card": [f"cat_{i}" for i in range(n_rows)],
            "many_zeros": [0] * 70 + [1] * 50,
            "normal": list(range(n_rows)),
        }
    )
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)

    assert flags["has_constant_columns"] is True
    assert flags["has_high_cardinality_categoricals"] is True
    assert flags["has_many_zero_values"] is True

    assert 0.0 <= flags["quality_score"] <= 1.0
    assert flags["quality_score"] < 0.8
