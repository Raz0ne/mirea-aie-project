"""Tests for the synthetic data generator and dataset loader."""
from __future__ import annotations

from src.data.dataset import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from src.data.synthetic import generate


def test_synthetic_generator_schema():
    df = generate(n_rows=2_000, random_state=0)
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]:
        assert col in df.columns, f"missing column: {col}"
    # Target is binary {0,1}.
    assert set(df[TARGET].unique()).issubset({0, 1})


def test_synthetic_positive_rate_realistic():
    """Positive rate should sit in a plausible credit-scoring range (1%-25%)."""
    df = generate(n_rows=20_000, random_state=0)
    rate = df[TARGET].mean()
    assert 0.01 <= rate <= 0.25, f"unrealistic positive rate: {rate:.3f}"
