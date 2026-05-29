"""Метрики оценки качества для кредитного скоринга."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)


@dataclass
class ScoringMetrics:
    roc_auc: float
    pr_auc: float
    f1_at_threshold: float
    ks_statistic: float
    log_loss: float
    brier: float
    threshold: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "roc_auc": float(self.roc_auc),
            "pr_auc": float(self.pr_auc),
            "f1_at_threshold": float(self.f1_at_threshold),
            "ks_statistic": float(self.ks_statistic),
            "log_loss": float(self.log_loss),
            "brier": float(self.brier),
            "threshold": float(self.threshold),
        }


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Статистика Колмогорова-Смирнова — стандартная метрика в скоринге.

    KS = max |F1(s) - F0(s)| по скорам s, где F1 — CDF скоров для положительного
    класса, F0 — для отрицательного. Чем больше — тем лучше разделение классов.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    thresholds = np.sort(np.unique(y_score))
    cdf_pos = np.searchsorted(np.sort(pos), thresholds, side="right") / len(pos)
    cdf_neg = np.searchsorted(np.sort(neg), thresholds, side="right") / len(neg)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def evaluate(y_true, y_score, threshold: float = 0.5) -> ScoringMetrics:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    return ScoringMetrics(
        roc_auc=roc_auc_score(y_true, y_score),
        pr_auc=average_precision_score(y_true, y_score),
        f1_at_threshold=f1_score(y_true, y_pred, zero_division=0),
        ks_statistic=ks_statistic(y_true, y_score),
        log_loss=log_loss(y_true, np.clip(y_score, 1e-7, 1 - 1e-7)),
        brier=brier_score_loss(y_true, y_score),
        threshold=threshold,
    )
