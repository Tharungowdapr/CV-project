"""Selective-prediction metrics: the deployment-facing numbers.

Risk-coverage is the standard for abstention, and false-association rate at a
fixed coverage is the figure an operator can actually act on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RCCurve:
    coverage: np.ndarray
    risk: np.ndarray


def risk_coverage_curve(confidence: np.ndarray, correct: np.ndarray) -> RCCurve:
    order = np.argsort(-np.asarray(confidence, dtype=np.float64))
    corr = np.asarray(correct, dtype=np.float64)[order]
    n = len(corr)
    accepted = np.arange(1, n + 1)
    risk = 1.0 - np.cumsum(corr) / accepted
    return RCCurve(accepted / n, risk)


_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy >= 2.0 renamed it


def aurc(curve: RCCurve) -> float:
    """Area under the risk-coverage curve. Lower is better."""
    return float(_trapezoid(curve.risk, curve.coverage))


def excess_aurc(curve: RCCurve, accuracy: float) -> float:
    """AURC minus the value an oracle-ranked perfect selector would achieve."""
    err = 1.0 - accuracy
    if err <= 0:
        return float(aurc(curve))
    optimal = err + (1 - err) * np.log(1 - err) if err < 1 else 0.0
    return float(aurc(curve) - optimal)


def far_at_coverage(confidence: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    """False-association rate among the top `coverage` fraction by confidence."""
    if not 0 < coverage <= 1:
        raise ValueError("coverage must lie in (0, 1]")
    order = np.argsort(-np.asarray(confidence, dtype=np.float64))
    corr = np.asarray(correct, dtype=np.float64)[order]
    k = max(1, int(round(coverage * len(corr))))
    return float(1.0 - corr[:k].mean())


def coverage_at_risk(confidence: np.ndarray, correct: np.ndarray, max_risk: float) -> float:
    """Largest coverage whose risk stays within the budget."""
    curve = risk_coverage_curve(confidence, correct)
    ok = np.nonzero(curve.risk <= max_risk)[0]
    return float(curve.coverage[ok[-1]]) if len(ok) else 0.0


def auroc_correct_vs_incorrect(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Does the confidence separate right from wrong at all? Rank-based, ties handled."""
    conf = np.asarray(confidence, dtype=np.float64)
    corr = np.asarray(correct, dtype=bool)
    n_pos, n_neg = int(corr.sum()), int((~corr).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(conf)
    ranks = np.empty(len(conf), dtype=np.float64)
    ranks[order] = np.arange(1, len(conf) + 1)
    return float((ranks[corr].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
