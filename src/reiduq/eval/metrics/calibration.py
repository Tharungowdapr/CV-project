"""Calibration metrics.

Default binning is equal-mass (quantile), not equal-width. Re-ID confidence
piles up near 1.0, so equal-width bins leave most bins nearly empty and the
resulting ECE is dominated by sampling noise - a subtle way to report a number
that is not reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReliabilityCurve:
    bin_confidence: np.ndarray
    bin_accuracy: np.ndarray
    bin_count: np.ndarray


@dataclass(frozen=True)
class BrierDecomposition:
    reliability: float
    resolution: float
    uncertainty: float


def _bin_edges(conf: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(conf, qs)
    edges[0], edges[-1] = 0.0, 1.0
    return np.unique(edges)


def reliability_curve(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15, strategy: str = "quantile"
) -> ReliabilityCurve:
    conf = np.asarray(confidence, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    if conf.shape != corr.shape:
        raise ValueError("confidence and correct must have the same shape")
    edges = _bin_edges(conf, n_bins, strategy)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, len(edges) - 2)
    n = len(edges) - 1
    bc, ba, cnt = np.zeros(n), np.zeros(n), np.zeros(n)
    for b in range(n):
        m = idx == b
        cnt[b] = m.sum()
        if cnt[b] > 0:
            bc[b], ba[b] = conf[m].mean(), corr[m].mean()
    return ReliabilityCurve(bc, ba, cnt)


def ece(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15, strategy: str = "quantile"
) -> float:
    """Expected calibration error: mass-weighted |confidence - accuracy|."""
    curve = reliability_curve(confidence, correct, n_bins, strategy)
    total = curve.bin_count.sum()
    if total == 0:
        return 0.0
    return float((curve.bin_count / total * np.abs(curve.bin_confidence - curve.bin_accuracy)).sum())


def adaptive_ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    return ece(confidence, correct, n_bins, strategy="quantile")


def mce(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15, strategy: str = "quantile"
) -> float:
    """Worst-bin calibration error - the operationally relevant one."""
    curve = reliability_curve(confidence, correct, n_bins, strategy)
    mask = curve.bin_count > 0
    if not mask.any():
        return 0.0
    return float(np.abs(curve.bin_confidence[mask] - curve.bin_accuracy[mask]).max())


def brier(confidence: np.ndarray, correct: np.ndarray) -> tuple[float, BrierDecomposition]:
    conf = np.asarray(confidence, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    score = float(np.mean((conf - corr) ** 2))
    curve = reliability_curve(conf, corr, 15, "quantile")
    total = max(curve.bin_count.sum(), 1.0)
    base = corr.mean()
    w = curve.bin_count / total
    rel = float((w * (curve.bin_confidence - curve.bin_accuracy) ** 2).sum())
    res = float((w * (curve.bin_accuracy - base) ** 2).sum())
    return score, BrierDecomposition(rel, res, float(base * (1 - base)))


def nll(confidence: np.ndarray, correct: np.ndarray) -> float:
    conf = np.clip(np.asarray(confidence, dtype=np.float64), 1e-12, 1 - 1e-12)
    corr = np.asarray(correct, dtype=np.float64)
    return float(-np.mean(corr * np.log(conf) + (1 - corr) * np.log(1 - conf)))
