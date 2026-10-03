"""Bootstrap confidence intervals and paired significance tests.

Single-run numbers with no variance estimate are one of the most common reasons
a Q1 submission comes back for revision, so every headline metric goes through
here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    point: float
    lower: float
    upper: float
    n_samples: int


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    confidence: np.ndarray,
    correct: np.ndarray,
    *,
    n_samples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Interval:
    rng = np.random.default_rng(seed)
    n = len(confidence)
    values = np.empty(n_samples)
    for i in range(n_samples):
        idx = rng.integers(0, n, n)
        values[i] = metric_fn(confidence[idx], correct[idx])
    return Interval(
        point=float(metric_fn(confidence, correct)),
        lower=float(np.quantile(values, alpha / 2)),
        upper=float(np.quantile(values, 1 - alpha / 2)),
        n_samples=n_samples,
    )


def paired_bootstrap_test(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    conf_a: np.ndarray,
    conf_b: np.ndarray,
    correct: np.ndarray,
    *,
    n_samples: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Compare two methods on the SAME queries. Returns (mean difference, p-value)."""
    rng = np.random.default_rng(seed)
    n = len(correct)
    observed = metric_fn(conf_a, correct) - metric_fn(conf_b, correct)
    diffs = np.empty(n_samples)
    for i in range(n_samples):
        idx = rng.integers(0, n, n)
        diffs[i] = metric_fn(conf_a[idx], correct[idx]) - metric_fn(conf_b[idx], correct[idx])
    p = float(np.mean(np.abs(diffs - diffs.mean()) >= abs(observed)))
    return float(observed), p
