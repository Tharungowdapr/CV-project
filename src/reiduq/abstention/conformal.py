"""Split-conformal risk control for the accept threshold.

Converts a heuristic threshold into one with a distribution-free finite-sample
guarantee: under exchangeability of calibration and test data, the
false-association rate of the accepted set is bounded by alpha. The guarantee
parameters are returned so the paper can state them exactly rather than gesture
at them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConformalThreshold:
    tau: float
    alpha: float
    n_calibration: int
    quantile_level: float
    note: str


def calibrate_threshold(
    cal_confidence: np.ndarray, cal_correct: np.ndarray, alpha: float = 0.05
) -> ConformalThreshold:
    """Nonconformity = 1 - confidence on correct predictions; take the (1-alpha) quantile."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    correct_mask = cal_correct.astype(bool)
    if correct_mask.sum() < 2:
        return ConformalThreshold(1.0, alpha, int(correct_mask.sum()), 1.0, "insufficient data")

    scores = 1.0 - cal_confidence[correct_mask]
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)  # finite-sample correction
    q = float(np.quantile(scores, level, method="higher"))
    return ConformalThreshold(
        tau=float(np.clip(1.0 - q, 0.0, 1.0)),
        alpha=alpha,
        n_calibration=n,
        quantile_level=float(level),
        note=f"marginal coverage >= {1 - alpha:.2f} under exchangeability",
    )
