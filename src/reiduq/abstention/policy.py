"""Threshold selection under an operator constraint.

Thresholds are never hand-tuned: they are chosen on the calibration split to
satisfy a stated constraint ("false-association rate <= 1%") and then applied
unchanged to test. Tuning them on test would be the same leak as calibrating on
test, one stage later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Thresholds:
    tau_high: float
    tau_low: float
    achieved_coverage: float
    achieved_risk: float
    method: str


def select_thresholds(
    confidence: np.ndarray,
    correct: np.ndarray,
    *,
    max_false_association_rate: float = 0.01,
    reject_below: float = 0.20,
) -> Thresholds:
    """Lowest tau_high whose accepted set meets the risk constraint (max coverage)."""
    order = np.argsort(-confidence)
    conf_sorted = confidence[order]
    correct_sorted = correct[order].astype(bool)

    cum_correct = np.cumsum(correct_sorted)
    n_accepted = np.arange(1, len(conf_sorted) + 1)
    risk = 1.0 - cum_correct / n_accepted

    feasible = np.nonzero(risk <= max_false_association_rate)[0]
    if len(feasible) == 0:
        # No operating point meets the constraint: abstain on everything rather
        # than silently returning a threshold that violates the guarantee.
        return Thresholds(1.0, reject_below, 0.0, 0.0, "infeasible")

    idx = int(feasible[-1])
    return Thresholds(
        tau_high=float(conf_sorted[idx]),
        tau_low=float(reject_below),
        achieved_coverage=float((idx + 1) / len(conf_sorted)),
        achieved_risk=float(risk[idx]),
        method="empirical_risk_constraint",
    )
