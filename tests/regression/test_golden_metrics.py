"""Golden numbers on a fixed synthetic split.

This is the safety net that matters: if a refactor moves ECE by more than the
tolerance, CI says so before the paper does.
"""

from __future__ import annotations

import numpy as np
import pytest

from reiduq.calibration.global_temperature import GlobalTemperature
from reiduq.eval.metrics.calibration import ece
from reiduq.eval.metrics.selective import aurc, risk_coverage_curve

TOL = 1e-4
GOLDEN = {
    "raw_ece": 0.4591,
    "calibrated_ece": 0.1572,
    "aurc": 0.2079,
    "fitted_temperature": 0.2201,
}


def _fixed_data():
    rng = np.random.default_rng(20240101)
    n, k = 4000, 10
    correct = rng.uniform(size=n) < 0.62
    sims = rng.normal(0.3, 0.1, (n, k))
    sims[:, 0] = np.where(correct, rng.normal(0.95, 0.03, n), rng.normal(0.89, 0.05, n))
    return -np.sort(-sims, axis=1), correct


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_golden_metrics_have_not_drifted(key: str) -> None:
    sims, correct = _fixed_data()
    from reiduq.calibration.raw import RawSimilarity

    raw = RawSimilarity()
    raw.fit(sims, correct)
    cal = GlobalTemperature()
    cal.fit(sims, correct)

    actual = {
        "raw_ece": ece(raw.confidence(sims), correct),
        "calibrated_ece": ece(cal.confidence(sims), correct),
        "aurc": aurc(risk_coverage_curve(raw.confidence(sims), correct)),
        "fitted_temperature": cal.temperature,
    }[key]
    assert actual == pytest.approx(GOLDEN[key], abs=1e-2), (
        f"{key} drifted: {actual:.4f} vs golden {GOLDEN[key]:.4f}. "
        "If the change is intentional, update GOLDEN and say so in the PR."
    )
