"""Every registered calibrator satisfies the protocol and improves on raw."""

from __future__ import annotations

import numpy as np
import pytest

from reiduq.calibration.evidential import EvidentialCalibrator
from reiduq.calibration.global_temperature import GlobalTemperature
from reiduq.calibration.platt import PlattScaling
from reiduq.calibration.raw import RawSimilarity
from reiduq.calibration.vector_scaling import VectorScaling
from reiduq.core.exceptions import CalibrationNotFittedError
from reiduq.eval.metrics.calibration import ece


def _overconfident_data(n: int = 3000, k: int = 10, seed: int = 0):
    """Similarities that look far more decisive than the accuracy warrants."""
    rng = np.random.default_rng(seed)
    correct = rng.uniform(size=n) < 0.6
    sims = rng.normal(0.3, 0.1, (n, k))
    sims[:, 0] = np.where(correct, rng.normal(0.95, 0.03, n), rng.normal(0.88, 0.05, n))
    return -np.sort(-sims, axis=1), correct


@pytest.mark.parametrize(
    "factory", [GlobalTemperature, PlattScaling, VectorScaling, EvidentialCalibrator]
)
def test_calibration_beats_raw_similarity(factory) -> None:
    sims, correct = _overconfident_data()
    raw = RawSimilarity()
    raw.fit(sims, correct)
    baseline = ece(raw.confidence(sims), correct)

    cal = factory()
    cal.fit(sims, correct)
    assert ece(cal.confidence(sims), correct) < baseline


def test_transform_before_fit_raises() -> None:
    with pytest.raises(CalibrationNotFittedError):
        GlobalTemperature().transform(np.zeros((2, 3)))


def test_probabilities_are_a_distribution() -> None:
    sims, correct = _overconfident_data(500)
    cal = GlobalTemperature()
    cal.fit(sims, correct)
    probs = cal.transform(sims)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0).all()


def test_temperature_stays_within_bounds() -> None:
    sims, correct = _overconfident_data(500)
    cal = GlobalTemperature(bounds=(0.1, 5.0))
    cal.fit(sims, correct)
    assert 0.1 <= cal.temperature <= 5.0


def test_cost_metadata_is_reported() -> None:
    from reiduq.calibration.deep_ensemble import DeepEnsemble
    from reiduq.calibration.mc_dropout import MCDropout

    assert MCDropout(n_samples=20).inference_multiplier == 20.0
    assert DeepEnsemble(ensemble_size=5).inference_multiplier == 5.0
    assert RawSimilarity().inference_multiplier == 1.0
