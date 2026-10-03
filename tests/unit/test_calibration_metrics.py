"""Calibration metric correctness, including hand-computable cases."""

from __future__ import annotations

import numpy as np

from reiduq.eval.metrics.calibration import brier, ece, mce, nll, reliability_curve


def test_perfect_calibration_gives_zero_ece() -> None:
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.0, 1.0, 20000)
    correct = rng.uniform(size=20000) < conf  # correctness matches stated confidence
    assert ece(conf, correct, n_bins=10) < 0.02


def test_maximal_overconfidence_is_detected() -> None:
    conf = np.ones(1000)
    correct = np.zeros(1000, dtype=bool)  # always sure, always wrong
    assert ece(conf, correct) > 0.95
    assert mce(conf, correct) > 0.95


def test_ece_is_bounded() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        conf = rng.uniform(size=500)
        correct = rng.integers(0, 2, 500).astype(bool)
        assert 0.0 <= ece(conf, correct) <= 1.0


def test_ece_is_permutation_invariant() -> None:
    rng = np.random.default_rng(2)
    conf, correct = rng.uniform(size=400), rng.integers(0, 2, 400).astype(bool)
    order = rng.permutation(400)
    assert ece(conf, correct) == ece(conf[order], correct[order])


def test_brier_decomposition_components_are_non_negative() -> None:
    rng = np.random.default_rng(3)
    conf, correct = rng.uniform(size=1000), rng.integers(0, 2, 1000).astype(bool)
    score, decomp = brier(conf, correct)
    assert 0.0 <= score <= 1.0
    assert decomp.reliability >= 0 and decomp.resolution >= 0


def test_nll_penalises_confident_errors_more_than_hedged_ones() -> None:
    correct = np.zeros(100, dtype=bool)
    assert nll(np.full(100, 0.99), correct) > nll(np.full(100, 0.55), correct)


def test_reliability_bins_hold_every_sample() -> None:
    rng = np.random.default_rng(4)
    conf, correct = rng.uniform(size=997), rng.integers(0, 2, 997).astype(bool)
    curve = reliability_curve(conf, correct, n_bins=15)
    assert curve.bin_count.sum() == 997


def test_top_k_entropy_handles_a_single_candidate_without_warning() -> None:
    import warnings

    from reiduq.calibration.features import top_k_entropy

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # promote the RuntimeWarning to a failure if it recurs
        result = top_k_entropy(np.array([[0.9], [0.5]]))
    assert (result == 0.0).all()
