"""Property-based invariants for the selective-prediction metrics."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from reiduq.eval.metrics.selective import (
    coverage_at_risk,
    far_at_coverage,
    risk_coverage_curve,
)


@settings(max_examples=50, deadline=None)
@given(st.integers(min_value=20, max_value=500), st.integers(min_value=0, max_value=10_000))
def test_coverage_is_monotone_and_bounded(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    conf, correct = rng.uniform(size=n), rng.integers(0, 2, n).astype(bool)
    curve = risk_coverage_curve(conf, correct)
    assert np.all(np.diff(curve.coverage) > 0)
    assert curve.coverage[-1] == 1.0
    assert np.all((curve.risk >= 0) & (curve.risk <= 1))


@settings(max_examples=50, deadline=None)
@given(st.integers(min_value=20, max_value=300), st.integers(min_value=0, max_value=10_000))
def test_full_coverage_risk_equals_overall_error(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    conf, correct = rng.uniform(size=n), rng.integers(0, 2, n).astype(bool)
    assert far_at_coverage(conf, correct, 1.0) == float(1.0 - correct.mean())


@settings(max_examples=30, deadline=None)
@given(st.integers(min_value=50, max_value=300), st.integers(min_value=0, max_value=10_000))
def test_a_looser_risk_budget_never_reduces_coverage(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    conf, correct = rng.uniform(size=n), rng.integers(0, 2, n).astype(bool)
    assert coverage_at_risk(conf, correct, 0.5) >= coverage_at_risk(conf, correct, 0.1)
