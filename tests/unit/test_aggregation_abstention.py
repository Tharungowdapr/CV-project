"""Tracklet fusion, ESS discounting and the decision layer."""

from __future__ import annotations

import numpy as np

from reiduq.abstention.conformal import calibrate_threshold
from reiduq.abstention.decision import decide
from reiduq.abstention.policy import select_thresholds
from reiduq.aggregation.best_frame import BestFrame
from reiduq.aggregation.ess import effective_sample_size
from reiduq.aggregation.visibility_weighted import VisibilityWeighted


def test_identical_frames_contribute_one_effective_sample() -> None:
    e = np.tile(np.array([[1.0, 0.0, 0.0]]), (10, 1))
    assert effective_sample_size(e) < 1.5


def test_orthogonal_frames_are_fully_independent() -> None:
    e = np.eye(5)
    assert effective_sample_size(e) > 4.0


def test_visibility_weighting_favours_the_clear_frame() -> None:
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    out = VisibilityWeighted(ess_correction=False).aggregate(
        probs, np.array([0.95, 0.10]), np.array([1.0, 1.0])
    )
    assert out[0] > out[1]


def test_best_frame_picks_the_most_visible() -> None:
    probs = np.array([[0.4, 0.6], [0.95, 0.05]])
    out = BestFrame().aggregate(probs, np.array([0.3, 0.99]), np.array([1.0, 1.0]))
    assert np.allclose(out, probs[1])


def test_decision_layer_three_way_split() -> None:
    assert decide(0.95, "a", 0.9, 0.2).verdict == "MATCH"
    assert decide(0.50, "a", 0.9, 0.2).verdict == "UNCERTAIN"
    assert decide(0.05, "a", 0.9, 0.2).verdict == "REJECT"
    assert decide(0.05, "a", 0.9, 0.2).matched_id is None


def test_threshold_selection_respects_the_risk_budget() -> None:
    rng = np.random.default_rng(0)
    conf = rng.uniform(size=5000)
    correct = rng.uniform(size=5000) < conf
    t = select_thresholds(conf, correct, max_false_association_rate=0.05)
    assert t.achieved_risk <= 0.05 or t.method == "infeasible"


def test_conformal_threshold_reports_its_guarantee() -> None:
    rng = np.random.default_rng(1)
    conf = rng.uniform(0.5, 1.0, 1000)
    correct = rng.uniform(size=1000) < conf
    ct = calibrate_threshold(conf, correct, alpha=0.1)
    assert 0.0 <= ct.tau <= 1.0
    assert "exchangeability" in ct.note
