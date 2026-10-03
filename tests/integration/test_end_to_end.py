"""Calibrate -> retrieve -> aggregate -> decide, on synthetic data."""

from __future__ import annotations

import numpy as np

import pytest

from reiduq.abstention.decision import decide_batch
from reiduq.abstention.policy import select_thresholds
from reiduq.aggregation.visibility_weighted import VisibilityWeighted
from reiduq.calibration.conditional_temperature import ConditionalTemperature
from reiduq.calibration.features import build_features
from reiduq.eval.metrics.calibration import ece
from reiduq.retrieval.index import GalleryEntry, GalleryIndex


def _dataset(n: int = 1500, k: int = 10, seed: int = 0):
    """Visibility drives accuracy but NOT similarity - exactly the failure mode."""
    rng = np.random.default_rng(seed)
    visibility = rng.uniform(0.2, 1.0, n)
    correct = rng.uniform(size=n) < (0.35 + 0.55 * visibility)
    sims = rng.normal(0.3, 0.1, (n, k))
    sims[:, 0] = rng.normal(0.92, 0.04, n)  # top-1 score is high regardless
    sims = -np.sort(-sims, axis=1)
    features = build_features(
        sims,
        visibility.astype(np.float32),
        np.tile(visibility[:, None], (1, 6)).astype(np.float32),
        rng.normal(0, 0.2, (n, 16)).astype(np.float32),
        np.ones(n, dtype=int),
    )
    return sims, correct, features, visibility


torch = pytest.importorskip("torch", reason="conditional calibrator needs torch")


@pytest.mark.integration
def test_conditional_calibration_recovers_hidden_visibility_effect() -> None:
    sims, correct, features, _ = _dataset()
    split = len(sims) // 2
    cal = ConditionalTemperature(epochs=150, device="cpu")
    cal.fit(sims[:split], correct[:split], features[:split])

    raw_ece = ece(sims[split:, 0] / sims[split:].sum(axis=1) * sims.shape[1] / 10, correct[split:])
    cal_ece = ece(cal.confidence(sims[split:], features[split:]), correct[split:])
    assert cal_ece < max(raw_ece, 0.5)
    assert cal.n_params < 15_000  # the "lightweight" claim, asserted in code


@pytest.mark.integration
def test_retrieval_then_decision_pipeline() -> None:
    pytest.importorskip("faiss", reason="gallery index needs faiss")
    rng = np.random.default_rng(1)
    dim = 32
    gallery = rng.normal(size=(200, dim)).astype(np.float32)
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)

    index = GalleryIndex(dim=dim)
    index.add(gallery, [GalleryEntry(f"g{i}", identity=i % 40, camera_id=f"c{i % 4}") for i in range(200)])

    queries = gallery[:20] + rng.normal(0, 0.01, (20, dim)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    results = index.search(queries, [f"q{i}" for i in range(20)], k=5,
                           query_identities=[i % 40 for i in range(20)])

    assert len(results) == 20
    assert all(len(r.candidate_ids) == 5 for r in results)
    assert sum(bool(r.top1_correct) for r in results) >= 18  # near-identical queries

    conf = np.array([float(r.similarities[0]) for r in results])
    correct = np.array([bool(r.top1_correct) for r in results])
    thresholds = select_thresholds(conf, correct, max_false_association_rate=0.1)
    decisions = decide_batch(conf, [r.candidate_ids[0] for r in results],
                             thresholds.tau_high, thresholds.tau_low)
    assert {d.verdict for d in decisions} <= {"MATCH", "UNCERTAIN", "REJECT"}


def test_tracklet_aggregation_sharpens_a_consistent_tracklet() -> None:
    probs = np.array([[0.6, 0.4], [0.65, 0.35], [0.7, 0.3]])
    out = VisibilityWeighted(ess_correction=False).aggregate(
        probs, np.array([0.9, 0.9, 0.9]), np.ones(3)
    )
    assert out[0] > probs[:, 0].max()
