"""Visibility estimation and banding."""

from __future__ import annotations

import numpy as np
import pytest

from reiduq.visibility.bands import to_band
from reiduq.visibility.estimator import from_synthetic, from_tracklet
from reiduq.visibility.parts import part_visibility


def test_synthetic_visibility_is_exact() -> None:
    original = np.ones((100, 100), dtype=np.uint8)
    visible = original.copy()
    visible[:40] = 0  # hide the top 40%
    assert from_synthetic(visible, original).value == pytest.approx(0.6, abs=1e-6)


def test_fully_visible_is_one() -> None:
    m = np.ones((50, 50), dtype=np.uint8)
    assert from_synthetic(m, m).value == 1.0


@pytest.mark.parametrize(
    ("v", "expected"), [(1.0, "O0"), (0.97, "O0"), (0.85, "O1"), (0.7, "O2"), (0.5, "O3"), (0.1, "O4")]
)
def test_band_assignment(v: float, expected: str) -> None:
    assert to_band(v) == expected


def test_band_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="outside"):
        to_band(1.5)


def test_part_visibility_shape_and_range() -> None:
    mask = np.ones((60, 30), dtype=np.uint8)
    p = part_visibility(mask)
    assert p.shape == (6,)
    assert ((p >= 0) & (p <= 1)).all()


def test_tracklet_estimator_uses_the_peak_frame() -> None:
    masks = [np.ones((10, 10), np.uint8), np.ones((10, 10), np.uint8)]
    masks[1][:5] = 0  # second frame half occluded
    est = from_tracklet(masks, [10.0, 10.0], index=1)
    assert est.value == pytest.approx(0.5, abs=0.05)
