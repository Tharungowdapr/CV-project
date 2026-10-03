"""Coarse part-visibility vector.

A vehicle 70% visible with its distinctive rear hidden is a different problem
from one 70% visible with the rear intact; the calibrator needs to tell them
apart, so visibility is reported per region as well as in aggregate.
"""

from __future__ import annotations

import numpy as np

PARTS = ("front", "rear", "left", "right", "roof", "glass")


def part_visibility(mask: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Return (6,) float32 visible fraction per coarse region.

    The crop is divided into a 3x3 grid; regions map onto the grid cells under
    the usual surveillance viewing geometry. Crude, but stable and cheap - and a
    learned part model would need part annotations this dataset does not have.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    ref = reference if reference is not None else np.ones_like(mask)
    h, w = mask.shape
    hs, ws = max(h // 3, 1), max(w // 3, 1)

    regions = {
        "front": (slice(0, h), slice(0, ws)),
        "rear": (slice(0, h), slice(2 * ws, w)),
        "left": (slice(2 * hs, h), slice(0, w)),
        "right": (slice(0, hs), slice(0, w)),
        "roof": (slice(0, hs), slice(ws, 2 * ws)),
        "glass": (slice(hs, 2 * hs), slice(ws, 2 * ws)),
    }
    out = np.zeros(6, dtype=np.float32)
    for i, name in enumerate(PARTS):
        rs, cs = regions[name]
        denom = float((ref[rs, cs] > 0).sum())
        out[i] = float((mask[rs, cs] > 0).sum()) / denom if denom > 0 else 0.0
    return np.clip(out, 0.0, 1.0)
