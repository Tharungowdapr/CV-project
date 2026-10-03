"""Occlusion bands O0-O4.

Bands rather than a continuous regression because per-band ECE is what makes
the degradation curve legible in a figure; the continuous value is retained for
conditioning the calibrator.
"""

from __future__ import annotations

from reiduq.core.types import Band

BAND_EDGES: list[tuple[Band, float, float, str]] = [
    ("O0", 0.95, 1.01, "Clean"),
    ("O1", 0.80, 0.95, "Light"),
    ("O2", 0.60, 0.80, "Moderate"),
    ("O3", 0.40, 0.60, "Heavy"),
    ("O4", -0.01, 0.40, "Severe"),
]


def to_band(visibility: float) -> Band:
    for name, lo, hi, _ in BAND_EDGES:
        if lo <= visibility < hi:
            return name
    raise ValueError(f"visibility outside [0,1]: {visibility}")


def band_label(band: Band) -> str:
    return next(label for name, _, _, label in BAND_EDGES if name == band)
