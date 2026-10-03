"""Mask quality control. The visibility metric is only as good as the masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_AREA_PX = 400
ASPECT_RANGE = (0.3, 4.0)
DISAGREEMENT_IOU = 0.15


@dataclass(frozen=True)
class QualityReport:
    accepted: bool
    reason: str


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a > 0, b > 0).sum())
    union = float(np.logical_or(a > 0, b > 0).sum())
    return inter / union if union > 0 else 0.0


def check_mask(mask: np.ndarray) -> QualityReport:
    area = float((mask > 0).sum())
    if area < MIN_AREA_PX:
        return QualityReport(False, f"mask area {area:.0f}px below {MIN_AREA_PX}")
    ys, xs = np.nonzero(mask > 0)
    h = float(ys.max() - ys.min() + 1)
    w = float(xs.max() - xs.min() + 1)
    aspect = w / h if h > 0 else 0.0
    if not ASPECT_RANGE[0] <= aspect <= ASPECT_RANGE[1]:
        return QualityReport(False, f"implausible aspect ratio {aspect:.2f}")
    return QualityReport(True, "ok")


def resolve_disagreement(
    primary: np.ndarray, secondary: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Prefer the higher-fidelity secondary mask when the two models disagree."""
    iou = mask_iou(primary, secondary)
    disagreed = iou < (1.0 - DISAGREEMENT_IOU)
    return (secondary if disagreed else primary), disagreed
