"""Find naturally occluded samples so results do not rest on synthetics alone.

A detection is treated as really occluded when another instance mask overlaps
it and sits nearer the camera (larger box bottom edge = closer on a ground
plane under a fixed surveillance camera).
"""

from __future__ import annotations

import numpy as np

from reiduq.core.types import Detection


def mask_overlap_ratio(host: np.ndarray, other: np.ndarray) -> float:
    inter = float(np.logical_and(host > 0, other > 0).sum())
    area = float((host > 0).sum())
    return inter / area if area > 0 else 0.0


def is_in_front(host: Detection, other: Detection) -> bool:
    """Larger bottom-edge y means nearer the camera on a ground plane."""
    return other.bbox[3] > host.bbox[3]


def mine(
    detections: list[Detection], masks: list[np.ndarray], min_occlusion: float = 0.05
) -> list[tuple[int, float]]:
    """Return (index, estimated_occluded_fraction) for naturally occluded hosts."""
    out: list[tuple[int, float]] = []
    for i, (det, mask) in enumerate(zip(detections, masks, strict=True)):
        occluded = 0.0
        for j, (other_det, other_mask) in enumerate(zip(detections, masks, strict=True)):
            if i == j or not is_in_front(det, other_det):
                continue
            occluded += mask_overlap_ratio(mask, other_mask)
        occluded = min(occluded, 1.0)
        if occluded >= min_occlusion:
            out.append((i, occluded))
    return out
