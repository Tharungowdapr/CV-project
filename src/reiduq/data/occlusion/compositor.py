"""Realistic occlusion compositing with fully replayable recipes.

Every composite is described by a recipe dict (occluder id, position, scale,
feather, jpeg quality, seed) so the entire synthetic dataset can be regenerated
from JSON - which is what makes the occlusion sweep reproducible by a reviewer.

The placement is binary-searched until the achieved visibility lands within a
tolerance of the requested target, so each sample can be assigned to an
occlusion band with a known, not estimated, visibility.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from reiduq.data.occlusion.geometry import Placement, estimate_horizon, is_plausible
from reiduq.data.occlusion.occluder_bank import OccluderPatch

_TOLERANCE = 0.02
_MAX_ITERS = 24


def _paste(
    image: np.ndarray,
    host_mask: np.ndarray,
    patch: OccluderPatch,
    placement: Placement,
    feather_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Alpha-composite the patch and return (image, remaining_visible_host_mask)."""
    import cv2

    oh, ow = patch.size
    new_h, new_w = max(1, int(oh * placement.scale)), max(1, int(ow * placement.scale))
    rgb = cv2.resize(patch.rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    alpha = cv2.resize(patch.mask.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if feather_px > 0:
        ksize = feather_px * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (ksize, ksize), 0)

    H, W = image.shape[:2]
    x0, y0 = max(0, placement.x), max(0, placement.y)
    x1, y1 = min(W, placement.x + new_w), min(H, placement.y + new_h)
    if x1 <= x0 or y1 <= y0:
        return image, host_mask

    sx0, sy0 = x0 - placement.x, y0 - placement.y
    a = alpha[sy0 : sy0 + (y1 - y0), sx0 : sx0 + (x1 - x0)][..., None]
    out = image.astype(np.float32).copy()
    region = out[y0:y1, x0:x1]
    out[y0:y1, x0:x1] = region * (1 - a) + rgb[sy0 : sy0 + (y1 - y0), sx0 : sx0 + (x1 - x0)] * a

    remaining = host_mask.astype(np.float32).copy()
    remaining[y0:y1, x0:x1] *= 1.0 - (a[..., 0] > 0.5)
    return out.astype(np.uint8), (remaining > 0.5).astype(np.uint8)


def _requantise_jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    """Re-encode so the composite carries the host image's compression signature."""
    import cv2

    ok, buf = cv2.imencode(".jpg", image[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:  # pragma: no cover - only on a broken opencv build
        return image
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1]


def composite(
    image: np.ndarray,
    host_mask: np.ndarray,
    host_box: tuple[float, float, float, float],
    patch: OccluderPatch,
    target_visibility: float,
    rng: np.random.Generator,
    *,
    feather_px: int = 2,
    jpeg_quality: int = 90,
    enforce_geometry: bool = True,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Composite `patch` over the host until visibility ~= target_visibility.

    Returns (image, achieved_visibility, recipe).
    """
    if not 0.0 <= target_visibility <= 1.0:
        raise ValueError("target_visibility must lie in [0,1]")
    base_area = float(host_mask.sum())
    if base_area <= 0:
        raise ValueError("host mask is empty")

    x1, y1, x2, y2 = host_box
    horizon = estimate_horizon(image.shape[0])
    lo, hi = 0.05, 4.0
    best: tuple[np.ndarray, float, Placement] | None = None
    rejections: list[str] = []

    for _ in range(_MAX_ITERS):
        scale = (lo + hi) / 2
        px = int(rng.uniform(x1 - 0.2 * (x2 - x1), x2 - 0.2 * (x2 - x1)))
        py = int(rng.uniform(y1, y2))
        placement = Placement(px, py, scale, patch.occluder_class)

        if enforce_geometry:
            ok, reason = is_plausible(placement, host_box, patch.size, horizon)
            if not ok:
                rejections.append(reason)
                hi = scale  # shrink and retry
                continue

        img_out, remaining = _paste(image, host_mask, patch, placement, feather_px)
        achieved = float(remaining.sum()) / base_area
        best = (img_out, achieved, placement)
        if abs(achieved - target_visibility) <= _TOLERANCE:
            break
        if achieved > target_visibility:
            lo = scale  # not enough occlusion -> bigger occluder
        else:
            hi = scale

    if best is None:
        return image, 1.0, {"status": "no_plausible_placement", "rejections": rejections}

    img_out, achieved, placement = best
    img_out = _requantise_jpeg(img_out, jpeg_quality)
    recipe = {
        "status": "ok",
        "patch_id": patch.patch_id,
        "occluder_class": patch.occluder_class,
        "x": placement.x,
        "y": placement.y,
        "scale": round(placement.scale, 4),
        "feather_px": feather_px,
        "jpeg_quality": jpeg_quality,
        "target_visibility": target_visibility,
        "achieved_visibility": round(achieved, 4),
        "rejections": rejections,
    }
    return img_out, achieved, recipe
