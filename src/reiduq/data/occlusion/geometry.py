"""Plausibility rules for synthetic occluder placement.

Reviewers reject synthetic occlusion that could not physically occur. These
checks are what let the paper claim the composites are realistic rather than
merely convenient.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Placement:
    x: int
    y: int
    scale: float
    occluder_class: str


GROUND_BORNE = {"car", "truck", "bus", "motorcycle", "pedestrian", "pole", "barrier"}


def is_plausible(
    placement: Placement,
    host_box: tuple[float, float, float, float],
    occluder_size: tuple[int, int],
    horizon_y: float,
) -> tuple[bool, str]:
    """Return (ok, reason). Reason is logged for every rejection."""
    x1, y1, x2, y2 = host_box
    oh, ow = occluder_size
    oh_s, ow_s = oh * placement.scale, ow * placement.scale
    bottom = placement.y + oh_s

    if placement.occluder_class in GROUND_BORNE and bottom < horizon_y:
        return False, "ground-borne occluder floating above the horizon"
    if placement.x + ow_s < x1 or placement.x > x2:
        return False, "occluder does not intersect the host vehicle"
    if placement.y + oh_s < y1 or placement.y > y2:
        return False, "occluder does not intersect the host vehicle"
    host_h = max(y2 - y1, 1.0)
    if not 0.15 <= oh_s / host_h <= 6.0:
        return False, f"implausible relative scale {oh_s / host_h:.2f}"
    return True, "ok"


def estimate_horizon(image_height: int) -> float:
    """Crude horizon prior for fixed surveillance cameras: upper third boundary.

    Replaced by a per-camera calibrated value when camera metadata is available;
    the prior only has to be good enough to reject sky placements.
    """
    return image_height / 3.0
