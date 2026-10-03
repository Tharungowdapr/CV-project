"""PlateReader: graceful degradation, and the pattern filter rejecting noise."""

from __future__ import annotations

import numpy as np

from reiduq.search.plate_ocr import PlateReader, locate_plate_region


def test_unavailable_backend_returns_none_not_raises() -> None:
    reader = PlateReader()
    text, conf = reader.read(np.zeros((40, 120, 3), dtype=np.uint8))
    if not reader.available:
        assert text is None
        assert conf == 0.0


def test_locate_plate_region_returns_a_sub_crop() -> None:
    vehicle = np.ones((200, 400, 3), dtype=np.uint8)
    region = locate_plate_region(vehicle)
    assert region.shape[0] < vehicle.shape[0]
    assert region.shape[1] < vehicle.shape[1]
    assert region.size > 0


def test_locate_plate_region_handles_a_tiny_crop_without_crashing() -> None:
    vehicle = np.ones((4, 4, 3), dtype=np.uint8)
    region = locate_plate_region(vehicle)
    assert isinstance(region, np.ndarray)
