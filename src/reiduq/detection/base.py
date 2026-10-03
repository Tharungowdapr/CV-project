"""Detector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from reiduq.core.types import Detection

VEHICLE_CLASSES = ("car", "truck", "bus", "motorcycle")


class BaseDetector(ABC):
    def __init__(self, score_threshold: float = 0.35, classes: tuple[str, ...] = VEHICLE_CLASSES):
        self.score_threshold = score_threshold
        self.classes = classes

    @abstractmethod
    def detect(self, frame: np.ndarray, frame_id: int, camera_id: str) -> list[Detection]:
        """(H, W, 3) uint8 RGB -> vehicle detections above threshold."""
