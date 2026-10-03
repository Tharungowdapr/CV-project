"""Segmenter interface and the mask-gating operation.

Mask gating fills the non-vehicle region with the channel mean rather than
black: a black fill introduces a hard artificial edge the encoder responds to.
Both variants plus the ungated baseline are kept so the ablation can measure it
instead of assuming it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from reiduq.core.types import Detection

IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)


class BaseSegmenter(ABC):
    @abstractmethod
    def segment(self, frame: np.ndarray, detections: list[Detection]) -> list[np.ndarray]:
        """Return one crop-local (h, w) uint8 {0,1} mask per detection."""


def gate_crop(
    crop: np.ndarray, mask: np.ndarray, mode: Literal["none", "mean_fill", "black_fill"]
) -> np.ndarray:
    """Suppress background pixels outside the vehicle mask."""
    if mode == "none":
        return crop
    if crop.shape[:2] != mask.shape[:2]:
        raise ValueError(f"crop {crop.shape[:2]} and mask {mask.shape[:2]} disagree")
    alpha = (mask > 0).astype(np.float32)[..., None]
    fill = np.zeros(3, dtype=np.float32) if mode == "black_fill" else IMAGENET_MEAN
    return (crop.astype(np.float32) * alpha + (1 - alpha) * fill).astype(np.uint8)
