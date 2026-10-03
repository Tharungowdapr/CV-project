"""Meta SAM 2.1 (Segment Anything 2.1) wrapper for vehicle segmentation."""

from __future__ import annotations

from typing import Any

import numpy as np

from reiduq.core.logging import get_logger
from reiduq.core.registry import SEGMENTERS
from reiduq.core.types import Detection
from reiduq.segmentation.base import BaseSegmenter

log = get_logger(__name__)


@SEGMENTERS.register("sam2")
class SAM2Segmenter(BaseSegmenter):
    """Meta Segment Anything 2.1 (SAM 2.1) segmenter for precise vehicle masks."""

    def __init__(
        self,
        checkpoint: str = "sam2.1_hiera_large.pt",
        device: str = "cpu",
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self._predictor: Any | None = None

    @property
    def predictor(self) -> Any:
        if self._predictor is None:
            try:
                from sam2.build_sam import build_sam2_camera_predictor
                self._predictor = build_sam2_camera_predictor(self.checkpoint, device=self.device)
                log.info("sam2.loaded", checkpoint=self.checkpoint)
            except Exception as e:
                log.warning("sam2.fallback", note=f"sam2 package not found ({e}), using bounding box binary mask fallback")
                self._predictor = "fallback"
        return self._predictor

    def segment(self, frame: np.ndarray, detections: list[Detection]) -> list[np.ndarray]:
        masks: list[np.ndarray] = []
        h_frame, w_frame = frame.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            crop_h = max(1, y2 - y1)
            crop_w = max(1, x2 - x1)

            if self.predictor == "fallback":
                # Create a crisp binary mask for the vehicle bounding box area
                mask = np.ones((crop_h, crop_w), dtype=np.uint8)
            else:
                # SAM 2.1 prompt-based segmentation using box prompt
                box = np.array([x1, y1, x2, y2])
                res_masks, scores, _ = self._predictor.predict(
                    point_coords=None, point_labels=None, box=box[None, :], multimask_output=False
                )
                full_mask = res_masks[0]
                mask = full_mask[y1:y2, x1:x2].astype(np.uint8)

            masks.append(mask)

        return masks
