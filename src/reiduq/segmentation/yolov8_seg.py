"""YOLOv8-seg: joint detection and segmentation in one forward pass."""

from __future__ import annotations

from typing import Any

import numpy as np

from reiduq.core.registry import SEGMENTERS
from reiduq.core.types import Detection
from reiduq.segmentation.base import BaseSegmenter


@SEGMENTERS.register("yolov8_seg")
class YOLOv8Segmenter(BaseSegmenter):
    def __init__(self, weights: str = "yolov8m-seg.pt", device: str = "cuda") -> None:
        self.weights, self.device = weights, device
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.weights)
        return self._model

    def segment(self, frame: np.ndarray, detections: list[Detection]) -> list[np.ndarray]:
        import cv2

        results = self.model.predict(frame, device=self.device, verbose=False)
        masks: list[np.ndarray] = []
        full: list[np.ndarray] = []
        for res in results:
            if res.masks is None:
                continue
            for m in res.masks.data.cpu().numpy():
                full.append(
                    cv2.resize(m, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                )
        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            best = np.zeros((max(y2 - y1, 1), max(x2 - x1, 1)), dtype=np.uint8)
            best_area = 0.0
            for m in full:
                sub = (m[y1:y2, x1:x2] > 0.5).astype(np.uint8)
                if sub.size and float(sub.sum()) > best_area:
                    best, best_area = sub, float(sub.sum())
            masks.append(best)
        return masks
