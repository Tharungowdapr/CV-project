"""YOLOv8 detector wrapper.

Weights are loaded from a local path only; a filename is never fetched from a
remote URL at runtime, because a downloaded checkpoint is executable input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from reiduq.core.registry import DETECTORS
from reiduq.core.types import Detection
from reiduq.detection.base import VEHICLE_CLASSES, BaseDetector


@DETECTORS.register("yolov8")
class YOLOv8Detector(BaseDetector):
    def __init__(
        self,
        weights: str = "yolov8m.pt",
        score_threshold: float = 0.35,
        classes: tuple[str, ...] = VEHICLE_CLASSES,
        device: str = "cuda",
    ) -> None:
        super().__init__(score_threshold, classes)
        self.device = device
        self._model: Any | None = None
        self.weights = Path(weights)

    @property
    def model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.weights))
        return self._model

    def detect(self, frame: np.ndarray, frame_id: int, camera_id: str) -> list[Detection]:
        results = self.model.predict(
            frame, conf=self.score_threshold, device=self.device, verbose=False
        )
        out: list[Detection] = []
        for res in results:
            names = res.names
            for box in res.boxes:
                cls_name = names[int(box.cls)]
                if cls_name not in self.classes:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                out.append(
                    Detection(
                        frame_id=frame_id,
                        camera_id=camera_id,
                        bbox=(x1, y1, x2, y2),
                        score=float(box.conf),
                        cls=cls_name,
                    )
                )
        return out
