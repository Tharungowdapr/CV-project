"""Model loading and single-query inference for the service.

Shares the identical calibration code path as the research pipeline; a second
implementation would drift and the demo would stop matching the paper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reiduq.abstention.decision import decide
from reiduq.calibration.base import BaseCalibrator
from reiduq.calibration.conditional_temperature import ConditionalTemperature
from reiduq.calibration.features import build_features
from reiduq.calibration.global_temperature import GlobalTemperature
from reiduq.calibration.camera_descriptor import CameraDescriptor
from reiduq.core.logging import get_logger
from reiduq.core.types import Decision

log = get_logger(__name__)


class Predictor:
    def __init__(
        self,
        calibrator: BaseCalibrator,
        camera_descriptor: CameraDescriptor,
        tau_high: float,
        tau_low: float,
        version: str = "dev",
    ) -> None:
        self.calibrator = calibrator
        self.camera_descriptor = camera_descriptor
        self.tau_high, self.tau_low = tau_high, tau_low
        self.version = version

    @classmethod
    def from_registry(cls, registry_path: Path | str = "./models") -> Predictor:
        """Load the promoted checkpoint, or fall back to an identity calibrator.

        The service must start even with no checkpoint present, otherwise a
        health check cannot distinguish "not deployed" from "crashed".
        """
        root = Path(registry_path)
        ckpt = root / "calibrator.pt"
        thresholds = root / "thresholds.npz"
        calibrator: BaseCalibrator
        if ckpt.exists():
            cond = ConditionalTemperature()
            cond.load(ckpt)
            calibrator = cond
        else:
            log.warning("predictor.no_checkpoint", path=str(ckpt))
            calibrator = GlobalTemperature()
            calibrator.fit(np.array([[0.9, 0.5]]), np.array([True]))
        tau_high, tau_low = 0.9, 0.2
        if thresholds.exists():
            blob = np.load(thresholds)
            tau_high, tau_low = float(blob["tau_high"]), float(blob["tau_low"])
        return cls(calibrator, CameraDescriptor(), tau_high, tau_low, version=ckpt.stem)

    def predict(
        self,
        similarities: list[float],
        candidate_ids: list[str],
        *,
        visibility: float = 1.0,
        part_visibility: list[float] | None = None,
        camera_stats: list[float] | None = None,
        n_frames: int = 1,
    ) -> tuple[Decision, float]:
        sims = np.asarray([similarities], dtype=np.float32)
        parts = np.asarray([part_visibility or [1.0] * 6], dtype=np.float32)
        cam_raw = np.asarray([camera_stats or [0.0] * 9], dtype=np.float32)
        features = build_features(
            sims,
            np.array([visibility], dtype=np.float32),
            parts,
            self.camera_descriptor(cam_raw),
            np.array([n_frames]),
        )
        needs = getattr(self.calibrator, "requires_features", False)
        probs = self.calibrator.transform(sims, features if needs else None)
        temps = self.calibrator.temperatures(sims, features if needs else None)
        confidence = float(probs[0, 0])
        decision = decide(
            confidence,
            candidate_ids[0] if candidate_ids else None,
            self.tau_high,
            self.tau_low,
            visibility=visibility,
        )
        return decision, float(temps[0])
