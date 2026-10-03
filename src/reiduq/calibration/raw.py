"""Uncalibrated similarity softmax - the status quo this project criticises."""

from __future__ import annotations

import numpy as np

from reiduq.calibration.base import BaseCalibrator, softmax
from reiduq.core.registry import CALIBRATORS


@CALIBRATORS.register("raw")
class RawSimilarity(BaseCalibrator):
    name = "raw"

    def fit(self, sims: np.ndarray, correct: np.ndarray, features: np.ndarray | None = None) -> None:
        self._fitted = True  # nothing to learn; kept for interface symmetry

    def transform(self, sims: np.ndarray, features: np.ndarray | None = None) -> np.ndarray:
        return softmax(sims, 1.0)

    @property
    def n_params(self) -> int:
        return 0
