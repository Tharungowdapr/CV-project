"""Deep Ensemble: M independently seeded encoders, averaged probabilities.

Usually the quality ceiling for uncertainty estimation, at M times the training
and inference cost. Reported so the trade-off statement in the paper has a real
upper bound rather than an assumed one.
"""

from __future__ import annotations

import numpy as np

from reiduq.calibration.base import BaseCalibrator, softmax
from reiduq.calibration.global_temperature import GlobalTemperature
from reiduq.core.registry import CALIBRATORS


@CALIBRATORS.register("deep_ensemble")
class DeepEnsemble(BaseCalibrator):
    name = "deep_ensemble"

    def __init__(self, ensemble_size: int = 5) -> None:
        super().__init__()
        self.ensemble_size = ensemble_size
        self.members: list[GlobalTemperature] = []

    def fit_members(self, sims_per_member: list[np.ndarray], correct: np.ndarray) -> None:
        """One temperature per member; each member sees the same calibration split."""
        self.members = []
        for sims in sims_per_member:
            gt = GlobalTemperature()
            gt.fit(sims, correct)
            self.members.append(gt)
        self._fitted = True

    def fit(self, sims: np.ndarray, correct: np.ndarray, features: np.ndarray | None = None) -> None:
        self.fit_members([sims] * self.ensemble_size, correct)

    def transform_members(self, sims_per_member: list[np.ndarray]) -> np.ndarray:
        self._check_fitted()
        if len(sims_per_member) != len(self.members):
            raise ValueError("number of similarity sets must match the fitted ensemble size")
        return np.mean(
            [softmax(s, m.temperature) for s, m in zip(sims_per_member, self.members, strict=True)],
            axis=0,
        )

    def transform(self, sims: np.ndarray, features: np.ndarray | None = None) -> np.ndarray:
        return self.transform_members([sims] * len(self.members))

    @property
    def n_params(self) -> int:
        return len(self.members)

    @property
    def inference_multiplier(self) -> float:
        return float(self.ensemble_size)
