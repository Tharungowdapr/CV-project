"""MC Dropout: N stochastic forward passes, predictive mean and entropy.

Strong but costs N encoder passes per query. It exists here to anchor the
efficiency argument: the proposed head must land near it at a fraction of the
cost, and that claim needs the expensive baseline actually measured.
"""

from __future__ import annotations

import numpy as np

from reiduq.calibration.base import BaseCalibrator, softmax
from reiduq.core.registry import CALIBRATORS


@CALIBRATORS.register("mc_dropout")
class MCDropout(BaseCalibrator):
    name = "mc_dropout"

    def __init__(self, n_samples: int = 20, temperature: float = 1.0) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.temperature = temperature

    def fit(self, sims: np.ndarray, correct: np.ndarray, features: np.ndarray | None = None) -> None:
        """Only the output temperature is fitted; dropout is a property of the encoder."""
        from reiduq.calibration.global_temperature import GlobalTemperature

        gt = GlobalTemperature()
        gt.fit(sims, correct)
        self.temperature = gt.temperature
        self._fitted = True

    def transform(self, sims: np.ndarray, features: np.ndarray | None = None) -> np.ndarray:
        self._check_fitted()
        return softmax(sims, self.temperature)

    def transform_samples(self, sim_samples: np.ndarray) -> np.ndarray:
        """(S, N, k) stochastic similarity samples -> (N, k) predictive mean."""
        self._check_fitted()
        return np.mean([softmax(s, self.temperature) for s in sim_samples], axis=0)

    @property
    def n_params(self) -> int:
        return 1

    @property
    def inference_multiplier(self) -> float:
        return float(self.n_samples)
