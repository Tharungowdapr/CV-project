"""Global temperature scaling (Guo et al., 2017) - the standard to beat.

One scalar fitted by minimising NLL on the calibration split with the encoder
frozen. The hypothesis of this project is that a single scalar is insufficient
under shift, so this class is both a baseline and the thing being argued
against; its per-band optimal value is itself a headline figure.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from reiduq.calibration.base import BaseCalibrator, softmax
from reiduq.core.registry import CALIBRATORS


@CALIBRATORS.register("global_temperature")
class GlobalTemperature(BaseCalibrator):
    name = "global_temperature"

    def __init__(self, bounds: tuple[float, float] = (0.05, 20.0)) -> None:
        super().__init__()
        self.bounds = bounds
        self.temperature: float = 1.0

    def _nll(self, t: float, sims: np.ndarray, correct: np.ndarray) -> float:
        probs = softmax(sims, t)
        p_true = np.where(correct, probs[:, 0], 1.0 - probs[:, 0])
        return float(-np.mean(np.log(np.clip(p_true, 1e-12, 1.0))))

    def fit(self, sims: np.ndarray, correct: np.ndarray, features: np.ndarray | None = None) -> None:
        res = minimize_scalar(
            self._nll, bounds=self.bounds, args=(sims, correct.astype(bool)), method="bounded"
        )
        self.temperature = float(res.x)
        self._fitted = True

    def transform(self, sims: np.ndarray, features: np.ndarray | None = None) -> np.ndarray:
        self._check_fitted()
        return softmax(sims, self.temperature)

    def temperatures(self, sims: np.ndarray, features: np.ndarray | None = None) -> np.ndarray:
        self._check_fitted()
        return np.full(len(sims), self.temperature, dtype=np.float64)

    @property
    def n_params(self) -> int:
        return 1
