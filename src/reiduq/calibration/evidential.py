"""Evidential / Dirichlet calibration - the most dangerous competitor.

Single forward pass like ours, and modern. If it matches the conditional head at
equal cost, the contribution narrows to the protocol plus tracklet aggregation,
and the paper should say so rather than bury it. Evidence is derived from the
shifted similarities; the Dirichlet mean gives the probability and the total
evidence gives an explicit "I don't know" mass.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from reiduq.calibration.base import BaseCalibrator
from reiduq.core.registry import CALIBRATORS


@CALIBRATORS.register("evidential")
class EvidentialCalibrator(BaseCalibrator):
    name = "evidential"

    def __init__(self) -> None:
        super().__init__()
        self.scale: float = 1.0
        self.shift: float = 0.0

    def _alpha(self, sims: np.ndarray, scale: float, shift: float) -> np.ndarray:
        evidence = np.maximum(np.exp(scale * (sims - shift)) - 1.0, 0.0)
        return evidence + 1.0  # Dirichlet concentration

    def _nll(self, params: np.ndarray, sims: np.ndarray, correct: np.ndarray) -> float:
        alpha = self._alpha(sims, params[0], params[1])
        strength = alpha.sum(axis=1, keepdims=True)
        probs = alpha / strength
        p_true = np.where(correct, probs[:, 0], 1.0 - probs[:, 0])
        return float(-np.mean(np.log(np.clip(p_true, 1e-12, 1.0))))

    def fit(self, sims: np.ndarray, correct: np.ndarray, features: np.ndarray | None = None) -> None:
        res = minimize(
            self._nll,
            x0=np.array([1.0, float(np.median(sims))]),
            args=(sims, correct.astype(bool)),
            method="Nelder-Mead",
            options={"maxiter": 500, "xatol": 1e-4},
        )
        self.scale, self.shift = float(res.x[0]), float(res.x[1])
        self._fitted = True

    def transform(self, sims: np.ndarray, features: np.ndarray | None = None) -> np.ndarray:
        self._check_fitted()
        alpha = self._alpha(sims, self.scale, self.shift)
        return alpha / alpha.sum(axis=1, keepdims=True)

    def vacuity(self, sims: np.ndarray) -> np.ndarray:
        """(N,) explicit ignorance mass K / sum(alpha): high means no evidence at all."""
        self._check_fitted()
        alpha = self._alpha(sims, self.scale, self.shift)
        return sims.shape[1] / alpha.sum(axis=1)

    @property
    def n_params(self) -> int:
        return 2
