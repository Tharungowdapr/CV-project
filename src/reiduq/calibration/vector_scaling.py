"""Vector scaling: per-rank affine transform of the top-k similarities.

Included specifically so a reviewer cannot attribute the conditional model's
gain merely to having more parameters than a single scalar.
"""

from __future__ import annotations

import numpy as np

from reiduq.calibration.base import BaseCalibrator, softmax
from reiduq.core.exceptions import CalibrationNotFittedError
from reiduq.core.registry import CALIBRATORS


@CALIBRATORS.register("vector_scaling")
class VectorScaling(BaseCalibrator):
    name = "vector_scaling"

    def __init__(self, lr: float = 0.05, epochs: int = 300) -> None:
        super().__init__()
        self.lr, self.epochs = lr, epochs
        self.w: np.ndarray | None = None
        self.b: np.ndarray | None = None

    def fit(self, sims: np.ndarray, correct: np.ndarray, features: np.ndarray | None = None) -> None:
        k = sims.shape[1]
        w = np.ones(k, dtype=np.float64)
        b = np.zeros(k, dtype=np.float64)
        y = correct.astype(np.float64)
        for _ in range(self.epochs):
            probs = softmax(sims * w + b)
            grad_logits = probs.copy()
            grad_logits[:, 0] -= y  # one-hot on the top-1 slot
            w -= self.lr * (grad_logits * sims).mean(axis=0)
            b -= self.lr * grad_logits.mean(axis=0)
        self.w, self.b = w, b
        self._fitted = True

    def transform(self, sims: np.ndarray, features: np.ndarray | None = None) -> np.ndarray:
        self._check_fitted()
        if self.w is None or self.b is None:  # pragma: no cover - fit() always sets both or raises
            raise CalibrationNotFittedError("vector_scaling: weights are unset after fit()")
        return softmax(sims * self.w + self.b)

    @property
    def n_params(self) -> int:
        return 0 if self.w is None else 2 * len(self.w)
