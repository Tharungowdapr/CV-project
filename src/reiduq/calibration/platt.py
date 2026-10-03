"""Platt scaling: logistic regression on the top-1 similarity and margin."""

from __future__ import annotations

import numpy as np

from reiduq.calibration.base import BaseCalibrator
from reiduq.core.exceptions import CalibrationNotFittedError
from reiduq.core.registry import CALIBRATORS


@CALIBRATORS.register("platt")
class PlattScaling(BaseCalibrator):
    name = "platt"

    def __init__(self) -> None:
        super().__init__()
        self._model: object | None = None

    @staticmethod
    def _design(sims: np.ndarray) -> np.ndarray:
        margin = sims[:, 0] - (sims[:, 1] if sims.shape[1] > 1 else 0.0)
        return np.stack([sims[:, 0], margin], axis=1)

    def fit(self, sims: np.ndarray, correct: np.ndarray, features: np.ndarray | None = None) -> None:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000)
        model.fit(self._design(sims), correct.astype(int))
        self._model = model
        self._fitted = True

    def transform(self, sims: np.ndarray, features: np.ndarray | None = None) -> np.ndarray:
        self._check_fitted()
        if self._model is None:  # pragma: no cover - fit() always sets it or raises
            raise CalibrationNotFittedError("platt: model is unset after fit()")
        p1 = self._model.predict_proba(self._design(sims))[:, 1]  # type: ignore[attr-defined]
        rest = sims[:, 1:]
        if rest.size == 0:
            return p1[:, None]
        remaining = np.clip(1.0 - p1, 1e-12, 1.0)[:, None]
        share = rest / np.maximum(rest.sum(axis=1, keepdims=True), 1e-12)
        return np.hstack([p1[:, None], remaining * share])

    @property
    def n_params(self) -> int:
        return 3
