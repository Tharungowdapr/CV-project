"""Calibrator protocol.

``n_params`` and ``inference_multiplier`` are part of the interface, not an
afterthought: the efficiency trade-off figure is a first-class deliverable, and
a method that cannot report its own cost cannot appear in it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from reiduq.core.exceptions import CalibrationNotFittedError


def softmax(logits: np.ndarray, temperature: np.ndarray | float = 1.0) -> np.ndarray:
    """Numerically stable row-wise softmax with per-row temperature.

    (N, k) -> (N, k). Temperature may be scalar or (N, 1).
    """
    t = np.asarray(temperature, dtype=np.float64)
    if t.ndim == 1:
        t = t[:, None]
    scaled = np.asarray(logits, dtype=np.float64) / np.maximum(t, 1e-8)
    scaled -= scaled.max(axis=-1, keepdims=True)
    exp = np.exp(scaled)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-12)


class BaseCalibrator(ABC):
    """Maps top-k similarities to a calibrated probability distribution."""

    name: str = "base"
    requires_features: bool = False

    def __init__(self) -> None:
        self._fitted = False

    @abstractmethod
    def fit(
        self, sims: np.ndarray, correct: np.ndarray, features: np.ndarray | None = None
    ) -> None:
        """sims (N, k) descending; correct (N,) bool: is candidate 0 the right identity."""

    @abstractmethod
    def transform(self, sims: np.ndarray, features: np.ndarray | None = None) -> np.ndarray:
        """(N, k) -> (N, k) calibrated probabilities."""

    def temperatures(self, sims: np.ndarray, features: np.ndarray | None = None) -> np.ndarray:
        """(N,) per-sample temperature. Constant methods return a filled array."""
        return np.ones(len(sims), dtype=np.float64)

    def confidence(self, sims: np.ndarray, features: np.ndarray | None = None) -> np.ndarray:
        return self.transform(sims, features)[:, 0]

    @property
    @abstractmethod
    def n_params(self) -> int: ...

    @property
    def inference_multiplier(self) -> float:
        """Forward passes of the encoder required per query, relative to raw."""
        return 1.0

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise CalibrationNotFittedError(f"{self.name}: fit() must be called before transform()")
