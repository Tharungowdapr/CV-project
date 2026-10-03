"""Tracklet fusion interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseAggregator(ABC):
    name: str = "base"

    def __init__(self, ess_correction: bool = True) -> None:
        self.ess_correction = ess_correction

    @abstractmethod
    def aggregate(
        self,
        probabilities: np.ndarray,  # (n, k) per-frame calibrated distributions
        visibility: np.ndarray,  # (n,)
        temperatures: np.ndarray,  # (n,)
        embeddings: np.ndarray | None = None,  # (n, D) for the ESS discount
    ) -> np.ndarray:
        """-> (k,) aggregated distribution for the whole tracklet."""

    @staticmethod
    def _renormalise(log_p: np.ndarray) -> np.ndarray:
        log_p = log_p - log_p.max()
        p = np.exp(log_p)
        return p / max(p.sum(), 1e-12)
