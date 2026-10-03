"""Inverse-temperature weighting: the calibrator's own estimate of evidence strength.

Self-consistent - it introduces no hyper-parameter the visibility rule needs.
"""

from __future__ import annotations

import numpy as np

from reiduq.aggregation.base import BaseAggregator
from reiduq.aggregation.ess import effective_sample_size
from reiduq.core.registry import AGGREGATORS


@AGGREGATORS.register("inverse_temperature")
class InverseTemperature(BaseAggregator):
    name = "inverse_temperature"

    def aggregate(
        self,
        probabilities: np.ndarray,
        visibility: np.ndarray,
        temperatures: np.ndarray,
        embeddings: np.ndarray | None = None,
    ) -> np.ndarray:
        # Scaled to mean 1 for the same reason as visibility weighting: pooling
        # accumulates evidence rather than averaging it.
        w = 1.0 / np.clip(temperatures, 1e-3, None)
        w = w / max(w.mean(), 1e-12)
        log_p = (w[:, None] * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(axis=0)
        if self.ess_correction and embeddings is not None:
            log_p *= effective_sample_size(embeddings) / len(probabilities)
        return self._renormalise(log_p)
