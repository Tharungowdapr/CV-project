"""Visibility-weighted log-pooling: frames with more visible vehicle count more."""

from __future__ import annotations

import numpy as np

from reiduq.aggregation.base import BaseAggregator
from reiduq.aggregation.ess import effective_sample_size
from reiduq.core.registry import AGGREGATORS


@AGGREGATORS.register("visibility_weighted")
class VisibilityWeighted(BaseAggregator):
    name = "visibility_weighted"

    def aggregate(
        self,
        probabilities: np.ndarray,
        visibility: np.ndarray,
        temperatures: np.ndarray,
        embeddings: np.ndarray | None = None,
    ) -> np.ndarray:
        if len(probabilities) == 0:
            raise ValueError("cannot aggregate an empty tracklet")
        # Weights are scaled to MEAN 1, not sum 1: log-pooling accumulates evidence
        # across frames, so the total weight must stay proportional to the frame
        # count. Normalising to sum 1 would collapse this to a geometric mean and
        # a consistent tracklet could never become more confident than one frame.
        w = np.clip(visibility, 1e-3, 1.0)
        w = w / max(w.mean(), 1e-12)
        log_p = (w[:, None] * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(axis=0)
        if self.ess_correction and embeddings is not None:
            n_eff = effective_sample_size(embeddings)
            log_p *= n_eff / len(probabilities)
        return self._renormalise(log_p)
