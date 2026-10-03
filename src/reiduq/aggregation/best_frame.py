"""Best-frame selection: take only the most visible frame.

Embarrassingly simple and surprisingly strong. The aggregation contribution only
stands if it beats this, so it is a required comparison, not an optional one.
"""

from __future__ import annotations

import numpy as np

from reiduq.aggregation.base import BaseAggregator
from reiduq.core.registry import AGGREGATORS


@AGGREGATORS.register("best_frame")
class BestFrame(BaseAggregator):
    name = "best_frame"

    def aggregate(
        self,
        probabilities: np.ndarray,
        visibility: np.ndarray,
        temperatures: np.ndarray,
        embeddings: np.ndarray | None = None,
    ) -> np.ndarray:
        return probabilities[int(np.argmax(visibility))]
