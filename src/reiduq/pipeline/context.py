"""Pipeline context: the object every stage receives and returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from reiduq.core.config import ExperimentConfig
from reiduq.core.types import Decision, Tracklet


@dataclass
class Context:
    cfg: ExperimentConfig
    run_id: str
    tracklets: list[Tracklet] = field(default_factory=list)
    embeddings: np.ndarray | None = None
    similarities: np.ndarray | None = None
    correct: np.ndarray | None = None
    features: np.ndarray | None = None
    confidence: np.ndarray | None = None
    decisions: list[Decision] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
