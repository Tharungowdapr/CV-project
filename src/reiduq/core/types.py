"""Frozen data contracts shared by every layer.

Defined once and never changed casually: the whole pipeline is typed against
them. Array shapes are documented inline and validated where cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Verdict = Literal["MATCH", "UNCERTAIN", "REJECT"]
Band = Literal["O0", "O1", "O2", "O3", "O4"]


@dataclass(frozen=True)
class Detection:
    """One detected vehicle in one frame."""

    frame_id: int
    camera_id: str
    bbox: tuple[float, float, float, float]  # xyxy, pixels
    score: float
    cls: str


@dataclass(frozen=True)
class Instance:
    """A detection enriched with its segmentation mask and measured visibility."""

    detection: Detection
    mask: np.ndarray  # (H, W) uint8 in {0,1}, crop-local
    visibility: float  # v in [0, 1]
    part_visibility: np.ndarray  # (6,) float32: front, rear, left, right, roof, glass
    identity: int | None = None  # ground truth, evaluation only

    def __post_init__(self) -> None:
        if not 0.0 <= self.visibility <= 1.0:
            raise ValueError(f"visibility must lie in [0,1], got {self.visibility}")
        if self.part_visibility.shape != (6,):
            raise ValueError(f"part_visibility must be (6,), got {self.part_visibility.shape}")


@dataclass(frozen=True)
class Tracklet:
    """Maximal same-track sequence within a single camera."""

    track_id: int
    camera_id: str
    instances: list[Instance]
    identity: int | None = None

    @property
    def n_frames(self) -> int:
        return len(self.instances)

    @property
    def mean_visibility(self) -> float:
        return float(np.mean([i.visibility for i in self.instances]))


@dataclass(frozen=True)
class Embedding:
    """L2-normalised appearance embedding, shape (D,) float32."""

    vector: np.ndarray
    instance: Instance
    backbone: str


@dataclass(frozen=True)
class RetrievalResult:
    """Top-k gallery candidates for one query, similarities descending."""

    query_id: str
    candidate_ids: list[str]
    similarities: np.ndarray  # (k,) float32
    query_identity: int | None = None
    candidate_identities: list[int | None] = field(default_factory=list)

    @property
    def top1_correct(self) -> bool | None:
        if self.query_identity is None or not self.candidate_identities:
            return None
        return self.candidate_identities[0] == self.query_identity


@dataclass(frozen=True)
class CalibratedResult:
    """Retrieval result plus a calibrated distribution over its candidates."""

    retrieval: RetrievalResult
    temperature: float
    probabilities: np.ndarray  # (k,) float32, sums to 1

    @property
    def confidence(self) -> float:
        return float(self.probabilities[0])


@dataclass(frozen=True)
class Decision:
    """Final three-way output handed to an operator."""

    verdict: Verdict
    matched_id: str | None
    confidence: float
    reason: str
