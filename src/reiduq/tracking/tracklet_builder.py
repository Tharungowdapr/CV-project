"""Tracklet construction and identity-switch hygiene.

An identity switch pools two vehicles as one and corrupts the aggregated
confidence - it would look exactly like a calibration failure in the results, so
it is split out explicitly and the split rate is reported in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reiduq.core.types import Instance, Tracklet


@dataclass(frozen=True)
class TrackletStats:
    n_built: int
    n_dropped_short: int
    n_split_consistency: int

    @property
    def split_rate(self) -> float:
        total = self.n_built + self.n_split_consistency
        return self.n_split_consistency / total if total else 0.0


def build_tracklets(
    instances_by_track: dict[int, list[Instance]],
    *,
    min_len: int = 5,
    max_gap: int = 15,
) -> tuple[list[Tracklet], TrackletStats]:
    tracklets: list[Tracklet] = []
    dropped = 0
    for track_id, instances in instances_by_track.items():
        ordered = sorted(instances, key=lambda i: i.detection.frame_id)
        current: list[Instance] = []
        for inst in ordered:
            if current and inst.detection.frame_id - current[-1].detection.frame_id > max_gap:
                if len(current) >= min_len:
                    tracklets.append(_make(track_id, current))
                else:
                    dropped += 1
                current = []
            current.append(inst)
        if len(current) >= min_len:
            tracklets.append(_make(track_id, current))
        elif current:
            dropped += 1
    return tracklets, TrackletStats(len(tracklets), dropped, 0)


def _make(track_id: int, instances: list[Instance]) -> Tracklet:
    ids = {i.identity for i in instances if i.identity is not None}
    return Tracklet(
        track_id=track_id,
        camera_id=instances[0].detection.camera_id,
        instances=instances,
        identity=next(iter(ids)) if len(ids) == 1 else None,
    )


def split_on_inconsistency(
    tracklet: Tracklet, embeddings: np.ndarray, threshold: float = 0.55, min_len: int = 5
) -> list[Tracklet]:
    """Cut wherever consecutive-frame cosine similarity drops below the threshold."""
    if len(embeddings) != tracklet.n_frames:
        raise ValueError("one embedding per instance is required")
    if tracklet.n_frames < 2:
        return [tracklet]
    sims = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    cuts = [0, *(np.nonzero(sims < threshold)[0] + 1).tolist(), tracklet.n_frames]
    pieces: list[Tracklet] = []
    for a, b in zip(cuts[:-1], cuts[1:], strict=True):
        if b - a >= min_len:
            pieces.append(_make(tracklet.track_id, tracklet.instances[a:b]))
    return pieces or []
