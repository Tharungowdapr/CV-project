"""Conditioning-feature construction for the conditional calibrator.

Every element earns its place - the ablation removes each in turn:

    v            measured visibility; less evidence should widen the posterior
    p_parts (6)  which regions are visible, not just how much
    c_cam  (16)  camera-domain descriptor, label-free
    s1           the raw top-1 similarity being corrected
    margin       s1 - s2; a thin margin is ambiguous regardless of s1
    entropy      flatness of the top-k profile: a gallery full of look-alikes
    n_frames     tracklet evidence available downstream
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = (
    ["visibility"]
    + [f"part_{i}" for i in range(6)]
    + [f"cam_{i}" for i in range(16)]
    + ["sim_top1", "margin", "entropy", "n_frames"]
)
FEATURE_DIM = len(FEATURE_NAMES)  # 30


def top_k_entropy(sims: np.ndarray) -> np.ndarray:
    """Normalised entropy of the similarity profile, (N, k) -> (N,) in [0, 1].

    With a single candidate (k=1) there is nothing to be uncertain about
    among candidates, so entropy is defined as 0 rather than 0/log(1).
    """
    if sims.shape[1] <= 1:
        return np.zeros(len(sims), dtype=np.float64)
    shifted = sims - sims.min(axis=1, keepdims=True)
    total = np.maximum(shifted.sum(axis=1, keepdims=True), 1e-12)
    p = shifted / total
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1)
    return ent / np.log(sims.shape[1])


def build_features(
    sims: np.ndarray,
    visibility: np.ndarray,
    part_visibility: np.ndarray,
    camera_descriptor: np.ndarray,
    n_frames: np.ndarray,
    *,
    use_visibility: bool = True,
    use_parts: bool = True,
    use_camera: bool = True,
) -> np.ndarray:
    """Assemble (N, 30). Disabled groups are zeroed so shapes stay constant.

    Zeroing rather than dropping columns keeps a checkpoint loadable across
    ablations - an ablation that changes the input width silently invalidates
    every saved normalisation statistic.
    """
    n = len(sims)
    vis = visibility.reshape(n, 1) if use_visibility else np.zeros((n, 1))
    parts = part_visibility.reshape(n, 6) if use_parts else np.zeros((n, 6))
    cam = camera_descriptor.reshape(n, 16) if use_camera else np.zeros((n, 16))
    s1 = sims[:, :1]
    margin = (sims[:, 0] - (sims[:, 1] if sims.shape[1] > 1 else 0.0)).reshape(n, 1)
    ent = top_k_entropy(sims).reshape(n, 1)
    frames = np.log1p(n_frames.reshape(n, 1).astype(np.float64))
    out = np.hstack([vis, parts, cam, s1, margin, ent, frames]).astype(np.float32)
    if out.shape[1] != FEATURE_DIM:  # pragma: no cover - guards a future edit
        raise ValueError(f"expected {FEATURE_DIM} features, built {out.shape[1]}")
    return out
