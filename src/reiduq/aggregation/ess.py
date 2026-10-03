"""Effective sample size for correlated tracklet frames.

Consecutive frames of one vehicle are near-duplicates. Naive log-pooling treats
them as independent evidence and manufactures confidence - the exact failure
this project exists to fix, reintroduced one stage later. The pooled evidence is
therefore discounted by n_eff / n.
"""

from __future__ import annotations

import numpy as np


def mean_pairwise_correlation(embeddings: np.ndarray) -> float:
    """(n, D) L2-normalised -> mean off-diagonal cosine similarity, clipped to [0, 1]."""
    if len(embeddings) < 2:
        return 0.0
    gram = embeddings @ embeddings.T
    n = len(embeddings)
    off = (gram.sum() - np.trace(gram)) / (n * (n - 1))
    return float(np.clip(off, 0.0, 1.0))


def effective_sample_size(embeddings: np.ndarray) -> float:
    """n_eff = n / (1 + (n-1) * rho), clamped to [1, n]."""
    n = len(embeddings)
    if n <= 1:
        return float(max(n, 1))
    rho = mean_pairwise_correlation(embeddings)
    return float(np.clip(n / (1.0 + (n - 1) * rho), 1.0, n))
