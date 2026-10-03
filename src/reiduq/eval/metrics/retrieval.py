"""Standard Re-ID retrieval metrics, plus mINP.

mINP is included because it is sensitive to the hardest correct match, which is
precisely what occlusion degrades - mAP can look healthy while the difficult
matches fall off the end of the ranking.
"""

from __future__ import annotations

import numpy as np


def cmc_and_map(
    ranked_identities: np.ndarray, query_identities: np.ndarray, max_rank: int = 20
) -> tuple[np.ndarray, float, float]:
    """(N, G) ranked gallery identities -> (CMC curve, mAP, mINP)."""
    n_queries = len(query_identities)
    matches = ranked_identities == query_identities[:, None]
    cmc = np.zeros(max_rank, dtype=np.float64)
    aps: list[float] = []
    inps: list[float] = []

    for i in range(n_queries):
        row = matches[i]
        if not row.any():
            continue
        first = int(np.argmax(row))
        cmc[min(first, max_rank - 1) :] += 1.0

        positions = np.nonzero(row)[0] + 1
        precisions = np.arange(1, len(positions) + 1) / positions
        aps.append(float(precisions.mean()))
        # mINP: penalise by where the LAST correct match sits
        inps.append(float(len(positions) / positions[-1]))

    valid = max(len([1 for i in range(n_queries) if matches[i].any()]), 1)
    return cmc / valid, float(np.mean(aps)) if aps else 0.0, float(np.mean(inps)) if inps else 0.0


def rank_k(cmc: np.ndarray, k: int) -> float:
    return float(cmc[min(k, len(cmc)) - 1])
