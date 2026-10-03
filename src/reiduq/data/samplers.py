"""PK identity sampler: P identities x K instances per batch.

Triplet loss needs multiple instances of the same identity inside one batch;
random shuffling gives batches with almost no positive pairs.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


class PKSampler:
    """Yields index batches of size P*K."""

    def __init__(
        self, identities: list[int], p: int = 16, k: int = 4, seed: int = 42
    ) -> None:
        if p < 2 or k < 2:
            raise ValueError("PK sampling needs p >= 2 and k >= 2")
        self.p, self.k = p, k
        self.rng = np.random.default_rng(seed)
        self.index_by_id: dict[int, list[int]] = {}
        for idx, ident in enumerate(identities):
            self.index_by_id.setdefault(ident, []).append(idx)
        self.ids = [i for i, v in self.index_by_id.items() if len(v) >= 2]

    def __iter__(self) -> Iterator[list[int]]:
        ids = list(self.ids)
        self.rng.shuffle(ids)
        for start in range(0, len(ids) - self.p + 1, self.p):
            batch: list[int] = []
            for ident in ids[start : start + self.p]:
                pool = self.index_by_id[ident]
                replace = len(pool) < self.k
                batch.extend(self.rng.choice(pool, self.k, replace=replace).tolist())
            yield batch

    def __len__(self) -> int:
        return len(self.ids) // self.p
