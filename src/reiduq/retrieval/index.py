"""Gallery index. Exact inner-product search for science, IVF-PQ for the scale test.

Exact search is used for every reported number; the approximate index appears
only in the throughput study, where its recall against exact is reported so the
speed-up is not quietly bought with accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from reiduq.core.types import RetrievalResult


@dataclass
class GalleryEntry:
    uid: str
    identity: int | None
    camera_id: str


class GalleryIndex:
    def __init__(self, dim: int = 512, kind: Literal["flat", "ivfpq"] = "flat") -> None:
        self.dim, self.kind = dim, kind
        self.entries: list[GalleryEntry] = []
        self._vectors: list[np.ndarray] = []
        self._index: Any | None = None

    def add(self, vectors: np.ndarray, entries: list[GalleryEntry]) -> None:
        if vectors.shape[0] != len(entries):
            raise ValueError("vector count and entry count disagree")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        self._vectors.append(np.ascontiguousarray(vectors, dtype=np.float32))
        self.entries.extend(entries)
        self._index = None  # rebuilt lazily

    @property
    def matrix(self) -> np.ndarray:
        if not self._vectors:
            raise RuntimeError("gallery is empty")
        return np.vstack(self._vectors)

    def build(self, nlist: int = 100, m: int = 16) -> None:
        import faiss

        mat = self.matrix
        if self.kind == "flat":
            index = faiss.IndexFlatIP(self.dim)
        else:
            quantiser = faiss.IndexFlatIP(self.dim)
            index = faiss.IndexIVFPQ(quantiser, self.dim, min(nlist, max(len(mat) // 40, 1)), m, 8)
            index.train(mat)
        index.add(mat)
        self._index = index

    def save(self, path: Path | str) -> None:
        """Persist the FAISS index and entry metadata alongside it.

        Building an index (running the encoder over an entire dataset) is a
        one-off offline job; searching happens later, often in a different
        process. Without persistence every search would need to re-embed the
        whole gallery first, which defeats the point of having an index.
        """
        import json

        import faiss

        if self._index is None:
            self.build()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "dim": self.dim,
                    "kind": self.kind,
                    "entries": [
                        {"uid": e.uid, "identity": e.identity, "camera_id": e.camera_id}
                        for e in self.entries
                    ],
                }
            )
        )

    @classmethod
    def load(cls, path: Path | str) -> GalleryIndex:
        import json

        import faiss

        path = Path(path)
        meta = json.loads(path.with_suffix(path.suffix + ".meta.json").read_text())
        obj = cls(dim=meta["dim"], kind=meta["kind"])
        obj._index = faiss.read_index(str(path))
        obj.entries = [
            GalleryEntry(uid=e["uid"], identity=e["identity"], camera_id=e["camera_id"])
            for e in meta["entries"]
        ]
        return obj

    def search(
        self,
        queries: np.ndarray,
        query_ids: list[str],
        k: int = 20,
        query_identities: list[int | None] | None = None,
        exclude_same_camera: list[str] | None = None,
    ) -> list[RetrievalResult]:
        if self._index is None:
            self.build()
        if self._index is None:  # pragma: no cover - build() always sets it or raises
            raise RuntimeError("gallery index failed to build")
        # Over-fetch so same-camera candidates can be removed without short results.
        fetch = k * 3 if exclude_same_camera else k
        sims, idx = self._index.search(np.ascontiguousarray(queries, dtype=np.float32), fetch)

        results: list[RetrievalResult] = []
        for i, qid in enumerate(query_ids):
            cand_idx, cand_sim = idx[i], sims[i]
            if exclude_same_camera is not None:
                keep = [
                    j
                    for j, gi in enumerate(cand_idx)
                    if gi >= 0 and self.entries[gi].camera_id != exclude_same_camera[i]
                ]
                cand_idx, cand_sim = cand_idx[keep][:k], cand_sim[keep][:k]
            else:
                cand_idx, cand_sim = cand_idx[:k], cand_sim[:k]
            results.append(
                RetrievalResult(
                    query_id=qid,
                    candidate_ids=[self.entries[j].uid for j in cand_idx if j >= 0],
                    similarities=np.asarray(cand_sim, dtype=np.float32),
                    query_identity=query_identities[i] if query_identities else None,
                    candidate_identities=[self.entries[j].identity for j in cand_idx if j >= 0],
                )
            )
        return results
