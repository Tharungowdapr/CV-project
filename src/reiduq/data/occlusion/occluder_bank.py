"""A bank of real, segmented occluder patches mined from the dataset itself.

Using real objects (with their own texture statistics and compression
artefacts) is what separates a defensible occlusion protocol from pasting black
rectangles, which reviewers reject on sight.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OccluderPatch:
    patch_id: str
    rgb: np.ndarray  # (h, w, 3) uint8
    mask: np.ndarray  # (h, w) uint8 in {0,1}
    occluder_class: str
    source_image: str

    @property
    def size(self) -> tuple[int, int]:
        return self.mask.shape[0], self.mask.shape[1]


class OccluderBank:
    """Loads patches saved as ``<id>.npz`` with an index.json describing them."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        index_file = self.root / "index.json"
        self.index: list[dict[str, str]] = (
            json.loads(index_file.read_text()) if index_file.exists() else []
        )

    def __len__(self) -> int:
        return len(self.index)

    def load(self, patch_id: str) -> OccluderPatch:
        blob = np.load(self.root / f"{patch_id}.npz")
        entry = next(e for e in self.index if e["patch_id"] == patch_id)
        return OccluderPatch(
            patch_id=patch_id,
            rgb=blob["rgb"],
            mask=blob["mask"],
            occluder_class=entry["occluder_class"],
            source_image=entry["source_image"],
        )

    def sample(self, rng: np.random.Generator, classes: list[str] | None = None) -> OccluderPatch:
        pool = [e for e in self.index if classes is None or e["occluder_class"] in classes]
        if not pool:
            raise ValueError("occluder bank is empty - run scripts/mine_occluders.py first")
        return self.load(pool[int(rng.integers(len(pool)))]["patch_id"])

    def add(self, patch: OccluderPatch) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.root / f"{patch.patch_id}.npz", rgb=patch.rgb, mask=patch.mask)
        self.index.append(
            {
                "patch_id": patch.patch_id,
                "occluder_class": patch.occluder_class,
                "source_image": patch.source_image,
            }
        )
        (self.root / "index.json").write_text(json.dumps(self.index, indent=2))
