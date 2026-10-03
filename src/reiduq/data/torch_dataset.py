"""torch.utils.data.Dataset wrapping a list of Sample records.

Kept separate from data/base.py so that base.py (the dataset adapter contract)
has no torch dependency and can be imported in torch-free contexts, such as
metric-only unit tests.
"""

from __future__ import annotations

from typing import Any

from reiduq.data.base import Sample


class ReIDImageDataset:
    """Maps identities to contiguous label indices; returns (image, label, camera_id)."""

    def __init__(self, samples: list[Sample], transform: Any = None) -> None:
        self.samples = samples
        self.transform = transform
        ids = sorted({s.identity for s in samples})
        self.label_map = {vid: i for i, vid in enumerate(ids)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int, str]:
        from PIL import Image

        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.label_map[sample.identity], sample.camera_id

    @property
    def num_identities(self) -> int:
        return len(self.label_map)

    @property
    def identities(self) -> list[int]:
        """One identity per sample, in dataset order - what PKSampler needs."""
        return [s.identity for s in self.samples]
