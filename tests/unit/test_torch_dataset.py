"""ReIDImageDataset: label remapping and PKSampler compatibility."""

from __future__ import annotations

from pathlib import Path

from reiduq.data.base import Sample
from reiduq.data.samplers import PKSampler
from reiduq.data.torch_dataset import ReIDImageDataset


def _samples(n_ids: int = 6, per_id: int = 4) -> list[Sample]:
    return [
        Sample(Path(f"/tmp/{i}_{j}.jpg"), identity=100 + i, camera_id="c001")
        for i in range(n_ids)
        for j in range(per_id)
    ]


def test_labels_are_remapped_to_contiguous_indices() -> None:
    ds = ReIDImageDataset(_samples())
    assert ds.num_identities == 6
    assert set(ds.label_map.values()) == set(range(6))


def test_identities_align_with_sampler_indices() -> None:
    samples = _samples()
    ds = ReIDImageDataset(samples)
    sampler = PKSampler(ds.identities, p=2, k=4, seed=0)
    batch = next(iter(sampler))
    labels = [ds.label_map[samples[i].identity] for i in batch]
    assert len(batch) == 8
    assert len(set(labels)) == 2  # exactly P distinct identities per batch
