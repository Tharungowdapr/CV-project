"""Build a searchable gallery: embeddings + metadata for every image in a dataset.

This is the offline job that makes the two search modes possible. It walks a
dataset (VeRi-776, VERI-Wild, ...), and for every image:

  1. computes an L2-normalised embedding with the trained encoder
  2. (optionally) attempts a best-effort plate read - see plate_ocr.py for
     why this is treated as a low-confidence hint, not a verified field

Dataset images (VeRi-776, VERI-Wild) are already single-vehicle crops, not
full scenes - that is how Re-ID benchmarks are packaged - so no detection
stage runs here. Detection only matters on the query side, where an
uploaded photo has not been pre-cropped to one vehicle.

Run via scripts/build_gallery_index.py, not imported ad hoc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from reiduq.core.logging import get_logger
from reiduq.data.base import BaseVehicleDataset, Sample
from reiduq.retrieval.index import GalleryEntry, GalleryIndex
from reiduq.search.gallery_store import GalleryEntryRow, GalleryStore

log = get_logger(__name__)


class GalleryIndexer:
    def __init__(
        self,
        encoder: Any,
        embed_dim: int,
        device: str = "cpu",
        plate_reader: Any | None = None,
    ) -> None:
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.device = device
        self.plate_reader = plate_reader

    def _embed(self, image: np.ndarray) -> np.ndarray:
        import torch
        import torchvision.transforms as T

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        tensor = transform(image).unsqueeze(0).to(self.device)
        self.encoder.eval()
        with torch.no_grad():
            vec = self.encoder(tensor)
            vec = torch.nn.functional.normalize(vec, dim=1)
        return vec.cpu().numpy()[0].astype(np.float32)

    def _read_plate(self, image: np.ndarray) -> tuple[str | None, float]:
        if self.plate_reader is None or not getattr(self.plate_reader, "available", False):
            return None, 0.0
        from reiduq.search.plate_ocr import locate_plate_region

        crop = locate_plate_region(image)
        if crop.size == 0:
            return None, 0.0
        return self.plate_reader.read(crop)

    def build(
        self,
        dataset: BaseVehicleDataset,
        index_out: Path | str,
        store_out: Path | str,
        limit: int | None = None,
    ) -> tuple[GalleryIndex, GalleryStore]:
        """Embed every sample, write incrementally to the metadata store, and
        write the FAISS index once at the end.

        NOTE on resumability: the metadata store (SQLite) is written one row
        at a time and survives a crash. The FAISS side does not yet support
        incremental append-and-resume - a crash partway through means
        re-running build() for the embeddings. For VERI-Wild-scale runs,
        prefer --limit for an incremental first pass rather than one giant
        call until FAISS-side resumability is added.
        """
        samples: list[Sample] = dataset.load()
        if limit:
            samples = samples[:limit]

        store = GalleryStore(store_out)
        vectors: list[np.ndarray] = []
        entries: list[GalleryEntry] = []

        for i, sample in enumerate(samples):
            image = dataset.read_image(sample.image_path)
            embedding = self._embed(image)
            plate_text, plate_conf = self._read_plate(image)

            uid = sample.uid
            vectors.append(embedding)
            entries.append(GalleryEntry(uid=uid, identity=sample.identity, camera_id=sample.camera_id))
            store.add(
                GalleryEntryRow(
                    faiss_position=i,
                    uid=uid,
                    image_path=str(sample.image_path),
                    dataset=dataset.name,
                    camera_id=sample.camera_id,
                    ground_truth_identity=sample.identity,
                    plate_text=plate_text,
                    plate_confidence=plate_conf if plate_text else None,
                    visibility=None,
                )
            )
            if (i + 1) % 500 == 0:
                log.info("indexer.progress", done=i + 1, total=len(samples))

        index = GalleryIndex(dim=self.embed_dim)
        index.add(np.stack(vectors), entries)
        index.build()
        index.save(index_out)
        log.info("indexer.done", entries=len(entries), index_path=str(index_out))
        return index, store
