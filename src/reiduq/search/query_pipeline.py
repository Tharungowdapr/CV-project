"""Shared query-time logic for both search modes.

Photo search: embed the uploaded image, retrieve top-k from the index,
calibrate, return sightings. Plate search: look the plate text up in the
gallery store to get an anchor image, then run that anchor's embedding
through the *same* photo-search path - this is why a plate hit can surface
sightings where the plate itself was never legible: once anchored, matching
runs on appearance, not text.

Every result carries a calibrated Match/Uncertain/Reject verdict, not a bare
similarity score - that is the one property this whole project exists to
provide, and the search feature would defeat the point if it silently
dropped back to raw similarity here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reiduq.abstention.decision import decide
from reiduq.calibration.base import BaseCalibrator
from reiduq.core.exceptions import DataIntegrityError
from reiduq.data.base import BaseVehicleDataset
from reiduq.retrieval.index import GalleryIndex
from reiduq.search.gallery_store import GalleryStore


@dataclass(frozen=True)
class Sighting:
    image_path: str
    camera_id: str
    dataset: str
    similarity: float
    confidence: float
    verdict: str
    reason: str
    plate_text: str | None = None
    plate_confidence: float | None = None


class SearchEngine:
    """Owns the loaded index, store, encoder and calibrator for one gallery."""

    def __init__(
        self,
        index: GalleryIndex,
        store: GalleryStore,
        encoder: Any,
        calibrator: BaseCalibrator,
        tau_high: float = 0.9,
        tau_low: float = 0.2,
        device: str = "cpu",
    ) -> None:
        self.index = index
        self.store = store
        self.encoder = encoder
        self.calibrator = calibrator
        self.tau_high, self.tau_low = tau_high, tau_low
        self.device = device

    @classmethod
    def load(
        cls,
        index_path: Path | str,
        store_path: Path | str,
        encoder: Any,
        calibrator: BaseCalibrator,
        **kw: Any,
    ) -> SearchEngine:
        index = GalleryIndex.load(index_path)
        store = GalleryStore(store_path)
        return cls(index, store, encoder, calibrator, **kw)

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

    def _to_sightings(self, sims: np.ndarray, uids: list[str], visibility: float) -> list[Sighting]:
        needs_features = getattr(self.calibrator, "requires_features", False)
        features = None
        if needs_features:
            from reiduq.calibration.camera_descriptor import CameraDescriptor
            from reiduq.calibration.features import build_features

            features = build_features(
                sims.reshape(1, -1),
                np.array([visibility], dtype=np.float32),
                np.tile(visibility, (1, 6)).astype(np.float32),
                CameraDescriptor()(np.zeros((1, 9), dtype=np.float32)),
                np.array([1]),
            )
        probs = self.calibrator.transform(sims.reshape(1, -1), features)[0]

        out: list[Sighting] = []
        for i, uid in enumerate(uids):
            row = self._row_for_uid(uid)
            if row is None:
                continue
            confidence = float(probs[i]) if i < len(probs) else float(sims[i])
            decision = decide(confidence, uid, self.tau_high, self.tau_low, visibility=visibility)
            out.append(
                Sighting(
                    image_path=str(row["image_path"]),
                    camera_id=str(row["camera_id"]),
                    dataset=str(row["dataset"]),
                    similarity=float(sims[i]),
                    confidence=confidence,
                    verdict=decision.verdict,
                    reason=decision.reason,
                    plate_text=row.get("plate_text"),  # type: ignore[arg-type]
                    plate_confidence=row.get("plate_confidence"),  # type: ignore[arg-type]
                )
            )
        return out

    def _row_for_uid(self, uid: str) -> dict[str, object] | None:
        for entry_idx, entry in enumerate(self.index.entries):
            if entry.uid == uid:
                return self.store.by_faiss_position(entry_idx)
        return None

    def search_by_photo(self, image: np.ndarray, k: int = 10, visibility: float = 1.0) -> list[Sighting]:
        embedding = self._embed(image)
        results = self.index.search(embedding.reshape(1, -1), ["query"], k=k)
        result = results[0]
        return self._to_sightings(result.similarities, result.candidate_ids, visibility)

    def search_by_plate(self, plate_query: str, k: int = 10) -> list[Sighting]:
        """Fuzzy plate lookup for an anchor, then Re-ID expansion from it.

        Returns sightings found via appearance matching from the best plate
        hit, NOT a plain list of every row whose OCR text happens to match -
        that distinction is the entire value of doing this over a bare text
        search: it recovers sightings where the plate itself was never
        legible.
        """
        candidates = self.store.search_plate(plate_query, limit=5)
        if not candidates:
            raise DataIntegrityError(f"no indexed plate reads matched '{plate_query}'")
        anchor = candidates[0]  # highest OCR confidence among the matches
        image = BaseVehicleDataset.read_image(Path(str(anchor["image_path"])))
        sightings = self.search_by_photo(image, k=k)
        # Surface the anchor's own OCR confidence on every result, since the
        # whole chain of trust starts there - a low-confidence plate read
        # should visibly discount everything downstream of it.
        anchor_conf = float(anchor.get("plate_confidence") or 0.0)
        return [Sighting(**{**s.__dict__, "plate_confidence": anchor_conf}) for s in sightings]
