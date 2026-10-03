"""End-to-end: index a tiny synthetic gallery, then search it both ways."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="search pipeline needs torch")
pytest.importorskip("faiss", reason="search pipeline needs faiss")

from reiduq.calibration.global_temperature import GlobalTemperature
from reiduq.models.builder import build_encoder
from reiduq.retrieval.index import GalleryEntry, GalleryIndex
from reiduq.search.gallery_store import GalleryEntryRow, GalleryStore


class _FakeDataset:
    name = "fake"

    @staticmethod
    def read_image(path: Path) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(str(path))) % (2**31))
        return rng.integers(0, 255, (256, 128, 3), dtype=np.uint8)


def _build_tiny_index(tmp_path: Path) -> tuple[Path, Path]:
    embed_dim = 32
    rng = np.random.default_rng(0)

    index = GalleryIndex(dim=embed_dim)
    store = GalleryStore(tmp_path / "store.db")
    vectors, entries = [], []
    for i in range(6):
        vec = rng.normal(size=embed_dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        vectors.append(vec)
        uid = f"u{i}"
        entries.append(GalleryEntry(uid=uid, identity=i % 3, camera_id=f"c{i % 2}"))
        store.add(
            GalleryEntryRow(
                faiss_position=i, uid=uid, image_path=f"/tmp/{uid}.jpg", dataset="fake",
                camera_id=f"c{i % 2}", ground_truth_identity=i % 3,
                plate_text="KA05MN1234" if i == 2 else None,
                plate_confidence=0.85 if i == 2 else None,
            )
        )
    index.add(np.stack(vectors), entries)
    index.build()
    index_path = tmp_path / "index.faiss"
    index.save(index_path)
    return index_path, tmp_path / "store.db"


def test_search_by_photo_returns_calibrated_sightings(tmp_path: Path) -> None:
    from reiduq.search.query_pipeline import SearchEngine

    index_path, store_path = _build_tiny_index(tmp_path)
    embed_dim = 32
    encoder = build_encoder("resnet50_ibn", embed_dim=embed_dim, pretrained=False)
    calibrator = GlobalTemperature()
    calibrator.fit(np.array([[0.9, 0.5, 0.3]]), np.array([True]))

    engine = SearchEngine.load(index_path, store_path, encoder, calibrator)
    image = np.zeros((256, 128, 3), dtype=np.uint8)
    sightings = engine.search_by_photo(image, k=3)

    assert len(sightings) > 0
    assert all(s.verdict in {"MATCH", "UNCERTAIN", "REJECT"} for s in sightings)
    assert all(0.0 <= s.confidence <= 1.0 for s in sightings)


def test_search_by_plate_expands_via_reid_not_just_text(tmp_path: Path) -> None:
    import reiduq.search.query_pipeline as qp

    index_path, store_path = _build_tiny_index(tmp_path)
    embed_dim = 32
    encoder = build_encoder("resnet50_ibn", embed_dim=embed_dim, pretrained=False)
    calibrator = GlobalTemperature()
    calibrator.fit(np.array([[0.9, 0.5, 0.3]]), np.array([True]))

    engine = qp.SearchEngine.load(index_path, store_path, encoder, calibrator)
    qp.BaseVehicleDataset = _FakeDataset  # type: ignore[attr-defined,misc]
    sightings = engine.search_by_plate("KA05MN1234", k=3)

    assert len(sightings) > 0
    assert all(s.plate_confidence == pytest.approx(0.85) for s in sightings)


def test_search_by_unknown_plate_raises_a_clear_error(tmp_path: Path) -> None:
    from reiduq.search.query_pipeline import SearchEngine

    index_path, store_path = _build_tiny_index(tmp_path)
    embed_dim = 32
    encoder = build_encoder("resnet50_ibn", embed_dim=embed_dim, pretrained=False)
    calibrator = GlobalTemperature()
    calibrator.fit(np.array([[0.9]]), np.array([True]))

    engine = SearchEngine.load(index_path, store_path, encoder, calibrator)
    from reiduq.core.exceptions import DataIntegrityError

    with pytest.raises(DataIntegrityError, match="no indexed plate reads"):
        engine.search_by_plate("ZZZ9999")
