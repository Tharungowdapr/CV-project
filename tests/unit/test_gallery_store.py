"""GalleryStore: persistence, plate search, and the not-a-guarantee semantics."""

from __future__ import annotations

from pathlib import Path

from reiduq.search.gallery_store import GalleryEntryRow, GalleryStore


def _row(faiss_pos: int, uid: str, plate: str | None = None, conf: float | None = None) -> GalleryEntryRow:
    return GalleryEntryRow(
        faiss_position=faiss_pos,
        uid=uid,
        image_path=f"/tmp/{uid}.jpg",
        dataset="veri776",
        camera_id="c001",
        ground_truth_identity=1,
        plate_text=plate,
        plate_confidence=conf,
    )


def test_round_trip_by_faiss_position(tmp_path: Path) -> None:
    store = GalleryStore(tmp_path / "s.db")
    store.add(_row(0, "u0", "AB1234", 0.8))
    row = store.by_faiss_position(0)
    assert row is not None
    assert row["plate_text"] == "AB1234"


def test_plate_search_is_case_insensitive_substring(tmp_path: Path) -> None:
    store = GalleryStore(tmp_path / "s.db")
    store.add(_row(0, "u0", "KA05MN1234", 0.9))
    store.add(_row(1, "u1", "TN10AB5555", 0.7))
    hits = store.search_plate("ka05")
    assert len(hits) == 1
    assert hits[0]["plate_text"] == "KA05MN1234"


def test_plate_search_orders_by_confidence(tmp_path: Path) -> None:
    store = GalleryStore(tmp_path / "s.db")
    store.add(_row(0, "u0", "MATCH999", 0.4))
    store.add(_row(1, "u1", "MATCH999", 0.9))
    hits = store.search_plate("MATCH999")
    assert hits[0]["plate_confidence"] == 0.9


def test_percent_wildcard_in_query_is_escaped_not_interpreted(tmp_path: Path) -> None:
    """A raw '%' in a LIKE query matches everything - must not let a query
    string accidentally become a wildcard scan."""
    store = GalleryStore(tmp_path / "s.db")
    store.add(_row(0, "u0", "ABC123", 0.9))
    hits = store.search_plate("%")
    assert hits == []  # a literal percent sign should not match every row


def test_entries_with_no_plate_are_never_returned(tmp_path: Path) -> None:
    store = GalleryStore(tmp_path / "s.db")
    store.add(_row(0, "u0", None, None))
    assert store.search_plate("anything") == []


def test_count(tmp_path: Path) -> None:
    store = GalleryStore(tmp_path / "s.db")
    for i in range(3):
        store.add(_row(i, f"u{i}"))
    assert store.count() == 3
