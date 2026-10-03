"""VERI-Wild adapter: list-file parsing, not filename parsing (unlike VeRi-776)."""

from __future__ import annotations

from pathlib import Path

import pytest

from reiduq.core.exceptions import DataIntegrityError
from reiduq.data.veriwild import VeriWild


def _make_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "veriwild"
    (root / "images" / "00001").mkdir(parents=True)
    (root / "images" / "00002").mkdir(parents=True)
    (root / "train_test_split").mkdir(parents=True)
    (root / "images" / "00001" / "000001.jpg").write_bytes(b"fake")
    (root / "images" / "00002" / "000002.jpg").write_bytes(b"fake")
    (root / "train_test_split" / "train_list.txt").write_text(
        "00001/000001 1 c001\n00002/000002 2 c002\n"
    )
    return root


def test_loads_identity_and_camera_from_list_file(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path)
    samples = VeriWild(root, split="train").load()
    assert {s.identity for s in samples} == {1, 2}
    assert {s.camera_id for s in samples} == {"c001", "c002"}


def test_missing_list_file_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(DataIntegrityError, match="list file missing"):
        VeriWild(root, split="train").load()


def test_missing_images_dir_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "no_images"
    (root / "train_test_split").mkdir(parents=True)
    (root / "train_test_split" / "train_list.txt").write_text("00001/000001 1 c001\n")
    with pytest.raises(DataIntegrityError, match="images/ directory"):
        VeriWild(root, split="train").load()


def test_unknown_split_rejected(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path)
    with pytest.raises(DataIntegrityError):
        VeriWild(root, split="bogus")
