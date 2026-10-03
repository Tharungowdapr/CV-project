"""Split hygiene. These tests protect the validity of every number in the paper."""

from __future__ import annotations

from pathlib import Path

import pytest

from reiduq.core.exceptions import SplitLeakageError
from reiduq.data.base import Sample
from reiduq.data.splits import SplitManifest, assert_disjoint, build_splits, load_manifest


def _samples(n_ids: int = 20, n_cams: int = 4) -> list[Sample]:
    return [
        Sample(Path(f"/tmp/{i}_{c}.jpg"), identity=i, camera_id=f"c{c:03d}", vehicle_type="sedan")
        for i in range(n_ids)
        for c in range(n_cams)
    ]


def test_build_splits_are_identity_disjoint() -> None:
    m = build_splits(_samples(), dataset="veri776", seed=0)
    assert not set(m.train_ids) & set(m.calib_ids)
    assert not set(m.train_ids) & set(m.test_ids)
    assert not set(m.calib_ids) & set(m.test_ids)


def test_leakage_is_detected_not_tolerated() -> None:
    bad = SplitManifest(
        protocol="in_distribution",
        dataset="veri776",
        seed=0,
        train_ids=[1, 2, 3],
        calib_ids=[3, 4],  # 3 leaks
        test_ids=[5],
    )
    with pytest.raises(SplitLeakageError, match="identity leakage"):
        assert_disjoint(bad)


def test_camera_leakage_detected_under_domain_shift() -> None:
    bad = SplitManifest(
        protocol="leave_camera_out",
        dataset="veri776",
        seed=0,
        train_ids=[1],
        calib_ids=[2],
        test_ids=[3],
        calib_cameras=["c001", "c002"],
        test_cameras=["c002"],
    )
    with pytest.raises(SplitLeakageError, match="camera leakage"):
        assert_disjoint(bad, require_camera_disjoint=True)


def test_manifest_roundtrip_detects_tampering(tmp_path: Path) -> None:
    m = build_splits(_samples(), dataset="veri776", seed=1)
    path = m.save(tmp_path / "split.json")
    assert load_manifest(path).content_hash == m.content_hash

    text = path.read_text().replace('"seed": 1', '"seed": 999')
    path.write_text(text)
    with pytest.raises(SplitLeakageError, match="edited after creation"):
        load_manifest(path)


def test_leave_camera_out_requires_matching_cameras() -> None:
    with pytest.raises(SplitLeakageError):
        build_splits(
            _samples(), dataset="veri776", protocol="leave_camera_out", held_out_cameras=["c999"]
        )
