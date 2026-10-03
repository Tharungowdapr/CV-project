"""Split construction and the leakage guard.

The single most damaging mistake available in this project is fitting a
calibrator on data that also appears in the test set: it manufactures a result
that no one can reproduce. ``assert_disjoint`` is therefore called automatically
on every manifest load, not left to the discipline of whoever runs the script.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from reiduq.core.exceptions import SplitLeakageError
from reiduq.data.base import Sample


@dataclass
class SplitManifest:
    """A fully reproducible description of one train/calibration/test division."""

    protocol: str
    dataset: str
    seed: int
    train_ids: list[int]
    calib_ids: list[int]
    test_ids: list[int]
    train_cameras: list[str] = field(default_factory=list)
    calib_cameras: list[str] = field(default_factory=list)
    test_cameras: list[str] = field(default_factory=list)
    held_out_types: list[str] = field(default_factory=list)
    content_hash: str = ""

    def compute_hash(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if k != "content_hash"}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def save(self, path: Path) -> Path:
        self.content_hash = self.compute_hash()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        return path


def assert_disjoint(manifest: SplitManifest, *, require_camera_disjoint: bool = False) -> None:
    """Raise SplitLeakageError naming the offending ids or cameras."""
    pairs = (
        ("train", set(manifest.train_ids), "calibration", set(manifest.calib_ids)),
        ("train", set(manifest.train_ids), "test", set(manifest.test_ids)),
        ("calibration", set(manifest.calib_ids), "test", set(manifest.test_ids)),
    )
    for a_name, a, b_name, b in pairs:
        overlap = a & b
        if overlap:
            raise SplitLeakageError(
                f"identity leakage between {a_name} and {b_name}: "
                f"{sorted(overlap)[:10]}{' ...' if len(overlap) > 10 else ''} "
                f"({len(overlap)} identities)"
            )
    if require_camera_disjoint:
        cam_overlap = set(manifest.calib_cameras) & set(manifest.test_cameras)
        if cam_overlap:
            raise SplitLeakageError(
                "camera leakage between calibration and test under a domain-shift "
                f"protocol: {sorted(cam_overlap)}"
            )


def load_manifest(path: Path, *, require_camera_disjoint: bool = False) -> SplitManifest:
    data = json.loads(Path(path).read_text())
    manifest = SplitManifest(**data)
    expected = manifest.compute_hash()
    if manifest.content_hash and manifest.content_hash != expected:
        raise SplitLeakageError(
            f"manifest {path} was edited after creation "
            f"(hash {manifest.content_hash} != {expected})"
        )
    assert_disjoint(manifest, require_camera_disjoint=require_camera_disjoint)
    return manifest


def build_splits(
    samples: list[Sample],
    *,
    dataset: str,
    protocol: str = "in_distribution",
    seed: int = 42,
    calibration_fraction: float = 0.25,
    held_out_cameras: list[str] | None = None,
    held_out_types: list[str] | None = None,
) -> SplitManifest:
    """Create an identity-disjoint split, camera-disjoint where the protocol demands."""
    rng = np.random.default_rng(seed)
    ids = np.array(sorted({s.identity for s in samples}))
    rng.shuffle(ids)

    n_test = max(1, int(0.3 * len(ids)))
    n_calib = max(1, int(calibration_fraction * (len(ids) - n_test)))
    test_ids = ids[:n_test].tolist()
    calib_ids = ids[n_test : n_test + n_calib].tolist()
    train_ids = ids[n_test + n_calib :].tolist()

    all_cams = sorted({s.camera_id for s in samples})
    held_out_cameras = held_out_cameras or []
    if protocol == "leave_camera_out":
        test_cams = [c for c in all_cams if c in held_out_cameras]
        seen_cams = [c for c in all_cams if c not in held_out_cameras]
        if not test_cams:
            raise SplitLeakageError(
                f"held_out_cameras {held_out_cameras} match none of {all_cams}"
            )
        train_cams, calib_cams = seen_cams, seen_cams
    else:
        train_cams = calib_cams = test_cams = all_cams

    if protocol == "unseen_types" and held_out_types:
        held = set(held_out_types)
        test_ids = sorted({s.identity for s in samples if s.vehicle_type in held})
        train_ids = [i for i in train_ids if i not in set(test_ids)]
        calib_ids = [i for i in calib_ids if i not in set(test_ids)]

    manifest = SplitManifest(
        protocol=protocol,
        dataset=dataset,
        seed=seed,
        train_ids=sorted(train_ids),
        calib_ids=sorted(calib_ids),
        test_ids=sorted(test_ids),
        train_cameras=train_cams,
        calib_cameras=calib_cams,
        test_cameras=test_cams,
        held_out_types=held_out_types or [],
    )
    assert_disjoint(manifest, require_camera_disjoint=(protocol == "leave_camera_out"))
    manifest.content_hash = manifest.compute_hash()
    return manifest
