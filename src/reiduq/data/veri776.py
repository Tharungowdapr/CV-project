"""VeRi-776 adapter.

Filenames follow ``<vehicleID>_c<cameraID>_<frame>_<seq>.jpg`` which is enough to
recover identity, camera and frame without parsing the XML metadata. Vehicle
type is read from the provided label file when present (needed for the
unseen-types protocol).
"""

from __future__ import annotations

import re
from pathlib import Path

from reiduq.core.exceptions import DataIntegrityError
from reiduq.core.registry import DATASETS
from reiduq.data.base import BaseVehicleDataset, Sample

_PATTERN = re.compile(r"^(?P<vid>\d+)_c(?P<cam>\d+)_(?P<frame>\d+)_")


@DATASETS.register("veri776")
class VeRi776(BaseVehicleDataset):
    name = "veri776"

    SPLIT_DIRS = {
        "train": "image_train",
        "query": "image_query",
        "gallery": "image_test",
    }

    def __init__(self, root: Path | str, split: str = "train") -> None:
        super().__init__(root)
        if split not in self.SPLIT_DIRS:
            raise DataIntegrityError(f"unknown split '{split}'")
        self.split = split
        self.types = self._load_types()

    def _load_types(self) -> dict[int, str]:
        label_file = self.root / "list_type.txt"
        if not label_file.exists():
            return {}
        out: dict[int, str] = {}
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                out[int(parts[0])] = parts[1]
        return out

    def load(self) -> list[Sample]:
        folder = self.root / self.SPLIT_DIRS[self.split]
        if not folder.is_dir():
            raise DataIntegrityError(f"expected directory missing: {folder}")
        samples: list[Sample] = []
        for path in sorted(folder.glob("*.jpg")):
            m = _PATTERN.match(path.name)
            if not m:
                continue
            vid = int(m.group("vid"))
            samples.append(
                Sample(
                    image_path=path,
                    identity=vid,
                    camera_id=f"c{int(m.group('cam')):03d}",
                    frame_id=int(m.group("frame")),
                    vehicle_type=self.types.get(vid),
                )
            )
        if not samples:
            raise DataIntegrityError(f"no parsable images under {folder}")
        return samples
