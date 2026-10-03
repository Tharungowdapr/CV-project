"""VERI-Wild / VERI-Wild 2.0 adapter.

VERI-Wild ships as ``images/<vehicle_id>/<image_id>.jpg`` plus plain-text list
files of the form ``<vehicle_id>/<image_id> <vehicle_id> <camera_id>`` under
``train_test_split/``. This is a different layout from VeRi-776 (which encodes
everything in the filename), so identity and camera come from the list file
rather than a filename regex.

Per the dataset's terms of access this adapter only reads local files already
downloaded and accepted under those terms; it never fetches anything.
"""

from __future__ import annotations

from pathlib import Path

from reiduq.core.exceptions import DataIntegrityError
from reiduq.core.registry import DATASETS
from reiduq.data.base import BaseVehicleDataset, Sample


@DATASETS.register("veriwild")
class VeriWild(BaseVehicleDataset):
    name = "veriwild"

    LIST_FILES = {
        "train": "train_test_split/train_list.txt",
        "query": "train_test_split/test_3000_id_query.txt",
        "gallery": "train_test_split/test_3000_id.txt",
    }

    def __init__(self, root: Path | str, split: str = "train") -> None:
        super().__init__(root)
        if split not in self.LIST_FILES:
            raise DataIntegrityError(f"unknown split '{split}'")
        self.split = split
        self.vehicle_info = self._load_vehicle_info()

    def _load_vehicle_info(self) -> dict[int, dict[str, str]]:
        """Optional vehicle_info.txt: <vid>;<camera>;<type>;<color>;<brand>."""
        info_file = self.root / "vehicle_info.txt"
        if not info_file.exists():
            return {}
        out: dict[int, dict[str, str]] = {}
        for line in info_file.read_text().splitlines()[1:]:  # skip header
            parts = line.strip().split(";")
            if len(parts) >= 5 and parts[0].isdigit():
                out[int(parts[0])] = {"type": parts[2], "color": parts[3], "brand": parts[4]}
        return out

    def load(self) -> list[Sample]:
        list_path = self.root / self.LIST_FILES[self.split]
        if not list_path.is_file():
            raise DataIntegrityError(f"expected list file missing: {list_path}")
        images_root = self.root / "images"
        if not images_root.is_dir():
            raise DataIntegrityError(f"expected images/ directory missing under {self.root}")

        samples: list[Sample] = []
        for line in list_path.read_text().splitlines():
            fields = line.strip().split()
            if not fields:
                continue
            rel = fields[0]  # "<vid>/<image_id>"
            vid_str, _, image_id = rel.partition("/")
            if not vid_str.isdigit():
                continue
            vid = int(vid_str)
            camera_id = fields[2] if len(fields) >= 3 else "c000"
            image_path = images_root / vid_str / f"{image_id}.jpg"
            samples.append(
                Sample(
                    image_path=image_path,
                    identity=vid,
                    camera_id=camera_id if camera_id.startswith("c") else f"c{camera_id}",
                    vehicle_type=self.vehicle_info.get(vid, {}).get("type"),
                )
            )
        if not samples:
            raise DataIntegrityError(f"no parsable rows in {list_path}")
        return samples
