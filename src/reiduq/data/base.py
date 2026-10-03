"""Dataset adapter interface. Every dataset emits the same Sample records.

Adding a dataset means writing one adapter; nothing downstream changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Sample:
    """One labelled vehicle image."""

    image_path: Path
    identity: int
    camera_id: str
    frame_id: int = 0
    vehicle_type: str | None = None  # make/model, used by the unseen-types protocol
    timestamp: float | None = None

    @property
    def uid(self) -> str:
        return f"{self.camera_id}:{self.identity}:{self.image_path.name}"


class BaseVehicleDataset(ABC):
    """Common surface for VeRi-776, VERI-Wild, VehicleID and CityFlow."""

    name: str

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"{type(self).__name__}: dataset root missing: {self.root}")

    @abstractmethod
    def load(self) -> list[Sample]:
        """Return every sample with identity and camera labels resolved."""

    @property
    def identities(self) -> list[int]:
        return sorted({s.identity for s in self.load()})

    @property
    def cameras(self) -> list[str]:
        return sorted({s.camera_id for s in self.load()})

    @staticmethod
    def read_image(path: Path) -> np.ndarray:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"unreadable image: {path}")
        return img[:, :, ::-1].copy()  # BGR -> RGB
