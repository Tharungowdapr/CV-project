"""Best-effort license plate OCR.

Read this before trusting a plate match: VeRi-776 and VERI-Wild were released
as *re-identification* benchmarks, not plate-recognition benchmarks. They
carry no ground-truth plate-text labels, and because they are fixed
surveillance footage, a meaningful fraction of plates are too small, angled,
or low-resolution to read reliably - this is a limitation of the source
images, not of the OCR engine. Treat every plate read here as a low-confidence
hint, never as a verified identifier, and always report plate_confidence
alongside plate_text rather than the text alone.

Two backends, in preference order:
  1. EasyOCR general-purpose OCR run on a plate-shaped crop, if installed.
     Not a purpose-built ANPR model, so expect a non-trivial error rate on
     oblique or low-resolution plates - it is what is realistically available
     without paying for a commercial ANPR SDK.
  2. A no-op fallback that returns (None, 0.0) so indexing and search still
     work end to end with plate lookup simply returning nothing - the system
     degrades, it does not crash, when the optional dependency is absent.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

# Plausible plate-text shape: letters/digits, a few characters, no lowercase.
# Filters obvious OCR noise (stray punctuation, single characters) before it
# ever reaches the gallery store.
_PLATE_PATTERN = re.compile(r"^[A-Z0-9]{4,10}$")


class PlateReader:
    def __init__(self, min_confidence: float = 0.35) -> None:
        self.min_confidence = min_confidence
        self._reader: Any | None = None
        self._available = self._try_load()

    def _try_load(self) -> bool:
        try:
            import easyocr

            self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            return True
        except ImportError:
            return False

    @property
    def available(self) -> bool:
        return self._available

    def read(self, plate_crop: np.ndarray) -> tuple[str | None, float]:
        """(H, W, 3) uint8 RGB crop, ideally already localised to the plate
        region -> (text, confidence) or (None, 0.0) if nothing legible was
        found or the OCR backend is unavailable.
        """
        if not self._available or self._reader is None:
            return None, 0.0
        try:
            results = self._reader.readtext(plate_crop)
        except Exception:  # pragma: no cover - depends on an optional native lib
            return None, 0.0

        best_text, best_conf = None, 0.0
        for _bbox, text, conf in results:
            cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
            if _PLATE_PATTERN.match(cleaned) and conf > best_conf:
                best_text, best_conf = cleaned, float(conf)

        if best_conf < self.min_confidence:
            return None, 0.0
        return best_text, best_conf


def locate_plate_region(vehicle_crop: np.ndarray) -> np.ndarray:
    """Heuristic plate-region crop: the lower-middle band of the vehicle box.

    Not a plate detector - a real deployment should run a small dedicated
    plate-detection model first and pass its crop to PlateReader.read()
    instead of this heuristic. This exists so the search pipeline has
    *something* to OCR without requiring a second trained model just to
    demonstrate the flow; its accuracy is correspondingly limited and it
    should be the first thing replaced in a serious deployment.
    """
    h, w = vehicle_crop.shape[:2]
    y0, y1 = int(h * 0.55), int(h * 0.85)
    x0, x1 = int(w * 0.15), int(w * 0.85)
    return vehicle_crop[y0:y1, x0:x1]
