"""Dataset processor: converts image-based datasets into lightweight CSV manifests.

Instead of keeping thousands of raw images in memory/storage, this module:
  1. Scans an image folder (or unpacked ZIP) and extracts all metadata into a
     CSV manifest (uid, identity, camera_id, frame_id, vehicle_type, split,
     relative_path, width, height, file_size_bytes, md5).
  2. Copies only the first N images as "preview" thumbnails (resized, JPEG).
  3. Optionally deletes the source images after export.

This is the core engine behind both:
  - ``scripts/process_dataset.py`` (offline CLI)
  - ``/v1/dataset/ingest`` (live API endpoint, ZIP or folder)
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ImageRecord:
    uid: str
    dataset_name: str
    split: str
    relative_path: str          # relative to dataset root
    identity: int | None
    camera_id: str | None
    frame_id: int | None
    vehicle_type: str | None
    width: int | None
    height: int | None
    file_size_bytes: int
    md5: str
    is_preview: bool = False
    preview_path: str | None = None   # relative path of the saved thumbnail

    def to_row(self) -> dict:
        return asdict(self)


@dataclass
class ProcessingResult:
    dataset_name: str
    total_images: int
    processed: int
    preview_count: int
    csv_path: str
    preview_dir: str
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


# ---------------------------------------------------------------------------
# Filename parsers (VeRi-776, VeriWild, generic)
# ---------------------------------------------------------------------------

_VERI776_PAT = re.compile(r"^(?P<vid>\d+)_c(?P<cam>\d+)_(?P<frame>\d+)_")
_VERIWILD_PAT = re.compile(r"^(?P<vid>\d+)/(?P<img_id>\d+)$")


def _parse_veri776(name: str) -> tuple[int | None, str | None, int | None]:
    m = _VERI776_PAT.match(name)
    if not m:
        return None, None, None
    return int(m.group("vid")), f"c{int(m.group('cam')):03d}", int(m.group("frame"))


def _parse_filename(rel: str) -> tuple[int | None, str | None, int | None]:
    """Best-effort metadata extraction from a relative path."""
    # VeRi-776 style: <vid>_c<cam>_<frame>_<seq>.jpg
    name = Path(rel).name
    identity, camera_id, frame_id = _parse_veri776(name)
    if identity is not None:
        return identity, camera_id, frame_id

    # VeriWild style: images/<vid>/<img_id>.jpg
    parts = Path(rel).parts
    if len(parts) >= 2 and parts[-2].isdigit():
        return int(parts[-2]), None, None

    # Fallback: try leading numeric prefix in filename
    m = re.match(r"^(\d+)", name)
    if m:
        return int(m.group(1)), None, None

    return None, None, None


# ---------------------------------------------------------------------------
# Image helpers (lightweight; avoids torch/cv2 for simple metadata reads)
# ---------------------------------------------------------------------------

def _image_size(path: Path) -> tuple[int | None, int | None]:
    """Read image dimensions without loading full pixel data."""
    try:
        from PIL import Image
        with Image.open(path) as img:
            return img.size  # (width, height)
    except Exception:
        return None, None


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_preview(src: Path, dst: Path, max_size: tuple[int, int] = (320, 240)) -> bool:
    """Save a small JPEG thumbnail. Returns True on success."""
    try:
        from PIL import Image
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as img:
            img.thumbnail(max_size, Image.LANCZOS)
            img.convert("RGB").save(dst, "JPEG", quality=75, optimize=True)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Core processor
# ---------------------------------------------------------------------------

class DatasetProcessor:
    """Scan a dataset root directory, emit CSV + preview images."""

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        out_dir: Path,
        dataset_name: str,
        n_preview: int = 20,
        delete_originals: bool = False,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.dataset_name = dataset_name
        self.n_preview = n_preview
        self.delete_originals = delete_originals
        self.progress_cb = progress_cb

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_directory(
        self,
        root: Path,
        split: str = "unknown",
        type_map: dict[int, str] | None = None,
    ) -> ProcessingResult:
        """Process all images under *root* for a single split."""
        import time

        t0 = time.perf_counter()
        image_paths = sorted(
            p for p in root.rglob("*") if p.suffix.lower() in self.IMAGE_EXTS
        )
        total = len(image_paths)
        records: list[ImageRecord] = []
        errors: list[str] = []
        preview_saved = 0

        csv_path = self.out_dir / f"{self.dataset_name}_{split}.csv"
        preview_dir = self.out_dir / "previews" / self.dataset_name
        preview_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        for i, img_path in enumerate(image_paths):
            if self.progress_cb:
                self.progress_cb(i, total)
            try:
                rel = str(img_path.relative_to(root))
                identity, camera_id, frame_id = _parse_filename(rel)
                if type_map and identity is not None:
                    vtype = type_map.get(identity)
                else:
                    vtype = None

                width, height = _image_size(img_path)
                file_size = img_path.stat().st_size
                md5 = _md5(img_path)

                is_preview = preview_saved < self.n_preview
                preview_rel: str | None = None
                if is_preview:
                    thumb_name = f"{self.dataset_name}_{split}_{i:06d}.jpg"
                    thumb_dst = preview_dir / thumb_name
                    if _save_preview(img_path, thumb_dst):
                        preview_rel = str(thumb_dst.relative_to(self.out_dir))
                        preview_saved += 1

                uid = f"{self.dataset_name}:{split}:{rel}"
                rec = ImageRecord(
                    uid=uid,
                    dataset_name=self.dataset_name,
                    split=split,
                    relative_path=rel,
                    identity=identity,
                    camera_id=camera_id,
                    frame_id=frame_id,
                    vehicle_type=vtype,
                    width=width,
                    height=height,
                    file_size_bytes=file_size,
                    md5=md5,
                    is_preview=is_preview,
                    preview_path=preview_rel,
                )
                records.append(rec)

                if self.delete_originals:
                    img_path.unlink(missing_ok=True)

            except Exception as exc:
                errors.append(f"{img_path}: {exc}")

        self._write_csv(csv_path, records)

        if self.progress_cb:
            self.progress_cb(total, total)

        return ProcessingResult(
            dataset_name=self.dataset_name,
            total_images=total,
            processed=len(records),
            preview_count=preview_saved,
            csv_path=str(csv_path),
            preview_dir=str(preview_dir),
            errors=errors,
            elapsed_seconds=time.perf_counter() - t0,
        )

    def process_zip(self, zip_path: Path, split: str = "train") -> ProcessingResult:
        """Unpack a ZIP to a temp directory, then call process_directory."""
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp)
            # If the ZIP contains a single top-level folder, descend into it
            children = list(Path(tmp).iterdir())
            root = Path(tmp) if len(children) != 1 or not children[0].is_dir() else children[0]
            return self.process_directory(root, split=split)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_csv(path: Path, records: list[ImageRecord]) -> None:
        if not records:
            path.write_text("", encoding="utf-8")
            return
        fieldnames = list(records[0].to_row().keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in records:
                writer.writerow(rec.to_row())


# ---------------------------------------------------------------------------
# Convenience: process_existing_raw_data
# ---------------------------------------------------------------------------

def process_existing_raw(
    raw_dir: Path,
    processed_dir: Path,
    n_preview: int = 20,
    delete_originals: bool = False,
) -> list[ProcessingResult]:
    """Process all images already sitting in data/raw/ into CSVs.

    Detects sub-directories as dataset splits; flat images fall under 'default'.
    Returns one ProcessingResult per detected dataset / split combination.
    """
    results: list[ProcessingResult] = []
    sub_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]

    if sub_dirs:
        for sub in sub_dirs:
            dataset_name = sub.name
            proc = DatasetProcessor(
                out_dir=processed_dir,
                dataset_name=dataset_name,
                n_preview=n_preview,
                delete_originals=delete_originals,
            )
            result = proc.process_directory(sub, split="all")
            results.append(result)
    else:
        # Flat: treat data/raw itself as one dataset
        dataset_name = "raw"
        proc = DatasetProcessor(
            out_dir=processed_dir,
            dataset_name=dataset_name,
            n_preview=n_preview,
            delete_originals=delete_originals,
        )
        result = proc.process_directory(raw_dir, split="all")
        results.append(result)

    return results
