#!/usr/bin/env python
"""Generate a synthetic VeRi-776 structured dataset manifest CSV.

Produces 1,200+ realistic records across train / query / gallery splits,
with full metadata: vehicle identity, camera ID, frame ID, vehicle type,
license plate text, visibility, confidence, etc.

The records mimic the VeRi-776 naming convention:
    <vehicle_id>_c<camera_id>_<frame_id>_<seq>.jpg

Usage
-----
    python scripts/generate_veri_dataset.py
    python scripts/generate_veri_dataset.py --records 2000 --seed 99
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import string
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_OUT_DIR = _REPO_ROOT / "data" / "processed"

# ── Configuration ─────────────────────────────────────────────────────────────

N_IDENTITIES = 200          # unique vehicle identities (VeRi-776 has 776)
N_CAMERAS = 20              # surveillance cameras
N_TRAIN_IDS = 160           # identities in training split
N_QUERY = 200               # query images (one per identity subset)
N_GALLERY = 800             # gallery images
N_TRAIN = 1200              # training images

VEHICLE_TYPES = ["sedan", "suv", "truck", "van", "bus", "motorcycle", "pickup"]
COLORS = ["black", "white", "silver", "red", "blue", "green", "yellow", "gray", "brown", "gold"]
BRANDS = ["Toyota", "Honda", "Ford", "BMW", "Audi", "Mercedes", "Hyundai", "Kia", "Nissan", "VW"]
PLATE_STATES = ["MH", "DL", "KA", "TN", "AP", "GJ", "UP", "RJ", "WB", "MP"]
VISIBILITY_LEVELS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _plate(rng: random.Random) -> str:
    state = rng.choice(PLATE_STATES)
    digits1 = rng.randint(10, 99)
    letters = "".join(rng.choices(string.ascii_uppercase, k=2))
    digits2 = rng.randint(1000, 9999)
    return f"{state}-{digits1:02d}-{letters}-{digits2:04d}"


def _fake_md5(seed: str) -> str:
    return hashlib.md5(seed.encode()).hexdigest()


def _timestamp(rng: random.Random, base: datetime) -> str:
    offset = timedelta(seconds=rng.randint(0, 86400 * 365))
    return (base + offset).isoformat()


@dataclass
class VehicleMeta:
    identity: int
    vehicle_type: str
    color: str
    brand: str
    plate: str


def _build_vehicle_pool(n: int, rng: random.Random) -> dict[int, VehicleMeta]:
    pool: dict[int, VehicleMeta] = {}
    for vid in range(1, n + 1):
        pool[vid] = VehicleMeta(
            identity=vid,
            vehicle_type=rng.choice(VEHICLE_TYPES),
            color=rng.choice(COLORS),
            brand=rng.choice(BRANDS),
            plate=_plate(rng),
        )
    return pool


def _make_filename(vid: int, cam: int, frame: int, seq: int) -> str:
    return f"{vid:04d}_c{cam:03d}_{frame:06d}_{seq:02d}.jpg"


def _make_record(
    *,
    split: str,
    dataset_name: str,
    vid: int,
    cam: int,
    frame: int,
    seq: int,
    meta: VehicleMeta,
    rng: random.Random,
    base_time: datetime,
    idx: int,
    n_preview: int = 50,
) -> dict:
    filename = _make_filename(vid, cam, frame, seq)
    rel_path = f"images/{split}/{filename}"
    uid = f"{dataset_name}:{split}:{rel_path}"
    is_preview = idx < n_preview
    width = rng.choice([128, 256, 320, 640])
    height = rng.choice([256, 128, 240, 480])
    file_size = rng.randint(4_000, 250_000)
    visibility = rng.choice(VISIBILITY_LEVELS)
    plate_conf = round(rng.uniform(0.55, 0.99), 4)
    timestamp = _timestamp(rng, base_time)

    return {
        "uid": uid,
        "dataset_name": dataset_name,
        "split": split,
        "relative_path": rel_path,
        "identity": vid,
        "camera_id": f"c{cam:03d}",
        "frame_id": frame,
        "sequence_id": seq,
        "vehicle_type": meta.vehicle_type,
        "color": meta.color,
        "brand": meta.brand,
        "plate_text": meta.plate,
        "plate_confidence": plate_conf,
        "visibility": visibility,
        "width": width,
        "height": height,
        "file_size_bytes": file_size,
        "md5": _fake_md5(uid),
        "is_preview": is_preview,
        "preview_path": f"previews/{dataset_name}/{dataset_name}_{split}_{idx:06d}.jpg" if is_preview else "",
        "timestamp": timestamp,
    }


FIELDNAMES = [
    "uid", "dataset_name", "split", "relative_path",
    "identity", "camera_id", "frame_id", "sequence_id",
    "vehicle_type", "color", "brand", "plate_text", "plate_confidence",
    "visibility", "width", "height", "file_size_bytes", "md5",
    "is_preview", "preview_path", "timestamp",
]


def generate(
    out_dir: Path,
    dataset_name: str = "veri776",
    seed: int = 42,
    n_train: int = N_TRAIN,
    n_query: int = N_QUERY,
    n_gallery: int = N_GALLERY,
    n_preview: int = 50,
) -> dict[str, str]:
    """Generate CSV manifests for train/query/gallery splits.

    Returns a dict mapping split name → CSV path.
    """
    rng = random.Random(seed)
    base_time = datetime(2023, 1, 1, tzinfo=UTC)
    vehicle_pool = _build_vehicle_pool(N_IDENTITIES, rng)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── TRAINING split ──────────────────────────────────────────────
    train_ids = list(range(1, N_TRAIN_IDS + 1))
    train_rows = []
    frame_counter: dict[tuple, int] = {}  # (vid, cam) -> last frame

    for i in range(n_train):
        vid = rng.choice(train_ids)
        cam = rng.randint(1, N_CAMERAS)
        key = (vid, cam)
        frame_counter[key] = frame_counter.get(key, 0) + rng.randint(5, 50)
        frame = frame_counter[key]
        seq = rng.randint(0, 5)
        train_rows.append(_make_record(
            split="train", dataset_name=dataset_name,
            vid=vid, cam=cam, frame=frame, seq=seq,
            meta=vehicle_pool[vid], rng=rng,
            base_time=base_time, idx=i, n_preview=n_preview,
        ))

    # ── QUERY split ─────────────────────────────────────────────────
    query_ids = list(range(N_TRAIN_IDS + 1, N_IDENTITIES + 1))
    # Pad with some train ids if needed
    while len(query_ids) < n_query:
        query_ids += list(range(1, N_TRAIN_IDS + 1))

    query_rows = []
    for i in range(n_query):
        vid = rng.choice(query_ids)
        cam = rng.randint(1, N_CAMERAS)
        frame = rng.randint(100, 5000)
        seq = 0
        query_rows.append(_make_record(
            split="query", dataset_name=dataset_name,
            vid=vid, cam=cam, frame=frame, seq=seq,
            meta=vehicle_pool[vid], rng=rng,
            base_time=base_time, idx=i, n_preview=n_preview,
        ))

    # ── GALLERY split ────────────────────────────────────────────────
    gallery_rows = []
    for i in range(n_gallery):
        vid = rng.randint(1, N_IDENTITIES)
        cam = rng.randint(1, N_CAMERAS)
        frame = rng.randint(100, 9999)
        seq = rng.randint(0, 5)
        gallery_rows.append(_make_record(
            split="gallery", dataset_name=dataset_name,
            vid=vid, cam=cam, frame=frame, seq=seq,
            meta=vehicle_pool[vid], rng=rng,
            base_time=base_time, idx=i, n_preview=n_preview,
        ))

    # ── Write CSVs ───────────────────────────────────────────────────
    result: dict[str, str] = {}
    for split, rows in [("train", train_rows), ("query", query_rows), ("gallery", gallery_rows)]:
        csv_path = out_dir / f"{dataset_name}_{split}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        result[split] = str(csv_path)
        print(f"  OK {split:8s}  {len(rows):5,} rows  ->  {csv_path.name}")

    # ── Write combined all.csv ───────────────────────────────────────
    all_rows = train_rows + query_rows + gallery_rows
    all_csv = out_dir / f"{dataset_name}_all.csv"
    with open(all_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)
    result["all"] = str(all_csv)
    print(f"  OK {'all':8s}  {len(all_rows):5,} rows  ->  {all_csv.name}")

    # ── Create placeholder preview directories ────────────────────────
    preview_dir = out_dir / "previews" / dataset_name
    preview_dir.mkdir(parents=True, exist_ok=True)

    # Create tiny placeholder preview images using PIL if available
    try:
        from PIL import Image, ImageDraw, ImageFont
        import random as _r
        _r2 = _r.Random(seed + 1)
        
        all_preview_rows = [r for r in all_rows if r["is_preview"] and r["preview_path"]]
        for row in all_preview_rows[:n_preview * 3]:  # up to 3 splits worth
            preview_rel = row["preview_path"]
            dst = out_dir / preview_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                # Create a colored placeholder thumbnail
                color = {
                    "black": (30, 30, 30), "white": (220, 220, 220),
                    "silver": (180, 180, 190), "red": (180, 40, 40),
                    "blue": (40, 80, 180), "green": (40, 140, 60),
                    "yellow": (200, 180, 30), "gray": (120, 120, 120),
                    "brown": (140, 80, 40), "gold": (200, 160, 40),
                }.get(row["color"], (100, 100, 100))
                img = Image.new("RGB", (128, 96), color=color)
                draw = ImageDraw.Draw(img)
                draw.text((5, 5), f"ID:{row['identity']}", fill=(255, 255, 255))
                draw.text((5, 20), row["plate_text"][:10], fill=(200, 200, 0))
                draw.text((5, 35), row["vehicle_type"][:8], fill=(180, 180, 255))
                img.save(dst, "JPEG", quality=70)
        print(f"  OK Preview images written to {preview_dir}")
    except ImportError:
        print("  WARNING: PIL not available - preview thumbnails skipped (install Pillow)")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=_OUT_DIR, help="Output directory")
    ap.add_argument("--name", default="veri776", help="Dataset name prefix")
    ap.add_argument("--train", type=int, default=N_TRAIN, help="Number of training records")
    ap.add_argument("--query", type=int, default=N_QUERY, help="Number of query records")
    ap.add_argument("--gallery", type=int, default=N_GALLERY, help="Number of gallery records")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--preview", type=int, default=50, help="Number of preview thumbnails per split")
    args = ap.parse_args()

    total = args.train + args.query + args.gallery
    print(f"Generating synthetic VeRi-776 dataset: {total:,} records")
    print(f"  Identities : {N_IDENTITIES}")
    print(f"  Cameras    : {N_CAMERAS}")
    print(f"  Splits     : train={args.train}, query={args.query}, gallery={args.gallery}")
    print(f"  Output     : {args.out}\n")

    paths = generate(
        out_dir=args.out.resolve(),
        dataset_name=args.name,
        seed=args.seed,
        n_train=args.train,
        n_query=args.query,
        n_gallery=args.gallery,
        n_preview=args.preview,
    )

    print(f"\nAll done — {total:,} records written")
    print("  CSVs:")
    for split, path in paths.items():
        print(f"    {split}: {path}")


if __name__ == "__main__":
    main()
