#!/usr/bin/env python
"""Process a dataset (folder or ZIP) → CSV manifest + preview thumbnails.

This is the offline CLI companion to the /v1/dataset/ingest API endpoint.
It extracts all image metadata into a memory-efficient CSV and saves only
a configurable number of small preview thumbnails, making it safe to delete
or archive the original images.

Usage examples
--------------
# Process a flat folder of images (e.g. data/raw/)
python scripts/process_dataset.py --name gallery_raw --root data/raw --split all --preview 20

# Process a ZIP archive
python scripts/process_dataset.py --name veri776_train --zip path/to/veri776.zip --split train

# Process ALL existing raw data (auto-detect subdirectories as datasets)
python scripts/process_dataset.py --all-raw

# Process and DELETE originals to free disk space
python scripts/process_dataset.py --name gallery_raw --root data/raw --delete-originals
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make sure we can import reiduq from the repo root regardless of install state
_repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(_repo_root / "src"))


def _print_result(result) -> None:  # type: ignore[type-arg]
    """Pretty-print a ProcessingResult."""
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  Dataset : {result.dataset_name}")
    print(f"  Images  : {result.total_images} found, {result.processed} processed")
    print(f"  Previews: {result.preview_count}")
    print(f"  CSV     : {result.csv_path}")
    print(f"  Elapsed : {result.elapsed_seconds:.2f}s")
    if result.errors:
        print(f"  Errors  : {len(result.errors)}")
        for e in result.errors[:5]:
            print(f"    - {e}")
        if len(result.errors) > 5:
            print(f"    ... and {len(result.errors) - 5} more")
    print(f"{sep}\n")



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    source = ap.add_mutually_exclusive_group()
    source.add_argument("--root", type=Path, help="Path to dataset root directory")
    source.add_argument("--zip", type=Path, help="Path to a ZIP archive")
    source.add_argument(
        "--all-raw",
        action="store_true",
        help="Process everything under data/raw/ (auto-detect datasets)",
    )

    ap.add_argument("--name", default="dataset", help="Dataset name (used as CSV filename prefix)")
    ap.add_argument("--split", default="all", help="Split label written into the CSV (default: all)")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for CSVs and previews (default: data/processed)",
    )
    ap.add_argument(
        "--preview",
        type=int,
        default=20,
        help="Number of preview thumbnail images to keep per split (default: 20)",
    )
    ap.add_argument(
        "--delete-originals",
        action="store_true",
        help="Delete source images after extracting metadata (IRREVERSIBLE)",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (for scripting / API integration)",
    )

    args = ap.parse_args()

    from reiduq.data.dataset_processor import DatasetProcessor, process_existing_raw

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all_raw:
        raw_dir = (_repo_root / "data" / "raw").resolve()
        print(f"Processing all datasets under {raw_dir} …")
        results = process_existing_raw(
            raw_dir=raw_dir,
            processed_dir=out_dir,
            n_preview=args.preview,
            delete_originals=args.delete_originals,
        )
    elif args.zip:
        proc = DatasetProcessor(
            out_dir=out_dir,
            dataset_name=args.name,
            n_preview=args.preview,
            delete_originals=False,  # deleting a ZIP we didn't create is dangerous
        )
        print(f"Unpacking {args.zip} …")
        result = proc.process_zip(args.zip.resolve(), split=args.split)
        results = [result]
    elif args.root:
        proc = DatasetProcessor(
            out_dir=out_dir,
            dataset_name=args.name,
            n_preview=args.preview,
            delete_originals=args.delete_originals,
        )
        print(f"Scanning {args.root} …")
        result = proc.process_directory(args.root.resolve(), split=args.split)
        results = [result]
    else:
        ap.error("Provide one of --root, --zip, or --all-raw")

    if args.json:
        import dataclasses
        print(json.dumps([dataclasses.asdict(r) for r in results], indent=2))
    else:
        for r in results:
            _print_result(r)
        print(f"All done. CSVs and previews written to: {out_dir}")


if __name__ == "__main__":
    main()
