#!/usr/bin/env python
"""Build a searchable gallery index from a dataset.

This is the one-time (or periodic) offline job that makes both search modes
work: it embeds every image in a dataset, best-effort-OCRs a plate crop, and
writes a FAISS index + SQLite metadata store to models/gallery/.

Usage:
  python scripts/build_gallery_index.py --dataset veri776 --root data/raw/VeRi
  python scripts/build_gallery_index.py --dataset veriwild --root data/raw/VeriWild --limit 5000

--limit is for a fast first pass to confirm the pipeline works before
committing to the multi-hour full-dataset run.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["veri776", "veriwild"], required=True)
    ap.add_argument("--root", type=Path, required=True, help="dataset root, e.g. data/raw/VeRi")
    ap.add_argument("--split", default="train", help="which split to index (default: train - the largest)")
    ap.add_argument("--out", type=Path, default=Path("models/gallery"))
    ap.add_argument("--embed-dim", type=int, default=512)
    ap.add_argument("--backbone", default="osnet_ain")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--checkpoint", type=Path, default=None, help="trained encoder checkpoint; omit to index with an untrained encoder for a pipeline smoke test")
    ap.add_argument("--limit", type=int, default=None, help="index only the first N images - for a fast smoke test")
    ap.add_argument("--no-ocr", action="store_true", help="skip plate OCR entirely (faster; plate search will find nothing)")
    args = ap.parse_args()

    from reiduq.core.logging import configure_logging, get_logger
    from reiduq.core.registry import DATASETS
    from reiduq.models.builder import build_encoder
    from reiduq.search.indexer import GalleryIndexer
    from reiduq.search.plate_ocr import PlateReader

    configure_logging("INFO", "console")
    log = get_logger(__name__)

    dataset = DATASETS.build(args.dataset, root=args.root, split=args.split)  # type: ignore[attr-defined]

    encoder = build_encoder(args.backbone, args.embed_dim, pretrained=args.checkpoint is None)
    if args.checkpoint:
        import torch

        blob = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        encoder.load_state_dict(blob["state_dict"])
        log.info("build_gallery_index.checkpoint_loaded", path=str(args.checkpoint))
    else:
        log.warning(
            "build_gallery_index.no_checkpoint",
            note="indexing with an untrained encoder - embeddings will not be meaningful; "
                 "this is fine for a pipeline smoke test, not for real search results",
        )
    encoder = encoder.to(args.device)

    plate_reader = None if args.no_ocr else PlateReader()
    if plate_reader is not None and not plate_reader.available:
        log.warning(
            "build_gallery_index.ocr_unavailable",
            note="easyocr not installed - plate_text will be null for every entry; "
                 "install with `pip install easyocr` or pass --no-ocr to silence this",
        )

    indexer = GalleryIndexer(encoder, args.embed_dim, device=args.device, plate_reader=plate_reader)
    args.out.mkdir(parents=True, exist_ok=True)
    index, store = indexer.build(
        dataset,
        index_out=args.out / "index.faiss",
        store_out=args.out / "store.db",
        limit=args.limit,
    )
    log.info("build_gallery_index.done", entries=store.count(), out=str(args.out))
    print(f"\nIndexed {store.count()} entries -> {args.out}")
    print("Start the API and try:")
    print('  curl -H "Authorization: Bearer <key>" "http://localhost:8000/v1/search/plate?plate_query=AB1234"')


if __name__ == "__main__":
    main()
