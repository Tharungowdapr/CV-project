#!/usr/bin/env python
"""Generate and commit split manifests for every protocol.

Run this once per dataset. The manifests are versioned in git so a reviewer can
regenerate every number without guessing which identities were held out.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reiduq.data.splits import build_splits
from reiduq.data.veri776 import VeRi776


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="dataset root")
    ap.add_argument("--out", type=Path, default=Path("configs/splits"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    samples = VeRi776(args.root, split="train").load()

    protocols = [
        ("in_distribution", {}),
        ("leave_camera_out", {"held_out_cameras": ["c016", "c017", "c018", "c019", "c020"]}),
        ("unseen_types", {"held_out_types": ["truck", "bus", "van"]}),
    ]
    for protocol, extra in protocols:
        manifest = build_splits(
            samples, dataset="veri776", protocol=protocol, seed=args.seed, **extra
        )
        path = manifest.save(args.out / f"veri776_{protocol}_seed{args.seed}.json")
        print(f"{protocol:20s} -> {path}  (hash {manifest.content_hash})")


if __name__ == "__main__":
    main()
