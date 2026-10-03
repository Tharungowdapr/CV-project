#!/usr/bin/env python
"""Build the occluder bank from real segmented objects in the dataset.

Using real objects, with their own texture and compression statistics, is what
makes the synthetic occlusion protocol defensible.
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import numpy as np

from reiduq.data.occlusion.occluder_bank import OccluderBank, OccluderPatch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, required=True, help="directory of source frames")
    ap.add_argument("--out", type=Path, default=Path("data/processed/occluders"))
    ap.add_argument("--max-patches", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from reiduq.detection.yolov8 import YOLOv8Detector
    from reiduq.segmentation.yolov8_seg import YOLOv8Segmenter

    detector = YOLOv8Detector(device=args.device)
    segmenter = YOLOv8Segmenter(device=args.device)
    bank = OccluderBank(args.out)

    import cv2

    for i, path in enumerate(sorted(args.images.glob("*.jpg"))):
        if len(bank) >= args.max_patches:
            break
        frame = cv2.imread(str(path))[:, :, ::-1].copy()
        dets = detector.detect(frame, frame_id=i, camera_id="bank")
        masks = segmenter.segment(frame, dets)
        for det, mask in zip(dets, masks, strict=True):
            if mask.sum() < 800:
                continue
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            bank.add(
                OccluderPatch(
                    patch_id=uuid.uuid4().hex[:12],
                    rgb=frame[y1:y2, x1:x2].copy(),
                    mask=mask.astype(np.uint8),
                    occluder_class=det.cls,
                    source_image=path.name,
                )
            )

    print(f"occluder bank now holds {len(bank)} patches at {args.out}")


if __name__ == "__main__":
    main()
