"""Train/eval image transforms.

Random erasing is kept for accuracy (it is standard Re-ID practice) but is
explicitly NOT the occlusion protocol used for the calibration study - see
data/occlusion/compositor.py for that. Conflating the two would let an
ablation's result be explained by either mechanism, which is exactly the
ambiguity configs/ablation/ exists to rule out.
"""

from __future__ import annotations

from typing import Any


def build_train_transform(image_size: tuple[int, int], random_erasing_p: float) -> Any:
    import torchvision.transforms as T

    return T.Compose(
        [
            T.Resize(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(p=random_erasing_p, value="random"),
        ]
    )


def build_eval_transform(image_size: tuple[int, int]) -> Any:
    import torchvision.transforms as T

    return T.Compose(
        [
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
