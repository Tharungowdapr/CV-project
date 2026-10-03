"""Re-ID encoder construction and embedding extraction.

The backbone is deliberately ordinary. A calibration finding obtained on an
unusual encoder invites the objection that the effect is an artefact of the
architecture, so the encoder follows standard practice and the interesting part
stays downstream.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from reiduq.core.registry import BACKBONES


def build_encoder(name: str, embed_dim: int = 512, pretrained: bool = True, **kw: Any) -> Any:
    """Construct a registered backbone; falls back to torchvision ResNet-50 + BNNeck.

    Registered backbones (OSNetAIN, ResNet50IBN, TransReID) are plain wrapper
    classes exposing a ``.build()`` method that returns the actual
    ``nn.Module`` - the wrapper itself is not one, so it must be unwrapped
    here rather than handed to ``.to(device)`` as-is.
    """
    if name in BACKBONES.keys():
        wrapper = BACKBONES.build(name, embed_dim=embed_dim, pretrained=pretrained, **kw)
        try:
            return wrapper.build()  # type: ignore[attr-defined]
        except ImportError as exc:
            # osnet_ain needs torchreid, transreid can use timm - both optional.
            # Degrading to the generic backbone keeps `make train` runnable
            # without every optional vision package installed; a real
            # experiment run should install the intended package instead of
            # relying on this fallback silently.
            from reiduq.core.logging import get_logger

            get_logger(__name__).warning(
                "build_encoder.optional_dependency_missing", backbone=name, error=str(exc)
            )
    return _generic_resnet50(embed_dim, pretrained)


def _generic_resnet50(embed_dim: int, pretrained: bool) -> Any:
    import torch
    import torchvision

    weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
    backbone = torchvision.models.resnet50(weights=weights)
    backbone.fc = torch.nn.Identity()
    return torch.nn.Sequential(
        backbone,
        torch.nn.BatchNorm1d(2048),
        torch.nn.Linear(2048, embed_dim, bias=False),
    )


def l2_normalise(x: np.ndarray) -> np.ndarray:
    """(N, D) -> (N, D) with unit rows; cosine similarity is then a dot product."""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


@BACKBONES.register("osnet_ain")
class OSNetAIN:
    """Thin wrapper over torchreid's OSNet-AIN; the lightweight Re-ID standard.

    torchreid's ``build_model`` returns a classification model with its own
    feature dimension (512 for osnet_ain_x1_0, but not guaranteed to equal
    ``embed_dim`` for other variants), so the classifier head is stripped and
    an explicit BNNeck + projection to ``embed_dim`` is added on top - the
    same pattern every other backbone in this module follows, so the trainer
    can treat all of them identically.
    """

    def __init__(self, embed_dim: int = 512, pretrained: bool = True) -> None:
        self.embed_dim, self.pretrained = embed_dim, pretrained

    def build(self) -> Any:
        import torch
        import torchreid  # optional dependency; ImportError is handled by build_encoder

        full = torchreid.models.build_model(
            name="osnet_ain_x1_0", num_classes=1, pretrained=self.pretrained
        )
        full.classifier = torch.nn.Identity()  # torchreid OSNet exposes .featuremaps + .classifier
        feat_dim = full.feature_dim if hasattr(full, "feature_dim") else 512
        return torch.nn.Sequential(
            full,
            torch.nn.BatchNorm1d(feat_dim),
            torch.nn.Linear(feat_dim, self.embed_dim, bias=False),
        )
