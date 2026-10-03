"""ResNet-50-IBN-a backbone: instance-batch-normalisation for domain robustness.

IBN layers mix instance norm (style-invariant) and batch norm (content-
discriminative) in the early blocks, which is specifically useful here: it
reduces sensitivity to per-camera colour/contrast style, which is part of what
the camera-domain shift the calibrator has to handle is made of. Falls back to
plain ResNet-50 if the ``resnet_ibn`` package is not installed, since IBN is an
optional refinement, not a hard requirement to run the pipeline.
"""

from __future__ import annotations

from typing import Any

from reiduq.core.registry import BACKBONES


@BACKBONES.register("resnet50_ibn")
class ResNet50IBN:
    def __init__(self, embed_dim: int = 512, pretrained: bool = True) -> None:
        self.embed_dim, self.pretrained = embed_dim, pretrained

    def build(self) -> Any:
        import torch

        try:
            from resnet_ibn import resnet50_ibn_a  # type: ignore[import-not-found]

            backbone = resnet50_ibn_a(pretrained=self.pretrained)
            backbone.fc = torch.nn.Identity()
            feat_dim = 2048
        except ImportError:
            import torchvision

            weights = torchvision.models.ResNet50_Weights.DEFAULT if self.pretrained else None
            backbone = torchvision.models.resnet50(weights=weights)
            backbone.fc = torch.nn.Identity()
            feat_dim = 2048

        return torch.nn.Sequential(
            backbone,
            torch.nn.BatchNorm1d(feat_dim),
            torch.nn.Linear(feat_dim, self.embed_dim, bias=False),
        )
