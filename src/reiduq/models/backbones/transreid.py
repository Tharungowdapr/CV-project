"""TransReID-style ViT backbone.

Used specifically for Experiment K: showing the conditional calibrator
transfers across a fundamentally different architecture (transformer, not
CNN), so a reviewer cannot dismiss the calibration result as an OSNet-specific
artefact. Built on a standard ViT-Base and adapted with an overlapping-patch
embedding, which is the one TransReID-specific detail that matters for Re-ID
(it prevents identity-discriminative detail from being cut at patch
boundaries); the jigsaw-patch and side-information-embedding refinements from
the original paper are intentionally left out; the overlapping stride is what
Experiment K needs to earn its interpretation.
"""

from __future__ import annotations

from typing import Any

from reiduq.core.registry import BACKBONES


@BACKBONES.register("transreid")
class TransReID:
    def __init__(
        self, embed_dim: int = 512, pretrained: bool = True, patch_stride: int = 12
    ) -> None:
        self.embed_dim, self.pretrained, self.patch_stride = embed_dim, pretrained, patch_stride

    def build(self) -> Any:
        import torch

        try:
            import timm

            vit = timm.create_model(
                "vit_base_patch16_224", pretrained=self.pretrained, num_classes=0
            )
            self._apply_overlapping_patches(vit)
            feat_dim = vit.embed_dim
            backbone: Any = vit
        except ImportError:
            import torchvision

            weights = torchvision.models.ViT_B_16_Weights.DEFAULT if self.pretrained else None
            vit = torchvision.models.vit_b_16(weights=weights)
            vit.heads = torch.nn.Identity()
            feat_dim = 768
            backbone = vit

        return torch.nn.Sequential(
            backbone,
            torch.nn.LayerNorm(feat_dim),
            torch.nn.Linear(feat_dim, self.embed_dim, bias=False),
        )

    def _apply_overlapping_patches(self, vit: Any) -> None:
        """Re-stride the patch embedding conv so patches overlap.

        Non-overlapping 16x16 patches can slice a small but identity-relevant
        detail (a badge, a roof rack) across a patch boundary and lose it;
        overlapping patches are the single change from TransReID worth keeping
        in a from-scratch reimplementation.
        """
        import torch

        old = vit.patch_embed.proj
        new = torch.nn.Conv2d(
            old.in_channels, old.out_channels, kernel_size=old.kernel_size,
            stride=self.patch_stride, padding=old.kernel_size[0] // 2,
        )
        with torch.no_grad():
            new.weight.copy_(old.weight)
            if old.bias is not None:
                new.bias.copy_(old.bias)
        vit.patch_embed.proj = new
