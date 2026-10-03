"""Center loss: pulls embeddings toward a learned per-identity centroid.

Triplet loss alone only pushes relative distances; center loss adds an
absolute pull toward class centroids, which tightens intra-identity clusters
and is standard practice alongside triplet + cross-entropy in Re-ID.
"""

from __future__ import annotations

from typing import Any


def build_center_loss(num_classes: int, embed_dim: int) -> Any:
    import torch

    class CenterLoss(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.centers = torch.nn.Parameter(torch.randn(num_classes, embed_dim) * 0.01)

        def forward(self, embeddings: Any, labels: Any) -> Any:
            centers_batch = self.centers[labels]
            return ((embeddings - centers_batch) ** 2).sum(dim=1).mean()

    return CenterLoss()
