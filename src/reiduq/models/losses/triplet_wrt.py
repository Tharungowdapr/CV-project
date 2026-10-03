"""Weighted-regularised triplet loss.

Standard hard-mining triplet loss is unstable early in training because the
hardest negative in a batch is often noise. Weighted-regularised triplet
(Ristani-style soft weighting) uses every negative in the batch, weighted by
how hard it is, instead of only the single hardest one - smoother gradients,
less sensitive to the margin hyperparameter.
"""

from __future__ import annotations

from typing import Any


def build_triplet_wrt() -> Any:
    import torch

    class WeightedRegularisedTriplet(torch.nn.Module):
        def forward(self, embeddings: Any, labels: Any) -> Any:
            dist = torch.cdist(embeddings, embeddings, p=2)
            same = labels.unsqueeze(0) == labels.unsqueeze(1)
            eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
            pos_mask, neg_mask = same & ~eye, ~same

            # Soft-weight every negative/positive by distance rather than
            # picking only the single hardest one.
            pos_w = torch.softmax(dist.masked_fill(~pos_mask, -1e9), dim=1)
            neg_w = torch.softmax(-dist.masked_fill(~neg_mask, -1e9), dim=1)
            pos_dist = (dist * pos_w * pos_mask).sum(dim=1)
            neg_dist = (dist * neg_w * neg_mask).sum(dim=1)

            valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
            loss = torch.nn.functional.softplus(pos_dist - neg_dist)
            return loss[valid].mean() if valid.any() else loss.mean() * 0.0

    return WeightedRegularisedTriplet()
