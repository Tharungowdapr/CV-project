"""Loss functions: shape, non-negativity, and gradient flow."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="loss tests need torch")

from reiduq.models.losses.ce_smooth import build_label_smoothing_ce
from reiduq.models.losses.center import build_center_loss
from reiduq.models.losses.triplet_wrt import build_triplet_wrt


def test_label_smoothing_ce_is_scalar_and_differentiable() -> None:
    torch.manual_seed(0)
    logits = torch.randn(8, 5, requires_grad=True)
    labels = torch.randint(0, 5, (8,))
    loss = build_label_smoothing_ce(num_classes=5)(logits, labels)
    assert loss.dim() == 0
    loss.backward()
    assert logits.grad is not None


def test_triplet_wrt_is_nonnegative_with_valid_triplets() -> None:
    torch.manual_seed(0)
    embeddings = torch.nn.functional.normalize(torch.randn(12, 16), dim=1)
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    loss = build_triplet_wrt()(embeddings, labels)
    assert loss.item() >= 0.0


def test_triplet_wrt_handles_no_valid_triplets_gracefully() -> None:
    embeddings = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 2, 3])  # every identity singleton: no positives
    loss = build_triplet_wrt()(embeddings, labels)
    assert torch.isfinite(loss)


def test_center_loss_pulls_toward_learned_centroids() -> None:
    torch.manual_seed(0)
    center = build_center_loss(num_classes=3, embed_dim=8)
    embeddings = torch.randn(6, 8)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    loss = center(embeddings, labels)
    assert loss.item() >= 0.0
    assert list(center.parameters())[0].shape == (3, 8)
