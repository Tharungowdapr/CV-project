"""Label-smoothed cross-entropy on identity logits.

Smoothing keeps the classifier from driving logits to extremes, which in turn
keeps the raw similarity scores from becoming needlessly overconfident before
calibration ever sees them - a small thing that makes the calibrator's job
slightly easier and is standard Re-ID practice regardless.
"""

from __future__ import annotations

from typing import Any


def build_label_smoothing_ce(num_classes: int, epsilon: float = 0.1) -> Any:
    import torch

    class LabelSmoothingCE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.eps = epsilon
            self.n = num_classes

        def forward(self, logits: Any, targets: Any) -> Any:
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            with torch.no_grad():
                smooth = torch.full_like(log_probs, self.eps / (self.n - 1))
                smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.eps)
            return (-smooth * log_probs).sum(dim=1).mean()

    return LabelSmoothingCE()
