"""Three estimators for the visibility ratio v = |M_visible| / |M_amodal|.

The amodal (unoccluded) extent is not observable in general, so which estimator
is valid depends on the protocol. Each is explicit about its assumption, and
the single-image estimator reports an error band that is propagated into the
analysis rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reiduq.visibility.parts import part_visibility


@dataclass(frozen=True)
class VisibilityEstimate:
    value: float
    method: str
    uncertainty: float = 0.0  # +/- band on the estimate


def from_synthetic(visible_mask: np.ndarray, original_mask: np.ndarray) -> VisibilityEstimate:
    """Exact: the unoccluded mask is known before compositing."""
    denom = float((original_mask > 0).sum())
    if denom <= 0:
        raise ValueError("original mask is empty")
    v = float((visible_mask > 0).sum()) / denom
    return VisibilityEstimate(float(np.clip(v, 0.0, 1.0)), "synthetic_exact", 0.0)


def from_tracklet(
    masks: list[np.ndarray], box_heights: list[float], index: int
) -> VisibilityEstimate:
    """A tracked vehicle is usually unoccluded at some point in its trajectory.

    Areas are normalised by the squared box height so that a vehicle appearing
    larger as it approaches the camera is not mistaken for one becoming less
    occluded.
    """
    if not masks:
        raise ValueError("empty tracklet")
    areas = np.array([float((m > 0).sum()) for m in masks], dtype=np.float64)
    heights = np.array(box_heights, dtype=np.float64)
    heights[heights <= 0] = 1.0
    normalised = areas / (heights**2)
    peak = float(normalised.max())
    if peak <= 0:
        return VisibilityEstimate(1.0, "tracklet_peak", 0.5)
    v = float(np.clip(normalised[index] / peak, 0.0, 1.0))
    # If the peak frame is itself occluded, v is optimistic. Spread of the top
    # quartile is a cheap proxy for how much we can trust the peak.
    top = np.sort(normalised)[-max(1, len(normalised) // 4) :]
    unc = float(np.std(top) / peak) if peak > 0 else 0.5
    return VisibilityEstimate(v, "tracklet_peak", min(unc, 0.5))


def features_for_regression(mask: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """(10,) float32 feature vector for the single-image visibility regressor."""
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    area = float((mask > 0).sum())
    fill = area / (bw * bh)
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return np.zeros(10, dtype=np.float32)
    spread_x = float(xs.std() / bw)
    spread_y = float(ys.std() / bh)
    return np.concatenate(
        [
            np.array([fill, spread_x, spread_y, bw / bh], dtype=np.float32),
            part_visibility(mask),
        ]
    )


class SingleImageVisibilityRegressor:
    """Ridge regression from mask geometry to v, trained on the synthetic set.

    Deliberately simple: it is a measurement instrument, not a contribution, and
    a small closed-form model is easier to justify and audit than a network.
    """

    def __init__(self) -> None:
        self.weights: np.ndarray | None = None
        self.residual_std: float = 0.0

    def fit(self, features: np.ndarray, targets: np.ndarray, alpha: float = 1.0) -> None:
        x = np.hstack([features, np.ones((len(features), 1), dtype=np.float32)])
        gram = x.T @ x + alpha * np.eye(x.shape[1], dtype=np.float32)
        self.weights = np.linalg.solve(gram, x.T @ targets)
        self.residual_std = float(np.std(targets - x @ self.weights))

    def predict(self, features: np.ndarray) -> VisibilityEstimate:
        if self.weights is None:
            raise RuntimeError("regressor not fitted")
        x = np.concatenate([features, [1.0]]).astype(np.float32)
        v = float(np.clip(x @ self.weights, 0.0, 1.0))
        return VisibilityEstimate(v, "single_image_regression", self.residual_std)
