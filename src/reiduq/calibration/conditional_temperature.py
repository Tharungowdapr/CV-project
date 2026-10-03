"""Conditional temperature T(v, c) - the primary method contribution.

A ~6k-parameter MLP predicts a per-sample temperature from the conditioning
vector built in ``features.py``, instead of fitting one global scalar. The
encoder stays frozen; only this head is trained, on the calibration split.

Three implementation details matter more than they look:

* ``softplus`` guarantees T > 0 without a clamp inside the optimisation.
* T is clamped to sane bounds afterwards and the clamp rate is logged -
  persistent clamping means a feature-scaling bug, not a hard dataset.
* The feature mean/std are saved with the checkpoint. Applying a calibrator
  with mismatched normalisation is the most likely silent failure in this whole
  system, and it produces plausible-looking numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from reiduq.calibration.base import BaseCalibrator, softmax
from reiduq.core.exceptions import CalibrationNotFittedError
from reiduq.calibration.features import FEATURE_DIM
from reiduq.core.logging import get_logger
from reiduq.core.registry import CALIBRATORS

log = get_logger(__name__)


def _torch() -> Any:
    import torch

    return torch


@CALIBRATORS.register("conditional_temperature")
class ConditionalTemperature(BaseCalibrator):
    name = "conditional_temperature"
    requires_features = True

    def __init__(
        self,
        hidden_dim: int = 64,
        epochs: int = 200,
        lr: float = 1e-3,
        bounds: tuple[float, float] = (0.05, 20.0),
        dropout: float = 0.1,
        patience: int = 20,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.hidden_dim, self.epochs, self.lr = hidden_dim, epochs, lr
        self.bounds, self.dropout, self.patience = bounds, dropout, patience
        self.seed, self.device = seed, device
        self.net: Any | None = None
        self.feature_mean: np.ndarray = np.zeros(FEATURE_DIM, dtype=np.float32)
        self.feature_std: np.ndarray = np.ones(FEATURE_DIM, dtype=np.float32)
        self.clamp_rate: float = 0.0

    def _build(self) -> Any:
        torch = _torch()
        torch.manual_seed(self.seed)
        return torch.nn.Sequential(
            torch.nn.Linear(FEATURE_DIM, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_dim, 1),
            torch.nn.Softplus(),
        ).to(self.device)

    def _normalise(self, features: np.ndarray) -> np.ndarray:
        return (features - self.feature_mean) / self.feature_std

    def fit(
        self, sims: np.ndarray, correct: np.ndarray, features: np.ndarray | None = None
    ) -> None:
        if features is None:
            raise ValueError("conditional_temperature requires conditioning features")
        torch = _torch()
        self.feature_mean = features.mean(axis=0).astype(np.float32)
        self.feature_std = np.maximum(features.std(axis=0), 1e-6).astype(np.float32)

        n_val = max(1, int(0.15 * len(features)))  # nested split, for early stopping
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(features))
        val_idx, tr_idx = order[:n_val], order[n_val:]

        x = torch.tensor(self._normalise(features), dtype=torch.float32, device=self.device)
        s = torch.tensor(sims, dtype=torch.float32, device=self.device)
        y = torch.tensor(correct.astype(np.float32), device=self.device)

        self.net = self._build()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        best_val, best_state, waited = float("inf"), None, 0

        def nll(idx: np.ndarray) -> Any:
            t = self.net(x[idx]) + 1e-6  # type: ignore[misc]
            probs = torch.softmax(s[idx] / t, dim=1)
            p_true = torch.where(y[idx] > 0.5, probs[:, 0], 1.0 - probs[:, 0])
            return -torch.log(p_true.clamp_min(1e-12)).mean()

        for epoch in range(self.epochs):
            self.net.train()
            opt.zero_grad()
            loss = nll(tr_idx)
            loss.backward()
            opt.step()

            self.net.eval()
            with torch.no_grad():
                val = float(nll(val_idx))
            if val < best_val - 1e-5:
                best_val, waited = val, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience:
                    log.info("conditional_temperature.early_stop", epoch=epoch, val_nll=val)
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self._fitted = True

    def temperatures(self, sims: np.ndarray, features: np.ndarray | None = None) -> np.ndarray:
        self._check_fitted()
        if features is None:
            raise ValueError("conditional_temperature requires conditioning features")
        torch = _torch()
        if self.net is None:  # pragma: no cover - fit() always sets it or raises
            raise CalibrationNotFittedError("conditional_temperature: net is unset after fit()")
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(self._normalise(features), dtype=torch.float32, device=self.device)
            t = self.net(x).cpu().numpy().ravel().astype(np.float64) + 1e-6
        lo, hi = self.bounds
        self.clamp_rate = float(np.mean((t < lo) | (t > hi)))
        if self.clamp_rate > 0.05:
            log.warning("conditional_temperature.high_clamp_rate", rate=self.clamp_rate)
        return np.clip(t, lo, hi)

    def transform(self, sims: np.ndarray, features: np.ndarray | None = None) -> np.ndarray:
        return softmax(sims, self.temperatures(sims, features))

    @property
    def n_params(self) -> int:
        if self.net is None:
            return 0
        return int(sum(p.numel() for p in self.net.parameters()))

    def save(self, path: Path) -> None:
        torch = _torch()
        if self.net is None:  # pragma: no cover - fit() always sets it or raises
            raise CalibrationNotFittedError("conditional_temperature: cannot save before fit()")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "feature_mean": self.feature_mean,
                "feature_std": self.feature_std,
                "hidden_dim": self.hidden_dim,
                "bounds": self.bounds,
            },
            path,
        )

    def load(self, path: Path) -> None:
        torch = _torch()
        # weights_only=True: a checkpoint is untrusted input and pickle executes code.
        blob = torch.load(path, map_location=self.device, weights_only=True)
        self.hidden_dim = int(blob["hidden_dim"])
        self.net = self._build()
        self.net.load_state_dict(blob["state_dict"])
        self.feature_mean = np.asarray(blob["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(blob["feature_std"], dtype=np.float32)
        self.bounds = tuple(blob["bounds"])  # type: ignore[assignment]
        self._fitted = True
