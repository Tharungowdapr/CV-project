"""Global determinism control.

Determinism is a correctness property here: a calibration number that moves
between runs cannot be reported in a paper. Whatever stays non-deterministic is
documented in docs/reproducibility.md rather than silently tolerated.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed python, numpy and torch; optionally force deterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch optional for metric-only tests
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def worker_init_fn(worker_id: int) -> None:
    """Dataloader worker seeding; without it augmentation differs between runs."""
    base = int(np.random.get_state()[1][0])  # type: ignore[index]
    np.random.seed((base + worker_id) % (2**32))
    random.seed((base + worker_id) % (2**32))
