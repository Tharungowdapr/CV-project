"""Experiment orchestration: one config in, one result JSON out.

Every result carries its config hash, git SHA, seed and hardware string, so any
number in the paper can be traced to the exact state that produced it.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess  # nosec B404 - only used below for `git rev-parse`, absolute path, shell=False
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from reiduq.core.config import ExperimentConfig
from reiduq.core.logging import bind_run, get_logger
from reiduq.core.paths import ensure_dir
from reiduq.core.seeding import seed_everything
from reiduq.eval.metrics import calibration as calib_metrics
from reiduq.eval.metrics import selective

log = get_logger(__name__)


def git_sha() -> str:
    git_bin = shutil.which("git")  # resolved once, absolute path - not a partial "git" lookup
    if git_bin is None:
        return "unknown"
    try:
        out = subprocess.run(  # nosec B603 - absolute path from shutil.which, fixed argv, shell=False
            [git_bin, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return "unknown"


@dataclass
class ExperimentResult:
    name: str
    run_id: str
    config_hash: str
    git_sha: str
    seed: int
    hardware: str
    timestamp: str
    metrics: dict[str, Any] = field(default_factory=dict)
    per_band: dict[str, dict[str, float]] = field(default_factory=dict)

    def save(self, root: Path | str = "./outputs/results") -> Path:
        path = ensure_dir(Path(root)) / f"{self.name}__{self.config_hash}__{self.run_id}.json"
        path.write_text(json.dumps(asdict(self), indent=2, default=str))
        return path


def hardware_string() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"{torch.cuda.get_device_name(0)} / torch {torch.__version__}"
    except ImportError:  # pragma: no cover
        pass
    return f"{platform.processor() or platform.machine()} / cpu"


def evaluate_confidence(
    confidence: np.ndarray, correct: np.ndarray, cfg: ExperimentConfig
) -> dict[str, float]:
    """The metric block computed identically for every method and every band."""
    curve = selective.risk_coverage_curve(confidence, correct)
    brier_score, decomp = calib_metrics.brier(confidence, correct)
    out = {
        "accuracy": float(np.mean(correct)),
        "ece": calib_metrics.ece(confidence, correct, cfg.eval.ece_bins, cfg.eval.ece_strategy),
        "mce": calib_metrics.mce(confidence, correct, cfg.eval.ece_bins, cfg.eval.ece_strategy),
        "brier": brier_score,
        "brier_reliability": decomp.reliability,
        "brier_resolution": decomp.resolution,
        "nll": calib_metrics.nll(confidence, correct),
        "aurc": selective.aurc(curve),
        "auroc": selective.auroc_correct_vs_incorrect(confidence, correct),
        "coverage_at_1pct_risk": selective.coverage_at_risk(confidence, correct, 0.01),
    }
    for cov in cfg.eval.coverage_points:
        out[f"far_at_{int(cov * 100)}pct_coverage"] = selective.far_at_coverage(
            confidence, correct, cov
        )
    return out


def run_experiment(
    cfg: ExperimentConfig,
    confidence: np.ndarray,
    correct: np.ndarray,
    bands: np.ndarray | None = None,
) -> ExperimentResult:
    run_id = uuid.uuid4().hex[:12]
    seed_everything(cfg.runtime.seed, deterministic=cfg.runtime.deterministic)
    bind_run(run_id=run_id, config_hash=cfg.hash, experiment=cfg.name)

    result = ExperimentResult(
        name=cfg.name,
        run_id=run_id,
        config_hash=cfg.hash,
        git_sha=git_sha(),
        seed=cfg.runtime.seed,
        hardware=hardware_string(),
        timestamp=datetime.now(UTC).isoformat(),
        metrics=evaluate_confidence(confidence, correct, cfg),
    )
    if bands is not None:
        for band in sorted(set(bands.tolist())):
            mask = bands == band
            if mask.sum() >= 20:  # too few samples makes per-band ECE meaningless
                result.per_band[str(band)] = evaluate_confidence(
                    confidence[mask], correct[mask], cfg
                )
    log.info("experiment.done", **{k: round(v, 4) for k, v in result.metrics.items()})
    return result
