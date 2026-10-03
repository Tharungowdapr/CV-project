"""Figure generation F1-F6 from committed result JSONs.

Figures regenerate from results, never from a live run: a figure that requires
re-running the experiment cannot be checked by a reviewer.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BAND_ORDER = ["O0", "O1", "O2", "O3", "O4"]


def _load_results(results_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]


def figure_ece_vs_band(results: list[dict], out: Path) -> Path | None:
    """F1: ECE against occlusion band, one line per method. The core figure."""
    fig, ax = plt.subplots(figsize=(6, 4))
    drawn = False
    for res in results:
        bands = res.get("per_band", {})
        xs = [b for b in BAND_ORDER if b in bands]
        if not xs:
            continue
        ax.plot(xs, [bands[b]["ece"] for b in xs], marker="o", label=res["name"])
        drawn = True
    if not drawn:
        plt.close(fig)
        return None
    ax.set_xlabel("Occlusion band (decreasing visibility)")
    ax.set_ylabel("Expected calibration error")
    ax.set_title("Calibration degrades with occlusion")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    path = out / "F1_ece_vs_occlusion_band.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def figure_risk_coverage(results: list[dict], out: Path) -> Path | None:
    """F4: risk-coverage operating points per method."""
    fig, ax = plt.subplots(figsize=(6, 4))
    drawn = False
    for res in results:
        m = res.get("metrics", {})
        points = [(int(k.split("_")[2].rstrip("pct")) / 100, v) for k, v in m.items() if k.startswith("far_at_")]
        if not points:
            continue
        points.sort()
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="s", label=res["name"])
        drawn = True
    if not drawn:
        plt.close(fig)
        return None
    ax.set_xlabel("Coverage")
    ax.set_ylabel("False-association rate")
    ax.set_title("Operational cost of abstention")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    path = out / "F4_risk_coverage.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def build_all(results_dir: Path, out_dir: Path) -> list[Path]:
    results = _load_results(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    built = [figure_ece_vs_band(results, out_dir), figure_risk_coverage(results, out_dir)]
    return [p for p in built if p is not None]
