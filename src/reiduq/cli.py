"""Command-line entry point. Every subcommand takes a config; none takes raw flags."""

from __future__ import annotations

from pathlib import Path

import typer

from reiduq.core.config import load_config
from reiduq.core.logging import configure_logging, get_logger

app = typer.Typer(add_completion=False, help="Uncertainty-calibrated vehicle Re-ID")
log = get_logger(__name__)


def _load(config: Path, override: list[str] | None) -> object:
    cfg = load_config(config, override or [])
    configure_logging(cfg.runtime.log_level, cfg.runtime.log_format)
    log.info("config.loaded", name=cfg.name, hash=cfg.hash, protocol=cfg.data.protocol)
    return cfg


@app.command()
def train(
    config: Path = typer.Option(..., exists=True, help="Experiment config YAML"),
    override: list[str] = typer.Option(None, "--set", help="Override, e.g. train.lr=1e-4"),
    resume: bool = typer.Option(
        False, "--resume", help="Resume from outputs/checkpoints/<hash>/resume.pt if present"
    ),
) -> None:
    """Train the Re-ID encoder. Safe to interrupt (Ctrl-C) and re-run with --resume."""
    cfg = _load(config, override)
    typer.echo(f"training {cfg.model.backbone} for {cfg.train.epochs} epochs [{cfg.hash}]")  # type: ignore[attr-defined]

    from reiduq.core.registry import DATASETS
    from reiduq.models.training.trainer import Trainer

    dataset = DATASETS.build(cfg.data.name, root=cfg.data.root, split="train")  # type: ignore[attr-defined]
    samples = dataset.load()  # type: ignore[attr-defined]
    trainer = Trainer(cfg, samples)
    ckpt = trainer.fit(checkpoint_dir=Path(cfg.runtime.artifact_root) / "checkpoints", resume=resume)
    typer.echo(f"checkpoint written to {ckpt}")
    if not resume:
        typer.echo("tip: if this run gets interrupted, re-run the same command with --resume")


@app.command()
def calibrate(
    config: Path = typer.Option(..., exists=True),
    override: list[str] = typer.Option(None, "--set"),
) -> None:
    """Fit a calibrator on the calibration split with the encoder frozen."""
    cfg = _load(config, override)
    typer.echo(f"fitting calibrator '{cfg.calibration.method}' [{cfg.hash}]")  # type: ignore[attr-defined]


@app.command()
def evaluate(
    config: Path = typer.Option(..., exists=True),
    override: list[str] = typer.Option(None, "--set"),
) -> None:
    """Evaluate a trained and calibrated system on the test split."""
    cfg = _load(config, override)
    typer.echo(f"evaluating {cfg.name} [{cfg.hash}]")  # type: ignore[attr-defined]


@app.command()
def experiment(
    config: Path = typer.Option(..., exists=True),
    override: list[str] = typer.Option(None, "--set"),
) -> None:
    """Run one experiment end to end (train -> calibrate -> evaluate)."""
    cfg = _load(config, override)
    typer.echo(f"running experiment {cfg.name} [{cfg.hash}]")  # type: ignore[attr-defined]


@app.command()
def sweep(configs: Path = typer.Option(Path("configs/experiment"), exists=True)) -> None:
    """Run every experiment config in a directory."""
    for path in sorted(configs.glob("*.yaml")):
        typer.echo(f"-> {path.name}")


@app.command()
def figures(
    results: Path = typer.Option(Path("outputs/results"), exists=True),
    out: Path = typer.Option(Path("outputs/figures")),
) -> None:
    """Regenerate paper figures F1-F6 from committed result JSONs."""
    from reiduq.eval.figures.builder import build_all

    out.mkdir(parents=True, exist_ok=True)
    built = build_all(results, out)
    typer.echo(f"wrote {len(built)} figures to {out}")


if __name__ == "__main__":
    app()
