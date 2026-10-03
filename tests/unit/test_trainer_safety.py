"""Trainer resilience: resumable checkpoints, OOM backoff, graceful stop.

These are the properties that matter for an unattended run on a single
laptop GPU - a crash should lose at most one epoch, and an OOM on one batch
should not end the process.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="trainer tests need torch")

from reiduq.core.config import load_config
from reiduq.data.base import Sample
from reiduq.models.training.trainer import Trainer, _cosine_warmup_lr


def _tiny_dataset(tmp_path: Path, n_ids: int = 4, per_id: int = 4) -> list[Sample]:
    from PIL import Image

    root = tmp_path / "images"
    root.mkdir()
    rng = np.random.default_rng(0)
    samples = []
    for vid in range(n_ids):
        color = rng.integers(0, 255, 3)
        for k in range(per_id):
            arr = np.tile(color, (32, 16, 1)).astype(np.uint8)
            path = root / f"{vid}_{k}.jpg"
            Image.fromarray(arr).save(path)
            samples.append(Sample(image_path=path, identity=vid, camera_id=f"c{k % 2}"))
    return samples


def _cfg(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "name: t\n"
        "data:\n  image_size: [32, 16]\n  num_workers: 0\n"
        "model:\n  backbone: resnet50_ibn\n  embed_dim: 16\n  pretrained: false\n"
        "train:\n"
        "  epochs: 2\n"
        "  batch_size: 4\n"
        "  p_identities: 2\n"
        "  k_instances: 2\n"
        "  amp: false\n"
        "  checkpoint_every: 1\n"
        "  max_cpu_threads: 1\n"
        "runtime:\n  device: cpu\n  seed: 0\n"
    )
    return load_config(cfg_path)


def test_cosine_warmup_schedule_ramps_then_decays() -> None:
    assert _cosine_warmup_lr(0, 10, 3) < _cosine_warmup_lr(2, 10, 3)  # ramping during warmup
    assert _cosine_warmup_lr(2, 10, 3) == pytest.approx(1.0, abs=1e-6)  # peak right at warmup end
    assert _cosine_warmup_lr(9, 10, 3) < _cosine_warmup_lr(4, 10, 3)  # decaying after warmup


def test_training_writes_a_resumable_checkpoint_every_epoch(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    samples = _tiny_dataset(tmp_path)
    trainer = Trainer(cfg, samples, device="cpu")
    trainer.fit(checkpoint_dir=tmp_path / "ckpt")

    resume_file = tmp_path / "ckpt" / cfg.hash / "resume.pt"
    assert resume_file.exists()
    blob = torch.load(resume_file, map_location="cpu", weights_only=True)
    assert blob["epoch"] == cfg.train.epochs - 1
    assert blob["config_hash"] == cfg.hash


def test_resume_continues_from_the_saved_epoch_not_from_scratch(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    samples = _tiny_dataset(tmp_path)
    ckpt_dir = tmp_path / "ckpt"

    first = Trainer(cfg, samples, device="cpu")
    first.fit(checkpoint_dir=ckpt_dir)  # runs both configured epochs

    second = Trainer(cfg, samples, device="cpu")
    resumed = second.load_checkpoint(ckpt_dir / cfg.hash)
    assert resumed is True
    assert second.start_epoch == cfg.train.epochs  # nothing left to run - already completed


def test_resume_with_no_checkpoint_starts_from_zero(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    samples = _tiny_dataset(tmp_path)
    trainer = Trainer(cfg, samples, device="cpu")
    found = trainer.load_checkpoint(tmp_path / "nonexistent")
    assert found is False
    assert trainer.start_epoch == 0


def test_oom_guard_retries_at_half_batch_before_giving_up(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    samples = _tiny_dataset(tmp_path)
    trainer = Trainer(cfg, samples, device="cpu")

    calls: list[int] = []

    def flaky_forward(images, labels):
        calls.append(images.shape[0])
        if images.shape[0] > 1:
            raise torch.cuda.OutOfMemoryError("synthetic OOM for the test")
        return {"loss": 0.0, "ce": 0.0, "triplet": 0.0, "center": 0.0}

    trainer._forward_backward = flaky_forward  # type: ignore[method-assign]
    images = torch.zeros(4, 3, 32, 16)
    labels = torch.zeros(4, dtype=torch.long)
    result = trainer._step_with_oom_guard(images, labels)

    assert result is not None  # eventually succeeded at a smaller batch size
    assert calls == sorted(calls, reverse=True)  # batch size only ever shrank


def test_oom_guard_gives_up_after_the_configured_retries_and_returns_none(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    samples = _tiny_dataset(tmp_path)
    trainer = Trainer(cfg, samples, device="cpu")

    def always_oom(images, labels):
        raise torch.cuda.OutOfMemoryError("synthetic, never recovers")

    trainer._forward_backward = always_oom  # type: ignore[method-assign]
    result = trainer._step_with_oom_guard(torch.zeros(4, 1), torch.zeros(4, dtype=torch.long))
    assert result is None  # skipped, not raised - the epoch loop must be able to continue
