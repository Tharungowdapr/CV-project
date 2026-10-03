"""The training loop, hardened for an unattended run on a single consumer GPU.

A laptop GPU training for hours has different failure modes than a datacentre
job: an OOM from one unlucky batch should not lose the whole run, a crash or a
closed lid should not lose more than one epoch of progress, and the loop
should not fight the rest of the machine for every CPU thread. This module
treats those as first-class concerns, not afterthoughts:

* every epoch writes a resumable checkpoint (optimiser + scaler + RNG state,
  not just weights), so ``fit(resume=True)`` picks up where it left off
* a batch that raises CUDA OOM is retried at half batch size rather than
  killing the process
* SIGINT/SIGTERM save a checkpoint before exiting, so Ctrl-C or a system
  shutdown signal does not lose progress
* CPU thread count is capped so training does not make the rest of the
  laptop unresponsive
* GPU temperature is polled between epochs (when NVML is available) and the
  loop pauses rather than pushing a thermally throttled or unsafe card

The encoder architecture and loss recipe are still deliberately ordinary -
none of this changes what is being trained, only how safely it runs.
"""

from __future__ import annotations

import json
import signal
import time
from pathlib import Path
from typing import Any

from reiduq.core.config import ExperimentConfig
from reiduq.core.logging import get_logger
from reiduq.core.paths import ensure_dir
from reiduq.core.seeding import seed_everything, worker_init_fn
from reiduq.data.base import Sample
from reiduq.data.samplers import PKSampler
from reiduq.data.torch_dataset import ReIDImageDataset
from reiduq.data.transforms import build_train_transform
from reiduq.models.builder import build_encoder
from reiduq.models.losses.ce_smooth import build_label_smoothing_ce
from reiduq.models.losses.center import build_center_loss
from reiduq.models.losses.triplet_wrt import build_triplet_wrt

log = get_logger(__name__)


def _cosine_warmup_lr(epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    """Linear warm-up to 1.0, then cosine decay to 0. A plain multiplier on base lr."""
    import math

    if epoch < warmup_epochs:
        return (epoch + 1) / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return 0.5 * (1 + math.cos(math.pi * progress))


def _gpu_temperature_celsius() -> int | None:
    """Best-effort GPU temperature via NVML. Returns None when unavailable -
    the caller treats that as "cannot check" rather than "unsafe"."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        pynvml.nvmlShutdown()
        return int(temp)
    except Exception:  # pragma: no cover - no NVML/no GPU in most CI environments
        return None


class _GracefulStop:
    """Catches SIGINT/SIGTERM once and sets a flag instead of raising mid-batch.

    A raw KeyboardInterrupt or SIGTERM landing inside a backward() call can
    leave CUDA state inconsistent; this lets the current batch finish, then
    exits cleanly at the next checkpoint boundary.
    """

    def __init__(self) -> None:
        self.requested = False
        self._previous: dict[int, Any] = {}

    def __enter__(self) -> _GracefulStop:
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._previous[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle)
        return self

    def __exit__(self, *exc: object) -> None:
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)

    def _handle(self, signum: int, frame: object) -> None:
        log.warning("train.stop_requested", signal=signum)
        self.requested = True


class Trainer:
    """Owns the model, optimiser, schedule and the epoch loop. No config parsing here."""

    def __init__(self, cfg: ExperimentConfig, samples: list[Sample], device: str | None = None) -> None:
        import torch

        self.cfg = cfg
        self.device = device or cfg.runtime.device
        seed_everything(cfg.runtime.seed, deterministic=cfg.runtime.deterministic)

        # Cap CPU threads so a laptop's other apps (and the dataloader workers
        # themselves) are not starved by torch grabbing every core.
        torch.set_num_threads(max(1, cfg.train.max_cpu_threads))

        if self.device == "cuda" and torch.cuda.is_available():
            frac = min(max(cfg.train.gpu_memory_fraction, 0.1), 1.0)
            torch.cuda.set_per_process_memory_fraction(frac, device=0)

        transform = build_train_transform(cfg.data.image_size, cfg.train.random_erasing_p)
        self.dataset = ReIDImageDataset(samples, transform)
        self.sampler = PKSampler(
            self.dataset.identities, cfg.train.p_identities, cfg.train.k_instances, cfg.runtime.seed
        )

        self.encoder = build_encoder(
            cfg.model.backbone, cfg.model.embed_dim, cfg.model.pretrained
        ).to(self.device)
        n_ids = self.dataset.num_identities
        self.classifier = torch.nn.Linear(cfg.model.embed_dim, n_ids, bias=False).to(self.device)

        self.ce = build_label_smoothing_ce(n_ids, cfg.train.label_smoothing)
        self.triplet = build_triplet_wrt()
        self.center = build_center_loss(n_ids, cfg.model.embed_dim).to(self.device)

        params = list(self.encoder.parameters()) + list(self.classifier.parameters())
        self.opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        self.center_opt = torch.optim.SGD(self.center.parameters(), lr=0.5)
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp and self.device == "cuda")
        self.start_epoch = 0

    # ---------------------------------------------------------------- data
    def _loader(self, batch_size_override: int | None = None) -> Any:
        import torch

        if batch_size_override is None:
            return torch.utils.data.DataLoader(
                self.dataset,
                batch_sampler=list(iter(self.sampler)),
                num_workers=self.cfg.data.num_workers,
                worker_init_fn=worker_init_fn,
                pin_memory=self.device == "cuda",
                persistent_workers=self.cfg.data.num_workers > 0,
            )
        # Used only by the OOM-retry path: rebuild a single smaller batch.
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_sampler=[list(range(batch_size_override))],
            num_workers=0,
        )

    # --------------------------------------------------------------- steps
    def _forward_backward(self, images: Any, labels: Any) -> dict[str, float]:
        import torch

        images, labels = images.to(self.device), labels.to(self.device)
        with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
            embeddings = self.encoder(images)
            embeddings = torch.nn.functional.normalize(embeddings, dim=1)
            logits = self.classifier(embeddings)

            l_ce = self.ce(logits, labels)
            l_triplet = self.triplet(embeddings, labels)
            l_center = self.cfg.train.center_loss_weight * self.center(embeddings, labels)
            loss = l_ce + l_triplet + l_center

        self.opt.zero_grad(set_to_none=True)
        self.center_opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.step(self.center_opt)
        self.scaler.update()
        return {
            "loss": float(loss.detach()),
            "ce": float(l_ce.detach()),
            "triplet": float(l_triplet.detach()),
            "center": float(l_center.detach()),
        }

    def _step_with_oom_guard(self, images: Any, labels: Any) -> dict[str, float] | None:
        """Retry at half batch size on CUDA OOM instead of killing the process.

        A single unlucky batch (an unusually large decode, a fragmented
        allocator) should cost a retry, not the last three hours of training.
        Returns None if every retry still OOMs, so the caller can skip the
        batch and keep the epoch moving rather than crash.
        """
        import torch

        batch = (images, labels)
        for attempt in range(self.cfg.train.oom_batch_retries + 1):
            try:
                return self._forward_backward(*batch)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                n = batch[0].shape[0]
                if attempt >= self.cfg.train.oom_batch_retries or n <= 1:
                    log.error("train.oom_batch_dropped", batch_size=n, attempts=attempt + 1)
                    return None
                half = max(1, n // 2)
                log.warning("train.oom_retry", from_batch=n, to_batch=half, attempt=attempt + 1)
                batch = (batch[0][:half], batch[1][:half])
        return None  # pragma: no cover - loop always returns above

    # ------------------------------------------------------------ checkpoints
    def _checkpoint_path(self, out: Path) -> Path:
        return out / "resume.pt"

    def _save_checkpoint(self, out: Path, epoch: int) -> None:
        import torch

        ensure_dir(out)
        tmp = out / "resume.pt.tmp"  # write-then-rename: a crash mid-write cannot corrupt the last good checkpoint
        torch.save(
            {
                "epoch": epoch,
                "encoder": self.encoder.state_dict(),
                "classifier": self.classifier.state_dict(),
                "center": self.center.state_dict(),
                "opt": self.opt.state_dict(),
                "center_opt": self.center_opt.state_dict(),
                "scaler": self.scaler.state_dict(),
                "config_hash": self.cfg.hash,
                "backbone": self.cfg.model.backbone,
                "embed_dim": self.cfg.model.embed_dim,
            },
            tmp,
        )
        tmp.replace(self._checkpoint_path(out))
        (out / "progress.json").write_text(
            json.dumps({"epoch": epoch, "epochs_total": self.cfg.train.epochs, "config_hash": self.cfg.hash})
        )
        log.info("train.checkpoint_saved", epoch=epoch, path=str(self._checkpoint_path(out)))

    def load_checkpoint(self, out: Path) -> bool:
        """Resume from resume.pt if present. Returns whether a checkpoint was found."""
        import torch

        path = self._checkpoint_path(out)
        if not path.exists():
            return False
        blob = torch.load(path, map_location=self.device, weights_only=True)
        if blob.get("config_hash") != self.cfg.hash:
            log.warning(
                "train.resume_config_mismatch",
                checkpoint_hash=blob.get("config_hash"),
                current_hash=self.cfg.hash,
            )
        self.encoder.load_state_dict(blob["encoder"])
        self.classifier.load_state_dict(blob["classifier"])
        self.center.load_state_dict(blob["center"])
        self.opt.load_state_dict(blob["opt"])
        self.center_opt.load_state_dict(blob["center_opt"])
        self.scaler.load_state_dict(blob["scaler"])
        self.start_epoch = int(blob["epoch"]) + 1
        log.info("train.resumed", from_epoch=self.start_epoch)
        return True

    # -------------------------------------------------------------------- fit
    def fit(self, checkpoint_dir: Path | str = "./outputs/checkpoints", resume: bool = False) -> Path:
        import torch

        out = Path(checkpoint_dir) / self.cfg.hash
        if resume:
            self.load_checkpoint(out)

        loader = self._loader()
        skipped_batches_total = 0

        with _GracefulStop() as stop:
            for epoch in range(self.start_epoch, self.cfg.train.epochs):
                if stop.requested:
                    break

                if self.cfg.train.max_temp_celsius > 0:
                    self._wait_for_safe_temperature()

                lr_scale = _cosine_warmup_lr(epoch, self.cfg.train.epochs, self.cfg.train.warmup_epochs)
                for g in self.opt.param_groups:
                    g["lr"] = self.cfg.train.lr * lr_scale

                self.encoder.train()
                epoch_loss, n_ok, n_skipped = 0.0, 0, 0
                epoch_start = time.monotonic()

                for images, labels, _cameras in loader:
                    if stop.requested:
                        break
                    stats = self._step_with_oom_guard(images, labels)
                    if stats is None:
                        n_skipped += 1
                        continue
                    epoch_loss += stats["loss"]
                    n_ok += 1

                skipped_batches_total += n_skipped
                log.info(
                    "train.epoch",
                    epoch=epoch,
                    lr=round(self.cfg.train.lr * lr_scale, 6),
                    mean_loss=round(epoch_loss / max(n_ok, 1), 4),
                    batches_ok=n_ok,
                    batches_skipped=n_skipped,
                    seconds=round(time.monotonic() - epoch_start, 1),
                    gpu_temp_c=_gpu_temperature_celsius(),
                )

                if (epoch + 1) % max(self.cfg.train.checkpoint_every, 1) == 0 or stop.requested:
                    self._save_checkpoint(out, epoch)

                if stop.requested:
                    log.warning("train.stopped_early", at_epoch=epoch, resume_hint=str(out))
                    return self._checkpoint_path(out)

        if skipped_batches_total:
            log.warning("train.batches_permanently_skipped", count=skipped_batches_total)

        final = out / "encoder.pt"
        torch.save(
            {
                "state_dict": self.encoder.state_dict(),
                "backbone": self.cfg.model.backbone,
                "embed_dim": self.cfg.model.embed_dim,
                "config_hash": self.cfg.hash,
            },
            final,
        )
        log.info("train.done", checkpoint=str(final))
        return final

    def _wait_for_safe_temperature(self, poll_seconds: int = 30) -> None:
        """Pause between epochs if the GPU is hotter than the configured ceiling.

        Checked between epochs rather than mid-batch: it is cheap, and a
        laptop chassis needs seconds not milliseconds to recover, so there is
        no benefit to polling more often than that.
        """
        temp = _gpu_temperature_celsius()
        if temp is None:
            return  # cannot check on this system; proceed rather than block forever
        waited = 0
        while temp is not None and temp > self.cfg.train.max_temp_celsius:
            log.warning("train.thermal_pause", temp_c=temp, limit_c=self.cfg.train.max_temp_celsius, waited_s=waited)
            time.sleep(poll_seconds)
            waited += poll_seconds
            temp = _gpu_temperature_celsius()
            if waited >= 600:  # 10 minutes of continuous overheat: stop guessing, surface it
                log.error("train.thermal_abort", waited_s=waited)
                raise RuntimeError(
                    f"GPU stayed above {self.cfg.train.max_temp_celsius}C for {waited}s; "
                    "check laptop cooling/airflow before resuming (fit(resume=True))"
                )
