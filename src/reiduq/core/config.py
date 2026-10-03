"""Pydantic config schemas, YAML loader with inheritance, and content hashing.

Two rules make this module load-bearing:

1. ``extra="forbid"`` on every model. A typo'd key crashes at load time instead
   of being silently ignored - which is how experiments quietly run with the
   wrong settings for a week.
2. Every resolved config is hashed. The hash is written into every artifact, so
   any number in the paper can be traced back to the exact configuration that
   produced it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from reiduq.core.exceptions import ConfigError

_Strict = ConfigDict(extra="forbid", frozen=True)


class DataConfig(BaseModel):
    model_config = _Strict
    name: Literal["veri776", "veriwild", "vehicleid", "cityflow"] = "veri776"
    root: str = "./data/raw/VeRi"
    image_size: tuple[int, int] = (256, 128)
    protocol: Literal[
        "in_distribution", "leave_camera_out", "cross_dataset", "unseen_types"
    ] = "in_distribution"
    held_out_cameras: list[str] = Field(default_factory=list)
    held_out_types: list[str] = Field(default_factory=list)
    target_dataset: str | None = None
    calibration_fraction: float = 0.25
    num_workers: int = 8


class OcclusionConfig(BaseModel):
    model_config = _Strict
    enabled: bool = False
    mode: Literal["synthetic", "real", "both"] = "synthetic"
    target_bands: list[str] = Field(default_factory=lambda: ["O0", "O1", "O2", "O3", "O4"])
    occluder_bank: str = "./data/processed/occluders"
    feather_px: int = 2
    match_jpeg_quality: bool = True
    enforce_geometry: bool = True


class ModelConfig(BaseModel):
    model_config = _Strict
    backbone: Literal["osnet_ain", "transreid", "resnet50_ibn"] = "osnet_ain"
    embed_dim: int = 512
    pretrained: bool = True
    dropout: float = 0.0
    mask_gating: Literal["none", "mean_fill", "black_fill"] = "mean_fill"


class TrainConfig(BaseModel):
    model_config = _Strict
    epochs: int = 120
    batch_size: int = 64
    lr: float = 3.5e-4
    weight_decay: float = 5e-4
    warmup_epochs: int = 10
    p_identities: int = 16
    k_instances: int = 4
    label_smoothing: float = 0.1
    triplet_margin: float = 0.3
    center_loss_weight: float = 5e-4
    random_erasing_p: float = 0.5
    amp: bool = True
    # Resource-safety knobs, all relevant on a single consumer laptop GPU
    checkpoint_every: int = 1          # epochs between resumable checkpoints
    max_cpu_threads: int = 4            # caps torch CPU threads so the rest of the laptop stays usable
    oom_batch_retries: int = 3          # times to shrink and retry a batch that OOMs before giving up
    gpu_memory_fraction: float = 0.9    # soft VRAM ceiling passed to torch, leaves headroom for the desktop
    max_temp_celsius: int = 85          # pause training if the GPU reports hotter than this (0 disables the check)


class CalibrationConfig(BaseModel):
    model_config = _Strict
    method: Literal[
        "raw",
        "platt",
        "global_temperature",
        "vector_scaling",
        "conditional_temperature",
        "mc_dropout",
        "deep_ensemble",
        "evidential",
    ] = "conditional_temperature"
    top_k: int = 20
    hidden_dim: int = 64
    epochs: int = 200
    lr: float = 1e-3
    temperature_bounds: tuple[float, float] = (0.05, 20.0)
    use_visibility: bool = True
    use_parts: bool = True
    use_camera_descriptor: bool = True
    camera_descriptor_dim: int = 16
    oracle_camera_id: bool = False  # ablation only
    mc_samples: int = 20
    ensemble_size: int = 5


class AggregationConfig(BaseModel):
    model_config = _Strict
    method: Literal["visibility_weighted", "inverse_temperature", "best_frame"] = (
        "visibility_weighted"
    )
    ess_correction: bool = True
    min_tracklet_len: int = 5
    max_gap_frames: int = 15


class AbstentionConfig(BaseModel):
    model_config = _Strict
    policy: Literal["threshold", "conformal"] = "conformal"
    max_false_association_rate: float = 0.01
    reject_below: float = 0.20
    alpha: float = 0.05


class EvalConfig(BaseModel):
    model_config = _Strict
    ece_bins: int = 15
    ece_strategy: Literal["quantile", "uniform"] = "quantile"
    bootstrap_samples: int = 2000
    coverage_points: list[float] = Field(default_factory=lambda: [0.5, 0.7, 0.9])
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])


class RuntimeConfig(BaseModel):
    model_config = _Strict
    device: str = "cuda"
    seed: int = 42
    deterministic: bool = True
    artifact_root: str = "./outputs"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"


class ExperimentConfig(BaseModel):
    model_config = _Strict
    name: str = "unnamed"
    description: str = ""
    data: DataConfig = DataConfig()
    occlusion: OcclusionConfig = OcclusionConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    aggregation: AggregationConfig = AggregationConfig()
    abstention: AbstentionConfig = AbstentionConfig()
    eval: EvalConfig = EvalConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    hash: str = ""

    @model_validator(mode="after")
    def _check_protocol_coherence(self) -> ExperimentConfig:
        if self.data.protocol == "leave_camera_out" and not self.data.held_out_cameras:
            raise ValueError("leave_camera_out protocol requires data.held_out_cameras")
        if self.data.protocol == "cross_dataset" and not self.data.target_dataset:
            raise ValueError("cross_dataset protocol requires data.target_dataset")
        if self.data.protocol == "unseen_types" and not self.data.held_out_types:
            raise ValueError("unseen_types protocol requires data.held_out_types")
        return self


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply CLI overrides of the form ``train.lr=1e-4``."""
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override must be key=value, got '{item}'")
        dotted, raw = item.split("=", 1)
        node: dict[str, Any] = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def config_hash(cfg: ExperimentConfig) -> str:
    payload = cfg.model_dump(exclude={"hash"})
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def load_config(path: Path | str, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load YAML with ``_base_`` inheritance, apply overrides, validate, hash."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    base_ref = raw.pop("_base_", None)
    if base_ref:
        base_path = (path.parent / base_ref).resolve()
        base_raw = yaml.safe_load(base_path.read_text()) or {}
        base_raw.pop("_base_", None)
        raw = _deep_merge(base_raw, raw)
    if overrides:
        raw = _apply_overrides(raw, overrides)
    try:
        cfg = ExperimentConfig(**raw)
    except Exception as exc:  # pydantic ValidationError - re-raise as domain error
        raise ConfigError(f"invalid config {path}: {exc}") from exc
    return cfg.model_copy(update={"hash": config_hash(cfg)})
