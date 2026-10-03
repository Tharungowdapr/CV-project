"""Efficiency measurement. A claim you make is a claim you measure.

The paper calls the proposed calibrator "lightweight" against a graph
transformer; that word is only allowed next to these numbers.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EfficiencyReport:
    name: str
    n_params: int
    inference_multiplier: float
    p50_latency_ms: float
    p95_latency_ms: float
    throughput_qps: float
    peak_memory_mb: float = 0.0
    hardware: str = "unknown"
    extra: dict[str, Any] = field(default_factory=dict)


def benchmark(
    fn: Callable[[], Any],
    *,
    name: str,
    n_params: int = 0,
    inference_multiplier: float = 1.0,
    warmup: int = 10,
    runs: int = 100,
    batch_size: int = 1,
    hardware: str = "unknown",
) -> EfficiencyReport:
    """Time a callable with warm-up discarded, reporting percentiles not means."""
    for _ in range(warmup):
        fn()
    timings: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - start) * 1000.0)
    arr = np.array(timings)
    p50 = float(np.percentile(arr, 50))
    return EfficiencyReport(
        name=name,
        n_params=n_params,
        inference_multiplier=inference_multiplier,
        p50_latency_ms=p50,
        p95_latency_ms=float(np.percentile(arr, 95)),
        throughput_qps=float(batch_size / (p50 / 1000.0)) if p50 > 0 else 0.0,
        peak_memory_mb=_peak_memory_mb(),
        hardware=hardware,
    )


def _peak_memory_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / 1024**2)
    except ImportError:  # pragma: no cover
        pass
    return 0.0


def hardware_string_safe() -> str:
    """Best-effort hardware label recorded alongside every timing."""
    import platform

    try:
        import torch

        if torch.cuda.is_available():
            return f"{torch.cuda.get_device_name(0)} / torch {torch.__version__}"
    except ImportError:  # pragma: no cover
        pass
    return f"{platform.processor() or platform.machine()} / cpu"
