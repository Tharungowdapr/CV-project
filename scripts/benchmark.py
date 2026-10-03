#!/usr/bin/env python
"""Measure params, latency and throughput for every calibrator.

This produces the efficiency table. The word "lightweight" is only allowed in
the paper next to these numbers.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from reiduq.calibration.conditional_temperature import ConditionalTemperature
from reiduq.calibration.evidential import EvidentialCalibrator
from reiduq.calibration.features import build_features
from reiduq.calibration.global_temperature import GlobalTemperature
from reiduq.calibration.platt import PlattScaling
from reiduq.calibration.raw import RawSimilarity
from reiduq.calibration.vector_scaling import VectorScaling
from reiduq.eval.metrics.efficiency import benchmark, hardware_string_safe


def synthetic(n: int = 2000, k: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    correct = rng.uniform(size=n) < 0.6
    sims = -np.sort(-rng.normal(0.4, 0.15, (n, k)), axis=1)
    feats = build_features(
        sims,
        rng.uniform(0.2, 1.0, n).astype(np.float32),
        rng.uniform(0.2, 1.0, (n, 6)).astype(np.float32),
        rng.normal(0, 0.2, (n, 16)).astype(np.float32),
        np.ones(n, dtype=int),
    )
    return sims, correct, feats


def main() -> None:
    sims, correct, feats = synthetic()
    reports = []
    for factory in (
        RawSimilarity,
        PlattScaling,
        GlobalTemperature,
        VectorScaling,
        EvidentialCalibrator,
        lambda: ConditionalTemperature(epochs=60, device="cpu"),
    ):
        cal = factory()
        needs = getattr(cal, "requires_features", False)
        cal.fit(sims, correct, feats if needs else None)
        report = benchmark(
            lambda c=cal, n=needs: c.transform(sims, feats if n else None),
            name=cal.name,
            n_params=cal.n_params,
            inference_multiplier=cal.inference_multiplier,
            batch_size=len(sims),
            hardware=hardware_string_safe(),
            runs=30,
        )
        reports.append(asdict(report))
        print(f"{report.name:26s} params={report.n_params:>7d}  p50={report.p50_latency_ms:7.2f}ms")

    out = Path("outputs/efficiency.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
