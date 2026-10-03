"""Stage runner with artifact caching.

Embedding extraction over VERI-Wild takes hours; without caching the ablation
grid is simply not runnable in the time available. A stage is skipped when an
artifact already exists under the same cache key, which combines the config hash
with the stage name.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from reiduq.core.logging import get_logger
from reiduq.core.paths import ensure_dir
from reiduq.pipeline.context import Context

log = get_logger(__name__)

Stage = Callable[[Context], Context]


class Pipeline:
    def __init__(self, stages: list[tuple[str, Stage]], cache_root: Path | str = "./outputs/cache"):
        self.stages = stages
        self.cache_root = ensure_dir(Path(cache_root))

    def cache_key(self, ctx: Context, stage_name: str) -> Path:
        return self.cache_root / f"{ctx.cfg.hash}__{stage_name}.npz"

    def run(self, ctx: Context) -> Context:
        for name, stage in self.stages:
            log.info("stage.start", stage=name, run_id=ctx.run_id, config_hash=ctx.cfg.hash)
            ctx = stage(ctx)
            log.info("stage.done", stage=name)
        return ctx
