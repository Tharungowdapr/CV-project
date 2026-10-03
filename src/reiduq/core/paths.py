"""Path containment. Every filesystem access driven by config or request uses this.

Threat: a config or API field such as ``../../etc/passwd`` reading arbitrary files.
Control: resolve both sides, assert containment, then return.
"""

from __future__ import annotations

from pathlib import Path

from reiduq.core.exceptions import UnsafePathError


def safe_path(root: Path | str, *parts: str) -> Path:
    """Join parts under root and guarantee the result stays inside root."""
    root_r = Path(root).resolve()
    candidate = root_r.joinpath(*parts).resolve()
    if root_r != candidate and root_r not in candidate.parents:
        raise UnsafePathError(f"path escapes root: {candidate} is not under {root_r}")
    return candidate


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
