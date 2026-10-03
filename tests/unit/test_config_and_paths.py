"""Config validation and path containment."""

from __future__ import annotations

from pathlib import Path

import pytest

from reiduq.core.config import load_config
from reiduq.core.exceptions import ConfigError, UnsafePathError
from reiduq.core.paths import safe_path


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(text)
    return p


def test_typo_in_config_key_is_rejected(tmp_path: Path) -> None:
    cfg = _write(tmp_path, "name: t\ntrain:\n  epocks: 10\n")  # 'epocks' typo
    with pytest.raises(ConfigError):
        load_config(cfg)


def test_config_hash_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    a = load_config(_write(tmp_path, "name: t\ntrain:\n  epochs: 10\n"))
    b = load_config(_write(tmp_path, "name: t\ntrain:\n  epochs: 10\n"))
    c = load_config(_write(tmp_path, "name: t\ntrain:\n  epochs: 11\n"))
    assert a.hash == b.hash != c.hash


def test_overrides_apply(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, "name: t\n"), ["train.lr=1e-5"])
    assert cfg.train.lr == 1e-5


def test_incoherent_protocol_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, "name: t\ndata:\n  protocol: leave_camera_out\n"))


@pytest.mark.security
def test_path_traversal_is_blocked(tmp_path: Path) -> None:
    with pytest.raises(UnsafePathError):
        safe_path(tmp_path, "../../etc/passwd")


@pytest.mark.security
def test_normal_path_is_allowed(tmp_path: Path) -> None:
    assert safe_path(tmp_path, "a", "b.txt").is_relative_to(tmp_path.resolve())
