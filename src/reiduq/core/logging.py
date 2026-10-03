"""Structured logging. Every record carries run_id and config_hash.

Security: user-supplied data is bound as key/value pairs, never interpolated
into the message string, so log injection is not possible. Raw image crops are
never logged - only ids and hashes.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level.upper())
    renderer = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=True)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


def bind_run(**kwargs: Any) -> None:
    """Attach run-scoped context (run_id, config_hash, git_sha) to every record."""
    structlog.contextvars.bind_contextvars(**kwargs)


def get_logger(name: str) -> Any:
    return structlog.get_logger(name)
