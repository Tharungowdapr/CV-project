"""Persistence for match decisions.

Before this module existed, a decision was logged and returned but nothing
survived past the log line - there was no way to answer "what did the system
decide about query X last Tuesday" without grepping structured logs. This adds
a minimal, append-only SQLite audit trail, which is enough for a
single-instance deployment and for review/compliance queries; a
multi-replica production deployment should point this at a real database
instead (the schema is trivial to port - see the note on AuditStore below).

What is stored is deliberately narrow: an id, a verdict, a confidence, a
temperature, a role and a timestamp. Never the image, never the raw
similarity vector, never anything that could re-identify a person - the same
"ids and hashes only, never imagery" rule the logging module follows.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

_SCHEMA = """
CREATE TABLE IF NOT EXISTS match_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    correlation_id TEXT NOT NULL,
    query_id TEXT NOT NULL,
    verdict TEXT NOT NULL CHECK (verdict IN ('MATCH', 'UNCERTAIN', 'REJECT')),
    matched_id TEXT,
    confidence REAL NOT NULL,
    temperature REAL NOT NULL,
    model_version TEXT NOT NULL,
    role TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_match_decisions_query_id ON match_decisions(query_id);
CREATE INDEX IF NOT EXISTS idx_match_decisions_created_at ON match_decisions(created_at);
"""


@dataclass(frozen=True)
class AuditRecord:
    correlation_id: str
    query_id: str
    verdict: str
    matched_id: str | None
    confidence: float
    temperature: float
    model_version: str
    role: str


class AuditStore:
    """Thread-safe-enough for a single Uvicorn worker: one connection, WAL mode.

    For more than one worker process (``uvicorn --workers N>1``) or more than
    one replica, each gets its own SQLite file/lock contention story - at
    that point swap this for a real database (Postgres) behind the same
    ``record`` / ``recent`` interface; nothing else in the service needs to
    change, which is the point of keeping the interface this narrow.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    def record(self, entry: AuditRecord) -> None:
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO match_decisions
                   (correlation_id, query_id, verdict, matched_id, confidence,
                    temperature, model_version, role, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.correlation_id,
                    entry.query_id,
                    entry.verdict,
                    entry.matched_id,
                    entry.confidence,
                    entry.temperature,
                    entry.model_version,
                    entry.role,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def recent(self, limit: int = 100) -> list[dict[str, object]]:
        """Newest first. Admin-only in the API - see require_admin on the route."""
        limit = max(1, min(limit, 1000))  # a caller-controlled unbounded query is its own DoS
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM match_decisions ORDER BY id DESC LIMIT ?", (limit,)
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

    def verdict_counts(self, since_hours: int = 24) -> dict[str, int]:
        """Feeds the drift signal described in the ops guide: a rising
        UNCERTAIN share is the earliest indicator of domain shift."""
        with self._cursor() as cur:
            cur.execute(
                """SELECT verdict, COUNT(*) FROM match_decisions
                   WHERE created_at >= datetime('now', ?)
                   GROUP BY verdict""",
                (f"-{max(1, since_hours)} hours",),
            )
            return dict(cur.fetchall())

    def close(self) -> None:
        self._conn.close()
