"""AuditStore: persistence, ordering, limit clamping."""

from __future__ import annotations

from pathlib import Path

from reiduq.serving.audit import AuditRecord, AuditStore


def _record(query_id: str, verdict: str = "MATCH") -> AuditRecord:
    return AuditRecord(
        correlation_id=f"c-{query_id}",
        query_id=query_id,
        verdict=verdict,
        matched_id="g1",
        confidence=0.9,
        temperature=1.5,
        model_version="test",
        role="analyst",
    )


def test_records_persist_across_store_instances(tmp_path: Path) -> None:
    db = tmp_path / "audit.db"
    AuditStore(db).record(_record("q1"))
    reopened = AuditStore(db)
    rows = reopened.recent()
    assert len(rows) == 1
    assert rows[0]["query_id"] == "q1"


def test_recent_returns_newest_first(tmp_path: Path) -> None:
    store = AuditStore(tmp_path / "audit.db")
    for i in range(5):
        store.record(_record(f"q{i}"))
    rows = store.recent(limit=3)
    assert [r["query_id"] for r in rows] == ["q4", "q3", "q2"]


def test_limit_is_clamped_to_a_sane_ceiling(tmp_path: Path) -> None:
    store = AuditStore(tmp_path / "audit.db")
    store.record(_record("q0"))
    # A caller-controlled unbounded limit is its own denial-of-service risk.
    rows = store.recent(limit=10_000_000)
    assert len(rows) == 1  # did not attempt to allocate anything absurd


def test_verdict_counts_groups_correctly(tmp_path: Path) -> None:
    store = AuditStore(tmp_path / "audit.db")
    store.record(_record("q1", "MATCH"))
    store.record(_record("q2", "UNCERTAIN"))
    store.record(_record("q3", "UNCERTAIN"))
    counts = store.verdict_counts(since_hours=24)
    assert counts == {"MATCH": 1, "UNCERTAIN": 2}
