"""API boundary tests: auth, headers, and payload validation."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="API tests need the serving extras installed")

from fastapi.testclient import TestClient  # noqa: E402

from reiduq.serving.app import app  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)
AUTH = {"Authorization": "Bearer devkey"}


@pytest.mark.security
def test_health_is_public() -> None:
    assert client.get("/health").status_code == 200


@pytest.mark.security
def test_match_requires_authentication() -> None:
    r = client.post("/v1/match", json={"query_id": "q1", "similarities": [0.9], "candidate_ids": ["g1"]})
    assert r.status_code == 401


@pytest.mark.security
def test_invalid_token_is_rejected() -> None:
    r = client.post(
        "/v1/match",
        json={"query_id": "q1", "similarities": [0.9], "candidate_ids": ["g1"]},
        headers={"Authorization": "Bearer wrong"},
    )
    assert r.status_code == 401


@pytest.mark.security
def test_security_headers_are_present() -> None:
    h = client.get("/health").headers
    assert h["X-Content-Type-Options"] == "nosniff"
    assert h["X-Frame-Options"] == "DENY"


@pytest.mark.security
def test_malformed_query_id_is_rejected() -> None:
    r = client.post(
        "/v1/match",
        json={"query_id": "../../etc/passwd", "similarities": [0.9], "candidate_ids": ["g1"]},
        headers=AUTH,
    )
    assert r.status_code == 422


@pytest.mark.security
def test_unknown_field_is_rejected() -> None:
    r = client.post(
        "/v1/match",
        json={"query_id": "q1", "similarities": [0.9], "candidate_ids": ["g1"], "evil": 1},
        headers=AUTH,
    )
    assert r.status_code == 422


@pytest.mark.security
def test_oversized_candidate_list_is_rejected() -> None:
    r = client.post(
        "/v1/match",
        json={"query_id": "q1", "similarities": [0.5] * 500, "candidate_ids": ["g"] * 500},
        headers=AUTH,
    )
    assert r.status_code == 422
