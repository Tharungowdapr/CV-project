"""Session-cookie authentication: signing, expiry, forgery resistance."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi", reason="session auth tests need fastapi")

from reiduq.serving.security import (
    SESSION_COOKIE_NAME,
    _verify_session_token,
    issue_session_token,
)


@pytest.mark.security
def test_valid_token_round_trips_to_its_role() -> None:
    token = issue_session_token("admin")
    assert _verify_session_token(token) == "admin"


@pytest.mark.security
def test_tampered_payload_is_rejected() -> None:
    token = issue_session_token("analyst")
    encoded, signature = token.split(".", 1)
    # Flip a character in the payload without recomputing the signature -
    # simulates an attacker editing the role client-side.
    forged = (encoded[:-1] + ("A" if encoded[-1] != "A" else "B")) + "." + signature
    assert _verify_session_token(forged) is None


@pytest.mark.security
def test_tampered_signature_is_rejected() -> None:
    token = issue_session_token("admin")
    encoded, signature = token.split(".", 1)
    forged = f"{encoded}.{'0' * len(signature)}"
    assert _verify_session_token(forged) is None


@pytest.mark.security
def test_malformed_token_is_rejected_not_raised() -> None:
    assert _verify_session_token("not-a-real-token") is None
    assert _verify_session_token("") is None


@pytest.mark.security
def test_expired_token_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import reiduq.serving.security as sec

    token = issue_session_token("admin")
    # Fast-forward past the TTL instead of sleeping for real in a test.
    future = time.time() + sec.SESSION_TTL_SECONDS + 1
    monkeypatch.setattr(time, "time", lambda: future)
    assert _verify_session_token(token) is None


@pytest.mark.security
def test_login_sets_httponly_samesite_cookie() -> None:
    import os

    os.environ.setdefault("API_KEYS", "devkey:analyst,adminkey:admin")
    os.environ.setdefault("SECRET_KEY", "test-secret-not-a-real-key")
    from fastapi.testclient import TestClient

    from reiduq.serving.app import app

    client = TestClient(app)
    r = client.post("/v1/auth/login", json={"api_key": "devkey"})
    assert r.status_code == 200
    set_cookie = r.headers.get("set-cookie", "")
    assert SESSION_COOKIE_NAME in set_cookie
    assert "httponly" in set_cookie.lower()
    assert "samesite=strict" in set_cookie.lower()


@pytest.mark.security
def test_login_never_reveals_whether_the_key_exists() -> None:
    from fastapi.testclient import TestClient

    from reiduq.serving.app import app

    client = TestClient(app)
    # Two different WRONG (but well-formed) keys - the response must not
    # let a caller distinguish "wrong" from "unknown" by any observable
    # difference in status code or message.
    r1 = client.post("/v1/auth/login", json={"api_key": "totally-made-up"})
    r2 = client.post("/v1/auth/login", json={"api_key": "also-not-real"})
    assert r1.status_code == r2.status_code == 401
    assert r1.json()["detail"] == r2.json()["detail"]
