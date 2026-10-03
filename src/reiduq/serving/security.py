"""Authentication, rate limiting and upload validation.

Two credential paths exist and both end up at the same role check:

1. ``Authorization: Bearer <api-key>`` - for server-to-server callers (the
   experiment runner, another backend service). The frontend never handles a
   raw API key: it exchanges one, once, at login for a session cookie.
2. A signed, httpOnly session cookie issued by ``POST /v1/auth/login`` - for
   the browser console. httpOnly means client-side JS (and anything injected
   into it) cannot read the cookie, which is why the frontend was written
   against ``credentials: "include"`` rather than storing a token in
   localStorage: anything in localStorage is readable by any injected script,
   a cookie marked httpOnly is not.

Deliberate choices worth naming:
  * API keys and session signatures are compared with a constant-time
    function - a plain `==` on a secret leaks its prefix through timing.
  * Every route except /health requires a credential. "Internal only, so no
    auth" is how internal services end up exposed.
  * Session tokens are HMAC-signed, not just base64: a client cannot forge a
    higher-privileged role by editing the cookie value, because it cannot
    reproduce the signature without SECRET_KEY.
  * Sessions expire; a token whose expiry has passed is rejected even if the
    signature is valid, so a leaked cookie has a shelf life.
  * Uploads are size- and dimension-capped and re-encoded before use, which
    defuses decompression bombs and malicious EXIF payloads in one step.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid

from fastapi import Depends, Header, HTTPException, Request, status

from reiduq.core.settings import settings

SESSION_COOKIE_NAME = "reiduq_session"
SESSION_TTL_SECONDS = 8 * 60 * 60  # 8h: long enough for a review shift, short enough to matter if leaked


def new_correlation_id() -> str:
    return uuid.uuid4().hex[:16]


def verify_api_key(presented: str) -> str | None:
    """Public: also used by /v1/auth/login to exchange a key for a session."""
    for key, role in settings.api_key_map.items():
        if hmac.compare_digest(presented, key):
            return role
    return None


def _sign(payload: bytes) -> str:
    return hmac.new(settings.secret_key.encode(), payload, hashlib.sha256).hexdigest()


def issue_session_token(role: str) -> str:
    """<base64 json payload>.<hmac signature>. Opaque to the browser, verifiable by us."""
    body = json.dumps({"role": role, "exp": int(time.time()) + SESSION_TTL_SECONDS}).encode()
    encoded = base64.urlsafe_b64encode(body).decode()
    return f"{encoded}.{_sign(encoded.encode())}"


def _verify_session_token(token: str) -> str | None:
    """Returns the role if the signature is valid and the token has not expired."""
    try:
        encoded, signature = token.split(".", 1)
    except ValueError:
        return None
    if not hmac.compare_digest(signature, _sign(encoded.encode())):
        return None  # signature check first: never JSON-parse an unverified payload
    try:
        payload = json.loads(base64.urlsafe_b64decode(encoded.encode()))
    except Exception:
        return None
    if payload.get("exp", 0) < time.time():
        return None
    role = payload.get("role")
    return role if isinstance(role, str) else None


async def require_api_key(
    request: Request, authorization: str = Header(default="")
) -> str:
    """Return the caller's role from either credential path, or 401.

    Checks the bearer header first (server-to-server callers set it and
    nothing else), then falls back to the session cookie (what the browser
    console sends). Never echoes the presented credential in an error.
    """
    if authorization.startswith("Bearer "):
        role = verify_api_key(authorization.removeprefix("Bearer ").strip())
        if role is not None:
            return role

    cookie = request.cookies.get(SESSION_COOKIE_NAME)
    if cookie:
        role = _verify_session_token(cookie)
        if role is not None:
            return role

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="missing or invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_admin(role: str = Depends(require_api_key)) -> str:
    if role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="admin role required")
    return role


def validate_image_bytes(blob: bytes) -> None:
    """Magic-byte, size and pixel-count checks before an image is ever decoded."""
    if len(blob) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="upload exceeds the size limit")
    magics = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n")
    if not any(blob.startswith(m) for m in magics):
        raise HTTPException(status_code=415, detail="only JPEG and PNG are accepted")


def validate_dimensions(width: int, height: int) -> None:
    if width * height > settings.max_image_pixels:
        raise HTTPException(status_code=413, detail="image dimensions exceed the limit")
