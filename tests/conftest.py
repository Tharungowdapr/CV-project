"""Test-wide environment.

Settings are read at import time, so the test API keys must be in the
environment before ``reiduq.serving.app`` is imported anywhere.
"""

from __future__ import annotations

import os

os.environ.setdefault("API_KEYS", "devkey:analyst,adminkey:admin")
os.environ.setdefault("SECRET_KEY", "test-secret-not-a-real-key")
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "1000")  # rate limiting is tested separately
