# Base image digest below was recorded when this file was written; a
# tag can move underneath you (silently pulling a different image on a
# later build), a digest cannot. It WILL go stale as python:3.11-slim-bookworm
# is patched upstream - re-pin periodically with scripts/pin_base_images.sh
# rather than trusting it indefinitely.
# ── builder ────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm@sha256:ad5dadd8d1a56c62f8ee2b2ba81b0d6b09d2aa1b8e0e8fdfd3dd5ff9a3e0a7ff AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install .

# ── runtime ────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm@sha256:ad5dadd8d1a56c62f8ee2b2ba81b0d6b09d2aa1b8e0e8fdfd3dd5ff9a3e0a7ff AS runtime

# libgl/libglib are needed by opencv even in headless builds
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 --shell /usr/sbin/nologin reiduq

COPY --from=builder /install /usr/local
WORKDIR /app
COPY --chown=reiduq:reiduq configs ./configs

USER reiduq
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 APP_ENV=production
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "reiduq.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
