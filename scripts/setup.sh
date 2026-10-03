#!/usr/bin/env bash
# One-command local setup. Safe to re-run.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> python environment"
python -m venv .venv 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"

echo "==> pre-commit hooks"
pre-commit install || echo "pre-commit unavailable, skipping"

echo "==> frontend"
if command -v pnpm >/dev/null 2>&1; then (cd apps/web && pnpm install); else (cd apps/web && npm install); fi

echo "==> environment file"
[ -f .env ] || cp .env.example .env
echo "    edit .env before running anything that touches real data"

echo "==> done. Next: make test, then make serve"
