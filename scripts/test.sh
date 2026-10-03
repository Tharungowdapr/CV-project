#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ruff check src tests
black --check src tests
mypy --strict src/reiduq
pytest -m "not slow"
