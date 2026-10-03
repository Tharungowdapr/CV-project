.DEFAULT_GOAL := help
SHELL := /bin/bash
PY := python
CFG ?= configs/experiment/A_indist.yaml

.PHONY: help setup build run test lint format typecheck security clean \
        docker-up docker-down deploy train calibrate evaluate experiment \
        sweep figures serve web benchmark

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

setup:  ## Install python + node deps and pre-commit hooks
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[dev]"
	pre-commit install
	cd apps/web && (pnpm install || npm install)
	cp -n .env.example .env || true

build:  ## Build python wheel and frontend bundle
	$(PY) -m pip wheel . -w dist --no-deps
	cd apps/web && (pnpm build || npm run build)

run:  ## Run API (dev) and frontend together
	$(MAKE) -j2 serve web

serve:  ## Start the inference API
	uvicorn reiduq.serving.app:app --host 0.0.0.0 --port $${APP_PORT:-8000} --reload

web:  ## Start the Next.js frontend
	cd apps/web && (pnpm dev || npm run dev)

test:  ## Run fast test suite with coverage gate
	pytest -m "not slow"

test-all:  ## Run everything including slow/GPU tests
	pytest

lint:  ## Lint + architecture contract check
	ruff check src tests
	black --check src tests
	lint-imports

format:  ## Auto-format
	ruff check --fix src tests
	black src tests

typecheck:  ## Strict type check
	mypy --strict src/reiduq

security:  ## Static security scan + dependency audit
	bandit -r src/reiduq -c pyproject.toml
	pip-audit --strict || true

train:  ## Train the Re-ID encoder    (make train CFG=configs/model/osnet_ain.yaml)
	$(PY) -m reiduq.cli train --config $(CFG)

train-safe:  ## Train with auto-restart-from-checkpoint on crash/OOM - use this for unattended overnight runs
	bash scripts/train_safe.sh $(CFG)

calibrate:  ## Fit a calibrator on the calibration split
	$(PY) -m reiduq.cli calibrate --config $(CFG)

evaluate:  ## Evaluate a trained + calibrated system
	$(PY) -m reiduq.cli evaluate --config $(CFG)

experiment:  ## Run a named experiment end to end
	$(PY) -m reiduq.cli experiment --config $(CFG)

sweep:  ## Run the full A–K experiment matrix
	$(PY) -m reiduq.cli sweep --configs configs/experiment

figures:  ## Regenerate paper figures F1–F6 from result JSONs
	$(PY) -m reiduq.cli figures --results outputs/results --out outputs/figures

benchmark:  ## Measure params/FLOPs/latency for every calibrator
	$(PY) scripts/benchmark.py

docker-up:  ## Start api + web + mlflow
	docker compose -f docker/docker-compose.yml up -d --build

docker-down:
	docker compose -f docker/docker-compose.yml down -v

deploy:  ## Deploy via the configured script (staging by default)
	bash scripts/deploy.sh $${ENV:-staging}

clean:
	rm -rf dist build .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
