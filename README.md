# 🚘 Vehicle Re-ID Uncertainty Platform

> **Uncertainty-calibrated, multi-camera vehicle re-identification with risk-bounded abstention (`MATCH / UNCERTAIN / REJECT`), a 10,000-record CSV dataset pipeline, and an Excel-style web management console.**

[![CI](https://github.com/Tharungowdapr/CV-project/actions/workflows/ci.yml/badge.svg)](https://github.com/Tharungowdapr/CV-project/actions/workflows/ci.yml)
[![Security](https://github.com/Tharungowdapr/CV-project/actions/workflows/security.yml/badge.svg)](https://github.com/Tharungowdapr/CV-project/actions/workflows/security.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🗺️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION & PROCESSING                          │
│                                                                              │
│   Video / RTSP ──▶ RF-DETR Detector ──▶ SAM 2.1 Segmenter ──▶ ByteTrack    │
│   ZIP / Images ──▶ Dataset Processor ──▶ CSV Manifest (veri776_*.csv)       │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼─────────────────────────────────────────┐
│                      FEATURE EXTRACTION & CALIBRATION                        │
│                                                                              │
│   Occlusion / Visibility Estimator ──▶ TransReID / ResNet-IBN Encoder       │
│   Camera Descriptor ──▶ Conditional Temperature Scaling T(v, c)             │
│   ──▶ Calibrated Confidence ──▶ Risk Estimator                              │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼─────────────────────────────────────────┐
│                       ABSTENTION & DECISION LAYER                            │
│                                                                              │
│   Confidence ≥ θ_high  ──▶  ✅ MATCH                                        │
│   θ_low ≤ Conf < θ_high ──▶  ⚠️  UNCERTAIN  (human adjudication queue)      │
│   Confidence < θ_low   ──▶  ❌ REJECT                                        │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼─────────────────────────────────────────┐
│                          SERVING & WEB CONSOLE                               │
│                                                                              │
│   FastAPI REST API (:8000) ── /v1/match  /v1/search  /v1/dataset            │
│   Next.js Console (:3000)  ── Dataset Manager · Search Console · Live Feed  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

| Feature | Detail |
|---|---|
| **10 k-Record Dataset** | Synthetic VeRi-776 generator + ZIP ingest pipeline; outputs `veri776_all/train/query/gallery.csv` |
| **Excel-Style Grid** | Sortable, filterable, paginated spreadsheet UI for exploring 10,000 records |
| **Hybrid Search** | Streaming CSV full-text search (no index needed) + FAISS vector similarity search |
| **Calibrated Uncertainty** | Conditional temperature scaling replaces raw Cosine distance with meaningful probabilities |
| **Risk Abstention** | Three-tier decision: MATCH / UNCERTAIN (queued for review) / REJECT |
| **Audit Log** | Every match decision stored in SQLite with correlation IDs |
| **CI / CD** | GitHub Actions: unit tests, lint, Next.js build, Docker image push to GHCR, Trivy scan |
| **Docker Ready** | Multi-stage hardened Dockerfile (uid 10001, read-only rootfs, all caps dropped) |

---

## 🛠️ Tech Stack

```
Backend   Python 3.11 · FastAPI · Uvicorn · PyTorch · FAISS · OpenCV · Structlog
Frontend  Next.js 14 · TypeScript · Tailwind CSS · Lucide Icons
CV / AI   TransReID · ResNet-IBN · RF-DETR · SAM 2.1 · YOLOv8 · ByteTrack
Infra     Docker · GitHub Actions · Terraform (optional)
```

---

## 📁 Repository Layout

```
CV-project/
├── .github/
│   └── workflows/
│       ├── ci.yml          # Python tests + Next.js build
│       ├── cd.yml          # Docker build → GHCR push → Trivy scan
│       └── security.yml    # Gitleaks · Bandit · pip-audit · Trivy FS scan
│
├── apps/
│   └── web/                # Next.js 14 frontend web console
│       ├── src/app/        # App Router pages
│       │   ├── dataset/    # ← MAIN PAGE: Excel grid dataset manager
│       │   ├── search/     # Search console (CSV + FAISS)
│       │   ├── database/   # Gallery vector browser
│       │   ├── ingest/     # Video / ZIP ingest UI
│       │   └── live/       # RTSP live stream monitor
│       └── src/
│           ├── components/ # Shared UI components
│           ├── lib/        # API client (api.ts)
│           └── types/      # TypeScript types
│
├── configs/
│   ├── base.yaml           # Shared hyperparameters
│   └── experiment/         # A–K ablation experiment configs
│
├── data/
│   ├── processed/          # Generated CSVs (gitignored — run generator)
│   ├── raw/                # Place VeRi-776 / VeriWild here (gitignored)
│   └── external/           # Third-party data (gitignored)
│
├── docker/
│   ├── Dockerfile          # Multi-stage hardened API container
│   ├── Dockerfile.web      # Next.js production container
│   └── docker-compose.yml  # Full stack: api + web + mlflow
│
├── docs/
│   ├── architecture.md     # Detailed layer descriptions
│   ├── api.md              # REST API reference
│   ├── data_card.md        # Dataset documentation
│   ├── ethics.md           # Surveillance ethics statement
│   ├── model_card.md       # Model card (accuracy, limitations)
│   ├── threat_model.md     # Security threat model
│   ├── reproducibility.md  # How to reproduce all experiments
│   └── FILE_MAP.md         # Every file explained
│
├── infra/                  # Terraform IaC (optional cloud deploy)
│
├── scripts/
│   ├── generate_veri_dataset.py  # Synthetic dataset generator (10 k records)
│   ├── process_dataset.py        # ZIP → CSV manifest pipeline
│   ├── build_gallery_index.py    # CSV → FAISS + SQLite gallery index
│   ├── prepare_splits.py         # Train / query / gallery split generator
│   ├── mine_occluders.py         # Occlusion pattern miner
│   ├── benchmark.py              # Calibrator latency benchmark
│   ├── setup.sh                  # One-shot Linux/macOS env setup
│   ├── train_safe.sh             # Auto-restart training wrapper
│   ├── deploy.sh                 # Terraform staging/production deploy
│   └── pin_base_images.sh        # Re-pin Docker base image digests
│
├── src/reiduq/             # Core Python package
│   ├── abstention/         # Risk decision: conformal, policy
│   ├── aggregation/        # Multi-frame score fusion
│   ├── calibration/        # Temperature scaling, Platt, evidential
│   ├── core/               # Config, settings, registry, logging, paths
│   ├── data/               # Dataset adapters (VeRi-776, VeriWild), samplers
│   ├── detection/          # RF-DETR / YOLOv8 detectors
│   ├── eval/               # Metrics: ECE, mAP, selective accuracy
│   ├── experiments/        # Experiment runner (A–K matrix)
│   ├── models/             # TransReID / ResNet-IBN backbones & losses
│   ├── pipeline/           # End-to-end processor
│   ├── retrieval/          # FAISS index wrapper
│   ├── search/             # Plate OCR, gallery store, query pipeline
│   ├── segmentation/       # SAM 2.1 / YOLOv8-seg wrappers
│   ├── serving/            # FastAPI app, schemas, security, audit
│   ├── tracking/           # ByteTrack tracklet builder
│   └── visibility/         # Part-level occlusion estimator
│
├── tests/
│   ├── unit/               # 74 fast tests (no GPU, no dataset)
│   ├── integration/        # End-to-end pipeline tests
│   ├── property/           # Hypothesis property tests
│   ├── regression/         # Golden metric regression tests
│   └── security/           # API boundary & auth tests
│
├── .gitignore
├── .pre-commit-config.yaml  # ruff · black · mypy · gitleaks · bandit
├── Makefile                 # All dev commands (see below)
└── pyproject.toml           # Build system, deps, tool config
```

---

## 🚀 Quick Start

### Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.11 – 3.12 | 3.11 recommended |
| Node.js | 20+ | Frontend only |
| Git | any | — |
| CUDA | 11.8+ | Optional; CPU works for the API |

### 1. Clone

```bash
git clone https://github.com/Tharungowdapr/CV-project.git
cd CV-project
```

### 2. Install Everything (One Command)

```bash
make setup
# Equivalent to:
#   pip install -e ".[dev]"
#   pre-commit install
#   cd apps/web && npm install
```

On **Windows** without `make`:
```powershell
python -m pip install -U pip
pip install -e ".[dev]"
cd apps/web; npm install; cd ..\..
```

### 3. Generate the Dataset

```bash
python scripts/generate_veri_dataset.py --train 5000 --query 1000 --gallery 4000
# Creates data/processed/veri776_all.csv  (10 000 rows)
```

### 4. Run Locally

```bash
# Terminal 1 – Backend API
make serve
# OR: uvicorn reiduq.serving.app:app --reload --port 8000

# Terminal 2 – Frontend Console
make web
# OR: cd apps/web && npm run dev
```

- **API**: [http://localhost:8000](http://localhost:8000) — Swagger docs at `/docs`
- **Console**: [http://localhost:3000](http://localhost:3000)
- **Dataset Manager**: [http://localhost:3000/dataset](http://localhost:3000/dataset) ← **Start here**

---

## 🛠️ Makefile Command Reference

```bash
make help          # List all commands
make setup         # Install Python + Node deps, pre-commit hooks
make run           # Start API + frontend together (parallel)
make serve         # FastAPI backend on :8000
make web           # Next.js frontend on :3000
make test          # Unit test suite (fast, no GPU needed)
make test-all      # Full suite including slow GPU tests
make lint          # ruff + black + lint-imports
make format        # Auto-fix formatting (ruff --fix + black)
make typecheck     # mypy strict type check
make security      # bandit static analysis + pip-audit
make benchmark     # Measure calibrator latency/params/FLOPs
make docker-up     # docker compose up: api + web + mlflow
make docker-down   # docker compose down -v
make train         # Train Re-ID encoder (requires dataset)
make train-safe    # Overnight training with auto-restart on OOM
make calibrate     # Fit calibrator on calibration split
make evaluate      # Evaluate trained + calibrated system
make experiment    # Run one named experiment end-to-end
make sweep         # Run full A–K experiment matrix
make figures       # Regenerate F1–F6 paper figures
make clean         # Remove build artefacts and caches
```

Pass a config: `make train CFG=configs/experiment/A_indist.yaml`

---

## 📜 Scripts Reference

| Script | How to Run | What It Does |
|---|---|---|
| `generate_veri_dataset.py` | `python scripts/generate_veri_dataset.py --train 5000 --query 1000 --gallery 4000` | Generates synthetic VeRi-776 CSV dataset manifests |
| `process_dataset.py` | `python scripts/process_dataset.py --zip dataset.zip --name my_dataset` | Ingests any image ZIP into a normalised CSV manifest |
| `build_gallery_index.py` | `python scripts/build_gallery_index.py --csv data/processed/veri776_all.csv` | Builds FAISS index + SQLite metadata store |
| `prepare_splits.py` | `python scripts/prepare_splits.py` | Generates leakage-free train/query/gallery splits |
| `mine_occluders.py` | `python scripts/mine_occluders.py` | Mines occlusion patches from training images |
| `benchmark.py` | `python scripts/benchmark.py` | Calibrator throughput + latency report |
| `setup.sh` | `bash scripts/setup.sh` | Linux/macOS one-shot setup (venv + deps) |
| `train_safe.sh` | `bash scripts/train_safe.sh configs/experiment/A_indist.yaml` | Safe overnight training with checkpoint resume |
| `deploy.sh` | `bash scripts/deploy.sh staging` | Terraform deploy to staging or production |
| `pin_base_images.sh` | `bash scripts/pin_base_images.sh` | Re-pin Docker base image SHA256 digests |

---

## 🔄 CI / CD Pipeline

```
Push / PR
   │
   ├──▶ CI (ci.yml)
   │       ├── Python: Install → ruff → black → pytest tests/unit
   │       └── Web: npm ci → tsc → eslint → next build
   │
   └──▶ CD (cd.yml)  [main branch only]
           ├── Docker build → push ghcr.io/Tharungowdapr/CV-project:sha
           ├── Trivy image vulnerability scan
           └── SBOM generation (spdx-json)

Weekly
   └──▶ Security (security.yml)
           ├── Gitleaks secret scan
           ├── Bandit static analysis
           ├── pip-audit dependency audit
           └── Trivy filesystem scan
```

---

## 🌐 API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `POST` | `/v1/match` | Re-ID match with calibrated confidence |
| `POST` | `/v1/search/photo` | Photo-to-gallery vector search |
| `GET` | `/v1/search/plate` | License plate text search |
| `GET` | `/v1/dataset/list` | List all processed dataset CSVs |
| `GET` | `/v1/dataset/{name}/rows` | Paginated CSV row access |
| `GET` | `/v1/dataset/search` | Full-text streaming search across CSVs |
| `POST` | `/v1/dataset/ingest` | Upload + process a ZIP dataset |
| `GET` | `/v1/gallery` | Paginated gallery vector entries |
| `GET` | `/v1/gallery/stats` | Gallery aggregate statistics |
| `GET` | `/v1/admin/audit` | Recent match audit log |

Full documentation: `docs/api.md` or browse `/docs` on a running server.

---

## 🔒 Security & Ethics

- API keys with role-based access (`analyst` / `admin`)
- Rate limiting per client (configurable via `RATE_LIMIT_PER_MINUTE`)
- Container runs as uid 10001, read-only rootfs, all Linux capabilities dropped
- `UNCERTAIN` results are **never** used for automatic action — always require human review
- See `docs/threat_model.md` and `docs/ethics.md` for full details

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
