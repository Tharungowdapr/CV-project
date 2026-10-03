# File map

What each file is, why it exists, and when you would touch it.

## Root

| Path | What / Why | Modify when |
|---|---|---|
| `pyproject.toml` | Single source for deps, ruff, black, mypy, pytest, coverage gate, import contracts | Adding a dependency (pin it) or changing a tool setting |
| `Makefile` | Every command anyone needs; `make help` lists them | Adding a workflow |
| `.env.example` | Every environment variable with a comment. Real values go in `.env`, never committed | Adding a config knob |
| `.pre-commit-config.yaml` | ruff, black, mypy, gitleaks, large-file guard | Adding a hook |
| `pnpm-workspace.yaml` | Declares `apps/*` as workspaces | Adding an app |

## `src/reiduq/core/` — infrastructure

| Path | What / Why | Modify when |
|---|---|---|
| `types.py` | Frozen dataclasses every layer is typed against. Freeze early; changing them later is expensive | A genuinely new data concept appears |
| `config.py` | Pydantic schemas, YAML `_base_` inheritance, CLI overrides, SHA-256 config hash. `extra="forbid"` turns a typo into a crash | Adding a config field |
| `settings.py` | Environment-driven runtime settings. No secret is ever hardcoded | Adding an env var |
| `registry.py` | String → class registries so configs name components without imports | Adding a component category |
| `paths.py` | `safe_path` containment check against traversal | Basically never |
| `seeding.py` | Seeds python/numpy/torch/CUDA and dataloader workers | Adding a new RNG source |
| `logging.py` | structlog JSON with `run_id`/`config_hash` on every record | Changing log shape |
| `exceptions.py` | Typed domain errors | Adding an error category |

## `src/reiduq/data/`

| Path | What / Why |
|---|---|
| `base.py` | `Sample` record and the dataset ABC every adapter implements |
| `veri776.py` | VeRi-776 adapter; parses identity/camera/frame from filenames |
| `veriwild.py` | VERI-Wild / VERI-Wild 2.0 adapter; reads identity/camera from `train_test_split/*.txt` list files (different layout from VeRi-776) |
| `torch_dataset.py` | `ReIDImageDataset` — remaps identities to contiguous labels for the classifier head |
| `transforms.py` | Train/eval image transforms; random erasing here is NOT the occlusion protocol (see `occlusion/compositor.py`) |
| `splits.py` | **Correctness-critical.** Builds manifests, hashes them, and raises `SplitLeakageError` automatically on load |
| `samplers.py` | PK identity sampler — triplet loss needs positives inside the batch |
| `occlusion/geometry.py` | Plausibility rules; rejects floating or implausibly scaled occluders |
| `occlusion/occluder_bank.py` | Bank of real segmented occluder patches |
| `occlusion/compositor.py` | Binary-searches placement to hit a target visibility; emits a replayable recipe |
| `occlusion/miner.py` | Finds naturally occluded samples for the real-occlusion protocol |

## `src/reiduq/visibility/`

| Path | What / Why |
|---|---|
| `estimator.py` | Three estimators for `v`, each explicit about its assumption; the single-image one reports an error band |
| `parts.py` | 6-region visibility vector — *which* parts are hidden, not just how much |
| `bands.py` | O0–O4 bucketing for per-band reporting |

## `src/reiduq/calibration/` — the contribution (100% coverage required)

| Path | What / Why |
|---|---|
| `base.py` | `BaseCalibrator` with `n_params` and `inference_multiplier` in the interface, because the efficiency table is a deliverable |
| `raw.py` | Uncalibrated softmax — the status quo being criticised |
| `global_temperature.py` | Guo et al. 2017, one scalar. Baseline and argument target |
| `platt.py`, `vector_scaling.py` | Classic post-hoc baselines; vector scaling rules out "more parameters" as the explanation |
| `conditional_temperature.py` | **The method.** ~6k-param MLP predicting `T(v, c)`. Saves feature mean/std with the checkpoint |
| `mc_dropout.py`, `deep_ensemble.py` | Expensive strong baselines; anchor the cost/quality trade-off |
| `evidential.py` | Single-pass modern competitor — the most dangerous baseline |
| `features.py` | Builds the 30-d conditioning vector; disabled groups are zeroed so checkpoints stay loadable across ablations |
| `camera_descriptor.py` | Label-free domain descriptor from image statistics |

## `src/reiduq/aggregation/`, `abstention/`

| Path | What / Why |
|---|---|
| `ess.py` | Effective-sample-size discount; without it, log-pooling near-duplicate frames manufactures confidence |
| `visibility_weighted.py`, `inverse_temperature.py` | Two fusion rules |
| `best_frame.py` | The simple baseline the aggregation contribution must beat |
| `policy.py` | Threshold selection under an operator risk constraint — never hand-tuned |
| `conformal.py` | Split-conformal risk control with a stated finite-sample guarantee |
| `decision.py` | Three-way verdict; keeps UNCERTAIN and REJECT distinct on purpose |

## `src/reiduq/eval/`

| Path | What / Why |
|---|---|
| `metrics/calibration.py` | ECE (equal-mass default), aECE, MCE, Brier + decomposition, NLL, reliability curves |
| `metrics/selective.py` | Risk–coverage, AURC, FAR@coverage, coverage@risk, AUROC |
| `metrics/retrieval.py` | CMC, mAP, mINP |
| `metrics/efficiency.py` | Params, p50/p95 latency, throughput, peak memory |
| `bootstrap.py` | CIs and paired bootstrap tests — single-run numbers get revision requests |
| `figures/builder.py` | F1–F6 from committed result JSONs, never from a live run |

## `src/reiduq/models/`

| Path | What / Why |
|---|---|
| `builder.py` | Encoder construction and L2 normalisation; falls back to a plain ResNet-50 if a registered backbone is unavailable |
| `backbones/resnet_ibn.py` | ResNet-50-IBN-a; falls back to plain ResNet-50 without the optional `resnet_ibn` package |
| `backbones/transreid.py` | ViT-B/16 with overlapping-patch embedding; used for Experiment K (backbone transfer) |
| `losses/{ce_smooth,triplet_wrt,center}.py` | Label-smoothed CE, weighted-regularised triplet, center loss — the standard three-loss Re-ID recipe |
| `training/trainer.py` | Training loop with laptop-safety built in: per-epoch resumable checkpoints, CUDA-OOM batch retry/skip, SIGINT/SIGTERM-safe shutdown, CPU thread cap, optional GPU thermal pause |

## `src/reiduq/search/` — the plate/photo search feature

| Path | What / Why |
|---|---|
| `gallery_store.py` | SQLite metadata (image path, camera, plate text/confidence) alongside the FAISS index; one row per indexed image |
| `indexer.py` | Offline job: embeds every image in a dataset, best-effort OCRs a plate crop, writes the FAISS index + metadata store. Run via `scripts/build_gallery_index.py` |
| `plate_ocr.py` | Best-effort plate reading (EasyOCR if installed, clean no-op fallback otherwise). Explicitly NOT a verified identifier - see the module docstring on why these datasets make plate OCR unreliable |
| `query_pipeline.py` | `SearchEngine`: photo search (embed → FAISS top-k → calibrate → verdict) and plate search (fuzzy text lookup → anchor image → re-run as a photo search). Every result carries a calibrated Match/Uncertain/Reject verdict, never a bare similarity |

## `src/reiduq/serving/`

| Path | What / Why |
|---|---|
| `app.py` | FastAPI wiring: CORS allowlist, rate limit, security headers, scrubbed error handler |
| `security.py` | Constant-time key comparison, HMAC-signed session tokens with expiry, role dependencies, upload validation |
| `audit.py` | SQLite-backed audit trail of match decisions (ids/verdict/confidence only, never images); swap for Postgres behind the same interface at multi-replica scale |
| `predictor.py` | Shares the research calibration path so demo and paper cannot drift |
| `schemas.py` | Pydantic request/response; `extra="forbid"` and a strict id pattern |

## `apps/web/`

| Path | What / Why |
|---|---|
| `next.config.mjs` | CSP, no inline scripts, standalone output |
| `src/lib/api.ts` | Typed fetch client; session cookie only, API key never held client-side, `UnauthorizedError` on a 401 |
| `src/hooks/useSession.ts` | Login/logout state; source of truth is the httpOnly cookie, not client state |
| `src/app/login/page.tsx` | Exchanges an API key for a session once; the key is never stored |
| `src/components/features/MatchConsole.tsx` | The console: adjust visibility and watch confidence move while similarity does not |
| `src/components/ui/ConfidenceBar.tsx` | Shows calibrated probability with both thresholds marked |

## `tests/`

| Path | What / Why |
|---|---|
| `unit/test_splits.py` | Asserts leakage is *detected*. The most important test in the repo |
| `unit/test_calibration_metrics.py` | Hand-computable ECE cases and bounds |
| `unit/test_calibrators.py` | Every calibrator beats raw on overconfident data |
| `property/` | Hypothesis invariants for the selective metrics |
| `regression/test_golden_metrics.py` | Golden numbers, tolerance 1e-2. CI catches drift before the paper does |
| `security/test_api_security.py` | Auth, headers, payload rejection |

## `docker/`, `infra/`, `.github/`

| Path | What / Why |
|---|---|
| `docker/Dockerfile` | Multi-stage, digest-pinned base, non-root uid 10001, healthcheck |
| `docker/docker-compose.yml` | Read-only rootfs, dropped capabilities, no-new-privileges |
| `infra/*.tf` | Cloud-agnostic scaffold with the same security posture |
| `.github/workflows/ci.yml` | lint → contracts → types → tests → coverage artifact |
| `.github/workflows/security.yml` | gitleaks, bandit, pip-audit, trivy; weekly schedule for new CVEs |
| `.github/workflows/cd.yml` | Build → scan → SBOM → cosign sign → staging → manual prod approval |
