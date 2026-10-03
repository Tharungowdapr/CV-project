# Uncertainty-Calibrated Vehicle Re-Identification

Cross-camera vehicle matching where the confidence number means what it claims, and the
system is allowed to say it does not know.

Current Re-ID systems always return a ranked answer with a similarity score, and that score
gets treated as a confidence. It is not one. Under occlusion and on unseen cameras, accuracy
collapses while the score stays high — the system is confidently wrong exactly when it matters.
This project measures how fast that happens, fixes it with a lightweight **conditional**
calibrator, and adds a Match / Uncertain / Reject decision layer with a stated risk guarantee.

---

## Architecture

```
  video ──▶ [1] detect ──▶ [2] segment ──▶ [3] track ──▶ [4] visibility
                                                              │
                                                              ▼
   decision ◀── [8] abstain ◀── [7] aggregate ◀── [6] calibrate ◀── [5] encode
       │                                               ▲
       │                                        T(v, c) head
       ▼
  MATCH / UNCERTAIN / REJECT

  Layers (inner never imports outer):
    serving ▸ experiments ▸ pipeline ▸ eval ▸ calibration ▸ data ▸ core
```

Stages 2, 4, 6, 7 and 8 carry the research contribution; the rest is standard machinery.

## Prerequisites

| Requirement | Version | Note |
|---|---|---|
| Python | 3.10 – 3.12 | 3.11 recommended |
| Node.js | 20+ | frontend only |
| CUDA | 11.8+ | optional; CPU works for metrics and the API |
| Docker | 24+ | optional |
| Disk | ~60 GB | VeRi-776 (~1 GB) + VERI-Wild (~40 GB) |

## Quickstart

```bash
git clone <repo> && cd vehicle-reid-uncertainty
make setup            # python deps, node deps, pre-commit, .env
make test             # 50+ tests, no dataset required
make serve            # API on :8000
make web              # console on :3000
```

Nothing above needs a dataset. Everything data-dependent starts at `scripts/prepare_splits.py`.

### Datasets

VeRi-776 and VERI-Wild are third-party research datasets, not redistributed here. Request access from the maintainers (VeRi-776: https://github.com/VehicleReId/VeRidataset; VERI-Wild: see the terms-of-access notice on the VERI-Wild release page) and accept their terms before downloading. Point `data.root` at the extracted folder:

```
data/raw/VeRi/        image_train/ image_query/ image_test/ list_type.txt
data/raw/VeriWild/    images/<vid>/<image>.jpg  train_test_split/*.txt  vehicle_info.txt
```

`data.name: veri776` and `data.name: veriwild` in a config select the matching adapter (`src/reiduq/data/veri776.py`, `src/reiduq/data/veriwild.py`); no other code changes.

## Common commands

| Command | What it does |
|---|---|
| `make setup` | Install everything and create `.env` |
| `make test` | Fast suite with the 85% coverage gate |
| `make lint` / `make typecheck` | ruff + black + import contracts / mypy strict |
| `make security` | bandit + pip-audit |
| `make train CFG=configs/experiment/A_indist.yaml` | Train the encoder |
| `make calibrate CFG=...` | Fit a calibrator on the calibration split |
| `make experiment CFG=...` | One experiment end to end |
| `make sweep` | The full A–K matrix |
| `make figures` | Regenerate F1–F6 from result JSONs |
| `make benchmark` | Efficiency table for every calibrator |
| `make docker-up` | api + web + mlflow |

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `APP_ENV` | `development` | Enables HSTS and disables `/docs` in production |
| `SECRET_KEY` | — | Generate with `openssl rand -hex 32`; never commit |
| `API_KEYS` | — | `key:role` pairs; roles are `analyst` and `admin` |
| `CORS_ORIGINS` | `http://localhost:3000` | Explicit allowlist, never `*` |
| `RATE_LIMIT_PER_MINUTE` | `60` | Per-client request cap |
| `MAX_UPLOAD_BYTES` | `10485760` | Upload size cap |
| `DATA_ROOT` | `./data` | All dataset access is confined under this root |
| `DEVICE` | `cuda` | `cuda`, `cpu` or `mps` |
| `SEED` | `42` | Global determinism |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Frontend → API |

## Folder structure

```
src/reiduq/        core · data · detection · segmentation · tracking · visibility
                   models · calibration · aggregation · abstention · retrieval
                   eval · pipeline · experiments · serving
apps/web/          Next.js review console
configs/           base.yaml + experiment/ + ablation/  (every run is a config)
tests/             unit · integration · property · regression · security
docker/ infra/     container and IaC
scripts/           setup, splits, occluder mining, benchmark, deploy
docs/              architecture, reproducibility, threat model, model card, ethics
```

`docs/FILE_MAP.md` explains every file.

## Testing

```bash
make test        # fast lane
pytest -m slow   # GPU / full-dataset tests
pytest tests/security -m security
```

The suite that matters most is `tests/unit/test_splits.py`: it asserts that **leakage is
detected**. If a refactor makes those tests pass trivially, the guard is gone and every
result after that point is unverifiable.

## Deployment

```bash
make docker-up                              # local stack
ENV=staging make deploy                     # terraform
CONFIRM_PRODUCTION=yes ENV=production make deploy
```

The container runs as uid 10001, read-only root filesystem, all capabilities dropped.
Production deploys require manual approval in the `production` GitHub environment.

## Contributing

1. Branch from `main`; direct pushes are blocked.
2. `pre-commit install` — ruff, black, mypy, gitleaks run on every commit.
3. New component → `@register("name")` + a config file. Never edit the runner.
4. Every new function gets a unit test; anything in `calibration/` or `eval/metrics/` needs 100%.
5. If a golden regression number changes, say why in the PR description.

## Ethics

Vehicle Re-ID is surveillance technology. This system exists to *reduce* wrongful identity
association: the abstention layer is on by default, `UNCERTAIN` results are intended for human
adjudication and never for automatic action, and only public research datasets are used. See
`docs/ethics.md`.

## License

MIT.
