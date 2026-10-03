# Reproducibility

## The rule

No result exists without its config. Every artifact records the config SHA-256, the git SHA,
the seed and the hardware string. If a number cannot be traced back to those four things, it
does not go in the paper.

## Seeding

`seed_everything()` covers `PYTHONHASHSEED`, `random`, `numpy`, `torch`, CUDA, and — via
`worker_init_fn` — every dataloader worker. Deterministic cuDNN algorithms are requested and
`CUBLAS_WORKSPACE_CONFIG` is set.

## Known non-determinism

| Source | Effect | Mitigation |
|---|---|---|
| `atomicAdd` in some CUDA kernels | Last-bit differences in gradients | Accepted; results are reported over 5 seeds with CIs |
| FAISS IVF-PQ training | Different centroids per run | Exact `IndexFlatIP` is used for every reported number; IVF-PQ appears only in the throughput study |
| Multi-threaded JPEG decode | Decode order, not content | No effect on results |
| `ultralytics` auto-download | Silent weight changes | Weights are loaded from a local pinned path only |

## Reproducing the results

```bash
make setup
python scripts/prepare_splits.py --root data/raw/VeRi     # commits split manifests
make experiment CFG=configs/experiment/A_indist.yaml      # must match published Rank-1/mAP ±1
make sweep                                                # A–K
make figures
```

Step 3 is a gate, not a formality. Calibration findings on an under-trained encoder are
worthless, so nothing downstream should be run until the baseline reproduces.

## Split manifests

Manifests are JSON, content-hashed, and committed. Editing one by hand invalidates the hash
and `load_manifest` refuses to load it. Disjointness is checked on every load, and camera
disjointness is additionally checked under domain-shift protocols.

## Golden regression tests

`tests/regression/test_golden_metrics.py` pins metric values on a fixed synthetic split. If a
refactor moves them, CI fails. When the change is intentional, update the constants and say why
in the PR — the failure is the mechanism, not an obstacle.

## Release checklist

- [ ] Split manifests committed
- [ ] Occlusion recipes exported as JSON
- [ ] Config hash recorded in every result file
- [ ] Model cards written for every released checkpoint
- [ ] Efficiency table regenerated on the stated hardware
- [ ] Figures regenerate from committed result JSONs alone
