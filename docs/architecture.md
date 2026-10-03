# Architecture

## Layering

```
L5 PRESENTATION   CLI · FastAPI service · Next.js console
L4 ORCHESTRATION  experiment runner · pipeline · sweeps
L3 DOMAIN         detect · segment · track · visibility · encode ·
                  calibrate · aggregate · abstain · retrieve · evaluate
L2 DATA           dataset adapters · split manager · occlusion compositor
L1 INFRA          config · logging · seeding · registry · paths
```

**Dependency rule:** an inner layer never imports an outer one. Enforced in CI by
`import-linter` via the contract in `pyproject.toml`; a violation fails the build rather than
being caught in review.

## Patterns and where they are used

| Pattern | Where | Why it earns its place |
|---|---|---|
| Registry + factory | detectors, segmenters, trackers, backbones, calibrators, aggregators | A new calibrator is a decorator plus a YAML file. The runner never changes, so adding method #8 cannot break methods #1–7 |
| Strategy | calibrators, fusion rules, abstention policies | Seven calibrators behind one protocol makes the comparison table mechanical |
| Adapter | dataset loaders | Four datasets, one `Sample` record. Cross-dataset experiments become a config change |
| Pipeline | stage execution | Each stage is `(Context) -> Context`: individually testable and cacheable |
| Ports and adapters | serving | Domain code has no knowledge of FastAPI, so the research path and the API path cannot drift |

## The two operating modes

**Offline research mode** — deterministic, batched, cached. Every number in the paper comes
from here. Fixed seeds, cached embeddings, versioned splits.

**Online inference mode** — streaming, bounded latency. Used for the efficiency study and the
console. Shares the identical weights and calibration head.

Keeping both on one codebase behind a shared interface prevents the classic failure where the
reported numbers cannot be reproduced by the shipped system.

## Data flow

1. Frames in, detections out (YOLOv8, conf ≥ 0.35, vehicle classes only).
2. Instance masks per detection. Masks do three jobs: measure visibility, suppress background,
   and give part-level structure.
3. BoT-SORT builds tracklets; an embedding-consistency check splits identity switches.
4. Visibility `v` is computed by whichever estimator the protocol allows.
5. The encoder produces an L2-normalised embedding from the mask-gated crop.
6. FAISS returns top-k similarities.
7. The calibrator converts similarities to a probability, conditioned on `v`, part visibility
   and a camera descriptor.
8. Tracklet fusion pools frame-level evidence with an ESS discount.
9. The abstention layer emits MATCH / UNCERTAIN / REJECT under a stated risk constraint.

## Caching

Embedding extraction over VERI-Wild takes hours. A stage is skipped when an artifact exists
under `{config_hash}__{stage}`. Without this the ablation grid is not runnable in the
available time.
