# Data card

## Datasets

| Dataset | Scale | Role | Access |
|---|---|---|---|
| VeRi-776 | ~50k images, 776 ids, 20 cameras | Training and in-distribution evaluation; camera ids enable leave-camera-out | Request from the authors |
| VERI-Wild | 400k+ images, 40k ids, 174 cameras | Large-scale cross-dataset stress test | Request from the authors |
| VehicleID | ~220k images | Same-model/same-colour confusion study | Request from the authors |
| CityFlow | City-scale MTMC | Tracklet-level and end-to-end evaluation | AI City Challenge registration |

None are redistributed with this repository. `data/` contains only `.gitkeep` files.

## Provenance and consent
All four are public research datasets collected from fixed traffic surveillance cameras and
released for research use. No consent was obtained from vehicle owners — this is a known
limitation of the field, not something this project resolves. Licence terms are more
restrictive than this repository's MIT licence and take precedence.

## Derived artifacts

| Artifact | How produced | Committed? |
|---|---|---|
| Split manifests | `scripts/prepare_splits.py`, content-hashed | Yes — reproducibility depends on it |
| Occluder bank | `scripts/mine_occluders.py` from real segmented objects | No — regenerable, and it contains image data |
| Occlusion recipes | Emitted by the compositor as JSON | Yes — this is what makes the synthetic set reproducible |
| Embeddings cache | Pipeline stage cache | No |

## Known biases
- Geographic and temporal concentration: each dataset comes from one city over a limited period.
- Vehicle-type imbalance: sedans dominate; trucks and buses are under-represented, which is
  exactly why the unseen-types protocol exists.
- Weather and lighting skew toward clear daytime conditions.
- Camera-height and viewpoint distributions are narrow relative to real deployments.

These are reasons the calibration question matters, not incidental caveats.
