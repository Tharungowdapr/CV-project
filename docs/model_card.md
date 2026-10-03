# Model card — calibrated vehicle Re-ID

## Details
- **Task:** cross-camera vehicle re-identification with calibrated confidence and abstention
- **Components:** YOLOv8 detection/segmentation → BoT-SORT → OSNet-AIN encoder → conditional
  temperature head → tracklet fusion → abstention layer
- **Calibrator size:** ~6k parameters; roughly 1.001× the encoder's inference cost
- **Training data:** VeRi-776 (public research dataset)
- **Licence:** MIT (code). Dataset licences apply separately and are more restrictive.

## Intended use
Research on uncertainty calibration in Re-ID, and operator-assisted forensic review where a
human adjudicates every `UNCERTAIN` result.

## Out of scope
- Automatic enforcement action from a `MATCH` verdict without human review
- Person re-identification or any inference about vehicle occupants
- Deployment outside the evaluated shift ranges (visibility < 0.2, camera types absent from
  training) — the calibration guarantee does not extend there, and it will fail quietly

## Performance
Populate from `outputs/results/` after running the A–K matrix. Report at minimum: Rank-1, mAP,
mINP, ECE, MCE, Brier, AURC, and false-association rate at 50/70/90% coverage, each per
occlusion band, as mean ± std over 5 seeds with bootstrap CIs.

## Limitations
- Visibility for single images is *estimated*, not measured; the error band is reported but it
  propagates into the conditioning.
- Conformal guarantees assume exchangeability between calibration and test data. Cross-dataset
  evaluation deliberately violates this, and the guarantee is correspondingly weaker there — it
  is reported as observed coverage, not as a claim.
- Identity switches inside a tracklet corrupt aggregated confidence. Mitigated by consistency
  splitting; the residual split rate is reported.
- Vehicle type is memorised more than generalised (a known field-wide result); performance on
  unseen makes and models degrades, and calibration may degrade faster than accuracy.

## Ethics
See `docs/ethics.md`.
