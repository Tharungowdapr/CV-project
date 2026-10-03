# Ethics and intended use

## The honest framing

Vehicle re-identification is surveillance technology. Publishing an improvement to it is a
choice with consequences, and the reason this particular improvement is worth making is that it
reduces a specific harm: **confident wrong identity association**.

A false cross-camera match does not stay inside the system. It propagates into a wrong
trajectory, a wrong location history, and a wrong conclusion about a person — and because the
score attached to it looks high, nobody downstream questions it. An abstention costs an analyst
a minute. A confident error can cost someone a great deal more.

## Commitments in this work

1. **Abstention is on by default.** The system is configured to refuse rather than guess.
2. **UNCERTAIN means a human decides.** Never an automatic action.
3. **Public research datasets only.** No scraped footage, no operational data.
4. **No occupant inference.** The system models vehicles. It does not attempt to identify,
   count, or characterise people.
5. **Documented failure ranges.** The model card states where the calibration guarantee does not
   hold, because a guarantee that is quietly out of scope is worse than none.
6. **Logs contain ids, never imagery.**

## Dual-use acknowledgement

Better Re-ID can be used for tracking that a person has not consented to. Withholding the
calibration work would not remove that capability — uncalibrated Re-ID is already deployed —
but it would leave those systems unable to express doubt. The argument for publishing is that
an honest system is safer than a confident one, not that the technology is neutral.

## For the paper

Include a short ethics statement covering: intended use, the abstention requirement, datasets
used and their licences, the absence of occupant inference, and the recommendation that
deployments route `UNCERTAIN` results to human review. Several Q1 venues now require this, and
it is the correct statement to make regardless.
