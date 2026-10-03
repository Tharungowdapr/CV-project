"""The three-way decision layer.

UNCERTAIN and REJECT are kept distinct on purpose. UNCERTAIN means the evidence
is ambiguous and a human can resolve it; REJECT means the evidence positively
indicates no match exists in the gallery - an open-set outcome. A two-way
accept/reject system throws away exactly the information a review queue needs.
"""

from __future__ import annotations

import numpy as np

from reiduq.core.types import Decision


def decide(
    confidence: float,
    matched_id: str | None,
    tau_high: float,
    tau_low: float,
    *,
    visibility: float | None = None,
) -> Decision:
    if confidence >= tau_high:
        return Decision("MATCH", matched_id, confidence, f"confidence {confidence:.3f} >= {tau_high:.3f}")
    if confidence >= tau_low:
        detail = f"confidence {confidence:.3f} in the ambiguous band [{tau_low:.3f}, {tau_high:.3f})"
        if visibility is not None and visibility < 0.5:
            detail += f"; only {visibility:.0%} of the vehicle is visible"
        return Decision("UNCERTAIN", matched_id, confidence, detail + " - route to human review")
    return Decision("REJECT", None, confidence, f"confidence {confidence:.3f} < {tau_low:.3f}; no match asserted")


def decide_batch(
    confidence: np.ndarray, matched_ids: list[str], tau_high: float, tau_low: float
) -> list[Decision]:
    return [
        decide(float(c), mid, tau_high, tau_low)
        for c, mid in zip(confidence, matched_ids, strict=True)
    ]
