"""Baseline fusion: confidence-weighted log-odds pooling.

This is deliberately a *baseline*, not the answer. It exists so the pipeline runs
end to end from day one and so later work has something to beat.

What is wrong with it, and must be fixed before any headline number is reported:

1. It assumes detectors are independent. They are not -- several compression
   detectors read the same underlying artefacts and will agree with each other
   for reasons that have nothing to do with the image being manipulated.
2. Its output is not calibrated. 0.9 does not currently mean "right 90% of the
   time", so the abstention thresholds below are not yet meaningful.
3. Weights are hand-set rather than learned.

Replace with a learned fusion (logistic regression or small GBM over detector
outputs) plus explicit calibration, and verify on a reliability diagram. Until
then, treat these numbers as ordering, not probability.
"""

from __future__ import annotations

import numpy as np

from ..core.types import Decision, Evidence, Verdict

# Below AUTO_CLEAR we clear it; above FLAG we flag it; the band between the two,
# and anything we are not confident about, goes to a human.
AUTO_CLEAR_BELOW = 0.30
FLAG_ABOVE = 0.70
MIN_CONFIDENCE_TO_DECIDE = 0.35

_EPS = 1e-6


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return float(np.log(p / (1.0 - p)))


def fuse(evidence: list[Evidence]) -> Verdict:
    """Combine detector evidence into a single verdict.

    Detectors that reported ``applicable=False`` are excluded entirely rather
    than folded in as neutral -- "I cannot speak to this" and "I looked and found
    nothing" are different claims and averaging them together is how a fused
    score quietly becomes meaningless.
    """
    usable = [e for e in evidence if e.applicable and e.confidence > 0]

    if not usable:
        return Verdict(
            manipulated_probability=0.5,
            decision=Decision.ROUTE_TO_HUMAN,
            evidence=evidence,
            explanation=(
                "No detector was able to assess this image. "
                "Routing to manual review."
            ),
        )

    weights = np.array([e.confidence for e in usable], dtype=float)
    logits = np.array([_logit(e.score) for e in usable], dtype=float)
    pooled = float((weights * logits).sum() / weights.sum())
    prob = float(1.0 / (1.0 + np.exp(-pooled)))

    # Aggregate confidence: dominated by the most confident detector, with
    # diminishing credit for corroboration. A single very sure detector should be
    # able to carry a decision; ten unsure ones should not.
    top = float(weights.max())
    corroboration = float(1.0 - np.prod(1.0 - weights))
    aggregate_confidence = max(top, corroboration)

    if aggregate_confidence < MIN_CONFIDENCE_TO_DECIDE:
        decision = Decision.ROUTE_TO_HUMAN
    elif prob < AUTO_CLEAR_BELOW:
        decision = Decision.AUTO_CLEAR
    elif prob > FLAG_ABOVE:
        decision = Decision.FLAG
    else:
        decision = Decision.ROUTE_TO_HUMAN

    verdict = Verdict(
        manipulated_probability=prob,
        decision=decision,
        evidence=evidence,
        explanation="",
    )
    verdict.explanation = _explain(verdict, aggregate_confidence)
    return verdict


def _explain(verdict: Verdict, confidence: float) -> str:
    """Adjuster-facing prose. The product is this, not the number."""
    firing = verdict.firing
    header = {
        Decision.AUTO_CLEAR: "No indication of manipulation.",
        Decision.FLAG: "Manipulation indicated.",
        Decision.ROUTE_TO_HUMAN: "Inconclusive -- manual review required.",
    }[verdict.decision]

    if not firing:
        skipped = [e for e in verdict.evidence if not e.applicable]
        tail = f" ({len(skipped)} detectors did not apply)" if skipped else ""
        return f"{header} No detector raised a concern{tail}."

    reasons = "\n".join(f"  - {e.explanation} [{e.detector_id}]" for e in firing[:5])
    return (
        f"{header} P(manipulated) = {verdict.manipulated_probability:.2f} "
        f"at confidence {confidence:.2f}.\nContributing findings:\n{reasons}"
    )
