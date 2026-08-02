"""Pipeline orchestration: run every applicable detector, then fuse."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..core.detector import Detector, all_detectors
from ..core.types import ImageCase, Verdict
from ..fusion.weighted import fuse

log = logging.getLogger(__name__)


def analyse(case: ImageCase, detectors: list[Detector] | None = None) -> Verdict:
    """Run the full pipeline over one image case.

    Detectors run cheapest-first. Every one is asked whether it applies before it
    runs, so an image lacking EXIF, or a claim lacking a policy date, simply drops
    the detectors that cannot speak to it rather than contributing noise.
    """
    detectors = detectors if detectors is not None else all_detectors()
    evidence = []
    for det in detectors:
        ev = det.run(case)
        log.debug(
            "%s applicable=%s score=%.2f conf=%.2f",
            det.id, ev.applicable, ev.score, ev.confidence,
        )
        evidence.append(ev)

    verdict = fuse(evidence)
    verdict.created_at = datetime.now(timezone.utc)
    return verdict
