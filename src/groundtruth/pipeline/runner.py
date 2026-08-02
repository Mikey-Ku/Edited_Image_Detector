"""Pipeline orchestration: run every applicable detector, fuse, localise."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from ..core.detector import Detector, all_detectors
from ..core.types import ImageCase, Verdict
from ..fusion.localisation import fuse_heatmaps
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

    try:
        shape = case.pixels().shape[:2]
    except Exception:
        log.exception("could not decode %s for localisation", case.image_path)
    else:
        verdict.heatmap, verdict.localised_by = fuse_heatmaps(evidence, shape)

    verdict.created_at = datetime.now(UTC)
    return verdict
