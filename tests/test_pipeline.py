from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from groundtruth import ClaimContext, Decision, ImageCase, analyse
from groundtruth.core.detector import all_detectors


def _write_jpeg(path: Path, captured: str | None = None, size=(320, 240)) -> Path:
    """A noise JPEG, optionally carrying an EXIF DateTimeOriginal."""
    rng = np.random.default_rng(0)
    arr = rng.normal(128, 25, (size[1], size[0], 3)).clip(0, 255).astype("uint8")
    img = Image.fromarray(arr)
    if captured is None:
        img.save(path, quality=88)
    else:
        exif = Image.Exif()
        exif[0x8769] = {36867: captured}  # Exif IFD -> DateTimeOriginal
        img.save(path, quality=88, exif=exif)
    return path


@pytest.fixture
def claim() -> ClaimContext:
    return ClaimContext(
        claim_id="CLM-1001",
        claimant_id="C-77",
        policy_inception=date(2026, 3, 1),
        loss_date=date(2026, 6, 15),
    )


def test_detectors_are_registered():
    ids = {d.id for d in all_detectors()}
    assert "context.policy_consistency" in ids
    assert "metadata.thumbnail_mismatch" in ids


def test_detectors_run_cheapest_first():
    costs = [d.cost for d in all_detectors()]
    assert costs == sorted(costs)


def test_photo_predating_policy_is_flagged(tmp_path, claim):
    """The dispositive case: damage photographed before coverage existed."""
    img = _write_jpeg(tmp_path / "before_policy.jpg", captured="2026:02:18 09:14:00")

    verdict = analyse(ImageCase(image_path=img, context=claim))

    assert verdict.decision is Decision.FLAG
    assert verdict.manipulated_probability > 0.9

    ev = next(e for e in verdict.evidence if e.detector_id == "context.policy_consistency")
    assert ev.applicable
    assert ev.details["days_before_inception"] == 11
    assert "predates coverage" in ev.explanation


def test_consistent_photo_is_not_flagged(tmp_path, claim):
    img = _write_jpeg(tmp_path / "consistent.jpg", captured="2026:06:16 14:02:00")

    verdict = analyse(ImageCase(image_path=img, context=claim))

    assert verdict.decision is not Decision.FLAG
    ev = next(e for e in verdict.evidence if e.detector_id == "context.policy_consistency")
    assert ev.score < 0.5


def test_photo_before_loss_date_is_suspicious(tmp_path, claim):
    img = _write_jpeg(tmp_path / "before_loss.jpg", captured="2026:05:01 08:00:00")

    ev = next(
        e
        for e in analyse(ImageCase(image_path=img, context=claim)).evidence
        if e.detector_id == "context.policy_consistency"
    )
    assert ev.score > 0.8
    assert ev.details["days_before_loss"] == 45


def test_thumbnail_detector_skips_non_jpeg(tmp_path, claim):
    png = tmp_path / "shot.png"
    Image.fromarray(np.zeros((64, 64, 3), "uint8")).save(png)

    ev = next(
        e
        for e in analyse(ImageCase(image_path=png, context=claim)).evidence
        if e.detector_id == "metadata.thumbnail_mismatch"
    )
    assert not ev.applicable
    assert "JPEG" in ev.explanation


def test_no_claim_context_drops_context_detector(tmp_path):
    img = _write_jpeg(tmp_path / "orphan.jpg", captured="2026:06:16 14:02:00")

    ev = next(
        e
        for e in analyse(ImageCase(image_path=img)).evidence
        if e.detector_id == "context.policy_consistency"
    )
    assert not ev.applicable


def test_verdict_with_no_usable_evidence_routes_to_human(tmp_path):
    """Inapplicable detectors must be excluded, not folded in as neutral."""
    png = tmp_path / "bare.png"
    Image.fromarray(np.zeros((64, 64, 3), "uint8")).save(png)

    verdict = analyse(ImageCase(image_path=png))

    assert verdict.decision is Decision.ROUTE_TO_HUMAN
    assert verdict.manipulated_probability == pytest.approx(0.5)
    assert not any(e.applicable for e in verdict.evidence)
