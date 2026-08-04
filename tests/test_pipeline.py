from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from groundtruth import ClaimContext, Decision, Evidence, ImageCase, analyse
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
    assert {
        "context.policy_consistency",
        "metadata.container_identity",
        "metadata.preview_mismatch",
        "compression.ela",
        "geometric.copy_move",
        "sensor.noiseprint_anomaly",
    } <= ids


def test_quarantined_detectors_stay_out_of_the_default_set():
    """Each of these failed measurement. The set is asserted, not just its size.

    Naming them means re-admitting one is a deliberate edit to this test with a
    number to justify it, rather than something that happens by accident when a
    registration import moves.
    """
    default = {d.id for d in all_detectors()}
    every = {d.id for d in all_detectors(include_experimental=True)}

    assert every - default == {
        "compression.block_grid",             # fires on chance-level phase agreement
        "geometric.sharpness_inconsistency",  # 7.2 sigma, 0% of pixels inside the edit
        "sensor.synthetic_region",            # no operating point separates the classes
        "sensor.noise_inconsistency",         # AUC 0.494 on 224 real photographs
    }


def test_detectors_run_cheapest_first():
    costs = [d.cost for d in all_detectors()]
    assert costs == sorted(costs)


def test_photo_predating_policy_is_flagged(tmp_path, claim):
    """The dispositive case: damage photographed before coverage existed."""
    img = _write_jpeg(tmp_path / "before_policy.jpg", captured="2026:02:18 09:14:00")

    verdict = analyse(ImageCase(image_path=img, context=claim))

    assert verdict.decision is Decision.FLAG
    # The decision is the contract, not the exact probability. Adding detectors
    # that correctly report "clean" moderates the fused number without changing
    # the call, and pinning it too tightly would punish that.
    assert verdict.manipulated_probability > 0.85

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


def test_ela_skips_non_jpeg_container(tmp_path, claim):
    png = tmp_path / "shot.png"
    Image.fromarray(np.zeros((64, 64, 3), "uint8")).save(png)

    ev = next(
        e
        for e in analyse(ImageCase(image_path=png, context=claim)).evidence
        if e.detector_id == "compression.ela"
    )
    assert not ev.applicable
    assert "png" in ev.explanation


def test_container_detector_is_silent_on_ordinary_files(tmp_path, claim):
    """A JPEG named .jpg is the default, not evidence. It must not dilute fusion."""
    img = _write_jpeg(tmp_path / "ordinary.jpg", captured="2026:06:16 14:02:00")

    ev = next(
        e
        for e in analyse(ImageCase(image_path=img, context=claim)).evidence
        if e.detector_id == "metadata.container_identity"
    )
    assert not ev.applicable


def test_container_detector_catches_extension_lie(tmp_path, claim):
    """Bytes are PNG, name says .jpg -- the file was converted or re-saved."""
    liar = tmp_path / "damage.jpg"
    Image.fromarray(np.zeros((64, 64, 3), "uint8")).save(liar, "PNG")

    ev = next(
        e
        for e in analyse(ImageCase(image_path=liar, context=claim)).evidence
        if e.detector_id == "metadata.container_identity"
    )
    assert ev.applicable
    assert ev.score > 0.6
    assert ev.details["actual"] == "png" and ev.details["claimed"] == "jpeg"


def test_no_claim_context_drops_context_detector(tmp_path):
    img = _write_jpeg(tmp_path / "orphan.jpg", captured="2026:06:16 14:02:00")

    ev = next(
        e
        for e in analyse(ImageCase(image_path=img)).evidence
        if e.detector_id == "context.policy_consistency"
    )
    assert not ev.applicable


def test_fusion_with_no_usable_evidence_routes_to_human():
    """Inapplicable detectors must be excluded, not folded in as neutral."""
    from groundtruth.core.types import Tier
    from groundtruth.fusion.weighted import fuse

    verdict = fuse(
        [
            Evidence.not_applicable("a", Tier.METADATA, "nope"),
            Evidence.not_applicable("b", Tier.SENSOR, "also nope"),
        ]
    )

    assert verdict.decision is Decision.ROUTE_TO_HUMAN
    assert verdict.manipulated_probability == pytest.approx(0.5)


def test_zero_confidence_evidence_is_excluded_from_fusion():
    """Confidence 0 means 'I have no opinion' and must not move the score."""
    from groundtruth.core.types import Tier
    from groundtruth.fusion.weighted import fuse

    strong = Evidence("real", Tier.SENSOR, True, score=0.95, confidence=0.9)
    mute = Evidence("mute", Tier.METADATA, True, score=0.05, confidence=0.0)

    assert fuse([strong, mute]).manipulated_probability == pytest.approx(
        fuse([strong]).manipulated_probability
    )
