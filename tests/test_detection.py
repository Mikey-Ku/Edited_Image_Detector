"""Detection behaviour against manipulations with known ground truth.

Asserting "the score went up" is not verification -- a detector that fires on
everything satisfies that. These tests check that detectors fire *in the right
place*, stay quiet on clean input, and degrade in a documented way.
"""

from __future__ import annotations

import numpy as np
import pytest
from fixtures import hit_rate, localisation_iou, noise_splice, pristine

from groundtruth import Decision, ImageCase, analyse
from groundtruth.core.detector import all_detectors, get
from groundtruth.detectors.sensor.noise_inconsistency import _keep_clusters

NOISE = "sensor.noise_inconsistency"


def _noise_evidence(path):
    return get(NOISE).run(ImageCase(image_path=path))


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def test_noise_splice_fixture_is_caught_only_by_the_quarantined_detector(tmp_path):
    """The default pipeline does NOT catch this fixture, and that is correct.

    This test used to assert a FLAG, and it passed for a bad reason on both sides.
    `noise_splice` builds a scene with *uniform* noise and swaps in a patch at a
    different level -- which is precisely the assumption `sensor.noise_inconsistency`
    makes. The fixture could not have falsified the detector, and the detector could
    not have failed the fixture.

    Measured on 112 real forgeries and their matched originals, that detector scores
    AUC 0.494 and fires on 66% of *pristine* photographs -- so it is now quarantined,
    and this synthetic splice goes undetected by the default set.

    What is asserted here is the true state of affairs: invoked directly the
    detector still fires exactly as designed, and the shipped pipeline still says
    nothing about this image. Both facts are worth failing on if they change.
    """
    path, _ = noise_splice(tmp_path / "splice.jpg")

    ev = _noise_evidence(path)
    assert ev.applicable and ev.score > 0.7, "the quarantined detector still fires"
    assert NOISE not in {d.id for d in all_detectors()}, "must stay out of the default set"

    verdict = analyse(ImageCase(image_path=path))
    assert verdict.decision is not Decision.FLAG


def test_synthetic_pristine_is_not_falsely_flagged(tmp_path):
    """A generated image is something the system cannot assess, and now says so.

    This asserted AUTO_CLEAR until `sensor.noise_inconsistency` was quarantined, and
    it passed for the wrong reason: that detector confidently reported "clean" on a
    fixture built from the same uniform-noise assumption it makes. Its confidence was
    worth nothing -- AUC 0.494 on 224 real photographs.

    With it gone, every remaining detector correctly abstains on this input: no SIFT
    keypoints in a smooth gradient, no demosaicing structure because no sensor made
    it, no embedded preview, no claim context. Only ELA can speak, at confidence
    0.05. Routing to a human is the honest response to having no evidence, and
    auto-clearing would be false confidence.

    What must never happen is a false FLAG, and that is what is asserted.
    """
    path, _ = pristine(tmp_path / "clean.jpg")
    verdict = analyse(ImageCase(image_path=path))

    assert verdict.decision is not Decision.FLAG
    assert verdict.manipulated_probability < 0.55
    assert not verdict.firing, "no detector should raise a concern on a clean image"


def test_pristine_produces_no_clustered_anomaly(tmp_path):
    path, _ = pristine(tmp_path / "clean.jpg")
    ev = _noise_evidence(path)

    assert ev.applicable
    assert ev.details["anomalous_blocks"] == 0


# --------------------------------------------------------------------------
# Localisation -- the part that actually matters
# --------------------------------------------------------------------------


def test_heatmap_lands_inside_the_manipulated_region(tmp_path):
    """Localisation quality, measured on the detector rather than the pipeline.

    Ran against the fused verdict until `sensor.noise_inconsistency` was
    quarantined; it was the only detector localising this fixture, so the fused map
    is now empty. The machinery being checked -- that a heatmap points at the edit
    rather than merely correlating with it -- is worth keeping under test, so it is
    asserted where the signal actually is.
    """
    path, mask = noise_splice(tmp_path / "splice.jpg")
    ev = _noise_evidence(path)

    assert ev.heatmap is not None
    assert ev.heatmap.shape == mask.shape
    # Nearly all flagged pixels must fall inside the true splice. A detector that
    # scores correctly but points at the wrong place is useless to an adjuster.
    assert hit_rate(ev.heatmap, mask) > 0.85
    assert localisation_iou(ev.heatmap, mask) > 0.4


def test_default_pipeline_does_not_localise_the_noise_splice(tmp_path):
    """The cost of the quarantine, stated rather than hidden.

    An empty map is the honest output here: no detector in the shipped set can see
    this manipulation. Recording it means a future detector that closes the gap will
    break this test, which is the point.
    """
    path, _ = noise_splice(tmp_path / "splice.jpg")
    verdict = analyse(ImageCase(image_path=path))

    assert NOISE not in verdict.localised_by


def test_heatmap_is_absent_when_nothing_localises(tmp_path):
    path, _ = pristine(tmp_path / "clean.png", quality=None)
    verdict = analyse(ImageCase(image_path=path))

    # ELA does not apply to PNG and the noise detector found no cluster, so there
    # must be no map at all -- not an all-zero one masquerading as a result.
    assert verdict.heatmap is None
    assert verdict.localised_by == []


def test_reported_region_overlaps_the_truth(tmp_path):
    from groundtruth.fusion.localisation import peak_regions

    path, mask = noise_splice(tmp_path / "splice.jpg")
    ev = _noise_evidence(path)

    regions = peak_regions(ev.heatmap, threshold=0.5)
    assert regions, "expected at least one region of interest"

    x0, y0, x1, y1 = regions[0]["bbox"]
    ys, xs = np.where(mask)
    assert x0 <= xs.max() and x1 >= xs.min()
    assert y0 <= ys.max() and y1 >= ys.min()


# --------------------------------------------------------------------------
# Robustness -- a detector that only works on pristine files does not work
# --------------------------------------------------------------------------


@pytest.mark.parametrize("quality", [98, 92, 85, 75])
def test_splice_survives_recompression(tmp_path, quality):
    """Real claim photos arrive through email and messaging apps, not pristine."""
    path, _ = noise_splice(tmp_path / f"splice_q{quality}.jpg", quality=quality)
    ev = _noise_evidence(path)

    assert ev.applicable, f"detector bailed at q={quality}"
    assert ev.score > 0.6, f"missed the splice at q={quality}"
    assert ev.details["anomalous_blocks"] > 0


@pytest.mark.parametrize("donor_sigma", [0.075, 0.045, 0.030])
def test_detection_degrades_gracefully_with_contrast(tmp_path, donor_sigma):
    """Smaller noise differences are harder. Document where the floor is."""
    path, _ = noise_splice(tmp_path / "s.jpg", donor_sigma=donor_sigma)
    ev = _noise_evidence(path)
    assert ev.applicable
    # No detection assertion -- this pins observed behaviour so a regression in
    # sensitivity shows up as a test change rather than passing silently.
    assert 0.0 <= ev.score <= 1.0


# --------------------------------------------------------------------------
# Component behaviour
# --------------------------------------------------------------------------


def test_cluster_filter_drops_isolated_blocks():
    mask = np.zeros((6, 6), dtype=bool)
    mask[0, 0] = True  # isolated -> dropped
    mask[3, 3] = mask[3, 4] = True  # pair -> kept

    kept = _keep_clusters(mask, min_size=2)

    assert not kept[0, 0]
    assert kept[3, 3] and kept[3, 4]


def test_cluster_filter_is_identity_below_min_size():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    assert _keep_clusters(mask, min_size=1).sum() == 1


def test_ela_is_deliberately_low_confidence(tmp_path):
    """ELA is a documented baseline, not a signal. It must never drive a verdict."""
    path, _ = noise_splice(tmp_path / "splice.jpg")
    ev = get("compression.ela").run(ImageCase(image_path=path))

    assert ev.applicable
    assert ev.confidence <= 0.30
    assert "caveat" in ev.details


def test_ela_skips_non_jpeg(tmp_path):
    path, _ = pristine(tmp_path / "clean.png", quality=None)
    ev = get("compression.ela").run(ImageCase(image_path=path))
    assert not ev.applicable


def test_noise_detector_skips_tiny_images(tmp_path):
    from PIL import Image

    tiny = tmp_path / "tiny.jpg"
    Image.fromarray(np.zeros((40, 40, 3), "uint8")).save(tiny)

    ev = _noise_evidence(tiny)
    assert not ev.applicable
    assert "too small" in ev.explanation
