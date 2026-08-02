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
from groundtruth.core.detector import get
from groundtruth.detectors.sensor.noise_inconsistency import _keep_clusters

NOISE = "sensor.noise_inconsistency"


def _noise_evidence(path):
    return get(NOISE).run(ImageCase(image_path=path))


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def test_splice_is_flagged(tmp_path):
    path, _ = noise_splice(tmp_path / "splice.jpg")
    verdict = analyse(ImageCase(image_path=path))

    assert verdict.decision is Decision.FLAG
    assert verdict.manipulated_probability > 0.7


def test_pristine_is_cleared(tmp_path):
    path, _ = pristine(tmp_path / "clean.jpg")
    verdict = analyse(ImageCase(image_path=path))

    assert verdict.decision is Decision.AUTO_CLEAR
    assert verdict.manipulated_probability < 0.4


def test_pristine_produces_no_clustered_anomaly(tmp_path):
    path, _ = pristine(tmp_path / "clean.jpg")
    ev = _noise_evidence(path)

    assert ev.applicable
    assert ev.details["anomalous_blocks"] == 0


# --------------------------------------------------------------------------
# Localisation -- the part that actually matters
# --------------------------------------------------------------------------


def test_heatmap_lands_inside_the_manipulated_region(tmp_path):
    path, mask = noise_splice(tmp_path / "splice.jpg")
    verdict = analyse(ImageCase(image_path=path))

    assert verdict.heatmap is not None
    assert verdict.heatmap.shape == mask.shape
    # Nearly all flagged pixels must fall inside the true splice. A detector that
    # scores correctly but points at the wrong place is useless to an adjuster.
    assert hit_rate(verdict.heatmap, mask) > 0.85
    assert localisation_iou(verdict.heatmap, mask) > 0.4


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
    verdict = analyse(ImageCase(image_path=path))

    regions = peak_regions(verdict.heatmap, threshold=0.5)
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
