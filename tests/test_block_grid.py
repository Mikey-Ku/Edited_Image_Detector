"""Block-grid misalignment -- currently an experimental detector that does not work.

These tests used to assert detection. They passed because the foreign-phase
cluster threshold was 2, which sits below the noise floor: across N confident
windows and 64 possible grid phases, chance supplies about N/64 windows on any
given wrong phase. The benchmark exposed this -- the detector fired on 3 of 4
pristine photographs and on every manipulation class equally.

With the threshold raised to beat chance, it detects nothing. These tests now
record that, so the failure is documented rather than rediscovered. They are the
specification the estimator has to meet before the detector leaves experimental.
"""

from __future__ import annotations

import numpy as np
import pytest
from fixtures import grid_clean, grid_splice
from PIL import Image

from groundtruth import ImageCase
from groundtruth.core.detector import get

BLOCK_GRID = "compression.block_grid"


def _run(path):
    return get(BLOCK_GRID).run(ImageCase(image_path=path))


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def test_detector_is_held_out_of_the_default_pipeline():
    from groundtruth.core.detector import all_detectors

    assert BLOCK_GRID not in {d.id for d in all_detectors()}
    assert BLOCK_GRID in {d.id for d in all_detectors(include_experimental=True)}


def test_grid_splice_is_currently_MISSED(tmp_path):
    """Known failure. Flip this to an assertion of detection when it is fixed."""
    ev = _run(grid_splice(tmp_path / "splice.jpg")[0])

    assert ev.applicable
    assert ev.details["windows_misaligned"] == 0


def test_clean_image_shows_a_single_coherent_grid(tmp_path):
    """Still true, and the one thing it does reliably."""
    ev = _run(grid_clean(tmp_path / "clean.jpg")[0])

    assert ev.applicable
    assert ev.score < 0.4
    assert ev.details["windows_misaligned"] == 0
    assert ev.details["host_agreement"] > 0.9


def test_cluster_threshold_scales_with_chance(tmp_path):
    """The requirement must grow with the frame, because chance does."""
    from groundtruth.detectors.compression.block_grid import _min_cluster

    assert _min_cluster(50) < _min_cluster(650)
    assert _min_cluster(650) >= 20  # ~10 expected by chance, so demand well above it


def test_phase_values_are_in_range(tmp_path):
    ev = _run(grid_splice(tmp_path / "splice.jpg")[0])

    for key in ("host_phase",):
        assert all(0 <= v < 8 for v in ev.details[key]), key


# --------------------------------------------------------------------------
# The envelope -- where this technique genuinely stops working
# --------------------------------------------------------------------------


def test_fails_when_the_composite_is_saved_at_low_quality(tmp_path):
    """Documented limitation, not a regression.

    Saving the composite at q=88 quantises hard enough that the re-save's own
    grid, anchored at the origin, overwrites the donor's. The foreign grid is
    physically gone from the pixels -- no estimator can recover it.
    """
    ev = _run(grid_splice(tmp_path / "low.jpg", final_quality=88)[0])

    assert ev.applicable
    assert ev.details["windows_misaligned"] == 0  # a miss, and an expected one


@pytest.mark.parametrize("final_quality", [92, 96, 99])
def test_currently_misses_across_the_whole_quality_range(tmp_path, final_quality):
    """Was the 'working envelope'. It was the threshold, not the envelope."""
    ev = _run(
        grid_splice(tmp_path / f"q{final_quality}.jpg", final_quality=final_quality)[0]
    )
    assert ev.details["windows_misaligned"] == 0


def test_aligned_paste_is_invisible(tmp_path):
    """The genuine 1-in-64 blind spot: a paste landing on the grid coincides."""
    ev = _run(grid_splice(tmp_path / "aligned.jpg", paste_offset=(0, 0))[0])

    assert ev.applicable
    assert ev.details["windows_misaligned"] == 0


def test_smooth_image_reports_no_readable_grid(tmp_path):
    """Flat content is barely quantised, so there is no grid to phase-align.

    This must return not-applicable rather than guessing -- guessing from
    featureless regions was what made the first version fire on every image.
    """
    smooth = tmp_path / "smooth.jpg"
    _, xx = np.mgrid[0:256, 0:256].astype(np.float32)
    ramp = (0.4 + 0.2 * xx / 256.0) * 255
    Image.fromarray(np.stack([ramp] * 3, -1).astype("uint8")).save(
        smooth, "JPEG", quality=95
    )

    ev = _run(smooth)
    assert not ev.applicable
    assert "readable block grid" in ev.explanation


def test_skips_containers_without_a_block_grid(tmp_path):
    png = tmp_path / "shot.png"
    Image.fromarray(np.zeros((256, 256, 3), "uint8")).save(png)

    ev = _run(png)
    assert not ev.applicable
    assert "8x8 compression grid" in ev.explanation


def test_skips_images_too_small_to_window(tmp_path):
    tiny = tmp_path / "tiny.jpg"
    Image.fromarray(np.zeros((64, 64, 3), "uint8")).save(tiny)

    ev = _run(tiny)
    assert not ev.applicable
    assert "too small" in ev.explanation
