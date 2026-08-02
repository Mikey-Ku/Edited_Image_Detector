"""Block-grid misalignment, and the envelope where it stops working.

Half of these tests assert detection; the other half pin the *limits*. A detector
whose failure modes are undocumented is one you cannot deploy, because you have no
idea when to believe it.
"""

from __future__ import annotations

import numpy as np
import pytest
from fixtures import grid_clean, grid_splice, hit_rate
from PIL import Image

from groundtruth import ImageCase
from groundtruth.core.detector import get

BLOCK_GRID = "compression.block_grid"


def _run(path):
    return get(BLOCK_GRID).run(ImageCase(image_path=path))


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def test_grid_splice_is_detected(tmp_path):
    ev = _run(grid_splice(tmp_path / "splice.jpg")[0])

    assert ev.applicable
    assert ev.score > 0.7
    assert ev.details["windows_misaligned"] > 0


def test_clean_image_shows_a_single_coherent_grid(tmp_path):
    ev = _run(grid_clean(tmp_path / "clean.jpg")[0])

    assert ev.applicable
    assert ev.score < 0.4
    assert ev.details["windows_misaligned"] == 0
    # Nearly every readable window should agree on one phase for a camera original.
    assert ev.details["host_agreement"] > 0.9


def test_detected_region_is_localised(tmp_path):
    path, mask = grid_splice(tmp_path / "splice.jpg")
    ev = _run(path)

    assert ev.heatmap is not None
    assert hit_rate(ev.heatmap, mask, 0.5) > 0.5


def test_foreign_phase_differs_from_host(tmp_path):
    """The reported region phase must be a phase some window actually had.

    An earlier version took the mode of the x and y phases independently, which
    could report a combination no window held -- and on clean images produced a
    'foreign' phase identical to the host's.
    """
    ev = _run(grid_splice(tmp_path / "splice.jpg")[0])

    assert ev.details["region_phase"] != ev.details["host_phase"]
    assert ev.details["phase_difference"] != [0, 0]


def test_phase_values_are_in_range(tmp_path):
    ev = _run(grid_splice(tmp_path / "splice.jpg")[0])

    for key in ("host_phase", "region_phase", "phase_difference"):
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
def test_detects_across_the_working_envelope(tmp_path, final_quality):
    ev = _run(
        grid_splice(tmp_path / f"q{final_quality}.jpg", final_quality=final_quality)[0]
    )
    assert ev.details["windows_misaligned"] > 0


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
