"""Copy-move detection, and the specificity that makes it useful.

Detection alone is cheap -- a detector that fires on everything scores perfectly.
Half of these tests assert that it stays silent on cases it should not see.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from groundtruth import ImageCase
from groundtruth.benchmark import make
from groundtruth.core.detector import get

COPY_MOVE = "geometric.copy_move"
KORUS = Path(__file__).resolve().parents[1] / "data/interim/korus/data-images"

pytestmark = pytest.mark.skipif(
    not KORUS.is_dir(), reason="needs the Korus photographs (scripts/salvage_zip.py)"
)


@pytest.fixture(scope="module")
def base() -> Path:
    return min((KORUS / "Nikon_D7000" / "pristine").glob("*.TIF"))


@pytest.fixture
def workdir(tmp_path_factory):
    d = tmp_path_factory.mktemp("copymove")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _run(base: Path, workdir: Path, op: str, size: float = 0.08):
    fx = make(base, workdir, op, size, "jpeg95", seed=3)
    return get(COPY_MOVE).run(ImageCase(image_path=fx.path)), fx


def test_finds_a_duplicated_region(base, workdir):
    ev, _ = _run(base, workdir, "duplicate")

    assert ev.applicable
    assert ev.score > 0.8
    assert ev.details["supporting_pairs"] >= 8
    assert ev.effect_size > 0.5


def test_finds_content_cloned_over_something(base, workdir):
    """Removal by cloning -- invisible to every statistical detector here."""
    ev, _ = _run(base, workdir, "clone_out")

    assert ev.applicable
    assert ev.score > 0.8


def test_localises_to_the_duplication(base, workdir):
    ev, fx = _run(base, workdir, "duplicate")

    assert ev.heatmap is not None
    pred = ev.heatmap >= 0.5
    # Half the flagged pixels sit on the SOURCE region, which is correct behaviour
    # and not in the mask -- the mask marks only where content was pasted.
    assert (pred & fx.mask).sum() > 0
    assert 0.25 < float((pred & fx.mask).sum() / pred.sum()) < 0.85


def test_silent_on_an_untouched_photograph(base, workdir):
    ev, _ = _run(base, workdir, "pristine", size=0.0)

    assert ev.applicable
    assert ev.score < 0.4
    assert ev.details["consensus_groups"] == 0


def test_silent_on_a_legitimate_exposure_change(base, workdir):
    """Brightening a dark claim photo is not fraud."""
    ev, _ = _run(base, workdir, "global_tone", size=0.0)

    assert ev.applicable
    assert ev.score < 0.4


def test_silent_on_a_rendered_overlay(base, workdir):
    """Nothing was duplicated, so this detector must not claim it was."""
    ev, _ = _run(base, workdir, "render_overlay")

    assert ev.score < 0.4


def test_reports_a_single_consistent_displacement(base, workdir):
    ev, _ = _run(base, workdir, "duplicate")

    dx, dy = ev.details["displacement_px"]
    assert abs(dx) + abs(dy) > 0


def test_skips_images_too_small_to_match(tmp_path):
    import numpy as np
    from PIL import Image

    tiny = tmp_path / "tiny.jpg"
    Image.fromarray(np.zeros((64, 64, 3), "uint8")).save(tiny)

    ev = get(COPY_MOVE).run(ImageCase(image_path=tiny))
    assert not ev.applicable
