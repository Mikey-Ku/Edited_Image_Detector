"""Recovering the pre-edit image, and container handling across formats."""

from __future__ import annotations

import numpy as np
import pytest
from fixtures import hit_rate, noise_splice, stale_preview
from PIL import Image

from groundtruth import ImageCase, analyse
from groundtruth.core.detector import get
from groundtruth.core.image_io import HEIF_AVAILABLE, Container, inspect, load_rgb, sniff
from groundtruth.recovery import Fidelity, extract_previews, reconstruct

PREVIEW = "metadata.preview_mismatch"


# --------------------------------------------------------------------------
# Preview extraction
# --------------------------------------------------------------------------


def test_finds_embedded_preview(tmp_path):
    path, _ = stale_preview(tmp_path / "edited.jpg")
    previews = extract_previews(path)

    assert previews, "expected to find the embedded EXIF thumbnail"
    assert (previews[0].width, previews[0].height) == (160, 120)


def test_previews_are_largest_first(tmp_path):
    """RAW files carry both a tiny thumbnail and a big preview; the big one wins."""
    path, _ = stale_preview(tmp_path / "edited.jpg")
    sizes = [p.pixels for p in extract_previews(path)]
    assert sizes == sorted(sizes, reverse=True)


def test_no_preview_when_metadata_is_stripped(tmp_path):
    path, _ = noise_splice(tmp_path / "bare.jpg")  # saved without EXIF
    assert reconstruct(path) is None


# --------------------------------------------------------------------------
# Reconstruction -- these are real recovered pixels, not a guess
# --------------------------------------------------------------------------


def test_recovers_the_original_and_localises_the_edit(tmp_path):
    path, mask = stale_preview(tmp_path / "edited.jpg")
    recon = reconstruct(path)

    assert recon is not None
    assert recon.fidelity is Fidelity.RECOVERED
    assert recon.is_evidence

    # The recovered "before" must actually differ from the current image inside
    # the edited region, and agree with it outside.
    before = recon.before.astype(float).mean(axis=2)
    after = recon.after.astype(float).mean(axis=2)
    assert abs(after[mask].mean() - before[mask].mean()) > 30
    assert abs(after[~mask].mean() - before[~mask].mean()) < 8


def test_change_map_lands_on_the_edit(tmp_path):
    path, mask = stale_preview(tmp_path / "edited.jpg")
    recon = reconstruct(path)

    assert recon.difference.shape == mask.shape
    assert hit_rate(recon.difference, mask, threshold=0.3) > 0.9


def test_reported_region_matches_the_known_edit(tmp_path):
    path, _ = stale_preview(tmp_path / "edited.jpg", edit_box=(380, 150, 560, 300))
    recon = reconstruct(path)

    x0, y0, x1, y1 = recon.regions[0]["bbox"]
    # Within a few pixels of the truth; the preview is 1/16th resolution.
    assert abs(x0 - 380) < 12 and abs(y0 - 150) < 12
    assert abs(x1 - 559) < 12 and abs(y1 - 299) < 12


def test_unedited_image_shows_no_change(tmp_path):
    """Negative control: preview and image agree, so nothing should light up."""
    path, _ = stale_preview(tmp_path / "clean.jpg", edit=False)
    recon = reconstruct(path)

    assert recon is not None
    assert recon.changed_fraction < 0.01
    assert not recon.cropped


def test_crop_is_detected_and_declared(tmp_path):
    path, _ = stale_preview(
        tmp_path / "cropped.jpg", edit=False, crop_to=(0, 0, 640, 320)
    )
    recon = reconstruct(path)

    assert recon.cropped
    assert "cropped" in recon.caveat


def test_low_resolution_preview_is_labelled_partial(tmp_path):
    path, _ = stale_preview(tmp_path / "tiny_thumb.jpg", thumb_size=(48, 36))
    recon = reconstruct(path)

    assert recon.fidelity is Fidelity.PARTIAL
    assert recon.is_evidence  # still real pixels, just coarse


def test_fidelity_never_claims_inferred_output_is_evidence():
    """The one invariant that must never break: a guess is not evidence."""
    assert Fidelity.INFERRED.value == "inferred"
    assert Fidelity.RECOVERED is not Fidelity.INFERRED


# --------------------------------------------------------------------------
# The detector built on top
# --------------------------------------------------------------------------


def test_preview_mismatch_flags_the_edit(tmp_path):
    path, _ = stale_preview(tmp_path / "edited.jpg")
    ev = get(PREVIEW).run(ImageCase(image_path=path))

    assert ev.applicable
    assert ev.score > 0.75
    assert ev.heatmap is not None
    assert "pre-edit original" in ev.explanation


def test_preview_mismatch_is_quiet_on_clean_images(tmp_path):
    path, _ = stale_preview(tmp_path / "clean.jpg", edit=False)
    ev = get(PREVIEW).run(ImageCase(image_path=path))

    assert ev.applicable
    assert ev.score < 0.3
    assert ev.heatmap is None


def test_edited_image_is_flagged_end_to_end(tmp_path):
    path, _ = stale_preview(tmp_path / "edited.jpg")
    verdict = analyse(ImageCase(image_path=path))

    assert verdict.manipulated_probability > 0.7
    assert PREVIEW in verdict.localised_by


# --------------------------------------------------------------------------
# Containers -- identified by magic bytes, never by extension
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fmt", "expected"),
    [
        ("JPEG", Container.JPEG),
        ("PNG", Container.PNG),
        ("WEBP", Container.WEBP),
        ("TIFF", Container.TIFF),
        ("BMP", Container.BMP),
        ("GIF", Container.GIF),
    ],
)
def test_container_sniffing(tmp_path, fmt, expected):
    path = tmp_path / f"x.{fmt.lower()}"
    Image.fromarray(np.full((64, 64, 3), 128, "uint8")).save(path, fmt)

    assert sniff(path) is expected
    assert load_rgb(path).shape == (64, 64, 3)


def test_extension_lie_is_detected(tmp_path):
    path = tmp_path / "damage.jpg"
    Image.fromarray(np.zeros((64, 64, 3), "uint8")).save(path, "PNG")

    info = inspect(path)
    assert info.actual is Container.PNG
    assert info.claimed is Container.JPEG
    assert info.extension_mismatch


def test_lossy_classification():
    assert Container.JPEG.lossy and Container.JPEG.block_compressed
    assert Container.HEIF.lossy and Container.HEIF.block_compressed
    assert not Container.PNG.lossy
    assert not Container.TIFF.lossy


@pytest.mark.skipif(not HEIF_AVAILABLE, reason="pillow-heif not installed")
def test_heic_round_trip(tmp_path):
    """iPhone default since 2017 -- in a claims pipeline this is the common case."""
    path = tmp_path / "iphone.heic"
    Image.fromarray(np.full((128, 96, 3), 90, "uint8")).save(path, "HEIF")

    assert sniff(path) is Container.HEIF
    assert inspect(path).decodable
    assert load_rgb(path).shape == (128, 96, 3)


def test_case_uses_magic_bytes_not_the_name(tmp_path):
    path = tmp_path / "damage.jpg"
    Image.fromarray(np.zeros((80, 80, 3), "uint8")).save(path, "PNG")

    case = ImageCase(image_path=path)
    assert case.suffix == ".jpg"
    assert not case.is_jpeg
    assert not case.is_lossy
