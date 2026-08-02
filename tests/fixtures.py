"""Synthetic manipulations with known ground truth.

Real forensic datasets are the eventual target, but synthetic splices with
pixel-exact masks are what let us assert that a detector fires *in the right
place* rather than merely producing a number. Every detector claim in the test
suite should be verifiable against a mask.

Each generator returns ``(image_path, mask)`` where ``mask`` is a boolean array,
True inside the manipulated region.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

Box = tuple[int, int, int, int]  # x0, y0, x1, y1


def _scene(shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    """Smooth synthetic content: gradients plus low-frequency structure.

    Deliberately low-texture. Texture is the main confound for noise-based
    estimators, so a clean test keeps it out until we test for it explicitly.
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = (
        0.45
        + 0.18 * (xx / w)
        + 0.12 * np.sin(2 * np.pi * yy / (h / 1.5))
        + 0.08 * np.cos(2 * np.pi * xx / (w / 2.0))
    )
    rng = np.random.default_rng(seed)
    base += rng.normal(0, 0.01, size=(h, w)).astype(np.float32)
    return np.clip(base, 0.05, 0.95)


def _with_noise(gray: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = gray + rng.normal(0, sigma, size=gray.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)


def noise_splice(
    path: Path,
    shape: tuple[int, int] = (384, 512),
    box: Box = (300, 120, 460, 260),
    host_sigma: float = 0.010,
    donor_sigma: float = 0.075,
    quality: int | None = 96,
    seed: int = 7,
) -> tuple[Path, np.ndarray]:
    """A region carrying a different sensor-noise level than its surroundings.

    This is the physical signature of a splice from a photo shot at a different
    ISO -- the most common real-world composite.
    """
    scene = _scene(shape, seed)
    host = _with_noise(scene, host_sigma, seed)
    donor = _with_noise(scene, donor_sigma, seed + 1)

    x0, y0, x1, y1 = box
    composite = host.copy()
    composite[y0:y1, x0:x1] = donor[y0:y1, x0:x1]

    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True

    rgb = (np.stack([composite] * 3, axis=-1) * 255).astype(np.uint8)
    img = Image.fromarray(rgb)
    if quality is None:
        img.save(path)
    else:
        img.save(path, "JPEG", quality=quality)
    return path, mask


def pristine(
    path: Path,
    shape: tuple[int, int] = (384, 512),
    sigma: float = 0.020,
    quality: int | None = 96,
    seed: int = 11,
) -> tuple[Path, np.ndarray]:
    """An unmanipulated image with uniform noise. The negative control."""
    img = Image.fromarray(
        (np.stack([_with_noise(_scene(shape, seed), sigma, seed)] * 3, -1) * 255).astype(
            np.uint8
        )
    )
    if quality is None:
        img.save(path)
    else:
        img.save(path, "JPEG", quality=quality)
    return path, np.zeros(shape, dtype=bool)


def localisation_iou(heat: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> float:
    """IoU between the thresholded heatmap and the ground-truth mask."""
    pred = heat >= threshold
    union = int((pred | mask).sum())
    return float((pred & mask).sum() / union) if union else 0.0


def hit_rate(heat: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> float:
    """Fraction of above-threshold heat that lands inside the true region.

    A more forgiving measure than IoU: a detector that flags a small part of the
    splice and nothing else is useful even though its IoU is poor.
    """
    pred = heat >= threshold
    total = int(pred.sum())
    return float((pred & mask).sum() / total) if total else 0.0


def stale_preview(
    path: Path,
    shape: tuple[int, int] = (480, 640),
    edit_box: Box = (380, 150, 560, 300),
    thumb_size: tuple[int, int] = (160, 120),
    crop_to: Box | None = None,
    edit: bool = True,
    seed: int = 3,
) -> tuple[Path, np.ndarray]:
    """A JPEG whose EXIF thumbnail still shows the UNEDITED original.

    This is the real-world failure that makes recovery possible: an editor rewrites
    the main image and leaves the embedded preview alone, so the container is still
    carrying a photograph of the original.

    Set ``edit=False`` for the negative control -- preview and image agree.
    """
    import io

    import piexif

    scene = _scene(shape, seed)
    original = (np.stack([_with_noise(scene, 0.02, seed)] * 3, -1) * 255).astype("uint8")

    # The thumbnail is made from the ORIGINAL, before any edit.
    tb = io.BytesIO()
    Image.fromarray(original).resize(thumb_size, Image.Resampling.LANCZOS).save(
        tb, "JPEG", quality=70
    )

    current = original.copy()
    mask = np.zeros(shape, dtype=bool)
    if edit:
        x0, y0, x1, y1 = edit_box
        current[y0:y1, x0:x1] = np.clip(
            current[y0:y1, x0:x1].astype(int) + 70, 0, 255
        ).astype("uint8")
        mask[y0:y1, x0:x1] = True
    if crop_to is not None:
        x0, y0, x1, y1 = crop_to
        current = current[y0:y1, x0:x1]
        mask = mask[y0:y1, x0:x1]

    mb = io.BytesIO()
    Image.fromarray(current).save(mb, "JPEG", quality=92)

    exif = {
        "0th": {piexif.ImageIFD.Software: b"Adobe Photoshop 26.0"},
        "Exif": {},
        "GPS": {},
        "1st": {},
        "thumbnail": tb.getvalue(),
    }
    out = io.BytesIO()
    piexif.insert(piexif.dump(exif), mb.getvalue(), out)
    path.write_bytes(out.getvalue())
    return path, mask


def _textured_scene(shape: tuple[int, int], seed: int) -> np.ndarray:
    """Content with enough detail for block artefacts to be readable.

    A smooth gradient carries no blocking signature -- JPEG barely has to quantise
    anything -- so grid-phase tests need real texture to measure against.
    """
    h, w = shape
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = 0.45 + 0.15 * np.sin(xx / 23.0) * np.cos(yy / 31.0)
    img += 0.10 * np.sin((xx + yy) / 17.0)
    # Sensor-realistic noise only. Heavy white noise destroys the block grid
    # outright -- JPEG spends its bits encoding the noise instead of quantising
    # smooth blocks -- so a noisy fixture tests nothing.
    img += 0.015 * rng.normal(0, 1, (h, w)).astype(np.float32)
    coarse = rng.normal(0, 1, (h // 24 + 1, w // 24 + 1)).astype(np.float32)
    img += 0.18 * np.asarray(
        Image.fromarray(((coarse - coarse.min()) / np.ptp(coarse) * 255).astype("uint8"))
        .resize((w, h), Image.Resampling.BICUBIC),
        dtype=np.float32,
    ) / 255.0
    return np.clip(img, 0.02, 0.98)


def grid_splice(
    path: Path,
    shape: tuple[int, int] = (384, 512),
    box: Box = (200, 120, 360, 264),
    paste_offset: tuple[int, int] = (3, 5),
    donor_quality: int = 60,
    final_quality: int = 96,
    seed: int = 21,
) -> tuple[Path, np.ndarray]:
    """A region carrying a JPEG block grid out of phase with its host.

    The donor is compressed first, which bakes an 8x8 grid into its pixels. A crop
    of it is then pasted at an offset that is not a multiple of 8, so the grid it
    carries no longer lines up with the frame it lands in. Re-saving writes a
    second grid on top without erasing the first.

    ``paste_offset`` must not be (0, 0) mod 8 or the grids coincide and there is
    nothing to detect -- which is the real 1-in-64 blind spot, not a test artefact.

    Defaults sit inside the detector's measured operating envelope: the donor is
    compressed hard enough to leave a strong grid, and the composite is saved at a
    high enough quality that the new grid does not overwrite it. Outside that
    envelope the technique genuinely fails -- see ``test_block_grid.py``.
    """
    host_arr = (np.stack([_textured_scene(shape, seed)] * 3, -1) * 255).astype("uint8")
    donor_arr = (np.stack([_textured_scene(shape, seed + 101)] * 3, -1) * 255).astype(
        "uint8"
    )

    # Compress the donor on its own so it acquires a grid anchored at ITS origin.
    donor_jpeg = path.parent / f".donor_{path.stem}.jpg"
    Image.fromarray(donor_arr).save(donor_jpeg, "JPEG", quality=donor_quality)
    donor = np.asarray(Image.open(donor_jpeg).convert("RGB"))

    x0, y0, x1, y1 = box
    dx, dy = paste_offset
    bh, bw = y1 - y0, x1 - x0

    composite = host_arr.copy()
    composite[y0:y1, x0:x1] = donor[y0 + dy : y0 + dy + bh, x0 + dx : x0 + dx + bw]

    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True

    Image.fromarray(composite).save(path, "JPEG", quality=final_quality)
    donor_jpeg.unlink(missing_ok=True)
    return path, mask


def grid_clean(
    path: Path,
    shape: tuple[int, int] = (384, 512),
    quality: int = 90,
    seed: int = 21,
) -> tuple[Path, np.ndarray]:
    """Textured, singly-compressed, unmanipulated. The negative control."""
    arr = (np.stack([_textured_scene(shape, seed)] * 3, -1) * 255).astype("uint8")
    Image.fromarray(arr).save(path, "JPEG", quality=quality)
    return path, np.zeros(shape, dtype=bool)
