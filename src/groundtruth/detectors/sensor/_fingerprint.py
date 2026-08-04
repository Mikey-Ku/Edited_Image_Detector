"""Shared block machinery for the two Noiseprint readouts.

Both fingerprint detectors run the same network over the same frame and differ only
in what they reduce each block to. That difference turned out to matter more than
anything else about them, and measurement says to keep both:

    matched FPR    energy-only hits   period2-only hits   exact binomial
    1.8%                  7                   0              p = 0.016
    5.0%                  2                   3              p = 1.000
    10.0%                 5                  11              p = 0.210

At the high-precision corner the energy readout catches seven forgeries the
structural one misses and *none* the other way round -- a nested win, not noise.
Above 5% the structural readout pulls ahead. They are complementary, so replacing
either with the other loses a measured capability.

The residual is cached here so running both costs one inference pass, not two.

Nonprofit use only -- see ``groundtruth.learned.noiseprint`` for the licence.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

# A block is unusable when this share of it sits at either rail. Clipped pixels
# carry no sensor noise by definition -- and including them made a legitimate
# exposure lift look like tampering, because brightening a photo blows out its
# highlights and every blown region then reads as a foreign fingerprint.
CLIP_LOW, CLIP_HIGH = 0.02, 0.98
MAX_CLIPPED_FRACTION = 0.20

MIN_BLOCKS = 32

# Frame-level period-2 structure below which there is no camera fingerprint here
# at all, and a detector must abstain rather than report.
#
# The saturated-block rule raised to the whole frame: an image that was never
# demosaiced carries no Bayer periodicity, so "this region disagrees with the
# frame" is a comparison against nothing. The two populations are not close --
# real photographs sit at 12.7 to 26.7, synthetic fixtures at 1.2 to 1.7. This is
# a categorical difference, not a tuned threshold.
#
# It also earns its place in production: screenshots, game captures, renders and
# fully generated images get an honest "I cannot speak to this" instead of a
# fabricated reading off a residual with no fingerprint in it.
MIN_DEMOSAIC_STRUCTURE = 6.0


def blocks(a: np.ndarray, block: int) -> np.ndarray:
    """Reshape to (rows, cols, block, block), preserving pixel order within blocks."""
    nr, nc = a.shape[0] // block, a.shape[1] // block
    return (
        a[: nr * block, : nc * block]
        .reshape(nr, block, nc, block)
        .transpose(0, 2, 1, 3)
    )


def energy(residual: np.ndarray, block: int) -> np.ndarray:
    """Per-block residual magnitude.

    Blind to spatial arrangement by construction -- shuffling a block's pixels
    leaves this unchanged. That is a real limitation and also, at the
    high-precision corner, not a fatal one.
    """
    return blocks(residual, block).std(axis=(2, 3))


def period2(residual: np.ndarray, block: int) -> np.ndarray:
    """Per-block energy at the 2x2 spatial frequency, normalised by total energy.

    A sensor captures one colour per pixel and interpolates the rest across a Bayer
    grid, so an authentic residual carries structure with period exactly 2. A region
    that was rendered, inpainted, or resampled was never demosaiced and loses it.

    Computed as the projection onto the three sign patterns that alternate every
    pixel -- along rows, along columns, and along both -- which is the discrete
    Fourier component at the Nyquist frequency in each direction. Normalising by the
    block's own energy makes this a measure of shape rather than amount, which is
    what distinguishes a fingerprint from mere texture.
    """
    r = blocks(residual, block)
    centred = r - r.mean(axis=(2, 3), keepdims=True)
    total = np.sqrt((centred**2).sum(axis=(2, 3))) + 1e-9

    alt = (-1.0) ** np.arange(block)
    row = np.abs((centred * alt[None, None, :, None]).sum(axis=(2, 3))) / total
    col = np.abs((centred * alt[None, None, None, :]).sum(axis=(2, 3))) / total
    diag = np.abs(
        (centred * (alt[:, None] * alt[None, :])[None, None]).sum(axis=(2, 3))
    ) / total
    return np.sqrt(row**2 + col**2 + diag**2)


def clipped_share(gray: np.ndarray, block: int) -> np.ndarray:
    g = blocks(gray, block)
    return ((g <= CLIP_LOW) | (g >= CLIP_HIGH)).mean(axis=(2, 3))


def robust_z(stat: np.ndarray, usable: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Deviation from the frame's own robust centre, in MAD units.

    Returns (z, centre, scale). Comparison is against the frame itself rather than
    any absolute level, because the fingerprint differs by camera model and there is
    no population baseline to compare a single photograph against.
    """
    values = stat[usable]
    centre = float(np.median(values))
    scale = float(np.median(np.abs(values - centre))) * 1.4826 + 1e-9
    z = np.zeros_like(stat)
    z[usable] = np.abs(stat[usable] - centre) / scale
    return z, centre, scale


@lru_cache(maxsize=4)
def _residual_cached(path: str, qf: int, mtime: float) -> np.ndarray:
    from PIL import Image

    from ...learned.noiseprint import extract

    with Image.open(path) as im:
        gray = np.asarray(im.convert("RGB"), dtype=np.float32).mean(axis=2) / 255.0
    out = extract(gray, qf)
    out.flags.writeable = False  # shared across detectors; must not be mutated
    return out


def residual_for(path: Path, qf: int) -> np.ndarray:
    """Noiseprint residual for a file, computed once and shared between detectors.

    Keyed on modification time as well as path, so an edited file is not served a
    stale residual from an earlier run in the same process.
    """
    return _residual_cached(str(path), qf, Path(path).stat().st_mtime)
