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
