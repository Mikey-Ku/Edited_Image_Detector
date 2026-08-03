"""Regions that were drawn rather than photographed.

A replaced licence plate, a pasted price, an overlaid document field -- these are
rendered graphics composited onto a photograph. They are visually obvious to a
person and almost invisible to every statistical detector, because they are small,
and because the traces those detectors read (compression phase, demosaicing
periodicity) are destroyed by the downscaling and format conversion such images
usually go through before anyone sees them.

What a rendered region cannot fake is the **absence of sensor noise**.

Every photographed pixel carries shot noise. It is present in flat areas, it scales
with brightness, and it cannot be removed by an editor without leaving the region
smoother than physics allows. A drawn graphic has none: its flat areas are exactly
flat, to the bit.

Neither half of that is sufficient alone, which is why an earlier attempt using
sharpness by itself failed. On a photograph of a damaged car:

    crumpled metal   strong edges  +  normal sensor noise   -> photographed
    rendered plate   strong edges  +  NO sensor noise       -> drawn
    blank sky        no edges      +  normal sensor noise   -> photographed

Sharpness alone flags the crumpled metal, which is real content. The conjunction
does not, because the metal is noisy. **The pair is the signal.**

Noise is measured only in the FLATTEST parts of each block. Measuring it across the
whole block would read the graphic's own hard edges as noise and hide exactly the
emptiness being looked for. The expected level comes from the noise level function
fitted across the frame, so a block is judged against what its own brightness
predicts rather than against a constant.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, label

from ...core.detector import Detector, register
from ...core.types import Evidence, ImageCase, Tier
from .noise_inconsistency import _fit_noise_level, _haar_hh

BLOCK = 24

# A block needs real structure before its silence means anything. Featureless sky
# is quiet because there is nothing in it, not because it was drawn.
_MIN_EDGE_ENERGY = 0.010

# Fraction of each block, taken from its flattest pixels, used to estimate noise.
# The graphic's own edges must be excluded or they masquerade as the noise that is
# supposed to be missing.
_FLAT_QUANTILE = 0.35

# How far below the predicted noise level counts as impossible, in log-sigma units.
# ln(0.45) ~ -0.8: under half the noise the frame's own physics predicts.
_SILENCE_THRESHOLD = 0.8

_MIN_CLUSTER_BLOCKS = 2
_MIN_BLOCKS = 40

# A handful of quiet blocks is expected in any photograph -- a patch of sky inside
# a structured area, an overexposed highlight. Below these floors the finding is
# reported as clean rather than as a weak positive, because a detector whose
# "nothing here" still scores 0.67 poisons every fused verdict it touches.
_MIN_SYNTHETIC_BLOCKS = 6
_MIN_SYNTHETIC_FRACTION = 0.01


def _block_stats(gray: np.ndarray) -> tuple[np.ndarray, ...]:
    """Per block: brightness, edge energy, and noise in its flattest pixels."""
    smooth = gaussian_filter(gray, 1.2)
    gy, gx = np.gradient(smooth)
    grad = np.hypot(gx, gy)

    # Noise carrier: finest-scale detail, at half resolution.
    hh = np.abs(_haar_hh(gray))
    half = BLOCK // 2

    nr = min(gray.shape[0] // BLOCK, hh.shape[0] // half)
    nc = min(gray.shape[1] // BLOCK, hh.shape[1] // half)

    brightness = np.zeros((nr, nc), np.float32)
    edge = np.zeros((nr, nc), np.float32)
    flat_noise = np.zeros((nr, nc), np.float32)

    for i in range(nr):
        for j in range(nc):
            blk = (slice(i * BLOCK, (i + 1) * BLOCK), slice(j * BLOCK, (j + 1) * BLOCK))
            sub = (slice(i * half, (i + 1) * half), slice(j * half, (j + 1) * half))

            brightness[i, j] = float(gray[blk].mean())
            edge[i, j] = float(grad[blk].std())

            g = grad[blk][::2, ::2]
            d = hh[sub]
            n = min(g.size, d.size)
            if n < 16:
                continue
            g, d = g.ravel()[:n], d.ravel()[:n]
            # Keep only the flattest pixels, then take a robust spread of the
            # detail there. On a drawn region this is essentially zero.
            keep = g <= np.quantile(g, _FLAT_QUANTILE)
            vals = d[keep]
            if vals.size >= 8:
                flat_noise[i, j] = float(np.median(vals)) / 0.6745
    return brightness, edge, flat_noise


def _keep_clusters(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1 or not mask.any():
        return mask
    labelled, count = label(mask, structure=np.ones((3, 3), dtype=int))
    sizes = np.bincount(labelled.ravel())
    keep = [i for i in range(1, count + 1) if sizes[i] >= min_size]
    return np.isin(labelled, keep) if keep else np.zeros_like(mask)


@register
class SyntheticRegionDetector(Detector):
    """Find structured regions carrying no sensor noise -- drawn, not photographed."""

    id = "sensor.synthetic_region"
    tier = Tier.SENSOR
    localises = True
    cost = 3

    # EXPERIMENTAL -- the hypothesis is sound, this implementation is not.
    #
    # Swept over four real photographs, varying the silence threshold and the
    # minimum flagged fraction, there is NO operating point:
    #
    #   silence  controls quiet  render_overlay detected
    #     0.8        0/8               4/4        <- flags every clean photo
    #     1.1        4/8               0/4
    #     1.4        8/8               0/4        <- detects nothing
    #
    # It goes straight from firing on everything to firing on nothing, which means
    # the measurement does not separate the classes at all. On a single photograph
    # it looked clean and selective; across four it fires on 3 of 4 pristine
    # images. Adding it took the benchmark to 68% detected but dropped controls
    # from 100% to 38% -- a system that flags 62% of clean photographs is unusable
    # whatever its hit rate.
    #
    # The likely cause is convergence under compression: after a JPEG save the
    # rendered region's flat areas pick up quantisation artefacts while genuinely
    # smooth photographed regions (sky, out-of-focus background) lose theirs, so
    # the two populations overlap. Testing on uncompressed input, and measuring the
    # noise SPECTRUM rather than its magnitude, are the obvious next things to try.
    experimental = True

    def applies_to(self, case: ImageCase) -> tuple[bool, str]:
        h, w = case.pixels().shape[:2]
        if min(h, w) < BLOCK * 8:
            return False, f"image too small for {BLOCK}px blocks ({w}x{h})"
        return True, ""

    def _run(self, case: ImageCase) -> Evidence:
        gray = case.pixels().astype(np.float32).mean(axis=2) / 255.0
        brightness, edge, flat_noise = _block_stats(gray)

        structured = edge > _MIN_EDGE_ENERGY
        measurable = structured & (flat_noise > 0)
        if int(measurable.sum()) < _MIN_BLOCKS:
            return Evidence.not_applicable(
                self.id,
                self.tier,
                f"only {int(measurable.sum())} blocks carry structure to judge; "
                f"image too flat or too small",
            )

        # What noise SHOULD be here, given each block's own brightness.
        a, b = _fit_noise_level(
            brightness[measurable], (flat_noise[measurable] ** 2).astype(np.float64)
        )
        predicted = np.maximum(a * brightness + b, 1e-12)

        silence = np.zeros_like(flat_noise)
        silence[measurable] = 0.5 * np.log(
            np.maximum(flat_noise[measurable] ** 2, 1e-12) / predicted[measurable]
        )
        # Centre on the frame's own median so a uniformly low-noise camera is not
        # mistaken for a frame full of drawn content.
        silence[measurable] -= float(np.median(silence[measurable]))

        # The conjunction: structured AND far quieter than its brightness predicts.
        flagged = measurable & (silence < -_SILENCE_THRESHOLD)
        synthetic = _keep_clusters(flagged, _MIN_CLUSTER_BLOCKS)
        fraction = float(synthetic.sum() / max(int(measurable.sum()), 1))

        if (
            int(synthetic.sum()) < _MIN_SYNTHETIC_BLOCKS
            or fraction < _MIN_SYNTHETIC_FRACTION
        ):
            synthetic = np.zeros_like(synthetic)
            fraction = 0.0

        details: dict[str, object] = {
            "block_px": BLOCK,
            "structured_blocks": int(measurable.sum()),
            "synthetic_blocks": int(synthetic.sum()),
            "isolated_discarded": int(flagged.sum() - synthetic.sum()),
            "deepest_silence": round(float(-silence.min()), 2),
            "shot_noise_coeff": round(float(a), 8),
        }

        support = min(1.0, int(measurable.sum()) / 200.0)
        confidence = float(np.clip(0.80 * support, 0.15, 0.80))

        if not synthetic.any():
            return Evidence(
                detector_id=self.id,
                tier=self.tier,
                applicable=True,
                score=0.26,
                confidence=confidence,
                effect_size=0.0,
                explanation=(
                    f"every structured region carries sensor noise consistent with "
                    f"its brightness ({int(measurable.sum())} blocks)"
                ),
                details=details,
            )

        heat = np.clip(-silence / (2.0 * _SILENCE_THRESHOLD), 0.0, 1.0)
        heat[~synthetic] = 0.0
        heat = np.kron(heat, np.ones((BLOCK, BLOCK), dtype=np.float32))

        return Evidence(
            detector_id=self.id,
            tier=self.tier,
            applicable=True,
            score=float(min(0.94, 0.60 + 8.0 * (fraction - _MIN_SYNTHETIC_FRACTION))),
            confidence=confidence,
            effect_size=float(min(1.0, fraction / 0.03)),
            explanation=(
                f"{int(synthetic.sum())} blocks have hard edges but carry no sensor "
                f"noise -- content that was drawn rather than photographed "
                f"(quietest region {float(-silence.min()):.1f}x below the noise its "
                f"brightness predicts)"
            ),
            heatmap=heat,
            details=details,
        )
