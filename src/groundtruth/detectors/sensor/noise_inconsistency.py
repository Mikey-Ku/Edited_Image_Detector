"""Local noise-level inconsistency.

Every sensor imposes noise whose magnitude is set by the physics of the capture --
ISO, exposure, and local brightness. Across a single authentic photograph that
noise level varies smoothly and predictably.

A region spliced in from a different photograph carries *its source's* noise
level. A region that was denoised, blurred, or synthesised carries almost none.
Either way it disagrees with its surroundings, and the disagreement is a physical
inconsistency rather than a statistical coincidence.

Estimation follows Donoho's robust estimator: the standard deviation of the
finest-scale wavelet detail coefficients, measured by median absolute deviation
so that edges and texture -- which are outliers, not noise -- do not inflate it.

    sigma ~= MAD(HH) / 0.6745

MAD matters here. A plain standard deviation over a block containing an edge
reports the edge, not the noise, and every texture boundary in the image becomes
a false positive.

**The noise level is not constant across a photograph, and assuming it is was
this detector's central bug.** Photon arrival is Poisson, so shot noise grows with
the square root of the signal: a brightly lit wall is genuinely noisier than a
shadow in the same untouched frame. Measured against a single global level,
hundreds of legitimate blocks deviate -- and they deviate identically in an edited
image and its own unedited original, which is exactly what the Korus evaluation
found (mean separation -0.065, the original scoring higher on 10 of 14 pairs).

So the baseline is a fitted **noise level function** rather than a number:

    sigma^2(mu) = a * mu + b

fitted robustly across the frame, with `a` carrying shot noise and `b` the
signal-independent read noise. A block is anomalous when it departs from the level
predicted *for its own brightness*.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, label

from ...core.detector import Detector, register
from ...core.types import Evidence, ImageCase, Tier

BLOCK = 32
_MAD_TO_SIGMA = 1.0 / 0.6745

# Structure must be measured on a SMOOTHED image. Measured on raw pixels, noise
# itself registers as structure -- which makes the filter discard exactly the
# high-noise regions this detector exists to find. Blur first so the measurement
# reflects real content (edges, objects) rather than the noise floor.
_STRUCTURE_BLUR_SIGMA = 2.0

# Exclusion is by ADAPTIVE QUANTILE, not an absolute threshold. An absolute cut on
# a normalised gradient behaves completely differently on a smooth scene than on a
# busy one -- on smoothly-varying content nearly every block reads as "textured"
# and coverage collapses. Dropping the busiest N% keeps coverage stable on any
# image, at the honest cost that a manipulation hidden inside genuine fine texture
# may be excluded from measurement.
_STRUCTURE_EXCLUDE_PERCENTILE = 85.0

# Robust z-score beyond which a block's noise level is called anomalous.
_Z_ANOMALOUS = 3.5

# A real manipulation is CONTIGUOUS -- it spans a region, not a scattered block
# here and there. Across ~160 blocks a handful will exceed any fixed z purely from
# estimation noise, so isolated anomalous blocks are discarded and only clusters
# survive. This is a physical constraint on what a splice looks like, which is a
# better false-positive control than simply raising the threshold until the noise
# goes away (that would cost real detections too).
_MIN_CLUSTER_BLOCKS = 2

_MIN_BLOCKS = 16

# Iterations of reweighted fitting for the noise level function. A manipulated
# region is an outlier to the model, so the fit is repeated with outliers
# down-weighted -- otherwise a large splice drags the baseline toward itself and
# hides in its own average.
_NLF_ITERATIONS = 3


def _haar_hh(gray: np.ndarray) -> np.ndarray:
    """Finest-scale diagonal detail coefficients (single-level Haar HH)."""
    h, w = gray.shape
    g = gray[: h - h % 2, : w - w % 2]
    return (g[0::2, 0::2] - g[0::2, 1::2] - g[1::2, 0::2] + g[1::2, 1::2]) * 0.5


def _keep_clusters(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Drop connected components smaller than ``min_size`` (8-connectivity)."""
    if min_size <= 1 or not mask.any():
        return mask
    labelled, count = label(mask, structure=np.ones((3, 3), dtype=int))
    if count == 0:
        return mask
    sizes = np.bincount(labelled.ravel())
    survivors = {i for i in range(1, count + 1) if sizes[i] >= min_size}
    return np.isin(labelled, list(survivors)) if survivors else np.zeros_like(mask)


def _fit_noise_level(mu: np.ndarray, var: np.ndarray) -> tuple[float, float]:
    """Robustly fit sigma^2 = a*mu + b across the frame.

    Iteratively reweighted least squares: fit, measure residuals, down-weight the
    blocks that disagree, refit. Without the reweighting a manipulated region large
    enough to matter would bias the very baseline it is being compared against.
    """
    a, b = 0.0, float(np.median(var))
    w = np.ones_like(mu)
    for _ in range(_NLF_ITERATIONS):
        sw = w.sum()
        if sw < 1e-9:
            break
        mx = float((w * mu).sum() / sw)
        my = float((w * var).sum() / sw)
        cov = float((w * (mu - mx) * (var - my)).sum() / sw)
        varx = float((w * (mu - mx) ** 2).sum() / sw)
        a = cov / varx if varx > 1e-12 else 0.0
        b = my - a * mx
        resid = np.abs(var - (a * mu + b))
        scale = float(np.median(resid)) * 1.4826 + 1e-9
        w = 1.0 / (1.0 + (resid / (3.0 * scale)) ** 2)
    # Noise variance cannot be negative; a downward-sloping fit means the model
    # does not describe this image, so fall back to a flat baseline.
    if a < 0:
        a, b = 0.0, float(np.median(var))
    return a, b


def _blockwise(a: np.ndarray, block: int):
    """Yield (row, col, tile) over non-overlapping blocks."""
    h, w = a.shape
    for r in range(0, h - block + 1, block):
        for c in range(0, w - block + 1, block):
            yield r // block, c // block, a[r : r + block, c : c + block]


@register
class NoiseInconsistencyDetector(Detector):
    """Map local sensor-noise level and flag regions that disagree with the frame."""

    id = "sensor.noise_inconsistency"
    tier = Tier.SENSOR
    localises = True
    cost = 3

    def applies_to(self, case: ImageCase) -> tuple[bool, str]:
        h, w = case.pixels().shape[:2]
        if min(h, w) < BLOCK * 4:
            return False, f"image too small for {BLOCK}px noise blocks ({w}x{h})"
        return True, ""

    def _run(self, case: ImageCase) -> Evidence:
        gray = case.pixels().astype(np.float32).mean(axis=2) / 255.0
        hh = _haar_hh(gray)

        half = BLOCK // 2  # HH plane is half resolution
        nrows = hh.shape[0] // half
        ncols = hh.shape[1] // half
        sigma = np.full((nrows, ncols), np.nan, dtype=np.float32)
        structure = np.zeros((nrows, ncols), dtype=np.float32)
        brightness = np.zeros((nrows, ncols), dtype=np.float32)

        smoothed = gaussian_filter(gray, sigma=_STRUCTURE_BLUR_SIGMA)

        for i, j, tile in _blockwise(hh, half):
            if i < nrows and j < ncols:
                sigma[i, j] = np.median(np.abs(tile - np.median(tile))) * _MAD_TO_SIGMA

        for i, j, tile in _blockwise(smoothed, BLOCK):
            if i < nrows and j < ncols:
                structure[i, j] = float(tile.std())
                brightness[i, j] = float(tile.mean())

        measurable = np.isfinite(sigma) & (sigma > 1e-6)
        if not measurable.any():
            return Evidence.not_applicable(
                self.id, self.tier, "no block yielded a usable noise estimate"
            )

        cutoff = float(np.percentile(structure[measurable], _STRUCTURE_EXCLUDE_PERCENTILE))
        valid = measurable & (structure <= cutoff)

        if int(valid.sum()) < _MIN_BLOCKS:
            return Evidence.not_applicable(
                self.id,
                self.tier,
                f"only {int(valid.sum())} low-structure blocks; noise estimate unreliable",
            )

        # Fit the noise level function, then judge each block against the level
        # predicted for ITS OWN brightness rather than against a global constant.
        a, b = _fit_noise_level(brightness[valid], (sigma[valid] ** 2).astype(np.float64))
        predicted = np.maximum(a * brightness + b, 1e-12)

        # Log ratio of observed to predicted: noise scales multiplicatively with
        # gain, so a region from a different exposure is a constant offset in logs.
        # Halved to express the deviation in SIGMA units rather than variance
        # units. log(var ratio) is twice log(sigma ratio), and every threshold and
        # spread constant downstream was calibrated against sigma.
        ratio = np.zeros_like(sigma)
        ratio[valid] = 0.5 * np.log(
            np.maximum(sigma[valid] ** 2, 1e-12) / predicted[valid]
        )
        centre = float(np.median(ratio[valid]))
        scale = float(np.median(np.abs(ratio[valid] - centre))) * _MAD_TO_SIGMA + 1e-6

        z = np.zeros_like(sigma)
        z[valid] = np.abs(ratio[valid] - centre) / scale

        flagged = valid & (z > _Z_ANOMALOUS)
        anomalous = _keep_clusters(flagged, _MIN_CLUSTER_BLOCKS)
        isolated = int(flagged.sum() - anomalous.sum())
        anomalous_fraction = float(anomalous.sum() / max(int(valid.sum()), 1))

        # The heatmap shows only clustered anomalies, so the overlay agrees with
        # the verdict. Painting isolated blocks the operator is told to ignore
        # would undermine the explanation.
        heat = np.clip(z / (2.0 * _Z_ANOMALOUS), 0.0, 1.0)
        heat[~anomalous] = 0.0
        heat = np.kron(heat, np.ones((BLOCK, BLOCK), dtype=np.float32))

        # Confidence reflects how well-determined the baseline is, not how alarmed
        # we are. Many measurable blocks with a tight noise distribution means the
        # frame has a clear noise signature to deviate FROM; a wide spread means we
        # have no stable baseline and any "anomaly" is unreliable.
        n_valid = int(valid.sum())
        support = min(1.0, n_valid / 128.0)
        homogeneity = float(np.clip(1.0 - scale / 0.60, 0.2, 1.0))
        confidence = float(np.clip(0.80 * support * homogeneity, 0.10, 0.80))

        # Effect size: area of the anomaly relative to ~10% of measurable blocks,
        # tempered by how far past threshold the strongest block went. A single
        # marginal block is a small effect however confident the estimate.
        effect = float(
            min(1.0, anomalous_fraction / 0.10)
            * min(1.0, float(z.max()) / (2.0 * _Z_ANOMALOUS))
        ) if anomalous.any() else 0.0

        if anomalous.any():
            score = float(min(0.92, 0.55 + 3.0 * anomalous_fraction))
            explanation = (
                f"{int(anomalous.sum())} of {int(valid.sum())} measurable blocks have a "
                f"noise level inconsistent with the rest of the frame "
                f"(max deviation {float(z.max()):.1f} sigma"
                + (f"; {isolated} isolated block(s) ignored)" if isolated else ")")
            )
        else:
            score = 0.25
            explanation = (
                f"sensor noise is uniform across {int(valid.sum())} measurable blocks"
            )

        return Evidence(
            detector_id=self.id,
            tier=self.tier,
            applicable=True,
            score=score,
            confidence=confidence,
            effect_size=effect,
            explanation=explanation,
            # No clustered anomaly means there is nothing to point at. Returning a
            # blank map instead of None would let fusion present "we looked and
            # found nothing" as a localisation result.
            heatmap=heat if anomalous.any() else None,
            details={
                "block_px": BLOCK,
                "measurable_blocks": n_valid,
                "anomalous_blocks": int(anomalous.sum()),
                "isolated_discarded": isolated,
                "excluded_as_structured": int(measurable.sum()) - n_valid,
                "shot_noise_coeff": round(float(a), 8),
                "read_noise_floor": round(float(b), 10),
                "residual_spread": round(scale, 4),
                "max_z": round(float(z.max()), 2),
            },
        )
