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

        smoothed = gaussian_filter(gray, sigma=_STRUCTURE_BLUR_SIGMA)

        for i, j, tile in _blockwise(hh, half):
            if i < nrows and j < ncols:
                sigma[i, j] = np.median(np.abs(tile - np.median(tile))) * _MAD_TO_SIGMA

        for i, j, tile in _blockwise(smoothed, BLOCK):
            if i < nrows and j < ncols:
                structure[i, j] = float(tile.std())

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

        # Work in log space: noise scales multiplicatively with gain, so a splice
        # from a higher-ISO source is a constant offset in logs, not in absolutes.
        logs = np.log(sigma[valid])
        centre = float(np.median(logs))
        scale = float(np.median(np.abs(logs - centre))) * _MAD_TO_SIGMA + 1e-6

        z = np.zeros_like(sigma)
        z[valid] = np.abs(np.log(sigma[valid]) - centre) / scale

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
                "median_sigma": round(float(np.exp(centre)), 6),
                "log_sigma_spread": round(scale, 4),
                "max_z": round(float(z.max()), 2),
            },
        )
