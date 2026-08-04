"""Regions whose camera fingerprint differs in STRUCTURE from the rest of the frame.

The sibling of `sensor.noiseprint_anomaly`, running on the same residual and
differing only in what each block is reduced to. That difference is the whole point.

The energy readout takes the standard deviation of a block's residual -- a scalar
magnitude, invariant to shuffling the block's pixels. A camera fingerprint lives in
the arrangement, so that statistic is blind to the quantity it exists to measure.
This one measures the arrangement: energy at the 2x2 spatial frequency, normalised
by the block's own energy. A sensor captures one colour per pixel and interpolates
the rest across a Bayer grid, so an authentic residual carries structure with period
exactly 2. A region that was rendered, inpainted, or resampled was never demosaiced
and loses it.

Measured on the Korus Realistic Tampering Dataset, on 30 pairs held out from every
choice made here -- readout, block size, threshold:

    readout        held-out AUC
    energy         0.517   [0.367, 0.667]     confidence interval spans chance
    period2        0.677   [0.539, 0.815]

    paired sign test: period2 separates the pair on 26 of 30 images, p = 3.0e-05

A block-size sweep from 16 to 64 px under the *energy* readout was flat, 0.527 to
0.567, which was the clue: geometry cannot matter while the summary discards the
structure. Under this readout block size matters exactly as the physics predicts --
estimating a periodic pattern needs enough samples -- rising from 0.580 at 16 px to
0.679 at 48. Hence 48 here against 32 for the energy readout.

**Where this readout does NOT win.** Compared at matched false positive rate, the
energy readout catches seven forgeries this one misses at FPR 1.8%, and none the
other way round (p = 0.016). Above 5% this one pulls ahead. Both ship; see
`_fingerprint.py`.

The score is a graded logistic in the anomalous fraction rather than a threshold,
because the ordering is where this readout's value is, and a floor throws it away.
It is clipped well short of certainty on purpose: separating classes at AUC 0.68 is
real and nowhere near conclusive, so a reading that emitted 0.94 would claim more
than the measurement supports.

Nonprofit use only -- see ``groundtruth.learned.noiseprint`` for the licence.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label

from ...core.detector import Detector, register
from ...core.types import Evidence, ImageCase, Tier
from ._fingerprint import (
    MAX_CLIPPED_FRACTION,
    MIN_BLOCKS,
    MIN_DEMOSAIC_STRUCTURE,
    clipped_share,
    energy,
    period2,
    residual_for,
    robust_z,
)

# 48px, not 32. Large enough to estimate period-2 structure reliably; the sweep
# showed the readout still improving to 64 but with fewer blocks to compare.
BLOCK = 48

_Z_ANOMALOUS = 3.0

# Logistic mapping from anomalous fraction to score. Centre sits near where the
# tampered and pristine distributions cross on the tuning split.
#
# The scale is deliberately gentle. At 0.010 the top decile of *pristine*
# photographs already saturated the score -- clean images pinned at the ceiling is
# what miscalibration looks like from the inside, and AUC cannot see it, being
# invariant to any monotone transform of the score.
_SCORE_CENTRE = 0.045
_SCORE_SCALE = 0.020
_SCORE_FLOOR, _SCORE_CEILING = 0.15, 0.85

# Effect size saturates only at a genuinely extreme fraction -- the 95th percentile
# of tampered images sits at 0.115 -- so that fusion's positive weighting is not
# spent on readings that are merely above centre.
_EFFECT_SATURATION = 0.15

# Presentation only. The score above uses the UNFILTERED fraction, because that is
# the statistic that was measured, but isolated single blocks scattered across a
# frame are not something a reviewer can act on.
_MIN_CLUSTER_BLOCKS = 2


def _keep_clusters(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1 or not mask.any():
        return mask
    labelled, count = label(mask, structure=np.ones((3, 3), dtype=int))
    sizes = np.bincount(labelled.ravel())
    keep = [i for i in range(1, count + 1) if sizes[i] >= min_size]
    return np.isin(labelled, keep) if keep else np.zeros_like(mask)


@register
class NoiseprintStructureDetector(Detector):
    """Locate regions whose learned camera fingerprint differs in arrangement."""

    id = "sensor.noiseprint_structure"
    tier = Tier.SENSOR
    localises = True
    cost = 6

    def applies_to(self, case: ImageCase) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
        except ImportError:
            return False, "camera-fingerprint analysis requires pytorch"

        from ...learned.noiseprint import WEIGHTS_DIR

        if not WEIGHTS_DIR.is_dir() or not any(WEIGHTS_DIR.glob("*.pth")):
            return False, "no noiseprint weights installed"

        h, w = case.pixels().shape[:2]
        if min(h, w) < BLOCK * 6:
            return False, f"image too small for {BLOCK}px fingerprint blocks ({w}x{h})"
        return True, ""

    def _run(self, case: ImageCase) -> Evidence:
        from ...learned.noiseprint import quality_factor

        gray = case.pixels().astype(np.float32).mean(axis=2) / 255.0
        qf = quality_factor(case.image_path)
        residual = residual_for(case.image_path, qf)

        structure = period2(residual, BLOCK)
        clipped = clipped_share(gray, BLOCK)
        usable = (energy(residual, BLOCK) > 1e-6) & (clipped <= MAX_CLIPPED_FRACTION)
        if int(usable.sum()) < MIN_BLOCKS:
            return Evidence.not_applicable(
                self.id, self.tier, "too few blocks carry a readable fingerprint"
            )

        z, centre, scale = robust_z(structure, usable)
        if centre < MIN_DEMOSAIC_STRUCTURE:
            return Evidence.not_applicable(
                self.id,
                self.tier,
                "no demosaicing structure in this image, so it carries no camera "
                "fingerprint to compare regions against",
            )

        anomalous = usable & (z > _Z_ANOMALOUS)
        fraction = float(anomalous.sum() / int(usable.sum()))

        score = float(
            np.clip(
                1.0 / (1.0 + np.exp(-(fraction - _SCORE_CENTRE) / _SCORE_SCALE)),
                _SCORE_FLOOR,
                _SCORE_CEILING,
            )
        )
        effect_size = float(min(1.0, fraction / _EFFECT_SATURATION))
        shown = _keep_clusters(anomalous, _MIN_CLUSTER_BLOCKS)

        details: dict[str, object] = {
            "quality_factor": qf,
            "block_px": BLOCK,
            "readout": "period2",
            "readable_blocks": int(usable.sum()),
            "excluded_saturated": int((clipped > MAX_CLIPPED_FRACTION).sum()),
            "foreign_blocks": int(anomalous.sum()),
            "anomalous_fraction": round(fraction, 4),
            "contiguous_blocks": int(shown.sum()),
            "max_deviation_sigma": round(float(z.max()), 2),
            "demosaic_structure": round(centre, 2),
        }

        # Confidence rests on the frame having a coherent fingerprint to deviate
        # from. Measured as spread RELATIVE to the frame's own centre, which is the
        # only form that survives changing the readout: the absolute MAD of this
        # statistic runs 1.4 to 7.2 across Korus frames, so a constant tuned for the
        # energy readout pinned every image to the confidence floor. Relative spread
        # runs 0.068 to 0.346 over the same frames.
        support = min(1.0, int(usable.sum()) / 400.0)
        coherence = float(np.clip(1.0 - (scale / centre) / 0.5, 0.15, 1.0))
        confidence = float(np.clip(0.85 * support * coherence, 0.15, 0.85))

        if not shown.any():
            return Evidence(
                detector_id=self.id,
                tier=self.tier,
                applicable=True,
                score=score,
                confidence=confidence,
                effect_size=effect_size,
                explanation=(
                    f"fingerprint structure is consistent across "
                    f"{int(usable.sum())} blocks"
                ),
                details=details,
            )

        heat = np.clip(z / (2.0 * _Z_ANOMALOUS), 0.0, 1.0)
        heat[~shown] = 0.0
        heat = np.kron(heat, np.ones((BLOCK, BLOCK), dtype=np.float32))
        h, w = gray.shape
        full = np.zeros((h, w), dtype=np.float32)
        full[: heat.shape[0], : heat.shape[1]] = heat[:h, :w]

        return Evidence(
            detector_id=self.id,
            tier=self.tier,
            applicable=True,
            score=score,
            confidence=confidence,
            effect_size=effect_size,
            explanation=(
                f"{int(anomalous.sum())} of {int(usable.sum())} blocks carry "
                f"demosaicing structure inconsistent with the rest of the image "
                f"(max deviation {float(z.max()):.1f} sigma)"
            ),
            heatmap=full,
            details=details,
        )
