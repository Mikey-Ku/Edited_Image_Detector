"""Regions whose camera fingerprint differs in MAGNITUDE from the rest of the frame.

This is the detector every hand-crafted attempt in this project was reaching for.
The physics was always right -- a manipulated region carries a different camera
signature -- but deriving that signature in closed form kept failing: the wavelet
residual, the block grid, and the flat-region silence each looked convincing on
one image and fell apart across several.

Noiseprint supplies the residual instead of deriving it. Because the network was
trained to suppress scene content and enhance camera-model artefacts, the same
per-block anomaly search that failed on a hand-derived residual has a real signal
to work with.

**This detector reduces each block to residual energy, which is blind to spatial
arrangement.** Shuffling a block's pixels leaves the reading unchanged, and a camera
fingerprint lives largely in the arrangement -- so on its own this readout separates
Korus forgeries from their matched originals at only AUC 0.517 on held-out images,
a confidence interval spanning chance. `sensor.noiseprint_structure` measures the
arrangement instead and separates them at 0.677.

**It is kept anyway, because it wins where it matters most.** Compared at matched
false positive rate, on the same 110 forgeries:

    matched FPR    energy-only hits   structure-only hits   exact binomial
    1.8%                  7                    0               p = 0.016
    5.0%                  2                    3               p = 1.000
    10.0%                 5                   11               p = 0.210

At the high-precision corner -- the operating point an insurance triage system
actually runs at, where a false accusation is expensive -- this readout catches
seven forgeries the structural one misses and none the other way round. That is a
nested win at p = 0.016, not noise. Its bimodal design is why: silent almost always,
confident when it speaks. AUC integrates over the whole curve and does not see this,
which is a good reason not to select detectors on AUC alone.

Two further details matter:

**The quality factor must match.** A separate network is trained per JPEG
quantisation level, and using the wrong one degrades the fingerprint badly. It is
recovered from the file's own quantisation table, never assumed.

**Saturated blocks are excluded**, and frames with no demosaicing structure are
abstained on entirely -- see `_fingerprint.py` for both rules.

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

BLOCK = 32

# Robust deviation beyond which a block's fingerprint is called foreign.
_Z_ANOMALOUS = 3.5

# A manipulation is contiguous; isolated blocks across a large frame are noise.
_MIN_CLUSTER_BLOCKS = 3

# Floors below which a finding is reported as clean rather than as a weak
# positive. A detector whose "nothing here" still scores highly poisons fusion --
# and this bimodal shape is exactly what buys the high-precision corner above.
_MIN_ANOMALOUS_BLOCKS = 4
_MIN_ANOMALOUS_FRACTION = 0.012


def _keep_clusters(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1 or not mask.any():
        return mask
    labelled, count = label(mask, structure=np.ones((3, 3), dtype=int))
    sizes = np.bincount(labelled.ravel())
    keep = [i for i in range(1, count + 1) if sizes[i] >= min_size]
    return np.isin(labelled, keep) if keep else np.zeros_like(mask)


@register
class NoiseprintAnomalyDetector(Detector):
    """Locate regions whose learned camera fingerprint differs in strength."""

    id = "sensor.noiseprint_anomaly"
    tier = Tier.SENSOR
    localises = True
    cost = 5

    def applies_to(self, case: ImageCase) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
        except ImportError:
            return False, "camera-fingerprint analysis requires pytorch"

        from ...learned.noiseprint import WEIGHTS_DIR

        if not WEIGHTS_DIR.is_dir() or not any(WEIGHTS_DIR.glob("*.pth")):
            return False, "no noiseprint weights installed"

        h, w = case.pixels().shape[:2]
        if min(h, w) < BLOCK * 8:
            return False, f"image too small for {BLOCK}px fingerprint blocks ({w}x{h})"
        return True, ""

    def _run(self, case: ImageCase) -> Evidence:
        from ...learned.noiseprint import quality_factor

        gray = case.pixels().astype(np.float32).mean(axis=2) / 255.0
        qf = quality_factor(case.image_path)
        residual = residual_for(case.image_path, qf)

        block_energy = energy(residual, BLOCK)
        clipped = clipped_share(gray, BLOCK)
        usable = (block_energy > 1e-6) & (clipped <= MAX_CLIPPED_FRACTION)
        if int(usable.sum()) < MIN_BLOCKS:
            return Evidence.not_applicable(
                self.id, self.tier, "too few blocks carry a readable fingerprint"
            )

        if float(np.median(period2(residual, BLOCK)[usable])) < MIN_DEMOSAIC_STRUCTURE:
            return Evidence.not_applicable(
                self.id,
                self.tier,
                "no demosaicing structure in this image, so it carries no camera "
                "fingerprint to compare regions against",
            )

        # Log space: fingerprint energy varies multiplicatively with local content
        # that the network did not fully suppress.
        logs = np.log(np.maximum(block_energy, 1e-6))
        z, _, scale = robust_z(logs, usable)

        flagged = usable & (z > _Z_ANOMALOUS)
        anomalous = _keep_clusters(flagged, _MIN_CLUSTER_BLOCKS)
        fraction = float(anomalous.sum() / int(usable.sum()))

        if (
            int(anomalous.sum()) < _MIN_ANOMALOUS_BLOCKS
            or fraction < _MIN_ANOMALOUS_FRACTION
        ):
            anomalous = np.zeros_like(anomalous)
            fraction = 0.0

        details: dict[str, object] = {
            "quality_factor": qf,
            "block_px": BLOCK,
            "readout": "energy",
            "readable_blocks": int(usable.sum()),
            "excluded_saturated": int((clipped > MAX_CLIPPED_FRACTION).sum()),
            "foreign_blocks": int(anomalous.sum()),
            "isolated_discarded": int(flagged.sum() - anomalous.sum()),
            "max_deviation_sigma": round(float(z.max()), 2),
            "fingerprint_spread": round(scale, 4),
        }

        # Confidence rests on the frame having a coherent fingerprint to deviate
        # from. A wide spread means the residual is dominated by leaked content and
        # any single anomaly is unreliable.
        support = min(1.0, int(usable.sum()) / 400.0)
        coherence = float(np.clip(1.0 - scale / 0.8, 0.15, 1.0))
        confidence = float(np.clip(0.85 * support * coherence, 0.15, 0.85))

        if not anomalous.any():
            return Evidence(
                detector_id=self.id,
                tier=self.tier,
                applicable=True,
                score=0.22,
                confidence=confidence,
                effect_size=0.0,
                explanation=(
                    f"fingerprint strength is consistent across "
                    f"{int(usable.sum())} blocks"
                ),
                details=details,
            )

        heat = np.clip(z / (2.0 * _Z_ANOMALOUS), 0.0, 1.0)
        heat[~anomalous] = 0.0
        heat = np.kron(heat, np.ones((BLOCK, BLOCK), dtype=np.float32))
        # Blocks are floor-divided, so pad the remainder back to the frame size.
        h, w = gray.shape
        full = np.zeros((h, w), dtype=np.float32)
        full[: heat.shape[0], : heat.shape[1]] = heat[:h, :w]

        return Evidence(
            detector_id=self.id,
            tier=self.tier,
            applicable=True,
            score=float(min(0.94, 0.62 + 6.0 * fraction)),
            confidence=confidence,
            effect_size=float(min(1.0, fraction / 0.04)),
            explanation=(
                f"{int(anomalous.sum())} of {int(usable.sum())} blocks carry a camera "
                f"fingerprint inconsistent with the rest of the image "
                f"(max deviation {float(z.max()):.1f} sigma)"
            ),
            heatmap=full,
            details=details,
        )
