"""Error Level Analysis.

Re-save the image at a known quality and measure how much each region changes.
Regions that have been through a different compression history than the rest of
the frame should change by a different amount.

IMPORTANT -- read this before trusting the output. ELA is the most famous and the
most over-trusted technique in image forensics. It is included here deliberately
as a *baseline to beat*, not as a reliable signal, because:

  1. It confounds texture with tampering. Sharp edges and busy texture always
     produce high error regardless of provenance, so a detailed region of an
     entirely authentic photo lights up exactly like a splice.
  2. It is meaningless on images that were never JPEG, or that were saved at a
     quality near the probe quality.
  3. It produces confident-looking output on random input, which is precisely
     what makes it dangerous -- it looks like evidence.

The honest thing to do is report it with low confidence, document why, and let
the physically-grounded detectors (block grid, CFA, PRNU) carry the verdict.
Quantifying how badly ELA underperforms them is a result worth publishing.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from ...core.detector import Detector, register
from ...core.types import Evidence, ImageCase, Tier

PROBE_QUALITY = 90

# Local texture energy above which we refuse to interpret ELA at all, because the
# texture confound dominates.
_TEXTURE_SUSPECT = 0.20


def _texture_energy(gray: np.ndarray) -> np.ndarray:
    """Normalised local gradient magnitude -- our proxy for 'busy region'."""
    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy)
    hi = float(np.percentile(mag, 99)) + 1e-6
    return np.clip(mag / hi, 0.0, 1.0)


@register
class ELADetector(Detector):
    """Compression-error map from a fixed-quality re-save."""

    id = "compression.ela"
    tier = Tier.COMPRESSION
    localises = True
    cost = 1

    def applies_to(self, case: ImageCase) -> tuple[bool, str]:
        if not case.is_jpeg:
            return False, "ELA is only interpretable on JPEG-compressed input"
        return True, ""

    def _run(self, case: ImageCase) -> Evidence:
        original = case.pixels().astype(np.float32)

        buf = io.BytesIO()
        Image.fromarray(case.pixels()).save(buf, "JPEG", quality=PROBE_QUALITY)
        buf.seek(0)
        with Image.open(buf) as resaved_img:
            resaved = np.asarray(resaved_img.convert("RGB"), dtype=np.float32)

        error = np.abs(original - resaved).mean(axis=2)
        peak = float(error.max()) + 1e-6
        heat = error / peak

        gray = original.mean(axis=2) / 255.0
        texture = _texture_energy(gray)

        # Suppress error that is explained by texture. What survives is error that
        # texture does NOT account for -- the only part that could mean anything.
        residual = np.clip(heat - texture, 0.0, 1.0)

        hot_fraction = float((residual > 0.25).mean())
        texture_share = float((texture > _TEXTURE_SUSPECT).mean())

        # Confidence is capped low by construction. This detector does not get to
        # drive a verdict; it exists to be compared against.
        confidence = float(np.clip(0.30 * (1.0 - texture_share), 0.05, 0.30))

        if hot_fraction > 0.02:
            score = float(min(0.75, 0.5 + 4.0 * hot_fraction))
            explanation = (
                f"compression error over {hot_fraction:.1%} of the frame is not "
                f"explained by local texture (ELA -- weak signal, see docs)"
            )
        else:
            score = 0.4
            explanation = "no texture-independent compression-error anomaly (ELA)"

        return Evidence(
            detector_id=self.id,
            tier=self.tier,
            applicable=True,
            score=score,
            confidence=confidence,
            explanation=explanation,
            heatmap=residual,
            details={
                "probe_quality": PROBE_QUALITY,
                "hot_fraction": round(hot_fraction, 4),
                "texture_share": round(texture_share, 4),
                "caveat": "ELA confounds texture with tampering; baseline only",
            },
        )
