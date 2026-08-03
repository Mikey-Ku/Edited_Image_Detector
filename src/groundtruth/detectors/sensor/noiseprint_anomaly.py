"""Regions whose camera fingerprint disagrees with the rest of the frame.

This is the detector every hand-crafted attempt in this project was reaching for.
The physics was always right -- a manipulated region carries a different camera
signature -- but deriving that signature in closed form kept failing: the wavelet
residual, the block grid, and the flat-region silence each looked convincing on
one image and fell apart across several.

Noiseprint supplies the residual instead of deriving it. Because the network was
trained to suppress scene content and enhance camera-model artefacts, the same
per-block anomaly search that failed on a hand-derived residual has a real signal
to work with.

The search itself is deliberately simple, and the same shape as every other
detector here: per-block statistics, a robust deviation from the frame's own
median, and a contiguity requirement so isolated blocks are discarded. The
difference is entirely in the quality of the residual it runs on -- which is the
finding worth recording.

Two details matter:

**The quality factor must match.** A separate network is trained per JPEG
quantisation level, and using the wrong one degrades the fingerprint badly. It is
recovered from the file's own quantisation table, never assumed.

**Content still leaks through.** Suppression is not perfect, so a heavily textured
region retains some scene energy. Blocks are compared in log space against a
robust centre, which handles the multiplicative part of that leak.

**Saturated blocks are excluded.** Where the sensor clipped, there is no noise to
fingerprint -- the pixels are pinned at the rail and carry no information about the
camera. Including them made a legitimate exposure lift look like tampering, because
brightening a photo blows out its highlights and every blown region then reads as a
foreign fingerprint. Absence of a fingerprint where physics says none can exist is
not evidence of anything.

Nonprofit use only -- see ``groundtruth.learned.noiseprint`` for the licence.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label

from ...core.detector import Detector, register
from ...core.types import Evidence, ImageCase, Tier

BLOCK = 32

# A block is unusable when this share of it sits at either rail. Clipped pixels
# carry no sensor noise by definition.
_CLIP_LOW, _CLIP_HIGH = 0.02, 0.98
_MAX_CLIPPED_FRACTION = 0.20

# Robust deviation beyond which a block's fingerprint is called foreign.
_Z_ANOMALOUS = 3.5

# A manipulation is contiguous; isolated blocks across a large frame are noise.
_MIN_CLUSTER_BLOCKS = 3

_MIN_BLOCKS = 32

# Floors below which a finding is reported as clean rather than as a weak
# positive. A detector whose "nothing here" still scores highly poisons fusion.
_MIN_ANOMALOUS_BLOCKS = 4
_MIN_ANOMALOUS_FRACTION = 0.012


def _block_stats(residual: np.ndarray, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-block fingerprint energy, and the share of clipped pixels."""
    h, w = residual.shape
    nr, nc = h // BLOCK, w // BLOCK
    energy = np.zeros((nr, nc), dtype=np.float32)
    clipped = np.zeros((nr, nc), dtype=np.float32)
    for i in range(nr):
        for j in range(nc):
            sl = (slice(i * BLOCK, (i + 1) * BLOCK), slice(j * BLOCK, (j + 1) * BLOCK))
            energy[i, j] = float(residual[sl].std())
            g = gray[sl]
            clipped[i, j] = float(((g <= _CLIP_LOW) | (g >= _CLIP_HIGH)).mean())
    return energy, clipped


def _keep_clusters(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1 or not mask.any():
        return mask
    labelled, count = label(mask, structure=np.ones((3, 3), dtype=int))
    sizes = np.bincount(labelled.ravel())
    keep = [i for i in range(1, count + 1) if sizes[i] >= min_size]
    return np.isin(labelled, keep) if keep else np.zeros_like(mask)


@register
class NoiseprintAnomalyDetector(Detector):
    """Locate regions whose learned camera fingerprint differs from the frame."""

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
        from ...learned.noiseprint import extract, quality_factor

        gray = case.pixels().astype(np.float32).mean(axis=2) / 255.0
        qf = quality_factor(case.image_path)
        residual = extract(gray, qf)

        energy, clipped = _block_stats(residual, gray)
        usable = (energy > 1e-6) & (clipped <= _MAX_CLIPPED_FRACTION)
        if int(usable.sum()) < _MIN_BLOCKS:
            return Evidence.not_applicable(
                self.id, self.tier, "too few blocks carry a readable fingerprint"
            )

        # Log space: fingerprint energy varies multiplicatively with local content
        # that the network did not fully suppress.
        logs = np.log(np.maximum(energy, 1e-6))
        centre = float(np.median(logs[usable]))
        scale = float(np.median(np.abs(logs[usable] - centre))) * 1.4826 + 1e-6

        z = np.zeros_like(energy)
        z[usable] = np.abs(logs[usable] - centre) / scale

        flagged = usable & (z > _Z_ANOMALOUS)
        anomalous = _keep_clusters(flagged, _MIN_CLUSTER_BLOCKS)
        fraction = float(anomalous.sum() / max(int(usable.sum()), 1))

        if (
            int(anomalous.sum()) < _MIN_ANOMALOUS_BLOCKS
            or fraction < _MIN_ANOMALOUS_FRACTION
        ):
            anomalous = np.zeros_like(anomalous)
            fraction = 0.0

        details: dict[str, object] = {
            "quality_factor": qf,
            "block_px": BLOCK,
            "readable_blocks": int(usable.sum()),
            "excluded_saturated": int((clipped > _MAX_CLIPPED_FRACTION).sum()),
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
                    f"the camera fingerprint is consistent across "
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
