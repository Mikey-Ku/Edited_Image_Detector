"""Local sharpness inconsistency -- content that is too crisp, or too soft.

A lens has a point spread function. Everything it photographs is blurred by it,
and by depth of field, and by whatever motion occurred during the exposure.
Content that did not come through that lens does not carry that blur.

The measurement is **two-sided**, because addition and removal are mirror images
and a one-sided test is blind to half the problem:

- **Too sharp** -- a rendered graphic or a crop from a sharper source pasted in.
  Hard edges, no optical falloff, uniformly crisp. This is how a replaced licence
  plate, a pasted price, or an overlaid document field looks.
- **Too soft** -- content cloned or inpainted *over* something. Generative fill
  and healing brushes synthesise from surrounding low-frequency context, so the
  result is smoother than anything the sensor actually recorded.

The sign of the anomaly therefore classifies the edit, which no other detector
here does.

The comparison is **local, not global**. Depth of field and lens softness vary
smoothly across a frame -- the far background is legitimately blurrier than the
near foreground, and a global threshold just rediscovers that. What is not
legitimate is a region that disagrees with its *immediate neighbours*, because
blur cannot change discontinuously across a few dozen pixels of a real
photograph. Each block is therefore scored against a ring of surrounding blocks
rather than against the whole image.

A first version normalised fine detail by coarse detail, on the theory that the
ratio approximates blur-kernel width independent of content. It does not work
here: a rendered graphic has hard edges, so fine and coarse energy rise together
and the ratio cancels the very signal being looked for.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, label

from ...core.detector import Detector, register
from ...core.types import Evidence, ImageCase, Tier

BLOCK = 24

_FINE = 1.0

# Radius, in blocks, of the neighbourhood a block is compared against. Wide enough
# to span a pasted object's surroundings, narrow enough that a real depth-of-field
# gradient across the frame stays inside the baseline rather than becoming an
# anomaly.
_RING = 4

# Blocks with almost no structure carry no usable measurement -- a blank sky has
# nothing to be sharp or soft about. Excluded rather than guessed at.
_MIN_ENERGY = 0.004

_Z_ANOMALOUS = 3.0
_MIN_CLUSTER_BLOCKS = 2
_MIN_BLOCKS = 24


def _acutance(gray: np.ndarray) -> np.ndarray:
    """Per-block fine-detail energy -- how much the block resolves."""
    fine = gray - gaussian_filter(gray, _FINE)
    h, w = gray.shape
    nr, nc = h // BLOCK, w // BLOCK
    out = np.zeros((nr, nc), dtype=np.float32)
    for i in range(nr):
        for j in range(nc):
            out[i, j] = float(
                fine[i * BLOCK : (i + 1) * BLOCK, j * BLOCK : (j + 1) * BLOCK].std()
            )
    return out


def _local_deviation(values: np.ndarray, usable: np.ndarray) -> tuple[np.ndarray, float]:
    """Robust z of each block against the ring of blocks around it.

    The block itself is excluded from its own baseline, so a large uniform patch
    cannot quietly define the level it is then judged against.
    """
    nr, nc = values.shape
    z = np.zeros_like(values)
    scales: list[float] = []
    for i in range(nr):
        for j in range(nc):
            if not usable[i, j]:
                continue
            r0, r1 = max(0, i - _RING), min(nr, i + _RING + 1)
            c0, c1 = max(0, j - _RING), min(nc, j + _RING + 1)
            ring = values[r0:r1, c0:c1][usable[r0:r1, c0:c1]]
            if ring.size < 8:
                continue
            centre = float(np.median(ring))
            scale = float(np.median(np.abs(ring - centre))) * 1.4826
            if scale < 1e-6:
                continue
            z[i, j] = (values[i, j] - centre) / scale
            scales.append(scale / (centre + 1e-6))
    return z, float(np.median(scales)) if scales else 1.0


def _keep_clusters(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1 or not mask.any():
        return mask
    labelled, count = label(mask, structure=np.ones((3, 3), dtype=int))
    sizes = np.bincount(labelled.ravel())
    keep = [i for i in range(1, count + 1) if sizes[i] >= min_size]
    return np.isin(labelled, keep) if keep else np.zeros_like(mask)


@register
class SharpnessInconsistencyDetector(Detector):
    """Find regions whose blur does not match the lens that took the photograph."""

    id = "geometric.sharpness_inconsistency"
    tier = Tier.GEOMETRIC
    localises = True
    cost = 2

    # EXPERIMENTAL -- measured, and it does not work yet.
    #
    # On a real photograph with a hand-replaced licence plate it fires at 7.2 sigma
    # but localises to the damaged bumper and the tree line instead: 0% of flagged
    # pixels fall inside the actual edit. On matched Korus pairs it flags the
    # unedited original as often as the edited version. Added to the synthetic
    # splice pipeline it dropped localisation hit rate below the passing threshold.
    #
    # The underlying physics is sound and the literature supports it, but a local
    # acutance z-score is not selective enough on its own: genuine depth-of-field
    # transitions and crumpled metal both deviate more strongly than a pasted
    # graphic does. It likely needs pairing with a second cue -- absence of sensor
    # noise inside the sharp region -- so that "sharp AND noiseless" is required
    # rather than "sharp".
    experimental = True

    def applies_to(self, case: ImageCase) -> tuple[bool, str]:
        h, w = case.pixels().shape[:2]
        if min(h, w) < BLOCK * 8:
            return False, f"image too small for {BLOCK}px sharpness blocks ({w}x{h})"
        return True, ""

    def _run(self, case: ImageCase) -> Evidence:
        gray = case.pixels().astype(np.float32).mean(axis=2) / 255.0
        acu = _acutance(gray)
        usable = acu > _MIN_ENERGY
        if int(usable.sum()) < _MIN_BLOCKS:
            return Evidence.not_applicable(
                self.id,
                self.tier,
                f"only {int(usable.sum())} blocks carry enough structure to measure blur",
            )

        signed, spread = _local_deviation(np.log(np.maximum(acu, 1e-6)), usable)

        sharp = _keep_clusters(usable & (signed > _Z_ANOMALOUS), _MIN_CLUSTER_BLOCKS)
        soft = _keep_clusters(usable & (signed < -_Z_ANOMALOUS), _MIN_CLUSTER_BLOCKS)
        anomalous = sharp | soft

        fraction = float(anomalous.sum() / max(int(usable.sum()), 1))
        details: dict[str, object] = {
            "block_px": BLOCK,
            "measurable_blocks": int(usable.sum()),
            "too_sharp_blocks": int(sharp.sum()),
            "too_soft_blocks": int(soft.sum()),
            "max_deviation_sigma": round(float(np.abs(signed).max()), 2),
            "neighbourhood_spread": round(spread, 4),
        }

        # Confidence tracks how well-determined the frame's own blur is. A photo
        # with a shallow depth of field genuinely varies, which widens the spread
        # and should make us less willing to call any single region anomalous.
        support = min(1.0, int(usable.sum()) / 200.0)
        homogeneity = float(np.clip(1.0 - spread / 0.55, 0.15, 1.0))
        confidence = float(np.clip(0.75 * support * homogeneity, 0.10, 0.75))

        if not anomalous.any():
            return Evidence(
                detector_id=self.id,
                tier=self.tier,
                applicable=True,
                score=0.28,
                confidence=confidence,
                effect_size=0.0,
                explanation=(
                    f"blur is consistent across {int(usable.sum())} measurable blocks"
                ),
                details=details,
            )

        heat = np.clip(np.abs(signed) / (2.0 * _Z_ANOMALOUS), 0.0, 1.0)
        heat[~anomalous] = 0.0
        heat = np.kron(heat, np.ones((BLOCK, BLOCK), dtype=np.float32))

        if sharp.sum() and not soft.sum():
            kind = (
                f"{int(sharp.sum())} blocks are sharper than the lens that took this "
                f"photograph could resolve -- consistent with rendered or pasted content"
            )
        elif soft.sum() and not sharp.sum():
            kind = (
                f"{int(soft.sum())} blocks are softer than the rest of the frame -- "
                f"consistent with content cloned or generated over something"
            )
        else:
            kind = (
                f"{int(sharp.sum())} blocks too sharp and {int(soft.sum())} too soft "
                f"for a single optical path"
            )

        return Evidence(
            detector_id=self.id,
            tier=self.tier,
            applicable=True,
            score=float(min(0.93, 0.58 + 2.5 * fraction)),
            confidence=confidence,
            effect_size=float(
                min(1.0, fraction / 0.06)
                * min(1.0, float(np.abs(signed).max()) / (2.0 * _Z_ANOMALOUS))
            ),
            explanation=(
                f"{kind} (max deviation "
                f"{float(np.abs(signed).max()):.1f} sigma)"
            ),
            heatmap=heat,
            details=details,
        )
