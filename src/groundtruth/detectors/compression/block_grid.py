"""Block Artefact Grid misalignment.

JPEG compresses in 8x8 blocks, quantising each independently. That leaves faint
discontinuities at every block boundary -- a grid, rigidly anchored to the image
origin, baked into the pixels.

Paste a region from another JPEG and it brings its own grid, shifted by wherever
it landed. Only 1 offset in 64 lands back in phase, so a spliced region almost
always carries a grid that disagrees with its host.

What makes this the strongest classical splice detector is that it is a
**geometric** argument rather than a statistical one. It is not "this region looks
unusual"; it is "this region's compression grid is offset by 3 pixels from its
surroundings, and no camera produces that." It also survives being re-saved,
because the misaligned grid is in the pixel values themselves -- the new
compression pass writes a second grid on top rather than erasing the first.

Estimation is separable: vertical block edges give the x phase, horizontal ones
the y phase, so this is two 8-way problems rather than one 64-way problem.

MEASURED OPERATING ENVELOPE -- this technique does not work everywhere, and the
limits are physical rather than fixable:

- **The composite must be saved at high quality (>= ~92).** Below that, the
  re-save quantises hard enough that its own grid, anchored at the origin,
  overwrites the foreign one. At q=88 detection fails completely on synthetic
  splices -- 0 of ~100 readable windows misaligned.
- **The donor must have been compressed at moderate-to-low quality.** A grid that
  was never strongly imposed cannot be read back out.
- **The image needs texture.** Smooth content is barely quantised, so there is no
  blocking signature to phase-align. Sensor-level noise is fine; heavy noise
  destroys the grid, because JPEG spends its bits encoding noise instead.
- **1 paste offset in 64 is invisible**, since the grids coincide.

That envelope is narrow, which is exactly why this is one signal among many rather
than the product.
"""

from __future__ import annotations

import numpy as np

from ...core.detector import Detector, register
from ...core.types import Evidence, ImageCase, Tier

BLOCK = 8

# Window over which one local grid phase is estimated. Eight blocks gives eight
# samples per phase -- enough to be stable, small enough to localise a splice.
WINDOW = 64
STRIDE = 32

# A phase estimate is only used when its peak stands this many robust deviations
# above the other seven. Flat regions (sky, blur) carry no grid to read, and
# guessing from them is how this technique generates false positives.
_MIN_PHASE_SNR = 3.0

# Minimum spread, in intensity levels, used when normalising the phase peak.
# Without an absolute floor the ratio diverges on flat content.
_SCALE_FLOOR = 0.05

# A cluster must beat CHANCE, and chance scales with the image.
#
# There are 64 possible grid phases. Across N confident windows, roughly N/64 will
# land on any particular wrong phase by estimation noise alone -- on a 1920x1080
# frame that is ~10 windows agreeing on a foreign phase in a completely untouched
# photograph. A fixed threshold of 2 was therefore guaranteed to fire on almost
# everything, and the benchmark confirmed it: 3 of 4 pristine images flagged.
#
# The requirement is now scaled against that null: a foreign phase has to be
# carried by several times more windows than coincidence would supply.
_PHASES = BLOCK * BLOCK
_CHANCE_MULTIPLE = 2.5
_MIN_CLUSTER_FLOOR = 4


def _min_cluster(n_confident: int) -> int:
    """Smallest believable foreign-phase cluster for this many windows."""
    expected = n_confident / _PHASES
    return max(_MIN_CLUSTER_FLOOR, int(np.ceil(_CHANCE_MULTIPLE * expected)))

_MIN_CONFIDENT_WINDOWS = 6


def _excess_step(gray: np.ndarray, axis: int) -> np.ndarray:
    """How much larger a step is than the steps immediately beside it.

    The naive measure -- absolute difference across a candidate boundary -- is
    useless on a textured image, because texture produces large differences
    everywhere and raises all eight phases together. What distinguishes a block
    boundary is that the step *there* exceeds the steps on either side of it.

    Subtracting the neighbouring steps cancels texture: busy content inflates the
    boundary and its neighbours equally, so the difference stays near zero, while a
    quantisation edge survives because its neighbours are not edges.
    """
    d = np.abs(np.diff(gray, axis=axis))
    if axis == 1:
        neighbours = (d[:, :-2] + d[:, 2:]) * 0.5
        return d[:, 1:-1] - neighbours
    neighbours = (d[:-2, :] + d[2:, :]) * 0.5
    return d[1:-1, :] - neighbours


def _phase_energy(profile: np.ndarray, axis: int) -> np.ndarray:
    """Mean excess-step at each of the 8 grid phases.

    Mean rather than median: the excess-step measure has already cancelled texture,
    so there is no heavy tail left for a median to protect against. Meanwhile a
    median over small integer pixel differences quantises hard -- it lands on
    0.0, 0.5, 1.0 -- which destroys the resolution needed to compare eight phases
    and collapses the spread of the non-peak phases to exactly zero.
    """
    n = profile.shape[axis]
    out = np.zeros(BLOCK, dtype=np.float32)
    for phase in range(BLOCK):
        idx = np.arange(phase, n, BLOCK)
        if idx.size:
            out[phase] = float(np.take(profile, idx, axis=axis).mean())
    return out


def _dominant_phase(energy: np.ndarray) -> tuple[int, float]:
    """Argmax phase and how far it stands above the other seven.

    The scale floor is ABSOLUTE and in intensity units. Dividing by the observed
    spread alone produces an unbounded ratio whenever the other phases happen to
    agree closely, which reads as overwhelming confidence in what is actually a
    featureless region -- the single reason the first version of this detector
    fired on every image it saw.
    """
    phase = int(np.argmax(energy))
    others = np.delete(energy, phase)
    spread = float(others.std())
    return phase, float((energy[phase] - float(others.mean())) / max(spread, _SCALE_FLOOR))


def _keep_clusters(mask: np.ndarray, min_size: int) -> np.ndarray:
    from scipy.ndimage import label

    if min_size <= 1 or not mask.any():
        return mask
    labelled, count = label(mask, structure=np.ones((3, 3), dtype=int))
    sizes = np.bincount(labelled.ravel())
    keep = [i for i in range(1, count + 1) if sizes[i] >= min_size]
    return np.isin(labelled, keep) if keep else np.zeros_like(mask)


@register
class BlockGridDetector(Detector):
    """Locate regions whose JPEG block grid is out of phase with the frame."""

    id = "compression.block_grid"
    tier = Tier.COMPRESSION
    localises = True
    cost = 3

    # EXPERIMENTAL -- the benchmark showed this was never working.
    #
    # It previously required only 2 windows to agree on a foreign phase. Across N
    # confident windows and 64 possible phases, chance alone supplies about N/64 --
    # roughly 10 on a full-frame photograph. The threshold sat below the noise
    # floor, so it fired on 3 of 4 PRISTINE images in the benchmark and on 16/16 of
    # every manipulation class equally. That is not detection, it is a constant.
    #
    # With the threshold raised to beat chance it now finds zero coherent foreign
    # phases even on the synthetic splice it was designed for, at both fixture and
    # full-frame size. So the phase estimator is not recovering the pasted region's
    # grid in enough windows to be distinguishable from noise.
    #
    # The physics is real and well documented; this implementation does not deliver
    # it. Held out of the default pipeline until the estimator is strong enough to
    # clear the chance threshold on a known splice.
    experimental = True

    def applies_to(self, case: ImageCase) -> tuple[bool, str]:
        if not case.container.actual.block_compressed:
            return False, (
                f"no 8x8 compression grid in {case.container.actual.value}"
            )
        h, w = case.pixels().shape[:2]
        if min(h, w) < WINDOW * 2:
            return False, f"image too small for {WINDOW}px grid windows ({w}x{h})"
        return True, ""

    def _run(self, case: ImageCase) -> Evidence:
        gray = case.pixels().astype(np.float32).mean(axis=2)
        h, w = gray.shape

        nrows = (h - WINDOW) // STRIDE + 1
        ncols = (w - WINDOW) // STRIDE + 1
        phase_x = np.full((nrows, ncols), -1, dtype=np.int16)
        phase_y = np.full((nrows, ncols), -1, dtype=np.int16)
        snr = np.zeros((nrows, ncols), dtype=np.float32)

        for i in range(nrows):
            for j in range(ncols):
                tile = gray[
                    i * STRIDE : i * STRIDE + WINDOW, j * STRIDE : j * STRIDE + WINDOW
                ]
                px, sx = _dominant_phase(_phase_energy(_excess_step(tile, 1), axis=1))
                py, sy = _dominant_phase(_phase_energy(_excess_step(tile, 0), axis=0))

                # Phase is measured within the window, which starts at a known
                # offset in the image. Convert to an absolute phase so windows are
                # comparable to one another.
                phase_x[i, j] = (px + j * STRIDE) % BLOCK
                phase_y[i, j] = (py + i * STRIDE) % BLOCK
                snr[i, j] = min(sx, sy)

        confident = snr >= _MIN_PHASE_SNR
        if int(confident.sum()) < _MIN_CONFIDENT_WINDOWS:
            return Evidence.not_applicable(
                self.id,
                self.tier,
                f"only {int(confident.sum())} windows carry a readable block grid; "
                f"image is too smooth or too heavily recompressed",
            )

        # The frame's own grid is whichever phase the confident windows agree on.
        codes = (phase_y[confident].astype(int) * BLOCK + phase_x[confident].astype(int))
        host_code = int(np.bincount(codes, minlength=64).argmax())
        host_y, host_x = divmod(host_code, BLOCK)

        # A spliced region carries ONE foreign grid, so its windows must agree with
        # EACH OTHER, not merely differ from the host. Requiring only "differs from
        # host" lets windows that disagree for unrelated reasons -- one wrong on the
        # x axis, another on y -- form a spurious cluster, and the phase reported for
        # it is then a mode over two axes that no single window actually had.
        codes_all = phase_y.astype(int) * BLOCK + phase_x.astype(int)
        disagrees = confident & (codes_all != host_code)

        min_cluster = _min_cluster(int(confident.sum()))
        clustered = np.zeros_like(disagrees)
        for code in np.unique(codes_all[disagrees]) if disagrees.any() else []:
            same = disagrees & (codes_all == code)
            clustered |= _keep_clusters(same, min_cluster)

        isolated = int(disagrees.sum() - clustered.sum())
        agreement = float((confident & ~disagrees).sum() / max(int(confident.sum()), 1))

        details: dict[str, object] = {
            "host_phase": [host_x, host_y],
            "windows_readable": int(confident.sum()),
            "windows_misaligned": int(clustered.sum()),
            "isolated_discarded": isolated,
            "host_agreement": round(agreement, 3),
            "min_cluster_required": min_cluster,
        }

        if not clustered.any():
            return Evidence(
                detector_id=self.id,
                tier=self.tier,
                applicable=True,
                score=0.2,
                confidence=float(np.clip(0.85 * agreement, 0.2, 0.85)),
                explanation=(
                    f"block grid is in phase ({host_x},{host_y}) across all "
                    f"{int(confident.sum())} readable windows"
                ),
                details=details,
            )

        # The phase difference between the foreign region and its host. This is
        # RELATED to the paste displacement modulo 8, but the exact geometric
        # correspondence has not been verified against ground truth: on synthetic
        # splices the y component tracks the true displacement while the x component
        # frequently reads zero, so the x grid is being recovered less reliably than
        # the y grid. Reported as an observed phase difference, not as a recovered
        # paste offset, until that is measured properly.
        region_code = int(np.bincount(codes_all[clustered], minlength=64).argmax())
        off_y, off_x = divmod(region_code, BLOCK)
        details["region_phase"] = [off_x, off_y]
        details["phase_difference"] = [
            (off_x - host_x) % BLOCK,
            (off_y - host_y) % BLOCK,
        ]

        # Weight each pixel by the FRACTION of windows covering it that disagreed,
        # not by how many did. Interior pixels are covered by four windows and edge
        # pixels by one, so a raw count would report the middle of the frame as
        # suspicious purely because more windows overlap there.
        # Attribute each window's verdict to its CENTRE, not its whole extent. A
        # 64px window overlapping the edge of a splice would otherwise smear its
        # finding 32px into untouched surroundings in every direction.
        pad = (WINDOW - STRIDE) // 2
        hot = np.zeros((h, w), dtype=np.float32)
        cover = np.zeros((h, w), dtype=np.float32)
        for i in range(nrows):
            for j in range(ncols):
                if not confident[i, j]:
                    continue
                ys, xs = i * STRIDE + pad, j * STRIDE + pad
                cover[ys : ys + STRIDE, xs : xs + STRIDE] += 1.0
                if clustered[i, j]:
                    hot[ys : ys + STRIDE, xs : xs + STRIDE] += 1.0
        heat = np.divide(hot, cover, out=np.zeros_like(hot), where=cover > 0)

        misaligned_fraction = float(clustered.sum() / max(int(confident.sum()), 1))
        effect = float(min(1.0, misaligned_fraction / 0.15))
        return Evidence(
            detector_id=self.id,
            tier=self.tier,
            applicable=True,
            score=float(min(0.95, 0.65 + 1.5 * misaligned_fraction)),
            confidence=float(np.clip(0.85 * agreement, 0.25, 0.85)),
            effect_size=effect,
            explanation=(
                f"{int(clustered.sum())} of {int(confident.sum())} readable windows "
                f"carry a block grid at phase ({off_x},{off_y}) while the frame is at "
                f"({host_x},{host_y}) -- a region compressed on a different 8x8 grid "
                f"was composited in, phase difference "
                f"({(off_x - host_x) % BLOCK},{(off_y - host_y) % BLOCK})"
            ),
            heatmap=heat,
            details=details,
        )
