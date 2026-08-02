"""Reconstruct what the image looked like before it was edited.

Given an embedded preview that the editor failed to regenerate, this recovers a
view of the pre-edit image and reports exactly what changed between then and now.

Two rules govern everything here, because this output could end up in front of an
adjuster deciding whether to pay a claim:

**Compare at the preview's resolution, not the main image's.** Upscaling a 160x120
thumbnail to 4000px and diffing invents detail that was never measured, and the
interpolation error swamps the real signal. The comparison happens where both
images actually carry information; only the resulting difference map is enlarged
for display.

**Never label a synthesised image as recovered.** :class:`Fidelity` is carried
through every result. RECOVERED means real pixels that were in the file. INFERRED
would mean a model's plausible guess, which is a hypothesis and never evidence. We
do not currently produce INFERRED output, and if we ever do it must stay visually
and structurally distinct from the recovered kind.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image

from ..core.image_io import load_rgb
from .preview import Preview, extract_previews

# Below this, a preview is too coarse for region-level claims about what changed.
_PARTIAL_PIXEL_RATIO = 0.02

# Aspect ratios differing by more than this indicate a crop, not rounding.
_ASPECT_TOLERANCE = 0.02

_CHANGE_THRESHOLD = 0.30

# Minimum mean absolute difference, in normalised intensity, before a pixel counts
# as changed. Re-encoding a preview at a lower quality typically costs 2-5 levels
# out of 255 (~0.02); an edit that matters moves them far further.
_ABS_CHANGE_FLOOR = 0.05

# How far above the threshold a difference must reach to read as fully changed.
# ~20 intensity levels out of 255: well beyond any re-encoding artefact, and well
# within the magnitude of an edit that would alter what a photograph depicts.
_CHANGE_RAMP = 0.08


class Fidelity(str, Enum):
    RECOVERED = "recovered"
    """Real pixels that were present in the file. Evidence."""

    PARTIAL = "partial"
    """Real pixels, but at a resolution too low to localise finely. Evidence, weakly."""

    INFERRED = "inferred"
    """Synthesised. A hypothesis about what may have been there. NEVER evidence."""


@dataclass
class Reconstruction:
    before: np.ndarray
    """Best available view of the pre-edit image, upscaled to the current size."""

    after: np.ndarray
    difference: np.ndarray
    """[0,1] change map at the current image's resolution."""

    regions: list[dict]
    fidelity: Fidelity
    source: str
    preview_size: tuple[int, int]
    current_size: tuple[int, int]
    cropped: bool
    changed_fraction: float
    caveat: str

    @property
    def is_evidence(self) -> bool:
        return self.fidelity is not Fidelity.INFERRED


def _match_tone(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Least-squares gain and offset per channel, mapping ``src`` onto ``ref``.

    A preview is encoded separately from the main image, so the two differ in
    brightness and colour even when the content is identical. Without correcting
    for that, the whole frame reads as changed and the real edit is invisible.
    Gain and offset are global, so a genuinely altered region cannot be absorbed.
    """
    out = np.empty_like(src, dtype=np.float32)
    for c in range(src.shape[2]):
        x = src[..., c].ravel().astype(np.float32)
        y = ref[..., c].ravel().astype(np.float32)
        var = float(x.var())
        if var < 1e-8:
            out[..., c] = src[..., c]
            continue
        gain = float(((x - x.mean()) * (y - y.mean())).mean() / var)
        offset = float(y.mean() - gain * x.mean())
        out[..., c] = np.clip(gain * src[..., c] + offset, 0.0, 255.0)
    return out


def _resize(a: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize an HxWx3 uint8-ish array to (width, height)."""
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return np.asarray(img.resize(size, Image.Resampling.LANCZOS), dtype=np.float32)


def _regions_from(diff: np.ndarray, threshold: float) -> list[dict]:
    from ..fusion.localisation import peak_regions

    return peak_regions(diff, threshold=threshold, min_area=16)


def reconstruct(path: Path, preview: Preview | None = None) -> Reconstruction | None:
    """Recover the pre-edit image from an embedded preview.

    Returns ``None`` when the container carries no usable preview -- which is the
    common case for images that have been through a messaging app, since those
    strip metadata wholesale.
    """
    current = load_rgb(path)
    ch, cw = current.shape[:2]

    if preview is None:
        candidates = [p for p in extract_previews(path) if p.pixels < ch * cw]
        if not candidates:
            return None
        preview = candidates[0]

    before_small = preview.as_rgb().astype(np.float32)
    pw, ph = preview.width, preview.height

    current_aspect = cw / ch
    preview_aspect = pw / ph
    cropped = abs(current_aspect - preview_aspect) / current_aspect > _ASPECT_TOLERANCE

    # Compare where both images carry real information: the preview's resolution.
    current_small = _resize(current, (pw, ph))
    before_matched = _match_tone(before_small, current_small)

    delta = np.abs(current_small - before_matched).mean(axis=2) / 255.0

    # The threshold must have an ABSOLUTE component. A purely relative one --
    # "top few percent of the difference distribution" -- reports the same
    # fraction of the frame as changed whether or not anything was edited, because
    # every distribution has a top few percent. The absolute floor is grounded in
    # what re-encoding physically costs: a preview stored at a lower quality
    # differs from the main image by a few intensity levels, whereas a real edit
    # moves them by tens.
    floor = float(np.median(delta))
    mad = float(np.median(np.abs(delta - floor))) * 1.4826
    threshold = max(_ABS_CHANGE_FLOOR, floor + 6.0 * mad)

    # Ramp above the threshold is ABSOLUTE, not proportional to the threshold.
    # Scaling the ramp by the threshold would mean a noisier image needs a larger
    # edit to register the same intensity on the map, which is backwards: how
    # visible a change is should not depend on how noisy its neighbours are. The
    # threshold adapts to the image; the ramp is a fixed physical quantity.
    change_small = np.clip((delta - threshold) / _CHANGE_RAMP, 0.0, 1.0)

    difference = np.asarray(
        Image.fromarray((change_small * 255).astype(np.uint8)).resize(
            (cw, ch), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    ) / 255.0

    changed_fraction = float((difference > _CHANGE_THRESHOLD).mean())

    pixel_ratio = preview.pixels / (ch * cw)
    fidelity = (
        Fidelity.RECOVERED if pixel_ratio >= _PARTIAL_PIXEL_RATIO else Fidelity.PARTIAL
    )

    caveat = (
        f"Recovered from an embedded preview at {pw}x{ph}, "
        f"{pixel_ratio:.2%} of the current resolution. These are real pixels from the "
        f"file, not a reconstruction -- but fine detail was never stored at this size."
    )
    if cropped:
        caveat += (
            f" Aspect ratio changed ({preview_aspect:.3f} -> {current_aspect:.3f}): "
            f"the image has also been cropped, so some original content is absent "
            f"from the current frame entirely."
        )

    return Reconstruction(
        before=_resize(before_matched, (cw, ch)).astype(np.uint8),
        after=current,
        difference=difference,
        regions=_regions_from(difference, _CHANGE_THRESHOLD),
        fidelity=fidelity,
        source=preview.source,
        preview_size=(pw, ph),
        current_size=(cw, ch),
        cropped=cropped,
        changed_fraction=changed_fraction,
        caveat=caveat,
    )
