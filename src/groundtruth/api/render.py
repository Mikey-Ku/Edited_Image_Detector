"""Render a verdict as a side-by-side image: original, overlay, raw heatmap.

Hand-rolled colour ramp rather than a matplotlib dependency -- this needs to run
in a service, and pulling a plotting library into an inference path to map three
floats is not worth it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..core.types import Verdict
from ..fusion.localisation import peak_regions

# Black -> deep red -> orange -> yellow -> white. Monotonic in luminance so it
# still reads correctly in greyscale or for a colourblind viewer.
_RAMP = np.array(
    [
        [0, 0, 0],
        [60, 0, 20],
        [140, 20, 20],
        [220, 80, 10],
        [250, 170, 30],
        [255, 240, 120],
        [255, 255, 255],
    ],
    dtype=np.float32,
)


def colourise(heat: np.ndarray) -> np.ndarray:
    """Map a [0,1] array to RGB uint8 via the ramp above."""
    h = np.clip(heat, 0.0, 1.0) * (len(_RAMP) - 1)
    lo = np.floor(h).astype(int)
    hi = np.minimum(lo + 1, len(_RAMP) - 1)
    t = (h - lo)[..., None]
    return (_RAMP[lo] * (1 - t) + _RAMP[hi] * t).astype(np.uint8)


def overlay(base: np.ndarray, heat: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend the colourised heatmap over the image, weighted by heat itself.

    Weighting by heat means cold regions stay legible instead of being washed out
    by a flat alpha -- the adjuster still needs to see the photograph.
    """
    colour = colourise(heat).astype(np.float32)
    a = (np.clip(heat, 0, 1) * alpha)[..., None]
    return (base.astype(np.float32) * (1 - a) + colour * a).astype(np.uint8)


def duplicate_pair(
    base: np.ndarray,
    regions: list[dict],
    displacement: tuple[float, float] | None = None,
    zoom: int = 2,
    gap: int = 24,
) -> tuple[np.ndarray, dict] | None:
    """Crop the two matched regions and set them side by side, with the numbers.

    Copy-move states its finding in units nobody outside the code can check: "212
    keypoint pairs share a displacement of (400, -10) px". True, and useless as
    evidence to a person. The same finding shown as two crops beside each other is
    checkable by looking, which is what evidence is supposed to be.

    **The pair comes from the measured displacement, not from the heatmap.** Picking
    the two hottest regions and hoping they are the matched pair works only when the
    heatmap has exactly two clean blobs. On a car panel copy-move lights up a dozen
    scattered fragments, and the top two were arbitrary neighbours: they scored a
    ratio of 1.3, which reads as evidence and is not. The detector already knows the
    offset, so the honest crop is one region and that same region shifted by it.

    The comparison against an unrelated patch of the same image matters as much as
    the pair itself. "These two look similar" is worth nothing without knowing what
    similar means for this photograph: a wall of repeating tiles would score well by
    accident. The control makes the ratio interpretable.

    Returns None when there is nothing trustworthy to show, which is most images.
    """
    if not regions:
        return None
    h, w = base.shape[:2]

    def crop_box(x0, y0, x1, y1) -> np.ndarray | None:
        x0, y0 = max(int(x0), 0), max(int(y0), 0)
        x1, y1 = min(int(x1), w), min(int(y1), h)
        if x1 - x0 < 24 or y1 - y0 < 24:
            return None
        return base[y0:y1, x0:x1]

    if displacement is not None:
        dx, dy = (round(v) for v in displacement)
        if abs(dx) < 8 and abs(dy) < 8:
            return None
        # Largest flagged region, and the same window shifted by the offset. One of
        # the two is the source and the other the paste; which is which is not
        # recoverable, and does not matter for showing they are the same pixels.
        r = max(regions, key=lambda r: r.get("area_px", 0))
        x0, y0, x1, y1 = (int(v) for v in r["bbox"])
        a = crop_box(x0, y0, x1, y1)
        b = crop_box(x0 + dx, y0 + dy, x1 + dx, y1 + dy)
        if a is None or b is None:
            # The partner falls outside the frame, so try the other direction.
            a = crop_box(x0 - dx, y0 - dy, x1 - dx, y1 - dy)
            b = crop_box(x0, y0, x1, y1)
            if a is None or b is None:
                return None
    else:
        if len(regions) < 2:
            return None
        top = sorted(regions, key=lambda r: -r.get("peak", 0))[:2]
        a = crop_box(*top[0]["bbox"])
        b = crop_box(*top[1]["bbox"])
        if a is None or b is None:
            return None

    # Compare on the overlap, since the two boxes are rarely the exact same size.
    ch, cw = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a, b = a[:ch, :cw], b[:ch, :cw]

    # Refine the alignment before measuring. The reported displacement is a median
    # over matched keypoints, so it is right to within a few pixels, and the crop
    # boundary comes from a heatmap blob rather than from the paste edge. A couple of
    # pixels of misalignment on a textured surface inflates the difference enormously:
    # on the wall it read 42.6 against a control of 50.8, a ratio of 1.2, which would
    # have been published as proof of nothing.
    if displacement is not None and min(ch, cw) > 48:
        pad = 6
        inner_a = a[pad:ch - pad, pad:cw - pad].astype(np.float32)
        best, best_off = None, (0, 0)
        for oy in range(-pad, pad + 1):
            for ox in range(-pad, pad + 1):
                shifted = b[pad + oy:ch - pad + oy, pad + ox:cw - pad + ox]
                if shifted.shape != inner_a.shape:
                    continue
                d = float(np.abs(inner_a - shifted.astype(np.float32)).mean())
                if best is None or d < best:
                    best, best_off = d, (ox, oy)
        if best is not None:
            ox, oy = best_off
            a = a[pad:ch - pad, pad:cw - pad]
            b = b[pad + oy:ch - pad + oy, pad + ox:cw - pad + ox]
            ch, cw = a.shape[0], a.shape[1]

    pair_diff = float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())

    # The control: a patch the same size taken from elsewhere in the frame, as far
    # from the flagged region as the image allows. Anchored on the largest region
    # rather than on whichever branch produced the pair, so both paths agree.
    anchor = max(regions, key=lambda r: r.get("area_px", 0))["bbox"]
    cx = 0 if int(anchor[0]) > w // 2 else max(w - cw, 0)
    cy = min(int(anchor[1]), max(h - ch, 0))
    control = base[cy:cy + ch, cx:cx + cw]
    control_diff = (
        float(np.abs(a.astype(np.float32) - control.astype(np.float32)).mean())
        if control.shape == a.shape
        else float("nan")
    )

    left = Image.fromarray(a).resize((cw * zoom, ch * zoom), Image.NEAREST)
    right = Image.fromarray(b).resize((cw * zoom, ch * zoom), Image.NEAREST)
    canvas = Image.new("RGB", (cw * zoom * 2 + gap, ch * zoom), (245, 246, 245))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (cw * zoom + gap, 0))

    ratio = round(control_diff / pair_diff, 1) if pair_diff > 0.01 else None

    # Refuse to publish weak evidence. This panel exists to let a person verify the
    # finding by looking, and two crops that are only marginally more alike than two
    # unrelated patches demonstrate nothing. Below 3x the honest move is to show no
    # proof panel at all rather than a number that reads as one.
    if ratio is None or ratio < 3.0:
        return None

    return np.asarray(canvas), {
        "pair_difference": round(pair_diff, 2),
        "control_difference": round(control_diff, 2),
        "ratio": ratio,
    }


def render_verdict(
    image_path: Path,
    verdict: Verdict,
    out_path: Path,
    box_threshold: float = 0.5,
) -> Path | None:
    """Write a three-panel PNG. Returns None if no detector localised anything."""
    if verdict.heatmap is None:
        return None

    with Image.open(image_path) as im:
        base = np.asarray(im.convert("RGB"))

    heat = verdict.heatmap
    panels = [base, overlay(base, heat), colourise(heat)]

    h, w = base.shape[:2]
    gap = 8
    canvas = Image.new("RGB", (w * 3 + gap * 2, h), (18, 18, 20))
    for i, panel in enumerate(panels):
        canvas.paste(Image.fromarray(panel), (i * (w + gap), 0))

    # Outline the regions worth looking at, on the overlay panel.
    draw = ImageDraw.Draw(canvas)
    for region in peak_regions(heat, threshold=box_threshold)[:6]:
        x0, y0, x1, y1 = region["bbox"]
        draw.rectangle(
            [x0 + w + gap, y0, x1 + w + gap, y1], outline=(80, 220, 255), width=3
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


_LABEL_BG = (18, 18, 20)
_LABEL_FG = (235, 235, 240)
_BAR = 26


def _panel_grid(
    panels: list[tuple[str, np.ndarray]], gap: int = 8
) -> tuple[Image.Image, int, int]:
    """Lay panels out horizontally with a caption bar under each."""
    h, w = panels[0][1].shape[:2]
    canvas = Image.new(
        "RGB", (w * len(panels) + gap * (len(panels) - 1), h + _BAR), _LABEL_BG
    )
    draw = ImageDraw.Draw(canvas)
    for i, (label, arr) in enumerate(panels):
        x = i * (w + gap)
        canvas.paste(Image.fromarray(arr.astype(np.uint8)), (x, 0))
        draw.text((x + 6, h + 6), label, fill=_LABEL_FG)
    return canvas, w, gap


def render_reconstruction(reconstruction, out_path: Path) -> Path:
    """Four panels: recovered original, current image, what changed, and where.

    The caption states the fidelity explicitly. Someone looking at this needs to
    know at a glance whether the left panel is real recovered pixels or a model's
    guess -- presenting the two identically would be the single most misleading
    thing this tool could do.
    """
    r = reconstruction
    diff = r.difference

    panels = [
        (f"BEFORE ({r.fidelity.value}, {r.preview_size[0]}x{r.preview_size[1]})", r.before),
        ("AFTER (current file)", r.after),
        ("CHANGED", colourise(diff)),
        ("REGIONS", overlay(r.after, diff)),
    ]
    canvas, w, gap = _panel_grid(panels)

    draw = ImageDraw.Draw(canvas)
    for region in r.regions[:6]:
        x0, y0, x1, y1 = region["bbox"]
        off = 3 * (w + gap)
        draw.rectangle([x0 + off, y0, x1 + off, y1], outline=(80, 220, 255), width=3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path
