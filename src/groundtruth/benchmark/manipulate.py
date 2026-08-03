"""Generate manipulations on top of real photographs, with exact masks.

Every fixture is built from a genuine camera image. That is not incidental: the
first version of this project generated synthetic scenes with *uniform* sensor
noise, which happened to be exactly what the noise detector assumed, so the tests
could not fail. They scored 0.99 and the system turned out to be at chance on real
photographs. A benchmark whose images share the detectors' assumptions measures
nothing.

Three independent axes, because a detector can be excellent on one and blind on
another:

**Operation** -- what was done. Addition and removal are mirror images: spliced
content brings foreign statistics, cloned content brings *the same photo's*
statistics and is invisible to anything looking for foreignness.

**Size** -- how much of the frame. A quarter of the image and half a percent are
different detection problems, not the same one at different difficulty.

**Laundering** -- what happened to the file afterwards. Re-saving, downscaling and
format conversion destroy compression and sensor traces in that order, and real
photographs arrive having been through some of it.

Controls are first class. `pristine` and `global_tone` must NOT be flagged --
brightening a dark claim photo is legitimate, and a system that calls it fraud is
worse than useless.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

Box = tuple[int, int, int, int]

OPERATIONS = (
    "pristine",        # control: untouched
    "global_tone",     # control: legitimate exposure/contrast adjustment
    "splice_in",       # foreign content pasted from another photograph
    "render_overlay",  # synthetic graphic drawn on -- the licence-plate case
    "clone_out",       # content covered by a copy from elsewhere in the same photo
    "inpaint_out",     # content covered by smooth fill synthesised from context
    "duplicate",       # a region copied and pasted within the same photo
)

LAUNDERING = (
    "none",        # saved losslessly, metadata intact
    "jpeg95",
    "jpeg85",
    "jpeg75",
    "downscale",   # resized to 70% then saved -- destroys the block grid
    "png",         # converted to a lossless container, all EXIF dropped
)


@dataclass
class Fixture:
    path: Path
    mask: np.ndarray
    operation: str
    size: float
    laundering: str

    @property
    def is_control(self) -> bool:
        return self.operation in {"pristine", "global_tone"}


def _box_for(shape: tuple[int, int], fraction: float, rng) -> Box:
    """A randomly placed box covering ``fraction`` of the frame, 4:3-ish."""
    h, w = shape
    area = fraction * h * w
    bw = int(np.sqrt(area * 1.33))
    bh = int(area / max(bw, 1))
    bw, bh = min(bw, w - 8), min(bh, h - 8)
    x0 = int(rng.integers(4, max(5, w - bw - 4)))
    y0 = int(rng.integers(4, max(5, h - bh - 4)))
    return x0, y0, x0 + bw, y0 + bh


def _render_plate(size: tuple[int, int], rng) -> Image.Image:
    """A synthetic graphic: flat fill, hard edges, crisp text, no sensor noise.

    This is what a replaced licence plate, a pasted price, or an overlaid document
    field actually is -- pixels that were drawn rather than photographed.
    """
    w, h = max(size[0], 8), max(size[1], 8)
    img = Image.new("RGB", (w, h), (238, 240, 244))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w - 1, h - 1], outline=(30, 40, 90), width=max(1, h // 20))
    d.rectangle([0, int(h * 0.78), w - 1, h - 1], fill=(28, 60, 140))
    step = max(6, w // 9)
    for i in range(1, 8):
        x = i * step
        if x < w - 4:
            d.rectangle(
                [x, int(h * 0.22), x + max(2, step // 3), int(h * 0.66)],
                fill=(20, 22, 28),
            )
    return img


def _apply(base: np.ndarray, op: str, box: Box, rng) -> tuple[np.ndarray, np.ndarray]:
    h, w = base.shape[:2]
    out = base.copy()
    mask = np.zeros((h, w), dtype=bool)
    x0, y0, x1, y1 = box
    bh, bw = y1 - y0, x1 - x0
    if bh < 4 or bw < 4:
        return out, mask

    if op == "pristine":
        return out, mask

    if op == "global_tone":
        # A legitimate edit: lift exposure and contrast across the whole frame.
        # Nothing is localised, so the mask stays empty and any flag is a miss.
        f = base.astype(np.float32)
        out = np.clip((f - 128.0) * 1.18 + 128.0 + 14.0, 0, 255).astype(np.uint8)
        return out, mask

    if op == "render_overlay":
        plate = np.asarray(_render_plate((bw, bh), rng))
        out[y0:y1, x0:x1] = plate
        mask[y0:y1, x0:x1] = True
        return out, mask

    if op == "inpaint_out":
        # Approximates content-aware fill: the region is rebuilt from a heavily
        # blurred version of its surroundings, so it is smoother than anything the
        # sensor recorded.
        pad = max(bw, bh) // 2
        r0, r1 = max(0, y0 - pad), min(h, y1 + pad)
        c0, c1 = max(0, x0 - pad), min(w, x1 + pad)
        context = Image.fromarray(base[r0:r1, c0:c1]).filter(
            ImageFilter.GaussianBlur(radius=max(6, min(bw, bh) / 6))
        )
        fill = np.asarray(context)[y0 - r0 : y1 - r0, x0 - c0 : x1 - c0]
        out[y0:y1, x0:x1] = fill
        mask[y0:y1, x0:x1] = True
        return out, mask

    # The remaining operations copy pixels from somewhere. clone_out and duplicate
    # take them from THIS photograph -- same sensor, same lens, same compression --
    # which is why detectors that hunt for foreign statistics cannot see them.
    src = _box_for((h, w), (bw * bh) / (h * w), rng)
    sx0, sy0 = src[0], src[1]
    sx0 = min(sx0, w - bw)
    sy0 = min(sy0, h - bh)
    out[y0:y1, x0:x1] = base[sy0 : sy0 + bh, sx0 : sx0 + bw]
    mask[y0:y1, x0:x1] = True
    return out, mask


def _launder(img: Image.Image, how: str, path: Path) -> Path:
    if how == "none":
        p = path.with_suffix(".png")
        img.save(p)
        return p
    if how == "png":
        p = path.with_suffix(".png")
        # Round-trip through JPEG first, then convert: the common laundering path,
        # and the one that leaves a lossless container with no compression history.
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=92)
        buf.seek(0)
        Image.open(buf).convert("RGB").save(p)
        return p
    if how == "downscale":
        p = path.with_suffix(".jpg")
        small = img.resize(
            (int(img.width * 0.7), int(img.height * 0.7)), Image.Resampling.LANCZOS
        )
        small.save(p, "JPEG", quality=92)
        return p
    quality = int(how.removeprefix("jpeg"))
    p = path.with_suffix(".jpg")
    img.save(p, "JPEG", quality=quality)
    return p


def make(
    base_path: Path,
    out_dir: Path,
    operation: str,
    size: float,
    laundering: str,
    seed: int = 0,
) -> Fixture:
    """Build one benchmark cell from a real photograph."""
    rng = np.random.default_rng(seed)
    with Image.open(base_path) as im:
        base = np.asarray(im.convert("RGB"))

    box = _box_for(base.shape[:2], max(size, 1e-4), rng)
    edited, mask = _apply(base, operation, box, rng)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{operation}_{size:.3f}_{laundering}_{base_path.stem}"
    path = _launder(Image.fromarray(edited), laundering, out_dir / stem)

    # Downscaling changes the frame, so the mask has to follow it or every
    # localisation score silently compares against the wrong geometry.
    if laundering == "downscale":
        with Image.open(path) as im:
            target = im.size
        mask = (
            np.asarray(
                Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    target, Image.Resampling.NEAREST
                )
            )
            > 127
        )

    return Fixture(
        path=path, mask=mask, operation=operation, size=size, laundering=laundering
    )
