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


def _dims(shape: tuple[int, int], fraction: float) -> tuple[int, int]:
    h, w = shape
    area = fraction * h * w
    bw = int(np.sqrt(area * 1.33))
    bh = int(area / max(bw, 1))
    return max(8, min(bw, w - 8)), max(8, min(bh, h - 8))


def _box_for(
    shape: tuple[int, int], fraction: float, rng, content: np.ndarray | None = None
) -> Box:
    """A box covering ``fraction`` of the frame, placed where there is content.

    Placement must be content-aware or the benchmark manufactures manipulations
    nobody would make. Uniform random placement put an 8% duplication into empty
    sky -- a region containing zero SIFT keypoints, so copy-move detection was
    being scored on a case that is undetectable in principle and pointless in
    practice. An attacker edits the damage, the plate, the face. Not the sky.

    Candidate boxes are scored by the texture they contain and the busiest of a
    random sample wins, which keeps placement varied without letting it drift into
    featureless regions.
    """
    h, w = shape
    bw, bh = _dims(shape, fraction)
    hi_x, hi_y = max(5, w - bw - 4), max(5, h - bh - 4)

    if content is None:
        return (x := int(rng.integers(4, hi_x))), (y := int(rng.integers(4, hi_y))), x + bw, y + bh

    best, best_score = None, -1.0
    for _ in range(24):
        x0 = int(rng.integers(4, hi_x))
        y0 = int(rng.integers(4, hi_y))
        score = float(content[y0 : y0 + bh, x0 : x0 + bw].mean())
        if score > best_score:
            best, best_score = (x0, y0, x0 + bw, y0 + bh), score
    return best  # type: ignore[return-value]


def _content_map(base: np.ndarray) -> np.ndarray:
    """Local detail energy -- a proxy for "is there anything here to edit"."""
    from scipy.ndimage import gaussian_filter

    gray = base.astype(np.float32).mean(axis=2) / 255.0
    return np.abs(gray - gaussian_filter(gray, 2.0))


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


def _apply(
    base: np.ndarray, op: str, box: Box, rng, donor: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
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

    # splice_in must come from a DIFFERENT photograph -- that is the whole point of
    # the case. Taking it from this one turns it into a copy-move, which is a
    # different detection problem entirely and made the keypoint detector look like
    # it was solving splicing when it was solving duplication.
    source = base
    if op == "splice_in":
        if donor is None:
            return out, mask
        source = donor

    sh, sw = source.shape[:2]
    if sh < bh or sw < bw:
        return out, mask

    src = _box_for((sh, sw), (bw * bh) / (sh * sw), rng, _content_map(source))
    sx0 = min(src[0], sw - bw)
    sy0 = min(src[1], sh - bh)
    out[y0:y1, x0:x1] = source[sy0 : sy0 + bh, sx0 : sx0 + bw]
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
    donor_path: Path | None = None,
) -> Fixture:
    """Build one benchmark cell from a real photograph.

    ``donor_path`` supplies the foreign content for ``splice_in`` and is required
    for it; every other operation works from the base image alone.
    """
    rng = np.random.default_rng(seed)
    with Image.open(base_path) as im:
        base = np.asarray(im.convert("RGB"))

    donor = None
    if operation == "splice_in" and donor_path is not None:
        with Image.open(donor_path) as im:
            donor = np.asarray(im.convert("RGB"))

    box = _box_for(base.shape[:2], max(size, 1e-4), rng, _content_map(base))
    edited, mask = _apply(base, operation, box, rng, donor)

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
