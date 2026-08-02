"""Extract embedded previews -- photographs of the image before it was edited.

Containers routinely carry a second, smaller copy of the image: an EXIF thumbnail
in a JPEG, a thumbnail item in a HEIC, a reduced-resolution SubIFD in a TIFF, and
in camera RAW files an embedded JPEG that is frequently FULL RESOLUTION.

The forensic value is that many editors rewrite the main image and leave the
preview untouched. When that happens the preview is not an estimate of the
original -- it *is* the original, at whatever resolution the container stored it.
Nothing is being guessed.

Recovery strategy, in order of reliability:

1. Container-specific extraction (EXIF IFD1, HEIF thumbnail items, TIFF SubIFDs)
2. A generic scan for embedded JPEG streams, which catches previews in containers
   we do not parse explicitly -- including most RAW formats, since nearly all of
   them are TIFF derivatives with a JPEG preview bolted in.

Candidates from the generic scan are validated by actually decoding them; the
byte pattern that starts a JPEG occurs by chance inside compressed data often
enough that pattern-matching alone produces garbage.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from ..core.image_io import HEIF_AVAILABLE, Container, sniff

log = logging.getLogger(__name__)

_SOI = b"\xff\xd8\xff"
_EOI = b"\xff\xd9"

# Previews live in headers. Scanning an entire multi-megabyte file for candidate
# streams is wasted work, and anything found deep in the entropy-coded data is
# far more likely to be a coincidence than a preview.
_SCAN_LIMIT_BYTES = 8 * 1024 * 1024

_MAX_CANDIDATES = 24
_MIN_PREVIEW_PX = 24


@dataclass
class Preview:
    """One embedded copy of the image found inside the container."""

    image: Image.Image = field(repr=False)
    source: str
    width: int
    height: int
    byte_offset: int | None = None

    @property
    def pixels(self) -> int:
        return self.width * self.height

    def as_rgb(self) -> np.ndarray:
        return np.asarray(self.image.convert("RGB"))


def _try_decode(blob: bytes, source: str, offset: int | None) -> Preview | None:
    try:
        img = Image.open(io.BytesIO(blob))
        img.load()
    except Exception:
        return None
    if img.width < _MIN_PREVIEW_PX or img.height < _MIN_PREVIEW_PX:
        return None
    return Preview(
        image=img, source=source, width=img.width, height=img.height, byte_offset=offset
    )


def _scan_embedded_jpegs(raw: bytes, skip_offset_zero: bool) -> list[Preview]:
    """Find and validate JPEG streams inside arbitrary container bytes."""
    found: list[Preview] = []
    window = raw[:_SCAN_LIMIT_BYTES]
    start = 3 if skip_offset_zero else 0  # a JPEG's own SOI sits at byte 0

    while len(found) < _MAX_CANDIDATES:
        soi = window.find(_SOI, start)
        if soi == -1:
            break
        eoi = window.find(_EOI, soi + 3)
        if eoi == -1:
            break
        preview = _try_decode(
            window[soi : eoi + 2], f"embedded_jpeg@0x{soi:x}", soi
        )
        if preview is not None:
            found.append(preview)
            start = eoi + 2
        else:
            start = soi + 3
    return found


def _heif_thumbnails(path: Path) -> list[Preview]:
    if not HEIF_AVAILABLE:
        return []
    try:
        import pillow_heif

        heif = pillow_heif.open_heif(str(path), convert_hdr_to_8bit=True)
    except Exception:
        log.debug("HEIF thumbnail extraction failed for %s", path, exc_info=True)
        return []

    out: list[Preview] = []
    for thumb in getattr(heif, "thumbnails", []) or []:
        try:
            img = Image.frombytes(thumb.mode, thumb.size, thumb.data, "raw")
        except Exception:
            log.debug("undecodable HEIF thumbnail in %s", path, exc_info=True)
            continue
        if min(img.size) >= _MIN_PREVIEW_PX:
            out.append(
                Preview(
                    image=img,
                    source="heif_thumbnail",
                    width=img.width,
                    height=img.height,
                )
            )
    return out


def _tiff_subifds(path: Path) -> list[Preview]:
    out: list[Preview] = []
    try:
        with Image.open(path) as im:
            n = getattr(im, "n_frames", 1)
            for i in range(1, min(n, 6)):  # frame 0 is the primary image
                im.seek(i)
                copy = im.copy()
                if min(copy.size) >= _MIN_PREVIEW_PX:
                    out.append(
                        Preview(
                            image=copy,
                            source=f"tiff_subifd[{i}]",
                            width=copy.width,
                            height=copy.height,
                        )
                    )
    except Exception:
        log.debug("TIFF SubIFD extraction failed for %s", path, exc_info=True)
    return out


def extract_previews(path: Path) -> list[Preview]:
    """Every embedded preview found, largest first.

    Largest-first matters: a RAW file may carry both a 160x120 thumbnail and a
    full-resolution JPEG preview, and the full-resolution one is worth far more
    when the question is what the image looked like before it was edited.
    """
    container = sniff(path)
    raw = path.read_bytes()

    previews: list[Preview] = []
    if container is Container.HEIF:
        previews.extend(_heif_thumbnails(path))
    elif container is Container.TIFF:
        previews.extend(_tiff_subifds(path))

    previews.extend(
        _scan_embedded_jpegs(raw, skip_offset_zero=container is Container.JPEG)
    )

    # Deduplicate by decoded size -- the same preview is frequently reachable via
    # more than one route, and reporting it twice would overstate the evidence.
    seen: set[tuple[int, int]] = set()
    unique: list[Preview] = []
    for p in sorted(previews, key=lambda p: p.pixels, reverse=True):
        key = (p.width, p.height)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique
