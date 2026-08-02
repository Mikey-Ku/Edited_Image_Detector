"""Embedded thumbnail mismatch.

JPEGs carry a small preview image inside the EXIF block. Many editors --
including some very well-known ones -- modify the main image and never
regenerate that thumbnail.

When that happens you are handed a photograph of the *original*. Downscale the
main image, diff it against the stale thumbnail, and the edited region lights up.

It is the cheapest localisation in the entire system and it requires no model at
all. It fails silently against any tool that updates the thumbnail correctly,
which is why it is one signal among many rather than the whole product.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from ...core.detector import Detector, register
from ...core.types import Evidence, ImageCase, Tier

_SOI = b"\xff\xd8"
_EOI = b"\xff\xd9"

# Below this, the thumbnail is too small for the difference map to mean anything.
_MIN_THUMB_PIXELS = 32 * 32

# Fraction of a normalised difference map that must be hot before we call it a region.
_HOT = 0.35


def _extract_embedded_thumbnail(raw: bytes) -> Image.Image | None:
    """Pull the EXIF thumbnail out of a JPEG byte stream.

    The thumbnail is itself a complete JPEG embedded inside the APP1 segment, so
    we locate APP1 and take the first nested SOI...EOI pair inside it.
    """
    if not raw.startswith(_SOI):
        return None

    pos = 2
    while pos < len(raw) - 4:
        if raw[pos] != 0xFF:
            pos += 1
            continue
        marker = raw[pos + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue

        seg_len = int.from_bytes(raw[pos + 2 : pos + 4], "big")
        segment = raw[pos + 4 : pos + 2 + seg_len]

        if marker == 0xE1 and segment.startswith(b"Exif\x00\x00"):
            start = segment.find(_SOI, 6)
            if start != -1:
                end = segment.find(_EOI, start)
                if end != -1:
                    try:
                        return Image.open(io.BytesIO(segment[start : end + 2]))
                    except Exception:
                        return None
            return None

        if marker == 0xDA:  # start of scan -- no more metadata segments
            break
        pos += 2 + seg_len
    return None


@register
class ThumbnailMismatchDetector(Detector):
    """Compare the stale EXIF preview against the current image content."""

    id = "metadata.thumbnail_mismatch"
    tier = Tier.METADATA
    localises = True
    cost = 2

    def applies_to(self, case: ImageCase) -> tuple[bool, str]:
        if not case.is_jpeg:
            return False, "embedded thumbnails only exist in JPEG containers"
        return True, ""

    def _run(self, case: ImageCase) -> Evidence:
        raw = case.image_path.read_bytes()
        thumb = _extract_embedded_thumbnail(raw)

        if thumb is None:
            return Evidence.not_applicable(
                self.id, self.tier, "no embedded EXIF thumbnail present"
            )
        if thumb.width * thumb.height < _MIN_THUMB_PIXELS:
            return Evidence.not_applicable(
                self.id, self.tier, f"thumbnail too small to compare ({thumb.size})"
            )

        thumb_rgb = np.asarray(thumb.convert("RGB"), dtype=np.float32) / 255.0

        main = Image.fromarray(case.pixels()).resize(thumb.size, Image.Resampling.LANCZOS)
        main_rgb = np.asarray(main, dtype=np.float32) / 255.0

        # Mean absolute difference per pixel across channels.
        diff = np.abs(main_rgb - thumb_rgb).mean(axis=2)

        # Normalise against the image's own noise floor so that global brightness
        # or re-encoding differences don't masquerade as a localised edit.
        floor = float(np.median(diff))
        spread = float(diff.std()) + 1e-6
        heat = np.clip((diff - floor) / (6.0 * spread), 0.0, 1.0)

        hot_fraction = float((heat > _HOT).mean())
        peak = float(heat.max())

        # A genuine edit is *localised*: a small fraction of the frame, very hot.
        # A re-encode or brightness shift moves the whole frame a little.
        localised = hot_fraction < 0.25 and peak > 0.8

        if localised:
            score = min(0.95, 0.6 + 2.0 * hot_fraction)
            confidence = 0.8
            explanation = (
                f"embedded thumbnail disagrees with the current image over "
                f"{hot_fraction:.1%} of the frame -- the preview appears to show the "
                f"pre-edit original"
            )
        elif hot_fraction > 0.5:
            score = 0.55
            confidence = 0.25
            explanation = (
                "thumbnail differs across most of the frame; consistent with a global "
                "re-encode or exposure change rather than a localised edit"
            )
        else:
            score = 0.25
            confidence = 0.7
            explanation = "embedded thumbnail matches the current image content"

        return Evidence(
            detector_id=self.id,
            tier=self.tier,
            applicable=True,
            score=score,
            confidence=confidence,
            explanation=explanation,
            heatmap=heat,
            details={
                "thumbnail_size": list(thumb.size),
                "hot_fraction": round(hot_fraction, 4),
                "peak": round(peak, 4),
                "localised": localised,
            },
        )
