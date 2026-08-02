"""Container identification and decoding.

Format is determined by **magic bytes, never by file extension.** Extensions are
metadata a user controls; the container is what the bytes actually are. The
disagreement between the two is itself a forensic signal -- a file named
``damage.jpg`` whose bytes are a PNG has been through a re-save that the claimant
did not mention.

HEIC matters more than it looks: it is the default on every iPhone since 2017, so
in a claims pipeline it is not an edge case, it is the common case.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

try:  # optional -- HEIC support ships separately from Pillow
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    HEIF_AVAILABLE = False
    log.info("pillow-heif not installed; HEIC/HEIF input will not decode")


class Container(str, Enum):
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    TIFF = "tiff"
    HEIF = "heif"
    GIF = "gif"
    BMP = "bmp"
    UNKNOWN = "unknown"

    @property
    def lossy(self) -> bool:
        """Whether the container discards information on save.

        Compression forensics only mean something on lossy containers -- there is
        no compression history to read out of a PNG.
        """
        return self in {Container.JPEG, Container.WEBP, Container.HEIF}

    @property
    def block_compressed(self) -> bool:
        """Whether the codec quantises on a fixed block grid.

        JPEG's rigid 8x8 grid is what makes block-artefact analysis possible.
        """
        return self in {Container.JPEG, Container.HEIF}


_MAGIC: list[tuple[bytes, Container]] = [
    (b"\xff\xd8\xff", Container.JPEG),
    (b"\x89PNG\r\n\x1a\n", Container.PNG),
    (b"GIF87a", Container.GIF),
    (b"GIF89a", Container.GIF),
    (b"II*\x00", Container.TIFF),
    (b"MM\x00*", Container.TIFF),
    (b"BM", Container.BMP),
]

# HEIF/AVIF brands appear at offset 8, after the box length and 'ftyp'.
_FTYP_BRANDS = {
    b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis",
    b"mif1", b"msf1", b"avif", b"avis",
}

_EXT_TO_CONTAINER = {
    ".jpg": Container.JPEG, ".jpeg": Container.JPEG, ".jpe": Container.JPEG,
    ".png": Container.PNG,
    ".webp": Container.WEBP,
    ".tif": Container.TIFF, ".tiff": Container.TIFF,
    ".heic": Container.HEIF, ".heif": Container.HEIF, ".avif": Container.HEIF,
    ".gif": Container.GIF,
    ".bmp": Container.BMP,
}


def sniff(path: Path) -> Container:
    """Identify the container from its leading bytes."""
    with path.open("rb") as fh:
        head = fh.read(32)

    for magic, container in _MAGIC:
        if head.startswith(magic):
            return container

    if len(head) >= 12 and head[4:8] == b"ftyp" and head[8:12] in _FTYP_BRANDS:
        return Container.HEIF
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return Container.WEBP

    return Container.UNKNOWN


def container_from_extension(path: Path) -> Container:
    return _EXT_TO_CONTAINER.get(path.suffix.lower(), Container.UNKNOWN)


@dataclass(frozen=True)
class ContainerInfo:
    actual: Container
    claimed: Container
    """What the file extension asserts. May disagree with `actual`."""

    decodable: bool
    note: str = ""

    @property
    def extension_mismatch(self) -> bool:
        return (
            self.claimed is not Container.UNKNOWN
            and self.actual is not Container.UNKNOWN
            and self.claimed != self.actual
        )


def inspect(path: Path) -> ContainerInfo:
    actual = sniff(path)
    claimed = container_from_extension(path)

    decodable, note = True, ""
    if actual is Container.HEIF and not HEIF_AVAILABLE:
        decodable, note = False, "HEIC/HEIF input requires pillow-heif"
    elif actual is Container.UNKNOWN:
        decodable, note = False, "unrecognised container"

    return ContainerInfo(actual=actual, claimed=claimed, decodable=decodable, note=note)


def load_rgb(path: Path) -> np.ndarray:
    """Decode to an HxWx3 uint8 array, whatever the container.

    EXIF orientation is deliberately NOT applied. Auto-rotating would silently
    resample the pixel grid, destroying the block alignment and noise structure
    that the compression and sensor detectors depend on.
    """
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))
