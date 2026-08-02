"""Extract files from a truncated ZIP by walking local file headers.

A normal ZIP reader starts at the central directory, which lives at the END of the
archive -- so a download that stops early is unreadable even though most of its
contents arrived intact. Every member is also preceded by its own local header,
so the archive can be walked forwards instead, recovering everything up to the
truncation point.

Only members whose declared length is fully present are written out; a partially
received file is dropped rather than silently emitted as a corrupt image.
"""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

_LFH = b"PK\x03\x04"
_HEADER = struct.Struct("<HHHHHIIIHH")  # after the 4-byte signature


def salvage(archive: Path, out_dir: Path, name_filter: str = "") -> tuple[int, int]:
    raw = archive.read_bytes()
    out_dir.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    pos = raw.find(_LFH)
    while pos != -1 and pos + 4 + _HEADER.size <= len(raw):
        (
            _ver, flags, method, _t, _d, _crc, comp_size, _uncomp, name_len, extra_len,
        ) = _HEADER.unpack_from(raw, pos + 4)

        start = pos + 4 + _HEADER.size
        name = raw[start : start + name_len].decode("utf-8", "replace")
        data_start = start + name_len + extra_len

        # Bit 3 means the sizes live in a trailing data descriptor, so the header
        # says zero. Recovering those needs the next signature; skip them rather
        # than guess at a boundary.
        if flags & 0x08 or comp_size == 0:
            skipped += 1
            pos = raw.find(_LFH, data_start)
            continue

        data_end = data_start + comp_size
        if data_end > len(raw):
            skipped += 1  # truncated mid-member: this is where the download died
            break

        if not name.endswith("/") and (not name_filter or name_filter in name):
            blob = raw[data_start:data_end]
            if method == 8:
                try:
                    blob = zlib.decompress(blob, -15)
                except zlib.error:
                    skipped += 1
                    pos = raw.find(_LFH, data_end)
                    continue
            elif method != 0:
                skipped += 1
                pos = raw.find(_LFH, data_end)
                continue

            target = out_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
            written += 1

        pos = raw.find(_LFH, data_end)

    return written, skipped


if __name__ == "__main__":
    archive = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    name_filter = sys.argv[3] if len(sys.argv) > 3 else ""
    w, s = salvage(archive, out_dir, name_filter)
    print(f"recovered {w} files, skipped {s}")
