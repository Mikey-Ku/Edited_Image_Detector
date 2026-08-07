"""Build the bundled claim-photo samples from their originals, reproducibly.

    python scripts/make_samples.py

The demo images are staged forgeries, and a staged forgery committed as an opaque
binary is exactly the kind of artefact nobody can audit. This script is the recipe:
it takes the untouched source photographs in `samples/` and produces the edited
versions beside them, so anyone can see precisely what was changed and re-derive it.

**These are demonstrations, never measurements.** They exist to show what a finding
looks like. Every accuracy figure in this project comes from the Korus dataset, which
is 112 forgeries made by other people who had never heard of this system. Scoring
edits I staged myself would be meaningless, because I chose them.

Two frauds, both realistic in a claims context and both a straight clone stamp,
mechanically identical to what Photoshop's clone tool does:

1. **Damage extended.** Copy a crumpled section of a car door onto a panel that was
   never hit, so the claim covers more than the accident did.
2. **Damage exaggerated, with the original left behind.** Duplicate part of a
   structural crack so subsidence looks worse, and leave the file's embedded
   thumbnail showing the pre-edit original, which is the mistake editors actually
   make.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def clone(img: Image.Image, box, dst, feather: int = 20) -> Image.Image:
    """Copy a region and blend it in with a soft edge, as a clone stamp does."""
    patch = img.crop(box)
    w, h = patch.size
    f = max(1, min(feather, w // 2 - 1, h // 2 - 1))
    mask = np.zeros((h, w), np.uint8)
    mask[f:h - f, f:w - f] = 255
    soft = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(f * 0.6))
    out = img.copy()
    out.paste(patch, dst, soft)
    return out


def with_stale_preview(original: Image.Image, edited: Image.Image, out: Path) -> None:
    """Write `edited`, but with a thumbnail generated from `original`.

    This reproduces the single most useful mistake in image forensics. Editors
    rewrite the main image and leave the embedded preview alone, so the file ends up
    carrying a photograph of its own pre-edit state.
    """
    import piexif

    thumb = original.copy()
    thumb.thumbnail((320, 320))
    tb = io.BytesIO()
    thumb.save(tb, "JPEG", quality=70)

    mb = io.BytesIO()
    edited.save(mb, "JPEG", quality=92, subsampling=0)

    exif = {
        "0th": {piexif.ImageIFD.Software: b"Adobe Photoshop 26.0"},
        "Exif": {},
        "GPS": {},
        "1st": {},
        "thumbnail": tb.getvalue(),
    }
    piexif.insert(piexif.dump(exif), mb.getvalue(), str(out))


def main() -> int:
    car_src = SAMPLES / "claim_car_original.jpg"
    brick_src = SAMPLES / "claim_wall_original.jpg"
    for p in (car_src, brick_src):
        if not p.is_file():
            print(f"missing source photograph: {p}", file=sys.stderr)
            return 1

    # 1. Car: extend the damage onto an undamaged rear door.
    car = Image.open(car_src).convert("RGB")
    edited = clone(car, (300, 440, 800, 800), (1050, 460))
    edited.save(SAMPLES / "claim_car_damage_extended.jpg", quality=92, subsampling=0)
    print("wrote claim_car_damage_extended.jpg")

    # 2. Wall: duplicate the crack, and leave the original in the thumbnail.
    brick = Image.open(brick_src).convert("RGB")
    worse = clone(brick, (430, 250, 780, 700), (400, 780), feather=16)
    with_stale_preview(brick, worse, SAMPLES / "claim_wall_crack_duplicated.jpg")
    print("wrote claim_wall_crack_duplicated.jpg  (with a stale preview)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
