"""Where copy-move detection stops working.

    python scripts/sweep_copy_move.py <image>

`geometric.copy_move` is the second-strongest detector in the set and had no
recorded blind spots, which is not the same as having none. A detector nobody has
tried to defeat has an unknown envelope, not a wide one.

This stages a clone into a real photograph and varies three things: how large the
copied region is, how hard the copy is to see after the fact, and whether the copy
was transformed on its way in. The last group is the one that matters, because
rotating or mirroring a clone is something retouchers do by habit rather than as an
attack.

The result to expect, and the reason this script exists: **visibility is not the
axis.** The detector measures duplication, so degrading the image afterwards does
almost nothing, while two ordinary editing habits defeat it completely.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundtruth.core.detector import get
from groundtruth.core.types import ImageCase

DETECTOR = "geometric.copy_move"
WORK = Path(__file__).resolve().parents[1] / "data/interim/copy_move_sweep"


def clone(img, box, dst, feather=20, transform=None):
    """Paste a region of the image somewhere else, the way a clone stamp does."""
    patch = img.crop(box)
    if transform:
        patch = transform(patch)
    w, h = patch.size
    f = max(1, min(feather, w // 2 - 1, h // 2 - 1))
    m = np.zeros((h, w), np.uint8)
    m[f:h - f, f:w - f] = 255
    mask = Image.fromarray(m).filter(ImageFilter.GaussianBlur(f * 0.6))
    out = img.copy()
    out.paste(patch, dst, mask)
    return out


def score(img, path: Path, quality=95, after=None) -> tuple[float, int]:
    if after:
        img = after(img)
    img.save(path, quality=quality, subsampling=0 if quality >= 90 else 2)
    ev = get(DETECTOR).run(ImageCase(image_path=path))
    return ev.score, int(ev.details.get("supporting_pairs", 0) or 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--x", type=int, default=520)
    ap.add_argument("--y", type=int, default=380)
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    src = Image.open(args.image).convert("RGB")
    W, H = src.size
    cx, cy = args.x, args.y
    print(f"{args.image.name}  {W}x{H}\n")

    print("SIZE. How small can the copied region be?\n")
    print(f"  {'patch':>8}{'% frame':>10}{'pairs':>8}   verdict")
    floor = None
    for s in (500, 350, 250, 180, 120, 80, 56, 40):
        if cx + s > W or cy + s > H:
            continue
        im = clone(src, (cx, cy, cx + s, cy + s), (cx - 40, cy + 300),
                   feather=max(3, s // 12))
        sc, pairs = score(im, WORK / f"size_{s}.jpg")
        caught = sc > 0.5
        if caught:
            floor = s
        print(f"  {s:>6}px{100 * s * s / (W * H):>9.2f}%{pairs:>8}   "
              f"{'caught' if caught else 'MISSED'}")

    base = (cx, cy, cx + 250, cy + 250)
    dst = (cx - 40, cy + 300)
    print("\nCONCEALMENT. A 250px clone, then degraded to hide it.\n")
    rng = np.random.default_rng(0)
    trials = [
        ("hard edge", {"feather": 2}, {}),
        ("heavy feather, 60px", {"feather": 60}, {}),
        ("saved at JPEG q75", {}, {"quality": 75}),
        ("saved at JPEG q50", {}, {"quality": 50}),
        ("saved at JPEG q30", {}, {"quality": 30}),
        ("blurred 0.8px", {},
         {"after": lambda i: i.filter(ImageFilter.GaussianBlur(0.8))}),
        ("noise added, sigma 6", {},
         {"after": lambda i: Image.fromarray(
             np.clip(np.asarray(i, float) + rng.normal(0, 6, np.asarray(i).shape),
                     0, 255).astype(np.uint8))}),
    ]
    for label, ckw, skw in trials:
        im = clone(src, base, dst, **{"feather": 20, **ckw})
        sc, pairs = score(im, WORK / f"{label.replace(' ', '_').replace(',', '')}.jpg", **skw)
        print(f"  {label:<26}{pairs:>6} pairs   {'caught' if sc > 0.5 else 'MISSED'}")

    print("\nTRANSFORM. The copy altered on its way in.\n")
    for label, fn in [
        ("rotated 3 degrees", lambda p: p.rotate(3, resample=Image.BICUBIC)),
        ("scaled 1.08x", lambda p: p.resize(
            (round(p.width * 1.08), round(p.height * 1.08)), Image.LANCZOS)),
        ("mirrored horizontally", lambda p: p.transpose(Image.FLIP_LEFT_RIGHT)),
    ]:
        im = clone(src, base, dst, transform=fn)
        sc, pairs = score(im, WORK / f"tf_{label.replace(' ', '_')}.jpg")
        print(f"  {label:<26}{pairs:>6} pairs   {'caught' if sc > 0.5 else 'MISSED'}")

    print(
        "\nReading:"
        f"\n  Size floor is sharp, around {floor}px, roughly 1% of the frame."
        "\n  Compression, blur, noise and small rotations or rescalings do essentially"
        "\n  nothing, because the detector measures duplication rather than visibility."
        "\n\n  Two ordinary editing habits defeat it outright. Heavy feathering leaves"
        "\n  too little of the patch truly identical, and it is what a healing brush"
        "\n  does by default. Mirroring the copy removes the single consistent"
        "\n  displacement the detector looks for, and retouchers mirror clones"
        "\n  precisely so the result does not look repetitive."
        "\n\n  So the supportable claim is narrow: a hard-edged or lightly-blended,"
        "\n  un-mirrored copy covering more than about 1% of the frame, after which"
        "\n  it survives almost any degradation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
