"""Does "carries no camera fingerprint" survive the trip through a real claims inbox?

    python scripts/sweep_provenance.py [--n 24]

`MIN_DEMOSAIC_STRUCTURE = 6.0` already separates camera originals from synthetic
images by a wide margin, and both fingerprint detectors abstain below it. That
abstention is currently silent, which throws away a fact a claims pipeline would want:
a photograph submitted as proof of damage that carries no sensor fingerprint at all is
a screenshot, a render, a heavily re-encoded forward, or a generated image. None of
those are what the claimant said they were sending.

Before that silence can become a *finding*, the false-positive cost has to be known,
because real photographs do not arrive pristine. They arrive through email, through
WhatsApp, resized by a phone, screenshotted out of a PDF. If ordinary degradation
pushes an honest photograph below the threshold, the finding accuses the innocent and
must not ship.

So this measures frame-level period-2 structure across the laundering the DESIGN
robustness sweep calls for: JPEG re-encode at q in {95, 75, 50}, downscale, and a
screenshot-style round trip. The question is not "is the threshold good" but "how much
abuse does an authentic photograph absorb before it looks like it never came from a
camera".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundtruth.detectors.sensor._fingerprint import (
    MAX_CLIPPED_FRACTION,
    MIN_BLOCKS,
    MIN_DEMOSAIC_STRUCTURE,
    clipped_share,
    energy,
    period2,
    residual_for,
    robust_z,
)
from groundtruth.detectors.sensor.noiseprint_structure import BLOCK
from groundtruth.learned.noiseprint import quality_factor

ROOT = Path(__file__).resolve().parents[1]
KORUS = ROOT / "data/interim/korus/data-images"
WORK = ROOT / "data/interim/provenance_sweep"


def structure_of(path: Path) -> float | None:
    """Frame-level period-2 structure, or None when too few blocks are readable.

    Identical to the computation both fingerprint detectors gate on, imported rather
    than reimplemented so this cannot drift away from what actually ships.
    """
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    gray = np.asarray(img, dtype=np.float32).mean(axis=2) / 255.0
    if min(gray.shape) < BLOCK * 6:
        return None
    residual = residual_for(path, quality_factor(path))
    usable = (energy(residual, BLOCK) > 1e-6) & (
        clipped_share(gray, BLOCK) <= MAX_CLIPPED_FRACTION
    )
    if int(usable.sum()) < MIN_BLOCKS:
        return None
    _, centre, _ = robust_z(period2(residual, BLOCK), usable)
    return float(centre)


def launder(src: Path, out_dir: Path) -> dict[str, Path]:
    """The ways an honest photograph gets mangled before it reaches an adjuster."""
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(src).convert("RGB")
    w, h = img.size
    made: dict[str, Path] = {}

    for q in (95, 75, 50):
        p = out_dir / f"{src.stem}_q{q}.jpg"
        img.save(p, quality=q, subsampling=0)
        made[f"jpeg q{q}"] = p

    # Messaging apps resize aggressively before re-encoding.
    p = out_dir / f"{src.stem}_resized.jpg"
    img.resize((w // 2, h // 2), Image.LANCZOS).save(p, quality=80)
    made["half size, q80"] = p

    # A screenshot resamples the whole frame and re-encodes it losslessly.
    p = out_dir / f"{src.stem}_screenshot.png"
    img.resize((int(w * 0.8), int(h * 0.8)), Image.BICUBIC).save(p)
    made["screenshot-style"] = p

    # The shape of a model-regenerated image: every pixel resynthesised, so per-pixel
    # sensor structure is gone everywhere at once rather than in one region.
    p = out_dir / f"{src.stem}_regen.jpg"
    img.resize((w // 2, h // 2), Image.LANCZOS).resize((w, h), Image.LANCZOS).save(
        p, quality=92, subsampling=0
    )
    made["whole-frame resynthesis"] = p

    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16, help="camera originals to sweep")
    args = ap.parse_args()

    if not KORUS.is_dir():
        raise SystemExit(f"needs the Korus photographs at {KORUS}")

    originals = sorted((KORUS / "Nikon_D7000" / "pristine").glob("*.TIF"))[: args.n]
    print(f"{len(originals)} camera originals, threshold = {MIN_DEMOSAIC_STRUCTURE}\n")

    rows: dict[str, list[float]] = {"camera original": []}
    for src in originals:
        s = structure_of(src)
        if s is not None:
            rows["camera original"].append(s)
        for label, path in launder(src, WORK).items():
            v = structure_of(path)
            if v is not None:
                rows.setdefault(label, []).append(v)

    print(f"{'condition':<26}{'n':>4}{'median':>9}{'min':>9}{'max':>9}"
          f"{'below thr':>11}")
    print("-" * 68)
    order = ["camera original", "jpeg q95", "jpeg q75", "jpeg q50",
             "half size, q80", "screenshot-style", "whole-frame resynthesis"]
    summary = {}
    for label in order:
        v = np.array(rows.get(label, []))
        if not len(v):
            continue
        below = float((v < MIN_DEMOSAIC_STRUCTURE).mean())
        summary[label] = below
        print(f"{label:<26}{len(v):>4}{np.median(v):>9.2f}{v.min():>9.2f}"
              f"{v.max():>9.2f}{below * 100:>10.0f}%")

    print("\nReading:")
    fp = max(summary.get(k, 0.0) for k in
             ("jpeg q95", "jpeg q75", "jpeg q50", "half size, q80"))
    print(f"  Worst false-positive rate across ordinary re-encoding and resizing: "
          f"{fp * 100:.0f}%.")
    print(f"  Screenshot-style round trip falls below the threshold "
          f"{summary.get('screenshot-style', 0) * 100:.0f}% of the time.")
    print(f"  Whole-frame resynthesis, the shape of a generated image, falls below "
          f"{summary.get('whole-frame resynthesis', 0) * 100:.0f}% of the time.")
    print("\n  A finding built on this can only be as strong as the gap between the"
          "\n  last two rows and the first four. Where honest laundering and synthesis"
          "\n  overlap, the honest report is 'this file is not camera-original', which"
          "\n  routes to a human, and never 'this image was edited'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
