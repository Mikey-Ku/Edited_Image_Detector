"""Run the detection-envelope benchmark over real photographs.

    python scripts/benchmark.py            # operations x size
    python scripts/benchmark.py laundering # operations x laundering
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image

from groundtruth.benchmark import by_detector, envelope, run

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
KORUS = ROOT / "data/interim/korus/data-images"
N_BASES = 4


def bases() -> list[Path]:
    found: list[Path] = []
    for cam in ("Nikon_D7000", "Nikon_D90"):
        found += sorted((KORUS / cam / "pristine").glob("*.TIF"))[: N_BASES // 2]
    return found


def main() -> int:
    photos = bases()
    if not photos:
        print("no base photographs -- run scripts/salvage_zip.py first", file=sys.stderr)
        return 1

    axis = sys.argv[1] if len(sys.argv) > 1 else "size"
    kwargs = (
        {"launderings": ("none", "jpeg95", "jpeg85", "jpeg75", "downscale", "png"),
         "sizes": (0.08,)}
        if axis == "laundering"
        else {}
    )

    print(f"{len(photos)} real base photographs, axis = {axis}\n", flush=True)
    started = time.time()
    results = run(photos, ROOT / "data/interim/benchmark", **kwargs)
    print(envelope(results, by="laundering" if axis == "laundering" else "size"))
    print("\n\nWHICH DETECTOR FIRES ON WHAT\n")
    print(by_detector(results))
    print(f"\n{len(results)} cells in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
