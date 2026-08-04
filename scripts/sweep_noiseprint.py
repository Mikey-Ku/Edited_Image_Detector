"""Sweep the block-analysis parameters of the fingerprint detector on real forgeries.

    python scripts/sweep_noiseprint.py --limit 40

The network is the expensive part and it does not depend on any parameter being
swept, so the residual is extracted once per image and cached. Every configuration
after that is cheap array arithmetic. Without this the sweep costs one full
inference pass per cell and stops being run at all.

Measured against the Korus Realistic Tampering Dataset -- human-made forgeries with
pixel-exact masks -- and, critically, against the *matched pristine originals*. A
block configuration that finds more anomalies on tampered images while finding just
as many on their untouched counterparts has learned nothing; that failure is exactly
what sank `sensor.noise_inconsistency`, and only the paired controls reveal it.

Reported per configuration:

- **AUC** of the anomalous-fraction statistic, tampered vs pristine. This is the
  question "does this parameter separate the classes at all", asked without
  reference to any threshold.
- **Localisation hit rate**, over tampered images only. A configuration can
  separate the classes while pointing at the wrong place, which is how
  `geometric.sharpness_inconsistency` survived longer than it deserved.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundtruth.learned.noiseprint import extract, quality_factor

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
KORUS = ROOT / "data/interim/korus/data-images"
CACHE = ROOT / "data/interim/noiseprint_cache"
CAMERAS = ("Nikon_D7000", "Nikon_D90")

BLOCKS = (16, 24, 32, 48, 64)
Z_VALUES = (2.5, 3.0, 3.5, 4.0)

_CLIP_LOW, _CLIP_HIGH = 0.02, 0.98
_MAX_CLIPPED_FRACTION = 0.20


def residual(path: Path) -> np.ndarray:
    """Noiseprint residual, cached to disk as float16.

    float16 because the residual is a small-amplitude map used only for relative
    comparisons within an image -- the precision that costs is not the precision
    that matters, and it halves 1.8 GB of cache.
    """
    key = CACHE / f"{path.parent.parent.name}_{path.parent.name}_{path.stem}.npy"
    if key.exists():
        return np.load(key).astype(np.float32)
    with Image.open(path) as im:
        gray = np.asarray(im.convert("RGB"), dtype=np.float32).mean(axis=2) / 255.0
    res = extract(gray, quality_factor(path))
    key.parent.mkdir(parents=True, exist_ok=True)
    np.save(key, res.astype(np.float16))
    return res


def gray_of(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32).mean(axis=2) / 255.0


def block_stats(res: np.ndarray, gray: np.ndarray, block: int):
    """Per-block fingerprint energy and clipped share, vectorised over blocks."""
    h, w = res.shape
    nr, nc = h // block, w // block
    r = res[: nr * block, : nc * block].reshape(nr, block, nc, block)
    g = gray[: nr * block, : nc * block].reshape(nr, block, nc, block)
    energy = r.std(axis=(1, 3))
    clipped = ((g <= _CLIP_LOW) | (g >= _CLIP_HIGH)).mean(axis=(1, 3))
    return energy, clipped


def anomaly_map(res, gray, block: int, z_thresh: float, min_cluster: int):
    """Blocks whose fingerprint deviates from the frame's own robust centre."""
    energy, clipped = block_stats(res, gray, block)
    usable = (energy > 1e-6) & (clipped <= _MAX_CLIPPED_FRACTION)
    if usable.sum() < 32:
        return None, 0.0

    logs = np.log(np.maximum(energy, 1e-6))
    centre = float(np.median(logs[usable]))
    scale = float(np.median(np.abs(logs[usable] - centre))) * 1.4826 + 1e-6

    z = np.zeros_like(energy)
    z[usable] = np.abs(logs[usable] - centre) / scale
    flagged = usable & (z > z_thresh)

    if min_cluster > 1 and flagged.any():
        lab, n = label(flagged, structure=np.ones((3, 3), dtype=int))
        sizes = np.bincount(lab.ravel())
        keep = [i for i in range(1, n + 1) if sizes[i] >= min_cluster]
        flagged = np.isin(lab, keep) if keep else np.zeros_like(flagged)

    return flagged, float(flagged.sum() / max(int(usable.sum()), 1))


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    from scipy.stats import rankdata

    if not len(pos) or not len(neg):
        return float("nan")
    s = np.concatenate([pos, neg])
    ranks = rankdata(s)
    n1, n0 = len(pos), len(neg)
    return float((ranks[: n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=40, help="tampered images per camera")
    ap.add_argument("--min-cluster", type=int, default=3)
    args = ap.parse_args()

    pairs = []
    for cam in CAMERAS:
        base = KORUS / cam
        for t in sorted((base / "tampered-realistic").glob("*.TIF"))[: args.limit]:
            gt, pr = base / "ground-truth" / f"{t.stem}.PNG", base / "pristine" / t.name
            if gt.exists() and pr.exists():
                pairs.append((t, pr, gt))
    if not pairs:
        print("no Korus pairs found", file=sys.stderr)
        return 1

    print(f"{len(pairs)} tampered/pristine pairs with ground truth\n", flush=True)

    started = time.time()
    cached = []
    for i, (t, pr, gt) in enumerate(pairs, 1):
        with Image.open(gt) as im:
            mask = np.asarray(im.convert("L")) > 127
        cached.append(
            (residual(t), gray_of(t), residual(pr), gray_of(pr), mask)
        )
        if i % 10 == 0 or i == len(pairs):
            print(f"  residuals {i}/{len(pairs)} ({time.time()-started:.0f}s)", flush=True)

    print(f"\n{'block':>6} {'z':>5} {'AUC':>7} {'hit':>7} {'IoU':>7} "
          f"{'t.frac':>8} {'p.frac':>8}")
    print("-" * 54)

    best = []
    for block in BLOCKS:
        for z in Z_VALUES:
            tf, pf, hits, ious = [], [], [], []
            for res_t, g_t, res_p, g_p, mask in cached:
                ft, frac_t = anomaly_map(res_t, g_t, block, z, args.min_cluster)
                _, frac_p = anomaly_map(res_p, g_p, block, z, args.min_cluster)
                tf.append(frac_t)
                pf.append(frac_p)
                if ft is not None and ft.any():
                    up = np.kron(ft, np.ones((block, block), dtype=bool))
                    h, w = mask.shape
                    full = np.zeros((h, w), dtype=bool)
                    full[: up.shape[0], : up.shape[1]] = up[:h, :w]
                    hits.append(float((full & mask).sum() / max(full.sum(), 1)))
                    union = int((full | mask).sum())
                    ious.append(float((full & mask).sum() / union) if union else 0.0)

            a = auc(np.array(tf), np.array(pf))
            hr = float(np.median(hits)) if hits else 0.0
            iou = float(np.median(ious)) if ious else 0.0
            best.append((a, block, z, hr, iou))
            print(f"{block:>6} {z:>5.1f} {a:>7.3f} {hr:>7.3f} {iou:>7.3f} "
                  f"{np.mean(tf):>8.4f} {np.mean(pf):>8.4f}")

    best.sort(reverse=True)
    a, block, z, hr, iou = best[0]
    print(f"\nbest AUC {a:.3f} at block={block} z={z} (hit {hr:.3f}, IoU {iou:.3f})")
    print("current shipped config is block=32 z=3.5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
