"""Compare readout statistics on the cached Noiseprint residual.

    python scripts/sweep_readout.py

`scripts/sweep_noiseprint.py` swept block geometry and came back flat: AUC 0.527 to
0.567 across every block size from 16 to 64 and every threshold. Geometry is not the
binding constraint, so multi-scale analysis cannot be the fix.

That result points somewhere more interesting. The shipped detector reduces each
block to **the standard deviation of its residual** -- a scalar energy. But a camera
fingerprint is a *pattern*, not an amount. Two blocks can carry identical residual
energy and completely different fingerprints, and the current readout cannot tell
them apart. Sweeping block size only ever varied how many pixels went into the wrong
summary.

So this sweeps the summary instead, over the same cached residuals:

**energy** -- the shipped statistic, standard deviation. Baseline to beat.

**period2** -- energy at the 2x2 spatial frequency, normalised by total block energy.
A sensor captures one colour per pixel and interpolates the rest on a Bayer grid, so
an authentic residual carries structure with period exactly 2. A rendered, inpainted,
or resampled region was never demosaiced and loses it. This is a *shape* measure: it
asks what the residual looks like rather than how large it is.

**global_corr** -- correlation of each block's residual against the frame's own mean
residual pattern. Same intuition, learned from the image rather than assumed from the
Bayer grid.

Each is scored by AUC against matched pristine originals, which is the only way to
tell a statistic that finds manipulations from one that finds texture.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
KORUS = ROOT / "data/interim/korus/data-images"
CACHE = ROOT / "data/interim/noiseprint_cache"
CAMERAS = ("Nikon_D7000", "Nikon_D90")

_CLIP_LOW, _CLIP_HIGH = 0.02, 0.98
_MAX_CLIPPED = 0.20


def blocks_of(a: np.ndarray, block: int) -> np.ndarray:
    nr, nc = a.shape[0] // block, a.shape[1] // block
    return a[: nr * block, : nc * block].reshape(nr, block, nc, block).transpose(
        0, 2, 1, 3
    )


def readouts(res: np.ndarray, gray: np.ndarray, block: int) -> dict[str, np.ndarray]:
    """Per-block statistics, all computed from the same block decomposition."""
    r = blocks_of(res, block)
    g = blocks_of(gray, block)
    nr, nc = r.shape[:2]

    energy = r.std(axis=(2, 3))
    clipped = ((g <= _CLIP_LOW) | (g >= _CLIP_HIGH)).mean(axis=(2, 3))

    # Period-2 structure: project each block onto the three sign patterns that
    # alternate every pixel. Normalised by total energy so this measures the
    # residual's SHAPE, independent of how strong it is.
    i = np.arange(block)
    sr = ((-1.0) ** i)[None, :]
    centred = r - r.mean(axis=(2, 3), keepdims=True)
    total = np.sqrt((centred ** 2).sum(axis=(2, 3))) + 1e-9
    p_row = np.abs((centred * sr[:, :, None]).sum(axis=(2, 3))) / total
    p_col = np.abs((centred * sr[:, None, :]).sum(axis=(2, 3))) / total
    p_diag = np.abs(
        (centred * (sr[:, :, None] * sr[:, None, :])).sum(axis=(2, 3))
    ) / total
    period2 = np.sqrt(p_row ** 2 + p_col ** 2 + p_diag ** 2)

    # Correlation against the frame's own mean residual pattern.
    flat = centred.reshape(nr, nc, -1)
    norms = np.linalg.norm(flat, axis=2) + 1e-9
    unit = flat / norms[:, :, None]
    template = unit.reshape(-1, flat.shape[-1]).mean(axis=0)
    template /= np.linalg.norm(template) + 1e-9
    global_corr = np.abs(unit @ template)

    return {
        "energy": energy,
        "period2": period2,
        "global_corr": global_corr,
        "_clipped": clipped,
    }


def anomalous_fraction(stat: np.ndarray, usable: np.ndarray, z: float) -> float:
    """Share of usable blocks deviating from the frame's own robust centre."""
    if usable.sum() < 32:
        return 0.0
    v = stat[usable]
    centre = float(np.median(v))
    scale = float(np.median(np.abs(v - centre))) * 1.4826 + 1e-9
    return float((np.abs(v - centre) / scale > z).sum() / usable.sum())


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cached_residual(path: Path) -> np.ndarray | None:
    key = CACHE / f"{path.parent.parent.name}_{path.parent.name}_{path.stem}.npy"
    return np.load(key).astype(np.float32) if key.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--block", type=int, default=32)
    args = ap.parse_args()

    pairs = []
    for cam in CAMERAS:
        base = KORUS / cam
        for t in sorted((base / "tampered-realistic").glob("*.TIF")):
            pr = base / "pristine" / t.name
            if pr.exists() and cached_residual(t) is not None:
                pairs.append((t, pr))
    if not pairs:
        print("no cached residuals -- run scripts/sweep_noiseprint.py first",
              file=sys.stderr)
        return 1
    print(f"{len(pairs)} pairs with cached residuals, block={args.block}\n")

    names = ("energy", "period2", "global_corr")
    z_values = (2.5, 3.0, 3.5, 4.0)
    scores: dict[tuple[str, float], tuple[list, list]] = {
        (n, z): ([], []) for n in names for z in z_values
    }

    for t, pr in pairs:
        for slot, path in ((0, t), (1, pr)):
            res = cached_residual(path)
            with Image.open(path) as im:
                gray = np.asarray(im.convert("RGB"), np.float32).mean(axis=2) / 255.0
            r = readouts(res, gray, args.block)
            usable = (r["energy"] > 1e-6) & (r["_clipped"] <= _MAX_CLIPPED)
            for n in names:
                for z in z_values:
                    scores[(n, z)][slot].append(anomalous_fraction(r[n], usable, z))

    print(f"{'readout':>12} {'z':>5} {'AUC':>8}   {'t.frac':>8} {'p.frac':>8}")
    print("-" * 48)
    best = []
    for n in names:
        for z in z_values:
            tf, pf = scores[(n, z)]
            a = auc(np.array(tf), np.array(pf))
            best.append((a, n, z))
            print(f"{n:>12} {z:>5.1f} {a:>8.3f}   "
                  f"{np.mean(tf):>8.4f} {np.mean(pf):>8.4f}")
        print()

    best.sort(reverse=True)
    a, n, z = best[0]
    print(f"best: {n} at z={z} -> AUC {a:.3f}")
    print("shipped readout is `energy`; block geometry was already swept and is flat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
