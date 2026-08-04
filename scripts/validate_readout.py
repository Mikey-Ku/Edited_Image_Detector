"""Validate the chosen readout on Korus pairs the sweep never saw.

    python scripts/validate_readout.py

`scripts/sweep_readout.py` scored 60 configurations -- 3 readouts x 5 block sizes x
4 thresholds -- on the first 40 tampered images per camera, and the best cell was
0.679. Reporting that number would be reporting the maximum of 60 draws, which is
biased upward by construction and is one of the easier ways to fool yourself.

So the configuration is fixed here in advance, and scored on the 15 pairs per camera
the sweep never touched. Two configurations only:

    baseline   energy  block=32  z=3.5    what currently ships
    candidate  period2 block=48  z=3.0    best cell on the sweep set

The comparison that matters is not whether the candidate hits 0.679 again -- it will
not, and it does not need to. It is whether it still beats the baseline on data that
had no say in choosing it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sweep_readout import _MAX_CLIPPED, anomalous_fraction, readouts

from groundtruth.learned.noiseprint import extract, quality_factor

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
KORUS = ROOT / "data/interim/korus/data-images"
CACHE = ROOT / "data/interim/noiseprint_cache"
CAMERAS = ("Nikon_D7000", "Nikon_D90")

# Index at which the sweep stopped. Everything from here on is unseen.
HELD_OUT_FROM = 40

BASELINE = ("energy", 32, 3.5)
CANDIDATE = ("period2", 48, 3.0)


def residual(path: Path) -> np.ndarray:
    key = CACHE / f"{path.parent.parent.name}_{path.parent.name}_{path.stem}.npy"
    if key.exists():
        return np.load(key).astype(np.float32)
    with Image.open(path) as im:
        gray = np.asarray(im.convert("RGB"), np.float32).mean(axis=2) / 255.0
    res = extract(gray, quality_factor(path))
    key.parent.mkdir(parents=True, exist_ok=True)
    np.save(key, res.astype(np.float16))
    return res


def auc(pos, neg) -> float:
    ranks = rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_ci(pos, neg, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main() -> int:
    pairs = []
    for cam in CAMERAS:
        base = KORUS / cam
        tam = sorted((base / "tampered-realistic").glob("*.TIF"))[HELD_OUT_FROM:]
        for t in tam:
            pr = base / "pristine" / t.name
            if pr.exists():
                pairs.append((cam, t, pr))
    if not pairs:
        print("no held-out pairs found", file=sys.stderr)
        return 1
    print(f"{len(pairs)} held-out pairs, unseen by the sweep\n", flush=True)

    out = {BASELINE: ([], [], []), CANDIDATE: ([], [], [])}
    for i, (cam, t, pr) in enumerate(pairs, 1):
        for slot, path in ((0, t), (1, pr)):
            res = residual(path)
            with Image.open(path) as im:
                gray = np.asarray(im.convert("RGB"), np.float32).mean(axis=2) / 255.0
            for cfg in (BASELINE, CANDIDATE):
                name, block, z = cfg
                r = readouts(res, gray, block)
                usable = (r["energy"] > 1e-6) & (r["_clipped"] <= _MAX_CLIPPED)
                out[cfg][slot].append(anomalous_fraction(r[name], usable, z))
            if slot == 0:
                out[BASELINE][2].append(cam)
                out[CANDIDATE][2].append(cam)
        if i % 5 == 0 or i == len(pairs):
            print(f"  {i}/{len(pairs)}", flush=True)

    print(f"\n{'config':<28} {'AUC':>7}  {'95% CI':>16}")
    print("-" * 56)
    for cfg, label in ((BASELINE, "baseline  (shipped)"), (CANDIDATE, "candidate (period2)")):
        t, p, cams = out[cfg]
        t, p = np.array(t), np.array(p)
        lo, hi = boot_ci(t, p)
        name, block, z = cfg
        print(f"{label:<28} {auc(t, p):>7.3f}  [{lo:>6.3f}, {hi:>6.3f}]   "
              f"{name} block={block} z={z}")

    print(f"\n{'per camera':<28} {'baseline':>9} {'candidate':>10}")
    print("-" * 56)
    cams = out[BASELINE][2]
    for cam in CAMERAS:
        idx = [i for i, c in enumerate(cams) if c == cam]
        if len(idx) < 4:
            continue
        row = []
        for cfg in (BASELINE, CANDIDATE):
            t, p, _ = out[cfg]
            row.append(auc(np.array(t)[idx], np.array(p)[idx]))
        print(f"{cam:<28} {row[0]:>9.3f} {row[1]:>10.3f}")

    # Paired: same images, both configs. Removes image difficulty as a confound.
    tb, pb, _ = out[BASELINE]
    tc, pc, _ = out[CANDIDATE]
    wins = sum(
        (tc[i] - pc[i]) > (tb[i] - pb[i]) for i in range(len(tb))
    )
    print(f"\ncandidate separates the pair more often on {wins}/{len(tb)} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
