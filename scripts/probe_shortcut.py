"""How much of CASIA can be solved with no forensic feature at all?

    python scripts/probe_shortcut.py

Prediction 5.6 of docs/METHOD_STUDY.md, as amended. The original probe read JPEG
quantization tables; the mirror re-encodes everything to PNG, so those are gone.
What survives is cruder and still worth measuring: if a classifier given only the
shape and gross statistics of a file can separate authentic from tampered, then
that much of any CASIA result is bookkeeping rather than forensics.

Nothing here looks at a forgery. No mask, no local structure, no comparison
between regions. Three feature sets, so the answer is attributable:

    shape    width, height, aspect ratio, pixel count, file size, bytes per pixel
    stats    per-channel mean and standard deviation
    both     the union

If `shape` alone clears 0.80 the dataset has a resolution tell, and every number
this study reports on CASIA is discounted by that amount.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CASIA = ROOT / "data/raw/casia2"
OUT = ROOT / "data/processed/shortcut_probe.json"

SHAPE = ["width", "height", "aspect", "pixels", "bytes", "bytes_per_pixel"]
STATS = ["r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std"]


def features(path: Path) -> list[float] | None:
    try:
        with Image.open(path) as im:
            w, h = im.size
            a = np.asarray(im.convert("RGB"), dtype=np.float32)
    except Exception:
        return None
    size = path.stat().st_size
    px = w * h
    return [
        float(w), float(h), w / h, float(px), float(size), size / max(px, 1),
        *[float(a[:, :, c].mean()) for c in range(3)],
        *[float(a[:, :, c].std()) for c in range(3)],
    ]


def collect() -> tuple[np.ndarray, np.ndarray]:
    rows, labels = [], []
    groups = [(CASIA / "authentic", 0), (CASIA / "tampered", 1)]
    for folder, label in groups:
        paths = [p for p in sorted(folder.glob("*.png")) if ".mask." not in p.name]
        print(f"  {folder.name}: {len(paths)} images", flush=True)
        for i, p in enumerate(paths):
            f = features(p)
            if f is not None:
                rows.append(f)
                labels.append(label)
            if (i + 1) % 2000 == 0:
                print(f"    {i+1}", flush=True)
    return np.array(rows, dtype=float), np.array(labels)


def evaluate(X: np.ndarray, y: np.ndarray, seed: int) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = {}
    for name, model in {
        "logistic": make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=2000)),
        "boosted": HistGradientBoostingClassifier(random_state=seed, max_depth=4),
    }.items():
        pred = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            model.fit(X[tr], y[tr])
            pred[te] = model.predict_proba(X[te])[:, 1]
        out[name] = float(roc_auc_score(y, pred))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not (CASIA / "authentic").is_dir():
        print(f"no CASIA at {CASIA} -- run scripts/fetch_casia.py", file=sys.stderr)
        return 1

    print("reading files...")
    X, y = collect()
    print(f"\n{len(y)} images, {int(y.sum())} tampered\n")

    names = SHAPE + STATS
    sets = {
        "shape": [names.index(n) for n in SHAPE],
        "stats": [names.index(n) for n in STATS],
        "both": list(range(len(names))),
    }

    results = {}
    for label, cols in sets.items():
        results[label] = evaluate(X[:, cols], y, args.seed)
        best = max(results[label].values())
        print(f"  {label:<7} logistic {results[label]['logistic']:.3f}   "
              f"boosted {results[label]['boosted']:.3f}   (best {best:.3f})")

    best_overall = max(v for r in results.values() for v in r.values())
    print(f"\n5.6 (>= 0.80 with no forensic feature): {best_overall:.3f} -> "
          f"{'HIT' if best_overall >= 0.80 else 'MISS'}")
    OUT.write_text(json.dumps({"n": len(y), "results": results}, indent=2))
    print(f"written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
