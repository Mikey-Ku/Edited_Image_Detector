"""Learn the fusion weights, and see whether learning them survives a sensor change.

    python scripts/learn_fusion.py

Predictions 5.4 and 5.5 of docs/METHOD_STUDY.md. `fusion/weighted.py` sets its
weights by hand and says so in its own docstring, listing it as defect 3 of 3 that
"must be fixed before any headline number is reported". This measures what fixing
it is worth.

Three arms over identical features, from the evidence `scripts/evaluate_korus.py`
already recorded, so nothing is re-inferred:

    hand-set     the shipped fuse(), read straight off the recorded probability
    logistic     L2 logistic regression
    boosted      gradient boosted trees

Two questions, and they are not the same question:

**Within a sensor.** Stratified 5-fold inside one camera. This is the friendly
case and the one a paper would report.

**Across sensors.** Train on one camera, test on the other, nothing shared. This
is the case a deployment actually meets, and there is already a reason to expect
it to fail: `fusion/calibration.py` found the two Korus sensors want significantly
different slopes (p = 0.02), so a single fitted map exports one camera's
correction onto another.

Confidence intervals are bootstrapped **on the paired difference** against the
hand-set baseline on the same test rows, never on the two AUCs separately. Two
overlapping intervals do not tell you whether the difference is real.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "data/processed/korus_results.json"
OUT = ROOT / "data/processed/fusion_comparison.json"

FEATURES = ("score", "confidence", "effect_size")


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based ROC AUC, ties averaged. No sklearn needed for this one."""
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within tied groups
    srt = s[order]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    pos, neg = y == 1, y == 0
    n1, n0 = int(pos.sum()), int(neg.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rows = json.loads(EVIDENCE.read_text())
    detectors = sorted({k for r in rows for k in r["evidence"]})
    names = [f"{d}.{f}" for d in detectors for f in FEATURES]

    X = np.array(
        [[r["evidence"].get(d, {}).get(f, 0.0) for d in detectors for f in FEATURES]
         for r in rows],
        dtype=float,
    )
    y = np.array([1 if r["label"] == "tampered" else 0 for r in rows])
    base = np.array([r["probability"] for r in rows], dtype=float)
    cam = np.array([r["camera"] for r in rows])

    keep = X.std(axis=0) > 0
    dropped = [n for n, k in zip(names, keep) if not k]
    if dropped:
        print(f"dropping {len(dropped)} constant features: "
              f"{', '.join(sorted({d.rsplit('.', 1)[0] for d in dropped}))}")
    return X[:, keep], y, base, cam, [n for n, k in zip(names, keep) if k]


def models(seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "logistic": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)
        ),
        # Shallow and heavily regularised on purpose. 224 rows is not a lot, and an
        # unconstrained booster on this many features would fit the split, not the
        # signal, then lose the cross-sensor test for the wrong reason.
        "boosted": HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.05,
            l2_regularization=1.0, min_samples_leaf=10, random_state=seed,
        ),
    }


def boot_delta(y, a, b, rng, n=5000) -> tuple[float, float]:
    """Percentile CI for AUC(a) - AUC(b), resampling test rows together."""
    deltas = np.empty(n)
    idx = np.arange(len(y))
    for i in range(n):
        take = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[take])) < 2:
            deltas[i] = np.nan
            continue
        deltas[i] = auc(y[take], a[take]) - auc(y[take], b[take])
    deltas = deltas[~np.isnan(deltas)]
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def within_sensor(X, y, base, cam, seed, rng) -> list[dict]:
    from sklearn.model_selection import StratifiedKFold

    out = []
    for camera in sorted(set(cam)):
        m = cam == camera
        if m.sum() < 40:
            print(f"  skipping {camera}: only {int(m.sum())} images")
            continue
        Xc, yc, bc = X[m], y[m], base[m]
        preds = {k: np.zeros(len(yc)) for k in models(seed)}
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, te in kf.split(Xc, yc):
            for name, mdl in models(seed).items():
                mdl.fit(Xc[tr], yc[tr])
                preds[name][te] = mdl.predict_proba(Xc[te])[:, 1]

        row = {"split": "within-sensor", "camera": camera, "n": int(m.sum()),
               "hand_set": auc(yc, bc)}
        for name, p in preds.items():
            lo, hi = boot_delta(yc, p, bc, rng)
            row[name] = auc(yc, p)
            row[f"{name}_delta"] = row[name] - row["hand_set"]
            row[f"{name}_ci"] = [lo, hi]
        out.append(row)
    return out


def cross_sensor(X, y, base, cam, seed, rng) -> list[dict]:
    out = []
    big = [c for c in sorted(set(cam)) if (cam == c).sum() >= 40]
    for train_cam in big:
        for test_cam in big:
            if train_cam == test_cam:
                continue
            tr, te = cam == train_cam, cam == test_cam
            row = {"split": "cross-sensor", "train": train_cam, "test": test_cam,
                   "n": int(te.sum()), "hand_set": auc(y[te], base[te])}
            for name, mdl in models(seed).items():
                mdl.fit(X[tr], y[tr])
                p = mdl.predict_proba(X[te])[:, 1]
                lo, hi = boot_delta(y[te], p, base[te], rng)
                row[name] = auc(y[te], p)
                row[f"{name}_delta"] = row[name] - row["hand_set"]
                row[f"{name}_ci"] = [lo, hi]
            out.append(row)
    return out


def show(rows: list[dict]) -> None:
    for r in rows:
        where = r.get("camera") or f"{r['train']} -> {r['test']}"
        print(f"\n  {where}  (n={r['n']})")
        print(f"    hand-set   AUC {r['hand_set']:.3f}")
        for name in ("logistic", "boosted"):
            lo, hi = r[f"{name}_ci"]
            sig = " " if lo <= 0 <= hi else "*"
            print(f"    {name:<10} AUC {r[name]:.3f}   delta {r[f'{name}_delta']:+.3f}"
                  f"  95% CI [{lo:+.3f}, {hi:+.3f}] {sig}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not EVIDENCE.exists():
        print(f"no evidence at {EVIDENCE} -- run scripts/evaluate_korus.py",
              file=sys.stderr)
        return 1

    X, y, base, cam, _names = load()
    print(f"{len(y)} images, {X.shape[1]} live features, "
          f"{len(set(cam))} cameras\n")

    rng = np.random.default_rng(args.seed)
    within = within_sensor(X, y, base, cam, args.seed, rng)
    print("WITHIN SENSOR (stratified 5-fold, * = CI excludes zero)")
    show(within)

    across = cross_sensor(X, y, base, cam, args.seed, rng)
    print("\n\nACROSS SENSORS (train one camera, test the other)")
    show(across)

    OUT.write_text(json.dumps({"within": within, "across": across}, indent=2))
    print(f"\nwritten to {OUT}")

    best_within = max(r["boosted_delta"] for r in within) if within else 0.0
    print(f"\n5.4 (learned fusion beats hand-set by >= 0.03 within a sensor): "
          f"best boosted delta {best_within:+.3f} -> "
          f"{'HIT' if best_within >= 0.03 else 'MISS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
