"""Choose the fingerprint detector's score mapping against the FUSED objective.

    python scripts/tune_fusion_score.py

A detector's own AUC is not the objective. What matters is the pipeline's AUC after
fusion, and the two can disagree: `fuse()` discounts negatives, weights positives by
effect size, and a detector that orders images well can still fuse badly if its
scores land in the wrong part of that machinery.

Measuring that properly would cost a 28-minute pipeline run per candidate mapping,
which in practice means one candidate gets tried. Instead the new readout is computed
offline from cached residuals, spliced into the recorded evidence from
`scripts/evaluate_korus.py`, and the real `fuse()` is replayed. Seconds per candidate.

**Split discipline.** The mapping is chosen on the 40-per-camera sweep set and
reported on the 15-per-camera held-out set, which had no say in the readout, the
block size, the threshold, or the mapping.

Two mapping families:

**bimodal** -- what ships now: below a floor report clean at a fixed low score, above
it report a graded positive. Suits a detector that is either silent or certain.

**graded** -- a logistic in the anomalous fraction, no floor. Passes the full ordering
to the fusion instead of discarding it at a threshold. Costs the ability to say
"definitely nothing", which is what the floor exists to protect.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sweep_readout import (
    _MAX_CLIPPED,
    CAMERAS,
    KORUS,
    anomalous_fraction,
    cached_residual,
    readouts,
)

from groundtruth.core.types import Evidence, Tier
from groundtruth.fusion.weighted import fuse

Image.MAX_IMAGE_PIXELS = None

RESULTS = Path(__file__).resolve().parents[1] / "data/processed/korus_results.json"
DETECTOR = "sensor.noiseprint_anomaly"

BLOCK, Z, READOUT = 48, 3.0, "period2"
SWEEP_N = 40  # first N tampered per camera chose everything; the rest are unseen


def auc(pos, neg) -> float:
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fractions() -> dict[str, tuple[float, bool]]:
    """name -> (anomalous fraction, is_held_out), for every cached image."""
    out: dict[str, tuple[float, bool]] = {}
    for cam in CAMERAS:
        base = KORUS / cam
        tam = sorted((base / "tampered-realistic").glob("*.TIF"))
        for i, t in enumerate(tam):
            for path in (t, base / "pristine" / t.name):
                res = cached_residual(path)
                if res is None or not path.exists():
                    continue
                with Image.open(path) as im:
                    g = np.asarray(im.convert("RGB"), np.float32).mean(axis=2) / 255.0
                r = readouts(res, g, BLOCK)
                usable = (r["energy"] > 1e-6) & (r["_clipped"] <= _MAX_CLIPPED)
                key = f"{cam}/{path.parent.name}/{path.name}"
                out[key] = (anomalous_fraction(r[READOUT], usable, Z), i >= SWEEP_N)
    return out


def bimodal(frac: float, floor: float) -> tuple[float, float]:
    """(score, effect_size) -- the shape the detector ships today."""
    if frac < floor:
        return 0.22, 0.0
    return min(0.94, 0.62 + 6.0 * frac), min(1.0, frac / (2 * floor))


def graded(frac: float, centre: float, scale: float) -> tuple[float, float]:
    """(score, effect_size) -- logistic in the fraction, no floor."""
    s = 1.0 / (1.0 + np.exp(-(frac - centre) / scale))
    return float(np.clip(s, 0.15, 0.85)), float(min(1.0, frac / (2 * centre)))


def replay(records, frac_by_key, mapper) -> tuple[float, float, float, float]:
    """(sweep AUC, held-out AUC, pristine flag rate, tampered flag rate).

    AUC alone cannot see calibration -- it is invariant to any monotone transform of
    the score, so a mapping that puts every clean photograph at 0.8 scores exactly as
    well as one that puts them at 0.2. The decision rates are what the thresholds act
    on, and they are how the first version of this readout got caught: AUC rose while
    pristine photographs drifted up to the flag boundary.
    """
    buckets = {False: ([], []), True: ([], [])}
    for r in records:
        key = f"{r['camera']}/{'tampered-realistic' if r['label']=='tampered' else 'pristine'}/{r['name']}"
        if key not in frac_by_key:
            continue
        frac, held = frac_by_key[key]
        score, effect = mapper(frac)
        ev = []
        for d, v in r["evidence"].items():
            if d == DETECTOR:
                v = {**v, "score": score, "effect_size": effect}
            ev.append(
                Evidence(
                    detector_id=d,
                    tier=Tier.SENSOR,
                    applicable=v["applicable"],
                    score=v["score"],
                    confidence=v["confidence"],
                    effect_size=v["effect_size"],
                    explanation="",
                )
            )
        p = fuse(ev).manipulated_probability
        buckets[held][0 if r["label"] == "tampered" else 1].append(p)

    tam = np.array(buckets[False][0] + buckets[True][0])
    pri = np.array(buckets[False][1] + buckets[True][1])
    return (
        auc(np.array(buckets[False][0]), np.array(buckets[False][1])),
        auc(np.array(buckets[True][0]), np.array(buckets[True][1])),
        float((pri > 0.70).mean()),
        float((tam > 0.70).mean()),
    )


def main() -> int:
    records = [
        r for r in json.loads(RESULTS.read_text()) if "evidence" in r
    ]
    if not records:
        print("no recorded evidence -- run scripts/evaluate_korus.py", file=sys.stderr)
        return 1
    print("computing readout over cached residuals...", flush=True)
    fbk = fractions()
    print(f"{len(fbk)} images with a cached residual\n")

    print(f"{'mapping':<34} {'sweep':>8} {'held-out':>10} {'FPR':>7} {'TPR':>7}")
    print("-" * 70)

    # Current shipped detector, unchanged, for reference.
    cur = {False: ([], []), True: ([], [])}
    for r in records:
        key = f"{r['camera']}/{'tampered-realistic' if r['label']=='tampered' else 'pristine'}/{r['name']}"
        if key not in fbk:
            continue
        held = fbk[key][1]
        ev = [
            Evidence(detector_id=d, tier=Tier.SENSOR, applicable=v["applicable"],
                     score=v["score"], confidence=v["confidence"],
                     effect_size=v["effect_size"], explanation="")
            for d, v in r["evidence"].items()
        ]
        cur[held][0 if r["label"] == "tampered" else 1].append(
            fuse(ev).manipulated_probability
        )
    ct, cp = np.array(cur[False][0] + cur[True][0]), np.array(cur[False][1] + cur[True][1])
    print(f"{'SHIPPED (energy readout)':<34} "
          f"{auc(np.array(cur[False][0]), np.array(cur[False][1])):>8.3f} "
          f"{auc(np.array(cur[True][0]), np.array(cur[True][1])):>10.3f} "
          f"{(cp > 0.70).mean():>7.1%} {(ct > 0.70).mean():>7.1%}")
    print()

    results = []
    for centre in (0.025, 0.035, 0.045, 0.055, 0.065):
        for scale in (0.015, 0.020, 0.030):
            s, h, fpr, tpr = replay(
                records, fbk, lambda f, c=centre, sc=scale: graded(f, c, sc)
            )
            label = f"graded centre={centre} scale={scale}"
            results.append((s, h, fpr, tpr, label))
            print(f"{label:<34} {s:>8.3f} {h:>10.3f} {fpr:>7.1%} {tpr:>7.1%}")
        print()

    # Rank by held-out AUC among mappings that keep the false positive rate at or
    # below what ships today. An uncalibrated gain in AUC is not a gain in the
    # product: the thresholds are what an adjuster actually sees.
    budget = float((cp > 0.70).mean())
    viable = [r for r in results if r[2] <= budget]
    if not viable:
        print(f"no mapping holds FPR at or below the shipped {budget:.1%}")
        return 1
    viable.sort(key=lambda r: -r[0])
    s, h, fpr, tpr, label = viable[0]
    print(f"best on SWEEP within the shipped FPR budget of {budget:.1%}:")
    print(f"  {label}  ->  held-out AUC {h:.3f}, FPR {fpr:.1%}, TPR {tpr:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
