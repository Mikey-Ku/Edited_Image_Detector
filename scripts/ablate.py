"""Ablate the detector set and the fusion, offline, from recorded evidence.

    python scripts/ablate.py

Reads the per-detector readings written by `scripts/evaluate_korus.py` and replays
the real `fuse()` over subsets of them. Nothing is re-inferred, so a question that
used to cost a 28-minute run costs a second, and questions that cost 28 minutes
tend not to get asked.

Three things are measured, and they answer different questions:

**Standalone AUC** -- can this detector, alone, order forgeries above their own
matched originals? A detector below 0.5 is actively anti-correlated and worse than
silence.

**Leave-one-out AUC** -- does removing this detector help or hurt the fused score?
A detector can carry real signal and still damage the fusion by correlating with a
stronger one, and a detector with no signal at all can only add variance.

**Localisation lift** -- flagged pixels landing inside the true mask, against the
mask's own area as the chance baseline. Hit rate alone is not interpretable: a
heatmap covering the whole frame scores the mask fraction for free.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundtruth.core.types import Evidence, Tier
from groundtruth.fusion.weighted import fuse

RESULTS = Path(__file__).resolve().parents[1] / "data/processed/korus_results.json"


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_ci(pos, neg, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [
        auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n)
    ]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def rebuild(ev: dict, keep: set[str]) -> list[Evidence]:
    """Reconstruct Evidence for the kept detectors and replay the real fusion."""
    return [
        Evidence(
            detector_id=d,
            tier=Tier.SENSOR,  # tier is not consulted by fuse()
            applicable=v["applicable"],
            score=v["score"],
            confidence=v["confidence"],
            effect_size=v["effect_size"],
            explanation="",
        )
        for d, v in ev.items()
        if d in keep
    ]


def fused_auc(records: list[dict], keep: set[str]) -> float:
    t, p = [], []
    for r in records:
        prob = fuse(rebuild(r["evidence"], keep)).manipulated_probability
        (t if r["label"] == "tampered" else p).append(prob)
    return auc(np.array(t), np.array(p))


def main() -> int:
    if not RESULTS.exists():
        print(f"no results at {RESULTS} -- run scripts/evaluate_korus.py", file=sys.stderr)
        return 1
    records = json.loads(RESULTS.read_text())
    if "evidence" not in records[0]:
        print("results predate per-detector recording -- re-run evaluate_korus.py",
              file=sys.stderr)
        return 1

    detectors = sorted(
        {d for r in records for d, v in r["evidence"].items() if v["applicable"]}
    )
    tam = [r for r in records if r["label"] == "tampered"]
    pri = [r for r in records if r["label"] == "pristine"]
    print(f"{len(tam)} tampered / {len(pri)} pristine, "
          f"{len(detectors)} applicable detectors\n")

    base = fused_auc(records, set(detectors))
    lo, hi = boot_ci(
        np.array([fuse(rebuild(r["evidence"], set(detectors))).manipulated_probability
                  for r in tam]),
        np.array([fuse(rebuild(r["evidence"], set(detectors))).manipulated_probability
                  for r in pri]),
    )
    print(f"FULL PIPELINE   AUC {base:.4f}   95% CI [{lo:.3f}, {hi:.3f}]\n")

    print(f"{'detector':<34} {'alone':>8} {'without':>9} {'delta':>8}")
    print("-" * 62)
    rows = []
    for d in detectors:
        a_alone = auc(
            np.array([r["evidence"][d]["score"] for r in tam]),
            np.array([r["evidence"][d]["score"] for r in pri]),
        )
        a_without = fused_auc(records, set(detectors) - {d})
        rows.append((d, a_alone, a_without, a_without - base))
        print(f"{d:<34} {a_alone:>8.3f} {a_without:>9.3f} {a_without-base:>+8.3f}")

    print("\nA positive delta means the pipeline scores HIGHER without that detector.")

    # Best subset. Small detector count, so this is exhaustive rather than greedy --
    # greedy selection would miss a pair that only helps together.
    print(f"\n{'best subsets':<52} {'AUC':>8}")
    print("-" * 62)
    subsets = []
    for k in range(1, len(detectors) + 1):
        for combo in combinations(detectors, k):
            subsets.append((fused_auc(records, set(combo)), combo))
    subsets.sort(reverse=True)
    for a, combo in subsets[:5]:
        names = ", ".join(c.split(".")[-1] for c in combo)
        print(f"{names:<52} {a:>8.4f}")

    # Localisation, against the mask's own area as the chance baseline.
    print(f"\n{'detector localisation (tampered only)':<40} {'hit':>7} {'lift':>7} {'n':>5}")
    print("-" * 62)
    for d in detectors:
        hits, lifts = [], []
        for r in tam:
            loc = r.get("detector_loc", {}).get(d)
            if loc and r.get("mask_fraction"):
                hits.append(loc["hit_rate"])
                lifts.append(loc["hit_rate"] / r["mask_fraction"])
        if hits:
            print(f"{d:<40} {np.median(hits):>7.3f} {np.median(lifts):>7.1f}x "
                  f"{len(hits):>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
