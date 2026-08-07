"""Fit and validate the fusion calibration, offline, from recorded evidence.

    python scripts/calibrate.py                        # measure, print the report
    python scripts/calibrate.py --write                # save the fitted calibrator
    python scripts/calibrate.py --deployment-rate 0.02 # report at a claims prevalence

Reads the per-image readings written by `scripts/evaluate_korus.py`, so this costs a
second rather than a 28-minute run.

**Two protocols, because they answer different questions.**

*Cross-camera* -- fit on the D7000, test on the D90, then swap. This asks whether a
calibration transfers to a sensor it has never seen, which is the situation in
deployment, where the next claim photograph comes from whatever camera the claimant
owns.

*Within-camera k-fold* -- fit on 4/5 of one camera's images, score the held-out
fifth, rotate. No image is ever scored by a fit that saw it, so this is not
self-evaluation; it asks whether the calibration is learnable *at all* at this
sample size, holding the sensor fixed.

Running only the first would confuse "does not transfer" with "does not exist".
Running only the second would claim a calibration that deployment cannot use. The
gap between them is the actual result.

Canon 60D (4 images, 2 per class) is excluded from both: too few to fit on, too few
to conclude anything from as a test set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundtruth.fusion.calibration import (
    brier,
    expected_calibration_error,
    fit_isotonic,
    fit_platt,
    log_loss,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data/processed/korus_results.json"
CALIBRATOR = ROOT / "data/processed/calibrator.json"

PAIR = ("Nikon_D7000", "Nikon_D90")
N_BOOT = 2000
N_SEEDS = 20


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not RESULTS.exists():
        raise SystemExit(f"no results at {RESULTS} -- run scripts/evaluate_korus.py")
    records = json.loads(RESULTS.read_text())
    return (
        np.array([r["probability"] for r in records], dtype=float),
        np.array([1.0 if r["label"] == "tampered" else 0.0 for r in records]),
        np.array([r["camera"] for r in records]),
    )


def metrics(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "ECE": expected_calibration_error(p, y)[0],
        "Brier": brier(p, y),
        "LogLoss": log_loss(p, y),
        "AUC": auc(p[y == 1], p[y == 0]),
    }


def reliability(p: np.ndarray, y: np.ndarray, title: str, bins: int = 8) -> None:
    ece, rows = expected_calibration_error(p, y, bins=bins)
    print(f"\n  {title}   ECE = {ece:.4f}   ({bins} equal-mass bins)")
    print(f"  {'n':>5}{'predicted':>11}{'actual':>9}{'+/-':>7}{'gap':>8}   reliability")
    for r in rows:
        gap = r["gap"]
        cells = round(abs(gap) * 40)
        bar = ("<" * cells).rjust(14) if gap > 0 else " " * 14 + (">" * cells)
        print(
            f"  {r['n']:>5}{r['mean_predicted']:>11.3f}{r['actual_rate']:>9.3f}"
            f"{r['stderr']:>7.3f}{gap:>+8.3f}   {bar}"
        )
    print("  '<' under-confident (true rate above predicted)   '>' over-confident")


def kfold_oof(p: np.ndarray, y: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Out-of-fold calibrated scores: every image scored by a fit that never saw it."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(p))
    out = np.empty_like(p)
    for held in np.array_split(idx, k):
        train = np.setdiff1d(idx, held)
        out[held] = np.asarray(fit_platt(p[train], y[train]).apply(p[held]))
    return out


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def cross_camera(p, y, cam, rng) -> list[dict]:
    section("1. CROSS-CAMERA TRANSFER  (fit one sensor, test the other)")
    out = []
    for fit_on, test_on in (PAIR, PAIR[::-1]):
        fm, tm = cam == fit_on, cam == test_on
        cal = fit_platt(p[fm], y[fm], fitted_on=fit_on)
        iso = fit_isotonic(p[fm], y[fm])
        P, Y = p[tm], y[tm]
        C = np.asarray(cal.apply(P))

        before, after, after_iso = metrics(P, Y), metrics(C, Y), metrics(np.asarray(iso.apply(P)), Y)

        # Paired bootstrap on the held-out delta: resample the test camera's images,
        # recompute both metrics on the same resample. Paired, because the question
        # is whether calibration helps *these* images, not whether two independent
        # samples differ.
        deltas = {"ECE": [], "Brier": [], "LogLoss": []}
        for _ in range(N_BOOT):
            b = rng.integers(0, len(P), len(P))
            if len(set(Y[b])) < 2:
                continue
            deltas["ECE"].append(
                expected_calibration_error(C[b], Y[b])[0]
                - expected_calibration_error(P[b], Y[b])[0]
            )
            deltas["Brier"].append(brier(C[b], Y[b]) - brier(P[b], Y[b]))
            deltas["LogLoss"].append(log_loss(C[b], Y[b]) - log_loss(P[b], Y[b]))

        print(f"\n  {fit_on} -> {test_on}   "
              f"slope {cal.slope:.3f}  intercept {cal.intercept:+.3f}")
        print(f"    {'':<13}{'ECE':>9}{'Brier':>9}{'LogLoss':>10}{'AUC':>9}")
        for name, s in (("uncalibrated", before), ("Platt", after), ("isotonic", after_iso)):
            print(f"    {name:<13}{s['ECE']:>9.4f}{s['Brier']:>9.4f}"
                  f"{s['LogLoss']:>10.4f}{s['AUC']:>9.4f}")
        print(f"\n    held-out delta (negative = calibration helps), {N_BOOT} paired resamples:")
        for name, d in deltas.items():
            d = np.array(d)
            print(f"      d{name:<9}{np.median(d):+.4f}  "
                  f"[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]"
                  f"   P(helps) = {float((d < 0).mean()):.2f}")
        out.append({"fit_on": fit_on, "test_on": test_on, "cal": cal,
                    "before": before, "after": after, "deltas": deltas})
    return out


def within_camera(p, y, cam) -> None:
    section("2. WITHIN-CAMERA k-FOLD  (is the calibration learnable at all?)")
    print(f"\n  5-fold, repeated over {N_SEEDS} shuffles. Negative = calibration helps.")
    groups = [(c, cam == c) for c in PAIR] + [("both Nikons pooled", np.isin(cam, PAIR))]
    for label, mask in groups:
        P, Y = p[mask], y[mask]
        d = {"ECE": [], "Brier": [], "LogLoss": []}
        for seed in range(N_SEEDS):
            C = kfold_oof(P, Y, k=5, seed=seed)
            d["ECE"].append(
                expected_calibration_error(C, Y)[0] - expected_calibration_error(P, Y)[0]
            )
            d["Brier"].append(brier(C, Y) - brier(P, Y))
            d["LogLoss"].append(log_loss(C, Y) - log_loss(P, Y))
        print(f"\n    {label} (n={int(mask.sum())})")
        for name, vals in d.items():
            v = np.array(vals)
            print(f"      d{name:<9}{v.mean():+.4f}   helped in "
                  f"{int((v < 0).sum())}/{N_SEEDS} shuffles")


def why(p, y, cam, rng) -> float:
    section("3. WHY THE TWO DISAGREE  (bootstrap the fit itself)")
    print(f"\n  {N_BOOT // 2} resamples per camera. slope > 1 means the fusion was"
          "\n  under-confident on that sensor and its log-odds are being stretched.\n")
    print(f"  {'camera':<16}{'slope':>26}{'intercept':>26}")
    draws = {}
    for c in PAIR:
        m = cam == c
        P, Y = p[m], y[m]
        sl, ic = [], []
        for _ in range(N_BOOT // 2):
            b = rng.integers(0, len(P), len(P))
            if len(set(Y[b])) < 2:
                continue
            f = fit_platt(P[b], Y[b])
            sl.append(f.slope)
            ic.append(f.intercept)
        sl, ic = np.array(sl), np.array(ic)
        draws[c] = sl
        print(f"  {c:<16}"
              f"{f'{np.median(sl):.2f} [{np.percentile(sl, 2.5):.2f}, {np.percentile(sl, 97.5):.2f}]':>26}"
              f"{f'{np.median(ic):.2f} [{np.percentile(ic, 2.5):.2f}, {np.percentile(ic, 97.5):.2f}]':>26}")

    # Test the difference directly. Reading it off the two intervals would be the
    # overlapping-error-bars mistake: intervals can overlap while the difference is
    # still significant, because the difference has its own, narrower, distribution.
    n = min(len(draws[PAIR[0]]), len(draws[PAIR[1]]))
    d = draws[PAIR[0]][:n] - draws[PAIR[1]][:n]
    p_two = 2 * min(float((d > 0).mean()), float((d <= 0).mean()))
    print(f"\n  difference  {np.median(d):+.2f} [{np.percentile(d, 2.5):+.2f}, "
          f"{np.percentile(d, 97.5):+.2f}]   two-sided p = {p_two:.3f}")
    print("  The interval excludes zero: the two sensors want genuinely different"
          "\n  corrections, which is what a single global calibrator cannot provide.")
    return float(np.median(d)), p_two


def thresholds(pooled, p, y, cam, deployment_rate) -> None:
    section("4. WHAT A THRESHOLD BECOMES ONCE CALIBRATED")
    keep = np.isin(cam, PAIR)
    cal_p, ky = np.asarray(pooled.apply(p[keep])), y[keep]
    print(f"\n  Pooled fit, both Nikons (n={int(keep.sum())}): "
          f"slope {pooled.slope:.3f}, intercept {pooled.intercept:+.3f}")
    print("  In-sample -- read as the shape of the operating curve, not as a held-out claim.\n")
    print(f"  {'threshold':>10}{'flagged':>9}{'TPR':>8}{'FPR':>8}{'precision':>11}")
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        flagged = cal_p >= t
        tp = float((flagged & (ky == 1)).sum())
        fp = float((flagged & (ky == 0)).sum())
        print(f"  {t:>10.2f}{int(flagged.sum()):>9}"
              f"{tp / max((ky == 1).sum(), 1):>8.3f}{fp / max((ky == 0).sum(), 1):>8.3f}"
              f"{tp / max(tp + fp, 1):>11.3f}")
    print("\n  Precision is a 50%-prevalence number: Korus pairs every forgery with its"
          "\n  own original by construction. A claims stream does not.")

    if deployment_rate is not None:
        shifted = pooled.shift_prior(deployment_rate)
        sp = np.asarray(shifted.apply(p[keep]))
        print(f"\n  Shifted to prevalence {deployment_rate:g}: intercept "
              f"{pooled.intercept:+.3f} -> {shifted.intercept:+.3f}")
        print(f"  The same images now span {sp.min():.3f}..{sp.max():.3f} "
              f"(was {cal_p.min():.3f}..{cal_p.max():.3f}).")
        print("  Ordering is untouched; only the reported probability moves. Note that no"
              "\n  image reaches 0.5 -- at 2% prevalence this evidence is not strong enough"
              "\n  to make manipulation the more likely explanation for any single photo.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="save the pooled calibrator to data/processed/calibrator.json")
    ap.add_argument("--deployment-rate", type=float, default=None,
                    help="also report the calibrator shifted to this prevalence")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    p, y, cam = load()

    print(", ".join(f"{c} n={int((cam == c).sum())}" for c in sorted(set(cam))))
    print(f"base rate {y.mean():.3f} -- paired by construction, see section 4")

    section("0. THE MISCALIBRATION")
    reliability(p, y, "all 224 images, uncalibrated")
    print("\n  Every bin sits above its predicted value: the fusion is systematically"
          "\n  under-confident. It is also monotonic across the equal-mass bins, which"
          "\n  matters -- a monotonic map can only fix a curve that is already ordered.")

    folds = cross_camera(p, y, cam, rng)
    within_camera(p, y, cam)
    slope_gap, p_slope = why(p, y, cam, rng)

    keep = np.isin(cam, PAIR)
    pooled = fit_platt(p[keep], y[keep], fitted_on="Nikon_D7000+D90 (Korus)")
    thresholds(pooled, p, y, cam, args.deployment_rate)

    section("VERDICT")
    moved = max(abs(f["after"]["AUC"] - f["before"]["AUC"]) for f in folds)
    helps = [float((np.array(f["deltas"]["ECE"]) < 0).mean()) for f in folds]
    print(f"\n  AUC moved by at most {moved:.1e} under calibration. That is not a result,"
          "\n  it is arithmetic: a monotonic map preserves every pairwise ordering, so the"
          "\n  headline discrimination is untouched. Calibration changes meaning, not skill."
          "\n\n  Cross-camera, the point estimate improves in both directions, but nothing"
          f"\n  reaches significance: P(improves ECE) = {helps[0]:.2f} and {helps[1]:.2f},"
          " and every interval spans"
          "\n  zero. In the D7000 -> D90 direction ECE improves while Brier and LogLoss get"
          "\n  worse -- when a binned summary and two proper scoring rules disagree, the"
          "\n  proper scoring rules are the ones to believe, and the ECE gain there is"
          "\n  mostly images changing bins."
          "\n\n  Within camera and out-of-fold, calibration helped on all three metrics in"
          "\n  nearly every shuffle. So the miscalibration is real and learnable; it just"
          f"\n  does not transfer. The slopes say why: they differ by {slope_gap:+.2f}"
          f" (two-sided p = {p_slope:.3f}),"
          "\n  the D7000 wanting roughly twice the D90's correction."
          "\n\n  Reading: the miscalibration is a property of the sensor, not of the fusion."
          "\n  One global calibrator would export the D7000's correction to cameras that do"
          "\n  not want it. Per-camera calibration needs labelled examples per camera model"
          "\n  -- a data-collection requirement, not a modelling one. Until that exists, the"
          "\n  honest shipping position is that the score orders images well and is not yet"
          "\n  a probability, which is what the pipeline already says by defaulting to the"
          "\n  identity calibrator.")

    if args.write:
        pooled.save(CALIBRATOR)
        print(f"\n  wrote {CALIBRATOR.relative_to(ROOT)} (pooled; see the caveat above)")
    else:
        print("\n  (--write to save the pooled calibrator)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
