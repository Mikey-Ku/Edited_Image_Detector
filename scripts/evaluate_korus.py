"""Evaluate the pipeline against the Korus Realistic Tampering Dataset.

Real photographs from real cameras, manipulated by hand in GIMP and Affinity
Photo, with pixel-exact ground-truth masks. Everything measured before this point
was measured against manipulations we generated ourselves, which proves detectors
fire in the right place but proves nothing about real images.

Reported metrics:

- **Detection rate** at a fixed operating point, and the score distributions that
  produced it, so the threshold can be moved without re-running.
- **False positive rate** on the matched pristine originals -- the same scenes,
  the same cameras, unmanipulated. This is the number that decides whether the
  system is deployable, and it is the one most easily hidden by reporting accuracy.
- **Localisation** as hit rate and IoU against the true mask, computed only over
  images the system flagged: localisation quality on images it missed is not a
  meaningful quantity.
- **Applicability**, per detector. These are TIFFs, so every JPEG-dependent
  detector should correctly abstain. If one does not, that is a bug.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundtruth import Decision, ImageCase, analyse  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1] / "data/interim/korus/data-images"
CAMERAS = ["Nikon_D7000", "Nikon_D90", "Canon_60D"]


def _mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"))
    return arr > 127


def _hit_rate(heat: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> float:
    pred = heat >= threshold
    return float((pred & mask).sum() / pred.sum()) if pred.any() else 0.0


def _iou(heat: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> float:
    pred = heat >= threshold
    union = int((pred | mask).sum())
    return float((pred & mask).sum() / union) if union else 0.0


def _pairs() -> list[tuple[str, Path, Path | None, Path | None]]:
    """(camera, tampered, pristine, ground-truth) for every recovered triple."""
    out = []
    for cam in CAMERAS:
        base = ROOT / cam
        tdir, pdir, gdir = (
            base / "tampered-realistic",
            base / "pristine",
            base / "ground-truth",
        )
        if not tdir.is_dir():
            continue
        for t in sorted(tdir.glob("*.TIF")):
            gt = gdir / f"{t.stem}.PNG"
            pr = pdir / t.name
            out.append((cam, t, pr if pr.exists() else None, gt if gt.exists() else None))
    return out


def main() -> int:
    pairs = _pairs()
    if not pairs:
        print("no data found -- run scripts/salvage_zip.py first", file=sys.stderr)
        return 1

    print(f"{len(pairs)} tampered images across {len({p[0] for p in pairs})} cameras\n")

    records: list[dict] = []
    applicability: dict[str, dict[str, int]] = {}
    started = time.time()

    for i, (cam, tampered, pristine, gt) in enumerate(pairs, 1):
        for label, path in (("tampered", tampered), ("pristine", pristine)):
            if path is None:
                continue
            verdict = analyse(ImageCase(image_path=path))

            for ev in verdict.evidence:
                slot = applicability.setdefault(
                    ev.detector_id, {"applied": 0, "skipped": 0}
                )
                slot["applied" if ev.applicable else "skipped"] += 1

            rec = {
                "camera": cam,
                "name": path.name,
                "label": label,
                "probability": verdict.manipulated_probability,
                "decision": verdict.decision.value,
                "localised_by": verdict.localised_by,
            }
            if label == "tampered" and gt is not None and verdict.heatmap is not None:
                mask = _mask(gt)
                if mask.shape == verdict.heatmap.shape:
                    rec["hit_rate"] = _hit_rate(verdict.heatmap, mask)
                    rec["iou"] = _iou(verdict.heatmap, mask)
                    rec["mask_fraction"] = float(mask.mean())
            records.append(rec)

        if i % 10 == 0 or i == len(pairs):
            print(f"  {i}/{len(pairs)} ({time.time() - started:.0f}s)", flush=True)

    tam = [r for r in records if r["label"] == "tampered"]
    pri = [r for r in records if r["label"] == "pristine"]

    def rate(rows, *decisions):
        return sum(r["decision"] in decisions for r in rows) / len(rows) if rows else 0.0

    print("\n" + "=" * 68)
    print("DETECTION")
    print("=" * 68)
    print(f"  tampered flagged            {rate(tam, 'flag'):.1%}  ({len(tam)} images)")
    print(f"  tampered flagged or routed  {rate(tam, 'flag', 'route_to_human'):.1%}")
    print(f"  pristine FALSELY flagged    {rate(pri, 'flag'):.1%}  ({len(pri)} images)")
    print(f"  pristine auto-cleared       {rate(pri, 'auto_clear'):.1%}")

    for name, rows in (("tampered", tam), ("pristine", pri)):
        if not rows:
            continue
        p = np.array([r["probability"] for r in rows])
        print(
            f"\n  {name} P(manipulated): "
            f"min {p.min():.3f}  p25 {np.percentile(p, 25):.3f}  "
            f"median {np.median(p):.3f}  p75 {np.percentile(p, 75):.3f}  max {p.max():.3f}"
        )

    loc = [r for r in tam if "hit_rate" in r and r["decision"] != "auto_clear"]
    print("\n" + "=" * 68)
    print(f"LOCALISATION  ({len(loc)} flagged/routed images with a heatmap)")
    print("=" * 68)
    if loc:
        hr = np.array([r["hit_rate"] for r in loc])
        iou = np.array([r["iou"] for r in loc])
        mf = np.array([r["mask_fraction"] for r in loc])
        print(f"  hit rate   median {np.median(hr):.3f}   mean {hr.mean():.3f}")
        print(f"  IoU        median {np.median(iou):.3f}   mean {iou.mean():.3f}")
        print(f"  true manipulated area: median {np.median(mf):.2%} of frame")
        print(f"  hit rate > 0.5 on {(hr > 0.5).mean():.1%} of them")
    else:
        print("  no flagged image produced a localisation map")

    print("\n" + "=" * 68)
    print("DETECTOR APPLICABILITY")
    print("=" * 68)
    for det, counts in sorted(applicability.items()):
        total = counts["applied"] + counts["skipped"]
        print(f"  {det:<34} applied {counts['applied']:>4}/{total}")

    out = Path(__file__).resolve().parents[1] / "data/processed/korus_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
