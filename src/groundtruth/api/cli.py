"""Command line entry point: analyse an image and optionally render the overlay."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from ..core.types import ClaimContext, ImageCase
from ..fusion.localisation import peak_regions
from ..pipeline.runner import analyse
from .render import render_verdict


def main() -> int:
    p = argparse.ArgumentParser(prog="groundtruth", description=__doc__)
    p.add_argument("image", type=Path)
    p.add_argument("--claim-id", default="unknown")
    p.add_argument("--claimant-id", default="unknown")
    p.add_argument("--policy-inception", type=date.fromisoformat)
    p.add_argument("--loss-date", type=date.fromisoformat)
    p.add_argument(
        "--render",
        type=Path,
        metavar="OUT.png",
        help="write a three-panel PNG: original | overlay | heatmap",
    )
    args = p.parse_args()

    if not args.image.exists():
        p.error(f"no such file: {args.image}")

    has_context = bool(args.policy_inception or args.loss_date)
    context = (
        ClaimContext(
            claim_id=args.claim_id,
            claimant_id=args.claimant_id,
            policy_inception=args.policy_inception,
            loss_date=args.loss_date,
        )
        if has_context
        else None
    )

    verdict = analyse(ImageCase(image_path=args.image, context=context))

    print(f"\n{args.image.name}")
    print(f"decision:       {verdict.decision.value.upper()}")
    print(f"P(manipulated): {verdict.manipulated_probability:.3f}\n")
    print(verdict.explanation)

    print("\ndetectors:")
    for e in verdict.evidence:
        state = f"{e.score:.2f} @ {e.confidence:.2f}" if e.applicable else "n/a"
        loc = " [map]" if e.heatmap is not None else ""
        print(f"  {e.detector_id:<32}{state:>14}{loc:<6}  {e.explanation}")

    if verdict.heatmap is not None:
        regions = peak_regions(verdict.heatmap)
        print(f"\nlocalised by: {', '.join(verdict.localised_by)}")
        if regions:
            print(f"regions of interest: {len(regions)}")
            for r in regions[:5]:
                x0, y0, x1, y1 = r["bbox"]
                print(
                    f"  bbox=({x0},{y0})-({x1},{y1})  "
                    f"{r['area_fraction']:.2%} of frame  peak={r['peak']:.2f}"
                )
        else:
            print("no contiguous region crossed the reporting threshold")
    else:
        print("\nno detector produced a localisation map")

    if args.render:
        out = render_verdict(args.image, verdict, args.render)
        print(f"\nwrote {out}" if out else "\nnothing to render (no heatmap)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
