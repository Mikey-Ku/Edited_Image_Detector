"""Command line entry point: analyse a single image."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from ..core.types import ClaimContext, ImageCase
from ..pipeline.runner import analyse


def main() -> int:
    p = argparse.ArgumentParser(prog="groundtruth", description=__doc__)
    p.add_argument("image", type=Path)
    p.add_argument("--claim-id", default="unknown")
    p.add_argument("--claimant-id", default="unknown")
    p.add_argument("--policy-inception", type=date.fromisoformat)
    p.add_argument("--loss-date", type=date.fromisoformat)
    args = p.parse_args()

    if not args.image.exists():
        p.error(f"no such file: {args.image}")

    context = ClaimContext(
        claim_id=args.claim_id,
        claimant_id=args.claimant_id,
        policy_inception=args.policy_inception,
        loss_date=args.loss_date,
    )
    verdict = analyse(ImageCase(image_path=args.image, context=context))

    print(f"\n{args.image.name}")
    print(f"decision: {verdict.decision.value.upper()}")
    print(f"P(manipulated): {verdict.manipulated_probability:.3f}\n")
    print(verdict.explanation)
    print("\ndetectors:")
    for e in verdict.evidence:
        state = f"{e.score:.2f} @ {e.confidence:.2f}" if e.applicable else "n/a"
        print(f"  {e.detector_id:<38} {state:>14}   {e.explanation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
