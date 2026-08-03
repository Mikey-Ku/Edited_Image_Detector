"""Run the benchmark matrix and report the detection envelope.

The output is a grid, not a number. A single accuracy figure hides the thing that
actually matters -- *which* manipulations are caught, at what size, after what
laundering -- and it lets a change that helps one case while breaking three look
like an improvement.

Two scores per cell, and both are needed:

- **detect**: did the system flag it (and, for the controls, did it correctly stay
  quiet). A detector that flags everything scores perfectly here alone.
- **locate**: of the pixels it flagged, how many were inside the real edit.
  Detecting for the wrong reason scores zero, which is what caught the sharpness
  detector firing on a crumpled bumper 7 sigma away from the actual manipulation.
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.detector import Detector, all_detectors
from ..core.types import Decision, ImageCase
from ..pipeline.runner import analyse
from .manipulate import OPERATIONS, make


@dataclass
class CellResult:
    operation: str
    size: float
    laundering: str
    flagged: bool
    probability: float
    hit_rate: float
    correct: bool
    fired: list[str]


def _hit_rate(heat: np.ndarray | None, mask: np.ndarray) -> float:
    if heat is None or not mask.any():
        return 0.0
    if heat.shape != mask.shape:
        return 0.0
    pred = heat >= 0.5
    return float((pred & mask).sum() / pred.sum()) if pred.any() else 0.0


def run(
    bases: list[Path],
    out_dir: Path,
    operations: tuple[str, ...] = OPERATIONS,
    sizes: tuple[float, ...] = (0.005, 0.02, 0.08, 0.25),
    launderings: tuple[str, ...] = ("jpeg95",),
    detectors: list[Detector] | None = None,
    keep_files: bool = False,
) -> list[CellResult]:
    """Generate every cell, analyse it, and return the results."""
    detectors = detectors if detectors is not None else all_detectors()
    results: list[CellResult] = []

    for op in operations:
        # Controls have no manipulated region, so size is meaningless for them.
        cell_sizes = (0.0,) if op in {"pristine", "global_tone"} else sizes
        for size in cell_sizes:
            for how in launderings:
                for i, base in enumerate(bases):
                    # A splice needs foreign content, so the donor is another
                    # photograph from the set -- never the base itself.
                    donor = bases[(i + 1) % len(bases)] if len(bases) > 1 else None
                    fx = make(
                        base, out_dir, op, size, how, seed=i * 17 + 3, donor_path=donor
                    )
                    verdict = analyse(
                        ImageCase(image_path=fx.path), detectors=detectors
                    )
                    flagged = verdict.decision is Decision.FLAG
                    correct = (not flagged) if fx.is_control else flagged
                    results.append(
                        CellResult(
                            operation=op,
                            size=size,
                            laundering=how,
                            flagged=flagged,
                            probability=verdict.manipulated_probability,
                            hit_rate=_hit_rate(verdict.heatmap, fx.mask),
                            correct=correct,
                            fired=[
                                e.detector_id
                                for e in verdict.evidence
                                if e.applicable and e.score > 0.6
                            ],
                        )
                    )
                    if not keep_files:
                        fx.path.unlink(missing_ok=True)

    if not keep_files:
        shutil.rmtree(out_dir, ignore_errors=True)
    return results


def envelope(results: list[CellResult], by: str = "size") -> str:
    """Render the results as a readable grid."""
    axis = sorted({getattr(r, by) for r in results})
    ops = list(dict.fromkeys(r.operation for r in results))

    grouped: dict[tuple[str, object], list[CellResult]] = defaultdict(list)
    for r in results:
        grouped[(r.operation, getattr(r, by))].append(r)

    head = f"{'operation':<16}" + "".join(
        f"{(f'{a:.1%}' if isinstance(a, float) else str(a)):>13}" for a in axis
    )
    lines = [head, "-" * len(head)]

    for op in ops:
        row = f"{op:<16}"
        for a in axis:
            cells = grouped.get((op, a), [])
            if not cells:
                row += f"{'-':>13}"
                continue
            rate = sum(c.correct for c in cells) / len(cells)
            control = cells[0].operation in {"pristine", "global_tone"}
            if control:
                row += f"{f'{rate:.0%} quiet':>13}"
            else:
                loc = np.mean([c.hit_rate for c in cells])
                row += f"{f'{rate:.0%}/{loc:.2f}':>13}"
        lines.append(row)

    detected = [r for r in results if r.operation not in {"pristine", "global_tone"}]
    controls = [r for r in results if r.operation in {"pristine", "global_tone"}]
    lines += [
        "",
        "cells show  detected% / mean localisation   (controls show % correctly quiet)",
        "",
        (
            f"manipulations detected : {sum(r.correct for r in detected)}"
            f"/{len(detected)}  "
            f"({sum(r.correct for r in detected) / max(len(detected), 1):.0%})"
        ),
        (
            f"controls left alone    : {sum(r.correct for r in controls)}"
            f"/{len(controls)}  "
            f"({sum(r.correct for r in controls) / max(len(controls), 1):.0%})"
        ),
    ]
    return "\n".join(lines)


def by_detector(results: list[CellResult]) -> str:
    """Which detector actually fires on which operation."""
    ops = list(dict.fromkeys(r.operation for r in results))
    dets = sorted({d for r in results for d in r.fired})
    if not dets:
        return "no detector fired on any cell"

    width = max(len(d) for d in dets) + 2
    head = f"{'detector':<{width}}" + "".join(f"{o[:11]:>13}" for o in ops)
    lines = [head, "-" * len(head)]
    for d in dets:
        row = f"{d:<{width}}"
        for op in ops:
            cells = [r for r in results if r.operation == op]
            hit = sum(d in r.fired for r in cells)
            row += f"{f'{hit}/{len(cells)}':>13}"
        lines.append(row)
    return "\n".join(lines)
