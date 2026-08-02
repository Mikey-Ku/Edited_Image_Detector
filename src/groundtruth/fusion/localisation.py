"""Combine per-detector localisation maps into a single overlay.

Detectors produce heatmaps at whatever resolution their method works at -- the
thumbnail detector at preview size, ELA at full resolution, the noise detector at
block granularity. They are resampled to a common frame and pooled.

Pooling is confidence-weighted *mean*, not maximum. Maximum would let a single
low-confidence detector paint the whole frame red, and the entire point of running
many detectors is that agreement between independent methods is what carries
weight. A region is suspicious because several methods that look at different
physical properties all point at it.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..core.types import Evidence

# Detectors below this confidence do not contribute to the overlay at all.
MIN_CONFIDENCE = 0.10


def _resample(heat: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a [0,1] map to (H, W) with bilinear interpolation."""
    if heat.shape == shape:
        return heat.astype(np.float32)
    img = Image.fromarray((np.clip(heat, 0, 1) * 255).astype(np.uint8))
    resized = img.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def fuse_heatmaps(
    evidence: list[Evidence], shape: tuple[int, int]
) -> tuple[np.ndarray | None, list[str]]:
    """Pool localisation maps.

    Returns ``(heatmap, contributing_detector_ids)``. The heatmap is ``None`` when
    no applicable detector produced one -- which is a meaningful outcome and must
    not be faked with an all-zero map.
    """
    maps: list[np.ndarray] = []
    weights: list[float] = []
    contributors: list[str] = []

    for ev in evidence:
        if not ev.applicable or ev.heatmap is None:
            continue
        if ev.confidence < MIN_CONFIDENCE:
            continue
        # Guard: a detector that reports an entirely blank map has nothing to
        # contribute, and averaging it in only dilutes detectors that do.
        if not np.any(ev.heatmap > 0):
            continue
        maps.append(_resample(ev.heatmap, shape))
        # Weight by confidence AND by how suspicious this detector found the image.
        # A confident "nothing here" should not drag the overlay toward its own noise.
        weights.append(ev.confidence * max(ev.score, 0.05))
        contributors.append(ev.detector_id)

    if not maps:
        return None, []

    stack = np.stack(maps)
    w = np.asarray(weights, dtype=np.float32).reshape(-1, 1, 1)
    pooled = (stack * w).sum(axis=0) / w.sum()
    return np.clip(pooled, 0.0, 1.0), contributors


def peak_regions(
    heat: np.ndarray, threshold: float = 0.5, min_area: int = 64
) -> list[dict]:
    """Connected components above ``threshold``, as bounding boxes.

    Gives the adjuster "look at this rectangle" rather than a diffuse wash of
    colour. Uses a simple flood fill -- no scipy.ndimage dependency, and the
    region count here is small by construction.
    """
    mask = heat >= threshold
    if not mask.any():
        return []

    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    regions: list[dict] = []

    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
            pts: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                pts.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(pts) < min_area:
                continue
            ys = [p[0] for p in pts]
            xs = [p[1] for p in pts]
            regions.append(
                {
                    "bbox": [min(xs), min(ys), max(xs), max(ys)],
                    "area_px": len(pts),
                    "area_fraction": round(len(pts) / (h * w), 5),
                    "peak": round(float(heat[mask & seen].max()), 3),
                }
            )

    return sorted(regions, key=lambda r: r["area_px"], reverse=True)
