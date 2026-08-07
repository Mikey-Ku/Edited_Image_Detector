"""Copy-move: a region duplicated from elsewhere in the same photograph.

This is the one manipulation that every statistical detector in this project is
structurally blind to. Cloning something out, or duplicating it, copies pixels
from *this* sensor, through *this* lens, with *this* compression history. There is
nothing foreign to find. Noise matches, the grid matches, the demosaicing pattern
matches -- because it is genuinely the same photograph.

What gives it away is not statistics but geometry: two regions that are the same
content in different places, related by a consistent transform.

The pipeline is the standard one from the forensics literature:

1. **SIFT keypoints.** Robust to the rotation, scaling and recompression an
   attacker applies to hide the duplication.
2. **g2NN matching within the image.** A keypoint's nearest neighbour is matched
   only if it stands clearly apart from the next candidate, which suppresses the
   dense mutual matches that repetitive texture (brickwork, foliage, gravel)
   produces.
3. **Spatial exclusion.** Neighbouring keypoints on the same object match each
   other trivially, so pairs closer together than a minimum distance are dropped.
   Without this, every textured surface reports itself as copy-moved.
4. **RANSAC on the offset.** A real copy-move has *one* consistent displacement
   shared by many pairs. Coincidental matches scatter. Requiring a consensus is
   what separates a duplication from a wall of similar-looking bricks.
"""

from __future__ import annotations

import numpy as np

from ...core.detector import Detector, register
from ...core.types import Evidence, ImageCase, Tier

# Keypoints closer than this cannot be a meaningful duplication -- they are two
# corners of the same object. Scaled to the image so it holds at any resolution.
_MIN_SEPARATION_FRAC = 0.02
_MIN_SEPARATION_PX = 32

# g2NN ratio. Lower is stricter; 0.5 is conservative and keeps repetitive texture
# from flooding the match set.
_RATIO = 0.5

# A duplication must be carried by this many pairs sharing one displacement.
_MIN_CONSENSUS = 8

# Displacements within this distance of each other count as the same transform.
_OFFSET_TOLERANCE = 12.0

_MAX_KEYPOINTS = 4000

# Radius, in pixels, painted around each matched keypoint when localising.
_PAINT = 18


def _sift_keypoints(gray: np.ndarray):
    import cv2

    sift = cv2.SIFT_create(nfeatures=_MAX_KEYPOINTS)
    kp, desc = sift.detectAndCompute(gray, None)
    return kp, desc


def _matched_pairs(kp, desc, min_sep: float) -> list[tuple[int, int]]:
    """Self-matched keypoint pairs surviving g2NN and spatial exclusion."""
    import cv2

    if desc is None or len(desc) < 4:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    # k=3 so a keypoint can skip its own trivial self-match at rank 0.
    knn = matcher.knnMatch(desc, desc, k=min(4, len(desc)))

    pts = np.array([k.pt for k in kp], dtype=np.float32)
    pairs: list[tuple[int, int]] = []
    for group in knn:
        candidates = [m for m in group if m.trainIdx != m.queryIdx]
        if len(candidates) < 2:
            continue
        best, second = candidates[0], candidates[1]
        if second.distance <= 1e-6 or best.distance / second.distance > _RATIO:
            continue
        i, j = best.queryIdx, best.trainIdx
        if np.linalg.norm(pts[i] - pts[j]) < min_sep:
            continue
        pairs.append((i, j) if i < j else (j, i))
    return list(dict.fromkeys(pairs))


def _consensus(pts: np.ndarray, pairs: list[tuple[int, int]]) -> list[tuple]:
    """Group pairs by shared displacement; keep groups big enough to be real."""
    if not pairs:
        return []
    offsets = np.array([pts[j] - pts[i] for i, j in pairs], dtype=np.float32)
    # Direction is arbitrary -- source and destination are interchangeable.
    offsets = np.where(offsets[:, :1] < 0, -offsets, offsets)

    used = np.zeros(len(pairs), dtype=bool)
    groups: list[tuple] = []
    for a in range(len(pairs)):
        if used[a]:
            continue
        close = ~used & (np.linalg.norm(offsets - offsets[a], axis=1) < _OFFSET_TOLERANCE)
        if int(close.sum()) >= _MIN_CONSENSUS:
            used |= close
            members = [pairs[k] for k in np.where(close)[0]]
            groups.append((tuple(np.round(offsets[close].mean(axis=0), 1)), members))
    return sorted(groups, key=lambda g: -len(g[1]))


@register
class CopyMoveDetector(Detector):
    """Find content duplicated from elsewhere in the same photograph."""

    id = "geometric.copy_move"
    tier = Tier.GEOMETRIC
    localises = True
    cost = 4

    def applies_to(self, case: ImageCase) -> tuple[bool, str]:
        try:
            import cv2  # noqa: F401
        except ImportError:
            return False, "copy-move detection requires opencv"
        h, w = case.pixels().shape[:2]
        if min(h, w) < 128:
            return False, f"image too small for keypoint matching ({w}x{h})"
        return True, ""

    def _run(self, case: ImageCase) -> Evidence:
        import cv2

        rgb = case.pixels()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        kp, desc = _sift_keypoints(gray)
        if desc is None or len(kp) < 16:
            return Evidence.not_applicable(
                self.id, self.tier, f"only {len(kp) if kp else 0} keypoints; too few to match"
            )

        min_sep = max(_MIN_SEPARATION_PX, _MIN_SEPARATION_FRAC * max(h, w))
        pairs = _matched_pairs(kp, desc, min_sep)
        pts = np.array([k.pt for k in kp], dtype=np.float32)
        groups = _consensus(pts, pairs)

        details: dict[str, object] = {
            "keypoints": len(kp),
            "matched_pairs": len(pairs),
            "consensus_groups": len(groups),
            "min_separation_px": round(min_sep, 1),
        }

        # Confidence rests on having had enough keypoints to make the search
        # meaningful. A near-featureless frame yields none and proves nothing.
        confidence = float(np.clip(0.85 * min(1.0, len(kp) / 600.0), 0.15, 0.85))

        if not groups:
            return Evidence(
                detector_id=self.id,
                tier=self.tier,
                applicable=True,
                score=0.22,
                confidence=confidence,
                effect_size=0.0,
                explanation=(
                    f"no duplicated region found among {len(kp)} keypoints "
                    f"({len(pairs)} candidate matches, none sharing a displacement)"
                ),
                details=details,
            )

        offset, members = groups[0]
        heat = np.zeros((h, w), dtype=np.float32)
        yy, xx = np.mgrid[0:h, 0:w]
        for i, j in members:
            for x, y in (pts[i], pts[j]):
                heat[(xx - x) ** 2 + (yy - y) ** 2 <= _PAINT**2] = 1.0

        # Plain floats, not the numpy scalars `offset` arrives as. `details` is
        # published straight out of the JSON API, and a numpy float32 in here took
        # the whole /api/analyse response down with a 500 -- but only on images
        # where copy-move actually found something, which is to say only on the
        # images worth demonstrating.
        details["displacement_px"] = [float(v) for v in offset]
        details["supporting_pairs"] = len(members)

        return Evidence(
            detector_id=self.id,
            tier=self.tier,
            applicable=True,
            score=float(min(0.95, 0.68 + 0.012 * len(members))),
            confidence=confidence,
            effect_size=float(min(1.0, len(members) / 40.0)),
            explanation=(
                f"{len(members)} keypoint pairs share a single displacement of "
                f"({offset[0]:.0f}, {offset[1]:.0f}) px -- a region of this image "
                f"appears twice"
            ),
            heatmap=heat,
            details=details,
        )
