"""Embedded preview disagrees with the image it is supposed to preview.

Containers carry a second, smaller copy of the image -- an EXIF thumbnail in a
JPEG, a thumbnail item in a HEIC, a reduced-resolution SubIFD in a TIFF, an
embedded JPEG in most RAW formats. Many editors rewrite the main image and leave
that copy untouched.

When they do, the preview is a photograph of the pre-edit original, and the
disagreement between the two localises the edit exactly. Unlike every other
detector here, this one does not infer that something was changed from statistical
traces -- it shows both versions.

Its blind spot is total: any tool that regenerates the preview correctly defeats
it completely, and messaging apps strip metadata wholesale. High precision, low
recall. One signal among many.
"""

from __future__ import annotations

from ...core.detector import Detector, register
from ...core.types import Evidence, ImageCase, Tier
from ...recovery.reconstruct import Fidelity, reconstruct

# Below this, the difference is consistent with independent re-encoding of the
# preview rather than an edit to the image.
_MIN_CHANGED_FRACTION = 0.004

# Above this, the whole frame differs, which points at a global operation
# (exposure, colour grade, full re-render) rather than a localised edit.
_GLOBAL_CHANGE_FRACTION = 0.55


@register
class PreviewMismatchDetector(Detector):
    """Compare the container's embedded preview against the current image."""

    id = "metadata.preview_mismatch"
    tier = Tier.METADATA
    localises = True
    cost = 2

    def _run(self, case: ImageCase) -> Evidence:
        recon = reconstruct(case.image_path)
        if recon is None:
            return Evidence.not_applicable(
                self.id, self.tier, "container carries no embedded preview"
            )

        changed = recon.changed_fraction
        details = {
            "preview_source": recon.source,
            "preview_size": list(recon.preview_size),
            "changed_fraction": round(changed, 5),
            "cropped": recon.cropped,
            "fidelity": recon.fidelity.value,
            "regions": recon.regions[:5],
        }

        if changed < _MIN_CHANGED_FRACTION and not recon.cropped:
            return Evidence(
                detector_id=self.id,
                tier=self.tier,
                applicable=True,
                score=0.15,
                confidence=0.75,
                explanation=(
                    f"embedded preview ({recon.preview_size[0]}x{recon.preview_size[1]}) "
                    f"matches the current image"
                ),
                details=details,
            )

        if changed > _GLOBAL_CHANGE_FRACTION:
            return Evidence(
                detector_id=self.id,
                tier=self.tier,
                applicable=True,
                score=0.60,
                confidence=0.35,
                explanation=(
                    f"embedded preview differs across {changed:.0%} of the frame; "
                    f"consistent with a global re-render or colour change rather than a "
                    f"localised edit"
                ),
                heatmap=recon.difference,
                details=details,
            )

        # A low-resolution preview can localise a region but cannot speak to fine
        # detail, so confidence is capped when the stored copy was tiny.
        confidence = 0.85 if recon.fidelity is Fidelity.RECOVERED else 0.6

        parts = [
            (
                f"embedded preview disagrees with the current image over {changed:.1%} "
                f"of the frame -- the preview shows the pre-edit original"
            )
        ]
        if recon.cropped:
            parts.append("aspect ratio also changed, so the image was cropped")
        if recon.regions:
            x0, y0, x1, y1 = recon.regions[0]["bbox"]
            parts.append(f"largest changed region at ({x0},{y0})-({x1},{y1})")

        return Evidence(
            detector_id=self.id,
            tier=self.tier,
            applicable=True,
            score=float(min(0.96, 0.75 + 2.0 * changed)),
            confidence=confidence,
            # Scaled against 5% of frame, not 10%. This is a DIRECT comparison
            # against a known original rather than a statistical inference, so a
            # localised change of a few percent is already an unambiguous effect --
            # the same fraction means much more here than it does to a detector
            # inferring manipulation from noise statistics.
            effect_size=float(min(1.0, changed / 0.05)),
            explanation="; ".join(parts),
            heatmap=recon.difference,
            details=details,
        )
