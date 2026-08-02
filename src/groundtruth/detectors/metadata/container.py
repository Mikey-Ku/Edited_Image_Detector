"""Container identity: is the file what it claims to be?

An original photograph off a phone is a JPEG or a HEIC whose extension matches its
bytes. A file named ``damage.jpg`` that is actually a PNG has been through a
conversion the claimant did not mention, and conversions happen inside editors.

This is weak evidence on its own -- plenty of innocent pipelines rewrite files --
but it is nearly free to compute and it sharpens what the other detectors mean. A
PNG cannot be assessed by compression forensics at all, and knowing that is more
useful than silently getting no signal.
"""

from __future__ import annotations

from ...core.detector import Detector, register
from ...core.image_io import Container
from ...core.types import Evidence, ImageCase, Tier


@register
class ContainerIdentityDetector(Detector):
    """Compare the declared extension against the actual container."""

    id = "metadata.container_identity"
    tier = Tier.METADATA
    localises = False
    cost = 1

    def _run(self, case: ImageCase) -> Evidence:
        info = case.container
        details = {
            "actual": info.actual.value,
            "claimed": info.claimed.value,
            "lossy": info.actual.lossy,
            "block_compressed": info.actual.block_compressed,
        }

        if info.actual is Container.UNKNOWN:
            return Evidence.not_applicable(
                self.id, self.tier, "container not recognised from magic bytes"
            )

        if info.extension_mismatch:
            return Evidence(
                detector_id=self.id,
                tier=self.tier,
                applicable=True,
                score=0.68,
                confidence=0.55,
                explanation=(
                    f"file is named .{info.claimed.value} but the bytes are "
                    f"{info.actual.value} -- the file has been converted or re-saved"
                ),
                details=details,
            )

        if not info.actual.lossy:
            return Evidence(
                detector_id=self.id,
                tier=self.tier,
                applicable=True,
                score=0.55,
                confidence=0.25,
                explanation=(
                    f"{info.actual.value} is a lossless container; cameras do not "
                    f"produce it, so this file was exported by software"
                ),
                details=details,
            )

        # A lossy container whose name matches its bytes is what an unmodified
        # camera file looks like. That is not evidence of anything -- it is the
        # default. Reporting it as weak exculpatory evidence would dilute every
        # fused score by a constant, which is worse than staying silent.
        return Evidence.not_applicable(
            self.id,
            self.tier,
            f"container is {info.actual.value} and matches its name; nothing anomalous",
        )
