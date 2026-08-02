"""Claim-context consistency: does the photograph agree with the claim?

The cheapest and most decisive detector in the system, and it uses no machine
learning whatsoever. A pixel-perfect, entirely authentic photograph is still
fraud if it was taken eleven days before the policy incepted -- the damage
predates the coverage.

Worth remembering while building the sophisticated tiers: the smartest system is
not always the most sophisticated one.
"""

from __future__ import annotations

from datetime import date, datetime

from PIL import ExifTags, Image

from ...core.detector import Detector, register
from ...core.types import Evidence, GeoPoint, ImageCase, Tier

_DATETIME_ORIGINAL = 36867
_GPS_IFD = 34853

# How far the photo's GPS may sit from the reported loss location before we care.
_LOCATION_TOLERANCE_KM = 50.0


def _read_capture_datetime(img: Image.Image) -> datetime | None:
    exif = img.getexif()
    raw = exif.get(_DATETIME_ORIGINAL)
    if raw is None:
        ifd = exif.get_ifd(ExifTags.IFD.Exif) if hasattr(ExifTags, "IFD") else {}
        raw = ifd.get(_DATETIME_ORIGINAL)
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def _read_gps(img: Image.Image) -> GeoPoint | None:
    exif = img.getexif()
    try:
        gps = exif.get_ifd(_GPS_IFD)
    except Exception:
        return None
    if not gps:
        return None

    def to_deg(dms, ref: str) -> float:
        d, m, s = (float(x) for x in dms)
        val = d + m / 60.0 + s / 3600.0
        return -val if ref in {"S", "W"} else val

    try:
        lat = to_deg(gps[2], gps[1])
        lon = to_deg(gps[4], gps[3])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    return GeoPoint(lat, lon)


@register
class PolicyConsistencyDetector(Detector):
    """Compare EXIF capture time and location against what the claim asserts."""

    id = "context.policy_consistency"
    tier = Tier.CONTEXT
    localises = False
    cost = 1

    def applies_to(self, case: ImageCase) -> tuple[bool, str]:
        if case.context is None:
            return False, "no claim context attached"
        ctx = case.context
        if ctx.policy_inception is None and ctx.loss_date is None and ctx.loss_location is None:
            return False, "claim context carries no date or location to check against"
        return True, ""

    def _run(self, case: ImageCase) -> Evidence:
        ctx = case.context
        assert ctx is not None

        with Image.open(case.image_path) as img:
            captured = _read_capture_datetime(img)
            gps = _read_gps(img)

        findings: list[str] = []
        details: dict[str, object] = {}
        score = 0.5
        confidence = 0.0

        if captured is None:
            details["capture_datetime"] = None
            findings.append("no capture timestamp in EXIF")
            # Absent EXIF is weak evidence of tampering on its own -- stripping it
            # is trivial and messaging apps do it routinely. Nudge, don't accuse.
            score = 0.55
            confidence = 0.15
        else:
            shot: date = captured.date()
            details["capture_datetime"] = captured.isoformat()

            if ctx.policy_inception and shot < ctx.policy_inception:
                days = (ctx.policy_inception - shot).days
                findings.append(
                    f"photograph was taken {days} days BEFORE the policy incepted "
                    f"({shot} vs {ctx.policy_inception}) -- the damage predates coverage"
                )
                details["days_before_inception"] = days
                # Dispositive. This is not a probabilistic signal.
                score = 0.98
                confidence = 0.95

            elif ctx.loss_date and shot < ctx.loss_date:
                days = (ctx.loss_date - shot).days
                findings.append(
                    f"photograph predates the reported loss date by {days} days "
                    f"({shot} vs {ctx.loss_date})"
                )
                details["days_before_loss"] = days
                score = 0.9
                confidence = 0.85

            elif ctx.loss_date and (shot - ctx.loss_date).days > 90:
                days = (shot - ctx.loss_date).days
                findings.append(f"photograph taken {days} days after the reported loss")
                details["days_after_loss"] = days
                score = 0.65
                confidence = 0.4

            else:
                findings.append("capture timestamp is consistent with the claim")
                score = 0.35
                confidence = 0.5

        if gps and ctx.loss_location:
            km = gps.km_to(ctx.loss_location)
            details["gps_distance_km"] = round(km, 1)
            if km > _LOCATION_TOLERANCE_KM:
                findings.append(
                    f"photograph GPS is {km:.0f} km from the reported loss location"
                )
                score = max(score, 0.9)
                confidence = max(confidence, 0.85)
            else:
                findings.append(f"photograph GPS is {km:.0f} km from the reported loss")
                confidence = max(confidence, 0.6)

        return Evidence(
            detector_id=self.id,
            tier=self.tier,
            applicable=True,
            score=score,
            confidence=confidence,
            explanation="; ".join(findings),
            details=details,
        )
