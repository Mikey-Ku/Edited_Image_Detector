"""Ground Truth -- image manipulation detection for insurance claims."""

from . import detectors  # noqa: F401  -- import registers every detector
from .core.types import ClaimContext, Decision, Evidence, GeoPoint, ImageCase, Tier, Verdict
from .pipeline.runner import analyse

__all__ = [
    "ClaimContext", "Decision", "Evidence", "GeoPoint",
    "ImageCase", "Tier", "Verdict", "analyse",
]
__version__ = "0.1.0"
