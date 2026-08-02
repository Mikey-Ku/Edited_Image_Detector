"""Core types.

Design note: `Evidence` separates *score* from *confidence* from *applicable*.
These are three different things and collapsing them is the most common way a
multi-detector system quietly produces nonsense:

    applicable = False  ->  this detector cannot speak about this input at all
    score      = 0.5    ->  this detector looked and found nothing conclusive
    confidence = 0.1    ->  this detector looked, has an opinion, but distrusts it

A detector that can't run must be *excluded* from fusion, not folded in as neutral.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .image_io import Container, ContainerInfo, load_rgb
from .image_io import inspect as inspect_container


class Tier(str, Enum):
    """Which physical or statistical property the detector exploits."""

    METADATA = "metadata"
    COMPRESSION = "compression"
    SENSOR = "sensor"
    GEOMETRIC = "geometric"
    GENERATIVE = "generative"
    CONTEXT = "context"


class Decision(str, Enum):
    """Three outcomes, not two. See docs/DESIGN.md."""

    AUTO_CLEAR = "auto_clear"
    FLAG = "flag"
    ROUTE_TO_HUMAN = "route_to_human"


@dataclass(frozen=True)
class GeoPoint:
    lat: float
    lon: float

    def km_to(self, other: GeoPoint) -> float:
        """Haversine distance in kilometres."""
        r = 6371.0
        p1, p2 = np.radians(self.lat), np.radians(other.lat)
        dp = p2 - p1
        dl = np.radians(other.lon - self.lon)
        a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
        return float(2 * r * np.arcsin(np.sqrt(a)))


@dataclass
class ClaimContext:
    """What the claim asserts about the photograph.

    This is the layer that turns image forensics into fraud detection. A
    pixel-perfect authentic photo is still fraud if it was taken before the
    policy existed.
    """

    claim_id: str
    claimant_id: str
    policy_inception: date | None = None
    loss_date: date | None = None
    loss_location: GeoPoint | None = None
    reported_peril: str | None = None  # "hail", "collision", "water", ...
    claimed_amount: float | None = None


@dataclass
class ImageCase:
    """One image plus everything the claim says about it."""

    image_path: Path
    context: ClaimContext | None = None
    _pixels: np.ndarray | None = field(default=None, repr=False)
    _container: ContainerInfo | None = field(default=None, repr=False)

    @property
    def suffix(self) -> str:
        return self.image_path.suffix.lower()

    @property
    def container(self) -> ContainerInfo:
        """What the file actually is, decided by magic bytes rather than by name."""
        if self._container is None:
            self._container = inspect_container(self.image_path)
        return self._container

    @property
    def is_jpeg(self) -> bool:
        return self.container.actual is Container.JPEG

    @property
    def is_lossy(self) -> bool:
        """Whether compression forensics can say anything about this container."""
        return self.container.actual.lossy

    def pixels(self) -> np.ndarray:
        """Lazily decoded RGB array, cached across detectors."""
        if self._pixels is None:
            self._pixels = load_rgb(self.image_path)
        return self._pixels


@dataclass
class Evidence:
    """One detector's finding."""

    detector_id: str
    tier: Tier
    applicable: bool
    score: float = 0.5
    """P(manipulated) according to this detector alone. 0.5 = no information."""

    confidence: float = 0.0
    """How much this detector trusts itself *on this input*. Fusion weights by this."""

    explanation: str = ""
    """Human-readable, adjuster-facing. Not a number."""

    heatmap: np.ndarray | None = field(default=None, repr=False)
    """Optional HxW float array in [0,1] localising the suspected manipulation."""

    details: dict[str, Any] = field(default_factory=dict)
    """Structured findings for audit and downstream analysis."""

    @classmethod
    def not_applicable(cls, detector_id: str, tier: Tier, why: str) -> Evidence:
        return cls(
            detector_id=detector_id,
            tier=tier,
            applicable=False,
            confidence=0.0,
            explanation=why,
        )


@dataclass
class Verdict:
    """The fused result for one image."""

    manipulated_probability: float
    decision: Decision
    evidence: list[Evidence]
    explanation: str
    created_at: datetime | None = None

    heatmap: np.ndarray | None = field(default=None, repr=False)
    """Fused localisation over the full image, or None if nothing localised."""

    localised_by: list[str] = field(default_factory=list)
    """Detector ids that contributed to :attr:`heatmap`."""

    @property
    def firing(self) -> list[Evidence]:
        """Applicable detectors that actually found something, most suspicious first."""
        return sorted(
            (e for e in self.evidence if e.applicable and e.score > 0.5),
            key=lambda e: e.score * e.confidence,
            reverse=True,
        )
