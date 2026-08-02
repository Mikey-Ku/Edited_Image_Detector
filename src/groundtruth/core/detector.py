"""Detector interface and registry.

Detectors self-register via ``@register``. The pipeline asks each one whether it
applies to a given case before running it, so an image with no EXIF, or a PNG, or
a claim with no policy date simply drops the detectors that cannot speak to it.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from .types import Evidence, ImageCase, Tier

log = logging.getLogger(__name__)

_REGISTRY: dict[str, type[Detector]] = {}


class Detector(ABC):
    """Base class for every detection method.

    Subclasses set the class attributes and implement :meth:`_run`. The public
    :meth:`run` wraps it so a detector that raises cannot take down the pipeline —
    a crashed detector becomes "not applicable", which is honest: it produced no
    evidence.
    """

    id: str
    tier: Tier
    localises: bool = False
    cost: int = 1
    """Rough compute cost, 1-5. Used to order cheap detectors first for early exit."""

    def applies_to(self, case: ImageCase) -> tuple[bool, str]:
        """Return (applicable, reason-if-not). Default: always applicable."""
        return True, ""

    @abstractmethod
    def _run(self, case: ImageCase) -> Evidence:
        """Actual detection. Only called when :meth:`applies_to` passed."""

    def run(self, case: ImageCase) -> Evidence:
        ok, why = self.applies_to(case)
        if not ok:
            return Evidence.not_applicable(self.id, self.tier, why)
        try:
            return self._run(case)
        except Exception:
            log.exception("detector %s failed on %s", self.id, case.image_path)
            return Evidence.not_applicable(
                self.id, self.tier, "detector raised; produced no evidence"
            )


def register(cls: type[Detector]) -> type[Detector]:
    """Class decorator adding a detector to the global registry."""
    if not getattr(cls, "id", None):
        raise ValueError(f"{cls.__name__} must define a non-empty `id`")
    if cls.id in _REGISTRY:
        raise ValueError(f"duplicate detector id: {cls.id}")
    _REGISTRY[cls.id] = cls
    return cls


def all_detectors() -> list[Detector]:
    """Instantiate every registered detector, cheapest first."""
    return sorted((c() for c in _REGISTRY.values()), key=lambda d: d.cost)


def get(detector_id: str) -> Detector:
    return _REGISTRY[detector_id]()
