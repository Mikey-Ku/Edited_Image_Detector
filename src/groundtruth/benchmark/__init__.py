"""Benchmark generation and the detection-envelope report."""

from .manipulate import LAUNDERING, OPERATIONS, Fixture, make
from .matrix import CellResult, by_detector, envelope, run

__all__ = [
    "LAUNDERING",
    "OPERATIONS",
    "CellResult",
    "Fixture",
    "by_detector",
    "envelope",
    "make",
    "run",
]
