"""Learned camera-fingerprint extraction.

Trained self-supervised on camera identity, never on manipulations -- see
``dataset`` for why that distinction is the whole point.
"""

from .dataset import PATCH, CameraPatches, Split, device_split
from .model import EMBED_DIM, FingerprintNet, supervised_contrastive_loss

__all__ = [
    "EMBED_DIM",
    "PATCH",
    "CameraPatches",
    "FingerprintNet",
    "Split",
    "device_split",
    "supervised_contrastive_loss",
]
