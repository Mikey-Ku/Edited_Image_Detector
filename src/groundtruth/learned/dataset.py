"""Camera-labelled patch sampling for contrastive fingerprint training.

The training task is deliberately **not** manipulation detection. It is:

    given two 64x64 patches, did the same camera model take them?

That needs no edited images and no masks -- only clean photographs labelled by
device. A model trained this way never learns what a manipulation looks like, so
it cannot go stale when a new editing tool ships. Manipulation shows up at
inference as a region whose fingerprint disagrees with the rest of its own image.

Two sampling rules matter more than the architecture:

**Split by device, never by patch.** Patches from one photograph are nearly
identical; splitting them randomly puts the same sensor on both sides of the wall
and the score measures memorisation. Whole devices are held out, so the evaluation
asks the only question worth asking -- does this work on a camera it has never
seen?

**Prefer textured patches, but not exclusively.** The fingerprint lives in the
noise residual, which needs some signal to modulate it; a patch of blown-out sky
carries almost nothing. But training only on busy patches produces a model that
fails on the flat regions where manipulations are easiest to hide, so a minority
of low-texture patches is kept on purpose.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

PATCH = 64

# Patches flatter than this carry no usable fingerprint -- blown highlights, blank
# walls. Measured as the standard deviation of the high-pass residual.
_MIN_DETAIL = 0.004

# Fraction of sampled patches allowed to fall below that bar anyway, so the model
# still sees the flat regions where edits are easiest to hide.
_FLAT_ALLOWANCE = 0.15

_SAMPLE_ATTEMPTS = 12


@dataclass(frozen=True)
class Split:
    train_devices: list[str]
    val_devices: list[str]

    def describe(self) -> str:
        return (
            f"{len(self.train_devices)} train devices, "
            f"{len(self.val_devices)} held-out devices "
            f"({', '.join(d.split('_')[0] for d in self.val_devices)})"
        )


def device_split(root: Path, val_fraction: float = 0.25, seed: int = 0) -> Split:
    """Hold out entire devices, stratified by brand.

    Stratifying by brand matters: Apple devices share an imaging pipeline, so a
    split that put every iPhone in training would leave the held-out set testing a
    brand the model had effectively already seen through its siblings.
    """
    devices = sorted(p.name for p in root.iterdir() if p.is_dir())
    by_brand: dict[str, list[str]] = {}
    for d in devices:
        brand = d.split("_")[1] if "_" in d else d
        by_brand.setdefault(brand, []).append(d)

    rng = random.Random(seed)
    train, val = [], []
    for brand_devices in by_brand.values():
        shuffled = brand_devices[:]
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_fraction)) if len(shuffled) > 1 else 0
        val += shuffled[:n_val]
        train += shuffled[n_val:]
    return Split(sorted(train), sorted(val))


def _high_pass(patch: np.ndarray) -> float:
    """Detail energy: how much fine structure the patch carries."""
    g = patch.mean(axis=2)
    return float(np.abs(g[1:, 1:] - g[:-1, 1:] - g[1:, :-1] + g[:-1, :-1]).std())


class CameraPatches(Dataset):
    """Yields (patch, device_index). Patches are sampled fresh each epoch."""

    def __init__(
        self,
        root: Path,
        devices: list[str],
        patches_per_image: int = 6,
        seed: int = 0,
    ) -> None:
        self.root = root
        self.devices = devices
        self.index = {d: i for i, d in enumerate(devices)}
        self.patches_per_image = patches_per_image
        self.rng = random.Random(seed)

        self.files: list[tuple[Path, int]] = []
        for d in devices:
            for f in sorted((root / d).glob("*.jpg")):
                self.files.append((f, self.index[d]))
        if not self.files:
            raise ValueError(f"no images under {root} for the given devices")

    def __len__(self) -> int:
        return len(self.files) * self.patches_per_image

    def _sample_patch(self, img: np.ndarray, rng: random.Random) -> np.ndarray:
        h, w = img.shape[:2]
        fallback = None
        for attempt in range(_SAMPLE_ATTEMPTS):
            y = rng.randrange(0, max(1, h - PATCH))
            x = rng.randrange(0, max(1, w - PATCH))
            patch = img[y : y + PATCH, x : x + PATCH]
            if patch.shape[:2] != (PATCH, PATCH):
                continue
            if fallback is None:
                fallback = patch
            if _high_pass(patch) >= _MIN_DETAIL:
                return patch
            # Occasionally keep a flat patch on purpose.
            if attempt == 0 and rng.random() < _FLAT_ALLOWANCE:
                return patch
        return fallback if fallback is not None else img[:PATCH, :PATCH]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        path, label = self.files[i // self.patches_per_image]
        rng = random.Random((i, path.stat().st_size))
        with Image.open(path) as im:
            img = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        patch = self._sample_patch(img, rng)
        # Channels-first, zero-centred. No normalisation beyond that: the signal is
        # a faint residual and per-channel standardisation would rescale it away.
        return torch.from_numpy(patch.transpose(2, 0, 1).copy() - 0.5), label
