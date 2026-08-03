"""Noiseprint: a pretrained camera-model fingerprint extractor.

This is the thing the hand-crafted phase of this project kept failing to build.
Every physically-motivated detector here tried to derive a noise residual in
closed form -- from wavelet detail, from block grids, from flat-region silence --
and each looked convincing on one image before falling apart across several.
Noiseprint learns the same residual instead, from pairs of patches labelled by the
camera that took them, and it is trained across quality factors 51-100 plus PNG so
it survives the recompression that destroyed the derived versions.

Architecture: 17 fully-convolutional levels, one grayscale channel in, one out.
No pooling and no stride, so the output is a residual *image* -- which is what
localisation needs. The BatchNorm is inference-only with statistics stored as
parameters, matching the original TensorFlow release so the published weights load
unchanged.

A separate network is trained per JPEG quality factor, because quantisation
changes the residual. Picking the wrong one degrades the fingerprint badly, so the
quality factor is recovered from the file's own quantisation table rather than
assumed; qf101 is the model for images with no JPEG history at all.

LICENSE -- this is the one dependency here that is not permissively licensed.

    Copyright (c) 2019 Image Processing Research Group of University Federico II
    of Naples ('GRIP-UNINA'). All rights reserved.
    This work should only be used for nonprofit purposes.

Weights and architecture are the authors' own, obtained via a faithful PyTorch
port (github.com/RonyAbecidan/noiseprint-pytorch) of the official TensorFlow
release. Informational and nonprofit use only; commercial use is expressly
prohibited. See vendor/NOISEPRINT_LICENSE.txt.

    Cozzolino & Verdoliva, "Noiseprint: A CNN-Based Camera Model Fingerprint",
    IEEE TIFS 2020.  https://arxiv.org/abs/1808.08396
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

log = logging.getLogger(__name__)

WEIGHTS_DIR = Path(__file__).resolve().parents[3] / "data/raw/noiseprint"

# Quality factor used when an image carries no JPEG quantisation table.
PNG_QF = 101

_LEVELS = 17
_WIDTH = 64
_BN_EPS = 1e-5

# Above this many pixels the residual is computed in overlapping tiles. The
# network is fully convolutional so a full frame is valid input, but memory grows
# linearly and a phone photograph will exhaust it.
_TILE_LIMIT = 1_050_000
_TILE = 1024
_OVERLAP = 34


class _AddBias(nn.Module):
    """Bias as its own layer, matching the original graph's parameter layout."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias.view(1, -1, 1, 1)


class _FrozenBatchNorm(nn.Module):
    """Inference-time BatchNorm with statistics held as parameters.

    Stored as parameters rather than buffers because that is how the released
    checkpoints are keyed; loading them into a standard ``nn.BatchNorm2d`` would
    silently mismatch.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(channels))
        self.moving_mean = nn.Parameter(torch.zeros(channels))
        self.moving_variance = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        mean = self.moving_mean.view(1, c, 1, 1)
        var = self.moving_variance.view(1, c, 1, 1)
        return self.gamma.view(1, c, 1, 1) * (x - mean) / torch.sqrt(var + _BN_EPS)


class NoiseprintNet(nn.Module):
    """Grayscale image in, camera-fingerprint residual out."""

    def __init__(self, levels: int = _LEVELS, width: int = _WIDTH) -> None:
        super().__init__()
        self.conv_layers = nn.ModuleList()
        for i in range(levels):
            first, last = i == 0, i == levels - 1
            in_ch = 1 if first else width
            out_ch = 1 if last else width
            block: list[nn.Module] = [
                nn.Conv2d(in_ch, out_ch, 3, padding="same", bias=first or last)
            ]
            if not first and not last:
                block.append(_FrozenBatchNorm(out_ch))
                block.append(_AddBias(out_ch))
            block.append(nn.Identity() if last else nn.ReLU())
            self.conv_layers.append(nn.Sequential(*block))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.conv_layers:
            x = layer(x)
        return x


def quality_factor(path: Path) -> int:
    """Recover the JPEG quality factor from the file's own quantisation table.

    The residual differs per quantisation level, so the matching network has to be
    used. Estimated by comparing the file's luminance table against the standard
    table scaled to each quality -- the same inversion the original implementation
    performs. Returns PNG_QF when the file carries no table at all.
    """
    from PIL import Image

    try:
        with Image.open(path) as im:
            tables = getattr(im, "quantization", None)
    except Exception:
        return PNG_QF
    if not tables or 0 not in tables:
        return PNG_QF

    std = np.array(
        [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
         14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
         18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
         49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99],
        dtype=np.float64,
    )
    table = np.array(tables[0], dtype=np.float64).ravel()[:64]

    best, best_err = PNG_QF, np.inf
    for qf in range(51, 101):
        scale = 5000 / qf if qf < 50 else 200 - 2 * qf
        predicted = np.clip(np.floor((std * scale + 50) / 100), 1, 255)
        err = float(np.abs(predicted - table).mean())
        if err < best_err:
            best, best_err = qf, err
    return best


@lru_cache(maxsize=8)
def load(qf: int, weights_dir: str | None = None) -> NoiseprintNet:
    """Load the network trained for this quality factor. Cached across calls."""
    root = Path(weights_dir) if weights_dir else WEIGHTS_DIR
    path = root / f"model_qf{qf}.pth"
    if not path.exists():
        raise FileNotFoundError(
            f"no noiseprint weights at {path} -- see scripts/fetch_noiseprint.py"
        )
    net = NoiseprintNet()
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    return net


@torch.no_grad()
def extract(gray: np.ndarray, qf: int, weights_dir: str | None = None) -> np.ndarray:
    """Residual for a grayscale image in [0,1]. Same shape as the input.

    Large frames are processed in overlapping tiles and the overlap is discarded,
    because the network's receptive field means tile borders are unreliable --
    stitching them in would draw a grid of artefacts straight through the result.
    """
    net = load(qf, weights_dir)
    h, w = gray.shape

    if h * w <= _TILE_LIMIT:
        x = torch.from_numpy(gray.astype(np.float32))[None, None]
        return net(x)[0, 0].numpy()

    out = np.zeros((h, w), dtype=np.float32)
    for y in range(0, h, _TILE):
        for x0 in range(0, w, _TILE):
            ys, ye = max(y - _OVERLAP, 0), min(y + _TILE + _OVERLAP, h)
            xs, xe = max(x0 - _OVERLAP, 0), min(x0 + _TILE + _OVERLAP, w)
            tile = torch.from_numpy(gray[ys:ye, xs:xe].astype(np.float32))[None, None]
            res = net(tile)[0, 0].numpy()
            # Trim back to the tile's own area, dropping the unreliable margin.
            out[y : min(y + _TILE, h), x0 : min(x0 + _TILE, w)] = res[
                y - ys : y - ys + min(_TILE, h - y),
                x0 - xs : x0 - xs + min(_TILE, w - x0),
            ]
    return out
