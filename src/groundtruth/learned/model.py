"""A small camera-fingerprint extractor.

The network's job is to strip scene content and leave the sensor's signature. Two
design choices carry most of that:

**No pooling, no stride.** Every layer is a 3x3 convolution at full resolution, so
the output is a per-pixel residual rather than a summary vector. That is what makes
the model usable for *localisation* -- at inference we need a fingerprint at every
position so a manipulated region can be spotted as a patch of the image whose
residual disagrees with its surroundings. A classifier that pools to a single
vector per image could tell you which camera took a photo and nothing about where
it was edited.

**A high-pass first layer, initialised and left free.** The fingerprint lives in
the noise residual, and the first layer is initialised to a fixed high-pass kernel
so training starts from the residual rather than having to discover it. It stays
trainable, because the hand-derived filter is a starting point, not a constraint --
the entire lesson of the hand-crafted phase of this project is that a fixed filter
chosen by a human tends to be slightly wrong in ways that matter.

Training is contrastive over camera identity: patches from the same device should
produce similar residuals, patches from different devices should not. The model
never sees a manipulation, which is exactly why it does not go stale when a new
editing tool appears.
"""

from __future__ import annotations

import torch
from torch import nn

# Output channels of the residual. Wide enough to separate a few dozen devices,
# narrow enough that the per-pixel map stays cheap to compute over a full frame.
EMBED_DIM = 32


def _high_pass_kernel() -> torch.Tensor:
    """3x3 second-order difference, applied per input channel."""
    k = torch.tensor(
        [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]
    ) / 8.0
    return k


class FingerprintNet(nn.Module):
    """Maps an RGB image to a per-pixel camera residual."""

    def __init__(self, width: int = 48, depth: int = 6, embed_dim: int = EMBED_DIM):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 3
        for i in range(depth):
            out_ch = width if i < depth - 1 else embed_dim
            layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=(i > 0)))
            if i < depth - 1:
                layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.body = nn.Sequential(*layers)

        # Projection head, on the pooled statistics rather than the residual.
        #
        # Needed because the pooled features are per-channel means and standard
        # deviations, and standard deviations are non-negative -- so every raw
        # embedding lands in the positive orthant and any two point nearly the same
        # way. Cosine similarity was ~1.000 for every pair before training started,
        # which left the contrastive loss no gradient to work with.
        #
        # BatchNorm1d centres each statistic across the batch, removing the shared
        # component; the MLP then has the freedom to place cameras apart. This is
        # the standard contrastive-learning projection head, and it is discarded at
        # inference -- localisation uses the per-pixel residual from `forward`.
        stat_dim = embed_dim * 2
        self.head = nn.Sequential(
            nn.BatchNorm1d(stat_dim),
            nn.Linear(stat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
        )
        self._init_high_pass()

    def _init_high_pass(self) -> None:
        first = self.body[0]
        assert isinstance(first, nn.Conv2d)
        with torch.no_grad():
            k = _high_pass_kernel()
            first.weight.zero_()
            for o in range(first.out_channels):
                first.weight[o, o % first.in_channels] = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Per-pixel residual, L2-normalised across channels.

        Normalising per pixel makes the fingerprint a *direction* rather than a
        magnitude, so a bright region and a dark region of the same sensor compare
        equal. Magnitude is dominated by local contrast, which is scene content --
        precisely what has to be discarded.
        """
        r = self.body(x)
        return r / (r.norm(dim=1, keepdim=True) + 1e-6)

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        """One embedding per patch, for the contrastive training objective.

        Pools the RAW residual, and by second-order statistics rather than a mean.

        Both details were bugs first. Averaging the per-pixel *normalised* residual
        collapses the representation: 4096 unit vectors average to their common
        component, so every patch produced a near-identical embedding (same-camera
        similarity .974, different-camera .969, and a loss sitting exactly at
        ln(batch) -- the value for having learned nothing).

        A mean is also the wrong summary in principle. A noise residual is
        zero-mean by construction, so its spatial average carries almost no
        information about the sensor. What distinguishes a camera is the residual's
        *distribution* -- its per-channel energy -- which is what the standard
        deviation captures. The mean is kept alongside it since a genuine DC offset
        per channel is still evidence.
        """
        r = self.body(x)
        stats = torch.cat([r.mean(dim=(2, 3)), r.std(dim=(2, 3))], dim=1)
        e = self.head(stats)
        return e / (e.norm(dim=1, keepdim=True) + 1e-6)


def supervised_contrastive_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """Pull same-device embeddings together, push different devices apart.

    Supervised contrastive rather than a plain classifier: the aim is a metric
    space where "same sensor" means "close", which is what the detector needs at
    inference. A softmax classifier would only ever learn to separate the devices
    it was trained on, and every device it meets in production is one it was not.
    """
    device = embeddings.device
    sim = embeddings @ embeddings.T / temperature

    # A patch must not be its own positive. Masked with a large finite negative
    # rather than -inf: -inf survives the logsumexp fine, but multiplying it by a
    # False mask afterwards gives -inf * 0 = NaN, which silently poisons the whole
    # loss. Finite masking keeps the arithmetic well defined everywhere.
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=device)
    sim = sim.masked_fill(self_mask, -1e4)

    positives = (labels[:, None] == labels[None, :]) & ~self_mask
    has_positive = positives.any(dim=1)
    if not has_positive.any():
        return embeddings.sum() * 0.0

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    # Zero the non-positive entries instead of multiplying by the mask, so no
    # masked value can reach the sum at all.
    per_sample = log_prob.masked_fill(~positives, 0.0).sum(dim=1) / positives.sum(
        dim=1
    ).clamp(min=1)
    return -per_sample[has_positive].mean()
