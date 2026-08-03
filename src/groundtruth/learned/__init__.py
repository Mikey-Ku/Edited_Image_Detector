"""Learned camera-fingerprint extraction.

Trained self-supervised on camera identity, never on manipulations -- see
``dataset`` for why that distinction is the whole point.

STATUS: the training pipeline is correct and does not converge at this scale.

Measured on VISION, 28 training devices and 7 held out:

    epoch 1   loss 4.931   train GAP +0.002   held-out GAP +0.002
    epoch 2   loss 4.822   train GAP +0.005   held-out GAP +0.004

The train and held-out gaps are identical, which rules out the interesting
failure. This is not a model memorising its training devices and failing to
transfer -- it is not learning the task at all. More devices would not fix it.

Two things were ruled out along the way:

- **Pooling.** Mean+std and a trace-normalised Gram matrix -- the style-transfer
  statistic, chosen because it captures texture while discarding spatial
  arrangement -- give the same +0.0045 separation on real patches. The bottleneck
  is not how the residual is summarised.
- **Architecture.** On synthetic patches whose only difference is noise level
  (sigma 0.02 vs 0.08, identical scene), the same network reaches a gap of +1.441
  in 120 steps. It can learn a sensor signature when one is cleanly present.

What remains is scale. Noiseprint++ was trained on **1,475 camera models across
512 processing pipelines**; this has 28 devices and 780 images. Between two real
photographs the residual statistics are dominated by scene content -- brickwork
against skin -- and the sensor difference is a small perturbation on top. Pulling
it out is what the extra three orders of magnitude of camera diversity buys.

The pipeline is kept because it is correct and reusable, and because the negative
result is worth as much as the code: it says plainly that this approach needs a
data scale this project does not have, rather than leaving the question open.
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
