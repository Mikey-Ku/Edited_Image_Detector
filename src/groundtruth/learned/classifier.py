"""A supervised real/fake classifier. The control arm, built to win.

This is the thing the rest of the project argues against: a network that looks at
labelled manipulations and learns what they look like. It exists so that the
argument can be tested instead of asserted, which means it has to be built well
enough that losing is informative. A weak control proves nothing.

**Trained from scratch, no ImageNet pretraining.** Partly a constraint, the
environment has no working torchvision, and partly a choice: pretrained features
carry ImageNet's own compression history, which is exactly the kind of signal this
study is trying to attribute. From-scratch is the cleaner comparison here. The
cost is that the control gets less prior knowledge than the literature usually
gives it, and if it fails to clear prediction 5.1 in-distribution then this
decision is the first suspect and the study has to say so.

The architecture is deliberately ordinary: a small residual stack on 224x224
crops. Nothing here is trying to be clever. The point of a control is to be the
obvious thing done competently.
"""

from __future__ import annotations

import torch
from torch import nn


class Block(nn.Module):
    """Pre-activation residual block, stride on the first convolution."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
            if (stride != 1 or in_ch != out_ch)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.bn1(x))
        out = self.conv1(h)
        out = self.conv2(torch.relu(self.bn2(out)))
        return out + self.skip(h)


class ManipulationCNN(nn.Module):
    """Binary real/fake over a whole image crop.

    One deliberate departure from a stock classifier: the stem is a plain 3x3 at
    stride 1 rather than the usual 7x7 stride-2. Manipulation traces live in
    high-frequency detail, and a stride-2 stem throws half of it away in the first
    operation. That is a free handicap this control does not need to carry.
    """

    def __init__(self, width: int = 32, num_blocks: tuple[int, ...] = (2, 2, 2, 2)):
        super().__init__()
        self.stem = nn.Conv2d(3, width, 3, padding=1, bias=False)
        layers: list[nn.Module] = []
        in_ch = width
        for i, n in enumerate(num_blocks):
            out_ch = width * (2 ** i)
            for j in range(n):
                layers.append(Block(in_ch, out_ch, stride=2 if j == 0 else 1))
                in_ch = out_ch
        self.body = nn.Sequential(*layers)
        self.norm = nn.BatchNorm2d(in_ch)
        self.head = nn.Linear(in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(self.stem(x))
        h = torch.relu(self.norm(h)).mean(dim=(2, 3))
        return self.head(h).squeeze(1)
