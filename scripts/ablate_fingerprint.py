"""Is the fingerprint network learning, or is the high-pass filter doing the work?

    python scripts/ablate_fingerprint.py --epochs 12

Prediction 5.10 of docs/METHOD_STUDY.md. `model.py` initialises the first
convolution to a hand-derived high-pass kernel and then leaves it trainable, on
the argument that "the hand-derived filter is a starting point, not a
constraint". That argument has never been tested. If a frozen filter does as well,
the training is decoration and the honest thing is to say so.

Four conditions, identical in every other respect, same seeds, same split:

    trained-highpass   the shipped configuration: high-pass init, everything trains
    trained-random     random init, everything trains -- does the init matter?
    frozen-highpass    high-pass first layer frozen, rest trains -- does training
                       THAT layer matter?
    untrained          no training at all -- does any of it matter?

`untrained` is the one that decides it. A network whose random body plus a fixed
filter matches a trained one is not a model, it is a feature extractor with extra
steps, and the project should stop describing it as learned.

Reported on **held-out devices**, because separating cameras already seen is not
the question. The metric is the same same-vs-different similarity gap the training
script reports, so the numbers are directly comparable to its output.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundtruth.learned import (
    CameraPatches,
    FingerprintNet,
    device_split,
    supervised_contrastive_loss,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/vision"
OUT = ROOT / "data/processed/fingerprint_ablation.json"

CONDITIONS = ("trained-highpass", "trained-random", "frozen-highpass", "untrained")


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def gap(model: FingerprintNet, loader: DataLoader, dev: torch.device) -> dict:
    model.eval()
    embeds, labels = [], []
    for x, y in loader:
        embeds.append(model.pooled(x.to(dev)).cpu())
        labels.append(y)
    e, y = torch.cat(embeds), torch.cat(labels)
    sim = e @ e.T
    same = (y[:, None] == y[None, :]) & ~torch.eye(len(y), dtype=torch.bool)
    diff = y[:, None] != y[None, :]
    if not same.any() or not diff.any():
        return {"same": 0.0, "diff": 0.0, "gap": 0.0}
    s, d = float(sim[same].mean()), float(sim[diff].mean())
    return {"same": s, "diff": d, "gap": s - d}


def build(condition: str, width: int, depth: int, seed: int) -> FingerprintNet:
    torch.manual_seed(seed)
    model = FingerprintNet(width=width, depth=depth)
    if condition == "trained-random":
        # Undo the high-pass initialisation, keeping everything else identical.
        first = model.body[0]
        torch.nn.init.kaiming_normal_(first.weight, nonlinearity="relu")
    if condition == "frozen-highpass":
        model.body[0].weight.requires_grad_(False)
    return model


def run(condition: str, args, split, dev: torch.device) -> dict:
    train_dl = DataLoader(
        CameraPatches(DATA, split.train_devices, args.patches_per_image),
        batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True,
    )
    val_dl = DataLoader(
        CameraPatches(DATA, split.val_devices, args.patches_per_image, seed=99),
        batch_size=args.batch, num_workers=4,
    )

    model = build(condition, args.width, args.depth, args.seed).to(dev)
    epochs = 0 if condition == "untrained" else args.epochs

    if epochs:
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        for _ in range(epochs):
            model.train()
            for x, y in train_dl:
                x, y = x.to(dev), y.to(dev)
                loss = supervised_contrastive_loss(model.pooled(x), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sched.step()

    m = gap(model, val_dl, dev)
    m["condition"] = condition
    m["epochs"] = epochs
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patches-per-image", type=int, default=6)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not DATA.is_dir():
        print(f"no data at {DATA} -- run scripts/fetch_vision.py", file=sys.stderr)
        return 1

    split = device_split(DATA)
    dev = pick_device()
    print(split.describe())
    print(f"device {dev}, {args.epochs} epochs per trained condition\n")

    results = []
    for condition in CONDITIONS:
        started = time.time()
        m = run(condition, args, split, dev)
        results.append(m)
        print(
            f"{condition:<18} held-out GAP {m['gap']:+.4f}"
            f"   (same {m['same']:+.4f} diff {m['diff']:+.4f})"
            f"   ({time.time()-started:.0f}s)",
            flush=True,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))

    by = {r["condition"]: r["gap"] for r in results}
    delta = by["trained-highpass"] - by["untrained"]
    print(f"\ntrained-highpass minus untrained: {delta:+.4f}")
    print(
        "5.10 HIT: training adds under 0.02, the filter was doing the work."
        if abs(delta) < 0.02 else
        "5.10 MISS: training moves the held-out gap by more than 0.02."
    )
    print(f"written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
