"""Train the camera-fingerprint extractor.

    python scripts/train_fingerprint.py --epochs 12

The objective is camera identity, not manipulation. Nothing in this script has
ever seen an edited image, and that is deliberate: a model trained on
manipulations learns the generator that made them, which is how this project's
first evaluation came to report 0.99 on a system that was at chance.

Validation is on **held-out devices**. The reported number answers "does this
separate cameras it has never seen", which is the only question that predicts
behaviour on a stranger's photograph.
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
OUT = ROOT / "data/processed/fingerprint.pt"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: FingerprintNet, loader: DataLoader, dev: torch.device) -> dict:
    """Same-camera vs different-camera separation on held-out devices.

    Reported as the gap between mean same-device similarity and mean
    different-device similarity. A model that has learned nothing scores ~0 --
    every pair looks alike -- so the gap is directly interpretable in a way that a
    contrastive loss value is not.
    """
    model.eval()
    embeds, labels = [], []
    for x, y in loader:
        embeds.append(model.pooled(x.to(dev)).cpu())
        labels.append(y)
    e = torch.cat(embeds)
    y = torch.cat(labels)

    sim = e @ e.T
    same = (y[:, None] == y[None, :]) & ~torch.eye(len(y), dtype=torch.bool)
    diff = (y[:, None] != y[None, :])
    if not same.any() or not diff.any():
        return {"same": 0.0, "diff": 0.0, "gap": 0.0}

    s, d = float(sim[same].mean()), float(sim[diff].mean())
    return {"same": s, "diff": d, "gap": s - d}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patches-per-image", type=int, default=6)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--depth", type=int, default=6)
    args = ap.parse_args()

    if not DATA.is_dir():
        print(f"no data at {DATA} -- run scripts/fetch_vision.py", file=sys.stderr)
        return 1

    split = device_split(DATA)
    print(split.describe())

    train_ds = CameraPatches(DATA, split.train_devices, args.patches_per_image)
    val_ds = CameraPatches(DATA, split.val_devices, args.patches_per_image, seed=99)
    print(f"{len(train_ds)} train patches, {len(val_ds)} held-out patches\n")

    train_dl = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True
    )
    val_dl = DataLoader(val_ds, batch_size=args.batch, num_workers=4)

    dev = pick_device()
    model = FingerprintNet(width=args.width, depth=args.depth).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    params = sum(p.numel() for p in model.parameters())
    print(f"device {dev}, {params/1000:.0f}k parameters\n")

    history = []
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, total, seen = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            loss = supervised_contrastive_loss(model.pooled(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss) * len(y)
            seen += len(y)
        sched.step()

        m = evaluate(model, val_dl, dev)
        history.append({"epoch": epoch, "loss": total / max(seen, 1), **m})
        print(
            f"epoch {epoch:>2}  loss {total/max(seen,1):.4f}   "
            f"held-out same {m['same']:+.3f}  diff {m['diff']:+.3f}  "
            f"GAP {m['gap']:+.3f}   ({time.time()-started:.0f}s)",
            flush=True,
        )

        if m["gap"] > best:
            best = m["gap"]
            OUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "width": args.width,
                    "depth": args.depth,
                    "val_devices": split.val_devices,
                    "gap": best,
                },
                OUT,
            )

    (OUT.parent / "fingerprint_history.json").write_text(json.dumps(history, indent=2))
    print(f"\nbest held-out gap {best:+.3f} -> {OUT}")
    if best < 0.05:
        print(
            "\nWARNING: the gap is near zero. The model is not separating cameras it "
            "has not seen, so it will not localise manipulations either. Do not wire "
            "it into the pipeline on the strength of a falling loss curve.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
