"""Train the supervised control and see whether it survives a change of dataset.

    python scripts/train_supervised.py --epochs 8

Predictions 5.1 and 5.2 of docs/METHOD_STUDY.md. Trains a real/fake classifier on
CASIA 2.0, reports it on a held-out CASIA split, then points it at Korus without
changing anything. The whole question is the size of the gap between those two
numbers.

**Scale, and why inference is tiled.** CASIA images are mostly 384x256. Korus
images are full-frame camera output, an order of magnitude larger. Resizing Korus
down to CASIA's scale would resample away the high-frequency detail that every
manipulation trace lives in, which would hand the control a loss it did not earn.
Cropping natively instead keeps the pixels honest but raises a different problem:
a single 224 crop of a 4 megapixel frame will usually miss the forgery entirely.

So Korus is tiled into a grid of native-resolution 224 crops and scored over all
of them. Both the max and the mean are reported. Max is the operationally sensible
reading, a forgery anywhere makes the image tampered, and it is applied to
pristine images identically, so the extra chances to fire are paid for on both
sides. Where the two disagree, that disagreement is itself a result.

Trained from scratch. See `learned/classifier.py` for why, and for what that
costs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundtruth.learned.classifier import ManipulationCNN

ROOT = Path(__file__).resolve().parents[1]
CASIA = ROOT / "data/raw/casia2"
KORUS = ROOT / "data/interim/korus/data-images"
OUT = ROOT / "data/processed/supervised_control.json"
CROP = 224

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def to_tensor(im: Image.Image) -> torch.Tensor:
    a = torch.from_numpy(np.asarray(im.convert("RGB"), dtype=np.uint8).copy())
    return (a.permute(2, 0, 1).float() / 255.0 - MEAN) / STD


def casia_items() -> list[tuple[Path, int]]:
    auth = [(p, 0) for p in sorted((CASIA / "authentic").glob("*.png"))]
    tamp = [(p, 1) for p in sorted((CASIA / "tampered").glob("*.png"))
            if ".mask." not in p.name]
    return auth + tamp


class CasiaCrops(Dataset):
    def __init__(self, items: list[tuple[Path, int]], train: bool):
        self.items, self.train = items, train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        im = Image.open(path).convert("RGB")
        # Pad rather than resize when an image is smaller than the crop. Resizing
        # up would invent high-frequency content, which is the signal under test.
        if im.width < CROP or im.height < CROP:
            canvas = Image.new("RGB", (max(CROP, im.width), max(CROP, im.height)))
            canvas.paste(im, (0, 0))
            im = canvas
        if self.train:
            x = random.randint(0, im.width - CROP)
            y = random.randint(0, im.height - CROP)
            im = im.crop((x, y, x + CROP, y + CROP))
            if random.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            x, y = (im.width - CROP) // 2, (im.height - CROP) // 2
            im = im.crop((x, y, x + CROP, y + CROP))
        return to_tensor(im), label


def auc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    pos, neg = y == 1, y == 0
    n1, n0 = int(pos.sum()), int(neg.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def korus_items() -> list[tuple[Path, int]]:
    items: list[tuple[Path, int]] = []
    for cam in sorted(p for p in KORUS.iterdir() if p.is_dir()):
        for sub, label in (("pristine", 0), ("tampered-realistic", 1)):
            d = cam / sub
            if d.is_dir():
                items += [(p, label) for p in sorted(d.glob("*.TIF"))]
    return items


@torch.no_grad()
def score_tiled(model, path: Path, dev, stride: int = CROP, cap: int = 64):
    """Score every native-resolution tile of one image; return (max, mean)."""
    im = Image.open(path).convert("RGB")
    if im.width < CROP or im.height < CROP:
        return None
    xs = list(range(0, im.width - CROP + 1, stride))
    ys = list(range(0, im.height - CROP + 1, stride))
    coords = [(x, y) for y in ys for x in xs][:cap]
    tiles = torch.stack([to_tensor(im.crop((x, y, x + CROP, y + CROP)))
                         for x, y in coords]).to(dev)
    logits = torch.cat([model(tiles[i:i + 16]) for i in range(0, len(tiles), 16)])
    p = torch.sigmoid(logits).cpu().numpy()
    return float(p.max()), float(p.mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not (CASIA / "authentic").is_dir():
        print(f"no CASIA at {CASIA} -- run scripts/fetch_casia.py", file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    items = casia_items()
    order = rng.permutation(len(items))
    n_test = int(0.15 * len(items))
    test = [items[i] for i in order[:n_test]]
    train = [items[i] for i in order[n_test:]]
    print(f"CASIA: {len(train)} train, {len(test)} held out "
          f"({sum(l for _, l in train)} tampered in train)")

    dev = (torch.device("mps") if torch.backends.mps.is_available()
           else torch.device("cuda") if torch.cuda.is_available()
           else torch.device("cpu"))
    model = ManipulationCNN().to(dev)
    params = sum(p.numel() for p in model.parameters())
    print(f"device {dev}, {params/1e6:.1f}M parameters\n")

    train_dl = DataLoader(CasiaCrops(train, True), batch_size=args.batch,
                          shuffle=True, num_workers=4, drop_last=True)
    test_dl = DataLoader(CasiaCrops(test, False), batch_size=args.batch,
                         num_workers=4)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = torch.nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        started, tot, seen = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev).float()
            loss = lossf(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(y)
            seen += len(y)
        sched.step()

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in test_dl:
                ps.append(torch.sigmoid(model(x.to(dev))).cpu().numpy())
                ys.append(y.numpy())
        a = auc(np.concatenate(ys), np.concatenate(ps))
        print(f"epoch {epoch:>2}  loss {tot/max(seen,1):.4f}   "
              f"CASIA held-out AUC {a:.4f}   ({time.time()-started:.0f}s)", flush=True)

    casia_auc = a
    print(f"\n5.1 (>= 0.95 in-distribution): {casia_auc:.4f} -> "
          f"{'HIT' if casia_auc >= 0.95 else 'MISS'}")

    print("\nscoring Korus, tiled at native resolution...")
    model.eval()
    ky, kmax, kmean = [], [], []
    for i, (path, label) in enumerate(korus_items()):
        r = score_tiled(model, path, dev)
        if r is None:
            continue
        ky.append(label)
        kmax.append(r[0])
        kmean.append(r[1])
        if (i + 1) % 50 == 0:
            print(f"  {i+1} images", flush=True)

    ky = np.array(ky)
    korus_max = auc(ky, np.array(kmax))
    korus_mean = auc(ky, np.array(kmean))
    print(f"\nKorus ({len(ky)} images)  AUC max-tile {korus_max:.4f}   "
          f"mean-tile {korus_mean:.4f}")
    worst = max(korus_max, korus_mean)
    print(f"5.2 (<= 0.70 on Korus): best reading {worst:.4f} -> "
          f"{'HIT' if worst <= 0.70 else 'MISS'}")

    OUT.write_text(json.dumps({
        "casia_heldout_auc": casia_auc,
        "korus_auc_max_tile": korus_max,
        "korus_auc_mean_tile": korus_mean,
        "n_korus": len(ky),
        "epochs": args.epochs,
        "params": params,
    }, indent=2))
    print(f"written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
