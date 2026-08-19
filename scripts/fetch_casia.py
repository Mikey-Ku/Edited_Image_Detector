"""Fetch CASIA 2.0, the second manipulation dataset for docs/METHOD_STUDY.md.

    python scripts/fetch_casia.py

CASIA 2.0 is 12,614 images, 7,491 authentic and 5,123 spliced or copy-moved in
Photoshop. It is here for one reason: Korus is 224 images from two cameras, which
is enough to *evaluate* a detector and nowhere near enough to *train* one. The
supervised classifier in the study needs a training set large enough that losing
on Korus means something. If it were trained on Korus it would lose for lack of
data and prove nothing.

    Mirror: huggingface.co/datasets/ductai199x/image-manipulation-dataset-compilation
    Original: Dong, Wang & Tan, "CASIA Image Tampering Detection Evaluation Database"

**This dataset is a training set here and never an evidence base.** CASIA is known
to carry a compression shortcut: the authentic and tampered halves were not saved
through the same pipeline, so a classifier can separate them without looking at
the forgery at all. Prediction 5.6 in the study measures exactly how much of it a
quantization-table-only classifier can solve. Whatever that number is, it is the
discount applied to every other CASIA figure.

Masks are not fetched. Localisation is measured on Korus, which has ground truth
this project has already validated.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/casia2"
REPO = "ductai199x/image-manipulation-dataset-compilation"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main"


def shards() -> list[str]:
    """Every CASIA2.0 tar in the mirror, discovered rather than hard-coded."""
    with urllib.request.urlopen(API, timeout=60) as r:
        tree = json.load(r)
    names = sorted(
        f["path"] for f in tree
        if f["path"].startswith("CASIA2.0-") and f["path"].endswith(".tar")
    )
    if not names:
        sys.exit("no CASIA2.0 tars found in the mirror; the layout may have changed")
    return names


def grab(name: str) -> tuple[str, int]:
    """Download one shard and extract it under authentic/ or tampered/."""
    kind = "authentic" if "-auth-" in name else "tampered"
    dest = OUT / kind
    dest.mkdir(parents=True, exist_ok=True)
    tmp = OUT / name

    if not tmp.exists():
        urllib.request.urlretrieve(f"{RESOLVE}/{name}", tmp)

    written = 0
    with tarfile.open(tmp) as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            target = dest / Path(member.name).name
            if target.exists():
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            target.write_bytes(src.read())
            written += 1
    tmp.unlink()
    return name, written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    names = shards()
    print(f"{len(names)} shards")
    OUT.mkdir(parents=True, exist_ok=True)

    total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(grab, n): n for n in names}
        for fut in as_completed(futures):
            name, n = fut.result()
            total += n
            print(f"  {name}  +{n}")

    auth = len(list((OUT / "authentic").glob("*")))
    tamp = len(list((OUT / "tampered").glob("*")))
    print(f"\n{total} extracted -> {auth} authentic, {tamp} tampered, in {OUT}")


if __name__ == "__main__":
    main()
