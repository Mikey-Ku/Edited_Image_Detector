"""Fetch Synthbuster+, the generated-image set for Q2 of docs/METHOD_STUDY.md.

    python scripts/fetch_synthbuster.py --per-source 500

Q2 asks whether an image came from a camera at all, and the interesting version of
that question is cross-generator: train a detector on one model's output and see
whether it recognises another model's. The published GenImage result says
in-generator accuracy runs above 98% while the best cross-generator average sits
near 70%, so there is a known answer for the harness to reproduce.

    Mirror: huggingface.co/datasets/marco-willi/synthbuster-plus
    Original: Bammey, "Synthbuster: Towards Detection of Diffusion Model Generated
    Images", IEEE OJSP 2024. Real photographs are RAISE.

**Why this and not GenImage.** GenImage is the larger benchmark, but every mirror
of it ships 38 GB multipart archives per generator that cannot be subset, and its
real images are ImageNet: web-scraped, re-encoded, of unknown provenance.
Synthbuster's real half is RAISE, which is uncompressed camera output with known
devices. That matters more here than scale, because half this repository reads
sensor and compression structure, and ImageNet's re-encoding would destroy exactly
the signal under test. A dataset that erases the thing you are measuring makes the
measurement meaningless no matter how many images it has.

**Splits are rebuilt.** The upstream train/validation/test split is by image, which
is the wrong axis: a detector could see dalle2 in training and dalle2 in test and
report a number that says nothing about an unseen generator. This script keeps the
`source` label on every file and lays the data out by generator so the study can
split on that instead. Upstream splits are used only to bound the download.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/synthbuster"
REPO = "marco-willi/synthbuster-plus"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/data"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main"


def shard_names(splits: tuple[str, ...]) -> list[str]:
    with urllib.request.urlopen(API, timeout=60) as r:
        tree = json.load(r)
    names = sorted(
        f["path"] for f in tree
        if f["path"].endswith(".parquet")
        and Path(f["path"]).name.split("-")[0] in splits
    )
    if not names:
        sys.exit(f"no parquet shards for splits {splits}; layout may have changed")
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=500,
                    help="images kept per generator, and per real-image source")
    ap.add_argument("--splits", nargs="+", default=["test", "validation"],
                    help="upstream splits to stream; they are re-partitioned by source")
    args = ap.parse_args()

    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow missing -- pip install -e '.[study]'")

    names = shard_names(tuple(args.splits))
    print(f"{len(names)} shards across {args.splits}, cap {args.per_source} per source")

    kept: dict[str, int] = defaultdict(int)
    tmp = OUT / "_shard.parquet"
    OUT.mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(names, 1):
        # Every shard is downloaded, read, and deleted. Only the retained images
        # stay on disk, so the cap bounds storage even though it cannot bound
        # transfer: parquet has to arrive before it can be filtered.
        urllib.request.urlretrieve(f"{RESOLVE}/{name}", tmp)
        table = pq.read_table(tmp, columns=["image", "label", "image_id", "source"])
        rows = table.to_pylist()
        tmp.unlink()

        added = 0
        for row in rows:
            label = "real" if row["label"] == 0 else row["source"]
            if kept[label] >= args.per_source:
                continue
            dest = OUT / label
            dest.mkdir(parents=True, exist_ok=True)
            payload = row["image"]
            raw = payload["bytes"] if isinstance(payload, dict) else payload
            if raw is None:
                continue
            suffix = Path(payload.get("path") or "x.png").suffix if isinstance(payload, dict) else ".png"
            (dest / f"{row['image_id']}{suffix or '.png'}").write_bytes(raw)
            kept[label] += 1
            added += 1

        print(f"  [{i}/{len(names)}] {Path(name).name}  +{added}")

    print()
    for source in sorted(kept):
        print(f"  {source:<24} {kept[source]}")
    print(f"\n{sum(kept.values())} images in {OUT}")


if __name__ == "__main__":
    main()
