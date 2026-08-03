"""Fetch a device-balanced subset of the VISION dataset.

VISION carries 35 phones across 11 brands, and for every natural scene it also
provides the same image after a WhatsApp and a Facebook round-trip. That is the
crucial part: the laundering that destroys hand-crafted forensic traces is already
in the data, so a model trained here learns to survive it rather than assuming it
away.

Only a balanced subset is pulled. The training signal is "which camera took this",
so what matters is covering many devices with varied scenes, not exhausting any one
device. Every device contributes the same number of images, because an unbalanced
set would let the model score well by learning the prior over devices instead of
learning their sensor signatures.

    https://lesc.dinfo.unifi.it/VISION/  --  CC BY-SA 4.0
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "data/raw/vision_files.txt"
OUT = ROOT / "data/raw/vision"

# Native scenes teach the sensor signature; the social-media copies teach it to
# survive recompression and resizing. Both are needed -- native alone produces a
# model that works only on files nobody actually receives.
MIX = {"nat": 18, "natWA": 8, "natFBL": 4}


def plan(per_device: dict[str, int]) -> dict[str, list[str]]:
    if not LIST.exists():
        sys.exit(f"missing {LIST} -- download VISION_files.txt first")

    by_device: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for line in LIST.read_text().splitlines():
        if "/images/" not in line:
            continue
        device = line.split("/dataset/")[1].split("/")[0]
        category = line.split("/images/")[1].split("/")[0]
        if category in per_device:
            by_device[device][category].append(line)

    chosen: dict[str, list[str]] = {}
    for device, cats in sorted(by_device.items()):
        urls: list[str] = []
        for category, n in per_device.items():
            # Evenly spaced rather than the first n, so scenes stay varied instead
            # of all coming from one shooting session.
            pool = cats.get(category, [])
            if not pool:
                continue
            step = max(1, len(pool) // n)
            urls += pool[::step][:n]
        chosen[device] = urls
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--devices", type=int, default=0, help="limit devices (0 = all)")
    # Politeness, not throughput. At 24 workers the host returned 503 for 863 of
    # 1050 requests -- it is a university server, not a CDN. Six concurrent
    # connections with backoff finishes sooner than hammering it and retrying.
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args()

    chosen = plan(MIX)
    if args.devices:
        chosen = dict(list(chosen.items())[: args.devices])

    total = sum(len(v) for v in chosen.values())
    print(f"{len(chosen)} devices, {total} images "
          f"({', '.join(f'{k}x{v}' for k, v in MIX.items())} each)")
    if args.dry_run:
        for d, urls in list(chosen.items())[:3]:
            print(f"  {d}: {len(urls)}  e.g. {urls[0].rsplit('/', 1)[1]}")
        return 0

    jobs: list[tuple[str, Path]] = []
    for device, urls in chosen.items():
        target = OUT / device
        target.mkdir(parents=True, exist_ok=True)
        for url in urls:
            jobs.append((url, target / url.rsplit("/", 1)[1]))

    def fetch(job: tuple[str, Path]) -> bool:
        url, dest = job
        if dest.exists() and dest.stat().st_size > 0:
            return True
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:
            print(f"  ! {dest.name}: {exc}", file=sys.stderr)
            dest.unlink(missing_ok=True)
            return False
        return True

    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            if fut.result():
                done += 1
            else:
                failed += 1
            if i % 50 == 0 or i == len(futures):
                print(f"  {i}/{total}  ({done} ok, {failed} failed)", flush=True)

    size = sum(p.stat().st_size for p in OUT.rglob("*.jpg")) / 1e9
    print(f"\n{done} images, {failed} failed, {size:.1f} GB in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
