"""How much does a content credential survive on its way to an adjuster?

    python scripts/sweep_credentials.py <image-with-credentials> [more...]

`provenance.content_credentials` turns a signed C2PA manifest into the strongest
single finding this pipeline can produce: a real ChatGPT export goes from
`AUTO_CLEAR` at 0.288 to `FLAG` at 0.876, because the file says under signature that
a model made it.

That strength is worth exactly as much as the manifest's survival, and metadata does
not survive much. The point of this sweep is to put a number on the failure rather
than mention it in a caveat, because the difference between "credentials are
strippable" and "credentials are gone after one screenshot" decides whether the
detector is a deployment feature or a demo trick.

Each condition is something that happens to a claim photograph without anyone
intending to attack the system: a phone re-encodes it, a messaging app resizes it,
someone screenshots it out of an email. Deliberate stripping is easier still.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/interim/credential_sweep"


def reads_credentials(path: Path) -> tuple[bool, str]:
    """Whether a verifiable manifest survives in this file."""
    try:
        import json

        from c2pa import Reader

        with Reader(str(path)) as r:
            report = json.loads(r.json())
        active = (report.get("manifests") or {}).get(report.get("active_manifest"))
        if not active:
            return False, "no manifest"
        return True, str(report.get("validation_state", "?"))
    except ImportError:
        raise SystemExit("needs the provenance extra: pip install -e '.[provenance]'")
    except Exception:
        return False, "no manifest"


def launder(src: Path, out: Path) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    img = Image.open(src).convert("RGB")
    w, h = img.size
    made: dict[str, Path] = {}

    p = out / f"{src.stem}_copy.png"
    img.save(p)
    made["re-saved as PNG"] = p

    for q in (95, 75):
        p = out / f"{src.stem}_q{q}.jpg"
        img.save(p, quality=q)
        made[f"re-encoded to JPEG q{q}"] = p

    p = out / f"{src.stem}_resized.jpg"
    img.resize((w // 2, h // 2), Image.LANCZOS).save(p, quality=85)
    made["resized by a messaging app"] = p

    # A screenshot never touches the original bytes at all: the pixels are recaptured
    # off the screen and written into a brand-new file.
    p = out / f"{src.stem}_screenshot.png"
    img.crop((0, 0, w, h)).save(p)
    made["screenshotted"] = p

    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", type=Path)
    args = ap.parse_args()

    rows: dict[str, list[bool]] = {}
    for src in args.images:
        if not src.is_file():
            print(f"skipping missing {src}", file=sys.stderr)
            continue
        ok, state = reads_credentials(src)
        print(f"{src.name}\n  as supplied: "
              f"{'credentials present, ' + state if ok else 'no credentials'}")
        rows.setdefault("as supplied", []).append(ok)
        for label, path in launder(src, WORK).items():
            survived, _ = reads_credentials(path)
            rows.setdefault(label, []).append(survived)

    if not rows:
        return 1

    print(f"\n{'condition':<34}{'n':>4}{'credentials survive':>22}")
    print("-" * 60)
    for label, results in rows.items():
        rate = sum(results) / len(results)
        print(f"{label:<34}{len(results):>4}{rate * 100:>21.0f}%")

    survived_any = any(
        any(v) for k, v in rows.items() if k != "as supplied"
    )
    print(
        "\nReading:"
        "\n  A signed manifest is the strongest finding the pipeline has, and it is"
        "\n  gone the moment anyone re-saves the file. That is not a flaw in C2PA, it"
        "\n  is what metadata is: the bytes are not the pixels, and nothing binds them"
        "\n  together once the chunk is dropped."
        "\n\n  So the detector is deliberately one-sided. Present and valid is close to"
        "\n  conclusive. Absent means nothing whatsoever, and it abstains rather than"
        "\n  reporting innocence, because every honest photograph that ever passed"
        "\n  through a messaging app looks identical to a stripped generated one."
    )
    if not survived_any:
        print("\n  Measured here: no laundering condition preserved the manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
