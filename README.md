# Ground Truth

**Detecting manipulated images in insurance claims — by understanding how cameras and
generators physically work, not by training a classifier and hoping.**

---

## The problem

Insurers settle claims from photographs. Until recently, faking one convincingly took
skill and a copy of Photoshop. Now it takes a prompt and ten seconds, and the fraud is
invisible to the human adjusters who review these images all day.

The obvious response — train a CNN on real-vs-fake — **provably does not work.** Learned
detectors collapse toward chance accuracy when they meet a generator they weren't trained
on, and new generators ship monthly.

## The approach

Manipulation leaves physical traces, because **an edited image is a physical impossibility**:

- A camera sensor captures **one colour per pixel** and interpolates the rest. AI-generated
  pixels were never demosaiced — they can't be, no sensor produced them.
- JPEG compresses on a rigid **8×8 grid**. Paste a region in and its grid almost never aligns.
- Every sensor has a **unique noise fingerprint** from manufacturing imperfections.
  Tampered regions lose it.
- Lenses fail to focus all wavelengths equally, and the error grows **radially from the
  optical centre**. A pasted object carries the aberration of wherever it came from.

None of these depend on which generator produced the fake, because they're properties of
**how physics works**, not of one model's output distribution.

On top of the forensics sits the layer that actually catches fraud: **claim context.**
A pixel-perfect authentic photo is still fraud if its EXIF timestamp predates the policy.

## The capability this is built toward

Sensor fingerprinting doesn't just localise tampering — it **identifies the individual
physical camera**. Cluster every image ever submitted by fingerprint, and any cluster
spanning supposedly-unrelated claimants is a **fraud ring**.

Each of those claims passes every check on its own. The fraud exists only in the
relationships between them.

## Usage

```bash
pip install -e ".[dev]"
groundtruth photo.jpg --render overlay.png \
    --policy-inception 2026-03-01 --loss-date 2026-06-15
```

```
decision:       FLAG
P(manipulated): 0.793

Manipulation indicated. P(manipulated) = 0.79 at confidence 0.63.
Contributing findings:
  - 14 of 163 measurable blocks have a noise level inconsistent with the rest
    of the frame (max deviation 6.8 sigma)  [sensor.noise_inconsistency]

regions of interest: 1
  bbox=(320,128)-(447,255)  6.77% of frame  peak=0.87
```

`--render` writes a three-panel PNG — original, heatmap overlay with regions of interest
boxed, and the raw localisation map.

## Status

Working end-to-end. Four detectors, 26 tests.

| Detector | Tier | Localises | State |
|---|---|---|---|
| `context.policy_consistency` | context | — | capture time vs. policy inception and loss date; GPS vs. loss location |
| `metadata.thumbnail_mismatch` | metadata | ✅ | diffs the stale EXIF preview against the image |
| `compression.ela` | compression | ✅ | baseline only — deliberately capped at low confidence, see source |
| `sensor.noise_inconsistency` | sensor | ✅ | MAD-based per-block noise estimation, adaptive structure exclusion, contiguity filter |

**Verified against synthetic splices with pixel-exact masks:** a noise splice at JPEG q=96
is flagged with a localisation hit rate of **0.99** and IoU **0.59**; the matching pristine
image auto-clears. Detection holds down to q=75. See `tests/test_detection.py`.

**Scope:** classical manipulation detection (splicing, copy-move, retouching, resampling)
comes first. Generative-AI detection is deliberately deferred until the classical stack is
solid — see the scope note in [`docs/DETECTORS.md`](docs/DETECTORS.md).

Next up: JPEG block-grid misalignment and double-compression detection.

## Layout

```
docs/DETECTORS.md      every detection method, what it exploits, and its blind spots
docs/DESIGN.md         architecture and evaluation protocol
src/groundtruth/
  core/                types, detector interface, registry
  detectors/           metadata · compression · sensor · geometric · generative · context
  fusion/              weighted combination, abstention, heatmap pooling
  pipeline/            orchestration
  api/                 CLI and overlay rendering
tests/fixtures.py      synthetic manipulations with pixel-exact ground-truth masks
```
