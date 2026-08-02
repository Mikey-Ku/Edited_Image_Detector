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

## Recovering the original

Often you don't have to infer what changed — **you can show it.**

Containers carry a second copy of the image: an EXIF thumbnail in a JPEG, a thumbnail item
in a HEIC, a reduced-resolution SubIFD in a TIFF, an embedded JPEG in most RAW files. Many
editors rewrite the main image and leave that copy untouched.

When that happens, **the preview is a photograph of the pre-edit original.** Not a
reconstruction, not a model's guess — real pixels that are still sitting in the file.
Diff the two and the edit localises itself.

```
groundtruth claim.jpg --recover before_after.png
```

```
recovered pre-edit image [recovered]
  source:  embedded_jpeg@0x59 at 160x120
  changed: 8.78% of the frame
```

A 160×120 thumbnail — 6% of the current resolution — located a synthetic edit at
`(379,151)-(560,300)` against a ground truth of `(380,150)-(560,300)`.

**One invariant governs this whole module.** Every result carries a `Fidelity`:
`RECOVERED` and `PARTIAL` are real pixels from the file and are evidence. `INFERRED` would
mean a model's plausible guess about what *might* have been there — a hypothesis, never
evidence. Nothing synthesised may ever be presented as recovered, and the distinction is
carried through the type system, the renderer's captions, and the tests.

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

### Review UI

```bash
pip install -e ".[ui]"
uvicorn groundtruth.api.server:app --port 8420
```

Drag in an image, or click one of the bundled examples in `samples/`. The UI is deliberately
a **review** tool rather than a verdict tool: every detector's score sits next to its
confidence, its explanation, and its raw measurements, and detectors that did not apply are
listed separately rather than hidden. A number without the reasoning behind it is exactly
what an adjuster cannot act on.

## Status

Working end-to-end. Six detectors, 65 tests.

| Detector | Tier | Localises | State |
|---|---|---|---|
| `context.policy_consistency` | context | — | capture time vs. policy inception and loss date; GPS vs. loss location |
| `metadata.container_identity` | metadata | — | magic bytes vs. filename; silent when nothing is anomalous |
| `metadata.preview_mismatch` | metadata | ✅ | recovers the pre-edit original from an embedded preview |
| `compression.block_grid` | compression | ✅ | 8×8 JPEG grid phase per window; flags regions carrying a foreign grid |
| `compression.ela` | compression | ✅ | baseline only — deliberately capped at low confidence, see source |
| `sensor.noise_inconsistency` | sensor | ✅ | MAD-based per-block noise estimation, adaptive structure exclusion, contiguity filter |

### ⚠️ Real-data result: this does not work yet

Evaluated on 112 tampered + 112 matched pristine originals from the **Korus Realistic
Tampering Dataset** — real photographs, hand-manipulated in GIMP and Affinity Photo:

| | synthetic fixtures | real photographs |
|---|---|---|
| localisation hit rate | 0.99 | **0.010** |
| clean images auto-cleared | 100% | **0%** |
| tampered flagged | 100% | 18.8% |
| pristine **falsely** flagged | 0% | **19.6%** |

The false positive rate exceeds the true positive rate and the two score distributions are
indistinguishable. **This is chance performance.**

Root cause: the noise detector assumes one global noise level per image, but photon shot
noise scales with the square root of signal, so bright regions of an *unmanipulated* photo
are legitimately noisier than dark ones. The synthetic fixture generated uniform noise —
**encoding the same false assumption the detector makes** — so the test could not fail.

Full analysis, including the matched-pair diagnostic, in [`docs/FINDINGS.md`](docs/FINDINGS.md).

**Every synthetic number in this repository should be read as an implementation check, not
as performance.** They verify a detector fires in the right place *given its assumptions*;
only real data tests whether those assumptions hold.

## Containers

Format is decided by **magic bytes, never by file extension** — the extension is metadata a
user controls, and a `damage.jpg` whose bytes are a PNG has been through a conversion the
claimant did not mention. That disagreement is itself reported as a finding.

JPEG · PNG · WebP · TIFF · **HEIC/HEIF** · BMP · GIF. HEIC matters more than it looks: it
has been the iPhone default since 2017, so in a claims pipeline it is the common case, not
an edge case.

**Scope:** classical manipulation detection (splicing, copy-move, retouching, resampling)
comes first. Generative-AI detection is deliberately deferred until the classical stack is
solid — see the scope note in [`docs/DETECTORS.md`](docs/DETECTORS.md).

**Documented limits.** Every detector's failure envelope is measured and pinned in
tests, not left to be discovered later. Block-grid analysis, for instance, fails entirely
when the composite is saved below ~q92 — the re-save's own grid overwrites the foreign one,
so the evidence is physically gone from the pixels. That is asserted as a test, because a
detector whose failure modes are undocumented is one you cannot deploy.

Next up: double-JPEG detection, then copy-move.

## Layout

```
docs/DETECTORS.md      every detection method, what it exploits, and its blind spots
docs/DESIGN.md         architecture and evaluation protocol
src/groundtruth/
  core/                types, detector interface, registry
  detectors/           metadata · compression · sensor · geometric · generative · context
  recovery/            embedded-preview extraction and pre-edit reconstruction
  fusion/              weighted combination, abstention, heatmap pooling
  pipeline/            orchestration
  api/                 CLI and overlay rendering
tests/fixtures.py      synthetic manipulations with pixel-exact ground-truth masks
```
