# Ground Truth

Detecting manipulated images in insurance claims using camera and compression
physics rather than a trained real/fake classifier.

---

## ⚠️ Status: does not work on real photographs yet

Measured on the [Korus Realistic Tampering Dataset](https://pkorus.pl/downloads/dataset-realistic-tampering)
— 112 tampered images and their 112 matched pristine originals, real cameras,
manipulated by hand in GIMP and Affinity Photo:

| | synthetic fixtures | real photographs |
|---|---|---|
| localisation hit rate | 0.99 | **0.010** |
| clean images auto-cleared | 100% | **0%** |
| tampered flagged | 100% | 18.8% |
| pristine **falsely** flagged | 0% | **19.6%** |

The false positive rate exceeds the true positive rate and the score distributions
are identical to three decimals. **That is chance performance.**

**Why:** the noise detector assumes one global noise level per image. Photon shot
noise scales with √signal, so bright regions of an untouched photo are genuinely
noisier than dark ones — hundreds of blocks legitimately deviate, identically in
both versions of the same scene.

**The lesson is the fixture, not the detector.** `noise_splice()` generated a scene
with *uniform* noise, encoding the same false assumption the detector makes, so the
synthetic test could not have failed. Synthetic fixtures verify a detector fires in
the right place *given its assumptions*; only real data tests whether those
assumptions hold.

**Every synthetic number in this repository is an implementation check, not a
performance claim.** Full analysis in [`docs/DESIGN.md`](docs/DESIGN.md#evaluation-results).

---

## Approach

An edited image is a physical impossibility, and the traces are properties of how
cameras work rather than of any particular editing tool:

- A sensor captures **one colour per pixel** and interpolates the rest. Synthesised
  pixels were never demosaiced — no sensor produced them.
- JPEG compresses on a rigid **8×8 grid** anchored to the origin. A pasted region
  brings its own grid, and only 1 offset in 64 lands back in phase.
- Every sensor carries a **unique noise fingerprint** from manufacturing variation.
- Lenses fail to focus all wavelengths equally, and the error grows **radially**
  from the optical centre.

On top sits the layer that actually catches fraud: **claim context.** A pixel-perfect
authentic photo is still fraud if its EXIF timestamp predates the policy.

### Recovering the original

Containers carry a second copy of the image — an EXIF thumbnail, a HEIC thumbnail
item, a TIFF SubIFD, the embedded JPEG in most RAW files. Editors routinely rewrite
the main image and leave it untouched, in which case **the preview is a photograph
of the pre-edit original**: real pixels, not a reconstruction.

Every result carries a `Fidelity` — `RECOVERED`/`PARTIAL` are evidence; `INFERRED`
would be a model's hypothesis and never is. That distinction is enforced in the
types, the renderer, the UI, and the tests.

---

## Use

```bash
pip install -e ".[dev]"

# CLI
groundtruth photo.jpg --render overlay.png --recover before_after.png \
    --policy-inception 2026-03-01 --loss-date 2026-06-15

# Review UI at http://localhost:8420
pip install -e ".[ui]"
uvicorn groundtruth.api.server:app --port 8420
```

The UI is a **review** tool, not a verdict tool: every score sits beside its
confidence, its explanation, and its raw measurements, and detectors that did not
apply are listed rather than hidden. Bundled examples in `samples/` pair every
manipulation with its control.

No API keys or environment configuration are required.

Containers are identified by **magic bytes, never by file extension** — JPEG, PNG,
WebP, TIFF, HEIC, BMP, GIF. A `damage.jpg` whose bytes are a PNG is reported as a
finding in its own right.

## Detectors

| Detector | Tier | Localises |
|---|---|---|
| `context.policy_consistency` | context | — |
| `metadata.container_identity` | metadata | — |
| `metadata.preview_mismatch` | metadata | ✅ |
| `compression.block_grid` | compression | ✅ |
| `compression.ela` | compression | ✅ |
| `sensor.noise_inconsistency` | sensor | ✅ |

Each detector's blind spots and measured operating envelope are in
[`docs/DETECTORS.md`](docs/DETECTORS.md). Block-grid analysis, for example, fails
entirely below ~q92 because the re-save's own grid overwrites the foreign one —
asserted as a test, because a detector whose failure modes are undocumented is one
you cannot deploy.

## Layout

```
docs/DETECTORS.md   detection methods, blind spots, build order
docs/DESIGN.md      architecture, evaluation protocol, results
src/groundtruth/
  core/             types, detector interface, registry, container IO
  detectors/        metadata · compression · sensor · context
  recovery/         embedded previews, pre-edit reconstruction
  fusion/           weighted combination, abstention, heatmap pooling
  api/              CLI, overlay rendering, review UI
scripts/            dataset salvage and evaluation
tests/fixtures.py   synthetic manipulations with ground-truth masks
```

## Next

1. Model noise as a function of local brightness instead of a global constant.
2. Matched-pair regression harness, so scene variation cancels.
3. A JPEG corpus — the Korus TIFFs meant the entire compression tier correctly
   abstained (0 of 224) and remains unvalidated on real data.
