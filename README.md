# Ground Truth

Detecting manipulated images in insurance claims using camera and compression
physics rather than a trained real/fake classifier.

---

## Status: measured, partly working

Every claim below comes from `scripts/benchmark.py`, which generates manipulations
**on top of real photographs** across operation × size × laundering, with pixel-exact
masks and explicit controls. 88 cells:

```
operation            0.5%      2.0%      8.0%     25.0%
--------------------------------------------------------
pristine        100% quiet   (control — must not flag)
global_tone     100% quiet   (control — legitimate exposure lift)
clone_out         75%/0.34  100%/0.49  100%/0.49  100%/0.50
duplicate         75%/0.34  100%/0.49  100%/0.49  100%/0.50
splice_in          0%/0.02    0%/0.12    0%/0.45    0%/0.60
render_overlay     0%/0.37    0%/0.36    0%/0.18    0%/0.22
inpaint_out        0%/0.02    0%/0.07    0%/0.04    0%/0.06

manipulations detected : 30/80  (38%)
controls left alone    :   8/8  (100%)
```

**Works:** copy-move — content cloned out or duplicated within a photograph.
100% at 2% of frame and above, and it fires on nothing else (0/4 pristine,
0/4 global_tone, 0/16 render_overlay). That specificity is the point.

**Does not work:** foreign content spliced in, rendered overlays such as a replaced
licence plate, and removal by smooth fill.

**Two detectors are quarantined as `experimental` and excluded from the pipeline**
because the benchmark showed they never worked — see below. A detector earns its
place by being measured, not by being written.

---

## The earlier failure, and what it taught

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

### Three detectors that did not survive measurement

The benchmark exists because eyeballing single images kept producing false confidence.
It immediately caught two:

**`compression.block_grid`** required 2 windows to agree on a foreign grid phase.
Across N confident windows and 64 possible phases, chance supplies about N/64 —
roughly **10 on a full-frame photo**. The threshold sat below the noise floor, so it
fired on 3 of 4 *pristine* images and on every manipulation class equally. Raised to
beat chance, it finds nothing even on the splice it was built for.

**`geometric.sharpness_inconsistency`** fires at 7.2σ on a real photo with a
hand-replaced licence plate — but localises to the crumpled bumper and the tree line.
**0% of flagged pixels inside the actual edit.**

**`sensor.synthetic_region`** tested the conjunction those two failures suggested:
a rendered graphic is *sharp **and** noiseless*, where crumpled metal is sharp and
noisy. On one photograph it looked clean and selective. Swept across four, there is
no operating point at all:

| silence threshold | controls quiet | overlays detected |
|---|---|---|
| 0.8 | **0/8** | 4/4 |
| 1.1 | 4/8 | 0/4 |
| 1.4 | 8/8 | **0/4** |

It goes straight from flagging every clean photograph to detecting nothing — the
measurement does not separate the classes. Adding it took detection to 68% while
dropping controls from 100% to 38%, which is not a trade worth making.

All three are still in the tree, fully documented, and excluded from the default
pipeline. **This is the pattern worth noticing:** physically-motivated cues keep
looking convincing on one image and failing to be selective across several. That is
what the benchmark is for.

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

| Detector | Tier | Localises | State |
|---|---|---|---|
| `geometric.copy_move` | geometric | ✅ | **works** — 100% on duplication ≥2% of frame |
| `metadata.preview_mismatch` | metadata | ✅ | **works** — when a stale preview survives |
| `context.policy_consistency` | context | — | works — dispositive, no ML involved |
| `metadata.container_identity` | metadata | — | weak structural signal |
| `sensor.noise_inconsistency` | sensor | ✅ | fitted noise level function; rarely fires |
| `compression.ela` | compression | ✅ | baseline only, confidence capped at 0.30 |
| `compression.block_grid` | compression | ✅ | **experimental** — never worked, see above |
| `geometric.sharpness_inconsistency` | geometric | ✅ | **experimental** — mislocalises, see above |

Each detector's blind spots and measured envelope are in
[`docs/DETECTORS.md`](docs/DETECTORS.md).

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

## Benchmark

```bash
python scripts/benchmark.py              # operation x size
python scripts/benchmark.py laundering   # operation x laundering
```

Cells report **detected% / mean localisation**. Both numbers are needed: a detector
that flags everything scores perfectly on the first alone, and localisation is what
caught the sharpness detector firing 7σ away from the actual edit.

## Next

1. **Rendered overlays** (`render_overlay`, 0%) — the replaced-licence-plate case.
   Likely needs sharpness paired with *absence of sensor noise*; neither cue alone
   separates a rendered plate from crumpled metal.
2. **Splicing** (`splice_in`, 0%) — the current benchmark splices between photos
   from the same camera model, which share a noise level function. Cross-camera
   donors would be the honest harder test.
3. **Removal by fill** (`inpaint_out`, 0%) — needs the "too smooth" side, which is
   the mirror of the sharpness cue and currently unbuilt.
4. A JPEG corpus for the compression tier, which remains unvalidated on real data.
