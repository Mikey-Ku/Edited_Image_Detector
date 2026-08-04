# Retrace

Detecting manipulated images in insurance claims using camera and compression
physics rather than a trained real/fake classifier.

---

## Status

Headline numbers come from the [Korus Realistic Tampering Dataset](https://pkorus.pl/downloads/dataset-realistic-tampering):
**112 forgeries made by hand in GIMP and Affinity Photo, and their 112 matched
pristine originals** — real cameras, someone else's data, pixel-exact ground truth.

```
ROC AUC                     0.792   95% CI [0.731, 0.847]
  Nikon D7000 / D90         consistent across both cameras

threshold     TPR     FPR
     0.70   26.8%    3.6%
     0.75   14.3%    0.0%    zero false positives on 112 clean photographs

clean photographs auto-cleared        74.1%
localisation, flagged images          hit rate 0.484, IoU 0.143
                                      11.2x better than the mask area alone
```

The 0.75 row is the one that matters for deployment: a high-confidence tier that
caught 1 forgery in 7 without a single false accusation across 112 clean images.

**What works, measured by how much removing it costs the pipeline:**

| detector | alone | pipeline without it | localises on | lift |
|---|---|---|---|---|
| `sensor.noiseprint_structure` | 0.687 | −0.105 | 95/112 | 11.2× |
| `geometric.copy_move` | 0.647 | −0.075 | 41/112 | 11.0× |
| `sensor.noiseprint_anomaly` | 0.526 | −0.004 | 10/112 | 3.7× |

Every detector in the default set earns its place: the full set *is* the
best-scoring subset, and no detector has a positive delta.

`noiseprint_anomaly` looks marginal by AUC and is kept deliberately. Compared at
matched false positive rate it catches **7 forgeries the structural readout misses
at FPR 1.8%, and none the other way round (p = 0.016)** — a nested win in the
high-precision corner that AUC, integrating over the whole curve, cannot see. That
is a good reason not to select detectors on AUC alone.

**Still unsolved:** `P(manipulated)` is uncalibrated, so 0.70 is a hand-set number
rather than a probability. Images with *perfect* localisation currently report 0.46
to 0.55 and route to human review. Calibration is the next piece of work.

**Four detectors are quarantined as `experimental`** and excluded from the pipeline
because measurement showed they do not work — see below. A detector earns its place
by being measured, not by being written.

### Synthetic benchmark

`scripts/benchmark.py` generates manipulations on top of real photographs across
operation × size × laundering with explicit controls. It is an **implementation
check** — does a detector fire in the right place given its assumptions — not a
performance claim. The Korus numbers above are the performance claim.

```
operation            0.5%      2.0%      8.0%     25.0%
--------------------------------------------------------
pristine        100% quiet   (control — must not flag)
global_tone     100% quiet   (control — legitimate exposure lift)
clone_out         75%/0.34  100%/0.58  100%/0.49  100%/0.50
duplicate         75%/0.34  100%/0.58  100%/0.49  100%/0.50
render_overlay     0%/0.37    0%/0.36   25%/0.98  100%/1.00
splice_in          0%/0.02    0%/0.12    0%/0.45    0%/0.60
inpaint_out        0%/0.02    0%/0.46    0%/0.24    0%/0.51
```

---

## The earlier failure, and what it taught

The first Korus evaluation, before any of the above:

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

### Four detectors that did not survive measurement

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

**`sensor.noise_inconsistency`** was the last hand-crafted detector still shipping,
and it survived one round of repair: the global-noise assumption above was replaced
with a fitted noise level function, σ²(μ) = aμ + b, which is a genuinely better model
of a photograph. Re-measured on all 224 Korus images it still scored **AUC 0.494 —
below chance** — firing on 68% of forgeries and 66% of untouched originals, with the
pristine mean fractionally *higher*. Removing it raised the pipeline from 0.665 to
0.686.

The reason is worth keeping: at 32px the residual variation around the fit — texture,
focus falloff, local content — is larger than the difference a careful human retoucher
leaves behind. **A better model of the wrong quantity does not become the right
quantity.**

All four are still in the tree, fully documented, and excluded from the default
pipeline. **This is the pattern worth noticing:** physically-motivated cues keep
looking convincing on one image and failing to be selective across several. That is
what measurement is for.

### What replaced them

The fingerprint detectors reduce each block of a learned residual to a single
statistic, and **which statistic** turned out to matter more than anything else about
them. The first version used the block's standard deviation — a scalar magnitude,
invariant to shuffling the block's pixels. A camera fingerprint lives in the
arrangement, so the statistic was blind to the quantity it existed to measure.

A block-size sweep from 16 to 64 px was flat, 0.527 to 0.567, which was the clue:
geometry cannot matter while the summary discards the structure. So the multi-scale
work that flat curve seemed to call for would have bought nothing.

Measuring **period-2 structure** instead — energy at the 2×2 spatial frequency,
normalised by the block's own energy, because a sensor interpolates across a Bayer
grid and a rendered or inpainted region was never demosaiced:

| readout | held-out AUC (30 pairs unseen by every choice made) |
|---|---|
| `energy` | 0.517 [0.367, 0.667] — interval spans chance |
| `period2` | **0.677** [0.539, 0.815] |

Paired sign test: `period2` separates the tampered/pristine pair on **26 of 30**
images, p = 3.0e-05. Block size then matters exactly as the physics predicts —
estimating a periodic pattern needs samples — rising from 0.580 at 16 px to 0.679
at 48.

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
| `sensor.noiseprint_structure` | sensor | ✅ | **works** — held-out AUC 0.677; localises 95/112 at 11.2× |
| `geometric.copy_move` | geometric | ✅ | **works** — AUC 0.647; localises 41/112 at 11.0× |
| `sensor.noiseprint_anomaly` | sensor | ✅ | **works at low FPR** — 7 catches the structural readout misses, p=0.016 |
| `metadata.preview_mismatch` | metadata | ✅ | **works** — when a stale preview survives |
| `context.policy_consistency` | context | — | works — dispositive, no ML involved |
| `metadata.container_identity` | metadata | — | weak structural signal; untestable on Korus (all TIFF) |
| `compression.ela` | compression | ✅ | baseline only, confidence capped at 0.30 |
| `compression.block_grid` | compression | ✅ | **experimental** — never worked, see above |
| `geometric.sharpness_inconsistency` | geometric | ✅ | **experimental** — mislocalises, see above |
| `sensor.synthetic_region` | sensor | ✅ | **experimental** — no operating point, see above |
| `sensor.noise_inconsistency` | sensor | ✅ | **experimental** — AUC 0.494 on 224 real photographs |

Both fingerprint detectors run on one shared inference pass. Both abstain on images
with no demosaicing structure — screenshots, renders, generated images — rather than
reading a fingerprint off a residual that cannot contain one.

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

## Measuring

```bash
python scripts/evaluate_korus.py      # the real number: 224 photographs, ground truth
python scripts/ablate.py              # replay fusion over detector subsets, offline
python scripts/benchmark.py           # synthetic implementation check
python scripts/sweep_readout.py       # compare block statistics on cached residuals
python scripts/validate_readout.py    # score a fixed config on held-out pairs
```

`evaluate_korus.py` records every detector's raw reading, so `ablate.py` answers
"what if this detector were removed" in a second rather than a 28-minute run.
Questions that cost 28 minutes do not get asked.

The Korus dataset is **educational and research use only** and is not redistributed
here — `scripts/salvage_zip.py` recovers it from the published archive.

## Next

1. **Calibrate the fusion.** `P(manipulated)` is not a probability: thresholds are
   hand-set and images with *perfect* localisation report 0.46–0.55. Fit on one
   camera, test on the other, report a reliability diagram. This is the single thing
   most limiting the system's usefulness.
2. **Splicing** (`splice_in`, 0%) — the synthetic benchmark splices between photos
   from the same camera model, which share a noise level function. Cross-camera
   donors would be the honest harder test.
3. **Removal by fill** (`inpaint_out`, 0%) — needs the "too smooth" side, which is
   the mirror of the sharpness cue and currently unbuilt.
4. A JPEG corpus for the compression tier, which remains unvalidated on real data —
   every Korus image is a TIFF, so `compression.ela` and `metadata.container_identity`
   have never been exercised on data that could falsify them.
