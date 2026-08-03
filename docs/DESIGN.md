# Design

## Architecture

```
                    ImageCase (image + claim context)
                                │
                    ┌───────────┴───────────┐
                    │   Detector Registry   │   runs every applicable detector
                    └───────────┬───────────┘
                                │
     ┌──────────┬──────────┬────┴─────┬──────────┬──────────┐
  metadata  compression  sensor   geometric  generative  context
     │          │          │          │          │          │
     └──────────┴──────────┴────┬─────┴──────────┴──────────┘
                                │  Evidence[]  (score, confidence, heatmap?)
                    ┌───────────┴───────────┐
                    │     Fusion layer      │   calibrated · handles missing detectors
                    └───────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    │    Decision layer     │   auto-clear / flag / route to human
                    └───────────┬───────────┘
                                │
                    Verdict + localisation + written explanation
                                │
                    ┌───────────┴───────────┐
                    │  Cross-claim index    │   PRNU clustering → ring detection
                    └───────────────────────┘
```

## Core design decisions

### Detectors are independent and know when they don't apply

Every detector implements `applicable(case) -> bool`. A JPEG detector on a PNG returns
`False` and is excluded from fusion — it does **not** return a neutral score. Missing
evidence and neutral evidence are different things and conflating them silently corrupts
the fused probability.

### Every detector reports confidence separately from score

`score` = how manipulated this looks. `confidence` = how much this detector trusts itself
on this input. A PRNU detector with four reference images is far less sure than one with
two hundred, and fusion has to know that.

### Three outcomes, not two

`AUTO_CLEAR` / `FLAG` / `ROUTE_TO_HUMAN`. A system that must decide on every input is
useless in a regulated setting. **The product is triage, not judgement** — the model's job
is to cut 100k images down to the 500 an adjuster must actually look at.

### Explanations are first-class

The output is not `0.87`. It's *"the embedded thumbnail shows an undamaged bumper; the JPEG
block grid is misaligned in that region; the capture timestamp precedes policy inception by
eleven days."* An adjuster has to act on this, and a regulator has to audit it.

---

## Evaluation protocol

### Headline metric: the risk–coverage curve

> At 94% auto-clear coverage, 99% of manipulations are still caught.

Accuracy is close to meaningless here — the base rate of fraud is low and the error costs
are wildly asymmetric. What a buyer evaluates is: *how much review labour does this remove,
and what does it miss?*

### Cost-weighted evaluation

A missed manipulation costs the claim value. A false positive costs one adjuster review
(~$50). These differ by three orders of magnitude, and the operating point must be chosen
in dollars, not in F1.

### Calibration

Reliability diagram + Expected Calibration Error. If the system says 0.9, it must be right
about 90% of the time. Uncalibrated confidence makes the abstention threshold meaningless.

### Generalisation — the experiment that matters

Hold out **entire generators and entire manipulation tools**, not random samples.

Train the learned detector on one generator, evaluate on an unseen one, and show accuracy
collapsing toward chance — while the physics-based detectors hold. This is the central
argument for the whole architecture, and it needs to be measured, not asserted.

### Robustness sweep

Every result reported at multiple degradation levels: JPEG re-encode at q ∈ {95, 75, 50},
downscale, screenshot round-trip, messaging-app re-encode. **A detector that only works on
pristine files is a detector that doesn't work**, because real claims arrive through email
and phone cameras.

### Splits

By **claimant**, never at random. The same claim's photos appearing in both train and test
leaks and inflates everything.

---

## Evaluation results

### Synthetic validation did not predict real performance — at all

**Date:** 2026-08-02 · **Data:** Korus Realistic Tampering Dataset, 112 tampered +
112 matched pristine originals, Nikon D7000 / D90 / Canon 60D, 1920×1080 TIFF,
hand-manipulated in GIMP and Affinity Photo with pixel-exact masks.

#### The result

| | synthetic fixtures | real photographs |
|---|---|---|
| localisation hit rate | **0.99** | **0.010** (median) |
| IoU | 0.59 | 0.004 (median) |
| clean images auto-cleared | 100% | **0%** |
| tampered flagged | 100% | 18.8% |
| pristine **falsely** flagged | 0% | **19.6%** |

The false positive rate is *higher* than the true positive rate. The score
distributions are indistinguishable:

```
tampered  P(manip): min 0.327  p25 0.424  median 0.561  p75 0.637  max 0.871
pristine  P(manip): min 0.327  p25 0.424  median 0.560  p75 0.655  max 0.871
```

This is not degradation. It is **chance performance**. The system has zero
discriminative power on real photographs.

#### Why — the noise detector's model of the world is wrong

Running `sensor.noise_inconsistency` alone on matched tampered/pristine pairs of
the *same scene*:

```
image            TAMPERED anom/meas   PRISTINE anom
r09696ba3t          189/1560              186
r1e3303ebt          192/1634              220     <- pristine higher
r2410fbc3t          256/1578              260     <- pristine higher
r36610622t            0/1683              317     <- pristine flagged, tampered clean
```

Mean separation: **−0.065**. Tampered scores higher than its own pristine original
on **4 of 14 pairs** — worse than a coin flip.

The detector assumes **one global noise level per image**, and treats local
deviation as evidence of splicing. That assumption is false for real photographs:

- **Photon shot noise scales with the square root of signal**, so bright regions
  are genuinely noisier than dark ones. Nothing was manipulated; the physics
  differs across the frame.
- In-camera processing, demosaicing, and local tone mapping vary the noise
  further.
- The adaptive structure filter drops the busiest 15% of blocks, which is enough
  on a smooth synthetic scene and nowhere near enough on a real photograph.

So on a real image, hundreds of blocks legitimately deviate — and they deviate
identically in the tampered and pristine versions, because those two files are the
same photograph apart from a region covering ~5% of the frame.

#### The actual lesson

**The fixture was built on the same false assumption as the detector.**
`noise_splice()` generates a scene with *uniform* noise and pastes in a region with
a different uniform level. That is precisely the world the detector models, so it
scored 0.99 — the test could not have failed. It was not measuring whether the
detector works; it was measuring whether the code matched its own premise.

Synthetic fixtures verify that a detector fires **in the right place given its
assumptions**. Only real data tests **whether those assumptions hold.** Both are
needed, and passing the first says nothing about the second.

#### What has to change

1. **Model noise as a function of local brightness** (a noise level function)
   rather than a global constant, and flag deviation from *that* model. This is
   the standard approach in the forensics literature and the synthetic fixture
   hid the need for it entirely.
2. **Add a matched-pair regression harness** — every future change measured on
   tampered vs. its own pristine original, since scene-level variation cancels out.
3. `metadata.container_identity` fires on all 224 images at score 0.55 because
   TIFF is lossless. For a corpus of RAW conversions that is a constant offset
   carrying no information, and it should abstain rather than nudge every score.
4. **A JPEG corpus is still needed.** These TIFFs mean `block_grid`, `ela`, and
   `preview_mismatch` all correctly abstained — 0 of 224 — so the compression tier
   is entirely unvalidated on real data.

#### Status of every earlier number in this repo

Every metric reported before this evaluation was measured on synthetic fixtures
and **should be read as an implementation check, not as performance**. They are
retained because they still serve that purpose, but no synthetic number in this
repository should be quoted as evidence that the system detects manipulation.

---

## Data

| Source | Use |
|---|---|
| `DocTamper` | document tampering, pixel-level masks |
| `CASIA v2`, `IMD2020`, `DEFACTO` | classical splicing/copy-move with masks |
| Korus Realistic Tampering | **primary source** — real cameras, hand-made edits, masks |
| Self-generated | implementation checks only — see the results above |

### Building the attacker

Generated manipulations give pixel-perfect ground truth and full control over attack
difficulty, which makes them good for verifying a detector fires in the right place. They
are **not** evidence that it works — see the results above for what happens when a fixture
encodes the same assumption as the detector it tests. Tiers:

1. **Crude** — copy-paste, no blending
2. **Competent** — Photoshop-equivalent: blended, colour-matched, resampled
3. **AI inpainting** — diffusion-based add/remove damage
4. **Fully synthetic** — generated damage photos end to end
5. **Laundered** — any of the above, then re-encoded/screenshotted to strip forensic traces

Reporting detection rate *per attack tier* is far more informative than a single number,
and tier 5 is where honest systems admit their limits.

---

## Non-goals

- Not a general-purpose forensics library — claim context is load-bearing
- Not a courtroom tool — output is triage, not proof
- Not attempting real-time video
