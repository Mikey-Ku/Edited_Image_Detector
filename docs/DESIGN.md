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

## Data

| Source | Use |
|---|---|
| `DocTamper` | document tampering, pixel-level masks |
| `CASIA v2`, `IMD2020`, `DEFACTO` | classical splicing/copy-move with masks |
| Self-generated | **primary source** — see below |

### Building the attacker

The most valuable dataset is the one we generate, because it gives pixel-perfect ground
truth and full control over attack difficulty:

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
