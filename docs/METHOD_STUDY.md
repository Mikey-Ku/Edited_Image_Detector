# What actually detects a manipulated or synthetic image?

A controlled comparison of the detection methods in this repository against the
two methods everyone reaches for instead: a supervised real/fake classifier, and
Error Level Analysis.

This document is written **before the experiments are run**. The predictions in
section 5 are recorded here so that the results cannot be quietly reinterpreted
to match whatever comes out. Where a prediction is wrong, it stays on the page
and the wrongness is the finding.

---

## 1. Why this study exists

Every detector in this repository was built on one bet: that reading the physics
of capture generalizes better than learning what edited images looked like last
year. That bet has never been measured against the alternative. The README
records the moment it nearly went the other way, an early evaluation that
"reported 0.99 on a system that was at chance", but an anecdote in a docstring is
not evidence.

So the alternative gets built properly, given the data it needs to win, and
measured on the axis that actually matters: what happens when it meets something
it was not trained on.

## 2. Two questions, not one

The repository already separates these and the separation drives the whole design.

| | question | what it presumes |
|---|---|---|
| **Q1** | Was a real photograph altered after capture? | there was a capture |
| **Q2** | Did this image come from a camera at all? | nothing |

A method that answers Q1 well can be worse than useless on Q2. Measured once on a
real ChatGPT export, the pixel tiers returned `AUTO_CLEAR` at 0.288, **lower than
a genuine untouched photograph**, while the provenance tier flagged it at 0.876.
That single observation is the seed of Q2 and it needs a sample size.

## 3. The arms

Each arm has to be implemented well enough that its failure is informative. An
arm that loses because it was built carelessly teaches nothing.

| # | arm | family | status |
|---|---|---|---|
| 1 | CFA / demosaicing consistency | physics | built |
| 2 | JPEG block grid | physics | built |
| 3 | Copy-move (SIFT) | geometric | built |
| 4 | Content credentials (C2PA) | provenance | built |
| 5 | Error Level Analysis | classic baseline | **to build** |
| 6 | Camera-fingerprint net (contrastive, camera identity) | learned, no manipulations seen | trained 2 epochs, **needs a real run** |
| 7 | Fusion: hand-set log-odds | fusion | built, and its own docstring calls the weights defect 3 of 3 |
| 8 | Fusion: logistic regression | fusion, learned | **to build** |
| 9 | Fusion: gradient boosted trees | fusion, learned | **to build** |
| 10 | Supervised real/fake CNN | learned, trained on manipulations | **to build**, this is the control |
| 11 | Quantization-table-only classifier | shortcut probe | **to build**, see 5.6 |

Arm 11 is not a detector anyone would ship. It exists to measure how much of a
dataset can be solved without looking at the pixels at all.

## 4. Data and the transfer axes

| dataset | role | note |
|---|---|---|
| Korus realistic tampering | Q1, held-out | 224 images, 2 sensors, already here |
| CASIA v2 | Q1, training | ~12.6k images; corrected ground truth from `SunnyHaze/CASIA2.0-Corrected-Groundtruth` |
| VISION | camera identity | already here, trains arm 6 |
| GenImage subset | Q2 | 3 generators plus matched real, cross-generator track |

Three transfer axes, and they are the point of the study. In-distribution numbers
are reported but they are not the result.

- **Cross-sensor.** Tune on Korus camera A, test on camera B. Calibration is
  already known to break here (two sensors want different slopes, p = 0.02).
- **Cross-dataset.** Train on CASIA v2, test on Korus. Different editors,
  different cameras, different pipeline.
- **Cross-generator.** Train the Q2 classifier on one generator, test on the
  others. This is the "can a model detect other models" question.

Splits are grouped by camera for Q1 and by generator for Q2. An ungrouped random
split leaks and would make every number here meaningless.

## 5. Predictions, recorded before running

Stated as falsifiable claims with numbers. Each will be marked HIT or MISS.

**5.1** The supervised CNN reaches ROC AUC >= 0.95 on a held-out CASIA v2 split.

**5.2** The same CNN, unchanged, falls to <= 0.70 on Korus.

**5.3** The physics detectors move by < 0.10 AUC between CASIA and Korus. They are
weaker in-distribution and steadier across it.

**5.4** Learned fusion (arm 9) beats hand-set weights (arm 7) by >= 0.03 AUC
within a sensor.

**5.5** Learned fusion does **not** survive the cross-sensor split, mirroring the
calibration result. If it does survive, that is more interesting than if it does
not.

**5.6** ~~A classifier given **only JPEG quantization tables**, no pixels, reaches
>= 0.80 AUC on CASIA v2.~~ **Amended 2026-08-17, before running, see section 9.**

A classifier given **only cheap global statistics** and no forensic feature at all
(pixel dimensions, aspect ratio, file size, mean and variance per channel, and 8x8
blockiness energy) reaches >= 0.80 AUC on CASIA v2. If this holds, a large part of
what the literature reports on this dataset is bookkeeping rather than forensics,
and every CASIA number in this study has to be read in that light.

**5.7** On generated images the pixel physics tiers score **lower** than on real
photographs. Not merely uninformative, actively inverted.

**5.8** The Q2 classifier reaches >= 0.98 within its training generator and loses
>= 0.20 AUC on unseen generators. The published GenImage benchmark reports
in-generator above 98% and the best cross-generator average near 70%, so this
prediction is really a check that the harness reproduces a known result.

**5.9** Content credentials are close to perfect where present and absent on most
files. The limiting factor for provenance is coverage, not accuracy.

**5.10** The camera-fingerprint net, trained to convergence, lands within 0.02 AUC
of the same architecture with its high-pass first layer **frozen and never
trained**. If that holds, the learning in arm 6 is decoration and the
hand-derived filter was doing the work.

## 6. Metrics

ROC AUC with bootstrap confidence intervals **on the difference** between arms,
never on the two numbers separately. Plus, where the arm produces a probability:
ECE and Brier for calibration. Plus localisation lift against the mask area as
the chance baseline, since a heatmap covering the frame scores the mask fraction
for free. Plus wall-clock cost per image, because a method nobody can afford to
run is not a method.

## 7. Threats to validity

- **Korus is small.** 224 images and 2 sensors. Every Korus interval will be
  wide and will be reported as such rather than rounded away.
- **CASIA v2 is known to be biased.** That is why arm 11 exists. If 5.6 holds,
  CASIA is a training set here and never an evidence base.
- **A negative control can be strawmanned.** The supervised CNN gets a real
  architecture, real augmentation, and a real tuning budget. If it loses, it has
  to lose fairly, and the tuning budget spent on it is recorded.
- **One author, one harness.** Nothing here is independently replicated.

## 8. Results

### 8.1 Learned fusion, predictions 5.4 and 5.5

`scripts/learn_fusion.py`, replayed over the evidence already recorded by
`scripts/evaluate_korus.py`. 224 images, 10 live features after 11 constant ones
were dropped, cameras as groups. Canon_60D is skipped at 4 images.

**Within a sensor**, stratified 5-fold:

| camera | hand-set | logistic | boosted |
|---|---|---|---|
| Nikon_D7000 | **0.846** | 0.806 (-0.040, CI [-0.071, -0.011]) | 0.708 (-0.138, CI [-0.211, -0.073]) |
| Nikon_D90 | **0.750** | 0.680 (-0.070, CI [-0.133, -0.011]) | 0.591 (-0.160, CI [-0.239, -0.084]) |

**Across sensors**, train one camera and test the other:

| direction | hand-set | logistic | boosted |
|---|---|---|---|
| D7000 -> D90 | 0.750 | 0.753 (+0.002, CI [-0.041, +0.046]) | 0.772 (+0.022, CI [-0.016, +0.065]) |
| D90 -> D7000 | **0.846** | 0.808 (-0.037, CI [-0.095, +0.015]) | 0.751 (-0.095, CI [-0.182, -0.013]) |

**5.4 MISS, and not narrowly.** I predicted learned fusion would beat the hand-set
weights by at least 0.03 within a sensor. It lost in both cameras, by 0.04 to 0.16,
with every confidence interval on the paired difference excluding zero. The
gradient booster, the more flexible model, lost by more than the linear one in
every single split. That ordering is the tell.

**5.5 MISS, in the surprising direction.** I predicted learned fusion would fail
across sensors, following the calibration result. Instead the cross-sensor deltas
are the *better* ones: three of four intervals straddle zero, and boosted actually
edges ahead going D7000 to D90. Learning did worse where I expected it to do well
and about the same where I expected it to collapse.

**What this actually says.** Within-sensor 5-fold trains on 88 images with 10
features. That is not enough data to discover a fusion rule, so the learned models
fit their fold and the hand-set weights, which encode prior knowledge about what
each detector means, beat them. Cross-sensor trains on 110 and tests on a disjoint
110, which is a cleaner and larger fit, and the gap closes. The variable that moves
the result is training size, not the sensor boundary.

So `fusion/weighted.py` calling its hand-set weights defect 3 of 3 was, on this
evidence, wrong. At this data scale the hand-set weights are the better estimator
and the defect note should be rewritten to say so. What is *not* shown is that
learning cannot help in principle: nothing here rules out a learned fusion winning
with an order of magnitude more labelled forgeries. It says the fix is not free and
the repository does not currently have the data to buy it.

### 8.2 The CASIA shortcut floor, prediction 5.6 (amended)

`scripts/probe_shortcut.py` over all 12,614 images. No forensic feature of any
kind: no mask, no local structure, no comparison between regions. Stratified
5-fold, ROC AUC.

| feature set | logistic | boosted |
|---|---|---|
| shape (dimensions, aspect, pixel count, file size, bytes per pixel) | 0.633 | 0.647 |
| stats (per-channel mean and standard deviation) | 0.642 | 0.655 |
| both | 0.694 | **0.708** |

**MISS.** I predicted at least 0.80 and got 0.708.

**The number matters more than the verdict.** A classifier that has never looked
at a forgery, and cannot, separates CASIA's two halves at 0.708. The floor for
this dataset is therefore not 0.5, and any CASIA figure in this study has to be
read against 0.708 rather than against chance. A model reporting 0.95 on CASIA has
earned 0.24 of separation over knowing nothing, not 0.45.

The two feature sets are partly complementary, 0.647 and 0.655 alone against 0.708
together, so the tell is not a single artefact. CASIA's authentic and tampered
halves differ in both what the images are of and how big they are.

This does not indict the dataset for the use it is put to here. CASIA is the
training set, and a model that learns a resolution prior from it will simply not
find that prior on Korus, which is the transfer test working as intended.

### 8.3 The supervised control, predictions 5.1 and 5.2

`scripts/train_supervised.py`, 2.8M parameters trained from scratch on CASIA 2.0
for 40 epochs, then pointed at Korus unchanged. Both scored the same way, tiled at
native resolution, max over tiles.

| tested on | AUC | |
|---|---|---|
| CASIA held-out, 1,887 images | **0.975** | in-distribution |
| Korus, 277 images | **0.618** | different dataset, different cameras, different editors |

**5.1 HIT** at 0.975 against a 0.95 threshold. **5.2 HIT** at 0.618 against a 0.70
threshold. Both predictions land, and the gap between them is 35.8 points.

**Put next to the physics arms on the same Korus images:**

| method | Korus AUC |
|---|---|
| hand-set physics fusion, Nikon_D7000 | **0.846** |
| hand-set physics fusion, Nikon_D90 | **0.750** |
| supervised CNN trained on CASIA | 0.618 |

The supervised model beats every physics detector by a wide margin on the dataset
it was trained on, and loses to all of them on the dataset it was not. At 0.618 it
is closer to chance than it is to its own in-distribution score, and it sits
*below* the 0.708 that dimensions and channel statistics alone reach on CASIA.

This is the bet the repository was built on, and it is now measured rather than
asserted.

**Two things this is not.** It is not evidence that supervised detection cannot
transfer; it is evidence that this one, trained on this dataset, does not. And the
control was not handicapped into losing: it reached 0.975 in-distribution from
scratch, so whatever it failed at, it did not fail for want of capacity or budget.
The first attempt did fail that way, plateauing at 0.667 on random crops that
mostly contained no forgery, and that run was thrown away rather than reported.

### 8.4 The rest

Not yet run: Q2 cross-generator, and the fingerprint initialisation ablation.

## 9. Amendments

Changes made to this document after it was first committed. Each says what
changed, when, and why, so the record shows the reasoning rather than just the
final wording.

**2026-08-17, prediction 5.6, before any experiment was run.** The CASIA 2.0
mirror used by `scripts/fetch_casia.py` re-encodes every image to PNG:
`Au_ani_30215.jpg.png` opens as PNG and carries no quantization tables. The
original-JPEG shortcut therefore cannot be measured on this copy, because the
headers that would carry it no longer exist.

Two things follow, and they point in opposite directions. The measurement I wanted
is impossible here. But the shortcut itself is also *absent* from the data every
other arm sees, which makes this copy of CASIA a fairer training set than the
original, not a worse one. The supervised control cannot win by reading a header
that is not there.

What survives re-encoding is still worth probing, so 5.6 now asks a weaker version
of the same question: how much of CASIA can be solved with no forensic feature at
all. Pixel-level 8x8 blocking survives a PNG round-trip, and so do the resolution
and file-size distributions, which are a documented source of bias in this dataset.

The stronger quantization-table probe stays on the table if the original JPEG
release is obtained later. If that happens it will be added as a separate
prediction with its own number rather than by editing this one.
