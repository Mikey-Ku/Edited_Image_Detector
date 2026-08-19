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

**5.6** A classifier given **only JPEG quantization tables**, no pixels, reaches
>= 0.80 AUC on CASIA v2. If this holds, a large part of what the literature
reports on this dataset is compression bookkeeping rather than forensics, and
every CASIA number in this study has to be read in that light.

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

Not yet run.
