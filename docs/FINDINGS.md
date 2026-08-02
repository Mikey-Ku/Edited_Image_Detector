# Findings

## 1. Synthetic validation did not predict real performance — at all

**Date:** 2026-08-02 · **Data:** Korus Realistic Tampering Dataset, 112 tampered +
112 matched pristine originals, Nikon D7000 / D90 / Canon 60D, 1920×1080 TIFF,
hand-manipulated in GIMP and Affinity Photo with pixel-exact masks.

### The result

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

### Why — the noise detector's model of the world is wrong

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

### The actual lesson

**The fixture was built on the same false assumption as the detector.**
`noise_splice()` generates a scene with *uniform* noise and pastes in a region with
a different uniform level. That is precisely the world the detector models, so it
scored 0.99 — the test could not have failed. It was not measuring whether the
detector works; it was measuring whether the code matched its own premise.

Synthetic fixtures verify that a detector fires **in the right place given its
assumptions**. Only real data tests **whether those assumptions hold.** Both are
needed, and passing the first says nothing about the second.

### What has to change

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

### Status of every earlier number in this repo

Every metric reported before this evaluation was measured on synthetic fixtures
and **should be read as an implementation check, not as performance**. They are
retained because they still serve that purpose, but no synthetic number in this
repository should be quoted as evidence that the system detects manipulation.
