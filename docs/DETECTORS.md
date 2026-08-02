# Detector Catalogue

Every way we know of to tell whether an image is what it claims to be.

**Organising principle:** no single detector works. Every one of these has a blind spot,
and an attacker who knows about one can defeat it. The system's value comes from running
many *independent* signals and fusing them — an attacker has to defeat all of them
simultaneously, and each one they defeat costs them something in another.

Each detector is scored on:

- **Exploits** — the physical or statistical property that makes it work
- **Catches / Misses** — honest failure modes
- **Cost** — implementation difficulty (1 = a day, 5 = a research project)
- **Localises?** — does it produce a heatmap, or only an image-level score

---

## Tier 1 — Container & Metadata Forensics

Cheap, fast, and embarrassingly effective. Most real-world fraud is lazy. Run these first;
they cost milliseconds and resolve a surprising fraction of cases.

### 1.1 EXIF field analysis
- **Exploits:** editors stamp themselves into the file. `Software: Adobe Photoshop 26.0` is a confession.
- **Catches:** careless edits, screenshots, re-saves, format conversions
- **Misses:** anyone who strips EXIF (trivial) — but *stripped EXIF is itself a signal*, since phone-camera originals essentially always carry it
- **Cost:** 1 · **Localises:** no

### 1.2 Quantisation table fingerprinting
- **Exploits:** JPEG encoders each use distinctive quantisation tables. Apple's differ from Samsung's, which differ from Photoshop's "Save for Web," which differ from Python's PIL.
- **Why it matters:** the claim says "photo from my iPhone." The table says libjpeg. Contradiction.
- **Catches:** any re-encode outside the claimed capture device
- **Misses:** re-encodes that coincidentally match; a sophisticated attacker can transplant tables
- **Cost:** 2 · **Localises:** no
- **Note:** build a reference library of tables keyed by device/software. This is a genuine asset and mostly nobody has one.

### 1.3 Embedded thumbnail mismatch ⭐
- **Exploits:** JPEGs carry a small embedded preview. **Many editors modify the main image and forget to regenerate the thumbnail.**
- **Why it's great:** you literally get a picture of the *original*. Diff thumbnail vs. downscaled main image and the edited region lights up.
- **Cost:** 2 · **Localises:** YES — and it's the cheapest localisation in the entire system
- **Verdict:** implement early. Low cost, high drama in a demo.

### 1.4 C2PA / Content Credentials
- **Exploits:** cryptographic provenance signed at capture. Increasingly present on new phones and in generative tools.
- **Asymmetry:** presence of a valid chain is *strong* evidence; absence proves nothing (yet). Treat as a positive-only signal.
- **Cost:** 2 · **Localises:** no

---

## Tier 2 — Compression Forensics

Exploits the fact that JPEG is lossy and **remembers its own history**. This is the first
tier with real mathematics, and it's where most classical forgery detection lives.

### 2.1 Double-JPEG detection
- **Exploits:** compressing an already-compressed image leaves *periodic artefacts in the DCT coefficient histograms*. Quantise by q₁, then by q₂, and the histogram develops characteristic gaps and spikes.
- **Catches:** any image saved, edited, and re-saved — which is nearly every manipulated image
- **Misses:** single-compression forgeries (edit a RAW/PNG, save once); q₂ that's a multiple of q₁
- **Cost:** 3 · **Localises:** partially (per-block)
- **Maths:** histogram of DCT coefficients per frequency bin; look for periodicity via FFT of the histogram.

### 2.2 Block Artefact Grid (BAG) misalignment ⭐
- **Exploits:** JPEG compresses in a rigid **8×8 grid** anchored to the image origin. Paste a region from another JPEG and its internal block grid almost never aligns with the host's.
- **Why it's strong:** it's a *geometric* argument, not a statistical one. Misalignment is nearly impossible to produce accidentally.
- **Catches:** splices, copy-paste from another JPEG
- **Misses:** edits that happen to land on a grid boundary (1 in 64); regions painted rather than pasted
- **Cost:** 3 · **Localises:** YES
- **✅ Implemented** as `compression.block_grid`. **Measured operating envelope** — narrower
  than the literature implies, and worth stating plainly:
  - The composite must be saved at **≥ ~q92**. At q88 detection fails completely (0 of ~100
    readable windows misaligned) because the re-save's own origin-anchored grid overwrites
    the foreign one. The evidence is gone from the pixels; no estimator recovers it.
  - The donor must have been compressed at moderate-to-low quality, or there is no grid to read.
  - The image needs texture. Heavy noise destroys the grid outright — JPEG spends its bits
    encoding noise instead of quantising smooth blocks.
- **Two implementation traps**, both of which produced a detector that fired on every image:
  - **Raw step size is useless on textured content** — texture raises all eight phases
    together. Measure the *excess* of a step over its immediate neighbours, which cancels
    texture because busy content inflates a boundary and its neighbours equally.
  - **The confidence ratio needs an absolute scale floor.** Normalising the phase peak by
    the observed spread of the other seven diverges whenever they happen to agree, reading
    as overwhelming confidence in what is actually a featureless region.
- **Not yet verified:** the reported phase difference is *related* to the paste displacement
  mod 8, but the geometric correspondence has not been checked against ground truth — on
  synthetic splices the y component tracks the true displacement while x frequently reads
  zero. Reported as an observed phase difference, not as a recovered paste offset.

### 2.3 JPEG ghosts
- **Exploits:** re-save the image at every quality 1–100 and measure error. A region originally compressed at q=70 shows an **error minimum at 70** while the rest of the image doesn't.
- **Catches:** spliced regions with a different compression history
- **Cost:** 2 · **Localises:** YES (but noisy)

### 2.4 Error Level Analysis (ELA)
- **Exploits:** re-save at fixed quality, diff against original; regions with different compression history show different error.
- **⚠️ Honest assessment:** ELA is the most famous and most *overrated* technique in this field. It's noisy, easy to misread, and produces confident-looking garbage on textured regions. **Include it — but as a baseline we beat, and write up why it's weak.** That writeup is itself a differentiator: it shows you evaluated a popular method and found it wanting, which is what an actual practitioner does.
- **Cost:** 1 · **Localises:** YES (unreliably)

---

## Tier 3 — Sensor & Camera Physics ⭐ *the deep tier*

This is where the real mathematics lives, and it's the tier almost nobody implements
because it requires understanding how a camera physically works. **This is your ECE
background paying rent.**

### 3.1 PRNU — Photo Response Non-Uniformity ⭐⭐
- **Exploits:** silicon manufacturing is imperfect. Every sensor photosite has a slightly different sensitivity, producing a **fixed multiplicative noise pattern unique to that individual sensor** — not the model, the specific physical unit. A camera fingerprint.
- **How:** denoise the image, take the residual, correlate against a reference fingerprint. Tampered regions have suppressed or foreign PRNU.
- **Two enormous payoffs:**
  1. **Tamper localisation** — regions where the fingerprint went missing
  2. **Source linking** — *"these fourteen claims from supposedly unrelated people were all photographed on the same physical phone."* That is a **fraud ring detector**, and it falls out of the same machinery for free.
- **Misses:** needs enough reference images to estimate a fingerprint; weakened by heavy compression, scaling, and social-media re-encoding
- **Cost:** 4 · **Localises:** YES
- **Verdict:** the single most valuable detector in the system. The cross-claim linking capability is a genuinely novel angle for a portfolio project.

### 3.2 CFA / demosaicing periodicity ⭐
- **Exploits:** a sensor captures **one colour per pixel** (Bayer pattern); the other two are interpolated. This leaves a rigid periodic correlation structure between neighbouring pixels.
- **Why it's powerful:** *synthesised pixels were never demosaiced.* AI-generated and inpainted regions have no Bayer structure — they can't, because no sensor produced them.
- **Catches:** AI inpainting, generated regions, heavy retouching
- **Misses:** survives poorly through aggressive resizing/recompression
- **Cost:** 4 · **Localises:** YES
- **Note:** this is one of the best AI-inpainting detectors that exists, and it predates AI entirely.

### 3.3 Noise level inconsistency
- **Exploits:** sensor noise scales predictably with ISO and local brightness. Estimate local noise variance across the image; a spliced region carries its *source's* noise level.
- **Cost:** 3 · **Localises:** YES

### 3.4 Chromatic aberration consistency
- **Exploits:** lenses fail to focus all wavelengths identically, and the effect grows **radially from the optical centre**. Every real photo has a smooth, physically-constrained CA field.
- **Why it's elegant:** a pasted object carries CA appropriate to *its original position in a different frame*. It's essentially impossible to fake without modelling optics.
- **Cost:** 4 · **Localises:** YES

---

## Tier 4 — Geometric & Physical Consistency

The "physics doesn't lie" tier. Slower, harder, but extremely convincing when it fires —
and highly explainable, which matters enormously for a human adjuster.

### 4.1 Resampling / interpolation detection
- **Exploits:** scaling or rotating a region requires interpolating between pixels, which induces **periodic linear correlations** among neighbours. Detectable as spectral peaks in the second derivative.
- **Method:** Popescu–Farid — EM to estimate a per-pixel probability of being interpolated, then look for periodicity in that map.
- **Cost:** 4 · **Localises:** YES

### 4.2 Copy-move detection
- **Exploits:** the most common real-world edit is duplicating content *within* the same image — cloning out a pre-existing scratch, or duplicating damage to inflate a claim.
- **Method:** SIFT/ORB keypoints → match within image → geometric verification (RANSAC) to reject coincidence.
- **Cost:** 3 · **Localises:** YES (and beautifully — you draw the correspondence)
- **Insurance relevance:** very high. "Same dent, three places."

### 4.3 Lighting direction consistency
- **Exploits:** every object in a scene is lit by the same light sources. Estimate light direction from shading on each object; inconsistency is physically impossible.
- **Cost:** 5 · **Localises:** per-object

### 4.4 Shadow geometry
- **Exploits:** shadow rays must converge to a single light position. Draw lines from object points through shadow points; they must meet.
- **Bonus:** with sun position + date + GPS, you can verify the **claimed time of day**.
- **Cost:** 5 · **Localises:** per-object

---

## Tier 5 — Generative-AI Specific

The new attack surface. These target how diffusion models physically construct images.

### 5.1 Frequency-domain spectral signature ⭐
- **Exploits:** generators build images through repeated **upsampling**, which leaves periodic grid artefacts in the Fourier spectrum. Real photographs have a smooth ~1/f falloff; generated ones have peaks.
- **Catches:** GAN and diffusion output, especially fully-synthetic images
- **Misses:** weakens after recompression/downscaling; newer architectures are getting cleaner
- **Cost:** 2 · **Localises:** patch-wise
- **Verdict:** cheap, mathematically clean, great early win.

### 5.2 DIRE — Diffusion Reconstruction Error ⭐
- **Exploits:** run the image backwards then forwards through a diffusion model. **Generated images live on the model's learned manifold and reconstruct almost perfectly; real photographs don't.**
- **Why it's clever:** you use the generator itself as the detector. Very much the "understand the mechanism" move.
- **Cost:** 4 · **Localises:** YES (reconstruction error map)

### 5.3 VAE / latent reconstruction residual
- **Exploits:** latent diffusion models push everything through a VAE. Anything that came out of one **reconstructs through that VAE with near-zero error** — it's already in the codebook. Real photos don't.
- **Cost:** 3 · **Localises:** YES
- **Verdict:** cheaper than DIRE, similar idea, strong on inpainting.

### 5.4 Inpainting boundary detection
- **Exploits:** the *most likely real fraud* — "add damage here" / "remove the pre-existing dent" — creates a seam where generated meets original. Local statistics change discontinuously.
- **Cost:** 3 · **Localises:** YES
- **Verdict:** highest-priority AI detector for the claims use case specifically.

### 5.5 Learned detector (CNN / ViT baseline)
- **Exploits:** whatever the network finds.
- **⚠️ The known failure — and it's the most interesting result in the project:** learned detectors **do not generalise across generators.** Train on Stable Diffusion, test on Midjourney, watch accuracy collapse toward chance. This is well documented and it's exactly why the physics-based detectors matter.
- **Cost:** 3 · **Localises:** with attention/patches
- **Verdict:** build it as the baseline, then *demonstrate its collapse under generator shift while the physics detectors hold*. **That experiment is the single most compelling result the project can produce.**

---

## Tier 6 — Claim Context ⭐ *the differentiator*

Everything above is generic image forensics. **This tier is what makes it an insurance
fraud system instead of a forensics library** — and it's where the biggest wins hide,
because these catch fraud even when the image is completely unedited.

### 6.1 Prior-submission / near-duplicate detection ⭐
- **Exploits:** perceptual hashing against every image ever submitted.
- **Catches:** the *same damage photo* filed twice, by the same person or across a ring
- **Cost:** 2 · **Verdict:** cheap, and catches real fraud that no pixel analysis would ever see.

### 6.2 Off-the-internet detection
- **Exploits:** the photo was lifted from a used-car listing, a repair-shop portfolio, or stock imagery.
- **Cost:** 3

### 6.3 EXIF timestamp vs. policy inception ⭐⭐
- **Exploits:** the photo was taken **before the policy started.** The damage predates coverage.
- **Why it's the best signal in the whole system:** it's a single integer comparison, requires no ML at all, and it is *dispositive*. Pixel-perfect authenticity is irrelevant if the timestamp proves the damage predates the policy.
- **Cost:** 1 · **Verdict:** implement this on day one. It's a reminder that the smartest system isn't always the most sophisticated one.

### 6.4 GPS vs. reported loss location
- **Exploits:** claimed accident in Boston, EXIF GPS says Ohio.
- **Cost:** 1

### 6.5 Weather / environment cross-check
- **Exploits:** hail damage claimed on a date with no hail at that location. Public historical weather APIs.
- **Also:** shadow direction vs. claimed time of day, foliage state vs. claimed season, wet pavement vs. recorded conditions.
- **Cost:** 2

### 6.6 Cross-claim PRNU clustering ⭐⭐
- **Exploits:** §3.1's fingerprint, applied *across* claims. Cluster all submitted images by sensor fingerprint. Any cluster spanning multiple unrelated claimants is a **fraud ring**.
- **Why it's the best idea in this document:** individual claims each pass every check. The fraud only exists in the relationship between them. This is "The Ring" and "Provenance" fused into one system, and it comes almost free once PRNU exists.
- **Cost:** 4 (given 3.1) · **Verdict:** this is the headline capability. Build toward it.

---

## Fusion

Each detector emits `(score, confidence, localisation_map | None)`. The fusion layer must:

1. **Handle missing detectors** — many won't apply (no EXIF, not a JPEG, no PRNU reference)
2. **Be calibrated** — output a probability that means what it says, verified on a reliability diagram
3. **Be explainable** — an adjuster needs *"line item 3 was inpainted; the block grid is misaligned; the timestamp predates the policy,"* not `0.87`
4. **Abstain** — three outcomes, not two: `auto-clear` / `flag` / `route to human`

**Headline metric: the risk–coverage curve.** *"At 94% auto-clear, we still catch 99% of
manipulations."* That single chart is worth more than any accuracy number, because it's
the one an actual buyer evaluates.

---

## Scope: classical manipulation first, generative detection deferred

**Tier 5 is deliberately not being built yet.** Everything else — splicing, copy-move,
retouching, resampling, recompression — is classical manipulation, and it is the right
place to start for three reasons:

1. **It is verifiable.** Classical forensics rests on properties of cameras and codecs
   that we can construct ground truth for and measure against. Generative detection rests
   on properties of models that change every few months.
2. **The maths is settled.** Double-JPEG, PRNU, CFA, and resampling detection are
   decades-old, well-founded results. We can implement them correctly and *prove* it.
3. **It is still most of the real fraud.** Doctored claim photos are overwhelmingly
   Photoshop-class edits — cloned-out damage, pasted damage, retouched severity.

Tier 5 gets added once the classical stack is solid and measured. Building it first would
mean chasing a moving target with no reliable baseline to compare against.

---

## Build order

Ordered by (value ÷ cost), not by tier:

| Phase | Detectors | Status |
|---|---|---|
| **0** | 6.3, 6.4, 1.1 — claim-context consistency | ✅ done |
| **1** | 1.3 preview mismatch · 2.4 ELA baseline · 3.3 noise inconsistency | ✅ done |
| **1b** | multi-container support · **pre-edit recovery from embedded previews** | ✅ done |
| **2** | 2.2 block-grid misalignment | ✅ done |
| **2b** | 2.1 double-JPEG | ← next |
| **3** | 4.2 copy-move · 4.1 resampling | |
| **4** | 3.1 PRNU · 3.2 CFA — the deep tier | |
| **5** | 6.1 near-duplicate · **6.6 cross-claim rings** | the finale |
| *later* | Tier 5 generative + generalisation experiment | deferred, see above |

**Ship a working end-to-end system before deepening any single detector.** Five detectors
that work and are measured beat twenty that are half-finished.

---

## The experiment that makes the project

Once Phase 4 lands, the argument writes itself: hold out entire *manipulation tools* rather
than random samples, and show that detectors grounded in camera physics (PRNU, CFA,
block-grid) transfer to tools they never saw, while a learned real/fake classifier trained
on one tool collapses toward chance on another.

That is a real result, it is the strongest possible argument for this architecture, and it
is exactly the kind of finding nobody in a portfolio project produces.
