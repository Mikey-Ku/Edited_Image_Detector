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

## The capability this is built toward

Sensor fingerprinting doesn't just localise tampering — it **identifies the individual
physical camera**. Cluster every image ever submitted by fingerprint, and any cluster
spanning supposedly-unrelated claimants is a **fraud ring**.

Each of those claims passes every check on its own. The fraud exists only in the
relationships between them.

## Status

🚧 Early. See [`docs/DETECTORS.md`](docs/DETECTORS.md) for the full detector catalogue and
build order, and [`docs/DESIGN.md`](docs/DESIGN.md) for architecture.

## Layout

```
docs/DETECTORS.md      every detection method, what it exploits, and its blind spots
docs/DESIGN.md         architecture and evaluation protocol
src/groundtruth/
  core/                types, detector interface, registry
  detectors/           metadata · compression · sensor · geometric · generative · context
  fusion/              calibrated combination + abstention
  pipeline/            orchestration
  api/                 service layer
```
