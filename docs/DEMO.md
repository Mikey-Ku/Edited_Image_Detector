# Running the demo

How to drive Retrace by hand, which images to use, and what each one is supposed to
prove. Written so a recorded walkthrough can follow it top to bottom.

```bash
pip install -e ".[dev,ui,learned]"
uvicorn groundtruth.api.server:app --port 8420
```

Then open <http://localhost:8420>. Nothing leaves the machine.

Skipping the `learned` extra is a supported install, but it drops the two
fingerprint detectors, which are the two that carry the interesting result. Do not
record a demo without it.

---

## The images, and what each one shows

### Bundled samples, one click from the page

These ship in `samples/` and appear as buttons under the drop zone. All are JPEG,
which matters: re-encoding destroys the demosaicing structure, so **the camera
fingerprint detectors abstain on every one of them**. Copy-move still runs, because it
works on keypoints rather than on sensor structure.

| Sample | Verdict | What it demonstrates |
|---|---|---|
| `courtyard cloned window` | **FLAG**, 0.795 | **Lead with this one.** A decorative window clone-stamped along the wall. The file has no embedded preview, so there is no original anywhere in the picture: copy-move catches 212 keypoint pairs sharing a displacement of (400, -10) px, and the image is convicted on its own internal inconsistency. |
| `edited stale preview` | **FLAG**, 0.90 | The editor rewrote the image and left the embedded preview behind, so the original is recoverable. Near-conclusive when it applies, and it is the one path that contributes *nothing* to the benchmark number. |
| `rooftops stale preview` | **FLAG**, 0.73 | Same mechanism on a real photograph. 10.4% of the frame changed. |
| `courtyard stale preview` | **ROUTE TO HUMAN**, 0.67 | The second worked example on the landing page. Sits in the middle band, so the third outcome is visible. |
| `splice noise mismatch` | **ROUTE TO HUMAN**, 0.60 | Compression error fires weakly. Note that its confidence is capped at 0.30, so it can never decide alone. |
| `clean uniform noise` | **ROUTE TO HUMAN**, 0.40 | The unedited control for the splice above. |
| `courtyard edited` | **AUTO CLEAR**, 0.239 | **Show this one.** A genuinely edited photograph the system does not catch. |
| `courtyard original` | **AUTO CLEAR**, 0.240 | Its untouched twin, scoring within 0.001 of the edit above. |

That last pair is the honest part of the demo and worth saying out loud: with no
embedded preview to compare against, a re-encoded JPEG gives the sensor detectors
nothing to read, and the system cannot separate a hand-edited photograph from its own
original. The UI groups and labels these as real photographs for exactly that reason.

**Say early that this is not a difference detector.** The preview path looks like one,
because it is one, and it is the first thing a viewer assumes. Two facts kill that
reading. `metadata.preview_mismatch` is applicable on **0 of the 224** benchmark
images, so none of the measured accuracy comes from it. And `courtyard cloned window`
flags at 0.795 with `preview_mismatch` reporting *not applicable*, because the file
carries no preview at all. What the system actually asks is whether one photograph is
internally consistent with having come from a single sensor in a single exposure.

A clone that was **not** caught is worth keeping in your pocket. Cloning a patch of the
brick paving over another part of the paving produces no detection at all: repeating
texture gives SIFT plenty of matches but no single consensus displacement. Cloning a
distinctive object is caught immediately; cloning flat or repeating texture is not.

### The stronger demo set, from the Korus dataset

Prepared in `data/interim/demo/` (gitignored, since Korus is research-use only and is
not redistributed). Regenerate with `scripts/salvage_zip.py` plus the copy commands in
the session notes if the folder is missing.

These are **TIFFs straight from the camera**, so the demosaicing structure survives
and the fingerprint detectors actually run. This is where the machine-learning work is
visible rather than the metadata trick.

| File | Verdict | Detectors that fired |
|---|---|---|
| `02-EDITED-nikon-d90.TIF` | **FLAG**, 0.810 | Copy-move found 61 keypoint pairs sharing one displacement of (172, -11) px. Camera fingerprint found 59 of 872 blocks inconsistent at 7.9 sigma. |
| `02-original-nikon-d90.TIF` | **AUTO CLEAR**, 0.253 | Same scene, untouched. |
| `01-EDITED-nikon-d7000.TIF` | **FLAG**, 0.810 | Copy-move, displacement (152, 8) px, plus 55 of 805 blocks at 8.2 sigma. |
| `01-original-nikon-d7000.TIF` | ROUTE TO HUMAN, 0.395 | Honest wobble: the fingerprint detector fires somewhat on the clean original too. |
| `03-EDITED-nikon-d90.TIF` | **FLAG**, 0.790 | 260 keypoint pairs at (356, -1) px, the clearest copy-move in the set. |
| `03-original-nikon-d90.TIF` | ROUTE TO HUMAN, 0.365 | As above. |

**Pair 02 is the demo.** Same scene, one edited, and the system separates them
cleanly: FLAG at 0.810 against AUTO CLEAR at 0.253. Pair 03 has the most legible
picture, a building facade with a window duplicated along the wall, which reads
instantly on screen.

Pairs 01 and 03 are worth showing precisely because their originals land in
ROUTE TO HUMAN rather than clearing. That is the system declining to decide, which is
the behaviour the three-outcome design exists for.

---

## A walkthrough that holds together

1. **Open the landing page.** Both worked examples are live output, not mockups. The
   page analyses the two sample files on load and shows whatever came back.
2. **Click `courtyard cloned window` first.** This is the one that sets the framing.
   One image in, no original, and the heatmap lights up *both* windows, which is what
   "a region of this image appears twice" looks like. Read the copy-move line aloud: a
   shared displacement across 212 keypoints is a concrete claim someone could check by
   hand, not a score.
3. **Then click `edited stale preview`.** Fast, unambiguous FLAG by a completely
   different route. Open *Before and after* to show the recovered original, and point
   at the `RECOVERED` badge: those are real pixels pulled out of the file, never a
   reconstruction, and the type system enforces the difference. Say that this path is
   luck rather than method.
4. **Drag in `02-EDITED-nikon-d90.TIF`.** The substantive one, and the only place the
   camera fingerprint appears, because this is a camera-original TIFF rather than a
   re-encoded JPEG. Walk the *What we checked* panel: every detector is listed with its
   own verdict and reason, including the ones that did not apply. Here copy-move and
   the fingerprint agree, from different physics.
5. **Drag in `02-original-nikon-d90.TIF`.** Same scene, clears at 0.253.
6. **Finish on `courtyard edited`.** It is edited, and it clears. Say why: no preview,
   re-encoded JPEG, nothing for the sensor tier to read.

Ending on the failure is the right call. The four quarantined detectors and the
chance-performance history are the strongest thing about this project, and a demo that
only shows wins throws that away.

---

## Talking about the score

Do not call the number a probability, because it is not one yet. `scripts/calibrate.py`
measures exactly this: the fusion is systematically under-confident, a Platt fit removes
most of that within a single camera out-of-fold, and it does **not** transfer across
sensors. The two Korus cameras want slopes of 2.05 and 1.02, differing by
+1.03 [+0.04, +2.08], p = 0.038.

So the honest line is: the score orders images well, an AUC of 0.792 says how well, and
turning it into a probability needs labelled images per camera model. `fuse()` defaults
to no calibration for that reason.

---

## Recording notes

- Run at 1280 wide or narrower. The results view is a two-column split above 860px and
  stacks below it, and the stacked layout reads better in a portrait clip.
- Analysis takes a few seconds per image. The progress bar names real pipeline stages,
  but the timing is an estimate, so do not cut it to imply per-stage timing.
- `data/interim/demo/*_result.png` holds pre-rendered three-panel overlays
  (original, detection, heatmap) for stills or thumbnails.
- The Korus images may not be redistributed, so keep the recording to your own screen
  rather than publishing the files.
