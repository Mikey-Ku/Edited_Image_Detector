# Bundled samples

Images the review UI offers under the drop zone, and the two worked examples on the
landing page.

## These are demonstrations, not measurements

Every edited file here is a forgery **I staged myself**, and staged forgeries prove
nothing about accuracy: I chose them, so of course they are caught. Every performance
number in this project comes from the [Korus Realistic Tampering
Dataset](https://pkorus.pl/downloads/dataset-realistic-tampering), which is 112
forgeries made by other people who had never heard of this system.

The staged edits are reproducible. `scripts/make_samples.py` rebuilds each one from
its untouched original, so anyone can see exactly what was changed.

## Provenance

| file | source |
|---|---|
| `claim_car_original.jpg` | Collected from the web. **Provenance unverified.** |
| `claim_wall_original.jpg` | Collected from the web. **Provenance unverified.** |
| `real_courtyard_*.jpg` · `real_rooftops_*.jpg` | Derived from the Korus dataset, research use |

Three synthetic files were removed once the landing page stopped offering them:
`clean_uniform_noise.jpg`, `splice_noise_mismatch.jpg` and `edited_stale_preview.jpg`.
They demonstrated a detector rather than a claim, and the claim photographs cover the
same ground with something a person recognises. The generators still exist as
`noise_splice`, `pristine` and `stale_preview` in `tests/fixtures.py`, and the files
themselves are in git history if they are ever wanted back.

**The two claim photographs were gathered from an online search and their licence has
not been confirmed.** They are used here as demonstration material only. If you are
reusing this repository, or if you are the rights holder, replace them: drop
equivalents at the same two filenames and run `python scripts/make_samples.py`, and
nothing else needs to change.

Suitable replacements need a damaged surface filling most of the frame, irregular
texture rather than flat paint, and at least 1500 px on the long side. The reasoning
behind those constraints is measured in `scripts/sweep_copy_move.py`.
