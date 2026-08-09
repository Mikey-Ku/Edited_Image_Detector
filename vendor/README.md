# vendor

Third-party material kept for provenance, not for execution.

## `noiseprint_ref.py`

The original authors' network definition, from the PyTorch port of the official
TensorFlow release ([RonyAbecidan/noiseprint-pytorch](https://github.com/RonyAbecidan/noiseprint-pytorch)).

**It does not run and is not meant to.** It imports `utilityRead`, which is not part
of this repository, and nothing here imports it. Ruff skips the whole directory
(`extend-exclude = ["vendor"]` in `pyproject.toml`), so it is not held to this
project's style either.

It is committed because `src/groundtruth/learned/noiseprint.py` claims to be a
*faithful* port, and that claim is worth nothing if the thing it was ported from is
not in the repository to check it against. The layer order, the inference-only
BatchNorm with statistics stored as parameters, and the bias-only `AddBias` layer are
all reproduced from this file so the published weights load unchanged.

## `NOISEPRINT_LICENSE.txt`

Noiseprint is the one dependency in this project that is **not permissively
licensed**: nonprofit use only, commercial use expressly prohibited. Copyright (c)
2019 GRIP-UNINA. The full terms are in that file, and the constraint is restated at
the top of `learned/noiseprint.py` where someone is likely to read it.

Cozzolino & Verdoliva, "Noiseprint: A CNN-Based Camera Model Fingerprint", IEEE TIFS
2020. <https://arxiv.org/abs/1808.08396>
