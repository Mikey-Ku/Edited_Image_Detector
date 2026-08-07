"""Turn the fused score into something that can be read as a probability.

`fuse()` pools detector log-odds into a number between 0 and 1. That number orders
images correctly -- it is what the 0.792 AUC measures -- but it is not a
probability. Measured on the 224 Korus photographs, every reliability bin sits
*above* its predicted value: the pipeline says 0.34 for a group of images that are
61% forgeries. It is systematically under-confident, so a hand-set 0.70 threshold
is a rank cut-off wearing a probability's clothing.

This module fits the missing monotonic map. Three things follow from that word:

**Calibration cannot change AUC.** A monotonic map preserves every pairwise
ordering, so the discrimination numbers in the README are untouched by anything
here. Calibration changes what the number *means*, never how well it separates.
Anyone reporting a calibration that improved their AUC has a bug.

**Platt scaling is the default, but not because isotonic lost.** Measured
cross-camera, the two trade wins: isotonic takes the D90 -> D7000 direction on every
metric and blows up in the other, posting a log loss of 1.67 against Platt's 0.67
because a step function fitted on one sensor extrapolates badly onto another. Two
parameters are preferred for that asymmetry -- a bounded worst case -- rather than
for a better average. `scripts/calibrate.py` prints both.

**The Korus base rate is 50% by construction** -- 112 forgeries paired with their
own 112 originals. A probability is only a probability with respect to a prior, and
a claims pipeline does not see one forgery per honest photograph. `shift_prior()`
moves a calibrator fitted at 50% to a deployment prevalence. Skipping that step is
how a model calibrated in a lab reports 0.5 on a population where the honest answer
is 0.02.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

_EPS = 1e-6


def _logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


@dataclass(frozen=True)
class Calibrator:
    """A fitted map from fused score to calibrated probability.

    ``P_calibrated = sigmoid(slope * logit(p_fused) + intercept)``

    Because `fuse()` builds its output as ``sigmoid(pooled_log_odds)``, the logit
    here recovers the pooled log-odds exactly. So the fit is a linear correction in
    the space the fusion actually works in: `slope` says how much to trust the
    spread of the pooled evidence, `intercept` where to centre it.

    A `slope` above 1 means the fusion was under-confident and its log-odds are
    being stretched; below 1, over-confident and being shrunk.
    """

    slope: float
    intercept: float
    fitted_on: str = ""
    """Which subset this was fitted on -- kept so a calibrator cannot be silently
    applied to the data that produced it."""

    n_fit: int = 0
    base_rate: float = 0.5
    """Prevalence of the fitting set. Not decoration: `shift_prior()` needs it, and
    a calibrated probability is meaningless without knowing the prior it assumes."""

    def apply(self, p_fused: np.ndarray | float) -> np.ndarray | float:
        out = _sigmoid(self.slope * _logit(p_fused) + self.intercept)
        return float(out) if np.isscalar(p_fused) or np.ndim(p_fused) == 0 else out

    def shift_prior(self, deployment_rate: float) -> Calibrator:
        """Re-target this calibrator at a different prevalence.

        Under a shift in the prior alone -- the class-conditional evidence
        distributions unchanged -- the correction is entirely in the intercept:
        add the difference of the prior log-odds. This is the same identity behind
        prior correction in naive Bayes.

        The assumption is worth stating plainly, because it is doing real work: it
        holds when deployment forgeries *look like* Korus forgeries and only their
        frequency differs. If the deployment population is also harder or easier,
        this moves the numbers to the right neighbourhood but does not make them
        true, and only labelled deployment data can settle it.
        """
        if not 0.0 < deployment_rate < 1.0:
            raise ValueError(f"deployment_rate must be in (0,1), got {deployment_rate}")
        delta = float(
            np.log(deployment_rate / (1.0 - deployment_rate))
            - np.log(self.base_rate / (1.0 - self.base_rate))
        )
        return Calibrator(
            slope=self.slope,
            intercept=self.intercept + delta,
            fitted_on=f"{self.fitted_on} @ prior {deployment_rate:g}",
            n_fit=self.n_fit,
            base_rate=deployment_rate,
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> Calibrator:
        return cls(**json.loads(Path(path).read_text()))

    @classmethod
    def identity(cls) -> Calibrator:
        """The no-op calibrator: what the pipeline uses when none is fitted."""
        return cls(slope=1.0, intercept=0.0, fitted_on="identity")


def fit_platt(p_fused: np.ndarray, labels: np.ndarray, fitted_on: str = "") -> Calibrator:
    """Fit slope/intercept by minimising log loss.

    Log loss rather than Brier because it is the proper scoring rule whose minimiser
    under this two-parameter family is the maximum-likelihood fit, and because it
    punishes confident mistakes hard -- which is the failure mode that matters when
    a flagged image becomes an accusation.
    """
    p_fused = np.asarray(p_fused, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if p_fused.shape != labels.shape:
        raise ValueError(f"shape mismatch: {p_fused.shape} vs {labels.shape}")
    if len(p_fused) == 0:
        raise ValueError("cannot fit a calibrator on an empty set")

    z = _logit(p_fused)

    def nll(theta: np.ndarray) -> float:
        q = np.clip(_sigmoid(theta[0] * z + theta[1]), _EPS, 1.0 - _EPS)
        return float(-np.mean(labels * np.log(q) + (1.0 - labels) * np.log(1.0 - q)))

    best = minimize(nll, x0=np.array([1.0, 0.0]), method="L-BFGS-B")
    return Calibrator(
        slope=float(best.x[0]),
        intercept=float(best.x[1]),
        fitted_on=fitted_on,
        n_fit=len(p_fused),
        base_rate=float(labels.mean()),
    )


def expected_calibration_error(
    p: np.ndarray, labels: np.ndarray, bins: int = 10, equal_mass: bool = True
) -> tuple[float, list[dict]]:
    """ECE plus the reliability diagram behind it.

    The per-bin rows are returned rather than only the scalar because ECE averages
    away direction: a model over-confident at one end and under-confident at the
    other can post the same ECE as a well-behaved one. The diagram shows which.

    **Equal-mass bins by default, and that default matters here.** These scores
    cluster hard -- with equal-width bins, 119 of the 224 Korus images land in
    [0.2,0.3) while other bins hold one or two, so most of the reported ECE comes
    from bins whose sampling error is larger than the effect being measured. Equal
    mass spends the same number of images on every bin. Equal width remains
    available because it is what most papers report, and a number that cannot be
    compared to the literature is its own kind of useless.
    """
    p = np.asarray(p, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if len(p) == 0:
        return float("nan"), []

    if equal_mass:
        order = np.argsort(p)
        groups = [g for g in np.array_split(order, min(bins, len(p))) if len(g)]
    else:
        edges = np.linspace(0.0, 1.0, bins + 1)
        groups = []
        for i in range(bins):
            upper = p <= edges[i + 1] if i == bins - 1 else p < edges[i + 1]
            idx = np.flatnonzero((p >= edges[i]) & upper)
            if len(idx):
                groups.append(idx)

    total, rows = 0.0, []
    for g in groups:
        conf, acc = float(p[g].mean()), float(labels[g].mean())
        rows.append(
            {
                "lo": float(p[g].min()),
                "hi": float(p[g].max()),
                "n": len(g),
                "mean_predicted": conf,
                "actual_rate": acc,
                "gap": acc - conf,
                # Binomial standard error on the bin's true rate. Printed alongside
                # the gap so a "miscalibrated" bin of five images is visibly noise.
                "stderr": float(np.sqrt(max(acc * (1.0 - acc), 1e-12) / len(g))),
            }
        )
        total += len(g) / len(p) * abs(acc - conf)
    return float(total), rows


def brier(p: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(labels, dtype=float)) ** 2))


def log_loss(p: np.ndarray, labels: np.ndarray) -> float:
    q = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    y = np.asarray(labels, dtype=float)
    return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))


def fit_isotonic(p_fused: np.ndarray, labels: np.ndarray) -> IsotonicCalibrator:
    """Pool-adjacent-violators isotonic fit -- the flexible alternative to Platt.

    Isotonic fits any monotonic map rather than a two-parameter one. Whether that
    freedom pays or overfits at n~110 is a question for measurement, not for a
    docstring: `scripts/calibrate.py` scores both on held-out data and, as it turns
    out, they trade wins depending on the fold.
    """
    p_fused = np.asarray(p_fused, dtype=float)
    labels = np.asarray(labels, dtype=float)
    order = np.argsort(p_fused)
    x, y = p_fused[order], labels[order].astype(float)

    # Pool adjacent violators: repeatedly merge any neighbouring blocks whose means
    # go the wrong way, replacing both with their weighted mean.
    values = list(y)
    weights = [1.0] * len(y)
    i = 0
    while i < len(values) - 1:
        if values[i] <= values[i + 1] + 1e-12:
            i += 1
            continue
        w = weights[i] + weights[i + 1]
        merged = (values[i] * weights[i] + values[i + 1] * weights[i + 1]) / w
        values[i : i + 2] = [merged]
        weights[i : i + 2] = [w]
        i = max(i - 1, 0)

    knots_x, knots_y, cursor = [], [], 0
    for value, weight in zip(values, weights, strict=True):
        span = round(weight)
        knots_x.append(float(x[cursor]))
        knots_y.append(float(value))
        cursor += span
    return IsotonicCalibrator(np.array(knots_x), np.array(knots_y))


@dataclass(frozen=True)
class IsotonicCalibrator:
    """Step function from the PAV fit. Comparison baseline only -- see `fit_isotonic`."""

    knots_x: np.ndarray
    knots_y: np.ndarray

    def apply(self, p_fused: np.ndarray | float) -> np.ndarray | float:
        out = np.interp(
            np.asarray(p_fused, dtype=float), self.knots_x, self.knots_y,
            left=float(self.knots_y[0]), right=float(self.knots_y[-1]),
        )
        return float(out) if np.ndim(p_fused) == 0 else out
