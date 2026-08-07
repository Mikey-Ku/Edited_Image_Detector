"""Calibration behaviour.

The property that matters most here is the one that is easiest to get silently
wrong: a calibrator must not change how images are *ordered*. If it does, the AUC
in the README stops describing the shipped system and every ablation result is
measuring something other than what it claims to.
"""

from __future__ import annotations

import numpy as np
import pytest

from groundtruth.core.types import Evidence, Tier
from groundtruth.fusion.calibration import (
    Calibrator,
    brier,
    expected_calibration_error,
    fit_isotonic,
    fit_platt,
    log_loss,
)
from groundtruth.fusion.weighted import fuse


def _scores(rng, n=400):
    """A miscalibrated but well-ordered score set: under-confident, like the real one."""
    y = rng.integers(0, 2, n).astype(float)
    z = rng.normal(loc=np.where(y == 1, 0.9, -0.9), scale=1.0)
    return 1.0 / (1.0 + np.exp(-0.4 * z)), y  # 0.4 squashes it toward 0.5


# --------------------------------------------------------------------------
# The ordering invariant
# --------------------------------------------------------------------------


def test_calibration_never_changes_the_ranking():
    rng = np.random.default_rng(0)
    p, y = _scores(rng)
    cal = fit_platt(p, y)
    out = np.asarray(cal.apply(p))
    assert np.array_equal(np.argsort(np.argsort(p)), np.argsort(np.argsort(out)))


def test_fitted_slope_is_positive_so_the_map_is_increasing():
    """A negative slope would fit the data by *inverting* it -- better log loss, and
    an utterly broken detector. Nothing in Platt scaling forbids it, so it is worth
    an explicit check rather than an assumption."""
    rng = np.random.default_rng(1)
    p, y = _scores(rng)
    assert fit_platt(p, y).slope > 0


def test_identity_calibrator_is_a_no_op():
    ident = Calibrator.identity()
    for v in (0.01, 0.25, 0.5, 0.75, 0.99):
        assert ident.apply(v) == pytest.approx(v, abs=1e-5)


# --------------------------------------------------------------------------
# Does it actually calibrate?
# --------------------------------------------------------------------------


def test_fitting_reduces_calibration_error_in_sample():
    rng = np.random.default_rng(2)
    p, y = _scores(rng)
    cal = fit_platt(p, y)
    before, _ = expected_calibration_error(p, y)
    after, _ = expected_calibration_error(np.asarray(cal.apply(p)), y)
    assert after < before
    assert log_loss(np.asarray(cal.apply(p)), y) < log_loss(p, y)
    assert brier(np.asarray(cal.apply(p)), y) < brier(p, y)


def test_under_confident_input_produces_a_slope_above_one():
    """The construction in `_scores` squashes log-odds by 0.4, so a correct fit has
    to stretch them back out. This pins the sign of the diagnostic the report leans
    on when it says the fusion was under-confident."""
    rng = np.random.default_rng(3)
    p, y = _scores(rng)
    assert fit_platt(p, y).slope > 1.0


def test_already_calibrated_input_is_left_roughly_alone():
    rng = np.random.default_rng(4)
    n = 4000
    p = rng.uniform(0.02, 0.98, n)
    y = (rng.uniform(size=n) < p).astype(float)  # true rate == predicted, by construction
    cal = fit_platt(p, y)
    assert cal.slope == pytest.approx(1.0, abs=0.15)
    assert cal.intercept == pytest.approx(0.0, abs=0.15)


# --------------------------------------------------------------------------
# Prior shift
# --------------------------------------------------------------------------


def test_prior_shift_moves_probabilities_down_but_not_the_order():
    cal = Calibrator(slope=1.0, intercept=0.0, base_rate=0.5)
    rare = cal.shift_prior(0.02)
    p = np.array([0.2, 0.5, 0.8, 0.95])
    shifted = np.asarray(rare.apply(p))
    assert np.all(shifted < np.asarray(cal.apply(p)))
    assert np.all(np.diff(shifted) > 0)
    assert rare.base_rate == 0.02


def test_prior_shift_of_the_same_rate_is_a_no_op():
    cal = Calibrator(slope=1.3, intercept=0.4, base_rate=0.5)
    assert cal.shift_prior(0.5).intercept == pytest.approx(cal.intercept)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_prior_shift_rejects_impossible_rates(bad):
    with pytest.raises(ValueError):
        Calibrator.identity().shift_prior(bad)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_equal_mass_bins_are_actually_equal_mass():
    rng = np.random.default_rng(5)
    p = np.concatenate([rng.uniform(0.2, 0.25, 90), rng.uniform(0.7, 0.9, 10)])
    y = rng.integers(0, 2, 100).astype(float)
    _, rows = expected_calibration_error(p, y, bins=10, equal_mass=True)
    assert {r["n"] for r in rows} == {10}

    # Equal width on the same clustered scores is exactly the failure mode that
    # motivated the default: nearly everything lands in one bin.
    _, wide = expected_calibration_error(p, y, bins=10, equal_mass=False)
    assert max(r["n"] for r in wide) >= 50


def test_perfect_predictions_score_zero_everywhere():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    p = np.array([1e-6, 1e-6, 1 - 1e-6, 1 - 1e-6])
    assert expected_calibration_error(p, y)[0] == pytest.approx(0.0, abs=1e-5)
    assert brier(p, y) == pytest.approx(0.0, abs=1e-9)


def test_isotonic_is_monotonic_non_decreasing():
    rng = np.random.default_rng(6)
    p, y = _scores(rng, n=200)
    iso = fit_isotonic(p, y)
    grid = np.linspace(p.min(), p.max(), 100)
    assert np.all(np.diff(np.asarray(iso.apply(grid))) >= -1e-9)


# --------------------------------------------------------------------------
# Wiring into the pipeline
# --------------------------------------------------------------------------


def _ev(score, conf=0.8):
    return [
        Evidence(detector_id="a", tier=Tier.SENSOR, applicable=True,
                 score=score, confidence=conf, effect_size=0.6)
    ]


def test_fuse_without_a_calibrator_is_unchanged():
    """The published numbers were produced with no calibrator. If this default ever
    flips, every result in the README silently stops describing the shipped code."""
    assert fuse(_ev(0.8)).manipulated_probability == pytest.approx(0.8, abs=1e-3)


def test_fuse_applies_a_supplied_calibrator():
    sharpen = Calibrator(slope=2.0, intercept=0.0)
    plain = fuse(_ev(0.8)).manipulated_probability
    calibrated = fuse(_ev(0.8), calibrator=sharpen).manipulated_probability
    assert calibrated > plain


def test_calibrator_round_trips_through_disk(tmp_path):
    cal = fit_platt(*_scores(np.random.default_rng(7)), fitted_on="test")
    path = tmp_path / "cal.json"
    cal.save(path)
    back = Calibrator.load(path)
    assert back.slope == pytest.approx(cal.slope)
    assert back.intercept == pytest.approx(cal.intercept)
    assert back.fitted_on == "test"


def test_fitting_on_nothing_is_an_error_rather_than_a_silent_identity():
    with pytest.raises(ValueError):
        fit_platt(np.array([]), np.array([]))
