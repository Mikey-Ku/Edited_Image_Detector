"""The training machinery, especially the parts that fail silently.

A contrastive loss that returns NaN, or a split that leaks a device across the
wall, both produce a training run that looks fine and a model that is worthless.
Neither shows up as an error.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from groundtruth.learned import (
    EMBED_DIM,
    FingerprintNet,
    device_split,
    supervised_contrastive_loss,
)


def _norm(t):
    return torch.nn.functional.normalize(t, dim=1)


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------


def test_loss_is_finite():
    """Regression: masking the diagonal with -inf gave -inf * 0 = NaN."""
    e = _norm(torch.randn(16, EMBED_DIM))
    y = torch.arange(16) % 4
    loss = supervised_contrastive_loss(e, y)

    assert torch.isfinite(loss), "loss must never be NaN or inf"


def test_loss_orders_separation_correctly():
    y = torch.tensor([0, 0, 1, 1])
    perfect = _norm(torch.tensor([[1.0, 0], [1.0, 0], [0, 1.0], [0, 1.0]]))
    inverted = _norm(torch.tensor([[1.0, 0], [0, 1.0], [1.0, 0], [0, 1.0]]))

    assert supervised_contrastive_loss(perfect, y) < supervised_contrastive_loss(
        inverted, y
    )


def test_loss_handles_a_batch_with_no_positive_pair():
    """Every label unique -- there is nothing to pull together."""
    e = _norm(torch.randn(4, EMBED_DIM))
    loss = supervised_contrastive_loss(e, torch.arange(4))

    assert torch.isfinite(loss)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def test_output_is_per_pixel_not_pooled():
    """Localisation needs a fingerprint at every position, not one per image."""
    m = FingerprintNet()
    r = m(torch.randn(2, 3, 64, 64))

    assert r.shape == (2, EMBED_DIM, 64, 64)


def test_residual_is_unit_norm_per_pixel():
    m = FingerprintNet()
    r = m(torch.randn(2, 3, 32, 32))

    assert torch.allclose(r.norm(dim=1), torch.ones(2, 32, 32), atol=1e-3)


def test_first_layer_starts_as_a_high_pass():
    """A constant image must produce ~no residual before training.

    Checked on the interior only: zero-padding makes the border respond by
    construction, which is a property of the padding rather than of the filter.
    """
    m = FingerprintNet()
    flat = torch.full((1, 3, 32, 32), 0.3)

    interior = m.body[0](flat)[..., 1:-1, 1:-1]
    assert float(interior.abs().max()) < 1e-4


def test_accepts_any_input_size():
    """Full frames at inference, patches during training."""
    m = FingerprintNet()
    assert m(torch.randn(1, 3, 96, 128)).shape[-2:] == (96, 128)


# --------------------------------------------------------------------------
# The split -- the easiest thing to get quietly wrong
# --------------------------------------------------------------------------


def test_split_holds_out_whole_devices(tmp_path):
    for brand in ("Apple", "Samsung", "Huawei"):
        for i in range(4):
            (tmp_path / f"D{i}{brand[:2]}_{brand}_Model{i}").mkdir()

    split = device_split(tmp_path, val_fraction=0.25, seed=0)

    assert not set(split.train_devices) & set(split.val_devices)
    assert len(split.train_devices) + len(split.val_devices) == 12


def test_split_stratifies_by_brand(tmp_path):
    """Apple devices share a pipeline; all of them in train would leak."""
    for brand in ("Apple", "Samsung"):
        for i in range(4):
            (tmp_path / f"D{i}{brand[:2]}_{brand}_Model{i}").mkdir()

    split = device_split(tmp_path, val_fraction=0.25, seed=0)
    val_brands = {d.split("_")[1] for d in split.val_devices}

    assert val_brands == {"Apple", "Samsung"}


def test_split_is_deterministic(tmp_path):
    for i in range(8):
        (tmp_path / f"D{i}_Brand{i % 2}_Model{i}").mkdir()

    assert device_split(tmp_path, seed=7) == device_split(tmp_path, seed=7)
