"""Tests for src/losses.py — WeightedBCE, FocalLoss, LabelSmoothingBCE."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from losses import (  # noqa: E402
    FocalLoss,
    LabelSmoothingBCELoss,
    WeightedBCELoss,
    balanced_alpha,
    compute_pos_weight,
)


# ──────────────────────────────────────────────────────────────────────────────
# class-weight helpers
# ──────────────────────────────────────────────────────────────────────────────


def test_compute_pos_weight_on_imbalanced():
    y = torch.tensor([0, 0, 0, 1])
    pos_weight = compute_pos_weight(y)
    assert pytest.approx(pos_weight.item(), rel=1e-6) == 3.0


def test_compute_pos_weight_all_negatives_raises():
    y = torch.zeros(10)
    with pytest.raises(ValueError):
        compute_pos_weight(y)


def test_balanced_alpha_sums_to_one():
    y = torch.cat([torch.zeros(78), torch.ones(22)])
    ap, an = balanced_alpha(y)
    assert pytest.approx(ap + an, rel=1e-9) == 1.0
    assert ap > an  # minority class gets higher weight


def test_balanced_alpha_single_class_raises():
    with pytest.raises(ValueError):
        balanced_alpha(torch.zeros(10))


# ──────────────────────────────────────────────────────────────────────────────
# WeightedBCELoss
# ──────────────────────────────────────────────────────────────────────────────


def test_wbce_matches_bce_when_pos_weight_is_one():
    logits = torch.randn(64)
    y = torch.randint(0, 2, (64,)).float()
    wbce = WeightedBCELoss(pos_weight=1.0)
    bce = F.binary_cross_entropy_with_logits(logits, y)
    assert torch.allclose(wbce(logits, y), bce, atol=1e-6)


def test_wbce_reduction_none_shape():
    logits = torch.randn(32)
    y = torch.randint(0, 2, (32,)).float()
    wbce = WeightedBCELoss(pos_weight=2.0, reduction="none")
    assert wbce(logits, y).shape == (32,)


def test_wbce_gradient_flow():
    logits = torch.randn(32, requires_grad=True)
    y = torch.randint(0, 2, (32,)).float()
    WeightedBCELoss(pos_weight=3.0)(logits, y).backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


# ──────────────────────────────────────────────────────────────────────────────
# FocalLoss
# ──────────────────────────────────────────────────────────────────────────────


def test_focal_gamma_zero_equals_bce():
    logits = torch.randn(128)
    y = torch.randint(0, 2, (128,)).float()
    fl = FocalLoss(gamma=0.0, alpha=None)
    bce = F.binary_cross_entropy_with_logits(logits, y)
    assert torch.allclose(fl(logits, y), bce, atol=1e-6)


def test_focal_strictly_decreases_with_gamma():
    """On the same data, larger γ ⇒ smaller loss (by construction)."""
    logits = torch.randn(1024)
    y = torch.randint(0, 2, (1024,)).float()
    losses = [FocalLoss(gamma=g, alpha=None)(logits, y).item() for g in (0.0, 1.0, 2.0, 3.0)]
    assert losses[0] > losses[1] > losses[2] > losses[3]


def test_focal_balanced_alpha_fits_once():
    y = torch.cat([torch.zeros(80), torch.ones(20)])
    logits = torch.randn(100)
    fl = FocalLoss(gamma=2.0, alpha="balanced")
    fl(logits, y)
    fl(logits, y)  # second call should reuse the fit
    ap, an = fl._balanced_fitted
    assert 0.75 < ap < 0.85  # minority is 20% so alpha_pos ≈ 0.80


def test_focal_rejects_bad_tuple():
    # alpha is resolved lazily on first forward — must actually call it.
    fl = FocalLoss(gamma=2.0, alpha=(0.5,))  # length 1 tuple
    logits = torch.randn(8)
    y = torch.randint(0, 2, (8,)).float()
    with pytest.raises(ValueError, match="alpha tuple must have length 2"):
        fl(logits, y)


def test_focal_rejects_negative_gamma():
    with pytest.raises(ValueError):
        FocalLoss(gamma=-0.1)


def test_focal_gradient_flow():
    logits = torch.randn(16, requires_grad=True)
    y = torch.randint(0, 2, (16,)).float()
    FocalLoss(gamma=2.0, alpha=0.75)(logits, y).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


# ──────────────────────────────────────────────────────────────────────────────
# LabelSmoothingBCELoss
# ──────────────────────────────────────────────────────────────────────────────


def test_label_smoothing_eps_zero_equals_wbce():
    logits = torch.randn(64)
    y = torch.randint(0, 2, (64,)).float()
    lsm = LabelSmoothingBCELoss(epsilon=0.0, pos_weight=3.0)
    wbce = WeightedBCELoss(pos_weight=3.0)
    assert torch.allclose(lsm(logits, y), wbce(logits, y), atol=1e-6)


def test_label_smoothing_increases_loss():
    logits = torch.randn(1024) * 2  # peaky
    y = torch.randint(0, 2, (1024,)).float()
    lsm_a = LabelSmoothingBCELoss(epsilon=0.0)
    lsm_b = LabelSmoothingBCELoss(epsilon=0.10)
    # Smoothing increases entropy of the target, so a peaky model pays a higher loss.
    assert lsm_b(logits, y).item() > lsm_a(logits, y).item()


def test_label_smoothing_rejects_bad_epsilon():
    with pytest.raises(ValueError):
        LabelSmoothingBCELoss(epsilon=1.5)


def test_label_smoothing_gradient_flow():
    logits = torch.randn(16, requires_grad=True)
    y = torch.randint(0, 2, (16,)).float()
    LabelSmoothingBCELoss(epsilon=0.05)(logits, y).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
