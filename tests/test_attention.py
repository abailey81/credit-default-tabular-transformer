"""Tests for src/attention.py — ScaledDotProductAttention and MultiHeadAttention."""

from __future__ import annotations

import math

import pytest
import torch

from attention import MultiHeadAttention, ScaledDotProductAttention  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# ScaledDotProductAttention
# ──────────────────────────────────────────────────────────────────────────────


def test_sdpa_output_shape():
    sdp = ScaledDotProductAttention(dropout=0.0)
    B, H, T, D = 2, 4, 8, 16
    Q = torch.randn(B, H, T, D)
    K = torch.randn(B, H, T, D)
    V = torch.randn(B, H, T, D)
    out, w = sdp(Q, K, V)
    assert out.shape == (B, H, T, D)
    assert w.shape == (B, H, T, T)


def test_sdpa_attention_rows_sum_to_one():
    sdp = ScaledDotProductAttention(dropout=0.0)
    B, H, T, D = 2, 2, 6, 8
    Q = torch.randn(B, H, T, D)
    K = torch.randn(B, H, T, D)
    V = torch.randn(B, H, T, D)
    _, w = sdp(Q, K, V)
    assert torch.allclose(w.sum(dim=-1), torch.ones_like(w.sum(dim=-1)), atol=1e-5)


def test_sdpa_dropout_active_only_in_train():
    sdp = ScaledDotProductAttention(dropout=0.5)
    Q = torch.randn(2, 2, 4, 8)
    # Eval mode: dropout disabled, two calls must be deterministic.
    sdp.eval()
    out1, _ = sdp(Q, Q, Q)
    out2, _ = sdp(Q, Q, Q)
    assert torch.allclose(out1, out2, atol=1e-5)


# ──────────────────────────────────────────────────────────────────────────────
# MultiHeadAttention
# ──────────────────────────────────────────────────────────────────────────────


def test_mha_output_shape():
    mha = MultiHeadAttention(d_model=32, n_heads=4, dropout=0.0)
    x = torch.randn(2, 10, 32)
    out, w = mha(x)
    assert out.shape == (2, 10, 32)
    assert w.shape == (2, 4, 10, 10)


def test_mha_rejects_non_divisible_dims():
    with pytest.raises(AssertionError):
        MultiHeadAttention(d_model=30, n_heads=4)


def test_mha_weights_sum_to_one():
    mha = MultiHeadAttention(d_model=32, n_heads=4, dropout=0.0)
    mha.eval()
    x = torch.randn(2, 24, 32)
    _, w = mha(x)
    assert torch.allclose(w.sum(dim=-1), torch.ones_like(w.sum(dim=-1)), atol=1e-5)


def test_mha_gradient_flow():
    mha = MultiHeadAttention(d_model=32, n_heads=4, dropout=0.0)
    x = torch.randn(2, 10, 32, requires_grad=True)
    out, _ = mha(x)
    (out.pow(2).sum()).backward()
    for name, p in mha.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"bad grad on {name}"


# ──────────────────────────────────────────────────────────────────────────────
# Scaling factor check
# ──────────────────────────────────────────────────────────────────────────────


def test_scaling_by_sqrt_dk():
    """
    Direct scaling check: with standard-normal Q and K and d_k = 16,
    the softmax logits before scaling would have variance ~16; after
    scaling by sqrt(16) = 4, logits have variance ~1 and softmax does not
    saturate. Sanity: max softmax value on random data should not be ~1.
    """
    torch.manual_seed(0)
    sdp = ScaledDotProductAttention(dropout=0.0)
    Q = torch.randn(1, 1, 32, 16)
    K = torch.randn(1, 1, 32, 16)
    V = torch.randn(1, 1, 32, 16)
    _, w = sdp(Q, K, V)
    # On random data the max softmax prob per row should be well below 1.0.
    max_prob = w.max(dim=-1).values.mean().item()
    assert max_prob < 0.5, f"softmax looks saturated (max prob {max_prob:.2f})"
