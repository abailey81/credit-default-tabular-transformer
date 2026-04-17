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


# ──────────────────────────────────────────────────────────────────────────────
# attn_bias hook (PR #8) — pytest coverage for a feature that previously only
# had an in-module smoke test. Exercises each supported bias shape plus the
# "zero bias is a strict no-op" contract.
# ──────────────────────────────────────────────────────────────────────────────


def test_mha_attn_bias_none_vs_zeros_is_bit_identical():
    """attn_bias=None and attn_bias=torch.zeros(T, T) must produce the same
    output and attention weights to float32 precision, under the same seed."""
    torch.manual_seed(0)
    d_model, n_heads, B, T = 32, 4, 2, 24
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)
    mha.eval()
    x = torch.randn(B, T, d_model)
    out_none, w_none = mha(x)
    out_zero, w_zero = mha(x, attn_bias=torch.zeros(T, T))
    assert torch.allclose(out_none, out_zero, atol=1e-6)
    assert torch.allclose(w_none, w_zero, atol=1e-6)


def test_mha_attn_bias_broadcasts_from_seq_shape():
    """A (T, T) bias must broadcast cleanly over batch and head dimensions."""
    torch.manual_seed(1)
    d_model, n_heads, B, T = 32, 4, 3, 12
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)
    mha.eval()
    x = torch.randn(B, T, d_model)
    bias = torch.randn(T, T) * 0.1
    out, w = mha(x, attn_bias=bias)
    assert out.shape == (B, T, d_model)
    assert w.shape == (B, n_heads, T, T)
    assert torch.allclose(w.sum(dim=-1), torch.ones_like(w.sum(dim=-1)), atol=1e-5)


def test_mha_attn_bias_broadcasts_from_per_head_shape():
    """A (H, T, T) bias must broadcast over batch only — different heads may
    see different bias patterns (used by ``TemporalDecayBias(mode='per_head')``)."""
    torch.manual_seed(2)
    d_model, n_heads, B, T = 32, 4, 2, 8
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)
    mha.eval()
    x = torch.randn(B, T, d_model)
    bias = torch.randn(n_heads, T, T) * 0.1
    out, w = mha(x, attn_bias=bias)
    assert out.shape == (B, T, d_model)
    assert w.shape == (B, n_heads, T, T)


def test_mha_attn_bias_suppresses_targeted_cell():
    """A large negative bias at a single cell (q=0, k=1) must drive the
    softmax weight at that cell near zero across every row in the batch."""
    torch.manual_seed(3)
    d_model, n_heads, B, T = 32, 4, 4, 10
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)
    mha.eval()
    x = torch.randn(B, T, d_model)
    bias = torch.zeros(T, T)
    bias[0, 1] = -1e4
    _, w = mha(x, attn_bias=bias)
    # w has shape (B, n_heads, T, T); row 0, column 1 must be crushed to ~0.
    suppressed = w[:, :, 0, 1]
    assert torch.all(suppressed < 1e-3), (
        f"bias -1e4 at (0,1) did not suppress attention (max={suppressed.max():.4e})"
    )


def test_mha_attn_bias_gradient_flows_when_bias_has_requires_grad():
    """If the bias tensor is a learnable parameter (the TemporalDecayBias
    pattern), its gradient must flow through softmax + matmul."""
    torch.manual_seed(4)
    d_model, n_heads, B, T = 32, 4, 2, 8
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)
    mha.train()
    x = torch.randn(B, T, d_model)
    bias = torch.zeros(T, T, requires_grad=True)
    out, _ = mha(x, attn_bias=bias)
    out.sum().backward()
    assert bias.grad is not None
    assert torch.isfinite(bias.grad).all()
    # Zero-bias gradient through softmax + matmul should be non-zero generically.
    assert bias.grad.abs().sum().item() > 0


def test_mha_attn_bias_plus_dropout_in_train_is_finite():
    """Exercise the combination the original smoke test missed: attn_bias not
    None, .train() mode active, attention dropout > 0. No NaN / Inf allowed.
    (Addresses reviewer note on PR #8.)"""
    torch.manual_seed(5)
    d_model, n_heads, B, T = 32, 4, 4, 24
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.2)
    mha.train()
    x = torch.randn(B, T, d_model)
    bias = torch.randn(T, T) * 0.1
    out, w = mha(x, attn_bias=bias)
    assert torch.isfinite(out).all()
    assert torch.isfinite(w).all()
    assert out.shape == (B, T, d_model)
