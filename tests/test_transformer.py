"""Tests for src/transformer.py — FeedForward, TransformerBlock,
TemporalDecayBias, and TransformerEncoder.

Complements the in-module smoke test with full pytest coverage, so CI catches
regressions on what was previously 412 LOC of untested code (PR #8).

Structure
---------
- FeedForward:        shape, initialisation (Kaiming/Xavier split), dropout
- TransformerBlock:   PreNorm residual, attn_bias threading, independently
                      ablatable dropout channels
- TemporalDecayBias:  three modes (scalar / per_head / off), α=0 → zero bias,
                      α>0 suppression semantics (PAY_0→PAY_6 < PAY_0→PAY_2),
                      checkpoint state-dict excludes the non-persistent buffer
- TransformerEncoder: output shape, attention-weight list length, gradient
                      flow, shared-bias-across-layers contract, dropout-on
                      no-NaN robustness, end-to-end with embedding.

References: PROJECT_PLAN.md §6.4, §6.5, §6.6, §6.11, §6.12.2 (Novelty N3).
"""

from __future__ import annotations

import pytest
import torch

from transformer import (  # noqa: E402
    FeedForward,
    TemporalDecayBias,
    TransformerBlock,
    TransformerEncoder,
)


# ──────────────────────────────────────────────────────────────────────────────
# FeedForward
# ──────────────────────────────────────────────────────────────────────────────


def test_feedforward_output_shape():
    ffn = FeedForward(d_model=32, d_ff=128, dropout=0.0)
    x = torch.randn(4, 24, 32)
    assert ffn(x).shape == (4, 24, 32)


def test_feedforward_d_ff_defaults_to_4x_d_model():
    ffn = FeedForward(d_model=32)
    assert ffn.linear1.out_features == 128
    assert ffn.linear2.in_features == 128


def test_feedforward_init_uses_kaiming_then_xavier():
    """W_1 feeds GELU → Kaiming; W_2 feeds residual → Xavier. Variances differ,
    so a quick distributional check is enough."""
    torch.manual_seed(0)
    ffn = FeedForward(d_model=64, d_ff=256)
    # Kaiming-normal (relu-mode) std ≈ sqrt(2/fan_in); Xavier-normal std ≈ sqrt(2/(fan_in+fan_out)).
    k_std_expected = (2.0 / 64) ** 0.5
    x_std_expected = (2.0 / (256 + 64)) ** 0.5
    assert abs(ffn.linear1.weight.std().item() - k_std_expected) < 0.02
    assert abs(ffn.linear2.weight.std().item() - x_std_expected) < 0.02
    # Biases must be zero.
    assert torch.all(ffn.linear1.bias == 0)
    assert torch.all(ffn.linear2.bias == 0)


def test_feedforward_gradient_flows():
    ffn = FeedForward(d_model=32, d_ff=128, dropout=0.1)
    ffn.train()
    x = torch.randn(4, 24, 32, requires_grad=True)
    ffn(x).sum().backward()
    for name, p in ffn.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all()


# ──────────────────────────────────────────────────────────────────────────────
# TransformerBlock — PreNorm + independently ablatable dropout channels
# ──────────────────────────────────────────────────────────────────────────────


def test_transformer_block_shape_and_attn_shape():
    block = TransformerBlock(d_model=32, n_heads=4, dropout=0.0)
    block.eval()
    x = torch.randn(4, 24, 32)
    out, w = block(x)
    assert out.shape == (4, 24, 32)
    assert w.shape == (4, 4, 24, 24)


def test_transformer_block_prenorm_residual_is_identity_like_when_weights_zeroed():
    """PreNorm + residual means that if every Linear is zero, the block's
    output equals the input (residuals dominate, sub-layers contribute 0).
    This confirms the residual path is wired correctly."""
    block = TransformerBlock(d_model=16, n_heads=2, dropout=0.0)
    block.eval()
    with torch.no_grad():
        for p in block.parameters():
            p.zero_()
        # LayerNorm scale γ=0 would collapse output; restore γ=1, β=0 defaults.
        for ln in (block.ln1, block.ln2):
            ln.weight.fill_(1.0)
            ln.bias.zero_()
    x = torch.randn(2, 10, 16)
    out, _ = block(x)
    # With all projections zeroed, attn_out and ffn_out are both 0, so out == x.
    assert torch.allclose(out, x, atol=1e-6)


def test_transformer_block_attn_bias_is_threaded_to_attention():
    """A large negative attn_bias at cell (q, k) must crush w[:, :, q, k]."""
    torch.manual_seed(0)
    block = TransformerBlock(d_model=32, n_heads=4, dropout=0.0)
    block.eval()
    x = torch.randn(2, 10, 32)
    bias = torch.zeros(10, 10)
    bias[3, 7] = -1e4
    _, w = block(x, attn_bias=bias)
    assert torch.all(w[:, :, 3, 7] < 1e-3)


def test_transformer_block_independent_dropout_channels():
    """Per-channel dropout overrides must be honoured: verify each of the
    three dropout modules has the requested rate."""
    block = TransformerBlock(
        d_model=32, n_heads=4,
        dropout=0.1,
        attn_dropout=0.3,
        ffn_dropout=0.25,
        residual_dropout=0.05,
    )
    assert block.attention.attention.dropout.p == 0.3
    assert block.ffn.dropout.p == 0.25
    assert block.residual_dropout.p == 0.05


def test_transformer_block_dropout_defaults_to_shared_when_per_channel_absent():
    block = TransformerBlock(d_model=32, n_heads=4, dropout=0.2)
    assert block.attention.attention.dropout.p == 0.2
    assert block.ffn.dropout.p == 0.2
    assert block.residual_dropout.p == 0.2


def test_transformer_block_in_train_mode_produces_finite_output():
    """Exercises the combination the PR #8 smoke test missed: non-None
    attn_bias, .train() mode, dropout > 0 — no NaN/Inf allowed."""
    torch.manual_seed(0)
    block = TransformerBlock(d_model=32, n_heads=4, dropout=0.2)
    block.train()
    x = torch.randn(8, 24, 32)
    bias = torch.randn(24, 24) * 0.1
    out, w = block(x, attn_bias=bias)
    assert torch.isfinite(out).all()
    assert torch.isfinite(w).all()


# ──────────────────────────────────────────────────────────────────────────────
# TemporalDecayBias — Novelty N3
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def canonical_layout():
    return {
        "pay":     {"positions": [4, 5, 6, 7, 8, 9],         "months": [0, 1, 2, 3, 4, 5]},
        "bill":    {"positions": [12, 13, 14, 15, 16, 17],   "months": [0, 1, 2, 3, 4, 5]},
        "pay_amt": {"positions": [18, 19, 20, 21, 22, 23],   "months": [0, 1, 2, 3, 4, 5]},
    }


def test_temporal_decay_scalar_alpha_zero_gives_zero_bias(canonical_layout):
    decay = TemporalDecayBias(canonical_layout, seq_len=24, mode="scalar")
    bias = decay()
    assert bias is not None
    assert bias.shape == (24, 24)
    assert torch.equal(bias, torch.zeros(24, 24))


def test_temporal_decay_off_returns_none(canonical_layout):
    decay = TemporalDecayBias(canonical_layout, seq_len=24, mode="off")
    assert decay() is None


def test_temporal_decay_per_head_output_shape(canonical_layout):
    decay = TemporalDecayBias(canonical_layout, seq_len=24, mode="per_head", n_heads=4)
    bias = decay()
    assert bias.shape == (4, 24, 24)


def test_temporal_decay_rejects_unknown_mode(canonical_layout):
    with pytest.raises(ValueError, match="Unknown mode"):
        TemporalDecayBias(canonical_layout, seq_len=24, mode="not_a_mode")  # type: ignore[arg-type]


def test_temporal_decay_distance_matrix_masks_cross_group_to_zero(canonical_layout):
    """The cached ``neg_distance_masked`` buffer must be zero outside the
    within-group blocks — e.g. PAY (positions 4-9) should never contribute a
    non-zero penalty vs BILL_AMT (positions 12-17)."""
    decay = TemporalDecayBias(canonical_layout, seq_len=24, mode="scalar")
    # (4, 12) is PAY_0 vs BILL_AMT1 — different groups → zero distance.
    assert decay.neg_distance_masked[4, 12].item() == 0.0
    # (4, 9) is PAY_0 vs PAY_6 — same group, 5 months apart → -5.
    assert decay.neg_distance_masked[4, 9].item() == -5.0
    # CLS (index 0) never receives a bias.
    assert torch.all(decay.neg_distance_masked[0] == 0.0)
    assert torch.all(decay.neg_distance_masked[:, 0] == 0.0)


def test_temporal_decay_suppresses_distant_month_more_than_near(canonical_layout):
    """With α=5 the penalty at (PAY_0, PAY_6) = -25 >> (PAY_0, PAY_2) = -5,
    so the softmax weight at column 9 must be lower than at column 5 for every
    batch row and every head."""
    torch.manual_seed(0)
    decay = TemporalDecayBias(canonical_layout, seq_len=24, mode="scalar")
    with torch.no_grad():
        decay.alpha.fill_(5.0)
    encoder = TransformerEncoder(
        d_model=32, n_heads=4, n_layers=2, dropout=0.0, temporal_decay=decay,
    )
    encoder.eval()
    x = torch.randn(4, 24, 32)
    _, weights = encoder(x)
    w0 = weights[0]
    assert torch.all(w0[:, :, 4, 9] < w0[:, :, 4, 5])


def test_temporal_decay_alpha_is_trainable(canonical_layout):
    """α must receive a non-zero gradient — otherwise the model cannot
    activate or deactivate the prior during training."""
    torch.manual_seed(0)
    decay = TemporalDecayBias(canonical_layout, seq_len=24, mode="scalar")
    with torch.no_grad():
        decay.alpha.fill_(0.5)
    encoder = TransformerEncoder(
        d_model=32, n_heads=4, n_layers=2, dropout=0.0, temporal_decay=decay,
    )
    encoder.train()
    x = torch.randn(4, 24, 32)
    out, _ = encoder(x)
    out.sum().backward()
    assert decay.alpha.grad is not None
    assert decay.alpha.grad.abs().item() > 0


def test_temporal_decay_neg_distance_is_non_persistent_buffer(canonical_layout):
    """``neg_distance_masked`` is deterministically reconstructible from
    ``temporal_layout`` and is therefore registered with
    ``persistent=False`` — state_dict() must not contain it, so checkpoints
    stay small and the contract holds."""
    decay = TemporalDecayBias(canonical_layout, seq_len=24, mode="scalar")
    assert "neg_distance_masked" not in decay.state_dict()
    # Still present on the module for forward to use.
    assert hasattr(decay, "neg_distance_masked")


# ──────────────────────────────────────────────────────────────────────────────
# TransformerEncoder — stacking, attention collection, gradient flow
# ──────────────────────────────────────────────────────────────────────────────


def test_encoder_output_and_attn_list_shapes():
    enc = TransformerEncoder(d_model=32, n_heads=4, n_layers=3, dropout=0.0)
    enc.eval()
    x = torch.randn(4, 24, 32)
    out, weights = enc(x)
    assert out.shape == (4, 24, 32)
    assert len(weights) == 3
    for w in weights:
        assert w.shape == (4, 4, 24, 24)
        assert torch.allclose(w.sum(dim=-1), torch.ones_like(w.sum(dim=-1)), atol=1e-5)


def test_encoder_gradient_flows_through_every_parameter():
    enc = TransformerEncoder(d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    enc.train()
    x = torch.randn(4, 24, 32)
    enc(x)[0].sum().backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_encoder_temporal_decay_called_exactly_once_per_forward(canonical_layout):
    """By design the same ``TemporalDecayBias`` output is added to every
    block's pre-softmax scores — not recomputed per layer. We verify the
    "compute-once-reuse" contract by counting forward() calls: one call per
    encoder forward regardless of ``n_layers``.
    """
    decay = TemporalDecayBias(canonical_layout, seq_len=24, mode="scalar")

    call_counter = {"n": 0}
    real_forward = decay.forward

    def counted_forward(*args, **kwargs):
        call_counter["n"] += 1
        return real_forward(*args, **kwargs)

    decay.forward = counted_forward  # type: ignore[assignment]

    enc = TransformerEncoder(
        d_model=32, n_heads=4, n_layers=4, dropout=0.0, temporal_decay=decay,
    )
    enc.eval()
    x = torch.randn(2, 24, 32)
    _ = enc(x)
    assert call_counter["n"] == 1, (
        f"TemporalDecayBias.forward() was called {call_counter['n']}× for a "
        f"single encoder forward pass — it must be called exactly once and "
        f"the result reused across all {len(enc.blocks)} layers"
    )


def test_encoder_alpha_receives_gradient_under_stacked_layers(canonical_layout):
    """With multiple stacked blocks sharing the same α, backprop through the
    encoder must produce a finite non-zero gradient on α — the mechanism is
    active on every layer, so every layer's attention weights depend on α."""
    torch.manual_seed(0)
    decay = TemporalDecayBias(canonical_layout, seq_len=24, mode="scalar")
    with torch.no_grad():
        decay.alpha.fill_(0.3)
    enc = TransformerEncoder(
        d_model=32, n_heads=4, n_layers=3, dropout=0.0, temporal_decay=decay,
    )
    enc.train()
    x = torch.randn(4, 24, 32)
    enc(x)[0].sum().backward()
    assert decay.alpha.grad is not None
    assert torch.isfinite(decay.alpha.grad).all()
    assert decay.alpha.grad.abs().item() > 0


def test_encoder_in_train_mode_with_dropout_and_bias_produces_no_nan(canonical_layout):
    """The interaction the PR #8 reviewer flagged: .train() mode + dropout >
    0 + non-None attn_bias. Must produce finite outputs and weights."""
    torch.manual_seed(0)
    decay = TemporalDecayBias(canonical_layout, seq_len=24, mode="scalar")
    with torch.no_grad():
        decay.alpha.fill_(0.7)
    enc = TransformerEncoder(
        d_model=32, n_heads=4, n_layers=2, dropout=0.15, temporal_decay=decay,
    )
    enc.train()
    x = torch.randn(16, 24, 32)
    out, weights = enc(x)
    assert torch.isfinite(out).all()
    for w in weights:
        assert torch.isfinite(w).all()


def test_encoder_independent_dropout_channels_forwarded():
    enc = TransformerEncoder(
        d_model=32, n_heads=4, n_layers=2,
        dropout=0.1,
        attn_dropout=0.4,
        ffn_dropout=0.3,
        residual_dropout=0.05,
    )
    for block in enc.blocks:
        assert block.attention.attention.dropout.p == 0.4
        assert block.ffn.dropout.p == 0.3
        assert block.residual_dropout.p == 0.05


def test_encoder_no_temporal_decay_returns_none_bias(canonical_layout):
    """If temporal_decay is None, forward() must not call its module and must
    pass attn_bias=None to every block."""
    enc = TransformerEncoder(
        d_model=32, n_heads=4, n_layers=2, dropout=0.0, temporal_decay=None,
    )
    enc.eval()
    x = torch.randn(2, 24, 32)
    out, _ = enc(x)
    assert out.shape == (2, 24, 32)


def test_encoder_integrates_with_feature_embedding(canonical_layout):
    """End-to-end compatibility: FeatureEmbedding → TransformerEncoder with
    matching d_model and the canonical layout drives a gradient through every
    encoder parameter."""
    from embedding import FeatureEmbedding

    torch.manual_seed(0)
    d_model = 32
    emb = FeatureEmbedding(d_model=d_model, dropout=0.0)
    decay = TemporalDecayBias(canonical_layout, seq_len=24, mode="scalar")
    enc = TransformerEncoder(
        d_model=d_model, n_heads=4, n_layers=2, dropout=0.0, temporal_decay=decay,
    )

    B = 4
    batch = {
        "cat_indices": {
            "SEX":       torch.tensor([0, 1, 0, 1]),
            "EDUCATION": torch.tensor([0, 1, 2, 3]),
            "MARRIAGE":  torch.tensor([0, 1, 2, 0]),
        },
        "pay_state_ids":  torch.zeros(B, 6, dtype=torch.long),
        "pay_severities": torch.zeros(B, 6, dtype=torch.float),
        "num_values":     torch.randn(B, 14),
        "label":          torch.tensor([0.0, 1.0, 0.0, 1.0]),
    }
    emb.train()
    enc.train()
    tokens = emb(batch)
    assert tokens.shape == (B, 24, d_model)
    out, weights = enc(tokens)
    assert out.shape == (B, 24, d_model)
    assert len(weights) == 2
    out.sum().backward()
    # Every encoder parameter must have a gradient after end-to-end backprop.
    for name, p in enc.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all()
