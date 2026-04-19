"""transformer.py — FeedForward / TransformerBlock / TemporalDecayBias (N3)
/ FeatureGroupBias (N2) / TransformerEncoder."""

from __future__ import annotations

import pytest
import torch

from src.models.transformer import (  # noqa: E402
    FeatureGroupBias,
    FeedForward,
    TemporalDecayBias,
    TransformerBlock,
    TransformerEncoder,
)
from src.tokenization.embedding import N_FEATURE_GROUPS, build_group_assignment  # noqa: E402


def test_feedforward_output_shape():
    ffn = FeedForward(d_model=32, d_ff=128, dropout=0.0)
    x = torch.randn(4, 24, 32)
    assert ffn(x).shape == (4, 24, 32)


def test_feedforward_d_ff_defaults_to_4x_d_model():
    ffn = FeedForward(d_model=32)
    assert ffn.linear1.out_features == 128
    assert ffn.linear2.in_features == 128


def test_feedforward_init_uses_kaiming_then_xavier():
    # W_1 → GELU → Kaiming, W_2 → residual → Xavier. variances differ
    # predictably, so a distributional check is enough
    torch.manual_seed(0)
    ffn = FeedForward(d_model=64, d_ff=256)
    k_std_expected = (2.0 / 64) ** 0.5
    x_std_expected = (2.0 / (256 + 64)) ** 0.5
    assert abs(ffn.linear1.weight.std().item() - k_std_expected) < 0.02
    assert abs(ffn.linear2.weight.std().item() - x_std_expected) < 0.02
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


def test_transformer_block_shape_and_attn_shape():
    block = TransformerBlock(d_model=32, n_heads=4, dropout=0.0)
    block.eval()
    x = torch.randn(4, 24, 32)
    out, w = block(x)
    assert out.shape == (4, 24, 32)
    assert w.shape == (4, 4, 24, 24)


def test_transformer_block_prenorm_residual_is_identity_like_when_weights_zeroed():
    # PreNorm + residual: if every Linear is zero, out == in. confirms the
    # residual path is wired correctly.
    block = TransformerBlock(d_model=16, n_heads=2, dropout=0.0)
    block.eval()
    with torch.no_grad():
        for p in block.parameters():
            p.zero_()
        for ln in (block.ln1, block.ln2):
            ln.weight.fill_(1.0)
            ln.bias.zero_()
    x = torch.randn(2, 10, 16)
    out, _ = block(x)
    assert torch.allclose(out, x, atol=1e-6)


def test_transformer_block_attn_bias_is_threaded_to_attention():
    torch.manual_seed(0)
    block = TransformerBlock(d_model=32, n_heads=4, dropout=0.0)
    block.eval()
    x = torch.randn(2, 10, 32)
    bias = torch.zeros(10, 10)
    bias[3, 7] = -1e4
    _, w = block(x, attn_bias=bias)
    assert torch.all(w[:, :, 3, 7] < 1e-3)


def test_transformer_block_independent_dropout_channels():
    block = TransformerBlock(
        d_model=32,
        n_heads=4,
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
    # exercises the trio PR #8 smoke missed: non-None attn_bias + .train()
    # + dropout > 0
    torch.manual_seed(0)
    block = TransformerBlock(d_model=32, n_heads=4, dropout=0.2)
    block.train()
    x = torch.randn(8, 24, 32)
    bias = torch.randn(24, 24) * 0.1
    out, w = block(x, attn_bias=bias)
    assert torch.isfinite(out).all()
    assert torch.isfinite(w).all()


@pytest.fixture()
def canonical_layout():
    return {
        "pay": {"positions": [4, 5, 6, 7, 8, 9], "months": [0, 1, 2, 3, 4, 5]},
        "bill": {"positions": [12, 13, 14, 15, 16, 17], "months": [0, 1, 2, 3, 4, 5]},
        "pay_amt": {"positions": [18, 19, 20, 21, 22, 23], "months": [0, 1, 2, 3, 4, 5]},
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
    decay = TemporalDecayBias(canonical_layout, seq_len=24, mode="scalar")
    # (4, 12) PAY_0 vs BILL_AMT1 — cross-group → 0
    assert decay.neg_distance_masked[4, 12].item() == 0.0
    # (4, 9) PAY_0 vs PAY_6 — same group, 5mo apart
    assert decay.neg_distance_masked[4, 9].item() == -5.0
    # CLS never gets a bias
    assert torch.all(decay.neg_distance_masked[0] == 0.0)
    assert torch.all(decay.neg_distance_masked[:, 0] == 0.0)


def test_temporal_decay_suppresses_distant_month_more_than_near(canonical_layout):
    # α=5: penalty at (PAY_0, PAY_6) = -25 vs (PAY_0, PAY_2) = -5, so w at
    # col 9 must drop below col 5 across every row/head
    torch.manual_seed(0)
    decay = TemporalDecayBias(canonical_layout, seq_len=24, mode="scalar")
    with torch.no_grad():
        decay.alpha.fill_(5.0)
    encoder = TransformerEncoder(
        d_model=32,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
        temporal_decay=decay,
    )
    encoder.eval()
    x = torch.randn(4, 24, 32)
    _, weights = encoder(x)
    w0 = weights[0]
    assert torch.all(w0[:, :, 4, 9] < w0[:, :, 4, 5])


def test_temporal_decay_alpha_is_trainable(canonical_layout):
    torch.manual_seed(0)
    decay = TemporalDecayBias(canonical_layout, seq_len=24, mode="scalar")
    with torch.no_grad():
        decay.alpha.fill_(0.5)
    encoder = TransformerEncoder(
        d_model=32,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
        temporal_decay=decay,
    )
    encoder.train()
    x = torch.randn(4, 24, 32)
    out, _ = encoder(x)
    out.sum().backward()
    assert decay.alpha.grad is not None
    assert decay.alpha.grad.abs().item() > 0


def test_temporal_decay_neg_distance_is_non_persistent_buffer(canonical_layout):
    # neg_distance_masked is reconstructible from the layout; persistent=False
    # means checkpoints shouldn't carry it
    decay = TemporalDecayBias(canonical_layout, seq_len=24, mode="scalar")
    assert "neg_distance_masked" not in decay.state_dict()
    assert hasattr(decay, "neg_distance_masked")


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
    # same bias is added to every block → forward() once, reuse; not per layer
    decay = TemporalDecayBias(canonical_layout, seq_len=24, mode="scalar")

    call_counter = {"n": 0}
    real_forward = decay.forward

    def counted_forward(*args, **kwargs):
        call_counter["n"] += 1
        return real_forward(*args, **kwargs)

    decay.forward = counted_forward  # type: ignore[assignment]

    enc = TransformerEncoder(
        d_model=32,
        n_heads=4,
        n_layers=4,
        dropout=0.0,
        temporal_decay=decay,
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
    torch.manual_seed(0)
    decay = TemporalDecayBias(canonical_layout, seq_len=24, mode="scalar")
    with torch.no_grad():
        decay.alpha.fill_(0.3)
    enc = TransformerEncoder(
        d_model=32,
        n_heads=4,
        n_layers=3,
        dropout=0.0,
        temporal_decay=decay,
    )
    enc.train()
    x = torch.randn(4, 24, 32)
    enc(x)[0].sum().backward()
    assert decay.alpha.grad is not None
    assert torch.isfinite(decay.alpha.grad).all()
    assert decay.alpha.grad.abs().item() > 0


def test_encoder_in_train_mode_with_dropout_and_bias_produces_no_nan(canonical_layout):
    # reviewer-flagged on PR #8: .train() + dropout>0 + attn_bias
    torch.manual_seed(0)
    decay = TemporalDecayBias(canonical_layout, seq_len=24, mode="scalar")
    with torch.no_grad():
        decay.alpha.fill_(0.7)
    enc = TransformerEncoder(
        d_model=32,
        n_heads=4,
        n_layers=2,
        dropout=0.15,
        temporal_decay=decay,
    )
    enc.train()
    x = torch.randn(16, 24, 32)
    out, weights = enc(x)
    assert torch.isfinite(out).all()
    for w in weights:
        assert torch.isfinite(w).all()


def test_encoder_independent_dropout_channels_forwarded():
    enc = TransformerEncoder(
        d_model=32,
        n_heads=4,
        n_layers=2,
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
    enc = TransformerEncoder(
        d_model=32,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
        temporal_decay=None,
    )
    enc.eval()
    x = torch.randn(2, 24, 32)
    out, _ = enc(x)
    assert out.shape == (2, 24, 32)


def test_encoder_integrates_with_feature_embedding(canonical_layout):
    from src.tokenization.embedding import FeatureEmbedding

    torch.manual_seed(0)
    d_model = 32
    emb = FeatureEmbedding(d_model=d_model, dropout=0.0)
    decay = TemporalDecayBias(canonical_layout, seq_len=24, mode="scalar")
    enc = TransformerEncoder(
        d_model=d_model,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
        temporal_decay=decay,
    )

    B = 4
    batch = {
        "cat_indices": {
            "SEX": torch.tensor([0, 1, 0, 1]),
            "EDUCATION": torch.tensor([0, 1, 2, 3]),
            "MARRIAGE": torch.tensor([0, 1, 2, 0]),
        },
        "pay_state_ids": torch.zeros(B, 6, dtype=torch.long),
        "pay_severities": torch.zeros(B, 6, dtype=torch.float),
        "num_values": torch.randn(B, 14),
        "label": torch.tensor([0.0, 1.0, 0.0, 1.0]),
    }
    emb.train()
    enc.train()
    tokens = emb(batch)
    assert tokens.shape == (B, 24, d_model)
    out, weights = enc(tokens)
    assert out.shape == (B, 24, d_model)
    assert len(weights) == 2
    out.sum().backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all()


@pytest.fixture()
def canonical_group_assignment():
    # canonical 24-slot groups: [CLS], demographic, PAY, BILL_AMT, PAY_AMT
    return build_group_assignment(cls_offset=1)


def test_feature_group_bias_scalar_output_shape(canonical_group_assignment):
    fgb = FeatureGroupBias(
        group_assignment=canonical_group_assignment,
        n_groups=N_FEATURE_GROUPS,
        mode="scalar",
    )
    bias = fgb()
    assert bias is not None
    assert bias.shape == (24, 24)


def test_feature_group_bias_per_head_output_shape(canonical_group_assignment):
    fgb = FeatureGroupBias(
        group_assignment=canonical_group_assignment,
        n_groups=N_FEATURE_GROUPS,
        mode="per_head",
        n_heads=4,
    )
    bias = fgb()
    assert bias is not None
    assert bias.shape == (4, 24, 24)


def test_feature_group_bias_off_returns_none(canonical_group_assignment):
    fgb = FeatureGroupBias(
        group_assignment=canonical_group_assignment,
        n_groups=N_FEATURE_GROUPS,
        mode="off",
    )
    assert fgb() is None
    assert fgb.bias_matrix is None


def test_feature_group_bias_zero_initialised(canonical_group_assignment):
    # default-safe: B=0 → plain encoder
    fgb_scalar = FeatureGroupBias(
        group_assignment=canonical_group_assignment,
        n_groups=N_FEATURE_GROUPS,
        mode="scalar",
    )
    assert fgb_scalar.bias_matrix is not None
    assert fgb_scalar.bias_matrix.shape == (N_FEATURE_GROUPS, N_FEATURE_GROUPS)
    assert torch.equal(
        fgb_scalar.bias_matrix,
        torch.zeros(N_FEATURE_GROUPS, N_FEATURE_GROUPS),
    )
    assert torch.equal(fgb_scalar(), torch.zeros(24, 24))

    fgb_ph = FeatureGroupBias(
        group_assignment=canonical_group_assignment,
        n_groups=N_FEATURE_GROUPS,
        mode="per_head",
        n_heads=4,
    )
    assert fgb_ph.bias_matrix is not None
    assert fgb_ph.bias_matrix.shape == (4, N_FEATURE_GROUPS, N_FEATURE_GROUPS)
    assert torch.equal(
        fgb_ph.bias_matrix,
        torch.zeros(4, N_FEATURE_GROUPS, N_FEATURE_GROUPS),
    )


def test_feature_group_bias_rejects_unknown_mode(canonical_group_assignment):
    with pytest.raises(ValueError, match="Unknown mode"):
        FeatureGroupBias(
            group_assignment=canonical_group_assignment,
            n_groups=N_FEATURE_GROUPS,
            mode="not_a_mode",  # type: ignore[arg-type]
        )


def test_feature_group_bias_rejects_out_of_range_assignments():
    # OOB indices would silently corrupt the fancy-indexing lookup
    with pytest.raises(ValueError, match=r"group_assignment"):
        FeatureGroupBias(
            group_assignment=[0, 1, 2, 3, 5],
            n_groups=5,
            mode="scalar",
        )
    with pytest.raises(ValueError, match=r"group_assignment"):
        FeatureGroupBias(
            group_assignment=[0, 1, -1, 2, 3],
            n_groups=5,
            mode="scalar",
        )


def test_feature_group_bias_index_lookup_correctness():
    # B[g_x, g_y] = 42 must light up exactly the (i, j) where i ∈ g_x and j ∈ g_y
    assignment = [0, 0, 1, 1, 2, 2]
    fgb = FeatureGroupBias(
        group_assignment=assignment,
        n_groups=3,
        mode="scalar",
    )
    with torch.no_grad():
        fgb.bias_matrix.zero_()
        fgb.bias_matrix[1, 2] = 42.0

    bias = fgb()
    assert bias.shape == (6, 6)
    for i in (2, 3):
        for j in (4, 5):
            assert bias[i, j].item() == 42.0, (i, j)
    mask_nonzero = torch.zeros(6, 6, dtype=torch.bool)
    for i in (2, 3):
        for j in (4, 5):
            mask_nonzero[i, j] = True
    assert torch.all(bias[~mask_nonzero] == 0.0)


def test_feature_group_bias_per_head_different_heads():
    # per-head mode: each head gets an independent B
    assignment = [0, 0, 1, 1, 2, 2]
    fgb = FeatureGroupBias(
        group_assignment=assignment,
        n_groups=3,
        mode="per_head",
        n_heads=2,
    )
    with torch.no_grad():
        fgb.bias_matrix.zero_()
        fgb.bias_matrix[0, 0, 1] = 7.0
        fgb.bias_matrix[1, 1, 0] = -3.0

    bias = fgb()
    assert bias.shape == (2, 6, 6)
    for i in (0, 1):
        for j in (2, 3):
            assert bias[0, i, j].item() == 7.0, ("head0", i, j)
            assert bias[1, i, j].item() == 0.0, ("head1 should be zero", i, j)
    for i in (2, 3):
        for j in (0, 1):
            assert bias[1, i, j].item() == -3.0, ("head1", i, j)
            assert bias[0, i, j].item() == 0.0, ("head0 should be zero", i, j)


def test_feature_group_bias_gradient_flows_to_matrix(canonical_group_assignment):
    torch.manual_seed(0)
    fgb = FeatureGroupBias(
        group_assignment=canonical_group_assignment,
        n_groups=N_FEATURE_GROUPS,
        mode="scalar",
    )
    with torch.no_grad():
        fgb.bias_matrix.fill_(0.2)

    enc = TransformerEncoder(
        d_model=32,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
        feature_group_bias=fgb,
    )
    enc.train()
    x = torch.randn(4, 24, 32)
    enc(x)[0].sum().backward()
    assert fgb.bias_matrix.grad is not None
    assert torch.isfinite(fgb.bias_matrix.grad).all()
    assert fgb.bias_matrix.grad.abs().sum().item() > 0


def test_encoder_composes_temporal_decay_and_feature_group_bias(
    canonical_layout,
    canonical_group_assignment,
):
    # composed attn_bias == sum of the two module outputs. with both zero-
    # init, attention matches a plain encoder at the same seed.
    torch.manual_seed(0)
    x = torch.randn(2, 24, 32)

    torch.manual_seed(42)
    enc_plain = TransformerEncoder(
        d_model=32,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
    )
    enc_plain.eval()
    _, weights_plain = enc_plain(x)

    torch.manual_seed(42)
    decay = TemporalDecayBias(canonical_layout, seq_len=24, mode="scalar")
    fgb = FeatureGroupBias(
        group_assignment=canonical_group_assignment,
        n_groups=N_FEATURE_GROUPS,
        mode="scalar",
    )
    enc_both = TransformerEncoder(
        d_model=32,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
        temporal_decay=decay,
        feature_group_bias=fgb,
    )
    enc_both.eval()
    _, weights_both_zero = enc_both(x)
    for w_plain, w_zero in zip(weights_plain, weights_both_zero):
        assert torch.allclose(
            w_plain, w_zero, atol=1e-6
        ), "zero-initialised biases must leave attention unchanged vs plain encoder"

    with torch.no_grad():
        decay.alpha.fill_(0.3)
        fgb.bias_matrix.fill_(0.5)
    expected = decay() + fgb()
    composed = enc_both._compose_attn_bias()
    assert composed is not None
    assert torch.allclose(composed, expected, atol=1e-6)

    _, weights_both_active = enc_both(x)
    total_diff = sum(
        (wa - wz).abs().sum().item() for wa, wz in zip(weights_both_active, weights_both_zero)
    )
    assert total_diff > 1e-3, "activating α and B should measurably shift the attention weights"


def test_encoder_no_bias_when_both_none():
    # plain path (no biases): pre-N2/N3 behaviour, _compose_attn_bias → None
    enc = TransformerEncoder(
        d_model=32,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
        temporal_decay=None,
        feature_group_bias=None,
    )
    enc.eval()
    x = torch.randn(4, 24, 32)
    out, weights = enc(x)
    assert out.shape == (4, 24, 32)
    assert len(weights) == 2
    assert enc._compose_attn_bias() is None
    for w in weights:
        assert w.shape == (4, 4, 24, 24)
        row_sums = w.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
