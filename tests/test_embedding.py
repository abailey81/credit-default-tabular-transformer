"""Tests for src/embedding.py — FeatureEmbedding across all modes."""

from __future__ import annotations

import pytest
import torch

from embedding import (  # noqa: E402
    CAT_VOCAB_SIZES,
    N_MONTHS,
    TOKEN_ORDER,
    FeatureEmbedding,
)


@pytest.fixture()
def dummy_batch():
    """A B=4 batch with mixed PAY states so non-zero severities are exercised."""
    B = 4
    return {
        "cat_indices": {
            "SEX":       torch.tensor([0, 1, 0, 1]),
            "EDUCATION": torch.tensor([0, 1, 2, 3]),
            "MARRIAGE":  torch.tensor([0, 1, 2, 0]),
        },
        "pay_state_ids":  torch.tensor(
            [[0, 1, 2, 3, 3, 3], [3, 3, 2, 1, 0, 0], [2, 2, 2, 2, 2, 2], [0, 0, 0, 3, 3, 3]],
            dtype=torch.long,
        ),
        "pay_severities": torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.25, 0.50, 0.75],
                [1.0, 0.875, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.375, 0.625, 1.0],
            ],
            dtype=torch.float,
        ),
        "num_values":     torch.randn(B, 14),
        "label":          torch.tensor([0.0, 1.0, 0.0, 1.0]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Baseline
# ──────────────────────────────────────────────────────────────────────────────


def test_output_shape_d32(dummy_batch):
    model = FeatureEmbedding(d_model=32, dropout=0.0)
    model.eval()
    out = model(dummy_batch)
    assert out.shape == (4, 24, 32)


def test_output_no_nan_inf(dummy_batch):
    model = FeatureEmbedding(d_model=32, dropout=0.0)
    model.eval()
    out = model(dummy_batch)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_all_params_have_gradient(dummy_batch):
    model = FeatureEmbedding(d_model=32, dropout=0.0)
    model.train()
    torch.manual_seed(7)
    proj = torch.randn(4, 24, 32)
    (model(dummy_batch) * proj).sum().backward()
    zero = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not zero, f"zero-grad params: {zero}"


def test_severity_projection_is_exercised(dummy_batch):
    """Regression for the earlier smoke-test bug where severity was all-zero."""
    model = FeatureEmbedding(d_model=32, dropout=0.0)
    model.train()
    torch.manual_seed(0)
    proj = torch.randn(4, 24, 32)
    (model(dummy_batch) * proj).sum().backward()
    assert model.pay_severity_proj.weight.grad.abs().sum().item() > 0


# ──────────────────────────────────────────────────────────────────────────────
# Temporal positional encoding
# ──────────────────────────────────────────────────────────────────────────────


def test_temporal_pos_embedding_created_when_enabled():
    model = FeatureEmbedding(d_model=32, use_temporal_pos=True)
    assert model.temporal_pos_embedding is not None
    assert model.temporal_pos_embedding.num_embeddings == N_MONTHS


def test_temporal_pos_embedding_absent_by_default():
    model = FeatureEmbedding(d_model=32)
    assert model.temporal_pos_embedding is None


def test_temporal_pos_receives_gradient(dummy_batch):
    model = FeatureEmbedding(d_model=32, dropout=0.0, use_temporal_pos=True)
    model.train()
    torch.manual_seed(0)
    proj = torch.randn(4, 24, 32)
    (model(dummy_batch) * proj).sum().backward()
    assert model.temporal_pos_embedding.weight.grad.abs().sum().item() > 0


def test_non_temporal_slots_are_correct():
    model = FeatureEmbedding(d_model=32, use_temporal_pos=True)
    non_temp = (model.temporal_index_per_token == -1).nonzero(as_tuple=True)[0].tolist()
    # SEX=0, EDUCATION=1, MARRIAGE=2, LIMIT_BAL=9, AGE=10 within the 23-block
    assert non_temp == [0, 1, 2, 9, 10]


# ──────────────────────────────────────────────────────────────────────────────
# [MASK] token
# ──────────────────────────────────────────────────────────────────────────────


def test_mask_token_created_when_enabled():
    model = FeatureEmbedding(d_model=32, use_mask_token=True)
    assert model.mask_token is not None and model.mask_token.shape == (32,)


def test_mask_token_absent_by_default():
    model = FeatureEmbedding(d_model=32)
    assert model.mask_token is None


def test_masked_slots_differ_from_unmasked(dummy_batch):
    model = FeatureEmbedding(d_model=32, dropout=0.0, use_mask_token=True)
    model.eval()
    B = 4

    mask_positions = torch.zeros(B, 23, dtype=torch.bool)
    mask_positions[0, 3] = True  # mask row 0's PAY_0
    mask_positions[1, 9] = True  # mask row 1's LIMIT_BAL

    batch_masked = {**dummy_batch, "mask_positions": mask_positions}
    batch_no_mask = {**dummy_batch, "mask_positions": torch.zeros(B, 23, dtype=torch.bool)}

    out_m = model(batch_masked)
    out_u = model(batch_no_mask)

    # Masked slots must differ from unmasked.
    # Output is (B, 24, d); +1 for CLS offset.
    diff_mask = (out_m[0, 1 + 3] - out_u[0, 1 + 3]).abs().sum().item()
    diff_mask_2 = (out_m[1, 1 + 9] - out_u[1, 1 + 9]).abs().sum().item()
    assert diff_mask > 1e-3
    assert diff_mask_2 > 1e-3


def test_unmasked_slots_identical_to_no_mask_run(dummy_batch):
    model = FeatureEmbedding(d_model=32, dropout=0.0, use_mask_token=True)
    model.eval()
    B = 4
    mask_positions = torch.zeros(B, 23, dtype=torch.bool)
    mask_positions[0, 3] = True  # mask PAY_0 only for row 0

    batch_masked = {**dummy_batch, "mask_positions": mask_positions}
    batch_no_mask = {**dummy_batch, "mask_positions": torch.zeros(B, 23, dtype=torch.bool)}

    out_m = model(batch_masked)
    out_u = model(batch_no_mask)

    # Row 3 (no masks at all) should be identical in both runs.
    assert torch.allclose(out_m[3], out_u[3], atol=1e-5)


def test_mask_token_receives_gradient_when_used(dummy_batch):
    model = FeatureEmbedding(d_model=32, dropout=0.0, use_mask_token=True)
    model.train()
    mask_positions = torch.zeros(4, 23, dtype=torch.bool)
    mask_positions[:, 3] = True  # mask PAY_0 for every row
    batch = {**dummy_batch, "mask_positions": mask_positions}
    torch.manual_seed(0)
    proj = torch.randn(4, 24, 32)
    (model(batch) * proj).sum().backward()
    assert model.mask_token.grad is not None
    assert model.mask_token.grad.abs().sum().item() > 0


def test_mask_positions_wrong_shape_raises(dummy_batch):
    model = FeatureEmbedding(d_model=32, dropout=0.0, use_mask_token=True)
    model.eval()
    bad = {**dummy_batch, "mask_positions": torch.zeros(4, 10, dtype=torch.bool)}
    with pytest.raises(ValueError, match="mask_positions shape"):
        model(bad)


# ──────────────────────────────────────────────────────────────────────────────
# Determinism
# ──────────────────────────────────────────────────────────────────────────────


def test_two_fresh_models_same_seed_identical_output(dummy_batch):
    torch.manual_seed(0)
    m1 = FeatureEmbedding(d_model=32, dropout=0.0)
    torch.manual_seed(0)
    m2 = FeatureEmbedding(d_model=32, dropout=0.0)
    m1.eval()
    m2.eval()
    assert torch.allclose(m1(dummy_batch), m2(dummy_batch), atol=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# Hardening coverage added after PR #8: MTLM × temporal-pos interaction,
# cat_vocab_sizes override, lazy metadata loader, build_temporal_layout helper.
# ──────────────────────────────────────────────────────────────────────────────


def test_mtlm_mask_preserves_temporal_pos_at_masked_temporal_tokens(dummy_batch):
    """
    Regression for the pre-hardening bug where MTLM-masked temporal tokens
    silently lost their temporal positional embedding (content path only
    re-added ``pos_embedding``, so a masked PAY_0 no longer carried a month-0
    signal). Concretely: perturb ``temporal_pos_embedding.weight[0]`` (month
    0) heterogeneously; a masked PAY_0 (month-0 temporal token at output
    position 4) must respond. A masked SEX (non-temporal, month index = -1)
    at output position 1 must NOT respond.
    """
    torch.manual_seed(0)
    model = FeatureEmbedding(
        d_model=32, dropout=0.0,
        use_temporal_pos=True, use_mask_token=True,
    )
    model.eval()
    B = 4

    mask_positions = torch.zeros(B, 23, dtype=torch.bool)
    mask_positions[0, 3] = True  # mask PAY_0 (month 0) on row 0
    mask_positions[1, 0] = True  # mask SEX (non-temporal) on row 1
    batch = {**dummy_batch, "mask_positions": mask_positions}

    out_before = model(batch)

    # Heterogeneous perturbation of month-0 temporal embedding so LayerNorm
    # can't subtract it out (a uniform shift would be invisible).
    torch.manual_seed(123)
    with torch.no_grad():
        model.temporal_pos_embedding.weight.data[0].normal_(mean=0.0, std=5.0)

    out_after = model(batch)

    # Masked PAY_0 (output position 1 + 3 = 4) should respond.
    diff_masked_pay0 = (out_before[0, 4] - out_after[0, 4]).abs().sum().item()
    # Masked SEX (output position 1 + 0 = 1) is non-temporal → no response.
    diff_masked_sex = (out_before[1, 1] - out_after[1, 1]).abs().sum().item()

    assert diff_masked_pay0 > 1e-2, (
        f"masked temporal token (PAY_0) did not respond to temporal_pos_embed[0] "
        f"perturbation (diff={diff_masked_pay0:.6f}) — regression to the pre-fix "
        f"temporal-pos-wipe behaviour on mask"
    )
    assert diff_masked_sex < 1e-3, (
        f"masked non-temporal token (SEX) should be insensitive to "
        f"temporal_pos_embed perturbation (diff={diff_masked_sex:.6f})"
    )


def test_mtlm_mask_preserves_temporal_pos_differs_across_months(dummy_batch):
    """Two sibling rows masking the same *kind* of token at different
    temporal positions (PAY_0 = month 0 vs PAY_6 = month 5) must produce
    distinguishable outputs once temporal_pos is active — the only
    difference between those two slots' content is their month embedding."""
    torch.manual_seed(0)
    model = FeatureEmbedding(
        d_model=32, dropout=0.0,
        use_temporal_pos=True, use_mask_token=True,
    )
    model.eval()
    B = 4

    # Row 0: mask PAY_0 (month 0).
    # Row 1: mask PAY_6 (month 5).
    mask_positions = torch.zeros(B, 23, dtype=torch.bool)
    mask_positions[0, 3] = True  # PAY_0 → output pos 4
    mask_positions[1, 8] = True  # PAY_6 → output pos 9

    # Make the two rows otherwise identical so the only systematic difference
    # in the masked slot's content is the month index.
    for key in ("pay_state_ids", "pay_severities", "num_values"):
        dummy_batch[key][1] = dummy_batch[key][0]
    for cat_key in dummy_batch["cat_indices"]:
        dummy_batch["cat_indices"][cat_key][1] = dummy_batch["cat_indices"][cat_key][0]

    batch = {**dummy_batch, "mask_positions": mask_positions}
    out = model(batch)

    masked_month0 = out[0, 4]  # PAY_0 slot
    masked_month5 = out[1, 9]  # PAY_6 slot

    # The two masked slots have (a) the same mask_token, (b) different
    # positional embeddings (pos 4 vs 9), (c) different temporal embeds
    # (month 0 vs 5). Their outputs must reflect those differences.
    diff = (masked_month0 - masked_month5).abs().sum().item()
    assert diff > 1e-3, (
        f"masked PAY_0 and masked PAY_6 produced near-identical "
        f"representations (diff={diff:.6f}) — temporal signal lost"
    )


def test_cat_vocab_sizes_override_is_honoured():
    """Passing an explicit cat_vocab_sizes dict must make the constructor
    hermetic (no disk I/O), and the resulting embedding tables must have
    the specified sizes."""
    custom = {"SEX": 7, "EDUCATION": 11, "MARRIAGE": 13}
    model = FeatureEmbedding(d_model=16, cat_vocab_sizes=custom)
    assert model.cat_embeddings["SEX"].num_embeddings == 7
    assert model.cat_embeddings["EDUCATION"].num_embeddings == 11
    assert model.cat_embeddings["MARRIAGE"].num_embeddings == 13


def test_cat_vocab_sizes_missing_feature_raises():
    bad = {"SEX": 2, "EDUCATION": 4}  # MARRIAGE missing
    with pytest.raises(KeyError, match="MARRIAGE"):
        FeatureEmbedding(d_model=16, cat_vocab_sizes=bad)


def test_load_cat_vocab_sizes_lazy_and_matches_defaults():
    """The lazy loader returns the canonical sizes — identical to the legacy
    ``CAT_VOCAB_SIZES`` module attribute (preserved via PEP 562)."""
    from embedding import load_cat_vocab_sizes
    loaded = load_cat_vocab_sizes()
    assert loaded == CAT_VOCAB_SIZES
    # Standard sizes for this dataset (2 SEX, 4 EDUCATION post-cleaning, 3 MARRIAGE).
    assert loaded["SEX"] == 2
    assert loaded["EDUCATION"] == 4
    assert loaded["MARRIAGE"] == 3


def test_build_temporal_layout_matches_canonical_token_order():
    """Drift guard: ensure ``build_temporal_layout()`` agrees with the
    canonical positions that ``TemporalDecayBias`` was designed around. If
    TOKEN_ORDER is ever reordered without updating the layout helper, this
    test catches it before any downstream consumer silently suppresses the
    wrong tokens."""
    from embedding import build_temporal_layout
    layout = build_temporal_layout()
    assert layout == {
        "pay":     {"positions": [4, 5, 6, 7, 8, 9],         "months": [0, 1, 2, 3, 4, 5]},
        "bill":    {"positions": [12, 13, 14, 15, 16, 17],   "months": [0, 1, 2, 3, 4, 5]},
        "pay_amt": {"positions": [18, 19, 20, 21, 22, 23],   "months": [0, 1, 2, 3, 4, 5]},
    }


def test_build_temporal_layout_respects_cls_offset_zero():
    """With cls_offset=0 positions should address the 23-feature-token
    sequence directly (no CLS prepended)."""
    from embedding import build_temporal_layout
    layout = build_temporal_layout(cls_offset=0)
    assert layout["pay"]["positions"] == [3, 4, 5, 6, 7, 8]
    assert layout["bill"]["positions"] == [11, 12, 13, 14, 15, 16]
    assert layout["pay_amt"]["positions"] == [17, 18, 19, 20, 21, 22]


def test_token_order_is_well_formed():
    """TOKEN_ORDER length matches the embedding's 23-feature-slot assumption."""
    assert len(TOKEN_ORDER) == 23
    # Categoricals first (positions 0-2), then PAY (3-8), then numerical (9-22).
    assert TOKEN_ORDER[:3] == ["SEX", "EDUCATION", "MARRIAGE"]
    assert TOKEN_ORDER[3:9] == ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    assert TOKEN_ORDER[9:11] == ["LIMIT_BAL", "AGE"]
    assert TOKEN_ORDER[11:17] == [f"BILL_AMT{i}" for i in range(1, 7)]
    assert TOKEN_ORDER[17:23] == [f"PAY_AMT{i}" for i in range(1, 7)]
