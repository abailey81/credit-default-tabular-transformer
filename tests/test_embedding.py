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
