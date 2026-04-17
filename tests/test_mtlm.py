"""Tests for src/mtlm.py — MTLMHead, mtlm_loss, MTLMModel (Plan §8.5 / Novelty N4).

Scope
-----
- MTLMHead output dict shape + per-feature head types
- MTLMHead gradient flow
- mtlm_loss: per-feature-type contributions, entropy / variance
  normalisation, zero-mask graceful behaviour, checkpoint compatibility
  with TabularTransformer via MTLMModel.encoder_state_dict()
- MTLMModel forward + state-dict key compatibility
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
import torch

from embedding import FeatureEmbedding, N_FEATURE_GROUPS, build_group_assignment, build_temporal_layout  # noqa: E402
from model import TabularTransformer  # noqa: E402
from mtlm import MTLMHead, MTLMLossComponents, MTLMModel, mtlm_loss  # noqa: E402
from tokenizer import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PAY_RAW_NUM_CLASSES,
    PAY_STATUS_FEATURES,
)
from transformer import FeatureGroupBias, TemporalDecayBias, TransformerEncoder  # noqa: E402


@pytest.fixture()
def tiny_cat_vocab_sizes() -> Dict[str, int]:
    return {"SEX": 2, "EDUCATION": 4, "MARRIAGE": 3}


@pytest.fixture()
def mtlm_batch() -> Dict[str, Any]:
    """Batch of size 4 with all six MTLMCollator output keys plus
    mask_positions selecting tokens across every feature type."""
    B = 4
    mask = torch.zeros(B, 23, dtype=torch.bool)
    # Mask: one categorical (SEX on row 0), two PAYs (row 1 PAY_0, row 2 PAY_6),
    # one numerical (row 3 LIMIT_BAL — position 9 in the 23-block).
    mask[0, 0] = True
    mask[1, 3] = True
    mask[2, 8] = True
    mask[3, 9] = True
    return {
        "cat_indices": {
            "SEX":       torch.tensor([0, 1, 0, 1]),
            "EDUCATION": torch.tensor([0, 1, 2, 3]),
            "MARRIAGE":  torch.tensor([0, 1, 2, 0]),
        },
        "pay_state_ids":  torch.zeros(B, 6, dtype=torch.long),
        "pay_severities": torch.zeros(B, 6, dtype=torch.float),
        "pay_raw":        torch.tensor(
            [[2, 2, 2, 2, 2, 2],
             [5, 2, 2, 2, 2, 2],
             [0, 0, 0, 0, 0, 10],
             [2, 2, 2, 2, 2, 2]],
            dtype=torch.long,
        ),
        "num_values":     torch.randn(B, 14),
        "label":          torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "mask_positions": mask,
    }


# ──────────────────────────────────────────────────────────────────────────────
# MTLMHead
# ──────────────────────────────────────────────────────────────────────────────


def test_mtlm_head_output_shapes(tiny_cat_vocab_sizes):
    head = MTLMHead(d_model=32, cat_vocab_sizes=tiny_cat_vocab_sizes)
    hidden = torch.randn(4, 24, 32)  # (B, seq, d_model)
    out = head(hidden)

    assert set(out.keys()) == {"cat", "pay", "num"}
    # Categoricals — per-feature vocab sizes.
    assert out["cat"]["SEX"].shape == (4, 2)
    assert out["cat"]["EDUCATION"].shape == (4, 4)
    assert out["cat"]["MARRIAGE"].shape == (4, 3)
    # PAY — unified 11-class vocab for every PAY feature.
    for feat in PAY_STATUS_FEATURES:
        assert out["pay"][feat].shape == (4, PAY_RAW_NUM_CLASSES)
    # Numerical — scalar per feature.
    for feat in NUMERICAL_FEATURES:
        assert out["num"][feat].shape == (4,)


def test_mtlm_head_gradient_flows(tiny_cat_vocab_sizes):
    head = MTLMHead(d_model=32, cat_vocab_sizes=tiny_cat_vocab_sizes)
    head.train()
    hidden = torch.randn(4, 24, 32, requires_grad=True)
    out = head(hidden)
    # Backprop through a reduction of every head's output.
    loss = (
        sum(v.sum() for v in out["cat"].values())
        + sum(v.sum() for v in out["pay"].values())
        + sum(v.sum() for v in out["num"].values())
    )
    loss.backward()
    for name, p in head.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all()
    assert hidden.grad is not None


def test_mtlm_head_rejects_non_canonical_numerical_order(tiny_cat_vocab_sizes):
    with pytest.raises(ValueError, match="canonical tokenizer.NUMERICAL_FEATURES"):
        MTLMHead(
            d_model=32,
            cat_vocab_sizes=tiny_cat_vocab_sizes,
            numerical_features=["LIMIT_BAL"],  # not the canonical order
        )


# ──────────────────────────────────────────────────────────────────────────────
# mtlm_loss
# ──────────────────────────────────────────────────────────────────────────────


def test_mtlm_loss_returns_components_and_backprops(tiny_cat_vocab_sizes, mtlm_batch):
    """Each feature-type loss component is positive, the joint scalar is
    finite, and the backward call lands a gradient on the heads for the
    *specific* features that were masked in the fixture.

    The fixture masks only four positions (SEX on row 0, PAY_0 on row 1,
    PAY_6 on row 2, LIMIT_BAL on row 3). Heads for never-masked features
    (EDUCATION, MARRIAGE, most PAYs, most numericals) won't receive any
    gradient, which is the expected behaviour — the loss is only computed
    on masked positions — not a bug."""
    head = MTLMHead(d_model=32, cat_vocab_sizes=tiny_cat_vocab_sizes)
    head.train()
    hidden = torch.randn(4, 24, 32, requires_grad=True)
    predictions = head(hidden)

    loss, comps = mtlm_loss(
        predictions=predictions,
        batch=mtlm_batch,
        mask_positions=mtlm_batch["mask_positions"],
    )
    assert isinstance(comps, MTLMLossComponents)
    # Every feature-type component active given the mask layout in fixture.
    assert comps.cat > 0
    assert comps.pay > 0
    assert comps.num > 0
    assert comps.n_masked == int(mtlm_batch["mask_positions"].sum())
    assert torch.isfinite(loss)
    loss.backward()

    # Only the heads for the specific masked features must receive gradient.
    masked_features = {
        "cat": ["SEX"],
        "pay": ["PAY_0", "PAY_6"],
        "num": ["LIMIT_BAL"],
    }
    for feat in masked_features["cat"]:
        for p in head.cat_heads[feat].parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), feat
    for feat in masked_features["pay"]:
        for p in head.pay_heads[feat].parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), feat
    for feat in masked_features["num"]:
        for p in head.num_heads[feat].parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), feat


def test_mtlm_loss_empty_mask_returns_zero(tiny_cat_vocab_sizes, mtlm_batch):
    """If no tokens are masked, the composite loss is exactly zero — no
    NaNs from empty stacks, no gradient flow surprises."""
    head = MTLMHead(d_model=32, cat_vocab_sizes=tiny_cat_vocab_sizes)
    hidden = torch.randn(4, 24, 32)
    predictions = head(hidden)
    empty_mask = torch.zeros(4, 23, dtype=torch.bool)
    loss, comps = mtlm_loss(predictions, mtlm_batch, empty_mask)
    assert float(loss.item()) == pytest.approx(0.0)
    assert comps.cat == pytest.approx(0.0)
    assert comps.pay == pytest.approx(0.0)
    assert comps.num == pytest.approx(0.0)
    assert comps.n_masked == 0


def test_mtlm_loss_entropy_normalised_within_cross_entropy_bounds(
    tiny_cat_vocab_sizes, mtlm_batch,
):
    """Entropy-normalised CE on a uniform-random head should hover around
    1.0 (per-feature ``CE / ln(n_cats)`` is the "information-theoretic
    baseline": a uniform prediction loses exactly ``ln(n)`` nats)."""
    torch.manual_seed(0)
    head = MTLMHead(d_model=32, cat_vocab_sizes=tiny_cat_vocab_sizes)
    # Zero all head weights — predictions become constant, softmax uniform,
    # CE per feature ≈ ln(n_cats), entropy-normalised ≈ 1.0.
    with torch.no_grad():
        for lin in (*head.cat_heads.values(), *head.pay_heads.values()):
            lin.weight.zero_()
            lin.bias.zero_()
    hidden = torch.randn(4, 24, 32)
    predictions = head(hidden)
    _, comps = mtlm_loss(predictions, mtlm_batch, mtlm_batch["mask_positions"])
    # Each normalised CE is exactly 1.0 under uniform predictions, and the
    # reported value is the mean across contributing features.
    assert 0.95 < comps.cat < 1.05
    assert 0.95 < comps.pay < 1.05


def test_mtlm_loss_variance_normalisation_on_numerical(
    tiny_cat_vocab_sizes, mtlm_batch,
):
    """Dividing MSE by a larger variance must scale the numerical component
    down proportionally."""
    head = MTLMHead(d_model=32, cat_vocab_sizes=tiny_cat_vocab_sizes)
    hidden = torch.randn(4, 24, 32)
    predictions = head(hidden)

    variance_1 = {feat: 1.0 for feat in NUMERICAL_FEATURES}
    variance_4 = {feat: 4.0 for feat in NUMERICAL_FEATURES}
    _, c1 = mtlm_loss(predictions, mtlm_batch, mtlm_batch["mask_positions"],
                      num_feature_variance=variance_1)
    _, c4 = mtlm_loss(predictions, mtlm_batch, mtlm_batch["mask_positions"],
                      num_feature_variance=variance_4)
    # Dividing by 4× the variance should give exactly 1/4 of the num loss.
    assert c4.num == pytest.approx(c1.num / 4.0, rel=1e-4)


# ──────────────────────────────────────────────────────────────────────────────
# MTLMModel — integration + checkpoint compatibility with TabularTransformer
# ──────────────────────────────────────────────────────────────────────────────


def _build_mtlm_model(d_model: int = 32, n_heads: int = 4, n_layers: int = 2) -> MTLMModel:
    cat_vocab_sizes = {"SEX": 2, "EDUCATION": 4, "MARRIAGE": 3}
    emb = FeatureEmbedding(
        d_model=d_model, dropout=0.0,
        cat_vocab_sizes=cat_vocab_sizes,
        use_temporal_pos=False, use_mask_token=True,
    )
    enc = TransformerEncoder(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=0.0,
    )
    head = MTLMHead(d_model=d_model, cat_vocab_sizes=cat_vocab_sizes)
    return MTLMModel(emb, enc, head)


def test_mtlm_model_forward_end_to_end(mtlm_batch):
    model = _build_mtlm_model()
    model.eval()
    out = model(mtlm_batch)
    # Sanity: all heads produced output at every feature slot.
    assert set(out["cat"].keys()) == set(CATEGORICAL_FEATURES)
    assert set(out["pay"].keys()) == set(PAY_STATUS_FEATURES)
    assert set(out["num"].keys()) == set(NUMERICAL_FEATURES)


def test_mtlm_model_encoder_state_dict_has_only_embedding_and_encoder():
    model = _build_mtlm_model()
    encoder_sd = model.encoder_state_dict()
    # Every key must start with 'embedding.' or 'encoder.' — that's exactly
    # the set TabularTransformer.load_pretrained_encoder consumes.
    for key in encoder_sd:
        assert key.startswith("embedding.") or key.startswith("encoder."), key
    # And every encoder / embedding parameter in the full model appears.
    full = model.state_dict()
    for key in full:
        if key.startswith("embedding.") or key.startswith("encoder."):
            assert key in encoder_sd, f"missing {key}"
        else:
            assert key not in encoder_sd, f"should not include {key}"


def test_tabular_transformer_loads_mtlm_encoder_state_dict(tmp_path):
    """Integration: pretrain an MTLMModel, save encoder_state_dict,
    construct a fresh TabularTransformer, call load_pretrained_encoder
    on the raw file. The embedding+encoder weights must transfer exactly;
    the classification head must stay at fresh init."""
    torch.manual_seed(0)
    cat_vocab_sizes = {"SEX": 2, "EDUCATION": 4, "MARRIAGE": 3}
    # Shared architecture — both models must agree on every layer shape.
    mtlm = _build_mtlm_model(d_model=32, n_heads=4, n_layers=2)

    # Snapshot a couple of encoder weights before any state transfer.
    mtlm_qweight = mtlm.encoder.blocks[0].attention.W_Q.weight.detach().clone()
    mtlm_emb_cls = mtlm.embedding.cls_token.detach().clone()

    encoder_sd = mtlm.encoder_state_dict()
    path = tmp_path / "encoder_pretrained.pt"
    torch.save(encoder_sd, path)

    torch.manual_seed(42)  # different seed → different random init
    tabular = TabularTransformer(
        d_model=32, n_heads=4, n_layers=2,
        cat_vocab_sizes=cat_vocab_sizes,
    )
    # Classifier weights before the load — different from any MTLM weight.
    head_w_before = tabular.classifier[0].weight.detach().clone()

    tabular.load_pretrained_encoder(path, strict=False)

    # Encoder/embedding should now match the MTLM snapshot.
    assert torch.allclose(
        tabular.encoder.blocks[0].attention.W_Q.weight, mtlm_qweight, atol=1e-6
    )
    assert torch.allclose(tabular.embedding.cls_token, mtlm_emb_cls, atol=1e-6)
    # Classifier head should remain untouched.
    assert torch.allclose(tabular.classifier[0].weight, head_w_before, atol=1e-6)


def test_tabular_transformer_load_raises_on_missing_file(tmp_path):
    tabular = TabularTransformer(cat_vocab_sizes={"SEX": 2, "EDUCATION": 4, "MARRIAGE": 3})
    with pytest.raises(FileNotFoundError):
        tabular.load_pretrained_encoder(tmp_path / "does_not_exist.pt")
