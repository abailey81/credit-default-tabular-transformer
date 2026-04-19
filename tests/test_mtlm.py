"""MTLMHead / mtlm_loss / MTLMModel (N4, Plan §8.5)."""

from __future__ import annotations

from typing import Any, Dict

import pytest
import torch

from src.models.model import TabularTransformer  # noqa: E402
from src.models.mtlm import MTLMHead, MTLMLossComponents, MTLMModel, mtlm_loss  # noqa: E402
from src.models.transformer import FeatureGroupBias, TemporalDecayBias, TransformerEncoder  # noqa: E402
from src.tokenization.embedding import FeatureEmbedding, N_FEATURE_GROUPS, build_group_assignment, build_temporal_layout  # noqa: E402
from src.tokenization.tokenizer import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PAY_RAW_NUM_CLASSES,
    PAY_STATUS_FEATURES,
)


@pytest.fixture()
def tiny_cat_vocab_sizes() -> Dict[str, int]:
    return {"SEX": 2, "EDUCATION": 4, "MARRIAGE": 3}


@pytest.fixture()
def mtlm_batch() -> Dict[str, Any]:
    B = 4
    mask = torch.zeros(B, 23, dtype=torch.bool)
    # 1 cat (SEX r0), 2 PAYs (PAY_0 r1, PAY_6 r2), 1 num (LIMIT_BAL r3 @ pos 9)
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


def test_mtlm_head_output_shapes(tiny_cat_vocab_sizes):
    head = MTLMHead(d_model=32, cat_vocab_sizes=tiny_cat_vocab_sizes)
    hidden = torch.randn(4, 24, 32)
    out = head(hidden)

    assert set(out.keys()) == {"cat", "pay", "num"}
    assert out["cat"]["SEX"].shape == (4, 2)
    assert out["cat"]["EDUCATION"].shape == (4, 4)
    assert out["cat"]["MARRIAGE"].shape == (4, 3)
    for feat in PAY_STATUS_FEATURES:
        assert out["pay"][feat].shape == (4, PAY_RAW_NUM_CLASSES)
    for feat in NUMERICAL_FEATURES:
        assert out["num"][feat].shape == (4,)


def test_mtlm_head_gradient_flows(tiny_cat_vocab_sizes):
    head = MTLMHead(d_model=32, cat_vocab_sizes=tiny_cat_vocab_sizes)
    head.train()
    hidden = torch.randn(4, 24, 32, requires_grad=True)
    out = head(hidden)
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
            numerical_features=["LIMIT_BAL"],
        )


def test_mtlm_loss_returns_components_and_backprops(tiny_cat_vocab_sizes, mtlm_batch):
    # only heads for the *specifically* masked features should get grad —
    # unmasked-feature heads at zero grad is correct, not a bug
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
    assert comps.cat > 0
    assert comps.pay > 0
    assert comps.num > 0
    assert comps.n_masked == int(mtlm_batch["mask_positions"].sum())
    assert torch.isfinite(loss)
    loss.backward()

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
    # uniform preds lose ln(n) nats/feature, so CE / ln(n_cats) ≈ 1.0
    torch.manual_seed(0)
    head = MTLMHead(d_model=32, cat_vocab_sizes=tiny_cat_vocab_sizes)
    with torch.no_grad():
        for lin in (*head.cat_heads.values(), *head.pay_heads.values()):
            lin.weight.zero_()
            lin.bias.zero_()
    hidden = torch.randn(4, 24, 32)
    predictions = head(hidden)
    _, comps = mtlm_loss(predictions, mtlm_batch, mtlm_batch["mask_positions"])
    assert 0.95 < comps.cat < 1.05
    assert 0.95 < comps.pay < 1.05


def test_mtlm_loss_variance_normalisation_on_numerical(
    tiny_cat_vocab_sizes, mtlm_batch,
):
    head = MTLMHead(d_model=32, cat_vocab_sizes=tiny_cat_vocab_sizes)
    hidden = torch.randn(4, 24, 32)
    predictions = head(hidden)

    variance_1 = {feat: 1.0 for feat in NUMERICAL_FEATURES}
    variance_4 = {feat: 4.0 for feat in NUMERICAL_FEATURES}
    _, c1 = mtlm_loss(predictions, mtlm_batch, mtlm_batch["mask_positions"],
                      num_feature_variance=variance_1)
    _, c4 = mtlm_loss(predictions, mtlm_batch, mtlm_batch["mask_positions"],
                      num_feature_variance=variance_4)
    assert c4.num == pytest.approx(c1.num / 4.0, rel=1e-4)


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
    assert set(out["cat"].keys()) == set(CATEGORICAL_FEATURES)
    assert set(out["pay"].keys()) == set(PAY_STATUS_FEATURES)
    assert set(out["num"].keys()) == set(NUMERICAL_FEATURES)


def test_mtlm_model_encoder_state_dict_has_only_embedding_and_encoder():
    model = _build_mtlm_model()
    encoder_sd = model.encoder_state_dict()
    # every key must be `embedding.*` or `encoder.*` — exactly what
    # TabularTransformer.load_pretrained_encoder consumes
    for key in encoder_sd:
        assert key.startswith("embedding.") or key.startswith("encoder."), key
    full = model.state_dict()
    for key in full:
        if key.startswith("embedding.") or key.startswith("encoder."):
            assert key in encoder_sd, f"missing {key}"
        else:
            assert key not in encoder_sd, f"should not include {key}"


def test_tabular_transformer_loads_mtlm_encoder_state_dict(tmp_path):
    # Plan §8.5.5 contract e2e: pretrain MTLM → persist encoder_state_dict
    # → load into fresh TabularTransformer. encoder transfers, head stays.
    torch.manual_seed(0)
    cat_vocab_sizes = {"SEX": 2, "EDUCATION": 4, "MARRIAGE": 3}
    mtlm = _build_mtlm_model(d_model=32, n_heads=4, n_layers=2)

    mtlm_qweight = mtlm.encoder.blocks[0].attention.W_Q.weight.detach().clone()
    mtlm_emb_cls = mtlm.embedding.cls_token.detach().clone()

    encoder_sd = mtlm.encoder_state_dict()
    path = tmp_path / "encoder_pretrained.pt"
    torch.save(encoder_sd, path)

    torch.manual_seed(42)
    tabular = TabularTransformer(
        d_model=32, n_heads=4, n_layers=2,
        cat_vocab_sizes=cat_vocab_sizes,
    )
    head_w_before = tabular.classifier[0].weight.detach().clone()

    tabular.load_pretrained_encoder(path, strict=False)

    assert torch.allclose(
        tabular.encoder.blocks[0].attention.W_Q.weight, mtlm_qweight, atol=1e-6
    )
    assert torch.allclose(tabular.embedding.cls_token, mtlm_emb_cls, atol=1e-6)
    assert torch.allclose(tabular.classifier[0].weight, head_w_before, atol=1e-6)


def test_tabular_transformer_load_raises_on_missing_file(tmp_path):
    tabular = TabularTransformer(cat_vocab_sizes={"SEX": 2, "EDUCATION": 4, "MARRIAGE": 3})
    with pytest.raises(FileNotFoundError):
        tabular.load_pretrained_encoder(tmp_path / "does_not_exist.pt")
