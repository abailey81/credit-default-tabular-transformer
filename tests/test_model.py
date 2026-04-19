"""TabularTransformer — shapes, pooling, param count, N3/N5 heads,
pretrained-encoder load, determinism."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from src.models.model import (  # noqa: E402
    FULL_SEQ_LEN,
    PAY_0_FEATURE_POSITION_23,
    PAY_0_OUTPUT_POSITION_24,
    TabularTransformer,
)
from src.tokenization.embedding import TOKEN_ORDER  # noqa: E402
from src.tokenization.tokenizer import PAY_RAW_NUM_CLASSES  # noqa: E402


@pytest.fixture()
def mixed_batch():
    # B=4 batch with non-zero PAY severities + pay_raw — exercises aux_pay0
    B = 4
    return {
        "cat_indices": {
            "SEX":       torch.tensor([0, 1, 0, 1]),
            "EDUCATION": torch.tensor([0, 1, 2, 3]),
            "MARRIAGE":  torch.tensor([0, 1, 2, 0]),
        },
        "pay_state_ids":  torch.tensor(
            [[0, 1, 2, 3, 3, 3], [3, 3, 2, 1, 0, 0],
             [2, 2, 2, 2, 2, 2], [0, 0, 0, 3, 3, 3]],
            dtype=torch.long,
        ),
        "pay_severities": torch.tensor(
            [[0.0, 0.0, 0.0, 0.25, 0.50, 0.75],
             [1.0, 0.875, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.375, 0.625, 1.0]],
            dtype=torch.float,
        ),
        "pay_raw": torch.tensor(
            # raw PAY shifted into [0, 10]: -2→0 … 8→10
            [[0, 1, 2, 3, 5, 8],
             [10, 9, 2, 1, 0, 0],
             [2, 2, 2, 2, 2, 2],
             [0, 0, 0, 5, 7, 10]],
            dtype=torch.long,
        ),
        "num_values":     torch.randn(B, 14),
        "label":          torch.tensor([0.0, 1.0, 0.0, 1.0]),
    }


def test_forward_returns_logit_of_shape_B(mixed_batch):
    torch.manual_seed(0)
    model = TabularTransformer()
    model.eval()
    out = model(mixed_batch)
    assert "logit" in out
    assert out["logit"].shape == (4,)
    assert torch.isfinite(out["logit"]).all()


def test_forward_without_attn_does_not_include_attn_key(mixed_batch):
    model = TabularTransformer()
    model.eval()
    out = model(mixed_batch)
    assert "attn_weights" not in out


def test_forward_with_attn_returns_per_layer_list(mixed_batch):
    model = TabularTransformer(n_layers=3)
    model.eval()
    out = model(mixed_batch, return_attn=True)
    assert "attn_weights" in out
    assert len(out["attn_weights"]) == 3
    for w in out["attn_weights"]:
        assert w.shape == (4, 4, FULL_SEQ_LEN, FULL_SEQ_LEN)


@pytest.mark.parametrize("pool", ["cls", "mean", "max"])
def test_pool_modes_each_produce_valid_logit(mixed_batch, pool):
    model = TabularTransformer(pool=pool)
    model.eval()
    out = model(mixed_batch)
    assert out["logit"].shape == (4,)
    assert torch.isfinite(out["logit"]).all()


def test_invalid_pool_rejected():
    with pytest.raises(ValueError, match="pool must be"):
        TabularTransformer(pool="not_a_pool")  # type: ignore[arg-type]


def test_mean_pool_excludes_cls_token(mixed_batch):
    # mean pool must skip CLS — else its small init anchors pooling to zero
    torch.manual_seed(0)
    model_mean = TabularTransformer(pool="mean")
    model_mean.eval()

    with torch.no_grad():
        model_mean.embedding.cls_token.zero_()
    out = model_mean(mixed_batch)
    assert out["logit"].shape == (4,)
    assert torch.isfinite(out["logit"]).all()


def test_param_count_at_plan_defaults_in_envelope():
    model = TabularTransformer()
    n = model.count_parameters()
    # Plan §6.9 targets ~28K; envelope allows natural drift
    assert 20_000 <= n <= 40_000, (
        f"TabularTransformer at plan defaults has {n:,} params — "
        f"outside the [20K, 40K] Plan §6.9 envelope"
    )


def test_param_count_scales_with_n_layers():
    small = TabularTransformer(n_layers=1).count_parameters()
    big = TabularTransformer(n_layers=4).count_parameters()
    assert big > small


def test_gradient_flows_through_every_parameter(mixed_batch):
    torch.manual_seed(0)
    model = TabularTransformer()
    model.train()
    out = model(mixed_batch)
    out["logit"].sum().backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert not no_grad, f"no grad on {no_grad}"
    for name, p in model.named_parameters():
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"


def test_temporal_decay_off_means_no_temporal_decay_submodule():
    model = TabularTransformer(temporal_decay_mode="off")
    assert model.encoder.temporal_decay is None


def test_temporal_decay_scalar_creates_module(mixed_batch):
    model = TabularTransformer(temporal_decay_mode="scalar")
    assert model.encoder.temporal_decay is not None
    assert model.encoder.temporal_decay.mode == "scalar"
    assert model.encoder.temporal_decay.alpha.shape == (1,)
    assert float(model.encoder.temporal_decay.alpha.item()) == 0.0


def test_temporal_decay_per_head_matches_n_heads(mixed_batch):
    model = TabularTransformer(n_heads=4, temporal_decay_mode="per_head")
    assert model.encoder.temporal_decay is not None
    assert model.encoder.temporal_decay.alpha.shape == (4,)


def test_temporal_decay_alpha_receives_gradient(mixed_batch):
    torch.manual_seed(0)
    model = TabularTransformer(temporal_decay_mode="scalar")
    decay = model.encoder.temporal_decay
    assert decay is not None
    with torch.no_grad():
        decay.alpha.fill_(0.3)
    model.train()
    out = model(mixed_batch)
    out["logit"].sum().backward()
    assert decay.alpha.grad is not None
    assert decay.alpha.grad.abs().item() > 0


def test_temporal_decay_uses_canonical_layout():
    model = TabularTransformer(temporal_decay_mode="scalar")
    decay = model.encoder.temporal_decay
    assert decay is not None
    # PAY_0 (pos 4) vs PAY_6 (pos 9) — same group, 5mo apart
    assert decay.neg_distance_masked[4, 9].item() == -5.0
    # PAY_0 vs BILL_AMT1 (pos 12) — different groups
    assert decay.neg_distance_masked[4, 12].item() == 0.0


def test_aux_pay0_head_created_when_enabled():
    model = TabularTransformer(aux_pay0=True)
    assert model.aux_pay0_head is not None
    assert model.aux_pay0 is True


def test_aux_pay0_absent_by_default():
    model = TabularTransformer()
    assert model.aux_pay0_head is None
    assert model.aux_pay0 is False


def test_aux_pay0_outputs_11_class_logits(mixed_batch):
    model = TabularTransformer(aux_pay0=True)
    model.eval()
    out = model(mixed_batch)
    assert "aux_pay0_logits" in out
    assert out["aux_pay0_logits"].shape == (4, PAY_RAW_NUM_CLASSES)


def test_aux_pay0_forces_use_mask_token_in_embedding():
    # aux_pay0=True must force use_mask_token on — no [MASK] = nothing to
    # replace PAY_0 with
    model = TabularTransformer(aux_pay0=True, use_mask_token=False)
    assert model.embedding.mask_token is not None


def test_aux_pay0_force_mask_includes_pay_0(mixed_batch):
    # perturb [MASK]; aux_pay0_logits must respond. otherwise PAY_0 isn't
    # being force-masked inside forward.
    torch.manual_seed(0)
    model = TabularTransformer(aux_pay0=True)
    model.eval()
    out1 = model(mixed_batch)
    with torch.no_grad():
        model.embedding.mask_token.normal_(mean=0.0, std=5.0)
    out2 = model(mixed_batch)
    diff = (out1["aux_pay0_logits"] - out2["aux_pay0_logits"]).abs().sum().item()
    assert diff > 1e-2, (
        f"aux_pay0_logits did not respond to mask_token perturbation "
        f"(diff={diff:.6f}) — PAY_0 may not be force-masked"
    )


def test_aux_pay0_loss_composable_with_primary_loss(mixed_batch):
    # joint backward (primary BCE + aux CE) must reach aux head AND shared
    # encoder — the N5 multi-task training contract
    import torch.nn.functional as F

    torch.manual_seed(0)
    model = TabularTransformer(aux_pay0=True)
    model.train()
    out = model(mixed_batch)

    primary = F.binary_cross_entropy_with_logits(out["logit"], mixed_batch["label"])
    aux = F.cross_entropy(out["aux_pay0_logits"], mixed_batch["pay_raw"][:, 0])
    joint = primary + 0.3 * aux
    joint.backward()

    assert model.aux_pay0_head is not None
    for _, p in model.aux_pay0_head.named_parameters():
        assert p.grad is not None and p.grad.abs().sum().item() > 0
    for _, p in model.classifier.named_parameters():
        assert p.grad is not None and p.grad.abs().sum().item() > 0
    enc_grads = [p.grad.abs().sum().item() for p in model.encoder.parameters()
                 if p.grad is not None]
    assert all(g >= 0 for g in enc_grads) and any(g > 0 for g in enc_grads)


def test_pay0_position_constants_align_with_token_order():
    # drift guard: aux-head slicing assumes PAY_0 is the 4th feature
    assert TOKEN_ORDER.index("PAY_0") == PAY_0_FEATURE_POSITION_23
    assert PAY_0_OUTPUT_POSITION_24 == PAY_0_FEATURE_POSITION_23 + 1


def test_load_pretrained_encoder_applies_matching_keys_and_skips_classifier(mixed_batch):
    # MTLM pretrain → fine-tune: encoder transfers, classifier stays fresh
    torch.manual_seed(0)
    pretrained = TabularTransformer()
    pretrained.eval()

    torch.manual_seed(42)
    fresh = TabularTransformer()
    fresh.eval()

    pre_head = pretrained.classifier[0].weight.clone()
    fresh_head_before = fresh.classifier[0].weight.clone()
    assert not torch.allclose(pre_head, fresh_head_before)

    pre_encoder_lin = pretrained.encoder.blocks[0].attention.W_Q.weight.clone()

    with tempfile.TemporaryDirectory() as tmp:
        from src.training.utils import build_checkpoint_metadata, save_checkpoint
        ckpt = Path(tmp) / "pretrained.pt"
        save_checkpoint(
            ckpt, pretrained,
            metadata=build_checkpoint_metadata(seed=0, extra={"role": "mtlm-pretrain"}),
        )

        fresh.load_pretrained_encoder(ckpt)

    fresh_encoder_lin = fresh.encoder.blocks[0].attention.W_Q.weight
    assert torch.allclose(fresh_encoder_lin, pre_encoder_lin, atol=1e-6), (
        "pretrained encoder weights were not applied"
    )


def test_two_fresh_models_same_seed_produce_identical_logits(mixed_batch):
    torch.manual_seed(7)
    m1 = TabularTransformer()
    torch.manual_seed(7)
    m2 = TabularTransformer()
    m1.eval()
    m2.eval()
    l1 = m1(mixed_batch)["logit"]
    l2 = m2(mixed_batch)["logit"]
    assert torch.allclose(l1, l2, atol=1e-6)


def test_cat_vocab_sizes_override_propagates(mixed_batch):
    custom = {"SEX": 3, "EDUCATION": 5, "MARRIAGE": 4}
    model = TabularTransformer(cat_vocab_sizes=custom)
    assert model.embedding.cat_embeddings["SEX"].num_embeddings == 3
    assert model.embedding.cat_embeddings["EDUCATION"].num_embeddings == 5
    assert model.embedding.cat_embeddings["MARRIAGE"].num_embeddings == 4


def test_train_mode_with_dropout_on_is_finite(mixed_batch):
    # full-feature compose (temporal pos + decay + aux head + dropout) must
    # stay finite — NaN-blowup guard
    torch.manual_seed(0)
    model = TabularTransformer(
        dropout=0.2,
        classification_dropout=0.2,
        use_temporal_pos=True,
        temporal_decay_mode="scalar",
        aux_pay0=True,
    )
    model.train()
    out = model(mixed_batch)
    assert torch.isfinite(out["logit"]).all()
    assert torch.isfinite(out["aux_pay0_logits"]).all()


def test_feature_group_bias_mode_off_means_no_submodule():
    model = TabularTransformer(feature_group_bias_mode="off")
    assert model.encoder.feature_group_bias is None


def test_feature_group_bias_mode_scalar_creates_module():
    model = TabularTransformer(feature_group_bias_mode="scalar")
    fgb = model.encoder.feature_group_bias
    assert fgb is not None
    assert fgb.mode == "scalar"
    assert fgb.bias_matrix is not None
    assert fgb.bias_matrix.shape == (5, 5)
    # zero-init: a fresh model recovers the plain encoder
    assert torch.equal(fgb.bias_matrix, torch.zeros(5, 5))


def test_feature_group_bias_mode_per_head_creates_module():
    model = TabularTransformer(n_heads=4, feature_group_bias_mode="per_head")
    fgb = model.encoder.feature_group_bias
    assert fgb is not None
    assert fgb.mode == "per_head"
    assert fgb.bias_matrix is not None
    assert fgb.bias_matrix.shape == (4, 5, 5)


def test_feature_group_bias_receives_gradient(mixed_batch):
    torch.manual_seed(0)
    model = TabularTransformer(feature_group_bias_mode="scalar")
    fgb = model.encoder.feature_group_bias
    assert fgb is not None
    # seed B≠0 so the mechanism is active from step 1 — B=0 + zero grad on
    # the B→B path would otherwise give a false negative
    with torch.no_grad():
        fgb.bias_matrix.fill_(0.1)
    model.train()
    out = model(mixed_batch)
    out["logit"].sum().backward()
    assert fgb.bias_matrix.grad is not None
    assert torch.isfinite(fgb.bias_matrix.grad).all()
    assert fgb.bias_matrix.grad.abs().sum().item() > 0


def test_both_novelty_biases_compose(mixed_batch):
    # N2 + N3 composed: both biases present, forward works, both learnable
    # matrices receive gradient under joint backprop
    torch.manual_seed(0)
    model = TabularTransformer(
        temporal_decay_mode="scalar",
        feature_group_bias_mode="scalar",
    )
    assert model.encoder.temporal_decay is not None
    assert model.encoder.feature_group_bias is not None

    with torch.no_grad():
        model.encoder.temporal_decay.alpha.fill_(0.3)
        model.encoder.feature_group_bias.bias_matrix.fill_(0.2)

    model.train()
    out = model(mixed_batch)
    assert out["logit"].shape == (4,)
    assert torch.isfinite(out["logit"]).all()

    out["logit"].sum().backward()
    assert model.encoder.temporal_decay.alpha.grad is not None
    assert model.encoder.temporal_decay.alpha.grad.abs().item() > 0
    assert model.encoder.feature_group_bias.bias_matrix.grad is not None
    assert model.encoder.feature_group_bias.bias_matrix.grad.abs().sum().item() > 0
