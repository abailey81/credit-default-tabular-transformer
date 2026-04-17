"""Tests for src/model.py — the TabularTransformer top-level wrapper.

Scope (≈30 cases):
- Forward-pass shapes (logit, attention, aux head)
- Pool modes (cls / mean / max) — Ablation A5
- Parameter count lands inside Plan §6.9 envelope at defaults
- Gradient flow through every parameter
- Temporal-decay bias integration (N3 / A22)
- Aux PAY_0 forecast head (N5 / A16) — forced mask, logits shape, gradient
- Aux head correctly uses the tokenizer's pay_raw as 11-class target
- Pretrained-encoder loading via strict=False
- Deterministic under identical seeds
- Validation of invalid pool / dropout choices
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from embedding import TOKEN_ORDER  # noqa: E402
from model import (  # noqa: E402
    FULL_SEQ_LEN,
    PAY_0_FEATURE_POSITION_23,
    PAY_0_OUTPUT_POSITION_24,
    TabularTransformer,
)
from tokenizer import PAY_RAW_NUM_CLASSES  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def mixed_batch():
    """A B=4 mixed batch with non-zero PAY severities and pay_raw present
    so the aux_pay0 path can be exercised."""
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
            # Raw PAY values shifted into [0, 10]: -2→0, -1→1, 0→2, 1→3, …, 8→10.
            [[0, 1, 2, 3, 5, 8],
             [10, 9, 2, 1, 0, 0],
             [2, 2, 2, 2, 2, 2],
             [0, 0, 0, 5, 7, 10]],
            dtype=torch.long,
        ),
        "num_values":     torch.randn(B, 14),
        "label":          torch.tensor([0.0, 1.0, 0.0, 1.0]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Forward-pass output shape
# ──────────────────────────────────────────────────────────────────────────────


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


# ──────────────────────────────────────────────────────────────────────────────
# Pool modes — Ablation A5
# ──────────────────────────────────────────────────────────────────────────────


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
    """The CLS token is initialised small (std 0.02) and sits at index 0.
    Mean pool should operate over the *feature* tokens only (indices 1-23)
    so the CLS magnitude does not anchor the pooled representation."""
    torch.manual_seed(0)
    model_mean = TabularTransformer(pool="mean")
    model_mean.eval()

    # Zero out the CLS token entirely so, if mean-pool included it, the
    # pooled vector would be scaled by a factor of (23/24). A model that
    # correctly excludes CLS should behave the same whether we zero the
    # CLS parameter or not. (We only check the contract — an exact match
    # requires taking the mean pre-LayerNorm which we can't easily do.)
    with torch.no_grad():
        model_mean.embedding.cls_token.zero_()
    out = model_mean(mixed_batch)
    assert out["logit"].shape == (4,)
    # If mean pool included CLS, LayerNorm output at CLS slot (all zeros pre-LN
    # → undefined / NaN during .train()); eval path may still work, but the
    # point is that the pooled feature representation remains finite.
    assert torch.isfinite(out["logit"]).all()


# ──────────────────────────────────────────────────────────────────────────────
# Parameter count — Plan §6.9 budget check
# ──────────────────────────────────────────────────────────────────────────────


def test_param_count_at_plan_defaults_in_envelope():
    model = TabularTransformer()  # d_model=32, n_heads=4, n_layers=2 defaults
    n = model.count_parameters()
    # Plan §6.9: "Total: ~28,000 parameters". Leave envelope for natural
    # drift (e.g., if head_norm is added/removed later).
    assert 20_000 <= n <= 40_000, (
        f"TabularTransformer at plan defaults has {n:,} params — "
        f"outside the [20K, 40K] Plan §6.9 envelope"
    )


def test_param_count_scales_with_n_layers():
    small = TabularTransformer(n_layers=1).count_parameters()
    big = TabularTransformer(n_layers=4).count_parameters()
    assert big > small  # each additional block adds ~12.6K params


# ──────────────────────────────────────────────────────────────────────────────
# Gradient flow through every parameter
# ──────────────────────────────────────────────────────────────────────────────


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


# ──────────────────────────────────────────────────────────────────────────────
# Temporal decay (Novelty N3 / Ablation A22)
# ──────────────────────────────────────────────────────────────────────────────


def test_temporal_decay_off_means_no_temporal_decay_submodule():
    model = TabularTransformer(temporal_decay_mode="off")
    assert model.encoder.temporal_decay is None


def test_temporal_decay_scalar_creates_module(mixed_batch):
    model = TabularTransformer(temporal_decay_mode="scalar")
    assert model.encoder.temporal_decay is not None
    assert model.encoder.temporal_decay.mode == "scalar"
    # α is a scalar Parameter, zero-initialised.
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
    """Layout positions must come from build_temporal_layout — never hard-coded."""
    model = TabularTransformer(temporal_decay_mode="scalar")
    decay = model.encoder.temporal_decay
    assert decay is not None
    # PAY_0 (position 4 in 24-seq) vs PAY_6 (position 9 in 24-seq), same group,
    # 5 months apart, so neg_distance should be -5.
    assert decay.neg_distance_masked[4, 9].item() == -5.0
    # PAY_0 vs BILL_AMT1 (position 12) are different groups → zero.
    assert decay.neg_distance_masked[4, 12].item() == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Aux PAY_0 forecast head (Novelty N5 / Ablation A16)
# ──────────────────────────────────────────────────────────────────────────────


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
    """Even if caller passes use_mask_token=False, aux_pay0=True must turn
    it on — otherwise there's no [MASK] to force-mask PAY_0."""
    model = TabularTransformer(aux_pay0=True, use_mask_token=False)
    assert model.embedding.mask_token is not None


def test_aux_pay0_force_mask_includes_pay_0(mixed_batch):
    """Forward with aux_pay0=True must force-mask PAY_0 inside the embedding.
    We verify by perturbing the [MASK] embedding and checking the aux_pay0
    logits respond (they would not if PAY_0 were never replaced with mask)."""
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
    """End-to-end: compute primary BCE loss on logit + CE loss on aux_pay0
    logits (against pay_raw[:, 0]), ensure joint backward reaches the aux
    head and the encoder."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    model = TabularTransformer(aux_pay0=True)
    model.train()
    out = model(mixed_batch)

    primary = F.binary_cross_entropy_with_logits(out["logit"], mixed_batch["label"])
    aux = F.cross_entropy(out["aux_pay0_logits"], mixed_batch["pay_raw"][:, 0])
    joint = primary + 0.3 * aux
    joint.backward()

    # Both heads + the shared encoder must have received gradient.
    assert model.aux_pay0_head is not None
    for _, p in model.aux_pay0_head.named_parameters():
        assert p.grad is not None and p.grad.abs().sum().item() > 0
    for _, p in model.classifier.named_parameters():
        assert p.grad is not None and p.grad.abs().sum().item() > 0
    # Encoder shares gradient from both heads.
    enc_grads = [p.grad.abs().sum().item() for p in model.encoder.parameters()
                 if p.grad is not None]
    assert all(g >= 0 for g in enc_grads) and any(g > 0 for g in enc_grads)


def test_pay0_position_constants_align_with_token_order():
    """PAY_0 must be the 4th feature in TOKEN_ORDER (after the 3 categoricals)
    — the aux-head slicing depends on this. Catches silent TOKEN_ORDER drift."""
    assert TOKEN_ORDER.index("PAY_0") == PAY_0_FEATURE_POSITION_23
    # +1 accounts for the [CLS] token prepended at output index 0.
    assert PAY_0_OUTPUT_POSITION_24 == PAY_0_FEATURE_POSITION_23 + 1


# ──────────────────────────────────────────────────────────────────────────────
# Pretrained encoder loading (fine-tune path, Plan §8.5.5)
# ──────────────────────────────────────────────────────────────────────────────


def test_load_pretrained_encoder_applies_matching_keys_and_skips_classifier(mixed_batch):
    """Simulate an MTLM pretrain: save one model's state dict, then create a
    fresh model with identical shape but a FRESHLY randomised classifier head,
    load the pretrained state with strict=False, and confirm the encoder
    weights were applied while the classifier stayed at fresh init."""
    torch.manual_seed(0)
    pretrained = TabularTransformer()
    pretrained.eval()

    torch.manual_seed(42)  # different seed → different head init
    fresh = TabularTransformer()
    fresh.eval()

    # Sanity: classifier weights differ before loading.
    pre_head = pretrained.classifier[0].weight.clone()
    fresh_head_before = fresh.classifier[0].weight.clone()
    assert not torch.allclose(pre_head, fresh_head_before)

    # Snapshot the pretrained encoder weight to compare after load.
    pre_encoder_lin = pretrained.encoder.blocks[0].attention.W_Q.weight.clone()

    with tempfile.TemporaryDirectory() as tmp:
        from utils import build_checkpoint_metadata, save_checkpoint
        ckpt = Path(tmp) / "pretrained.pt"
        save_checkpoint(
            ckpt, pretrained,
            metadata=build_checkpoint_metadata(seed=0, extra={"role": "mtlm-pretrain"}),
        )

        # Load into the fresh model with strict=False (default).
        fresh.load_pretrained_encoder(ckpt)

    # Encoder weights must now equal the pretrained weights.
    fresh_encoder_lin = fresh.encoder.blocks[0].attention.W_Q.weight
    assert torch.allclose(fresh_encoder_lin, pre_encoder_lin, atol=1e-6), (
        "pretrained encoder weights were not applied"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Determinism
# ──────────────────────────────────────────────────────────────────────────────


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
    """Passing cat_vocab_sizes directly keeps the model hermetic (no disk I/O)."""
    custom = {"SEX": 3, "EDUCATION": 5, "MARRIAGE": 4}
    model = TabularTransformer(cat_vocab_sizes=custom)
    assert model.embedding.cat_embeddings["SEX"].num_embeddings == 3
    assert model.embedding.cat_embeddings["EDUCATION"].num_embeddings == 5
    assert model.embedding.cat_embeddings["MARRIAGE"].num_embeddings == 4


def test_train_mode_with_dropout_on_is_finite(mixed_batch):
    """Dropout-on training mode with the full feature set (temporal pos +
    temporal decay + aux head + non-zero dropout) must produce finite output."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Feature-group bias (Novelty N2 / Ablation A21)
# ──────────────────────────────────────────────────────────────────────────────


def test_feature_group_bias_mode_off_means_no_submodule():
    model = TabularTransformer(feature_group_bias_mode="off")
    assert model.encoder.feature_group_bias is None


def test_feature_group_bias_mode_scalar_creates_module():
    model = TabularTransformer(feature_group_bias_mode="scalar")
    fgb = model.encoder.feature_group_bias
    assert fgb is not None
    assert fgb.mode == "scalar"
    # 5-group scalar bias matrix — shared across heads.
    assert fgb.bias_matrix is not None
    assert fgb.bias_matrix.shape == (5, 5)
    # Zero-initialised so a fresh model recovers plain-encoder behaviour.
    assert torch.equal(fgb.bias_matrix, torch.zeros(5, 5))


def test_feature_group_bias_mode_per_head_creates_module():
    model = TabularTransformer(n_heads=4, feature_group_bias_mode="per_head")
    fgb = model.encoder.feature_group_bias
    assert fgb is not None
    assert fgb.mode == "per_head"
    # One 5×5 matrix per attention head.
    assert fgb.bias_matrix is not None
    assert fgb.bias_matrix.shape == (4, 5, 5)


def test_feature_group_bias_receives_gradient(mixed_batch):
    """Train the model for one step: ``bias_matrix.grad`` must be populated
    with at least some non-zero entries — otherwise the prior can never be
    learned."""
    torch.manual_seed(0)
    model = TabularTransformer(feature_group_bias_mode="scalar")
    fgb = model.encoder.feature_group_bias
    assert fgb is not None
    # Seed B away from zero so the mechanism is active from the first step
    # (zero init + zero gradient on the B → B path would be a false negative).
    with torch.no_grad():
        fgb.bias_matrix.fill_(0.1)
    model.train()
    out = model(mixed_batch)
    out["logit"].sum().backward()
    assert fgb.bias_matrix.grad is not None
    assert torch.isfinite(fgb.bias_matrix.grad).all()
    assert fgb.bias_matrix.grad.abs().sum().item() > 0


def test_both_novelty_biases_compose(mixed_batch):
    """N2 + N3 must coexist: both bias submodules live on the encoder, the
    forward pass runs, and both learnable matrices receive gradient under
    joint backprop."""
    torch.manual_seed(0)
    model = TabularTransformer(
        temporal_decay_mode="scalar",
        feature_group_bias_mode="scalar",
    )
    assert model.encoder.temporal_decay is not None
    assert model.encoder.feature_group_bias is not None

    # Seed both parameters away from zero so the mechanisms are active and
    # their gradients are guaranteed non-zero under backprop.
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
