"""Masked Tabular Language Modelling (MTLM) — Novelty N4.

Purpose
-------
Provide the BERT-style self-supervised pretraining objective for the
tabular transformer: per-feature prediction heads that reconstruct masked
tokens, plus a loss function that normalises each feature-type's error to
a comparable scale, plus an ``MTLMModel`` wrapper that shares the encoder
with the supervised ``TabularTransformer``.

Key public symbols
------------------
- ``MTLMHead`` — per-feature reconstruction heads: 3 categorical, 6 PAY
  status, 14 numerical.
- ``MTLMModel`` — embedding + encoder + head; state-dict keys align
  exactly with ``TabularTransformer`` so the encoder checkpoints drop in.
- ``mtlm_loss`` — entropy-normalised CE for classification heads +
  variance-normalised MSE for regression heads.
- ``MTLMLossComponents`` — immutable record of per-term losses for
  logging.

Design choice
-------------
Losses are normalised per-feature-type so the total gradient isn't
dominated by whichever feature happens to have the largest vocab or the
widest numeric range. CE divided by ``ln(n_classes)`` puts every
classification head on the same "0 = perfect, 1 = uniform posterior"
scale; numerical MSE divided by pre-computed per-feature variance does
the analogous thing for regression. This is an adaptation of the
Rubachev+ 2022 tabular pretraining recipe.

Critical invariant
------------------
``MTLMModel``'s submodule names — ``embedding``, ``encoder``, ``mtlm_head``
— are plain attribute assignment (not an ``nn.ModuleList`` or similar) so
that the encoder/embedding state-dict keys match ``TabularTransformer``'s
byte-for-byte. Renaming any of those three attributes silently breaks
pretrain -> finetune transfer.

Dependencies
------------
Takes a pre-built ``FeatureEmbedding`` + ``TransformerEncoder`` by
injection (``MTLMModel`` does not construct them) so the pretrain script
can share the same instances as the supervised model if it wants to.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..tokenization.embedding import FeatureEmbedding
from ..tokenization.tokenizer import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PAY_RAW_NUM_CLASSES,
    PAY_STATUS_FEATURES,
)
from .transformer import TransformerEncoder

logger = logging.getLogger(__name__)

# 24-token layout: [CLS, 3 cat, 6 PAY, 14 num]. Mirrors the ordering in
# FeatureEmbedding.forward so slicing here can't drift away from the
# tokeniser's canonical order. The _SLICE_* constants index the 24-slot
# tensor; the _FIRST_*_POS constants are derived starts kept for clarity
# even though the slices are the ones actually used.
_CLS_OFFSET = 1
_N_CAT = len(CATEGORICAL_FEATURES)  # 3
_N_PAY = len(PAY_STATUS_FEATURES)  # 6
_N_NUM = len(NUMERICAL_FEATURES)  # 14
_SLICE_CAT = slice(_CLS_OFFSET, _CLS_OFFSET + _N_CAT)
_SLICE_PAY = slice(_CLS_OFFSET + _N_CAT, _CLS_OFFSET + _N_CAT + _N_PAY)
_SLICE_NUM = slice(_CLS_OFFSET + _N_CAT + _N_PAY, _CLS_OFFSET + _N_CAT + _N_PAY + _N_NUM)
_FIRST_FEATURE_POS = _CLS_OFFSET  # 1
_FIRST_PAY_POS = _CLS_OFFSET + _N_CAT  # 4
_FIRST_NUM_POS = _CLS_OFFSET + _N_CAT + _N_PAY  # 10


class MTLMHead(nn.Module):
    """Per-feature reconstruction heads for masked-token pretraining.

    One independent linear head per feature:

    - 3 categorical heads (SEX, EDUCATION, MARRIAGE) -> logits over
      their own vocab.
    - 6 PAY-status heads -> logits over ``PAY_RAW_NUM_CLASSES`` (11,
      covering values -2..8 plus reserved codes).
    - 14 numerical heads -> single scalar prediction (regression).

    Position indices come from the tokeniser's TOKEN_ORDER; the head
    does not re-slice the sequence itself, it just uses the precomputed
    slice constants at module level.

    The dropout applied at the top of ``forward`` is shared across every
    head — applying it once to the hidden states is equivalent to
    applying per-head dropout and saves parameters.
    """

    def __init__(
        self,
        d_model: int,
        cat_vocab_sizes: Dict[str, int],
        numerical_features: Optional[List[str]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # The numerical-feature ordering is load-bearing: each head maps to
        # a specific slot, and shuffling would silently mis-align the
        # regression targets. Only allow the canonical order to prevent
        # accidental drift during experimentation.
        num_features = list(numerical_features or NUMERICAL_FEATURES)
        if num_features != list(NUMERICAL_FEATURES):
            raise ValueError(
                "MTLMHead expects numerical_features to equal the canonical "
                "tokenizer.NUMERICAL_FEATURES ordering; override this only if "
                "the tokenizer's token-ordering invariant has changed."
            )

        self.dropout = nn.Dropout(p=dropout)

        # ModuleDict so each head is addressable by feature name — makes
        # the loss computation readable and lets future per-feature
        # sweeps pick out a single head cleanly.
        self.cat_heads = nn.ModuleDict(
            {feat: nn.Linear(d_model, cat_vocab_sizes[feat]) for feat in CATEGORICAL_FEATURES}
        )
        self.pay_heads = nn.ModuleDict(
            {feat: nn.Linear(d_model, PAY_RAW_NUM_CLASSES) for feat in PAY_STATUS_FEATURES}
        )
        self.num_heads = nn.ModuleDict({feat: nn.Linear(d_model, 1) for feat in num_features})

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-normal on every head, zero biases.

        Heads feed directly into the loss (no subsequent nonlinearity),
        so variance-preserving init is the right call.
        """
        for head in (*self.cat_heads.values(), *self.pay_heads.values(), *self.num_heads.values()):
            nn.init.xavier_normal_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, hidden: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """Produce per-feature predictions from the encoder's hidden states.

        Parameters
        ----------
        hidden
            Encoder output of shape ``(B, 24, d_model)``. CLS at 0,
            feature tokens at 1..23.

        Returns
        -------
        dict
            ``{"cat": {feat: (B, vocab_feat)}, "pay": {feat: (B, 11)},
            "num": {feat: (B,)}}`` — logits for cat/PAY, scalar predictions
            for num.
        """
        h = self.dropout(hidden)

        # Slice the hidden states by the semantic-group layout once, so
        # each head sees only its designated token.
        h_cat = h[:, _SLICE_CAT, :]  # (B, 3, d)
        h_pay = h[:, _SLICE_PAY, :]  # (B, 6, d)
        h_num = h[:, _SLICE_NUM, :]  # (B, 14, d)

        cat_logits: Dict[str, torch.Tensor] = {}
        for i, feat in enumerate(CATEGORICAL_FEATURES):
            cat_logits[feat] = self.cat_heads[feat](h_cat[:, i, :])

        pay_logits: Dict[str, torch.Tensor] = {}
        for i, feat in enumerate(PAY_STATUS_FEATURES):
            pay_logits[feat] = self.pay_heads[feat](h_pay[:, i, :])

        num_preds: Dict[str, torch.Tensor] = {}
        for i, feat in enumerate(NUMERICAL_FEATURES):
            # squeeze(-1) so predictions are (B,) — matches target shape
            # and avoids an implicit broadcast in the MSE.
            num_preds[feat] = self.num_heads[feat](h_num[:, i, :]).squeeze(-1)

        return {"cat": cat_logits, "pay": pay_logits, "num": num_preds}


@dataclass(frozen=True)
class MTLMLossComponents:
    """Immutable snapshot of the per-term loss values for a single step.

    All fields are plain ``float`` / ``int`` (already ``.item()``'d) so
    they can be logged or serialised without pulling tensors into the
    epoch-level accumulators. ``n_masked`` is the total number of
    masked positions in the batch across all features, useful for
    normalising running averages.
    """

    total: float
    cat: float
    pay: float
    num: float
    n_masked: int


def mtlm_loss(
    predictions: Dict[str, Dict[str, torch.Tensor]],
    batch: Dict[str, torch.Tensor],
    mask_positions: torch.Tensor,
    *,
    num_feature_variance: Optional[Dict[str, float]] = None,
    w_cat: float = 1.0,
    w_pay: float = 1.0,
    w_num: float = 1.0,
) -> tuple[torch.Tensor, MTLMLossComponents]:
    """Masked-token reconstruction loss with per-type normalisation.

    Categorical and PAY losses are cross-entropy divided by
    ``ln(n_classes)`` — because a uniform posterior loses exactly
    ``ln(n)`` nats, that normalisation puts every classification head on
    the same "0 = perfect, 1 = uniform" scale regardless of vocab size.
    Numerical losses are MSE, optionally divided by each feature's
    marginal variance so a wide-range feature cannot dominate the total
    by simple unit-scale accident.

    Gracefully handles the "no masks in this slot this batch" case: that
    feature just drops out of the mean rather than contributing a NaN.
    If every slot in a whole type is mask-free, that type's mean is
    taken as zero.

    Parameters
    ----------
    predictions
        Output of ``MTLMHead.forward``.
    batch
        Must contain ``cat_indices`` (dict per feature), ``pay_raw``
        ``(B, 6)``, ``num_values`` ``(B, 14)``. Extra keys are ignored.
    mask_positions
        Boolean ``(B, 23)`` tensor over the PRE-CLS layout: slots 0-2
        are categorical, 3-8 are PAY, 9-22 are numerical. True means
        "this slot was masked in the input and should be predicted".
    num_feature_variance
        Optional map from numerical feature name to its training-set
        variance. Missing entries fall back to 1.0 (no scaling); variances
        below 1e-6 are clipped to 1e-6 to avoid division blow-ups on
        near-constant features.
    w_cat, w_pay, w_num
        Scalar weights on each sub-total. Default 1.0 each gives equal
        weight after per-feature normalisation.

    Returns
    -------
    tuple
        ``(total_loss_tensor, MTLMLossComponents)``. The tensor keeps its
        graph so ``.backward()`` works; the dataclass is the
        already-detached per-term record for logging.
    """
    # Pre-CLS slot indices. Matches the layout of mask_positions, which
    # is constructed in the 23-slot feature-only frame (no CLS).
    CAT_START = 0
    PAY_START = _N_CAT
    NUM_START = _N_CAT + _N_PAY

    # Pull device/dtype off the first prediction tensor so the zero
    # placeholder used below lives on the same device as the real losses.
    device = next(iter(predictions["cat"].values())).device
    dtype = next(iter(predictions["cat"].values())).dtype
    total_mask = int(mask_positions.sum().item())

    # --- Categorical CE, entropy-normalised ---------------------------------
    losses_cat: List[torch.Tensor] = []
    for i, feat in enumerate(CATEGORICAL_FEATURES):
        slot_mask = mask_positions[:, CAT_START + i]
        # Skip features with no masked rows this batch — including them
        # would mean CE on an empty tensor (NaN).
        if not slot_mask.any():
            continue
        logits = predictions["cat"][feat][slot_mask]
        target = batch["cat_indices"][feat][slot_mask].to(device)
        ce = F.cross_entropy(logits, target, reduction="mean")
        # Normalise by ln(n) — uniform posterior loses exactly ln(n)
        # nats, so ce/ln(n) is in roughly [0, 1] regardless of vocab.
        # max(2, ...) guards against a pathological unary head whose
        # ln(1)=0 would divide-by-zero.
        norm = math.log(max(2, logits.size(-1)))
        losses_cat.append(ce / norm)

    # --- PAY-status CE, entropy-normalised ----------------------------------
    losses_pay: List[torch.Tensor] = []
    for i, feat in enumerate(PAY_STATUS_FEATURES):
        slot_mask = mask_positions[:, PAY_START + i]
        if not slot_mask.any():
            continue
        logits = predictions["pay"][feat][slot_mask]
        # pay_raw is (B, 6) with the 6 PAY months on axis 1 — index by
        # `i` to get the column for this specific head.
        target = batch["pay_raw"][:, i][slot_mask].to(device)
        ce = F.cross_entropy(logits, target, reduction="mean")
        norm = math.log(PAY_RAW_NUM_CLASSES)
        losses_pay.append(ce / norm)

    # --- Numerical MSE, variance-normalised ---------------------------------
    losses_num: List[torch.Tensor] = []
    for i, feat in enumerate(NUMERICAL_FEATURES):
        slot_mask = mask_positions[:, NUM_START + i]
        if not slot_mask.any():
            continue
        pred = predictions["num"][feat][slot_mask]
        # Cast target to the prediction's dtype — batch tensors come out
        # of the collate_fn as float32, but autocast mixed-precision
        # paths could narrow pred to fp16; without the explicit cast
        # we'd get a dtype-mismatch error on the subtraction.
        target = batch["num_values"][:, i][slot_mask].to(device).to(dtype)
        se = (pred - target).pow(2).mean()
        # Divide by per-feature training variance so one wide-range
        # feature doesn't dominate the sum. Clip to 1e-6 so a constant
        # feature (variance ~ 0) doesn't blow up the denominator.
        var = 1.0
        if num_feature_variance is not None:
            var = max(1e-6, float(num_feature_variance.get(feat, 1.0)))
        losses_num.append(se / var)

    # Fallback zero for all-mask-free types. Keeping it on-device/dtype
    # means torch.stack([...]).mean() always returns a compatible tensor.
    zero = torch.zeros((), device=device, dtype=dtype)
    mean_cat = torch.stack(losses_cat).mean() if losses_cat else zero
    mean_pay = torch.stack(losses_pay).mean() if losses_pay else zero
    mean_num = torch.stack(losses_num).mean() if losses_num else zero

    total = w_cat * mean_cat + w_pay * mean_pay + w_num * mean_num

    return total, MTLMLossComponents(
        total=float(total.item()),
        cat=float(mean_cat.item()),
        pay=float(mean_pay.item()),
        num=float(mean_num.item()),
        n_masked=total_mask,
    )


class MTLMModel(nn.Module):
    """Embedding + encoder + ``MTLMHead`` wrapper for pretraining.

    The three submodules (``embedding``, ``encoder``, ``mtlm_head``) are
    assigned as plain attributes — not wrapped in an ``nn.ModuleList`` or
    ``nn.Sequential`` — so the state-dict keys read
    ``embedding.*`` / ``encoder.*`` / ``mtlm_head.*``. That matches the
    ``TabularTransformer`` naming exactly, which is what lets
    ``load_pretrained_encoder`` lift the pretrained weights into the
    supervised model without a key-translation step.

    The MTLM head travels with this wrapper and is NOT persisted into
    the post-pretraining encoder checkpoint — see ``encoder_state_dict``.
    """

    def __init__(
        self,
        embedding: FeatureEmbedding,
        encoder: TransformerEncoder,
        mtlm_head: MTLMHead,
    ):
        super().__init__()
        # Plain attribute assignment (PyTorch still registers them as
        # submodules because they're nn.Module instances). Do NOT switch
        # to nn.ModuleList etc. — that would change the state_dict keys
        # and break the pretrain -> finetune transfer invariant.
        self.embedding = embedding
        self.encoder = encoder
        self.mtlm_head = mtlm_head

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Run embedding -> encoder -> head and return per-feature predictions."""
        tokens = self.embedding(batch)
        # Encoder returns (x, attn_weights); we drop attention here —
        # MTLM pretraining doesn't need it and keeping it alive would
        # bloat memory on long pretraining runs.
        hidden, _attn = self.encoder(tokens)
        return self.mtlm_head(hidden)

    def encoder_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return only the ``embedding.*`` + ``encoder.*`` keys.

        The MTLM head is task-specific and doesn't travel with the
        encoder — the supervised model replaces it with a classification
        head. Filtering on the string prefix is deliberately simple: if
        the submodule-attribute names ever change, this filter must
        change with them (same invariant as the constructor).
        """
        full = self.state_dict()
        keep = {}
        for key, tensor in full.items():
            if key.startswith("embedding.") or key.startswith("encoder."):
                keep[key] = tensor
        return keep
