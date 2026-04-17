"""
mtlm.py — Masked Tabular Language Modelling head and loss (Plan §8.5 / Novelty N4).

This is the single most "language-model-like" component of the project and
the direct answer to the coursework PDF's explicit framing — a "small
transformer-based **language model**". Adapts the BERT masked-language-
modelling objective to heterogeneous tabular features so the encoder learns
feature dependencies from the data distribution *before* the supervised
default-prediction objective is applied.

Two classes + a pure function:

* :class:`MTLMHead` — per-feature prediction heads applied on top of the
  shared encoder hidden states. Three sub-heads:

  - **Categorical (3 features)** — one ``nn.Linear(d_model, n_categories)``
    per feature. Target: the integer category id from the tokenizer
    vocabulary. Loss: ``nn.CrossEntropyLoss``.
  - **PAY status (6 features)** — one ``nn.Linear(d_model, 11)`` per feature
    over the unified PAY vocabulary (``-2..8`` shifted into ``[0, 10]`` by
    the tokenizer). Loss: ``nn.CrossEntropyLoss`` on ``batch["pay_raw"]``.
  - **Numerical (14 features)** — one ``nn.Linear(d_model, 1)`` per feature,
    regressing on the *scaled* numerical value the tokenizer emits. Loss:
    ``nn.MSELoss``.

  Every head shares the **same encoder hidden state at its own token
  position**, so training naturally back-propagates feature-dependency
  gradients through the shared encoder.

* :class:`MTLMModel` — thin wrapper tying a :class:`FeatureEmbedding` +
  :class:`TransformerEncoder` + :class:`MTLMHead` together. Designed to be
  *checkpoint-compatible* with :class:`model.TabularTransformer`: the
  ``embedding.*`` and ``encoder.*`` sub-module names are identical, so
  :meth:`model.TabularTransformer.load_pretrained_encoder` transparently
  picks up only the matching keys and leaves the classification head at
  fresh init (Plan §8.5.5 two-stage fine-tuning).

* :func:`mtlm_loss` — computes the composite pretraining loss. Each
  feature-type sub-loss is normalised so numerical MSE and categorical
  CE contribute comparably regardless of their natural scales:

      L_cat = CE(pred, target) / ln(n_categories)       # entropy-normalised
      L_pay = CE(pred, target) / ln(11)                 # entropy-normalised
      L_num = MSE(pred, target) / feature_variance      # variance-normalised

  Then composed as ``w_cat · L̄_cat + w_pay · L̄_pay + w_num · L̄_num``
  with user-specified weights (default uniform = 1/3 each).

Novelty and plan references
---------------------------

* Rubachev, I., Alekberov, A., Gorishniy, Y. & Babenko, A. (2022).
  "Revisiting Pretraining Objectives for Tabular Deep Learning." arXiv:2207.03208.
* Plan §8.5 (Phase 6A — MTLM pretraining) / §1.6 (Novelty N4) /
  Ablation A15.
* BERT / Devlin et al. (2019) — the 80/10/10 masking split is handled by
  :class:`tokenizer.MTLMCollator`; this module consumes the mask metadata.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding import FeatureEmbedding
from tokenizer import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PAY_RAW_NUM_CLASSES,
    PAY_STATUS_FEATURES,
)
from transformer import TransformerEncoder

logger = logging.getLogger(__name__)

# Token position slicing in the full 24-token output sequence (CLS at index 0).
# Must mirror FeatureEmbedding.forward's layout — this is the canonical source
# of truth for where each feature's hidden state lives.
_CLS_OFFSET = 1
_N_CAT = len(CATEGORICAL_FEATURES)          # 3
_N_PAY = len(PAY_STATUS_FEATURES)           # 6
_N_NUM = len(NUMERICAL_FEATURES)            # 14
_SLICE_CAT = slice(_CLS_OFFSET, _CLS_OFFSET + _N_CAT)
_SLICE_PAY = slice(_CLS_OFFSET + _N_CAT, _CLS_OFFSET + _N_CAT + _N_PAY)
_SLICE_NUM = slice(_CLS_OFFSET + _N_CAT + _N_PAY, _CLS_OFFSET + _N_CAT + _N_PAY + _N_NUM)
_FIRST_FEATURE_POS = _CLS_OFFSET                         # 1
_FIRST_PAY_POS = _CLS_OFFSET + _N_CAT                    # 4
_FIRST_NUM_POS = _CLS_OFFSET + _N_CAT + _N_PAY           # 10


# ──────────────────────────────────────────────────────────────────────────────
# MTLMHead — per-feature prediction heads
# ──────────────────────────────────────────────────────────────────────────────


class MTLMHead(nn.Module):
    """
    Per-feature prediction heads for the Masked Tabular Language Modelling
    pretraining objective (Plan §8.5 / Novelty N4).

    The head consumes the ``(B, 24, d_model)`` hidden state produced by
    :class:`TransformerEncoder` and produces, per feature type, a target
    distribution at that feature's token position:

    * Categorical / PAY — per-feature logits over the feature's discrete
      vocabulary.
    * Numerical — a single scalar per feature (regression target is the
      same standardised value the tokenizer emitted).

    The ``forward`` method returns a dict:

        {
            "cat": {"SEX": (B, 2), "EDUCATION": (B, 4), "MARRIAGE": (B, 3)},
            "pay": {"PAY_0": (B, 11), ..., "PAY_6": (B, 11)},
            "num": {"LIMIT_BAL": (B,), ..., "PAY_AMT6": (B,)},
        }

    Parameters
    ----------
    d_model
        Encoder hidden dimension.
    cat_vocab_sizes
        Mapping ``{feature_name: vocabulary_size}`` for the three categoricals
        (SEX / EDUCATION / MARRIAGE). Same dict the :class:`FeatureEmbedding`
        constructor receives.
    numerical_features
        Ordered list of numerical feature names. Defaults to the canonical
        :data:`tokenizer.NUMERICAL_FEATURES`.
    dropout
        Dropout applied to the hidden state before each prediction head.
        Helps the model use the full token sequence rather than collapsing
        onto a single feature. Default 0.1.
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

        num_features = list(numerical_features or NUMERICAL_FEATURES)
        if num_features != list(NUMERICAL_FEATURES):
            raise ValueError(
                "MTLMHead expects numerical_features to equal the canonical "
                "tokenizer.NUMERICAL_FEATURES ordering; override this only if "
                "the tokenizer's token-ordering invariant has changed."
            )

        # Shared pre-head dropout — applied once before each per-feature head.
        self.dropout = nn.Dropout(p=dropout)

        # Per-feature heads.
        self.cat_heads = nn.ModuleDict(
            {
                feat: nn.Linear(d_model, cat_vocab_sizes[feat])
                for feat in CATEGORICAL_FEATURES
            }
        )
        self.pay_heads = nn.ModuleDict(
            {
                feat: nn.Linear(d_model, PAY_RAW_NUM_CLASSES)
                for feat in PAY_STATUS_FEATURES
            }
        )
        self.num_heads = nn.ModuleDict(
            {feat: nn.Linear(d_model, 1) for feat in num_features}
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-normal on linear weights; zeros on biases. Classification-
        style heads; numerical heads treat the bias as a per-feature mean
        prior that the model can easily shift with a single scalar."""
        for head in (*self.cat_heads.values(), *self.pay_heads.values(),
                     *self.num_heads.values()):
            nn.init.xavier_normal_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, hidden: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        hidden : (B, 24, d_model)
            Encoder output. Position 0 is ``[CLS]``; positions 1-23 are the
            feature tokens in :data:`embedding.TOKEN_ORDER`.

        Returns
        -------
        Dict with three sub-dicts (``cat``, ``pay``, ``num``), each mapping
        feature-name → logits tensor.
        """
        h = self.dropout(hidden)

        # Slice each feature type. Using the precomputed slice constants
        # keeps this drift-safe: any change to TOKEN_ORDER would immediately
        # land a different shape and fail the pytest check on predictions.
        h_cat = h[:, _SLICE_CAT, :]          # (B, 3, d)
        h_pay = h[:, _SLICE_PAY, :]          # (B, 6, d)
        h_num = h[:, _SLICE_NUM, :]          # (B, 14, d)

        cat_logits: Dict[str, torch.Tensor] = {}
        for i, feat in enumerate(CATEGORICAL_FEATURES):
            cat_logits[feat] = self.cat_heads[feat](h_cat[:, i, :])

        pay_logits: Dict[str, torch.Tensor] = {}
        for i, feat in enumerate(PAY_STATUS_FEATURES):
            pay_logits[feat] = self.pay_heads[feat](h_pay[:, i, :])

        num_preds: Dict[str, torch.Tensor] = {}
        for i, feat in enumerate(NUMERICAL_FEATURES):
            num_preds[feat] = self.num_heads[feat](h_num[:, i, :]).squeeze(-1)

        return {"cat": cat_logits, "pay": pay_logits, "num": num_preds}


# ──────────────────────────────────────────────────────────────────────────────
# MTLM loss — composite per-feature-type loss with entropy/variance weighting
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MTLMLossComponents:
    """Bookkeeping container so the training loop can log every component
    individually. Every value is a Python float (already reduced across
    the batch)."""

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
    """
    Composite MTLM loss — entropy-normalised CE on cat/PAY, variance-normalised
    MSE on numerical. Loss is computed **only on masked positions**, following
    BERT.

    Parameters
    ----------
    predictions
        Output of :meth:`MTLMHead.forward`.
    batch
        The original training batch (before masking). Must contain the
        tokenizer-produced ground-truth tensors:

        * ``cat_indices[feat]`` : ``(B,)`` int64 — per-feature category id
        * ``pay_raw`` : ``(B, 6)`` int64 in ``[0, 10]``
        * ``num_values`` : ``(B, 14)`` float — *scaled* numerical values

    mask_positions
        ``(B, 23)`` bool tensor from :class:`tokenizer.MTLMCollator`. True
        at positions selected for the prediction loss. Slots 0-2 are the
        three categoricals, 3-8 are the PAY features, 9-22 are the
        numericals (pre-CLS positions).
    num_feature_variance
        Optional ``{feature_name: variance}`` dict — *on the training set*.
        When provided, each numerical MSE is divided by the feature's
        variance so each numerical feature contributes comparable signal
        regardless of scale. Defaults to 1.0 for every feature (no
        normalisation).
    w_cat, w_pay, w_num
        Relative weights on the three feature-type components. Default
        ``(1, 1, 1)`` — a reasonable starting point that the plan §8.5.4
        GradNorm-style variant could later replace.

    Returns
    -------
    ``(loss, components)`` — a scalar tensor suitable for ``.backward()``
    plus a :class:`MTLMLossComponents` record for logging.

    Notes
    -----
    * Categorical CE is divided by ``ln(n_categories)`` so a 2-class CE and
      an 11-class CE contribute comparably. This is the standard
      information-theoretic normalisation; a uniform posterior loses exactly
      ``ln(n)`` nats regardless of ``n``.
    * Numerical MSE is divided by feature variance so no single high-variance
      numerical (e.g., BILL_AMT) dominates the loss.
    """
    # Category-index slices inside mask_positions: 0-2 cat, 3-8 pay, 9-22 num.
    # These mirror the FeatureEmbedding pre-CLS layout (the collator acts on
    # the 23-feature block, not the 24-token post-CLS block).
    CAT_START, CAT_STOP = 0, _N_CAT
    PAY_START, PAY_STOP = _N_CAT, _N_CAT + _N_PAY
    NUM_START, NUM_STOP = _N_CAT + _N_PAY, _N_CAT + _N_PAY + _N_NUM

    device = next(iter(predictions["cat"].values())).device
    dtype = next(iter(predictions["cat"].values())).dtype
    total_mask = int(mask_positions.sum().item())

    losses_cat: List[torch.Tensor] = []
    for i, feat in enumerate(CATEGORICAL_FEATURES):
        slot_mask = mask_positions[:, CAT_START + i]          # (B,)
        if not slot_mask.any():
            continue
        logits = predictions["cat"][feat][slot_mask]          # (M, n_cat)
        target = batch["cat_indices"][feat][slot_mask].to(device)
        ce = F.cross_entropy(logits, target, reduction="mean")
        norm = math.log(max(2, logits.size(-1)))              # entropy of uniform
        losses_cat.append(ce / norm)

    losses_pay: List[torch.Tensor] = []
    for i, feat in enumerate(PAY_STATUS_FEATURES):
        slot_mask = mask_positions[:, PAY_START + i]          # (B,)
        if not slot_mask.any():
            continue
        logits = predictions["pay"][feat][slot_mask]          # (M, 11)
        target = batch["pay_raw"][:, i][slot_mask].to(device)
        ce = F.cross_entropy(logits, target, reduction="mean")
        norm = math.log(PAY_RAW_NUM_CLASSES)
        losses_pay.append(ce / norm)

    losses_num: List[torch.Tensor] = []
    for i, feat in enumerate(NUMERICAL_FEATURES):
        slot_mask = mask_positions[:, NUM_START + i]          # (B,)
        if not slot_mask.any():
            continue
        pred = predictions["num"][feat][slot_mask]            # (M,)
        target = batch["num_values"][:, i][slot_mask].to(device).to(dtype)
        se = (pred - target).pow(2).mean()
        var = 1.0
        if num_feature_variance is not None:
            var = max(1e-6, float(num_feature_variance.get(feat, 1.0)))
        losses_num.append(se / var)

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


# ──────────────────────────────────────────────────────────────────────────────
# MTLMModel — embedding + encoder + MTLMHead, checkpoint-compatible with
# TabularTransformer (shares the embedding.* / encoder.* state-dict prefixes).
# ──────────────────────────────────────────────────────────────────────────────


class MTLMModel(nn.Module):
    """
    Thin wrapper tying a shared :class:`FeatureEmbedding` +
    :class:`TransformerEncoder` + :class:`MTLMHead`.

    Designed so the pretrained state dict has the same ``embedding.*`` /
    ``encoder.*`` prefixes as :class:`model.TabularTransformer`, meaning
    :meth:`model.TabularTransformer.load_pretrained_encoder` (with
    ``strict=False``) transparently picks up the pretrained encoder weights
    and leaves the supervised classification head at fresh init. Plan §8.5.5.
    """

    def __init__(
        self,
        embedding: FeatureEmbedding,
        encoder: TransformerEncoder,
        mtlm_head: MTLMHead,
    ):
        super().__init__()
        # NOTE: assigning a Module attribute triggers automatic registration as
        # a submodule — we do NOT wrap in ModuleList, because we want the
        # canonical ``embedding`` / ``encoder`` / ``mtlm_head`` state-dict
        # names for checkpoint interoperability with TabularTransformer.
        self.embedding = embedding
        self.encoder = encoder
        self.mtlm_head = mtlm_head

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        tokens = self.embedding(batch)            # (B, 24, d)
        hidden, _attn = self.encoder(tokens)      # (B, 24, d)
        return self.mtlm_head(hidden)

    def encoder_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Return only the ``embedding.*`` + ``encoder.*`` state-dict entries.

        Useful when persisting a pretrained checkpoint for downstream
        supervised fine-tuning — the MTLM-head weights are task-specific
        and don't need to travel with the encoder. Callers can still save
        the full model via :func:`utils.save_checkpoint` if they want the
        prediction heads for inspection / continued pretraining.
        """
        full = self.state_dict()
        keep = {}
        for key, tensor in full.items():
            if key.startswith("embedding.") or key.startswith("encoder."):
                keep[key] = tensor
        return keep
