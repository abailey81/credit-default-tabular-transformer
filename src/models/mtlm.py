"""MTLM: per-feature heads + loss + wrapper. BERT-style MLM on tabular
tokens — entropy-normalised CE for cat/PAY, variance-normalised MSE for
numericals. MTLMModel shares embedding.*/encoder.* state-dict keys with
TabularTransformer so the encoder is drop-in for fine-tuning."""

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

# 24-token layout: [CLS, 3 cat, 6 PAY, 14 num]. Mirrors FeatureEmbedding.forward.
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


class MTLMHead(nn.Module):
    """Per-feature heads: 3 cat + 6 PAY + 14 num. Positions come from TOKEN_ORDER."""

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

        self.dropout = nn.Dropout(p=dropout)

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
        for head in (*self.cat_heads.values(), *self.pay_heads.values(),
                     *self.num_heads.values()):
            nn.init.xavier_normal_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, hidden: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """(B, 24, d) → {cat, pay, num}. CLS at 0, feature tokens at 1..23."""
        h = self.dropout(hidden)

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


@dataclass(frozen=True)
class MTLMLossComponents:
    """per-term record (already .item()'d) so the loop can log each piece."""

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
    """Masked reconstruction loss.

    cat/PAY CE divided by ln(n_classes) — a uniform posterior loses exactly
    ln(n) nats, so this puts every feature on the same scale regardless of
    vocab size. num MSE optionally divided by per-feature variance so a
    wide-range feature doesn't swamp everything else.

    mask_positions is (B, 23), pre-CLS: slots 0-2 cat, 3-8 PAY, 9-22 num.
    batch needs cat_indices, pay_raw, num_values.
    """
    CAT_START, CAT_STOP = 0, _N_CAT
    PAY_START, PAY_STOP = _N_CAT, _N_CAT + _N_PAY
    NUM_START, NUM_STOP = _N_CAT + _N_PAY, _N_CAT + _N_PAY + _N_NUM

    device = next(iter(predictions["cat"].values())).device
    dtype = next(iter(predictions["cat"].values())).dtype
    total_mask = int(mask_positions.sum().item())

    losses_cat: List[torch.Tensor] = []
    for i, feat in enumerate(CATEGORICAL_FEATURES):
        slot_mask = mask_positions[:, CAT_START + i]
        if not slot_mask.any():
            continue
        logits = predictions["cat"][feat][slot_mask]
        target = batch["cat_indices"][feat][slot_mask].to(device)
        ce = F.cross_entropy(logits, target, reduction="mean")
        norm = math.log(max(2, logits.size(-1)))
        losses_cat.append(ce / norm)

    losses_pay: List[torch.Tensor] = []
    for i, feat in enumerate(PAY_STATUS_FEATURES):
        slot_mask = mask_positions[:, PAY_START + i]
        if not slot_mask.any():
            continue
        logits = predictions["pay"][feat][slot_mask]
        target = batch["pay_raw"][:, i][slot_mask].to(device)
        ce = F.cross_entropy(logits, target, reduction="mean")
        norm = math.log(PAY_RAW_NUM_CLASSES)
        losses_pay.append(ce / norm)

    losses_num: List[torch.Tensor] = []
    for i, feat in enumerate(NUMERICAL_FEATURES):
        slot_mask = mask_positions[:, NUM_START + i]
        if not slot_mask.any():
            continue
        pred = predictions["num"][feat][slot_mask]
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


class MTLMModel(nn.Module):
    """embedding + encoder + MTLMHead. Shares embedding.*/encoder.* state-dict
    keys with TabularTransformer so the encoder is drop-in for fine-tuning."""

    def __init__(
        self,
        embedding: FeatureEmbedding,
        encoder: TransformerEncoder,
        mtlm_head: MTLMHead,
    ):
        super().__init__()
        # plain attribute assignment (not ModuleList) → state-dict keys match
        # TabularTransformer exactly
        self.embedding = embedding
        self.encoder = encoder
        self.mtlm_head = mtlm_head

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        tokens = self.embedding(batch)
        hidden, _attn = self.encoder(tokens)
        return self.mtlm_head(hidden)

    def encoder_state_dict(self) -> Dict[str, torch.Tensor]:
        """only embedding.* + encoder.* keys — the MTLM head is task-specific
        and doesn't travel with the encoder."""
        full = self.state_dict()
        keep = {}
        for key, tensor in full.items():
            if key.startswith("embedding.") or key.startswith("encoder."):
                keep[key] = tensor
        return keep
