"""
embedding.py — Feature embedding layer for the Credit Card Default Transformer.

Takes the tokenizer output (cat_indices, pay_state_ids, pay_severities, num_values)
and converts it into a sequence of 24 token embeddings of shape (B, 24, d_model).

Token order (fixed, determines positional index):
    Position  0:     [CLS] token                           (learnable, BERT-style)
    Positions 1-3:   SEX, EDUCATION, MARRIAGE              (categorical)
    Positions 4-9:   PAY_0, PAY_2, PAY_3, PAY_4,
                     PAY_5, PAY_6                          (PAY hybrid: state + severity)
    Positions 10-23: LIMIT_BAL, AGE, BILL_AMT1-6,
                     PAY_AMT1-6                            (numerical: identity + value)

Each token is assembled as:
    categorical → cat_embed(local_idx)      + pos_embed(position)
    PAY         → pay_state_embed(state_id)
                  + severity_proj(severity) + pos_embed(position)
    numerical   → num_feat_embed(feat_idx)
                  + value_proj(value)       + pos_embed(position)
"""

import json
import torch
import torch.nn as nn
from pathlib import Path

from tokenizer import (
    CATEGORICAL_FEATURES,
    PAY_STATUS_FEATURES,
    NUMERICAL_FEATURES,
)

# ──────────────────────────────────────────────────────────────────────────────
# Vocabulary sizes for categorical features — loaded from feature_metadata.json
# so they stay in sync if preprocessing changes.
# ──────────────────────────────────────────────────────────────────────────────
_metadata_path = Path(__file__).parent.parent / "data" / "processed" / "feature_metadata.json"
with open(_metadata_path) as _f:
    _meta = json.load(_f)
CAT_VOCAB_SIZES = {
    feat: info["n_categories"]
    for feat, info in _meta["categorical_features"].items()
}

# Fixed token order — position in this list + 1 = positional embedding index
# (index 0 is reserved for [CLS], BERT-style)
TOKEN_ORDER = CATEGORICAL_FEATURES + PAY_STATUS_FEATURES + NUMERICAL_FEATURES
# ["SEX","EDUCATION","MARRIAGE","PAY_0","PAY_2",...,"PAY_6",
#  "LIMIT_BAL","AGE","BILL_AMT1",...,"PAY_AMT6"]
# Total: 3 + 6 + 14 = 23 feature tokens  (+1 CLS = 24 output tokens)


class FeatureEmbedding(nn.Module):
    """
    Converts tokenizer output into a sequence of token embeddings.

    Input:  batch dict from CreditDefaultDataset / DataLoader
    Output: tensor of shape (B, 24, d_model)
    """

    def __init__(self, d_model: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # ── 1. Categorical embeddings (SEX, EDUCATION, MARRIAGE) ──────────────
        # One nn.Embedding per feature — each feature has its own table.
        self.cat_embeddings = nn.ModuleDict({
            feat: nn.Embedding(CAT_VOCAB_SIZES[feat], d_model)
            for feat in CATEGORICAL_FEATURES
        })

        # ── 2. PAY hybrid embeddings ──────────────────────────────────────────
        # State embedding: one of 4 semantic states (no_bill/paid/minimum/delinquent)
        self.pay_state_embedding = nn.Embedding(4, d_model)

        # Severity projection: scalar severity (0..1) → d_model vector.
        # Shared across all 6 PAY features — the state embedding already
        # identifies *which* PAY feature this is (via positional embedding).
        self.pay_severity_proj = nn.Linear(1, d_model, bias=True)

        # ── 3. Numerical embeddings ───────────────────────────────────────────
        # Feature identity embedding: tells the model "I am LIMIT_BAL" etc.
        self.num_feature_embedding = nn.Embedding(len(NUMERICAL_FEATURES), d_model)

        # Value projection: scaled scalar → d_model vector.
        # Shared across all 14 numerical features (identity comes from above).
        self.value_proj = nn.Linear(1, d_model, bias=True)

        # ── 4. Positional embedding ───────────────────────────────────────────
        # 24 positions: index 0 = [CLS], indices 1-23 = feature tokens (BERT-style).
        self.pos_embedding = nn.Embedding(len(TOKEN_ORDER) + 1, d_model)

        # [CLS] token — learnable parameter, takes positional embedding at index 0
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Pre-register feature position indices as a buffer (no grad needed).
        # Features occupy positions 1-23; position 0 belongs to [CLS].
        positions = torch.arange(1, len(TOKEN_ORDER) + 1)
        self.register_buffer("positions", positions)

        # ── 5. Output normalisation ───────────────────────────────────────────
        # LayerNorm brings categorical (std≈0.02) and projected numerical
        # (xavier, std≈0.2) tokens to the same scale before attention.
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialise all embedding tables and projection layers."""
        for emb in self.cat_embeddings.values():
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

        nn.init.normal_(self.pay_state_embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_normal_(self.pay_severity_proj.weight)
        nn.init.zeros_(self.pay_severity_proj.bias)

        nn.init.normal_(self.num_feature_embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_normal_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)

        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: dict from DataLoader with keys:
                cat_indices    — dict {feat: (B,) long tensor}
                pay_state_ids  — (B, 6) long tensor
                pay_severities — (B, 6) float tensor
                num_values     — (B, 14) float tensor

        Returns:
            tokens — (B, 24, d_model) float tensor
        """
        B = batch["num_values"].shape[0]
        device = batch["num_values"].device

        # Positional embeddings for all 23 feature positions at once (indices 1-23)
        all_pos_emb = self.pos_embedding(self.positions)  # (23, d_model)

        # ── Categorical tokens (sequence positions 1-3) ───────────────────────
        # Each feature has its own embedding table so we still loop, but only 3 iters.
        cat_tokens = torch.stack([
            self.cat_embeddings[feat](batch["cat_indices"][feat])  # (B, d_model)
            + all_pos_emb[pos]                                     # broadcast (d_model,)
            for pos, feat in enumerate(CATEGORICAL_FEATURES)
        ], dim=1)  # (B, 3, d_model)

        # ── PAY tokens (sequence positions 4-9) ──────────────────────────────
        # Vectorised: pass (B, 6) tensors directly — one kernel launch each.
        n_cat = len(CATEGORICAL_FEATURES)  # = 3
        state_emb = self.pay_state_embedding(batch["pay_state_ids"])          # (B, 6, d_model)
        sev_emb   = self.pay_severity_proj(batch["pay_severities"].unsqueeze(-1))  # (B, 6, d_model)
        pay_pos   = all_pos_emb[n_cat: n_cat + len(PAY_STATUS_FEATURES)]     # (6, d_model)
        pay_tokens = state_emb + sev_emb + pay_pos                            # (B, 6, d_model)

        # ── Numerical tokens (sequence positions 10-23) ───────────────────────
        # Vectorised: look up all 14 feature identities and project all values at once.
        n_cat_pay  = n_cat + len(PAY_STATUS_FEATURES)  # = 9
        feat_idx   = torch.arange(len(NUMERICAL_FEATURES), dtype=torch.long, device=device)
        feat_emb   = self.num_feature_embedding(feat_idx)                    # (14, d_model)
        value_emb  = self.value_proj(batch["num_values"].unsqueeze(-1))      # (B, 14, d_model)
        num_pos    = all_pos_emb[n_cat_pay:]                                 # (14, d_model)
        num_tokens = value_emb + feat_emb + num_pos                          # (B, 14, d_model)

        # ── Prepend [CLS] token at position 0 (BERT-style) ───────────────────
        cls_pos = self.pos_embedding(torch.zeros(1, dtype=torch.long, device=device))  # (1, d_model)
        cls = (self.cls_token + cls_pos).expand(B, -1, -1)                  # (B, 1, d_model)

        # Concatenate all tokens: [CLS] + categorical + PAY + numerical
        out = torch.cat([cls, cat_tokens, pay_tokens, num_tokens], dim=1)   # (B, 24, d_model)
        return self.dropout(self.norm(out))


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B = 4  # batch size

    dummy_batch = {
        "cat_indices": {
            "SEX":      torch.tensor([0, 1, 0, 1]),
            "EDUCATION":torch.tensor([0, 1, 2, 3]),
            "MARRIAGE": torch.tensor([0, 1, 2, 0]),
        },
        "pay_state_ids":  torch.zeros(B, 6, dtype=torch.long),   # all "no_bill"
        "pay_severities": torch.zeros(B, 6, dtype=torch.float),
        "num_values":     torch.randn(B, 14),
        "label":          torch.tensor([0.0, 1.0, 0.0, 1.0]),
    }

    model = FeatureEmbedding(d_model=64)
    out = model(dummy_batch)

    print(f"Output shape: {out.shape}")   # expect torch.Size([4, 24, 64])
    assert out.shape == (B, 24, 64), "Shape mismatch!"
    print("Smoke test passed.")
