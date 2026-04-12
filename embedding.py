"""
embedding.py — Feature embedding layer for the Credit Card Default Transformer.

Takes the tokenizer output (cat_indices, pay_state_ids, pay_severities, num_values)
and converts it into a sequence of 23 token embeddings of shape (B, 23, d_model).

Token order (fixed, determines positional index):
    Positions 0-2:   SEX, EDUCATION, MARRIAGE          (categorical)
    Positions 3-8:   PAY_0, PAY_2, PAY_3, PAY_4,
                     PAY_5, PAY_6                       (PAY hybrid: state + severity)
    Positions 9-22:  LIMIT_BAL, AGE, BILL_AMT1-6,
                     PAY_AMT1-6                         (numerical: identity + value)

Each token is assembled as:
    categorical → cat_embed(local_idx)      + pos_embed(position)
    PAY         → pay_state_embed(state_id)
                  + severity_proj(severity) + pos_embed(position)
    numerical   → num_feat_embed(feat_idx)
                  + value_proj(value)       + pos_embed(position)
"""

import torch
import torch.nn as nn

from tokenizer import (
    CATEGORICAL_FEATURES,
    PAY_STATUS_FEATURES,
    NUMERICAL_FEATURES,
)

# ──────────────────────────────────────────────────────────────────────────────
# Vocabulary sizes for categorical features (after cleaning in Phase 1)
# ──────────────────────────────────────────────────────────────────────────────
CAT_VOCAB_SIZES = {
    "SEX":      2,   # 1=Male, 2=Female
    "EDUCATION": 4,  # 1=Grad school, 2=Uni, 3=High school, 4=Others
    "MARRIAGE":  3,  # 1=Married, 2=Single, 3=Others
}

# Fixed token order — position in this list = positional embedding index
TOKEN_ORDER = CATEGORICAL_FEATURES + PAY_STATUS_FEATURES + NUMERICAL_FEATURES
# ["SEX","EDUCATION","MARRIAGE","PAY_0","PAY_2",...,"PAY_6",
#  "LIMIT_BAL","AGE","BILL_AMT1",...,"PAY_AMT6"]
# Total: 3 + 6 + 14 = 23 tokens


class FeatureEmbedding(nn.Module):
    """
    Converts tokenizer output into a sequence of token embeddings.

    Input:  batch dict from CreditDefaultDataset / DataLoader
    Output: tensor of shape (B, 23, d_model)
    """

    def __init__(self, d_model: int = 64):
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
        # One learned vector per position (0-22). Tells the model the order.
        self.pos_embedding = nn.Embedding(len(TOKEN_ORDER), d_model)

        # [CLS] token — learnable parameter, prepended at position 0
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Pre-register position indices as a buffer (no grad needed)
        # +1 because position 0 is now [CLS]
        positions = torch.arange(1, len(TOKEN_ORDER) + 1)
        self.register_buffer("positions", positions)

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
            tokens — (B, 23, d_model) float tensor
        """
        B = batch["num_values"].shape[0]
        device = batch["num_values"].device
        tokens = []  # will collect 23 tensors of shape (B, d_model)

        # Positional embeddings for all 23 positions at once
        all_pos_emb = self.pos_embedding(self.positions)  # (23, d_model)

        # ── Categorical tokens (positions 0, 1, 2) ───────────────────────────
        for pos, feat in enumerate(CATEGORICAL_FEATURES):
            local_idx = batch["cat_indices"][feat]          # (B,)
            token = (
                self.cat_embeddings[feat](local_idx)        # (B, d_model)
                + all_pos_emb[pos]                          # broadcast (d_model,)
            )
            tokens.append(token)

        # ── PAY tokens (positions 3–8) ────────────────────────────────────────
        # pay_state_ids:  (B, 6)
        # pay_severities: (B, 6)
        n_cat = len(CATEGORICAL_FEATURES)  # = 3, offset for PAY positions
        for i in range(len(PAY_STATUS_FEATURES)):
            pos = n_cat + i                                 # positions 3..8

            state_ids = batch["pay_state_ids"][:, i]       # (B,)
            severities = batch["pay_severities"][:, i]     # (B,)

            state_emb = self.pay_state_embedding(state_ids)          # (B, d_model)
            sev_emb   = self.pay_severity_proj(severities.unsqueeze(-1))  # (B, d_model)

            token = state_emb + sev_emb + all_pos_emb[pos]
            tokens.append(token)

        # ── Numerical tokens (positions 9–22) ────────────────────────────────
        # num_values: (B, 14)
        n_cat_pay = n_cat + len(PAY_STATUS_FEATURES)  # = 9, offset for numerical
        for j in range(len(NUMERICAL_FEATURES)):
            pos = n_cat_pay + j                             # positions 9..22

            feat_idx = torch.tensor(j, dtype=torch.long, device=device)
            values   = batch["num_values"][:, j]            # (B,)

            feat_emb  = self.num_feature_embedding(feat_idx)         # (d_model,)
            value_emb = self.value_proj(values.unsqueeze(-1))        # (B, d_model)

            token = value_emb + feat_emb + all_pos_emb[pos]         # (B, d_model)
            tokens.append(token)

        # Prepend [CLS] token at position 0
        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, d_model)
        feature_tokens = torch.stack(tokens, dim=1)      # (B, 23, d_model)
        return torch.cat([cls, feature_tokens], dim=1)   # (B, 24, d_model)


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
