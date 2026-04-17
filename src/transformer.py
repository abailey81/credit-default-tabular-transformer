"""
transformer.py — PreNorm transformer block, encoder, and temporal-decay bias.

Four classes, stacked on top of src/attention.py's MultiHeadAttention:

    1. FeedForward        — position-wise two-layer MLP (Linear → GELU → Linear).
    2. TransformerBlock   — one PreNorm block: attention → FFN with residuals.
    3. TemporalDecayBias  — EDA-motivated learnable prior on attention scores:
                            penalises attention between distant months within
                            each temporal group (PAY / BILL_AMT / PAY_AMT).
                            Starts at α=0 (inactive). Adaptation of ALiBi
                            (Press et al. 2022) to tabular temporal structure.
    4. TransformerEncoder — stacks N blocks, optionally applies TemporalDecayBias,
                            collects per-layer attention weights for rollout.

The TemporalDecayBias is the single principled architectural extension tied to
EDA Fig 9 (BILL_AMT autocorrelation decays from 0.95 at lag 1 to 0.7 at lag 5).
The model activates it only if helpful — α is zero-initialised, so the default
behaviour is a standard FT-Transformer-style encoder.

References:
    Vaswani et al. (2017), "Attention Is All You Need", NeurIPS.
    Gorishniy et al. (2021), "Revisiting Deep Learning Models for Tabular Data", NeurIPS.
    Press et al. (2022), "Train Short, Test Long: Attention with Linear Biases" (ALiBi), ICLR.
"""

from typing import Literal

import torch
import torch.nn as nn
from attention import MultiHeadAttention

# ──────────────────────────────────────────────────────────────────────────────
# 1. Position-wise feed-forward network
# ──────────────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Two-layer MLP applied independently to every token:

        FFN(x) = W_2 · Dropout(GELU(W_1 · x + b_1)) + b_2

    Expansion factor d_ff / d_model = 4 is standard (Vaswani 2017).

    Args:
        d_model: Token embedding dimension.
        d_ff:    Hidden dimension. Defaults to 4 * d_model.
        dropout: Dropout applied between the two linear layers.
    """

    def __init__(
        self,
        d_model: int = 64,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self._init_weights()

    def _init_weights(self):
        # W_1 feeds into GELU — Kaiming preserves variance through the non-linearity.
        # (We use nonlinearity='relu' because PyTorch has no 'gelu' mode; GELU and
        # ReLU have very similar variance-preservation characteristics.)
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity="relu")
        nn.init.zeros_(self.linear1.bias)

        # W_2 output feeds the residual connection directly (no activation) — Xavier.
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


# ──────────────────────────────────────────────────────────────────────────────
# 2. One PreNorm transformer block
# ──────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    One PreNorm transformer block:

        x = x + Dropout(MultiHeadAttention(LayerNorm(x), attn_bias))
        x = x + Dropout(FeedForward(LayerNorm(x)))

    PreNorm (as in GPT-2 and FT-Transformer) keeps the residual path clean —
    gradients flow directly through the addition without passing through
    LayerNorm, giving more stable training than PostNorm (Vaswani 2017).

    Args:
        d_model: Token embedding dimension.
        n_heads: Number of attention heads.
        d_ff:    FFN hidden dimension. Defaults to 4 * d_model.
        dropout: Shared dropout rate (attention weights, FFN inner, residual).
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.residual_dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Sub-layer 1: attention with a residual connection
        attn_out, attn_weights = self.attention(self.ln1(x), attn_bias=attn_bias)
        x = x + self.residual_dropout(attn_out)

        # Sub-layer 2: position-wise FFN with a residual connection
        x = x + self.residual_dropout(self.ffn(self.ln2(x)))

        return x, attn_weights


# ──────────────────────────────────────────────────────────────────────────────
# 3. Temporal-decay attention bias — EDA-motivated inductive prior
# ──────────────────────────────────────────────────────────────────────────────

class TemporalDecayBias(nn.Module):
    """
    Learnable ALiBi-style decay on attention scores within each temporal group.

    Motivation (report §3): EDA Fig 9 shows BILL_AMT autocorrelation drops from
    0.95 at lag 1 to 0.7 at lag 5 — recent months carry more mutual information
    than distant ones for default prediction. We bake this prior into the
    attention mechanism as an additive bias on pre-softmax scores:

        bias[i, j] = -α · |month(i) - month(j)|   if i, j in the same temporal group
        bias[i, j] = 0                             otherwise

    α is a learnable scalar (or per-head vector) initialised to zero — if the
    prior is unhelpful, the model simply leaves α ≈ 0 and recovers standard
    attention.

    The [CLS] token and non-temporal features (LIMIT_BAL, SEX, EDUCATION,
    MARRIAGE, AGE) are all given zero bias — they can attend freely across
    every other token.

    Args:
        temporal_layout: Dict mapping group name → {"positions": list[int],
                         "months": list[int]}. Positions index into the
                         seq_len dimension; months are 0-indexed (0 = most
                         recent). Example for the 24-token layout:

                             {
                                 "pay":     {"positions": [4, 5, 6, 7, 8, 9],
                                             "months":    [0, 1, 2, 3, 4, 5]},
                                 "bill":    {"positions": [12, 13, 14, 15, 16, 17],
                                             "months":    [0, 1, 2, 3, 4, 5]},
                                 "pay_amt": {"positions": [18, 19, 20, 21, 22, 23],
                                             "months":    [0, 1, 2, 3, 4, 5]},
                             }

        seq_len: Total sequence length (including [CLS]).
        mode:    One of:
                   "scalar"   — single learnable α (default).
                   "per_head" — one α per attention head.
                   "off"      — disabled; forward() returns None.
        n_heads: Only used when mode="per_head".
    """

    def __init__(
        self,
        temporal_layout: dict,
        seq_len: int,
        mode: Literal["scalar", "per_head", "off"] = "scalar",
        n_heads: int = 1,
    ):
        super().__init__()
        self.mode = mode
        self.n_heads = n_heads
        self.seq_len = seq_len

        # Pre-compute the static (seq_len, seq_len) distance matrix, masked to
        # within-group positions. All other entries (CLS rows/cols, static
        # features, cross-group pairs) are zero — those tokens receive no
        # penalty regardless of α.
        neg_dist = torch.zeros(seq_len, seq_len)
        for group_name, group_spec in temporal_layout.items():
            positions = group_spec["positions"]
            months = group_spec["months"]
            assert len(positions) == len(months), (
                f"positions/months length mismatch in group '{group_name}'"
            )
            for a_pos, a_month in zip(positions, months):
                for b_pos, b_month in zip(positions, months):
                    neg_dist[a_pos, b_pos] = -abs(a_month - b_month)

        # Buffer, not parameter — no gradient, moves with .to(device) / .cuda().
        self.register_buffer("neg_distance_masked", neg_dist, persistent=False)

        # Learnable α — zero-initialised so the prior is inactive at training start.
        if mode == "scalar":
            self.alpha = nn.Parameter(torch.zeros(1))
        elif mode == "per_head":
            self.alpha = nn.Parameter(torch.zeros(n_heads))
        elif mode == "off":
            self.alpha = None
        else:
            raise ValueError(f"Unknown mode '{mode}'")

    def forward(self) -> torch.Tensor | None:
        """
        Returns the additive bias for attention scores, or None if disabled.

        Output shapes by mode:
            scalar:   (seq_len, seq_len)              — broadcasts to (B, n_heads)
            per_head: (n_heads, seq_len, seq_len)     — broadcasts to batch
            off:      None
        """
        if self.mode == "off":
            return None
        if self.mode == "scalar":
            return self.alpha * self.neg_distance_masked  # (seq_len, seq_len)
        # per_head — α: (n_heads,), dist: (seq_len, seq_len)
        # → (n_heads, 1, 1) * (1, seq_len, seq_len) = (n_heads, seq_len, seq_len)
        return self.alpha.view(-1, 1, 1) * self.neg_distance_masked.unsqueeze(0)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Transformer encoder — stacks blocks, threads temporal bias, collects attn
# ──────────────────────────────────────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    """
    Stacks n_layers TransformerBlocks, threading the same temporal-decay bias
    (if provided) through every block. Collects per-layer attention weights
    for attention rollout (Abnar & Zuidema, 2020) in the interpretability phase.

    Args:
        d_model, n_heads, d_ff, n_layers, dropout — block/layer hyperparameters.
        temporal_decay: Optional TemporalDecayBias module. When provided, its
                        output is added to every block's pre-softmax attention
                        scores. When None, the encoder behaves as a plain
                        FT-Transformer-style stack.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int | None = None,
        n_layers: int = 2,
        dropout: float = 0.1,
        temporal_decay: TemporalDecayBias | None = None,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )
        # Register as submodule so α shows up in .parameters() and moves with .to(device).
        self.temporal_decay = temporal_decay

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Compute the temporal bias once per forward pass (same for every block).
        attn_bias = (
            self.temporal_decay() if self.temporal_decay is not None else None
        )

        attn_weights_per_layer: list[torch.Tensor] = []
        for block in self.blocks:
            x, attn_weights = block(x, attn_bias=attn_bias)
            attn_weights_per_layer.append(attn_weights)
        return x, attn_weights_per_layer


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, seq_len, d_model, n_heads, n_layers = 4, 24, 64, 4, 2

    # Temporal layout matching embedding.py's TOKEN_ORDER:
    # CLS=0, SEX/EDU/MAR=1-3, PAY_0..PAY_6=4-9, LIMIT_BAL/AGE=10-11,
    # BILL_AMT1..6=12-17, PAY_AMT1..6=18-23.
    layout = {
        "pay":     {"positions": [4, 5, 6, 7, 8, 9],       "months": [0, 1, 2, 3, 4, 5]},
        "bill":    {"positions": [12, 13, 14, 15, 16, 17], "months": [0, 1, 2, 3, 4, 5]},
        "pay_amt": {"positions": [18, 19, 20, 21, 22, 23], "months": [0, 1, 2, 3, 4, 5]},
    }

    # ── Test A: no temporal bias — plain encoder ─────────────────────────
    torch.manual_seed(0)
    x = torch.randn(B, seq_len, d_model)

    torch.manual_seed(42)
    encoder = TransformerEncoder(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=0.0,
        temporal_decay=None,
    )
    encoder.eval()

    out, weights = encoder(x)

    assert out.shape == (B, seq_len, d_model), f"output shape {out.shape}"
    assert len(weights) == n_layers, f"expected {n_layers} attn tensors, got {len(weights)}"
    for w in weights:
        assert w.shape == (B, n_heads, seq_len, seq_len), f"attn shape {w.shape}"
        row_sums = w.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
            "Attention rows don't sum to 1"

    encoder.train()
    out_train, _ = encoder(x)
    out_train.sum().backward()
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"

    print("Test A — no temporal bias")
    print(f"  Output shape:          {out.shape}")
    print(f"  Attention layers:      {len(weights)}")
    print(f"  Per-layer attn shape:  {weights[0].shape}")
    print("  All gradients flow:    OK")

    # ── Test B: TemporalDecayBias mechanism (scalar mode) ────────────────
    # α=0 by default → bias is the zero matrix (trivial but worth asserting).
    decay_check = TemporalDecayBias(layout, seq_len=seq_len, mode="scalar")
    bias_at_zero = decay_check()
    assert torch.equal(bias_at_zero, torch.zeros(seq_len, seq_len)), \
        "α=0 should produce a zero bias matrix"

    # α > 0 → within-group distant-month attention should be softened.
    torch.manual_seed(42)
    decay = TemporalDecayBias(layout, seq_len=seq_len, mode="scalar")
    encoder_b = TransformerEncoder(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=0.0,
        temporal_decay=decay,
    )
    encoder_b.eval()

    # Strong α so the decay dominates any raw-score noise in this smoke test.
    with torch.no_grad():
        decay.alpha.fill_(5.0)
    _, weights_decayed = encoder_b(x)
    w0 = weights_decayed[0]  # first layer's attention — clean test of the mechanism

    # PAY_0 = pos 4 (month 0); PAY_2 = pos 5 (month 1); PAY_6 = pos 9 (month 5).
    # With α=5 the score at [4, 9] is penalised by -25 vs [4, 5] at -5:
    # the softmax weight at column 9 should be lower than at column 5.
    pay0_to_pay2 = w0[:, :, 4, 5]
    pay0_to_pay6 = w0[:, :, 4, 9]
    assert (pay0_to_pay6 < pay0_to_pay2).all(), \
        "Temporal decay should make PAY_0→PAY_6 weaker than PAY_0→PAY_2"

    # α must receive non-zero gradient during training.
    with torch.no_grad():
        decay.alpha.fill_(0.5)
    encoder_b.train()
    out_grad, _ = encoder_b(x)
    out_grad.sum().backward()
    assert decay.alpha.grad is not None, "α has no .grad attribute"
    assert decay.alpha.grad.abs().item() > 0, "α gradient is exactly zero"

    print("\nTest B — TemporalDecayBias (scalar)")
    print("  α=0 produces zero bias:            OK")
    print("  α=5 suppresses distant-month attn: OK")
    print("  α receives non-zero gradient:      OK")

    # ── Test C: ablation modes ───────────────────────────────────────────
    decay_off = TemporalDecayBias(layout, seq_len=seq_len, mode="off")
    assert decay_off() is None, "mode='off' should return None"

    decay_ph = TemporalDecayBias(
        layout, seq_len=seq_len, mode="per_head", n_heads=n_heads
    )
    bias_ph = decay_ph()
    assert bias_ph.shape == (n_heads, seq_len, seq_len), \
        f"per-head bias shape {bias_ph.shape}"

    encoder_ph = TransformerEncoder(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=0.0,
        temporal_decay=decay_ph,
    )
    encoder_ph.eval()
    out_ph, _ = encoder_ph(x)
    assert out_ph.shape == (B, seq_len, d_model)

    print("\nTest C — ablation modes")
    print("  mode='off' returns None:           OK")
    print(f"  mode='per_head' shape {bias_ph.shape}: OK")
    print("  encoder with per-head decay runs:  OK")

    print("\nAll checks passed.")
