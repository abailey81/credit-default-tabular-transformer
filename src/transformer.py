"""PreNorm transformer blocks + encoder + two learnable attention biases.
TemporalDecayBias (N3) = ALiBi-style within-group distance penalty,
FeatureGroupBias (N2) = 5×5 learnable bias by (group, group). Both zero-init."""

from typing import List, Literal, Optional

import torch
import torch.nn as nn
from attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Linear → GELU → Dropout → Linear."""

    def __init__(
        self,
        d_model: int = 32,
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
        # Kaiming('relu') on GELU input (torch has no 'gelu' mode, variance
        # ≈ same). Xavier on the residual-feeding output.
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity="relu")
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """pre-norm: z = x + attn(LN(x)); out = z + ffn(LN(z)).
    attn/ffn/residual dropout are each independently overridable (ablation
    A12); None → inherit from `dropout`."""

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 4,
        d_ff: int | None = None,
        dropout: float = 0.1,
        *,
        attn_dropout: float | None = None,
        ffn_dropout: float | None = None,
        residual_dropout: float | None = None,
    ):
        super().__init__()
        attn_p = dropout if attn_dropout is None else attn_dropout
        ffn_p = dropout if ffn_dropout is None else ffn_dropout
        res_p = dropout if residual_dropout is None else residual_dropout

        self.ln1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads, dropout=attn_p
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=ffn_p)
        self.residual_dropout = nn.Dropout(p=res_p)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attention(self.ln1(x), attn_bias=attn_bias)
        x = x + self.residual_dropout(attn_out)
        x = x + self.residual_dropout(self.ffn(self.ln2(x)))
        return x, attn_weights


class TemporalDecayBias(nn.Module):
    """ALiBi-style decay within each temporal group (PAY / BILL_AMT / PAY_AMT):
    bias[i, j] = -α·|month(i) - month(j)| for same-group pairs, else 0.
    α zero-init so the prior starts inactive. per_head → α is (n_heads,)."""

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

        # not persistent — rebuilt from temporal_layout at construction
        self.register_buffer("neg_distance_masked", neg_dist, persistent=False)

        if mode == "scalar":
            self.alpha = nn.Parameter(torch.zeros(1))
        elif mode == "per_head":
            self.alpha = nn.Parameter(torch.zeros(n_heads))
        elif mode == "off":
            self.alpha = None
        else:
            raise ValueError(f"Unknown mode '{mode}'")

    def forward(self) -> torch.Tensor | None:
        """(T, T) scalar, (H, T, T) per-head, None off."""
        if self.mode == "off":
            return None
        if self.mode == "scalar":
            return self.alpha * self.neg_distance_masked
        return self.alpha.view(-1, 1, 1) * self.neg_distance_masked.unsqueeze(0)


class FeatureGroupBias(nn.Module):
    """Additive bias indexed by (q_group, k_group). 5 groups: CLS, demo, PAY,
    BILL_AMT, PAY_AMT. Zero-init. group_assignment[i] = group of token i —
    pass embedding.build_group_assignment() for drift-safety."""

    def __init__(
        self,
        group_assignment: List[int],
        n_groups: int = 5,
        mode: Literal["scalar", "per_head", "off"] = "scalar",
        n_heads: int = 1,
    ):
        super().__init__()
        if mode not in ("scalar", "per_head", "off"):
            raise ValueError(f"Unknown mode '{mode}'")
        if any(g < 0 or g >= n_groups for g in group_assignment):
            raise ValueError(
                f"group_assignment values must lie in [0, {n_groups}); "
                f"got {group_assignment}"
            )

        self.mode = mode
        self.n_groups = n_groups
        self.n_heads = n_heads

        # non-persistent — rebuilt from TOKEN_ORDER at construction
        assignment = torch.as_tensor(group_assignment, dtype=torch.long)
        self.register_buffer("group_assignment", assignment, persistent=False)

        if mode == "scalar":
            self.bias_matrix: Optional[nn.Parameter] = nn.Parameter(
                torch.zeros(n_groups, n_groups)
            )
        elif mode == "per_head":
            self.bias_matrix = nn.Parameter(
                torch.zeros(n_heads, n_groups, n_groups)
            )
        else:
            self.bias_matrix = None

    def forward(self) -> torch.Tensor | None:
        """(T, T) scalar, (H, T, T) per-head, None off."""
        if self.mode == "off" or self.bias_matrix is None:
            return None
        g = self.group_assignment
        if self.mode == "scalar":
            return self.bias_matrix[g.unsqueeze(-1), g.unsqueeze(0)]
        return self.bias_matrix[:, g.unsqueeze(-1), g.unsqueeze(0)]


class TransformerEncoder(nn.Module):
    """Stack of n_layers TransformerBlocks. Any supplied bias modules are
    summed once and shared across every block (ALiBi-style single schedule).
    Returns (x, [attn_weights per layer]) so rollout can see every layer.

    XXX: neg_distance_masked is non-persistent — loading a chkpt into an
    encoder built with a different temporal_layout silently uses a different
    distance matrix. Always go through embedding.build_temporal_layout."""

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 4,
        d_ff: int | None = None,
        n_layers: int = 2,
        dropout: float = 0.1,
        *,
        attn_dropout: float | None = None,
        ffn_dropout: float | None = None,
        residual_dropout: float | None = None,
        temporal_decay: TemporalDecayBias | None = None,
        feature_group_bias: FeatureGroupBias | None = None,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    ffn_dropout=ffn_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.temporal_decay = temporal_decay
        self.feature_group_bias = feature_group_bias

    def _compose_attn_bias(self) -> torch.Tensor | None:
        """sum enabled biases. (T, T) broadcasts against (H, T, T), so mixing
        scalar and per-head modes composes correctly."""
        biases: list[torch.Tensor] = []
        if self.temporal_decay is not None:
            t = self.temporal_decay()
            if t is not None:
                biases.append(t)
        if self.feature_group_bias is not None:
            g = self.feature_group_bias()
            if g is not None:
                biases.append(g)
        if not biases:
            return None
        out = biases[0]
        for b in biases[1:]:
            out = out + b
        return out

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # compose once, reuse across blocks
        attn_bias = self._compose_attn_bias()

        attn_weights_per_layer: list[torch.Tensor] = []
        for block in self.blocks:
            x, attn_weights = block(x, attn_bias=attn_bias)
            attn_weights_per_layer.append(attn_weights)
        return x, attn_weights_per_layer


if __name__ == "__main__":
    import sys
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            try:
                _s.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass

    B, seq_len, d_model, n_heads, n_layers = 4, 24, 64, 4, 2

    from embedding import build_temporal_layout

    layout = build_temporal_layout()

    # drift guard: a TOKEN_ORDER change in embedding.py would quietly
    # shift these positions and break the test's assumptions
    EXPECTED_LAYOUT = {
        "pay":     {"positions": [4, 5, 6, 7, 8, 9],         "months": [0, 1, 2, 3, 4, 5]},
        "bill":    {"positions": [12, 13, 14, 15, 16, 17],   "months": [0, 1, 2, 3, 4, 5]},
        "pay_amt": {"positions": [18, 19, 20, 21, 22, 23],   "months": [0, 1, 2, 3, 4, 5]},
    }
    assert layout == EXPECTED_LAYOUT, (
        f"TOKEN_ORDER drift detected: build_temporal_layout() returned "
        f"{layout} but the canonical expected layout is {EXPECTED_LAYOUT}."
    )

    # A: plain encoder, no bias
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

    print("A: no bias")
    print(f"  out:  {out.shape}")
    print(f"  attn: {len(weights)} layers, {weights[0].shape}")
    print("  grads flow OK")

    # B: scalar TemporalDecayBias
    decay_check = TemporalDecayBias(layout, seq_len=seq_len, mode="scalar")
    bias_at_zero = decay_check()
    assert torch.equal(bias_at_zero, torch.zeros(seq_len, seq_len)), \
        "alpha=0 should produce a zero bias matrix"

    torch.manual_seed(42)
    decay = TemporalDecayBias(layout, seq_len=seq_len, mode="scalar")
    encoder_b = TransformerEncoder(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=0.0,
        temporal_decay=decay,
    )
    encoder_b.eval()

    with torch.no_grad():
        decay.alpha.fill_(5.0)
    _, weights_decayed = encoder_b(x)
    w0 = weights_decayed[0]

    # α=5 → 4→9 penalised by -25 vs 4→5 at -5, so col 9 softmax < col 5
    pay0_to_pay2 = w0[:, :, 4, 5]
    pay0_to_pay6 = w0[:, :, 4, 9]
    assert (pay0_to_pay6 < pay0_to_pay2).all(), \
        "Temporal decay should make PAY_0 -> PAY_6 weaker than PAY_0 -> PAY_2"

    with torch.no_grad():
        decay.alpha.fill_(0.5)
    encoder_b.train()
    out_grad, _ = encoder_b(x)
    out_grad.sum().backward()
    assert decay.alpha.grad is not None, "alpha has no .grad attribute"
    assert decay.alpha.grad.abs().item() > 0, "alpha gradient is exactly zero"

    print("\nB: scalar decay")
    print("  α=0 → 0 bias                OK")
    print("  α=5 kills distant-month attn OK")
    print("  α gets grad                  OK")

    # C: other modes
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

    print("\nC: modes")
    print("  off → None            OK")
    print(f"  per_head {bias_ph.shape} OK")
    print("  encoder runs          OK")

    print("\nall checks passed.")
