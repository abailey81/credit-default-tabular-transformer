"""
transformer.py — PreNorm transformer block, encoder, and architectural novelty biases.

Classes
-------
1. :class:`FeedForward`        — position-wise two-layer MLP (Linear → GELU → Linear).
2. :class:`TransformerBlock`   — one PreNorm block: attention → FFN with residuals,
                                 with independently ablatable attention / FFN /
                                 residual dropout channels (Plan §6.3 / A12).
3. :class:`TemporalDecayBias`  — EDA-motivated learnable prior on attention scores:
                                 penalises attention between distant months within
                                 each temporal group (PAY / BILL_AMT / PAY_AMT).
                                 Starts at α=0 (inactive). Adaptation of ALiBi
                                 (Press et al. 2022) to tabular temporal structure.
                                 Plan §6.12.2 / **Novelty N3** / Ablation A22.
4. :class:`FeatureGroupBias`   — credit-risk-specific inductive bias: a learnable
                                 (n_groups × n_groups) bias matrix indexed by the
                                 semantic group of (query, key) tokens. Groups:
                                 ``{CLS, demographic, PAY, BILL_AMT, PAY_AMT}``.
                                 Soft prior that within-group attention is easier
                                 than cross-group. Zero-initialised so the default
                                 behaviour is unchanged. Plan §6.12.1 /
                                 **Novelty N2** / Ablation A21.
5. :class:`TransformerEncoder` — stacks N blocks, optionally composes multiple
                                 attention biases (temporal decay + feature-group),
                                 and collects per-layer attention weights for
                                 Phase 10 rollout.

References
----------
* Vaswani et al. (2017). "Attention Is All You Need". *NeurIPS*.
* Gorishniy et al. (2021). "Revisiting Deep Learning Models for Tabular Data". *NeurIPS*.
* Press et al. (2022). "Train Short, Test Long: Attention with Linear Biases (ALiBi)". *ICLR*.
"""

from typing import List, Literal, Optional

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
        d_model: Token embedding dimension. Default ``32`` matches Plan §6.11.
        d_ff:    Hidden dimension. Defaults to ``4 * d_model``.
        dropout: Hidden-layer dropout, applied between GELU and ``W_2``.
    """

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

        x = x + residual_dropout( MultiHeadAttention( LayerNorm(x), attn_bias ) )
        x = x + residual_dropout( FeedForward( LayerNorm(x) ) )

    PreNorm (as in GPT-2 and FT-Transformer) keeps the residual path clean —
    gradients flow directly through the addition without passing through
    LayerNorm, giving more stable training than PostNorm (Vaswani 2017).

    The three dropout hyperparameters are **independently ablatable** per
    Plan §6.3 and Ablation A12:

    * ``attn_dropout``     — post-softmax attention-weight dropout
    * ``ffn_dropout``      — FFN hidden-layer dropout (between GELU and W_2)
    * ``residual_dropout`` — dropout on each sub-layer's output before the
      residual add

    Passing only ``dropout`` sets all three to that value (legacy behaviour);
    explicit kwargs override per-channel. This matters for A12 where the plan
    treats attention-weight dropout and FFN/residual dropout as separate axes.

    Args:
        d_model: Token embedding dimension. Default ``32`` matches Plan §6.11.
        n_heads: Number of attention heads (default 4).
        d_ff:    FFN hidden dimension. Defaults to ``4 * d_model``.
        dropout: Shared default for all three dropout channels.
        attn_dropout: Attention-weight dropout. Defaults to ``dropout``.
        ffn_dropout:  FFN-hidden dropout. Defaults to ``dropout``.
        residual_dropout: Residual-branch dropout. Defaults to ``dropout``.
    """

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
# 4. Feature-group attention bias — credit-risk inductive prior (Novelty N2)
# ──────────────────────────────────────────────────────────────────────────────

class FeatureGroupBias(nn.Module):
    """
    Learnable additive bias on pre-softmax attention scores indexed by the
    *semantic group* of the (query, key) token pair. The 24-token sequence
    naturally partitions into five groups:

        0 = [CLS]
        1 = demographic (SEX / EDUCATION / MARRIAGE / LIMIT_BAL / AGE)
        2 = PAY status (PAY_0 .. PAY_6)
        3 = BILL_AMT (BILL_AMT1 .. BILL_AMT6)
        4 = PAY_AMT (PAY_AMT1 .. PAY_AMT6)

    The model learns a 5×5 bias matrix ``B`` (or a stack of ``n_heads`` such
    matrices in ``per_head`` mode). At forward time we expand ``B`` into a
    ``(seq_len, seq_len)`` (or ``(n_heads, seq_len, seq_len)``) tensor via
    group-index lookup and return it as an additive bias.

    Rationale (Plan §6.12.1): the six PAY tokens share semantics the twelve
    amount tokens do not; a soft prior that within-group attention is easier
    than cross-group may speed convergence and improve generalisation. The
    bias is ``B = 0`` at init, so the default behaviour is a plain encoder —
    the model *chooses* whether to learn group structure.

    Parameters
    ----------
    group_assignment
        ``List[int]`` of length ``seq_len``; ``group_assignment[i]`` is the
        group index of token *i*. Pass :func:`embedding.build_group_assignment`
        to stay drift-safe under any TOKEN_ORDER reordering.
    n_groups
        Number of distinct groups. Default 5 for this dataset.
    mode
        * ``"scalar"``    — one 5×5 bias matrix shared across heads.
        * ``"per_head"``  — one 5×5 bias matrix per attention head.
        * ``"off"``       — disabled; ``forward()`` returns ``None``.
    n_heads
        Only used when ``mode="per_head"``.

    References
    ----------
    Plan §6.12.1 / Novelty register N2 / Ablation A21.
    """

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

        # Group index for each of the seq_len token positions. Registered as a
        # non-persistent buffer (deterministically reconstructible from the
        # caller's group_assignment list, so excluding it from state_dict keeps
        # checkpoints smaller).
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
        else:  # "off"
            self.bias_matrix = None

    def forward(self) -> torch.Tensor | None:
        """
        Returns the additive bias tensor, or None if disabled.

        Output shapes:
            scalar:   (seq_len, seq_len)            — broadcasts over batch & heads
            per_head: (n_heads, seq_len, seq_len)   — broadcasts over batch
            off:      None
        """
        if self.mode == "off" or self.bias_matrix is None:
            return None
        g = self.group_assignment
        # Fancy-indexing: for every (i, j), look up B[g[i], g[j]].
        if self.mode == "scalar":
            return self.bias_matrix[g.unsqueeze(-1), g.unsqueeze(0)]  # (seq, seq)
        # per_head
        # B: (H, n_groups, n_groups); index into the last two dims with (g[i], g[j]).
        return self.bias_matrix[:, g.unsqueeze(-1), g.unsqueeze(0)]  # (H, seq, seq)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Transformer encoder — stacks blocks, composes bias modules, collects attn
# ──────────────────────────────────────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    """
    Stacks ``n_layers`` :class:`TransformerBlock`s, threading the same
    :class:`TemporalDecayBias` output (if provided) through *every* block
    and collecting per-layer attention weights for attention rollout
    (Abnar & Zuidema, 2020) used in the interpretability phase (Plan §12.2).

    The temporal-decay bias is **shared across layers** rather than per-layer
    by design: ALiBi (Press et al., 2022) uses a single static bias schedule
    across depth; in our adaptation the only learnable parameter is α (one
    scalar, or ``n_heads`` scalars in per-head mode). Sharing keeps the
    novelty parameter count at its floor and gives Ablation A22 a single
    clean on/off axis. A per-layer variant would multiply α by ``n_layers``
    and complicate the ablation matrix without a principled motivation.

    .. note::
        :class:`TemporalDecayBias` registers ``neg_distance_masked`` with
        ``persistent=False`` — the distance matrix is deterministically
        reconstructable from ``temporal_layout`` and is therefore kept out of
        the state-dict. If a checkpoint is loaded into a fresh encoder built
        with a *different* ``temporal_layout`` the matrix will silently
        differ. Callers that load checkpoints must construct the encoder
        with an identical layout (use :func:`embedding.build_temporal_layout`
        to avoid drift).

    Args:
        d_model: Token embedding dimension. Default ``32`` matches Plan §6.11.
        n_heads: Attention heads (default 4, per Plan §6.2; Ablation A3
            sweeps {1, 2, 4, 8}).
        d_ff:    FFN hidden dim. Defaults to ``4 * d_model``.
        n_layers: Number of stacked blocks (default 2; Ablation A2 sweeps
            {1, 2, 3, 4}).
        dropout: Shared default for all three dropout channels.
        attn_dropout / ffn_dropout / residual_dropout: Optional per-channel
            overrides; unset channels inherit ``dropout``. Forwarded to every
            :class:`TransformerBlock` in the stack. Enables Ablation A12.
        temporal_decay: Optional :class:`TemporalDecayBias` (Novelty N3 /
            Ablation A22).
        feature_group_bias: Optional :class:`FeatureGroupBias` (Novelty N2 /
            Ablation A21).

    Both bias modules, when provided, have their per-forward outputs **summed**
    (broadcasting rules: ``(seq, seq)`` and ``(n_heads, seq, seq)`` compose
    naturally) and threaded into every block's attention call as a single
    ``attn_bias`` tensor. This keeps the per-layer dispatch path at one
    additive tensor regardless of how many novelty modules are active.
    """

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
        # Register both bias modules as submodules so their learnable
        # parameters (α for TemporalDecayBias, B for FeatureGroupBias) appear
        # in .parameters() and move with .to(device).
        self.temporal_decay = temporal_decay
        self.feature_group_bias = feature_group_bias

    def _compose_attn_bias(self) -> torch.Tensor | None:
        """Compute each bias module's output and sum them elementwise.

        Each module returns either ``None`` (disabled), a ``(T, T)`` tensor
        (scalar mode — broadcasts to any batch/head), or a ``(H, T, T)``
        tensor (per-head mode). Summing composes per-head into per-head and
        scalar-into-scalar; mixing broadcasts correctly.
        """
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
        # Compute the composed bias ONCE per forward pass; the same tensor is
        # threaded into every block's attention (Plan §6.12 share-across-
        # layers convention). Skipping the recompute keeps the forward pass
        # O(1) in the number of bias modules.
        attn_bias = self._compose_attn_bias()

        attn_weights_per_layer: list[torch.Tensor] = []
        for block in self.blocks:
            x, attn_weights = block(x, attn_bias=attn_bias)
            attn_weights_per_layer.append(attn_weights)
        return x, attn_weights_per_layer


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # UTF-8 stdout so the box-drawing separators print cleanly on Windows.
    import sys
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            try:
                _s.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass

    B, seq_len, d_model, n_heads, n_layers = 4, 24, 64, 4, 2

    # Derive the temporal layout from embedding.TOKEN_ORDER so any future
    # reordering there is caught by the drift-detection assertion below.
    from embedding import build_temporal_layout

    layout = build_temporal_layout()

    # Drift detection: compare against the canonical layout that TemporalDecayBias
    # was designed for. A mismatch means somebody reordered TOKEN_ORDER in
    # embedding.py without updating the temporal groups — the smoke test would
    # then quietly test the wrong thing.
    EXPECTED_LAYOUT = {
        "pay":     {"positions": [4, 5, 6, 7, 8, 9],         "months": [0, 1, 2, 3, 4, 5]},
        "bill":    {"positions": [12, 13, 14, 15, 16, 17],   "months": [0, 1, 2, 3, 4, 5]},
        "pay_amt": {"positions": [18, 19, 20, 21, 22, 23],   "months": [0, 1, 2, 3, 4, 5]},
    }
    assert layout == EXPECTED_LAYOUT, (
        f"TOKEN_ORDER drift detected: build_temporal_layout() returned "
        f"{layout} but the canonical expected layout is {EXPECTED_LAYOUT}. "
        f"Update EXPECTED_LAYOUT here together with embedding.TOKEN_ORDER."
    )

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
