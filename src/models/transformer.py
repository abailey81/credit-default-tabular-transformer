"""PreNorm transformer encoder + two learnable attention-bias priors.

Purpose
-------
Build the encoder stack that sits between ``FeatureEmbedding`` and the
classification / MTLM heads. Everything here is "plain transformer" except
the two additive-bias modules: ``TemporalDecayBias`` (N3, ALiBi-style decay
within PAY / BILL_AMT / PAY_AMT groups) and ``FeatureGroupBias`` (N2, a
learnable 5x5 by-group bias).

Key public symbols
------------------
- ``FeedForward`` — the standard 2-layer MLP with GELU used inside each
  transformer block.
- ``TransformerBlock`` — PreNorm residual block (attention + FFN).
- ``TransformerEncoder`` — stack of ``n_layers`` blocks; owns the optional
  bias modules and composes them once per forward.
- ``TemporalDecayBias`` (N3) — bias[i, j] = -alpha * |month(i) - month(j)|
  for same-group pairs, zero otherwise; alpha zero-init.
- ``FeatureGroupBias`` (N2) — additive bias indexed by (q_group, k_group);
  5 groups (CLS, demo, PAY, BILL_AMT, PAY_AMT); zero-init.

Design choice
-------------
We use PreNorm (LN before each sublayer, residual bypasses LN) rather than
the original PostNorm from Vaswani+ 2017. PostNorm was unstable without a
learning-rate warmup on our 21K-row regime — PreNorm (Xiong+ 2020) trains
cleanly from step 0 and was kept for reproducibility across ablations.

Critical invariant
------------------
Both bias modules register their geometry buffers (``neg_distance_masked``,
``group_assignment``) as NON-persistent, so they are rebuilt from the
tokenizer layout at construction time. Always go through
``embedding.build_temporal_layout`` / ``build_group_assignment`` so a
``TOKEN_ORDER`` change in the tokenizer can never silently misalign a
loaded checkpoint.
"""

from typing import Literal, Optional

import torch
import torch.nn as nn

from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear -> GELU -> Dropout -> Linear.

    The canonical Vaswani+ 2017 FFN uses ReLU and ``d_ff = 4 * d_model``;
    we keep the 4x expansion but swap to GELU (Hendrycks & Gimpel 2016),
    matching BERT / ViT / FT-Transformer and giving a small but consistent
    AUROC bump in ablation A09.
    """

    def __init__(
        self,
        d_model: int = 32,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_ff is None:
            # 4x expansion is the transformer-era convention. At d_model=32
            # this is d_ff=128 — small enough that the param budget stays
            # dominated by attention projections.
            d_ff = 4 * d_model

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self._init_weights()

    def _init_weights(self):
        """Kaiming on the pre-GELU linear; Xavier on the residual-feeding linear.

        PyTorch has no ``nonlinearity='gelu'`` mode; ReLU's variance profile
        is a close stand-in (both are ~half-linear through origin). The
        output layer feeds a residual connection, where Xavier's
        variance-preserving property avoids compounding drift across
        stacked blocks.
        """
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity="relu")
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, d_model) -> (B, T, d_model)`` via 4x expansion then project down."""
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """PreNorm residual block: ``z = x + attn(LN(x)); out = z + ffn(LN(z))``.

    PreNorm (Xiong+ 2020) was chosen over the original PostNorm because
    PostNorm needed a warmup schedule to train stably on this dataset.
    With PreNorm the residual path never passes through LN, which keeps
    gradients flowing even with the default LR and no warmup.

    Dropout is exposed at three independent sites — attention, FFN, and
    residual — so ablation A12 can vary them separately. Any of the three
    that is left ``None`` inherits the base ``dropout`` value.
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
        # Inherit from the base `dropout` knob unless the caller overrides.
        # This keeps the simple two-arg constructor useful while letting
        # A12 sweep the three dropout sites independently.
        attn_p = dropout if attn_dropout is None else attn_dropout
        ffn_p = dropout if ffn_dropout is None else ffn_dropout
        res_p = dropout if residual_dropout is None else residual_dropout

        self.ln1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=attn_p)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=ffn_p)
        # One Dropout module is reused for both residual sites — this is
        # safe because dropout is stateless at inference and the two calls
        # sample independently under train().
        self.residual_dropout = nn.Dropout(p=res_p)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One PreNorm block pass. Returns ``(x_out, attn_weights)``."""
        # LN inside each sublayer, residual around the sublayer — the
        # PreNorm ordering. Key property: the residual path is an
        # unobstructed identity, so gradients reach early layers cleanly.
        attn_out, attn_weights = self.attention(self.ln1(x), attn_bias=attn_bias)
        x = x + self.residual_dropout(attn_out)
        x = x + self.residual_dropout(self.ffn(self.ln2(x)))
        return x, attn_weights


class TemporalDecayBias(nn.Module):
    """ALiBi-style temporal-decay attention bias (Novelty N3).

    For tokens in the same temporal group (PAY, BILL_AMT, PAY_AMT) we add
    ``-alpha * |month(i) - month(j)|`` to the attention logits. Cross-group
    and demographic/CLS pairs get zero. ``alpha`` is zero-initialised so
    the prior starts INACTIVE — we want the model to discover the decay
    if it's useful, not bake it in from step 0.

    Parameters
    ----------
    temporal_layout
        Dict of ``{group_name: {positions: [...], months: [...]}}`` as
        emitted by ``embedding.build_temporal_layout``. The position
        indices are 24-slot (CLS at 0).
    seq_len
        Total sequence length including CLS (24 for this project).
    mode
        ``"scalar"`` (one shared alpha), ``"per_head"`` (alpha is
        ``(n_heads,)``, enables ablation A13), or ``"off"`` (no-op — kept
        so a constructor call with ``mode='off'`` still type-checks).
    n_heads
        Only read when ``mode='per_head'``.

    Ablation
    --------
    Toggles ablation A05 (no N3) when set to ``"off"`` at the encoder.
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

        # Build the negative-distance matrix once at construction. Entry
        # (i, j) is -|month(i) - month(j)| when i and j are in the same
        # temporal group, else 0 — so CLS/demo/cross-group pairs contribute
        # nothing when multiplied by alpha.
        neg_dist = torch.zeros(seq_len, seq_len)
        for group_name, group_spec in temporal_layout.items():
            positions = group_spec["positions"]
            months = group_spec["months"]
            assert len(positions) == len(
                months
            ), f"positions/months length mismatch in group '{group_name}'"
            for a_pos, a_month in zip(positions, months):
                for b_pos, b_month in zip(positions, months):
                    neg_dist[a_pos, b_pos] = -abs(a_month - b_month)

        # persistent=False: the matrix is derived from temporal_layout and
        # must be rebuilt at construction time, never restored from a
        # checkpoint — if TOKEN_ORDER drifts between save and load, the
        # non-persistent buffer stays authoritative to the current layout.
        self.register_buffer("neg_distance_masked", neg_dist, persistent=False)

        if mode == "scalar":
            # One shared decay rate across every head — simplest case.
            self.alpha = nn.Parameter(torch.zeros(1))
        elif mode == "per_head":
            # Per-head decay lets different heads specialise to different
            # time scales (ablation A13).
            self.alpha = nn.Parameter(torch.zeros(n_heads))
        elif mode == "off":
            # No learnable alpha; forward() returns None and the encoder
            # omits this bias from its composition.
            self.alpha = None
        else:
            raise ValueError(f"Unknown mode '{mode}'")

    def forward(self) -> torch.Tensor | None:
        """Build the bias tensor for the current alpha.

        Returns
        -------
        torch.Tensor or None
            ``(T, T)`` for scalar mode, ``(H, T, T)`` for per-head mode,
            or ``None`` when mode is ``"off"``.
        """
        if self.mode == "off":
            return None
        if self.mode == "scalar":
            return self.alpha * self.neg_distance_masked
        # Per-head: reshape alpha to (H, 1, 1) and broadcast over (T, T).
        return self.alpha.view(-1, 1, 1) * self.neg_distance_masked.unsqueeze(0)


class FeatureGroupBias(nn.Module):
    """Learnable by-group attention bias (Novelty N2).

    Adds a bias ``B[g_q, g_k]`` to every (query, key) pair where ``g_q``
    and ``g_k`` are the tokens' semantic groups. Five groups are used
    here: CLS, demographic, PAY status, BILL_AMT, PAY_AMT (see
    ``embedding.build_group_assignment``). The bias matrix is zero-init,
    so the prior starts inactive and the model learns which group pairs
    deserve extra/less attention.

    Parameters
    ----------
    group_assignment
        Length-``seq_len`` list mapping each token index to its group id.
        Always produced by ``embedding.build_group_assignment`` — passing
        a hand-rolled list risks drifting from ``TOKEN_ORDER``.
    n_groups
        Number of distinct groups; defaults to 5. All values in
        ``group_assignment`` must lie in ``[0, n_groups)``.
    mode
        ``"scalar"`` (one 5x5 matrix, shared across heads), ``"per_head"``
        (``(H, 5, 5)``), or ``"off"`` (no-op).
    n_heads
        Only read when ``mode='per_head'``.

    Ablation
    --------
    Toggles ablation A04 (no N2) when set to ``"off"`` at the encoder.
    """

    def __init__(
        self,
        group_assignment: list[int],
        n_groups: int = 5,
        mode: Literal["scalar", "per_head", "off"] = "scalar",
        n_heads: int = 1,
    ):
        super().__init__()
        if mode not in ("scalar", "per_head", "off"):
            raise ValueError(f"Unknown mode '{mode}'")
        # Defensive validation: a stray -1 or an out-of-range id here
        # produces a hard-to-debug silent gather at forward time.
        if any(g < 0 or g >= n_groups for g in group_assignment):
            raise ValueError(
                f"group_assignment values must lie in [0, {n_groups}); " f"got {group_assignment}"
            )

        self.mode = mode
        self.n_groups = n_groups
        self.n_heads = n_heads

        # Non-persistent: rebuilt from TOKEN_ORDER at construction, never
        # restored from a checkpoint. See module-level invariant note.
        assignment = torch.as_tensor(group_assignment, dtype=torch.long)
        self.register_buffer("group_assignment", assignment, persistent=False)

        if mode == "scalar":
            # Single (n_groups, n_groups) matrix shared across heads —
            # only n_groups**2 new parameters (25 at the default).
            self.bias_matrix: Optional[nn.Parameter] = nn.Parameter(torch.zeros(n_groups, n_groups))
        elif mode == "per_head":
            # Per-head lets different heads emphasise different group
            # pairs — useful when heads specialise (ablation A14).
            self.bias_matrix = nn.Parameter(torch.zeros(n_heads, n_groups, n_groups))
        else:
            self.bias_matrix = None

    def forward(self) -> torch.Tensor | None:
        """Build the bias tensor by gathering with ``group_assignment``.

        Returns
        -------
        torch.Tensor or None
            ``(T, T)`` for scalar mode, ``(H, T, T)`` for per-head mode,
            or ``None`` when mode is ``"off"``.
        """
        if self.mode == "off" or self.bias_matrix is None:
            return None
        g = self.group_assignment
        # Outer-index gather: g.unsqueeze(-1) is (T, 1), g.unsqueeze(0) is
        # (1, T). Indexing bias_matrix with these two broadcasts to (T, T)
        # where entry (i, j) = bias_matrix[group(i), group(j)].
        if self.mode == "scalar":
            return self.bias_matrix[g.unsqueeze(-1), g.unsqueeze(0)]
        # Per-head: same gather, with the leading head axis preserved.
        return self.bias_matrix[:, g.unsqueeze(-1), g.unsqueeze(0)]


class TransformerEncoder(nn.Module):
    """Stack of ``n_layers`` ``TransformerBlock``s with optional bias priors.

    The bias modules (if provided) are evaluated once per forward and the
    result is broadcast to every block — ALiBi-style single schedule, i.e.
    the same additive bias at every depth. This is a deliberate departure
    from per-layer learnable biases; with only 2-3 layers typical here,
    per-layer versions overfit the small training set.

    Returns ``(x, [attn_weights_per_layer])`` so Phase 10 rollout can see
    every layer's distribution.

    Gotcha
    ------
    ``neg_distance_masked`` and ``group_assignment`` are NON-persistent
    buffers. Loading a checkpoint into an encoder built with a different
    ``temporal_layout`` or ``group_assignment`` silently uses the newly
    constructed geometry, not the one at save time. Always build these
    through ``embedding.build_*`` helpers so drift is impossible.
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
        # Both bias modules are owned (not just referenced) — nn.Module
        # auto-registers them so their parameters appear in .parameters()
        # and their state_dict keys persist in the checkpoint.
        self.temporal_decay = temporal_decay
        self.feature_group_bias = feature_group_bias

    def _compose_attn_bias(self) -> torch.Tensor | None:
        """Sum every active bias into a single additive tensor.

        Mixing shapes works by PyTorch's broadcasting rules: a ``(T, T)``
        scalar-mode bias broadcasts cleanly against a ``(H, T, T)``
        per-head one, so a scalar N3 plus per-head N2 (or vice versa)
        composes correctly with no explicit reshape.
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
        # Explicit accumulator loop rather than `sum(biases)` — `sum`
        # starts from int(0) and would type-error on the first tensor add.
        out = biases[0]
        for b in biases[1:]:
            out = out + b
        return out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the stack. Returns ``(x_out, [attn per layer])``."""
        # Compose once and reuse — bias modules don't depend on x, so
        # recomputing per block would be wasted work.
        attn_bias = self._compose_attn_bias()

        attn_weights_per_layer: list[torch.Tensor] = []
        for block in self.blocks:
            x, attn_weights = block(x, attn_bias=attn_bias)
            attn_weights_per_layer.append(attn_weights)
        return x, attn_weights_per_layer


if __name__ == "__main__":
    # Smoke test. Three sections:
    #   A: plain encoder (no bias) — shapes, attention-row sums, gradients
    #   B: scalar temporal decay — alpha=0 must be a no-op; alpha=5 must
    #      actually suppress distant-month attention; alpha must get grads
    #   C: mode='off' returns None; mode='per_head' has the right shape
    import sys

    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            try:
                _s.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass

    B, seq_len, d_model, n_heads, n_layers = 4, 24, 64, 4, 2

    from ..tokenization.embedding import build_temporal_layout

    layout = build_temporal_layout()

    # Drift guard: if someone reshuffles TOKEN_ORDER in embedding.py and
    # forgets to update this test, it fires loudly instead of silently
    # passing against a different layout.
    EXPECTED_LAYOUT = {
        "pay": {"positions": [4, 5, 6, 7, 8, 9], "months": [0, 1, 2, 3, 4, 5]},
        "bill": {"positions": [12, 13, 14, 15, 16, 17], "months": [0, 1, 2, 3, 4, 5]},
        "pay_amt": {"positions": [18, 19, 20, 21, 22, 23], "months": [0, 1, 2, 3, 4, 5]},
    }
    assert layout == EXPECTED_LAYOUT, (
        f"TOKEN_ORDER drift detected: build_temporal_layout() returned "
        f"{layout} but the canonical expected layout is {EXPECTED_LAYOUT}."
    )

    # A: plain encoder, no bias -----------------------------------------------
    torch.manual_seed(0)
    x = torch.randn(B, seq_len, d_model)

    torch.manual_seed(42)
    encoder = TransformerEncoder(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=0.0,
        temporal_decay=None,
    )
    encoder.eval()

    out, weights = encoder(x)

    assert out.shape == (B, seq_len, d_model), f"output shape {out.shape}"
    assert len(weights) == n_layers, f"expected {n_layers} attn tensors, got {len(weights)}"
    for w in weights:
        assert w.shape == (B, n_heads, seq_len, seq_len), f"attn shape {w.shape}"
        row_sums = w.sum(dim=-1)
        assert torch.allclose(
            row_sums, torch.ones_like(row_sums), atol=1e-5
        ), "Attention rows don't sum to 1"

    encoder.train()
    out_train, _ = encoder(x)
    out_train.sum().backward()
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"

    print("A: no bias")
    print(f"  out:  {out.shape}")
    print(f"  attn: {len(weights)} layers, {weights[0].shape}")
    print("  grads flow OK")

    # B: scalar TemporalDecayBias --------------------------------------------
    decay_check = TemporalDecayBias(layout, seq_len=seq_len, mode="scalar")
    bias_at_zero = decay_check()
    # alpha zero-init must yield the no-op bias — otherwise "N3 off at
    # step 0" is silently different from "N3 disabled".
    assert torch.equal(
        bias_at_zero, torch.zeros(seq_len, seq_len)
    ), "alpha=0 should produce a zero bias matrix"

    torch.manual_seed(42)
    decay = TemporalDecayBias(layout, seq_len=seq_len, mode="scalar")
    encoder_b = TransformerEncoder(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=0.0,
        temporal_decay=decay,
    )
    encoder_b.eval()

    # Crank alpha up so the penalty clearly dominates. At alpha=5 the
    # (PAY_0, PAY_6) pair gets a -25 logit vs (PAY_0, PAY_2) at -5 — the
    # softmax has to push the distant-month weight below the near one.
    with torch.no_grad():
        decay.alpha.fill_(5.0)
    _, weights_decayed = encoder_b(x)
    w0 = weights_decayed[0]

    pay0_to_pay2 = w0[:, :, 4, 5]
    pay0_to_pay6 = w0[:, :, 4, 9]
    assert (
        pay0_to_pay6 < pay0_to_pay2
    ).all(), "Temporal decay should make PAY_0 -> PAY_6 weaker than PAY_0 -> PAY_2"

    # alpha must be trainable — zero-gradient here would mean the prior
    # is stuck at zero-init forever (silent N3 disable).
    with torch.no_grad():
        decay.alpha.fill_(0.5)
    encoder_b.train()
    out_grad, _ = encoder_b(x)
    out_grad.sum().backward()
    assert decay.alpha.grad is not None, "alpha has no .grad attribute"
    assert decay.alpha.grad.abs().item() > 0, "alpha gradient is exactly zero"

    print("\nB: scalar decay")
    print("  alpha=0 -> 0 bias            OK")
    print("  alpha=5 kills distant-month  OK")
    print("  alpha gets grad              OK")

    # C: other modes ---------------------------------------------------------
    decay_off = TemporalDecayBias(layout, seq_len=seq_len, mode="off")
    assert decay_off() is None, "mode='off' should return None"

    decay_ph = TemporalDecayBias(layout, seq_len=seq_len, mode="per_head", n_heads=n_heads)
    bias_ph = decay_ph()
    assert bias_ph.shape == (n_heads, seq_len, seq_len), f"per-head bias shape {bias_ph.shape}"

    encoder_ph = TransformerEncoder(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=0.0,
        temporal_decay=decay_ph,
    )
    encoder_ph.eval()
    out_ph, _ = encoder_ph(x)
    assert out_ph.shape == (B, seq_len, d_model)

    print("\nC: modes")
    print("  off -> None           OK")
    print(f"  per_head {bias_ph.shape} OK")
    print("  encoder runs          OK")

    print("\nall checks passed.")
