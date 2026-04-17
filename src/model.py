"""
model.py — TabularTransformer: top-level end-to-end model.

Wires the whole stack in one place:

    batch  ─▶  FeatureEmbedding      ─▶  (B, 24, d)
           ─▶  TransformerEncoder    ─▶  (B, 24, d) + per-layer attn
           ─▶  pool (CLS | mean | max)  ─▶  (B, d)
           ─▶  LayerNorm + 2-layer MLP head  ─▶  (B,) logit

Optional heads / returns:
    ∘ ``aux_pay0=True``  →  parallel 11-class head reading the PAY_0 token
                            after it has been forced-masked. This is
                            Plan §8.6 / Novelty N5 (multi-task auxiliary
                            PAY-forecast). Target supplied by tokenizer's
                            ``pay_raw`` (shifted into [0, 10]).
    ∘ ``return_attn``    →  include per-layer attention weights in the
                            forward output (for Phase 10 attention rollout).

Architectural switches that cover the relevant ablations:
    ∘ ``d_model``, ``n_heads``, ``n_layers``, ``d_ff``  →  A2, A3, A4
    ∘ ``pool ∈ {cls, mean, max}``                        →  A5
    ∘ ``use_temporal_pos``                               →  A7
    ∘ ``temporal_decay_mode ∈ {off, scalar, per_head}``  →  A22 / Novelty N3
    ∘ independently-ablatable attn/ffn/residual dropout  →  A12
    ∘ ``use_mask_token``                                 →  enables MTLM
                                                            pretraining (N4)
    ∘ ``aux_pay0``                                       →  N5 / A16

References (Plan sections): §6.7 classification head, §6.9 forward diagram,
§6.10 module layout, §6.11 `src/transformer.py` spec, §8.5.5 fine-tuning
protocol, §8.6 multi-task auxiliary (N5).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn as nn

from embedding import (
    FeatureEmbedding,
    N_FEATURE_GROUPS,
    build_group_assignment,
    build_temporal_layout,
)
from tokenizer import PAY_RAW_NUM_CLASSES
from transformer import FeatureGroupBias, TemporalDecayBias, TransformerEncoder

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants tied to the canonical TOKEN_ORDER in embedding.py. Using these
# instead of hard-coded integers makes the auxiliary-head wiring drift-safe
# under any future TOKEN_ORDER reordering (which would also fail the
# drift-detection guard in transformer.py's smoke test).
# ──────────────────────────────────────────────────────────────────────────────

#: Position of PAY_0 in the 23-feature block *before* [CLS] is prepended.
#: Used by the MTLM mask path (which works on the 23-block) and by the
#: force-mask that the ``aux_pay0`` head relies on.
PAY_0_FEATURE_POSITION_23 = 3

#: Position of PAY_0 in the full 24-token output sequence (CLS at index 0).
#: Used to slice the encoder's hidden state when reading the PAY_0 token
#: for the auxiliary forecast head.
PAY_0_OUTPUT_POSITION_24 = PAY_0_FEATURE_POSITION_23 + 1

#: Total number of token positions the encoder sees (CLS + 23 feature tokens).
FULL_SEQ_LEN = 24


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────


class TabularTransformer(nn.Module):
    """
    End-to-end transformer for the Credit Card Default task.

    Parameters
    ----------
    d_model, n_heads, n_layers, d_ff
        Encoder hyperparameters. Defaults match Plan §6.11 (~28K encoder
        parameters + ~1.2K classification head ≈ 30K total, appropriate for
        21K training rows).
    dropout
        Shared dropout default for the attention-weight, FFN-hidden, and
        residual channels in every ``TransformerBlock``.
    attn_dropout, ffn_dropout, residual_dropout
        Optional per-channel overrides (Plan §6.3 / Ablation A12). Unset
        channels inherit ``dropout``.
    classification_dropout
        Dropout between the two linear layers of the classification head
        (and of the ``aux_pay0_head`` when present).
    pool
        Aggregation of the encoder's (B, 24, d) output into (B, d):

        * ``"cls"``   — take the CLS token (position 0). BERT convention.
        * ``"mean"``  — mean over the 23 feature tokens (CLS excluded so
          it cannot dominate by magnitude).
        * ``"max"``   — element-wise max over the 23 feature tokens.

        Plan Ablation A5 sweeps all three.
    use_temporal_pos
        If True, the :class:`FeatureEmbedding` adds a learnable
        ``nn.Embedding(6, d_model)`` to every PAY / BILL_AMT / PAY_AMT
        token. Plan §5.4 / Ablation A7.
    use_mask_token
        If True, enables the ``[MASK]`` embedding path inside
        :class:`FeatureEmbedding`. Required for MTLM pretraining (Plan §8.5
        / Novelty N4). Implied-True whenever ``aux_pay0=True`` because the
        aux head force-masks PAY_0.
    temporal_decay_mode
        Selects the :class:`TemporalDecayBias` variant (Plan §6.12.2 /
        Novelty N3 / Ablation A22):

        * ``"off"``       — no bias. Plain FT-Transformer.
        * ``"scalar"``    — single learnable α (default for A22 on).
        * ``"per_head"``  — one α per attention head.

        The temporal layout is derived from :func:`embedding.build_temporal_layout`
        so it cannot silently drift from the canonical TOKEN_ORDER.
    feature_group_bias_mode
        Selects the :class:`FeatureGroupBias` variant (Plan §6.12.1 /
        Novelty N2 / Ablation A21):

        * ``"off"``       — no group bias.
        * ``"scalar"``    — one 5×5 learnable bias matrix shared across heads.
        * ``"per_head"``  — one 5×5 bias matrix per attention head.

        Groups are: [CLS], demographic, PAY, BILL_AMT, PAY_AMT. The
        assignment is derived from :func:`embedding.build_group_assignment`
        so it stays drift-safe under TOKEN_ORDER changes. Bias matrix
        initialised to zero — model activates the prior only if it helps.
    aux_pay0
        If True, attach a parallel 11-class classification head that
        predicts the raw PAY_0 value from the encoder's PAY_0 token
        after the token has been force-masked. Plan §8.6 / Novelty N5 /
        Ablation A16. Target is provided in the batch as ``pay_raw[:, 0]``
        (values in [0, 10]; see tokenizer.py).
    cat_vocab_sizes
        Optional override for ``FeatureEmbedding`` — primarily for
        hermetic unit tests (no disk I/O).

    Forward output
    --------------
    ``forward(batch, *, return_attn=False)`` returns a dict with keys:

    * ``"logit"`` — (B,) unnormalised default logit (losses consume logits;
      no sigmoid applied).
    * ``"aux_pay0_logits"`` — (B, 11) present only if ``aux_pay0=True``.
    * ``"attn_weights"`` — ``list[Tensor (B, n_heads, 24, 24)]`` of length
      ``n_layers``, present only if ``return_attn=True``.

    References
    ----------
    Plan §6.7 / §6.11 for the head design (LayerNorm → Linear → GELU →
    Dropout → Linear → sigmoid via BCEWithLogits-family loss). Plan §6.8
    for the weight-initialisation policy (Kaiming for GELU-feeding linears,
    Xavier for residual-feeding linears, zeros for biases).
    """

    def __init__(
        self,
        *,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        attn_dropout: Optional[float] = None,
        ffn_dropout: Optional[float] = None,
        residual_dropout: Optional[float] = None,
        classification_dropout: float = 0.1,
        pool: Literal["cls", "mean", "max"] = "cls",
        use_temporal_pos: bool = False,
        use_mask_token: bool = False,
        temporal_decay_mode: Literal["off", "scalar", "per_head"] = "off",
        feature_group_bias_mode: Literal["off", "scalar", "per_head"] = "off",
        aux_pay0: bool = False,
        cat_vocab_sizes: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        if pool not in ("cls", "mean", "max"):
            raise ValueError(f"pool must be one of 'cls'/'mean'/'max', got {pool!r}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.pool = pool
        self.aux_pay0 = aux_pay0

        # Embedding. The aux_pay0 path force-masks PAY_0 so it needs the
        # [MASK] embedding machinery — we implicitly enable use_mask_token
        # in that case even if the caller forgot.
        self.embedding = FeatureEmbedding(
            d_model=d_model,
            dropout=dropout,
            cat_vocab_sizes=cat_vocab_sizes,
            use_temporal_pos=use_temporal_pos,
            use_mask_token=use_mask_token or aux_pay0,
        )

        # Temporal-decay bias (Novelty N3) — derive the layout from TOKEN_ORDER
        # so the within-group distance matrix is drift-safe.
        if temporal_decay_mode != "off":
            temporal_decay: Optional[TemporalDecayBias] = TemporalDecayBias(
                temporal_layout=build_temporal_layout(cls_offset=1),
                seq_len=FULL_SEQ_LEN,
                mode=temporal_decay_mode,
                n_heads=n_heads,
            )
        else:
            temporal_decay = None

        # Feature-group attention bias (Novelty N2) — drift-safe group
        # assignment derived from TOKEN_ORDER.
        if feature_group_bias_mode != "off":
            feature_group_bias: Optional[FeatureGroupBias] = FeatureGroupBias(
                group_assignment=build_group_assignment(cls_offset=1),
                n_groups=N_FEATURE_GROUPS,
                mode=feature_group_bias_mode,
                n_heads=n_heads,
            )
        else:
            feature_group_bias = None

        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            dropout=dropout,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
            residual_dropout=residual_dropout,
            temporal_decay=temporal_decay,
            feature_group_bias=feature_group_bias,
        )

        # Classification head (Plan §6.7): final LayerNorm on the pooled
        # representation, then a 2-layer MLP projecting to a single logit.
        self.head_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),     # GELU-feeding — Kaiming
            nn.GELU(),
            nn.Dropout(classification_dropout),
            nn.Linear(d_model, 1),           # output — Xavier
        )

        # Auxiliary PAY_0 forecast head (Novelty N5). Symmetric to the main
        # head but with its own LayerNorm so its representation space is
        # decoupled from the primary classifier.
        if aux_pay0:
            self.aux_pay0_head: Optional[nn.Sequential] = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),                 # GELU-feeding — Kaiming
                nn.GELU(),
                nn.Dropout(classification_dropout),
                nn.Linear(d_model, PAY_RAW_NUM_CLASSES),     # 11-class — Xavier
            )
        else:
            self.aux_pay0_head = None

        self._init_heads()

    # ------------------------------------------------------------------ init
    def _init_heads(self) -> None:
        """Initialise both heads per Plan §6.8:
        GELU-feeding linears get Kaiming (He) init; final/residual-feeding
        linears get Xavier; biases are zero. LayerNorm uses PyTorch defaults
        (γ=1, β=0)."""
        for container in (self.classifier, self.aux_pay0_head):
            if container is None:
                continue
            linears = [m for m in container if isinstance(m, nn.Linear)]
            for idx, lin in enumerate(linears):
                if idx < len(linears) - 1:
                    nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_normal_(lin.weight)
                nn.init.zeros_(lin.bias)

    # ----------------------------------------------------------------- pool
    def _pool(self, hidden: torch.Tensor) -> torch.Tensor:
        """(B, 24, d) → (B, d) by CLS / mean / max aggregation."""
        if self.pool == "cls":
            return hidden[:, 0, :]
        # Exclude CLS (index 0) from mean/max so it cannot dominate.
        feature_hidden = hidden[:, 1:, :]
        if self.pool == "mean":
            return feature_hidden.mean(dim=1)
        # pool == "max"
        return feature_hidden.max(dim=1).values

    # ----------------------------------------------------------- _force_mask
    def _apply_aux_force_mask(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """For ``aux_pay0=True`` force PAY_0 to be masked (OR-ed with any
        existing mask_positions). The aux head then predicts PAY_0 from the
        surrounding context via the forced [MASK] embedding."""
        B = batch["num_values"].shape[0]
        device = batch["num_values"].device
        force = torch.zeros(B, 23, dtype=torch.bool, device=device)
        force[:, PAY_0_FEATURE_POSITION_23] = True
        existing = batch.get("mask_positions")
        if existing is not None:
            force = existing.to(device) | force
        return {**batch, "mask_positions": force}

    # --------------------------------------------------------------- forward
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        return_attn: bool = False,
    ) -> Dict[str, Any]:
        if self.aux_pay0:
            batch = self._apply_aux_force_mask(batch)

        tokens = self.embedding(batch)                       # (B, 24, d)
        hidden, attn_weights = self.encoder(tokens)          # (B, 24, d), list[L]
        pooled = self._pool(hidden)                          # (B, d)
        logit = self.classifier(self.head_norm(pooled)).squeeze(-1)  # (B,)

        out: Dict[str, Any] = {"logit": logit}

        if self.aux_pay0:
            assert self.aux_pay0_head is not None  # for the type checker
            pay0_hidden = hidden[:, PAY_0_OUTPUT_POSITION_24, :]         # (B, d)
            out["aux_pay0_logits"] = self.aux_pay0_head(pay0_hidden)     # (B, 11)

        if return_attn:
            out["attn_weights"] = attn_weights

        return out

    # -------------------------------------------------------- accounting API
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Total (trainable) parameter count. At plan defaults (d_model=32,
        n_heads=4, n_layers=2, d_ff=128, dropout=0.1, pool='cls',
        temporal_decay=off, feature_group_bias=off, aux_pay0=False) expect
        ~28K parameters — matches the Plan §6.9 budget of ~28,000 params
        for 21K training rows."""
        return sum(
            p.numel()
            for p in self.parameters()
            if (not trainable_only) or p.requires_grad
        )

    def parameter_count_by_module(
        self, trainable_only: bool = True,
    ) -> Dict[str, int]:
        """Return a dict mapping top-level sub-module name → parameter count.

        Useful for:
        * Sanity-checking the ~28K Plan §6.9 budget by module contribution.
        * Driving the parameter-breakdown plots in the training notebook.
        * Ensuring the encoder carries the bulk of parameters (not the head).

        Params NOT contained in any named sub-module (e.g. buffers, naked
        ``nn.Parameter`` attributes on ``self``) are grouped under ``"_leaf"``.
        """
        counts: Dict[str, int] = {}
        # Only count Parameters, not buffers.
        seen_param_ids: set[int] = set()
        for name, module in self.named_children():
            if module is None:
                continue
            total = 0
            for p in module.parameters():
                if (not trainable_only) or p.requires_grad:
                    total += p.numel()
                    seen_param_ids.add(id(p))
            if total:
                counts[name] = total
        # Leaf params directly on self.
        leaf = 0
        for p in self.parameters(recurse=False):
            if id(p) in seen_param_ids:
                continue
            if (not trainable_only) or p.requires_grad:
                leaf += p.numel()
        if leaf:
            counts["_leaf"] = leaf
        return counts

    # ------------------------------------------------- parameter-group APIs
    def get_head_params(self) -> List[torch.nn.Parameter]:
        """Return the parameters of the fresh classification head and
        (if present) the auxiliary PAY_0 forecast head.

        Used by ``train.build_optimizer`` to assign the *peak* learning rate
        to the freshly-initialised head parameters during MTLM → supervised
        fine-tuning (Plan §8.5.5). The embedding + encoder parameters get
        the smaller ``lr * encoder_lr_ratio``.
        """
        params: List[torch.nn.Parameter] = []
        for module in (self.head_norm, self.classifier, self.aux_pay0_head):
            if module is None:
                continue
            params.extend(module.parameters())
        return params

    def get_encoder_params(self) -> List[torch.nn.Parameter]:
        """Return the parameters of everything that *isn't* the classification
        head — i.e. the embedding + encoder + any novelty bias modules. This
        is the "pretrained" side of the Plan §8.5.5 two-stage optimiser."""
        head_ids = {id(p) for p in self.get_head_params()}
        return [p for p in self.parameters() if id(p) not in head_ids]

    # ---------------------------------------------------- architecture summary
    def summary(self) -> str:
        """Return a human-readable multi-line summary of the model's
        configuration, parameter budget, and active novelty switches.

        Intended for notebook / report inclusion and as a sanity check when
        swapping ablation configurations. The string is constructed entirely
        from live attributes — if a switch is inadvertently misconfigured,
        ``summary()`` will show it. Does not re-create or mutate any
        parameters.
        """
        n_total = self.count_parameters()
        breakdown = self.parameter_count_by_module()
        lines: List[str] = []
        lines.append("TabularTransformer — architecture summary")
        lines.append("-" * 56)
        lines.append(f"  d_model           : {self.d_model}")
        lines.append(f"  n_heads           : {self.n_heads}")
        lines.append(f"  n_layers          : {self.n_layers}")
        lines.append(f"  pool              : {self.pool}")
        td = self.encoder.temporal_decay
        fg = self.encoder.feature_group_bias
        lines.append(f"  temporal_decay    : {td.mode if td is not None else 'off'}")
        lines.append(f"  feature_group_bias: {fg.mode if fg is not None else 'off'}")
        lines.append(f"  use_temporal_pos  : {self.use_temporal_pos}")
        lines.append(f"  aux_pay0 head     : {self.aux_pay0}")
        lines.append("")
        lines.append("Parameter breakdown (trainable):")
        for name, count in sorted(breakdown.items(), key=lambda kv: -kv[1]):
            pct = 100.0 * count / max(1, n_total)
            lines.append(f"  {name:18s} {count:>8,d}  ({pct:5.1f}%)")
        lines.append("-" * 56)
        lines.append(f"  Total             : {n_total:,d} parameters")
        plan_lo, plan_hi = 20_000, 40_000
        in_plan = plan_lo <= n_total <= plan_hi
        status = "✓ within Plan §6.9 envelope" if in_plan else "⚠ outside Plan §6.9 envelope"
        lines.append(f"  Plan §6.9 budget  : ~28,000  →  {status}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # noqa: D401 — override the long torch repr
        return (
            f"TabularTransformer(d_model={self.d_model}, n_heads={self.n_heads}, "
            f"n_layers={self.n_layers}, pool={self.pool!r}, "
            f"temporal_decay={self.encoder.temporal_decay.mode if self.encoder.temporal_decay else 'off'!r}, "
            f"feature_group_bias={self.encoder.feature_group_bias.mode if self.encoder.feature_group_bias else 'off'!r}, "
            f"aux_pay0={self.aux_pay0}, "
            f"params={self.count_parameters():,})"
        )

    # ---------------------------------------------------- pretrained loading
    def load_pretrained_encoder(
        self,
        checkpoint_path: str | os.PathLike[str],
        *,
        strict: bool = False,
        trust_source: bool = False,
        map_location: Optional[torch.device | str] = None,
    ) -> Dict[str, Any]:
        """
        Load a previously-trained state dict into this model.

        Intended for the MTLM → supervised fine-tuning transition (Plan
        §8.5.5). The MTLM pretrain checkpoint contains ``embedding.*`` and
        ``encoder.*`` keys (plus MTLM-head keys that this model doesn't
        have); with ``strict=False`` the matching keys are applied and the
        extras are ignored silently. The ``classifier``,  ``head_norm``,
        and optional ``aux_pay0_head`` stay at their fresh initialisation
        — per the plan, the supervised objective re-learns the head.

        Use the caller-side two-stage learning-rate recipe from Plan §8.5.5
        (peak LR for the head, ~5× smaller for the pretrained encoder) in
        ``train.py``; this method does not touch the optimiser.

        Parameters mirror ``utils.load_checkpoint``:

        * ``strict``        — ``False`` is the intended default here.
        * ``trust_source``  — set ``True`` only for checkpoints you
          produced yourself. Default uses the weights-only sidecar
          (no pickle execution).
        * ``map_location``  — forwarded to ``torch.load``.
        """
        from utils import load_checkpoint  # local import to avoid circularity

        checkpoint = load_checkpoint(
            checkpoint_path,
            self,
            strict=strict,
            trust_source=trust_source,
            map_location=map_location,
        )
        logger.info(
            "TabularTransformer: loaded pretrained encoder from %s "
            "(classification head left at fresh init)",
            checkpoint_path,
        )
        return checkpoint


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test — runs only with preprocessing outputs available.
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys

    # UTF-8 stdout so box-drawing separators print cleanly on Windows.
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass

    import pandas as pd

    from dataset import make_loader
    from tokenizer import CreditDefaultDataset, build_categorical_vocab

    root = Path(__file__).resolve().parent.parent
    meta_path = root / "data/processed/feature_metadata.json"
    csv_path = root / "data/processed/train_scaled.csv"
    if not (meta_path.is_file() and csv_path.is_file()):
        print(
            "[SKIP] model.py smoke test requires preprocessing output.\n"
            "       Run `poetry run python run_pipeline.py --preprocess-only` "
            "first to materialise data/processed/*.csv."
        )
        sys.exit(0)

    meta = json.loads(meta_path.read_text())
    df = pd.read_csv(csv_path).head(256)
    cat_vocab = build_categorical_vocab(meta)
    ds = CreditDefaultDataset(df, cat_vocab, verbose=False)
    loader = make_loader(ds, batch_size=32, mode="val")

    # ── A: plan-default configuration (~28K params, no aux head) ─────────
    torch.manual_seed(0)
    model = TabularTransformer()
    model.eval()
    batch = next(iter(loader))
    with torch.no_grad():
        out = model(batch, return_attn=True)
    n_params = model.count_parameters()
    print(f"── A: plan defaults (d=32, h=4, L=2) ──")
    print(f"  Logit shape:        {tuple(out['logit'].shape)}")
    print(f"  Attn layers:        {len(out['attn_weights'])}")
    print(f"  Per-layer attn:     {tuple(out['attn_weights'][0].shape)}")
    print(f"  Param count:        {n_params:,} (~28K expected)")
    assert out["logit"].shape == (32,)
    assert len(out["attn_weights"]) == 2
    assert 20_000 <= n_params <= 40_000, (
        f"param count {n_params:,} outside the [20K, 40K] plan envelope"
    )

    # ── B: gradient flow through every parameter ──────────────────────────
    model.train()
    grad_out = model(batch)
    grad_out["logit"].sum().backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert not no_grad, f"no grad on {no_grad}"
    print("\n── B: gradient flow ──")
    print(f"  All {sum(1 for _ in model.parameters())} param groups receive gradient ✓")

    # ── C: pool modes ──────────────────────────────────────────────────────
    print("\n── C: pool modes ──")
    for pool in ("cls", "mean", "max"):
        m = TabularTransformer(pool=pool)
        m.eval()
        with torch.no_grad():
            logit = m(batch)["logit"]
        assert logit.shape == (32,), pool
        print(f"  pool={pool}: logit shape {tuple(logit.shape)} ✓")

    # ── D: temporal decay + aux_pay0 + multi-flag composition ─────────────
    torch.manual_seed(0)
    m = TabularTransformer(
        d_model=32, n_heads=4, n_layers=2,
        use_temporal_pos=True,
        temporal_decay_mode="scalar",
        aux_pay0=True,
    )
    m.train()
    out = m(batch, return_attn=True)
    assert "aux_pay0_logits" in out
    assert out["aux_pay0_logits"].shape == (32, PAY_RAW_NUM_CLASSES)
    print("\n── D: temporal_pos + temporal_decay + aux_pay0 ──")
    print(f"  Primary logit shape:    {tuple(out['logit'].shape)}")
    print(f"  Aux PAY_0 logits shape: {tuple(out['aux_pay0_logits'].shape)}")
    print(f"  TemporalDecayBias α:    {m.encoder.temporal_decay.alpha.item():.4f}")

    # ── E: aux_pay0 gradient flows through the aux head ──────────────────
    loss = out["aux_pay0_logits"].sum()
    loss.backward()
    assert m.aux_pay0_head is not None
    for name, p in m.aux_pay0_head.named_parameters():
        assert p.grad is not None, f"no grad on aux head {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on aux head {name}"
    print(f"  Aux head gradient flow: OK ({sum(1 for _ in m.aux_pay0_head.parameters())} param groups)")

    # ── F: deterministic re-run ───────────────────────────────────────────
    torch.manual_seed(1)
    m1 = TabularTransformer(d_model=32, n_heads=4, n_layers=2)
    torch.manual_seed(1)
    m2 = TabularTransformer(d_model=32, n_heads=4, n_layers=2)
    m1.eval()
    m2.eval()
    with torch.no_grad():
        l1 = m1(batch)["logit"]
        l2 = m2(batch)["logit"]
    assert torch.allclose(l1, l2, atol=1e-6)
    print("\n── F: deterministic under identical seed ✓")

    print("\nAll TabularTransformer smoke checks passed.")
