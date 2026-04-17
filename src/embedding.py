"""
embedding.py — Feature embedding layer for the Credit Card Default Transformer.

Takes the tokenizer output (cat_indices, pay_state_ids, pay_severities,
num_values) and converts it into a sequence of 24 token embeddings of shape
(B, 24, d_model).

Token order (fixed, determines positional index):
    Position  0:     [CLS] token                           (learnable, BERT-style)
    Positions 1-3:   SEX, EDUCATION, MARRIAGE              (categorical)
    Positions 4-9:   PAY_0, PAY_2, PAY_3, PAY_4,
                     PAY_5, PAY_6                          (PAY hybrid: state + severity)
    Positions 10-23: LIMIT_BAL, AGE, BILL_AMT1-6,
                     PAY_AMT1-6                            (numerical: identity + value)

Each token is assembled as:
    categorical → cat_embed(local_idx)      + pos_embed(position) + temporal_pos?
    PAY         → pay_state_embed(state_id)
                  + severity_proj(severity) + pos_embed(position) + temporal_pos
    numerical   → num_feat_embed(feat_idx)
                  + value_proj(value)       + pos_embed(position) + temporal_pos?

Optional features (all opt-in via constructor flags, default off for
backward-compatibility):

* **Temporal positional encoding** (Plan §5.4) — a learnable (6, d_model)
  embedding indexed by month (0=most-recent, 5=oldest), added to every
  temporal token (PAY, BILL_AMT, PAY_AMT). Non-temporal tokens are unaffected.

* **[MASK] token** support (Plan §5.4B, Novelty N4) — a learnable ``[MASK]``
  vector that replaces the content embedding (feature identity + value/state)
  at positions flagged by the MTLM collator's ``mask_positions`` tensor. The
  positional embedding is preserved so the model still knows *which* feature
  is masked. Used only during MTLM pretraining.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from tokenizer import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PAY_STATUS_FEATURES,
)

# ──────────────────────────────────────────────────────────────────────────────
# Fixed token order — position in this list + 1 = positional embedding index
# (index 0 is reserved for [CLS], BERT-style).
# ──────────────────────────────────────────────────────────────────────────────
TOKEN_ORDER: List[str] = (
    CATEGORICAL_FEATURES + PAY_STATUS_FEATURES + NUMERICAL_FEATURES
)
# ["SEX","EDUCATION","MARRIAGE","PAY_0","PAY_2",...,"PAY_6",
#  "LIMIT_BAL","AGE","BILL_AMT1",...,"PAY_AMT6"]
# Total: 3 + 6 + 14 = 23 feature tokens  (+1 CLS = 24 output tokens)

# Sequence-position slices — feature-token indices within the (B, 23, d) block
# *before* [CLS] is prepended. The final output has everything shifted by +1.
_SLICE_CAT = slice(0, 3)
_SLICE_PAY = slice(3, 9)
_SLICE_NUM = slice(9, 23)

# ──────────────────────────────────────────────────────────────────────────────
# Temporal layout — which feature tokens carry a month index, and which month
# (0 = September 2005, most recent; 5 = April 2005, oldest).
# ──────────────────────────────────────────────────────────────────────────────
N_MONTHS = 6
_TEMPORAL_MONTH: Dict[str, int] = {}
for _i, _feat in enumerate(PAY_STATUS_FEATURES):
    _TEMPORAL_MONTH[_feat] = _i
for _i, _feat in enumerate([f"BILL_AMT{_m}" for _m in range(1, 7)]):
    _TEMPORAL_MONTH[_feat] = _i
for _i, _feat in enumerate([f"PAY_AMT{_m}" for _m in range(1, 7)]):
    _TEMPORAL_MONTH[_feat] = _i

# Pre-computed per-token temporal index. -1 means non-temporal.
_TEMPORAL_INDEX_PER_TOKEN = torch.tensor(
    [_TEMPORAL_MONTH.get(feat, -1) for feat in TOKEN_ORDER],
    dtype=torch.long,
)


# ──────────────────────────────────────────────────────────────────────────────
# Metadata loading — lazy, so importing this module is side-effect-free and
# works on a fresh clone before preprocessing has generated the JSON.
# ──────────────────────────────────────────────────────────────────────────────

_METADATA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "feature_metadata.json"
)


def load_cat_vocab_sizes(
    metadata_path: Optional[Path] = None,
) -> Dict[str, int]:
    """
    Load categorical vocabulary sizes from ``data/processed/feature_metadata.json``.

    Raises a ``FileNotFoundError`` with an actionable message if preprocessing
    has not yet been run. Prefer passing an explicit ``cat_vocab_sizes`` kwarg
    to :class:`FeatureEmbedding` in unit tests so the constructor is hermetic.

    Parameters
    ----------
    metadata_path
        Override the default path. Primarily useful in tests and in notebooks
        that want to exercise a non-canonical metadata file.
    """
    path = Path(metadata_path) if metadata_path is not None else _METADATA_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"feature_metadata.json not found at {path}. "
            "Run `poetry run python run_pipeline.py --preprocess-only` to "
            "generate preprocessing outputs before instantiating FeatureEmbedding, "
            "or pass an explicit cat_vocab_sizes kwarg."
        )
    with open(path) as f:
        meta = json.load(f)
    return {
        feat: info["n_categories"]
        for feat, info in meta["categorical_features"].items()
    }


def __getattr__(name: str):
    """
    PEP 562 module-level ``__getattr__`` — preserves the legacy
    ``embedding.CAT_VOCAB_SIZES`` attribute while keeping the module import
    itself side-effect-free (no disk I/O unless somebody asks for the dict).
    Cached on first resolution so subsequent accesses are free.
    """
    if name == "CAT_VOCAB_SIZES":
        value = load_cat_vocab_sizes()
        globals()["CAT_VOCAB_SIZES"] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Drift-safe temporal layout helper — the canonical source of truth for
# TemporalDecayBias / FeatureGroupBias and any consumer that needs positions
# of the three temporal groups (PAY / BILL_AMT / PAY_AMT) in the full
# 24-token sequence (CLS at index 0).
# ──────────────────────────────────────────────────────────────────────────────

_TEMPORAL_GROUPS: Dict[str, List[str]] = {
    "pay":     list(PAY_STATUS_FEATURES),
    "bill":    [f"BILL_AMT{_m}" for _m in range(1, 7)],
    "pay_amt": [f"PAY_AMT{_m}" for _m in range(1, 7)],
}


# ──────────────────────────────────────────────────────────────────────────────
# Feature-group assignment — drift-safe source for Novelty N2
# (:class:`transformer.FeatureGroupBias`). The five semantic groups are:
#   0 = [CLS]
#   1 = demographic (SEX / EDUCATION / MARRIAGE / LIMIT_BAL / AGE)
#   2 = PAY status (PAY_0 .. PAY_6)
#   3 = BILL_AMT  (BILL_AMT1 .. BILL_AMT6)
#   4 = PAY_AMT   (PAY_AMT1 .. PAY_AMT6)
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_GROUP_CLS = 0
FEATURE_GROUP_DEMOGRAPHIC = 1
FEATURE_GROUP_PAY = 2
FEATURE_GROUP_BILL = 3
FEATURE_GROUP_PAY_AMT = 4
N_FEATURE_GROUPS = 5

# Human-readable names for each feature group. Used for attention heatmaps,
# attribution panels and any diagnostic code that prefers "PAY" over "2".
# Plan §7.1 — report-appendix visualisations consume this mapping directly.
FEATURE_GROUP_NAMES: Dict[int, str] = {
    FEATURE_GROUP_CLS:         "CLS",
    FEATURE_GROUP_DEMOGRAPHIC: "demographic",
    FEATURE_GROUP_PAY:         "PAY",
    FEATURE_GROUP_BILL:        "BILL_AMT",
    FEATURE_GROUP_PAY_AMT:     "PAY_AMT",
}

_DEMOGRAPHIC_FEATURES = set(CATEGORICAL_FEATURES) | {"LIMIT_BAL", "AGE"}
_BILL_FEATURES = {f"BILL_AMT{_m}" for _m in range(1, 7)}
_PAY_AMT_FEATURES = {f"PAY_AMT{_m}" for _m in range(1, 7)}


def _group_for_feature(feature_name: str) -> int:
    if feature_name in _DEMOGRAPHIC_FEATURES:
        return FEATURE_GROUP_DEMOGRAPHIC
    if feature_name in PAY_STATUS_FEATURES:
        return FEATURE_GROUP_PAY
    if feature_name in _BILL_FEATURES:
        return FEATURE_GROUP_BILL
    if feature_name in _PAY_AMT_FEATURES:
        return FEATURE_GROUP_PAY_AMT
    raise KeyError(f"No group defined for feature {feature_name!r}")


def build_group_assignment(cls_offset: int = 1) -> List[int]:
    """
    Build the drift-safe group-index list consumed by
    :class:`transformer.FeatureGroupBias` (Novelty N2 / Ablation A21).

    Length is ``len(TOKEN_ORDER) + cls_offset`` — 24 by default (one [CLS]
    prepended). Each entry is an integer in ``[0, N_FEATURE_GROUPS)``:

        0  [CLS]
        1  demographic  (SEX / EDUCATION / MARRIAGE / LIMIT_BAL / AGE)
        2  PAY status   (PAY_0 .. PAY_6)
        3  BILL_AMT     (BILL_AMT1 .. BILL_AMT6)
        4  PAY_AMT      (PAY_AMT1 .. PAY_AMT6)

    Derived from :data:`TOKEN_ORDER` so the assignment cannot silently drift
    if a future refactor reorders the canonical feature list.

    Parameters
    ----------
    cls_offset
        Number of CLS-style tokens prepended ahead of the 23 feature tokens.
        Default 1 — matches the 24-slot output of :class:`FeatureEmbedding`.
        Pass ``0`` to get assignments for the pre-CLS 23-block.
    """
    assignment: List[int] = [FEATURE_GROUP_CLS] * cls_offset
    assignment.extend(_group_for_feature(f) for f in TOKEN_ORDER)
    return assignment


def describe_token_layout(cls_offset: int = 1) -> str:
    """
    Return a human-readable multi-line summary of the token layout.

    Useful for notebook pretty-printing, report appendices, and quick
    sanity checks when debugging position-indexed biases (Novelties N2/N3)
    or MTLM mask placement (Novelty N4). Each line describes one position
    in the ``(cls_offset + len(TOKEN_ORDER))``-token output sequence:

        position  group_name       feature_name   month
        --------  ---------------  -------------  -----
              0   CLS              [CLS]          N/A
              1   demographic      SEX            N/A
              2   demographic      EDUCATION      N/A
              ...
              4   PAY              PAY_0          0
              ...
             12   BILL_AMT         BILL_AMT1      0
              ...

    ``month`` is ``"N/A"`` for non-temporal tokens. The layout is derived
    from the canonical :data:`TOKEN_ORDER`, so the output cannot silently
    desynchronise from the live embedding layer.

    Parameters
    ----------
    cls_offset
        Number of CLS-style tokens prepended. Default ``1`` — matches the
        24-token output sequence of :class:`FeatureEmbedding`.
    """
    lines: List[str] = []
    header = f"{'pos':>4}  {'group':<13}  {'feature':<13}  {'month':>5}"
    sep = "-" * len(header)
    lines.append(header)
    lines.append(sep)

    for cls_i in range(cls_offset):
        lines.append(
            f"{cls_i:>4}  {FEATURE_GROUP_NAMES[FEATURE_GROUP_CLS]:<13}  "
            f"{'[CLS]':<13}  {'N/A':>5}"
        )

    for local_i, feat in enumerate(TOKEN_ORDER):
        pos = cls_offset + local_i
        group_id = _group_for_feature(feat)
        group_name = FEATURE_GROUP_NAMES[group_id]
        month = _TEMPORAL_MONTH.get(feat)
        month_str = "N/A" if month is None else str(month)
        lines.append(f"{pos:>4}  {group_name:<13}  {feat:<13}  {month_str:>5}")

    return "\n".join(lines)


def build_temporal_layout(
    cls_offset: int = 1,
) -> Dict[str, Dict[str, List[int]]]:
    """
    Derive the ``temporal_layout`` dict expected by
    ``transformer.TemporalDecayBias`` from this module's canonical
    :data:`TOKEN_ORDER`. Uses this helper keeps callers drift-free against
    any reordering of ``TOKEN_ORDER``; hard-coded position lists in other
    modules will silently desync if the order changes.

    Parameters
    ----------
    cls_offset
        Offset applied to every position index. Default ``1`` — positions
        address the full 24-token output sequence with ``[CLS]`` at index 0.
        Pass ``0`` to get positions in the 23-feature-token sequence
        (no CLS prepended), e.g. for direct indexing into the pre-CLS
        ``feature_tokens`` tensor.

    Returns
    -------
    Dict mapping group name ∈ {"pay", "bill", "pay_amt"} to a
    ``{"positions": List[int], "months": List[int]}`` record. Months are
    0-indexed (0 = most recent, 5 = oldest). Positions are in the order
    given by :data:`TOKEN_ORDER`.
    """
    layout: Dict[str, Dict[str, List[int]]] = {}
    for group_name, feat_list in _TEMPORAL_GROUPS.items():
        layout[group_name] = {
            "positions": [cls_offset + TOKEN_ORDER.index(f) for f in feat_list],
            "months":    list(range(len(feat_list))),
        }
    return layout


class FeatureEmbedding(nn.Module):
    """
    Convert the tokenizer output into a sequence of token embeddings.

    Input (from :class:`tokenizer.CreditDefaultDataset` / :class:`DataLoader`):
        cat_indices    — dict {feat: (B,) long tensor}
        pay_state_ids  — (B, 6) long tensor
        pay_severities — (B, 6) float tensor
        num_values     — (B, 14) float tensor
        mask_positions — (optional) (B, 23) bool, True where MTLM masks the token

    Output: tensor of shape (B, 24, d_model).

    Parameters
    ----------
    d_model
        Token embedding dimension. Default ``32`` matches Plan §6.11.
    dropout
        Dropout applied after LayerNorm at the output.
    cat_vocab_sizes
        Optional explicit ``{feature_name: vocabulary_size}`` mapping for the
        categorical features. Defaults to lazily loading from
        ``data/processed/feature_metadata.json``; passing the dict explicitly
        makes unit tests hermetic and skips disk I/O.
    use_temporal_pos
        If True, add a learnable temporal-position embedding
        (``nn.Embedding(6, d)``) to every temporal token (PAY, BILL_AMT,
        PAY_AMT). Non-temporal tokens (CLS, SEX, EDUCATION, MARRIAGE,
        LIMIT_BAL, AGE) are unaffected. Plan §5.4 — tested in Ablation A7.
    use_mask_token
        If True, create a learnable ``[MASK]`` embedding for MTLM pretraining
        (Plan §5.4B / Phase 6A / Novelty N4). When forward() receives a batch
        containing ``mask_positions``, the *content* component at flagged
        slots is replaced by the shared ``[MASK]`` vector; both the feature
        positional embedding **and** the temporal positional embedding (if
        active) are preserved so the model still knows *where* the masked
        slot sits in the sequence and in time. No effect when absent from
        the batch.
    """

    def __init__(
        self,
        d_model: int = 32,
        dropout: float = 0.1,
        *,
        cat_vocab_sizes: Optional[Dict[str, int]] = None,
        use_temporal_pos: bool = False,
        use_mask_token: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_temporal_pos = use_temporal_pos
        self.use_mask_token = use_mask_token

        # Lazily resolve vocabulary sizes so that importing this module is
        # side-effect-free even on a fresh clone where preprocessing has not
        # yet run. Tests can override the dict to skip disk I/O entirely.
        if cat_vocab_sizes is None:
            cat_vocab_sizes = load_cat_vocab_sizes()
        missing = set(CATEGORICAL_FEATURES) - cat_vocab_sizes.keys()
        if missing:
            raise KeyError(
                f"cat_vocab_sizes is missing entries for {sorted(missing)}; "
                f"got {sorted(cat_vocab_sizes.keys())}"
            )
        self.cat_vocab_sizes: Dict[str, int] = dict(cat_vocab_sizes)

        # ── Categorical embeddings (SEX, EDUCATION, MARRIAGE) ──────────────
        self.cat_embeddings = nn.ModuleDict(
            {
                feat: nn.Embedding(cat_vocab_sizes[feat], d_model)
                for feat in CATEGORICAL_FEATURES
            }
        )

        # ── PAY hybrid embeddings ──────────────────────────────────────────
        self.pay_state_embedding = nn.Embedding(4, d_model)
        self.pay_severity_proj = nn.Linear(1, d_model, bias=True)

        # ── Numerical embeddings ───────────────────────────────────────────
        self.num_feature_embedding = nn.Embedding(len(NUMERICAL_FEATURES), d_model)
        self.value_proj = nn.Linear(1, d_model, bias=True)

        # ── Positional embedding over the full 24 sequence positions ───────
        self.pos_embedding = nn.Embedding(len(TOKEN_ORDER) + 1, d_model)

        # ── Temporal positional embedding (optional) ───────────────────────
        if use_temporal_pos:
            self.temporal_pos_embedding = nn.Embedding(N_MONTHS, d_model)
        else:
            self.temporal_pos_embedding = None

        # ── [MASK] token (optional, for MTLM pretraining) ──────────────────
        if use_mask_token:
            self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)
        else:
            self.mask_token = None

        # ── [CLS] token ────────────────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Pre-register feature position indices as a non-persistent buffer.
        self.register_buffer(
            "positions", torch.arange(1, len(TOKEN_ORDER) + 1), persistent=False
        )

        # Temporal-index map for the 23 feature tokens (−1 for non-temporal).
        self.register_buffer(
            "temporal_index_per_token", _TEMPORAL_INDEX_PER_TOKEN.clone(), persistent=False
        )

        # ── Output normalisation + dropout ─────────────────────────────────
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

        self._init_weights()

    # ------------------------------------------------------------------ init

    def _init_weights(self) -> None:
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

        if self.temporal_pos_embedding is not None:
            nn.init.normal_(self.temporal_pos_embedding.weight, mean=0.0, std=0.02)

    # --------------------------------------------------------------- forward

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        B = batch["num_values"].shape[0]
        device = batch["num_values"].device

        all_pos_emb = self.pos_embedding(self.positions)  # (23, d_model)

        # ── Categorical tokens (sequence positions 1-3) ────────────────────
        cat_tokens = torch.stack(
            [
                self.cat_embeddings[feat](batch["cat_indices"][feat])  # (B, d_model)
                + all_pos_emb[pos]
                for pos, feat in enumerate(CATEGORICAL_FEATURES)
            ],
            dim=1,
        )  # (B, 3, d_model)

        # ── PAY tokens (sequence positions 4-9) ────────────────────────────
        n_cat = len(CATEGORICAL_FEATURES)
        state_emb = self.pay_state_embedding(batch["pay_state_ids"])            # (B, 6, d)
        sev_emb = self.pay_severity_proj(batch["pay_severities"].unsqueeze(-1))  # (B, 6, d)
        pay_pos = all_pos_emb[n_cat : n_cat + len(PAY_STATUS_FEATURES)]         # (6, d)
        pay_tokens = state_emb + sev_emb + pay_pos                              # (B, 6, d)

        # ── Numerical tokens (sequence positions 10-23) ────────────────────
        n_cat_pay = n_cat + len(PAY_STATUS_FEATURES)
        feat_idx = torch.arange(len(NUMERICAL_FEATURES), dtype=torch.long, device=device)
        feat_emb = self.num_feature_embedding(feat_idx)                          # (14, d)
        value_emb = self.value_proj(batch["num_values"].unsqueeze(-1))           # (B, 14, d)
        num_pos = all_pos_emb[n_cat_pay:]                                        # (14, d)
        num_tokens = value_emb + feat_emb + num_pos                              # (B, 14, d)

        # Stitch the 23 feature tokens together in fixed order.
        feature_tokens = torch.cat([cat_tokens, pay_tokens, num_tokens], dim=1)  # (B, 23, d)

        # ── Optional: add temporal positional encoding to temporal tokens ──
        # Computed once (zero vector at non-temporal slots); re-used below so
        # the MTLM mask path can subtract and re-add it cleanly.
        if self.temporal_pos_embedding is not None:
            temporal_idx = self.temporal_index_per_token  # (23,)
            is_temporal = (temporal_idx >= 0).unsqueeze(-1).to(feature_tokens.dtype)  # (23, 1)
            # Clamp -1 → 0 so the lookup is safe; gate the result by is_temporal.
            safe_idx = temporal_idx.clamp(min=0)
            temporal_emb = self.temporal_pos_embedding(safe_idx) * is_temporal  # (23, d)
            feature_tokens = feature_tokens + temporal_emb  # broadcast over batch
        else:
            temporal_emb = None

        # ── Optional: swap masked slots with the [MASK] embedding ──────────
        # BERT convention: [MASK] replaces *content* (value + feature identity)
        # while *all* positional signals are preserved. We strip both the
        # feature-positional embedding and the temporal-positional embedding
        # (zero on non-temporal slots) before mask substitution, then restore
        # them — so a masked PAY_0 still carries "position 4, month 0" and a
        # masked PAY_6 still carries "position 9, month 5". This matters for
        # MTLM: without it, masked temporal tokens would lose their month
        # signal while unmasked ones kept it.
        if self.mask_token is not None and "mask_positions" in batch:
            mask_positions = batch["mask_positions"].to(device)
            if mask_positions.shape != (B, 23):
                raise ValueError(
                    f"mask_positions shape must be (B, 23), got {tuple(mask_positions.shape)}"
                )
            pos_emb_23 = all_pos_emb  # (23, d)  — feature positional embeds
            content = feature_tokens - pos_emb_23
            if temporal_emb is not None:
                content = content - temporal_emb
            mask_vec = self.mask_token.unsqueeze(0).unsqueeze(0)  # (1, 1, d)
            mask_bcast = mask_positions.unsqueeze(-1).to(feature_tokens.dtype)  # (B, 23, 1)
            new_content = content * (1.0 - mask_bcast) + mask_vec * mask_bcast
            feature_tokens = new_content + pos_emb_23
            if temporal_emb is not None:
                feature_tokens = feature_tokens + temporal_emb

        # ── Prepend [CLS] at position 0 ────────────────────────────────────
        cls_pos = self.pos_embedding(torch.zeros(1, dtype=torch.long, device=device))  # (1, d)
        cls = (self.cls_token + cls_pos).expand(B, -1, -1)                              # (B, 1, d)

        out = torch.cat([cls, feature_tokens], dim=1)                                    # (B, 24, d)
        return self.dropout(self.norm(out))

    # ----------------------------------------------------- fine-tuning helpers

    def freeze_encoder(self, freeze: bool = True) -> None:
        """
        Freeze (or unfreeze) every encoder-side parameter of this module.

        "Encoder-side" means everything *except* a downstream classification
        head — for :class:`FeatureEmbedding` the model *is* the encoder, so
        this toggles :attr:`requires_grad` on every owned parameter:

            * ``cat_embeddings.*``
            * ``pay_state_embedding``, ``pay_severity_proj``
            * ``num_feature_embedding``, ``value_proj``
            * ``pos_embedding``, ``temporal_pos_embedding`` (if present)
            * ``mask_token`` (if present), ``cls_token``
            * ``norm`` (LayerNorm weights)

        Intended for the two-stage fine-tuning workflow in Plan §8.5.5:
        after loading an MTLM-pretrained checkpoint, a caller may want to
        freeze the encoder for the first few epochs of supervised training
        so the classification head can settle before the encoder weights
        move. Subsequently calling ``freeze_encoder(False)`` re-enables
        gradient flow for the joint-finetuning phase.

        Parameters
        ----------
        freeze
            If True (default), all parameters have ``requires_grad=False``.
            If False, they are re-enabled.
        """
        for param in self.parameters():
            param.requires_grad = not freeze

    def init_from_pretrained_statedict(
        self,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = False,
    ) -> Dict[str, List[str]]:
        """
        Load embedding-relevant tensors from a larger (e.g. full-transformer
        or MTLM) checkpoint state_dict.

        Only keys that correspond to owned parameters/buffers are consulted;
        any additional keys (transformer blocks, classifier heads, optimiser
        state, etc.) are silently filtered out. This makes the helper robust
        to checkpoints produced by arbitrary wrapper models — the caller does
        not need to strip prefixes or hand-curate the subset.

        Keys considered (prefix-matched against owned state-dict entries):

            * ``cat_embeddings.*``
            * ``pay_state_embedding.*``, ``pay_severity_proj.*``
            * ``num_feature_embedding.*``, ``value_proj.*``
            * ``pos_embedding.*``, ``temporal_pos_embedding.*``
            * ``mask_token``, ``cls_token``
            * ``norm.*``

        Also tolerates state dicts saved under a common wrapper prefix such
        as ``"embedding."`` (from ``TabularTransformer.embedding``) or
        ``"feature_embedding."``: a key like ``"embedding.cls_token"`` is
        normalised to ``"cls_token"`` before matching.

        Parameters
        ----------
        state_dict
            Full or partial state dict — any key whose tail matches one of
            our own parameter names (after prefix stripping) is loaded.
        strict
            If True, raise :class:`KeyError` when any of this module's
            parameters has no corresponding tensor in ``state_dict`` (after
            prefix stripping). If False (default), missing keys are
            collected into the report and loading proceeds for the rest.
            Shape mismatches on matched keys always raise, regardless.

        Returns
        -------
        dict
            Small diagnostic report::

                {
                    "loaded":   list[str],   # parameter names we populated
                    "missing":  list[str],   # owned names with no match
                    "unexpected": list[str], # state_dict keys we ignored
                }

            Useful for unit tests and logging during MTLM → finetune
            handovers (Plan §8.5.5).
        """
        own_keys = set(self.state_dict().keys())

        # Candidate prefix strips — try in order, pick the first whose
        # stripped remainder is one of our owned keys. "" must come first so
        # checkpoints saved directly from a FeatureEmbedding skip the
        # prefix-hunt entirely.
        candidate_prefixes = ("", "embedding.", "feature_embedding.", "module.")

        # Map ``owned-name → tensor`` for every incoming key that resolves
        # cleanly to one of our parameters. Duplicate owned names (same key
        # appearing under multiple prefixes) keep the first hit.
        normalised: Dict[str, torch.Tensor] = {}
        unexpected: List[str] = []
        for raw_key, tensor in state_dict.items():
            matched_own: Optional[str] = None
            for prefix in candidate_prefixes:
                if prefix and not raw_key.startswith(prefix):
                    continue
                stripped = raw_key[len(prefix):] if prefix else raw_key
                if stripped in own_keys:
                    matched_own = stripped
                    break
            if matched_own is None:
                unexpected.append(raw_key)
                continue
            if matched_own not in normalised:
                normalised[matched_own] = tensor

        # Load what we have; shape mismatches raise with a clear message.
        loaded: List[str] = []
        for key, tensor in normalised.items():
            own = self.state_dict()[key]
            if own.shape != tensor.shape:
                raise ValueError(
                    f"init_from_pretrained_statedict: shape mismatch for "
                    f"{key!r} (own {tuple(own.shape)} vs incoming "
                    f"{tuple(tensor.shape)})"
                )
            # Use load_state_dict with a single-key dict so nn.Module's
            # native type-coercion / device-placement logic runs (no manual
            # .data.copy_).
            self.load_state_dict({key: tensor}, strict=False)
            loaded.append(key)

        missing = sorted(own_keys - set(loaded))
        if strict and missing:
            raise KeyError(
                f"init_from_pretrained_statedict(strict=True): missing keys "
                f"{missing}"
            )

        return {
            "loaded":     sorted(loaded),
            "missing":    missing,
            "unexpected": sorted(unexpected),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)

    B = 4
    # A mixed batch — non-zero severity on delinquent PAY states exercises the
    # severity-projection path (addresses prior review gap).
    batch: Dict[str, torch.Tensor] = {
        "cat_indices": {
            "SEX":       torch.tensor([0, 1, 0, 1]),
            "EDUCATION": torch.tensor([0, 1, 2, 3]),
            "MARRIAGE":  torch.tensor([0, 1, 2, 0]),
        },
        "pay_state_ids":  torch.tensor(
            [
                [0, 1, 2, 3, 3, 3],     # no_bill, paid, min, three delinquencies
                [3, 3, 2, 1, 0, 0],
                [2, 2, 2, 2, 2, 2],
                [0, 0, 0, 3, 3, 3],
            ],
            dtype=torch.long,
        ),
        "pay_severities": torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.25, 0.50, 0.75],
                [1.0, 0.875, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.375, 0.625, 1.0],
            ],
            dtype=torch.float,
        ),
        "num_values":     torch.randn(B, 14),
        "label":          torch.tensor([0.0, 1.0, 0.0, 1.0]),
    }

    # ── A: baseline (no temporal pos, no mask token) ──
    print("── A: baseline (no temporal, no mask) ──")
    model_a = FeatureEmbedding(d_model=32, dropout=0.0)
    model_a.eval()
    out_a = model_a(batch)
    print(f"  output shape: {tuple(out_a.shape)}")
    assert out_a.shape == (B, 24, 32)
    assert not torch.isnan(out_a).any() and not torch.isinf(out_a).any()

    # Gradient flow — non-zero severity guarantees pay_severity_proj is
    # exercised through its non-bias path. Use a random-weighted loss (not
    # bare .sum()) because LayerNorm's output has zero mean per-token, which
    # makes sum-loss gradients collapse for constant-input parameters like
    # [CLS]. A random dot-product mimics what a downstream classification
    # head does and exercises every parameter.
    torch.manual_seed(1)
    proj = torch.randn_like(model_a(batch))
    model_a.train()
    out_a_train = model_a(batch)
    (out_a_train * proj).sum().backward()
    no_grad = [n for n, p in model_a.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not no_grad, f"zero-grad params: {no_grad}"

    # Specifically verify the severity projection weight received a gradient
    # (this is the row 11 regression from the prior review — old smoke test
    # only passed all-zero severities).
    sev_w_grad = model_a.pay_severity_proj.weight.grad
    assert sev_w_grad is not None and sev_w_grad.abs().sum().item() > 0, (
        "pay_severity_proj not exercised — non-zero severity regressed"
    )
    print(
        f"  pay_severity_proj grad L1: {sev_w_grad.abs().sum().item():.4f} ✓"
    )
    print("  all params have non-zero gradients ✓")

    # ── B: temporal positional encoding ──
    print("\n── B: temporal positional encoding on ──")
    model_b = FeatureEmbedding(d_model=32, dropout=0.0, use_temporal_pos=True)
    model_b.eval()
    out_b = model_b(batch)
    assert out_b.shape == (B, 24, 32)
    assert not torch.isnan(out_b).any()
    # Gradient through the temporal embedding.
    model_b.train()
    model_b(batch).sum().backward()
    assert model_b.temporal_pos_embedding.weight.grad.abs().sum().item() > 0
    print("  temporal_pos_embedding receives gradient ✓")
    # Sanity: temporal embedding is only added to temporal tokens. The CLS
    # token (position 0) and SEX token (position 1) should not see it.
    # We can't easily prove "unchanged" end-to-end because LayerNorm mixes
    # everything, but we can verify the non-temporal indices in the buffer.
    non_temp = (model_b.temporal_index_per_token == -1).nonzero(as_tuple=True)[0]
    # SEX(0), EDUCATION(1), MARRIAGE(2), LIMIT_BAL(9), AGE(10) are non-temporal
    # within the 23-token block. PAY_0..PAY_6 are at 3..8, BILL_AMT1..6 at
    # 11..16, PAY_AMT1..6 at 17..22 — all temporal.
    expected_non_temp = [0, 1, 2, 9, 10]
    assert non_temp.tolist() == expected_non_temp, f"unexpected non-temporal set {non_temp.tolist()}"
    print(f"  non-temporal token indices (in 23-block): {expected_non_temp} ✓")

    # ── C: [MASK] token + mask_positions ──
    print("\n── C: [MASK] token on ──")
    model_c = FeatureEmbedding(d_model=32, dropout=0.0, use_mask_token=True)
    model_c.eval()
    mask_positions = torch.zeros(B, 23, dtype=torch.bool)
    mask_positions[0, 3] = True  # mask row 0's PAY_0 token
    mask_positions[1, 9] = True  # mask row 1's LIMIT_BAL token
    mask_positions[2, 0] = True  # mask row 2's SEX token
    batch_masked = {**batch, "mask_positions": mask_positions}
    out_c = model_c(batch_masked)
    assert out_c.shape == (B, 24, 32)
    assert not torch.isnan(out_c).any()

    # An unmasked batch should be insensitive to the mask_token's value:
    # every row unmasked ⇒ the mask_token appears at no slot.
    batch_no_mask = {**batch, "mask_positions": torch.zeros(B, 23, dtype=torch.bool)}
    out_c_nomask = model_c(batch_no_mask)
    # Now perturb the mask_token weights and re-run: output must be identical.
    saved = model_c.mask_token.data.clone()
    with torch.no_grad():
        model_c.mask_token.data.add_(100.0)
    out_c_nomask_perturbed = model_c(batch_no_mask)
    assert torch.allclose(out_c_nomask, out_c_nomask_perturbed, atol=1e-5), (
        "mask_token value influenced output with no masked positions"
    )
    with torch.no_grad():
        model_c.mask_token.data.copy_(saved)
    print("  with empty mask, mask_token perturbation has no effect ✓")

    # With masked positions, the masked slots must differ from the unmasked
    # slots. We can't perturb the mask_token with a uniform shift and expect
    # a change — LayerNorm is translation-invariant along the feature dim.
    # Instead, compare masked vs unmasked runs directly on the affected slots.
    # Slot (0, 3) in out_c is masked; in out_c_nomask it is not. The two
    # representations must differ there.
    diff_at_mask = (out_c[0, 1 + 3] - out_c_nomask[0, 1 + 3]).abs().sum().item()
    diff_at_unmasked = (out_c[3, 1 + 0] - out_c_nomask[3, 1 + 0]).abs().sum().item()
    assert diff_at_mask > 1e-3, (
        f"masked slot (0, PAY_0) must differ between masked and unmasked runs, "
        f"got L1 {diff_at_mask:.6f}"
    )
    assert diff_at_unmasked < 1e-4, (
        f"unmasked slot (3, SEX) must be identical between runs, got L1 {diff_at_unmasked:.6f}"
    )
    print(f"  masked slot differs (L1 {diff_at_mask:.3f}), unmasked identical ✓")

    # ── D: deterministic re-run ──
    torch.manual_seed(0)
    model_d = FeatureEmbedding(d_model=32, dropout=0.0)
    model_d.eval()
    out_d1 = model_d(batch)
    torch.manual_seed(0)
    model_d2 = FeatureEmbedding(d_model=32, dropout=0.0)
    model_d2.eval()
    out_d2 = model_d2(batch)
    assert torch.allclose(out_d1, out_d2, atol=1e-6), "re-seeded re-run not deterministic"
    print("\n  deterministic under identical seed ✓")

    # ── E: freeze_encoder ──
    print("\n── E: freeze_encoder / unfreeze ──")
    model_e = FeatureEmbedding(d_model=32, dropout=0.0, use_mask_token=True)
    model_e.freeze_encoder(True)
    assert all(not p.requires_grad for p in model_e.parameters()), "freeze leaked"
    print(f"  all {sum(1 for _ in model_e.parameters())} params frozen ✓")
    model_e.freeze_encoder(False)
    assert all(p.requires_grad for p in model_e.parameters()), "unfreeze leaked"
    print("  unfreeze restores requires_grad=True on every param ✓")

    # ── F: init_from_pretrained_statedict ──
    print("\n── F: init_from_pretrained_statedict ──")
    src_model = FeatureEmbedding(d_model=32, dropout=0.0, use_mask_token=True)
    # Build a "wrapper" state dict mimicking the shape of a full transformer
    # checkpoint: embedding keys under "embedding." prefix + some spurious
    # transformer-block keys that must be filtered out.
    wrapper_sd = {
        f"embedding.{k}": v.clone() for k, v in src_model.state_dict().items()
    }
    wrapper_sd["classifier.weight"] = torch.randn(1, 32)
    wrapper_sd["transformer_block_0.attn.weight"] = torch.randn(32, 32)

    dst_model = FeatureEmbedding(d_model=32, dropout=0.0, use_mask_token=True)
    report = dst_model.init_from_pretrained_statedict(wrapper_sd, strict=False)
    # Every owned key should have been loaded.
    assert not report["missing"], f"missing after wrapper load: {report['missing']}"
    # The two spurious keys should appear in "unexpected".
    assert "classifier.weight" in report["unexpected"]
    assert "transformer_block_0.attn.weight" in report["unexpected"]
    # Loaded tensors must match source.
    for key in report["loaded"]:
        assert torch.equal(dst_model.state_dict()[key], src_model.state_dict()[key]), (
            f"{key} not correctly copied"
        )
    print(
        f"  loaded={len(report['loaded'])}  unexpected={len(report['unexpected'])}  "
        f"missing={len(report['missing'])} ✓"
    )

    # Strict mode raises on missing keys.
    partial_sd = {"cls_token": src_model.cls_token.data.clone()}
    try:
        FeatureEmbedding(d_model=32, use_mask_token=True).init_from_pretrained_statedict(
            partial_sd, strict=True
        )
    except KeyError as e:
        print(f"  strict=True raises on partial dict: {str(e)[:60]}... ✓")
    else:
        raise AssertionError("strict=True should have raised on missing keys")

    # Shape mismatch raises clearly.
    bad_sd = {"cls_token": torch.randn(1, 1, 64)}  # wrong d_model
    try:
        FeatureEmbedding(d_model=32).init_from_pretrained_statedict(bad_sd)
    except ValueError as e:
        print(f"  shape mismatch raises: {str(e)[:60]}... ✓")
    else:
        raise AssertionError("shape mismatch should have raised")

    # ── G: describe_token_layout ──
    print("\n── G: describe_token_layout ──")
    layout_str = describe_token_layout()
    # Basic sanity: one header + one separator + 24 rows = 26 lines.
    assert layout_str.count("\n") >= 25, "layout should have at least 26 lines"
    assert "CLS" in layout_str and "PAY_0" in layout_str and "BILL_AMT1" in layout_str
    # First three rows of body are demographic non-temporal.
    assert "demographic" in layout_str
    assert "N/A" in layout_str  # non-temporal tokens
    # Also verify cls_offset=0 gives a 23-row body.
    short = describe_token_layout(cls_offset=0)
    assert short.count("\n") >= 24, "cls_offset=0 layout should have at least 25 lines"
    print(f"  {layout_str.count(chr(10)) + 1} lines, cls_offset=0 variant also valid ✓")

    # ── H: FEATURE_GROUP_NAMES ──
    assert FEATURE_GROUP_NAMES[FEATURE_GROUP_CLS] == "CLS"
    assert FEATURE_GROUP_NAMES[FEATURE_GROUP_PAY] == "PAY"
    assert len(FEATURE_GROUP_NAMES) == N_FEATURE_GROUPS
    print("  FEATURE_GROUP_NAMES consistent with group constants ✓")

    print("\nAll embedding smoke tests passed.")
