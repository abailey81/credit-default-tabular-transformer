"""Feature embedding: tokeniser output -> ``(B, 24, d_model)`` sequence.

The layer takes the tensor dict produced by
:mod:`src.tokenization.tokenizer` and produces the input the transformer
encoder consumes. The output sequence is:

    pos 0     [CLS]              — learnable, pooled for classification
    pos 1..3  demographics       — SEX, EDUCATION, MARRIAGE
    pos 4..9  PAY (hybrid)       — state emb + severity projection
    pos 10..23 numerical         — feature emb + value projection (14 cols)

Key design choices:

* **Hybrid PAY (Novelty N1)** is produced in the tokeniser; here we sum
  the state embedding and the severity projection so both signals
  share the same attention pathway.
* **Per-feature projections** for numerical columns: each of the 14
  columns gets its own learned feature id + a shared value projection,
  so the model can tell LIMIT_BAL apart from AGE even though both feed
  through the same scalar -> d_model projection.
* **Optional temporal positional encoding** (Ablation A7) adds a
  month-indexed embedding to PAY_i / BILL_AMT_i / PAY_AMT_i. Non-temporal
  tokens (demographics, LIMIT_BAL, AGE) pass through unchanged.
* **Optional [MASK] token** (Novelty N4, MTLM) swaps the *content* at
  flagged positions while preserving feature + temporal positional
  signals. A masked PAY_0 still tells the model "pos 4, month 0" — only
  its value is hidden.

The module is self-contained: it can instantiate itself from
``feature_metadata.json`` on disk, or from an explicit ``cat_vocab_sizes``
dict (preferred in tests to avoid filesystem dependencies).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .tokenizer import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PAY_STATUS_FEATURES,
)

# ---------------------------------------------------------------------------
# Token layout
# ---------------------------------------------------------------------------
# Position 0 is [CLS]; positions 1..23 follow TOKEN_ORDER.
# 3 categorical + 6 PAY + 14 numerical = 23 feature tokens (+1 [CLS] = 24).
# Every consumer of per-token attention, per-token importance, or
# per-group masking should go through TOKEN_ORDER (or build_group_assignment
# / build_temporal_layout below) rather than hard-code slice boundaries.
TOKEN_ORDER: list[str] = CATEGORICAL_FEATURES + PAY_STATUS_FEATURES + NUMERICAL_FEATURES

# Slice views into the 23-token pre-CLS block. Used by TemporalDecayBias /
# FeatureGroupBias and anything else that masks by group.
_SLICE_CAT = slice(0, 3)
_SLICE_PAY = slice(3, 9)
_SLICE_NUM = slice(9, 23)

# ---------------------------------------------------------------------------
# Temporal layout
# ---------------------------------------------------------------------------
# UCI dataset covers Apr 2005 -> Sep 2005. Our convention: month 0 is the
# most recent (Sep 2005), month 5 is the oldest (Apr 2005). That way
# PAY_0 / BILL_AMT1 / PAY_AMT1 all share month 0 — they correspond to
# the same calendar period even though the UCI naming is inconsistent.
N_MONTHS = 6
_TEMPORAL_MONTH: dict[str, int] = {}
for _i, _feat in enumerate(PAY_STATUS_FEATURES):
    _TEMPORAL_MONTH[_feat] = _i
for _i, _feat in enumerate([f"BILL_AMT{_m}" for _m in range(1, 7)]):
    _TEMPORAL_MONTH[_feat] = _i
for _i, _feat in enumerate([f"PAY_AMT{_m}" for _m in range(1, 7)]):
    _TEMPORAL_MONTH[_feat] = _i

# Per-token month index, with -1 for non-temporal tokens. Used to gate the
# temporal positional lookup cleanly at forward time without a Python-level
# branch over the 23 positions.
_TEMPORAL_INDEX_PER_TOKEN = torch.tensor(
    [_TEMPORAL_MONTH.get(feat, -1) for feat in TOKEN_ORDER],
    dtype=torch.long,
)


# Metadata loaded lazily so a fresh clone can import this module before
# the preprocessing pipeline has ever run. Tests typically pass
# cat_vocab_sizes directly to avoid touching disk.
_METADATA_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "feature_metadata.json"
)


def load_cat_vocab_sizes(
    metadata_path: Optional[Path] = None,
) -> dict[str, int]:
    """Read categorical vocab sizes from ``feature_metadata.json``.

    Prefer passing ``cat_vocab_sizes`` directly to ``FeatureEmbedding`` in
    tests and libraries — this helper exists for notebook / CLI use where
    hitting disk is fine and the JSON is always present. Raises
    ``FileNotFoundError`` with a fix-me pointer if the file is missing.
    """
    path = Path(metadata_path) if metadata_path is not None else _METADATA_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"feature_metadata.json not found at {path}. "
            "Run `poetry run python scripts/run_pipeline.py --preprocess-only` to "
            "generate preprocessing outputs before instantiating FeatureEmbedding, "
            "or pass an explicit cat_vocab_sizes kwarg."
        )
    with open(path) as f:
        meta = json.load(f)
    return {feat: info["n_categories"] for feat, info in meta["categorical_features"].items()}


def __getattr__(name: str):
    """PEP 562 lazy attribute shim.

    ``CAT_VOCAB_SIZES`` used to be a module-level constant computed at
    import time, but that breaks fresh clones that don't have
    ``feature_metadata.json`` yet. This shim defers the load until the
    constant is actually accessed and caches the result in ``globals()``.
    """
    if name == "CAT_VOCAB_SIZES":
        value = load_cat_vocab_sizes()
        globals()["CAT_VOCAB_SIZES"] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Feature groups (N2 + N3 + interpretability)
# ---------------------------------------------------------------------------
# Source of truth for TemporalDecayBias (N3), FeatureGroupBias (N2), and
# every interpretability consumer that needs "which positions are PAY".

_TEMPORAL_GROUPS: dict[str, list[str]] = {
    "pay": list(PAY_STATUS_FEATURES),
    "bill": [f"BILL_AMT{_m}" for _m in range(1, 7)],
    "pay_amt": [f"PAY_AMT{_m}" for _m in range(1, 7)],
}


# Group ids used by FeatureGroupBias's per-group attention bias.
# CLS is its own group so attention into / out of CLS is learnable
# independently of demographics.
FEATURE_GROUP_CLS = 0
FEATURE_GROUP_DEMOGRAPHIC = 1
FEATURE_GROUP_PAY = 2
FEATURE_GROUP_BILL = 3
FEATURE_GROUP_PAY_AMT = 4
N_FEATURE_GROUPS = 5

# Readable names for attention heatmaps and attribution panels.
FEATURE_GROUP_NAMES: dict[int, str] = {
    FEATURE_GROUP_CLS: "CLS",
    FEATURE_GROUP_DEMOGRAPHIC: "demographic",
    FEATURE_GROUP_PAY: "PAY",
    FEATURE_GROUP_BILL: "BILL_AMT",
    FEATURE_GROUP_PAY_AMT: "PAY_AMT",
}

_DEMOGRAPHIC_FEATURES = set(CATEGORICAL_FEATURES) | {"LIMIT_BAL", "AGE"}
_BILL_FEATURES = {f"BILL_AMT{_m}" for _m in range(1, 7)}
_PAY_AMT_FEATURES = {f"PAY_AMT{_m}" for _m in range(1, 7)}


def _group_for_feature(feature_name: str) -> int:
    """Map a feature name to its group id. Raises on unknown names."""
    if feature_name in _DEMOGRAPHIC_FEATURES:
        return FEATURE_GROUP_DEMOGRAPHIC
    if feature_name in PAY_STATUS_FEATURES:
        return FEATURE_GROUP_PAY
    if feature_name in _BILL_FEATURES:
        return FEATURE_GROUP_BILL
    if feature_name in _PAY_AMT_FEATURES:
        return FEATURE_GROUP_PAY_AMT
    raise KeyError(f"No group defined for feature {feature_name!r}")


def build_group_assignment(cls_offset: int = 1) -> list[int]:
    """Group-index list for ``FeatureGroupBias``.

    Length ``len(TOKEN_ORDER) + cls_offset`` (24 by default with one [CLS]
    prepended). Derived from ``TOKEN_ORDER`` so it can't drift when a
    feature is added or reordered. Pass ``cls_offset=0`` to address the
    pre-CLS 23-block directly (used by a few interpretability tools).
    """
    assignment: list[int] = [FEATURE_GROUP_CLS] * cls_offset
    assignment.extend(_group_for_feature(f) for f in TOKEN_ORDER)
    return assignment


def describe_token_layout(cls_offset: int = 1) -> str:
    """Pretty ``pos | group | feature | month`` table for the 24-seq.

    Used in the notebook and the report appendix. Derived from
    ``TOKEN_ORDER`` so it can never desync with the live layer — handy
    when reviewing attention rollouts where "position 9" needs to map
    back to a feature name.
    """
    lines: list[str] = []
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
) -> dict[str, dict[str, list[int]]]:
    """Build the ``temporal_layout`` dict that ``TemporalDecayBias`` consumes.

    Maps group name (``"pay"`` / ``"bill"`` / ``"pay_amt"``) to
    ``{"positions": [...], "months": [...]}``, months 0-indexed with 0 =
    most recent. ``cls_offset=0`` addresses the pre-CLS 23-block, useful
    when the caller prepends their own special tokens or works on a flat
    23-sequence view.
    """
    layout: dict[str, dict[str, list[int]]] = {}
    for group_name, feat_list in _TEMPORAL_GROUPS.items():
        layout[group_name] = {
            "positions": [cls_offset + TOKEN_ORDER.index(f) for f in feat_list],
            "months": list(range(len(feat_list))),
        }
    return layout


# ---------------------------------------------------------------------------
# FeatureEmbedding
# ---------------------------------------------------------------------------


class FeatureEmbedding(nn.Module):
    """Tokenizer output -> ``(B, 24, d_model)`` ready for the encoder.

    Expected batch dict:

    * ``cat_indices``    — per-feature ``(B,)`` longs.
    * ``pay_state_ids``  — ``(B, 6)`` longs.
    * ``pay_severities`` — ``(B, 6)`` floats in [0, 1].
    * ``num_values``     — ``(B, 14)`` floats.
    * ``mask_positions`` — ``(B, 23)`` bool, optional (MTLM only).

    Flags:

    * ``cat_vocab_sizes``   — pass explicitly to skip the JSON load; if
      omitted, falls back to ``feature_metadata.json`` on disk.
    * ``use_temporal_pos``  — add learnable ``nn.Embedding(6, d)`` per
      month on PAY / BILL_AMT / PAY_AMT tokens. Demographics + LIMIT_BAL
      + AGE pass through unchanged (they have no month).
    * ``use_mask_token``    — create a learnable [MASK] vector for MTLM
      (Novelty N4). When the batch carries ``mask_positions``, the
      *content* embedding at flagged slots is replaced by the [MASK]
      vector while positional signals (feature id + month) are preserved.

    Invariants
    ----------
    * ``cat_vocab_sizes`` MUST match the training-split vocab used by the
      tokeniser — mismatched sizes silently truncate the embedding table
      lookup on unseen codes.
    * When ``use_mask_token=True`` and ``mask_positions`` is absent from
      the batch, the mask vector has no effect. (This is what makes the
      module drop-in for MTLM fine-tuning without code changes.)
    """

    def __init__(
        self,
        d_model: int = 32,
        dropout: float = 0.1,
        *,
        cat_vocab_sizes: Optional[dict[str, int]] = None,
        use_temporal_pos: bool = False,
        use_mask_token: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_temporal_pos = use_temporal_pos
        self.use_mask_token = use_mask_token

        if cat_vocab_sizes is None:
            cat_vocab_sizes = load_cat_vocab_sizes()
        missing = set(CATEGORICAL_FEATURES) - cat_vocab_sizes.keys()
        if missing:
            raise KeyError(
                f"cat_vocab_sizes is missing entries for {sorted(missing)}; "
                f"got {sorted(cat_vocab_sizes.keys())}"
            )
        self.cat_vocab_sizes: dict[str, int] = dict(cat_vocab_sizes)

        # One embedding table per categorical feature. Using a ModuleDict
        # rather than a single big table with offsets keeps state-dict
        # keys self-documenting and makes per-feature freezing trivial.
        self.cat_embeddings = nn.ModuleDict(
            {feat: nn.Embedding(cat_vocab_sizes[feat], d_model) for feat in CATEGORICAL_FEATURES}
        )

        # PAY hybrid: 4-state embedding + 1-dim severity projection. The
        # two outputs are summed (not concatenated) so the attention
        # pathway has uniform d_model across all token types.
        self.pay_state_embedding = nn.Embedding(4, d_model)
        self.pay_severity_proj = nn.Linear(1, d_model, bias=True)

        # Numerical tokens: feature-identity embedding + shared scalar
        # projection. The identity embedding is what lets attention
        # distinguish LIMIT_BAL from AGE; the value projection only
        # needs to know "how big is the number".
        self.num_feature_embedding = nn.Embedding(len(NUMERICAL_FEATURES), d_model)
        self.value_proj = nn.Linear(1, d_model, bias=True)

        # Absolute positional embedding across the full 24 slots (0 = [CLS]).
        self.pos_embedding = nn.Embedding(len(TOKEN_ORDER) + 1, d_model)

        if use_temporal_pos:
            self.temporal_pos_embedding = nn.Embedding(N_MONTHS, d_model)
        else:
            self.temporal_pos_embedding = None

        if use_mask_token:
            # Small random init so the mask vector doesn't dominate the
            # signal at the start of pretraining. 0.02 matches the
            # embedding tables' init scale.
            self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)
        else:
            self.mask_token = None

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Non-persistent buffers: they move with .to(device) but don't
        # clutter state_dict or bloat checkpoint files.
        self.register_buffer("positions", torch.arange(1, len(TOKEN_ORDER) + 1), persistent=False)
        self.register_buffer(
            "temporal_index_per_token", _TEMPORAL_INDEX_PER_TOKEN.clone(), persistent=False
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """N(0, 0.02) on embedding tables, xavier-normal on projections.

        0.02 is the GPT-2 / BERT convention for transformer embeddings;
        anything larger tends to dominate the positional signal at init
        and make the first few hundred steps noisy. Projection layers use
        xavier-normal so the scalar -> d_model expansion preserves variance
        without knowing d_model at init time.
        """
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

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        B = batch["num_values"].shape[0]
        device = batch["num_values"].device

        # Pull the pre-CLS positional slab once; positions [0..22] index
        # into it for all three feature groups.
        all_pos_emb = self.pos_embedding(self.positions)  # (23, d)

        # Categoricals at seq 1..3 (pos 0 is [CLS] - added last).
        cat_tokens = torch.stack(
            [
                self.cat_embeddings[feat](batch["cat_indices"][feat]) + all_pos_emb[pos]
                for pos, feat in enumerate(CATEGORICAL_FEATURES)
            ],
            dim=1,
        )  # (B, 3, d)

        # PAY at seq 4..9 — hybrid sum: state emb + severity proj + pos emb.
        n_cat = len(CATEGORICAL_FEATURES)
        state_emb = self.pay_state_embedding(batch["pay_state_ids"])  # (B, 6, d)
        sev_emb = self.pay_severity_proj(batch["pay_severities"].unsqueeze(-1))  # (B, 6, d)
        pay_pos = all_pos_emb[n_cat : n_cat + len(PAY_STATUS_FEATURES)]
        pay_tokens = state_emb + sev_emb + pay_pos  # (B, 6, d)

        # Numerical at seq 10..23 — value proj + feature-id emb + pos emb.
        n_cat_pay = n_cat + len(PAY_STATUS_FEATURES)
        feat_idx = torch.arange(len(NUMERICAL_FEATURES), dtype=torch.long, device=device)
        feat_emb = self.num_feature_embedding(feat_idx)  # (14, d)
        value_emb = self.value_proj(batch["num_values"].unsqueeze(-1))  # (B, 14, d)
        num_pos = all_pos_emb[n_cat_pay:]
        num_tokens = value_emb + feat_emb + num_pos  # (B, 14, d)

        feature_tokens = torch.cat([cat_tokens, pay_tokens, num_tokens], dim=1)  # (B, 23, d)

        # Temporal positional encoding on temporal tokens only. Computed
        # outside the mask branch so MTLM can strip + re-add it cleanly —
        # a masked PAY_0 still needs its month signal.
        if self.temporal_pos_embedding is not None:
            temporal_idx = self.temporal_index_per_token  # (23,)
            is_temporal = (temporal_idx >= 0).unsqueeze(-1).to(feature_tokens.dtype)
            # Clamp -1 -> 0 so the lookup is safe; is_temporal gates the
            # result back to zero for non-temporal tokens.
            safe_idx = temporal_idx.clamp(min=0)
            temporal_emb = self.temporal_pos_embedding(safe_idx) * is_temporal  # (23, d)
            feature_tokens = feature_tokens + temporal_emb
        else:
            temporal_emb = None

        # MTLM masking (BERT convention): [MASK] replaces the *content*
        # (value + feature id). All positional signals are preserved. The
        # strip-then-readd pattern is equivalent to masking on the content
        # alone but keeps the code path symmetric with / without temporal
        # pos. A masked PAY_0 still carries "pos 4, month 0"; a masked
        # PAY_6 still carries "pos 9, month 5".
        if self.mask_token is not None and "mask_positions" in batch:
            mask_positions = batch["mask_positions"].to(device)
            if mask_positions.shape != (B, 23):
                raise ValueError(
                    f"mask_positions shape must be (B, 23), got {tuple(mask_positions.shape)}"
                )
            pos_emb_23 = all_pos_emb  # (23, d)
            content = feature_tokens - pos_emb_23
            if temporal_emb is not None:
                content = content - temporal_emb
            mask_vec = self.mask_token.unsqueeze(0).unsqueeze(0)  # (1, 1, d)
            mask_bcast = mask_positions.unsqueeze(-1).to(feature_tokens.dtype)  # (B, 23, 1)
            new_content = content * (1.0 - mask_bcast) + mask_vec * mask_bcast
            feature_tokens = new_content + pos_emb_23
            if temporal_emb is not None:
                feature_tokens = feature_tokens + temporal_emb

        # Prepend [CLS] at pos 0. We add its positional embedding here (not
        # at init) so the [CLS] parameter stays pure "content" and the
        # pos_embedding table stays the single source of position.
        cls_pos = self.pos_embedding(torch.zeros(1, dtype=torch.long, device=device))
        cls = (self.cls_token + cls_pos).expand(B, -1, -1)  # (B, 1, d)

        out = torch.cat([cls, feature_tokens], dim=1)  # (B, 24, d)
        return self.dropout(self.norm(out))

    def freeze_encoder(self, freeze: bool = True) -> None:
        """Toggle ``requires_grad`` on every owned parameter.

        Used in two-stage fine-tuning (§8.5.5): after loading an MTLM
        checkpoint we freeze the embedding for the first few supervised
        epochs so the classification head settles on stable features,
        then unfreeze for joint fine-tuning at a lower LR.
        """
        for param in self.parameters():
            param.requires_grad = not freeze

    def init_from_pretrained_statedict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = False,
    ) -> dict[str, list[str]]:
        """Copy embedding-relevant tensors out of a larger checkpoint.

        Designed for the MTLM -> supervised handover: an MTLM model is a
        thin wrapper over this layer plus a prediction head, so its
        state-dict keys are prefixed with ``"embedding."`` or
        ``"feature_embedding."``. This method tolerates those prefixes,
        silently drops unrelated keys (transformer blocks, classifier
        heads, MTLM heads), and returns a report that tests can assert
        against.

        Parameters
        ----------
        state_dict
            Flat mapping from some source model's state_dict.
        strict
            If ``True`` and any owned parameter has no source tensor,
            raise ``KeyError``. ``False`` (default) is appropriate for
            MTLM -> supervised where some parts of the destination
            (e.g. the [MASK] token) may legitimately start fresh.

        Returns
        -------
        dict
            ``{"loaded": [...], "missing": [...], "unexpected": [...]}``.

        Raises
        ------
        ValueError
            Shape mismatch on any matched key. This is always fatal —
            silently reusing a 64-d checkpoint in a 32-d model would
            corrupt training.
        KeyError
            ``strict=True`` and some owned param had no source tensor.
        """
        own_keys = set(self.state_dict().keys())

        # Empty prefix first so a direct FeatureEmbedding checkpoint
        # (no wrapper) skips the prefix hunt entirely.
        candidate_prefixes = ("", "embedding.", "feature_embedding.", "module.")

        normalised: dict[str, torch.Tensor] = {}
        unexpected: list[str] = []
        for raw_key, tensor in state_dict.items():
            matched_own: Optional[str] = None
            for prefix in candidate_prefixes:
                if prefix and not raw_key.startswith(prefix):
                    continue
                stripped = raw_key[len(prefix) :] if prefix else raw_key
                if stripped in own_keys:
                    matched_own = stripped
                    break
            if matched_own is None:
                unexpected.append(raw_key)
                continue
            # First match wins — if two prefixes map to the same own key
            # we keep the first (most specific) source.
            if matched_own not in normalised:
                normalised[matched_own] = tensor

        loaded: list[str] = []
        for key, tensor in normalised.items():
            own = self.state_dict()[key]
            if own.shape != tensor.shape:
                raise ValueError(
                    f"init_from_pretrained_statedict: shape mismatch for "
                    f"{key!r} (own {tuple(own.shape)} vs incoming "
                    f"{tuple(tensor.shape)})"
                )
            # Single-key load_state_dict per tensor reuses nn.Module's
            # dtype/device handling — copying via param.data would miss
            # the parameter-registration bookkeeping.
            self.load_state_dict({key: tensor}, strict=False)
            loaded.append(key)

        missing = sorted(own_keys - set(loaded))
        if strict and missing:
            raise KeyError(
                f"init_from_pretrained_statedict(strict=True): missing keys " f"{missing}"
            )

        return {
            "loaded": sorted(loaded),
            "missing": missing,
            "unexpected": sorted(unexpected),
        }


if __name__ == "__main__":
    # Smoke test: exercises every branch of forward() — baseline,
    # temporal on, mask on. Each sub-block ends with explicit assertions
    # so a regression (e.g. a dead gradient path) fails loudly.
    torch.manual_seed(0)

    B = 4
    # Build a batch with non-zero severity on delinquent states so the
    # severity projection actually gets exercised (zero severity on every
    # row would leave its gradient empty and mask a bug).
    batch: dict[str, torch.Tensor] = {
        "cat_indices": {
            "SEX": torch.tensor([0, 1, 0, 1]),
            "EDUCATION": torch.tensor([0, 1, 2, 3]),
            "MARRIAGE": torch.tensor([0, 1, 2, 0]),
        },
        "pay_state_ids": torch.tensor(
            [
                [0, 1, 2, 3, 3, 3],
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
        "num_values": torch.randn(B, 14),
        "label": torch.tensor([0.0, 1.0, 0.0, 1.0]),
    }

    print("A: baseline (no temporal, no mask):")
    model_a = FeatureEmbedding(d_model=32, dropout=0.0)
    model_a.eval()
    out_a = model_a(batch)
    print(f"  output shape: {tuple(out_a.shape)}")
    assert out_a.shape == (B, 24, 32)
    assert not torch.isnan(out_a).any() and not torch.isinf(out_a).any()

    # Random-weighted loss, not bare .sum(): LayerNorm is zero-mean per
    # token so a sum-loss kills gradients for constant-input params like
    # [CLS]. A random dot-product mimics a downstream classification head
    # and ensures every parameter sees non-zero gradient.
    torch.manual_seed(1)
    proj = torch.randn_like(model_a(batch))
    model_a.train()
    out_a_train = model_a(batch)
    (out_a_train * proj).sum().backward()
    no_grad = [
        n for n, p in model_a.named_parameters() if p.grad is None or p.grad.abs().sum() == 0
    ]
    assert not no_grad, f"zero-grad params: {no_grad}"

    sev_w_grad = model_a.pay_severity_proj.weight.grad
    assert (
        sev_w_grad is not None and sev_w_grad.abs().sum().item() > 0
    ), "pay_severity_proj not exercised -- non-zero severity regressed"
    print(f"  pay_severity_proj grad L1: {sev_w_grad.abs().sum().item():.4f} OK")
    print("  all params have non-zero gradients OK")

    print("\nB: temporal positional encoding on:")
    model_b = FeatureEmbedding(d_model=32, dropout=0.0, use_temporal_pos=True)
    model_b.eval()
    out_b = model_b(batch)
    assert out_b.shape == (B, 24, 32)
    assert not torch.isnan(out_b).any()
    model_b.train()
    model_b(batch).sum().backward()
    assert model_b.temporal_pos_embedding.weight.grad.abs().sum().item() > 0
    print("  temporal_pos_embedding receives gradient OK")
    non_temp = (model_b.temporal_index_per_token == -1).nonzero(as_tuple=True)[0]
    # In the 23-block: non-temporal tokens are SEX(0), EDUCATION(1),
    # MARRIAGE(2), LIMIT_BAL(9), AGE(10). Everything else has a month.
    expected_non_temp = [0, 1, 2, 9, 10]
    assert (
        non_temp.tolist() == expected_non_temp
    ), f"unexpected non-temporal set {non_temp.tolist()}"
    print(f"  non-temporal token indices (in 23-block): {expected_non_temp} OK")

    print("\nC: [MASK] token on:")
    model_c = FeatureEmbedding(d_model=32, dropout=0.0, use_mask_token=True)
    model_c.eval()
    mask_positions = torch.zeros(B, 23, dtype=torch.bool)
    mask_positions[0, 3] = True  # row 0 PAY_0
    mask_positions[1, 9] = True  # row 1 LIMIT_BAL
    mask_positions[2, 0] = True  # row 2 SEX
    batch_masked = {**batch, "mask_positions": mask_positions}
    out_c = model_c(batch_masked)
    assert out_c.shape == (B, 24, 32)
    assert not torch.isnan(out_c).any()

    # With no mask positions, the output must be independent of mask_token.
    # Verify by perturbing mask_token and confirming the output doesn't move.
    batch_no_mask = {**batch, "mask_positions": torch.zeros(B, 23, dtype=torch.bool)}
    out_c_nomask = model_c(batch_no_mask)
    saved = model_c.mask_token.data.clone()
    with torch.no_grad():
        model_c.mask_token.data.add_(100.0)
    out_c_nomask_perturbed = model_c(batch_no_mask)
    assert torch.allclose(
        out_c_nomask, out_c_nomask_perturbed, atol=1e-5
    ), "mask_token value influenced output with no masked positions"
    with torch.no_grad():
        model_c.mask_token.data.copy_(saved)
    print("  with empty mask, mask_token perturbation has no effect OK")

    # Masked slots must differ run-to-run; unmasked slots must not. Note:
    # LayerNorm is translation-invariant on the feature dim, so comparing
    # raw mask_token shifts directly won't work — compare the two runs.
    diff_at_mask = (out_c[0, 1 + 3] - out_c_nomask[0, 1 + 3]).abs().sum().item()
    diff_at_unmasked = (out_c[3, 1 + 0] - out_c_nomask[3, 1 + 0]).abs().sum().item()
    assert diff_at_mask > 1e-3, (
        f"masked slot (0, PAY_0) must differ between masked and unmasked runs, "
        f"got L1 {diff_at_mask:.6f}"
    )
    assert (
        diff_at_unmasked < 1e-4
    ), f"unmasked slot (3, SEX) must be identical between runs, got L1 {diff_at_unmasked:.6f}"
    print(f"  masked slot differs (L1 {diff_at_mask:.3f}), unmasked identical OK")

    torch.manual_seed(0)
    model_d = FeatureEmbedding(d_model=32, dropout=0.0)
    model_d.eval()
    out_d1 = model_d(batch)
    torch.manual_seed(0)
    model_d2 = FeatureEmbedding(d_model=32, dropout=0.0)
    model_d2.eval()
    out_d2 = model_d2(batch)
    assert torch.allclose(out_d1, out_d2, atol=1e-6), "re-seeded re-run not deterministic"
    print("\n  deterministic under identical seed OK")

    print("\nE: freeze_encoder / unfreeze:")
    model_e = FeatureEmbedding(d_model=32, dropout=0.0, use_mask_token=True)
    model_e.freeze_encoder(True)
    assert all(not p.requires_grad for p in model_e.parameters()), "freeze leaked"
    print(f"  all {sum(1 for _ in model_e.parameters())} params frozen OK")
    model_e.freeze_encoder(False)
    assert all(p.requires_grad for p in model_e.parameters()), "unfreeze leaked"
    print("  unfreeze restores requires_grad=True on every param OK")

    print("\nF: init_from_pretrained_statedict:")
    src_model = FeatureEmbedding(d_model=32, dropout=0.0, use_mask_token=True)
    # Simulate a full-transformer checkpoint: embedding keys under
    # "embedding." + a couple of junk keys the loader should ignore.
    wrapper_sd = {f"embedding.{k}": v.clone() for k, v in src_model.state_dict().items()}
    wrapper_sd["classifier.weight"] = torch.randn(1, 32)
    wrapper_sd["transformer_block_0.attn.weight"] = torch.randn(32, 32)

    dst_model = FeatureEmbedding(d_model=32, dropout=0.0, use_mask_token=True)
    report = dst_model.init_from_pretrained_statedict(wrapper_sd, strict=False)
    assert not report["missing"], f"missing after wrapper load: {report['missing']}"
    assert "classifier.weight" in report["unexpected"]
    assert "transformer_block_0.attn.weight" in report["unexpected"]
    for key in report["loaded"]:
        assert torch.equal(
            dst_model.state_dict()[key], src_model.state_dict()[key]
        ), f"{key} not correctly copied"
    print(
        f"  loaded={len(report['loaded'])}  unexpected={len(report['unexpected'])}  "
        f"missing={len(report['missing'])} OK"
    )

    partial_sd = {"cls_token": src_model.cls_token.data.clone()}
    try:
        FeatureEmbedding(d_model=32, use_mask_token=True).init_from_pretrained_statedict(
            partial_sd, strict=True
        )
    except KeyError as e:
        print(f"  strict=True raises on partial dict: {str(e)[:60]}... OK")
    else:
        raise AssertionError("strict=True should have raised on missing keys")

    bad_sd = {"cls_token": torch.randn(1, 1, 64)}
    try:
        FeatureEmbedding(d_model=32).init_from_pretrained_statedict(bad_sd)
    except ValueError as e:
        print(f"  shape mismatch raises: {str(e)[:60]}... OK")
    else:
        raise AssertionError("shape mismatch should have raised")

    print("\nG: describe_token_layout:")
    layout_str = describe_token_layout()
    # header + separator + 24 rows = 26 lines.
    assert layout_str.count("\n") >= 25, "layout should have at least 26 lines"
    assert "CLS" in layout_str and "PAY_0" in layout_str and "BILL_AMT1" in layout_str
    assert "demographic" in layout_str
    assert "N/A" in layout_str
    short = describe_token_layout(cls_offset=0)
    assert short.count("\n") >= 24, "cls_offset=0 layout should have at least 25 lines"
    print(f"  {layout_str.count(chr(10)) + 1} lines, cls_offset=0 variant also valid OK")

    assert FEATURE_GROUP_NAMES[FEATURE_GROUP_CLS] == "CLS"
    assert FEATURE_GROUP_NAMES[FEATURE_GROUP_PAY] == "PAY"
    assert len(FEATURE_GROUP_NAMES) == N_FEATURE_GROUPS
    print("  FEATURE_GROUP_NAMES consistent with group constants OK")

    print("\nall embedding smoke tests passed.")
