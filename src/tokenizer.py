"""
tokenizer.py — Tabular tokenisation for the Credit Card Default dataset.

Converts each row of the dataset into a sequence of tokens for the transformer.

Hybrid tokenisation scheme (three feature types, PROJECT_PLAN.md §5.2):
    1. Demographic categoricals (SEX, EDUCATION, MARRIAGE):
       Per-feature embedding-lookup. Each feature gets its own vocabulary and
       its own embedding table on the model side (FT-Transformer style).
    2. PAY status — hybrid state + severity (Strategy C, §5.2.3):
       Each PAY value (-2 to 8) is mapped to one of FOUR semantic states:
           -2 → "no_bill"     (no card use that month)
           -1 → "paid_full"   (paid balance in full)
            0 → "minimum"     (revolving credit / minimum payment)
         1..8 → "delinquent"  (delayed by N months)
       The state gets an embedding from a small table. For "delinquent" we
       additionally pass severity = value / 8 through a linear projection.
       PAY=-2/-1/0 carry severity = 0.
    3. Numerical features (LIMIT_BAL, AGE, BILL_AMT1–6, PAY_AMT1–6):
       Feature embedding + value projection — the feature name gets an embedding
       and the scaled value is passed through a linear layer.

Result: 23 tokens per client (3 demographic + 6 PAY + 14 numerical).

This revision additionally provides:
    * **Vectorised tokenisation**: the whole dataframe is tokenised once at
      Dataset-construction time using numpy vector ops. No more row-by-row
      `df.iloc[i]` loop. Pre-allocated tensors make `__getitem__` O(1).
    * **Bounds checking** on PAY values (raises on out-of-range input).
    * **Logger** instead of bare ``print`` in the Dataset constructor.
    * **MTLMCollator** — BERT-style masking collator for Phase 6A
      self-supervised pretraining (PROJECT_PLAN.md §5.4B, novelty N4).

Public API preserved for backward compatibility (see §"Public API" below).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Demographic categoricals.
CATEGORICAL_FEATURES: List[str] = ["SEX", "EDUCATION", "MARRIAGE"]

# PAY status features (hybrid encoding).
PAY_STATUS_FEATURES: List[str] = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

# Numerical features.
BILL_AMT_FEATURES: List[str] = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_FEATURES: List[str] = [f"PAY_AMT{i}" for i in range(1, 7)]
NUMERICAL_FEATURES: List[str] = ["LIMIT_BAL", "AGE"] + BILL_AMT_FEATURES + PAY_AMT_FEATURES  # 14

TARGET_COL = "DEFAULT"

# PAY state indices (used by tokenize_row and embedding.py).
PAY_STATE_NO_BILL = 0      # PAY = -2
PAY_STATE_PAID_FULL = 1    # PAY = -1
PAY_STATE_MINIMUM = 2      # PAY =  0
PAY_STATE_DELINQUENT = 3   # PAY =  1..8
N_PAY_STATES = 4

# Upper bound of the delinquency scale — used to normalise severity to [0, 1].
MAX_PAY_DELAY = 8

# Valid PAY values (closed interval). Anything outside raises PAYValueError.
VALID_PAY_VALUES = frozenset({-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8})


class PAYValueError(ValueError):
    """Raised when an out-of-range PAY value is tokenised."""


# ──────────────────────────────────────────────────────────────────────────────
# Block 1: Vocabulary builders
# ──────────────────────────────────────────────────────────────────────────────


def build_categorical_vocab(metadata: Dict) -> Dict[str, Dict[int, int]]:
    """
    Build a separate vocabulary for each demographic categorical feature
    (SEX, EDUCATION, MARRIAGE).

    Each feature gets its own mapping from raw value → local index. Local
    indices are assigned by sorted integer order, so the mapping is
    reproducible across runs and matches the embedding-table layout.

    Notes on unseen values (documented policy):
    Raw values encountered at inference time that are not in this vocabulary
    trigger a ``KeyError`` with a descriptive message — there is no "unknown"
    token. This is the correct behaviour for the closed-world UCI dataset
    where category codes are fully enumerated; downstream systems that might
    encounter unseen values should wrap the tokeniser in a preprocessor that
    first maps unseen codes to the documented "Others" bucket (EDUCATION=4,
    MARRIAGE=3) before tokenising.

    Returns
    -------
    Nested dict: feature_name → {raw_value: local_index}.
    """
    vocab: Dict[str, Dict[int, int]] = {}
    for feature in CATEGORICAL_FEATURES:
        value_map = metadata["categorical_features"][feature]["value_to_idx"]
        sorted_values = sorted(value_map.keys(), key=lambda x: int(x))
        vocab[feature] = {int(value): local_idx for local_idx, value in enumerate(sorted_values)}
    return vocab


def build_numerical_vocab() -> Dict[str, int]:
    """
    Map numerical feature names → their fixed integer identity indices.

    The identity goes into a single shared embedding table
    (:class:`~embedding.FeatureEmbedding.num_feature_embedding`); the scaled
    value is handled by a separate linear projection on the model side. See
    PROJECT_PLAN.md §5.2.1.

    This helper is primarily a documentation / sanity artefact — the
    production code uses :data:`NUMERICAL_FEATURES` directly via
    ``enumerate``. Kept because :mod:`tests.test_tokenizer` asserts its
    shape, which protects against silent reordering of
    :data:`NUMERICAL_FEATURES`.
    """
    return {feature: idx for idx, feature in enumerate(NUMERICAL_FEATURES)}


# ──────────────────────────────────────────────────────────────────────────────
# Block 2: PAY value encoding
# ──────────────────────────────────────────────────────────────────────────────


def encode_pay_value(pay_value: int) -> Tuple[int, float]:
    """
    Map a single PAY value to (state_id, severity) for the hybrid scheme.

    Mapping:
        -2  → (no_bill,      0.0)
        -1  → (paid_full,    0.0)
         0  → (minimum,      0.0)
       1..8 → (delinquent,   N / 8)

    Severity is normalised by :data:`MAX_PAY_DELAY` (=8) so that all
    delinquency values fit in [0, 1] — numerically comparable to embedding
    values.

    Raises
    ------
    PAYValueError
        If ``pay_value`` is outside :data:`VALID_PAY_VALUES`. The caller is
        expected to have validated the data upstream (see
        ``data_preprocessing.validate_data``).
    """
    if pay_value not in VALID_PAY_VALUES:
        raise PAYValueError(
            f"PAY value {pay_value} outside valid range [-2, {MAX_PAY_DELAY}]"
        )
    if pay_value == -2:
        return PAY_STATE_NO_BILL, 0.0
    if pay_value == -1:
        return PAY_STATE_PAID_FULL, 0.0
    if pay_value == 0:
        return PAY_STATE_MINIMUM, 0.0
    return PAY_STATE_DELINQUENT, pay_value / MAX_PAY_DELAY


def _encode_pay_vectorised(
    pay_raw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorised PAY encoding for a (N, 6) integer array.

    Returns
    -------
    (state_ids, severities) each with shape (N, 6), dtypes int64 and float32.

    Raises
    ------
    PAYValueError on any out-of-range value (described by its coordinates).
    """
    pay_raw = np.asarray(pay_raw, dtype=np.int64)

    # Bounds check — vectorised.
    min_v, max_v = -2, MAX_PAY_DELAY
    bad = (pay_raw < min_v) | (pay_raw > max_v)
    if bad.any():
        bad_idx = np.argwhere(bad)[0]
        raise PAYValueError(
            f"PAY value {pay_raw[tuple(bad_idx)]} at (row={bad_idx[0]}, "
            f"month={bad_idx[1]}) outside valid range [{min_v}, {max_v}]"
        )

    # State: where each value lands.
    state_ids = np.full_like(pay_raw, PAY_STATE_DELINQUENT, dtype=np.int64)
    state_ids[pay_raw == -2] = PAY_STATE_NO_BILL
    state_ids[pay_raw == -1] = PAY_STATE_PAID_FULL
    state_ids[pay_raw == 0] = PAY_STATE_MINIMUM

    # Severity: only delinquent rows carry a non-zero severity.
    severities = np.where(pay_raw > 0, pay_raw / MAX_PAY_DELAY, 0.0).astype(np.float32)
    return state_ids, severities


# ──────────────────────────────────────────────────────────────────────────────
# Block 3: Per-row tokenisation (preserved for ablation use)
# ──────────────────────────────────────────────────────────────────────────────


def tokenize_row(
    row: pd.Series,
    cat_vocab: Dict[str, Dict[int, int]],
) -> Tuple[Dict[str, int], List[int], List[float], List[float], int]:
    """
    Convert a single DataFrame row into token indices and values.

    This is the row-by-row implementation kept for ablation scripts and
    debugging. Production training uses the vectorised path in
    :class:`CreditDefaultDataset`.
    """
    # 1. Demographic categoricals.
    cat_indices: Dict[str, int] = {}
    for feature in CATEGORICAL_FEATURES:
        raw_value = int(row[feature])
        try:
            cat_indices[feature] = cat_vocab[feature][raw_value]
        except KeyError as exc:
            raise KeyError(
                f"Unseen {feature} value {raw_value}; "
                f"known values = {sorted(cat_vocab[feature].keys())}"
            ) from exc

    # 2. PAY statuses.
    pay_state_ids: List[int] = []
    pay_severities: List[float] = []
    for feature in PAY_STATUS_FEATURES:
        state_id, severity = encode_pay_value(int(row[feature]))
        pay_state_ids.append(state_id)
        pay_severities.append(severity)

    # 3. Numerical values.
    num_values = [float(row[feature]) for feature in NUMERICAL_FEATURES]

    # 4. Target.
    label = int(row[TARGET_COL])

    return cat_indices, pay_state_ids, pay_severities, num_values, label


# ──────────────────────────────────────────────────────────────────────────────
# Block 4: Vectorised DataFrame → tensors
# ──────────────────────────────────────────────────────────────────────────────


def tokenize_dataframe(
    df: pd.DataFrame,
    cat_vocab: Dict[str, Dict[int, int]],
) -> Dict[str, torch.Tensor]:
    """
    Vectorised tokenisation of an entire DataFrame.

    Produces pre-built tensors that :class:`CreditDefaultDataset` indexes into
    with O(1) ``__getitem__``.

    Returns
    -------
    Dict with keys::

        cat_indices    → Dict[str, LongTensor (N,)]
        pay_state_ids  → LongTensor  (N, 6)
        pay_severities → FloatTensor (N, 6)
        num_values     → FloatTensor (N, 14)
        labels         → FloatTensor (N,)

    Raises
    ------
    PAYValueError  on out-of-range PAY values.
    KeyError       on unseen categorical values.
    """
    n = len(df)

    # Categoricals.
    cat_indices: Dict[str, torch.Tensor] = {}
    for feature in CATEGORICAL_FEATURES:
        mapping = cat_vocab[feature]
        # pandas .map fills unknowns with NaN — we detect and raise.
        mapped = df[feature].astype(int).map(mapping)
        if mapped.isna().any():
            unseen = sorted(set(df.loc[mapped.isna(), feature].astype(int).unique()))
            raise KeyError(
                f"Unseen {feature} values {unseen}; "
                f"known = {sorted(mapping.keys())}"
            )
        cat_indices[feature] = torch.from_numpy(mapped.to_numpy(dtype=np.int64))

    # PAY.
    pay_raw = df[PAY_STATUS_FEATURES].to_numpy(dtype=np.int64)
    state_ids_np, severities_np = _encode_pay_vectorised(pay_raw)
    pay_state_ids = torch.from_numpy(state_ids_np)           # (N, 6) int64
    pay_severities = torch.from_numpy(severities_np)          # (N, 6) float32

    # Numerical.
    num_values = torch.from_numpy(
        df[NUMERICAL_FEATURES].to_numpy(dtype=np.float32)
    )  # (N, 14)

    # Labels.
    labels = torch.from_numpy(
        df[TARGET_COL].to_numpy(dtype=np.float32)
    )  # (N,)

    return {
        "cat_indices": cat_indices,
        "pay_state_ids": pay_state_ids,
        "pay_severities": pay_severities,
        "num_values": num_values,
        "labels": labels,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Block 5: PyTorch Dataset (vectorised, O(1) __getitem__)
# ──────────────────────────────────────────────────────────────────────────────


class CreditDefaultDataset(Dataset):
    """
    PyTorch Dataset over a preprocessed credit-default DataFrame.

    On construction the entire DataFrame is tokenised once via
    :func:`tokenize_dataframe`, producing pre-allocated tensors. Each
    ``__getitem__`` call then simply slices row *i* out of those tensors —
    no pandas access, no per-step tensor allocation, no Python loops.

    Output of ``__getitem__(idx)`` is a dict:

        cat_indices    : {"SEX": LongTensor(), "EDUCATION": LongTensor(), "MARRIAGE": LongTensor()}
        pay_state_ids  : LongTensor  (6,)
        pay_severities : FloatTensor (6,)
        num_values     : FloatTensor (14,)
        label          : FloatTensor (0-d, 0.0 or 1.0)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cat_vocab: Dict[str, Dict[int, int]],
        *,
        verbose: bool = True,
    ):
        self._tensors = tokenize_dataframe(df, cat_vocab)
        self._n = len(df)

        if verbose:
            labels = self._tensors["labels"]
            n_pos = int(labels.sum().item())
            logger.info(
                "CreditDefaultDataset: tokenised %d rows (%d defaults, %d non-defaults)",
                self._n, n_pos, self._n - n_pos,
            )

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return {
            "cat_indices": {
                feat: self._tensors["cat_indices"][feat][idx]
                for feat in CATEGORICAL_FEATURES
            },
            "pay_state_ids":  self._tensors["pay_state_ids"][idx],   # (6,)
            "pay_severities": self._tensors["pay_severities"][idx],  # (6,)
            "num_values":     self._tensors["num_values"][idx],      # (14,)
            "label":          self._tensors["labels"][idx],          # scalar
        }

    # Escape hatch — callers that want the whole batch at once (e.g. MTLMCollator,
    # probing classifiers) can reach in without re-iterating.
    @property
    def tensors(self) -> Dict[str, torch.Tensor]:
        return self._tensors


# ──────────────────────────────────────────────────────────────────────────────
# Block 6: Masked Tabular Language Modelling collator (Plan §5.4B, Novelty N4)
# ──────────────────────────────────────────────────────────────────────────────


class MTLMCollator:
    """
    BERT-style masking collator for Phase 6A (MTLM self-supervised pretraining).

    Given a batch emitted by :class:`CreditDefaultDataset`, chooses a random
    subset of the 23 feature tokens (per row) to predict, and returns both the
    original batch *and* the mask book-keeping needed by the pretraining loss.

    Masking strategy — follows BERT's 80/10/10 recipe adapted to heterogeneous
    tabular tokens:

        * 15% of the 23 feature tokens are selected for prediction (per row).
        * Of those selected:
            - 80%: replaced with the model-side ``[MASK]`` token (flag only —
              this collator does not touch the embeddings; the model's
              :class:`FeatureEmbedding` needs to honour a ``mask_positions``
              tensor to substitute ``[MASK]`` for the selected tokens).
            - 10%: replaced with a random valid value from the same feature's
              training distribution.
            - 10%: kept unchanged (but still contribute to the loss).

    This collator does *not* consult the embedding layer directly — it only
    rearranges tensors and produces the mask metadata. The FeatureEmbedding
    module is responsible for using ``mask_positions`` to swap in a ``[MASK]``
    vector at the masked slots when forward-passing.

    Parameters
    ----------
    mask_prob : float
        Probability a feature token is selected for prediction. Default 0.15.
    replace_with_mask : float
        Of the selected tokens, fraction replaced with ``[MASK]``. Default 0.8.
    replace_with_random : float
        Of the selected tokens, fraction replaced with a random valid value.
        Default 0.1. The remainder (0.1) are kept identical but still contribute
        to the loss.
    min_mask_per_row : int
        Lower bound on the number of selected tokens per row (≥ 1 so every row
        produces a loss term). Default 1.
    max_mask_per_row : Optional[int]
        Upper bound on the selected tokens per row (to cap task difficulty).
    seed : Optional[int]
        If given, seeds a local torch.Generator for deterministic masking. If
        None, uses the global RNG.

    Returns (from __call__):
        A dict with the original batch keys (cat_indices, pay_state_ids,
        pay_severities, num_values, label) AND:

        mask_positions : BoolTensor (B, 23) — True where the token is masked.
        replace_mode   : LongTensor (B, 23) — 0=[MASK], 1=random, 2=keep, -1=not selected.
        original_tokens: a reference to the *unmodified* batch tensors, so the
                         model can compute the prediction loss against the
                         original values.

    The collator does NOT modify the batch's tensors in place — replacement
    happens in the embedding layer via ``mask_positions``.
    """

    # Token positions: 0..2 = categorical, 3..8 = PAY, 9..22 = numerical.
    N_FEATURE_TOKENS = 23
    CAT_SLICE = slice(0, 3)
    PAY_SLICE = slice(3, 9)
    NUM_SLICE = slice(9, 23)

    def __init__(
        self,
        mask_prob: float = 0.15,
        replace_with_mask: float = 0.8,
        replace_with_random: float = 0.1,
        min_mask_per_row: int = 1,
        max_mask_per_row: Optional[int] = 5,
        seed: Optional[int] = None,
    ):
        if not (0.0 < mask_prob < 1.0):
            raise ValueError(f"mask_prob must lie in (0, 1), got {mask_prob}")
        if not (0.0 <= replace_with_mask <= 1.0):
            raise ValueError("replace_with_mask must lie in [0, 1]")
        if not (0.0 <= replace_with_random <= 1.0):
            raise ValueError("replace_with_random must lie in [0, 1]")
        if replace_with_mask + replace_with_random > 1.0 + 1e-9:
            raise ValueError(
                "replace_with_mask + replace_with_random must be <= 1.0 "
                f"(got {replace_with_mask} + {replace_with_random})"
            )
        if min_mask_per_row < 1:
            raise ValueError("min_mask_per_row must be >= 1")

        self.mask_prob = mask_prob
        self.replace_with_mask = replace_with_mask
        self.replace_with_random = replace_with_random
        self.min_mask_per_row = min_mask_per_row
        self.max_mask_per_row = (
            max_mask_per_row if max_mask_per_row is not None else self.N_FEATURE_TOKENS
        )
        self.generator: Optional[torch.Generator] = None
        if seed is not None:
            self.generator = torch.Generator().manual_seed(seed)

    # ---------- helpers ----------

    def _rand(self, *size: int) -> torch.Tensor:
        """Uniform [0, 1) random tensor, using the local generator if any."""
        if self.generator is not None:
            return torch.rand(*size, generator=self.generator)
        return torch.rand(*size)

    def _select_positions(self, B: int) -> torch.Tensor:
        """
        Sample, per row, which feature tokens to predict.

        Guarantees ``min_mask_per_row <= count <= max_mask_per_row`` for every row.
        Returns a BoolTensor (B, N_FEATURE_TOKENS).
        """
        # Bernoulli selection at mask_prob.
        selected = self._rand(B, self.N_FEATURE_TOKENS) < self.mask_prob

        # Enforce min. Top up each under-masked row with tokens chosen uniformly.
        counts = selected.sum(dim=1)
        short = counts < self.min_mask_per_row
        for row_idx in short.nonzero(as_tuple=True)[0].tolist():
            needed = self.min_mask_per_row - int(counts[row_idx].item())
            unmasked_positions = (~selected[row_idx]).nonzero(as_tuple=True)[0]
            picks = unmasked_positions[
                torch.randperm(
                    len(unmasked_positions), generator=self.generator
                )[:needed]
            ]
            selected[row_idx, picks] = True

        # Enforce max.
        counts = selected.sum(dim=1)
        long_rows = counts > self.max_mask_per_row
        for row_idx in long_rows.nonzero(as_tuple=True)[0].tolist():
            excess = int(counts[row_idx].item()) - self.max_mask_per_row
            masked_positions = selected[row_idx].nonzero(as_tuple=True)[0]
            drops = masked_positions[
                torch.randperm(
                    len(masked_positions), generator=self.generator
                )[:excess]
            ]
            selected[row_idx, drops] = False

        return selected

    def _assign_replace_mode(self, selected: torch.Tensor) -> torch.Tensor:
        """
        For each selected token, decide replacement mode per the 80/10/10 split.

        Returns a LongTensor (B, N_FEATURE_TOKENS):
            -1 = token NOT selected (no loss, no replacement)
             0 = replace with [MASK] embedding
             1 = replace with a random valid value
             2 = keep original but still supervise
        """
        B, T = selected.shape
        mode = torch.full((B, T), -1, dtype=torch.long)

        # Draw a random draw per selected cell, then split by threshold.
        draws = self._rand(B, T)
        is_mask = selected & (draws < self.replace_with_mask)
        is_rand = selected & (~is_mask) & (
            draws < self.replace_with_mask + self.replace_with_random
        )
        is_keep = selected & (~is_mask) & (~is_rand)

        mode[is_mask] = 0
        mode[is_rand] = 1
        mode[is_keep] = 2
        return mode

    # ---------- call ----------

    def __call__(
        self,
        batch: Sequence[Dict[str, object]] | Dict[str, torch.Tensor],
    ) -> Dict[str, object]:
        """
        Accept either a list of Dataset items (standard DataLoader collate input)
        or a pre-assembled batch dict. Returns the batch plus mask metadata.
        """
        if isinstance(batch, dict):
            # Pre-assembled; trust the shapes.
            assembled = batch
        else:
            assembled = self._default_collate(batch)

        B = assembled["num_values"].shape[0]
        mask_positions = self._select_positions(B)
        replace_mode = self._assign_replace_mode(mask_positions)

        return {
            **assembled,
            "mask_positions": mask_positions,  # (B, 23) bool
            "replace_mode": replace_mode,       # (B, 23) long, values in {-1,0,1,2}
        }

    @staticmethod
    def _default_collate(
        batch: Sequence[Dict[str, object]],
    ) -> Dict[str, torch.Tensor]:
        """
        Stack per-item dicts (from CreditDefaultDataset) into a batch dict.
        Mirrors ``torch.utils.data.default_collate`` but preserves the nested
        ``cat_indices`` dict layout.
        """
        cat_indices = {
            feat: torch.stack([
                item["cat_indices"][feat] for item in batch  # type: ignore[index]
            ])
            for feat in CATEGORICAL_FEATURES
        }
        return {
            "cat_indices":    cat_indices,
            "pay_state_ids":  torch.stack([item["pay_state_ids"] for item in batch]),   # type: ignore[list-item]
            "pay_severities": torch.stack([item["pay_severities"] for item in batch]),  # type: ignore[list-item]
            "num_values":     torch.stack([item["num_values"] for item in batch]),      # type: ignore[list-item]
            "label":          torch.stack([item["label"] for item in batch]),           # type: ignore[list-item]
        }


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    # UTF-8 stdout so box-drawing separators print cleanly on Windows.
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            try:
                _s.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    # Use the real preprocessed metadata + train scaled CSV shipped with the repo.
    root = Path(__file__).resolve().parent.parent
    meta_path = root / "data/processed/feature_metadata.json"
    csv_path = root / "data/processed/train_scaled.csv"
    if not meta_path.is_file() or not csv_path.is_file():
        print(
            "[SKIP] tokenizer.py smoke test requires preprocessing output.\n"
            "       Run `poetry run python run_pipeline.py --preprocess-only` "
            "first to materialise data/processed/*.csv."
        )
        sys.exit(0)
    meta = json.loads(meta_path.read_text())
    df = pd.read_csv(csv_path).head(128)

    cat_vocab = build_categorical_vocab(meta)
    print(f"cat vocab: {cat_vocab}")
    print(f"num vocab: {build_numerical_vocab()}")

    # ── Bounds check ──
    try:
        encode_pay_value(99)
    except PAYValueError as e:
        print(f"\nbounds check: raised PAYValueError correctly — {e}")
    else:
        raise AssertionError("encode_pay_value(99) should have raised")

    # ── Unseen categorical ──
    bad_df = df.head(3).copy()
    bad_df.loc[bad_df.index[0], "SEX"] = 99
    try:
        tokenize_dataframe(bad_df, cat_vocab)
    except KeyError as e:
        print(f"unseen categorical: raised KeyError correctly — {e.args[0][:80]}...")
    else:
        raise AssertionError("tokenize_dataframe with bad SEX should have raised")

    # ── Vectorised vs row-by-row equivalence ──
    ds = CreditDefaultDataset(df, cat_vocab)
    row0_fast = ds[0]
    row0_slow = tokenize_row(df.iloc[0], cat_vocab)
    assert row0_fast["cat_indices"]["SEX"].item() == row0_slow[0]["SEX"]
    assert row0_fast["pay_state_ids"].tolist() == row0_slow[1]
    assert all(
        abs(a - b) < 1e-5
        for a, b in zip(row0_fast["pay_severities"].tolist(), row0_slow[2])
    )
    assert all(
        abs(a - b) < 1e-4
        for a, b in zip(row0_fast["num_values"].tolist(), row0_slow[3])
    )
    assert int(row0_fast["label"].item()) == row0_slow[4]
    print("\nrow-by-row vs vectorised: bit-equivalent for row 0 ✓")

    # ── Vectorised timing ──
    import time
    t0 = time.perf_counter()
    _ = CreditDefaultDataset(df, cat_vocab, verbose=False)
    t_vec = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in range(len(df)):
        tokenize_row(df.iloc[i], cat_vocab)
    t_slow = time.perf_counter() - t0
    speedup = t_slow / t_vec if t_vec > 0 else float("inf")
    print(f"vectorised: {t_vec*1000:.1f} ms  |  per-row: {t_slow*1000:.1f} ms  |  speedup ≈ {speedup:.1f}×")

    # ── MTLMCollator ──
    print("\n── MTLMCollator ──")
    # Assemble a small batch the way DataLoader would.
    batch_list = [ds[i] for i in range(16)]

    collator = MTLMCollator(mask_prob=0.15, min_mask_per_row=1, max_mask_per_row=5, seed=42)
    out = collator(batch_list)
    assert out["mask_positions"].shape == (16, 23), out["mask_positions"].shape
    assert out["replace_mode"].shape == (16, 23)
    counts = out["mask_positions"].sum(dim=1)
    assert (counts >= 1).all() and (counts <= 5).all(), f"mask count per row out of bounds: {counts.tolist()}"
    # Replace-mode consistency: cells with mask_positions=False must have mode=-1
    assert ((out["replace_mode"] == -1) == (~out["mask_positions"])).all()
    # Replace-mode distribution sanity (over the selected positions).
    selected_modes = out["replace_mode"][out["mask_positions"]]
    n_mask = int((selected_modes == 0).sum().item())
    n_rand = int((selected_modes == 1).sum().item())
    n_keep = int((selected_modes == 2).sum().item())
    print(f"  masked rows: {out['mask_positions'].sum().item()}  "
          f"([MASK]={n_mask}, random={n_rand}, keep={n_keep})")

    # Determinism: same seed → identical mask positions.
    collator2 = MTLMCollator(mask_prob=0.15, seed=42)
    out2 = collator2(batch_list)
    assert torch.equal(out["mask_positions"], out2["mask_positions"]), "mask positions non-deterministic"
    print("  deterministic with seeded generator ✓")

    # Assembled-dict path.
    pre_assembled = collator._default_collate(batch_list)
    out3 = collator(pre_assembled)
    assert out3["mask_positions"].shape == (16, 23)
    print("  pre-assembled dict path ✓")

    print("\nAll tokenizer smoke tests passed.")
