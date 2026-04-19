"""Hybrid PAY tokeniser: 4-state category + severity N/8 per payment month.
Builds CreditDefaultDataset and MTLMCollator. Numerical features stay continuous.
Tokenisation is vectorised at construction so __getitem__ is O(1)."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# feature groups

CATEGORICAL_FEATURES: List[str] = ["SEX", "EDUCATION", "MARRIAGE"]

# mirrors data_preprocessing.DEMOGRAPHIC_FEATURES; kept local so
# validate_dataframe_schema has no cross-module import
DEMOGRAPHIC_FEATURES: List[str] = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]

PAY_STATUS_FEATURES: List[str] = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

BILL_AMT_FEATURES: List[str] = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_FEATURES: List[str] = [f"PAY_AMT{i}" for i in range(1, 7)]
NUMERICAL_FEATURES: List[str] = ["LIMIT_BAL", "AGE"] + BILL_AMT_FEATURES + PAY_AMT_FEATURES  # 14

TARGET_COL = "DEFAULT"

# PAY state ids
PAY_STATE_NO_BILL = 0      # PAY = -2
PAY_STATE_PAID_FULL = 1    # PAY = -1
PAY_STATE_MINIMUM = 2      # PAY =  0
PAY_STATE_DELINQUENT = 3   # PAY =  1..8
N_PAY_STATES = 4

# readable names for logs, notebooks, the §7.1 diagnostic panels
PAY_STATE_NAMES: Dict[int, str] = {
    PAY_STATE_NO_BILL:    "no_bill",
    PAY_STATE_PAID_FULL:  "paid_full",
    PAY_STATE_MINIMUM:    "minimum",
    PAY_STATE_DELINQUENT: "delinquent",
}

# divisor to squash severity into [0, 1]
MAX_PAY_DELAY = 8

# closed interval of valid PAY values; anything else raises PAYValueError
VALID_PAY_VALUES = frozenset({-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8})


class PAYValueError(ValueError):
    """PAY value outside [-2, 8]."""


# vocab builders


def build_categorical_vocab(metadata: Dict) -> Dict[str, Dict[int, int]]:
    """Per-feature {raw_value -> local_idx} for the demographics. Indices are
    sorted-integer order so the layout matches the embedding tables.

    No unknown bucket. Unseen codes at inference raise KeyError — callers that
    expect them should map to the documented "Others" bin (EDUCATION=4,
    MARRIAGE=3) upstream.
    """
    vocab: Dict[str, Dict[int, int]] = {}
    for feature in CATEGORICAL_FEATURES:
        value_map = metadata["categorical_features"][feature]["value_to_idx"]
        sorted_values = sorted(value_map.keys(), key=lambda x: int(x))
        vocab[feature] = {int(value): local_idx for local_idx, value in enumerate(sorted_values)}
    return vocab


def build_numerical_vocab() -> Dict[str, int]:
    """{feature -> identity_idx} for the numerical features. Production just
    enumerates NUMERICAL_FEATURES; kept because test_tokenizer asserts its
    shape and catches silent reorderings."""
    return {feature: idx for idx, feature in enumerate(NUMERICAL_FEATURES)}


# schema validation + diagnostics


def validate_dataframe_schema(
    df: pd.DataFrame,
    strict: bool = True,
) -> Dict[str, object]:
    """Check a post-rename DataFrame against the tokeniser's schema.

    Run right after load + column rename, before feature engineering or scaling.
    Checks: required columns present, no NaN, PAY_* ∈ VALID_PAY_VALUES, and
    categoricals are integer-coercible (so downstream .astype(int) won't
    silently truncate floats).

    strict=True raises ValueError with every detected problem concatenated, so
    the caller sees the full diff in one pass. strict=False returns the
    diagnostic dict and lets the caller decide. Report keys: n_rows,
    missing_columns, invalid_pay_values (column -> sorted bad values),
    nan_columns, non_integer_columns, ok.
    """
    required: List[str] = (
        DEMOGRAPHIC_FEATURES
        + PAY_STATUS_FEATURES
        + BILL_AMT_FEATURES
        + PAY_AMT_FEATURES
        + [TARGET_COL]
    )

    missing = [c for c in required if c not in df.columns]
    present = [c for c in required if c in df.columns]

    # NaN check, present columns only
    nan_cols = [c for c in present if df[c].isna().any()]

    invalid_pay: Dict[str, List[int]] = {}
    for col in PAY_STATUS_FEATURES:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        # float columns like 0.0 pass a naive `in VALID_PAY_VALUES` check; cast first
        try:
            as_int = series.astype(int)
        except (TypeError, ValueError):
            invalid_pay[col] = ["<non-integer column>"]  # type: ignore[list-item]
            continue
        bad_vals = sorted(set(int(v) for v in as_int.unique() if int(v) not in VALID_PAY_VALUES))
        if bad_vals:
            invalid_pay[col] = bad_vals

    # integer-coercible, not strict int — 1.5 should fail but 1.0 is fine
    non_int_cat: List[str] = []
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        try:
            as_int = series.astype(int)
            # round-trip: did the cast drop info?
            if (as_int.astype(series.dtype) != series).any():
                non_int_cat.append(col)
        except (TypeError, ValueError):
            non_int_cat.append(col)

    report: Dict[str, object] = {
        "n_rows":              int(len(df)),
        "missing_columns":     missing,
        "invalid_pay_values":  invalid_pay,
        "nan_columns":         nan_cols,
        "non_integer_columns": non_int_cat,
        "ok":                  not (missing or invalid_pay or nan_cols or non_int_cat),
    }

    if strict and not report["ok"]:
        issues: List[str] = []
        if missing:
            issues.append(f"missing columns: {missing}")
        if nan_cols:
            issues.append(f"NaN values present in: {nan_cols}")
        if invalid_pay:
            issues.append(f"invalid PAY values: {invalid_pay}")
        if non_int_cat:
            issues.append(f"non-integer categorical columns: {non_int_cat}")
        raise ValueError(
            "validate_dataframe_schema: " + "; ".join(issues)
        )

    return report


def tokenization_summary(
    tensors_dict: Dict[str, object],
) -> Dict[str, object]:
    """Diagnostic summary of a tokenize_dataframe output.

    Single pass over the pre-built tensors, cheap enough to call once per run
    or at dataset-load. Keys: n_rows, positive_rate, cat_value_counts
    (per-feature {local_idx: count}), pay_state_distribution (by state name),
    pay_severity ({min, max, mean}), num_feature_range (per-col {min, max}).
    """
    labels: torch.Tensor = tensors_dict["labels"]  # type: ignore[assignment]
    n_rows = int(labels.shape[0])
    positive_rate = float(labels.float().mean().item()) if n_rows > 0 else 0.0

    # per-feature local index space
    cat_value_counts: Dict[str, Dict[int, int]] = {}
    cat_tensors: Dict[str, torch.Tensor] = tensors_dict["cat_indices"]  # type: ignore[assignment]
    for feat in CATEGORICAL_FEATURES:
        values, counts = torch.unique(cat_tensors[feat], return_counts=True)
        cat_value_counts[feat] = {int(v): int(c) for v, c in zip(values.tolist(), counts.tolist())}

    pay_state_ids: torch.Tensor = tensors_dict["pay_state_ids"]  # type: ignore[assignment]
    state_distribution: Dict[str, int] = {}
    for sid, sname in PAY_STATE_NAMES.items():
        state_distribution[sname] = int((pay_state_ids == sid).sum().item())

    pay_severities: torch.Tensor = tensors_dict["pay_severities"]  # type: ignore[assignment]
    pay_severity_stats = {
        "min":  float(pay_severities.min().item()) if pay_severities.numel() else 0.0,
        "max":  float(pay_severities.max().item()) if pay_severities.numel() else 0.0,
        "mean": float(pay_severities.float().mean().item()) if pay_severities.numel() else 0.0,
    }

    # per-col min/max catches post-scaling outliers and columns the scaler missed
    num_values: torch.Tensor = tensors_dict["num_values"]  # type: ignore[assignment]
    num_feature_range: Dict[str, Dict[str, float]] = {}
    if num_values.numel() > 0:
        col_min = num_values.min(dim=0).values
        col_max = num_values.max(dim=0).values
        for i, feat in enumerate(NUMERICAL_FEATURES):
            num_feature_range[feat] = {
                "min": float(col_min[i].item()),
                "max": float(col_max[i].item()),
            }
    else:
        for feat in NUMERICAL_FEATURES:
            num_feature_range[feat] = {"min": 0.0, "max": 0.0}

    return {
        "n_rows":                 n_rows,
        "positive_rate":          positive_rate,
        "cat_value_counts":       cat_value_counts,
        "pay_state_distribution": state_distribution,
        "pay_severity":           pay_severity_stats,
        "num_feature_range":      num_feature_range,
    }


# PAY encoding


def encode_pay_value(pay_value: int) -> Tuple[int, float]:
    """Single PAY -> (state_id, severity). -2=no_bill, -1=paid_full, 0=minimum,
    1..8=delinquent with severity N/8 so it sits in [0, 1]. Out of range raises."""
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
    """(N, 6) int -> (state_ids int64, severities float32), both (N, 6).
    Out-of-range raises with the offending (row, month)."""
    pay_raw = np.asarray(pay_raw, dtype=np.int64)

    min_v, max_v = -2, MAX_PAY_DELAY
    bad = (pay_raw < min_v) | (pay_raw > max_v)
    if bad.any():
        bad_idx = np.argwhere(bad)[0]
        raise PAYValueError(
            f"PAY value {pay_raw[tuple(bad_idx)]} at (row={bad_idx[0]}, "
            f"month={bad_idx[1]}) outside valid range [{min_v}, {max_v}]"
        )

    state_ids = np.full_like(pay_raw, PAY_STATE_DELINQUENT, dtype=np.int64)
    state_ids[pay_raw == -2] = PAY_STATE_NO_BILL
    state_ids[pay_raw == -1] = PAY_STATE_PAID_FULL
    state_ids[pay_raw == 0] = PAY_STATE_MINIMUM

    # only delinquent rows have non-zero severity
    severities = np.where(pay_raw > 0, pay_raw / MAX_PAY_DELAY, 0.0).astype(np.float32)
    return state_ids, severities


# per-row tokenisation — ablations + debugging only


def tokenize_row(
    row: pd.Series,
    cat_vocab: Dict[str, Dict[int, int]],
) -> Tuple[Dict[str, int], List[int], List[float], List[float], int]:
    """One row -> token indices + values. Production uses the vectorised path
    in CreditDefaultDataset; this stays for ablation scripts."""
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

    pay_state_ids: List[int] = []
    pay_severities: List[float] = []
    for feature in PAY_STATUS_FEATURES:
        state_id, severity = encode_pay_value(int(row[feature]))
        pay_state_ids.append(state_id)
        pay_severities.append(severity)

    num_values = [float(row[feature]) for feature in NUMERICAL_FEATURES]
    label = int(row[TARGET_COL])

    return cat_indices, pay_state_ids, pay_severities, num_values, label


# vectorised DataFrame -> tensors


def tokenize_dataframe(
    df: pd.DataFrame,
    cat_vocab: Dict[str, Dict[int, int]],
) -> Dict[str, torch.Tensor]:
    """Vectorised tokenisation of a full DataFrame. Output tensors are what
    CreditDefaultDataset slices into.

    Returns:
      cat_indices     per-feature LongTensor(N,)
      pay_state_ids   LongTensor(N, 6), the 4-state hybrid ids
      pay_severities  FloatTensor(N, 6) in [0, 1]
      pay_raw         LongTensor(N, 6), raw PAY shifted -2..8 -> 0..10 so it
                      works directly as the 11-class target for MTLM and N5
      num_values      FloatTensor(N, 14)
      labels          FloatTensor(N,)

    Raises PAYValueError on out-of-range PAY, KeyError on unseen categoricals.
    """
    n = len(df)

    cat_indices: Dict[str, torch.Tensor] = {}
    for feature in CATEGORICAL_FEATURES:
        mapping = cat_vocab[feature]
        # pandas .map silently turns unknowns into NaN — catch and raise
        mapped = df[feature].astype(int).map(mapping)
        if mapped.isna().any():
            unseen = sorted(set(df.loc[mapped.isna(), feature].astype(int).unique()))
            raise KeyError(
                f"Unseen {feature} values {unseen}; "
                f"known = {sorted(mapping.keys())}"
            )
        cat_indices[feature] = torch.from_numpy(mapped.to_numpy(dtype=np.int64))

    # PAY: hybrid (state_ids + severities) for the primary tokeniser + raw
    # values shifted to [0, 10] for MTLM and the N5 PAY_0 forecast head
    pay_raw_np = df[PAY_STATUS_FEATURES].to_numpy(dtype=np.int64)
    state_ids_np, severities_np = _encode_pay_vectorised(pay_raw_np)
    pay_state_ids = torch.from_numpy(state_ids_np)
    pay_severities = torch.from_numpy(severities_np)
    # -2..8 -> 0..10 so CrossEntropyLoss gets non-negative targets.
    # _encode_pay_vectorised already rejected anything outside the range.
    pay_raw_shifted = torch.from_numpy(pay_raw_np + 2).long()

    num_values = torch.from_numpy(
        df[NUMERICAL_FEATURES].to_numpy(dtype=np.float32)
    )

    labels = torch.from_numpy(
        df[TARGET_COL].to_numpy(dtype=np.float32)
    )

    return {
        "cat_indices": cat_indices,
        "pay_state_ids": pay_state_ids,
        "pay_severities": pay_severities,
        "pay_raw": pay_raw_shifted,
        "num_values": num_values,
        "labels": labels,
    }


# class count for the shifted [0, 10] PAY target — used by the N5 head in model.py
PAY_RAW_NUM_CLASSES = 11


# Dataset


class CreditDefaultDataset(Dataset):
    """Dataset over a preprocessed credit-default DataFrame.

    Whole frame gets tokenised once at __init__ into pre-allocated tensors, so
    __getitem__ is a plain tensor slice — no pandas, no per-step allocation.
    Returns a dict with cat_indices (per-feature scalar longs), pay_state_ids
    (6,), pay_severities (6,), pay_raw (6,), num_values (14,), label (scalar).
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
            "pay_state_ids":  self._tensors["pay_state_ids"][idx],
            "pay_severities": self._tensors["pay_severities"][idx],
            "pay_raw":        self._tensors["pay_raw"][idx],
            "num_values":     self._tensors["num_values"][idx],
            "label":          self._tensors["labels"][idx],
        }

    # escape hatch for callers that want the whole batch (MTLMCollator,
    # probing classifiers) without re-iterating
    @property
    def tensors(self) -> Dict[str, torch.Tensor]:
        return self._tensors


# MTLM collator (N4)


class MTLMCollator:
    """BERT-style masking collator for phase-6A MTLM pretraining.

    Picks a random subset of the 23 feature tokens per row for the pretraining
    loss, then lets FeatureEmbedding swap in the shared [MASK] vector at the
    flagged slots. This collator only arranges tensors and produces mask
    metadata; it does not touch embeddings.

    Standard BERT 80/10/10: select mask_prob fraction, of which
    replace_with_mask become [MASK], replace_with_random become a random valid
    value, rest kept unchanged but still counted. min_mask_per_row /
    max_mask_per_row bound per-row counts so every row produces a loss term
    and task difficulty stays capped. seed -> local torch.Generator for
    deterministic masking.

    Returns the stacked batch + mask_positions BoolTensor(B, 23) (True where
    masked) + replace_mode LongTensor(B, 23) with values in
    {-1=not selected, 0=[MASK], 1=random, 2=keep}.
    """

    # token layout: 0..2 cat, 3..8 PAY, 9..22 numerical
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

    def _rand(self, *size: int) -> torch.Tensor:
        """U[0, 1) tensor, local generator if set."""
        if self.generator is not None:
            return torch.rand(*size, generator=self.generator)
        return torch.rand(*size)

    def _select_positions(self, B: int) -> torch.Tensor:
        """Per-row selection, clamped to [min_mask_per_row, max_mask_per_row].
        Returns BoolTensor(B, N_FEATURE_TOKENS)."""
        selected = self._rand(B, self.N_FEATURE_TOKENS) < self.mask_prob

        # top up under-masked rows
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
        """80/10/10 split per selected token. LongTensor(B, T): -1 not selected,
        0 [MASK], 1 random, 2 keep."""
        B, T = selected.shape
        mode = torch.full((B, T), -1, dtype=torch.long)

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

    def __call__(
        self,
        batch: Sequence[Dict[str, object]] | Dict[str, torch.Tensor],
    ) -> Dict[str, object]:
        """Takes a list of Dataset items or a pre-assembled batch dict.
        Returns the stacked batch + mask metadata."""
        if isinstance(batch, dict):
            assembled = batch
        else:
            assembled = self._default_collate(batch)

        B = assembled["num_values"].shape[0]
        mask_positions = self._select_positions(B)
        replace_mode = self._assign_replace_mode(mask_positions)

        return {
            **assembled,
            "mask_positions": mask_positions,
            "replace_mode": replace_mode,
        }

    @staticmethod
    def _default_collate(
        batch: Sequence[Dict[str, object]],
    ) -> Dict[str, torch.Tensor]:
        """Stack per-item dicts, keep the nested cat_indices layout."""
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
            "pay_raw":        torch.stack([item["pay_raw"] for item in batch]),         # type: ignore[list-item]
            "num_values":     torch.stack([item["num_values"] for item in batch]),      # type: ignore[list-item]
            "label":          torch.stack([item["label"] for item in batch]),           # type: ignore[list-item]
        }


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            try:
                _s.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    root = Path(__file__).resolve().parent.parent.parent
    meta_path = root / "data/processed/feature_metadata.json"
    csv_path = root / "data/processed/train_scaled.csv"
    if not meta_path.is_file() or not csv_path.is_file():
        print(
            "[SKIP] tokenizer.py smoke test requires preprocessing output.\n"
            "       Run `poetry run python scripts/run_pipeline.py --preprocess-only` "
            "first to materialise data/processed/*.csv."
        )
        sys.exit(0)
    meta = json.loads(meta_path.read_text())
    df = pd.read_csv(csv_path).head(128)

    cat_vocab = build_categorical_vocab(meta)
    print(f"cat vocab: {cat_vocab}")
    print(f"num vocab: {build_numerical_vocab()}")

    try:
        encode_pay_value(99)
    except PAYValueError as e:
        print(f"\nbounds check: raised PAYValueError correctly -- {e}")
    else:
        raise AssertionError("encode_pay_value(99) should have raised")

    bad_df = df.head(3).copy()
    bad_df.loc[bad_df.index[0], "SEX"] = 99
    try:
        tokenize_dataframe(bad_df, cat_vocab)
    except KeyError as e:
        print(f"unseen categorical: raised KeyError correctly -- {e.args[0][:80]}...")
    else:
        raise AssertionError("tokenize_dataframe with bad SEX should have raised")

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
    print("\nrow-by-row vs vectorised: bit-equivalent for row 0 OK")

    import time
    t0 = time.perf_counter()
    _ = CreditDefaultDataset(df, cat_vocab, verbose=False)
    t_vec = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in range(len(df)):
        tokenize_row(df.iloc[i], cat_vocab)
    t_slow = time.perf_counter() - t0
    speedup = t_slow / t_vec if t_vec > 0 else float("inf")
    print(f"vectorised: {t_vec*1000:.1f} ms  |  per-row: {t_slow*1000:.1f} ms  |  speedup ~ {speedup:.1f}x")

    print("\nMTLMCollator:")
    batch_list = [ds[i] for i in range(16)]

    collator = MTLMCollator(mask_prob=0.15, min_mask_per_row=1, max_mask_per_row=5, seed=42)
    out = collator(batch_list)
    assert out["mask_positions"].shape == (16, 23), out["mask_positions"].shape
    assert out["replace_mode"].shape == (16, 23)
    counts = out["mask_positions"].sum(dim=1)
    assert (counts >= 1).all() and (counts <= 5).all(), f"mask count per row out of bounds: {counts.tolist()}"
    assert ((out["replace_mode"] == -1) == (~out["mask_positions"])).all()
    selected_modes = out["replace_mode"][out["mask_positions"]]
    n_mask = int((selected_modes == 0).sum().item())
    n_rand = int((selected_modes == 1).sum().item())
    n_keep = int((selected_modes == 2).sum().item())
    print(f"  masked rows: {out['mask_positions'].sum().item()}  "
          f"([MASK]={n_mask}, random={n_rand}, keep={n_keep})")

    collator2 = MTLMCollator(mask_prob=0.15, seed=42)
    out2 = collator2(batch_list)
    assert torch.equal(out["mask_positions"], out2["mask_positions"]), "mask positions non-deterministic"
    print("  deterministic with seeded generator OK")

    pre_assembled = collator._default_collate(batch_list)
    out3 = collator(pre_assembled)
    assert out3["mask_positions"].shape == (16, 23)
    print("  pre-assembled dict path OK")

    print("\nvalidate_dataframe_schema:")
    raw_report = validate_dataframe_schema(df, strict=False)
    assert raw_report["n_rows"] == len(df)
    assert not raw_report["invalid_pay_values"], raw_report["invalid_pay_values"]
    assert not raw_report["nan_columns"], raw_report["nan_columns"]
    assert not raw_report["missing_columns"], raw_report["missing_columns"]
    print(f"  strict=False on clean frame: ok={raw_report['ok']} OK")

    bad = df.head(5).copy()
    bad.loc[bad.index[0], "PAY_0"] = 42
    try:
        validate_dataframe_schema(bad, strict=True)
    except ValueError as e:
        msg = str(e)[:80]
        print(f"  strict=True raises on bad frame: {msg}... OK")
    else:
        raise AssertionError("validate_dataframe_schema should raise on PAY=42")

    bad_report = validate_dataframe_schema(bad, strict=False)
    assert "PAY_0" in bad_report["invalid_pay_values"]
    assert 42 in bad_report["invalid_pay_values"]["PAY_0"]
    assert bad_report["ok"] is False
    print("  strict=False returns structured report for bad frame OK")

    missing_df = df.drop(columns=["PAY_6"]).head(5)
    miss_report = validate_dataframe_schema(missing_df, strict=False)
    assert "PAY_6" in miss_report["missing_columns"]
    print(f"  missing-column detection: {miss_report['missing_columns']} OK")

    print("\ntokenization_summary:")
    tensors = tokenize_dataframe(df, cat_vocab)
    summary = tokenization_summary(tensors)
    assert summary["n_rows"] == len(df)
    assert 0.0 <= summary["positive_rate"] <= 1.0
    for sname in PAY_STATE_NAMES.values():
        assert sname in summary["pay_state_distribution"], f"missing {sname}"
    assert sum(summary["pay_state_distribution"].values()) == len(df) * 6
    assert set(summary["num_feature_range"].keys()) == set(NUMERICAL_FEATURES)
    print(
        f"  n_rows={summary['n_rows']}  pos_rate={summary['positive_rate']:.3f}  "
        f"pay_dist={summary['pay_state_distribution']} OK"
    )
    print(f"  severity: {summary['pay_severity']} OK")

    assert PAY_STATE_NAMES[PAY_STATE_NO_BILL] == "no_bill"
    assert PAY_STATE_NAMES[PAY_STATE_DELINQUENT] == "delinquent"
    assert len(PAY_STATE_NAMES) == N_PAY_STATES
    print("  PAY_STATE_NAMES mapping consistent with state constants OK")

    print("\nall tokenizer smoke tests passed.")
