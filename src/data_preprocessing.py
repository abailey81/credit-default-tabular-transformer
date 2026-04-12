"""
data_preprocessing.py — Data loading, cleaning, feature engineering, and splitting
for the UCI Credit Card Default dataset.

Reference: Yeh, I.C. & Lien, C.H. (2009). Expert Systems with Applications, 36(2), 2473–2480.

This module handles:
    1. Raw data ingestion (XLS format from UCI repository)
    2. Schema validation and column renaming
    3. Cleaning of undocumented categorical codes
    4. Feature engineering (utilisation ratios, repayment ratios, delinquency aggregates)
    5. Stratified train / validation / test splitting
    6. Standardisation of numerical features (fitted on train only to prevent leakage)
    7. Metadata export for downstream tokenisation
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

RANDOM_SEED = 42

# Canonical column names (the raw file has inconsistent naming across versions)
DEMOGRAPHIC_FEATURES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
PAY_STATUS_FEATURES = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_AMT_FEATURES = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_FEATURES = [f"PAY_AMT{i}" for i in range(1, 7)]
TARGET_COL = "DEFAULT"

CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]
NUMERICAL_FEATURES = (
    ["LIMIT_BAL", "AGE"]
    + BILL_AMT_FEATURES
    + PAY_AMT_FEATURES
)
ORDINAL_PAY_FEATURES = PAY_STATUS_FEATURES  # treated specially in tokenisation

ALL_FEATURE_COLS = (
    DEMOGRAPHIC_FEATURES + PAY_STATUS_FEATURES + BILL_AMT_FEATURES + PAY_AMT_FEATURES
)

# Valid value ranges (for sanity checking)
VALID_SEX = {1, 2}
VALID_EDUCATION_RAW = {0, 1, 2, 3, 4, 5, 6}
VALID_EDUCATION_CLEAN = {1, 2, 3, 4}
VALID_MARRIAGE_RAW = {0, 1, 2, 3}
VALID_MARRIAGE_CLEAN = {1, 2, 3}
VALID_PAY_STATUS = {-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8}


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_raw_data(
    filepath: Optional[str] = None,
    *,
    mode: str = "auto",
    allow_fallback: bool = True,
) -> pd.DataFrame:
    """
    Load the raw UCI credit card default dataset.

    This is a thin wrapper over :mod:`data_sources`, which provides a layered
    multi-source ingestion pipeline with graceful API → local-file failover.

    Parameters
    ----------
    filepath
        Optional explicit path to a local ``.xls``/``.xlsx`` file. When set,
        only the local file is consulted (no network attempt).
    mode
        Source preference. One of:

        * ``"auto"`` (default) — try the UCI API first, fall back to the
          local manual dataset if the API is unavailable.
        * ``"api"`` — UCI API only; raises on failure.
        * ``"local"`` — local file only; uses the default candidate list
          plus the ``data/raw/`` directory.

        Ignored when ``filepath`` is supplied.
    allow_fallback
        Only meaningful in ``"auto"`` mode. If ``False``, disables the local
        fallback so API failures propagate as a hard error.

    Returns
    -------
    pandas.DataFrame
        Raw dataframe; downstream callers should still apply
        :func:`normalise_schema` and :func:`clean_categoricals`.
    """
    # Imported here so the module remains importable even if ``data_sources``
    # has not been added to ``sys.path`` yet (e.g., during introspection).
    from data_sources import build_default_data_source

    source = build_default_data_source(
        data_path=filepath,
        mode=mode,  # type: ignore[arg-type]
        allow_fallback=allow_fallback,
    )
    result = source.load()
    print(f"[INFO] {result.summary()}")
    if result.failed_attempts:
        for src_name, err in result.failed_attempts:
            print(f"[INFO]   ↳ fell back from '{src_name}': {err}")
    return result.dataframe


# ──────────────────────────────────────────────────────────────────────────────
# Schema normalisation
# ──────────────────────────────────────────────────────────────────────────────

def normalise_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names to a canonical schema.
    Handles known inconsistencies:
      - 'PAY_1' → 'PAY_0' (some file versions use PAY_1 for September status)
      - 'default payment next month' or 'default.payment.next.month' → 'DEFAULT'
      - Drop 'ID' column if present
    """
    df = df.copy()

    # Build rename map
    rename_map = {}
    for col in df.columns:
        col_upper = col.strip().upper().replace(" ", "_").replace(".", "_")
        if col_upper == "PAY_1":
            rename_map[col] = "PAY_0"
        elif "DEFAULT" in col_upper:
            rename_map[col] = "DEFAULT"
        elif col_upper == "ID":
            rename_map[col] = "ID"

    df.rename(columns=rename_map, inplace=True)

    # Drop ID
    if "ID" in df.columns:
        df.drop(columns=["ID"], inplace=True)
        print("[INFO] Dropped ID column")

    # Verify all expected columns exist
    expected = set(ALL_FEATURE_COLS + [TARGET_COL])
    actual = set(df.columns)
    missing = expected - actual
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Reorder columns to canonical order
    df = df[ALL_FEATURE_COLS + [TARGET_COL]]

    print(f"[INFO] Schema normalised: {list(df.columns)}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Cleaning
# ──────────────────────────────────────────────────────────────────────────────

def clean_categoricals(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Clean undocumented category codes.

    EDUCATION: Values 0, 5, 6 are undocumented in the original dataset description.
               Following standard practice, we merge them into category 4 ('Others').
    MARRIAGE:  Value 0 is undocumented. Merged into category 3 ('Others').

    These decisions are justified because:
      - The original paper (Yeh & Lien, 2009) only defines codes 1-4 for EDUCATION
        and 1-3 for MARRIAGE.
      - Values 0, 5, 6 appear infrequently and lack semantic definition.
      - Merging into 'Others' is the most conservative choice, avoiding the introduction
        of spurious categories that could confuse embedding layers.
    """
    df = df.copy()

    # EDUCATION: merge 0, 5, 6 → 4
    edu_before = df["EDUCATION"].value_counts().sort_index()
    mask = df["EDUCATION"].isin([0, 5, 6])
    n_edu_fixed = mask.sum()
    df.loc[mask, "EDUCATION"] = 4

    # MARRIAGE: merge 0 → 3
    mar_before = df["MARRIAGE"].value_counts().sort_index()
    mask = df["MARRIAGE"] == 0
    n_mar_fixed = mask.sum()
    df.loc[mask, "MARRIAGE"] = 3

    if verbose:
        print(f"[CLEAN] EDUCATION: merged {n_edu_fixed} undocumented values (0/5/6) → 4 (Others)")
        print(f"        Before: {edu_before.to_dict()}")
        print(f"        After:  {df['EDUCATION'].value_counts().sort_index().to_dict()}")
        print(f"[CLEAN] MARRIAGE: merged {n_mar_fixed} undocumented values (0) → 3 (Others)")
        print(f"        Before: {mar_before.to_dict()}")
        print(f"        After:  {df['MARRIAGE'].value_counts().sort_index().to_dict()}")

    return df


def validate_data(df: pd.DataFrame) -> Dict:
    """
    Run comprehensive data quality checks and return a validation report.
    """
    report = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicated_rows": int(df.duplicated().sum()),
        "target_distribution": df[TARGET_COL].value_counts().to_dict(),
        "issues": [],
    }

    # Check for missing values
    if report["total_missing"] > 0:
        report["issues"].append(f"Found {report['total_missing']} missing values")

    # Check for duplicates
    if report["duplicated_rows"] > 0:
        report["issues"].append(
            f"Found {report['duplicated_rows']} duplicate rows "
            f"({100*report['duplicated_rows']/len(df):.2f}%)"
        )

    # Validate SEX
    invalid_sex = set(df["SEX"].unique()) - VALID_SEX
    if invalid_sex:
        report["issues"].append(f"Invalid SEX values: {invalid_sex}")

    # Validate EDUCATION (post-cleaning)
    invalid_edu = set(df["EDUCATION"].unique()) - VALID_EDUCATION_CLEAN
    if invalid_edu:
        report["issues"].append(f"Invalid EDUCATION values: {invalid_edu}")

    # Validate MARRIAGE (post-cleaning)
    invalid_mar = set(df["MARRIAGE"].unique()) - VALID_MARRIAGE_CLEAN
    if invalid_mar:
        report["issues"].append(f"Invalid MARRIAGE values: {invalid_mar}")

    # Validate PAY status
    for col in PAY_STATUS_FEATURES:
        invalid_pay = set(df[col].unique()) - VALID_PAY_STATUS
        if invalid_pay:
            report["issues"].append(f"Invalid {col} values: {invalid_pay}")

    # Check for negative LIMIT_BAL
    if (df["LIMIT_BAL"] <= 0).any():
        report["issues"].append("Found non-positive LIMIT_BAL values")

    # Check AGE range
    if df["AGE"].min() < 18 or df["AGE"].max() > 100:
        report["issues"].append(f"AGE range suspicious: [{df['AGE'].min()}, {df['AGE'].max()}]")

    # Check target is binary
    if set(df[TARGET_COL].unique()) != {0, 1}:
        report["issues"].append(f"Target not binary: {df[TARGET_COL].unique()}")

    # Note: BILL_AMT can legitimately be negative (overpayment / credit balance)
    for col in BILL_AMT_FEATURES:
        n_neg = (df[col] < 0).sum()
        if n_neg > 0:
            report[f"{col}_negative_count"] = int(n_neg)

    if len(report["issues"]) == 0:
        print("[VALIDATE] All checks passed.")
    else:
        for issue in report["issues"]:
            print(f"[VALIDATE] WARNING: {issue}")

    return report


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features that capture domain-relevant interactions.

    These engineered features serve two purposes:
      1. For the Random Forest: explicit interaction features that trees can split on directly.
      2. For analysis / EDA: revealing structures in the data that motivate the transformer
         architecture (e.g., temporal patterns the attention mechanism should capture).

    Note: For the transformer, we typically feed raw features and let attention discover
    interactions. But engineered features can also be added as additional tokens.
    """
    df = df.copy()

    # ── Credit utilisation ratios ──
    # Ratio of bill amount to credit limit — a key risk indicator
    for i in range(1, 7):
        bill_col = f"BILL_AMT{i}"
        df[f"UTIL_RATIO_{i}"] = (df[bill_col] / df["LIMIT_BAL"].replace(0, np.nan)).fillna(0)

    # ── Repayment ratios ──
    # Ratio of payment amount to bill amount — how much of the bill was paid
    for i in range(1, 7):
        bill_col = f"BILL_AMT{i}"
        pay_col = f"PAY_AMT{i}"
        denom = df[bill_col].replace(0, np.nan).abs()
        # Clip to [-5, 5]: ratios beyond this indicate data artefacts (e.g., tiny
        # bill with large payment, or negative bill with large payment). Values of
        # ±5 represent 500% overpayment/underpayment — beyond practical range.
        df[f"REPAY_RATIO_{i}"] = (df[pay_col] / denom).fillna(0).clip(-5, 5)

    # ── Delinquency summary statistics ──
    pay_cols = PAY_STATUS_FEATURES
    pay_vals = df[pay_cols].values

    # Number of months with any delay (PAY > 0)
    df["N_MONTHS_DELAYED"] = (pay_vals > 0).sum(axis=1)
    # Maximum delay severity across 6 months
    df["MAX_DELAY"] = pay_vals.max(axis=1)
    # Most recent delay (PAY_0)
    df["RECENT_DELAY"] = df["PAY_0"]
    # Whether delay is worsening: PAY_0 > mean(PAY_2..PAY_6)
    df["DELAY_TREND"] = df["PAY_0"] - pay_vals[:, 1:].mean(axis=1)
    # Number of months with no consumption (-2)
    df["N_MONTHS_NO_USE"] = (pay_vals == -2).sum(axis=1)

    # ── Bill amount dynamics ──
    bill_vals = df[BILL_AMT_FEATURES].values
    # Bill amount trend (are bills growing or shrinking?)
    # Linear regression slope of BILL_AMT over 6 months per customer
    months = np.arange(6).reshape(1, -1)
    bill_mean = bill_vals.mean(axis=1, keepdims=True)
    month_mean = months.mean()
    numerator = ((months - month_mean) * (bill_vals - bill_mean)).sum(axis=1)
    denominator = ((months - month_mean) ** 2).sum()
    df["BILL_SLOPE"] = numerator / denominator

    # Average utilisation
    util_cols = [f"UTIL_RATIO_{i}" for i in range(1, 7)]
    df["AVG_UTIL_RATIO"] = df[util_cols].mean(axis=1)

    # ── Payment amount dynamics ──
    pay_amt_vals = df[PAY_AMT_FEATURES].values
    df["AVG_PAY_AMT"] = pay_amt_vals.mean(axis=1)
    df["PAY_AMT_VOLATILITY"] = pay_amt_vals.std(axis=1)

    # Balance: total payments vs total bills over 6 months
    df["TOTAL_BILL"] = df[BILL_AMT_FEATURES].sum(axis=1)
    df["TOTAL_PAY"] = df[PAY_AMT_FEATURES].sum(axis=1)
    # Same [-5, 5] clip rationale as per-month REPAY_RATIO above
    df["PAY_BILL_RATIO_TOTAL"] = (df["TOTAL_PAY"] / df["TOTAL_BILL"].replace(0, np.nan).abs()).fillna(0).clip(-5, 5)

    print(f"[ENGINEER] Created {len(df.columns) - len(ALL_FEATURE_COLS) - 1} engineered features")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Splitting and scaling
# ──────────────────────────────────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified three-way split into train / validation / test.

    Stratification ensures the class balance (~22% default) is preserved in each split.
    We split test first, then split the remainder into train and validation. This ensures
    the test set is never seen during any model selection or hyperparameter tuning.
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    # Second split: separate validation from training
    val_fraction = val_size / (1 - test_size)  # adjust for reduced total
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction, stratify=y_temp, random_state=seed
    )

    train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    val_df = pd.concat([X_val, y_val], axis=1).reset_index(drop=True)
    test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        default_rate = split[TARGET_COL].mean()
        print(f"[SPLIT] {name:5s}: {len(split):,} rows | default rate = {default_rate:.4f}")

    return train_df, val_df, test_df


def fit_scalers(
    train_df: pd.DataFrame,
    numerical_cols: List[str],
) -> Dict[str, StandardScaler]:
    """
    Fit StandardScalers on the training set only.

    Returns a dict mapping column name → fitted scaler.
    These scalers are then applied to val/test sets to prevent data leakage.
    """
    scalers = {}
    for col in numerical_cols:
        scaler = StandardScaler()
        scaler.fit(train_df[[col]])
        scalers[col] = scaler
    print(f"[SCALE] Fitted {len(scalers)} scalers on training data")
    return scalers


def apply_scalers(
    df: pd.DataFrame,
    scalers: Dict[str, StandardScaler],
) -> pd.DataFrame:
    """Apply pre-fitted scalers to a dataframe (train, val, or test)."""
    df = df.copy()
    for col, scaler in scalers.items():
        if col in df.columns:
            df[col] = scaler.transform(df[[col]])
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Metadata export (for tokeniser)
# ──────────────────────────────────────────────────────────────────────────────

def compute_feature_metadata(train_df: pd.DataFrame) -> Dict:
    """
    Compute metadata needed by the tokeniser:
      - For categoricals: number of unique categories and mapping
      - For PAY features: unique values and mapping
      - For numericals: mean, std (for reference)
      - Feature ordering for token sequence
    """
    metadata = {
        "categorical_features": {},
        "numerical_features": {},
        "pay_features": {},
        "feature_order": ALL_FEATURE_COLS,
        "n_features": len(ALL_FEATURE_COLS),
    }

    for col in CATEGORICAL_FEATURES:
        unique_vals = sorted(train_df[col].unique())
        metadata["categorical_features"][col] = {
            "n_categories": len(unique_vals),
            "value_to_idx": {int(v): i for i, v in enumerate(unique_vals)},
        }

    for col in PAY_STATUS_FEATURES:
        unique_vals = sorted(train_df[col].unique())
        metadata["pay_features"][col] = {
            "n_categories": len(unique_vals),
            "value_to_idx": {int(v): i for i, v in enumerate(unique_vals)},
        }

    for col in NUMERICAL_FEATURES:
        metadata["numerical_features"][col] = {
            "mean": float(train_df[col].mean()),
            "std": float(train_df[col].std()),
            "min": float(train_df[col].min()),
            "max": float(train_df[col].max()),
        }

    return metadata


# ──────────────────────────────────────────────────────────────────────────────
# Master pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_preprocessing_pipeline(
    data_path: Optional[str] = None,
    output_dir: str = "data/processed",
    include_engineered: bool = True,
    *,
    mode: str = "auto",
    allow_fallback: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict, Dict]:
    """
    End-to-end preprocessing pipeline.

    Args:
        data_path: Optional explicit local .xls/.xlsx path. Bypasses the
            chained loader when set.
        output_dir: Directory for serialized splits, metadata, and reports.
        include_engineered: If True, also save engineered-feature splits.
        mode: Data source mode (``"auto"``/``"api"``/``"local"``).
            ``"auto"`` enables API → local fallback.
        allow_fallback: If False, disables fallback in ``"auto"`` mode.

    Returns:
        train_df, val_df, test_df: Cleaned, split, scaled DataFrames
        metadata: Feature metadata for tokeniser
        validation_report: Data quality report
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load
    df = load_raw_data(data_path, mode=mode, allow_fallback=allow_fallback)

    # 2. Normalise schema
    df = normalise_schema(df)

    # 3. Clean categoricals
    df = clean_categoricals(df)

    # 4. Validate
    validation_report = validate_data(df)

    # 5. Feature engineering (optional — primarily for RF and EDA)
    if include_engineered:
        df_engineered = engineer_features(df)
    else:
        df_engineered = df.copy()

    # 6. Split (stratified)
    train_df, val_df, test_df = split_data(df)  # split the clean (non-engineered) data
    train_eng, val_eng, test_eng = split_data(df_engineered)  # split engineered version too

    # 7. Compute metadata before scaling
    metadata = compute_feature_metadata(train_df)

    # 8. Fit scalers on training set only
    all_numerical = NUMERICAL_FEATURES.copy()
    scalers = fit_scalers(train_df, all_numerical)

    # 9. Apply scalers
    train_scaled = apply_scalers(train_df, scalers)
    val_scaled = apply_scalers(val_df, scalers)
    test_scaled = apply_scalers(test_df, scalers)

    # 10. Save everything
    train_df.to_csv(f"{output_dir}/train_raw.csv", index=False)
    val_df.to_csv(f"{output_dir}/val_raw.csv", index=False)
    test_df.to_csv(f"{output_dir}/test_raw.csv", index=False)
    train_scaled.to_csv(f"{output_dir}/train_scaled.csv", index=False)
    val_scaled.to_csv(f"{output_dir}/val_scaled.csv", index=False)
    test_scaled.to_csv(f"{output_dir}/test_scaled.csv", index=False)
    train_eng.to_csv(f"{output_dir}/train_engineered.csv", index=False)
    val_eng.to_csv(f"{output_dir}/val_engineered.csv", index=False)
    test_eng.to_csv(f"{output_dir}/test_engineered.csv", index=False)

    with open(f"{output_dir}/feature_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(f"{output_dir}/validation_report.json", "w") as f:
        json.dump(validation_report, f, indent=2, default=str)

    print(f"\n[DONE] All outputs saved to {output_dir}/")
    return train_scaled, val_scaled, test_scaled, metadata, validation_report


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_preprocessing_pipeline()
