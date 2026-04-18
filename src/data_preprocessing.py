"""UCI credit-card-default preprocessing.

load → normalise schema → clean categoricals → validate → (optional) engineer
features → stratified 70/15/15 split → fit scalers on train only → apply to
val/test. Raw ingestion lives in :mod:`data_sources`.
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


# Constants

RANDOM_SEED = 42

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
ORDINAL_PAY_FEATURES = PAY_STATUS_FEATURES

ALL_FEATURE_COLS = (
    DEMOGRAPHIC_FEATURES + PAY_STATUS_FEATURES + BILL_AMT_FEATURES + PAY_AMT_FEATURES
)

VALID_SEX = {1, 2}
VALID_EDUCATION_RAW = {0, 1, 2, 3, 4, 5, 6}
VALID_EDUCATION_CLEAN = {1, 2, 3, 4}
VALID_MARRIAGE_RAW = {0, 1, 2, 3}
VALID_MARRIAGE_CLEAN = {1, 2, 3}
VALID_PAY_STATUS = {-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8}


# Loading

def load_raw_data(
    filepath: Optional[str] = None,
    *,
    mode: str = "auto",
    allow_fallback: bool = True,
) -> pd.DataFrame:
    """Load the raw dataframe. ``filepath`` pins a local file; otherwise
    ``mode`` selects auto (API → local), api, or local. ``allow_fallback=False``
    in auto mode turns API failure into a hard error."""
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
            print(f"[INFO]   fell back from '{src_name}': {err}")
    return result.dataframe


# Schema normalisation

def normalise_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalise column names. PAY_1→PAY_0, any `default payment...`
    variant → DEFAULT, drop ID if present."""
    df = df.copy()

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

    if "ID" in df.columns:
        df.drop(columns=["ID"], inplace=True)
        print("[INFO] Dropped ID column")

    expected = set(ALL_FEATURE_COLS + [TARGET_COL])
    actual = set(df.columns)
    missing = expected - actual
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df = df[ALL_FEATURE_COLS + [TARGET_COL]]

    print(f"[INFO] Schema normalised: {list(df.columns)}")
    return df


# Cleaning

def clean_categoricals(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Fold undocumented codes into the Others bucket.
    EDUCATION {0,5,6} → 4, MARRIAGE {0} → 3. The codes are rare and
    unlabelled in the source paper."""
    df = df.copy()

    edu_before = df["EDUCATION"].value_counts().sort_index()
    mask = df["EDUCATION"].isin([0, 5, 6])
    n_edu_fixed = mask.sum()
    df.loc[mask, "EDUCATION"] = 4

    mar_before = df["MARRIAGE"].value_counts().sort_index()
    mask = df["MARRIAGE"] == 0
    n_mar_fixed = mask.sum()
    df.loc[mask, "MARRIAGE"] = 3

    if verbose:
        print(f"[CLEAN] EDUCATION: merged {n_edu_fixed} undocumented values (0/5/6) -> 4 (Others)")
        print(f"        Before: {edu_before.to_dict()}")
        print(f"        After:  {df['EDUCATION'].value_counts().sort_index().to_dict()}")
        print(f"[CLEAN] MARRIAGE: merged {n_mar_fixed} undocumented values (0) -> 3 (Others)")
        print(f"        Before: {mar_before.to_dict()}")
        print(f"        After:  {df['MARRIAGE'].value_counts().sort_index().to_dict()}")

    return df


def validate_data(df: pd.DataFrame) -> Dict:
    """Quality-check the frame; return a report dict with any issues found."""
    report = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicated_rows": int(df.duplicated().sum()),
        "target_distribution": df[TARGET_COL].value_counts().to_dict(),
        "issues": [],
    }

    if report["total_missing"] > 0:
        report["issues"].append(f"Found {report['total_missing']} missing values")

    if report["duplicated_rows"] > 0:
        report["issues"].append(
            f"Found {report['duplicated_rows']} duplicate rows "
            f"({100*report['duplicated_rows']/len(df):.2f}%)"
        )

    invalid_sex = set(df["SEX"].unique()) - VALID_SEX
    if invalid_sex:
        report["issues"].append(f"Invalid SEX values: {invalid_sex}")

    invalid_edu = set(df["EDUCATION"].unique()) - VALID_EDUCATION_CLEAN
    if invalid_edu:
        report["issues"].append(f"Invalid EDUCATION values: {invalid_edu}")

    invalid_mar = set(df["MARRIAGE"].unique()) - VALID_MARRIAGE_CLEAN
    if invalid_mar:
        report["issues"].append(f"Invalid MARRIAGE values: {invalid_mar}")

    for col in PAY_STATUS_FEATURES:
        invalid_pay = set(df[col].unique()) - VALID_PAY_STATUS
        if invalid_pay:
            report["issues"].append(f"Invalid {col} values: {invalid_pay}")

    if (df["LIMIT_BAL"] <= 0).any():
        report["issues"].append("Found non-positive LIMIT_BAL values")

    if df["AGE"].min() < 18 or df["AGE"].max() > 100:
        report["issues"].append(f"AGE range suspicious: [{df['AGE'].min()}, {df['AGE'].max()}]")

    if set(df[TARGET_COL].unique()) != {0, 1}:
        report["issues"].append(f"Target not binary: {df[TARGET_COL].unique()}")

    # neg BILL_AMT is legit (overpayment / credit balance), just record the count
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


# Feature engineering

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived features the RF consumes directly (also surfaced in EDA).
    Ratios get clipped to ±5 so a near-zero denominator can't blow up."""
    df = df.copy()

    for i in range(1, 7):
        bill_col = f"BILL_AMT{i}"
        df[f"UTIL_RATIO_{i}"] = (df[bill_col] / df["LIMIT_BAL"].replace(0, np.nan)).fillna(0)

    for i in range(1, 7):
        bill_col = f"BILL_AMT{i}"
        pay_col = f"PAY_AMT{i}"
        denom = df[bill_col].replace(0, np.nan).abs()
        df[f"REPAY_RATIO_{i}"] = (df[pay_col] / denom).fillna(0).clip(-5, 5)

    pay_cols = PAY_STATUS_FEATURES
    pay_vals = df[pay_cols].values

    df["N_MONTHS_DELAYED"] = (pay_vals > 0).sum(axis=1)
    df["MAX_DELAY"] = pay_vals.max(axis=1)
    df["RECENT_DELAY"] = df["PAY_0"]
    df["DELAY_TREND"] = df["PAY_0"] - pay_vals[:, 1:].mean(axis=1)
    df["N_MONTHS_NO_USE"] = (pay_vals == -2).sum(axis=1)

    bill_vals = df[BILL_AMT_FEATURES].values
    months = np.arange(6).reshape(1, -1)
    bill_mean = bill_vals.mean(axis=1, keepdims=True)
    month_mean = months.mean()
    numerator = ((months - month_mean) * (bill_vals - bill_mean)).sum(axis=1)
    denominator = ((months - month_mean) ** 2).sum()
    df["BILL_SLOPE"] = numerator / denominator

    util_cols = [f"UTIL_RATIO_{i}" for i in range(1, 7)]
    df["AVG_UTIL_RATIO"] = df[util_cols].mean(axis=1)

    pay_amt_vals = df[PAY_AMT_FEATURES].values
    df["AVG_PAY_AMT"] = pay_amt_vals.mean(axis=1)
    df["PAY_AMT_VOLATILITY"] = pay_amt_vals.std(axis=1)

    df["TOTAL_BILL"] = df[BILL_AMT_FEATURES].sum(axis=1)
    df["TOTAL_PAY"] = df[PAY_AMT_FEATURES].sum(axis=1)
    df["PAY_BILL_RATIO_TOTAL"] = (df["TOTAL_PAY"] / df["TOTAL_BILL"].replace(0, np.nan).abs()).fillna(0).clip(-5, 5)

    print(f"[ENGINEER] Created {len(df.columns) - len(ALL_FEATURE_COLS) - 1} engineered features")
    return df


# Splitting and scaling

def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 (train/val/test) on DEFAULT.

    Test comes out first so it never touches model selection; val is carved
    from the remainder. Every split preserves the base rate.
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    val_fraction = val_size / (1 - test_size)
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
    """Per-column StandardScaler, fit on train only (leakage guard)."""
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
    """Apply pre-fitted scalers in-place-of-values."""
    df = df.copy()
    for col, scaler in scalers.items():
        if col in df.columns:
            df[col] = scaler.transform(df[[col]])
    return df


# Metadata export

def compute_feature_metadata(train_df: pd.DataFrame) -> Dict:
    """Tokeniser-ready metadata: cat mappings, num stats, feature order."""
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


# Master pipeline

def run_preprocessing_pipeline(
    data_path: Optional[str] = None,
    output_dir: str = "data/processed",
    include_engineered: bool = True,
    *,
    mode: str = "auto",
    allow_fallback: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict, Dict]:
    """End-to-end pipeline. Returns (train, val, test, metadata, report) and
    writes raw/scaled/engineered CSVs + metadata/report JSON under ``output_dir``.
    ``mode``/``allow_fallback`` forward to :mod:`data_sources`."""
    os.makedirs(output_dir, exist_ok=True)

    df = load_raw_data(data_path, mode=mode, allow_fallback=allow_fallback)
    df = normalise_schema(df)
    df = clean_categoricals(df)
    validation_report = validate_data(df)

    if include_engineered:
        df_engineered = engineer_features(df)
    else:
        df_engineered = df.copy()

    train_df, val_df, test_df = split_data(df)
    train_eng, val_eng, test_eng = split_data(df_engineered)

    metadata = compute_feature_metadata(train_df)

    all_numerical = NUMERICAL_FEATURES.copy()
    scalers = fit_scalers(train_df, all_numerical)

    train_scaled = apply_scalers(train_df, scalers)
    val_scaled = apply_scalers(val_df, scalers)
    test_scaled = apply_scalers(test_df, scalers)

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


if __name__ == "__main__":
    run_preprocessing_pipeline()
