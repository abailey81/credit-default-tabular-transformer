"""UCI credit-card-default preprocessing pipeline.

End-to-end flow:

    load -> normalise schema -> clean categoricals -> validate ->
    (optional) engineer features -> stratified 70/15/15 split ->
    fit scalers on train only -> apply to val/test -> write metadata.

Raw ingestion (API vs local .xls) lives in :mod:`src.data.sources`; this
module picks up the frame and owns everything model-relevant from there.

The hard rule this file enforces is *train-only fitting*. Every statistic
that can leak test information (StandardScaler mean/std, categorical
vocab, min/max bounds) is computed on the training split and applied
unchanged to val + test. Code that needs to see full-dataset statistics
belongs in EDA, not here.

Writes the following under ``output_dir`` (default ``data/processed/``):

* ``train_raw.csv`` / ``val_raw.csv`` / ``test_raw.csv``       — pre-scaling
* ``train_scaled.csv`` / ``val_scaled.csv`` / ``test_scaled.csv`` — post-scaling
* ``train_engineered.csv`` + siblings                          — with 22 FE cols
* ``feature_metadata.json``                                    — tokeniser input
* ``validation_report.json``                                   — QA trace
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

# pandas / sklearn emit UserWarnings on perfectly fine inputs (categorical
# dtype nags, deprecation nudges). They drown the useful [INFO] logs this
# pipeline prints so we silence at import time.
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical split seed for the whole project. Every split_data call should
# end up with identical row assignments if this stays fixed — the split
# hashes committed in data/processed/SPLIT_HASHES.md are computed with it.
RANDOM_SEED = 42

# Feature groups, ordered to match the 23-token layout used by the tokeniser
# and embedding. Changing the order here without also updating
# ``src.tokenization.embedding.TOKEN_ORDER`` silently desyncs attribution
# plots, so treat this list as part of the public contract.
DEMOGRAPHIC_FEATURES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
PAY_STATUS_FEATURES = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_AMT_FEATURES = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_FEATURES = [f"PAY_AMT{i}" for i in range(1, 7)]
TARGET_COL = "DEFAULT"

CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]
NUMERICAL_FEATURES = ["LIMIT_BAL", "AGE"] + BILL_AMT_FEATURES + PAY_AMT_FEATURES
ORDINAL_PAY_FEATURES = PAY_STATUS_FEATURES

ALL_FEATURE_COLS = DEMOGRAPHIC_FEATURES + PAY_STATUS_FEATURES + BILL_AMT_FEATURES + PAY_AMT_FEATURES

# Domain-allowed value sets used by ``validate_data`` and
# ``clean_categoricals``. The "RAW" variants include undocumented codes
# from the UCI dump; the "CLEAN" variants are what downstream code sees.
VALID_SEX = {1, 2}
VALID_EDUCATION_RAW = {0, 1, 2, 3, 4, 5, 6}
VALID_EDUCATION_CLEAN = {1, 2, 3, 4}
VALID_MARRIAGE_RAW = {0, 1, 2, 3}
VALID_MARRIAGE_CLEAN = {1, 2, 3}
VALID_PAY_STATUS = {-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_raw_data(
    filepath: Optional[str] = None,
    *,
    mode: str = "auto",
    allow_fallback: bool = True,
) -> pd.DataFrame:
    """Load the raw dataframe through the project's canonical source chain.

    Thin wrapper around ``src.data.sources.build_default_data_source``.
    Emits ``[INFO]`` logs that match the ones written elsewhere in the
    pipeline so a full preprocessing run produces one readable transcript.

    Parameters
    ----------
    filepath
        Pin a local ``.xls``/``.xlsx`` file. When set, ``mode`` is ignored.
    mode
        ``"auto"`` (API -> local), ``"api"``, or ``"local"``.
    allow_fallback
        Only meaningful for ``mode="auto"``. ``False`` makes API failure
        fatal instead of silently dropping to the local file.
    """
    from .sources import build_default_data_source

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


# ---------------------------------------------------------------------------
# Schema normalisation
# ---------------------------------------------------------------------------


def normalise_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalise column names to the internal vocabulary.

    UCI has shipped the same dataset with subtly different headers over the
    years: the PAY lag column is sometimes ``PAY_1`` (logically equivalent
    to ``PAY_0`` — both refer to the most recent month), the target column
    appears as "default payment next month" / "DEFAULT.PAYMENT" / similar,
    and an ``ID`` row-index column sometimes sneaks in. We rename everything
    to the project's canonical names (``PAY_0`` for the most recent, plain
    ``DEFAULT`` for the target) and drop ``ID`` if present.

    Raises
    ------
    ValueError
        If any of the 23 feature columns or the target is missing after
        renaming. This is a fail-fast guard: silently producing a subset
        of features would cascade into mysterious shape errors downstream.
    """
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

    # Reorder to the canonical layout. This also implicitly drops any extra
    # columns the source might have attached (UCI sometimes includes an
    # unnamed index column that survives the rename).
    df = df[ALL_FEATURE_COLS + [TARGET_COL]]

    print(f"[INFO] Schema normalised: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def clean_categoricals(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Fold undocumented categorical codes into the "Others" bucket.

    The source paper (Yeh & Lien 2009) documents four EDUCATION levels
    (1=grad school, 2=university, 3=high school, 4=others) and three
    MARRIAGE levels (1=married, 2=single, 3=others). The UCI dump, however,
    contains a handful of rows with EDUCATION in {0, 5, 6} and MARRIAGE=0
    — undocumented and rare (<0.5%). We fold them into the Others bucket
    rather than drop them: dropping would shrink an already imbalanced
    minority class, and treating them as a separate category would leave
    the embedding table with vocab entries that barely get any gradient.

    Verbose mode prints the before/after distribution so a CI log shows
    the fix quantitatively.
    """
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
    """Quality-check the frame and return a structured report.

    Returns a dict (not raises) so the caller decides how to react — the
    master pipeline writes it to ``validation_report.json`` for the audit
    trail. The ``issues`` list is human-readable; everything else is
    machine-parseable for downstream consumers (tests, the reproducibility
    gate) that just want counts.

    Checks performed:

    * Missing value counts per column + total.
    * Duplicate rows (percentage included for context).
    * Categorical values vs ``VALID_*_CLEAN`` — cleaning should already
      have happened, so any hit here is a real bug.
    * PAY status codes in [-2, 8].
    * ``LIMIT_BAL`` strictly positive (a zero/negative limit is almost
      certainly corrupt data, not a legitimate account state).
    * ``AGE`` in [18, 100].
    * Binary target.
    * Negative ``BILL_AMT`` counts are *recorded* but not flagged —
      overpayment / credit balance is a legitimate state.
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

    # Negative BILL_AMT is a *legitimate* state — it means the cardholder
    # overpaid and the bank owes them money. Record the count for audit
    # but do not flag it as an issue.
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


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the 22 engineered features the RF baseline consumes directly.

    The transformer does NOT see these — it ingests the 23 raw tokens and
    lets attention discover interactions. We produce them for the RF and
    the §4 EDA figures, where hand-crafted ratios make the story readable.

    Ratio features are clipped to [-5, 5] so a near-zero denominator can't
    blow up the column statistics. 5 is the smallest value that keeps every
    legitimate case intact in the training split (the empirical 99.9th
    percentile is ~3.4); anything tighter would clip real signal.

    Derived features (grouped):

    * ``UTIL_RATIO_{1..6}``  — bill as fraction of credit limit per month.
    * ``REPAY_RATIO_{1..6}`` — payment / |prev bill| per month, clipped.
    * ``N_MONTHS_DELAYED``   — count of PAY > 0 across the 6 months.
    * ``MAX_DELAY``          — worst PAY status over the window.
    * ``RECENT_DELAY``       — PAY_0 (most recent lag).
    * ``DELAY_TREND``        — PAY_0 minus the mean of PAY_2..PAY_6.
    * ``N_MONTHS_NO_USE``    — count of PAY == -2 (inactive months).
    * ``BILL_SLOPE``         — OLS slope of BILL_AMT over months 0..5.
    * ``AVG_UTIL_RATIO``     — mean utilisation ratio.
    * ``AVG_PAY_AMT``        — mean PAY_AMT.
    * ``PAY_AMT_VOLATILITY`` — std of PAY_AMT.
    * ``TOTAL_BILL`` / ``TOTAL_PAY`` / ``PAY_BILL_RATIO_TOTAL``.
    """
    df = df.copy()

    # Utilisation: bill as fraction of credit limit. LIMIT_BAL should never
    # be zero after validation, but guard anyway — a corrupted row would
    # otherwise silently contaminate the column with inf.
    for i in range(1, 7):
        bill_col = f"BILL_AMT{i}"
        df[f"UTIL_RATIO_{i}"] = (df[bill_col] / df["LIMIT_BAL"].replace(0, np.nan)).fillna(0)

    # Repayment ratio: how much of the previous bill the customer actually
    # paid. |denom| because overpayment (negative BILL_AMT) should produce
    # a positive ratio, not flip the sign.
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
    # Recency trend: is the most recent month worse than the earlier average?
    # Positive values indicate deterioration; the RF picks this up strongly.
    df["DELAY_TREND"] = df["PAY_0"] - pay_vals[:, 1:].mean(axis=1)
    df["N_MONTHS_NO_USE"] = (pay_vals == -2).sum(axis=1)

    # BILL_SLOPE: OLS slope of bill amount regressed on month index. A closed
    # form beats sklearn here — we have 6 points per row across 30k rows and
    # the broadcast-friendly formula is two orders of magnitude faster than
    # iterating np.polyfit per row.
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
    df["PAY_BILL_RATIO_TOTAL"] = (
        (df["TOTAL_PAY"] / df["TOTAL_BILL"].replace(0, np.nan).abs()).fillna(0).clip(-5, 5)
    )

    print(f"[ENGINEER] Created {len(df.columns) - len(ALL_FEATURE_COLS) - 1} engineered features")
    return df


# ---------------------------------------------------------------------------
# Splitting and scaling
# ---------------------------------------------------------------------------


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 train/val/test split on ``DEFAULT``.

    Test is carved off first and then held untouched — it never informs
    model selection. Val comes out of the remainder so the training split
    is always 70% of the original. Every split preserves the base default
    rate (roughly 22%) through stratification; without that, a random
    split can drift the minority-class rate by ~1 percentage point and
    confuse calibration.

    Parameters
    ----------
    df
        Frame containing ``TARGET_COL`` plus any feature columns.
    test_size, val_size
        Fractions of the original. Defaults give 0.70 / 0.15 / 0.15.
    seed
        Passed through to ``train_test_split``. Change this and the
        committed SPLIT_HASHES will no longer match.

    Returns
    -------
    (train_df, val_df, test_df)
        Three frames with contiguous 0..N-1 indexes, target column retained.
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Carve off test first so the remainder (X_temp) is exactly train+val.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    # val_size is a fraction of the *original* frame; to get that same
    # number of rows from the remainder, rescale by (1 - test_size).
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
    """Fit a per-column ``StandardScaler`` on the training split only.

    One scaler per column rather than one multi-column scaler, so callers
    can drop / reorder columns without rebuilding the full transform. The
    "train only" part is the leak guard: if val or test statistics ever
    enter the scaler's fit, downstream metrics become optimistic.
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
    """Apply pre-fitted scalers; columns not in ``scalers`` pass through."""
    df = df.copy()
    for col, scaler in scalers.items():
        if col in df.columns:
            df[col] = scaler.transform(df[[col]])
    return df


# ---------------------------------------------------------------------------
# Metadata export
# ---------------------------------------------------------------------------


def compute_feature_metadata(train_df: pd.DataFrame) -> Dict:
    """Build the tokeniser-ready metadata JSON.

    Produces the contract FeatureEmbedding consumes on instantiation:
    per-feature categorical vocabularies (with a stable sorted-integer
    ordering), per-feature numerical statistics, and the canonical feature
    order. Because vocabularies are computed from the training split only,
    unseen codes at inference raise rather than silently remapping — see
    :mod:`src.tokenization.tokenizer` for the enforcement.
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


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------


def run_preprocessing_pipeline(
    data_path: Optional[str] = None,
    output_dir: str = "data/processed",
    include_engineered: bool = True,
    *,
    mode: str = "auto",
    allow_fallback: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict, Dict]:
    """End-to-end pipeline: load -> normalise -> clean -> split -> scale -> write.

    Returns ``(train_scaled, val_scaled, test_scaled, metadata, validation_report)``
    and writes the full artefact set under ``output_dir``:

    * ``{train,val,test}_raw.csv``        — post-clean, pre-scaling frames.
    * ``{train,val,test}_scaled.csv``     — with fitted StandardScalers applied.
    * ``{train,val,test}_engineered.csv`` — raw + the 22 FE columns (for RF).
    * ``feature_metadata.json``           — tokeniser vocabularies + stats.
    * ``validation_report.json``          — quality-check trace.

    Splitting is done twice: once on the unengineered frame (for the raw /
    scaled CSVs the transformer reads) and once on the engineered frame
    (for the RF). Because both use ``RANDOM_SEED``, the two splits pick the
    *same row indices* — the RF eval and transformer eval compare like-for-like.

    Parameters
    ----------
    data_path, mode, allow_fallback
        Forwarded to :mod:`src.data.sources` — see ``load_raw_data``.
    output_dir
        Destination for the CSVs and JSON artefacts.
    include_engineered
        ``False`` skips feature engineering (the ``*_engineered.csv`` files
        are still written, but identical to the raw frames).
    """
    os.makedirs(output_dir, exist_ok=True)

    df = load_raw_data(data_path, mode=mode, allow_fallback=allow_fallback)
    df = normalise_schema(df)
    df = clean_categoricals(df)
    validation_report = validate_data(df)

    if include_engineered:
        df_engineered = engineer_features(df)
    else:
        df_engineered = df.copy()

    # Two splits, same seed -> same row assignments. This is what makes the
    # per-row comparison in src.evaluation.significance legitimate.
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
        # default=str so numpy ints/bools serialise cleanly. Without this,
        # json.dumps raises TypeError on the NumPy scalars inside the report.
        json.dump(validation_report, f, indent=2, default=str)

    print(f"\n[DONE] All outputs saved to {output_dir}/")
    return train_scaled, val_scaled, test_scaled, metadata, validation_report


if __name__ == "__main__":
    # Smoke-test entry point: runs the full pipeline against the default
    # data path + output dir. Used during local development; CI invokes
    # the dedicated scripts/run_pipeline.py with explicit flags.
    run_preprocessing_pipeline()
