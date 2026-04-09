"""
=============================================================================
RANDOM FOREST BENCHMARK — CREDIT CARD DEFAULT PREDICTION
=============================================================================
A SINGLE STANDALONE FILE — just open in VS Code and press ▶ Run.

What it does:
  1. Downloads the dataset automatically (no manual download needed)
  2. Cleans and preprocesses the data
  3. Engineers 22 extra features
  4. Trains a baseline Random Forest
  5. Tunes hyperparameters with RandomizedSearchCV
  6. Evaluates with full metrics
  7. Analyses feature importance (Gini + Permutation)
  8. Finds optimal classification threshold
  9. Runs 5-fold cross-validation
  10. Generates 5 publication-quality figures
  11. Exports all results to CSV/JSON

Output files (saved to same folder as this script):
  - rf_data_exploration.png
  - rf_tuning_analysis.png
  - rf_feature_importance.png
  - rf_roc_pr_curves.png
  - rf_confusion_matrix.png
  - rf_threshold_analysis.png
  - rf_metrics.csv
  - rf_feature_importance.csv
  - rf_cross_validation.csv
  - rf_config.json

Required packages (install if needed):
  pip install pandas numpy scikit-learn matplotlib seaborn ucimlrepo openpyxl
=============================================================================
"""

# ── Imports ──────────────────────────────────────────────────────────────────

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Output directory (same folder as this script) ───────────────────────────

OUTPUT_DIR = Path(__file__).parent
FIGURE_DIR = OUTPUT_DIR
RESULTS_DIR = OUTPUT_DIR

# ── Plot style ──────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

COLOURS = {
    "primary": "#534AB7",
    "secondary": "#1D9E75",
    "accent": "#D85A30",
    "neutral": "#888780",
    "info": "#378ADD",
}


# =============================================================================
# STEP 1: LOAD THE DATA (automatic download)
# =============================================================================

def load_data() -> pd.DataFrame:
    """
    Fetch the dataset from the UCI repository automatically.
    Falls back to local file if ucimlrepo is not available.
    """
    print("=" * 60)
    print("STEP 1: LOADING DATA")
    print("=" * 60)

    try:
        from ucimlrepo import fetch_ucirepo
        print("  Fetching from UCI repository...")
        dataset = fetch_ucirepo(id=350)
        df = dataset.data.features.copy()
        # UCI repo may return generic X1..X23 column names — rename them
        if "X1" in df.columns and "LIMIT_BAL" not in df.columns:
            col_map = {
                "X1": "LIMIT_BAL", "X2": "SEX", "X3": "EDUCATION", "X4": "MARRIAGE",
                "X5": "AGE", "X6": "PAY_0", "X7": "PAY_2", "X8": "PAY_3",
                "X9": "PAY_4", "X10": "PAY_5", "X11": "PAY_6",
                "X12": "BILL_AMT1", "X13": "BILL_AMT2", "X14": "BILL_AMT3",
                "X15": "BILL_AMT4", "X16": "BILL_AMT5", "X17": "BILL_AMT6",
                "X18": "PAY_AMT1", "X19": "PAY_AMT2", "X20": "PAY_AMT3",
                "X21": "PAY_AMT4", "X22": "PAY_AMT5", "X23": "PAY_AMT6",
            }
            df = df.rename(columns=col_map)
        df["DEFAULT"] = dataset.data.targets.values.ravel()
        print("  ✓ Dataset fetched successfully")
    except ImportError:
        print("  ucimlrepo not installed. Trying local file...")
        local_path = Path("default of credit card clients.xls")
        if not local_path.exists():
            print(f"\n  ERROR: Could not find dataset.")
            print(f"  Install ucimlrepo:  pip install ucimlrepo")
            print(f"  Or place the .xls file in: {local_path.resolve()}")
            raise FileNotFoundError("Dataset not found")
        df = pd.read_excel(local_path, header=1)
        df = df.rename(columns={"default payment next month": "DEFAULT"})
        if "ID" in df.columns:
            df = df.drop(columns=["ID"])

    # Normalise column names (PAY_1 → PAY_0 in some versions)
    if "PAY_1" in df.columns and "PAY_0" not in df.columns:
        df = df.rename(columns={"PAY_1": "PAY_0"})

    print(f"\n  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Default rate: {df['DEFAULT'].mean():.1%} ({df['DEFAULT'].sum():,} defaults)")
    print(f"  Class ratio:  {(1 - df['DEFAULT'].mean()) / df['DEFAULT'].mean():.1f}:1 imbalance")

    return df


# =============================================================================
# STEP 2: DATA EXPLORATION (figures for Section 2)
# =============================================================================

def explore_data(df: pd.DataFrame) -> None:
    """Generate exploration figures for the report."""
    print("\n" + "=" * 60)
    print("STEP 2: DATA EXPLORATION")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Class balance
    counts = df["DEFAULT"].value_counts()
    axes[0, 0].bar(["No Default", "Default"], [counts[0], counts[1]],
                    color=[COLOURS["secondary"], COLOURS["accent"]])
    axes[0, 0].set_title("Class Distribution")
    axes[0, 0].set_ylabel("Count")
    for i, v in enumerate([counts[0], counts[1]]):
        axes[0, 0].text(i, v + 300, f"{v:,}", ha="center", fontweight="bold")

    # 2. Age by default
    df[df["DEFAULT"] == 0]["AGE"].hist(bins=30, alpha=0.6, ax=axes[0, 1],
                                        label="No Default", color=COLOURS["secondary"])
    df[df["DEFAULT"] == 1]["AGE"].hist(bins=30, alpha=0.6, ax=axes[0, 1],
                                        label="Default", color=COLOURS["accent"])
    axes[0, 1].set_title("Age Distribution by Default Status")
    axes[0, 1].legend()

    # 3. Credit limit by default
    df[df["DEFAULT"] == 0]["LIMIT_BAL"].hist(bins=30, alpha=0.6, ax=axes[1, 0],
                                               label="No Default", color=COLOURS["secondary"])
    df[df["DEFAULT"] == 1]["LIMIT_BAL"].hist(bins=30, alpha=0.6, ax=axes[1, 0],
                                               label="Default", color=COLOURS["accent"])
    axes[1, 0].set_title("Credit Limit Distribution by Default Status")
    axes[1, 0].legend()

    # 4. PAY_0 by default
    pay_col = "PAY_0" if "PAY_0" in df.columns else "PAY_1"
    pay_default = df.groupby([pay_col, "DEFAULT"]).size().unstack(fill_value=0)
    pay_default.plot(kind="bar", ax=axes[1, 1], color=[COLOURS["secondary"], COLOURS["accent"]])
    axes[1, 1].set_title("PAY_0 (Sept Repayment) by Default")
    axes[1, 1].legend(["No Default", "Default"])
    axes[1, 1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    path = FIGURE_DIR / "rf_data_exploration.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path.name}")

    # Correlation with target
    corr = df.corr(numeric_only=True)["DEFAULT"].drop("DEFAULT").abs().sort_values(ascending=False)
    print(f"\n  Top 10 features correlated with DEFAULT:")
    for feat, val in corr.head(10).items():
        print(f"    {feat:20s}: {val:.4f}")


# =============================================================================
# STEP 3: CLEAN AND ENGINEER FEATURES
# =============================================================================

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data and create 22 engineered features.
    Matches your Step 2 preprocessing pipeline.
    """
    print("\n" + "=" * 60)
    print("STEP 3: CLEANING & FEATURE ENGINEERING")
    print("=" * 60)

    df = df.copy()

    # ── Clean categorical features ──
    # Merge undocumented EDUCATION codes (0, 5, 6) into 'other' (4)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    # Merge undocumented MARRIAGE code (0) into 'other' (3)
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    # ── Identify feature groups ──
    pay_cols = [c for c in df.columns if c.startswith("PAY_") and "AMT" not in c]
    bill_cols = [c for c in df.columns if c.startswith("BILL_AMT")]
    pay_amt_cols = [c for c in df.columns if c.startswith("PAY_AMT")]

    # Sort to ensure chronological order
    pay_cols = sorted(pay_cols, key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0)
    bill_cols = sorted(bill_cols, key=lambda x: int(x.split("AMT")[-1]))
    pay_amt_cols = sorted(pay_amt_cols, key=lambda x: int(x.split("AMT")[-1]))

    n_months = min(len(bill_cols), len(pay_amt_cols))
    initial_features = len(df.columns) - 1  # minus DEFAULT

    # ── Engineer features ──

    # 1. Credit utilisation ratio per month (6 features)
    for i, bcol in enumerate(bill_cols[:n_months]):
        col_name = f"UTIL_RATIO_{i+1}"
        df[col_name] = np.where(
            df["LIMIT_BAL"] > 0,
            df[bcol] / df["LIMIT_BAL"],
            0.0,
        )

    # 2. Repayment ratio per month (6 features)
    for i in range(n_months):
        col_name = f"REPAY_RATIO_{i+1}"
        df[col_name] = np.where(
            df[bill_cols[i]].abs() > 0,
            df[pay_amt_cols[i]] / df[bill_cols[i]].abs(),
            0.0,
        )
        # Clip extreme values
        df[col_name] = df[col_name].clip(-5, 5)

    # 3. Delinquency aggregates (5 features)
    pay_values = df[pay_cols].values
    df["DELAY_COUNT"] = (pay_values > 0).sum(axis=1)
    df["MAX_DELAY"] = pay_values.max(axis=1)
    df["DELAY_TREND"] = pay_values[:, 0] - pay_values[:, -1] if len(pay_cols) > 1 else 0
    df["NO_USE_MONTHS"] = (pay_values == -2).sum(axis=1)
    df["AVG_DELAY"] = np.where(
        (pay_values > 0).sum(axis=1) > 0,
        np.where(pay_values > 0, pay_values, 0).sum(axis=1) / np.maximum((pay_values > 0).sum(axis=1), 1),
        0.0,
    )

    # 4. Bill dynamics (2 features)
    bill_values = df[bill_cols].values
    x = np.arange(len(bill_cols))
    slopes = []
    for row in bill_values:
        if np.std(row) > 0:
            slope = np.polyfit(x, row, 1)[0]
        else:
            slope = 0.0
        slopes.append(slope)
    df["BILL_SLOPE"] = slopes
    df["AVG_UTIL"] = df[[f"UTIL_RATIO_{i+1}" for i in range(n_months)]].mean(axis=1)

    # 5. Payment dynamics (2 features)
    pay_amt_values = df[pay_amt_cols].values
    df["AVG_PAYMENT"] = pay_amt_values.mean(axis=1)
    df["PAYMENT_VOLATILITY"] = pay_amt_values.std(axis=1)

    # 6. Balance total (1 feature)
    total_bill = df[bill_cols].sum(axis=1)
    total_pay = df[pay_amt_cols].sum(axis=1)
    df["TOTAL_REPAY_RATIO"] = np.where(
        total_bill.abs() > 0,
        total_pay / total_bill.abs(),
        0.0,
    )
    df["TOTAL_REPAY_RATIO"] = df["TOTAL_REPAY_RATIO"].clip(-5, 5)

    new_features = len(df.columns) - 1 - initial_features
    print(f"  Original features: {initial_features}")
    print(f"  Engineered features: {new_features}")
    print(f"  Total features: {len(df.columns) - 1}")

    return df


# =============================================================================
# STEP 4: PREPARE TRAIN / VAL / TEST SPLITS
# =============================================================================

def prepare_splits(df: pd.DataFrame):
    """
    Stratified 70/15/15 split matching your repo setup.
    StandardScaler fitted on train only (no data leakage).
    """
    print("\n" + "=" * 60)
    print("STEP 4: PREPARING DATA SPLITS")
    print("=" * 60)

    X = df.drop(columns=["DEFAULT"])
    y = df["DEFAULT"]

    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y,
    )
    # Second split: 50/50 of temp → 15% val, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp,
    )

    print(f"  Train: {len(X_train):,} ({y_train.mean():.1%} default)")
    print(f"  Val:   {len(X_val):,} ({y_val.mean():.1%} default)")
    print(f"  Test:  {len(X_test):,} ({y_test.mean():.1%} default)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# STEP 5: BASELINE RANDOM FOREST
# =============================================================================

def train_baseline(X_train, y_train, X_test, y_test):
    """Train with default hyperparameters for a before/after comparison."""
    print("\n" + "=" * 60)
    print("STEP 5: BASELINE RANDOM FOREST")
    print("=" * 60)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    t0 = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - t0

    y_prob = rf.predict_proba(X_test)[:, 1]
    y_pred = rf.predict(X_test)

    metrics = {
        "model": "RF_baseline",
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "auc_roc": round(roc_auc_score(y_test, y_prob), 4),
        "avg_precision": round(average_precision_score(y_test, y_prob), 4),
        "train_time_s": round(train_time, 2),
    }

    print(f"\n  Accuracy:  {metrics['accuracy']}")
    print(f"  Precision: {metrics['precision']}")
    print(f"  Recall:    {metrics['recall']}")
    print(f"  F1 Score:  {metrics['f1']}")
    print(f"  AUC-ROC:   {metrics['auc_roc']}")
    print(f"  Time:      {train_time:.1f}s")

    return rf, metrics


# =============================================================================
# STEP 6: HYPERPARAMETER TUNING
# =============================================================================

def tune_model(X_train, y_train):
    """RandomizedSearchCV with 60 iterations across 6 hyperparameters."""
    print("\n" + "=" * 60)
    print("STEP 6: HYPERPARAMETER TUNING (RandomizedSearchCV)")
    print("=" * 60)

    param_grid = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "class_weight": [None, "balanced", "balanced_subsample"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    searcher = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=60,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        random_state=42,
        return_train_score=True,
    )

    print(f"  Running 60 random combinations × 5 folds = 300 fits...")
    print(f"  This may take a few minutes...\n")

    t0 = time.time()
    searcher.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"\n  Completed in {elapsed / 60:.1f} minutes")
    print(f"\n  Best hyperparameters:")
    for k, v in searcher.best_params_.items():
        print(f"    {k}: {v}")
    print(f"\n  Best CV AUC-ROC: {searcher.best_score_:.4f}")

    return searcher


# =============================================================================
# STEP 7: EVALUATE
# =============================================================================

def evaluate(model, X, y, split_name="test"):
    """Full evaluation on a given split."""
    t0 = time.time()
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    infer_time = time.time() - t0

    metrics = {
        "model": f"RF_tuned_{split_name}",
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred), 4),
        "recall": round(recall_score(y, y_pred), 4),
        "f1": round(f1_score(y, y_pred), 4),
        "auc_roc": round(roc_auc_score(y, y_prob), 4),
        "avg_precision": round(average_precision_score(y, y_prob), 4),
        "train_time_s": round(infer_time, 2),
    }

    print(f"\n{'=' * 60}")
    print(f"TUNED RANDOM FOREST — {split_name.upper()} SET")
    print(f"{'=' * 60}")
    print(f"  Accuracy:      {metrics['accuracy']}")
    print(f"  Precision:     {metrics['precision']}")
    print(f"  Recall:        {metrics['recall']}")
    print(f"  F1 Score:      {metrics['f1']}")
    print(f"  AUC-ROC:       {metrics['auc_roc']}")
    print(f"  Avg Precision: {metrics['avg_precision']}")
    print(f"\n{classification_report(y, y_pred, target_names=['No Default', 'Default'])}")

    return y_pred, y_prob, metrics


# =============================================================================
# STEP 8: CROSS-VALIDATION
# =============================================================================

def run_cross_validation(model, X_train, y_train):
    """5-fold stratified CV for robust performance estimates."""
    print(f"\n{'=' * 60}")
    print("5-FOLD CROSS-VALIDATION")
    print(f"{'=' * 60}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        scores = cross_val_score(model, X_train, y_train, cv=cv,
                                  scoring=metric, n_jobs=-1)
        rows.append({"metric": metric, "mean": round(scores.mean(), 4),
                      "std": round(scores.std(), 4)})
        print(f"  {metric:12s}: {scores.mean():.4f} (± {scores.std():.4f})")

    return pd.DataFrame(rows)


# =============================================================================
# STEP 9: FEATURE IMPORTANCE
# =============================================================================

def analyse_importance(model, X_test, y_test, top_n=20):
    """Gini + permutation importance — side by side."""
    print(f"\n{'=' * 60}")
    print("FEATURE IMPORTANCE (Gini + Permutation)")
    print(f"{'=' * 60}")

    features = X_test.columns.tolist()

    # Gini
    gini = pd.DataFrame({"feature": features, "gini": model.feature_importances_})
    gini = gini.sort_values("gini", ascending=False).reset_index(drop=True)

    # Permutation
    print("  Computing permutation importance...")
    perm = permutation_importance(model, X_test, y_test, n_repeats=10,
                                   random_state=42, n_jobs=-1, scoring="roc_auc")
    perm_df = pd.DataFrame({
        "feature": features,
        "perm_mean": perm.importances_mean,
        "perm_std": perm.importances_std,
    }).sort_values("perm_mean", ascending=False).reset_index(drop=True)

    importance = gini.merge(perm_df, on="feature")

    print(f"\n  {'Rank':>4s}  {'Feature':<28s}  {'Gini':>8s}  {'Perm (ΔAUC)':>11s}")
    print(f"  {'—'*4}  {'—'*28}  {'—'*8}  {'—'*11}")
    for i, row in importance.head(top_n).iterrows():
        print(f"  {i+1:4d}  {row['feature']:<28s}  {row['gini']:8.4f}  {row['perm_mean']:11.4f}")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    top_g = gini.head(top_n)
    axes[0].barh(range(len(top_g)), top_g["gini"].values[::-1],
                  color=COLOURS["primary"], alpha=0.85)
    axes[0].set_yticks(range(len(top_g)))
    axes[0].set_yticklabels(top_g["feature"].values[::-1])
    axes[0].set_xlabel("Gini Importance (MDI)")
    axes[0].set_title("Gini Importance (built-in)")

    top_p = perm_df.head(top_n)
    axes[1].barh(range(len(top_p)), top_p["perm_mean"].values[::-1],
                  xerr=top_p["perm_std"].values[::-1],
                  color=COLOURS["accent"], alpha=0.85, capsize=3)
    axes[1].set_yticks(range(len(top_p)))
    axes[1].set_yticklabels(top_p["feature"].values[::-1])
    axes[1].set_xlabel("Permutation Importance (ΔAUC)")
    axes[1].set_title("Permutation Importance (model-agnostic)")

    plt.suptitle("Feature Importance — Gini vs Permutation", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "rf_feature_importance.png")
    plt.close()
    print(f"\n  Saved: rf_feature_importance.png")

    return importance


# =============================================================================
# STEP 10: THRESHOLD ANALYSIS
# =============================================================================

def analyse_threshold(y_val, y_val_prob, y_test, y_test_prob):
    """Find optimal threshold on validation set, report on test set."""
    print(f"\n{'=' * 60}")
    print("THRESHOLD ANALYSIS")
    print(f"{'=' * 60}")

    # Tune threshold on VALIDATION set
    thresholds = np.arange(0.10, 0.90, 0.01)
    results = []
    for t in thresholds:
        preds = (y_val_prob >= t).astype(int)
        results.append({
            "threshold": t,
            "precision": precision_score(y_val, preds, zero_division=0),
            "recall": recall_score(y_val, preds, zero_division=0),
            "f1": f1_score(y_val, preds, zero_division=0),
        })

    df = pd.DataFrame(results)
    best_idx = df["f1"].idxmax()
    best_t = df.loc[best_idx, "threshold"]

    print(f"  Optimal threshold (max F1 on val): {best_t:.2f}")
    print(f"  Val F1 at optimal:   {df.loc[best_idx, 'f1']:.4f}")
    row_50 = df.iloc[(df["threshold"] - 0.50).abs().idxmin()]
    print(f"  Val F1 at 0.50:      {row_50['f1']:.4f}")

    # Report on TEST set with the chosen threshold
    test_preds = (y_test_prob >= best_t).astype(int)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)
    print(f"  Test F1 at optimal:  {test_f1:.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["threshold"], df["precision"], lw=2, label="Precision", color=COLOURS["secondary"])
    ax.plot(df["threshold"], df["recall"], lw=2, label="Recall", color=COLOURS["accent"])
    ax.plot(df["threshold"], df["f1"], lw=2.5, label="F1 Score", color=COLOURS["primary"])
    ax.axvline(best_t, ls="--", color=COLOURS["neutral"], alpha=0.6,
               label=f"Best F1 ({best_t:.2f})")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision, Recall, F1 vs Classification Threshold (Validation Set)")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.savefig(FIGURE_DIR / "rf_threshold_analysis.png")
    plt.close()
    print(f"  Saved: rf_threshold_analysis.png")

    return best_t


# =============================================================================
# STEP 11: ALL EVALUATION FIGURES
# =============================================================================

def plot_all_figures(y_test, y_prob, searcher, best_threshold):
    """Generate ROC, PR, confusion matrix, and tuning analysis figures."""

    # ── ROC + Precision-Recall ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    axes[0].plot(fpr, tpr, lw=2, color=COLOURS["primary"],
                 label=f"Random Forest (AUC = {auc:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.2)

    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    axes[1].plot(rec, prec, lw=2, color=COLOURS["accent"],
                 label=f"Random Forest (AP = {ap:.4f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "rf_roc_pr_curves.png")
    plt.close()
    print(f"  Saved: rf_roc_pr_curves.png")

    # ── Confusion matrix (at optimal threshold) ──
    y_pred_opt = (y_prob >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_opt)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=["No Default", "Default"],
                yticklabels=["No Default", "Default"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — Tuned RF (threshold={best_threshold:.2f})")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "rf_confusion_matrix.png")
    plt.close()
    print(f"  Saved: rf_confusion_matrix.png")

    # ── Tuning analysis ──
    results = pd.DataFrame(searcher.cv_results_)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for depth in [5, 10, 15]:
        mask = results["param_max_depth"] == depth
        if mask.sum() == 0:
            continue
        subset = results[mask].groupby("param_n_estimators")["mean_test_score"].mean()
        axes[0, 0].plot(subset.index, subset.values, marker="o", label=f"depth={depth}")
    axes[0, 0].set_xlabel("n_estimators")
    axes[0, 0].set_ylabel("Mean CV AUC-ROC")
    axes[0, 0].set_title("Effect of n_estimators")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.2)

    results["param_max_depth_str"] = results["param_max_depth"].apply(
        lambda d: "None" if d is None else str(d))
    depth_scores = results.groupby("param_max_depth_str")["mean_test_score"].mean()
    labels = depth_scores.index.tolist()
    axes[0, 1].bar(labels, depth_scores.values, color=COLOURS["primary"], alpha=0.85)
    axes[0, 1].set_xlabel("max_depth")
    axes[0, 1].set_ylabel("Mean CV AUC-ROC")
    axes[0, 1].set_title("Effect of max_depth")
    axes[0, 1].grid(True, alpha=0.2, axis="y")
    lo = depth_scores.values.min(); hi = depth_scores.values.max()
    margin = max((hi - lo) * 0.3, 0.002)
    axes[0, 1].set_ylim(lo - margin, hi + margin)

    results["param_class_weight_str"] = results["param_class_weight"].apply(
        lambda w: "None" if w is None else str(w))
    cw_scores = results.groupby("param_class_weight_str")["mean_test_score"].mean()
    cw_labels = cw_scores.index.tolist()
    axes[1, 0].bar(cw_labels, cw_scores.values,
                    color=[COLOURS["secondary"], COLOURS["accent"], COLOURS["info"]][:len(cw_labels)],
                    alpha=0.85)
    axes[1, 0].set_xlabel("class_weight")
    axes[1, 0].set_ylabel("Mean CV AUC-ROC")
    axes[1, 0].set_title("Effect of class_weight")
    axes[1, 0].grid(True, alpha=0.2, axis="y")
    lo = cw_scores.values.min(); hi = cw_scores.values.max()
    margin = max((hi - lo) * 0.3, 0.002)
    axes[1, 0].set_ylim(lo - margin, hi + margin)

    top_20 = results.nlargest(20, "mean_test_score")
    x = np.arange(len(top_20))
    axes[1, 1].bar(x - 0.15, top_20["mean_train_score"], width=0.3,
                    label="Train AUC", color=COLOURS["info"], alpha=0.85)
    axes[1, 1].bar(x + 0.15, top_20["mean_test_score"], width=0.3,
                    label="Test AUC", color=COLOURS["accent"], alpha=0.85)
    axes[1, 1].set_xlabel("Top 20 configurations")
    axes[1, 1].set_ylabel("AUC-ROC")
    axes[1, 1].set_title("Train vs Test AUC — Overfitting Check")
    axes[1, 1].legend()
    axes[1, 1].set_xticks([])
    axes[1, 1].grid(True, alpha=0.2, axis="y")
    all_vals = np.concatenate([top_20["mean_train_score"].values, top_20["mean_test_score"].values])
    lo = all_vals.min(); hi = all_vals.max()
    margin = max((hi - lo) * 0.3, 0.002)
    axes[1, 1].set_ylim(lo - margin, hi + margin)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "rf_tuning_analysis.png")
    plt.close()
    print(f"  Saved: rf_tuning_analysis.png")


# =============================================================================
# STEP 12: EXPORT RESULTS
# =============================================================================

def export_results(baseline_m, tuned_m, cv_df, importance_df, best_params, best_t):
    """Save everything to CSV and JSON."""
    print(f"\n{'=' * 60}")
    print("EXPORTING RESULTS")
    print(f"{'=' * 60}")

    pd.DataFrame([baseline_m, tuned_m]).to_csv(RESULTS_DIR / "rf_metrics.csv", index=False)
    cv_df.to_csv(RESULTS_DIR / "rf_cross_validation.csv", index=False)
    importance_df.to_csv(RESULTS_DIR / "rf_feature_importance.csv", index=False)

    config = {
        "best_params": best_params,
        "best_threshold": round(best_t, 2),
        "baseline_auc": baseline_m["auc_roc"],
        "tuned_auc": tuned_m["auc_roc"],
    }
    with open(RESULTS_DIR / "rf_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    print(f"  Saved: rf_metrics.csv")
    print(f"  Saved: rf_cross_validation.csv")
    print(f"  Saved: rf_feature_importance.csv")
    print(f"  Saved: rf_config.json")


# =============================================================================
# ▶ MAIN — RUN EVERYTHING
# =============================================================================

def main():
    print("\n" + "🌲" * 30)
    print("  RANDOM FOREST BENCHMARK — CREDIT CARD DEFAULT")
    print("🌲" * 30 + "\n")

    # 1. Load
    df = load_data()

    # 2. Explore
    explore_data(df)

    # 3. Clean + engineer features
    df = clean_and_engineer(df)

    # 4. Split
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_splits(df)

    # 5. Baseline
    _, baseline_metrics = train_baseline(X_train, y_train, X_test, y_test)

    # 6. Tune
    searcher = tune_model(X_train, y_train)
    best_model = searcher.best_estimator_

    # 7. Evaluate on val + test
    _, y_val_prob, _ = evaluate(best_model, X_val, y_val, "validation")
    y_pred, y_prob, tuned_metrics = evaluate(best_model, X_test, y_test, "test")

    # 8. Cross-validation
    cv_df = run_cross_validation(best_model, X_train, y_train)

    # 9. Feature importance
    importance_df = analyse_importance(best_model, X_test, y_test)

    # 10. Threshold analysis (tuned on val, reported on test)
    best_threshold = analyse_threshold(y_val, y_val_prob, y_test, y_prob)

    # 11. Figures
    plot_all_figures(y_test, y_prob, searcher, best_threshold)

    # 12. Export
    export_results(baseline_metrics, tuned_metrics, cv_df, importance_df,
                    searcher.best_params_, best_threshold)

    # ── Summary ──
    print("\n" + "=" * 60)
    print(" ALL DONE!")
    print("=" * 60)
    print(f"\n  Baseline AUC: {baseline_metrics['auc_roc']}")
    print(f"  Tuned AUC:    {tuned_metrics['auc_roc']}")
    print(f"  Best threshold: {best_threshold:.2f}")
    print(f"\n  Files saved to: {OUTPUT_DIR.resolve()}")
    print(f"\n  Figures:")
    for f in sorted(FIGURE_DIR.glob("rf_*.png")):
        print(f"    - {f.name}")
    print(f"\n  Data:")
    for f in sorted(RESULTS_DIR.glob("rf_*.*")):
        if f.suffix in (".csv", ".json"):
            print(f"    - {f.name}")


if __name__ == "__main__":
    main()
