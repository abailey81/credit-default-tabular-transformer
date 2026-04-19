"""Tuned Random Forest benchmark.

Shares preprocessing with the transformer (identical features), then runs a
200-iter RandomizedSearchCV on a 7-param grid, 5-fold CV on the winner, and
writes metrics + feature importance + figures under results/baseline/ and
figures/baseline/. Mirrors train.py's artefact layout."""

import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
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

from ..data.preprocessing import (
    load_raw_data,
    normalise_schema,
    clean_categoricals,
    engineer_features,
    split_data,
    TARGET_COL,
    RANDOM_SEED,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# Constants

TUNING_GRID: Dict[str, list] = {
    "n_estimators":      [100, 200, 300, 500, 1000],
    "max_depth":         [5, 10, 15, 20, 30, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf":  [1, 2, 4, 8],
    "max_features":      ["sqrt", "log2", 0.3, 0.5, 0.7],
    "class_weight":      [None, "balanced", "balanced_subsample"],
    "criterion":         ["gini", "entropy"],
}

# palette matches eda.py
PALETTE = {
    "primary": "#534AB7",
    "secondary": "#1D9E75",
    "accent": "#D85A30",
    "neutral": "#6B7280",
    "info": "#378ADD",
}


def set_rf_style() -> None:
    """Matplotlib rc tweaks for 300 DPI serif figures."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


# Training

def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = RANDOM_SEED,
) -> Tuple[RandomForestClassifier, float]:
    """100-tree RF at sklearn defaults; returns ``(model, seconds)``."""
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    t0 = time.time()
    rf.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"[RF] Baseline trained in {elapsed:.1f}s (100 estimators, default params)")
    return rf, elapsed


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Optional[Dict[str, list]] = None,
    n_iter: int = 60,
    n_cv_folds: int = 5,
    seed: int = RANDOM_SEED,
) -> RandomizedSearchCV:
    """RandomizedSearchCV on stratified folds, scoring AUC-ROC. Returns the
    full searcher (caller wants ``best_estimator_`` + ``cv_results_``)."""
    if param_grid is None:
        param_grid = TUNING_GRID

    cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=seed)
    rf = RandomForestClassifier(random_state=seed, n_jobs=-1)

    searcher = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        random_state=seed,
        return_train_score=True,
    )

    total_fits = n_iter * n_cv_folds
    print(f"[RF] Tuning: {n_iter} combinations x {n_cv_folds} folds = {total_fits} fits")

    t0 = time.time()
    searcher.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"[RF] Tuning completed in {elapsed / 60:.1f} minutes")
    print(f"[RF] Best CV AUC-ROC: {searcher.best_score_:.4f}")
    print(f"[RF] Best parameters:")
    for k, v in searcher.best_params_.items():
        print(f"      {k}: {v}")

    return searcher


# Evaluation

def evaluate_model(
    model: RandomForestClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str = "test",
    threshold: float = 0.5,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Score a split at the given threshold.

    Returns ``(metrics, y_pred, y_prob)``; metrics cover acc/prec/rec/F1/
    AUC-ROC/AP + inference time.
    """
    t0 = time.time()
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    infer_time = time.time() - t0

    metrics = {
        "split": split_name,
        "threshold": threshold,
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y, y_pred, zero_division=0), 4),
        "auc_roc": round(roc_auc_score(y, y_prob), 4),
        "avg_precision": round(average_precision_score(y, y_prob), 4),
        "inference_time_s": round(infer_time, 3),
    }

    print(f"[RF] {split_name.upper()} (threshold={threshold:.2f}):")
    print(f"      AUC-ROC:       {metrics['auc_roc']}")
    print(f"      F1:            {metrics['f1']}")
    print(f"      Precision:     {metrics['precision']}")
    print(f"      Recall:        {metrics['recall']}")
    print(f"      Avg Precision: {metrics['avg_precision']}")

    return metrics, y_pred, y_prob


def get_classification_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> str:
    """Pretty-printed sklearn classification report."""
    return classification_report(
        y_true, y_pred, target_names=["No Default", "Default"]
    )


def cross_validate_model(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_folds: int = 5,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """k-fold stratified CV; mean/std/min/max for acc/prec/rec/F1/AUC-ROC."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scoring_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = []

    print(f"[CV] {n_folds}-fold stratified cross-validation:")
    for metric in scoring_metrics:
        scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring=metric, n_jobs=-1
        )
        rows.append({
            "metric": metric,
            "mean": round(scores.mean(), 4),
            "std": round(scores.std(), 4),
            "min": round(scores.min(), 4),
            "max": round(scores.max(), 4),
        })
        print(f"      {metric:12s}: {scores.mean():.4f} (+/- {scores.std():.4f})")

    return pd.DataFrame(rows)


def compute_feature_importance(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
    top_n: int = 20,
) -> pd.DataFrame:
    """Gini (MDI) + permutation importance, sorted by permutation.
    Columns: feature, gini_importance, perm_importance, perm_std."""
    features = X_test.columns.tolist()

    gini = pd.DataFrame({
        "feature": features,
        "gini_importance": model.feature_importances_,
    }).sort_values("gini_importance", ascending=False).reset_index(drop=True)

    print(f"[RF] Computing permutation importance ({n_repeats} repeats)...")
    perm = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats, random_state=RANDOM_SEED,
        n_jobs=-1, scoring="roc_auc",
    )
    perm_df = pd.DataFrame({
        "feature": features,
        "perm_importance": perm.importances_mean,
        "perm_std": perm.importances_std,
    })

    importance = gini.merge(perm_df, on="feature")
    importance = importance.sort_values(
        "perm_importance", ascending=False
    ).reset_index(drop=True)

    print(f"\n[RF] Top {min(top_n, len(importance))} features (by permutation):")
    for i, row in importance.head(top_n).iterrows():
        print(
            f"      {i+1:2d}. {row['feature']:<28s}  "
            f"Gini={row['gini_importance']:.4f}  "
            f"Perm={row['perm_importance']:.4f}"
        )

    return importance


def optimize_threshold(
    y_val: pd.Series,
    y_val_prob: np.ndarray,
) -> Tuple[float, pd.DataFrame]:
    """Sweep τ ∈ [0.10, 0.90] in 0.01 steps, pick argmax val-F1.
    Returns (best_threshold, per-τ metric table)."""
    thresholds = np.arange(0.10, 0.90, 0.01)
    rows = []
    for t in thresholds:
        preds = (y_val_prob >= t).astype(int)
        rows.append({
            "threshold": round(t, 2),
            "precision": precision_score(y_val, preds, zero_division=0),
            "recall": recall_score(y_val, preds, zero_division=0),
            "f1": f1_score(y_val, preds, zero_division=0),
        })

    df = pd.DataFrame(rows)
    best_idx = df["f1"].idxmax()
    best_t = df.loc[best_idx, "threshold"]

    f1_at_50 = df.iloc[(df["threshold"] - 0.50).abs().idxmin()]["f1"]
    print(f"[RF] Optimal threshold (max F1 on val): {best_t:.2f}")
    print(f"      Val F1 at optimal:  {df.loc[best_idx, 'f1']:.4f}")
    print(f"      Val F1 at 0.50:     {f1_at_50:.4f}")

    return best_t, df


# Visualisation

def plot_roc_pr_curves(
    y_test: pd.Series,
    y_prob: np.ndarray,
    save_dir: Optional[str] = None,
) -> plt.Figure:
    """ROC + PR curves, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    axes[0].plot(fpr, tpr, lw=2.5, color=PALETTE["primary"],
                 label=f"Random Forest (AUC = {auc_val:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random classifier")
    axes[0].fill_between(fpr, tpr, alpha=0.06, color=PALETTE["primary"])
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.15)

    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    baseline = y_test.mean()
    axes[1].plot(rec, prec, lw=2.5, color=PALETTE["accent"],
                 label=f"Random Forest (AP = {ap:.4f})")
    axes[1].axhline(baseline, ls="--", color=PALETTE["neutral"], alpha=0.5,
                    label=f"No-skill baseline ({baseline:.2f})")
    axes[1].fill_between(rec, prec, alpha=0.06, color=PALETTE["accent"])
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision\u2013Recall Curve")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.15)

    plt.tight_layout()
    if save_dir:
        path = Path(save_dir) / "rf_roc_pr_curves.png"
        plt.savefig(path)
        print(f"[FIG] Saved: {path}")
    return fig


def plot_confusion_matrix(
    y_test: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
    save_dir: Optional[str] = None,
) -> plt.Figure:
    """Confusion-matrix heatmap at τ, with counts and % overlaid."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Purples",
        xticklabels=["No Default", "Default"],
        yticklabels=["No Default", "Default"],
        ax=ax, cbar_kws={"shrink": 0.8},
        annot_kws={"fontsize": 14, "fontweight": "bold"},
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(
        f"Confusion Matrix \u2014 Tuned RF (threshold = {threshold:.2f})",
        fontsize=12,
    )

    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            ax.text(
                j + 0.5, i + 0.72, f"({pct:.1f}%)",
                ha="center", va="center", fontsize=9, color="gray",
            )

    plt.tight_layout()
    if save_dir:
        path = Path(save_dir) / "rf_confusion_matrix.png"
        plt.savefig(path)
        print(f"[FIG] Saved: {path}")
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    save_dir: Optional[str] = None,
) -> plt.Figure:
    """Gini vs permutation importance — side-by-side horizontal bars."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    top_gini = importance_df.nlargest(top_n, "gini_importance")
    y_pos = range(len(top_gini))
    axes[0].barh(
        y_pos, top_gini["gini_importance"].values[::-1],
        color=PALETTE["primary"], alpha=0.85,
        edgecolor="white", linewidth=0.5,
    )
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(top_gini["feature"].values[::-1])
    axes[0].set_xlabel("Gini Importance (MDI)")
    axes[0].set_title("Gini Importance (built-in)", fontweight="bold")
    axes[0].grid(True, axis="x", alpha=0.15)

    top_perm = importance_df.nlargest(top_n, "perm_importance")
    y_pos = range(len(top_perm))
    axes[1].barh(
        y_pos, top_perm["perm_importance"].values[::-1],
        xerr=top_perm["perm_std"].values[::-1],
        color=PALETTE["accent"], alpha=0.85, capsize=3,
        edgecolor="white", linewidth=0.5,
    )
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(top_perm["feature"].values[::-1])
    axes[1].set_xlabel("Permutation Importance (\u0394AUC-ROC)")
    axes[1].set_title("Permutation Importance (model-agnostic)", fontweight="bold")
    axes[1].grid(True, axis="x", alpha=0.15)

    plt.suptitle(
        "Feature Importance \u2014 Gini vs Permutation",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if save_dir:
        path = Path(save_dir) / "rf_feature_importance.png"
        plt.savefig(path)
        print(f"[FIG] Saved: {path}")
    return fig


def plot_threshold_analysis(
    threshold_df: pd.DataFrame,
    best_threshold: float,
    save_dir: Optional[str] = None,
) -> plt.Figure:
    """P / R / F1 vs classification threshold, with optimal + default τ marked."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(threshold_df["threshold"], threshold_df["precision"],
            lw=2, label="Precision", color=PALETTE["secondary"])
    ax.plot(threshold_df["threshold"], threshold_df["recall"],
            lw=2, label="Recall", color=PALETTE["accent"])
    ax.plot(threshold_df["threshold"], threshold_df["f1"],
            lw=2.5, label="F1 Score", color=PALETTE["primary"])
    ax.axvline(best_threshold, ls="--", color=PALETTE["neutral"], alpha=0.7,
               label=f"Optimal threshold ({best_threshold:.2f})")
    ax.axvline(0.50, ls=":", color=PALETTE["neutral"], alpha=0.35,
               label="Default threshold (0.50)")

    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision, Recall, F1 vs Classification Threshold (Validation Set)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.grid(True, alpha=0.15)
    ax.set_xlim(0.10, 0.90)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    if save_dir:
        path = Path(save_dir) / "rf_threshold_analysis.png"
        plt.savefig(path)
        print(f"[FIG] Saved: {path}")
    return fig


def plot_tuning_analysis(
    searcher: RandomizedSearchCV,
    save_dir: Optional[str] = None,
) -> plt.Figure:
    """Four-panel tuning diagnostics: n_estimators × max_depth, marginals on
    max_depth and class_weight, and a train-vs-CV AUC overfitting check on
    the top-20 configs."""
    results = pd.DataFrame(searcher.cv_results_)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for depth in [5, 10, 15]:
        mask = results["param_max_depth"] == depth
        if mask.sum() == 0:
            continue
        subset = results[mask].groupby("param_n_estimators")["mean_test_score"].mean()
        axes[0, 0].plot(
            subset.index, subset.values, marker="o",
            label=f"depth={depth}", lw=1.5,
        )
    axes[0, 0].set_xlabel("n_estimators")
    axes[0, 0].set_ylabel("Mean CV AUC-ROC")
    axes[0, 0].set_title("n_estimators vs AUC-ROC (by max_depth)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.15)

    results["depth_str"] = results["param_max_depth"].apply(
        lambda d: "None" if d is None else str(int(d))
    )
    depth_scores = results.groupby("depth_str")["mean_test_score"].mean()
    axes[0, 1].bar(
        depth_scores.index, depth_scores.values,
        color=PALETTE["primary"], alpha=0.85, edgecolor="white",
    )
    axes[0, 1].set_xlabel("max_depth")
    axes[0, 1].set_ylabel("Mean CV AUC-ROC")
    axes[0, 1].set_title("Effect of max_depth")
    axes[0, 1].grid(True, alpha=0.15, axis="y")
    lo, hi = depth_scores.values.min(), depth_scores.values.max()
    margin = max((hi - lo) * 0.3, 0.002)
    axes[0, 1].set_ylim(lo - margin, hi + margin)

    results["cw_str"] = results["param_class_weight"].apply(
        lambda w: "None" if w is None else str(w)
    )
    cw_scores = results.groupby("cw_str")["mean_test_score"].mean()
    cw_colours = [
        PALETTE["secondary"], PALETTE["accent"], PALETTE["info"]
    ][:len(cw_scores)]
    axes[1, 0].bar(
        cw_scores.index, cw_scores.values,
        color=cw_colours, alpha=0.85, edgecolor="white",
    )
    axes[1, 0].set_xlabel("class_weight")
    axes[1, 0].set_ylabel("Mean CV AUC-ROC")
    axes[1, 0].set_title("Effect of class_weight")
    axes[1, 0].grid(True, alpha=0.15, axis="y")
    lo, hi = cw_scores.values.min(), cw_scores.values.max()
    margin = max((hi - lo) * 0.3, 0.002)
    axes[1, 0].set_ylim(lo - margin, hi + margin)

    top = results.nlargest(20, "mean_test_score")
    x = np.arange(len(top))
    axes[1, 1].bar(
        x - 0.15, top["mean_train_score"], width=0.3,
        label="Train AUC", color=PALETTE["info"], alpha=0.85,
    )
    axes[1, 1].bar(
        x + 0.15, top["mean_test_score"], width=0.3,
        label="CV AUC", color=PALETTE["accent"], alpha=0.85,
    )
    axes[1, 1].set_xlabel("Top 20 Configurations (ranked)")
    axes[1, 1].set_ylabel("AUC-ROC")
    axes[1, 1].set_title("Train vs CV AUC \u2014 Overfitting Check")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_xticks([])
    axes[1, 1].grid(True, alpha=0.15, axis="y")
    all_vals = np.concatenate(
        [top["mean_train_score"].values, top["mean_test_score"].values]
    )
    lo, hi = all_vals.min(), all_vals.max()
    margin = max((hi - lo) * 0.3, 0.002)
    axes[1, 1].set_ylim(lo - margin, hi + margin)

    plt.suptitle(
        "Hyperparameter Tuning Analysis", fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    if save_dir:
        path = Path(save_dir) / "rf_tuning_analysis.png"
        plt.savefig(path)
        print(f"[FIG] Saved: {path}")
    return fig


# Results export

def export_results(
    baseline_metrics: Dict,
    tuned_metrics: Dict,
    cv_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    best_params: Dict,
    best_threshold: float,
    output_dir: str = "results/baseline",
) -> None:
    """Write metrics, CV summary, importance, and config under ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([baseline_metrics, tuned_metrics]).to_csv(
        out / "rf_metrics.csv", index=False,
    )

    cv_df.to_csv(out / "rf_cross_validation.csv", index=False)
    importance_df.to_csv(out / "rf_feature_importance.csv", index=False)

    config = {
        "best_params": {str(k): str(v) for k, v in best_params.items()},
        "best_threshold": round(best_threshold, 2),
        "baseline_auc": baseline_metrics["auc_roc"],
        "tuned_auc": tuned_metrics["auc_roc"],
        "tuning_grid": {
            k: [str(v) for v in vs] for k, vs in TUNING_GRID.items()
        },
    }
    with open(out / "rf_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"[EXPORT] Saved to {out}/:")
    print(f"         rf_metrics.csv, rf_cross_validation.csv")
    print(f"         rf_feature_importance.csv, rf_config.json")


# Master pipeline

def run_rf_benchmark(
    data_path: Optional[str] = None,
    output_dir: str = "results/baseline",
    figure_dir: str = "figures/baseline",
    n_iter: int = 200,
    n_cv_folds: int = 5,
    seed: int = RANDOM_SEED,
    *,
    mode: str = "auto",
    allow_fallback: bool = True,
) -> Dict[str, Any]:
    """Full RF benchmark: preprocess → baseline → tune → eval → CV →
    importance → threshold sweep → figures + CSV/JSON artefacts."""
    set_rf_style()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(figure_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  RANDOM FOREST BENCHMARK - CREDIT CARD DEFAULT PREDICTION")
    print("=" * 65)

    print("\n-- Step 1: Data Pipeline --")
    df = load_raw_data(data_path, mode=mode, allow_fallback=allow_fallback)
    df = normalise_schema(df)
    df = clean_categoricals(df)
    df = engineer_features(df)

    train_df, val_df, test_df = split_data(df, seed=seed)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_val = val_df.drop(columns=[TARGET_COL])
    y_val = val_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    n_original = 23
    n_engineered = X_train.shape[1] - n_original
    print(f"[RF] Features: {X_train.shape[1]} ({n_engineered} engineered)")
    print(f"[RF] Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    print("\n-- Step 2: Baseline Random Forest --")
    baseline_model, baseline_time = train_baseline(X_train, y_train, seed)
    baseline_metrics, _, _ = evaluate_model(
        baseline_model, X_test, y_test, "test_baseline",
    )
    baseline_metrics["model"] = "RF_baseline"
    baseline_metrics["train_time_s"] = round(baseline_time, 2)

    print("\n-- Step 3: Hyperparameter Tuning --")
    searcher = tune_hyperparameters(
        X_train, y_train, n_iter=n_iter, n_cv_folds=n_cv_folds, seed=seed,
    )
    best_model = searcher.best_estimator_

    print("\n-- Step 4: Tuned Model Evaluation --")
    _, _, y_val_prob = evaluate_model(best_model, X_val, y_val, "validation")
    test_metrics, y_test_pred, y_test_prob = evaluate_model(
        best_model, X_test, y_test, "test",
    )
    test_metrics["model"] = "RF_tuned"

    print("\n-- Step 5: Cross-Validation --")
    cv_df = cross_validate_model(best_model, X_train, y_train, n_cv_folds, seed)

    print("\n-- Step 6: Feature Importance --")
    importance_df = compute_feature_importance(best_model, X_test, y_test)

    print("\n-- Step 7: Threshold Optimisation --")
    best_threshold, threshold_df = optimize_threshold(y_val, y_val_prob)

    test_pred_opt = (y_test_prob >= best_threshold).astype(int)
    test_f1_opt = f1_score(y_test, test_pred_opt, zero_division=0)
    print(f"      Test F1 at optimal:  {test_f1_opt:.4f}")

    print("\n-- Step 8: Generating Figures --")
    fig1 = plot_roc_pr_curves(y_test, y_test_prob, figure_dir)
    plt.close(fig1)
    fig2 = plot_confusion_matrix(y_test, y_test_prob, best_threshold, figure_dir)
    plt.close(fig2)
    fig3 = plot_feature_importance(importance_df, 20, figure_dir)
    plt.close(fig3)
    fig4 = plot_threshold_analysis(threshold_df, best_threshold, figure_dir)
    plt.close(fig4)
    fig5 = plot_tuning_analysis(searcher, figure_dir)
    plt.close(fig5)

    print("\n-- Step 9: Exporting Results --")
    export_results(
        baseline_metrics, test_metrics, cv_df, importance_df,
        searcher.best_params_, best_threshold, output_dir,
    )

    print("\n" + "=" * 65)
    print("  BENCHMARK COMPLETE")
    print("=" * 65)
    print(f"  Baseline AUC:    {baseline_metrics['auc_roc']}")
    print(f"  Tuned AUC:       {test_metrics['auc_roc']}")
    print(f"  Best threshold:  {best_threshold:.2f}")
    print(f"  Test F1 (opt):   {test_f1_opt:.4f}")
    print(f"  Figures:         {figure_dir}/rf_*.png")
    print(f"  Results:         {output_dir}/rf_*.csv / rf_*.json")

    return {
        "baseline_metrics": baseline_metrics,
        "tuned_metrics": test_metrics,
        "cv_results": cv_df,
        "importance": importance_df,
        "best_params": searcher.best_params_,
        "best_threshold": best_threshold,
        "searcher": searcher,
        "best_model": best_model,
    }


if __name__ == "__main__":
    run_rf_benchmark()
