"""EDA for the UCI credit-default dataset.

12 figures + one summary table (class balance, cat/num distributions, temporal
trajectories, correlations, PAY semantics). Loading goes through
:mod:`src.data.preprocessing`."""

import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import chi2_contingency, mannwhitneyu, pointbiserialr

warnings.filterwarnings("ignore")


def set_publication_style():
    """Matplotlib rc: 300 DPI, serif, top/right spines off."""
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
        "legend.fontsize": 8,
        "figure.titlesize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


COLORS = {"No Default": "#2C73D2", "Default": "#D63031"}
C_ND, C_D = COLORS["No Default"], COLORS["Default"]

CMAP_DIV = "RdBu_r"

from ..data.preprocessing import (
    PAY_STATUS_FEATURES, BILL_AMT_FEATURES, PAY_AMT_FEATURES,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COL,
    ALL_FEATURE_COLS,
)

MONTH_LABELS = ["Sep", "Aug", "Jul", "Jun", "May", "Apr"]


def plot_class_distribution(df: pd.DataFrame, save_dir: str):
    """DEFAULT counts (bar) + proportions (pie) with imbalance ratio."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    counts = df[TARGET_COL].value_counts().sort_index()
    labels = ["No Default", "Default"]
    colors = [C_ND, C_D]

    ax = axes[0]
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.5)
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{count:,}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_ylabel("Count")
    ax.set_title("(a) Class Counts")
    ax.set_ylim(0, counts.max() * 1.15)

    ax = axes[1]
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 10},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for t in autotexts:
        t.set_fontweight("bold")
    ax.set_title("(b) Class Proportions")

    imbalance = counts[0] / counts[1]
    fig.suptitle(
        f"Target Distribution - Imbalance Ratio {imbalance:.1f}:1",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(f"{save_dir}/fig01_class_distribution.png")
    plt.close(fig)
    print("[FIG 01] Class distribution")


def plot_categorical_by_target(df: pd.DataFrame, save_dir: str):
    """Default rate per SEX/EDUCATION/MARRIAGE level, with χ² annotations."""
    cat_features = {
        "SEX": {1: "Male", 2: "Female"},
        "EDUCATION": {1: "Grad School", 2: "University", 3: "High School", 4: "Others"},
        "MARRIAGE": {1: "Married", 2: "Single", 3: "Others"},
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for idx, (feat, label_map) in enumerate(cat_features.items()):
        ax = axes[idx]

        grouped = df.groupby(feat)[TARGET_COL].agg(["mean", "count"])
        grouped.index = grouped.index.map(label_map)

        contingency = pd.crosstab(df[feat], df[TARGET_COL])
        chi2, p_val, dof, expected = chi2_contingency(contingency)

        n_cats = len(grouped)
        x = np.arange(n_cats)
        default_rates = grouped["mean"]
        counts = grouped["count"]

        bars_nd = ax.bar(x, 1 - default_rates, color=C_ND, label="No Default",
                         edgecolor="white", linewidth=0.5)
        bars_d = ax.bar(x, default_rates, bottom=1 - default_rates, color=C_D,
                        label="Default", edgecolor="white", linewidth=0.5)

        for i, (rate, n) in enumerate(zip(default_rates, counts)):
            ax.text(i, 1.02, f"{rate:.1%}\n(n={n:,})", ha="center", fontsize=7, va="bottom")

        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index, rotation=15, ha="right")
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Proportion")
        p_str = f"p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
        ax.set_title(f"{feat}\n\u03c7\u00b2 = {chi2:.1f}, {p_str}")

        if idx == 0:
            ax.legend(loc="lower left", frameon=True, framealpha=0.9)

    fig.suptitle("Categorical Features: Default Rate by Category",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(f"{save_dir}/fig02_categorical_by_target.png")
    plt.close(fig)
    print("[FIG 02] Categorical features by target")


def plot_numerical_distributions(df: pd.DataFrame, save_dir: str):
    """LIMIT_BAL / AGE KDE+hist by default status, Mann-Whitney U annotated."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for idx, feat in enumerate(["LIMIT_BAL", "AGE"]):
        ax = axes[idx]
        d0 = df[df[TARGET_COL] == 0][feat]
        d1 = df[df[TARGET_COL] == 1][feat]

        U, p_val = mannwhitneyu(d0, d1, alternative="two-sided")
        n0, n1 = len(d0), len(d1)
        r_rb = 1 - (2 * U) / (n0 * n1)

        ax.hist(d0, bins=50, density=True, alpha=0.4, color=C_ND, label="No Default")
        ax.hist(d1, bins=50, density=True, alpha=0.4, color=C_D, label="Default")
        if d0.std() > 0:
            d0.plot.kde(ax=ax, color=C_ND, linewidth=2)
        if d1.std() > 0:
            d1.plot.kde(ax=ax, color=C_D, linewidth=2)

        p_str = f"p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
        ax.set_title(f"{feat}\nMann-Whitney {p_str}, r_rb = {r_rb:.3f}")
        ax.set_ylabel("Density")
        ax.legend()

        if feat == "LIMIT_BAL":
            ax.set_xlabel("Credit Limit (NT$)")
        else:
            ax.set_xlabel("Age (years)")

    fig.suptitle("Numerical Feature Distributions by Default Status",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(f"{save_dir}/fig03_numerical_distributions.png")
    plt.close(fig)
    print("[FIG 03] Numerical distributions")


def plot_pay_status_analysis(df: pd.DataFrame, save_dir: str):
    """PAY_0 dist by default + default rate vs PAY_0 + cross-month PAY heatmap.
    Motivates the hybrid PAY tokenisation (N1)."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    pay_vals = sorted(df["PAY_0"].unique())
    d0_counts = df[df[TARGET_COL] == 0]["PAY_0"].value_counts().reindex(pay_vals, fill_value=0)
    d1_counts = df[df[TARGET_COL] == 1]["PAY_0"].value_counts().reindex(pay_vals, fill_value=0)

    x = np.arange(len(pay_vals))
    w = 0.35
    ax.bar(x - w / 2, d0_counts.values, w, color=C_ND, label="No Default", edgecolor="white")
    ax.bar(x + w / 2, d1_counts.values, w, color=C_D, label="Default", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(pay_vals)
    ax.set_xlabel("PAY_0 Value")
    ax.set_ylabel("Count")
    ax.set_title("(a) PAY_0 Distribution by Default Status")
    ax.legend()

    ax.annotate("No bill", xy=(0, d0_counts.iloc[0]), xytext=(-0.5, d0_counts.iloc[0] * 1.1),
                fontsize=7, fontstyle="italic", color="gray")

    ax = fig.add_subplot(gs[0, 1])
    default_rates = df.groupby("PAY_0")[TARGET_COL].mean()
    counts = df.groupby("PAY_0").size()
    default_rates = default_rates.reindex(pay_vals)
    counts = counts.reindex(pay_vals)

    bars = ax.bar(x, default_rates.values, color=[C_D if r > 0.5 else C_ND for r in default_rates],
                  edgecolor="white", alpha=0.8)

    for i, (rate, n) in enumerate(zip(default_rates.values, counts.values)):
        ax.text(i, rate + 0.02, f"{rate:.1%}\n(n={n:,})", ha="center", fontsize=6, va="bottom")

    ax.axhline(y=df[TARGET_COL].mean(), color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(len(pay_vals) - 1, df[TARGET_COL].mean() + 0.01, "Overall rate",
            fontsize=7, color="gray", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(pay_vals)
    ax.set_xlabel("PAY_0 Value")
    ax.set_ylabel("Default Rate")
    ax.set_title("(b) Default Rate by PAY_0 - Non-linear Risk Profile")

    ax.axvspan(-0.5, 2.5, alpha=0.08, color="green")
    ax.axvspan(2.5, len(pay_vals) - 0.5, alpha=0.08, color="red")
    ax.text(1, ax.get_ylim()[1] * 0.92, "Categorical\nzone (-2,-1,0)",
            ha="center", fontsize=7, color="green", fontweight="bold")
    ax.text(len(pay_vals) - 3, ax.get_ylim()[1] * 0.92, "Ordinal\ndelinquency",
            ha="center", fontsize=7, color="red", fontweight="bold")

    ax = fig.add_subplot(gs[1, :])

    all_pay_vals = sorted(set().union(*[set(df[c].unique()) for c in PAY_STATUS_FEATURES]))

    heatmap_data = []
    row_labels = []
    for target_val, label in [(0, "No Default"), (1, "Default")]:
        subset = df[df[TARGET_COL] == target_val]
        for col, month in zip(PAY_STATUS_FEATURES, MONTH_LABELS):
            row_label = f"{label} - {month}"
            proportions = subset[col].value_counts(normalize=True).reindex(all_pay_vals, fill_value=0)
            heatmap_data.append(proportions.values)
            row_labels.append(row_label)

    heatmap_df = pd.DataFrame(heatmap_data, index=row_labels, columns=all_pay_vals)

    sns.heatmap(heatmap_df, ax=ax, cmap="YlOrRd", annot=True, fmt=".2f",
                linewidths=0.5, cbar_kws={"label": "Proportion"}, annot_kws={"size": 6})
    ax.set_xlabel("PAY Status Value")
    ax.set_title("(c) PAY Status Distribution Across Months - Defaulters vs Non-Defaulters")

    fig.suptitle("PAY Feature Semantic Analysis - Motivating Tokenisation Design",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(f"{save_dir}/fig04_pay_status_analysis.png")
    plt.close(fig)
    print("[FIG 04] PAY status semantic analysis")


def plot_temporal_trajectories(df: pd.DataFrame, save_dir: str):
    """6-month mean trajectories (PAY / BILL_AMT / PAY_AMT) with 95% CI."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    feature_groups = [
        ("Repayment Status", PAY_STATUS_FEATURES, "Mean PAY Value"),
        ("Bill Amount", BILL_AMT_FEATURES, "Mean Amount (NT$)"),
        ("Payment Amount", PAY_AMT_FEATURES, "Mean Amount (NT$)"),
    ]

    for idx, (title, cols, ylabel) in enumerate(feature_groups):
        ax = axes[idx]

        for target_val, label, color in [(0, "No Default", C_ND), (1, "Default", C_D)]:
            subset = df[df[TARGET_COL] == target_val][cols]
            means = subset.mean()
            stds = subset.std() / np.sqrt(len(subset))

            ax.plot(MONTH_LABELS, means.values, marker="o", color=color,
                    label=label, linewidth=2, markersize=5)
            ax.fill_between(MONTH_LABELS,
                            means.values - 1.96 * stds.values,
                            means.values + 1.96 * stds.values,
                            alpha=0.15, color=color)

        ax.set_xlabel("Month (2005)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"({chr(97 + idx)}) {title}")
        ax.legend()
        ax.invert_xaxis()

    fig.suptitle(
        "Temporal Trajectories: Diverging Patterns Between Defaulters and Non-Defaulters\n"
        "(Shaded regions = 95% CI of the mean)",
        fontsize=13, fontweight="bold", y=1.05,
    )
    plt.tight_layout()
    fig.savefig(f"{save_dir}/fig05_temporal_trajectories.png")
    plt.close(fig)
    print("[FIG 05] Temporal trajectories")


def plot_utilisation_analysis(df: pd.DataFrame, save_dir: str):
    """BILL_AMT/LIMIT_BAL utilisation: Sep distribution + 6-month trajectory."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    df_util = df.copy()
    for i in range(1, 7):
        df_util[f"_UTIL_{i}"] = df_util[f"BILL_AMT{i}"] / df_util["LIMIT_BAL"].replace(0, np.nan)

    ax = axes[0]
    for target_val, label, color in [(0, "No Default", C_ND), (1, "Default", C_D)]:
        subset = df_util[df_util[TARGET_COL] == target_val]["_UTIL_1"].clip(-0.5, 2)
        ax.hist(subset, bins=80, density=True, alpha=0.4, color=color, label=label)
        if subset.std() > 0:
            subset.plot.kde(ax=ax, color=color, linewidth=2)

    ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(1.02, ax.get_ylim()[1] * 0.8, "100% utilisation", fontsize=7,
            rotation=90, color="gray")
    ax.set_xlabel("Credit Utilisation Ratio (BILL_AMT1 / LIMIT_BAL)")
    ax.set_ylabel("Density")
    ax.set_title("(a) September Utilisation Distribution")
    ax.legend()

    ax = axes[1]
    util_cols = [f"_UTIL_{i}" for i in range(1, 7)]
    for target_val, label, color in [(0, "No Default", C_ND), (1, "Default", C_D)]:
        subset = df_util[df_util[TARGET_COL] == target_val][util_cols]
        means = subset.mean()
        ax.plot(MONTH_LABELS, means.values, marker="o", color=color,
                label=label, linewidth=2, markersize=5)

    ax.set_xlabel("Month (2005)")
    ax.set_ylabel("Mean Utilisation Ratio")
    ax.set_title("(b) Utilisation Trajectory")
    ax.legend()
    ax.invert_xaxis()

    fig.suptitle("Credit Utilisation Analysis - A Key Risk Indicator",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(f"{save_dir}/fig06_utilisation_analysis.png")
    plt.close(fig)
    print("[FIG 06] Utilisation analysis")


def plot_correlation_heatmap(df: pd.DataFrame, save_dir: str):
    """Lower-triangle Pearson r over the 23 features + DEFAULT."""
    numeric_cols = [c for c in ALL_FEATURE_COLS if c in df.select_dtypes(include=[np.number]).columns]
    corr = df[numeric_cols + [TARGET_COL]].corr()

    fig, ax = plt.subplots(figsize=(14, 12))

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap=CMAP_DIV, center=0, ax=ax,
                square=True, linewidths=0.3,
                cbar_kws={"shrink": 0.8, "label": "Pearson r"},
                annot=True, fmt=".2f", annot_kws={"size": 5},
                vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Matrix (Lower Triangle)\n"
                 "Strong BILL_AMT autocorrelation; PAY features most correlated with target",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{save_dir}/fig07_correlation_heatmap.png")
    plt.close(fig)
    print("[FIG 07] Correlation heatmap")


def plot_feature_target_association(df: pd.DataFrame, save_dir: str):
    """Feature → DEFAULT ranking. |r_pb| for numerical + PAY, Cramér's V for cat."""
    def cramers_v(x, y):
        confusion = pd.crosstab(x, y)
        chi2, _, _, _ = chi2_contingency(confusion)
        n = confusion.sum().sum()
        min_dim = min(confusion.shape[0], confusion.shape[1]) - 1
        return np.sqrt(chi2 / (n * max(min_dim, 1)))

    associations = {}

    for col in NUMERICAL_FEATURES:
        r, p = pointbiserialr(df[TARGET_COL], df[col])
        associations[col] = {"strength": abs(r), "metric": "|r_pb|", "p_value": p}

    for col in CATEGORICAL_FEATURES:
        v = cramers_v(df[col], df[TARGET_COL])
        associations[col] = {"strength": v, "metric": "Cramer's V", "p_value": np.nan}

    for col in PAY_STATUS_FEATURES:
        r, p = pointbiserialr(df[TARGET_COL], df[col])
        associations[col] = {"strength": abs(r), "metric": "|r_pb|", "p_value": p}

    assoc_df = pd.DataFrame(associations).T.sort_values("strength", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 10))

    colors = []
    for feat in assoc_df.index:
        if feat in PAY_STATUS_FEATURES:
            colors.append("#E17055")
        elif feat in CATEGORICAL_FEATURES:
            colors.append("#00B894")
        elif feat in BILL_AMT_FEATURES:
            colors.append("#6C5CE7")
        elif feat in PAY_AMT_FEATURES:
            colors.append("#FDCB6E")
        else:
            colors.append("#74B9FF")

    ax.barh(range(len(assoc_df)), assoc_df["strength"], color=colors, edgecolor="white")
    ax.set_yticks(range(len(assoc_df)))
    ax.set_yticklabels(assoc_df.index)
    ax.set_xlabel("Association Strength")
    ax.set_title("Feature-Target Association Strength\n"
                 "(Point-biserial |r| for numerical, Cramer's V for categorical)",
                 fontweight="bold")

    legend_elements = [
        Patch(facecolor="#E17055", label="PAY Status"),
        Patch(facecolor="#74B9FF", label="Demographic (num.)"),
        Patch(facecolor="#00B894", label="Demographic (cat.)"),
        Patch(facecolor="#6C5CE7", label="Bill Amount"),
        Patch(facecolor="#FDCB6E", label="Payment Amount"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True)

    plt.tight_layout()
    fig.savefig(f"{save_dir}/fig08_feature_target_association.png")
    plt.close(fig)
    print("[FIG 08] Feature-target association")


def plot_bill_amt_autocorrelation(df: pd.DataFrame, save_dir: str):
    """BILL_AMT cross-month corr heatmap + autocorr-vs-lag by default status.
    Motivates self-attention across time steps."""
    bill_corr = df[BILL_AMT_FEATURES].corr()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    sns.heatmap(bill_corr, ax=ax, cmap="YlOrRd", annot=True, fmt=".3f",
                square=True, linewidths=0.5, vmin=0.5, vmax=1,
                xticklabels=MONTH_LABELS, yticklabels=MONTH_LABELS)
    ax.set_title("(a) BILL_AMT Cross-Month Correlation")

    ax = axes[1]
    lags = []
    corrs_nd = []
    corrs_d = []
    for lag in range(1, 6):
        pairs = []
        for i in range(6 - lag):
            r_nd = df[df[TARGET_COL] == 0][BILL_AMT_FEATURES[i]].corr(
                df[df[TARGET_COL] == 0][BILL_AMT_FEATURES[i + lag]])
            r_d = df[df[TARGET_COL] == 1][BILL_AMT_FEATURES[i]].corr(
                df[df[TARGET_COL] == 1][BILL_AMT_FEATURES[i + lag]])
            pairs.append((r_nd, r_d))
        lags.append(lag)
        corrs_nd.append(np.mean([p[0] for p in pairs]))
        corrs_d.append(np.mean([p[1] for p in pairs]))

    ax.plot(lags, corrs_nd, marker="o", color=C_ND, label="No Default", linewidth=2)
    ax.plot(lags, corrs_d, marker="o", color=C_D, label="Default", linewidth=2)
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("Mean Pearson Correlation")
    ax.set_title("(b) Autocorrelation Decay by Default Status")
    ax.legend()
    ax.set_ylim(0.5, 1.0)

    fig.suptitle("Bill Amount Temporal Structure - Motivating Self-Attention over Time Steps",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(f"{save_dir}/fig09_bill_amt_autocorrelation.png")
    plt.close(fig)
    print("[FIG 09] BILL_AMT autocorrelation")


def plot_feature_interactions(df: pd.DataFrame, save_dir: str):
    """LIMIT_BAL × BILL_AMT1 scatter, LIMIT_BAL-by-PAY_0 box, default rate
    across Age × Education."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    sample = df.sample(n=5000, random_state=42)
    ax.scatter(sample[sample[TARGET_COL] == 0]["LIMIT_BAL"],
               sample[sample[TARGET_COL] == 0]["BILL_AMT1"],
               alpha=0.15, s=5, color=C_ND, label="No Default")
    ax.scatter(sample[sample[TARGET_COL] == 1]["LIMIT_BAL"],
               sample[sample[TARGET_COL] == 1]["BILL_AMT1"],
               alpha=0.3, s=5, color=C_D, label="Default")
    ax.plot([0, 1e6], [0, 1e6], "k--", alpha=0.3, linewidth=1)
    ax.text(600000, 650000, "100% utilisation", fontsize=7, color="gray", rotation=35)
    ax.set_xlabel("Credit Limit (NT$)")
    ax.set_ylabel("September Bill Amount (NT$)")
    ax.set_title("(a) Credit Utilisation Pattern")
    ax.legend(markerscale=3)

    ax = axes[1]
    pay0_vals = sorted(df["PAY_0"].unique())
    data_boxes = [df[df["PAY_0"] == v]["LIMIT_BAL"] for v in pay0_vals]
    bp = ax.boxplot(data_boxes, positions=range(len(pay0_vals)), widths=0.6,
                    patch_artist=True, showfliers=False)
    for i, patch in enumerate(bp["boxes"]):
        default_rate = df[df["PAY_0"] == pay0_vals[i]][TARGET_COL].mean()
        patch.set_facecolor(plt.cm.RdYlBu_r(default_rate))
        patch.set_alpha(0.7)
    ax.set_xticks(range(len(pay0_vals)))
    ax.set_xticklabels(pay0_vals)
    ax.set_xlabel("PAY_0 Status")
    ax.set_ylabel("Credit Limit (NT$)")
    ax.set_title("(b) Credit Limit by Payment Status\n(colour = default rate)")

    ax = axes[2]
    edu_labels = {1: "Grad School", 2: "University", 3: "High School", 4: "Others"}
    for edu_val, edu_label in edu_labels.items():
        subset = df[df["EDUCATION"] == edu_val]
        age_bins = pd.cut(subset["AGE"], bins=np.arange(20, 80, 5))
        default_by_age = subset.groupby(age_bins, observed=True)[TARGET_COL].mean()
        midpoints = [interval.mid for interval in default_by_age.index]
        ax.plot(midpoints, default_by_age.values, marker="o", label=edu_label,
                linewidth=1.5, markersize=4)

    ax.set_xlabel("Age")
    ax.set_ylabel("Default Rate")
    ax.set_title("(c) Default Rate by Age x Education")
    ax.legend(fontsize=7)

    fig.suptitle("Feature Interactions - Pairwise Patterns Self-Attention Can Capture",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(f"{save_dir}/fig10_feature_interactions.png")
    plt.close(fig)
    print("[FIG 10] Feature interactions")


def plot_pay_transitions(df: pd.DataFrame, save_dir: str):
    """Aug → Sep PAY transition heatmaps, one per default class."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for target_val, label, ax in [(0, "No Default", axes[0]), (1, "Default", axes[1])]:
        subset = df[df[TARGET_COL] == target_val]

        transitions = pd.crosstab(
            subset["PAY_2"].clip(-2, 4),
            subset["PAY_0"].clip(-2, 4),
            normalize="index",
        )

        sns.heatmap(transitions, ax=ax, cmap="YlOrRd", annot=True, fmt=".2f",
                    linewidths=0.5, annot_kws={"size": 7}, vmin=0, vmax=0.8)
        ax.set_xlabel("PAY_0 (September)")
        ax.set_ylabel("PAY_2 (August)")
        ax.set_title(f"{label}: Aug -> Sep Transition Probabilities")

    fig.suptitle("Payment Status Transitions - Sequential Dependencies\n"
                 "Defaulters show higher transition probabilities toward delinquency",
                 fontsize=12, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(f"{save_dir}/fig11_pay_transitions.png")
    plt.close(fig)
    print("[FIG 11] PAY transitions")


def generate_summary_statistics(df: pd.DataFrame, save_dir: str):
    """Per-feature mean/std/median/skew/kurt split by default. CSV + LaTeX."""
    stats_list = []
    for col in ALL_FEATURE_COLS:
        row = {
            "Feature": col,
            "Mean (All)": df[col].mean(),
            "Std (All)": df[col].std(),
            "Mean (Default=0)": df[df[TARGET_COL] == 0][col].mean(),
            "Mean (Default=1)": df[df[TARGET_COL] == 1][col].mean(),
            "Median": df[col].median(),
            "Skewness": df[col].skew(),
            "Kurtosis": df[col].kurtosis(),
        }
        stats_list.append(row)

    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(f"{save_dir}/summary_statistics.csv", index=False, float_format="%.3f")

    latex = stats_df.to_latex(index=False, float_format="%.2f", escape=True)
    with open(f"{save_dir}/summary_statistics.tex", "w") as f:
        f.write(latex)

    print("[TABLE] Summary statistics saved")
    return stats_df


def plot_repayment_ratio(df: pd.DataFrame, save_dir: str):
    """Repayment ratio (PAY_AMT / |BILL_AMT|): Sep dist + 6-month medians."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    df_repay = df.copy()
    repay_cols = []
    for i in range(1, 7):
        col = f"_REPAY_{i}"
        denom = df_repay[f"BILL_AMT{i}"].replace(0, np.nan).abs()
        df_repay[col] = (df_repay[f"PAY_AMT{i}"] / denom).clip(0, 3)
        repay_cols.append(col)

    ax = axes[0]
    for target_val, label, color in [(0, "No Default", C_ND), (1, "Default", C_D)]:
        vals = df_repay[df_repay[TARGET_COL] == target_val]["_REPAY_1"].dropna().clip(0, 2)
        ax.hist(vals, bins=60, density=True, alpha=0.4, color=color, label=label)
    ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1)
    ax.text(1.02, ax.get_ylim()[1] * 0.8, "Paid in full", fontsize=7, color="gray", rotation=90)
    ax.set_xlabel("Repayment Ratio (PAY_AMT1 / |BILL_AMT1|)")
    ax.set_ylabel("Density")
    ax.set_title("(a) September Repayment Ratio")
    ax.legend()

    ax = axes[1]
    for target_val, label, color in [(0, "No Default", C_ND), (1, "Default", C_D)]:
        subset = df_repay[df_repay[TARGET_COL] == target_val][repay_cols]
        medians = subset.median()
        ax.plot(MONTH_LABELS, medians.values, marker="o", color=color,
                label=label, linewidth=2, markersize=5)
    ax.set_xlabel("Month (2005)")
    ax.set_ylabel("Median Repayment Ratio")
    ax.set_title("(b) Repayment Ratio Trajectory")
    ax.legend()
    ax.invert_xaxis()

    fig.suptitle("Repayment Ratio Analysis - Defaulters Consistently Underpay",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(f"{save_dir}/fig13_repayment_ratio.png")
    plt.close(fig)
    print("[FIG 13] Repayment ratio analysis")


# Master EDA runner

def run_eda(
    data_path: Optional[str] = None,
    save_dir: str = "figures",
    *,
    mode: str = "auto",
    allow_fallback: bool = True,
):
    """Run the full EDA: 12 figures + summary stats table.
    ``mode`` / ``allow_fallback`` forward to :mod:`src.data.sources`."""
    os.makedirs(save_dir, exist_ok=True)
    set_publication_style()

    from ..data.preprocessing import load_raw_data, normalise_schema, clean_categoricals

    df = load_raw_data(data_path, mode=mode, allow_fallback=allow_fallback)
    df = normalise_schema(df)
    df = clean_categoricals(df, verbose=False)

    print(f"\n{'='*60}")
    print("EXPLORATORY DATA ANALYSIS")
    print(f"{'='*60}")
    print(f"Dataset: {len(df):,} rows x {len(df.columns)} columns")
    print(f"Default rate: {df[TARGET_COL].mean():.4f} ({df[TARGET_COL].sum():,} / {len(df):,})")
    print(f"{'='*60}\n")

    plot_class_distribution(df, save_dir)
    plot_categorical_by_target(df, save_dir)
    plot_numerical_distributions(df, save_dir)
    plot_pay_status_analysis(df, save_dir)
    plot_temporal_trajectories(df, save_dir)
    plot_utilisation_analysis(df, save_dir)
    plot_correlation_heatmap(df, save_dir)
    plot_feature_target_association(df, save_dir)
    plot_bill_amt_autocorrelation(df, save_dir)
    plot_feature_interactions(df, save_dir)
    plot_pay_transitions(df, save_dir)
    plot_repayment_ratio(df, save_dir)
    stats_df = generate_summary_statistics(df, save_dir)

    print(f"\n{'='*60}")
    print(f"EDA COMPLETE -- 12 figures + 1 table saved to {save_dir}/")
    print(f"{'='*60}")

    return df, stats_df


if __name__ == "__main__":
    run_eda(None)
