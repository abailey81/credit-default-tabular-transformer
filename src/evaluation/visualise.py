"""§4 report figures: ROC, PR, confusion matrices, training curves, reliability.
Reads per-seed artefacts from train.py; RF appears only as a numeric annotation
(RF's own plots live in random_forest.py)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

logger = logging.getLogger(__name__)


#: pos base rate on the stratified test split → PR no-skill line
TEST_BASE_RATE: float = 0.2211

#: matches train.py's ECE bin count
RELIABILITY_N_BINS: int = 10

#: keeps the same label for the same seed across every figure
RUN_DISPLAY_NAMES: Dict[str, str] = {
    "seed_42": "Transformer (seed 42)",
    "seed_1": "Transformer (seed 1)",
    "seed_2": "Transformer (seed 2)",
    "seed_42_mtlm_finetune": "Transformer + MTLM (seed 42)",
}

#: Okabe-Ito (CVD-safe), same palette as eda.py
RUN_COLOURS: Dict[str, str] = {
    "seed_42": "#0072B2",
    "seed_1": "#D55E00",
    "seed_2": "#009E73",
    "seed_42_mtlm_finetune": "#CC79A7",
}

DEFAULT_FIG_DPI: int = 150


# I/O

def load_predictions(run_dir: Path) -> Dict[str, Any]:
    """test preds + test metrics for one run."""
    run_dir = Path(run_dir)
    preds = np.load(run_dir / "test_predictions.npz")
    metrics = json.loads((run_dir / "test_metrics.json").read_text())
    return {
        "run_name": run_dir.name,
        "y_true": preds["y_true"].astype(int),
        "y_prob": preds["y_prob"].astype(float),
        "y_pred": preds["y_pred"].astype(int),
        "auc_roc": float(metrics["metrics"]["auc_roc"]),
        "auc_pr": float(metrics["metrics"]["auc_pr"]),
    }


def load_training_log(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(Path(run_dir) / "train_log.csv")


def load_rf_reference(csv_path: Path) -> Optional[Dict[str, float]]:
    """tuned RF {auc_roc, auc_pr} or None if CSV missing."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    tuned = df[df["model"] == "RF_tuned"]
    if tuned.empty:
        return None
    row = tuned.iloc[0]
    return {"auc_roc": float(row["auc_roc"]), "auc_pr": float(row["avg_precision"])}


def _style(run_name: str) -> Tuple[str, str]:
    """(label, colour) — same mapping across every figure."""
    label = RUN_DISPLAY_NAMES.get(run_name, run_name)
    colour = RUN_COLOURS.get(run_name, "#555555")
    return label, colour


# ROC curves

def plot_roc_curves(
    runs: List[Dict[str, Any]],
    rf_ref: Optional[Dict[str, float]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    for run in runs:
        fpr, tpr, _ = roc_curve(run["y_true"], run["y_prob"])
        label, colour = _style(run["run_name"])
        ax.plot(fpr, tpr, color=colour, linewidth=1.8,
                label=f"{label} (AUC = {run['auc_roc']:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0,
            color="#999999", label="Chance")

    # RF: text annotation only (no raw probs loaded here)
    if rf_ref is not None:
        ax.text(0.60, 0.08,
                f"RF (tuned): AUC = {rf_ref['auc_roc']:.3f}",
                fontsize=10,
                bbox=dict(facecolor="white", edgecolor="#999999", boxstyle="round,pad=0.3"))

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves - transformer runs")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_FIG_DPI)
    plt.close(fig)


# Precision-Recall curves

def plot_pr_curves(
    runs: List[Dict[str, Any]],
    rf_ref: Optional[Dict[str, float]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    for run in runs:
        precision, recall, _ = precision_recall_curve(run["y_true"], run["y_prob"])
        label, colour = _style(run["run_name"])
        ax.plot(recall, precision, color=colour, linewidth=1.8,
                label=f"{label} (AP = {run['auc_pr']:.3f})")

    ax.axhline(TEST_BASE_RATE, linestyle="--", linewidth=1.0,
               color="#999999",
               label=f"No-skill ({TEST_BASE_RATE:.3f})")

    if rf_ref is not None:
        ax.text(0.40, 0.90,
                f"RF (tuned): AP = {rf_ref['auc_pr']:.3f}",
                fontsize=10,
                bbox=dict(facecolor="white", edgecolor="#999999", boxstyle="round,pad=0.3"))

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves - transformer runs")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_FIG_DPI)
    plt.close(fig)


# Confusion matrices

def plot_confusion_matrices(runs: List[Dict[str, Any]], out_path: Path) -> None:
    n = len(runs)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, run in zip(axes, runs):
        cm = confusion_matrix(run["y_true"], run["y_pred"], labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=11)

        label, _ = _style(run["run_name"])
        ax.set_title(label, fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No default", "Default"], fontsize=9)
        ax.set_yticklabels(["No default", "Default"], fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(runs):]:
        ax.set_visible(False)

    fig.suptitle("Confusion matrices at the default 0.5 threshold", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_FIG_DPI)
    plt.close(fig)


# Training curves

def plot_training_curves(
    run_dirs: List[Path],
    out_path: Path,
) -> None:
    """2-panel: train loss (L), val AUC (R)."""
    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(12, 5))

    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        log = load_training_log(run_dir)
        label, colour = _style(run_dir.name)
        ax_loss.plot(log["epoch"], log["train_loss"],
                     color=colour, linewidth=1.6, label=label)
        ax_auc.plot(log["epoch"], log["val_auc_roc"],
                    color=colour, linewidth=1.6, label=label)

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Training loss")
    ax_loss.set_title("Training loss over epochs")
    ax_loss.legend(fontsize=9)
    ax_loss.grid(alpha=0.3)

    ax_auc.set_xlabel("Epoch")
    ax_auc.set_ylabel("Validation AUC-ROC")
    ax_auc.set_title("Validation AUC-ROC over epochs")
    ax_auc.legend(fontsize=9)
    ax_auc.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_FIG_DPI)
    plt.close(fig)


# Reliability diagrams

def _reliability_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = RELIABILITY_N_BINS,
) -> Tuple[np.ndarray, np.ndarray]:
    """equal-width prob bins → (mean p, observed pos rate) per populated bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    means = []
    rates = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # last bin is right-inclusive so p=1.0 lands somewhere
        in_bin = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if not in_bin.any():
            continue
        means.append(y_prob[in_bin].mean())
        rates.append(y_true[in_bin].mean())
    return np.array(means), np.array(rates)


def plot_reliability_diagrams(runs: List[Dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0,
            color="#999999", label="Perfect calibration")

    for run in runs:
        means, rates = _reliability_bins(run["y_true"], run["y_prob"])
        label, colour = _style(run["run_name"])
        ax.plot(means, rates, marker="o", color=colour, linewidth=1.6,
                markersize=5, label=label)

    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Observed default rate (bin)")
    ax.set_title(f"Reliability diagrams ({RELIABILITY_N_BINS} equal-width bins)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_FIG_DPI)
    plt.close(fig)


# CLI

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate the five report figures from committed training artefacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--runs", nargs="+", type=Path,
        default=[
            Path("results/transformer/seed_42"),
            Path("results/transformer/seed_1"),
            Path("results/transformer/seed_2"),
            Path("results/transformer/seed_42_mtlm_finetune"),
        ],
        help="Training run directories. Must each contain test_predictions.npz, "
             "test_metrics.json, and train_log.csv.",
    )
    p.add_argument(
        "--rf", type=Path, default=Path("results/baseline/rf_metrics.csv"),
        help="RF metrics CSV (used for reference annotations on the ROC / PR plots).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("figures/evaluation/comparison"),
        help="Directory to write the PNG files into.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args(argv)

    run_dirs = [Path(d) for d in args.runs]

    runs = []
    for d in run_dirs:
        if not d.is_dir():
            logger.warning("Skipping %s - not a directory", d)
            continue
        try:
            runs.append(load_predictions(d))
        except FileNotFoundError as e:
            logger.warning("Skipping %s - %s", d, e)

    if not runs:
        logger.error("No loadable runs. Check --runs paths.")
        return 1

    rf_ref = load_rf_reference(args.rf)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        ("roc_curves_transformer.png",
         lambda p: plot_roc_curves(runs, rf_ref, p)),
        ("pr_curves_transformer.png",
         lambda p: plot_pr_curves(runs, rf_ref, p)),
        ("confusion_matrices_transformer.png",
         lambda p: plot_confusion_matrices(runs, p)),
        ("training_curves.png",
         lambda p: plot_training_curves(run_dirs, p)),
        ("reliability_diagrams.png",
         lambda p: plot_reliability_diagrams(runs, p)),
    ]

    for filename, render in targets:
        path = args.output_dir / filename
        render(path)
        logger.info("Wrote %s", path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
