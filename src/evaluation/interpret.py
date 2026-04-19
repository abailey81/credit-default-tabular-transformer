"""
interpret.py - attention interpretability for the trained transformer.

Reads the attention weights that train.py saves alongside its predictions
(test_attn_weights.npz, one array per layer) and produces the figures
for section 4 of the report:

    figures/evaluation/interpret/attention_rollout.png
    figures/evaluation/interpret/cls_feature_importance.png
    figures/evaluation/interpret/attention_per_head.png
    figures/evaluation/interpret/defaulter_vs_nondefaulter_attention.png
    figures/evaluation/interpret/feature_importance_comparison.png

Plus results/evaluation/interpret.json with the raw importance rankings.

If test_attn_weights.npz is missing (it is gitignored; about 30 MB per
seed), rerun train.py without --no-save-attn to regenerate it.

References: Abnar & Zuidema (2020), "Quantifying Attention Flow in
Transformers" for the rollout formulation.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..tokenization.embedding import TOKEN_ORDER

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

#: The CLS token sits at index 0; the 23 feature tokens follow in TOKEN_ORDER.
#: This list gives every sequence position a human-readable name for plots.
POSITION_LABELS: List[str] = ["[CLS]"] + list(TOKEN_ORDER)
SEQ_LEN: int = len(POSITION_LABELS)
CLS_INDEX: int = 0

#: Per-feature label order for the importance bar chart. Excludes CLS because
#: CLS attending to itself is not meaningful for feature importance.
FEATURE_LABELS: List[str] = list(TOKEN_ORDER)

DEFAULT_FIG_DPI: int = 150


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────


def load_attention(run_dir: Path) -> np.ndarray:
    """
    Load test_attn_weights.npz and stack its per-layer arrays.

    train.py writes one array per layer with keys layer_0, layer_1, etc.
    Each has shape (N, n_heads, seq_len, seq_len). We stack them so the
    leading axis is the layer index.

    Returns shape (n_layers, N, n_heads, seq_len, seq_len).
    """
    path = Path(run_dir) / "test_attn_weights.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Rerun train.py without --no-save-attn "
            "to generate attention weights for this run."
        )
    data = np.load(path)
    # Keys are layer_0, layer_1, ... so ordered sort gives the right sequence.
    layer_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
    return np.stack([data[k] for k in layer_keys], axis=0)


def load_predictions(run_dir: Path) -> Dict[str, np.ndarray]:
    """Load test_predictions.npz and return y_true, y_prob, y_pred."""
    path = Path(run_dir) / "test_predictions.npz"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found.")
    data = np.load(path)
    return {
        "y_true": data["y_true"].astype(int),
        "y_prob": data["y_prob"].astype(float),
        "y_pred": data["y_pred"].astype(int),
    }


def load_rf_gini(csv_path: Path) -> Dict[str, float]:
    """
    Read the RF feature-importance CSV and return a feature-name to Gini map.

    The CSV includes engineered features (RECENT_DELAY, AVG_UTIL_RATIO, etc.)
    that do not exist in the transformer's input. The caller can filter to
    the overlap when needed.
    """
    df = pd.read_csv(csv_path)
    return dict(zip(df["feature"], df["gini_importance"]))


# ──────────────────────────────────────────────────────────────────────────────
# Core analyses
# ──────────────────────────────────────────────────────────────────────────────


def attention_rollout(attn: np.ndarray) -> np.ndarray:
    """
    Effective attention flow from input to the top of the stack.

    The Abnar-Zuidema recipe: at each layer, average over heads, blend with
    the identity (modelling the residual stream), then chain-multiply from
    the bottom layer upwards. The result is what the top layer effectively
    attends to once residual connections are accounted for.

    Args:
        attn: shape (n_layers, N, n_heads, seq_len, seq_len).

    Returns:
        shape (N, seq_len, seq_len). Each row sums to 1.
    """
    n_layers, n_samples, _, seq_len, _ = attn.shape

    # Average over heads. The rollout formulation treats all heads equally.
    head_avg = attn.mean(axis=2)  # (n_layers, N, seq_len, seq_len)

    # Add identity to account for the residual connection. Each row still
    # sums to 1 because the original attention row sums to 1 and the identity
    # row sums to 1, and we take a 50/50 blend.
    identity = np.eye(seq_len)
    augmented = 0.5 * head_avg + 0.5 * identity  # broadcasts over (n_layers, N)

    # Chain-multiply from layer 0 upward. Each sample gets its own rollout.
    rollout = augmented[0]
    for layer_idx in range(1, n_layers):
        rollout = np.einsum("nij,njk->nik", augmented[layer_idx], rollout)
    return rollout


def cls_to_feature_scores(rollout: np.ndarray) -> Dict[str, float]:
    """
    Per-feature importance from the CLS row of the rollout matrix.

    Averages the rollout over all test samples, takes the CLS row (which
    is what the model uses for classification), drops the CLS-to-CLS entry,
    and returns a dict mapping feature name to score.
    """
    mean_rollout = rollout.mean(axis=0)  # (seq_len, seq_len)
    cls_row = mean_rollout[CLS_INDEX]  # (seq_len,)

    # Drop CLS attending to itself and re-normalise so the feature scores
    # sum to 1. This makes the bar chart comparable across runs.
    feature_scores = cls_row[1:]
    total = feature_scores.sum()
    if total > 0:
        feature_scores = feature_scores / total

    return {name: float(feature_scores[i]) for i, name in enumerate(FEATURE_LABELS)}


def attention_by_class(
    rollout: np.ndarray,
    y_true: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split the rollout by default label and average within each group.

    Returns (defaulter_mean, nondefaulter_mean), each of shape (seq_len, seq_len).
    """
    defaulter_mask = y_true == 1
    if defaulter_mask.sum() == 0 or (~defaulter_mask).sum() == 0:
        raise ValueError("Need at least one sample in each class.")

    defaulter_mean = rollout[defaulter_mask].mean(axis=0)
    nondefaulter_mean = rollout[~defaulter_mask].mean(axis=0)
    return defaulter_mean, nondefaulter_mean


def per_head_entropy(attn: np.ndarray) -> np.ndarray:
    """
    Entropy of each head's CLS attention, averaged over the test set.

    Low entropy means the head concentrates on a small number of features
    (specialised). High entropy means it spreads attention widely (diffuse).

    Args:
        attn: shape (n_layers, N, n_heads, seq_len, seq_len).

    Returns:
        shape (n_layers, n_heads).
    """
    # CLS row only. Shape becomes (n_layers, N, n_heads, seq_len).
    cls_attn = attn[:, :, :, CLS_INDEX, :]

    # Shannon entropy. The epsilon avoids log(0) where a head attends 0 to
    # a position; those positions contribute 0 to the entropy in the limit.
    eps = 1e-12
    entropy = -np.sum(cls_attn * np.log(cls_attn + eps), axis=-1)  # (L, N, H)

    return entropy.mean(axis=1)  # (L, H)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────


def _style_heatmap_axes(ax, title: str) -> None:
    """Apply the token-label tick marks used on every attention heatmap."""
    ax.set_xticks(range(SEQ_LEN))
    ax.set_yticks(range(SEQ_LEN))
    ax.set_xticklabels(POSITION_LABELS, rotation=90, fontsize=7)
    ax.set_yticklabels(POSITION_LABELS, fontsize=7)
    ax.set_title(title, fontsize=10)


def plot_rollout_heatmap(rollout: np.ndarray, out_path: Path) -> None:
    """24x24 heatmap of the average rollout across the test set."""
    mean_rollout = rollout.mean(axis=0)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mean_rollout, cmap="Blues")
    _style_heatmap_axes(ax, "Attention rollout (test-set average)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_FIG_DPI)
    plt.close(fig)


def plot_cls_feature_bars(scores: Dict[str, float], out_path: Path) -> None:
    """Horizontal bar chart of CLS attention to every feature, sorted."""
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(range(len(names)), values, color="#0072B2")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Normalised CLS attention")
    ax.set_title("Feature importance: CLS attention weight per feature")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_FIG_DPI)
    plt.close(fig)


def plot_per_head_heatmaps(attn: np.ndarray, out_path: Path) -> None:
    """Grid of per-head attention heatmaps, one row per layer."""
    n_layers, _, n_heads, _, _ = attn.shape
    # Mean over samples for each (layer, head).
    mean_attn = attn.mean(axis=1)  # (n_layers, n_heads, S, S)

    fig, axes = plt.subplots(
        n_layers, n_heads,
        figsize=(3 * n_heads, 3 * n_layers),
        squeeze=False,
    )

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            ax = axes[layer_idx, head_idx]
            ax.imshow(mean_attn[layer_idx, head_idx], cmap="Blues")
            ax.set_title(f"L{layer_idx} H{head_idx}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Per-head attention (test-set average)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_FIG_DPI)
    plt.close(fig)


def plot_class_conditional(
    defaulter: np.ndarray,
    nondefaulter: np.ndarray,
    out_path: Path,
) -> None:
    """Three panels: defaulter attention, non-defaulter attention, and the
    difference. The difference panel highlights where the two classes draw
    different conclusions from the same features."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    im0 = axes[0].imshow(nondefaulter, cmap="Blues")
    _style_heatmap_axes(axes[0], "Non-defaulters")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(defaulter, cmap="Blues")
    _style_heatmap_axes(axes[1], "Defaulters")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    diff = defaulter - nondefaulter
    # Symmetric colour range so zero is always white-ish.
    vmax = np.abs(diff).max()
    im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    _style_heatmap_axes(axes[2], "Defaulters minus non-defaulters")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_FIG_DPI)
    plt.close(fig)


def plot_vs_rf_importance(
    attn_scores: Dict[str, float],
    rf_gini: Dict[str, float],
    out_path: Path,
) -> None:
    """
    Side-by-side bar chart for features present in both models.

    RF is trained on engineered features; the transformer uses the 23 raw
    ones. Only the overlap is shown here. The two scores are on different
    scales (normalised attention vs Gini importance) so we plot them as
    separate panels rather than overlaying.
    """
    shared = [f for f in FEATURE_LABELS if f in rf_gini]
    # Sort by transformer score so the left panel is ranked.
    shared.sort(key=lambda f: attn_scores.get(f, 0.0), reverse=True)

    attn_vals = [attn_scores.get(f, 0.0) for f in shared]
    rf_vals = [rf_gini.get(f, 0.0) for f in shared]

    fig, (ax_attn, ax_rf) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    ax_attn.barh(range(len(shared)), attn_vals, color="#0072B2")
    ax_attn.set_yticks(range(len(shared)))
    ax_attn.set_yticklabels(shared, fontsize=8)
    ax_attn.invert_yaxis()
    ax_attn.set_xlabel("Transformer CLS attention")
    ax_attn.set_title("Transformer attention rollout (CLS row)")
    ax_attn.grid(axis="x", alpha=0.3)

    ax_rf.barh(range(len(shared)), rf_vals, color="#D55E00")
    ax_rf.set_xlabel("RF Gini importance")
    ax_rf.set_title("Random Forest Gini importance")
    ax_rf.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Feature importance: transformer vs RF (shared raw features only)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_FIG_DPI)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Produce attention interpretability figures and a JSON of per-feature "
            "importance rankings. Needs test_attn_weights.npz and "
            "test_predictions.npz in the run directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run-dir", type=Path,
        default=Path("results/transformer/seed_42"),
        help="Training run directory to pull attention weights + predictions from.",
    )
    p.add_argument(
        "--rf-importance", type=Path,
        default=Path("results/baseline/rf_feature_importance.csv"),
        help="RF feature-importance CSV for the side-by-side comparison.",
    )
    p.add_argument(
        "--figures-dir", type=Path, default=Path("figures/evaluation/interpret"),
        help="Directory to write the 5 PNG figures into.",
    )
    p.add_argument(
        "--output-json", type=Path,
        default=Path("results/evaluation/interpret.json"),
        help="Where to write the raw importance rankings.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args(argv)

    attn = load_attention(args.run_dir)
    preds = load_predictions(args.run_dir)

    rollout = attention_rollout(attn)
    attn_scores = cls_to_feature_scores(rollout)
    defaulter_mean, nondefaulter_mean = attention_by_class(rollout, preds["y_true"])
    entropy = per_head_entropy(attn)

    rf_gini: Dict[str, float] = {}
    if args.rf_importance.is_file():
        rf_gini = load_rf_gini(args.rf_importance)
    else:
        logger.warning("RF importance CSV not found at %s; skipping comparison plot",
                       args.rf_importance)

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    plot_rollout_heatmap(rollout, args.figures_dir / "attention_rollout.png")
    plot_cls_feature_bars(attn_scores, args.figures_dir / "cls_feature_importance.png")
    plot_per_head_heatmaps(attn, args.figures_dir / "attention_per_head.png")
    plot_class_conditional(
        defaulter_mean, nondefaulter_mean,
        args.figures_dir / "defaulter_vs_nondefaulter_attention.png",
    )
    if rf_gini:
        plot_vs_rf_importance(
            attn_scores, rf_gini,
            args.figures_dir / "feature_importance_comparison.png",
        )

    summary = {
        "run_dir": Path(args.run_dir).as_posix(),
        "attention_scores": attn_scores,
        "attention_ranked": sorted(
            attn_scores.items(), key=lambda kv: kv[1], reverse=True,
        ),
        "per_head_entropy": entropy.tolist(),
        "n_defaulters": int((preds["y_true"] == 1).sum()),
        "n_nondefaulters": int((preds["y_true"] == 0).sum()),
    }
    args.output_json.write_text(json.dumps(summary, indent=2))

    logger.info("Figures written to %s", args.figures_dir)
    logger.info("Summary at %s", args.output_json)
    print("\nTop 10 features by CLS attention:")
    for name, score in summary["attention_ranked"][:10]:
        print(f"  {name:<15} {score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
