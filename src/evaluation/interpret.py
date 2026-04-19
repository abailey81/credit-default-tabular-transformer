"""Attention interpretability for the trained transformer (§4 of report).

Consumes the per-layer attention matrices that ``train.py`` saves to
``test_attn_weights.npz`` (one array per layer; the file is ~30 MB per
seed and is gitignored -- regenerate with ``python -m src.training.train
... --save-attn``) and produces the five interpretation figures the
report references:

* ``attention_rollout.png``                      -- overall token-to-token flow (Abnar-Zuidema rollout).
* ``cls_feature_importance.png``                 -- CLS row of the rollout as a ranked bar chart.
* ``attention_per_head.png``                     -- grid of per-head heatmaps (diagnostics, not in report body).
* ``defaulter_vs_nondefaulter_attention.png``    -- class-conditional rollout + difference.
* ``feature_importance_comparison.png``          -- transformer attention vs RF Gini for shared features.

Plus a machine-readable JSON of the importance rankings for cross-checks.

Attention rollout (Abnar & Zuidema 2020) is the standard way to collapse
a stack of attention matrices into a single input-to-output flow. At each
layer: average over heads, mix with the identity to model the residual
connection (50 / 50), then chain-multiply layer-by-layer upwards. The
result at the top is "what the classifier effectively attends to" once
residuals are accounted for.

Reference: S. Abnar, W. Zuidema, *Quantifying Attention Flow in
Transformers*, ACL 2020.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..tokenization.embedding import TOKEN_ORDER

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: CLS sits at sequence position 0; the 23 feature tokens follow in
#: TOKEN_ORDER. Tick labels on every heatmap use this list so axis
#: positions stay human-readable.
POSITION_LABELS: List[str] = ["[CLS]"] + list(TOKEN_ORDER)
SEQ_LEN: int = len(POSITION_LABELS)
CLS_INDEX: int = 0

#: Per-feature ordering for the importance bar chart. CLS itself is
#: excluded because "CLS attending to CLS" is not a meaningful feature
#: importance -- it's a self-loop.
FEATURE_LABELS: List[str] = list(TOKEN_ORDER)

DEFAULT_FIG_DPI: int = 150


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_attention(run_dir: Path) -> np.ndarray:
    """Stack the per-layer attention arrays from one run into a single tensor.

    ``train.py`` writes one ``layer_<i>`` array into the npz; we reorder
    by integer index rather than relying on dict iteration order (npz
    key order is not guaranteed stable across numpy versions).

    Returns
    -------
    array, shape (n_layers, N, n_heads, seq_len, seq_len)
    """
    path = Path(run_dir) / "test_attn_weights.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Rerun train.py without --no-save-attn "
            "to generate attention weights for this run."
        )
    data = np.load(path)
    # Keys are layer_0, layer_1, ...; parse the trailing int so the
    # stack order matches the actual layer order.
    layer_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
    return np.stack([data[k] for k in layer_keys], axis=0)


def load_predictions(run_dir: Path) -> Dict[str, np.ndarray]:
    """Load the per-row predictions npz for class-conditional splits."""
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
    """Read the RF feature-importance CSV into a {feature: gini} map.

    The RF baseline trains on engineered features (RECENT_DELAY,
    AVG_UTIL_RATIO, etc.) that don't exist as transformer input tokens.
    The side-by-side plot filters to the overlap; this loader returns
    everything and lets the caller decide.
    """
    df = pd.read_csv(csv_path)
    return dict(zip(df["feature"], df["gini_importance"]))


# ---------------------------------------------------------------------------
# Core analyses
# ---------------------------------------------------------------------------


def attention_rollout(attn: np.ndarray) -> np.ndarray:
    """Abnar-Zuidema attention rollout across the layer stack.

    Pseudo-code recipe::

        for layer l in 0..L-1:
            A_l = mean over heads of attn[l]      # (N, S, S)
            A_l = 0.5 * A_l + 0.5 * I             # mix with identity (residual)
        R = A_0
        for l in 1..L-1:
            R = A_l @ R                           # chain-multiply upwards
        return R

    The 50 / 50 identity mix models the residual stream: at each layer
    the signal is a blend of attention output and the identity pass-through.
    Other papers use other mixing coefficients; 50 / 50 is the Abnar
    default and what the report quotes.

    Invariant: every row of the returned matrix must sum to 1. If it
    doesn't, something upstream (a non-softmax attention, a normalisation
    bug) has broken the rollout interpretation.

    Parameters
    ----------
    attn : array, shape (n_layers, N, n_heads, seq_len, seq_len)

    Returns
    -------
    rollout : array, shape (N, seq_len, seq_len)
    """
    n_layers, n_samples, _, seq_len, _ = attn.shape

    # Average over heads -- rollout treats the heads as interchangeable.
    # Per-head heatmaps are produced separately by plot_per_head_heatmaps.
    head_avg = attn.mean(axis=2)  # (n_layers, N, seq_len, seq_len)

    # Residual-aware mix: softmax rows sum to 1 and the identity rows
    # sum to 1, so 0.5 * A + 0.5 * I also sums row-wise to 1 -- preserves
    # the "row is a probability distribution" invariant.
    identity = np.eye(seq_len)
    augmented = 0.5 * head_avg + 0.5 * identity  # broadcasts over (n_layers, N)

    # Chain-multiply bottom-up. Each sample gets its own rollout because
    # attention is input-dependent; einsum keeps the batch dimension
    # explicit instead of requiring per-sample loops.
    rollout = augmented[0]
    for layer_idx in range(1, n_layers):
        rollout = np.einsum("nij,njk->nik", augmented[layer_idx], rollout)
    return rollout


def cls_to_feature_scores(rollout: np.ndarray) -> Dict[str, float]:
    """Collapse the test-set-average rollout into per-feature importances.

    Averages across rows (one test sample each), pulls the CLS row of
    the mean matrix (that's what the classifier reads), drops the CLS
    self-attention cell, and re-normalises the remainder to sum to 1.
    The re-normalisation is what makes the scores comparable across
    runs; without it, a run where CLS attends 40 % to itself would
    have half the "feature attention mass" of one where it attends 80 %
    to itself, even if the relative feature ranking is identical.
    """
    mean_rollout = rollout.mean(axis=0)  # (seq_len, seq_len)
    cls_row = mean_rollout[CLS_INDEX]  # (seq_len,)

    # Drop CLS->CLS; re-normalise to sum to 1 for cross-run comparability.
    feature_scores = cls_row[1:]
    total = feature_scores.sum()
    if total > 0:
        feature_scores = feature_scores / total

    return {name: float(feature_scores[i]) for i, name in enumerate(FEATURE_LABELS)}


def attention_by_class(
    rollout: np.ndarray,
    y_true: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split the rollout by default label and average within each group.

    Returned in (defaulter, non-defaulter) order to match the downstream
    plot's "positive case on the left" convention. The difference plot
    uses ``defaulter - nondefaulter`` so positive cells highlight where
    the model leans more heavily for the positive class.
    """
    defaulter_mask = y_true == 1
    if defaulter_mask.sum() == 0 or (~defaulter_mask).sum() == 0:
        raise ValueError("Need at least one sample in each class.")

    defaulter_mean = rollout[defaulter_mask].mean(axis=0)
    nondefaulter_mean = rollout[~defaulter_mask].mean(axis=0)
    return defaulter_mean, nondefaulter_mean


def per_head_entropy(attn: np.ndarray) -> np.ndarray:
    """Shannon entropy of each head's CLS-row attention distribution.

    Low entropy => the head concentrates on one or two features
    (specialised; easy to interpret).
    High entropy => the head spreads attention uniformly (diffuse;
    classical "background" head).

    Averaged over the test set per (layer, head). Entropy is in nats
    because that's what everyone else in the report uses; switch the
    log base for bits if preferred.

    Returns
    -------
    array, shape (n_layers, n_heads)
    """
    # Keep only the CLS row: the classifier only reads that one, so the
    # other rows' entropies are interpretability "noise".
    cls_attn = attn[:, :, :, CLS_INDEX, :]

    # H = - sum p log p. The epsilon prevents log(0) in the theoretical
    # limit; a head that attends 0 mass to a token contributes 0 to the
    # entropy anyway (0 log 0 = 0).
    eps = 1e-12
    entropy = -np.sum(cls_attn * np.log(cls_attn + eps), axis=-1)  # (L, N, H)

    return entropy.mean(axis=1)  # (L, H)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _style_heatmap_axes(ax, title: str) -> None:
    """Tick-mark & title helper shared by every 24x24 attention heatmap."""
    ax.set_xticks(range(SEQ_LEN))
    ax.set_yticks(range(SEQ_LEN))
    ax.set_xticklabels(POSITION_LABELS, rotation=90, fontsize=7)
    ax.set_yticklabels(POSITION_LABELS, fontsize=7)
    ax.set_title(title, fontsize=10)


def plot_rollout_heatmap(rollout: np.ndarray, out_path: Path) -> None:
    """24 x 24 heatmap of the test-set-average rollout matrix."""
    mean_rollout = rollout.mean(axis=0)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mean_rollout, cmap="Blues")
    _style_heatmap_axes(ax, "Attention rollout (test-set average)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_FIG_DPI)
    plt.close(fig)


def plot_cls_feature_bars(scores: Dict[str, float], out_path: Path) -> None:
    """Horizontal ranked bar chart of CLS -> feature attention."""
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
    """(layers x heads) grid of mean attention heatmaps.

    Diagnostic figure -- the report doesn't feature this one directly,
    but it's invaluable for debugging ("why is head 3 picking up the
    noise?") and for the appendix.
    """
    n_layers, _, n_heads, _, _ = attn.shape
    # Mean over test samples for each (layer, head).
    mean_attn = attn.mean(axis=1)  # (n_layers, n_heads, S, S)

    fig, axes = plt.subplots(
        n_layers,
        n_heads,
        figsize=(3 * n_heads, 3 * n_layers),
        squeeze=False,
    )

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            ax = axes[layer_idx, head_idx]
            ax.imshow(mean_attn[layer_idx, head_idx], cmap="Blues")
            ax.set_title(f"L{layer_idx} H{head_idx}", fontsize=9)
            # Axis labels hidden on this grid -- there are too many to
            # read at typical publication sizes. Full labels live on
            # the rollout heatmap.
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
    """Three-panel class-conditional attention: ND | D | (D - ND).

    The difference panel is diverging-colour (RdBu_r) with a symmetric
    range so zero always reads as white-ish; positive cells = defaulter
    draws more attention to that token pair, negative = non-defaulter.
    This is the panel the report reads for the "what does the model
    look at for defaulters?" question.
    """
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    im0 = axes[0].imshow(nondefaulter, cmap="Blues")
    _style_heatmap_axes(axes[0], "Non-defaulters")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(defaulter, cmap="Blues")
    _style_heatmap_axes(axes[1], "Defaulters")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    diff = defaulter - nondefaulter
    # Symmetric colour range -> zero is always the neutral colour, so
    # the eye catches sign direction immediately.
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
    """Side-by-side bar chart: transformer attention vs RF Gini.

    Only features present in both models are plotted; the RF is trained
    on engineered features (RECENT_DELAY etc.) that the transformer
    doesn't see, and showing them in an empty transformer bar would be
    misleading. The two metrics live on different scales (normalised
    attention vs Gini importance), so they're plotted in separate panels
    rather than overlaid.
    """
    shared = [f for f in FEATURE_LABELS if f in rf_gini]
    # Sort by transformer score so the left panel reads as a ranking;
    # the right panel then shows whether RF agrees with that ranking.
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
        "--run-dir",
        type=Path,
        default=Path("results/transformer/seed_42"),
        help="Training run directory to pull attention weights + predictions from.",
    )
    p.add_argument(
        "--rf-importance",
        type=Path,
        default=Path("results/baseline/rf_feature_importance.csv"),
        help="RF feature-importance CSV for the side-by-side comparison.",
    )
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures/evaluation/interpret"),
        help="Directory to write the 5 PNG figures into.",
    )
    p.add_argument(
        "--output-json",
        type=Path,
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
        logger.warning(
            "RF importance CSV not found at %s; skipping comparison plot", args.rf_importance
        )

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    plot_rollout_heatmap(rollout, args.figures_dir / "attention_rollout.png")
    plot_cls_feature_bars(attn_scores, args.figures_dir / "cls_feature_importance.png")
    plot_per_head_heatmaps(attn, args.figures_dir / "attention_per_head.png")
    plot_class_conditional(
        defaulter_mean,
        nondefaulter_mean,
        args.figures_dir / "defaulter_vs_nondefaulter_attention.png",
    )
    if rf_gini:
        plot_vs_rf_importance(
            attn_scores,
            rf_gini,
            args.figures_dir / "feature_importance_comparison.png",
        )

    summary = {
        "run_dir": Path(args.run_dir).as_posix(),
        "attention_scores": attn_scores,
        "attention_ranked": sorted(
            attn_scores.items(),
            key=lambda kv: kv[1],
            reverse=True,
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
