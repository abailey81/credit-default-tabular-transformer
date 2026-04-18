"""
uncertainty.py — Phase 11B: Monte-Carlo-Dropout predictive uncertainty (N11).

Motivation
----------
Credit decisions are regulated — a production model that lends (or denies)
based on an over-confident but miscalibrated probability is an operational
risk. Point estimates hide that risk; a well-designed system surfaces
"predict with confidence" vs "abstain / escalate for manual review".

Monte-Carlo dropout (Gal & Ghahramani, 2016) approximates posterior
predictive uncertainty from a trained dropout network by *keeping dropout
active at inference*, sampling the output distribution, and reporting
moments of that distribution. This is the cheapest practical Bayesian-
deep-learning approximation: no architecture change, no re-training, just
T forward passes.

Pipeline
--------
1. Load a trained :class:`TabularTransformer` checkpoint.
2. Flip every ``nn.Dropout`` module to ``train`` mode (and only those —
   LayerNorm's γ/β are already deterministic, so leaving them eval-mode
   keeps the statistics stable).
3. For each test row, run ``T`` forward passes; stack predictions into a
   ``(T, N)`` tensor of probabilities.
4. Compute per-row ``mean``, ``std``, predictive entropy, and a
   mutual-information proxy (total − aleatoric). Report the
   **refuse-to-predict** curve: as we defer the top-k uncertain rows,
   how does accuracy on the retained subset evolve? A well-calibrated
   uncertainty signal is monotone on this curve.

Why MC-dropout here
-------------------
Deep ensembles are a stronger uncertainty signal but require training N
independent networks; we already have 3 seeds for the ensemble row in
the comparison table. MC-dropout is a complementary, single-seed signal
that isolates *aleatoric* uncertainty from the *epistemic* contribution
a deep ensemble captures. For the report we can compare both.

Artefacts
---------
* ``results/uncertainty/mc_dropout.npz``      — per-row arrays: mean, std,
  predictive_entropy, mutual_info, y_true.
* ``results/uncertainty/uncertainty_summary.json`` — headline scalars.
* ``figures/uncertainty_refuse_curve.png``    — accuracy / AUC as a
  function of abstention fraction.
* ``figures/uncertainty_entropy_hist.png``    — histogram of predictive
  entropy split by correct / incorrect prediction.

References: Plan §13.6 / N11, Gal & Ghahramani (2016) "Dropout as a
Bayesian approximation".
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from dataset import CreditDefaultDataset, make_loader  # noqa: E402
from model import TabularTransformer  # noqa: E402
from tokenizer import build_categorical_vocab  # noqa: E402
from utils import load_checkpoint  # noqa: E402

logger = logging.getLogger(__name__)

EPS = 1e-12


# ──────────────────────────────────────────────────────────────────────────────
# MC-dropout machinery
# ──────────────────────────────────────────────────────────────────────────────


def enable_dropout(model: nn.Module) -> int:
    """Flip every ``nn.Dropout`` / ``nn.Dropout2d`` submodule to training
    mode, leaving every other module (LayerNorm etc.) in eval mode.

    Returns the number of dropout modules switched — ≥1 is the precondition
    for MC-dropout to produce non-degenerate samples.
    """
    count = 0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.train(True)
            count += 1
    return count


@torch.no_grad()
def mc_dropout_predict(
    model: TabularTransformer,
    loader,
    n_samples: int = 50,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Run ``n_samples`` stochastic forward passes with dropout active.

    Returns a dict with keys:

    * ``probs``    — ``(T, N)`` per-sample probabilities.
    * ``y_true``   — ``(N,)`` labels.
    """
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2 for MC-dropout, got {n_samples}")
    if device is None:
        device = next(model.parameters()).device

    # Capture eval-mode shape for everything, then re-enable dropout only.
    model.eval()
    n_dropout = enable_dropout(model)
    if n_dropout == 0:
        raise RuntimeError(
            "No nn.Dropout modules found — this model has no MC-dropout "
            "signal. Enable dropout>0 at construction and re-train."
        )

    torch.manual_seed(seed)

    all_probs: List[np.ndarray] = []
    all_labels: Optional[np.ndarray] = None

    for t in range(n_samples):
        # Re-seed per-sample so reruns are reproducible.
        torch.manual_seed(seed + t)
        probs_chunks: List[np.ndarray] = []
        labels_chunks: List[np.ndarray] = []
        for batch in loader:
            moved: Dict[str, Any] = {}
            for k, v in batch.items():
                if isinstance(v, dict):
                    moved[k] = {kk: vv.to(device, non_blocking=True) for kk, vv in v.items()}
                elif isinstance(v, torch.Tensor):
                    moved[k] = v.to(device, non_blocking=True)
                else:
                    moved[k] = v
            out = model(moved, return_attn=False)
            p = torch.sigmoid(out["logit"]).detach().cpu().numpy()
            probs_chunks.append(p)
            labels_chunks.append(batch["label"].detach().cpu().numpy())

        all_probs.append(np.concatenate(probs_chunks, axis=0))
        if all_labels is None:
            all_labels = np.concatenate(labels_chunks, axis=0)

    probs = np.stack(all_probs, axis=0)  # (T, N)
    return {"probs": probs, "y_true": all_labels}


# ──────────────────────────────────────────────────────────────────────────────
# Uncertainty metrics derived from the MC-dropout distribution
# ──────────────────────────────────────────────────────────────────────────────


def _bernoulli_entropy(p: np.ndarray) -> np.ndarray:
    """H(Bern(p)) in nats. Clamp the edges to avoid log(0)."""
    p_c = np.clip(p, EPS, 1.0 - EPS)
    return -(p_c * np.log(p_c) + (1.0 - p_c) * np.log(1.0 - p_c))


@dataclass
class UncertaintyArrays:
    mean: np.ndarray            # (N,)  posterior-mean prediction
    std: np.ndarray             # (N,)  posterior std — a dispersion proxy
    predictive_entropy: np.ndarray   # (N,) total uncertainty
    aleatoric: np.ndarray            # (N,) expected data-uncertainty
    mutual_info: np.ndarray          # (N,) epistemic uncertainty (BALD)
    y_true: np.ndarray               # (N,)


def uncertainty_from_samples(
    probs: np.ndarray, y_true: np.ndarray,
) -> UncertaintyArrays:
    """Derive the uncertainty arrays from the raw ``(T, N)`` MC-dropout
    probability matrix.

    * ``predictive_entropy = H(E_T[p_t])``   — total predictive uncertainty.
    * ``aleatoric          = E_T[H(p_t)]``   — expected data uncertainty.
    * ``mutual_info        = predictive_entropy − aleatoric``  (BALD; Houlsby
      et al. 2011). This is the epistemic signal; zero when all samples
      agree regardless of their confidence.
    """
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2-D (T, N), got shape {probs.shape}")
    mean = probs.mean(axis=0)
    std = probs.std(axis=0)
    predictive_entropy = _bernoulli_entropy(mean)
    per_sample_entropy = _bernoulli_entropy(probs)           # (T, N)
    aleatoric = per_sample_entropy.mean(axis=0)
    mutual_info = predictive_entropy - aleatoric
    return UncertaintyArrays(
        mean=mean, std=std,
        predictive_entropy=predictive_entropy,
        aleatoric=aleatoric,
        mutual_info=mutual_info,
        y_true=y_true.astype(int),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Refuse-to-predict analysis
# ──────────────────────────────────────────────────────────────────────────────


def refuse_curve(
    arrays: UncertaintyArrays,
    threshold: float = 0.5,
    signal: str = "predictive_entropy",
    fractions: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """For each abstention fraction, defer the most uncertain rows and
    recompute accuracy + AUC on the retained subset. A useful uncertainty
    signal produces a monotone increase in accuracy as we abstain more.

    * ``signal`` — which uncertainty column to sort by: ``"predictive_entropy"``,
      ``"aleatoric"``, ``"mutual_info"``, or ``"std"``.
    * ``fractions`` — abstention fractions in [0, 1). Default: 0, 5%, 10%, …, 60%.
    """
    if signal not in {"predictive_entropy", "aleatoric", "mutual_info", "std"}:
        raise ValueError(f"Unknown signal: {signal}")
    u = getattr(arrays, signal)
    order = np.argsort(-u)   # most uncertain first

    if fractions is None:
        fractions = np.arange(0.0, 0.65, 0.05)

    y_true = arrays.y_true
    mean = arrays.mean
    y_pred_all = (mean >= threshold).astype(int)
    N = len(y_true)
    rows: List[Dict[str, float]] = []
    for frac in fractions:
        n_defer = int(round(N * frac))
        keep = np.ones(N, dtype=bool)
        if n_defer > 0:
            keep[order[:n_defer]] = False
        n_kept = int(keep.sum())
        if n_kept < 20 or len(np.unique(y_true[keep])) < 2:
            continue
        rows.append({
            "signal": signal,
            "fraction_deferred": float(frac),
            "n_deferred": int(N - n_kept),
            "n_retained": n_kept,
            "accuracy_retained": float(accuracy_score(y_true[keep], y_pred_all[keep])),
            "auc_roc_retained":  float(roc_auc_score(y_true[keep], mean[keep])),
            "defer_accuracy":    float(accuracy_score(y_true[~keep], y_pred_all[~keep]))
                                  if (N - n_kept) > 0 else float("nan"),
            "defer_positive_rate": float(y_true[~keep].mean())
                                    if (N - n_kept) > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────


def plot_refuse_curve(df: pd.DataFrame, out_path: Path) -> None:
    fig, (ax_acc, ax_auc) = plt.subplots(1, 2, figsize=(10, 4))
    for signal, sub in df.groupby("signal"):
        sub = sub.sort_values("fraction_deferred")
        ax_acc.plot(sub["fraction_deferred"], sub["accuracy_retained"],
                    marker="o", lw=1.6, label=signal)
        ax_auc.plot(sub["fraction_deferred"], sub["auc_roc_retained"],
                    marker="o", lw=1.6, label=signal)
    for ax, title, ylabel in (
        (ax_acc, "Accuracy on retained subset", "Accuracy"),
        (ax_auc, "AUC-ROC on retained subset", "AUC-ROC"),
    ):
        ax.set_xlabel("Abstention fraction")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_entropy_hist(arrays: UncertaintyArrays, out_path: Path, threshold: float = 0.5) -> None:
    y_pred = (arrays.mean >= threshold).astype(int)
    correct = (y_pred == arrays.y_true)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(arrays.predictive_entropy[correct],  bins=30, alpha=0.6,
            label=f"correct (n={correct.sum()})", color="#0072B2")
    ax.hist(arrays.predictive_entropy[~correct], bins=30, alpha=0.6,
            label=f"incorrect (n={(~correct).sum()})", color="#D55E00")
    ax.set_xlabel("Predictive entropy (nats)")
    ax.set_ylabel("Number of rows")
    ax.set_title("MC-dropout predictive entropy — correct vs incorrect rows")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end
# ──────────────────────────────────────────────────────────────────────────────


def _load_model_from_run(run_dir: Path, trust_source: bool = True) -> TabularTransformer:
    """Construct a :class:`TabularTransformer` from the run's config.json
    and load the checkpoint weights. ``trust_source=True`` for artefacts we
    generated ourselves."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())

    import json as _json
    meta_path = run_dir / "best.pt.meta.json"
    cat_vocab_sizes: Optional[Dict[str, int]] = None
    if meta_path.is_file():
        meta = _json.loads(meta_path.read_text())
        cat_vocab_sizes = (meta.get("cat_vocab_sizes")
                           if isinstance(meta, dict) else None)

    model = TabularTransformer(
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg.get("d_ff"),
        dropout=cfg.get("dropout", 0.1),
        attn_dropout=cfg.get("attn_dropout"),
        ffn_dropout=cfg.get("ffn_dropout"),
        residual_dropout=cfg.get("residual_dropout"),
        classification_dropout=cfg.get("classification_dropout", 0.1),
        pool=cfg.get("pool", "cls"),
        use_temporal_pos=cfg.get("use_temporal_pos", False),
        temporal_decay_mode=cfg.get("temporal_decay_mode", "off"),
        feature_group_bias_mode=cfg.get("feature_group_bias_mode", "off"),
        aux_pay0=bool(cfg.get("aux_pay0_lambda", 0.0) > 0),
        cat_vocab_sizes=cat_vocab_sizes,
    )
    load_checkpoint(run_dir / "best.pt", model, trust_source=trust_source)
    return model


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Phase 11B Monte-Carlo dropout uncertainty quantification on a "
            "trained transformer run. T forward passes with dropout active."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run", type=Path,
                   default=Path("results/transformer/seed_42"),
                   help="Run directory containing best.pt + config.json.")
    p.add_argument("--test-csv", type=Path,
                   default=Path("data/processed/test_scaled.csv"))
    p.add_argument("--metadata", type=Path,
                   default=Path("data/processed/feature_metadata.json"))
    p.add_argument("--n-samples", type=int, default=50,
                   help="Number of MC-dropout forward passes.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("results/uncertainty"))
    p.add_argument("--figures-dir", type=Path, default=Path("figures"))
    p.add_argument("--trust-source", action="store_true", default=True,
                   help="Load our own checkpoints with weights_only=False.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args(argv)

    if not (args.test_csv.is_file() and args.metadata.is_file()):
        logger.error("Missing preprocessing outputs — run run_pipeline.py first.")
        return 1
    if not (args.run / "best.pt").is_file():
        logger.error("No checkpoint at %s — run train.py first.", args.run)
        return 1

    logger.info("Loading checkpoint from %s", args.run)
    model = _load_model_from_run(args.run, trust_source=args.trust_source)
    device = torch.device("cpu")
    model.to(device)

    df = pd.read_csv(args.test_csv)
    meta = json.loads(args.metadata.read_text())
    cat_vocab = build_categorical_vocab(meta)
    ds = CreditDefaultDataset(df, cat_vocab, verbose=False)
    loader = make_loader(ds, batch_size=args.batch_size, mode="test", seed=args.seed)

    logger.info("Running %d MC-dropout forward passes over %d rows",
                args.n_samples, len(df))
    samples = mc_dropout_predict(
        model, loader,
        n_samples=args.n_samples, device=device, seed=args.seed,
    )
    arrays = uncertainty_from_samples(samples["probs"], samples["y_true"])

    # Headline scalars.
    y_pred = (arrays.mean >= 0.5).astype(int)
    correct_mask = y_pred == arrays.y_true
    headline = {
        "n_samples_per_row": int(args.n_samples),
        "run": args.run.name,
        "auc_roc_mean": float(roc_auc_score(arrays.y_true, arrays.mean)),
        "accuracy_at_0.5": float(correct_mask.mean()),
        "mean_predictive_entropy": float(arrays.predictive_entropy.mean()),
        "mean_aleatoric_entropy": float(arrays.aleatoric.mean()),
        "mean_mutual_information": float(arrays.mutual_info.mean()),
        "mean_std": float(arrays.std.mean()),
        # Spearman correlation between predictive entropy and the
        # misclassification indicator — a "calibration of uncertainty" score.
        "entropy_misclass_correlation": float(
            pd.Series(arrays.predictive_entropy)
            .corr(pd.Series((~correct_mask).astype(int)), method="spearman")
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "mc_dropout.npz",
        mean=arrays.mean.astype(np.float32),
        std=arrays.std.astype(np.float32),
        predictive_entropy=arrays.predictive_entropy.astype(np.float32),
        aleatoric=arrays.aleatoric.astype(np.float32),
        mutual_info=arrays.mutual_info.astype(np.float32),
        y_true=arrays.y_true.astype(np.int64),
        probs=samples["probs"].astype(np.float32),
    )
    (args.output_dir / "uncertainty_summary.json").write_text(
        json.dumps(headline, indent=2, default=str)
    )

    # Refuse-to-predict curves for each signal.
    dfs = [refuse_curve(arrays, signal=s)
           for s in ("predictive_entropy", "mutual_info", "std")]
    refuse = pd.concat(dfs, ignore_index=True)
    refuse.to_csv(args.output_dir / "refuse_curve.csv", index=False)

    plot_refuse_curve(refuse, args.figures_dir / "uncertainty_refuse_curve.png")
    plot_entropy_hist(arrays, args.figures_dir / "uncertainty_entropy_hist.png")

    logger.info("Uncertainty artefacts in %s", args.output_dir)
    print()
    for k, v in headline.items():
        print(f"  {k:40s} {v}")
    print("\nRefuse-curve head (predictive_entropy):")
    print(refuse[refuse["signal"] == "predictive_entropy"].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
