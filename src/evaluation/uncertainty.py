"""MC-dropout uncertainty (novelty N11): Gal & Ghahramani 2016.

Keep the nn.Dropout modules in train-mode at inference time and run the
same batch through the network T times. The T samples approximate the
posterior predictive over network weights under the variational
interpretation of dropout. From those samples we compute:

* Per-row mean and std of the probabilities.
* Predictive entropy (total uncertainty): ``H(E_T[p_t])``.
* Aleatoric entropy (data uncertainty): ``E_T[H(p_t)]``.
* Mutual information (epistemic / model uncertainty, the BALD score of
  Houlsby et al. 2011): ``predictive - aleatoric``.

These three quantities reveal *why* a prediction is uncertain: high
aleatoric means the inputs themselves are ambiguous and no more data
helps, high mutual information means the model is uncertain because it
hasn't seen enough of this kind of row. A refuse-to-predict curve then
defers the top-k most uncertain rows and reports accuracy / AUC on the
retained subset -- a well-calibrated uncertainty signal makes the
retained subset monotonically cleaner.

Reference: Y. Gal, Z. Ghahramani, *Dropout as a Bayesian Approximation*,
ICML 2016. BALD follows N. Houlsby et al., *Bayesian Active Learning
by Disagreement*, 2011.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score

from ..models.model import TabularTransformer
from ..tokenization.tokenizer import build_categorical_vocab
from ..training.dataset import CreditDefaultDataset, make_loader
from ..training.utils import load_checkpoint

logger = logging.getLogger(__name__)

__all__ = [
    "UncertaintyArrays",
    "enable_dropout",
    "mc_dropout_predict",
    "uncertainty_from_samples",
    "refuse_curve",
    "plot_refuse_curve",
    "plot_entropy_hist",
    "main",
    "EPS",
]

#: Tiny epsilon for log(p). Tighter than the calibration module's EPS because
#: log(1e-12) is still finite and the only use is inside an entropy sum.
EPS = 1e-12


def enable_dropout(model: nn.Module) -> int:
    """Flip every ``nn.Dropout*`` module into ``train(True)``.

    Leaves BatchNorm / LayerNorm / etc alone -- we only want the
    Bernoulli masks on, not the rest of the stochastic train-mode
    machinery (which would corrupt the batch statistics).

    Returns
    -------
    int
        Count of dropout modules flipped. A count of 0 means the model
        has no stochastic path and MC-dropout is a no-op -- the caller
        should treat that as an error.
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
    """Run ``n_samples`` stochastic forward passes through ``loader``.

    Parameters
    ----------
    model : TabularTransformer
        Trained model. Must contain at least one ``nn.Dropout`` module;
        otherwise we raise, since a deterministic forward pass is
        meaningless here.
    loader : DataLoader
        Iterable yielding the training-style batch dict with keys
        ``cat``, ``num``, etc. (see :class:`CreditDefaultDataset`).
    n_samples : int
        Number of stochastic passes. ``>= 2`` is required for variance.
        Gal & Ghahramani suggest 50-100 for stable BALD estimates; we
        default to 50 as the accuracy/cost knee on this model.
    device : torch.device or None
        Defaults to the model's current device.
    seed : int
        Base seed. Every pass reseeds to ``seed + t`` so a rerun with
        the same ``seed`` is bit-identical.

    Returns
    -------
    dict with keys
        ``probs`` : array, shape (T, N). Per-pass positive-class probs.
        ``y_true``: array, shape (N,). Collected from the loader once.
    """
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2 for MC-dropout, got {n_samples}")
    if device is None:
        device = next(model.parameters()).device

    # Eval first (drops train-mode flags on non-dropout layers), then
    # selectively flip dropout back on.
    model.eval()
    n_dropout = enable_dropout(model)
    if n_dropout == 0:
        raise RuntimeError(
            "No nn.Dropout modules found; this model has no MC-dropout signal. "
            "Enable dropout>0 at construction and re-train."
        )

    torch.manual_seed(seed)

    all_probs: List[np.ndarray] = []
    all_labels: Optional[np.ndarray] = None

    for t in range(n_samples):
        # Per-pass reseed so reruns with the same seed are bit-identical
        # even under multiprocessing loaders (which may not preserve a
        # single global RNG state across passes).
        torch.manual_seed(seed + t)
        probs_chunks: List[np.ndarray] = []
        labels_chunks: List[np.ndarray] = []
        for batch in loader:
            # Move to device manually rather than via a pin_memory /
            # collate hook so this works for any loader the caller
            # hands in, including the plain sequential test loader.
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
        # Labels only collected on the first pass -- they don't change.
        if all_labels is None:
            all_labels = np.concatenate(labels_chunks, axis=0)

    # Restore eval-mode dropout (off). Without this the caller's next
    # model(batch) silently returns stochastic output -- a very subtle
    # bug that would only show up as non-reproducible downstream scores.
    model.eval()
    probs = np.stack(all_probs, axis=0)  # (T, N)
    return {"probs": probs, "y_true": all_labels}


def _bernoulli_entropy(p: np.ndarray) -> np.ndarray:
    """H(Bern(p)) in nats, elementwise.

    Clipped into ``[EPS, 1 - EPS]`` so ``log(0)`` doesn't bite on
    confident predictions. The limit ``p * log(p) -> 0`` as ``p -> 0``,
    so clipping instead of zeroing-out contributes a negligible (~1e-12)
    bias for the cost of staying finite.
    """
    p_c = np.clip(p, EPS, 1.0 - EPS)
    return -(p_c * np.log(p_c) + (1.0 - p_c) * np.log(1.0 - p_c))


@dataclass
class UncertaintyArrays:
    """Packaged per-row uncertainty arrays.

    All arrays are aligned row-for-row with the test split; shape is
    ``(N,)`` throughout. Labels are carried so the caller can filter
    without having to re-thread ``y_true`` through every plot.
    """

    mean: np.ndarray  # (N,)
    std: np.ndarray  # (N,)
    predictive_entropy: np.ndarray  # (N,) total uncertainty
    aleatoric: np.ndarray  # (N,) data / irreducible uncertainty
    mutual_info: np.ndarray  # (N,) BALD = predictive - aleatoric (epistemic)
    y_true: np.ndarray  # (N,)


def uncertainty_from_samples(
    probs: np.ndarray,
    y_true: np.ndarray,
) -> UncertaintyArrays:
    """Collapse ``(T, N)`` samples into the per-row uncertainty bundle.

    The BALD / information-theoretic decomposition::

        predictive_entropy  = H(E_T[p_t])             # total
        aleatoric           = E_T[H(p_t)]             # data / irreducible
        mutual_info         = predictive - aleatoric  # model / epistemic (BALD)

    Intuition: if all T samples agree on ``p``, the per-sample entropies
    match the mean's entropy exactly, mutual information is ~0 -- the
    model is confident about the prediction regardless of whether it's
    confident in the class. If the samples disagree, mean entropy is
    high but individual entropies may be low, driving BALD up.

    Parameters
    ----------
    probs : array, shape (T, N)
    y_true : array, shape (N,)

    Returns
    -------
    UncertaintyArrays

    Raises
    ------
    ValueError
        If ``probs`` is not 2-D.
    """
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2-D (T, N), got shape {probs.shape}")
    mean = probs.mean(axis=0)
    std = probs.std(axis=0)
    predictive_entropy = _bernoulli_entropy(mean)
    # Entropy per (t, n), then mean over t -> aleatoric per n.
    per_sample_entropy = _bernoulli_entropy(probs)  # (T, N)
    aleatoric = per_sample_entropy.mean(axis=0)
    # BALD. Theoretically >= 0 but can be slightly negative from float
    # rounding -- we leave it as-is so downstream Spearman correlations
    # don't get clipped artefacts.
    mutual_info = predictive_entropy - aleatoric
    return UncertaintyArrays(
        mean=mean,
        std=std,
        predictive_entropy=predictive_entropy,
        aleatoric=aleatoric,
        mutual_info=mutual_info,
        y_true=y_true.astype(int),
    )


def refuse_curve(
    arrays: UncertaintyArrays,
    threshold: float = 0.5,
    signal: str = "predictive_entropy",
    fractions: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Refuse-to-predict sweep: defer top-k most uncertain rows, score the rest.

    A well-calibrated uncertainty signal makes ``accuracy_retained`` rise
    monotonically with deferral fraction -- the model was right about the
    rows it was confident on, wrong about the rows it wasn't. A flat
    (or falling) curve means the signal doesn't carry information
    about correctness.

    ``signal`` chooses which uncertainty to order by:

    * ``predictive_entropy`` -- total uncertainty. Usually the best
      single signal for a classifier.
    * ``aleatoric``          -- data ambiguity only. Low-aleatoric
      rows can still be mis-predicted if the model is wrong.
    * ``mutual_info``        -- BALD. Epistemic uncertainty; good
      active-learning signal but noisier for deferral.
    * ``std``                -- per-row std of the T probabilities.
      Cheap proxy; correlates with predictive_entropy.

    Rows where the retained subset drops below 20 or becomes single-class
    are dropped from the output -- the metrics are undefined there.
    """
    if signal not in {"predictive_entropy", "aleatoric", "mutual_info", "std"}:
        raise ValueError(f"Unknown signal: {signal}")
    u = getattr(arrays, signal)
    # Descending sort: most uncertain first, so ``order[:k]`` is the
    # set we defer.
    order = np.argsort(-u)

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
        # AUC needs both classes; accuracy reads poorly on n < 20.
        if n_kept < 20 or len(np.unique(y_true[keep])) < 2:
            continue
        rows.append(
            {
                "signal": signal,
                "fraction_deferred": float(frac),
                "n_deferred": int(N - n_kept),
                "n_retained": n_kept,
                "accuracy_retained": float(accuracy_score(y_true[keep], y_pred_all[keep])),
                "auc_roc_retained": float(roc_auc_score(y_true[keep], mean[keep])),
                # Also report the deferred-subset stats: if the uncertainty
                # signal is useful, the deferred subset should have both low
                # accuracy and an elevated positive rate.
                "defer_accuracy": (
                    float(accuracy_score(y_true[~keep], y_pred_all[~keep]))
                    if (N - n_kept) > 0
                    else float("nan")
                ),
                "defer_positive_rate": (
                    float(y_true[~keep].mean()) if (N - n_kept) > 0 else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_refuse_curve(df: pd.DataFrame, out_path: Path) -> None:
    """Two-panel refuse curves: accuracy (L) and AUC (R) on retained subset.

    One line per signal so their shapes can be compared directly; a
    signal that sits flat here is the one to drop from the ``run_all``
    report.
    """
    fig, (ax_acc, ax_auc) = plt.subplots(1, 2, figsize=(10, 4))
    for signal, sub in df.groupby("signal"):
        sub = sub.sort_values("fraction_deferred")
        ax_acc.plot(
            sub["fraction_deferred"], sub["accuracy_retained"], marker="o", lw=1.6, label=signal
        )
        ax_auc.plot(
            sub["fraction_deferred"], sub["auc_roc_retained"], marker="o", lw=1.6, label=signal
        )
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
    """Stacked predictive-entropy histogram split by correct vs incorrect.

    The diagnostic we look for: the incorrect distribution should sit to
    the *right* of the correct one (incorrect predictions are, on
    average, more uncertain). Overlapping distributions mean the
    uncertainty is not a useful correctness signal.
    """
    y_pred = (arrays.mean >= threshold).astype(int)
    correct = y_pred == arrays.y_true
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(
        arrays.predictive_entropy[correct],
        bins=30,
        alpha=0.6,
        label=f"correct (n={correct.sum()})",
        color="#0072B2",
    )
    ax.hist(
        arrays.predictive_entropy[~correct],
        bins=30,
        alpha=0.6,
        label=f"incorrect (n={(~correct).sum()})",
        color="#D55E00",
    )
    ax.set_xlabel("Predictive entropy (nats)")
    ax.set_ylabel("Number of rows")
    ax.set_title("MC-dropout predictive entropy, correct vs incorrect rows")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _load_model_from_run(run_dir: Path, trust_source: bool = False) -> TabularTransformer:
    """Rebuild the model skeleton from ``config.json`` and load weights.

    ``trust_source=False`` (the default) uses the ``.weights`` sidecar
    with ``weights_only=True`` -- this is the C-1 safe path from
    ``SECURITY_AUDIT.md`` that prevents arbitrary pickle execution on
    untrusted checkpoints. ``True`` flips to ``weights_only=False`` for
    checkpoints you produced yourself, where the attack surface is
    nil. The shipped ``best.pt`` always has a sidecar, so leaving
    this off is the normal path.
    """
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())

    # Aliased json import to avoid shadowing the module-level ``json``;
    # historical artefact from the first version of the function.
    import json as _json

    meta_path = run_dir / "best.pt.meta.json"
    cat_vocab_sizes: Optional[Dict[str, int]] = None
    if meta_path.is_file():
        meta = _json.loads(meta_path.read_text())
        cat_vocab_sizes = meta.get("cat_vocab_sizes") if isinstance(meta, dict) else None

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
        description="MC-dropout uncertainty: T passes with dropout active.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run",
        type=Path,
        default=Path("results/transformer/seed_42"),
        help="Run dir with best.pt + config.json.",
    )
    p.add_argument("--test-csv", type=Path, default=Path("data/processed/splits/test_scaled.csv"))
    p.add_argument("--metadata", type=Path, default=Path("data/processed/feature_metadata.json"))
    p.add_argument("--n-samples", type=int, default=50, help="Number of MC-dropout passes.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("results/evaluation/uncertainty"))
    p.add_argument("--figures-dir", type=Path, default=Path("figures/evaluation/uncertainty"))
    p.add_argument(
        "--trust-source",
        action="store_true",
        default=False,
        help="weights_only=False. Opts out of SECURITY_AUDIT C-1; "
        "only use for checkpoints you trained yourself. The "
        "shipped best.pt has a .weights sidecar, so leaving "
        "this off is the normal path.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args(argv)

    if not (args.test_csv.is_file() and args.metadata.is_file()):
        logger.error("Missing preprocessing outputs; run scripts/run_pipeline.py first.")
        return 1
    if not (args.run / "best.pt").is_file():
        logger.error("No checkpoint at %s; run train.py first.", args.run)
        return 1

    logger.info("Loading checkpoint from %s", args.run)
    model = _load_model_from_run(args.run, trust_source=args.trust_source)
    # CPU is plenty for 50 forward passes on ~6k rows. Passing device
    # explicitly keeps this deterministic across machines (no silent
    # cuDNN selection).
    device = torch.device("cpu")
    model.to(device)

    df = pd.read_csv(args.test_csv)
    meta = json.loads(args.metadata.read_text())
    cat_vocab = build_categorical_vocab(meta)
    ds = CreditDefaultDataset(df, cat_vocab, verbose=False)
    loader = make_loader(ds, batch_size=args.batch_size, mode="test", seed=args.seed)

    logger.info("Running %d MC-dropout forward passes over %d rows", args.n_samples, len(df))
    samples = mc_dropout_predict(
        model,
        loader,
        n_samples=args.n_samples,
        device=device,
        seed=args.seed,
    )
    arrays = uncertainty_from_samples(samples["probs"], samples["y_true"])

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
        # Spearman of (entropy, misclass indicator): the headline
        # "does uncertainty track correctness?" number. >0.1 is a
        # usable signal on this dataset; >0.2 is strong.
        "entropy_misclass_correlation": float(
            pd.Series(arrays.predictive_entropy).corr(
                pd.Series((~correct_mask).astype(int)), method="spearman"
            )
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

    # Three signals on one plot makes the "which signal wins?" comparison
    # obvious; std is omitted since it correlates nearly 1:1 with
    # predictive_entropy for this model.
    dfs = [refuse_curve(arrays, signal=s) for s in ("predictive_entropy", "mutual_info", "std")]
    refuse = pd.concat(dfs, ignore_index=True)
    refuse.to_csv(args.output_dir / "refuse_curve.csv", index=False)

    plot_refuse_curve(refuse, args.figures_dir / "uncertainty_refuse_curve.png")
    plot_entropy_hist(arrays, args.figures_dir / "uncertainty_entropy_hist.png")

    logger.info("Uncertainty artefacts in %s", args.output_dir)
    print()
    for k, v in headline.items():
        print(f"  {k:40s} {v}")
    print("\nRefuse-curve head (predictive_entropy):")
    print(
        refuse[refuse["signal"] == "predictive_entropy"].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
