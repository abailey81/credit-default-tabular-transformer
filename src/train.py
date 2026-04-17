"""
train.py — Supervised training loop for the TabularTransformer.

Implements the full Plan §8 specification:

    Optimiser         AdamW with decoupled weight decay              (§8.1)
    LR schedule       Linear warmup + cosine decay to ~1% of peak    (§8.2)
    Gradient clipping max_norm = 1.0                                 (§8.3)
    Regularisation    Independently-ablatable attn / FFN / residual  (§8.4)
                      dropout channels on the encoder + class-head
    Early stopping    On validation AUC-ROC, min_delta-aware         (§8.5)
    Batching          Batch size 256, optional stratified sampler    (§8.6/§8.8)
    Two-stage LR      Pretrained encoder at ``encoder_lr_ratio × lr``
                      when a MTLM checkpoint is loaded               (§8.5.5)
    Multi-task        Joint BCE + CE loss with λ-weighted PAY_0
                      forecast head when ``--aux-pay0-lambda > 0``   (§8.6, N5)
    Logging           Per-epoch CSV + JSON config snapshot           (§8.9)

Every ablation flag in Plan §11 that can be toggled from a training
invocation is exposed via argparse — one script drives A2 (depth), A3 (heads),
A4 (d_model), A5 (pool), A7 (temporal pos), A10 (loss family), A11 (focal γ),
A12 (dropout), A19 (data fraction), A22 (temporal decay), plus the N5
auxiliary objective (A16).

CLI example
-----------

.. code-block:: bash

    poetry run python src/train.py \\
        --seed 42 --d-model 32 --n-heads 4 --n-layers 2 \\
        --loss focal --focal-gamma 2.0 --focal-alpha balanced \\
        --lr 3e-4 --weight-decay 1e-5 \\
        --warmup-frac 0.10 --min-lr-frac 0.01 \\
        --epochs 200 --patience 20 --batch-size 256 \\
        --stratified-batches \\
        --temporal-decay-mode scalar

Outputs (under ``--output-dir``, defaults to
``results/transformer/seed_{seed}/``):

    config.json            → resolved argparse + runtime metadata + param count
    train_log.csv          → per-epoch train/val metrics, LR, grad norm, wall-clock
    best.pt                → checkpoint bundle (+ .weights + .meta.json sidecars)
    test_metrics.json      → final metrics at τ=0.5 + threshold sweep
    test_predictions.npz   → (y_true, y_prob, y_pred) for evaluate.py / interpret.py
    test_attn_weights.npz  → per-layer attention on the test set for Phase 10
                             rollout (skip with ``--no-save-attn``)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

# Make ``src/`` importable when invoked either as ``python src/train.py`` or
# as ``python -m src.train``.
_HERE = Path(__file__).resolve()
_SRC = _HERE.parent
_REPO = _HERE.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dataset import make_loader  # noqa: E402
from losses import (  # noqa: E402
    FocalLoss,
    LabelSmoothingBCELoss,
    WeightedBCELoss,
    balanced_alpha,
    compute_pos_weight,
)
from model import TabularTransformer  # noqa: E402
from tokenizer import (  # noqa: E402
    CreditDefaultDataset,
    PAY_RAW_NUM_CLASSES,
    build_categorical_vocab,
)
from utils import (  # noqa: E402
    EarlyStopping,
    build_checkpoint_metadata,
    configure_logging,
    count_parameters,
    describe_device,
    format_parameter_count,
    get_device,
    save_checkpoint,
    set_deterministic,
)

logger = logging.getLogger(__name__)

ECE_N_BINS = 15  # Plan §13.2


# ──────────────────────────────────────────────────────────────────────────────
# Argparse
# ──────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Supervised training loop for the TabularTransformer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Reproducibility + output
    g = p.add_argument_group("reproducibility / output")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to write artefacts. Default: results/transformer/seed_{seed}.",
    )
    g.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    g.add_argument("--determinism", action="store_true",
                   help="Enable torch.use_deterministic_algorithms(warn_only=True).")

    # Model architecture (Plan §6.11 defaults)
    g = p.add_argument_group("model architecture")
    g.add_argument("--d-model", type=int, default=32)
    g.add_argument("--n-heads", type=int, default=4)
    g.add_argument("--n-layers", type=int, default=2)
    g.add_argument("--d-ff", type=int, default=None,
                   help="FFN hidden dim. Defaults to 4 * d_model.")
    g.add_argument("--pool", choices=["cls", "mean", "max"], default="cls",
                   help="Plan §6.7 / Ablation A5.")
    g.add_argument("--use-temporal-pos", action="store_true",
                   help="Plan §5.4 / Ablation A7.")
    g.add_argument(
        "--temporal-decay-mode",
        choices=["off", "scalar", "per_head"],
        default="off",
        help="Plan §6.12.2 (Novelty N3) / Ablation A22.",
    )
    g.add_argument(
        "--feature-group-bias-mode",
        choices=["off", "scalar", "per_head"],
        default="off",
        help="Plan §6.12.1 (Novelty N2) / Ablation A21 — feature-group "
             "attention bias. Groups: CLS / demographic / PAY / BILL_AMT / PAY_AMT.",
    )

    # Dropout channels (Ablation A12)
    g = p.add_argument_group("regularisation")
    g.add_argument("--dropout", type=float, default=0.1)
    g.add_argument("--attn-dropout", type=float, default=None)
    g.add_argument("--ffn-dropout", type=float, default=None)
    g.add_argument("--residual-dropout", type=float, default=None)
    g.add_argument("--classification-dropout", type=float, default=0.1)

    # Primary loss (Ablations A10 / A11)
    g = p.add_argument_group("loss")
    g.add_argument("--loss", choices=["focal", "wbce", "label-smoothing"],
                   default="focal")
    g.add_argument("--focal-gamma", type=float, default=2.0,
                   help="Plan §7.2 / Ablation A11.")
    g.add_argument(
        "--focal-alpha", type=str, default="balanced",
        help="'balanced' / scalar α_pos in [0, 1] / tuple '(α_pos, α_neg)' / 'none'.",
    )
    g.add_argument("--label-smoothing-eps", type=float, default=0.05,
                   help="Plan §7.3 / Ablation (optional).")

    # Multi-task (N5 / A16)
    g.add_argument("--aux-pay0-lambda", type=float, default=0.0,
                   help="Joint-loss weight λ on the PAY_0 forecast auxiliary "
                        "(Plan §8.6 / Novelty N5 / Ablation A16). 0 disables.")

    # Optimiser (Plan §8.1-§8.3)
    g = p.add_argument_group("optimisation")
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--weight-decay", type=float, default=1e-5)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--beta1", type=float, default=0.9)
    g.add_argument("--beta2", type=float, default=0.999)
    g.add_argument("--eps", type=float, default=1e-8)
    g.add_argument("--warmup-frac", type=float, default=0.10,
                   help="Plan §8.2: 5–10%% warmup. Default 10%%.")
    g.add_argument("--min-lr-frac", type=float, default=0.01,
                   help="Cosine floor as a fraction of peak LR.")

    # Training schedule (Plan §8.6)
    g = p.add_argument_group("training schedule")
    g.add_argument("--epochs", type=int, default=200)
    g.add_argument("--patience", type=int, default=20,
                   help="Early stopping patience on val AUC-ROC. Plan §8.5.")
    g.add_argument("--min-delta", type=float, default=1e-4)
    g.add_argument("--batch-size", type=int, default=256)
    g.add_argument("--stratified-batches", action="store_true",
                   help="Plan §8.8 — every batch contains ~22.1%% positives.")
    g.add_argument("--num-workers", type=int, default=0)

    # Fine-tuning after MTLM (Plan §8.5.5)
    g = p.add_argument_group("fine-tuning from MTLM pretraining")
    g.add_argument("--pretrained-encoder", type=str, default=None,
                   help="Path to a MTLM pretrain .pt (Plan §8.5.5). When set, "
                        "the encoder+embedding weights are loaded strict=False "
                        "into the fresh model; the classification head stays "
                        "at fresh init.")
    g.add_argument("--encoder-lr-ratio", type=float, default=0.2,
                   help="Pretrained encoder LR = encoder_lr_ratio × --lr.")
    g.add_argument("--trust-checkpoint", action="store_true",
                   help="Only pass for checkpoints YOU produced. Enables the "
                        "full pickle load path of torch.load (required to "
                        "restore optimiser state); default is weights-only.")

    # Ablation A19 — scaling curve
    g = p.add_argument_group("data scaling")
    g.add_argument("--data-frac", type=float, default=1.0,
                   help="Fraction of the training split to use (Ablation A19).")

    # Artefact control
    g = p.add_argument_group("artefact control")
    g.add_argument("--no-save-attn", action="store_true",
                   help="Skip writing test_attn_weights.npz (saves ~30 MB per run).")
    g.add_argument("--no-save-predictions", action="store_true",
                   help="Skip writing test_predictions.npz.")
    g.add_argument("--log-every", type=int, default=1,
                   help="Print a log line every N epochs (file log is per-epoch).")

    # CI / smoke test
    g = p.add_argument_group("debug")
    g.add_argument("--smoke-test", action="store_true",
                   help="Run 2 epochs on ~500 training rows then exit. "
                        "For CI; skips attention-weight dump.")

    return p


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _git_sha(default: str = "unknown") -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=2,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return default


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        out = _REPO / "results" / "transformer" / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _resolve_focal_alpha(spec: str) -> Any:
    """Parse ``--focal-alpha`` into the shape FocalLoss expects."""
    spec = spec.strip().lower()
    if spec == "balanced":
        return "balanced"
    if spec == "none":
        return None
    # Tuple form: "(0.75, 0.25)" or "0.75,0.25".
    if spec.startswith("(") and spec.endswith(")"):
        spec = spec[1:-1]
    if "," in spec:
        parts = [float(x.strip()) for x in spec.split(",")]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"--focal-alpha tuple must have 2 values, got {parts}"
            )
        return tuple(parts)
    try:
        return float(spec)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--focal-alpha must be 'balanced', 'none', a scalar in [0,1], or "
            f"a tuple '(α_pos, α_neg)'; got {spec!r}"
        ) from exc


def _load_splits(
    seed: int, data_frac: float, smoke_test: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    meta_path = _REPO / "data/processed/feature_metadata.json"
    paths = {
        "train": _REPO / "data/processed/train_scaled.csv",
        "val":   _REPO / "data/processed/val_scaled.csv",
        "test":  _REPO / "data/processed/test_scaled.csv",
    }
    missing = [str(p) for p in (meta_path, *paths.values()) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing preprocessing outputs:\n  "
            + "\n  ".join(missing)
            + "\nRun `poetry run python run_pipeline.py --preprocess-only`."
        )

    meta = json.loads(meta_path.read_text())
    train_df = pd.read_csv(paths["train"])
    val_df = pd.read_csv(paths["val"])
    test_df = pd.read_csv(paths["test"])

    def _stratified_sample(df: pd.DataFrame, *, n: Optional[int] = None,
                           frac: Optional[float] = None) -> pd.DataFrame:
        """Stratified sample from a DataFrame by the ``DEFAULT`` column,
        preserving class ratios. Uses ``include_groups=False`` so pandas does
        not emit a FutureWarning about grouping-column behaviour."""
        pieces = []
        for _, group in df.groupby("DEFAULT", group_keys=False):
            if n is not None:
                take = min(n, len(group))
                pieces.append(group.sample(n=take, random_state=seed))
            else:
                assert frac is not None
                pieces.append(group.sample(frac=frac, random_state=seed))
        return pd.concat(pieces, ignore_index=True)

    if smoke_test:
        # Keep enough positives for AUC to be defined.
        train_df = _stratified_sample(train_df, n=250)
        val_df   = _stratified_sample(val_df,   n=100)
        test_df  = _stratified_sample(test_df,  n=100)
    elif data_frac < 1.0:
        train_df = _stratified_sample(train_df, frac=data_frac)
        logger.info("Ablation A19: reduced train to %.0f%% → %d rows",
                    data_frac * 100, len(train_df))

    return train_df, val_df, test_df, meta


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = {kk: vv.to(device, non_blocking=True) for kk, vv in v.items()}
        elif isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# ──────────────────────────────────────────────────────────────────────────────
# LR schedule
# ──────────────────────────────────────────────────────────────────────────────


def build_cosine_warmup_schedule(
    optim: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_frac: float = 0.01,
) -> LambdaLR:
    """Linear warmup from 0 → peak over ``warmup_steps``; cosine decay to
    ``min_lr_frac * peak`` over the remainder. Plan §8.2."""
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")
    warmup_steps = max(0, int(warmup_steps))
    min_lr_frac = float(max(0.0, min(1.0, min_lr_frac)))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        denom = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / denom))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_frac + (1.0 - min_lr_frac) * cosine

    return LambdaLR(optim, lr_lambda)


# ──────────────────────────────────────────────────────────────────────────────
# Loss factory
# ──────────────────────────────────────────────────────────────────────────────


def build_primary_loss(args: argparse.Namespace, y_train: torch.Tensor) -> nn.Module:
    if args.loss == "wbce":
        return WeightedBCELoss(pos_weight=compute_pos_weight(y_train))

    if args.loss == "label-smoothing":
        return LabelSmoothingBCELoss(
            epsilon=args.label_smoothing_eps,
            pos_weight=compute_pos_weight(y_train),
        )

    # Default — focal (Plan §7.2 primary).
    alpha = _resolve_focal_alpha(args.focal_alpha)
    if alpha == "balanced":
        alpha = balanced_alpha(y_train)  # (α_pos, α_neg)
    return FocalLoss(gamma=args.focal_gamma, alpha=alpha)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = ECE_N_BINS) -> float:
    """Expected Calibration Error with equal-width bins (Plan §13.2)."""
    y_true = y_true.astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_prob)
    if n == 0:
        return float("nan")
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def _safe_metric(fn, *args, **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except (ValueError, ZeroDivisionError):
        return float("nan")


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    prefix: str = "",
) -> Dict[str, float]:
    y_true_int = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        f"{prefix}auc_roc":   _safe_metric(roc_auc_score, y_true_int, y_prob),
        f"{prefix}auc_pr":    _safe_metric(average_precision_score, y_true_int, y_prob),
        f"{prefix}f1":        _safe_metric(f1_score, y_true_int, y_pred, zero_division=0),
        f"{prefix}accuracy":  _safe_metric(accuracy_score, y_true_int, y_pred),
        f"{prefix}precision": _safe_metric(precision_score, y_true_int, y_pred, zero_division=0),
        f"{prefix}recall":    _safe_metric(recall_score, y_true_int, y_pred, zero_division=0),
        f"{prefix}brier":     _safe_metric(brier_score_loss, y_true_int, y_prob),
        f"{prefix}ece":       compute_ece(y_true_int, y_prob),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation / training loops
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    collect_attn: bool = False,
) -> Dict[str, Any]:
    """One forward pass over a loader. Returns a dict with per-sample
    predictions, ground truth, metrics, and (optionally) stacked attention
    weights per layer."""
    model.eval()
    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    attns_per_layer: Optional[List[List[np.ndarray]]] = None

    for batch in loader:
        batch = _to_device(batch, device)
        out = model(batch, return_attn=collect_attn)
        prob = torch.sigmoid(out["logit"]).detach().cpu().numpy()
        ps.append(prob)
        ys.append(batch["label"].detach().cpu().numpy())
        if collect_attn:
            if attns_per_layer is None:
                attns_per_layer = [[] for _ in range(len(out["attn_weights"]))]
            for i, w in enumerate(out["attn_weights"]):
                attns_per_layer[i].append(w.detach().cpu().numpy())

    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    result: Dict[str, Any] = {
        "y_true": y_true,
        "y_prob": y_prob,
        "metrics": compute_classification_metrics(y_true, y_prob),
    }
    if collect_attn and attns_per_layer is not None:
        result["attn_weights"] = [np.concatenate(a, axis=0) for a in attns_per_layer]
    return result


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    primary_loss_fn: nn.Module,
    device: torch.device,
    *,
    grad_clip: float = 1.0,
    aux_loss_fn: Optional[nn.Module] = None,
    aux_lambda: float = 0.0,
) -> Dict[str, float]:
    model.train()
    primary_losses: List[float] = []
    aux_losses: List[float] = []
    grad_norms: List[float] = []

    for batch in loader:
        batch = _to_device(batch, device)
        out = model(batch)

        loss_primary = primary_loss_fn(out["logit"], batch["label"])
        loss = loss_primary
        if aux_loss_fn is not None and aux_lambda > 0 and "aux_pay0_logits" in out:
            # pay_raw is already shifted into [0, 10]; PAY_0 is index 0.
            target_pay0 = batch["pay_raw"][:, 0].long()
            loss_aux = aux_loss_fn(out["aux_pay0_logits"], target_pay0)
            loss = loss_primary + aux_lambda * loss_aux
            aux_losses.append(float(loss_aux.item()))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        grad_norms.append(float(grad_norm))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        primary_losses.append(float(loss_primary.item()))

    return {
        "train_loss":     float(np.mean(primary_losses)),
        "train_aux_loss": float(np.mean(aux_losses)) if aux_losses else float("nan"),
        "grad_norm_mean": float(np.mean(grad_norms)),
        "grad_norm_max":  float(np.max(grad_norms)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Optimiser construction (two-group when a pretrained encoder is loaded)
# ──────────────────────────────────────────────────────────────────────────────


def build_optimizer(
    model: TabularTransformer,
    args: argparse.Namespace,
    pretrained: bool,
) -> AdamW:
    betas = (args.beta1, args.beta2)
    if not pretrained:
        return AdamW(
            model.parameters(),
            lr=args.lr,
            betas=betas,
            eps=args.eps,
            weight_decay=args.weight_decay,
        )

    # Two-group: fresh head at peak LR, pretrained encoder (+ embedding +
    # novelty bias modules) at the smaller ``lr * encoder_lr_ratio``. The
    # TabularTransformer.get_head_params / get_encoder_params helpers are
    # the canonical source for the head-vs-encoder split and stay correct
    # as new sub-modules (e.g. FeatureGroupBias) land.
    head_params = model.get_head_params()
    encoder_params = model.get_encoder_params()

    logger.info(
        "Fine-tuning two-group AdamW: encoder_lr=%.2e  head_lr=%.2e  "
        "(%d encoder params vs %d head params)",
        args.lr * args.encoder_lr_ratio,
        args.lr,
        sum(p.numel() for p in encoder_params),
        sum(p.numel() for p in head_params),
    )

    return AdamW(
        [
            {"params": encoder_params, "lr": args.lr * args.encoder_lr_ratio,
             "weight_decay": args.weight_decay},
            {"params": head_params,    "lr": args.lr,
             "weight_decay": args.weight_decay},
        ],
        betas=betas,
        eps=args.eps,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    configure_logging()
    set_deterministic(args.seed, warn_only=not args.determinism)
    device = get_device(args.device)

    logger.info("=" * 72)
    logger.info("TabularTransformer supervised training — seed %d", args.seed)
    logger.info("Device: %s", describe_device(device))
    logger.info("=" * 72)

    output_dir = _resolve_output_dir(args)

    # ── Data ──────────────────────────────────────────────────────────────
    train_df, val_df, test_df, meta = _load_splits(
        seed=args.seed, data_frac=args.data_frac, smoke_test=args.smoke_test,
    )
    cat_vocab = build_categorical_vocab(meta)
    train_ds = CreditDefaultDataset(train_df, cat_vocab, verbose=False)
    val_ds = CreditDefaultDataset(val_df, cat_vocab, verbose=False)
    test_ds = CreditDefaultDataset(test_df, cat_vocab, verbose=False)
    logger.info(
        "Splits: train %d (%.1f%% pos) | val %d (%.1f%% pos) | test %d (%.1f%% pos)",
        len(train_ds), float(train_ds.tensors["labels"].mean()) * 100,
        len(val_ds),   float(val_ds.tensors["labels"].mean()) * 100,
        len(test_ds),  float(test_ds.tensors["labels"].mean()) * 100,
    )

    train_loader = make_loader(
        train_ds, batch_size=args.batch_size, mode="train",
        stratified=args.stratified_batches,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        seed=args.seed,
    )
    val_loader = make_loader(
        val_ds, batch_size=args.batch_size, mode="val",
        num_workers=args.num_workers,
    )
    test_loader = make_loader(
        test_ds, batch_size=args.batch_size, mode="test",
        num_workers=args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    aux_enabled = args.aux_pay0_lambda > 0
    model = TabularTransformer(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        ffn_dropout=args.ffn_dropout,
        residual_dropout=args.residual_dropout,
        classification_dropout=args.classification_dropout,
        pool=args.pool,
        use_temporal_pos=args.use_temporal_pos,
        temporal_decay_mode=args.temporal_decay_mode,
        feature_group_bias_mode=args.feature_group_bias_mode,
        aux_pay0=aux_enabled,
        cat_vocab_sizes={
            feat: info["n_categories"]
            for feat, info in meta["categorical_features"].items()
        },
    ).to(device)
    n_params = model.count_parameters()
    logger.info("Model: %s params — %s", format_parameter_count(n_params), str(model)[:120])

    # Optional MTLM fine-tuning starting point (Plan §8.5.5).
    pretrained = args.pretrained_encoder is not None
    if pretrained:
        model.load_pretrained_encoder(
            args.pretrained_encoder,
            strict=False,
            trust_source=args.trust_checkpoint,
            map_location=device,
        )

    # ── Losses ────────────────────────────────────────────────────────────
    y_train = train_ds.tensors["labels"]
    primary_loss_fn = build_primary_loss(args, y_train).to(device)
    aux_loss_fn: Optional[nn.Module] = None
    if aux_enabled:
        aux_loss_fn = nn.CrossEntropyLoss().to(device)
        logger.info(
            "Multi-task objective (N5): λ=%.2f on PAY_0 CE (11 classes)",
            args.aux_pay0_lambda,
        )

    # ── Optimiser + schedule ──────────────────────────────────────────────
    optimizer = build_optimizer(model, args, pretrained=pretrained)
    steps_per_epoch = len(train_loader)
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = int(args.warmup_frac * total_steps)
    scheduler = build_cosine_warmup_schedule(
        optimizer, warmup_steps=warmup_steps, total_steps=total_steps,
        min_lr_frac=args.min_lr_frac,
    )

    early = EarlyStopping(patience=args.patience, mode="max", min_delta=args.min_delta)

    # ── Config snapshot ───────────────────────────────────────────────────
    config = {
        **vars(args),
        "param_count": n_params,
        "device":       str(device),
        "device_desc":  describe_device(device),
        "git_sha":      _git_sha(),
        "torch_version": torch.__version__,
        "total_steps":  total_steps,
        "warmup_steps": warmup_steps,
        "train_size":   len(train_ds),
        "val_size":     len(val_ds),
        "test_size":    len(test_ds),
        "pos_weight":   float(compute_pos_weight(y_train)),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    logger.info("Config snapshot → %s", output_dir / "config.json")

    # ── Training loop ─────────────────────────────────────────────────────
    log_rows: List[Dict[str, Any]] = []
    start = time.perf_counter()
    logger.info("Starting training: %d epochs (≤ %d steps, warmup %d steps).",
                args.epochs, total_steps, warmup_steps)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_stats = train_one_epoch(
            model, train_loader, optimizer, scheduler, primary_loss_fn, device,
            grad_clip=args.grad_clip,
            aux_loss_fn=aux_loss_fn,
            aux_lambda=args.aux_pay0_lambda,
        )
        val = evaluate_on_loader(model, val_loader, device)
        val_metrics = val["metrics"]
        elapsed = time.perf_counter() - epoch_start

        log_row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[-1]["lr"],  # head LR when two-group, else the only group
            **train_stats,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "epoch_s": elapsed,
        }
        log_rows.append(log_row)

        if epoch == 1 or epoch % args.log_every == 0:
            logger.info(
                "Epoch %3d | train_loss=%.4f | val_auc_roc=%.4f | "
                "val_auc_pr=%.4f | val_f1=%.4f | val_ece=%.4f | lr=%.2e | "
                "grad_norm=%.2f | %.1fs",
                epoch, log_row["train_loss"], val_metrics["auc_roc"],
                val_metrics["auc_pr"], val_metrics["f1"], val_metrics["ece"],
                log_row["lr"], log_row["grad_norm_mean"], elapsed,
            )

        if early.step(val_metrics["auc_roc"], state=model.state_dict()):
            logger.info(
                "Early stopping at epoch %d — best val AUC-ROC %.4f at epoch %d",
                epoch, early.best_score, early.best_epoch,
            )
            break

    total_elapsed = time.perf_counter() - start
    logger.info("Training complete in %.1fs (%d epochs).", total_elapsed, epoch)

    # Restore best-model weights.
    if early.best_state is not None:
        model.load_state_dict(early.best_state)
        logger.info("Restored best-epoch weights (epoch %d, val AUC-ROC %.4f).",
                    early.best_epoch, early.best_score)

    # ── Write training log ────────────────────────────────────────────────
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(output_dir / "train_log.csv", index=False)
    logger.info("Training log → %s", output_dir / "train_log.csv")

    # ── Final test evaluation ─────────────────────────────────────────────
    collect_attn = not (args.no_save_attn or args.smoke_test)
    test = evaluate_on_loader(model, test_loader, device, collect_attn=collect_attn)
    test_metrics = test["metrics"]
    logger.info(
        "Test metrics @ τ=0.5: AUC-ROC %.4f | AUC-PR %.4f | F1 %.4f | "
        "ECE %.4f | Brier %.4f",
        test_metrics["auc_roc"], test_metrics["auc_pr"], test_metrics["f1"],
        test_metrics["ece"], test_metrics["brier"],
    )

    # Supplementary threshold sweep — helpful for evaluate.py downstream.
    threshold_sweep = {
        f"tau_{t:.2f}": compute_classification_metrics(
            test["y_true"], test["y_prob"], threshold=t, prefix="",
        )
        for t in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60)
    }

    test_payload = {
        "threshold": 0.5,
        "metrics": test_metrics,
        "threshold_sweep": threshold_sweep,
        "epochs_trained": epoch,
        "best_epoch": early.best_epoch,
        "best_val_auc_roc": early.best_score,
        "training_seconds": total_elapsed,
        "param_count": n_params,
    }
    (output_dir / "test_metrics.json").write_text(
        json.dumps(test_payload, indent=2, default=float)
    )

    if not args.no_save_predictions:
        y_pred = (test["y_prob"] >= 0.5).astype(np.int64)
        np.savez_compressed(
            output_dir / "test_predictions.npz",
            y_true=test["y_true"].astype(np.int64),
            y_prob=test["y_prob"].astype(np.float32),
            y_pred=y_pred,
        )

    if collect_attn and "attn_weights" in test:
        np.savez_compressed(
            output_dir / "test_attn_weights.npz",
            **{f"layer_{i}": a.astype(np.float32)
               for i, a in enumerate(test["attn_weights"])},
        )

    # ── Checkpoint ────────────────────────────────────────────────────────
    save_checkpoint(
        output_dir / "best.pt",
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=build_checkpoint_metadata(
            seed=args.seed,
            epoch=early.best_epoch,
            step=(early.best_epoch or 0) * steps_per_epoch,
            extra={
                "role": "supervised-finetune" if pretrained else "supervised-from-scratch",
                "best_val_auc_roc": early.best_score,
                "test_metrics": test_metrics,
                "config": {
                    k: v for k, v in vars(args).items()
                    if isinstance(v, (int, float, str, bool, type(None)))
                },
            },
        ),
    )

    logger.info("All artefacts saved to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
