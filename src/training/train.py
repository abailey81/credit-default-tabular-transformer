"""Supervised training entry point for :class:`TabularTransformer`.

Primary public API
------------------
* :func:`main` — CLI entry point, ``python -m src.training.train``.
* :func:`build_optimizer` — AdamW builder with optional two-group LR split.
* :func:`build_cosine_warmup_schedule` — linear-warmup → cosine-decay.
* :func:`train_one_epoch`, :func:`evaluate_on_loader` — re-usable loop
  primitives (imported by ablation scripts and the Colab notebook).
* :func:`compute_classification_metrics`, :func:`compute_ece` — metric
  helpers shared with :mod:`src.baselines.rf_predictions`.

Design choices
--------------
1. **AdamW over Adam** (Loshchilov & Hutter 2019). Decoupled weight decay
   matters on transformers because the additive-to-loss L2 form of vanilla
   Adam gets rescaled by the adaptive step, which effectively disables
   regularisation on dimensions with large running gradients — exactly the
   attention-heavy dimensions we most want to keep in check.

2. **Cosine warmup** (Loshchilov & Hutter 2017 / widely adopted in Vaswani+
   2017). 10 % linear warmup avoids the early-training divergence that AdamW
   occasionally hits on transformers with LayerNorm, then cosine decay to a
   1 % floor keeps the final-phase LR small enough to settle without fully
   zeroing out progress.

3. **Two-group LR on fine-tune**. See :func:`build_optimizer` — the
   freshly-initialised classification head needs the full peak LR while the
   pretrained encoder benefits from a lower LR to avoid catastrophic
   forgetting of the MTLM representations. We default to
   ``encoder_lr_ratio=0.2``.

4. **Early stop on val AUC-ROC** (not loss). AUC-ROC is threshold-agnostic
   and correlates better with the downstream tasks than a class-weighted
   loss that can rise for benign reasons (e.g. calibration shift).

Non-obvious dependency: this module imports from :mod:`src.models.model`
which itself depends on the embedding + transformer stack — do **not** move
train.py higher in the import graph, circular imports will result."""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

from ..models.model import TabularTransformer
from ..tokenization.tokenizer import (
    CreditDefaultDataset,
    build_categorical_vocab,
)
from .dataset import make_loader
from .losses import (
    FocalLoss,
    LabelSmoothingBCELoss,
    WeightedBCELoss,
    balanced_alpha,
    compute_pos_weight,
)
from .utils import (
    EarlyStopping,
    build_checkpoint_metadata,
    configure_logging,
    describe_device,
    format_parameter_count,
    get_device,
    save_checkpoint,
    set_deterministic,
)

# repo root for locating data/ and results/ from any CWD
_REPO = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger(__name__)

ECE_N_BINS = 15


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Supervised training loop for the TabularTransformer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("reproducibility / output")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write artefacts. Default: results/transformer/seed_{seed}.",
    )
    g.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    g.add_argument(
        "--determinism",
        action="store_true",
        help="Enable torch.use_deterministic_algorithms(warn_only=True).",
    )

    g = p.add_argument_group("model architecture")
    g.add_argument("--d-model", type=int, default=32)
    g.add_argument("--n-heads", type=int, default=4)
    g.add_argument("--n-layers", type=int, default=2)
    g.add_argument(
        "--d-ff", type=int, default=None, help="FFN hidden dim. Defaults to 4 * d_model."
    )
    g.add_argument("--pool", choices=["cls", "mean", "max"], default="cls")
    g.add_argument("--use-temporal-pos", action="store_true")
    g.add_argument(
        "--temporal-decay-mode",
        choices=["off", "scalar", "per_head"],
        default="off",
    )
    g.add_argument(
        "--feature-group-bias-mode",
        choices=["off", "scalar", "per_head"],
        default="off",
        help="Feature-group attention bias across CLS / demographic / PAY / " "BILL_AMT / PAY_AMT.",
    )

    g = p.add_argument_group("regularisation")
    g.add_argument("--dropout", type=float, default=0.1)
    g.add_argument("--attn-dropout", type=float, default=None)
    g.add_argument("--ffn-dropout", type=float, default=None)
    g.add_argument("--residual-dropout", type=float, default=None)
    g.add_argument("--classification-dropout", type=float, default=0.1)

    g = p.add_argument_group("loss")
    g.add_argument("--loss", choices=["focal", "wbce", "label-smoothing"], default="focal")
    g.add_argument("--focal-gamma", type=float, default=2.0)
    g.add_argument(
        "--focal-alpha",
        type=str,
        default="balanced",
        help="'balanced' / scalar alpha_pos in [0, 1] / tuple '(alpha_pos, alpha_neg)' / 'none'.",
    )
    g.add_argument("--label-smoothing-eps", type=float, default=0.05)

    g.add_argument(
        "--aux-pay0-lambda",
        type=float,
        default=0.0,
        help="Joint-loss weight on the PAY_0 forecast auxiliary head. "
        "0 disables the aux head entirely.",
    )

    g = p.add_argument_group("optimisation")
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--weight-decay", type=float, default=1e-5)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--beta1", type=float, default=0.9)
    g.add_argument("--beta2", type=float, default=0.999)
    g.add_argument("--eps", type=float, default=1e-8)
    g.add_argument(
        "--warmup-frac",
        type=float,
        default=0.10,
        help="Fraction of total steps used for linear warmup.",
    )
    g.add_argument(
        "--min-lr-frac", type=float, default=0.01, help="Cosine floor as a fraction of peak LR."
    )

    g = p.add_argument_group("training schedule")
    g.add_argument("--epochs", type=int, default=200)
    g.add_argument(
        "--patience", type=int, default=20, help="Early stopping patience on val AUC-ROC."
    )
    g.add_argument("--min-delta", type=float, default=1e-4)
    g.add_argument("--batch-size", type=int, default=256)
    g.add_argument(
        "--stratified-batches", action="store_true", help="Every batch contains ~22.1%% positives."
    )
    g.add_argument("--num-workers", type=int, default=0)

    g = p.add_argument_group("fine-tuning from MTLM pretraining")
    g.add_argument(
        "--pretrained-encoder",
        type=str,
        default=None,
        help="Path to a MTLM pretrain .pt artefact. Loaded strict=False "
        "so the classification head stays at fresh init.",
    )
    g.add_argument(
        "--encoder-lr-ratio",
        type=float,
        default=0.2,
        help="Pretrained encoder LR = encoder_lr_ratio * --lr.",
    )
    g.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Only pass for checkpoints YOU produced. Enables the "
        "full pickle load path of torch.load; default is weights-only.",
    )

    g = p.add_argument_group("data scaling")
    g.add_argument(
        "--data-frac",
        type=float,
        default=1.0,
        help="Fraction of the training split to use (scaling-curve ablation).",
    )

    g = p.add_argument_group("artefact control")
    g.add_argument(
        "--no-save-attn",
        action="store_true",
        help="Skip writing test_attn_weights.npz (~30 MB per run).",
    )
    g.add_argument("--no-save-predictions", action="store_true")
    g.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print a log line every N epochs; the CSV log is per-epoch.",
    )

    g = p.add_argument_group("debug")
    g.add_argument(
        "--smoke-test", action="store_true", help="Run 2 epochs on ~500 training rows then exit."
    )

    return p


def _git_sha(default: str = "unknown") -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
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
    """parse --focal-alpha into whatever FocalLoss wants."""
    spec = spec.strip().lower()
    if spec == "balanced":
        return "balanced"
    if spec == "none":
        return None
    if spec.startswith("(") and spec.endswith(")"):
        spec = spec[1:-1]
    if "," in spec:
        parts = [float(x.strip()) for x in spec.split(",")]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"--focal-alpha tuple must have 2 values, got {parts}")
        return tuple(parts)
    try:
        return float(spec)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--focal-alpha must be 'balanced', 'none', a scalar in [0,1], or "
            f"a tuple '(alpha_pos, alpha_neg)'; got {spec!r}"
        ) from exc


def _load_splits(
    seed: int,
    data_frac: float,
    smoke_test: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    meta_path = _REPO / "data/processed/feature_metadata.json"
    paths = {
        "train": _REPO / "data/processed/train_scaled.csv",
        "val": _REPO / "data/processed/val_scaled.csv",
        "test": _REPO / "data/processed/test_scaled.csv",
    }
    missing = [str(p) for p in (meta_path, *paths.values()) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing preprocessing outputs:\n  "
            + "\n  ".join(missing)
            + "\nRun `poetry run python scripts/run_pipeline.py --preprocess-only`."
        )

    meta = json.loads(meta_path.read_text())
    train_df = pd.read_csv(paths["train"])
    val_df = pd.read_csv(paths["val"])
    test_df = pd.read_csv(paths["test"])

    def _stratified_sample(
        df: pd.DataFrame, *, n: Optional[int] = None, frac: Optional[float] = None
    ) -> pd.DataFrame:
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
        # need enough positives for AUC to be defined
        train_df = _stratified_sample(train_df, n=250)
        val_df = _stratified_sample(val_df, n=100)
        test_df = _stratified_sample(test_df, n=100)
    elif data_frac < 1.0:
        train_df = _stratified_sample(train_df, frac=data_frac)
        logger.info(
            "Data-frac ablation: reduced train to %.0f%% -> %d rows", data_frac * 100, len(train_df)
        )

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


def build_cosine_warmup_schedule(
    optim: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_frac: float = 0.01,
) -> LambdaLR:
    """Linear-warmup + cosine-decay LR schedule as a :class:`LambdaLR`.

    The schedule is::

        step < warmup_steps        : lr = peak · (step / warmup_steps)
        warmup_steps ≤ step < T    : lr = peak · (ε + (1-ε) · ½(1+cos(π·p)))

    where ``T = total_steps``, ``p = (step - warmup_steps) / (T - warmup_steps)``,
    and ``ε = min_lr_frac``. At ``step == T`` the multiplier equals exactly
    ``min_lr_frac``; past T it stays pinned there (``progress`` is clamped).

    Parameters
    ----------
    optim
        Optimiser whose param-group LRs are treated as the *peak* values.
        Works transparently for the two-group fine-tune path because
        :class:`LambdaLR` applies the same multiplier to every group.
    warmup_steps
        Number of linear-warmup steps. Clamped to ≥ 0; 0 disables warmup.
    total_steps
        Total scheduled steps (typically ``epochs * steps_per_epoch``).
    min_lr_frac
        Cosine floor as a fraction of peak LR. 0.01 is a conservative default
        that keeps the late-training LR non-trivial so fine adjustments still
        happen past epoch ~80 % without fully flattening out.

    Raises
    ------
    ValueError
        If ``total_steps`` is non-positive (would divide-by-zero on progress).
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")
    warmup_steps = max(0, int(warmup_steps))
    # Clamp to a valid probability; silently correct bad input rather than
    # bounce here — this runs inside every step.
    min_lr_frac = float(max(0.0, min(1.0, min_lr_frac)))

    def lr_lambda(step: int) -> float:
        # Warmup phase: linear ramp 0 → 1.
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        # Cosine phase: denom guarded against zero when warmup==total.
        denom = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / denom))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_frac + (1.0 - min_lr_frac) * cosine

    return LambdaLR(optim, lr_lambda)


def build_primary_loss(args: argparse.Namespace, y_train: torch.Tensor) -> nn.Module:
    """Instantiate the primary classification loss dispatched by ``--loss``.

    ``pos_weight`` and ``"balanced"`` alpha are resolved from the actual
    training labels (not a hard-coded constant) so this stays correct under
    ``--data-frac`` scaling ablations.
    """
    if args.loss == "wbce":
        return WeightedBCELoss(pos_weight=compute_pos_weight(y_train))

    if args.loss == "label-smoothing":
        return LabelSmoothingBCELoss(
            epsilon=args.label_smoothing_eps,
            pos_weight=compute_pos_weight(y_train),
        )

    # Focal path — resolve "balanced" eagerly so the fitted (α_pos, α_neg) is
    # visible in the config snapshot rather than being deferred to the first
    # batch (which FocalLoss would do internally).
    alpha = _resolve_focal_alpha(args.focal_alpha)
    if alpha == "balanced":
        alpha = balanced_alpha(y_train)
    return FocalLoss(gamma=args.focal_gamma, alpha=alpha)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = ECE_N_BINS) -> float:
    """Expected Calibration Error with equal-width probability bins.

    ECE is the weighted mean of ``|accuracy − confidence|`` across N bins::

        ECE = Σ_b (|B_b| / N) · |acc(B_b) − conf(B_b)|

    where ``B_b`` is the set of predictions whose probability falls into
    bin ``b``. We use the 15-bin default from Guo+ 2017 (see ``ECE_N_BINS``).

    The last bin is closed on the right (``[lo, hi]``) so predictions with
    ``p_prob == 1.0`` are still counted — a common off-by-one gotcha.
    Returns NaN on an empty input rather than raising.
    """
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
    """Build the eight-metric classification report used across the project.

    Parameters
    ----------
    y_true
        1-D int array of 0/1 labels, shape ``(N,)``.
    y_prob
        1-D float array of predicted default probabilities, shape ``(N,)``.
    threshold
        Decision threshold for the rate metrics (F1/precision/recall/acc).
        The ranking metrics (AUC-ROC / AUC-PR / Brier / ECE) are
        threshold-agnostic and ignore this parameter.
    prefix
        Optional key prefix (e.g. ``"val_"``). Lets callers merge multiple
        splits' metrics into a single flat dict without collisions.

    Returns
    -------
    dict
        Keys ``{prefix}auc_roc / auc_pr / f1 / accuracy / precision / recall /
        brier / ece`` — all float. Failed metrics (e.g. AUC with one class
        present in ``y_true``) are returned as NaN rather than raising so the
        training log survives smoke tests with tiny splits.
    """
    y_true_int = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        f"{prefix}auc_roc": _safe_metric(roc_auc_score, y_true_int, y_prob),
        f"{prefix}auc_pr": _safe_metric(average_precision_score, y_true_int, y_prob),
        f"{prefix}f1": _safe_metric(f1_score, y_true_int, y_pred, zero_division=0),
        f"{prefix}accuracy": _safe_metric(accuracy_score, y_true_int, y_pred),
        f"{prefix}precision": _safe_metric(precision_score, y_true_int, y_pred, zero_division=0),
        f"{prefix}recall": _safe_metric(recall_score, y_true_int, y_pred, zero_division=0),
        f"{prefix}brier": _safe_metric(brier_score_loss, y_true_int, y_prob),
        f"{prefix}ece": compute_ece(y_true_int, y_prob),
    }


@torch.no_grad()
def evaluate_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    collect_attn: bool = False,
) -> Dict[str, Any]:
    """Run one pass of a model over a loader and return predictions + metrics.

    Parameters
    ----------
    model
        Transformer module; MUST be callable as ``model(batch, return_attn=…)``
        (i.e. our :class:`TabularTransformer` API, not an arbitrary nn.Module).
    loader
        Any loader yielding batches compatible with :func:`default_collate`.
    device
        Target device — batches are moved via :func:`_to_device`.
    collect_attn
        When True, also capture per-layer attention weights (shape
        ``(n_layers, N, n_heads, seq_len, seq_len)`` after concatenation).
        Disabled by default because attention tensors are ~30 MB / layer for
        the 6 000-row test split and we don't want them in every eval.

    Returns
    -------
    dict
        ``{"y_true": np.ndarray, "y_prob": np.ndarray, "metrics": dict,
        ["attn_weights": List[np.ndarray]]}``.

    Notes
    -----
    The decorator invariant: :func:`torch.no_grad` + ``model.eval()``. The
    caller is responsible for calling ``model.train()`` afterwards if training
    will resume — otherwise dropout and any running-stat modules stay in eval
    mode and silently change the loss dynamics.
    """
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
    """Run one SGD epoch and return per-epoch diagnostics.

    Composite loss
    --------------
    When ``aux_lambda > 0`` and the model's forward output contains
    ``aux_pay0_logits`` (the N-novel PAY_0-forecasting auxiliary head), the
    optimised objective is::

        L = L_primary + λ · L_aux_pay0

    The aux head is an 11-way cross-entropy on the shifted ``pay_raw[:, 0]``
    (shifted because the raw values live in ``[-2, 8]`` but embedding tables
    want non-negative indices). λ = 0 disables the branch cleanly — the aux
    tensor is never even allocated on the non-aux path.

    Gradient clipping
    -----------------
    ``clip_grad_norm_(grad_clip)`` rescales the concatenated parameter
    gradients when their L2 norm exceeds ``grad_clip``. This is a standard
    transformer-stability guard (Pascanu+ 2013 for the general case; nearly
    universal on transformer training since Vaswani+ 2017). We keep the
    default at 1.0 — high enough that healthy epochs rarely clip, low enough
    to prevent the rare 10×-spike from destabilising AdamW's second-moment
    estimates. ``grad_norm_{mean,max}`` are logged so regressions are visible.

    Scheduler ticking
    -----------------
    The scheduler is stepped **per batch** (not per epoch). This is intentional:
    the warmup+cosine schedule is defined in units of training steps so the
    100 steps of warmup happen at the start of epoch 1, not spread over 10 %
    of the run.

    Returns
    -------
    dict
        ``train_loss`` (mean primary loss), ``train_aux_loss`` (NaN when aux
        disabled), ``grad_norm_{mean,max}``. The returned aux loss is the
        pre-λ value, so downstream diagnostics can separate "aux is
        converging" from "aux is being weighted down".
    """
    # Put modules with train-only behaviour (dropout, BN if any) in train mode;
    # MUST be mirrored by model.eval() in evaluate_on_loader.
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
            # pay_raw stored as shifted-to-non-negative ints in [0, 10];
            # column 0 is PAY_0 (the most recent repayment-status token).
            target_pay0 = batch["pay_raw"][:, 0].long()
            loss_aux = aux_loss_fn(out["aux_pay0_logits"], target_pay0)
            loss = loss_primary + aux_lambda * loss_aux
            aux_losses.append(float(loss_aux.item()))

        # set_to_none=True: faster than zeroing, and clarifies ownership —
        # a NoneType grad means "never populated this step".
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Always clip before opt.step so AdamW's running moments see the
        # clipped gradient, not the raw one.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        grad_norms.append(float(grad_norm))
        optimizer.step()
        if scheduler is not None:
            # Per-step tick — see docstring note on schedule units.
            scheduler.step()

        primary_losses.append(float(loss_primary.item()))

    return {
        "train_loss": float(np.mean(primary_losses)),
        "train_aux_loss": float(np.mean(aux_losses)) if aux_losses else float("nan"),
        "grad_norm_mean": float(np.mean(grad_norms)),
        "grad_norm_max": float(np.max(grad_norms)),
    }


def build_optimizer(
    model: TabularTransformer,
    args: argparse.Namespace,
    pretrained: bool,
) -> AdamW:
    """Construct the AdamW optimiser, optionally with a two-group LR split.

    From-scratch path
    -----------------
    Single param group at ``args.lr``. Standard AdamW
    (Loshchilov & Hutter 2019) with ``(β1, β2, ε)`` from the CLI defaults
    and ``weight_decay`` applied decoupled from the gradient step.

    Fine-tune path (``pretrained=True``)
    ------------------------------------
    Two param groups:

    * **Encoder** — pretrained by MTLM. LR =
      ``args.lr * args.encoder_lr_ratio`` (default 0.2, so 6e-5 at peak).
      Motivation: the MTLM encoder has already learned useful token /
      positional representations over 50 epochs of masked reconstruction;
      cranking the full peak LR through it risks catastrophic forgetting —
      the head-side gradients push the encoder off the MTLM basin before the
      newly-initialised head has even established its own directions.
    * **Head** — freshly initialised at each ``train.py`` run. LR = full
      ``args.lr``. The head has no prior state worth preserving and needs
      the full peak LR for the warmup phase to actually make progress
      within the ~15 min training budget.

    The per-group ``lr`` values here are the **peak** values the
    :class:`LambdaLR` will multiply against — both groups are scaled by the
    same warmup/cosine multiplier, so the encoder's LR trajectory tracks the
    head's at a constant 0.2× offset.

    Parameters
    ----------
    model
        TabularTransformer; must expose ``get_head_params`` /
        ``get_encoder_params`` when ``pretrained=True``.
    args
        Parsed CLI namespace — supplies ``lr``, ``betas``, ``eps``,
        ``weight_decay``, ``encoder_lr_ratio``.
    pretrained
        Selects the optimiser topology, NOT whether weights are loaded (the
        caller does the loading separately).
    """
    betas = (args.beta1, args.beta2)
    if not pretrained:
        return AdamW(
            model.parameters(),
            lr=args.lr,
            betas=betas,
            eps=args.eps,
            weight_decay=args.weight_decay,
        )

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
            # Encoder group — pretrained, down-weighted to avoid forgetting.
            {
                "params": encoder_params,
                "lr": args.lr * args.encoder_lr_ratio,
                "weight_decay": args.weight_decay,
            },
            # Head group — fresh init, full peak LR.
            {"params": head_params, "lr": args.lr, "weight_decay": args.weight_decay},
        ],
        betas=betas,
        eps=args.eps,
    )


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: parse args → build data/model/opt → train → persist.

    Exits with code 0 on success. Side effects: writes to ``output_dir``
    (``config.json``, ``train_log.csv``, ``{train,val,test}_metrics.json``,
    ``{train,val,test}_predictions.npz``, ``best.pt`` + sidecars, optionally
    ``test_attn_weights.npz``). See the argparser for the full knob list.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    configure_logging()
    # Seed EVERYTHING before we build the model — weight init depends on it.
    set_deterministic(args.seed, warn_only=not args.determinism)
    device = get_device(args.device)

    logger.info("=" * 72)
    logger.info("TabularTransformer supervised training - seed %d", args.seed)
    logger.info("Device: %s", describe_device(device))
    logger.info("=" * 72)

    output_dir = _resolve_output_dir(args)

    # data
    train_df, val_df, test_df, meta = _load_splits(
        seed=args.seed,
        data_frac=args.data_frac,
        smoke_test=args.smoke_test,
    )
    cat_vocab = build_categorical_vocab(meta)
    train_ds = CreditDefaultDataset(train_df, cat_vocab, verbose=False)
    val_ds = CreditDefaultDataset(val_df, cat_vocab, verbose=False)
    test_ds = CreditDefaultDataset(test_df, cat_vocab, verbose=False)
    logger.info(
        "Splits: train %d (%.1f%% pos) | val %d (%.1f%% pos) | test %d (%.1f%% pos)",
        len(train_ds),
        float(train_ds.tensors["labels"].mean()) * 100,
        len(val_ds),
        float(val_ds.tensors["labels"].mean()) * 100,
        len(test_ds),
        float(test_ds.tensors["labels"].mean()) * 100,
    )

    train_loader = make_loader(
        train_ds,
        batch_size=args.batch_size,
        mode="train",
        stratified=args.stratified_batches,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        seed=args.seed,
    )
    val_loader = make_loader(
        val_ds,
        batch_size=args.batch_size,
        mode="val",
        num_workers=args.num_workers,
    )
    test_loader = make_loader(
        test_ds,
        batch_size=args.batch_size,
        mode="test",
        num_workers=args.num_workers,
    )

    # model
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
            feat: info["n_categories"] for feat, info in meta["categorical_features"].items()
        },
    ).to(device)
    n_params = model.count_parameters()
    logger.info("Model: %s params - %s", format_parameter_count(n_params), str(model)[:120])

    pretrained = args.pretrained_encoder is not None
    if pretrained:
        model.load_pretrained_encoder(
            args.pretrained_encoder,
            strict=False,
            trust_source=args.trust_checkpoint,
            map_location=device,
        )

    # losses
    y_train = train_ds.tensors["labels"]
    primary_loss_fn = build_primary_loss(args, y_train).to(device)
    aux_loss_fn: Optional[nn.Module] = None
    if aux_enabled:
        aux_loss_fn = nn.CrossEntropyLoss().to(device)
        logger.info(
            "Multi-task objective: lambda=%.2f on PAY_0 CE (11 classes)",
            args.aux_pay0_lambda,
        )

    # opt + sched
    optimizer = build_optimizer(model, args, pretrained=pretrained)
    steps_per_epoch = len(train_loader)
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = int(args.warmup_frac * total_steps)
    scheduler = build_cosine_warmup_schedule(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_frac=args.min_lr_frac,
    )

    early = EarlyStopping(patience=args.patience, mode="max", min_delta=args.min_delta)

    # config snapshot
    config = {
        **vars(args),
        "param_count": n_params,
        "device": str(device),
        "device_desc": describe_device(device),
        "git_sha": _git_sha(),
        "torch_version": torch.__version__,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "pos_weight": float(compute_pos_weight(y_train)),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    logger.info("Config snapshot -> %s", output_dir / "config.json")

    # train
    log_rows: List[Dict[str, Any]] = []
    start = time.perf_counter()
    logger.info(
        "Starting training: %d epochs (<= %d steps, warmup %d steps).",
        args.epochs,
        total_steps,
        warmup_steps,
    )

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            primary_loss_fn,
            device,
            grad_clip=args.grad_clip,
            aux_loss_fn=aux_loss_fn,
            aux_lambda=args.aux_pay0_lambda,
        )
        val = evaluate_on_loader(model, val_loader, device)
        val_metrics = val["metrics"]
        elapsed = time.perf_counter() - epoch_start

        log_row = {
            "epoch": epoch,
            # two-group → last group is head; single-group → only group
            "lr": optimizer.param_groups[-1]["lr"],
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
                epoch,
                log_row["train_loss"],
                val_metrics["auc_roc"],
                val_metrics["auc_pr"],
                val_metrics["f1"],
                val_metrics["ece"],
                log_row["lr"],
                log_row["grad_norm_mean"],
                elapsed,
            )

        if early.step(val_metrics["auc_roc"], state=model.state_dict()):
            logger.info(
                "Early stopping at epoch %d - best val AUC-ROC %.4f at epoch %d",
                epoch,
                early.best_score,
                early.best_epoch,
            )
            break

    total_elapsed = time.perf_counter() - start
    logger.info("Training complete in %.1fs (%d epochs).", total_elapsed, epoch)

    if early.best_state is not None:
        model.load_state_dict(early.best_state)
        logger.info(
            "Restored best-epoch weights (epoch %d, val AUC-ROC %.4f).",
            early.best_epoch,
            early.best_score,
        )

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(output_dir / "train_log.csv", index=False)
    logger.info("Training log -> %s", output_dir / "train_log.csv")

    # --- final eval on all splits with best-epoch weights ---
    # re-wrap train in val mode so numbers reflect fitted model, not mid-SGD
    final_train_loader = make_loader(
        train_ds,
        batch_size=args.batch_size,
        mode="val",
        num_workers=args.num_workers,
    )
    collect_attn = not (args.no_save_attn or args.smoke_test)
    train_eval = evaluate_on_loader(model, final_train_loader, device)
    val_eval = evaluate_on_loader(model, val_loader, device)
    test = evaluate_on_loader(model, test_loader, device, collect_attn=collect_attn)
    test_metrics = test["metrics"]

    for split, ev in (("Train", train_eval), ("Val", val_eval), ("Test", test)):
        m = ev["metrics"]
        logger.info(
            "%-5s metrics @ tau=0.5: AUC-ROC %.4f | AUC-PR %.4f | F1 %.4f | "
            "acc %.4f | ECE %.4f | Brier %.4f",
            split,
            m["auc_roc"],
            m["auc_pr"],
            m["f1"],
            m["accuracy"],
            m["ece"],
            m["brier"],
        )

    threshold_sweep = {
        f"tau_{t:.2f}": compute_classification_metrics(
            test["y_true"],
            test["y_prob"],
            threshold=t,
            prefix="",
        )
        for t in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60)
    }

    def _write_split(split: str, ev: Dict[str, Any], *, include_test_extras: bool) -> None:
        payload: Dict[str, Any] = {
            "split": split,
            "threshold": 0.5,
            "metrics": ev["metrics"],
        }
        if include_test_extras:
            payload.update(
                {
                    "threshold_sweep": threshold_sweep,
                    "epochs_trained": epoch,
                    "best_epoch": early.best_epoch,
                    "best_val_auc_roc": early.best_score,
                    "training_seconds": total_elapsed,
                    "param_count": n_params,
                }
            )
        (output_dir / f"{split}_metrics.json").write_text(
            json.dumps(payload, indent=2, default=float)
        )
        if not args.no_save_predictions:
            y_pred = (ev["y_prob"] >= 0.5).astype(np.int64)
            np.savez_compressed(
                output_dir / f"{split}_predictions.npz",
                y_true=ev["y_true"].astype(np.int64),
                y_prob=ev["y_prob"].astype(np.float32),
                y_pred=y_pred,
            )

    _write_split("train", train_eval, include_test_extras=False)
    _write_split("val", val_eval, include_test_extras=False)
    _write_split("test", test, include_test_extras=True)

    if collect_attn and "attn_weights" in test:
        np.savez_compressed(
            output_dir / "test_attn_weights.npz",
            **{f"layer_{i}": a.astype(np.float32) for i, a in enumerate(test["attn_weights"])},
        )

    # chkpt
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
                    k: v
                    for k, v in vars(args).items()
                    if isinstance(v, (int, float, str, bool, type(None)))
                },
            },
        ),
    )

    logger.info("All artefacts saved to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
