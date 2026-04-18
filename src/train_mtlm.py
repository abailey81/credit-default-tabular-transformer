"""MTLM self-supervised pretraining loop. Reconstructs masked tokens with
BERT 80/10/10 masking (done by tokenizer.MTLMCollator). Writes
encoder_pretrained.pt which train.py --pretrained-encoder consumes."""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve()
_SRC = _HERE.parent
_REPO = _HERE.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dataset import make_loader  # noqa: E402
from embedding import (  # noqa: E402
    FeatureEmbedding,
    N_FEATURE_GROUPS,
    build_group_assignment,
    build_temporal_layout,
)
from model import FULL_SEQ_LEN  # noqa: E402
from mtlm import MTLMHead, MTLMLossComponents, MTLMModel, mtlm_loss  # noqa: E402
from tokenizer import (  # noqa: E402
    CreditDefaultDataset,
    MTLMCollator,
    NUMERICAL_FEATURES,
    build_categorical_vocab,
)
from transformer import FeatureGroupBias, TemporalDecayBias, TransformerEncoder  # noqa: E402
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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MTLM self-supervised pretraining.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("reproducibility / output")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--output-dir", type=str, default=None,
                   help="Defaults to results/mtlm/run_{seed}.")
    g.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    g.add_argument("--determinism", action="store_true")

    # arch must mirror downstream TabularTransformer for drop-in fine-tune
    g = p.add_argument_group("model architecture (must match downstream fine-tune)")
    g.add_argument("--d-model", type=int, default=32)
    g.add_argument("--n-heads", type=int, default=4)
    g.add_argument("--n-layers", type=int, default=2)
    g.add_argument("--d-ff", type=int, default=None)
    g.add_argument("--use-temporal-pos", action="store_true")
    g.add_argument("--temporal-decay-mode",
                   choices=["off", "scalar", "per_head"], default="off")
    g.add_argument("--feature-group-bias-mode",
                   choices=["off", "scalar", "per_head"], default="off")
    g.add_argument("--dropout", type=float, default=0.1)
    g.add_argument("--attn-dropout", type=float, default=None)
    g.add_argument("--ffn-dropout", type=float, default=None)
    g.add_argument("--residual-dropout", type=float, default=None)
    g.add_argument("--mtlm-head-dropout", type=float, default=0.1)

    g = p.add_argument_group("MTLM masking")
    g.add_argument("--mask-prob", type=float, default=0.15,
                   help="Per-token selection probability.")
    g.add_argument("--min-mask", type=int, default=1,
                   help="Minimum masked tokens per row (>=1 so every row contributes).")
    g.add_argument("--max-mask", type=int, default=5,
                   help="Upper bound on masked tokens per row.")
    g.add_argument("--replace-with-mask", type=float, default=0.8,
                   help="Fraction of selected tokens replaced with [MASK].")
    g.add_argument("--replace-with-random", type=float, default=0.1,
                   help="Fraction replaced with a random valid value (BERT 80/10/10).")

    g.add_argument("--w-cat", type=float, default=1.0)
    g.add_argument("--w-pay", type=float, default=1.0)
    g.add_argument("--w-num", type=float, default=1.0)
    g.add_argument("--no-variance-normalise", action="store_true",
                   help="Disable variance-normalisation on numerical MSE.")

    g = p.add_argument_group("optimisation")
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--weight-decay", type=float, default=1e-5)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--beta1", type=float, default=0.9)
    g.add_argument("--beta2", type=float, default=0.999)
    g.add_argument("--eps", type=float, default=1e-8)
    g.add_argument("--warmup-frac", type=float, default=0.10)
    g.add_argument("--min-lr-frac", type=float, default=0.01)

    g = p.add_argument_group("training schedule")
    g.add_argument("--epochs", type=int, default=50)
    g.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience on held-out reconstruction loss.")
    g.add_argument("--min-delta", type=float, default=1e-4)
    g.add_argument("--batch-size", type=int, default=256)
    g.add_argument("--num-workers", type=int, default=0)
    g.add_argument("--log-every", type=int, default=1)
    g.add_argument("--smoke-test", action="store_true",
                   help="Run 2 epochs on ~500 training rows then exit.")

    return p


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
        out = _REPO / "results" / "mtlm" / f"run_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_splits(
    seed: int, smoke_test: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """train + val + meta. test is reserved for supervised eval."""
    meta_path = _REPO / "data/processed/feature_metadata.json"
    paths = {
        "train": _REPO / "data/processed/train_scaled.csv",
        "val":   _REPO / "data/processed/val_scaled.csv",
    }
    missing = [str(p) for p in (meta_path, *paths.values()) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing preprocessing outputs:\n  " + "\n  ".join(missing)
            + "\nRun `poetry run python run_pipeline.py --preprocess-only`."
        )
    meta = json.loads(meta_path.read_text())
    train_df = pd.read_csv(paths["train"])
    val_df = pd.read_csv(paths["val"])
    if smoke_test:
        train_df = train_df.sample(n=min(500, len(train_df)), random_state=seed).reset_index(drop=True)
        val_df = val_df.sample(n=min(200, len(val_df)), random_state=seed).reset_index(drop=True)
    return train_df, val_df, meta


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
    """linear warmup → cosine decay. same as train.build_cosine_warmup_schedule."""
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


def build_mtlm_model(
    args: argparse.Namespace,
    cat_vocab_sizes: Dict[str, int],
) -> MTLMModel:
    """embedding + encoder + MTLMHead. Arch mirrors TabularTransformer exactly."""
    # mask path must fire at pretrain time
    embedding = FeatureEmbedding(
        d_model=args.d_model,
        dropout=args.dropout,
        cat_vocab_sizes=cat_vocab_sizes,
        use_temporal_pos=args.use_temporal_pos,
        use_mask_token=True,
    )

    if args.temporal_decay_mode != "off":
        td = TemporalDecayBias(
            temporal_layout=build_temporal_layout(cls_offset=1),
            seq_len=FULL_SEQ_LEN,
            mode=args.temporal_decay_mode,
            n_heads=args.n_heads,
        )
    else:
        td = None

    if args.feature_group_bias_mode != "off":
        fg = FeatureGroupBias(
            group_assignment=build_group_assignment(cls_offset=1),
            n_groups=N_FEATURE_GROUPS,
            mode=args.feature_group_bias_mode,
            n_heads=args.n_heads,
        )
    else:
        fg = None

    encoder = TransformerEncoder(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        ffn_dropout=args.ffn_dropout,
        residual_dropout=args.residual_dropout,
        temporal_decay=td,
        feature_group_bias=fg,
    )

    mtlm_head = MTLMHead(
        d_model=args.d_model,
        cat_vocab_sizes=cat_vocab_sizes,
        dropout=args.mtlm_head_dropout,
    )

    return MTLMModel(embedding=embedding, encoder=encoder, mtlm_head=mtlm_head)


def train_one_epoch(
    model: MTLMModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    device: torch.device,
    *,
    grad_clip: float = 1.0,
    num_feature_variance: Optional[Dict[str, float]] = None,
    w_cat: float = 1.0,
    w_pay: float = 1.0,
    w_num: float = 1.0,
) -> Dict[str, float]:
    """one epoch → mean total/cat/pay/num loss + grad stats."""
    model.train()
    totals: List[float] = []
    cats: List[float] = []
    pays: List[float] = []
    nums: List[float] = []
    grad_norms: List[float] = []
    masked_counts: List[int] = []

    for batch in loader:
        batch = _to_device(batch, device)
        predictions = model(batch)
        loss, components = mtlm_loss(
            predictions=predictions,
            batch=batch,
            mask_positions=batch["mask_positions"],
            num_feature_variance=num_feature_variance,
            w_cat=w_cat, w_pay=w_pay, w_num=w_num,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        totals.append(components.total)
        cats.append(components.cat)
        pays.append(components.pay)
        nums.append(components.num)
        grad_norms.append(float(grad_norm))
        masked_counts.append(components.n_masked)

    return {
        "train_loss":     float(np.mean(totals)),
        "train_loss_cat": float(np.mean(cats)),
        "train_loss_pay": float(np.mean(pays)),
        "train_loss_num": float(np.mean(nums)),
        "grad_norm_mean": float(np.mean(grad_norms)),
        "grad_norm_max":  float(np.max(grad_norms)),
        "masked_mean":    float(np.mean(masked_counts)),
    }


@torch.no_grad()
def evaluate_on_loader(
    model: MTLMModel,
    loader: DataLoader,
    device: torch.device,
    *,
    num_feature_variance: Optional[Dict[str, float]] = None,
    w_cat: float = 1.0,
    w_pay: float = 1.0,
    w_num: float = 1.0,
) -> Dict[str, float]:
    """held-out reconstruction loss, mean over batches."""
    model.eval()
    totals: List[float] = []
    cats: List[float] = []
    pays: List[float] = []
    nums: List[float] = []

    for batch in loader:
        batch = _to_device(batch, device)
        predictions = model(batch)
        _, components = mtlm_loss(
            predictions=predictions,
            batch=batch,
            mask_positions=batch["mask_positions"],
            num_feature_variance=num_feature_variance,
            w_cat=w_cat, w_pay=w_pay, w_num=w_num,
        )
        totals.append(components.total)
        cats.append(components.cat)
        pays.append(components.pay)
        nums.append(components.num)

    return {
        "val_loss":     float(np.mean(totals)),
        "val_loss_cat": float(np.mean(cats)),
        "val_loss_pay": float(np.mean(pays)),
        "val_loss_num": float(np.mean(nums)),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    configure_logging()
    set_deterministic(args.seed, warn_only=not args.determinism)
    device = get_device(args.device)

    logger.info("=" * 72)
    logger.info("MTLM pretraining - seed %d", args.seed)
    logger.info("Device: %s", describe_device(device))
    logger.info("=" * 72)

    output_dir = _resolve_output_dir(args)

    # data
    train_df, val_df, meta = _load_splits(seed=args.seed, smoke_test=args.smoke_test)
    cat_vocab = build_categorical_vocab(meta)
    cat_vocab_sizes = {
        feat: info["n_categories"]
        for feat, info in meta["categorical_features"].items()
    }

    train_ds = CreditDefaultDataset(train_df, cat_vocab, verbose=False)
    val_ds = CreditDefaultDataset(val_df, cat_vocab, verbose=False)
    logger.info(
        "Splits: train %d (%.1f%% pos) | val %d (%.1f%% pos) | test reserved for supervised eval",
        len(train_ds), float(train_ds.tensors["labels"].mean()) * 100,
        len(val_ds),   float(val_ds.tensors["labels"].mean()) * 100,
    )

    # separate train/val collators → independent mask draws
    train_collator = MTLMCollator(
        mask_prob=args.mask_prob,
        replace_with_mask=args.replace_with_mask,
        replace_with_random=args.replace_with_random,
        min_mask_per_row=args.min_mask,
        max_mask_per_row=args.max_mask,
        seed=args.seed,
    )
    val_collator = MTLMCollator(
        mask_prob=args.mask_prob,
        replace_with_mask=args.replace_with_mask,
        replace_with_random=args.replace_with_random,
        min_mask_per_row=args.min_mask,
        max_mask_per_row=args.max_mask,
        seed=args.seed + 1,
    )

    train_loader = make_loader(
        train_ds, batch_size=args.batch_size, mode="mtlm",
        mtlm=train_collator,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        seed=args.seed,
    )
    val_loader = make_loader(
        val_ds, batch_size=args.batch_size, mode="mtlm",
        mtlm=val_collator,
        num_workers=args.num_workers,
        # keep drop_last=False so smoke tests don't silently yield 0 batches
        drop_last=False,
    )

    # model
    model = build_mtlm_model(args, cat_vocab_sizes).to(device)
    n_params = count_parameters(model)
    logger.info("Model: %s params - %s", format_parameter_count(n_params),
                str(model)[:120])

    if args.no_variance_normalise:
        num_feature_variance = None
    else:
        num_var = {}
        for i, feat in enumerate(NUMERICAL_FEATURES):
            vals = train_ds.tensors["num_values"][:, i].numpy()
            num_var[feat] = float(vals.var()) or 1.0
        num_feature_variance = num_var
        logger.info(
            "Variance-normalising numerical MSE (mean var = %.3f across 14 features)",
            float(np.mean(list(num_var.values()))),
        )

    # opt + sched
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = len(train_loader)
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = int(args.warmup_frac * total_steps)
    scheduler = build_cosine_warmup_schedule(
        optimizer, warmup_steps=warmup_steps, total_steps=total_steps,
        min_lr_frac=args.min_lr_frac,
    )

    early = EarlyStopping(patience=args.patience, mode="min",
                          min_delta=args.min_delta)

    # config snapshot
    config = {
        **vars(args),
        "param_count": n_params,
        "device":       str(device),
        "device_desc":  describe_device(device),
        "git_sha":      _git_sha(),
        "torch_version": torch.__version__,
        "total_steps":   total_steps,
        "warmup_steps":  warmup_steps,
        "train_size":    len(train_ds),
        "val_size":      len(val_ds),
        "variance_normalised": not args.no_variance_normalise,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    logger.info("Config snapshot -> %s", output_dir / "config.json")

    # train
    log_rows: List[Dict[str, Any]] = []
    start = time.perf_counter()
    logger.info(
        "Starting MTLM pretraining: %d epochs (<= %d steps, warmup %d steps, mask_prob=%.2f).",
        args.epochs, total_steps, warmup_steps, args.mask_prob,
    )

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_stats = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            grad_clip=args.grad_clip,
            num_feature_variance=num_feature_variance,
            w_cat=args.w_cat, w_pay=args.w_pay, w_num=args.w_num,
        )
        val_stats = evaluate_on_loader(
            model, val_loader, device,
            num_feature_variance=num_feature_variance,
            w_cat=args.w_cat, w_pay=args.w_pay, w_num=args.w_num,
        )
        elapsed = time.perf_counter() - epoch_start

        log_row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **train_stats,
            **val_stats,
            "epoch_s": elapsed,
        }
        log_rows.append(log_row)

        if epoch == 1 or epoch % args.log_every == 0:
            logger.info(
                "Epoch %3d | train=%.4f (cat=%.4f pay=%.4f num=%.4f) | "
                "val=%.4f | lr=%.2e | gnorm=%.2f | %.1fs",
                epoch, train_stats["train_loss"],
                train_stats["train_loss_cat"], train_stats["train_loss_pay"],
                train_stats["train_loss_num"], val_stats["val_loss"],
                log_row["lr"], train_stats["grad_norm_mean"], elapsed,
            )

        if early.step(val_stats["val_loss"], state=model.state_dict()):
            logger.info(
                "Early stopping at epoch %d - best val loss %.4f at epoch %d",
                epoch, early.best_score, early.best_epoch,
            )
            break

    total_elapsed = time.perf_counter() - start
    logger.info("MTLM pretraining complete in %.1fs (%d epochs).", total_elapsed, epoch)

    if early.best_state is not None:
        model.load_state_dict(early.best_state)
        logger.info("Restored best-epoch weights (epoch %d, val loss %.4f).",
                    early.best_epoch, early.best_score)

    pd.DataFrame(log_rows).to_csv(output_dir / "pretrain_log.csv", index=False)
    logger.info("Pretrain log -> %s", output_dir / "pretrain_log.csv")

    # full chkpt + encoder-only artefact
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
                "role": "mtlm-pretrain",
                "best_val_loss": early.best_score,
                "config": {
                    k: v for k, v in vars(args).items()
                    if isinstance(v, (int, float, str, bool, type(None)))
                },
            },
        ),
    )

    # embedding.* + encoder.* only — consumed by
    # TabularTransformer.load_pretrained_encoder
    encoder_only = model.encoder_state_dict()
    encoder_only_path = output_dir / "encoder_pretrained.pt"
    torch.save(encoder_only, encoder_only_path)
    logger.info(
        "Encoder-only state dict (%d tensors) -> %s",
        len(encoder_only), encoder_only_path,
    )
    logger.info(
        "Fine-tune via: python src/train.py --pretrained-encoder %s",
        encoder_only_path,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
