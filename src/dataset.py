"""
dataset.py — DataLoader construction helpers for the credit-default task.

This module does NOT define the Dataset itself — that stays in :mod:`tokenizer`
as :class:`tokenizer.CreditDefaultDataset` for historical reasons and to keep
all tokenisation concerns in one place. Instead, this module provides:

    * :class:`StratifiedBatchSampler` — a sampler that guarantees every batch
      contains the positive-class rate observed in the training set, rather
      than relying on random sampling to approximately hit it. Useful for
      small-batch training on imbalanced data (Plan §8.8).

    * :func:`make_loader` — one-liner DataLoader factory that chooses the
      right sampler, collate function, and DataLoader flags based on whether
      we're in supervised training, MTLM pretraining, or plain evaluation.

All components respect :mod:`utils.set_deterministic`: an explicit ``seed``
parameter drives a local :class:`torch.Generator` so that sampling is
reproducible independently of the global RNG state.

References: Plan §8.8 (data loading), §8.5 (MTLM).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from tokenizer import (
    CATEGORICAL_FEATURES,
    CreditDefaultDataset,  # re-exported for convenience
    MTLMCollator,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CreditDefaultDataset",
    "StratifiedBatchSampler",
    "default_collate",
    "make_loader",
]


# ──────────────────────────────────────────────────────────────────────────────
# Collate function (hoisted from tokenizer.MTLMCollator for reuse)
# ──────────────────────────────────────────────────────────────────────────────


def default_collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Stack per-item dicts from :class:`CreditDefaultDataset` into a batched dict.

    Identical in shape contract to
    ``torch.utils.data.default_collate`` but preserves the nested
    ``cat_indices`` dict layout that :class:`tokenizer.FeatureEmbedding` expects.
    """
    cat_indices = {
        feat: torch.stack([item["cat_indices"][feat] for item in batch])
        for feat in CATEGORICAL_FEATURES
    }
    return {
        "cat_indices":    cat_indices,
        "pay_state_ids":  torch.stack([item["pay_state_ids"] for item in batch]),
        "pay_severities": torch.stack([item["pay_severities"] for item in batch]),
        "num_values":     torch.stack([item["num_values"] for item in batch]),
        "label":          torch.stack([item["label"] for item in batch]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# StratifiedBatchSampler
# ──────────────────────────────────────────────────────────────────────────────


class StratifiedBatchSampler(Sampler[List[int]]):
    """
    Samples batches that preserve the overall positive-class rate.

    Motivation: with a 22.1% default rate and ``batch_size=256``, a naïvely
    random batch has a standard deviation of ~2.6% on the realised positive
    rate. That occasionally produces batches with < 17% or > 28% positives,
    which hurts gradient stability for class-weighted losses. Stratified
    sampling caps this variance by construction — every batch contains
    exactly ``round(batch_size * positive_rate)`` positive samples.

    The sampler:

    1. Partitions the dataset into positive and negative indices.
    2. On each epoch, reshuffles each partition (optionally with ``generator``).
    3. Yields batches formed by sampling ``k_pos`` from the positive pool and
       ``k_neg`` from the negative pool without replacement, where
       ``k_pos + k_neg == batch_size`` and the ratio matches the overall rate.

    If either pool is exhausted before the other, ``drop_last=True`` ignores
    the trailing partial batch; ``drop_last=False`` emits what's left
    (possibly with slightly different ratio).

    Parameters
    ----------
    labels : array-like of shape (N,)
        Binary class labels (0 or 1). Typically ``dataset.tensors["labels"]``
        or an equivalent 1-D tensor/ndarray.
    batch_size : int
        Total samples per batch.
    drop_last : bool
        If True, drop the final partial batch (standard PyTorch semantics).
    shuffle : bool
        If False, emit batches in a deterministic order (useful for val/test).
    generator : Optional[torch.Generator]
        Seeded generator for reproducible sampling.

    Yields
    ------
    List[int]
        A list of dataset indices of length ``batch_size`` (or less, on the
        final batch when ``drop_last=False``).
    """

    def __init__(
        self,
        labels: Sequence[int] | torch.Tensor | np.ndarray,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        # Normalise labels to a 1-D int64 tensor.
        if isinstance(labels, torch.Tensor):
            lbl = labels.detach().cpu().long().view(-1)
        else:
            lbl = torch.as_tensor(np.asarray(labels).reshape(-1), dtype=torch.long)
        self._labels = lbl

        pos_idx = (lbl == 1).nonzero(as_tuple=True)[0]
        neg_idx = (lbl == 0).nonzero(as_tuple=True)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            raise ValueError(
                "StratifiedBatchSampler requires both classes to be present; "
                f"got pos={len(pos_idx)}, neg={len(neg_idx)}"
            )
        self._pos_indices = pos_idx
        self._neg_indices = neg_idx

        self._pos_rate = float(len(pos_idx) / len(lbl))
        # Compute how many positives / negatives per batch.
        self._k_pos = max(1, int(round(batch_size * self._pos_rate)))
        self._k_neg = batch_size - self._k_pos
        if self._k_neg <= 0:
            raise ValueError(
                f"Derived k_neg is non-positive (batch_size={batch_size}, "
                f"pos_rate={self._pos_rate:.3f}). Increase batch_size."
            )

        self._batch_size = batch_size
        self._drop_last = drop_last
        self._shuffle = shuffle
        self._generator = generator

        # How many full batches we can produce in one epoch — bounded by
        # whichever class runs out first.
        n_pos_batches = len(pos_idx) // self._k_pos
        n_neg_batches = len(neg_idx) // self._k_neg
        self._n_batches_full = min(n_pos_batches, n_neg_batches)

    # --------------------------------------------------------------- utils

    def _shuffled(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a shuffled view of ``tensor`` using the local generator."""
        if not self._shuffle:
            return tensor
        perm = torch.randperm(len(tensor), generator=self._generator)
        return tensor[perm]

    # --------------------------------------------------------------- iter

    def __iter__(self) -> Iterator[List[int]]:
        pos = self._shuffled(self._pos_indices)
        neg = self._shuffled(self._neg_indices)

        # Emit as many stratified batches as both pools can support.
        for b in range(self._n_batches_full):
            pos_chunk = pos[b * self._k_pos : (b + 1) * self._k_pos]
            neg_chunk = neg[b * self._k_neg : (b + 1) * self._k_neg]
            batch = torch.cat([pos_chunk, neg_chunk])

            if self._shuffle:
                batch = batch[torch.randperm(len(batch), generator=self._generator)]
            yield batch.tolist()

        if self._drop_last:
            return

        # Emit any residual as a final partial batch if possible.
        pos_used = self._n_batches_full * self._k_pos
        neg_used = self._n_batches_full * self._k_neg
        pos_remain = pos[pos_used:]
        neg_remain = neg[neg_used:]
        if len(pos_remain) + len(neg_remain) == 0:
            return
        residual = torch.cat([pos_remain, neg_remain])
        if self._shuffle:
            residual = residual[torch.randperm(len(residual), generator=self._generator)]
        yield residual.tolist()

    def __len__(self) -> int:
        if self._drop_last:
            return self._n_batches_full
        # +1 if there's residual
        pos_rem = len(self._pos_indices) - self._n_batches_full * self._k_pos
        neg_rem = len(self._neg_indices) - self._n_batches_full * self._k_neg
        return self._n_batches_full + (1 if (pos_rem + neg_rem) > 0 else 0)


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────


def make_loader(
    dataset: Dataset,
    batch_size: int = 256,
    *,
    mode: str = "train",
    mtlm: Optional[MTLMCollator] = None,
    stratified: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: Optional[int] = None,
    drop_last: Optional[bool] = None,
) -> DataLoader:
    """
    Construct a DataLoader with the right sampler / collate for each pipeline mode.

    Parameters
    ----------
    dataset
        Typically a :class:`CreditDefaultDataset`.
    batch_size
        Minibatch size.
    mode
        One of:

        * ``"train"``  — shuffled, ``drop_last=True`` by default
        * ``"val"``    — unshuffled, ``drop_last=False``
        * ``"test"``   — unshuffled, ``drop_last=False``
        * ``"mtlm"``   — shuffled, MTLM collator added automatically (requires
          ``mtlm`` to be passed or a default is created)
    mtlm
        :class:`MTLMCollator` instance, used only when ``mode="mtlm"``.
        If None and mode is ``"mtlm"``, a default collator with seed=``seed``
        is created.
    stratified
        If True, use :class:`StratifiedBatchSampler`. Only valid with
        ``mode="train"`` or ``"mtlm"``. For val/test it is explicitly ignored.
    num_workers
        Forwarded to DataLoader.
    pin_memory
        Forwarded to DataLoader. Set True for CUDA training.
    seed
        If provided, seeds the sampler's local generator for reproducibility.
    drop_last
        Overrides the per-mode default if set.
    """
    if mode not in ("train", "val", "test", "mtlm"):
        raise ValueError(f"Unknown mode: {mode!r}")

    shuffle = mode in ("train", "mtlm")
    drop_last_default = mode in ("train", "mtlm")
    drop_last = drop_last_default if drop_last is None else drop_last

    generator: Optional[torch.Generator] = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)

    # Choose the collate function.
    if mode == "mtlm":
        collator = mtlm if mtlm is not None else MTLMCollator(seed=seed)
        collate_fn: Callable = collator
    else:
        collate_fn = default_collate

    # Choose the sampler.
    if stratified and mode in ("train", "mtlm"):
        # Fish the labels out of the dataset. CreditDefaultDataset exposes them
        # via the `.tensors` property.
        if not hasattr(dataset, "tensors"):
            raise ValueError(
                "Stratified sampling requires a dataset exposing `.tensors['labels']` "
                "(e.g. CreditDefaultDataset); got " + type(dataset).__name__
            )
        labels = dataset.tensors["labels"]
        batch_sampler = StratifiedBatchSampler(
            labels=labels,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            generator=generator,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=generator,
        )

    logger.info(
        "make_loader(mode=%s, batch=%d, stratified=%s) → %d batches",
        mode, batch_size, stratified, len(loader),
    )
    return loader


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    # UTF-8 stdout so box-drawing separators print cleanly on Windows.
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            try:
                _s.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    # Load real preprocessed data.
    root = Path(__file__).resolve().parent.parent
    meta_path = root / "data/processed/feature_metadata.json"
    train_csv = root / "data/processed/train_scaled.csv"
    val_csv = root / "data/processed/val_scaled.csv"
    if not (meta_path.is_file() and train_csv.is_file() and val_csv.is_file()):
        print(
            "[SKIP] dataset.py smoke test requires preprocessing output.\n"
            "       Run `poetry run python run_pipeline.py --preprocess-only` "
            "first to materialise data/processed/*.csv."
        )
        sys.exit(0)

    meta = json.loads(meta_path.read_text())
    import pandas as pd
    df_train = pd.read_csv(train_csv)
    from tokenizer import build_categorical_vocab
    cat_vocab = build_categorical_vocab(meta)
    ds = CreditDefaultDataset(df_train, cat_vocab, verbose=False)

    # ── 1. Standard supervised train loader ──
    print("── 1. Standard train loader ──")
    loader = make_loader(ds, batch_size=256, mode="train", seed=42)
    batch = next(iter(loader))
    assert batch["num_values"].shape == (256, 14)
    print(f"  shape OK: {tuple(batch['num_values'].shape)}, labels rate={batch['label'].mean().item():.3f}")

    # ── 2. Stratified train loader ──
    print("\n── 2. Stratified train loader ──")
    loader_s = make_loader(ds, batch_size=256, mode="train", stratified=True, seed=42)
    rates = []
    for b in loader_s:
        rates.append(b["label"].mean().item())
    pos_rate = float((ds.tensors["labels"] == 1).float().mean().item())
    mean_rate = float(np.mean(rates))
    std_rate = float(np.std(rates))
    print(f"  dataset pos rate: {pos_rate:.4f}")
    print(f"  batch rate mean±std: {mean_rate:.4f} ± {std_rate:.4f}")
    assert std_rate < 0.01, f"stratified batches should have tiny variance, got std={std_rate}"
    print(f"  stratified variance < 0.01 ✓")

    # ── 3. Val loader (unshuffled, no drop_last) ──
    print("\n── 3. Val loader ──")
    df_val = pd.read_csv(val_csv)
    ds_val = CreditDefaultDataset(df_val, cat_vocab, verbose=False)
    loader_v = make_loader(ds_val, batch_size=256, mode="val", seed=42)
    total_seen = sum(b["num_values"].shape[0] for b in loader_v)
    assert total_seen == len(ds_val), f"val loader dropped rows: {total_seen} vs {len(ds_val)}"
    print(f"  every val row visited: {total_seen}/{len(ds_val)} ✓")

    # ── 4. MTLM loader ──
    print("\n── 4. MTLM pretraining loader ──")
    from tokenizer import MTLMCollator
    mtlm = MTLMCollator(mask_prob=0.15, min_mask_per_row=1, max_mask_per_row=5, seed=7)
    loader_m = make_loader(ds, batch_size=128, mode="mtlm", mtlm=mtlm, seed=7)
    batch_m = next(iter(loader_m))
    assert "mask_positions" in batch_m and batch_m["mask_positions"].shape == (128, 23)
    assert "replace_mode" in batch_m
    counts = batch_m["mask_positions"].sum(dim=1)
    assert (counts >= 1).all() and (counts <= 5).all()
    print(f"  mtlm batch keys: {sorted(batch_m.keys())}")
    print(f"  mask counts per row in [1, 5] ✓")

    # ── 5. Reproducibility ──
    print("\n── 5. Reproducibility under same seed ──")
    l1 = make_loader(ds, batch_size=256, mode="train", stratified=True, seed=42)
    l2 = make_loader(ds, batch_size=256, mode="train", stratified=True, seed=42)
    b1 = next(iter(l1))
    b2 = next(iter(l2))
    # Compare the stacked labels tensor (same indices → same labels).
    assert torch.equal(b1["label"], b2["label"]), "stratified sampling not reproducible under same seed"
    print("  identical batches for identical seed ✓")

    print("\nAll dataset smoke tests passed.")
