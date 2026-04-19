"""DataLoader helpers. Dataset lives in tokenizer.py; this file has
StratifiedBatchSampler (fixed pos-rate per batch) and make_loader
(picks sampler + collate + flags for train/val/test/mtlm)."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ..tokenization.tokenizer import (
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


def default_collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Stack per-item dicts into a batch dict, keeping the nested cat_indices
    layout that FeatureEmbedding expects."""
    cat_indices = {
        feat: torch.stack([item["cat_indices"][feat] for item in batch])
        for feat in CATEGORICAL_FEATURES
    }
    return {
        "cat_indices":    cat_indices,
        "pay_state_ids":  torch.stack([item["pay_state_ids"] for item in batch]),
        "pay_severities": torch.stack([item["pay_severities"] for item in batch]),
        "pay_raw":        torch.stack([item["pay_raw"] for item in batch]),
        "num_values":     torch.stack([item["num_values"] for item in batch]),
        "label":          torch.stack([item["label"] for item in batch]),
    }


class StratifiedBatchSampler(Sampler[List[int]]):
    """Batches with a fixed positive-class rate.

    At 22.1% defaults and bs=256, random batches have ~2.6% std on the realised
    rate and occasionally dip below 17% or above 28%, which hurts gradients for
    class-weighted losses. This one splits the dataset into pos/neg pools and
    yields exactly round(bs × pos_rate) positives + the rest negatives per batch.

    drop_last=True drops the trailing partial; drop_last=False emits it at a
    slightly off ratio.
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

        # bounded by whichever pool runs out first
        n_pos_batches = len(pos_idx) // self._k_pos
        n_neg_batches = len(neg_idx) // self._k_neg
        self._n_batches_full = min(n_pos_batches, n_neg_batches)

    def _shuffled(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self._shuffle:
            return tensor
        perm = torch.randperm(len(tensor), generator=self._generator)
        return tensor[perm]

    def __iter__(self) -> Iterator[List[int]]:
        pos = self._shuffled(self._pos_indices)
        neg = self._shuffled(self._neg_indices)

        for b in range(self._n_batches_full):
            pos_chunk = pos[b * self._k_pos : (b + 1) * self._k_pos]
            neg_chunk = neg[b * self._k_neg : (b + 1) * self._k_neg]
            batch = torch.cat([pos_chunk, neg_chunk])

            if self._shuffle:
                batch = batch[torch.randperm(len(batch), generator=self._generator)]
            yield batch.tolist()

        if self._drop_last:
            return

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
        pos_rem = len(self._pos_indices) - self._n_batches_full * self._k_pos
        neg_rem = len(self._neg_indices) - self._n_batches_full * self._k_neg
        return self._n_batches_full + (1 if (pos_rem + neg_rem) > 0 else 0)


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
    """DataLoader factory for the four pipeline modes.

    mode:
      "train"       shuffled, drop_last=True
      "val"/"test"  unshuffled, drop_last=False
      "mtlm"        shuffled with MTLMCollator (default one built if mtlm is None)

    stratified=True swaps in StratifiedBatchSampler (train/mtlm only, ignored for val/test).
    seed seeds a local torch.Generator so sampling is reproducible.
    drop_last overrides the mode default.
    """
    if mode not in ("train", "val", "test", "mtlm"):
        raise ValueError(f"Unknown mode: {mode!r}")

    shuffle = mode in ("train", "mtlm")
    drop_last_default = mode in ("train", "mtlm")
    drop_last = drop_last_default if drop_last is None else drop_last

    generator: Optional[torch.Generator] = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)

    if mode == "mtlm":
        collator = mtlm if mtlm is not None else MTLMCollator(seed=seed)
        collate_fn: Callable = collator
    else:
        collate_fn = default_collate

    if stratified and mode in ("train", "mtlm"):
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
        "make_loader(mode=%s, batch=%d, stratified=%s) -> %d batches",
        mode, batch_size, stratified, len(loader),
    )
    return loader


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            try:
                _s.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    root = Path(__file__).resolve().parent.parent.parent
    meta_path = root / "data/processed/feature_metadata.json"
    train_csv = root / "data/processed/train_scaled.csv"
    val_csv = root / "data/processed/val_scaled.csv"
    if not (meta_path.is_file() and train_csv.is_file() and val_csv.is_file()):
        print(
            "[SKIP] dataset.py smoke test requires preprocessing output.\n"
            "       Run `poetry run python scripts/run_pipeline.py --preprocess-only` "
            "first to materialise data/processed/*.csv."
        )
        sys.exit(0)

    meta = json.loads(meta_path.read_text())
    import pandas as pd
    df_train = pd.read_csv(train_csv)
    from ..tokenization.tokenizer import build_categorical_vocab
    cat_vocab = build_categorical_vocab(meta)
    ds = CreditDefaultDataset(df_train, cat_vocab, verbose=False)

    print("1. standard train loader:")
    loader = make_loader(ds, batch_size=256, mode="train", seed=42)
    batch = next(iter(loader))
    assert batch["num_values"].shape == (256, 14)
    print(f"  shape OK: {tuple(batch['num_values'].shape)}, labels rate={batch['label'].mean().item():.3f}")

    print("\n2. stratified train loader:")
    loader_s = make_loader(ds, batch_size=256, mode="train", stratified=True, seed=42)
    rates = []
    for b in loader_s:
        rates.append(b["label"].mean().item())
    pos_rate = float((ds.tensors["labels"] == 1).float().mean().item())
    mean_rate = float(np.mean(rates))
    std_rate = float(np.std(rates))
    print(f"  dataset pos rate: {pos_rate:.4f}")
    print(f"  batch rate mean+/-std: {mean_rate:.4f} +/- {std_rate:.4f}")
    assert std_rate < 0.01, f"stratified batches should have tiny variance, got std={std_rate}"
    print(f"  stratified variance < 0.01 OK")

    print("\n3. val loader (unshuffled, no drop_last):")
    df_val = pd.read_csv(val_csv)
    ds_val = CreditDefaultDataset(df_val, cat_vocab, verbose=False)
    loader_v = make_loader(ds_val, batch_size=256, mode="val", seed=42)
    total_seen = sum(b["num_values"].shape[0] for b in loader_v)
    assert total_seen == len(ds_val), f"val loader dropped rows: {total_seen} vs {len(ds_val)}"
    print(f"  every val row visited: {total_seen}/{len(ds_val)} OK")

    print("\n4. MTLM pretraining loader:")
    from ..tokenization.tokenizer import MTLMCollator
    mtlm = MTLMCollator(mask_prob=0.15, min_mask_per_row=1, max_mask_per_row=5, seed=7)
    loader_m = make_loader(ds, batch_size=128, mode="mtlm", mtlm=mtlm, seed=7)
    batch_m = next(iter(loader_m))
    assert "mask_positions" in batch_m and batch_m["mask_positions"].shape == (128, 23)
    assert "replace_mode" in batch_m
    counts = batch_m["mask_positions"].sum(dim=1)
    assert (counts >= 1).all() and (counts <= 5).all()
    print(f"  mtlm batch keys: {sorted(batch_m.keys())}")
    print(f"  mask counts per row in [1, 5] OK")

    print("\n5. reproducibility under same seed:")
    l1 = make_loader(ds, batch_size=256, mode="train", stratified=True, seed=42)
    l2 = make_loader(ds, batch_size=256, mode="train", stratified=True, seed=42)
    b1 = next(iter(l1))
    b2 = next(iter(l2))
    assert torch.equal(b1["label"], b2["label"]), "stratified sampling not reproducible under same seed"
    print("  identical batches for identical seed OK")

    print("\nall dataset smoke tests passed.")
