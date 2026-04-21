"""DataLoader factory and stratified-batch sampler for the credit-default task.

The :class:`~..tokenization.tokenizer.CreditDefaultDataset` itself lives in
``tokenization/tokenizer.py`` alongside the tokenisation logic — this module
is the *loader* layer on top of it. It provides three public symbols:

* :func:`default_collate` — stacks the per-item dicts into a batch, preserving
  the nested ``cat_indices`` dict layout that :class:`FeatureEmbedding` expects
  rather than flattening into a single tensor.
* :class:`StratifiedBatchSampler` — yields batches with a **fixed** positive-
  class ratio (see docstring for the quantitative motivation).
* :func:`make_loader` — the one-stop factory used by the training entry points
  to configure the right sampler / collator / shuffle / drop-last combination
  for each of the four pipeline modes (train / val / test / mtlm).

Design choice
-------------
We chose stratified batching over plain uniform sampling because the 22.1 %
positive base rate leaves ~2.6 % std on the realised per-batch positive rate
at ``bs=256``, with some batches dipping below 17 % or climbing above 28 %.
Class-weighted losses (focal, WBCE) are sensitive to that oscillation: the
gradient contribution from positives swings by ±20 % batch-to-batch even
before the model's own output variance is considered. Empirically the
stratified sampler drops the realised pos-rate std to <0.01 (verified in the
smoke test at the bottom of this file), which produces visibly smoother
training curves and meaningfully faster convergence on AUC-PR.

Non-obvious dependency: :class:`MTLMCollator` is imported solely so the
``mtlm`` mode can default-construct one when the caller doesn't provide it —
there's no runtime dep on the MTLM pipeline otherwise."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ..tokenization.tokenizer import CreditDefaultDataset  # re-exported for convenience
from ..tokenization.tokenizer import (
    CATEGORICAL_FEATURES,
    MTLMCollator,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CreditDefaultDataset",
    "StratifiedBatchSampler",
    "default_collate",
    "make_loader",
]


def default_collate(batch: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Stack per-item dicts into a batched dict.

    The key point is that we keep ``cat_indices`` as a *nested* dict
    (one entry per categorical feature) rather than collapsing it into a
    single (B, n_cat) tensor. :class:`FeatureEmbedding` looks each feature
    up in its own :class:`nn.Embedding` table, so it's the feature-wise
    layout that's convenient for the model, not the stacked one.

    Parameters
    ----------
    batch
        Sequence of per-row dicts as produced by
        :class:`CreditDefaultDataset.__getitem__`. Each dict has keys
        ``cat_indices`` (nested dict of scalar long tensors), ``pay_state_ids``
        (shape ``(6,)``), ``pay_severities`` (shape ``(6,)``), ``pay_raw``
        (shape ``(6,)``), ``num_values`` (shape ``(14,)``), and ``label``
        (scalar float).

    Returns
    -------
    Dict[str, torch.Tensor]
        Batched dict with ``cat_indices`` of shape ``(B,)`` per feature and
        everything else of shape ``(B, ...)``.
    """
    # Preserve feature-keyed structure — FeatureEmbedding iterates
    # CATEGORICAL_FEATURES and would have to un-stack otherwise.
    cat_indices = {
        feat: torch.stack([item["cat_indices"][feat] for item in batch])
        for feat in CATEGORICAL_FEATURES
    }
    return {
        "cat_indices": cat_indices,
        "pay_state_ids": torch.stack([item["pay_state_ids"] for item in batch]),
        "pay_severities": torch.stack([item["pay_severities"] for item in batch]),
        "pay_raw": torch.stack([item["pay_raw"] for item in batch]),
        "num_values": torch.stack([item["num_values"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
    }


class StratifiedBatchSampler(Sampler[list[int]]):
    """Yield batches with a fixed, globally-calibrated positive-class rate.

    Why this exists
    ---------------
    On the UCI Default-of-Credit-Card-Clients split we use, the training set
    is 22.1 % positive. Uniform random batches at ``batch_size=256`` produce a
    realised per-batch positive rate with **std ≈ 0.026** and long tails
    (batches occasionally dip below 17 % or cross 28 %). Class-weighted losses
    (focal, WBCE) amplify that noise because the ``pos_weight`` multiplier
    multiplies a randomly-swinging count. The net effect is noisier gradient
    estimates and a visibly rougher AUC-PR trajectory.

    This sampler splits the dataset into positive and negative pools, shuffles
    each pool independently, and deterministically takes
    ``k_pos = round(bs * pos_rate)`` from the positive pool and the remaining
    ``bs - k_pos`` from the negative pool for every batch. The realised
    per-batch positive rate then has **std < 0.01** (see smoke test) — a
    ~3× reduction in batch-level class-composition noise.

    Parameters
    ----------
    labels
        1-D int/bool sequence of length N with values in ``{0, 1}``.
    batch_size
        Target batch size. Must be large enough that ``k_neg = bs - k_pos`` is
        at least 1; with 22.1 % positives that means ``bs >= 2``.
    drop_last
        If True, drop the trailing partial batch. If False, emit a final
        residual batch that may be smaller and at a slightly off ratio.
    shuffle
        Shuffle both pools at the start of each epoch and the rows within each
        batch. Pass False for debugging / deterministic replay.
    generator
        Optional :class:`torch.Generator` for reproducible sampling. When
        supplied by ``make_loader(seed=…)`` two loaders with the same seed
        produce identical batch sequences (asserted by smoke test §5).

    Raises
    ------
    ValueError
        If either class is absent, or if ``batch_size`` is too small to leave
        room for negatives at the empirical ``pos_rate``.
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

        # Normalise to a 1-D long tensor regardless of input type so downstream
        # comparisons are type-stable.
        if isinstance(labels, torch.Tensor):
            lbl = labels.detach().cpu().long().view(-1)
        else:
            lbl = torch.as_tensor(np.asarray(labels).reshape(-1), dtype=torch.long)
        self._labels = lbl

        pos_idx = (lbl == 1).nonzero(as_tuple=True)[0]
        neg_idx = (lbl == 0).nonzero(as_tuple=True)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            # Single-class datasets silently degrade to uniform sampling under a
            # naive fallback; we'd rather fail loudly than hide it.
            raise ValueError(
                "StratifiedBatchSampler requires both classes to be present; "
                f"got pos={len(pos_idx)}, neg={len(neg_idx)}"
            )
        self._pos_indices = pos_idx
        self._neg_indices = neg_idx

        # Anchor the stratification to the *empirical* pos rate of this split,
        # not a hard-coded 0.221 — keeps the sampler robust across CV folds.
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

        # Epoch length is bounded by whichever pool depletes first. With 22.1 %
        # positives and bs=256 the positive pool is the limiting factor (k_pos=57,
        # ~5312 positives → ~93 full batches); the negative pool has slack.
        n_pos_batches = len(pos_idx) // self._k_pos
        n_neg_batches = len(neg_idx) // self._k_neg
        self._n_batches_full = min(n_pos_batches, n_neg_batches)

    def _shuffled(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a view permuted by ``self._generator`` (identity if not shuffling)."""
        if not self._shuffle:
            return tensor
        perm = torch.randperm(len(tensor), generator=self._generator)
        return tensor[perm]

    def __iter__(self) -> Iterator[list[int]]:
        # Per-epoch shuffle of each pool independently, then stride through.
        pos = self._shuffled(self._pos_indices)
        neg = self._shuffled(self._neg_indices)

        for b in range(self._n_batches_full):
            pos_chunk = pos[b * self._k_pos : (b + 1) * self._k_pos]
            neg_chunk = neg[b * self._k_neg : (b + 1) * self._k_neg]
            batch = torch.cat([pos_chunk, neg_chunk])

            # Intra-batch shuffle so the model doesn't see a positives-first
            # ordering (which would e.g. bias BatchNorm running stats).
            if self._shuffle:
                batch = batch[torch.randperm(len(batch), generator=self._generator)]
            yield batch.tolist()

        if self._drop_last:
            return

        # Residual emission: trailing rows from both pools concatenated.
        # This batch has a slightly off positive rate by construction — callers
        # who need strict stratification should leave drop_last=True.
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
    """Build a :class:`DataLoader` configured for one of the four pipeline modes.

    This is the single place the four loaders (supervised train / val / test /
    MTLM pretrain) pick up their shuffle, drop-last, collator, and sampler
    choices — centralising the decisions keeps the two training entry points
    symmetric.

    Parameters
    ----------
    dataset
        Any :class:`torch.utils.data.Dataset` producing dicts compatible with
        :func:`default_collate`. Stratified mode additionally requires the
        dataset to expose ``.tensors['labels']`` (true for
        :class:`CreditDefaultDataset`).
    batch_size
        Target batch size. Ignored when ``stratified=True`` (the sampler
        owns batch composition) except for the k_pos / k_neg split.
    mode
        One of ``"train"`` / ``"val"`` / ``"test"`` / ``"mtlm"``. Defaults are:

        ============  ========  =========  =============
        mode          shuffle   drop_last  collator
        ============  ========  =========  =============
        train         True      True       default_collate
        val, test     False     False      default_collate
        mtlm          True      True       MTLMCollator
        ============  ========  =========  =============
    mtlm
        Explicit MTLM collator (seed / mask-prob configured by the caller).
        Only consulted when ``mode='mtlm'``; if ``None`` a fresh
        :class:`MTLMCollator` is constructed with the same ``seed``.
    stratified
        If True, swap in :class:`StratifiedBatchSampler`. Only applies to
        ``train`` / ``mtlm``; silently ignored for ``val`` / ``test`` because
        evaluation needs to visit *every* row exactly once.
    num_workers, pin_memory
        Standard DataLoader knobs. ``pin_memory=True`` is worth setting on CUDA
        for measurable H→D copy speedups.
    seed
        Seeds a **local** :class:`torch.Generator` so shuffle and mask draws
        are reproducible without touching the global RNG state. Two loaders
        built with the same seed produce identical batch sequences.
    drop_last
        Overrides the mode default when set.

    Returns
    -------
    DataLoader
        A configured loader ready for iteration.

    Raises
    ------
    ValueError
        On unknown ``mode``, or when ``stratified=True`` is requested for a
        dataset that doesn't expose a labels tensor.
    """
    if mode not in ("train", "val", "test", "mtlm"):
        raise ValueError(f"Unknown mode: {mode!r}")

    # Mode-specific defaults; either can still be overridden explicitly.
    shuffle = mode in ("train", "mtlm")
    drop_last_default = mode in ("train", "mtlm")
    drop_last = drop_last_default if drop_last is None else drop_last

    # Local generator → reproducibility without polluting the global torch RNG.
    generator: Optional[torch.Generator] = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)

    if mode == "mtlm":
        # Default-construct an MTLMCollator if the caller didn't supply one so
        # make_loader(mode="mtlm", seed=s) "just works" in smoke tests.
        collator = mtlm if mtlm is not None else MTLMCollator(seed=seed)
        collate_fn: Callable = collator
    else:
        collate_fn = default_collate

    if stratified and mode in ("train", "mtlm"):
        # Stratification needs access to labels — the DataFrame-backed
        # CreditDefaultDataset exposes them via .tensors; arbitrary datasets
        # would need an equivalent hook.
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
        # NB: when passing batch_sampler, DataLoader forbids batch_size / shuffle
        # / drop_last / sampler kwargs — the sampler owns those decisions.
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
        mode,
        batch_size,
        stratified,
        len(loader),
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
    train_csv = root / "data/processed/splits/train_scaled.csv"
    val_csv = root / "data/processed/splits/val_scaled.csv"
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
    print(
        f"  shape OK: {tuple(batch['num_values'].shape)}, labels rate={batch['label'].mean().item():.3f}"
    )

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
    print("  stratified variance < 0.01 OK")

    print("\n3. val loader (unshuffled, no drop_last):")
    df_val = pd.read_csv(val_csv)
    ds_val = CreditDefaultDataset(df_val, cat_vocab, verbose=False)
    loader_v = make_loader(ds_val, batch_size=256, mode="val", seed=42)
    total_seen = sum(b["num_values"].shape[0] for b in loader_v)
    assert total_seen == len(ds_val), f"val loader dropped rows: {total_seen} vs {len(ds_val)}"
    print(f"  every val row visited: {total_seen}/{len(ds_val)} OK")

    print("\n4. MTLM pretraining loader:")
    mtlm = MTLMCollator(mask_prob=0.15, min_mask_per_row=1, max_mask_per_row=5, seed=7)
    loader_m = make_loader(ds, batch_size=128, mode="mtlm", mtlm=mtlm, seed=7)
    batch_m = next(iter(loader_m))
    assert "mask_positions" in batch_m and batch_m["mask_positions"].shape == (128, 23)
    assert "replace_mode" in batch_m
    counts = batch_m["mask_positions"].sum(dim=1)
    assert (counts >= 1).all() and (counts <= 5).all()
    print(f"  mtlm batch keys: {sorted(batch_m.keys())}")
    print("  mask counts per row in [1, 5] OK")

    print("\n5. reproducibility under same seed:")
    l1 = make_loader(ds, batch_size=256, mode="train", stratified=True, seed=42)
    l2 = make_loader(ds, batch_size=256, mode="train", stratified=True, seed=42)
    b1 = next(iter(l1))
    b2 = next(iter(l2))
    assert torch.equal(
        b1["label"], b2["label"]
    ), "stratified sampling not reproducible under same seed"
    print("  identical batches for identical seed OK")

    print("\nall dataset smoke tests passed.")
