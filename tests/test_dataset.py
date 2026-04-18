"""StratifiedBatchSampler + make_loader."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dataset import (  # noqa: E402
    StratifiedBatchSampler,
    default_collate,
    make_loader,
)
from tokenizer import MTLMCollator


def _labels(n_pos: int, n_neg: int) -> torch.Tensor:
    return torch.cat([torch.ones(n_pos), torch.zeros(n_neg)]).long()


def test_stratified_batch_preserves_rate():
    labels = _labels(220, 780)
    sampler = StratifiedBatchSampler(
        labels=labels, batch_size=100, drop_last=True, shuffle=True,
        generator=torch.Generator().manual_seed(0),
    )
    rates = []
    for batch in sampler:
        batch_labels = labels[torch.tensor(batch)]
        rates.append(batch_labels.float().mean().item())
    assert all(abs(r - 0.22) < 0.01 for r in rates), f"rates: {rates}"


def test_stratified_batch_no_duplicates_within_batch():
    labels = _labels(100, 400)
    sampler = StratifiedBatchSampler(
        labels=labels, batch_size=50, drop_last=True, shuffle=True,
        generator=torch.Generator().manual_seed(1),
    )
    for batch in sampler:
        assert len(batch) == len(set(batch))


def test_stratified_batch_reproducible():
    labels = _labels(100, 400)
    s1 = StratifiedBatchSampler(
        labels=labels, batch_size=50, generator=torch.Generator().manual_seed(42)
    )
    s2 = StratifiedBatchSampler(
        labels=labels, batch_size=50, generator=torch.Generator().manual_seed(42)
    )
    assert list(s1) == list(s2)


def test_stratified_batch_rejects_single_class():
    with pytest.raises(ValueError):
        StratifiedBatchSampler(labels=torch.zeros(50), batch_size=10)


def test_stratified_batch_rejects_small_batch_size():
    with pytest.raises(ValueError):
        StratifiedBatchSampler(labels=_labels(99, 1), batch_size=2)


def test_stratified_batch_len():
    labels = _labels(100, 400)
    sampler = StratifiedBatchSampler(labels=labels, batch_size=50, drop_last=True)
    assert len(sampler) == 10


def test_default_collate_shapes(small_dataset):
    batch_list = [small_dataset[i] for i in range(8)]
    batch = default_collate(batch_list)
    assert batch["num_values"].shape == (8, 14)
    assert batch["pay_state_ids"].shape == (8, 6)
    assert batch["pay_severities"].shape == (8, 6)
    assert batch["label"].shape == (8,)
    assert batch["cat_indices"]["SEX"].shape == (8,)


def test_make_loader_train_mode(small_dataset):
    loader = make_loader(small_dataset, batch_size=32, mode="train", seed=0)
    batch = next(iter(loader))
    assert batch["num_values"].shape[0] == 32


def test_make_loader_stratified_variance_small(small_dataset):
    loader = make_loader(
        small_dataset, batch_size=64, mode="train", stratified=True, seed=0
    )
    rates = [b["label"].mean().item() for b in loader]
    assert len(rates) > 0
    if len(rates) > 1:
        assert np.std(rates) < 0.05


def test_make_loader_val_visits_every_row(small_dataset):
    loader = make_loader(small_dataset, batch_size=40, mode="val")
    seen = sum(b["num_values"].shape[0] for b in loader)
    assert seen == len(small_dataset)


def test_make_loader_mtlm_emits_mask_positions(small_dataset):
    mtlm = MTLMCollator(mask_prob=0.15, seed=0)
    loader = make_loader(
        small_dataset, batch_size=32, mode="mtlm", mtlm=mtlm, seed=0
    )
    batch = next(iter(loader))
    assert "mask_positions" in batch
    assert batch["mask_positions"].shape == (32, 23)


def test_make_loader_rejects_unknown_mode(small_dataset):
    with pytest.raises(ValueError):
        make_loader(small_dataset, batch_size=32, mode="banana")


def test_make_loader_stratified_with_val_ignored(small_dataset):
    loader = make_loader(
        small_dataset, batch_size=32, mode="val", stratified=True, seed=0
    )
    _ = next(iter(loader))
