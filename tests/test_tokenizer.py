"""tokenizer.py — hybrid PAY encoding, vectorisation, MTLM collator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from tokenizer import (  # noqa: E402
    CATEGORICAL_FEATURES,
    MAX_PAY_DELAY,
    NUMERICAL_FEATURES,
    PAY_STATUS_FEATURES,
    PAY_STATE_DELINQUENT,
    PAY_STATE_MINIMUM,
    PAY_STATE_NO_BILL,
    PAY_STATE_PAID_FULL,
    PAYValueError,
    CreditDefaultDataset,
    MTLMCollator,
    build_categorical_vocab,
    build_numerical_vocab,
    encode_pay_value,
    tokenize_dataframe,
    tokenize_row,
)


@pytest.mark.parametrize(
    "pay,expected",
    [
        (-2, (PAY_STATE_NO_BILL, 0.0)),
        (-1, (PAY_STATE_PAID_FULL, 0.0)),
        (0,  (PAY_STATE_MINIMUM, 0.0)),
        (1,  (PAY_STATE_DELINQUENT, 1 / 8)),
        (4,  (PAY_STATE_DELINQUENT, 4 / 8)),
        (8,  (PAY_STATE_DELINQUENT, 1.0)),
    ],
)
def test_encode_pay_value_in_range(pay, expected):
    assert encode_pay_value(pay) == expected


@pytest.mark.parametrize("bad", [-3, -10, 9, 100])
def test_encode_pay_value_out_of_range(bad):
    with pytest.raises(PAYValueError):
        encode_pay_value(bad)


def test_max_pay_delay_constant():
    assert MAX_PAY_DELAY == 8


def test_build_categorical_vocab_structure(metadata):
    vocab = build_categorical_vocab(metadata)
    assert set(vocab.keys()) == set(CATEGORICAL_FEATURES)
    for feat, mapping in vocab.items():
        assert all(isinstance(k, int) for k in mapping.keys())
        assert all(isinstance(v, int) for v in mapping.values())
        assert sorted(mapping.values()) == list(range(len(mapping)))


def test_build_numerical_vocab_shape():
    vocab = build_numerical_vocab()
    assert len(vocab) == len(NUMERICAL_FEATURES)
    assert sorted(vocab.values()) == list(range(len(NUMERICAL_FEATURES)))


def test_tokenize_row_vs_vectorised(train_df_small, cat_vocab):
    ds = CreditDefaultDataset(train_df_small, cat_vocab, verbose=False)
    for i in range(10):
        fast = ds[i]
        slow = tokenize_row(train_df_small.iloc[i], cat_vocab)
        assert fast["cat_indices"]["SEX"].item() == slow[0]["SEX"]
        assert fast["cat_indices"]["EDUCATION"].item() == slow[0]["EDUCATION"]
        assert fast["cat_indices"]["MARRIAGE"].item() == slow[0]["MARRIAGE"]
        assert fast["pay_state_ids"].tolist() == slow[1]
        assert fast["pay_severities"].tolist() == pytest.approx(slow[2], abs=1e-5)
        assert fast["num_values"].tolist() == pytest.approx(slow[3], abs=1e-4)
        assert int(fast["label"].item()) == slow[4]


def test_dataset_length(train_df_small, cat_vocab):
    ds = CreditDefaultDataset(train_df_small, cat_vocab, verbose=False)
    assert len(ds) == len(train_df_small)


def test_dataset_item_shapes(small_dataset):
    item = small_dataset[0]
    assert item["pay_state_ids"].shape == (6,)
    assert item["pay_severities"].shape == (6,)
    assert item["num_values"].shape == (14,)
    assert item["label"].ndim == 0


def test_dataset_getitem_is_o1_shaped(small_dataset):
    # __getitem__ must not re-iterate the frame
    import time

    t0 = time.perf_counter()
    for i in range(len(small_dataset)):
        _ = small_dataset[i]
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5, f"__getitem__ too slow: {elapsed*1000:.1f} ms for 128 rows"


def test_tokenize_dataframe_rejects_unseen_categorical(train_df_small, cat_vocab):
    bad = train_df_small.copy()
    bad.loc[bad.index[0], "SEX"] = 99
    with pytest.raises(KeyError, match="Unseen SEX"):
        tokenize_dataframe(bad, cat_vocab)


def test_tokenize_dataframe_rejects_out_of_range_pay(train_df_small, cat_vocab):
    bad = train_df_small.copy()
    bad.loc[bad.index[0], "PAY_0"] = 99
    with pytest.raises(PAYValueError):
        tokenize_dataframe(bad, cat_vocab)


def test_mtlm_collator_output_shape(small_dataset):
    batch = [small_dataset[i] for i in range(16)]
    collator = MTLMCollator(mask_prob=0.15, seed=42)
    out = collator(batch)
    assert out["mask_positions"].shape == (16, 23)
    assert out["replace_mode"].shape == (16, 23)
    assert out["num_values"].shape == (16, 14)


def test_mtlm_collator_respects_min_max(small_dataset):
    batch = [small_dataset[i] for i in range(32)]
    collator = MTLMCollator(
        mask_prob=0.15, min_mask_per_row=2, max_mask_per_row=4, seed=0
    )
    out = collator(batch)
    counts = out["mask_positions"].sum(dim=1)
    assert (counts >= 2).all()
    assert (counts <= 4).all()


def test_mtlm_collator_replace_mode_consistency(small_dataset):
    batch = [small_dataset[i] for i in range(16)]
    collator = MTLMCollator(mask_prob=0.15, seed=7)
    out = collator(batch)
    # replace_mode=-1 iff mask_positions=False
    assert ((out["replace_mode"] == -1) == (~out["mask_positions"])).all()
    selected_modes = out["replace_mode"][out["mask_positions"]]
    assert (selected_modes >= 0).all() and (selected_modes <= 2).all()


def test_mtlm_collator_deterministic(small_dataset):
    batch = [small_dataset[i] for i in range(16)]
    out_a = MTLMCollator(mask_prob=0.15, seed=123)(batch)
    out_b = MTLMCollator(mask_prob=0.15, seed=123)(batch)
    assert torch.equal(out_a["mask_positions"], out_b["mask_positions"])


def test_mtlm_collator_rejects_bad_probs():
    with pytest.raises(ValueError):
        MTLMCollator(mask_prob=0.0)
    with pytest.raises(ValueError):
        MTLMCollator(mask_prob=1.0)
    with pytest.raises(ValueError):
        MTLMCollator(replace_with_mask=0.7, replace_with_random=0.5)


def test_pay_raw_present_and_shifted_into_zero_ten(train_df_small, cat_vocab):
    ds = CreditDefaultDataset(train_df_small, cat_vocab, verbose=False)
    item = ds[0]
    assert "pay_raw" in item, "pay_raw missing from __getitem__ output"
    assert item["pay_raw"].dtype == torch.int64
    assert item["pay_raw"].shape == (6,)
    assert int(item["pay_raw"].min()) >= 0
    assert int(item["pay_raw"].max()) <= 10


def test_pay_raw_matches_raw_dataframe_values(train_df_small, cat_vocab):
    ds = CreditDefaultDataset(train_df_small, cat_vocab, verbose=False)
    expected = train_df_small[PAY_STATUS_FEATURES].to_numpy(dtype=np.int64)
    observed = ds.tensors["pay_raw"].numpy() - 2  # undo +2 shift
    assert np.array_equal(observed, expected)


def test_pay_raw_consistent_with_hybrid_state_severity(train_df_small, cat_vocab):
    # pay_raw must be recoverable from (state_id, severity) — the N5 aux
    # head's supervision target relies on this invariant
    ds = CreditDefaultDataset(train_df_small, cat_vocab, verbose=False)
    state = ds.tensors["pay_state_ids"]
    sev = ds.tensors["pay_severities"]
    raw_shifted = ds.tensors["pay_raw"]
    raw = raw_shifted - 2
    reconstructed = torch.zeros_like(raw)
    reconstructed[state == PAY_STATE_NO_BILL] = -2
    reconstructed[state == PAY_STATE_PAID_FULL] = -1
    reconstructed[state == PAY_STATE_MINIMUM] = 0
    delinquent = state == PAY_STATE_DELINQUENT
    reconstructed[delinquent] = (sev[delinquent] * MAX_PAY_DELAY).round().long()
    assert torch.equal(reconstructed, raw), (
        "pay_raw is inconsistent with (state, severity) — the N5 aux head "
        "target would not reflect the primary tokenisation"
    )


def test_pay_raw_flows_through_default_collate(small_dataset):
    from dataset import default_collate

    batch = default_collate([small_dataset[i] for i in range(8)])
    assert "pay_raw" in batch
    assert batch["pay_raw"].shape == (8, 6)
    assert batch["pay_raw"].dtype == torch.int64


def test_pay_raw_flows_through_mtlm_collator(small_dataset):
    out = MTLMCollator(mask_prob=0.15, seed=7)([small_dataset[i] for i in range(8)])
    assert "pay_raw" in out
    assert out["pay_raw"].shape == (8, 6)
