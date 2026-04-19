"""train.py — LR schedule, loss factory, metrics, eval loop, optimiser
construction, e2e smoke."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW

REPO = Path(__file__).resolve().parent.parent.parent

from src.models.model import TabularTransformer  # noqa: E402
from src.training import train as train_mod  # noqa: E402
from src.training.losses import FocalLoss, LabelSmoothingBCELoss, WeightedBCELoss  # noqa: E402


def test_cosine_warmup_schedule_starts_at_zero_and_peaks():
    params = [torch.nn.Parameter(torch.zeros(1))]
    opt = AdamW(params, lr=1.0)
    sched = train_mod.build_cosine_warmup_schedule(
        opt,
        warmup_steps=10,
        total_steps=100,
        min_lr_frac=0.0,
    )
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(0.0, abs=1e-9)
    assert lrs[10] == pytest.approx(1.0, abs=1e-9)
    assert lrs[-1] < 0.01


def test_cosine_warmup_schedule_floor_respects_min_lr_frac():
    params = [torch.nn.Parameter(torch.zeros(1))]
    opt = AdamW(params, lr=1.0)
    sched = train_mod.build_cosine_warmup_schedule(
        opt,
        warmup_steps=0,
        total_steps=50,
        min_lr_frac=0.1,
    )
    for _ in range(50):
        opt.step()
        sched.step()
    final = opt.param_groups[0]["lr"]
    assert 0.095 <= final <= 0.105


def test_cosine_warmup_rejects_zero_total_steps():
    params = [torch.nn.Parameter(torch.zeros(1))]
    opt = AdamW(params, lr=1.0)
    with pytest.raises(ValueError):
        train_mod.build_cosine_warmup_schedule(opt, 0, 0)


def test_cosine_warmup_schedule_monotonic_warmup_then_decay():
    params = [torch.nn.Parameter(torch.zeros(1))]
    opt = AdamW(params, lr=1.0)
    sched = train_mod.build_cosine_warmup_schedule(
        opt,
        warmup_steps=20,
        total_steps=100,
        min_lr_frac=0.0,
    )
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    for i in range(1, 20):
        assert lrs[i] >= lrs[i - 1]
    for i in range(21, 100):
        assert lrs[i] <= lrs[i - 1] + 1e-9


def _fake_args(**overrides: Any) -> Namespace:
    defaults = {
        "loss": "focal",
        "focal_gamma": 2.0,
        "focal_alpha": "balanced",
        "label_smoothing_eps": 0.05,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_loss_factory_focal_default():
    y = torch.tensor([0.0, 0.0, 0.0, 1.0])
    loss = train_mod.build_primary_loss(_fake_args(), y)
    assert isinstance(loss, FocalLoss)
    # balanced α = (α_pos = N_neg/N = 0.75, α_neg = 0.25)
    assert isinstance(loss.alpha, tuple)
    assert loss.alpha == pytest.approx((0.75, 0.25))


def test_loss_factory_wbce():
    y = torch.tensor([0.0, 0.0, 0.0, 1.0])
    loss = train_mod.build_primary_loss(_fake_args(loss="wbce"), y)
    assert isinstance(loss, WeightedBCELoss)


def test_loss_factory_label_smoothing():
    y = torch.tensor([0.0, 0.0, 0.0, 1.0])
    loss = train_mod.build_primary_loss(_fake_args(loss="label-smoothing"), y)
    assert isinstance(loss, LabelSmoothingBCELoss)
    assert loss.epsilon == pytest.approx(0.05)


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("balanced", "balanced"),
        ("none", None),
        ("0.75", 0.75),
        ("(0.6, 0.4)", (0.6, 0.4)),
        ("0.6,0.4", (0.6, 0.4)),
    ],
)
def test_resolve_focal_alpha(spec, expected):
    result = train_mod._resolve_focal_alpha(spec)
    if isinstance(expected, tuple):
        assert result == pytest.approx(expected)
    else:
        assert result == expected


def test_resolve_focal_alpha_rejects_garbage():
    import argparse as _ap

    with pytest.raises(_ap.ArgumentTypeError):
        train_mod._resolve_focal_alpha("not-a-number")


def test_compute_ece_on_perfect_predictions_is_zero():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])
    assert train_mod.compute_ece(y_true, y_prob, n_bins=10) == pytest.approx(0.0)


def test_compute_ece_on_systematically_overconfident_model_nonzero():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=1000)
    y_prob = np.full(1000, 0.9)
    ece = train_mod.compute_ece(y_true, y_prob)
    assert ece > 0.3


def test_compute_classification_metrics_returns_expected_keys():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.random(size=200)
    m = train_mod.compute_classification_metrics(y_true, y_prob)
    for key in ("auc_roc", "auc_pr", "f1", "accuracy", "precision", "recall", "brier", "ece"):
        assert key in m


def test_compute_classification_metrics_handles_single_class_gracefully():
    y_true = np.zeros(100, dtype=int)
    y_prob = np.full(100, 0.3)
    m = train_mod.compute_classification_metrics(y_true, y_prob)
    assert np.isnan(m["auc_roc"])


def _mini_batch(B: int = 4) -> dict[str, Any]:
    torch.manual_seed(0)
    sex = torch.arange(B) % 2
    edu = torch.arange(B) % 4
    mar = torch.arange(B) % 3
    labels = (torch.arange(B) % 2).float()
    return {
        "cat_indices": {
            "SEX": sex,
            "EDUCATION": edu,
            "MARRIAGE": mar,
        },
        "pay_state_ids": torch.zeros(B, 6, dtype=torch.long),
        "pay_severities": torch.zeros(B, 6, dtype=torch.float),
        "pay_raw": torch.full((B, 6), 2, dtype=torch.long),  # PAY=0 after +2 shift
        "num_values": torch.randn(B, 14),
        "label": labels,
    }


class _TrivialLoader:
    def __init__(self, batch, n_batches=2):
        self.batch = batch
        self.n_batches = n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            yield self.batch


def test_evaluate_on_loader_returns_expected_payload():
    model = TabularTransformer()
    model.eval()
    result = train_mod.evaluate_on_loader(
        model,
        _TrivialLoader(_mini_batch(B=4), n_batches=3),
        device=torch.device("cpu"),
    )
    assert result["y_true"].shape == (12,)
    assert result["y_prob"].shape == (12,)
    assert "metrics" in result
    assert "attn_weights" not in result


def test_evaluate_on_loader_collects_attn_when_requested():
    model = TabularTransformer(n_layers=2)
    model.eval()
    result = train_mod.evaluate_on_loader(
        model,
        _TrivialLoader(_mini_batch(B=4), n_batches=3),
        device=torch.device("cpu"),
        collect_attn=True,
    )
    assert "attn_weights" in result
    assert len(result["attn_weights"]) == 2
    assert result["attn_weights"][0].shape == (12, 4, 24, 24)


def test_single_group_optimiser_when_no_pretrain():
    model = TabularTransformer()
    args = Namespace(
        lr=3e-4,
        weight_decay=1e-5,
        encoder_lr_ratio=0.2,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
    )
    opt = train_mod.build_optimizer(model, args, pretrained=False)
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["lr"] == pytest.approx(3e-4)


def test_two_group_optimiser_assigns_smaller_lr_to_encoder_when_pretrain():
    model = TabularTransformer(aux_pay0=True)
    args = Namespace(
        lr=3e-4,
        weight_decay=1e-5,
        encoder_lr_ratio=0.2,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
    )
    opt = train_mod.build_optimizer(model, args, pretrained=True)
    assert len(opt.param_groups) == 2
    encoder_group, head_group = opt.param_groups[0], opt.param_groups[1]
    assert encoder_group["lr"] == pytest.approx(3e-4 * 0.2)
    assert head_group["lr"] == pytest.approx(3e-4)
    head_param_ids = {id(p) for p in head_group["params"]}
    aux_head_ids = {id(p) for p in model.aux_pay0_head.parameters()}  # type: ignore[union-attr]
    assert aux_head_ids.issubset(head_param_ids)


def test_train_one_epoch_reduces_loss_on_a_trivial_task():
    torch.manual_seed(0)
    model = TabularTransformer()
    opt = AdamW(model.parameters(), lr=1e-3)
    sched = train_mod.build_cosine_warmup_schedule(opt, 0, 20)
    loss_fn = WeightedBCELoss()

    batch = _mini_batch(B=16)
    loader = _TrivialLoader(batch, n_batches=5)
    with torch.no_grad():
        start_logit = model(batch)["logit"]
        start_loss = loss_fn(start_logit, batch["label"]).item()

    stats = train_mod.train_one_epoch(
        model,
        loader,
        opt,
        sched,
        loss_fn,
        torch.device("cpu"),
        grad_clip=1.0,
    )
    with torch.no_grad():
        end_logit = model(batch)["logit"]
        end_loss = loss_fn(end_logit, batch["label"]).item()

    assert end_loss <= start_loss + 1e-3, f"loss went up: {start_loss:.4f} → {end_loss:.4f}"
    assert np.isfinite(stats["train_loss"])
    assert stats["grad_norm_mean"] > 0


def test_train_one_epoch_with_aux_loss_updates_aux_head():
    torch.manual_seed(0)
    model = TabularTransformer(aux_pay0=True)
    opt = AdamW(model.parameters(), lr=1e-3)
    sched = train_mod.build_cosine_warmup_schedule(opt, 0, 20)
    primary_loss = WeightedBCELoss()
    aux_loss = nn.CrossEntropyLoss()

    batch = _mini_batch(B=8)
    assert model.aux_pay0_head is not None
    before = next(iter(model.aux_pay0_head.parameters())).detach().clone()

    train_mod.train_one_epoch(
        model,
        _TrivialLoader(batch, n_batches=3),
        opt,
        sched,
        primary_loss,
        torch.device("cpu"),
        aux_loss_fn=aux_loss,
        aux_lambda=0.5,
    )
    after = next(iter(model.aux_pay0_head.parameters())).detach().clone()
    assert not torch.allclose(before, after), "aux head did not update"


def test_main_smoke_test_produces_all_expected_artefacts(tmp_path: Path):
    if not (REPO / "data/processed/splits/train_scaled.csv").is_file():
        pytest.skip("preprocessing outputs not present; run run_pipeline.py first")

    output_dir = tmp_path / "run"
    rc = train_mod.main(
        [
            "--seed",
            "0",
            "--smoke-test",
            "--epochs",
            "2",
            "--patience",
            "10",
            "--batch-size",
            "64",
            "--no-save-attn",
            "--log-every",
            "1",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert rc == 0

    config = json.loads((output_dir / "config.json").read_text())
    for key in (
        "seed",
        "param_count",
        "total_steps",
        "warmup_steps",
        "train_size",
        "val_size",
        "test_size",
    ):
        assert key in config

    import pandas as _pd

    log = _pd.read_csv(output_dir / "train_log.csv")
    assert len(log) == 2
    assert "train_loss" in log.columns
    assert "val_auc_roc" in log.columns
    assert "lr" in log.columns

    tm = json.loads((output_dir / "test_metrics.json").read_text())
    assert "metrics" in tm
    assert "threshold_sweep" in tm

    preds = np.load(output_dir / "test_predictions.npz")
    assert set(preds.files) == {"y_true", "y_prob", "y_pred"}
    assert preds["y_true"].shape == preds["y_prob"].shape == preds["y_pred"].shape

    assert (output_dir / "best.pt").is_file()
    assert (output_dir / "best.pt.weights").is_file()
    assert (output_dir / "best.pt.meta.json").is_file()
