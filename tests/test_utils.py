"""utils.py — determinism / device / checkpoint / early stopping / accounting."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from utils import (  # noqa: E402
    EarlyStopping,
    Timer,
    build_checkpoint_metadata,
    configure_logging,
    count_parameters,
    derive_seed,
    describe_device,
    format_parameter_count,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_deterministic,
)


def test_set_deterministic_reproduces_tensors():
    set_deterministic(seed=42, warn_only=True)
    a = torch.randn(100)
    set_deterministic(seed=42, warn_only=True)
    b = torch.randn(100)
    assert torch.equal(a, b)


def test_set_deterministic_different_seeds_differ():
    set_deterministic(seed=0, warn_only=True)
    a = torch.randn(10)
    set_deterministic(seed=1, warn_only=True)
    b = torch.randn(10)
    assert not torch.equal(a, b)


def test_derive_seed_deterministic():
    assert derive_seed(42, "phase_a", "run_0") == derive_seed(42, "phase_a", "run_0")


def test_derive_seed_differs_by_tag():
    assert derive_seed(42, "phase_a", "run_0") != derive_seed(42, "phase_a", "run_1")


def test_derive_seed_differs_by_parent():
    assert derive_seed(42, "x") != derive_seed(43, "x")


def test_get_device_cpu():
    assert get_device("cpu") == torch.device("cpu")


def test_get_device_auto_returns_valid_device():
    dev = get_device("auto")
    assert dev.type in {"cpu", "cuda", "mps"}


def test_describe_device_cpu():
    assert "CPU" in describe_device(torch.device("cpu"))


def test_get_device_invalid_raises():
    with pytest.raises(ValueError):
        get_device("banana")


def test_checkpoint_roundtrip_trusted():
    # trust_source=True: optimizer state round-trips
    model = torch.nn.Linear(8, 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "ckpt.pt"
        md = build_checkpoint_metadata(seed=42, epoch=3, extra={"note": "test"})
        save_checkpoint(p, model, optimizer=opt, metadata=md)
        assert p.is_file()
        assert (p.with_suffix(".pt.meta.json")).is_file()

        meta = json.loads(p.with_suffix(".pt.meta.json").read_text())
        assert meta["seed"] == 42
        assert meta["epoch"] == 3
        assert meta["extra"]["note"] == "test"

        model2 = torch.nn.Linear(8, 4)
        opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        loaded = load_checkpoint(p, model2, optimizer=opt2, trust_source=True)

        for (_, v1), (_, v2) in zip(
            model.state_dict().items(), model2.state_dict().items()
        ):
            assert torch.equal(v1, v2)
        assert loaded["metadata"]["seed"] == 42


def test_checkpoint_roundtrip_safe_default():
    # safe default (trust_source=False): weights_only=True, weights-only restore
    model = torch.nn.Linear(8, 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "ckpt.pt"
        save_checkpoint(p, model, optimizer=opt)
        model2 = torch.nn.Linear(8, 4)
        loaded = load_checkpoint(p, model2)
        for (_, v1), (_, v2) in zip(
            model.state_dict().items(), model2.state_dict().items()
        ):
            assert torch.equal(v1, v2)
        assert loaded["metadata"]["seed"] is None


def test_load_checkpoint_warns_when_optimizer_requested_in_safe_mode(caplog):
    # safe mode can't restore optimizer state. asking for one must warn,
    # otherwise training state goes silently missing.
    import logging

    model = torch.nn.Linear(8, 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "ckpt.pt"
        save_checkpoint(p, model, optimizer=opt)
        model2 = torch.nn.Linear(8, 4)
        opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        with caplog.at_level(logging.WARNING):
            load_checkpoint(p, model2, optimizer=opt2)
        assert any("optimizer" in r.getMessage().lower() for r in caplog.records)


def test_checkpoint_load_missing_file():
    model = torch.nn.Linear(4, 2)
    with pytest.raises(FileNotFoundError):
        load_checkpoint("/tmp/does_not_exist_xyz.pt", model)


def test_early_stopping_max_mode():
    es = EarlyStopping(patience=3, mode="max", min_delta=1e-4)
    scores = [0.70, 0.75, 0.75, 0.75, 0.75]
    stopped = [es.step(s) for s in scores]
    assert stopped == [False, False, False, False, True]
    assert es.best_score == 0.75
    assert es.best_epoch == 2


def test_early_stopping_min_mode():
    es = EarlyStopping(patience=2, mode="min")
    assert not es.step(1.0)
    assert not es.step(0.5)
    assert not es.step(0.6)
    assert es.step(0.7)
    assert es.best_score == 0.5


def test_early_stopping_rejects_bad_mode():
    with pytest.raises(ValueError):
        EarlyStopping(patience=3, mode="invalid")


def test_early_stopping_stashes_state():
    # best_state must be a detached copy — mutating the model afterwards
    # must not leak into it
    es = EarlyStopping(patience=2, mode="max")
    model = torch.nn.Linear(4, 2)
    es.step(0.5, state=model.state_dict())
    original = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(100.0)
    for k, v in es.best_state.items():
        assert torch.equal(v, original[k])


def test_count_parameters_trainable_only():
    model = torch.nn.Linear(5, 3)
    assert count_parameters(model) == 5 * 3 + 3


def test_count_parameters_includes_frozen():
    model = torch.nn.Linear(5, 3)
    for p in model.parameters():
        p.requires_grad = False
    assert count_parameters(model, trainable_only=True) == 0
    assert count_parameters(model, trainable_only=False) == 5 * 3 + 3


def test_format_parameter_count():
    assert format_parameter_count(500) == "500"
    assert format_parameter_count(28_512) == "28.5K"
    assert format_parameter_count(12_345_678) == "12.35M"
    assert format_parameter_count(3_000_000_000) == "3.00B"


def test_timer_measures_elapsed():
    with Timer("x") as t:
        for _ in range(10_000):
            pass
    assert t.elapsed >= 0


def test_configure_logging_is_idempotent():
    configure_logging()
    configure_logging()
    configure_logging()
    import logging

    handlers = [h for h in logging.getLogger().handlers if getattr(h, "_credit_default_handler", False)]
    assert len(handlers) == 1
