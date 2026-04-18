"""Tests for src/uncertainty.py.

The MC-dropout pipeline requires a trained checkpoint, so the heavy
integration test is gated on its presence. Unit tests cover the
pure-numpy helpers (entropy, metrics derivation, refuse-curve) with
synthetic sample matrices.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import uncertainty as un  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# enable_dropout
# ──────────────────────────────────────────────────────────────────────────────


def test_enable_dropout_flips_only_dropout_modules():
    m = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.1),
                      nn.LayerNorm(4), nn.Dropout(0.2))
    m.eval()
    for mod in m:
        assert mod.training is False
    n = un.enable_dropout(m)
    assert n == 2
    # Dropouts now in train mode.
    assert m[1].training is True
    assert m[3].training is True
    # LayerNorm / Linear stayed in eval.
    assert m[0].training is False
    assert m[2].training is False


def test_enable_dropout_returns_zero_when_model_has_none():
    m = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    assert un.enable_dropout(m) == 0


# ──────────────────────────────────────────────────────────────────────────────
# uncertainty_from_samples
# ──────────────────────────────────────────────────────────────────────────────


def test_uncertainty_shapes_and_bounds():
    rng = np.random.default_rng(0)
    T, N = 20, 100
    probs = rng.uniform(0.0, 1.0, size=(T, N))
    y = rng.integers(0, 2, size=N)
    u = un.uncertainty_from_samples(probs, y)
    assert u.mean.shape == (N,)
    assert u.std.shape == (N,)
    assert u.predictive_entropy.shape == (N,)
    # Entropy of a Bernoulli is in [0, log 2].
    assert (u.predictive_entropy >= 0).all()
    assert (u.predictive_entropy <= np.log(2) + 1e-9).all()
    # Mutual info non-negative (within numerical tolerance).
    assert (u.mutual_info >= -1e-9).all()


def test_uncertainty_point_estimates_collapse_mc_spread():
    """If every MC sample agrees, std/MI collapse to zero."""
    T, N = 15, 50
    probs = np.tile(np.linspace(0.1, 0.9, N), (T, 1))
    y = np.zeros(N, dtype=int)
    u = un.uncertainty_from_samples(probs, y)
    assert u.std.max() < 1e-12
    assert u.mutual_info.max() < 1e-9


def test_mc_dropout_predict_refuses_n_less_than_2():
    with pytest.raises(ValueError):
        un.mc_dropout_predict(nn.Dropout(0.1), [], n_samples=1)


def test_refuse_curve_is_monotone_on_informative_signal():
    rng = np.random.default_rng(5)
    T, N = 40, 1000
    # Build data where predictive entropy correlates with misclassification.
    y = rng.integers(0, 2, size=N)
    base = np.where(y == 1, 0.75, 0.25)
    # Noisy — higher noise rows are harder to predict.
    noise_level = rng.uniform(0.0, 0.4, size=N)
    probs = np.clip(
        base + noise_level[None, :] * rng.standard_normal((T, N)),
        0.01, 0.99,
    )
    u = un.uncertainty_from_samples(probs, y)
    df = un.refuse_curve(u, signal="predictive_entropy")
    # Accuracy on retained should (weakly) increase as we defer more.
    accs = df.sort_values("fraction_deferred")["accuracy_retained"].values
    # Allow a tiny number of non-monotone steps from finite-sample noise.
    violations = int(np.sum(np.diff(accs) < -0.005))
    assert violations <= 1, f"too many monotonicity violations: {accs}"


def test_refuse_curve_rejects_unknown_signal():
    u = un.UncertaintyArrays(
        mean=np.zeros(10), std=np.zeros(10),
        predictive_entropy=np.zeros(10), aleatoric=np.zeros(10),
        mutual_info=np.zeros(10), y_true=np.zeros(10, dtype=int),
    )
    with pytest.raises(ValueError):
        un.refuse_curve(u, signal="nope")


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end — requires a real checkpoint.
# ──────────────────────────────────────────────────────────────────────────────


def test_main_end_to_end(tmp_path):
    seed_42 = REPO / "results" / "transformer" / "seed_42"
    proc = REPO / "data" / "processed"
    if (not (seed_42 / "best.pt").is_file()
            or not (proc / "test_scaled.csv").is_file()):
        pytest.skip("no committed checkpoint or preprocessing outputs")
    rc = un.main([
        "--run", str(seed_42),
        "--n-samples", "5",       # small for CI speed
        "--output-dir", str(tmp_path / "out"),
        "--figures-dir", str(tmp_path / "fig"),
    ])
    assert rc == 0
    assert (tmp_path / "out" / "mc_dropout.npz").is_file()
    summary = (tmp_path / "out" / "uncertainty_summary.json")
    assert summary.is_file()
