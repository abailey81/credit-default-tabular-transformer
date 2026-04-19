"""Tests for src/interpret.py.

Attention weights files are large (and gitignored), so we use synthetic
tensors of the same shape for unit tests. The end-to-end case is skipped
when no real weights are present, which will be the common case on a
fresh clone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent

from src.evaluation import interpret as interp  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# Synthetic attention fixture
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_attn() -> np.ndarray:
    """Random attention tensor of the shape train.py actually writes:
    (n_layers, N, n_heads, seq_len, seq_len). Rows are normalised so they
    pass the 'rows sum to 1' contract that softmax attention satisfies."""
    rng = np.random.default_rng(0)
    n_layers, n_samples, n_heads, seq_len = 2, 100, 4, 24
    raw = rng.random((n_layers, n_samples, n_heads, seq_len, seq_len))
    # Normalise the last dim so each row sums to 1.
    raw = raw / raw.sum(axis=-1, keepdims=True)
    return raw.astype(np.float32)


@pytest.fixture
def synthetic_y_true() -> np.ndarray:
    """Balanced-ish labels so the defaulter/non-defaulter split has
    samples on both sides."""
    rng = np.random.default_rng(1)
    return rng.integers(0, 2, size=100).astype(int)


# ──────────────────────────────────────────────────────────────────────────────
# Core analysis functions
# ──────────────────────────────────────────────────────────────────────────────


def test_attention_rollout_output_shape(synthetic_attn):
    rollout = interp.attention_rollout(synthetic_attn)
    _, n_samples, _, seq_len, _ = synthetic_attn.shape
    assert rollout.shape == (n_samples, seq_len, seq_len)


def test_attention_rollout_rows_sum_to_one(synthetic_attn):
    """The 0.5*A + 0.5*I blend preserves row-stochastic property. If it
    didn't, the rollout would not be a probability distribution and every
    downstream interpretation would be wrong."""
    rollout = interp.attention_rollout(synthetic_attn)
    row_sums = rollout.sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)


def test_cls_to_feature_scores_normalised_and_complete(synthetic_attn):
    rollout = interp.attention_rollout(synthetic_attn)
    scores = interp.cls_to_feature_scores(rollout)
    # One entry per feature, excluding CLS.
    assert len(scores) == len(interp.FEATURE_LABELS)
    # Scores are normalised to sum to 1.
    assert abs(sum(scores.values()) - 1.0) < 1e-5
    # Every score is non-negative.
    assert all(v >= 0 for v in scores.values())


def test_attention_by_class_shapes_match(synthetic_attn, synthetic_y_true):
    rollout = interp.attention_rollout(synthetic_attn)
    defaulter, nondefaulter = interp.attention_by_class(rollout, synthetic_y_true)
    # Each is a single (seq_len, seq_len) mean matrix.
    assert defaulter.shape == nondefaulter.shape
    assert defaulter.shape == (rollout.shape[1], rollout.shape[2])


def test_attention_by_class_raises_when_one_class_absent(synthetic_attn):
    """All-zero labels would crash averaging by class. The function should
    flag the problem explicitly rather than return garbage."""
    rollout = interp.attention_rollout(synthetic_attn)
    y_all_zero = np.zeros(rollout.shape[0], dtype=int)
    with pytest.raises(ValueError):
        interp.attention_by_class(rollout, y_all_zero)


def test_per_head_entropy_shape_and_finite(synthetic_attn):
    entropy = interp.per_head_entropy(synthetic_attn)
    n_layers, _, n_heads, _, _ = synthetic_attn.shape
    assert entropy.shape == (n_layers, n_heads)
    assert np.isfinite(entropy).all()
    # Entropy is non-negative.
    assert (entropy >= 0).all()


# ──────────────────────────────────────────────────────────────────────────────
# Plot functions produce non-empty PNGs
# ──────────────────────────────────────────────────────────────────────────────


def test_plot_rollout_heatmap_writes_png(tmp_path: Path, synthetic_attn):
    rollout = interp.attention_rollout(synthetic_attn)
    out = tmp_path / "rollout.png"
    interp.plot_rollout_heatmap(rollout, out)
    assert out.is_file() and out.stat().st_size > 0


def test_plot_cls_feature_bars_writes_png(tmp_path: Path, synthetic_attn):
    rollout = interp.attention_rollout(synthetic_attn)
    scores = interp.cls_to_feature_scores(rollout)
    out = tmp_path / "bars.png"
    interp.plot_cls_feature_bars(scores, out)
    assert out.is_file() and out.stat().st_size > 0


def test_plot_per_head_heatmaps_writes_png(tmp_path: Path, synthetic_attn):
    out = tmp_path / "per_head.png"
    interp.plot_per_head_heatmaps(synthetic_attn, out)
    assert out.is_file() and out.stat().st_size > 0


def test_plot_class_conditional_writes_png(tmp_path: Path, synthetic_attn, synthetic_y_true):
    rollout = interp.attention_rollout(synthetic_attn)
    defaulter, nondefaulter = interp.attention_by_class(rollout, synthetic_y_true)
    out = tmp_path / "class.png"
    interp.plot_class_conditional(defaulter, nondefaulter, out)
    assert out.is_file() and out.stat().st_size > 0


def test_plot_vs_rf_importance_writes_png(tmp_path: Path, synthetic_attn):
    rollout = interp.attention_rollout(synthetic_attn)
    attn_scores = interp.cls_to_feature_scores(rollout)
    # Partial overlap: only a few features in common between RF and transformer.
    rf_gini = {"PAY_0": 0.09, "SEX": 0.01, "BILL_AMT1": 0.02}
    out = tmp_path / "vs_rf.png"
    interp.plot_vs_rf_importance(attn_scores, rf_gini, out)
    assert out.is_file() and out.stat().st_size > 0


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def test_build_parser_help_renders_without_error(capsys):
    parser = interp._build_parser()
    parser.print_help()
    captured = capsys.readouterr()
    assert captured.out.startswith("usage:")
    assert "--run-dir" in captured.out


# ──────────────────────────────────────────────────────────────────────────────
# Loading helpers
# ──────────────────────────────────────────────────────────────────────────────


def test_load_attention_missing_file_raises(tmp_path: Path):
    """Loading should fail with an actionable message pointing at the
    --no-save-attn flag, not a generic FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="--no-save-attn"):
        interp.load_attention(tmp_path)


def test_load_attention_stacks_layers_in_order(tmp_path: Path):
    """Keys layer_0, layer_1, ... must stack with layer_0 at index 0 even
    when numpy's dict order is alphabetical (which would put layer_10
    before layer_2)."""
    rng = np.random.default_rng(42)
    path = tmp_path / "test_attn_weights.npz"
    tensors = {f"layer_{i}": rng.random((3, 2, 24, 24)).astype(np.float32) for i in range(3)}
    np.savez_compressed(path, **tensors)

    loaded = interp.load_attention(tmp_path)
    assert loaded.shape == (3, 3, 2, 24, 24)
    # The first layer in the stack must match layer_0, not layer_2.
    assert np.allclose(loaded[0], tensors["layer_0"])


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end smoke
# ──────────────────────────────────────────────────────────────────────────────


def test_main_end_to_end(tmp_path: Path):
    """Run main() against the committed seed_42 artefacts if the attention
    file is present. Skips cleanly otherwise so fresh clones are not
    forced to regenerate 30 MB of weights before pytest passes."""
    seed_42 = REPO / "results" / "transformer" / "seed_42"
    attn_path = seed_42 / "test_attn_weights.npz"
    if not attn_path.is_file():
        pytest.skip(
            "test_attn_weights.npz not present. Rerun train.py without "
            "--no-save-attn to generate it."
        )

    rc = interp.main(
        [
            "--run-dir",
            str(seed_42),
            "--rf-importance",
            str(REPO / "results" / "baseline" / "rf_feature_importance.csv"),
            "--figures-dir",
            str(tmp_path / "figures"),
            "--output-json",
            str(tmp_path / "interpret.json"),
        ]
    )
    assert rc == 0

    for name in (
        "attention_rollout.png",
        "cls_feature_importance.png",
        "attention_per_head.png",
        "defaulter_vs_nondefaulter_attention.png",
        "feature_importance_comparison.png",
    ):
        path = tmp_path / "figures" / name
        assert path.is_file()

    summary = json.loads((tmp_path / "interpret.json").read_text())
    assert "attention_scores" in summary
    assert "per_head_entropy" in summary
