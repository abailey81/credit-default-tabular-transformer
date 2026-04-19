"""visualise.py — PNG output, reliability-bin edges, CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent

from src.evaluation import visualise as vis  # noqa: E402


@pytest.fixture
def fake_runs() -> List[Dict[str, Any]]:
    rng = np.random.default_rng(0)
    n = 500
    y_true = rng.integers(0, 2, size=n)
    return [
        {
            "run_name": "seed_42",
            "y_true": y_true,
            "y_prob": np.clip(0.3 + 0.4 * y_true + 0.2 * rng.standard_normal(n), 0, 1),
            "y_pred": ((0.3 + 0.4 * y_true) > 0.5).astype(int),
            "auc_roc": 0.78,
            "auc_pr": 0.56,
        },
        {
            "run_name": "seed_42_mtlm_finetune",
            "y_true": y_true,
            "y_prob": np.clip(0.2 + 0.5 * y_true + 0.1 * rng.standard_normal(n), 0, 1),
            "y_pred": ((0.2 + 0.5 * y_true) > 0.5).astype(int),
            "auc_roc": 0.79,
            "auc_pr": 0.57,
        },
    ]


@pytest.fixture
def fake_rf_ref() -> Dict[str, float]:
    return {"auc_roc": 0.7845, "auc_pr": 0.5673}


def test_plot_roc_curves_writes_png(tmp_path: Path, fake_runs, fake_rf_ref):
    out = tmp_path / "roc.png"
    vis.plot_roc_curves(fake_runs, fake_rf_ref, out)
    assert out.is_file() and out.stat().st_size > 0


def test_plot_roc_curves_without_rf_reference(tmp_path: Path, fake_runs):
    out = tmp_path / "roc.png"
    vis.plot_roc_curves(fake_runs, None, out)
    assert out.is_file() and out.stat().st_size > 0


def test_plot_pr_curves_writes_png(tmp_path: Path, fake_runs, fake_rf_ref):
    out = tmp_path / "pr.png"
    vis.plot_pr_curves(fake_runs, fake_rf_ref, out)
    assert out.is_file() and out.stat().st_size > 0


def test_plot_confusion_matrices_writes_png(tmp_path: Path, fake_runs):
    out = tmp_path / "cm.png"
    vis.plot_confusion_matrices(fake_runs, out)
    assert out.is_file() and out.stat().st_size > 0


def test_plot_reliability_diagrams_writes_png(tmp_path: Path, fake_runs):
    out = tmp_path / "rel.png"
    vis.plot_reliability_diagrams(fake_runs, out)
    assert out.is_file() and out.stat().st_size > 0


def test_reliability_bins_handles_empty_bins():
    y_true = np.array([0, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.15, 0.05, 0.12, 0.08])  # all in one bin
    means, rates = vis._reliability_bins(y_true, y_prob, n_bins=10)
    assert len(means) == len(rates)
    assert len(means) >= 1
    assert not np.isnan(means).any()
    assert not np.isnan(rates).any()


def test_reliability_bins_endpoint_handling():
    # p = 1.0 must land in the final bin, not vanish between boundaries
    y_true = np.array([1, 0])
    y_prob = np.array([1.0, 0.0])
    means, rates = vis._reliability_bins(y_true, y_prob, n_bins=10)
    assert len(means) == 2


def test_build_parser_help_renders_without_error(capsys):
    parser = vis._build_parser()
    parser.print_help()
    captured = capsys.readouterr()
    assert captured.out.startswith("usage:")
    assert "--runs" in captured.out
    assert "--output-dir" in captured.out


def test_main_produces_all_five_figures(tmp_path: Path):
    seed_42 = REPO / "results" / "transformer" / "seed_42"
    if not (seed_42 / "test_predictions.npz").is_file():
        pytest.skip("no committed training artefacts; skipping end-to-end test")

    rc = vis.main(
        [
            "--runs",
            str(seed_42),
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0

    for name in (
        "roc_curves_transformer.png",
        "pr_curves_transformer.png",
        "confusion_matrices_transformer.png",
        "training_curves.png",
        "reliability_diagrams.png",
    ):
        path = tmp_path / name
        assert path.is_file(), f"Missing {name}"
        assert path.stat().st_size > 0, f"{name} is empty"
