"""calibration.py — synthetic unit tests + a committed-artefact e2e."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import calibration as cal  # noqa: E402


@pytest.fixture
def well_calibrated():
    rng = np.random.default_rng(0)
    n = 5000
    p = rng.uniform(0.0, 1.0, size=n)
    y = rng.binomial(1, p).astype(int)
    return y, p


@pytest.fixture
def miscalibrated():
    # overconfident by ×2: scores are σ(true_logit * 2)
    rng = np.random.default_rng(1)
    n = 5000
    true_logits = rng.normal(0.0, 1.5, size=n)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-true_logits))).astype(int)
    p = 1.0 / (1.0 + np.exp(-true_logits * 2.0))
    return y, p


def test_ece_zero_for_well_calibrated(well_calibrated):
    y, p = well_calibrated
    ece = cal.expected_calibration_error(y, p, n_bins=10)
    assert 0 <= ece < 0.03, f"ECE={ece:.4f} on well-calibrated sample"


def test_ece_positive_for_miscalibrated(miscalibrated):
    y, p = miscalibrated
    ece = cal.expected_calibration_error(y, p, n_bins=10)
    assert ece > 0.05, f"ECE={ece:.4f} too small for overconfident sample"


def test_mce_bounds(miscalibrated):
    y, p = miscalibrated
    mce = cal.maximum_calibration_error(y, p, n_bins=10)
    ece = cal.expected_calibration_error(y, p, n_bins=10)
    assert mce >= ece


def test_brier_decomposition_identity(miscalibrated):
    y, p = miscalibrated
    d = cal.brier_decomposition(y, p, n_bins=10)
    approx = d.reliability - d.resolution + d.uncertainty
    assert abs(approx - d.brier) < 0.02, (
        f"Brier decomposition identity broken: {approx} vs {d.brier}"
    )


def test_equal_mass_bins_never_empty():
    rng = np.random.default_rng(2)
    p = rng.beta(1, 9, size=1000)
    y = rng.binomial(1, p).astype(int)
    bins = cal._bin_indices(p, n_bins=10, strategy="equal_mass")
    counts = np.bincount(bins)
    assert (counts > 0).all(), f"empty bins found: {counts}"


def test_identity_roundtrip(miscalibrated):
    y, p = miscalibrated
    c = cal.IdentityCalibrator().fit(y, p)
    assert np.allclose(c.transform(p), np.clip(p, cal.EPS, 1 - cal.EPS))


def test_temperature_reduces_ece_on_overconfident(miscalibrated):
    y, p = miscalibrated
    ece_raw = cal.expected_calibration_error(y, p)
    ts = cal.TemperatureScaling().fit(y, p)
    p_cal = ts.transform(p)
    ece_cal = cal.expected_calibration_error(y, p_cal)
    assert ece_cal < ece_raw * 0.7, (
        f"Temperature didn't help: {ece_raw:.4f} -> {ece_cal:.4f}"
    )
    assert ts.temperature_ > 1.0


def test_platt_handles_weak_signal():
    rng = np.random.default_rng(3)
    n = 2000
    y = rng.binomial(1, 0.3, size=n).astype(int)
    p = rng.uniform(0.0, 1.0, size=n)
    platt = cal.PlattScaling().fit(y, p)
    p_cal = platt.transform(p)
    assert abs(p_cal.mean() - y.mean()) < 0.05
    assert p_cal.std() < 0.2


def test_isotonic_monotone(miscalibrated):
    y, p = miscalibrated
    iso = cal.IsotonicCalibrator().fit(y, p)
    grid = np.linspace(0.0, 1.0, 50)
    p_cal = iso.transform(grid)
    assert (np.diff(p_cal) >= -1e-9).all()


def test_calibrator_not_fitted_raises():
    ts = cal.TemperatureScaling()
    with pytest.raises(RuntimeError):
        ts.transform(np.array([0.5, 0.6]))
    platt = cal.PlattScaling()
    with pytest.raises(RuntimeError):
        platt.transform(np.array([0.5, 0.6]))
    iso = cal.IsotonicCalibrator()
    with pytest.raises(RuntimeError):
        iso.transform(np.array([0.5, 0.6]))


def test_calibrate_and_score_produces_every_calibrator(miscalibrated):
    y, p = miscalibrated
    n = len(y) // 2
    results = cal.calibrate_and_score(
        y[:n], p[:n], y[n:], p[n:],
        run_name="unit",
    )
    names = [r.calibrator for r in results]
    assert names == ["identity", "temperature", "platt", "isotonic"]
    ece_identity = next(r for r in results if r.calibrator == "identity").metrics["ece_equal_width"]
    for r in results:
        if r.calibrator == "identity":
            continue
        if r.calibrator in ("platt", "isotonic"):
            assert r.metrics["ece_equal_width"] < ece_identity, (
                f"{r.calibrator} didn't improve ECE"
            )


def test_probs_to_logits_invertible():
    p = np.array([0.1, 0.5, 0.9])
    z = cal._probs_to_logits(p)
    p_back = cal._sigmoid(z)
    assert np.allclose(p_back, p, atol=1e-6)


def test_main_writes_artefacts(tmp_path):
    seed_42 = REPO / "results" / "transformer" / "seed_42"
    if not (seed_42 / "val_predictions.npz").is_file():
        pytest.skip("no committed transformer runs; skipping e2e test")
    rc = cal.main([
        "--runs", str(seed_42),
        "--output-dir", str(tmp_path / "out"),
        "--figures-dir", str(tmp_path / "fig"),
    ])
    assert rc == 0
    assert (tmp_path / "out" / "calibration_metrics.csv").is_file()
    summary = json.loads(
        (tmp_path / "out" / "calibration_summary.json").read_text()
    )
    assert len(summary) >= 3
