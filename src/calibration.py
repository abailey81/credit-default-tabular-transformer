"""
calibration.py — Phase 11: post-hoc probability calibration.

The Phase 8 comparison table reveals a ~20× gap in Expected Calibration
Error between the transformer (ECE ≈ 0.26) and the tuned Random Forest
(ECE ≈ 0.012) even though their AUC-ROC scores are within 0.005 of each
other. In a credit-risk deployment this calibration gap is load-bearing:
Basel-III capital reserves scale with ``P(default_i) · LGD · EAD_i``, so a
miscalibrated probability directly misprices risk.

This module implements the four calibrators discussed in Plan §13.3
(A20) — temperature scaling, Platt scaling, and isotonic regression,
plus a no-op "identity" baseline — fits them on the held-out validation
split, and reports the per-calibrator ECE / MCE / Brier / Brier-
decomposition on the test split. Every calibrator exposes the same
``fit(y_val, p_val) → transform(p_test) → p_test_cal`` interface so they
plug interchangeably into the downstream pipeline.

Design choices & implementation notes
-------------------------------------
* Temperature scaling operates in log-odds space. We recover logits from
  stored probabilities via ``logit = log(p / (1-p))`` — exact up to
  numerical precision after clipping away 0.0/1.0. We minimise the
  negative log-likelihood (``-Σ y·log σ(z/T) + (1-y)·log σ(-z/T)``) with
  bounded Brent search on ``T ∈ [0.05, 10]``. This is Guo et al. (2017)
  "On Calibration of Modern Neural Networks", single-parameter form.
* Platt scaling is ``σ(a·logit + b)`` where ``(a, b)`` come from a
  logistic regression on the validation logits — implemented directly
  rather than via sklearn so the dependency surface stays minimal.
* Isotonic regression via sklearn; we cap at ``y_min=0, y_max=1`` and
  set ``out_of_bounds="clip"`` so the function is well-defined outside
  the training support.
* Brier decomposition follows Murphy (1973): ``Brier = reliability –
  resolution + uncertainty``, with equal-width bins on the predicted
  probability. We expose the three components individually; all are
  positive, and ``resolution - reliability`` is a proper scoring-rule
  "skill score".
* ECE offered in both equal-width (default) and equal-mass ("quantile")
  bins — equal-mass better handles skewed probability distributions where
  most predictions sit in the low-probability region and equal-width bins
  become near-empty there.

Artefacts written by ``main()``
-------------------------------
* ``results/calibration/calibration_metrics.csv`` — one row per
  (run, calibrator) pair with ECE_equal_width, ECE_equal_mass, MCE,
  Brier, reliability, resolution, uncertainty, AUC-ROC (unchanged, sanity
  check that calibration doesn't shift the ranking).
* ``results/calibration/calibration_summary.json`` — the full structured
  output for downstream programmatic consumption.
* ``figures/calibration_reliability.png`` — reliability diagrams
  pre/post calibration per calibrator.
* ``figures/calibration_ece_bar.png`` — grouped bar chart of ECE per
  run × calibrator.

References: Plan §13 (calibration), §13.2 (ECE/MCE), §13.3 / Ablation A20.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

logger = logging.getLogger(__name__)

EPS = 1e-7

# ──────────────────────────────────────────────────────────────────────────────
# Calibrators — all share fit/transform so they plug interchangeably.
# ──────────────────────────────────────────────────────────────────────────────


def _clip_probs(p: np.ndarray) -> np.ndarray:
    return np.clip(p.astype(np.float64), EPS, 1.0 - EPS)


def _probs_to_logits(p: np.ndarray) -> np.ndarray:
    p = _clip_probs(p)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable elementwise sigmoid.
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


class IdentityCalibrator:
    """No-op baseline. Useful for uniform reporting."""
    name = "identity"

    def fit(self, y: np.ndarray, p: np.ndarray) -> "IdentityCalibrator":
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return _clip_probs(p)


class TemperatureScaling:
    """Single-parameter logit scaling: ``σ(logit / T)``. Guo et al. 2017."""

    name = "temperature"

    def __init__(self, bounds: Tuple[float, float] = (0.05, 10.0)):
        self.bounds = bounds
        self.temperature_: Optional[float] = None

    def _nll(self, T: float, logits: np.ndarray, y: np.ndarray) -> float:
        if T <= 0:
            return float("inf")
        z = logits / T
        # log σ(z) = -softplus(-z);  log(1-σ(z)) = -softplus(z)
        # Use np.logaddexp for numerical stability.
        log_sig = -np.logaddexp(0.0, -z)
        log_1_sig = -np.logaddexp(0.0, z)
        return -float(np.sum(y * log_sig + (1.0 - y) * log_1_sig))

    def fit(self, y: np.ndarray, p: np.ndarray) -> "TemperatureScaling":
        y = y.astype(np.float64)
        logits = _probs_to_logits(p)
        res = minimize_scalar(
            self._nll,
            args=(logits, y),
            bounds=self.bounds,
            method="bounded",
            options={"xatol": 1e-5},
        )
        self.temperature_ = float(res.x)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if self.temperature_ is None:
            raise RuntimeError("TemperatureScaling.fit must be called first")
        return _sigmoid(_probs_to_logits(p) / self.temperature_)


class PlattScaling:
    """Two-parameter logistic regression on the model's logits.
    Fits ``σ(a·logit + b)`` by IRLS-free Newton descent on the NLL."""

    name = "platt"

    def __init__(self, max_iter: int = 100, tol: float = 1e-7):
        self.max_iter = max_iter
        self.tol = tol
        self.a_: Optional[float] = None
        self.b_: Optional[float] = None

    def _nll(self, a: float, b: float, logits: np.ndarray, y: np.ndarray) -> float:
        z = a * logits + b
        log_sig = -np.logaddexp(0.0, -z)
        log_1_sig = -np.logaddexp(0.0, z)
        return -float(np.sum(y * log_sig + (1.0 - y) * log_1_sig))

    def fit(self, y: np.ndarray, p: np.ndarray) -> "PlattScaling":
        y = y.astype(np.float64)
        x = _probs_to_logits(p)
        # Gradient descent on (a, b) — the NLL of a 1D logistic regression
        # is convex, so any bounded descent converges. Use scipy's BFGS for
        # robustness rather than rolling our own line search.
        from scipy.optimize import minimize
        res = minimize(
            lambda params: self._nll(params[0], params[1], x, y),
            x0=np.array([1.0, 0.0]),
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )
        self.a_ = float(res.x[0])
        self.b_ = float(res.x[1])
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if self.a_ is None:
            raise RuntimeError("PlattScaling.fit must be called first")
        z = self.a_ * _probs_to_logits(p) + self.b_
        return _sigmoid(z)


class IsotonicCalibrator:
    """Monotone-non-decreasing piecewise-constant map fitted on (p, y)."""

    name = "isotonic"

    def __init__(self):
        self._iso = IsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip",
        )
        self._fitted = False

    def fit(self, y: np.ndarray, p: np.ndarray) -> "IsotonicCalibrator":
        self._iso.fit(_clip_probs(p), y.astype(np.float64))
        self._fitted = True
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("IsotonicCalibrator.fit must be called first")
        return _clip_probs(self._iso.predict(_clip_probs(p)))


CALIBRATORS: Dict[str, type] = {
    "identity": IdentityCalibrator,
    "temperature": TemperatureScaling,
    "platt": PlattScaling,
    "isotonic": IsotonicCalibrator,
}


# ──────────────────────────────────────────────────────────────────────────────
# Calibration metrics
# ──────────────────────────────────────────────────────────────────────────────


def _bin_indices(p: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    """Assign each probability to a bin.

    * ``equal_width`` — bins span [0, 1] in equal intervals.
    * ``equal_mass`` (aka quantile) — each bin has approximately the same
      number of samples. Better when probabilities are concentrated in
      one region.
    """
    if strategy == "equal_width":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "equal_mass":
        edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
        # Avoid collapsed bins on heavily-duplicated probabilities.
        edges[0] = 0.0
        edges[-1] = 1.0
        edges = np.unique(edges)
    else:
        raise ValueError(f"strategy must be equal_width or equal_mass, got {strategy!r}")
    idx = np.digitize(p, edges[1:-1], right=False)
    return np.clip(idx, 0, len(edges) - 2)


def expected_calibration_error(
    y_true: np.ndarray,
    p: np.ndarray,
    n_bins: int = 10,
    strategy: str = "equal_width",
) -> float:
    """Sample-weighted ECE = Σ_b (n_b/N) · |acc_b − conf_b|."""
    y = y_true.astype(np.float64)
    p = p.astype(np.float64)
    bins = _bin_indices(p, n_bins, strategy)
    n = len(y)
    ece = 0.0
    for b in np.unique(bins):
        mask = bins == b
        if mask.sum() == 0:
            continue
        acc = y[mask].mean()
        conf = p[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def maximum_calibration_error(
    y_true: np.ndarray,
    p: np.ndarray,
    n_bins: int = 10,
    strategy: str = "equal_width",
) -> float:
    """Worst-case bin gap — relevant for safety-critical thresholding."""
    y = y_true.astype(np.float64)
    p = p.astype(np.float64)
    bins = _bin_indices(p, n_bins, strategy)
    gaps: List[float] = []
    for b in np.unique(bins):
        mask = bins == b
        if mask.sum() == 0:
            continue
        gaps.append(abs(y[mask].mean() - p[mask].mean()))
    return float(max(gaps)) if gaps else float("nan")


@dataclass
class BrierDecomposition:
    """Murphy (1973) three-component decomposition of the Brier score.

    ``Brier = reliability − resolution + uncertainty``.

    * ``reliability`` (lower is better): weighted squared gap between
      predicted confidence and observed frequency within each bin.
    * ``resolution`` (higher is better): how much bin outcomes vary
      around the overall base rate — the signal the forecaster extracts.
    * ``uncertainty``: dataset-level base-rate entropy-equivalent (Ȳ(1−Ȳ)).
      Independent of the forecaster; reported for completeness.
    """

    reliability: float
    resolution: float
    uncertainty: float
    brier: float


def brier_decomposition(
    y_true: np.ndarray, p: np.ndarray, n_bins: int = 10,
) -> BrierDecomposition:
    y = y_true.astype(np.float64)
    p = p.astype(np.float64)
    n = len(y)
    base_rate = y.mean()
    bins = _bin_indices(p, n_bins, "equal_width")
    reliability = 0.0
    resolution = 0.0
    for b in np.unique(bins):
        mask = bins == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        conf_b = p[mask].mean()
        acc_b = y[mask].mean()
        reliability += (n_b / n) * (conf_b - acc_b) ** 2
        resolution += (n_b / n) * (acc_b - base_rate) ** 2
    uncertainty = float(base_rate * (1.0 - base_rate))
    brier = float(brier_score_loss(y.astype(int), p))
    return BrierDecomposition(
        reliability=float(reliability),
        resolution=float(resolution),
        uncertainty=uncertainty,
        brier=brier,
    )


def calibration_metric_bundle(
    y_true: np.ndarray, p: np.ndarray, n_bins: int = 10,
) -> Dict[str, float]:
    decomp = brier_decomposition(y_true, p, n_bins=n_bins)
    return {
        "auc_roc": float(roc_auc_score(y_true.astype(int), p)),
        "ece_equal_width": expected_calibration_error(
            y_true, p, n_bins=n_bins, strategy="equal_width"),
        "ece_equal_mass": expected_calibration_error(
            y_true, p, n_bins=n_bins, strategy="equal_mass"),
        "mce": maximum_calibration_error(y_true, p, n_bins=n_bins),
        "brier": decomp.brier,
        "reliability": decomp.reliability,
        "resolution": decomp.resolution,
        "uncertainty": decomp.uncertainty,
        "brier_skill_score": (
            float((decomp.resolution - decomp.reliability) / decomp.uncertainty)
            if decomp.uncertainty > 0 else float("nan")
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline: fit on val, score on test, record everything.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class CalibrationResult:
    run_name: str
    calibrator: str
    metrics: Dict[str, float]
    params: Dict[str, Any] = field(default_factory=dict)


def calibrate_and_score(
    y_val: np.ndarray, p_val: np.ndarray,
    y_test: np.ndarray, p_test: np.ndarray,
    *,
    calibrator_names: Sequence[str] = ("identity", "temperature", "platt", "isotonic"),
    n_bins: int = 10,
    run_name: str = "run",
) -> List[CalibrationResult]:
    results: List[CalibrationResult] = []
    for name in calibrator_names:
        if name not in CALIBRATORS:
            raise ValueError(f"Unknown calibrator: {name}")
        cal = CALIBRATORS[name]()
        cal.fit(y_val, p_val)
        p_test_cal = cal.transform(p_test)
        metrics = calibration_metric_bundle(y_test, p_test_cal, n_bins=n_bins)
        params: Dict[str, Any] = {}
        if isinstance(cal, TemperatureScaling):
            params["temperature"] = cal.temperature_
        elif isinstance(cal, PlattScaling):
            params["a"] = cal.a_
            params["b"] = cal.b_
        results.append(CalibrationResult(
            run_name=run_name, calibrator=name, metrics=metrics, params=params,
        ))
    return results


def results_to_dataframe(results: Sequence[CalibrationResult]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for r in results:
        row: Dict[str, Any] = {"run": r.run_name, "calibrator": r.calibrator}
        row.update(r.metrics)
        for k, v in r.params.items():
            row[f"param_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────


def _reliability_points(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    confs: List[float] = []
    accs: List[float] = []
    counts: List[int] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if not mask.any():
            continue
        confs.append(p[mask].mean())
        accs.append(y[mask].mean())
        counts.append(int(mask.sum()))
    return np.array(confs), np.array(accs), np.array(counts)


def plot_reliability_panel(
    panels: List[Tuple[str, np.ndarray, np.ndarray]],
    out_path: Path,
    n_bins: int = 10,
) -> None:
    """One subplot per (label, y_true, p) tuple. Diagonal = perfect."""
    n = len(panels)
    ncols = min(4, max(1, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows),
                             squeeze=False)
    for ax, (label, y, p) in zip(axes.flat, panels):
        ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="#999999")
        confs, accs, _ = _reliability_points(y, p, n_bins=n_bins)
        ax.plot(confs, accs, marker="o", color="#0072B2", lw=1.6, ms=5)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.grid(alpha=0.3)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.suptitle(f"Reliability diagrams ({n_bins} bins)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ece_bar(df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart of ECE per (run, calibrator) — visual summary."""
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = df.pivot(index="run", columns="calibrator", values="ece_equal_width")
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("ECE (10 equal-width bins)")
    ax.set_title("Calibration error by run × calibrator — lower is better")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def _load_run_val_test(run_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    """Return y/p on val and test splits for a single transformer run, or
    None if the run's artefacts are missing."""
    run_dir = Path(run_dir)
    val = run_dir / "val_predictions.npz"
    test = run_dir / "test_predictions.npz"
    if not (val.is_file() and test.is_file()):
        return None
    v = np.load(val)
    t = np.load(test)
    return {
        "run_name": run_dir.name,
        "y_val": v["y_true"],
        "p_val": v["y_prob"],
        "y_test": t["y_true"],
        "p_test": t["y_prob"],
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Phase 11 calibration: fit temperature, Platt, and isotonic "
            "calibrators on each run's validation split, then score every "
            "post-hoc-calibrated probability vector on the test split."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--runs", nargs="*", type=Path,
        default=[
            Path("results/transformer/seed_42"),
            Path("results/transformer/seed_1"),
            Path("results/transformer/seed_2"),
            Path("results/transformer/seed_42_mtlm_finetune"),
        ],
        help="Training run directories, each containing val_predictions.npz "
             "+ test_predictions.npz.",
    )
    p.add_argument("--rf-dir", type=Path, default=Path("results/rf"),
                   help="Optional: RF artefact directory with test_predictions.npz. "
                        "Included in the table as an additional row for comparison.")
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--output-dir", type=Path, default=Path("results/calibration"))
    p.add_argument("--figures-dir", type=Path, default=Path("figures"))
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args(argv)

    all_results: List[CalibrationResult] = []
    reliability_panels: List[Tuple[str, np.ndarray, np.ndarray]] = []

    for run_dir in args.runs:
        payload = _load_run_val_test(run_dir)
        if payload is None:
            logger.warning("Skipping %s — predictions missing", run_dir)
            continue
        results = calibrate_and_score(
            payload["y_val"], payload["p_val"],
            payload["y_test"], payload["p_test"],
            n_bins=args.n_bins,
            run_name=payload["run_name"],
        )
        all_results.extend(results)
        # Add a pre-calibration reliability panel + one post-temperature panel
        # per run for a visual before/after.
        reliability_panels.append(
            (f"{payload['run_name']} (raw)",
             payload["y_test"], payload["p_test"]))
        ts = TemperatureScaling().fit(payload["y_val"], payload["p_val"])
        reliability_panels.append(
            (f"{payload['run_name']} (T={ts.temperature_:.2f})",
             payload["y_test"], ts.transform(payload["p_test"])))

    # RF — treat as a run with no calibration applied (it's already a
    # well-calibrated classifier) so the report shows a clean baseline.
    rf_dir = Path(args.rf_dir)
    if (rf_dir / "test_predictions.npz").is_file():
        rf = np.load(rf_dir / "test_predictions.npz")
        metrics = calibration_metric_bundle(rf["y_true"], rf["y_prob"], n_bins=args.n_bins)
        all_results.append(CalibrationResult(
            run_name="rf_tuned", calibrator="identity", metrics=metrics,
        ))
        reliability_panels.append(("rf_tuned (raw)", rf["y_true"], rf["y_prob"]))

    if not all_results:
        logger.error("No results produced — every run directory was missing predictions.")
        return 1

    df = results_to_dataframe(all_results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "calibration_metrics.csv"
    json_path = args.output_dir / "calibration_summary.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps([
        {"run": r.run_name, "calibrator": r.calibrator,
         "metrics": r.metrics, "params": r.params}
        for r in all_results
    ], indent=2, default=str))

    fig_rel = args.figures_dir / "calibration_reliability.png"
    fig_bar = args.figures_dir / "calibration_ece_bar.png"
    plot_reliability_panel(reliability_panels, fig_rel, n_bins=args.n_bins)
    plot_ece_bar(df, fig_bar)

    logger.info("Calibration metrics → %s", csv_path)
    logger.info("Reliability panel   → %s", fig_rel)
    logger.info("ECE bar chart       → %s", fig_bar)

    print()
    display_cols = [
        "run", "calibrator", "auc_roc",
        "ece_equal_width", "ece_equal_mass", "mce", "brier",
        "reliability", "resolution", "brier_skill_score",
    ]
    print(df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
