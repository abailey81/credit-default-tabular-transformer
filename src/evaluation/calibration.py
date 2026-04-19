"""Post-hoc calibrators (identity / temperature / Platt / isotonic) plus
ECE, MCE, and Murphy's Brier decomposition. Equal-mass bins are offered
alongside equal-width because our probs cluster low and the top
equal-width bins end up near-empty."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

logger = logging.getLogger(__name__)

__all__ = [
    # calibrators
    "IdentityCalibrator",
    "TemperatureScaling",
    "PlattScaling",
    "IsotonicCalibrator",
    # result types
    "BrierDecomposition",
    "CalibrationResult",
    # metrics + drivers
    "expected_calibration_error",
    "maximum_calibration_error",
    "brier_decomposition",
    "calibration_metric_bundle",
    "calibrate_and_score",
    "results_to_dataframe",
    "plot_reliability_panel",
    "plot_ece_bar",
    "main",
    # constants
    "CALIBRATORS",
    "EPS",
]

EPS = 1e-7


def _clip_probs(p: np.ndarray) -> np.ndarray:
    return np.clip(p.astype(np.float64), EPS, 1.0 - EPS)


def _probs_to_logits(p: np.ndarray) -> np.ndarray:
    p = _clip_probs(p)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # stable sigmoid
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


class IdentityCalibrator:
    """No-op. Here so every calibrator shares the fit/transform API."""

    name = "identity"

    def fit(self, y: np.ndarray, p: np.ndarray) -> "IdentityCalibrator":
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return _clip_probs(p)


class TemperatureScaling:
    """σ(z / T). Guo+ 2017."""

    name = "temperature"

    def __init__(self, bounds: Tuple[float, float] = (0.05, 10.0)):
        self.bounds = bounds
        self.temperature_: Optional[float] = None

    def _nll(self, T: float, logits: np.ndarray, y: np.ndarray) -> float:
        if T <= 0:
            return float("inf")
        z = logits / T
        # logσ(z) = -softplus(-z); log(1-σ(z)) = -softplus(z)
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
    """σ(a·z + b). LR on the model's logits."""

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
        # convex in (a, b); L-BFGS-B is plenty, no need for IRLS
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
    """Monotone step-fn fit on (p, y)."""

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


def _bin_indices(p: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    """Bin index per row, equal-width or equal-mass."""
    if strategy == "equal_width":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "equal_mass":
        edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
        # heavy ties can collapse bins — dedupe after pinning the endpoints
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
    """ECE = Σ_b (n_b / N) |acc_b − conf_b|."""
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
    """Worst-bin gap. Matters when one decision threshold is safety-critical."""
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
    """Brier = reliability − resolution + uncertainty (Murphy 1973).

    `uncertainty` is base-rate-only, so `resolution − reliability` is the
    skill score."""

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
    """Reliability grid, one subplot per panel. Dashed diagonal = perfect."""
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
    """ECE grouped bars: run × calibrator."""
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = df.pivot(index="run", columns="calibrator", values="ece_equal_width")
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("ECE (10 equal-width bins)")
    ax.set_title("Calibration error by run x calibrator (lower is better)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _load_run_val_test(run_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    """(y, p) on val+test, or None if the run hasn't produced predictions yet."""
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
            "Fit T/Platt/isotonic on each run's val split, score on test."
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
        help="Run dirs, each with val_predictions.npz + test_predictions.npz.",
    )
    p.add_argument("--rf-dir", type=Path, default=Path("results/baseline/rf"),
                   help="Optional RF run dir. Adds a comparison row.")
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--output-dir", type=Path, default=Path("results/evaluation/calibration"))
    p.add_argument("--figures-dir", type=Path, default=Path("figures/evaluation/calibration"))
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
            logger.warning("Skipping %s, predictions missing", run_dir)
            continue
        results = calibrate_and_score(
            payload["y_val"], payload["p_val"],
            payload["y_test"], payload["p_test"],
            n_bins=args.n_bins,
            run_name=payload["run_name"],
        )
        all_results.extend(results)
        # one raw + one post-T panel per run
        reliability_panels.append(
            (f"{payload['run_name']} (raw)",
             payload["y_test"], payload["p_test"]))
        ts = TemperatureScaling().fit(payload["y_val"], payload["p_val"])
        reliability_panels.append(
            (f"{payload['run_name']} (T={ts.temperature_:.2f})",
             payload["y_test"], ts.transform(payload["p_test"])))

    # RF is already ~calibrated, so stick it in under identity
    rf_dir = Path(args.rf_dir)
    if (rf_dir / "test_predictions.npz").is_file():
        rf = np.load(rf_dir / "test_predictions.npz")
        metrics = calibration_metric_bundle(rf["y_true"], rf["y_prob"], n_bins=args.n_bins)
        all_results.append(CalibrationResult(
            run_name="rf_tuned", calibrator="identity", metrics=metrics,
        ))
        reliability_panels.append(("rf_tuned (raw)", rf["y_true"], rf["y_prob"]))

    if not all_results:
        logger.error("No results produced; every run directory was missing predictions.")
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

    logger.info("Calibration metrics -> %s", csv_path)
    logger.info("Reliability panel   -> %s", fig_rel)
    logger.info("ECE bar chart       -> %s", fig_bar)

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
