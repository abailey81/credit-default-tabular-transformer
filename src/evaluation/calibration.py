"""Post-hoc probability calibration for the credit-default classifier.

This module implements the four calibrators we benchmark against in §4 of
the report -- identity (uncalibrated baseline), temperature scaling,
Platt scaling, and isotonic regression -- together with the metrics we
use to score them: expected and maximum calibration error (ECE / MCE)
and Murphy's reliability / resolution / uncertainty decomposition of
the Brier score.

Why two ECE binning strategies? The raw transformer probabilities on this
dataset cluster heavily in ``[0, 0.3]``; the top equal-width bins end up
near-empty, which makes equal-width ECE look artificially low (empty bins
contribute nothing). We therefore report equal-width ECE (the number most
papers quote, Guo+ 2017) *and* equal-mass ECE (Nixon+ 2019) side by side
-- the equal-mass variant is usually the worse -- and let the reader
choose their poison.

Key public symbols
------------------
* ``IdentityCalibrator`` / ``TemperatureScaling`` / ``PlattScaling`` /
  ``IsotonicCalibrator`` -- shared ``fit`` / ``transform`` API.
* ``expected_calibration_error`` / ``maximum_calibration_error``.
* ``brier_decomposition`` -- reliability / resolution / uncertainty.
* ``calibrate_and_score`` -- the driver that produces one row per
  (run, calibrator) pair.

Reference: C. Guo, G. Pleiss, Y. Sun, K. Weinberger, *On Calibration of
Modern Neural Networks*, ICML 2017. Equal-mass binning follows
J. Nixon et al., *Measuring Calibration in Deep Learning*, CVPR-W 2019.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

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

#: Clamp bound used everywhere probabilities get turned into logits. Below
#: this (or above ``1 - EPS``) ``log(p / (1 - p))`` overflows float64 and
#: the optimiser starts chasing inf.
EPS = 1e-7


def _clip_probs(p: np.ndarray) -> np.ndarray:
    # Force float64 before clipping -- the saved predictions are float32
    # and the logit step below easily overflows the narrower dtype at
    # p ~ 1e-4 / 1 - 1e-4.
    return np.clip(p.astype(np.float64), EPS, 1.0 - EPS)


def _probs_to_logits(p: np.ndarray) -> np.ndarray:
    p = _clip_probs(p)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid split by sign of ``z``.

    Naive ``1 / (1 + exp(-z))`` overflows for ``z << 0`` because ``exp(-z)``
    blows up; the dual ``exp(z) / (1 + exp(z))`` mirrors the issue for
    ``z >> 0``. We pick the form that keeps the dominant exponent negative.
    """
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


class IdentityCalibrator:
    """No-op calibrator (the ``raw`` / uncalibrated baseline).

    Exists purely so every calibrator in the module shares the same
    ``.fit(y, p) -> self`` / ``.transform(p) -> p_cal`` API. The driver
    loop then doesn't need to special-case the "no calibration" row --
    it just fits and transforms like every other calibrator.

    The ``transform`` call still clips into ``[EPS, 1 - EPS]`` so that
    downstream log-based metrics (NLL, Brier with log-clipped inputs) can
    never hit a NaN. This is the one visible side-effect.
    """

    name = "identity"

    def fit(self, y: np.ndarray, p: np.ndarray) -> IdentityCalibrator:
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return _clip_probs(p)


class TemperatureScaling:
    """Single-parameter post-hoc calibrator (Guo+ 2017).

    Rescales logits by a scalar temperature ``T`` before the sigmoid::

        p_hat = sigma(z / T)

    where ``z = logit(p)`` and ``T`` is fit on the held-out validation
    set by minimising NLL. ``T > 1`` softens over-confident probabilities
    (the common failure mode on our transformer -- raw ECE ~0.26 on test);
    ``T < 1`` sharpens under-confident ones (rare here, mostly seen on RF).

    Compared with Platt or isotonic:

    * single parameter, so needs the least validation data;
    * strictly monotone, so the ROC curve is preserved exactly
      (AUC post-T is bit-identical to raw);
    * often under-fits severely miscalibrated classifiers -- our
      transformer post-T still sits at ECE ~0.22, while Platt drops it
      toward ~0.01. See ``results/evaluation/calibration/`` for the
      per-calibrator numbers.

    Parameters
    ----------
    bounds : tuple of (float, float)
        Search range for scipy's bounded Brent method. Default
        ``(0.05, 10.0)`` is wide enough for every failure mode we've seen
        on this dataset (T ~ 2-3 for the raw transformer, ~1.0 for RF)
        but narrow enough that the 1-D search doesn't wander into the
        numerical-blow-up tail.

    Attributes
    ----------
    temperature_ : float or None
        Fitted temperature after :meth:`fit`. ``None`` until ``fit`` runs.
    """

    name = "temperature"

    def __init__(self, bounds: tuple[float, float] = (0.05, 10.0)):
        self.bounds = bounds
        self.temperature_: Optional[float] = None

    def _nll(self, T: float, logits: np.ndarray, y: np.ndarray) -> float:
        """Negative log-likelihood of the data under ``sigma(logits / T)``.

        Uses ``log sigma(z) = -softplus(-z) = -logaddexp(0, -z)``, which
        is stable for both signs of ``z``. A direct
        ``-sum(y log p + (1-y) log(1-p))`` formulation underflows for
        confident-and-correct predictions where ``p`` is very close to 0
        or 1 -- we'd silently feed the optimiser NaNs.
        """
        if T <= 0:
            return float("inf")
        z = logits / T
        # logaddexp(0, x) = softplus(x); stable at large |x|.
        log_sig = -np.logaddexp(0.0, -z)
        log_1_sig = -np.logaddexp(0.0, z)
        return -float(np.sum(y * log_sig + (1.0 - y) * log_1_sig))

    def fit(self, y: np.ndarray, p: np.ndarray) -> TemperatureScaling:
        """Fit ``T`` by 1-D bounded Brent minimisation of NLL.

        Parameters
        ----------
        y : array, shape (N,)
            Binary labels (``{0, 1}``).
        p : array, shape (N,)
            Uncalibrated probabilities from the base model on the *validation*
            split -- never on the same data that trained the base model
            (would severely under-estimate T).

        Returns
        -------
        self : TemperatureScaling
            With ``temperature_`` populated.
        """
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
        """Apply the fitted temperature to fresh probabilities.

        Raises :class:`RuntimeError` if called before :meth:`fit` -- the
        alternative (silently returning raw ``p``) would make calibration
        bugs invisible in downstream metrics.
        """
        if self.temperature_ is None:
            raise RuntimeError("TemperatureScaling.fit must be called first")
        return _sigmoid(_probs_to_logits(p) / self.temperature_)


class PlattScaling:
    """Two-parameter post-hoc calibrator (Platt 1999).

    Fits a logistic regression ``sigma(a z + b)`` on the validation
    ``(logit(p), y)`` pairs. Strictly more flexible than temperature
    scaling (temperature is the ``b = 0, a = 1 / T`` special case) and
    typically wins when the classifier is miscalibrated both in
    temperature *and* in bias -- which is exactly the transformer's
    behaviour here (shifted toward the class-imbalance base rate).

    Parameters
    ----------
    max_iter : int
        L-BFGS-B iteration cap. The problem is convex and 2-D, so 100 is
        wildly sufficient -- left configurable for stress-testing only.
    tol : float
        Function-value tolerance for L-BFGS-B.

    Attributes
    ----------
    a_, b_ : float or None
        Fitted slope / intercept on the logit scale.
    """

    name = "platt"

    def __init__(self, max_iter: int = 100, tol: float = 1e-7):
        self.max_iter = max_iter
        self.tol = tol
        self.a_: Optional[float] = None
        self.b_: Optional[float] = None

    def _nll(self, a: float, b: float, logits: np.ndarray, y: np.ndarray) -> float:
        # Same stable-logsigmoid trick as TemperatureScaling._nll.
        z = a * logits + b
        log_sig = -np.logaddexp(0.0, -z)
        log_1_sig = -np.logaddexp(0.0, z)
        return -float(np.sum(y * log_sig + (1.0 - y) * log_1_sig))

    def fit(self, y: np.ndarray, p: np.ndarray) -> PlattScaling:
        y = y.astype(np.float64)
        x = _probs_to_logits(p)
        # NLL is convex in (a, b); L-BFGS-B converges in ~10 iters. Niculescu-Mizil
        # & Caruana (2005) argue for IRLS but that's overkill for a 2-D problem.
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
    """Non-parametric monotone step-function calibrator (Zadrozny & Elkan 2002).

    Fits a monotone piecewise-constant mapping ``p -> p_hat`` that
    minimises squared error under a non-decreasing constraint. Much more
    flexible than Platt / T-scaling -- can correct arbitrary monotone
    miscalibration -- at the cost of needing more validation data, since
    every unique probability value gets its own segment.

    Shipped with ``out_of_bounds="clip"`` so probabilities outside the
    validation range at test time map to the nearest observed bin rather
    than raising. In practice this matters only for the tails; the bulk
    of the test probabilities lands well inside the fitted range.
    """

    name = "isotonic"

    def __init__(self):
        self._iso = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            increasing=True,
            out_of_bounds="clip",
        )
        self._fitted = False

    def fit(self, y: np.ndarray, p: np.ndarray) -> IsotonicCalibrator:
        self._iso.fit(_clip_probs(p), y.astype(np.float64))
        self._fitted = True
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("IsotonicCalibrator.fit must be called first")
        # Clip twice: once on input (handles rare <EPS or >1-EPS probs that
        # upstream floats may produce) and once on output (isotonic can
        # hand back values very slightly outside [0, 1] due to float rounding).
        return _clip_probs(self._iso.predict(_clip_probs(p)))


#: Registry of all calibrators. String key is the short name used in the
#: CLI / CSV, value is the constructor. Drivers iterate this to stay
#: order-preserving without hard-coding the class list.
CALIBRATORS: dict[str, type] = {
    "identity": IdentityCalibrator,
    "temperature": TemperatureScaling,
    "platt": PlattScaling,
    "isotonic": IsotonicCalibrator,
}


def _bin_indices(p: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    """Assign a bin index to every probability.

    Parameters
    ----------
    p : array, shape (N,)
    n_bins : int
        Number of bins requested. For equal-mass this is an *upper* bound
        -- tied quantile edges collapse and we return fewer bins.
    strategy : {"equal_width", "equal_mass"}
        ``equal_width`` is Guo+ 2017 default; ``equal_mass`` (Nixon+ 2019)
        is the fairer option when the probability distribution is skewed,
        as it is here.

    Returns
    -------
    idx : array, shape (N,), int
        Bin index in ``[0, n_bins - 1]`` after clipping.
    """
    if strategy == "equal_width":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "equal_mass":
        edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
        # Many tied probabilities (e.g. when the calibrator saturates) can
        # collapse adjacent quantile edges. Pin the endpoints to [0, 1]
        # first so even a degenerate distribution has one valid bin.
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
    """Weighted mean absolute gap between confidence and accuracy per bin.

    Defined as::

        ECE = sum_b (n_b / N) * |acc_b - conf_b|

    where ``conf_b`` is the mean predicted probability in bin ``b`` and
    ``acc_b`` is the observed positive rate.

    Parameters
    ----------
    y_true : array, shape (N,)
        Binary labels.
    p : array, shape (N,)
        Predicted probabilities for the positive class.
    n_bins, strategy : see :func:`_bin_indices`.

    Returns
    -------
    float
        ECE in ``[0, 1]``. 0 = perfect calibration.
    """
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
    """Worst-bin |conf - acc| gap.

    Relevant when a specific decision threshold is safety-critical --
    the mean ECE can be low while a single tail bin is badly miscalibrated.
    """
    y = y_true.astype(np.float64)
    p = p.astype(np.float64)
    bins = _bin_indices(p, n_bins, strategy)
    gaps: list[float] = []
    for b in np.unique(bins):
        mask = bins == b
        if mask.sum() == 0:
            continue
        gaps.append(abs(y[mask].mean() - p[mask].mean()))
    return float(max(gaps)) if gaps else float("nan")


@dataclass
class BrierDecomposition:
    """Murphy (1973) decomposition of the Brier score.

    Identity::

        Brier = reliability - resolution + uncertainty

    where

    * ``reliability`` is the weighted squared gap between bin confidence
      and bin accuracy -- what ECE captures in absolute value, Brier in
      squared value. Smaller is better.
    * ``resolution`` is the weighted squared gap between bin accuracy
      and the overall base rate -- how sharply the model separates the
      classes. Larger is better.
    * ``uncertainty`` is the base-rate-only Brier, ``p(1 - p)``.
      Constant for a given dataset; a model that always predicts the
      base rate hits exactly ``Brier = uncertainty``.

    The *Brier skill score* ``(resolution - reliability) / uncertainty``
    lives on this dataclass via :func:`calibration_metric_bundle`.
    """

    reliability: float
    resolution: float
    uncertainty: float
    brier: float


def brier_decomposition(
    y_true: np.ndarray,
    p: np.ndarray,
    n_bins: int = 10,
) -> BrierDecomposition:
    """Compute the reliability / resolution / uncertainty breakdown.

    Parameters
    ----------
    y_true : array, shape (N,)
    p : array, shape (N,)
    n_bins : int
        Equal-width bins; equal-mass is harder to interpret because the
        ``uncertainty`` term is base-rate only and stays constant.

    Returns
    -------
    BrierDecomposition

    Notes
    -----
    The three components sum to the Brier score up to a discretisation
    error (finite-bin approximation of the population identity). In
    practice the residual is ``< 1e-4`` for the 10-bin default on our
    test split of ~6k rows.
    """
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
        # Murphy identity, per-bin squared terms weighted by occupancy.
        reliability += (n_b / n) * (conf_b - acc_b) ** 2
        resolution += (n_b / n) * (acc_b - base_rate) ** 2
    uncertainty = float(base_rate * (1.0 - base_rate))
    # Separate sklearn call: gives the exact Brier (not the binned one),
    # so the decomposition and the headline number don't drift.
    brier = float(brier_score_loss(y.astype(int), p))
    return BrierDecomposition(
        reliability=float(reliability),
        resolution=float(resolution),
        uncertainty=uncertainty,
        brier=brier,
    )


def calibration_metric_bundle(
    y_true: np.ndarray,
    p: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Every calibration number we report for one (run, calibrator) pair.

    AUC is included so the caller can verify monotonicity-preserving
    calibrators (identity / temperature) haven't secretly changed the
    ranking -- if they have, something is broken upstream.
    """
    decomp = brier_decomposition(y_true, p, n_bins=n_bins)
    return {
        "auc_roc": float(roc_auc_score(y_true.astype(int), p)),
        "ece_equal_width": expected_calibration_error(
            y_true, p, n_bins=n_bins, strategy="equal_width"
        ),
        "ece_equal_mass": expected_calibration_error(
            y_true, p, n_bins=n_bins, strategy="equal_mass"
        ),
        "mce": maximum_calibration_error(y_true, p, n_bins=n_bins),
        "brier": decomp.brier,
        "reliability": decomp.reliability,
        "resolution": decomp.resolution,
        "uncertainty": decomp.uncertainty,
        "brier_skill_score": (
            float((decomp.resolution - decomp.reliability) / decomp.uncertainty)
            if decomp.uncertainty > 0
            else float("nan")
        ),
    }


@dataclass
class CalibrationResult:
    """One row of the calibration output table.

    ``params`` carries the fitted calibrator parameters (``T`` for
    temperature, ``(a, b)`` for Platt) so the CSV can be reconstructed
    or inspected without rerunning the fit.
    """

    run_name: str
    calibrator: str
    metrics: dict[str, float]
    params: dict[str, Any] = field(default_factory=dict)


def calibrate_and_score(
    y_val: np.ndarray,
    p_val: np.ndarray,
    y_test: np.ndarray,
    p_test: np.ndarray,
    *,
    calibrator_names: Sequence[str] = ("identity", "temperature", "platt", "isotonic"),
    n_bins: int = 10,
    run_name: str = "run",
) -> list[CalibrationResult]:
    """Fit each calibrator on (y_val, p_val); score on (y_test, p_test).

    Parameters
    ----------
    y_val, p_val : arrays
        Validation labels and raw probabilities. Calibrators are fit here.
    y_test, p_test : arrays
        Test split. Metrics are reported on this split only -- fitting on
        val and scoring on test is the whole point of post-hoc calibration.
    calibrator_names : sequence of str
        Must be keys of :data:`CALIBRATORS`.
    n_bins : int
        Number of ECE / Brier-decomposition bins.
    run_name : str
        Carried verbatim onto every result row.

    Returns
    -------
    list of CalibrationResult
        One entry per calibrator, in the input order.
    """
    results: list[CalibrationResult] = []
    for name in calibrator_names:
        if name not in CALIBRATORS:
            raise ValueError(f"Unknown calibrator: {name}")
        cal = CALIBRATORS[name]()
        cal.fit(y_val, p_val)
        p_test_cal = cal.transform(p_test)
        metrics = calibration_metric_bundle(y_test, p_test_cal, n_bins=n_bins)
        params: dict[str, Any] = {}
        # Expose fitted parameters so the result is reproducible without
        # re-running the fit (important for the report's calibration appendix).
        if isinstance(cal, TemperatureScaling):
            params["temperature"] = cal.temperature_
        elif isinstance(cal, PlattScaling):
            params["a"] = cal.a_
            params["b"] = cal.b_
        results.append(
            CalibrationResult(
                run_name=run_name,
                calibrator=name,
                metrics=metrics,
                params=params,
            )
        )
    return results


def results_to_dataframe(results: Sequence[CalibrationResult]) -> pd.DataFrame:
    """Flatten a list of :class:`CalibrationResult` into a tidy frame."""
    rows: list[dict[str, Any]] = []
    for r in results:
        row: dict[str, Any] = {"run": r.run_name, "calibrator": r.calibrator}
        row.update(r.metrics)
        # Namespace the fitted parameters with "param_" so they don't clash
        # with metric column names when different calibrators are joined.
        for k, v in r.params.items():
            row[f"param_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def _reliability_points(
    y: np.ndarray,
    p: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin (mean confidence, observed frequency, count) for plotting.

    Unlike :func:`_bin_indices`, the last bin is right-*inclusive* so a
    prediction of exactly 1.0 lands somewhere; otherwise the rightmost bin
    would always look empty.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    confs: list[float] = []
    accs: list[float] = []
    counts: list[int] = []
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
    panels: list[tuple[str, np.ndarray, np.ndarray]],
    out_path: Path,
    n_bins: int = 10,
) -> None:
    """Grid of reliability diagrams, one subplot per (label, y, p) triple.

    Dashed diagonal is the perfect-calibration reference. Empty bins are
    simply skipped rather than plotted at 0 -- this keeps the curves
    faithful to the data density.
    """
    n = len(panels)
    ncols = min(4, max(1, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
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
    # Blank out any trailing subplots when n is not a multiple of ncols.
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.suptitle(f"Reliability diagrams ({n_bins} bins)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ece_bar(df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart: one bar group per run, one bar per calibrator.

    Uses equal-width ECE because that's the number quoted throughout the
    report; flip the pivot column to ``ece_equal_mass`` for the stricter
    version.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = df.pivot(index="run", columns="calibrator", values="ece_equal_width")
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("ECE (10 equal-width bins)")
    ax.set_title("Calibration error by run x calibrator (lower is better)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _load_run_val_test(run_dir: Path) -> Optional[dict[str, np.ndarray]]:
    """Read val+test prediction npz files from a single run directory.

    Returns ``None`` when either file is missing -- the caller logs a
    warning and moves on instead of crashing mid-loop.
    """
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
        description=("Fit T/Platt/isotonic on each run's val split, score on test."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--runs",
        nargs="*",
        type=Path,
        default=[
            Path("results/transformer/seed_42"),
            Path("results/transformer/seed_1"),
            Path("results/transformer/seed_2"),
            Path("results/transformer/seed_42_mtlm_finetune"),
        ],
        help="Run dirs, each with val_predictions.npz + test_predictions.npz.",
    )
    p.add_argument(
        "--rf-dir",
        type=Path,
        default=Path("results/baseline/rf"),
        help="Optional RF run dir. Adds a comparison row.",
    )
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--output-dir", type=Path, default=Path("results/evaluation/calibration"))
    p.add_argument("--figures-dir", type=Path, default=Path("figures/evaluation/calibration"))
    return p


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args(argv)

    all_results: list[CalibrationResult] = []
    reliability_panels: list[tuple[str, np.ndarray, np.ndarray]] = []

    for run_dir in args.runs:
        payload = _load_run_val_test(run_dir)
        if payload is None:
            logger.warning("Skipping %s, predictions missing", run_dir)
            continue
        results = calibrate_and_score(
            payload["y_val"],
            payload["p_val"],
            payload["y_test"],
            payload["p_test"],
            n_bins=args.n_bins,
            run_name=payload["run_name"],
        )
        all_results.extend(results)
        # Two panels per run: raw and post-temperature, so the report
        # figure shows the calibrator in action without a third subplot.
        reliability_panels.append(
            (f"{payload['run_name']} (raw)", payload["y_test"], payload["p_test"])
        )
        ts = TemperatureScaling().fit(payload["y_val"], payload["p_val"])
        reliability_panels.append(
            (
                f"{payload['run_name']} (T={ts.temperature_:.2f})",
                payload["y_test"],
                ts.transform(payload["p_test"]),
            )
        )

    # The RF baseline is already approximately calibrated (sklearn RF probs
    # come from leaf-level class frequencies), so we report it under
    # ``identity`` rather than refitting every calibrator on its probs.
    rf_dir = Path(args.rf_dir)
    if (rf_dir / "test_predictions.npz").is_file():
        rf = np.load(rf_dir / "test_predictions.npz")
        metrics = calibration_metric_bundle(rf["y_true"], rf["y_prob"], n_bins=args.n_bins)
        all_results.append(
            CalibrationResult(
                run_name="rf_tuned",
                calibrator="identity",
                metrics=metrics,
            )
        )
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
    json_path.write_text(
        json.dumps(
            [
                {
                    "run": r.run_name,
                    "calibrator": r.calibrator,
                    "metrics": r.metrics,
                    "params": r.params,
                }
                for r in all_results
            ],
            indent=2,
            default=str,
        )
    )

    fig_rel = args.figures_dir / "calibration_reliability.png"
    fig_bar = args.figures_dir / "calibration_ece_bar.png"
    plot_reliability_panel(reliability_panels, fig_rel, n_bins=args.n_bins)
    plot_ece_bar(df, fig_bar)

    logger.info("Calibration metrics -> %s", csv_path)
    logger.info("Reliability panel   -> %s", fig_rel)
    logger.info("ECE bar chart       -> %s", fig_bar)

    print()
    display_cols = [
        "run",
        "calibrator",
        "auc_roc",
        "ece_equal_width",
        "ece_equal_mass",
        "mce",
        "brier",
        "reliability",
        "resolution",
        "brier_skill_score",
    ]
    print(df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
