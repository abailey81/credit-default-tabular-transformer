"""Paired significance tests for model comparison.

Contains four pairwise tests plus a power-analysis helper:

* :func:`mcnemar_test` -- discordant-pairs test on classification accuracy.
* :func:`delong_auc_test` -- paired DeLong test on AUC-ROC, using the
  Sun-Xu (2014) O(N log N) structural-components formulation.
* :func:`paired_bootstrap` -- generic paired-resample CI for any metric.
* :func:`bh_fdr` -- Benjamini-Hochberg FDR correction across a family.
* :func:`min_n_for_auc_difference` -- Hanley-McNeil (1982) power analysis.

The CLI runs every test on every model pair, corrects p-values with
Benjamini-Hochberg *per (test, metric) family* (not globally -- we don't
want the DeLong across pairs to inflate the bootstrap-F1 corrections),
and writes ``pairwise_tests.csv`` plus a p-value heatmap for one test.

References:
E. DeLong, D. DeLong, D. Clarke-Pearson (1988) for the original
non-parametric AUC covariance. Sun & Xu (2014), "Fast Implementation of
DeLong's Algorithm", IEEE Sig Proc Lett 21(11), for the O(N log N)
structural-components recipe we use. Hanley & McNeil (1982), "The
Meaning and Use of the Area Under a ROC Curve", Radiology 143, for
the power formula.
"""

from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from .calibration import expected_calibration_error

logger = logging.getLogger(__name__)

__all__ = [
    # Dataclasses
    "TestResult",
    # Tests
    "mcnemar_test",
    "delong_auc_test",
    "paired_bootstrap",
    "bh_fdr",
    "min_n_for_auc_difference",
    # Drivers
    "run_all_pairs",
    "plot_pvalue_heatmap",
    "main",
    # Metric dispatch table
    "METRIC_FNS",
]


@dataclass
class TestResult:
    """One row of the pairwise-test output.

    ``effect`` is always the observed ``A - B`` (positive = A better on
    that metric, for monotone-in-quality metrics like AUC / F1). For
    "lower is better" metrics like Brier or ECE the sign flips, so
    readers should always check the metric column before acting on
    the sign.
    """

    test: str
    model_a: str
    model_b: str
    metric: str
    statistic: float
    p_value: float
    effect: float  # observed A - B
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    n: Optional[int] = None
    extra: Optional[dict[str, Any]] = None

    def as_row(self) -> dict[str, Any]:
        """Flatten for DataFrame consumption; namespaces the extras."""
        row = {
            "test": self.test,
            "model_a": self.model_a,
            "model_b": self.model_b,
            "metric": self.metric,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "effect": self.effect,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
        }
        if self.extra:
            # Prefix with "extra_" so the flattened column names don't
            # clash with headline fields if a test adds, say, an "n" extra.
            row.update({f"extra_{k}": v for k, v in self.extra.items()})
        return row


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    *,
    model_a: str = "A",
    model_b: str = "B",
    exact_threshold: int = 25,
) -> TestResult:
    """McNemar's test on two classifiers over the same labels.

    The idea: tabulate the 2x2 of discordant pairs::

                               B correct     B wrong
        A correct               (a)             b
        A wrong                  c             (d)

    We never use ``a`` or ``d`` -- they're the rows where both models
    agree and tell us nothing about their *difference*. Under H0
    ("the two classifiers are equally accurate"), ``b ~ Binom(b + c,
    0.5)``.

    Two p-value flavours:

    * **Exact binomial** when ``b + c < exact_threshold``. The
      asymptotic chi-square test distorts badly in the small-discordant
      regime (Fagerland et al. 2013). We use the two-sided
      ``2 * P(X <= min(b, c))`` convention with Fisher's double-the-min
      rule clipped at 1.
    * **Continuity-corrected chi-square** for larger samples:
      ``((|b - c| - 1)^2) / (b + c)``, 1 df. Standard Edwards
      correction; the -1 reduces the bias from the discrete sampling
      distribution.

    Parameters
    ----------
    y_true, y_pred_a, y_pred_b : arrays, shape (N,)
        Labels and the two models' hard predictions at whatever
        threshold the caller applied.
    exact_threshold : int
        Cut-off for switching between exact-binomial and chi-square.
        25 is the conventional choice (see Fagerland 2013 for the
        recommendation).
    """
    y = y_true.astype(int)
    a = y_pred_a.astype(int)
    bn = y_pred_b.astype(int)
    a_correct = a == y
    b_correct = bn == y

    # b = A-right, B-wrong; c = A-wrong, B-right. Concordant pairs
    # (both right or both wrong) are excluded by design.
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    n_discordant = b + c
    effect = (a == y).mean() - (bn == y).mean()

    if n_discordant == 0:
        # Identical accuracy signature; every pair concordant. No
        # evidence against H0.
        return TestResult(
            test="mcnemar",
            model_a=model_a,
            model_b=model_b,
            metric="accuracy",
            statistic=0.0,
            p_value=1.0,
            effect=0.0,
            n=len(y),
            extra={"b": b, "c": c, "method": "no_discordant"},
        )

    if n_discordant < exact_threshold:
        # Exact two-sided binomial. Fisher convention: 2 * P(X <= min(b, c)),
        # clipped at 1 so the two-sided value never exceeds 1 at the
        # boundary.
        k = min(b, c)
        p_val = float(2.0 * stats.binom.cdf(k, n_discordant, 0.5))
        p_val = min(p_val, 1.0)
        stat = float(k)
        method = "exact_binomial"
    else:
        # Edwards continuity correction.
        stat = (abs(b - c) - 1.0) ** 2 / (b + c)
        p_val = float(1.0 - stats.chi2.cdf(stat, df=1))
        method = "chi2_continuity"

    return TestResult(
        test="mcnemar",
        model_a=model_a,
        model_b=model_b,
        metric="accuracy",
        statistic=float(stat),
        p_value=p_val,
        effect=float(effect),
        n=len(y),
        extra={"b": b, "c": c, "method": method},
    )


def _structural_components(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """V10, V01 structural components via mid-ranks (Sun-Xu 2014).

    These are the per-row contributions to the Mann-Whitney-U form of
    AUC. Sums ``mean(v10) = mean(v01) = AUC``, and the paired covariance
    over two models can be computed from the per-row vectors alone --
    which is what makes this O(N log N) instead of DeLong's original
    O(N^2).
    """
    pos = labels == 1
    neg = labels == 0
    scores_pos = scores[pos]
    scores_neg = scores[neg]
    n1 = len(scores_pos)
    n0 = len(scores_neg)

    # Pooled mid-ranks: ties get the average of their would-be ranks,
    # which keeps the Mann-Whitney-U unbiased under ties.
    combined = np.concatenate([scores_pos, scores_neg])
    order = np.argsort(combined)
    ranks = np.empty_like(order, dtype=np.float64)

    sorted_scores = combined[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0  # 1-based ranks
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    ranks_pos = ranks[:n1]
    ranks_neg = ranks[n1:]
    # Subtracting within-class ranks strips off the rank each positive
    # has among positives (and similarly for negatives), leaving the
    # per-row AUC contribution.
    v10 = (ranks_pos - _rank_within(scores_pos)) / n0
    v01 = 1.0 - (ranks_neg - _rank_within(scores_neg)) / n1
    return v10, v01


def _rank_within(a: np.ndarray) -> np.ndarray:
    """Mid-ranks within a single-class score vector.

    Same tie-handling as ``_structural_components``; broken out so the
    pos/neg streams can be ranked independently.
    """
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < len(sorted_a):
        j = i
        while j + 1 < len(sorted_a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def delong_auc_test(
    y_true: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    *,
    model_a: str = "A",
    model_b: str = "B",
) -> TestResult:
    """Paired DeLong test for ``AUC(A) - AUC(B)``.

    The Sun-Xu (2014) structural-components form decomposes::

        Var(AUC) = Var(V10) / n_pos  +  Var(V01) / n_neg

    for one model, and extends to a 2x2 block-covariance across two
    models' V-components. From the variance of the difference::

        Var(AUC_A - AUC_B) = Var(AUC_A) + Var(AUC_B) - 2 Cov(AUC_A, AUC_B)

    we get a Z-statistic ``(AUC_A - AUC_B) / sqrt(Var(diff))``, which
    is asymptotically standard normal under H0.

    The 95 % CI reported is the DeLong Wald form ``effect +/- 1.96 *
    sqrt(var)``. For very close AUCs we can numerically get
    ``var_diff <= 0`` (floating-point cancellation when the two models
    produce near-identical ranks); we fall back to ``p = 1``, CI = point
    effect, which is what a strict interpretation says anyway.

    Parameters
    ----------
    y_true : array, shape (N,)
        Binary labels. Must have at least one of each class.
    p_a, p_b : arrays, shape (N,)
        Paired probabilities from the two models.
    """
    y = y_true.astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("DeLong requires at least one row of each class.")

    v10_a, v01_a = _structural_components(p_a.astype(np.float64), y)
    v10_b, v01_b = _structural_components(p_b.astype(np.float64), y)

    auc_a = v10_a.mean()
    auc_b = v10_b.mean()
    effect = float(auc_a - auc_b)

    def _cov(x, y):
        # Sample covariance with Bessel correction; Sun-Xu uses the same.
        n = len(x)
        return float(np.sum((x - x.mean()) * (y - y.mean())) / (n - 1))

    n1 = (y == 1).sum()
    n0 = (y == 0).sum()
    # 2x2 covariance blocks over (A, B) for positive and negative classes.
    s10 = np.array(
        [
            [_cov(v10_a, v10_a), _cov(v10_a, v10_b)],
            [_cov(v10_b, v10_a), _cov(v10_b, v10_b)],
        ]
    )
    s01 = np.array(
        [
            [_cov(v01_a, v01_a), _cov(v01_a, v01_b)],
            [_cov(v01_b, v01_a), _cov(v01_b, v01_b)],
        ]
    )
    cov = s10 / n1 + s01 / n0
    var_diff = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if var_diff <= 0:
        # Near-identical AUCs: numerical cancellation in var_diff. The
        # DeLong test is degenerate here and the correct answer is "no
        # detectable difference".
        return TestResult(
            test="delong",
            model_a=model_a,
            model_b=model_b,
            metric="auc_roc",
            statistic=0.0,
            p_value=1.0,
            effect=effect,
            ci_low=effect,
            ci_high=effect,
            n=len(y),
        )
    z = effect / math.sqrt(var_diff)
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    ci_half = 1.96 * math.sqrt(var_diff)
    return TestResult(
        test="delong",
        model_a=model_a,
        model_b=model_b,
        metric="auc_roc",
        statistic=float(z),
        p_value=p_val,
        effect=effect,
        ci_low=float(effect - ci_half),
        ci_high=float(effect + ci_half),
        n=len(y),
        extra={"auc_a": float(auc_a), "auc_b": float(auc_b), "var_diff": var_diff},
    )


def paired_bootstrap(
    y_true: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    p_a: np.ndarray,
    p_b: np.ndarray,
    *,
    n_resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    model_a: str = "A",
    model_b: str = "B",
    metric_name: str = "metric",
) -> TestResult:
    """Paired bootstrap CI + p-value on ``metric(y, p_a) - metric(y, p_b)``.

    Joint index resampling (same ``idx`` for both models per replicate)
    pairs out the row variance, giving tighter CIs than two independent
    bootstraps. Pairing is what makes this applicable to any metric the
    caller defines, including non-closed-form ones like F1 at a fixed
    threshold.

    Degenerate-class replicates (all-zero or all-one ``y[idx]``) are
    logged as NaN and excluded from the quantile; if more than 10 % of
    replicates are degenerate we warn (the CI may be narrowed
    artificially by the implicit stratification that our filter
    applies).

    The two-sided p-value uses the usual bootstrap sign test:
    ``2 * min(P(diff <= 0), P(diff >= 0))``, clipped at 1.
    """
    rng = np.random.default_rng(seed)
    y = y_true.astype(int)
    N = len(y)
    observed = float(metric_fn(y, p_a)) - float(metric_fn(y, p_b))
    diffs = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, N, size=N)
        if len(np.unique(y[idx])) < 2:
            # Single-class replicate: metric undefined (AUC, AP, etc.).
            # NaN rather than silently imputing 0 -- this is a degenerate
            # replicate, not a zero difference.
            diffs[i] = np.nan
            continue
        diffs[i] = float(metric_fn(y[idx], p_a[idx])) - float(metric_fn(y[idx], p_b[idx]))
    valid = diffs[np.isfinite(diffs)]
    if len(valid) < 0.9 * n_resamples:
        logger.warning(
            "%d/%d bootstrap resamples had a degenerate class distribution; "
            "CI may be narrow. Consider stratified resampling.",
            n_resamples - len(valid),
            n_resamples,
        )
    ci_low = float(np.quantile(valid, alpha / 2))
    ci_high = float(np.quantile(valid, 1.0 - alpha / 2))
    p_val = 2.0 * min(float((valid <= 0).mean()), float((valid >= 0).mean()))
    p_val = min(p_val, 1.0)
    return TestResult(
        test="paired_bootstrap",
        model_a=model_a,
        model_b=model_b,
        metric=metric_name,
        statistic=observed,
        p_value=p_val,
        effect=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        n=N,
        extra={"n_resamples": n_resamples, "n_valid": int(len(valid))},
    )


def bh_fdr(p_values: Sequence[float], q: float = 0.05) -> dict[str, np.ndarray]:
    """Benjamini-Hochberg step-up FDR correction.

    In words: rank the p-values ascending, multiply the k-th smallest by
    ``m / k`` (bigger correction for bigger ranks), then enforce
    monotonicity by walking *down* from the largest and taking the
    running minimum. The resulting q-values are the smallest FDR
    levels at which each test would be rejected.

    Rejecting ``q_i <= q`` controls the *expected* false-discovery
    proportion at level ``q`` under independence or positive dependence
    (Benjamini-Yekutieli 2001 drops the independence assumption at the
    cost of a ``log(m)`` factor, which we don't apply here).
    """
    p = np.asarray(p_values, dtype=np.float64)
    n = len(p)
    if n == 0:
        return {"q_values": p.copy(), "rejected": np.zeros(0, dtype=bool)}
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    q_adjusted = p * n / ranks
    # Monotone step-up: walk from the largest p down, take the running
    # minimum, so q-values are non-decreasing in p.
    q_sorted = np.minimum.accumulate(q_adjusted[order[::-1]])
    q_final = np.empty(n, dtype=np.float64)
    q_final[order[::-1]] = np.clip(q_sorted, 0.0, 1.0)
    rejected = q_final <= q
    return {"q_values": q_final, "rejected": rejected}


def min_n_for_auc_difference(
    auc_a: float,
    auc_b: float,
    prevalence: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Hanley-McNeil (1982) sample size for detecting ``|AUC_A - AUC_B|``.

    Uses the negative-exponential assumption for the score distributions
    with parameters

    * ``Q1 = AUC / (2 - AUC)``        -- Pr(one positive's score > another negative's)
    * ``Q2 = 2 AUC^2 / (1 + AUC)``    -- Pr(two positives' scores > one negative's)

    and the variance formula::

        Var(AUC) = [AUC (1 - AUC) + (P - 1)(Q1 - AUC^2)
                                   + (1 - P)(Q2 - AUC^2) ] / (n P (1 - P))

    Returns the total N (positive + negative) needed so a two-sided
    test at level ``alpha`` has power ``power`` to detect
    ``delta = |AUC_A - AUC_B|``. The conservative average variance
    ``0.5 (Var_A + Var_B)`` is used in the numerator; this matches the
    standard Hanley-McNeil table.

    Returns ``1e9`` for degenerate inputs (identical AUCs or zero
    prevalence) so the caller's arithmetic doesn't blow up.
    """
    if auc_a == auc_b:
        return int(1e9)
    z_a = stats.norm.ppf(1.0 - alpha / 2)
    z_b = stats.norm.ppf(power)

    def _var(auc: float) -> float:
        Q1 = auc / (2 - auc)
        Q2 = (2 * auc**2) / (1 + auc)
        return (
            auc * (1 - auc) + (prevalence - 1) * (Q1 - auc**2) + (-prevalence) * (Q2 - auc**2)
        )

    var = 0.5 * (_var(auc_a) + _var(auc_b))  # conservative average of the two variances
    delta = abs(auc_a - auc_b)
    n1_over_n = prevalence * (1 - prevalence)
    if n1_over_n <= 0:
        return int(1e9)
    n = ((z_a + z_b) ** 2 * var) / (delta**2 * n1_over_n)
    return int(math.ceil(n))


def _load_run(run_dir: Path) -> Optional[dict[str, np.ndarray]]:
    """Load test preds for one run. ``None`` if the npz is missing."""
    run_dir = Path(run_dir)
    npz = run_dir / "test_predictions.npz"
    if not npz.is_file():
        return None
    d = np.load(npz)
    return {
        "run_name": run_dir.name,
        "y_true": d["y_true"],
        "y_prob": d["y_prob"],
        "y_pred": d["y_pred"],
    }


#: Metric dispatch table for the paired bootstrap. Every entry takes
#: ``(y, p)`` and returns a float -- the bootstrap loop doesn't need
#: to know what the metric actually is, only that it's row-stable.
METRIC_FNS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "auc_roc": lambda y, p: float(roc_auc_score(y, p)),
    "auc_pr": lambda y, p: float(average_precision_score(y, p)),
    "brier": lambda y, p: float(brier_score_loss(y, p)),
    "ece": lambda y, p: float(expected_calibration_error(y, p, n_bins=10)),
    "f1": lambda y, p: float(f1_score(y, (p >= 0.5).astype(int), zero_division=0)),
    "accuracy": lambda y, p: float(accuracy_score(y, (p >= 0.5).astype(int))),
}


def run_all_pairs(
    runs: Sequence[dict[str, np.ndarray]],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Run every test on every pair of runs; return a tidy long-form frame.

    BH-FDR is applied *per (test, metric) family*. Rationale: the
    bootstrap on F1 and the DeLong on AUC measure two different
    things; correcting them jointly would inflate whichever family has
    fewer tests. Correcting within family keeps each family's false-
    discovery rate at the nominal level.
    """
    results: list[TestResult] = []
    for i, ra in enumerate(runs):
        for rb in runs[i + 1 :]:
            y = ra["y_true"]
            # y_true must be bit-identical across runs; otherwise the
            # pairing is invalid and any p-value is meaningless.
            if not np.array_equal(y, rb["y_true"]):
                raise ValueError(f"y_true mismatch between {ra['run_name']} and {rb['run_name']}.")
            results.append(
                mcnemar_test(
                    y,
                    ra["y_pred"],
                    rb["y_pred"],
                    model_a=ra["run_name"],
                    model_b=rb["run_name"],
                )
            )
            results.append(
                delong_auc_test(
                    y,
                    ra["y_prob"],
                    rb["y_prob"],
                    model_a=ra["run_name"],
                    model_b=rb["run_name"],
                )
            )
            for metric_name, fn in METRIC_FNS.items():
                results.append(
                    paired_bootstrap(
                        y,
                        fn,
                        ra["y_prob"],
                        rb["y_prob"],
                        n_resamples=n_resamples,
                        seed=seed,
                        model_a=ra["run_name"],
                        model_b=rb["run_name"],
                        metric_name=metric_name,
                    )
                )
    df = pd.DataFrame([r.as_row() for r in results])
    df["q_value"] = np.nan
    # Family = (test, metric). Every bootstrap-F1 cell gets corrected
    # against every other bootstrap-F1 cell, but not against DeLong or
    # McNemar -- those live in different families.
    for (test, metric), grp in df.groupby(["test", "metric"]):
        bh = bh_fdr(grp["p_value"].tolist(), q=0.05)
        df.loc[grp.index, "q_value"] = bh["q_values"]
        df.loc[grp.index, "bh_reject_05"] = bh["rejected"]
    return df


def plot_pvalue_heatmap(df: pd.DataFrame, out_path: Path, test: str = "delong") -> None:
    """Symmetric p-value heatmap for one test.

    Each cell shows p-value and signed effect. Diagonal forced to 1 / 0
    (identity rows are not tested). The ``RdYlGn_r`` colour map puts
    small p-values (significant) in red so the eye catches them; the
    ``vmax=0.2`` cap keeps the gradient meaningful -- anything above 0.2
    reads the same.
    """
    sub = df[df["test"] == test]
    if sub.empty:
        return
    models = sorted(set(sub["model_a"]).union(sub["model_b"]))
    M = len(models)
    idx = {m: i for i, m in enumerate(models)}
    mat = np.full((M, M), np.nan)
    effect = np.full((M, M), np.nan)
    for _, row in sub.iterrows():
        i, j = idx[row["model_a"]], idx[row["model_b"]]
        mat[i, j] = row["p_value"]
        mat[j, i] = row["p_value"]
        effect[i, j] = row["effect"]
        # Flip sign on the mirror so A-B and B-A both read correctly.
        effect[j, i] = -row["effect"]
    np.fill_diagonal(mat, 1.0)
    np.fill_diagonal(effect, 0.0)

    fig, ax = plt.subplots(figsize=(1 + 0.8 * M, 1 + 0.8 * M))
    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0.0, vmax=0.2)
    ax.set_xticks(range(M))
    ax.set_yticks(range(M))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_yticklabels(models)
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            # White text when the background is dark (small p-value) so
            # the annotation stays legible.
            ax.text(
                j,
                i,
                f"p={mat[i, j]:.3f}\nΔ={effect[i, j]:+.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color=("white" if mat[i, j] < 0.05 else "black"),
            )
    ax.set_title(f"{test} p-values (Δ = effect A − B)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="p-value")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Significance testing: McNemar, DeLong, paired bootstrap, BH-FDR "
            "across every model pair."
        ),
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
            Path("results/baseline/rf"),
        ],
    )
    p.add_argument("--n-resamples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("results/evaluation/significance"))
    p.add_argument("--figures-dir", type=Path, default=Path("figures/evaluation/significance"))
    return p


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args(argv)

    loaded = []
    for d in args.runs:
        r = _load_run(d)
        if r is None:
            logger.warning("Skipping %s — no test_predictions.npz", d)
            continue
        loaded.append(r)
    if len(loaded) < 2:
        logger.error("Need at least 2 runs with predictions.")
        return 1

    df = run_all_pairs(loaded, n_resamples=args.n_resamples, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "pairwise_tests.csv"
    df.to_csv(csv_path, index=False)

    # Power analysis: how big a sample would we need to detect a given
    # AUC gap at 80 % power? Anchored at AUC ~ 0.78 (where this project
    # sits) and sweeping deltas we'd actually care about.
    prevalence = float(np.mean(loaded[0]["y_true"]))
    power_rows: list[dict[str, Any]] = []
    for delta in (0.005, 0.01, 0.02):
        auc0 = 0.78
        auc1 = auc0 + delta
        n_req = min_n_for_auc_difference(auc1, auc0, prevalence=prevalence)
        power_rows.append(
            {
                "auc_gap": delta,
                "n_required_80pct_power": n_req,
                "prevalence": prevalence,
                "current_n": len(loaded[0]["y_true"]),
            }
        )
    power_df = pd.DataFrame(power_rows)
    power_df.to_csv(args.output_dir / "power_analysis.csv", index=False)

    plot_pvalue_heatmap(df, args.figures_dir / "significance_pvalue_heatmap.png", test="delong")
    logger.info("Pairwise tests written to %s", csv_path)

    print()
    print("-- Pairwise significance summary (DeLong + McNemar only) --")
    mask = df["test"].isin(["delong", "mcnemar"])
    cols = [
        "test",
        "model_a",
        "model_b",
        "metric",
        "statistic",
        "p_value",
        "effect",
        "ci_low",
        "ci_high",
        "q_value",
    ]
    present = [c for c in cols if c in df.columns]
    print(df.loc[mask, present].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n-- Power analysis --")
    print(power_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
