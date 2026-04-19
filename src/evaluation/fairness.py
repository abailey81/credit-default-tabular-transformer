"""Subgroup fairness audit across SEX / EDUCATION / MARRIAGE (novelty N10).

Reports per-subgroup discrimination (AUC, TPR, FPR), calibration
(ECE, Brier), and the three standard fairness criteria:

* Demographic parity (DP) -- selection rate at the operating threshold
  should be equal across subgroups. Difference vs the reference
  subgroup.
* Equal opportunity (EO) -- TPR equal across subgroups (``| TPR_g -
  TPR_ref |``). Cares only about the positive class.
* Equalised odds (EO_2) -- TPR *and* FPR equal across subgroups
  (``max(|Δ TPR|, |Δ FPR|)``).

Chouldechova (2017) and Kleinberg-Mullainathan-Raghavan (2016) showed
these three criteria cannot all hold simultaneously when base rates
differ across subgroups, which is the case here: default rates vary
by both SEX and EDUCATION. We therefore report all three, annotate
which subgroup is the reference (always the largest-n one, so the
reference has the tightest CIs), and let the reader pick the trade-off.

The audit Platt-scales the transformer probabilities by default -- raw
probabilities are miscalibrated enough that the calibration columns
would dominate the disparity rows. RF is left uncalibrated (its raw
probs are already approximately calibrated).

Key public symbols: :class:`SubgroupMetrics`, :func:`audit_attribute`,
:func:`audit_run`, :func:`disparity_table`.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from .calibration import (
    PlattScaling,
    expected_calibration_error,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SubgroupMetrics",
    "audit_attribute",
    "audit_run",
    "disparity_table",
    "plot_disparity",
    "plot_subgroup_reliability",
    "main",
    "ATTRIBUTE_LABELS",
    "SEX_LABELS",
    "EDUCATION_LABELS",
    "MARRIAGE_LABELS",
    "MIN_SUBGROUP_N",
]

# UCI credit-default codebook. Kept here (rather than imported from the
# preprocessing module) because preprocessing folds EDUCATION 5/6 and
# MARRIAGE 0 into 4 / 3 respectively *before* dumping test_raw.csv; the
# maps below reflect the cleaned post-preprocessing coding.
SEX_LABELS = {1: "Male", 2: "Female"}
EDUCATION_LABELS = {
    1: "Grad school",
    2: "University",
    3: "High school",
    4: "Other",
}
MARRIAGE_LABELS = {1: "Married", 2: "Single", 3: "Other"}

ATTRIBUTE_LABELS: Dict[str, Dict[int, str]] = {
    "SEX": SEX_LABELS,
    "EDUCATION": EDUCATION_LABELS,
    "MARRIAGE": MARRIAGE_LABELS,
}

#: Subgroups smaller than this get flagged ``underpowered=True``. 50 is
#: a conservative binomial-CI floor on this test split (rule of thumb:
#: for a ±5 pp 95% CI on a binary outcome around 0.2, you need n >~ 250;
#: below 50 the CI is ±10 pp or wider). Rows still get recorded -- we
#: don't want to hide the subgroup -- but the flag lets the reader
#: discount the numbers appropriately.
MIN_SUBGROUP_N = 50


@dataclass
class SubgroupMetrics:
    """Per-subgroup metrics row.

    One instance per (run, attribute, subgroup) cell. All float fields
    can be NaN -- most commonly when the subgroup is single-class (AUC
    needs both) or too small to compute ECE reliably.
    """

    run_name: str
    attribute: str
    subgroup_code: int
    subgroup_label: str
    n: int
    base_rate: float
    selection_rate: float
    auc_roc: float
    ece: float
    brier: float
    tpr: float
    fpr: float
    calibrator: str
    underpowered: bool


def _subgroup_metrics(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute every metric we report for one subgroup.

    Defensive around single-class subgroups: AUC returns NaN (sklearn
    would raise), TPR / FPR return NaN if no positives / negatives.
    """
    y = y.astype(int)
    y_pred = (p >= threshold).astype(int)
    n = len(y)
    pos = y == 1
    neg = y == 0

    tpr = float(y_pred[pos].mean()) if pos.any() else float("nan")
    fpr = float(y_pred[neg].mean()) if neg.any() else float("nan")
    selection_rate = float(y_pred.mean()) if n > 0 else float("nan")
    base_rate = float(y.mean()) if n > 0 else float("nan")

    # AUC is undefined when the subgroup has only one class; sklearn
    # raises in that case, we NaN.
    if pos.any() and neg.any():
        auc = float(roc_auc_score(y, p))
    else:
        auc = float("nan")

    ece = float(expected_calibration_error(y, p, n_bins=10))
    brier = float(brier_score_loss(y, p))

    return {
        "n": n,
        "base_rate": base_rate,
        "selection_rate": selection_rate,
        "tpr": tpr,
        "fpr": fpr,
        "auc_roc": auc,
        "ece": ece,
        "brier": brier,
    }


def audit_attribute(
    y: np.ndarray,
    p: np.ndarray,
    attribute_values: np.ndarray,
    attribute_name: str,
    run_name: str,
    calibrator: str,
    threshold: float = 0.5,
) -> List[SubgroupMetrics]:
    """Audit one protected attribute on one run's predictions.

    Produces one :class:`SubgroupMetrics` per unique value of
    ``attribute_values``. The three fairness criteria this enables:

    * Demographic parity (DP): ``selection_rate`` should match across
      subgroups. The delta vs the reference subgroup is in
      :func:`disparity_table` as ``demographic_parity_diff``.
    * Equal opportunity (EO): TPR should match. ``| TPR_g - TPR_ref |``
      in the disparity table as ``equal_opportunity_violation``.
    * Equalised odds (EO_2): both TPR and FPR should match.
      ``max(| TPR_g - TPR_ref |, | FPR_g - FPR_ref |)`` as
      ``equalised_odds_violation``.

    Parameters
    ----------
    y, p : arrays, shape (N,)
        Labels and probabilities, aligned row-for-row with
        ``attribute_values``.
    attribute_values : array, shape (N,)
        Integer-coded protected attribute. The unique-value order
        (sorted ascending) drives the output order.
    attribute_name : str
        Must be a key of :data:`ATTRIBUTE_LABELS` if you want the
        human-readable labels to appear; otherwise the subgroup_code
        is also used as the label.
    run_name, calibrator : str
        Carried onto every returned row for the aggregated table.
    threshold : float
        Decision threshold for selection-rate / TPR / FPR. 0.5 by
        default to match the comparison table.

    Returns
    -------
    list of SubgroupMetrics
    """
    unique_vals = sorted(np.unique(attribute_values).tolist())
    labels = ATTRIBUTE_LABELS.get(attribute_name, {})
    out: List[SubgroupMetrics] = []
    for v in unique_vals:
        mask = attribute_values == v
        if mask.sum() == 0:
            continue
        m = _subgroup_metrics(y[mask], p[mask], threshold=threshold)
        out.append(
            SubgroupMetrics(
                run_name=run_name,
                attribute=attribute_name,
                subgroup_code=int(v),
                subgroup_label=labels.get(int(v), str(v)),
                n=int(m["n"]),
                base_rate=m["base_rate"],
                selection_rate=m["selection_rate"],
                auc_roc=m["auc_roc"],
                ece=m["ece"],
                brier=m["brier"],
                tpr=m["tpr"],
                fpr=m["fpr"],
                calibrator=calibrator,
                underpowered=m["n"] < MIN_SUBGROUP_N,
            )
        )
    return out


def disparity_table(
    subgroups: Sequence[SubgroupMetrics],
) -> pd.DataFrame:
    """Turn per-subgroup metrics into (subgroup - reference) deltas and ratios.

    The reference subgroup is the largest-n one *within each
    (run, attribute, calibrator) group*. That choice means the reference
    has the tightest CI, so differences against it are as meaningful
    as the data allows. It also matches the "demographic majority"
    reading some audit regimes use, though here it's driven by sample
    size, not demographics.
    """
    df = pd.DataFrame([s.__dict__ for s in subgroups])
    if df.empty:
        return df
    rows: List[Dict[str, Any]] = []
    for (run, attr, cal), grp in df.groupby(["run_name", "attribute", "calibrator"]):
        ref = grp.loc[grp["n"].idxmax()]
        for _, row in grp.iterrows():

            def _safe_ratio(a: float, b: float) -> float:
                # NaN on degenerate inputs -- better than a silent
                # inf / 0 cell.
                return (
                    float("nan") if (not np.isfinite(a) or not np.isfinite(b) or b == 0) else a / b
                )

            rows.append(
                {
                    "run": run,
                    "attribute": attr,
                    "calibrator": cal,
                    "subgroup_code": int(row["subgroup_code"]),
                    "subgroup_label": row["subgroup_label"],
                    "ref_subgroup_label": ref["subgroup_label"],
                    "n": int(row["n"]),
                    # Demographic parity: selection rate relative to reference.
                    "selection_rate_diff": row["selection_rate"] - ref["selection_rate"],
                    "selection_rate_ratio": _safe_ratio(
                        row["selection_rate"], ref["selection_rate"]
                    ),
                    # Equal opportunity (EO): TPR gap. Cares only about the
                    # positive class, so FPR disparity can still be non-zero.
                    "tpr_diff": row["tpr"] - ref["tpr"],
                    "fpr_diff": row["fpr"] - ref["fpr"],
                    "equal_opportunity_violation": abs(row["tpr"] - ref["tpr"]),
                    # Equalised odds: stricter -- both TPR and FPR must match.
                    "equalised_odds_violation": max(
                        abs(row["tpr"] - ref["tpr"]), abs(row["fpr"] - ref["fpr"])
                    ),
                    # Demographic parity alias kept for clarity when the
                    # reader is scanning for DP specifically.
                    "demographic_parity_diff": row["selection_rate"] - ref["selection_rate"],
                    "auc_diff": row["auc_roc"] - ref["auc_roc"],
                    "ece_diff": row["ece"] - ref["ece"],
                }
            )
    return pd.DataFrame(rows)


def plot_disparity(disp_df: pd.DataFrame, out_path: Path) -> None:
    """One grouped-bar panel per fairness metric, first run only.

    The report shows a single run (the primary transformer) in this
    figure; cross-run comparison lives in the underlying CSV.
    """
    if disp_df.empty:
        return
    metrics_to_plot = [
        ("demographic_parity_diff", "Demographic parity delta"),
        ("equal_opportunity_violation", "Equal-opportunity violation"),
        ("equalised_odds_violation", "Equalised-odds violation"),
        ("ece_diff", "ECE delta"),
    ]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 4))
    # First run only -- the story is qualitatively the same across runs
    # and a single panel row reads faster than a grid.
    priority = disp_df["run"].unique().tolist()
    df = disp_df[disp_df["run"] == priority[0]]
    for ax, (col, title) in zip(axes, metrics_to_plot):
        sub = df.pivot_table(
            index="subgroup_label",
            columns="attribute",
            values=col,
            aggfunc="mean",
        )
        sub.plot(kind="bar", ax=ax, width=0.8)
        ax.axhline(0.0, color="#333333", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Subgroup")
        ax.set_ylabel(col)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(f"Subgroup disparities ({priority[0]})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_subgroup_reliability(
    y: np.ndarray,
    p: np.ndarray,
    attribute_values: np.ndarray,
    attribute_name: str,
    out_path: Path,
    n_bins: int = 10,
) -> None:
    """One reliability curve per subgroup on shared axes.

    Subgroups with fewer than :data:`MIN_SUBGROUP_N` rows are skipped
    entirely (the reliability curve would be dominated by binning noise
    and would mislead). Palette cycles through Okabe-Ito so up to six
    subgroups stay distinguishable.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="#999999", label="Perfect calibration")
    labels = ATTRIBUTE_LABELS.get(attribute_name, {})
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"]
    for i, v in enumerate(sorted(np.unique(attribute_values).tolist())):
        mask = attribute_values == v
        if mask.sum() < MIN_SUBGROUP_N:
            continue
        p_sub = p[mask]
        y_sub = y[mask]
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        confs: List[float] = []
        accs: List[float] = []
        for j in range(n_bins):
            lo, hi = edges[j], edges[j + 1]
            # Right-inclusive last bin for the same reason as calibration.py:
            # lets p=1.0 land somewhere.
            m = (p_sub >= lo) & (p_sub < hi if j < n_bins - 1 else p_sub <= hi)
            if not m.any():
                continue
            confs.append(p_sub[m].mean())
            accs.append(y_sub[m].mean())
        ax.plot(
            confs,
            accs,
            marker="o",
            ms=4,
            lw=1.6,
            color=palette[i % len(palette)],
            label=f"{labels.get(int(v), str(v))} (n={mask.sum()})",
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed default rate")
    ax.set_title(f"Subgroup reliability ({attribute_name})")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def audit_run(
    run_dir: Path,
    test_raw: pd.DataFrame,
    *,
    use_platt: bool = True,
    attribute_names: Sequence[str] = ("SEX", "EDUCATION", "MARRIAGE"),
    threshold: float = 0.5,
) -> Tuple[List[SubgroupMetrics], Dict[str, np.ndarray]]:
    """Audit a single run end-to-end.

    Row-order alignment between ``test_predictions.npz`` and
    ``test_raw.csv`` is a hard contract enforced by the no-shuffle
    ``make_loader(mode="test")`` path in ``src.training.dataset``. The
    length check below will catch any breach of that contract; a more
    subtle per-row check would require a row-id column, which we don't
    currently emit (the split hash in SPLIT_HASHES.md is the integrity
    check for the split itself).
    """
    run_dir = Path(run_dir)
    val = np.load(run_dir / "val_predictions.npz")
    test = np.load(run_dir / "test_predictions.npz")

    if use_platt:
        # Platt on val, apply on test. Raw probs are miscalibrated enough
        # that leaving them raw would dominate the ECE-disparity columns.
        cal = PlattScaling().fit(val["y_true"], val["y_prob"])
        p_test = cal.transform(test["y_prob"])
        cal_name = "platt"
    else:
        p_test = test["y_prob"].astype(np.float64)
        cal_name = "identity"

    y_test = test["y_true"].astype(int)
    if len(y_test) != len(test_raw):
        raise ValueError(
            f"Test split row count mismatch: predictions={len(y_test)}, "
            f"raw={len(test_raw)}. Check alignment."
        )

    results: List[SubgroupMetrics] = []
    for attr in attribute_names:
        if attr not in test_raw.columns:
            logger.warning("Attribute %s missing from test_raw.csv, skipping", attr)
            continue
        values = test_raw[attr].astype(int).values
        results.extend(
            audit_attribute(
                y_test,
                p_test,
                values,
                attr,
                run_name=run_dir.name,
                calibrator=cal_name,
                threshold=threshold,
            )
        )

    # Return the arrays the subgroup-reliability plot needs so the CLI
    # caller doesn't have to re-load val+test.
    extras = {"y_test": y_test, "p_test": p_test}
    for attr in attribute_names:
        if attr in test_raw.columns:
            extras[attr] = test_raw[attr].astype(int).values
    return results, extras


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Subgroup fairness audit (SEX / EDUCATION / MARRIAGE) on the "
            "tuned transformer run(s) and the RF baseline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--runs",
        nargs="*",
        type=Path,
        default=[
            Path("results/transformer/seed_42"),
            Path("results/transformer/seed_42_mtlm_finetune"),
        ],
    )
    p.add_argument("--rf-dir", type=Path, default=Path("results/baseline/rf"))
    p.add_argument("--test-raw", type=Path, default=Path("data/processed/test_raw.csv"))
    p.add_argument("--attributes", nargs="+", default=["SEX", "EDUCATION", "MARRIAGE"])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument(
        "--no-platt", action="store_true", help="Audit raw probs instead of Platt-scaled ones."
    )
    p.add_argument("--output-dir", type=Path, default=Path("results/evaluation/fairness"))
    p.add_argument("--figures-dir", type=Path, default=Path("figures/evaluation/fairness"))
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args(argv)

    test_raw_path = Path(args.test_raw)
    if not test_raw_path.is_file():
        logger.error("Missing %s; run preprocessing first.", test_raw_path)
        return 1
    test_raw = pd.read_csv(test_raw_path)

    all_results: List[SubgroupMetrics] = []
    primary_extras: Optional[Dict[str, np.ndarray]] = None

    for run_dir in args.runs:
        run_dir = Path(run_dir)
        if not (run_dir / "test_predictions.npz").is_file():
            logger.warning("Skipping %s, no predictions", run_dir)
            continue
        results, extras = audit_run(
            run_dir,
            test_raw,
            use_platt=not args.no_platt,
            attribute_names=tuple(args.attributes),
            threshold=args.threshold,
        )
        all_results.extend(results)
        # The first run's extras drive the subgroup-reliability plot.
        # Subsequent runs add rows to the tables only.
        if primary_extras is None:
            primary_extras = extras

    # RF: skip Platt. RF probs come from leaf-level class frequencies
    # and are already approximately calibrated globally; re-Platting
    # would just add a thin layer of noise on top.
    rf_dir = Path(args.rf_dir)
    if (rf_dir / "test_predictions.npz").is_file():
        rf = np.load(rf_dir / "test_predictions.npz")
        for attr in args.attributes:
            if attr in test_raw.columns:
                all_results.extend(
                    audit_attribute(
                        rf["y_true"].astype(int),
                        rf["y_prob"].astype(np.float64),
                        test_raw[attr].astype(int).values,
                        attr,
                        run_name="rf_tuned",
                        calibrator="identity",
                        threshold=args.threshold,
                    )
                )

    if not all_results:
        logger.error("No subgroup results produced.")
        return 1

    df_subs = pd.DataFrame([s.__dict__ for s in all_results])
    df_disp = disparity_table(all_results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    df_subs.to_csv(args.output_dir / "subgroup_metrics.csv", index=False)
    df_disp.to_csv(args.output_dir / "disparity_metrics.csv", index=False)

    summary = {
        "subgroups": [s.__dict__ for s in all_results],
        "disparity": df_disp.to_dict(orient="records"),
        "minimum_subgroup_n": MIN_SUBGROUP_N,
    }
    (args.output_dir / "fairness_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    fig_disp = args.figures_dir / "fairness_disparity.png"
    plot_disparity(df_disp, fig_disp)
    logger.info("Disparity plot -> %s", fig_disp)

    if primary_extras is not None:
        for attr in args.attributes:
            if attr in primary_extras:
                fig_path = args.figures_dir / f"fairness_reliability_{attr.lower()}.png"
                plot_subgroup_reliability(
                    primary_extras["y_test"],
                    primary_extras["p_test"],
                    primary_extras[attr],
                    attr,
                    fig_path,
                )
                logger.info("Subgroup reliability (%s) -> %s", attr, fig_path)

    print()
    print("-- Subgroup metrics (abridged) --")
    print(
        df_subs[
            [
                "run_name",
                "attribute",
                "subgroup_label",
                "n",
                "base_rate",
                "selection_rate",
                "tpr",
                "fpr",
                "auc_roc",
                "ece",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )
    print("\n-- Disparity vs reference (largest subgroup) --")
    print(
        df_disp[
            [
                "run",
                "attribute",
                "subgroup_label",
                "demographic_parity_diff",
                "equal_opportunity_violation",
                "equalised_odds_violation",
                "auc_diff",
                "ece_diff",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:+.3f}")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
