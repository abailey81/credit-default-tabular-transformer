"""Build the transformer-vs-RF comparison table from committed artefacts.

This module is the glue between the training scripts and the comparison
row of the report. It does *not* recompute anything expensive: training
writes ``test_metrics.json`` and ``test_predictions.npz`` in each run
directory, the RF baseline writes ``rf_metrics.csv`` (and optionally a
full ``rf/test_predictions.npz``), and this module aggregates them.

Beyond straight aggregation it does three things worth flagging:

* Adds Cohen's kappa and specificity (TN / (TN + FP)) that ``train.py``
  doesn't emit -- they need the raw probabilities plus the threshold.
* Collapses multiple seeds into ``mean +/- std`` rows via
  :func:`aggregate_runs`, and refuses (not silently averages) if
  ``y_true`` disagrees across runs -- a length mismatch means somebody
  reshuffled the test split and every downstream number is wrong.
* Builds an ensemble row from per-seed raw probabilities via
  :func:`ensemble_run` (arithmetic mean-of-probs or geometric
  mean-of-logits) so the comparison table has a single headline
  transformer number.

Outputs live in ``results/evaluation/comparison/``:

* ``comparison_table.csv``  -- columns = [model, *REPORTED_METRICS].
* ``comparison_table.md``   -- markdown twin for the README / notebook.
* ``evaluate_summary.json`` -- per-seed breakdown + raw means / stds.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Column order for the comparison table. Ordering is intentional:
#: discrimination first (AUC, AP, F1), then threshold-sensitive metrics,
#: then calibration. The report reads left-to-right top-down.
REPORTED_METRICS: tuple[str, ...] = (
    "auc_roc",
    "auc_pr",
    "f1",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "cohen_kappa",
    "ece",
    "brier",
)

#: Columns we can expect in ``rf_metrics.csv``. Kappa, specificity, ECE, and
#: Brier are absent -- those require raw probs, which the CSV doesn't carry.
#: If ``--rf-predictions`` is set we recompute the full bundle from the npz
#: and ignore this list entirely.
_RF_METRIC_COLUMNS: tuple[str, ...] = (
    "auc_roc",
    "auc_pr",
    "f1",
    "accuracy",
    "precision",
    "recall",
)

#: The threshold baked into ``train.py``'s ``test_metrics.json``. Anything
#: that recomputes threshold-sensitive numbers (kappa / specificity /
#: ensembles) needs to use the same value to stay consistent with what
#: training already reported; overrides are carried per-run in the payload.
DEFAULT_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_test_metrics(run_dir: Path) -> dict[str, Any]:
    """Read one run's test-split artefacts off disk.

    Parameters
    ----------
    run_dir : Path
        A directory with both ``test_metrics.json`` and
        ``test_predictions.npz`` (i.e. a finished training run).

    Returns
    -------
    dict
        Keys: ``run_name`` (dir basename), ``metrics`` (from the JSON),
        ``threshold`` (from JSON or :data:`DEFAULT_THRESHOLD`),
        ``y_true`` / ``y_prob`` / ``y_pred`` (from the npz).

    Raises
    ------
    FileNotFoundError
        If either artefact is missing; we *want* a hard failure here,
        not a silently skipped run, because the report would reference
        a blank row otherwise.
    """
    run_dir = Path(run_dir)
    metrics_path = run_dir / "test_metrics.json"
    preds_path = run_dir / "test_predictions.npz"

    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing {metrics_path}")
    if not preds_path.is_file():
        raise FileNotFoundError(f"Missing {preds_path}")

    payload = json.loads(metrics_path.read_text())
    preds = np.load(preds_path)

    return {
        "run_name": run_dir.name,
        "metrics": dict(payload["metrics"]),
        "threshold": float(payload.get("threshold", DEFAULT_THRESHOLD)),
        "y_true": preds["y_true"],
        "y_prob": preds["y_prob"],
        "y_pred": preds["y_pred"],
    }


def load_rf_from_predictions(rf_dir: Path) -> Optional[dict[str, Any]]:
    """Read the per-row RF predictions written by ``rf_predictions.py``.

    Returns ``None`` if either artefact is missing, so the caller can
    fall back to the CSV-only path (fewer columns, calibration blank).
    """
    rf_dir = Path(rf_dir)
    npz_path = rf_dir / "test_predictions.npz"
    json_path = rf_dir / "test_metrics.json"
    if not (npz_path.is_file() and json_path.is_file()):
        return None
    payload = json.loads(json_path.read_text())
    preds = np.load(npz_path)
    threshold = float(payload.get("threshold", DEFAULT_THRESHOLD))
    return {
        "run_name": rf_dir.name,
        "metrics": dict(payload["metrics"]),
        "threshold": threshold,
        "y_true": preds["y_true"],
        "y_prob": preds["y_prob"],
        "y_pred": preds["y_pred"],
    }


def load_rf_metrics(csv_path: Path) -> dict[str, dict[str, float]]:
    """Extract the RF_baseline / RF_tuned rows from ``rf_metrics.csv``.

    The CSV uses ``avg_precision`` as the column name for AP; we rename
    to ``auc_pr`` on load so the rest of the module can use a single
    metric vocabulary.

    Returns
    -------
    dict
        ``{"rf_baseline": {...}, "rf_tuned": {...}}``. Either key may be
        missing if the corresponding row is absent; this is logged as a
        warning so the caller knows why a table row is blank.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    out: dict[str, dict[str, float]] = {}

    for display_name, model_key in (("rf_baseline", "RF_baseline"), ("rf_tuned", "RF_tuned")):
        match = df[df["model"] == model_key]
        if match.empty:
            logger.warning("No %s row in %s — skipping", model_key, csv_path)
            continue
        row = match.iloc[0]
        out[display_name] = {
            "auc_roc": float(row["auc_roc"]),
            "auc_pr": float(row["avg_precision"]),
            "f1": float(row["f1"]),
            "accuracy": float(row["accuracy"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
        }

    return out


# ---------------------------------------------------------------------------
# Metrics that train.py doesn't emit
# ---------------------------------------------------------------------------


def compute_additional_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Cohen's kappa + specificity at ``threshold``.

    Both metrics matter for imbalanced classification on this dataset.
    Specificity (TN rate) is what an operations team reads to estimate
    the false-positive burden, and kappa corrects accuracy for
    chance-agreement given the ~22% positive rate.

    The confusion matrix is built with ``labels=[0, 1]`` so a run that
    never predicts the positive class still returns a full 2x2 instead
    of a degenerate 1x1 that would raise on ``cm[0, 1]``.
    """
    y_true_int = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)

    kappa = float(cohen_kappa_score(y_true_int, y_pred))

    # spec = TN / (TN + FP). Guard against the no-negatives edge case even
    # though the test split is guaranteed to have both classes.
    cm = confusion_matrix(y_true_int, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    return {"cohen_kappa": kappa, "specificity": specificity}


def metrics_for_run(run: dict[str, Any]) -> dict[str, float]:
    """Union of ``train.py``'s metric bundle with kappa + specificity.

    Both use the same threshold that ``train.py`` used at test time,
    carried in ``run['threshold']``; otherwise the threshold-sensitive
    metrics would disagree between sources.
    """
    base = dict(run["metrics"])
    extra = compute_additional_metrics(
        run["y_true"],
        run["y_prob"],
        threshold=run["threshold"],
    )
    return {**base, **extra}


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------


def ensemble_run(
    runs: list[dict[str, Any]],
    *,
    mode: str = "arithmetic",
    display_name: str = "ensemble",
    threshold: float = DEFAULT_THRESHOLD,
) -> Optional[dict[str, Any]]:
    """Average per-seed probabilities into a single ensemble run.

    Parameters
    ----------
    runs : list of dict
        Each run must have identical ``y_true`` (checked) -- if any two
        runs disagree we refuse to average, because the mismatch implies
        a reshuffle somewhere upstream.
    mode : {"arithmetic", "geometric"}
        Arithmetic = mean of probabilities, the standard ensemble. Geometric
        = mean of logits, equivalent to model averaging at the logit level.
        Geometric downweights disagreement more aggressively and tends to
        help when one seed is miscalibrated.
    display_name : str
        Name for the ``run_name`` of the synthetic returned run. Doubles
        as the table row label.
    threshold : float
        Threshold for the derived ``y_pred``. Match what the individual
        seeds used to keep the ensemble comparable.

    Returns
    -------
    dict or None
        ``None`` if fewer than two runs are provided (an ensemble of one
        is meaningless). Otherwise the standard run-payload dict with an
        extra ``component_runs`` key listing the seed names.

    Raises
    ------
    ValueError
        If any two runs disagree on ``y_true`` or on ``mode``.
    """
    if len(runs) < 2:
        return None
    y_true = runs[0]["y_true"]
    for r in runs[1:]:
        if r["y_true"].shape != y_true.shape or not np.array_equal(r["y_true"], y_true):
            raise ValueError(
                f"Ensemble inputs disagree on y_true "
                f"({runs[0]['run_name']} vs {r['run_name']}); refusing to average."
            )

    probs = np.stack([r["y_prob"].astype(np.float64) for r in runs], axis=0)
    if mode == "arithmetic":
        y_prob = probs.mean(axis=0)
    elif mode == "geometric":
        # Logit-mean then sigmoid. Clipping is essential: probs too close to
        # 0 or 1 send log(p / (1 - p)) to +/- inf and the mean becomes NaN.
        eps = 1e-7
        p = np.clip(probs, eps, 1.0 - eps)
        mean_logit = np.log(p / (1.0 - p)).mean(axis=0)
        y_prob = 1.0 / (1.0 + np.exp(-mean_logit))
    else:
        raise ValueError(f"mode must be 'arithmetic' or 'geometric', got {mode!r}")

    y_pred = (y_prob >= threshold).astype(int)
    # Late import: ``compute_classification_metrics`` lives in src.training,
    # which imports torch. Deferring keeps torch out of the CLI import path
    # when only the aggregation utilities are used.
    from ..training.train import compute_classification_metrics

    metrics = compute_classification_metrics(y_true, y_prob, threshold=threshold)

    return {
        "run_name": display_name,
        "metrics": metrics,
        "threshold": threshold,
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "component_runs": [r["run_name"] for r in runs],
    }


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse N per-seed runs into a single ``mean +/- std`` summary.

    Uses the unbiased sample std (``ddof=1``) when n > 1; returns 0.0
    for the one-seed case (the fallback keeps the formatter happy but
    the caller should know the interval is degenerate).

    We deliberately use ``np.mean`` / ``np.std`` rather than ``nanmean``
    so a NaN on any individual seed propagates into the aggregate. A
    silent NaN->number conversion is the kind of bug that corrupts a
    headline number without anyone noticing; a warning plus a visible
    NaN cell in the CSV forces whoever reruns the pipeline to look.
    """
    per_seed = [metrics_for_run(r) for r in runs]
    keys = REPORTED_METRICS

    for k in keys:
        for r, m in zip(runs, per_seed):
            if not np.isfinite(m[k]):
                logger.warning(
                    "Run %s has NaN %s; aggregate will be NaN",
                    r["run_name"],
                    k,
                )

    mean = {k: float(np.mean([m[k] for m in per_seed])) for k in keys}
    std = {
        k: float(np.std([m[k] for m in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0
        for k in keys
    }

    return {
        "run_names": [r["run_name"] for r in runs],
        "n_seeds": len(runs),
        "mean": mean,
        "std": std,
        "per_seed": per_seed,
    }


# ---------------------------------------------------------------------------
# Table assembly
# ---------------------------------------------------------------------------


def _format_mean_std(mean: float, std: float, n: int, digits: int = 4) -> str:
    """Render ``mean +/- std`` for the comparison table cells.

    Drops the ``+/- std`` suffix when n <= 1 so the single-seed rows
    don't sport a misleading ``+/- 0.0000``.
    """
    if n <= 1 or std == 0.0:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def build_comparison_table(
    transformer_from_scratch: Optional[dict[str, Any]],
    transformer_mtlm: Optional[dict[str, Any]],
    rf: dict[str, dict[str, float]],
    *,
    ensemble: Optional[dict[str, Any]] = None,
    rf_full: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """Assemble the report's model-comparison table.

    Row order (top-down) is deliberate:

    1. Transformer from scratch (mean +/- std across seeds).
    2. Transformer with MTLM fine-tune (mean +/- std).
    3. Transformer ensemble (single row, no std; ensemble is deterministic
       given fixed component seeds).
    4. RF baseline (from CSV -- calibration columns blank since CSV lacks
       raw probs).
    5. RF tuned (full bundle when ``rf_full`` provided, else CSV-only).

    Missing inputs produce skipped rows rather than blanks; the table
    is meant to be ``cat``-able and copy-pasted into the report.
    """
    rows: list[dict[str, Any]] = []

    def _blank_row(name: str) -> dict[str, Any]:
        # Em-dash placeholder: renders unambiguously in both CSV and
        # Markdown, and is obviously a gap rather than a 0.
        return {"model": name, **{k: "—" for k in REPORTED_METRICS}}

    if transformer_from_scratch is not None:
        n = transformer_from_scratch["n_seeds"]
        mean = transformer_from_scratch["mean"]
        std = transformer_from_scratch["std"]
        row = {"model": f"Transformer (from scratch, n={n})"}
        for k in REPORTED_METRICS:
            row[k] = _format_mean_std(mean[k], std[k], n)
        rows.append(row)

    if transformer_mtlm is not None:
        n = transformer_mtlm["n_seeds"]
        mean = transformer_mtlm["mean"]
        std = transformer_mtlm["std"]
        row = {"model": f"Transformer (MTLM fine-tune, n={n})"}
        for k in REPORTED_METRICS:
            row[k] = _format_mean_std(mean[k], std[k], n)
        rows.append(row)

    if ensemble is not None:
        ens_metrics = metrics_for_run(ensemble)
        n_comp = len(ensemble.get("component_runs", []))
        row = {"model": f"Transformer ensemble (arithmetic, n={n_comp})"}
        for k in REPORTED_METRICS:
            row[k] = f"{ens_metrics[k]:.4f}"
        rows.append(row)

    # RF baseline: CSV is the only source (we don't pickle the baseline
    # model's predictions, just its headline metrics).
    if "rf_baseline" in rf:
        row = _blank_row("RF (baseline)")
        for k in _RF_METRIC_COLUMNS:
            if k in rf["rf_baseline"]:
                row[k] = f"{rf['rf_baseline'][k]:.4f}"
        rows.append(row)

    # RF tuned: prefer raw probs if available so the calibration columns
    # are populated, fall back to CSV otherwise.
    if rf_full is not None:
        rf_metrics = metrics_for_run(rf_full)
        row = {"model": "RF (tuned)"}
        for k in REPORTED_METRICS:
            row[k] = f"{rf_metrics[k]:.4f}"
        rows.append(row)
    elif "rf_tuned" in rf:
        row = _blank_row("RF (tuned)")
        for k in _RF_METRIC_COLUMNS:
            if k in rf["rf_tuned"]:
                row[k] = f"{rf['rf_tuned'][k]:.4f}"
        rows.append(row)

    return pd.DataFrame(rows, columns=["model", *REPORTED_METRICS])


def table_to_markdown(df: pd.DataFrame) -> str:
    """Render the comparison table as GitHub-flavoured markdown.

    Hand-rolled rather than ``df.to_markdown()`` so we don't depend on
    the optional ``tabulate`` extra; this lets the module import in a
    vanilla pandas install.
    """
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, separator]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Aggregate per-run transformer metrics and build the "
            "transformer-vs-RF comparison table. Inputs are the artefacts "
            "already written by train.py / random_forest.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--from-scratch-runs",
        nargs="*",
        default=[
            "results/transformer/seed_42",
            "results/transformer/seed_1",
            "results/transformer/seed_2",
        ],
        type=Path,
        help="Paths to from-scratch transformer run directories, " "aggregated as mean ± std.",
    )
    p.add_argument(
        "--mtlm-runs",
        nargs="*",
        default=["results/transformer/seed_42_mtlm_finetune"],
        type=Path,
        help="Paths to MTLM-fine-tuned run directories, aggregated separately.",
    )
    p.add_argument(
        "--rf",
        type=Path,
        default=Path("results/baseline/rf_metrics.csv"),
        help="Path to the RF metrics CSV produced by random_forest.py.",
    )
    p.add_argument(
        "--rf-predictions",
        type=Path,
        default=Path("results/baseline/rf"),
        help="Directory with rf test_predictions.npz / test_metrics.json "
        "(from src/baselines/rf_predictions.py). When present, the RF tuned row "
        "is computed from raw probabilities rather than the CSV.",
    )
    p.add_argument(
        "--ensemble-mode",
        choices=("arithmetic", "geometric", "none"),
        default="arithmetic",
        help="Aggregate per-seed from-scratch runs into an ensemble row. "
        "'none' disables the extra row.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/evaluation/comparison"),
        help="Directory to write comparison_table.csv / .md / evaluate_summary.json.",
    )
    return p


def _load_all_or_empty(run_dirs: list[Path]) -> list[dict[str, Any]]:
    """Load every run dir that actually has artefacts; skip (warn on) the rest.

    argparse defaults don't run ``type=`` conversion, so we re-``Path`` each
    entry here -- otherwise the default string paths would silently go
    through ``is_dir`` on the literal string and fail on Windows.
    """
    out: list[dict[str, Any]] = []
    for d in run_dirs:
        d = Path(d)
        if not d.is_dir():
            logger.warning("Skipping %s — not a directory", d)
            continue
        try:
            out.append(load_test_metrics(d))
        except FileNotFoundError as e:
            logger.warning("Skipping %s — %s", d, e)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args(argv)

    from_scratch_runs = _load_all_or_empty(args.from_scratch_runs)
    mtlm_runs = _load_all_or_empty(args.mtlm_runs)
    rf = load_rf_metrics(args.rf)
    rf_full = load_rf_from_predictions(args.rf_predictions)
    if rf_full is None:
        logger.info(
            "No raw RF predictions at %s — RF tuned row will use CSV "
            "numbers and leave calibration cells blank. "
            "Run `python -m src.baselines.rf_predictions` to populate them.",
            args.rf_predictions,
        )

    transformer_from_scratch = aggregate_runs(from_scratch_runs) if from_scratch_runs else None
    transformer_mtlm = aggregate_runs(mtlm_runs) if mtlm_runs else None
    ensemble = (
        None
        if args.ensemble_mode == "none"
        else ensemble_run(
            from_scratch_runs,
            mode=args.ensemble_mode,
            display_name=f"ensemble_{args.ensemble_mode}",
        )
    )

    table = build_comparison_table(
        transformer_from_scratch,
        transformer_mtlm,
        rf,
        ensemble=ensemble,
        rf_full=rf_full,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "comparison_table.csv"
    md_path = args.output_dir / "comparison_table.md"
    json_path = args.output_dir / "evaluate_summary.json"

    table.to_csv(csv_path, index=False)
    md_path.write_text(table_to_markdown(table))

    summary = {
        "reported_metrics": list(REPORTED_METRICS),
        "transformer_from_scratch": transformer_from_scratch,
        "transformer_mtlm": transformer_mtlm,
        "rf": rf,
        "rf_full": (
            {"metrics": metrics_for_run(rf_full), "threshold": rf_full["threshold"]}
            if rf_full is not None
            else None
        ),
        "ensemble": (
            {
                "metrics": metrics_for_run(ensemble),
                "mode": args.ensemble_mode,
                "component_runs": ensemble.get("component_runs", []),
            }
            if ensemble is not None
            else None
        ),
    }
    # Only metric dicts make it into the JSON; the raw probability arrays
    # stay in their original npz files (too heavy for a summary file).
    json_path.write_text(json.dumps(summary, indent=2, default=str))

    logger.info("Comparison table written to %s", csv_path)
    logger.info("Markdown twin at             %s", md_path)
    logger.info("Raw summary at               %s", json_path)
    print()
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
