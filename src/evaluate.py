"""
evaluate.py — Portfolio-level evaluation and model comparison.

Aggregates the per-run artefacts that ``train.py`` already emits into a single
report-ready comparison of the transformer against the random-forest baseline.

What this module does NOT do
----------------------------
It does not recompute metrics from scratch. ``train.py`` already writes:

    results/transformer/<run>/test_metrics.json    (metrics @ τ=0.5 + sweep)
    results/transformer/<run>/test_predictions.npz (y_true, y_prob, y_pred)
    results/rf_metrics.csv                         (RF baseline + tuned)

This module loads those, aggregates the transformer runs across seeds, and
adds a handful of metrics that ``train.py`` doesn't produce (Cohen's kappa,
specificity — useful for the report's §4 comparison table).

Outputs
-------
    results/comparison_table.csv    — side-by-side model comparison
    results/comparison_table.md     — markdown twin for the report appendix
    results/evaluate_summary.json   — raw per-seed numbers + aggregates

CLI
---
.. code-block:: bash

    poetry run python src/evaluate.py \\
        --from-scratch-runs results/transformer/seed_42 \\
                            results/transformer/seed_1 \\
                            results/transformer/seed_2 \\
        --mtlm-runs results/transformer/seed_42_mtlm_finetune \\
        --rf results/rf_metrics.csv \\
        --output-dir results

References (Plan sections): §10 evaluation, §10.4 model comparison table.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# Make src/ importable when this file is run as a script.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from train import compute_classification_metrics, compute_ece  # noqa: E402

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

#: Metrics reported in every row of the comparison table. Order is deliberate:
#: discrimination first, then threshold-dependent, then calibration.
REPORTED_METRICS: Tuple[str, ...] = (
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

#: RF metrics carried by the aggregate ``rf_metrics.csv``. When a full RF
#: prediction vector is available (via ``--rf-predictions``), every entry
#: in :data:`REPORTED_METRICS` is computed instead — including the ECE,
#: Brier, kappa, specificity columns that the aggregate CSV omits.
_RF_METRIC_COLUMNS: Tuple[str, ...] = (
    "auc_roc", "auc_pr", "f1", "accuracy", "precision", "recall",
)

#: Default decision threshold. Matches what ``train.py`` writes into the
#: ``metrics`` block of ``test_metrics.json``.
DEFAULT_THRESHOLD: float = 0.5


# ──────────────────────────────────────────────────────────────────────────────
# I/O — load per-run artefacts
# ──────────────────────────────────────────────────────────────────────────────


def load_test_metrics(run_dir: Path) -> Dict[str, Any]:
    """
    Load ``test_metrics.json`` + ``test_predictions.npz`` for a single run.

    Returns a dict with::

        {
            "run_name":   <basename of run_dir>,
            "metrics":    {auc_roc, auc_pr, f1, ... from test_metrics.json},
            "threshold":  float,
            "y_true":     np.ndarray (N,),
            "y_prob":     np.ndarray (N,),
            "y_pred":     np.ndarray (N,),
        }
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


def load_rf_from_predictions(rf_dir: Path) -> Optional[Dict[str, Any]]:
    """Load a full per-row RF test-set prediction artefact (produced by
    ``src/rf_predictions.py``) and compute every entry in
    :data:`REPORTED_METRICS` from it.

    Returns None if the directory or its artefacts are missing — callers
    fall back to the aggregate-CSV path in that case.
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


def load_rf_metrics(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Read ``results/rf_metrics.csv`` and return the baseline + tuned rows in a
    uniform dict keyed by display name.

    The CSV rows we care about are identified by the ``model`` column:

        * ``RF_baseline``  (split == ``test_baseline``)
        * ``RF_tuned``     (split == ``test``)
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    out: Dict[str, Dict[str, float]] = {}

    for display_name, model_key in (("rf_baseline", "RF_baseline"),
                                     ("rf_tuned", "RF_tuned")):
        match = df[df["model"] == model_key]
        if match.empty:
            logger.warning("No %s row in %s — skipping", model_key, csv_path)
            continue
        row = match.iloc[0]
        out[display_name] = {
            "auc_roc":   float(row["auc_roc"]),
            "auc_pr":    float(row["avg_precision"]),
            "f1":        float(row["f1"]),
            "accuracy":  float(row["accuracy"]),
            "precision": float(row["precision"]),
            "recall":    float(row["recall"]),
        }

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Metrics train.py doesn't already compute
# ──────────────────────────────────────────────────────────────────────────────


def compute_additional_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, float]:
    """
    Compute Cohen's kappa and specificity @ ``threshold``.

    These two are report-relevant but absent from ``train.py``'s default
    metrics bundle, so we compute them here from the predictions alone.
    """
    y_true_int = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)

    kappa = float(cohen_kappa_score(y_true_int, y_pred))

    # Specificity = TN / (TN + FP). Guard against degenerate single-class
    # prediction vectors (confusion_matrix returns a 1×1 array in that case).
    cm = confusion_matrix(y_true_int, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    return {"cohen_kappa": kappa, "specificity": specificity}


def metrics_for_run(run: Dict[str, Any]) -> Dict[str, float]:
    """
    Full per-run metrics dict — ``train.py``'s bundle plus Cohen's kappa and
    specificity, all at the same threshold.
    """
    base = dict(run["metrics"])
    extra = compute_additional_metrics(
        run["y_true"], run["y_prob"], threshold=run["threshold"],
    )
    return {**base, **extra}


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation across seeds
# ──────────────────────────────────────────────────────────────────────────────


def ensemble_run(
    runs: List[Dict[str, Any]],
    *,
    mode: str = "arithmetic",
    display_name: str = "ensemble",
    threshold: float = DEFAULT_THRESHOLD,
) -> Optional[Dict[str, Any]]:
    """Build a synthetic "run" whose predictions are the arithmetic or
    geometric mean of the inputs' per-row probabilities.

    Returns None when fewer than two runs are supplied (an ensemble of one
    is just that run). All inputs must share the same ``y_true`` — we
    enforce that with an explicit check because mismatched test splits
    produce silent nonsense otherwise.
    """
    if len(runs) < 2:
        return None
    y_true = runs[0]["y_true"]
    for r in runs[1:]:
        if r["y_true"].shape != y_true.shape or not np.array_equal(
            r["y_true"], y_true
        ):
            raise ValueError(
                f"Ensemble inputs disagree on y_true "
                f"({runs[0]['run_name']} vs {r['run_name']}) — refusing to average."
            )

    probs = np.stack([r["y_prob"].astype(np.float64) for r in runs], axis=0)
    if mode == "arithmetic":
        y_prob = probs.mean(axis=0)
    elif mode == "geometric":
        eps = 1e-7
        p = np.clip(probs, eps, 1.0 - eps)
        mean_logit = np.log(p / (1.0 - p)).mean(axis=0)
        y_prob = 1.0 / (1.0 + np.exp(-mean_logit))
    else:
        raise ValueError(f"mode must be 'arithmetic' or 'geometric', got {mode!r}")

    y_pred = (y_prob >= threshold).astype(int)
    # Reuse train.py's metric bundle so the ensemble row has the same
    # metric set as every other row in the table.
    from train import compute_classification_metrics  # late import ok
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


def aggregate_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate multiple single-seed runs into one ``mean ± std`` row.

    Returns::

        {
            "run_names": [...],
            "n_seeds":   int,
            "mean":      {metric_name: float, ...},
            "std":       {metric_name: float, ...},
            "per_seed":  [{metric_name: float, ...}, ...],   # raw per-seed
        }
    """
    per_seed = [metrics_for_run(r) for r in runs]
    keys = REPORTED_METRICS

    mean = {k: float(np.mean([m[k] for m in per_seed])) for k in keys}
    std = {k: float(np.std([m[k] for m in per_seed], ddof=1))
           if len(per_seed) > 1 else 0.0
           for k in keys}

    return {
        "run_names": [r["run_name"] for r in runs],
        "n_seeds": len(runs),
        "mean": mean,
        "std": std,
        "per_seed": per_seed,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Comparison table assembly
# ──────────────────────────────────────────────────────────────────────────────


def _format_mean_std(mean: float, std: float, n: int, digits: int = 4) -> str:
    """``0.7797 ± 0.0022`` when n > 1 else plain mean."""
    if n <= 1 or std == 0.0:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def build_comparison_table(
    transformer_from_scratch: Optional[Dict[str, Any]],
    transformer_mtlm: Optional[Dict[str, Any]],
    rf: Dict[str, Dict[str, float]],
    *,
    ensemble: Optional[Dict[str, Any]] = None,
    rf_full: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Build the report-ready comparison table.

    Columns: ``model`` + every entry in :data:`REPORTED_METRICS`. Cells are
    ``mean ± std`` strings for aggregated transformer rows, plain numbers
    elsewhere, and ``—`` where the metric isn't available.

    Parameters
    ----------
    transformer_from_scratch, transformer_mtlm
        Aggregated mean±std dicts from :func:`aggregate_runs`.
    rf
        Dict from :func:`load_rf_metrics` (from the aggregate CSV) — the
        fallback when no full RF prediction artefact is available.
    ensemble
        Optional single "run"-shaped dict from :func:`ensemble_run`, added
        as a row between the MTLM and the RF rows.
    rf_full
        Optional "run"-shaped RF dict from :func:`load_rf_from_predictions`.
        When present, the RF-tuned row is computed from the raw probability
        vector and every metric in :data:`REPORTED_METRICS` is populated —
        ECE, Brier, kappa, specificity included.
    """
    rows: List[Dict[str, Any]] = []

    def _blank_row(name: str) -> Dict[str, Any]:
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

    # RF baseline stays CSV-derived (no raw predictions for it on disk).
    if "rf_baseline" in rf:
        row = _blank_row("RF (baseline)")
        for k in _RF_METRIC_COLUMNS:
            if k in rf["rf_baseline"]:
                row[k] = f"{rf['rf_baseline'][k]:.4f}"
        rows.append(row)

    # RF tuned — prefer the full-prediction path when available.
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
    """
    Hand-rolled markdown table — avoids the optional ``tabulate`` dependency
    that ``DataFrame.to_markdown`` would pull in.
    """
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, separator]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    return "\n".join(lines) + "\n"


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


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
        help="Paths to from-scratch transformer run directories, "
             "aggregated as mean ± std.",
    )
    p.add_argument(
        "--mtlm-runs",
        nargs="*",
        default=["results/transformer/seed_42_mtlm_finetune"],
        type=Path,
        help="Paths to MTLM-fine-tuned run directories, aggregated separately.",
    )
    p.add_argument(
        "--rf", type=Path, default=Path("results/rf_metrics.csv"),
        help="Path to the RF metrics CSV produced by random_forest.py.",
    )
    p.add_argument(
        "--rf-predictions", type=Path, default=Path("results/rf"),
        help="Directory containing rf test_predictions.npz / test_metrics.json "
             "(from src/rf_predictions.py). When present, the RF tuned row is "
             "computed from the raw probability vector rather than the CSV.",
    )
    p.add_argument(
        "--ensemble-mode",
        choices=("arithmetic", "geometric", "none"),
        default="arithmetic",
        help="Aggregate per-seed from-scratch runs into a synthetic ensemble "
             "row. 'none' disables the extra row.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        help="Directory to write comparison_table.csv / .md / evaluate_summary.json.",
    )
    return p


def _load_all_or_empty(run_dirs: List[Path]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    # argparse default values bypass the ``type=`` conversion, so ensure we're
    # always working with Path instances regardless of how main() was called.
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


def main(argv: Optional[List[str]] = None) -> int:
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
            "Run `python src/rf_predictions.py` to populate them.",
            args.rf_predictions,
        )

    transformer_from_scratch = (
        aggregate_runs(from_scratch_runs) if from_scratch_runs else None
    )
    transformer_mtlm = (
        aggregate_runs(mtlm_runs) if mtlm_runs else None
    )
    ensemble = (
        None if args.ensemble_mode == "none"
        else ensemble_run(from_scratch_runs, mode=args.ensemble_mode,
                          display_name=f"ensemble_{args.ensemble_mode}")
    )

    table = build_comparison_table(
        transformer_from_scratch, transformer_mtlm, rf,
        ensemble=ensemble, rf_full=rf_full,
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
            if rf_full is not None else None
        ),
        "ensemble": (
            {"metrics": metrics_for_run(ensemble), "mode": args.ensemble_mode,
             "component_runs": ensemble.get("component_runs", [])}
            if ensemble is not None else None
        ),
    }
    # NumPy arrays from per_seed payloads get dropped — we only kept the
    # derived metrics dicts there. Raw arrays live in the *.npz files.
    json_path.write_text(json.dumps(summary, indent=2, default=str))

    logger.info("Comparison table written to %s", csv_path)
    logger.info("Markdown twin at             %s", md_path)
    logger.info("Raw summary at               %s", json_path)
    print()
    print(table.to_string(index=False))
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    raise SystemExit(main())
