"""Aggregate per-run artefacts into one transformer-vs-RF comparison table.
Reuses whatever train.py / random_forest.py already wrote; doesn't recompute.
Adds kappa + specificity (train.py doesn't emit those). Writes
comparison_table.{csv,md} and evaluate_summary.json."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from ..training.train import compute_classification_metrics

logger = logging.getLogger(__name__)

# constants

#: table columns. order: discrimination → threshold → calibration
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

#: columns in rf_metrics.csv. if --rf-predictions is set we compute the
#: full REPORTED_METRICS set from raw probs instead (CSV lacks ECE/Brier/kappa/spec).
_RF_METRIC_COLUMNS: Tuple[str, ...] = (
    "auc_roc", "auc_pr", "f1", "accuracy", "precision", "recall",
)

#: matches the threshold train.py bakes into test_metrics.json
DEFAULT_THRESHOLD: float = 0.5


# I/O


def load_test_metrics(run_dir: Path) -> Dict[str, Any]:
    """load test_metrics.json + test_predictions.npz for one run."""
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
    """per-row RF preds from rf_predictions.py. None → caller falls back to CSV."""
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
    """rf_metrics.csv → {rf_baseline, rf_tuned} keyed dict. picks the RF_baseline
    and RF_tuned rows off the `model` column."""
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


# metrics train.py skips


def compute_additional_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, float]:
    """κ + specificity @ threshold. train.py doesn't emit either."""
    y_true_int = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)

    kappa = float(cohen_kappa_score(y_true_int, y_pred))

    # spec = TN/(TN+FP). labels=[0,1] avoids 1×1 CM on single-class preds
    cm = confusion_matrix(y_true_int, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    return {"cohen_kappa": kappa, "specificity": specificity}


def metrics_for_run(run: Dict[str, Any]) -> Dict[str, float]:
    """train.py bundle + κ + specificity, same threshold throughout."""
    base = dict(run["metrics"])
    extra = compute_additional_metrics(
        run["y_true"], run["y_prob"], threshold=run["threshold"],
    )
    return {**base, **extra}


# aggregation across seeds


def ensemble_run(
    runs: List[Dict[str, Any]],
    *,
    mode: str = "arithmetic",
    display_name: str = "ensemble",
    threshold: float = DEFAULT_THRESHOLD,
) -> Optional[Dict[str, Any]]:
    """build a synthetic run from mean(probs). arithmetic=mean p, geometric=mean logit.
    <2 runs → None. Raises if y_true mismatches (would silently corrupt averages)."""
    if len(runs) < 2:
        return None
    y_true = runs[0]["y_true"]
    for r in runs[1:]:
        if r["y_true"].shape != y_true.shape or not np.array_equal(
            r["y_true"], y_true
        ):
            raise ValueError(
                f"Ensemble inputs disagree on y_true "
                f"({runs[0]['run_name']} vs {r['run_name']}); refusing to average."
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
    # reuse train.py's bundle so the ensemble row matches table shape
    from ..training.train import compute_classification_metrics  # late import ok
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
    """collapse N single-seed runs → one mean ± std row."""
    per_seed = [metrics_for_run(r) for r in runs]
    keys = REPORTED_METRICS

    # warn on NaN, but keep mean/std (not nanmean) so the NaN hits the CSV —
    # silent blanks are worse than obvious NaN
    for k in keys:
        for r, m in zip(runs, per_seed):
            if not np.isfinite(m[k]):
                logger.warning(
                    "Run %s has NaN %s; aggregate will be NaN",
                    r["run_name"], k,
                )

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


# table assembly


def _format_mean_std(mean: float, std: float, n: int, digits: int = 4) -> str:
    """'0.7797 ± 0.0022' if n>1 else plain mean."""
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
    """assemble the comparison table. cols = [model, *REPORTED_METRICS].
    transformer rows: mean±std. RF tuned uses rf_full probs if present,
    else CSV (leaves calibration cells as —). ensemble slots between MTLM and RF."""
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

    # RF baseline: CSV only, no raw preds on disk
    if "rf_baseline" in rf:
        row = _blank_row("RF (baseline)")
        for k in _RF_METRIC_COLUMNS:
            if k in rf["rf_baseline"]:
                row[k] = f"{rf['rf_baseline'][k]:.4f}"
        rows.append(row)

    # prefer raw-pred path for RF tuned if available
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
    """hand-rolled md table. avoids the optional tabulate dep on to_markdown."""
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, separator]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    return "\n".join(lines) + "\n"


# CLI


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
        "--rf", type=Path, default=Path("results/baseline/rf_metrics.csv"),
        help="Path to the RF metrics CSV produced by random_forest.py.",
    )
    p.add_argument(
        "--rf-predictions", type=Path, default=Path("results/baseline/rf"),
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
        "--output-dir", type=Path, default=Path("results/evaluation/comparison"),
        help="Directory to write comparison_table.csv / .md / evaluate_summary.json.",
    )
    return p


def _load_all_or_empty(run_dirs: List[Path]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    # argparse defaults skip type= conversion → always re-Path here
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
            "Run `python -m src.baselines.rf_predictions` to populate them.",
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
    # per_seed holds metric dicts only; raw arrays live in the npz files
    json_path.write_text(json.dumps(summary, indent=2, default=str))

    logger.info("Comparison table written to %s", csv_path)
    logger.info("Markdown twin at             %s", md_path)
    logger.info("Raw summary at               %s", json_path)
    print()
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
