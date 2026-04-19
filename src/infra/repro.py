"""Reproducibility gate: regenerate, diff, and fail-fast on drift.

This is the CI check that catches silent regressions. It runs the
subset of the pipeline that is cheap to regenerate (RF predictions, the
evaluate comparison table) from the *committed* preprocessed splits and
diffs the fresh output against the committed artefacts. If anything
moves — a dependency bump changes scikit-learn's RNG path, someone
edits a config without regenerating the CSV, the split hashes fall out
of sync — this exits non-zero and CI fails.

Checks (run by ``run_all`` in order):

1. ``artefacts_exist``             — every committed output is present.
2. ``transformer_run_files``       — every per-seed run dir is complete.
3. ``split_hashes_match``          — SHA-256 of split CSVs vs
                                     ``SPLIT_HASHES.md``.
4. ``python_pins``                 — pyproject pins python + torch.
5. ``git_clean``                   — informational; never fails.
6. ``rf_predictions_regenerate``   — regen + bitwise diff against
                                     committed RF predictions.
7. ``evaluate_regenerates``        — regen + numeric-tolerance diff
                                     against the comparison table.

Full taxonomy + add-a-check runbook: docs/REPRODUCIBILITY.md.

Invocation
----------

    python -m src.infra.repro                 # default: check repo root
    python -m src.infra.repro --repo /path    # alt root
    python -m src.infra.repro --scratch /tmp  # alt scratch dir

The JSON report at ``--report`` is a machine-parseable archive of every
check's pass/fail + detail + metadata; CI uploads it as a build artefact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    # Result dataclasses
    "Check",
    "Report",
    # Individual check functions
    "check_artefacts_exist",
    "check_transformer_run_files",
    "check_evaluate_regenerates",
    "check_rf_predictions_regenerate",
    "check_split_hashes_match",
    "check_python_pins",
    "check_git_clean",
    # Driver
    "run_all",
    "main",
]


@dataclass
class Check:
    """One check's outcome.

    ``detail`` is human-readable (appears in the console output);
    ``metadata`` is machine-parseable (goes into the JSON report so
    CI dashboards / post-mortem scripts can read it without parsing
    the detail string).
    """

    name: str
    passed: bool
    detail: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Report:
    """Aggregated result across every ``Check``."""

    checks: List[Check] = field(default_factory=list)

    def add(self, check: Check) -> None:
        self.checks.append(check)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def as_dict(self) -> Dict[str, Any]:
        """JSON-ready view of the report. Used to persist to disk."""
        return {
            "all_passed": self.all_passed,
            "n_checks": len(self.checks),
            "n_passed": sum(c.passed for c in self.checks),
            "checks": [c.__dict__ for c in self.checks],
        }


def _sha256(path: Path) -> str:
    """Chunked SHA-256 of a file (64 KB reads) so large artefacts
    don't force-load into memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: List[str], cwd: Path) -> Tuple[int, str, str]:
    """Subprocess wrapper returning ``(rc, stdout, stderr)``."""
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def _compare_dataframes(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    rtol: float = 1e-4,
) -> Tuple[bool, str]:
    """Tolerant dataframe equality. Numeric columns compared with
    ``np.allclose``; non-numeric (which in the comparison table is
    things like ``"0.7797 ± 0.0023"``) fall back to string equality."""
    if a.shape != b.shape:
        return False, f"shape mismatch: {a.shape} vs {b.shape}"
    if list(a.columns) != list(b.columns):
        return False, "column order mismatch"
    for col in a.columns:
        ser_a, ser_b = a[col], b[col]
        try:
            arr_a = pd.to_numeric(ser_a, errors="raise").values
            arr_b = pd.to_numeric(ser_b, errors="raise").values
            if not np.allclose(arr_a, arr_b, rtol=rtol, atol=1e-6, equal_nan=True):
                return False, f"column {col!r} differs beyond rtol={rtol}"
        except (ValueError, TypeError):
            # Non-numeric column — string compare handles unicode dashes,
            # "mean ± std" cells, and similar formatted values.
            if (ser_a.astype(str).values != ser_b.astype(str).values).any():
                return False, f"column {col!r} string mismatch"
    return True, "ok"


def check_artefacts_exist(repo: Path) -> Check:
    """Fail fast if any committed artefact is missing.

    We check only a curated list (not every file under ``results/``) so
    that adding an optional figure to a notebook doesn't break repro.
    The list is the hard contract — every entry here is something a
    downstream module imports.
    """
    required = [
        "results/baseline/rf_metrics.csv",
        "results/baseline/rf_config.json",
        "results/evaluation/comparison/comparison_table.csv",
        "results/evaluation/comparison/comparison_table.md",
        "results/evaluation/comparison/evaluate_summary.json",
        "data/processed/splits/train_scaled.csv",
        "data/processed/splits/val_scaled.csv",
        "data/processed/splits/test_scaled.csv",
        "data/processed/splits/test_raw.csv",
        "data/processed/feature_metadata.json",
    ]
    missing = [r for r in required if not (repo / r).is_file()]
    return Check(
        name="artefacts_exist",
        passed=not missing,
        detail=("all present" if not missing else f"missing: {missing}"),
        metadata={"missing": missing, "checked_files": required},
    )


def check_transformer_run_files(repo: Path) -> Check:
    """Every ``results/transformer/seed_*`` directory must carry the
    files downstream modules consume (calibration reads ``val_predictions.npz``,
    significance reads ``test_predictions.npz``, etc.). Partial runs are
    not merged, so any missing file here is a regression."""
    run_root = repo / "results" / "transformer"
    required_per_run = [
        "config.json",
        "train_log.csv",
        "train_metrics.json",
        "train_predictions.npz",
        "val_metrics.json",
        "val_predictions.npz",
        "test_metrics.json",
        "test_predictions.npz",
    ]
    missing: List[str] = []
    checked_runs: List[str] = []
    for run in sorted(run_root.glob("seed_*")):
        if not run.is_dir():
            continue
        checked_runs.append(run.name)
        for f in required_per_run:
            if not (run / f).is_file():
                missing.append(f"{run.name}/{f}")
    return Check(
        name="transformer_run_files",
        passed=not missing,
        detail=(
            f"{len(checked_runs)} runs OK" if not missing else f"missing files: {missing[:5]}..."
        ),
        metadata={"runs": checked_runs, "missing": missing},
    )


def check_evaluate_regenerates(repo: Path, scratch: Path) -> Check:
    """Regenerate the comparison table + diff against the committed copy.

    Uses ``rtol=1e-4`` rather than bitwise equality because the comparison
    table carries aggregate statistics (mean AUC-ROC, etc.) that can shift
    by ULPs between numpy versions. 1e-4 is tight enough to catch any real
    regression and loose enough to ignore numpy noise.
    """
    out_dir = scratch / "eval"
    rc, stdout, stderr = _run(
        [
            sys.executable,
            "-m",
            "src.evaluation.evaluate",
            "--output-dir",
            str(out_dir),
            "--ensemble-mode",
            "arithmetic",
        ],
        cwd=repo,
    )
    if rc != 0:
        return Check(
            name="evaluate_regenerates",
            passed=False,
            detail="src.evaluation.evaluate returned non-zero exit",
            metadata={"stderr": stderr[-400:]},
        )
    try:
        committed = pd.read_csv(
            repo / "results" / "evaluation" / "comparison" / "comparison_table.csv"
        )
        regen = pd.read_csv(out_dir / "comparison_table.csv")
    except FileNotFoundError as e:
        return Check(
            name="evaluate_regenerates",
            passed=False,
            detail=f"read failed: {e}",
        )
    ok, detail = _compare_dataframes(committed, regen, rtol=1e-4)
    return Check(
        name="evaluate_regenerates",
        passed=ok,
        detail=detail,
        metadata={"n_rows": len(regen), "n_cols": len(regen.columns)},
    )


def check_rf_predictions_regenerate(repo: Path, scratch: Path) -> Check:
    """Regenerate RF test predictions + diff bitwise against the committed
    copy. Threshold 1e-6 is effectively bitwise given sklearn's float64
    RNG; any drift here means the RF fit changed. Historic observed max
    |Δp| = 0 for clean rebuilds."""
    committed_dir = repo / "results" / "baseline" / "rf"
    committed_npz = committed_dir / "test_predictions.npz"
    if not committed_npz.is_file():
        return Check(
            name="rf_predictions_regenerate",
            passed=False,
            detail=f"no committed artefact at {committed_npz}",
        )
    out_dir = scratch / "rf"
    rc, stdout, stderr = _run(
        [sys.executable, "-m", "src.baselines.rf_predictions", "--output-dir", str(out_dir)],
        cwd=repo,
    )
    if rc != 0:
        return Check(
            name="rf_predictions_regenerate",
            passed=False,
            detail="src.baselines.rf_predictions returned non-zero",
            metadata={"stderr": stderr[-400:]},
        )
    regen = np.load(out_dir / "test_predictions.npz")
    comm = np.load(committed_npz)
    if regen["y_prob"].shape != comm["y_prob"].shape:
        return Check(
            name="rf_predictions_regenerate",
            passed=False,
            detail=f"shape mismatch: {regen['y_prob'].shape} vs {comm['y_prob'].shape}",
        )
    # Promote to float64 before subtraction — sklearn emits float64, but a
    # future optimisation could emit float32 and the diff would be 1e-7
    # noise on every entry.
    max_diff = float(
        np.abs(regen["y_prob"].astype(np.float64) - comm["y_prob"].astype(np.float64)).max()
    )
    ok = max_diff < 1e-6
    return Check(
        name="rf_predictions_regenerate",
        passed=ok,
        detail=f"max |delta p| = {max_diff:.2e} (threshold 1e-6)",
        metadata={"max_prob_diff": max_diff},
    )


def check_split_hashes_match(repo: Path) -> Check:
    """Compare SHA-256 of ``data/processed/`` artefacts against
    ``SPLIT_HASHES.md``.

    A miss here is the single most important signal in this whole module:
    if the splits have drifted, *every* downstream metric is computed on
    different rows and all comparisons are unsafe. It is the one check
    that is worth fixing before any other failing check.

    Lookup strategy: the hash ledger is keyed by bare filename (e.g.
    ``train_raw.csv``). After the 2026 reorg the split CSVs live under
    ``data/processed/splits/`` while the metadata JSON stays at
    ``data/processed/``. We probe both locations in a deterministic order
    (splits first, then root) so either layout round-trips cleanly.
    """
    import re

    hashes_md = repo / "data" / "processed" / "SPLIT_HASHES.md"
    if not hashes_md.is_file():
        return Check(
            name="split_hashes_match",
            passed=False,
            detail=f"missing {hashes_md.relative_to(repo)}",
        )

    expected: Dict[str, str] = {}
    # Markdown format: "| `filename.csv` | `sha256...` | ..."
    hash_re = re.compile(r"\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|")
    for line in hashes_md.read_text().splitlines():
        m = hash_re.search(line)
        if m:
            expected[m.group(1)] = m.group(2)
    if not expected:
        return Check(
            name="split_hashes_match",
            passed=False,
            detail="no hash rows parsed from SPLIT_HASHES.md",
        )

    processed_root = repo / "data" / "processed"
    probe_dirs = (processed_root / "splits", processed_root)

    mismatches: List[str] = []
    checked: List[str] = []
    for name, want in expected.items():
        path: Optional[Path] = None
        for d in probe_dirs:
            candidate = d / name
            if candidate.is_file():
                path = candidate
                break
        if path is None:
            mismatches.append(f"{name}: missing")
            continue
        got = _sha256(path)
        checked.append(name)
        if got != want:
            mismatches.append(f"{name}: got {got[:12]}... want {want[:12]}...")
    ok = not mismatches
    return Check(
        name="split_hashes_match",
        passed=ok,
        detail=(
            f"{len(checked)}/{len(expected)} match"
            if ok
            else f"{len(mismatches)} mismatches: {mismatches[:3]}"
        ),
        metadata={"n_files": len(expected), "mismatches": mismatches},
    )


def check_python_pins(repo: Path) -> Check:
    """``pyproject.toml`` must pin python + torch.

    Smoke test, not a full dependency audit. A real audit lives in
    SECURITY_AUDIT; this just catches the case where someone deletes the
    python/torch constraint block entirely.
    """
    p = repo / "pyproject.toml"
    if not p.is_file():
        return Check(name="python_pins", passed=False, detail="pyproject.toml missing")
    text = p.read_text(encoding="utf-8", errors="ignore")
    has_py = "python" in text.lower() and ("3.10" in text or "3.11" in text or "3.12" in text)
    has_torch = "torch" in text.lower()
    return Check(
        name="python_pins",
        passed=has_py and has_torch,
        detail="python + torch pinned" if (has_py and has_torch) else "missing python or torch pin",
    )


def check_git_clean(repo: Path) -> Check:
    """Informational working-tree status.

    Never fails — a dirty tree is expected during local development. CI
    uses this to surface "you forgot to commit a regenerated artefact"
    as a warning rather than a hard stop.
    """
    rc, stdout, _ = _run(["git", "status", "--porcelain"], cwd=repo)
    dirty = [ln for ln in stdout.splitlines() if ln.strip()]
    return Check(
        name="git_clean",
        passed=True,
        detail=("clean" if not dirty else f"{len(dirty)} tracked changes"),
        metadata={"dirty_files": dirty[:20], "is_clean": not dirty},
    )


def run_all(repo: Path, scratch: Path) -> Report:
    """Run every check. Cheap checks first so a failure surfaces fast;
    the subprocess-driven ones (RF regen, evaluate regen) run last so
    the local feedback loop stays snappy when iterating on the cheap
    ones."""
    rep = Report()
    rep.add(check_artefacts_exist(repo))
    rep.add(check_transformer_run_files(repo))
    rep.add(check_split_hashes_match(repo))
    rep.add(check_python_pins(repo))
    rep.add(check_git_clean(repo))
    rep.add(check_rf_predictions_regenerate(repo, scratch))
    rep.add(check_evaluate_regenerates(repo, scratch))
    return rep


def _build_parser() -> argparse.ArgumentParser:
    """CLI parser. Defaults align with the repo layout so the typical
    invocation is just ``python -m src.infra.repro``."""
    p = argparse.ArgumentParser(
        description=(
            "Reproducibility verification: regenerates every derivative "
            "artefact and diffs it against the committed copy."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--scratch", type=Path, default=Path("results/repro/_scratch"))
    p.add_argument("--report", type=Path, default=Path("results/repro/reproducibility_report.json"))
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Run every check, write the JSON report, print the table, return
    exit code (0 on all-pass, 1 on any failure)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args(argv)
    repo = Path(args.repo).resolve()
    scratch = Path(args.scratch).resolve()
    scratch.mkdir(parents=True, exist_ok=True)

    rep = run_all(repo, scratch)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(rep.as_dict(), indent=2, default=str))

    print()
    print("-- Reproducibility check results --")
    for c in rep.checks:
        flag = "PASS" if c.passed else "FAIL"
        print(f"  [{flag}] {c.name:32s} {c.detail}")
    print()
    if rep.all_passed:
        print("All reproducibility checks passed.")
        return 0
    failed = [c.name for c in rep.checks if not c.passed]
    print(f"FAILED: {failed}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
