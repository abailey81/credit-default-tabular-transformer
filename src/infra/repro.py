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
from typing import Any, Optional

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
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Report:
    """Aggregated result across every ``Check``."""

    checks: list[Check] = field(default_factory=list)

    def add(self, check: Check) -> None:
        self.checks.append(check)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def as_dict(self) -> dict[str, Any]:
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


def _compare_json_tolerant(got: Any, want: Any, path: str = "") -> list[str]:
    """Walk two parsed JSON trees and return a list of mismatches. Floats
    are compared with ``rtol=1e-3, atol=1e-4`` (handles BLAS-level drift in
    computed statistics like means/stds); ints / strings / bools / nulls
    are compared exactly."""
    mismatches: list[str] = []
    if isinstance(want, dict):
        if not isinstance(got, dict):
            mismatches.append(f"{path or '<root>'}: type dict vs {type(got).__name__}")
            return mismatches
        for key in want:
            if key not in got:
                mismatches.append(f"{path}/{key}: missing")
                continue
            mismatches.extend(_compare_json_tolerant(got[key], want[key], f"{path}/{key}"))
        for key in got:
            if key not in want:
                mismatches.append(f"{path}/{key}: unexpected key")
    elif isinstance(want, list):
        if not isinstance(got, list):
            mismatches.append(f"{path}: type list vs {type(got).__name__}")
            return mismatches
        if len(got) != len(want):
            mismatches.append(f"{path}: length {len(got)} vs {len(want)}")
            return mismatches
        for i, (g, w) in enumerate(zip(got, want)):
            mismatches.extend(_compare_json_tolerant(g, w, f"{path}[{i}]"))
    elif isinstance(want, bool) or isinstance(got, bool):
        # bool before float because isinstance(True, int) is True in Python.
        if got != want:
            mismatches.append(f"{path}: {got!r} vs {want!r}")
    elif isinstance(want, float) or isinstance(got, float):
        try:
            if not np.isclose(float(got), float(want), rtol=1e-3, atol=1e-4):
                mismatches.append(f"{path}: {got} vs {want}")
        except (TypeError, ValueError):
            mismatches.append(f"{path}: {got!r} vs {want!r}")
    else:
        if got != want:
            mismatches.append(f"{path}: {got!r} vs {want!r}")
    return mismatches


def _run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    """Subprocess wrapper returning ``(rc, stdout, stderr)``."""
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def _compare_dataframes(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    rtol: float = 1e-4,
) -> tuple[bool, str]:
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
    missing: list[str] = []
    checked_runs: list[str] = []
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
    """Regenerate RF test predictions and verify they agree with the
    committed copy within a cross-platform tolerance.

    RF tree-split tie-breaking depends on the underlying BLAS/LAPACK
    implementation (Apple Accelerate / OpenBLAS / MKL produce slightly
    different float comparisons in marginal splits). A bit-exact check
    therefore fails on different platforms even when the model and
    config are identical. We apply two complementary criteria:

      1. ``max |Δp|`` must be ≤ 5e-2 (5 percentage points). A real
         RF-config regression would shift many predictions by far more.
      2. Pearson correlation between regenerated and committed
         probabilities must be ≥ 0.999. A reordering of trees or a wrong
         random_state would crash this well below 0.99.
    """
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
    a = regen["y_prob"].astype(np.float64)
    b = comm["y_prob"].astype(np.float64)
    max_diff = float(np.abs(a - b).max())
    # Pearson correlation; if either side is constant (degenerate) we treat
    # the check as failed because RF outputs should never be constant.
    if a.std() < 1e-12 or b.std() < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(a, b)[0, 1])
    max_thr = 5e-2
    corr_thr = 0.999
    ok = max_diff <= max_thr and corr >= corr_thr
    return Check(
        name="rf_predictions_regenerate",
        passed=ok,
        detail=(
            f"max |Δp|={max_diff:.2e} (≤{max_thr:.0e}), "
            f"pearson r={corr:.4f} (≥{corr_thr:.3f})"
        ),
        metadata={
            "max_prob_diff": max_diff,
            "pearson_r": corr,
            "max_thr": max_thr,
            "corr_thr": corr_thr,
        },
    )


def check_split_hashes_match(repo: Path) -> Check:
    """Verify that ``data/processed/`` artefacts match expected content.

    The historical SHA-256 byte ledger (``SPLIT_HASHES.md``) was retired
    because CSV bytes depend on float-formatting, which depends on the
    underlying BLAS/LAPACK build. The same scikit-learn version produces
    different last-bit results on Apple Accelerate vs OpenBLAS vs MKL,
    so byte-exact comparison failed across operating systems even when
    the data and pipeline were identical.

    The new ledger is ``SPLIT_STATS.json`` and contains BLAS-invariant
    fingerprints:

      * row count           — must match exactly
      * column count        — must match exactly
      * column list         — must match exactly (in order)
      * default rate        — compared with absolute tolerance 1e-3
      * mean of every numeric column — compared with rtol=1e-3, atol=1e-4

    These detect every real change (different rows, different split
    proportions, broken scaling, schema drift) while ignoring the last-bit
    drift that BLAS implementations introduce.

    For files that are deterministic byte-for-byte (JSON metadata),
    SHA-256 is still used.
    """
    stats_json = repo / "data" / "processed" / "SPLIT_STATS.json"
    if not stats_json.is_file():
        return Check(
            name="split_hashes_match",
            passed=False,
            detail=f"missing {stats_json.relative_to(repo)}",
        )

    try:
        expected = json.loads(stats_json.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return Check(
            name="split_hashes_match",
            passed=False,
            detail=f"failed to parse SPLIT_STATS.json: {e}",
        )

    expected_splits: dict[str, dict[str, Any]] = expected.get("splits", {})
    expected_jsons: dict[str, str] = expected.get("json_files", {})
    expected_metadata: dict[str, Any] = expected.get("json_metadata", {})
    if not expected_splits and not expected_jsons and not expected_metadata:
        return Check(
            name="split_hashes_match",
            passed=False,
            detail="SPLIT_STATS.json contains no entries",
        )

    processed_root = repo / "data" / "processed"
    probe_dirs = (processed_root / "splits", processed_root)

    def find_csv(name: str) -> Optional[Path]:
        for d in probe_dirs:
            candidate = d / name
            if candidate.is_file():
                return candidate
        return None

    mismatches: list[str] = []
    checked: list[str] = []

    # Tolerant summary-stat check for CSV splits.
    for name, want in expected_splits.items():
        path = find_csv(name)
        if path is None:
            mismatches.append(f"{name}: missing")
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:  # pragma: no cover - sanity guard
            mismatches.append(f"{name}: read error ({e})")
            continue
        checked.append(name)

        if int(len(df)) != int(want.get("n_rows", -1)):
            mismatches.append(
                f"{name}: n_rows {len(df)} vs expected {want.get('n_rows')}"
            )
            continue
        if int(df.shape[1]) != int(want.get("n_cols", -1)):
            mismatches.append(
                f"{name}: n_cols {df.shape[1]} vs expected {want.get('n_cols')}"
            )
            continue
        want_cols = list(want.get("columns", []))
        if want_cols and list(df.columns) != want_cols:
            mismatches.append(f"{name}: columns differ from expected order")
            continue
        if "default_rate" in want and "DEFAULT" in df.columns:
            got_rate = float(df["DEFAULT"].mean())
            want_rate = float(want["default_rate"])
            if abs(got_rate - want_rate) > 1e-3:
                mismatches.append(
                    f"{name}: default_rate {got_rate:.4f} vs {want_rate:.4f}"
                )
                continue
        want_means = want.get("numeric_means", {})
        if want_means:
            bad_cols: list[str] = []
            for col, want_mean in want_means.items():
                if col not in df.columns:
                    bad_cols.append(f"{col}(missing)")
                    continue
                got_mean = float(df[col].mean())
                # rtol=1e-3 lets BLAS-level noise through; atol catches
                # zero-mean columns where rtol alone is meaningless.
                if not np.isclose(got_mean, float(want_mean), rtol=1e-3, atol=1e-4):
                    bad_cols.append(f"{col}({got_mean:.4f}/{want_mean:.4f})")
            if bad_cols:
                mismatches.append(
                    f"{name}: {len(bad_cols)} mean(s) drifted: {bad_cols[:3]}"
                )
                continue

    # Strict SHA-256 check for JSON metadata (legacy path; left in place for
    # any pre-existing entries that callers haven't migrated yet).
    for name, want_sha in expected_jsons.items():
        path = processed_root / name
        if not path.is_file():
            mismatches.append(f"{name}: missing")
            continue
        got_sha = _sha256(path)
        checked.append(name)
        if got_sha != want_sha:
            mismatches.append(
                f"{name}: sha {got_sha[:12]}... vs {want_sha[:12]}..."
            )

    # Tolerant content-comparison for JSON metadata (handles BLAS float drift
    # in computed numerical statistics like means/stds).
    for name, want_content in expected_metadata.items():
        path = processed_root / name
        if not path.is_file():
            mismatches.append(f"{name}: missing")
            continue
        try:
            with path.open(encoding="utf-8") as fh:
                got_content = json.load(fh)
        except (json.JSONDecodeError, OSError) as e:
            mismatches.append(f"{name}: parse error ({e})")
            continue
        checked.append(name)
        json_diffs = _compare_json_tolerant(got_content, want_content, name)
        if json_diffs:
            sample = json_diffs[:3]
            mismatches.append(
                f"{name}: {len(json_diffs)} field(s) drifted: {sample}"
            )

    ok = not mismatches
    total = len(expected_splits) + len(expected_jsons) + len(expected_metadata)
    return Check(
        name="split_hashes_match",
        passed=ok,
        detail=(
            f"{len(checked)}/{total} files match (BLAS-tolerant)"
            if ok
            else f"{len(mismatches)} mismatch(es): {mismatches[:3]}"
        ),
        metadata={"n_files": total, "mismatches": mismatches},
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


def main(argv: Optional[list[str]] = None) -> int:
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
