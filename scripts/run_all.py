#!/usr/bin/env python3
"""
run_all.py -- One-command end-to-end orchestrator for the credit-default
tabular transformer project.

Walks every stage of the pipeline in the correct order: preprocessing,
exploratory data analysis, Random Forest benchmark + prediction regeneration,
supervised transformer training (one run per seed), MTLM self-supervised
pretraining + fine-tune, the full evaluation battery (comparison table,
figures, calibration, fairness, uncertainty, significance, interpret), and a
final reproducibility check.

Each stage is executed as a subprocess via ``sys.executable`` so the logic
lives exactly once in ``src/`` and this file is a thin, auditable driver.
Stdout/stderr for every stage is tee'd both to the terminal and to a stage-
labelled log under ``results/pipeline/logs/`` for post-hoc debugging. Existing
artefacts (``results/transformer/seed_*/best.pt``, ``results/mtlm/run_*/
encoder_pretrained.pt``) are auto-detected and the expensive training stages
are skipped unless ``--force`` is supplied. A timing summary table is printed
at the end.

Examples
--------
    # full end-to-end run with defaults
    poetry run python scripts/run_all.py

    # fast smoke (one seed, small MC-dropout + bootstrap budgets)
    poetry run python scripts/run_all.py --n-samples 5 --n-resamples 200 --seeds 42

    # single-stage debugging
    poetry run python scripts/run_all.py --only rf
    poetry run python scripts/run_all.py --only evaluate

This script is deliberately ASCII-only in source to stay robust under the
Windows cp1252 console default. At runtime we set PYTHONIOENCODING=utf-8 on
child processes so the stage modules can still emit their unicode logs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

# ---------------------------------------------------------------------------
# repo layout
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "results" / "pipeline" / "logs"
COMPARISON_TABLE = REPO_ROOT / "results" / "evaluation" / "comparison" / "comparison_table.md"

DEFAULT_SEEDS: List[int] = [42, 1, 2]
DEFAULT_MTLM_SEED: int = 42
DEFAULT_N_SAMPLES: int = 50
DEFAULT_N_RESAMPLES: int = 2000

STAGE_NAMES = (
    "data",
    "eda",
    "rf",
    "rf_pred",
    "train",
    "mtlm",
    "evaluate",
    "visualise",
    "calibration",
    "fairness",
    "uncertainty",
    "significance",
    "interpret",
    "repro",
)

# Only these are exposed via the public --only flag (coarse-grained).
ONLY_CHOICES = ("data", "eda", "rf", "train", "mtlm", "evaluate")

# ---------------------------------------------------------------------------
# ANSI status helpers (plain, no third-party deps)
# ---------------------------------------------------------------------------

_ANSI = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _wrap(code: str, msg: str) -> str:
    return f"\033[{code}m{msg}\033[0m" if _ANSI else msg


def _green(msg: str) -> str:
    return _wrap("32", msg)


def _red(msg: str) -> str:
    return _wrap("31", msg)


def _yellow(msg: str) -> str:
    return _wrap("33", msg)


def _cyan(msg: str) -> str:
    return _wrap("36", msg)


def _bold(msg: str) -> str:
    return _wrap("1", msg)


# ---------------------------------------------------------------------------
# stage result bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    name: str
    status: str  # "OK", "SKIP", "FAIL"
    duration_s: float
    note: str = ""


@dataclass
class RunState:
    stages: List[StageResult] = field(default_factory=list)

    def record(self, name: str, status: str, duration: float, note: str = "") -> None:
        self.stages.append(StageResult(name, status, duration, note))

    @property
    def failed(self) -> bool:
        return any(s.status == "FAIL" for s in self.stages)


# ---------------------------------------------------------------------------
# low-level subprocess runner
# ---------------------------------------------------------------------------


def _run_subprocess(stage: str, argv: Sequence[str], state: RunState) -> bool:
    """Run argv as a subprocess; tee output to terminal + a stage log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"run_all.{stage}.log"
    print(_cyan(f"[{stage}] $ ") + " ".join(str(a) for a in argv))
    t0 = time.perf_counter()

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("MPLBACKEND", "Agg")

    with log_path.open("w", encoding="utf-8", errors="replace") as log_f:
        proc = subprocess.Popen(
            list(argv),
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_f.write(line)
        rc = proc.wait()

    dur = time.perf_counter() - t0
    if rc == 0:
        state.record(stage, "OK", dur)
        print(_green(f"[{stage}] OK  ({dur:6.1f}s) -> {log_path.as_posix()}"))
        return True

    state.record(stage, "FAIL", dur, note=f"exit={rc}")
    print(_red(f"[{stage}] FAIL (exit={rc}, {dur:6.1f}s) -> {log_path.as_posix()}"))
    return False


def _skip(stage: str, state: RunState, note: str) -> None:
    state.record(stage, "SKIP", 0.0, note=note)
    print(_yellow(f"[{stage}] SKIP  ({note})"))


# ---------------------------------------------------------------------------
# stage dispatchers -- each returns True on success (or skip), False on failure
# ---------------------------------------------------------------------------


def _py_module(*args: str) -> List[str]:
    return [sys.executable, "-m", *args]


def stage_data(state: RunState) -> bool:
    return _run_subprocess(
        "data",
        [sys.executable, "scripts/run_pipeline.py", "--preprocess-only"],
        state,
    )


def stage_eda(state: RunState) -> bool:
    return _run_subprocess(
        "eda",
        [sys.executable, "scripts/run_pipeline.py", "--eda-only"],
        state,
    )


def stage_rf(state: RunState) -> bool:
    return _run_subprocess("rf", _py_module("src.baselines.random_forest"), state)


def stage_rf_pred(state: RunState) -> bool:
    return _run_subprocess("rf_pred", _py_module("src.baselines.rf_predictions"), state)


def stage_train(state: RunState, seeds: Sequence[int], force: bool) -> bool:
    all_ok = True
    for seed in seeds:
        out_dir = REPO_ROOT / "results" / "transformer" / f"seed_{seed}"
        stage = f"train_seed_{seed}"
        if not force and (out_dir / "best.pt").is_file():
            _skip(stage, state, note=f"{out_dir.relative_to(REPO_ROOT).as_posix()}/best.pt exists")
            continue
        ok = _run_subprocess(
            stage,
            _py_module(
                "src.training.train", "--seed", str(seed), "--output-dir", out_dir.as_posix()
            ),
            state,
        )
        all_ok = all_ok and ok
    return all_ok


def stage_mtlm(state: RunState, seed: int, force: bool) -> bool:
    pre_dir = REPO_ROOT / "results" / "mtlm" / f"run_{seed}"
    ft_dir = REPO_ROOT / "results" / "transformer" / f"seed_{seed}_mtlm_finetune"

    ok = True

    if not force and (pre_dir / "encoder_pretrained.pt").is_file():
        _skip(
            "mtlm_pretrain",
            state,
            note=f"{pre_dir.relative_to(REPO_ROOT).as_posix()}/encoder_pretrained.pt exists",
        )
    else:
        ok &= _run_subprocess(
            "mtlm_pretrain",
            _py_module(
                "src.training.train_mtlm", "--seed", str(seed), "--output-dir", pre_dir.as_posix()
            ),
            state,
        )

    if not ok:
        return False

    if not force and (ft_dir / "best.pt").is_file():
        _skip(
            "mtlm_finetune",
            state,
            note=f"{ft_dir.relative_to(REPO_ROOT).as_posix()}/best.pt exists",
        )
        return True

    return _run_subprocess(
        "mtlm_finetune",
        _py_module(
            "src.training.train",
            "--seed",
            str(seed),
            "--pretrained-encoder",
            (pre_dir / "encoder_pretrained.pt").as_posix(),
            "--output-dir",
            ft_dir.as_posix(),
        ),
        state,
    )


def stage_evaluate(state: RunState) -> bool:
    return _run_subprocess("evaluate", _py_module("src.evaluation.evaluate"), state)


def stage_visualise(state: RunState) -> bool:
    return _run_subprocess("visualise", _py_module("src.evaluation.visualise"), state)


def stage_calibration(state: RunState) -> bool:
    return _run_subprocess("calibration", _py_module("src.evaluation.calibration"), state)


def stage_fairness(state: RunState) -> bool:
    return _run_subprocess("fairness", _py_module("src.evaluation.fairness"), state)


def stage_uncertainty(state: RunState, n_samples: int) -> bool:
    return _run_subprocess(
        "uncertainty",
        _py_module("src.evaluation.uncertainty", "--n-samples", str(n_samples)),
        state,
    )


def stage_significance(state: RunState, n_resamples: int) -> bool:
    return _run_subprocess(
        "significance",
        _py_module("src.evaluation.significance", "--n-resamples", str(n_resamples)),
        state,
    )


def stage_interpret(state: RunState) -> bool:
    # interpret depends on test_attn_weights.npz -- skip gracefully if missing.
    run_dir = REPO_ROOT / "results" / "transformer" / "seed_42"
    if not (run_dir / "test_attn_weights.npz").is_file():
        _skip("interpret", state, note="test_attn_weights.npz absent")
        return True
    return _run_subprocess("interpret", _py_module("src.evaluation.interpret"), state)


def stage_repro(state: RunState) -> bool:
    return _run_subprocess("repro", _py_module("src.infra.repro"), state)


# ---------------------------------------------------------------------------
# summary helpers
# ---------------------------------------------------------------------------


def _print_summary(state: RunState) -> None:
    print()
    print(_bold("=" * 72))
    print(_bold(" run_all.py -- stage summary"))
    print(_bold("=" * 72))
    print(f"  {'stage':<24}  {'status':<6}  {'duration':>10}  note")
    print(f"  {'-' * 24}  {'-' * 6}  {'-' * 10}  {'-' * 24}")
    for s in state.stages:
        colour = _green if s.status == "OK" else (_yellow if s.status == "SKIP" else _red)
        print(f"  {s.name:<24}  {colour(s.status):<15}  {s.duration_s:>8.1f}s  {s.note}")
    total = sum(s.duration_s for s in state.stages)
    print(f"  {'-' * 24}  {'-' * 6}  {'-' * 10}")
    print(f"  {'TOTAL':<24}  {'':<6}  {total:>8.1f}s")
    print()


def _print_comparison_table() -> None:
    if not COMPARISON_TABLE.is_file():
        print(_yellow(f"[summary] {COMPARISON_TABLE.as_posix()} not found -- skipped"))
        return
    print(_bold("-- comparison_table.md --"))
    print(COMPARISON_TABLE.read_text(encoding="utf-8", errors="replace"))


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "End-to-end pipeline runner: preprocess -> EDA -> RF -> transformer "
            "training (per seed) -> MTLM pretrain + fine-tune -> evaluate -> "
            "visualise -> calibration/fairness/uncertainty/significance/"
            "interpret -> reproducibility check."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip supervised transformer training stages entirely.",
    )
    p.add_argument(
        "--skip-mtlm",
        action="store_true",
        help="Skip MTLM pretraining + fine-tune stages entirely.",
    )
    p.add_argument("--skip-eda", action="store_true", help="Skip the EDA stage.")
    p.add_argument(
        "--only",
        choices=ONLY_CHOICES,
        default=None,
        help=(
            "Run only one coarse stage group and nothing else. "
            "Useful for debugging a single step."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-run training/MTLM even if checkpoints already "
            "exist on disk. Default: auto-skip when present."
        ),
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="MC-dropout passes for uncertainty stage.",
    )
    p.add_argument(
        "--n-resamples",
        type=int,
        default=DEFAULT_N_RESAMPLES,
        help="Paired bootstrap resamples for significance stage.",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Seeds for supervised training runs.",
    )
    p.add_argument(
        "--mtlm-seed",
        type=int,
        default=DEFAULT_MTLM_SEED,
        help="Seed for MTLM pretrain + fine-tune.",
    )
    return p


# ---------------------------------------------------------------------------
# main orchestrator
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    state = RunState()

    if os.environ.get("PYTHONIOENCODING", "").lower() not in ("utf-8", "utf8"):
        print(
            _yellow(
                "[warn] PYTHONIOENCODING is not set to utf-8. Child processes "
                "may fail to emit unicode characters on Windows consoles. "
                "Recommended: set PYTHONIOENCODING=utf-8 before invoking."
            )
        )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(_bold(f"[run_all] repo={REPO_ROOT.as_posix()}"))
    print(_bold(f"[run_all] logs -> {LOG_DIR.as_posix()}"))
    t_total = time.perf_counter()

    only = args.only

    # 1) preprocessing
    if only in (None, "data") and not stage_data(state):
        return _finalise(state, t_total, fail=True)
    if only == "data":
        return _finalise(state, t_total)

    # 2) EDA
    if only in (None, "eda"):
        if args.skip_eda and only is None:
            _skip("eda", state, note="--skip-eda")
        elif not stage_eda(state):
            return _finalise(state, t_total, fail=True)
    if only == "eda":
        return _finalise(state, t_total)

    # 3) RF benchmark + 4) RF predictions
    if only in (None, "rf"):
        if not stage_rf(state):
            return _finalise(state, t_total, fail=True)
        if not stage_rf_pred(state):
            return _finalise(state, t_total, fail=True)
    if only == "rf":
        return _finalise(state, t_total)

    # 5) supervised training (per seed, with auto-skip)
    if only in (None, "train"):
        if args.skip_train and only is None:
            _skip("train", state, note="--skip-train")
        elif not stage_train(state, args.seeds, args.force):
            return _finalise(state, t_total, fail=True)
    if only == "train":
        return _finalise(state, t_total)

    # 6) MTLM pretrain + fine-tune
    if only in (None, "mtlm"):
        if args.skip_mtlm and only is None:
            _skip("mtlm", state, note="--skip-mtlm")
        elif not stage_mtlm(state, args.mtlm_seed, args.force):
            return _finalise(state, t_total, fail=True)
    if only == "mtlm":
        return _finalise(state, t_total)

    # 7)-13) evaluation battery
    if not stage_evaluate(state):
        return _finalise(state, t_total, fail=True)
    if only == "evaluate":
        return _finalise(state, t_total)

    if not stage_visualise(state):
        return _finalise(state, t_total, fail=True)
    if not stage_calibration(state):
        return _finalise(state, t_total, fail=True)
    if not stage_fairness(state):
        return _finalise(state, t_total, fail=True)
    if not stage_uncertainty(state, args.n_samples):
        return _finalise(state, t_total, fail=True)
    if not stage_significance(state, args.n_resamples):
        return _finalise(state, t_total, fail=True)
    if not stage_interpret(state):
        return _finalise(state, t_total, fail=True)

    # 14) reproducibility check -- HARD failure if this doesn't pass
    if not stage_repro(state):
        return _finalise(state, t_total, fail=True)

    return _finalise(state, t_total)


def _finalise(state: RunState, t0: float, fail: bool = False) -> int:
    total = time.perf_counter() - t0
    _print_summary(state)
    print(_bold(f"[run_all] wall-clock total: {total:6.1f}s"))
    _print_comparison_table()
    if fail or state.failed:
        print(_red("[run_all] one or more stages FAILED"))
        return 1
    print(_green("[run_all] all stages OK"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
