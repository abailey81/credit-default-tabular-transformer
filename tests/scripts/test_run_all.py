"""scripts/run_all.py -- end-to-end orchestrator smoke tests.

These tests exercise the CLI plumbing of ``run_all.py`` without paying the
cost of any real stage: we stub subprocess.Popen so each stage dispatcher
sees a zero-exit fake child and the orchestrator walks its state machine in
milliseconds. The intent is to catch regressions in argument parsing, stage
ordering, and the --only / --skip-* / --force gating.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_all  # noqa: E402


def test_help_exits_zero(capsys):
    """--help must print usage and exit 0 (argparse SystemExit(0))."""
    with pytest.raises(SystemExit) as exc:
        run_all.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "End-to-end pipeline runner" in out
    # every advertised public flag must be visible in --help
    for flag in (
        "--skip-train",
        "--skip-mtlm",
        "--skip-eda",
        "--only",
        "--n-samples",
        "--n-resamples",
        "--seeds",
        "--mtlm-seed",
    ):
        assert flag in out, f"missing flag in help: {flag}"


class _FakePopen:
    """Minimal subprocess.Popen replacement for the tests."""

    last_argv: list = []
    invocations: list = []

    def __init__(self, argv, *args, **kwargs):
        _FakePopen.last_argv = list(argv)
        _FakePopen.invocations.append(list(argv))
        self.stdout = io.StringIO("stub stdout from FakePopen\n")
        self.returncode = 0

    def wait(self):
        return self.returncode


@pytest.fixture(autouse=True)
def _reset_fake():
    _FakePopen.last_argv = []
    _FakePopen.invocations = []
    yield


def test_only_data_runs_only_the_data_stage(monkeypatch, tmp_path):
    """With --only data, the runner should invoke exactly one subprocess."""
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    # Redirect the log dir into tmp so we don't pollute the repo
    monkeypatch.setattr(run_all, "LOG_DIR", tmp_path / "logs")

    rc = run_all.main(["--only", "data"])
    assert rc == 0
    assert len(_FakePopen.invocations) == 1, _FakePopen.invocations
    argv = _FakePopen.invocations[0]
    assert "run_pipeline.py" in " ".join(argv)
    assert "--preprocess-only" in argv


def test_only_rf_runs_rf_plus_rf_predictions(monkeypatch, tmp_path):
    """--only rf should run the RF benchmark AND the rf_predictions regen."""
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(run_all, "LOG_DIR", tmp_path / "logs")

    rc = run_all.main(["--only", "rf"])
    assert rc == 0
    mods = [" ".join(a) for a in _FakePopen.invocations]
    assert any("src.baselines.random_forest" in m for m in mods)
    assert any("src.baselines.rf_predictions" in m for m in mods)


def test_skip_flags_mark_stages_skipped(monkeypatch, tmp_path):
    """--skip-train and --skip-mtlm should produce SKIP records and skip
    the expensive stages while still running the downstream eval battery."""
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(run_all, "LOG_DIR", tmp_path / "logs")

    rc = run_all.main(
        [
            "--skip-eda",
            "--skip-train",
            "--skip-mtlm",
            "--n-samples",
            "2",
            "--n-resamples",
            "10",
        ]
    )
    assert rc == 0
    mods = [" ".join(a) for a in _FakePopen.invocations]
    # must NOT have launched train.py / train_mtlm.py
    assert not any(
        "src.training.train " in m or m.endswith("src.training.train") for m in mods
    ), mods
    assert not any("src.training.train_mtlm" in m for m in mods), mods
    # must have launched the evaluation battery
    for mod in (
        "src.evaluation.evaluate",
        "src.evaluation.visualise",
        "src.evaluation.calibration",
        "src.evaluation.fairness",
        "src.evaluation.uncertainty",
        "src.evaluation.significance",
        "src.infra.repro",
    ):
        assert any(mod in m for m in mods), f"expected {mod} in stages"
