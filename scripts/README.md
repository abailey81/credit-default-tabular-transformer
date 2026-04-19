# `scripts/` — End-to-end pipeline drivers

Thin driver scripts that sit on top of the `src/` subpackages. The
goal is: every piece of logic lives exactly once in `src/`; these
files are auditable orchestrators that shell out and write logs.

## What lives here

| File               | Purpose |
|---|---|
| `run_all.py`       | **Option A — one-command end-to-end.** Walks every pipeline stage in order: preprocessing → EDA → RF benchmark + per-row predictions → supervised transformer training (one run per seed) → MTLM pretrain + fine-tune → evaluation battery (comparison, figures, calibration, fairness, uncertainty, significance, interpret) → reproducibility check. Each stage runs as a subprocess via `sys.executable`; existing artefacts are auto-detected and expensive stages skipped unless `--force`. |
| `run_pipeline.py`  | **Option B — preprocessing-only shim.** Fine-grained control over data ingestion: API vs local fallback, single-stage flags (`--eda-only`, `--preprocess-only`, `--rf-benchmark`). Useful for debugging the data layer in isolation. |

## Option A — full run

```bash
poetry run python scripts/run_all.py
# Smoke-speed:
poetry run python scripts/run_all.py --n-samples 5 --n-resamples 200 --seeds 42
# Single stage:
poetry run python scripts/run_all.py --only rf
```

## Option B — preprocessing only

```bash
poetry run python scripts/run_pipeline.py --preprocess-only
poetry run python scripts/run_pipeline.py --source local
```

See the repo-root `README.md` Quick Start section for the full
matrix of flags.

## Log location

Every subprocess invocation is tee'd to both the terminal and a
stage-labelled log under `results/pipeline/logs/`. A timing summary
is printed at the end of `run_all.py`.

## Cross-reference

- `docs/ARCHITECTURE.md` — full pipeline graph (nodes = stages,
  edges = artefact dependencies). Read this before modifying stage
  ordering.
- `tests/scripts/test_run_all.py` — orchestrator smoke tests. Any new
  stage must come with a dispatch test here.
