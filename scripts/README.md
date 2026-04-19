# scripts/

> **Breadcrumb**: [↑ repo root](../) > **scripts/**

**End-to-end pipeline drivers** — thin driver scripts on top of the `src/` subpackages. Every piece of logic lives exactly once in `src/`; these files are auditable orchestrators that shell out and write logs. Consumed by Appendix 8 (Reproducibility) of the report and referenced from the top-level `README.md` Quick Start.

Two drivers, two flavours: `run_all.py` is Option A (one-command end-to-end), `run_pipeline.py` is Option B (preprocessing-only shim for fine-grained control over the data layer). Every subprocess invocation is tee'd to both the terminal and a stage-labelled log under [`../results/pipeline/logs/`](../results/pipeline/logs/).

## What's here

| File | Contents |
|---|---|
| [`run_all.py`](run_all.py) | **Option A — one-command end-to-end.** Walks every pipeline stage in order: preprocessing → EDA → RF benchmark + per-row predictions → supervised transformer training (one run per seed) → MTLM pretrain + fine-tune → evaluation battery (comparison, figures, calibration, fairness, uncertainty, significance, interpret) → reproducibility check. Each stage runs as a subprocess via `sys.executable`; existing artefacts auto-detected and expensive stages skipped unless `--force`. |
| [`run_pipeline.py`](run_pipeline.py) | **Option B — preprocessing-only shim.** Fine-grained control over data ingestion: API vs local fallback, single-stage flags (`--eda-only`, `--preprocess-only`, `--rf-benchmark`). Useful for debugging the data layer in isolation. |

## How it was produced

Hand-written Python drivers. Both shell out via `sys.executable` with deterministic stage ordering; the only non-deterministic thing about their output is the log-file timestamps.

```bash
# Option A — full run:
poetry run python scripts/run_all.py
poetry run python scripts/run_all.py --n-samples 5 --n-resamples 200 --seeds 42   # smoke
poetry run python scripts/run_all.py --only rf

# Option B — preprocessing-only:
poetry run python scripts/run_pipeline.py --preprocess-only
poetry run python scripts/run_pipeline.py --source local
```

## How it's consumed

- Developers running the full pipeline locally or in CI.
- [`../tests/scripts/test_run_all.py`](../tests/scripts/test_run_all.py) — orchestrator smoke tests. Any new stage must come with a dispatch test.
- [`../results/pipeline/logs/`](../results/pipeline/logs/) — per-stage log output.
- Report **Appendix 8** and the top-level `README.md` Quick Start.

## How to regenerate

```bash
poetry run python scripts/run_all.py
```

See [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) for the full pipeline graph.

## Neighbours

- **↑ Parent**: [`../`](../) — repo root
- **↔ Siblings**: [`../src/`](../src/), [`../tests/`](../tests/), [`../data/`](../data/), [`../results/`](../results/), [`../figures/`](../figures/), [`../notebooks/`](../notebooks/), [`../docs/`](../docs/)
- **↓ Children**: none
