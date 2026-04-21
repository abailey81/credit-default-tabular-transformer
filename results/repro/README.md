# results/repro/

> **Breadcrumb**: [↑ repo root](../../) > [↑ results](../) > **repro/**

**Reproducibility harness output** — the machine-readable "does `python -m src.infra.repro` still produce bit-stable artefacts?" report. Consumed by CI (`.github/workflows/*`), local pre-merge checks, and cited in CHANGELOG per-phase entries.

Headline: **7/7 PASS** required before any PR merges. The `_scratch/` subfolder holds side-by-side regenerated copies of the canonical RF and evaluate outputs, kept only to diff against the committed versions; overwritten each repro run. Consumed by Appendix 8 (Reproducibility) of the report — every bullet in `docs/REPRODUCIBILITY.md` maps to a `Check` name in this report.

## What's here

| File / Subfolder | Contents |
|---|---|
| [`reproducibility_report.json`](reproducibility_report.json) | Per-stage PASS/FAIL + hash comparisons across the 7 checks: EDA, RF retune, RF predictions, transformer evaluate, calibration, fairness, uncertainty, significance, interpret, visualise. |
| [`_scratch/`](_scratch/) | Side-by-side regenerated copies: `_scratch/rf/{test_metrics.json,test_predictions.npz}` and `_scratch/eval/{comparison_table.csv,comparison_table.md,evaluate_summary.json}`. Overwritten each repro run. |

## How it was produced

[`src/infra/repro.py`](../../src/infra/repro.py) — the umbrella harness that re-invokes every `src/<module>` CLI and hashes outputs. Deterministic — regenerates bit-stably by construction.

```bash
poetry run python -m src.infra.repro
```

## How it's consumed

- CI (`.github/workflows/*`) asserts 7/7 PASS.
- CHANGELOG entries cite the pass status per phase.
- [`docs/REPRODUCIBILITY.md`](../../docs/REPRODUCIBILITY.md) is the human-readable companion.

## How to regenerate

```bash
poetry run python -m src.infra.repro
```

If any stage flips to FAIL, diff against `_scratch/` to find the drifting artefact.

## Neighbours

- **↑ Parent**: [`../`](../) — results/ index
- **↔ Siblings**: [`../analysis/`](../analysis/), [`../baseline/`](../baseline/), [`../transformer/`](../transformer/), [`../mtlm/`](../mtlm/), [`../evaluation/`](../evaluation/), [`../pipeline/`](../pipeline/)
- **↓ Children**: none (only `_scratch/`, an ephemeral diff buffer with no README)
