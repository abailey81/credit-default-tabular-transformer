# results/evaluation/comparison/

> **Breadcrumb**: [↑ repo root](../../../) > [↑ results](../../) > [↑ evaluation](../) > **comparison/**

**Head-to-head comparison table** — every transformer run vs the RF baseline. Consumed by Section 4 / "Head-to-head evaluation" of the report and by the top-level `README.md` performance callouts.

This is the canonical numeric summary for the "does the transformer beat RF?" question. The Markdown version is a drop-in for the report; the JSON is consumed programmatically; the TXT is a human-readable one-liner used in README and CHANGELOG. Aggregate across the three supervised seeds + MTLM fine-tune as `mean ± std`.

## What's here

| File | Contents |
|---|---|
| [`comparison_table.csv`](comparison_table.csv) | One row per (run, metric): auc_roc / auc_pr / f1 / accuracy / precision / recall / specificity / cohen_kappa / ece / brier, with `mean ± std` aggregates. |
| [`comparison_table.md`](comparison_table.md) | Same table as GitHub-flavoured Markdown, drop-in for the report. |
| [`evaluate_summary.json`](evaluate_summary.json) | Structured: per-run metrics, aggregates, winner-per-metric record. |
| [`head_to_head_summary.txt`](head_to_head_summary.txt) | Plain-text narrative ("Transformer beats RF on AUC by +X ...") used in README / CHANGELOG. |

## How it was produced

[`src/evaluation/evaluate.py`](../../../src/evaluation/evaluate.py) — loads every `results/transformer/seed_*/test_predictions.npz` plus `results/baseline/rf/test_predictions.npz`. Deterministic — regenerates bit-stably via `python -m src.infra.repro`.

```bash
poetry run python -m src.evaluation.evaluate
```

## How it's consumed

- Report **Section 4** (table 2 + narrative).
- Top-level [`README.md`](../../../README.md) pulls numbers from the `.md` version.
- [`../../../figures/evaluation/comparison/`](../../../figures/evaluation/comparison/) — visual counterpart.

## How to regenerate

```bash
poetry run python -m src.evaluation.evaluate
```

## Neighbours

- **↑ Parent**: [`../`](../) — evaluation/ index
- **↔ Siblings**: [`../calibration/`](../calibration/), [`../fairness/`](../fairness/), [`../uncertainty/`](../uncertainty/), [`../significance/`](../significance/)
- **↓ Children**: none
