# results/evaluation/fairness/

> **Breadcrumb**: [↑ repo root](../../../) > [↑ results](../../) > [↑ evaluation](../) > **fairness/**

**Subgroup fairness audit artefacts** — Novelty **N10**. Consumed by Section 4 / "Fairness audit" of the report and by the Model Card's Intended-use / Limitations section.

Protected attributes: SEX, EDUCATION, MARRIAGE (original UCI encoding plus small regrouping described in [`src/data/preprocessing.py`](../../../src/data/preprocessing.py)). Two CSVs + a JSON summary provide both per-subgroup detail and headline disparity numbers for the report narrative.

## What's here

| File | Contents |
|---|---|
| [`subgroup_metrics.csv`](subgroup_metrics.csv) | Per-(attribute, subgroup) AUC / F1 / FPR / FNR / ECE / support count on the test set. |
| [`disparity_metrics.csv`](disparity_metrics.csv) | Max-min and ratio disparities per attribute for every metric (headline numbers). |
| [`fairness_summary.json`](fairness_summary.json) | Structured summary used by the plot backend and the model card. |

## How it was produced

[`src/evaluation/fairness.py`](../../../src/evaluation/fairness.py) — reads `results/transformer/seed_*/test_predictions.npz` and the raw test-split attributes from `data/processed/splits/test_raw.csv`. Deterministic — regenerates bit-stably via `python -m src.infra.repro`.

```bash
poetry run python -m src.evaluation.fairness
```

## How it's consumed

- Report **Section 4** / "Fairness audit".
- [`docs/MODEL_CARD.md`](../../../docs/MODEL_CARD.md) §Intended-use / Limitations.
- [`../../../figures/evaluation/fairness/`](../../../figures/evaluation/fairness/) — plots rendered from this data.

## How to regenerate

```bash
poetry run python -m src.evaluation.fairness
```

## Neighbours

- **↑ Parent**: [`../`](../) — evaluation/ index
- **↔ Siblings**: [`../comparison/`](../comparison/), [`../calibration/`](../calibration/), [`../uncertainty/`](../uncertainty/), [`../significance/`](../significance/)
- **↓ Children**: none
