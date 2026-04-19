# figures/

> **Breadcrumb**: [↑ repo root](../) > **figures/**

**Committed PNG artefacts** — every figure referenced by the report and the top-level `README.md`. Feeds Section 2 (EDA), Section 3 (Model/baseline), Section 4 (Experiments), and Appendix 8 (interpretability).

Every PNG here is regenerable from the CSV / JSON data in [`../results/`](../results/) via a single `python -m src.<module>` invocation. All figures are deterministic — regenerate bit-stably via `python -m src.infra.repro`. Subfolders mirror the report's narrative arc: EDA first, then baseline diagnostics, then the transformer evaluation battery.

## What's here

| Subfolder | Contents |
|---|---|
| [`eda/`](eda/) | 12 exploratory data figures (fig01-fig13, no fig12). |
| [`baseline/`](baseline/) | 5 Random Forest diagnostic plots. |
| [`evaluation/`](evaluation/) | Transformer evaluation figures across 6 subfolders: comparison / calibration / fairness / uncertainty / significance / interpret. |

## How it was produced

- `eda/` from [`src/analysis/eda.py`](../src/analysis/eda.py).
- `baseline/` from [`src/baselines/random_forest.py`](../src/baselines/random_forest.py).
- `evaluation/` from `src/evaluation/*.py` (see each subfolder README).

Deterministic across the suite; the MC-dropout and bootstrap figures under `evaluation/uncertainty/` and `evaluation/significance/` are stochastic, seeded — regenerate bit-stably under the committed seeds.

## How it's consumed

- Report: Section 2 (EDA), Section 3 (baseline), Section 4 (evaluation), Appendix 8 (interpretability).
- [`docs/MODEL_CARD.md`](../docs/MODEL_CARD.md) pulls calibration + fairness figures.
- Top-level [`README.md`](../README.md) embeds the training-curves PNG.

## How to regenerate

```bash
poetry run python -m src.infra.repro
```

Re-runs EDA, RF predictions, and the full evaluation suite and rewrites every PNG. All hashes must match the committed versions.

## Novelty cross-reference

- `evaluation/fairness/` → **N10** (subgroup fairness audit).
- `evaluation/uncertainty/` → **N11** (MC-dropout refusal curves).
- `evaluation/significance/` → **N12** (multi-seed significance).
- `evaluation/interpret/` → attention interpretability (Appendix, not on N1-N12).
- N2 feature-group bias + N3 temporal decay appear as annotations on training / evaluation plots, no standalone figures.

## Neighbours

- **↑ Parent**: [`../`](../) — repo root
- **↔ Siblings**: [`../data/`](../data/), [`../results/`](../results/), [`../src/`](../src/), [`../tests/`](../tests/), [`../scripts/`](../scripts/), [`../notebooks/`](../notebooks/), [`../docs/`](../docs/)
- **↓ Children**: [`eda/`](eda/), [`baseline/`](baseline/), [`evaluation/`](evaluation/)
