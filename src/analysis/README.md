# src/analysis/

> **Breadcrumb**: [↑ repo root](../../) > [↑ src](../) > **analysis/**

**Exploratory Data Analysis** — produces publication-quality EDA figures and statistical tests that back Section 2 (Data Exploration) of the report. Reads only the committed CSVs under [`../../data/processed/splits/`](../../data/processed/splits/) — never touches raw data.

Pure data-science module: no model code, no tokenizer dependency. Uses `feature_metadata.json` (produced by `src.data.preprocessing`) to keep categorical labels aligned with tokenizer vocabularies. Writes 12 figures to [`../../figures/eda/`](../../figures/eda/) and the summary statistics table to [`../../results/analysis/`](../../results/analysis/).

## What's here

| File | Contents |
|---|---|
| [`eda.py`](eda.py) | Self-contained CLI. Emits `fig01..fig13` under `figures/eda/` (no fig12 — dropped in Phase 3) covering class balance, numerical / categorical distributions with KS / chi-square / Cramér-V tests, PAY-sequence heatmaps, correlation matrices. Also writes `results/analysis/summary_statistics.{csv,tex}`. |
| [`__init__.py`](__init__.py) | Package marker. |

## How it was produced

Hand-written module; deterministic given a seed — regenerates bit-stably via `python -m src.infra.repro`.

```bash
python -m src.analysis.eda --data-dir data/processed --fig-dir figures/eda
# Or via the Stage-2 driver:
python scripts/run_all.py --only eda
```

## How it's consumed

- Report **Section 2** (all 12 figures + Table 1).
- [`../../figures/eda/`](../../figures/eda/) — rendered PNGs.
- [`../../results/analysis/`](../../results/analysis/) — summary stats CSV + LaTeX.
- [`../../notebooks/01_exploratory_data_analysis.ipynb`](../../notebooks/01_exploratory_data_analysis.ipynb) mirrors this module cell-by-cell.

## How to regenerate

```bash
python -m src.analysis.eda
```

No dedicated unit tests (figures are visually inspected); smoke-level coverage via `tests/scripts/test_run_all.py` CLI dispatch.

## Neighbours

- **↑ Parent**: [`../`](../) — src/ index
- **↔ Siblings**: [`../data/`](../data/), [`../tokenization/`](../tokenization/), [`../models/`](../models/), [`../training/`](../training/), [`../baselines/`](../baselines/), [`../evaluation/`](../evaluation/), [`../infra/`](../infra/)
- **↓ Children**: none
