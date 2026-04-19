# notebooks/

> **Breadcrumb**: [↑ repo root](../) > **notebooks/**

**Exploration notebooks** — four Jupyter notebooks for reviewers who want to step through the pipeline cell-by-cell rather than reading `src/` or running `scripts/run_all.py`. Mirror the report's Section 2 (Data Exploration), Section 3 (Model), and Section 4 (Experiments) narratives.

These notebooks are for **exploration and reviewer walkthrough** only. The production, reproducible pipeline is [`../scripts/run_all.py`](../scripts/run_all.py); nothing in `notebooks/` feeds downstream artefacts. If you catch yourself copy-pasting notebook code into `src/`, stop and add a proper module + test instead.

## What's here

| File | Contents |
|---|---|
| [`01_exploratory_data_analysis.ipynb`](01_exploratory_data_analysis.ipynb) | Walks Section 2 EDA figures inline: class balance, KS / chi-square tests, PAY-sequence heatmaps, correlation matrices. Mirrors [`src/analysis/eda.py`](../src/analysis/eda.py). |
| [`02_data_preprocessing.ipynb`](02_data_preprocessing.ipynb) | Demonstrates the 22-feature engineering, stratified 70/15/15 split, and leak-free scaling produced by [`src/data/preprocessing.py`](../src/data/preprocessing.py). |
| [`03_random_forest_benchmark.ipynb`](03_random_forest_benchmark.ipynb) | Reproduces the 200-iteration randomised-search RF benchmark from [`src/baselines/random_forest.py`](../src/baselines/random_forest.py) and explores the tuned feature importance. |
| [`04_train_transformer.ipynb`](04_train_transformer.ipynb) | Walks the tokenizer → embedding → encoder → head path end-to-end, then launches a short training run via [`src/training/train.py`](../src/training/train.py). |

## How it was produced

Hand-authored Jupyter. The notebooks assume the preprocessing CSVs already exist under [`../data/processed/splits/`](../data/processed/splits/). Run `python scripts/run_pipeline.py --preprocess-only` once before opening them, or data-loading cells will fail.

```bash
poetry install
poetry run jupyter lab notebooks/
# Or for VS Code / Cursor:
poetry run python -m ipykernel install --user --name credit-default-tabular-transformer
```

## How it's consumed

- Reviewers stepping through the pipeline cell-by-cell.
- Report **Section 2** (notebook 01) + **Section 3** (notebooks 02, 04) + implicit support for the RF benchmark description (notebook 03).

## How to regenerate

Notebooks are committed as-is; the scripted equivalents live under `src/`:

- [`src/analysis/eda.py`](../src/analysis/eda.py) — programmatic version of `01_*`.
- [`src/data/preprocessing.py`](../src/data/preprocessing.py) — programmatic version of `02_*`.
- [`src/baselines/random_forest.py`](../src/baselines/random_forest.py) — programmatic version of `03_*`.
- [`src/training/train.py`](../src/training/train.py) — programmatic version of `04_*`.

## Neighbours

- **↑ Parent**: [`../`](../) — repo root
- **↔ Siblings**: [`../src/`](../src/), [`../tests/`](../tests/), [`../data/`](../data/), [`../results/`](../results/), [`../figures/`](../figures/), [`../scripts/`](../scripts/), [`../docs/`](../docs/)
- **↓ Children**: none
