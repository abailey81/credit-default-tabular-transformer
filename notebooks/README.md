# `notebooks/` — Exploration notebooks

Four Jupyter notebooks for reviewers who want to step through the
pipeline cell-by-cell rather than reading the `src/` modules or
running `scripts/run_all.py`.

## Notebooks

| Notebook                                      | What it does | Report section |
|---|---|---|
| `01_exploratory_data_analysis.ipynb`          | Walks the Section 4 EDA figures inline: class balance, KS / chi-square tests, PAY-sequence heatmaps, correlation matrices. Mirrors `src/analysis/eda.py`. | Section 4 |
| `02_data_preprocessing.ipynb`                 | Demonstrates the 22-feature engineering, stratified 70/15/15 split, and leak-free scaling produced by `src/data/preprocessing.py`. | Section 3 |
| `03_random_forest_benchmark.ipynb`            | Reproduces the 200-iteration randomised-search RF benchmark from `src/baselines/random_forest.py` and explores the tuned feature importance. | Section 9 |
| `04_train_transformer.ipynb`                  | Walks through the tokenizer → embedding → encoder → head path end-to-end, then launches a short training run via `src/training/train.py`. | Section 6 + 8 |

## Setup

```bash
poetry install
poetry run jupyter lab notebooks/
# Or, if VS Code / Cursor:
poetry run python -m ipykernel install --user --name credit-default-tabular-transformer
# then pick the kernel from the notebook UI.
```

The notebooks assume the preprocessing CSVs already exist under
`data/processed/splits/`. Run `python scripts/run_pipeline.py
--preprocess-only` once before opening them, or the data-loading cells
will fail.

## Notes

These notebooks are for **exploration and reviewer walkthrough** only.
The production, reproducible pipeline is `scripts/run_all.py`;
nothing in `notebooks/` feeds downstream artefacts. If you catch
yourself copy-pasting notebook code into `src/`, stop and add a
proper module + test instead.

## Cross-reference

- `src/analysis/eda.py`               — programmatic version of `01_*`.
- `src/data/preprocessing.py`         — programmatic version of `02_*`.
- `src/baselines/random_forest.py`    — programmatic version of `03_*`.
- `src/training/train.py`             — programmatic version of `04_*`.
