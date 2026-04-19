# `src/analysis/` — Exploratory Data Analysis

Produces the publication-quality EDA figures and statistical tests that
back Section 4 of the report. Reads only the committed
`data/processed/splits/train_*.csv` CSVs — never touches raw data.

## Key modules

| Module   | Purpose |
|---|---|
| `eda.py` | Emits 12 EDA figures under `figures/eda/`: class balance, numerical / categorical distributions with KS / chi-square / Cramér-V tests, PAY-sequence heatmaps, and correlation matrices. Self-contained CLI; all figures are deterministic given a seed. |

## Non-obvious dependencies

Uses `feature_metadata.json` produced by `src.data.preprocessing` to
keep categorical labels aligned with the tokenizer vocabularies. It
does NOT depend on `src.tokenization` or any model code — this is a
pure data-science module.

## Invocation

```bash
python -m src.analysis.eda --data-dir data/processed --fig-dir figures/eda
```

Also runnable via Stage 2 of the end-to-end pipeline:

```bash
python scripts/run_all.py --only eda
```

## Tests

No dedicated unit tests yet (figures are visually inspected during
review); smoke-level coverage comes from the `scripts/run_all.py` test
in `tests/scripts/test_run_all.py` which exercises the CLI dispatch.

## Report section

- Section 4 (Exploratory Data Analysis) — every figure in that section
  is produced by `eda.py`.
- The notebook `notebooks/01_exploratory_data_analysis.ipynb` mirrors
  this module for reviewers who want to step through the analysis
  cell-by-cell.
