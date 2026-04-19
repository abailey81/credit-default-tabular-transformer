# `tests/` — Test suite

Mirrors `src/` one-for-one so every subpackage has a dedicated home
for its tests. 320 tests total; the suite runs in ~20s on a laptop
and produces a coverage report under the ASCII summary.

## Layout

```
tests/
├── conftest.py              # Session fixtures (repo_root, train_df, metadata, cat_vocab, ...)
├── tokenization/            # Tokenizer + embedding
├── models/                  # Attention, Transformer, TabularTransformer, MTLM
├── training/                # Losses, dataset, utils, train, train_mtlm
├── baselines/               # RF refit + per-row predictions
├── evaluation/              # Calibration, fairness, uncertainty, significance, interpret, visualise, evaluate
├── infra/                   # Reproducibility gate
└── scripts/                 # run_all.py orchestrator smoke tests
```

The `data/` and `analysis/` subpackages have no dedicated tests yet
(coverage is transitive via `tests/infra/` + `tests/scripts/`).

## How to run

```bash
# Full suite:
MPLBACKEND=Agg PYTHONIOENCODING=utf-8 python -m pytest tests/ -q

# A single subpackage:
python -m pytest tests/models/ -q
python -m pytest tests/evaluation/ -q

# A single file or test:
python -m pytest tests/evaluation/test_calibration.py -q
python -m pytest tests/models/test_model.py::test_forward_shape -q

# Without coverage (faster iteration):
python -m pytest tests/ -q --no-cov
```

Set `MPLBACKEND=Agg` on all platforms — `tests/evaluation/test_visualise.py`
writes PNGs and would otherwise open a GUI window on macOS / Windows.

## Fixtures (session-scoped, defined in `conftest.py`)

| Fixture          | Produces |
|---|---|
| `repo_root`      | `Path` to the repository root. |
| `metadata`       | Parsed `data/processed/feature_metadata.json`. Skips if missing. |
| `train_df`       | Parsed `data/processed/splits/train_scaled.csv`. Skips if missing. |
| `train_df_small` | First 128 rows of `train_df`. |
| `cat_vocab`      | Categorical vocabulary built from `metadata`. |
| `small_dataset`  | `CreditDefaultDataset` on `train_df_small`. |
| `seeded_rng`     | Per-test `torch.Generator` seeded to 42. |

Tests that need committed runs (`results/transformer/seed_42/`,
`results/baseline/rf/`, etc.) skip gracefully if the artefacts are not
on disk — CI runs them, local dev does not require them.

## Path idiom

Tests at the top level used `REPO = Path(__file__).resolve().parent.parent`.
After the 2026 restructure the tests moved one level deeper, so files
under `tests/X/` use `parent.parent.parent`. Every moved file was
patched accordingly; `conftest.py` stays at `tests/` root and keeps
the original two-level climb.
