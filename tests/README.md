# tests/

> **Breadcrumb**: [↑ repo root](../) > **tests/**

**Test suite** — mirrors `src/` one-for-one so every subpackage has a dedicated home for its tests. 320 tests total; the suite runs in ~20s on a laptop. Consumed by CI, by local pre-merge checks, and cited in Appendix 8 (Reproducibility) of the report.

The `data/` and `analysis/` subpackages have no dedicated tests yet — coverage is transitive via `tests/infra/` + `tests/scripts/`. Tests that need committed runs (under `results/transformer/seed_42/`, `results/baseline/rf/`) skip gracefully if the artefacts are not on disk; CI runs them, local dev does not require them.

## What's here

| Subfolder | Contents |
|---|---|
| [`data/`](data/) | Placeholder for `src/data/` unit tests (future). |
| [`tokenization/`](tokenization/) | Tokenizer + embedding. |
| [`models/`](models/) | Attention, Transformer, TabularTransformer, MTLM. |
| [`training/`](training/) | Losses, dataset, utils, train, train_mtlm. |
| [`baselines/`](baselines/) | RF refit + per-row predictions. |
| [`evaluation/`](evaluation/) | Calibration, fairness, uncertainty, significance, interpret, visualise, evaluate. |
| [`infra/`](infra/) | Reproducibility gate. |
| [`scripts/`](scripts/) | `run_all.py` orchestrator smoke tests. |
| [`conftest.py`](conftest.py) | Session fixtures: `repo_root`, `metadata`, `train_df`, `train_df_small`, `cat_vocab`, `small_dataset`, `seeded_rng`. |

## How it was produced

Hand-written pytest. Tests are deterministic (seed=42 where needed); the MTLM mask-ratio assertion is statistical and can flake on bad RNG — reroll with `pytest --randomly-seed=0`.

```bash
# Full suite:
MPLBACKEND=Agg PYTHONIOENCODING=utf-8 python -m pytest tests/ -q

# Single subpackage / file / test:
python -m pytest tests/models/ -q
python -m pytest tests/models/test_model.py::test_forward_shape -q

# Without coverage (faster):
python -m pytest tests/ -q --no-cov
```

Set `MPLBACKEND=Agg` on all platforms — `tests/evaluation/test_visualise.py` writes PNGs and would otherwise open a GUI window.

## How it's consumed

- CI (`.github/workflows/*`) runs the full suite.
- Coverage report printed below the ASCII summary.
- Report **Appendix 8** cites the 320-test tally as part of the reproducibility claim.

## How to regenerate

```bash
MPLBACKEND=Agg PYTHONIOENCODING=utf-8 python -m pytest tests/ -q
```

## Path idiom

Tests at the top level used `REPO = Path(__file__).resolve().parent.parent`. After the 2026 restructure the tests moved one level deeper, so files under `tests/X/` use `parent.parent.parent`. `conftest.py` stays at `tests/` root with the original two-level climb.

## Neighbours

- **↑ Parent**: [`../`](../) — repo root
- **↔ Siblings**: [`../src/`](../src/), [`../data/`](../data/), [`../results/`](../results/), [`../figures/`](../figures/), [`../scripts/`](../scripts/), [`../notebooks/`](../notebooks/), [`../docs/`](../docs/)
- **↓ Children**: [`data/`](data/), [`tokenization/`](tokenization/), [`models/`](models/), [`training/`](training/), [`baselines/`](baselines/), [`evaluation/`](evaluation/), [`infra/`](infra/), [`scripts/`](scripts/)
