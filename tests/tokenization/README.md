# tests/tokenization/

> **Breadcrumb**: [↑ repo root](../../) > [↑ tests](../) > **tokenization/**

**Tokenizer + embedding tests** — covers [`src/tokenization/`](../../src/tokenization/) (the 24-token sequence contract and the `FeatureEmbedding` module). Gated by Appendix 8 (Reproducibility) of the report.

Both files use `metadata`, `cat_vocab`, `train_df_small`, and `small_dataset` fixtures from the root `conftest.py`. Tests that do not need real data build tiny synthetic `DataFrame`s inline. The tokenizer invariants are strict: if you add a feature upstream in `src/data/preprocessing.py`, these tests fail until `TOKEN_ORDER` is updated in lock-step.

## What's here

| File | Contents |
|---|---|
| [`test_tokenizer.py`](test_tokenizer.py) | `TOKEN_ORDER` invariants, `build_categorical_vocab` round-trip, `CreditDefaultDataset` shape and dtype, `MTLMCollator` masking probability and vocabulary integrity. |
| [`test_embedding.py`](test_embedding.py) | Per-feature projection shapes, `[CLS]` insertion, optional temporal positional encoding (Ablation A7), optional `[MASK]` token behaviour, `build_temporal_layout`. |

## How it was produced

Hand-written pytest. The MTLM mask ratio is checked statistically over a sample — a bad random seed can flake rarely. Reroll with `pytest --randomly-seed=0`.

```bash
python -m pytest tests/tokenization/ -q
```

## How it's consumed

- CI runs this subpackage.
- Pinned by Report **Appendix 8** as part of the 320-test suite.

## How to regenerate

```bash
python -m pytest tests/tokenization/ -q
```

## Neighbours

- **↑ Parent**: [`../`](../) — tests/ index
- **↔ Siblings**: [`../data/`](../data/), [`../models/`](../models/), [`../training/`](../training/), [`../baselines/`](../baselines/), [`../evaluation/`](../evaluation/), [`../infra/`](../infra/), [`../scripts/`](../scripts/)
- **↓ Children**: none
