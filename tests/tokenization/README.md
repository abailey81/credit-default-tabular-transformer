# `tests/tokenization/` — Tokenizer + embedding tests

Covers `src/tokenization/` — the 24-token sequence contract and the
`FeatureEmbedding` module.

## What's covered

| File                 | Subject |
|---|---|
| `test_tokenizer.py`  | `TOKEN_ORDER` invariants, `build_categorical_vocab` round-trip, `CreditDefaultDataset` shape and dtype, `MTLMCollator` masking probability and vocabulary integrity. |
| `test_embedding.py`  | Per-feature projection shapes, `[CLS]` insertion, optional temporal positional encoding (Ablation A7), optional `[MASK]` token behaviour, `build_temporal_layout` helper. |

## Fixtures used

Both files use `metadata`, `cat_vocab`, `train_df_small`, and
`small_dataset` from the root `conftest.py`. Tests that do not need
real data build tiny synthetic `DataFrame`s inline.

## Running

```bash
python -m pytest tests/tokenization/ -q
```

## Gotchas

- The tokenizer invariants are strict: if you add a feature upstream
  in `src/data/preprocessing.py`, these tests will fail until
  `TOKEN_ORDER` is updated in lock-step.
- The MTLM mask ratio is checked statistically over a sample — a
  bad random seed can flake it rarely. Reroll with
  `pytest --randomly-seed=0`.
