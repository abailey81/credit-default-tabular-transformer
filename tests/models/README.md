# `tests/models/` — Model tests

Covers `src/models/` — every custom attention op, transformer block,
the full `TabularTransformer`, and the MTLM head.

## What's covered

| File                 | Subject |
|---|---|
| `test_attention.py`  | `ScaledDotProductAttention` numerics, mask semantics, `MultiHeadAttention` head-split / concat, the `attn_bias` hook used by Novelties N2 / N3. |
| `test_transformer.py`| `FeedForward` shapes, `TransformerBlock` PreNorm order, independent dropout channels (A11), `TemporalDecayBias` (Novelty N3) numerics, `TransformerEncoder` stacking. |
| `test_model.py`      | `TabularTransformer` end-to-end forward, pool modes (`cls` / `mean`), `load_pretrained_encoder` strictness flag, aux-PAY0 head wiring (Novelty N5). |
| `test_mtlm.py`       | `MTLMHead` per-feature shapes, `mtlm_loss` numerics on synthetic batches, state-dict prefix compatibility with `TabularTransformer.load_pretrained_encoder`. |

## Fixtures used

Pure-unit tests build tiny synthetic batches inline. The end-to-end
path in `test_model.py` uses `small_dataset` from `conftest.py`.

## Running

```bash
python -m pytest tests/models/ -q
# Single ablation switch:
python -m pytest tests/models/test_transformer.py::test_temporal_decay_bias_monotone -q
```

## Gotchas

- `test_mtlm.py::test_state_dict_prefixes_align` is the canary for any
  rename in the encoder state. If it fails after a refactor, rename
  the prefix in `src/models/mtlm.py` to match rather than mutating
  the test.
