# tests/models/

> **Breadcrumb**: [↑ repo root](../../) > [↑ tests](../) > **models/**

**Model tests** — covers [`src/models/`](../../src/models/): every custom attention op, the transformer block, the full `TabularTransformer`, and the MTLM head. Gated by Appendix 8 (Reproducibility) of the report.

Pure-unit tests build tiny synthetic batches inline; the end-to-end path in `test_model.py` uses `small_dataset` from `conftest.py`. `test_mtlm.py::test_state_dict_prefixes_align` is the canary for encoder-state renames — if it fails after a refactor, rename the prefix in `src/models/mtlm.py` to match rather than mutating the test.

## What's here

| File | Contents |
|---|---|
| [`test_attention.py`](test_attention.py) | `ScaledDotProductAttention` numerics, mask semantics, `MultiHeadAttention` head-split / concat, `attn_bias` hook used by N2 / N3. |
| [`test_transformer.py`](test_transformer.py) | `FeedForward` shapes, `TransformerBlock` PreNorm order, independent dropout channels (A11), `TemporalDecayBias` (N3) numerics, `TransformerEncoder` stacking. |
| [`test_model.py`](test_model.py) | `TabularTransformer` end-to-end forward, pool modes (`cls` / `mean`), `load_pretrained_encoder` strictness flag, aux-PAY0 head wiring (N5). |
| [`test_mtlm.py`](test_mtlm.py) | `MTLMHead` per-feature shapes, `mtlm_loss` numerics on synthetic batches, state-dict prefix compatibility with `TabularTransformer.load_pretrained_encoder`. |

## How it was produced

Hand-written pytest.

```bash
python -m pytest tests/models/ -q
# Single ablation switch:
python -m pytest tests/models/test_transformer.py::test_temporal_decay_bias_monotone -q
```

## How it's consumed

- CI runs this subpackage.
- Pinned by Report **Appendix 8** as part of the 320-test suite.

## How to regenerate

```bash
python -m pytest tests/models/ -q
```

## Neighbours

- **↑ Parent**: [`../`](../) — tests/ index
- **↔ Siblings**: [`../data/`](../data/), [`../tokenization/`](../tokenization/), [`../training/`](../training/), [`../baselines/`](../baselines/), [`../evaluation/`](../evaluation/), [`../infra/`](../infra/), [`../scripts/`](../scripts/)
- **↓ Children**: none
