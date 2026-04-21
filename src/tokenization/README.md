# src/tokenization/

> **Breadcrumb**: [↑ repo root](../../) > [↑ src](../) > **tokenization/**

**Tokenizer + feature embedding** — turns a pandas row into the 24-token sequence (`[CLS]` + 23 features) the transformer consumes. Owns the hybrid PAY tokenisation scheme (Novelty **N1**) and the MTLM `[MASK]` vocabulary (Novelty **N4**). Consumed by Section 3 (Model build-up) of the report.

`TOKEN_ORDER` is the only contract between `src.data.preprocessing` and everything downstream. If a new feature is added upstream, it must be inserted at the correct index here **and** in `feature_metadata.json` — otherwise `tests/tokenization/test_tokenizer.py` refuses to build the vocabulary. `CreditDefaultDataset` expects the scaled CSV layout: numerical columns must already be leak-free scaled before they hit the tokenizer.

## What's here

| File | Contents |
|---|---|
| [`tokenizer.py`](tokenizer.py) | `TOKEN_ORDER` (canonical 23-feature layout), `build_categorical_vocab`, `CreditDefaultDataset` (with `pay_raw` targets for MTLM / N5), `MTLMCollator` (random-mask sampler). |
| [`embedding.py`](embedding.py) | `FeatureEmbedding`: per-feature projections into `d_model`, prepended `[CLS]`, optional temporal positional encoding (Ablation A7), optional `[MASK]` token for MTLM. Exposes `build_temporal_layout`. |
| [`__init__.py`](__init__.py) | Package marker. |

## How it was produced

Hand-written. Not a standalone CLI; consumed by training + evaluation modules.

## How it's consumed

- [`../training/train.py`](../training/train.py), [`../training/train_mtlm.py`](../training/train_mtlm.py) — training loops.
- [`../models/`](../models/) — `FeatureEmbedding` is the input path for every model.
- [`../baselines/rf_predictions.py`](../baselines/rf_predictions.py) — feature-metadata consumer (categorical vocab).
- [`../evaluation/`](../evaluation/) — every evaluation module.
- Report **Section 3** — ships **N1** (hybrid PAY tokens) and part of **N4** (MTLM `[MASK]`). Appendix 8 cross-references Ablation A7.

## How to regenerate

Not regenerated directly — invoked by training / evaluation modules. Tests:

```bash
python -m pytest tests/tokenization/ -q
```

## Neighbours

- **↑ Parent**: [`../`](../) — src/ index
- **↔ Siblings**: [`../data/`](../data/), [`../analysis/`](../analysis/), [`../models/`](../models/), [`../training/`](../training/), [`../baselines/`](../baselines/), [`../evaluation/`](../evaluation/), [`../infra/`](../infra/)
- **↓ Children**: none
