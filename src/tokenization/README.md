# `src/tokenization/` — Tokenizer + Feature embedding

Turns a pandas row into the 24-token sequence (`[CLS]` + 23 features)
that the transformer consumes. Owns the hybrid PAY tokenisation scheme
(Novelty N1) and the MTLM `[MASK]` vocabulary (Novelty N4).

## Key modules

| Module         | Purpose |
|---|---|
| `tokenizer.py` | Defines `TOKEN_ORDER` (the canonical 23-feature layout), `build_categorical_vocab`, `CreditDefaultDataset` (with `pay_raw` targets for MTLM / N5), and `MTLMCollator` (random-mask sampler). |
| `embedding.py` | `FeatureEmbedding`: per-feature projections into `d_model`, prepended `[CLS]`, optional temporal positional encoding (Ablation A7), optional `[MASK]` token used by MTLM. Also exposes `build_temporal_layout` for drift-safe sequence layout. |

## Non-obvious dependencies

`TOKEN_ORDER` is the only contract between `src.data.preprocessing` and
everything downstream. If a new feature is added upstream, it must be
inserted here at the correct index *and* in the `feature_metadata.json`
schema — otherwise `tests/tokenization/test_tokenizer.py` will refuse
to build the vocabulary.

`CreditDefaultDataset` expects the scaled CSV layout from
`data/processed/splits/`. Numerical columns must already be leak-free
scaled before they hit the tokenizer.

## Invocation

Not a standalone CLI; consumed by `src.training.train`,
`src.training.train_mtlm`, `src.baselines.rf_predictions`, and every
evaluation module.

## Tests

- `tests/tokenization/test_tokenizer.py`  — vocabulary construction,
  TOKEN_ORDER invariants, MTLM masking ratios.
- `tests/tokenization/test_embedding.py`  — per-feature projection
  shapes, [CLS] insertion, temporal positional encoding, mask token
  behaviour.

## Report section

- Section 6.1-6.3 (Input representation, tokenizer, embedding) —
  ships the architectural novelties N1 (hybrid PAY tokens) and part of
  N4 (MTLM [MASK]). Appendix cross-references the ablation matrix for
  A7 (temporal positional encoding).
