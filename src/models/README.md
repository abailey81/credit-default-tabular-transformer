# `src/models/` — From-scratch Transformer + MTLM

The architectural core: every attention op, residual connection and
positional-bias mechanism is implemented from torch primitives (no
`nn.MultiheadAttention`, no `nn.TransformerEncoder`). Houses Novelties
N2, N3, N4, and the aux-PAY0 head (N5).

## Key modules

| Module           | Purpose |
|---|---|
| `attention.py`   | `ScaledDotProductAttention` and `MultiHeadAttention` with an `attn_bias` hook — additive pre-softmax bias used by N2 / N3. |
| `transformer.py` | `FeedForward`, `TransformerBlock` (PreNorm; independently ablatable attn / FFN / residual dropout), `TemporalDecayBias` (Novelty N3; ALiBi-style learnable temporal decay), and `TransformerEncoder`. |
| `model.py`       | `TabularTransformer` — tokenizer → embedding → encoder → pool → head. Exposes every ablation switch and a `load_pretrained_encoder` helper for the §8.5.5 MTLM two-stage fine-tune. |
| `mtlm.py`        | `MTLMHead` (per-feature decoders: 3 categorical + 6 PAY + 14 numerical), `mtlm_loss` (entropy-normalised CE + variance-normalised MSE), `MTLMModel` whose state-dict is drop-in for `load_pretrained_encoder`. |

## Non-obvious dependencies

All modules import `src.tokenization.embedding.FeatureEmbedding` for
the input path. `mtlm.py` reuses `TabularTransformer`'s encoder by
design — the prefixes in its `state_dict` must stay aligned so
checkpoints round-trip between pretraining and fine-tuning.

`TabularTransformer.aux_pay0=True` enables Novelty N5: an auxiliary
PAY_0 reconstruction head trained jointly with the main classifier.
The head reads `pay_raw` targets built in `src.tokenization.tokenizer`.

## Invocation

Not a standalone CLI; invoked by `src.training.train` and
`src.training.train_mtlm`. For interactive exploration, see
`notebooks/04_train_transformer.ipynb`.

## Tests

- `tests/models/test_attention.py`   — mask semantics, attn_bias hook,
  head-splitting numerics.
- `tests/models/test_transformer.py` — PreNorm wiring, dropout
  independence (A11), `TemporalDecayBias` numerical behaviour (N3).
- `tests/models/test_model.py`       — end-to-end forward, pool modes,
  `load_pretrained_encoder` strictness flag, aux-PAY0 head.
- `tests/models/test_mtlm.py`        — `MTLMHead` shapes, `mtlm_loss`
  on synthetic batches, state-dict prefix compatibility.

## Report section

- Section 6.4-6.7 (Attention, Block, Encoder, full model).
- Section 8.5.5 (Two-stage MTLM fine-tune) cross-references
  `load_pretrained_encoder`.
- Appendix (Ablation matrix) — A1..A16 switches all live here.
