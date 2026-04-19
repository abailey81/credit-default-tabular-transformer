# src/models/

> **Breadcrumb**: [↑ repo root](../../) > [↑ src](../) > **models/**

**From-scratch Transformer + MTLM** — every attention op, residual connection, and positional-bias mechanism is implemented from torch primitives. No `nn.MultiheadAttention`, no `nn.TransformerEncoder`. Houses Novelties **N2**, **N3**, **N4**, and the aux-PAY0 head (**N5**). Consumed by Section 3 (Model build-up) of the report.

All modules import `src.tokenization.embedding.FeatureEmbedding` for the input path. `mtlm.py` reuses `TabularTransformer`'s encoder by design — state-dict prefixes must stay aligned so checkpoints round-trip between pretraining and fine-tuning. `TabularTransformer.aux_pay0=True` enables N5 (auxiliary PAY_0 reconstruction head trained jointly with the main classifier).

## What's here

| File | Contents |
|---|---|
| [`attention.py`](attention.py) | `ScaledDotProductAttention` and `MultiHeadAttention` with an `attn_bias` hook — additive pre-softmax bias used by N2 / N3. |
| [`transformer.py`](transformer.py) | `FeedForward`, `TransformerBlock` (PreNorm; independent attn / FFN / residual dropout), `TemporalDecayBias` (**N3**; ALiBi-style learnable temporal decay), `TransformerEncoder`. |
| [`model.py`](model.py) | `TabularTransformer` — tokenizer → embedding → encoder → pool → head. Every ablation switch + `load_pretrained_encoder` helper for the MTLM two-stage fine-tune. |
| [`mtlm.py`](mtlm.py) | `MTLMHead` (per-feature decoders: 3 categorical + 6 PAY + 14 numerical), `mtlm_loss` (entropy-normalised CE + variance-normalised MSE), `MTLMModel`. |
| [`__init__.py`](__init__.py) | Package marker. |

## How it was produced

Hand-written torch modules. Not a standalone CLI; invoked by training loops. For interactive exploration, see [`../../notebooks/04_train_transformer.ipynb`](../../notebooks/04_train_transformer.ipynb).

## How it's consumed

- [`../training/train.py`](../training/train.py), [`../training/train_mtlm.py`](../training/train_mtlm.py) — construct and train these modules.
- [`../evaluation/`](../evaluation/) — loads saved checkpoints for inference.
- Report **Section 3** (Attention, Block, Encoder, full model) + "Two-stage MTLM fine-tune" subsection.
- Appendix 8 — Ablation matrix A1..A16 all flip switches here.

## How to regenerate

Not regenerated directly. Tests:

```bash
python -m pytest tests/models/ -q
```

## Neighbours

- **↑ Parent**: [`../`](../) — src/ index
- **↔ Siblings**: [`../data/`](../data/), [`../analysis/`](../analysis/), [`../tokenization/`](../tokenization/), [`../training/`](../training/), [`../baselines/`](../baselines/), [`../evaluation/`](../evaluation/), [`../infra/`](../infra/)
- **↓ Children**: none
