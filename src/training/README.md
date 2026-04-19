# src/training/

> **Breadcrumb**: [↑ repo root](../../) > [↑ src](../) > **training/**

**Training loops + loss functions** — owns the supervised loop (Phase 6/7/8), the MTLM pretraining loop (Phase 6A, **N4**), the batch sampler, and the three custom losses from Plan §7.2. Consumed by Section 3 (Model build-up) and Section 4 (Experiments) of the report.

Both training CLIs build their models via `src.models.model.TabularTransformer` and `src.models.mtlm.MTLMModel`, and their datasets via `src.tokenization.tokenizer.CreditDefaultDataset`. `utils.py`'s checkpoint helpers enforce `torch.load(weights_only=True)` by default — do not relax that without touching the security audit. Both CLIs have a `--smoke-test` flag that runs the loop end-to-end in seconds for test coverage.

## What's here

| File | Contents |
|---|---|
| [`losses.py`](losses.py) | `WeightedBCELoss`, `FocalLoss` (Ablation A11), `LabelSmoothingBCELoss`. All accept `reduction={'mean','sum','none'}`. |
| [`dataset.py`](dataset.py) | `StratifiedBatchSampler` (class-balanced per-batch) + `make_loader` factory (supervised or MTLM mode). |
| [`utils.py`](utils.py) | Determinism protocol (seed + deterministic algorithms), device selection, hardened weights-only checkpoint save/load, `EarlyStopping`, parameter accounting, UTF-8-safe logging. |
| [`train.py`](train.py) | Supervised training: AdamW + cosine warmup + gradient clipping + early stopping + optional two-stage encoder-LR fine-tune + optional aux-PAY0 head (N5). Emits per-run `config.json`, `train_log.csv`, `test_metrics.json`, `test_predictions.npz`, `test_attn_weights.npz`. |
| [`train_mtlm.py`](train_mtlm.py) | Phase 6A MTLM pretraining (**N4**). Emits `encoder_pretrained.pt` which `train.py` picks up via `--from-mtlm`. |
| [`__init__.py`](__init__.py) | Package marker. |

## How it was produced

Hand-written. Deterministic under fixed seeds — regenerates bit-stably via `python -m src.infra.repro`.

```bash
# MTLM pretrain (Phase 6A):
python -m src.training.train_mtlm --seed 42 --run-name run_42

# Supervised train with optional pretrained encoder:
python -m src.training.train --seed 42 \
    --from-mtlm results/mtlm/run_42/encoder_pretrained.pt
```

## How it's consumed

- [`../../results/transformer/`](../../results/transformer/) — per-seed directories written by `train.py`.
- [`../../results/mtlm/`](../../results/mtlm/) — MTLM encoder written by `train_mtlm.py`.
- [`../evaluation/`](../evaluation/) — consumes the per-run checkpoints and predictions.
- Report **Section 3** (Loss functions, training protocol, two-stage MTLM fine-tune). Appendix 8 ablations A11-A13 flip switches here.

## How to regenerate

```bash
python -m src.training.train --seed 42
```

## Neighbours

- **↑ Parent**: [`../`](../) — src/ index
- **↔ Siblings**: [`../data/`](../data/), [`../analysis/`](../analysis/), [`../tokenization/`](../tokenization/), [`../models/`](../models/), [`../baselines/`](../baselines/), [`../evaluation/`](../evaluation/), [`../infra/`](../infra/)
- **↓ Children**: none
