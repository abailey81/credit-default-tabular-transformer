# `src/training/` — Training loops + loss functions

Owns the supervised loop (Phase 6/7/8), the MTLM pretraining loop
(Phase 6A), the batch sampler, and the three custom loss functions
defined in Plan §7.2.

## Key modules

| Module            | Purpose |
|---|---|
| `losses.py`       | `WeightedBCELoss`, `FocalLoss` (Plan §7.2 / Ablation A11), `LabelSmoothingBCELoss`. All accept `reduction={'mean','sum','none'}` and broadcast over a batch dim. |
| `dataset.py`      | `StratifiedBatchSampler` (class-balanced per-batch sampling) + `make_loader` factory that switches between supervised and MTLM modes. |
| `utils.py`        | Determinism protocol (seed + deterministic algorithms), device selection, hardened weights-only checkpoint save/load, `EarlyStopping`, parameter accounting, UTF-8-safe logging setup. |
| `train.py`        | Plan §8 supervised training: AdamW + cosine warmup + gradient clipping + early stopping + optional two-stage encoder-LR fine-tune + optional aux-PAY0 multi-task head (N5). Emits `config.json`, `train_log.csv`, `test_metrics.json`, `test_predictions.npz`, `test_attn_weights.npz` per run. |
| `train_mtlm.py`   | Phase 6A MTLM pretraining (Novelty N4). Emits `encoder_pretrained.pt` which `train.py` picks up via `--pretrained-encoder`. |

## Non-obvious dependencies

`train.py` and `train_mtlm.py` both build their models via
`src.models.model.TabularTransformer` and `src.models.mtlm.MTLMModel`,
and their datasets via `src.tokenization.tokenizer.CreditDefaultDataset`.
`utils.py`'s checkpoint helpers enforce `torch.load(weights_only=True)`
by default — do not relax that without touching the security audit.

## Invocation

```bash
# MTLM pretrain (Phase 6A):
python -m src.training.train_mtlm --seed 0 --output-dir results/mtlm_pretrain

# Supervised train (Phase 6/7/8) with optional pretrained encoder:
python -m src.training.train \
    --seed 42 --output-dir results/transformer/seed_42 \
    --pretrained-encoder results/mtlm_pretrain/encoder_pretrained.pt
```

Both CLIs have a `--smoke-test` flag that runs the loop end-to-end in
seconds for test coverage.

## Tests

- `tests/training/test_losses.py`     — loss numerics on synthetic
  edge cases.
- `tests/training/test_dataset.py`    — sampler class balance, loader
  shapes, MTLM collation.
- `tests/training/test_utils.py`      — determinism, checkpoint
  round-trip, early-stopping state machine.
- `tests/training/test_train.py`      — LR schedule, loss factory,
  metrics, optimiser construction, e2e smoke.
- `tests/training/test_train_mtlm.py` — argparse, model construction,
  e2e smoke pretrain.

## Report section

- Section 7 (Loss functions).
- Section 8 (Training protocol).
- Section 8.5.5 specifically covers the two-stage MTLM fine-tune.
- Appendix (Ablation matrix) A11 / A12 / A13 all flip switches here.
