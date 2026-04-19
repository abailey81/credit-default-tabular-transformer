# tests/training/

> **Breadcrumb**: [↑ repo root](../../) > [↑ tests](../) > **training/**

**Training loop + losses + utils tests** — covers [`src/training/`](../../src/training/): losses, sampler, determinism, checkpoint helpers, and the two training loops (`train.py`, `train_mtlm.py`). Gated by Appendix 8 (Reproducibility) of the report.

E2E smoke tests shell out to `torch.save` / `torch.load`; they write to the pytest `tmp_path` fixture so nothing lands in the repo. `test_train.py::test_compute_classification_metrics_handles_single_class_gracefully` warns from sklearn — the warning is expected.

## What's here

| File | Contents |
|---|---|
| [`test_losses.py`](test_losses.py) | `WeightedBCELoss` / `FocalLoss` / `LabelSmoothingBCELoss` numerics, reduction modes, gradient sanity. |
| [`test_dataset.py`](test_dataset.py) | `StratifiedBatchSampler` class-balance invariant, `make_loader` shapes for supervised and MTLM modes. |
| [`test_utils.py`](test_utils.py) | Determinism protocol, device selection, checkpoint weights-only round-trip, `EarlyStopping` state machine, parameter accounting. |
| [`test_train.py`](test_train.py) | Cosine warmup LR schedule, loss factory dispatch, metric computation (AUC / PR / ECE), `evaluate_on_loader` payload, optimiser group construction, e2e smoke. |
| [`test_train_mtlm.py`](test_train_mtlm.py) | argparse, `MTLMModel` construction, e2e smoke pretrain that drops an `encoder_pretrained.pt` and confirms `TabularTransformer.load_pretrained_encoder` consumes it. |

## How it was produced

Hand-written pytest. Uses `metadata`, `train_df_small`, `cat_vocab`, and `small_dataset` fixtures.

```bash
python -m pytest tests/training/ -q
# Only the fast unit parts (skip e2e smoke):
python -m pytest tests/training/ -q -k 'not end_to_end'
```

## How it's consumed

- CI runs this subpackage.
- Pinned by Report **Appendix 8** as part of the 320-test suite.

## How to regenerate

```bash
python -m pytest tests/training/ -q
```

## Neighbours

- **↑ Parent**: [`../`](../) — tests/ index
- **↔ Siblings**: [`../data/`](../data/), [`../tokenization/`](../tokenization/), [`../models/`](../models/), [`../baselines/`](../baselines/), [`../evaluation/`](../evaluation/), [`../infra/`](../infra/), [`../scripts/`](../scripts/)
- **↓ Children**: none
