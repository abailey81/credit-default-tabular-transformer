# figures/evaluation/interpret/

> **Breadcrumb**: [↑ repo root](../../../) > [↑ figures](../../) > [↑ evaluation](../) > **interpret/**

**Attention-based interpretability figures** — consumed by Appendix 8 / "Interpretability" of the report and referenced from the Conclusion (Section 5). Not on the N1-N12 novelty register but contributes to the discussion of N2 (feature-group bias).

Attention rollout (Abnar & Zuidema 2020) aggregates attention across layers to produce a single per-sample importance score. The RF-vs-attention comparison plot is the load-bearing visual for the "does attention recover tree-based signal?" discussion. Numeric summary JSON lives at [`../../../results/evaluation/interpret.json`](../../../results/evaluation/interpret.json) (the one loose file in `results/evaluation/` — see that folder's README for why).

## What's here

| File | Contents |
|---|---|
| [`attention_rollout.png`](attention_rollout.png) | Attention rollout aggregated across layers, averaged over the test set. |
| [`cls_feature_importance.png`](cls_feature_importance.png) | CLS-token attention magnitude per input feature (feature-importance analogue). |
| [`attention_per_head.png`](attention_per_head.png) | Per-head attention matrices for the last encoder layer. |
| [`defaulter_vs_nondefaulter_attention.png`](defaulter_vs_nondefaulter_attention.png) | Class-conditional mean attention maps (defaulter - non-defaulter). |
| [`feature_importance_comparison.png`](feature_importance_comparison.png) | CLS-attention importance vs RF Gini importance (shared feature axis). |

## How it was produced

[`src/evaluation/interpret.py`](../../../src/evaluation/interpret.py) — reads `results/transformer/seed_*/test_attn_weights.npz` (**gitignored**, ~76 MB per seed) emitted during training / evaluate. Deterministic — regenerates bit-stably via `python -m src.infra.repro`.

```bash
poetry run python -m src.evaluation.interpret
```

If `test_attn_weights.npz` is missing, run `python -m src.evaluation.evaluate --save-attn` first (or let `python -m src.infra.repro` handle it).

## How it's consumed

- Report **Appendix 8** / "Interpretability".
- Discussion of N2 cross-references `feature_importance_comparison.png`.
- [`../../../results/evaluation/interpret.json`](../../../results/evaluation/interpret.json) — numeric summary (CLS importances, rollout stats, RF-vs-attn correlation).

## How to regenerate

```bash
poetry run python -m src.evaluation.interpret
```

## Neighbours

- **↑ Parent**: [`../`](../) — evaluation/ index
- **↔ Siblings**: [`../comparison/`](../comparison/), [`../calibration/`](../calibration/), [`../fairness/`](../fairness/), [`../uncertainty/`](../uncertainty/), [`../significance/`](../significance/)
- **↓ Children**: none
