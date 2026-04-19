# figures/evaluation/interpret/

Attention-based interpretability figures — Appendix (interpretability).
Not yet on the N1–N12 novelty register, but referenced from the Conclusion.

## Files

| File | Shows |
|---|---|
| `attention_rollout.png` | Attention rollout (Abnar & Zuidema 2020) aggregated across layers, averaged over the test set. |
| `cls_feature_importance.png` | CLS-token attention magnitude per input feature (analogue of feature importance). |
| `attention_per_head.png` | Per-head attention matrices for the last encoder layer. |
| `defaulter_vs_nondefaulter_attention.png` | Class-conditional mean attention maps (defaulter − non-defaulter). |
| `feature_importance_comparison.png` | CLS-attention importance vs RF Gini importance (shared feature axis). |

## Produced by

[`src/evaluation/interpret.py`](../../../src/evaluation/interpret.py) —
reads `results/transformer/seed_*/test_attn_weights.npz`
(**gitignored**, ~76 MB per seed) emitted during training / evaluate.

If `test_attn_weights.npz` is not present, run
`python -m src.evaluation.evaluate --save-attn` first (or let
`python -m src.infra.repro` do it).

## Consumed by

- Report **Appendix** / "Interpretability".
- Discussion-of-N2 paragraph cross-references `feature_importance_comparison.png`.

## Regenerate

```bash
poetry run python -m src.evaluation.interpret
```

Deterministic — regenerates bit-stably via `python -m src.infra.repro`.
