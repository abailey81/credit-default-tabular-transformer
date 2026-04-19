# `src/baselines/` — Random Forest benchmark

Implements the Plan §9 Random Forest benchmark that the transformer is
compared against in Section 10. Keeps training and per-row prediction
emission in separate modules so significance / calibration / fairness
consumers can refit without rerunning the 200-iteration search.

## Key modules

| Module              | Purpose |
|---|---|
| `random_forest.py`  | Hyperparameter-tuned RF benchmark (Plan §9): 200-iter randomised search across a 7-dimensional grid, class-balanced, stratified 5-fold inner CV. Persists `rf_config.json` + cross-validation diagnostics under `results/baseline/rf/`. |
| `rf_predictions.py` | Refits the tuned RF from `rf_config.json` and emits deterministic per-row probabilities + metrics for the calibration, significance, and fairness consumers. |

## Non-obvious dependencies

Reads the engineered CSV (`train_engineered.csv`) from
`data/processed/splits/`, not the scaled CSV — the transformer and RF
consume different feature views deliberately (tree models do not need
scaled features). Writes `rf_config.json` that `rf_predictions.py`
refits from, so the two modules are tightly coupled by the JSON
schema.

## Invocation

```bash
# Tune + fit (slow; ~10 min on a laptop):
python -m src.baselines.random_forest --output-dir results/baseline/rf

# Refit from a saved config + emit per-row probabilities (fast):
python -m src.baselines.rf_predictions \
    --config results/baseline/rf/rf_config.json \
    --output-dir results/baseline/rf
```

## Tests

- `tests/baselines/test_rf_predictions.py` — JSON round-trip,
  deterministic refit, per-row shape + bound invariants. A slower
  committed-data e2e test is gated on `rf_config.json` being present.

## Report section

- Section 9 (Random Forest baseline) — every hyperparameter table and
  feature-importance figure references `random_forest.py`.
- Section 10 (Comparison) consumes per-row predictions from
  `rf_predictions.py`.
- Appendix (Reproducibility) — `rf_predictions_regenerate` check in
  `src.infra.repro` pins `max|Δp| < 1e-6` against the committed copy.
