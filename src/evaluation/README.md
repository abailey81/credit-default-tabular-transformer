# `src/evaluation/` — Metrics, calibration, fairness, significance, interpretability, visualisation

Consumes the per-seed transformer runs under `results/transformer/`
and the RF runs under `results/baseline/rf/` to produce every number
and figure in Sections 4 and 10 of the report, plus the Appendix audit
artefacts.

## Key modules

| Module            | Purpose |
|---|---|
| `evaluate.py`     | Aggregates per-seed metrics into a transformer-vs-RF comparison table (ensemble of 4 seeds). Picks up RF from `rf_predictions.py`. |
| `visualise.py`    | §4 report figures: ROC / PR / confusion matrices / training curves / reliability diagrams. Deterministic; all seeds fixed. |
| `calibration.py`  | Post-hoc calibration (temperature, Platt, isotonic) + ECE / MCE / Brier decomposition + reliability-bin edges. |
| `fairness.py`     | Subgroup fairness audit across SEX / EDUCATION / MARRIAGE: AUC / TPR / FPR / selection rate disparity table. |
| `uncertainty.py`  | MC-dropout predictive entropy + BALD mutual information + refuse curve. |
| `significance.py` | Paired tests: McNemar (hard labels), DeLong (AUC), bootstrap (any metric), BH-FDR correction, Hanley-McNeil power. |
| `interpret.py`    | Attention rollout + per-feature importance vs RF Gini. Writes importance CSV + rollout heatmaps. |

## Non-obvious dependencies

All seven modules read committed artefacts under `results/` — never
raw data or raw model code. `calibration.py` is consumed by
`fairness.py` and `uncertainty.py` (calibrated probabilities feed
their decision-threshold analyses). `evaluate.py` is the central
orchestrator: pipeline stage "Section 10" runs only this module after
every `train.py` invocation.

## Invocation

```bash
python -m src.evaluation.evaluate     --runs results/transformer --rf-dir results/baseline/rf
python -m src.evaluation.visualise    --runs results/transformer --rf-dir results/baseline/rf --figures-dir figures
python -m src.evaluation.calibration  --runs results/transformer
python -m src.evaluation.fairness     --runs results/transformer/seed_42 --data-dir data/processed
python -m src.evaluation.uncertainty  --run  results/transformer/seed_42 --n-samples 30
python -m src.evaluation.significance --runs results/transformer --rf-dir results/baseline/rf
python -m src.evaluation.interpret    --run  results/transformer/seed_42
```

## Tests

Each module has a dedicated file under `tests/evaluation/` —
`test_calibration.py`, `test_fairness.py`, `test_uncertainty.py`,
`test_significance.py`, `test_interpret.py`, `test_visualise.py`, and
`test_evaluate_ensemble.py`. Every test has a synthetic unit path and
an end-to-end path that is gated on committed artefacts being present.

## Report section

- Section 4 (EDA) — `visualise.py`.
- Section 10 (Comparison) — `evaluate.py`, `calibration.py`,
  `significance.py`.
- Appendix (Fairness / Uncertainty / Interpretability) — the remaining
  four modules.
