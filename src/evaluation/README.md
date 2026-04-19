# src/evaluation/

> **Breadcrumb**: [↑ repo root](../../) > [↑ src](../) > **evaluation/**

**Metrics, calibration, fairness, significance, interpretability, visualisation** — consumes the per-seed transformer runs under [`../../results/transformer/`](../../results/transformer/) and the RF under [`../../results/baseline/rf/`](../../results/baseline/rf/) to produce every number and figure in Section 4 (Experiments) of the report plus Appendix 8 audit artefacts.

All seven modules read committed artefacts — never raw data or raw model code. `calibration.py` is consumed by `fairness.py` and `uncertainty.py` (calibrated probabilities feed their decision-threshold analyses). `evaluate.py` is the central orchestrator: every `train.py` invocation is followed by this module.

## What's here

| File | Contents |
|---|---|
| [`evaluate.py`](evaluate.py) | Aggregates per-seed metrics into the transformer-vs-RF comparison table (ensemble of 4 seeds). Picks up RF from `rf_predictions.py`. |
| [`visualise.py`](visualise.py) | ROC / PR / confusion matrices / training curves / reliability diagrams. |
| [`calibration.py`](calibration.py) | Post-hoc calibration (temperature, Platt, isotonic) + ECE / MCE / Brier decomposition. |
| [`fairness.py`](fairness.py) | Subgroup audit across SEX / EDUCATION / MARRIAGE: AUC / TPR / FPR / selection-rate disparity (**N10**). |
| [`uncertainty.py`](uncertainty.py) | MC-dropout predictive entropy + BALD mutual information + refuse curve (**N11**). |
| [`significance.py`](significance.py) | Paired tests: McNemar, DeLong, bootstrap, BH-FDR correction, Hanley-McNeil power (**N12**). |
| [`interpret.py`](interpret.py) | Attention rollout + per-feature importance vs RF Gini. |
| [`__init__.py`](__init__.py) | Package marker. |

## How it was produced

Hand-written. Deterministic where seeded; stochastic-and-seeded for `uncertainty.py` (MC-dropout) and `significance.py` (bootstrap) — regenerate bit-stably under committed seeds via `python -m src.infra.repro`.

```bash
python -m src.evaluation.evaluate
python -m src.evaluation.visualise
python -m src.evaluation.calibration
python -m src.evaluation.fairness
python -m src.evaluation.uncertainty
python -m src.evaluation.significance
python -m src.evaluation.interpret
```

## How it's consumed

- [`../../results/evaluation/`](../../results/evaluation/) — numeric outputs.
- [`../../figures/evaluation/`](../../figures/evaluation/) — rendered plots.
- [`docs/MODEL_CARD.md`](../../docs/MODEL_CARD.md) pulls calibration + fairness + uncertainty numbers.
- Report **Section 4** (all subsections) + **Appendix 8** (interpretability, audits).

## How to regenerate

```bash
python -m src.infra.repro
```

## Neighbours

- **↑ Parent**: [`../`](../) — src/ index
- **↔ Siblings**: [`../data/`](../data/), [`../analysis/`](../analysis/), [`../tokenization/`](../tokenization/), [`../models/`](../models/), [`../training/`](../training/), [`../baselines/`](../baselines/), [`../infra/`](../infra/)
- **↓ Children**: none
