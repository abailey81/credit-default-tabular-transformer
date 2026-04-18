# Model Card — `TabularTransformer` (Credit-Default Prediction)

Following Mitchell et al. (2019) "Model Cards for Model Reporting". This
card covers the small from-scratch transformer trained for the UCL MSc
Finance & AI coursework on the UCI Credit Card Default dataset. It is a
coursework artefact, not a production model.

---

## 1. Model details

| Item | Value |
|---|---|
| Model type | Small encoder-only transformer (FT-Transformer family) |
| Layers / heads / d_model / d_ff | 2 / 4 / 32 / 128 |
| Parameter count | **~28,600** (trainable) |
| Inputs | 23 tabular features + `[CLS]` token = 24-token sequence |
| Aggregation | `[CLS]` pooling (ablation A5 sweep available) |
| Output | Single logit; `sigmoid` gives P(default) |
| Novel inductive biases | N2 feature-group attention bias; N3 temporal-decay attention bias |
| Pretraining objective (optional) | N4 — masked tabular language modelling (MTLM) |
| Framework | PyTorch 2.2, Python 3.10–3.12 |
| License | Academic use (coursework); MIT on the repo |
| Developers | Tamer Atesyakar (integration, PR merges, hardening), abailey81 (EDA, preprocessing, plan), FardeenIdrus (attention / transformer / eval), Idaliia19 (tokenizer / dataset), LexieLee1020 (embedding), Shakhzod555 (RF benchmark) |
| Paper / reference | Plan §6 (architecture), Gorishniy et al. 2021 "Revisiting Deep Learning Models for Tabular Data" |
| Repository | `abailey81/credit-default-tabular-transformer` |
| Cite this model | This model card + PROJECT_PLAN.md commit SHA on the `additional` branch |

## 2. Intended use

**In scope**
- Demonstrating that a from-scratch self-attention transformer can be
  trained end-to-end on a small (30k-row) tabular classification dataset
  and competitively benchmarked against a tuned Random Forest.
- A tokeniser / embedding / attention walkthrough in a pedagogical
  context.
- A case study for calibration methods (§11), subgroup fairness (§11A),
  and uncertainty quantification (§11B) on tabular data.

**Out of scope / prohibited use**
- Real-world credit-decision deployment. This model was trained on a
  single 2005 Taiwanese bank's data; distribution drift is extreme.
- Any automated decision that affects a consumer's access to credit,
  employment, housing, or insurance.
- Transfer to any other tabular task without re-training.

## 3. Training data

- **Source**: UCI Machine Learning Repository, dataset ID 350
  (Yeh & Lien, 2009, *Expert Systems with Applications*).
- **Size**: 30,000 rows, 23 features + 1 binary target (`DEFAULT`).
- **Base rate**: 22.1 % positive class.
- **Temporal coverage**: April – September 2005; October 2005 target.
- **Country**: Taiwan (single bank).
- **Split**: 70 % / 15 % / 15 % stratified on `DEFAULT`
  → 21,000 / 4,500 / 4,500.
- **Provenance**: loaded via `src/data_sources.py` which attempts UCI
  API first, falls back to a committed local `.xls` — same bytes on
  every machine.

See `docs/DATA_SHEET.md` for the full Gebru-et-al. data sheet.

## 4. Evaluation data

Identical to training data's 15 % test split. Not touched during
hyperparameter tuning — decisions were made on the validation split
only.

## 5. Performance measures

Reported in `results/comparison_table.{csv,md}` and
`results/evaluate_summary.json`. Aggregates from
`results/transformer/seed_*/test_metrics.json` (n=3 seeds) plus MTLM
fine-tune (n=1).

| Model | AUC-ROC | AUC-PR | ECE (raw) | ECE (Platt) | Brier |
|---|---|---|---|---|---|
| Transformer from scratch (n=3) | 0.7797 ± 0.0023 | 0.5592 ± 0.0043 | 0.2589 | **0.009** | 0.209 |
| Transformer + MTLM (n=1) | 0.7801 | 0.5605 | 0.2515 | **0.007** | 0.206 |
| Transformer ensemble (arithmetic, n=3) | 0.7815 | 0.5646 | 0.2606 | — | 0.209 |
| RF baseline | 0.7654 | 0.5389 | — | — | — |
| RF tuned | **0.7852** | 0.5648 | **0.0103** | — | 0.133 |

Statistical significance (Phase 12): DeLong AUC-difference RF vs
transformer p = 0.023 (raw), q = 0.23 after BH-FDR across 15 pairwise
comparisons. **We do not claim a statistically-significant AUC gap** at
FDR 0.05. McNemar accuracy differences at τ = 0.5 are significant
because the two models trade off sensitivity / specificity differently
at the default threshold.

**Power**: the 4,500-row test split has 80 % power to detect an
AUC gap of 0.02 or larger at α = 0.05. A 0.005 gap requires ~240,000
rows to adjudicate — the observed RF-vs-transformer gap sits inside the
noise band on our test set.

## 6. Calibration

Raw transformer probabilities are poorly calibrated
(ECE ≈ 0.26) — confidences cluster around the positive-class base rate
but a small number are over-confident. Post-hoc Platt or isotonic
scaling fitted on the validation split drops ECE to 0.007 – 0.019
without moving AUC, matching the RF's calibration level. **Deploy with
Platt scaling; never the raw logits.** See `results/calibration/` and
`figures/calibration_reliability.png`.

## 7. Ethical considerations & limitations

- **Subgroup fairness (Phase 11A, N10)**. On SEX the transformer shows
  a small demographic-parity gap (Male vs Female: Δ = +0.02 selection
  rate); AUC differs by Δ = −0.011 (Female has slightly higher AUC).
  On EDUCATION the subgroup "Other" (n = 61) is underpowered and shows
  extreme disparity (AUC drops by 0.19 – 0.31); the raw-category merge
  0/5/6 → 4 may be concentrating rare cases. This is a coursework
  dataset; we do **not** recommend drawing demographic inferences.
  Full tables: `results/fairness/subgroup_metrics.csv`.
- **Kleinberg–Mullainathan–Raghavan impossibility** (2016). Demographic
  parity, equalised odds, and calibration parity cannot simultaneously
  hold on a non-trivial dataset with base-rate heterogeneity — our
  audit surfaces *which* criterion is violated.
- **Dataset vintage**. 2005 Taiwan; not transferable to present-day
  retail credit, which has different default-correlation structure,
  covariate distributions, and legal protections.
- **Label definition**. Only "default in October 2005" is labelled —
  no future-repayment follow-up, so performance on longer horizons is
  unknown.
- **Single-bank bias**. One Taiwanese bank's underwriting policy;
  selection bias is not quantified.
- **No differential privacy**. The training procedure is not
  differentially private; membership-inference risk on this open
  dataset is low but non-zero.

## 8. Caveats & recommendations

- Always pair predictions with the Phase 11B MC-dropout predictive
  entropy — refuse to predict (defer to human review) for the top
  ~30 % most-uncertain rows lifts retained AUC from 0.779 to 0.82.
  See `figures/uncertainty_refuse_curve.png`.
- When reporting headline metrics in the report, report the ensemble
  row AND the RF row. The from-scratch vs MTLM-pretrained difference
  is within seed noise on this test set (DeLong p = 0.26, Phase 12).
- For any extension of this work, re-fit a fresh calibrator on the
  downstream target distribution rather than assuming the Platt
  coefficients transfer.

## 9. Trained-model artefacts

| Artefact | Location |
|---|---|
| Full checkpoint bundle | `results/transformer/seed_42/best.pt` + `.pt.weights` + `.pt.meta.json` |
| Encoder-only state dict (for MTLM → fine-tune transitions) | produced by `src/train_mtlm.py` → `encoder_pretrained.pt` |
| Per-seed configs | `results/transformer/seed_*/config.json` |
| Per-seed training logs | `results/transformer/seed_*/train_log.csv` |
| Test-set predictions | `results/transformer/seed_*/test_predictions.npz` |

Loaded under `weights_only=True` by default (SECURITY_AUDIT C-1);
`trust_source=True` only for our own bundles.

## 10. Reproducibility

See `docs/REPRODUCIBILITY.md` and `src/repro.py`. Every derivative
artefact (`comparison_table.csv`, `rf/test_predictions.npz`, every
evaluation CSV under `results/calibration/`, `results/fairness/`,
`results/uncertainty/`, `results/significance/`) regenerates bit-stably
from the committed source.

Deterministic transformer retraining requires matching
torch == 2.2.2+cpu, Python 3.12, `CUDA_LAUNCH_BLOCKING=1`, and the
same deterministic-algorithm flags recorded in each run's `config.json`.
