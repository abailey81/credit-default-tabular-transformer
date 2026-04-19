# Model Card — TabularTransformer (Credit-Default Prediction)

A small from-scratch transformer for the UCI credit-default task, built
for UCL MSc Finance & AI coursework. Not production.

## Model details

| Item | Value |
|---|---|
| Type | Encoder-only transformer, FT-Transformer family |
| Layers / heads / d_model / d_ff | 2 / 4 / 32 / 128 |
| Parameters | ~28,600 trainable |
| Inputs | 23 tabular features + `[CLS]` = 24-token sequence |
| Aggregation | `[CLS]` pooling (A5 ablation available) |
| Output | single logit; σ → P(default) |
| Novel biases | N2 feature-group attention bias, N3 temporal-decay bias |
| Pretraining (opt.) | N4 masked tabular LM |
| Framework | PyTorch 2.2, Python 3.10–3.12 |
| License | academic (coursework); MIT on the repo |
| Developers | Tamer Atesyakar (integration, PR merges, hardening), abailey81 (EDA, preprocessing, plan), FardeenIdrus (attention / transformer / eval), Idaliia19 (tokenizer / dataset), LexieLee1020 (embedding), Shakhzod555 (RF benchmark) |
| Reference arch | Gorishniy et al. 2021 |
| Repo | `abailey81/credit-default-tabular-transformer` |
| Cite | this card + `PROJECT_PLAN.md` commit SHA on the `additional` branch |

## Intended use

In scope: showing that a from-scratch self-attention transformer can be
trained on a 30k-row tabular task and benchmarked against a tuned RF; a
pedagogical walkthrough of tokenisation / embedding / attention on
tabular data; calibration, subgroup fairness, and MC-dropout uncertainty
as a case study.

Out of scope: any real credit decision; any automated decision affecting
access to credit, employment, housing, or insurance; transfer to other
tabular tasks without retraining. Training data is one Taiwanese bank,
2005 — distribution drift to any modern setting is severe.

## Training data

UCI ML Repository dataset 350 (Yeh & Lien 2009, *Expert Systems with
Applications*). 30,000 rows, 23 features, binary `DEFAULT`, base rate
22.1 %. April–Sep 2005 observations, Oct 2005 as the prediction target;
single Taiwanese bank. Split 70/15/15 stratified on `DEFAULT`
(21,000 / 4,500 / 4,500). Loaded via `src.data.sources` (UCI API, with
a committed local `.xls` fallback so every machine sees identical bytes).
See `docs/DATA_SHEET.md`.

## Evaluation data

The 15 % test split above. Held out from every tuning decision; model
selection was done on val.

## Performance measures

Reported in `results/evaluation/comparison/comparison_table.{csv,md}` +
`results/evaluation/comparison/evaluate_summary.json`, aggregated from
`results/transformer/seed_*/test_metrics.json` (n=3 supervised seeds) and
the MTLM fine-tune (n=1).

| Model | AUC-ROC | AUC-PR | ECE (raw) | ECE (Platt) | Brier |
|---|---|---|---|---|---|
| Transformer from scratch (n=3) | 0.7797 ± 0.0023 | 0.5592 ± 0.0043 | 0.2589 | 0.011 ± 0.003 | 0.209 |
| Transformer + MTLM (n=1) | 0.7801 | 0.5605 | 0.2515 | 0.007 | 0.206 |
| Transformer ensemble (arithmetic, n=3) | 0.7815 | 0.5646 | 0.2606 | — | 0.209 |
| RF baseline | 0.7654 | 0.5389 | — | — | — |
| RF tuned | 0.7852 | 0.5648 | 0.0103 | — | 0.133 |

DeLong on RF vs transformer (seed_42): p = 0.023 raw, q = 0.23 after BH
over 15 pairwise comparisons. We do not claim a significant AUC gap at
FDR 0.05. McNemar on τ=0.5 accuracy differences *is* significant — the
two models trade sensitivity for specificity differently at that
threshold.

Power: on this 4,500-row test split, detecting an AUC gap of 0.02 at
α = 0.05 / 80 % power would need ~14,500 rows (Hanley-McNeil); a 0.005
gap needs ~237,000. The observed RF-vs-transformer gap (0.008) sits
inside the noise band on this test set. Source:
`results/evaluation/significance/power_analysis.csv`.

## Calibration

Raw transformer probabilities are poorly calibrated (ECE ≈ 0.26) — most
confidences cluster near the base rate, a few are over-confident.
Post-hoc Platt or isotonic fit on the val split drops ECE to 0.007–0.013
across the four runs (3-seed mean 0.011 ± 0.003; MTLM 0.007), no change
in AUC. That matches the tuned RF's native 0.010. Deploy with Platt, not
raw logits. See `results/evaluation/calibration/calibration_metrics.csv`
and `figures/evaluation/calibration/calibration_reliability.png`.

## Ethical considerations and limitations

Phase 11A subgroup audit (N10): on `SEX`, demographic-parity Δ = +0.02
(Male − Female selection rate) and female AUC is 0.011 higher than male.
On `EDUCATION`, the "Other" subgroup (n=61) is underpowered and shows
extreme disparity (AUC 0.19–0.31 below the other buckets); the
preprocessing merge 0/5/6 → 4 may be concentrating rare cases. Full
tables in `results/evaluation/fairness/subgroup_metrics.csv`. This is a coursework
dataset — do not draw demographic inferences from it.

Kleinberg–Mullainathan–Raghavan (2016): demographic parity, equalised
odds, and calibration parity can't all hold when base rates differ
across groups. The audit surfaces which criterion is violated; it does
not promise all three.

Dataset vintage: 2005 Taiwan. Default-correlation structure, covariate
distributions, and legal protections have all shifted. The label only
covers default in October 2005, so longer-horizon performance is
unknown. Selection bias from the bank's underwriting policy is
unquantified. Training isn't differentially private; membership-
inference risk on this public dataset is low but non-zero.

## Caveats and recommendations

Pair predictions with Phase 11B MC-dropout predictive entropy: deferring
the top ~30 % most-uncertain rows lifts retained AUC from 0.779 to 0.82.
See `figures/evaluation/uncertainty/uncertainty_refuse_curve.png`.

When reporting headline metrics, show both the ensemble row and the RF
row. From-scratch vs MTLM pretrained is within seed noise on this test
set (DeLong p = 0.26).

Extending this work? Refit the calibrator on the downstream target
distribution — don't reuse these Platt coefficients.

## Trained-model artefacts

| Artefact | Location |
|---|---|
| Full checkpoint bundle | `results/transformer/seed_42/best.pt` + `.pt.weights` + `.pt.meta.json` |
| Encoder-only (MTLM → fine-tune) | `results/mtlm/run_42/encoder_pretrained.pt` |
| Per-seed config | `results/transformer/seed_*/config.json` |
| Per-seed training log | `results/transformer/seed_*/train_log.csv` |
| Test predictions | `results/transformer/seed_*/test_predictions.npz` |

Checkpoints load with `weights_only=True` by default (SECURITY_AUDIT
C-1); `trust_source=True` only applies to our own bundles.

## Reproducibility

See `docs/REPRODUCIBILITY.md` for the full deterministic / stochastic
taxonomy and `docs/ARCHITECTURE.md` for the folder layout that these
paths live under. Every derivative artefact
(`evaluation/comparison/comparison_table.csv`,
`baseline/rf/test_predictions.npz`, every CSV under
`results/evaluation/calibration/`, `results/evaluation/fairness/`,
`results/evaluation/uncertainty/`, `results/evaluation/significance/`)
regenerates bit-stably from the committed source. Deterministic
transformer retraining wants `torch == 2.2.2+cpu`, Python 3.12,
`CUDA_LAUNCH_BLOCKING=1`, and the deterministic-algorithm flags
recorded in each run's `config.json`.

## References

Mitchell et al. 2019 (Model Cards for Model Reporting); Gorishniy et al.
2021 (Revisiting Deep Learning Models for Tabular Data);
Kleinberg–Mullainathan–Raghavan 2016.
