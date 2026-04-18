# Data Sheet — UCI Credit Card Default (Taiwan, 2005)

The dataset every model in this repo is trained on.

## Motivation

Assembled to benchmark classification methods (including neural nets)
for short-horizon credit-default prediction on records from a Taiwanese
commercial bank. Reference paper: Yeh & Lien (2009), *Expert Systems
with Applications*. Funding details for the original work aren't
published alongside the data. One of the most widely used tabular
benchmarks on UCI.

## Composition

30,000 anonymised credit-card holders. 23 input features + binary
`DEFAULT`, base rate 22.1 %.

| Block | Features | Notes |
|---|---|---|
| Demographic | `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | Credit limit, sex, education, marital status, age. |
| Repayment status (6 mo) | `PAY_0`, `PAY_2..PAY_6` | Ordinal delinquency: −2 no consumption, −1 paid in full, 0 revolving, 1..8 months delayed. |
| Bill amount (6 mo) | `BILL_AMT1..BILL_AMT6` | NT$. Negative = overpayment. |
| Payment amount (6 mo) | `PAY_AMT1..PAY_AMT6` | NT$. Many zeros. |
| Target | `DEFAULT` | 1 if defaulted in Oct 2005. |

No missing values at the UCI API level. Some categoricals carry
undocumented codes (0/5/6 in `EDUCATION`, 0 in `MARRIAGE`);
`src/data_preprocessing.py` folds them into "Other". No direct
identifiers — `AGE`, `SEX`, `EDUCATION`, `MARRIAGE`, `LIMIT_BAL` could
in principle be linked against an external table for re-ID, but no such
table exists in practice. `SEX`, `EDUCATION`, `MARRIAGE` are quasi-
protected under the UK Equality Act and analogues; subgroup performance
is audited in `src/fairness.py` (Phase 11A). 70/15/15 split, stratified
on `DEFAULT`, fixed random state.

## Collection process

April–September 2005, single Taiwanese commercial bank (name unpublished
with the dataset). Individual consent unknown; the bank held consent to
process its own customers' records. No contemporary ethics-board review
is documented; 2005 predates Taiwan's current PDPA framework (in force
from 2012).

## Preprocessing, cleaning, labelling

`src/data_preprocessing.py` and `src/data_sources.py`:

- `EDUCATION` {0, 5, 6} → 4 ("Other").
- `MARRIAGE` {0} → 3 ("Other").
- Clip `BILL_AMT` / `PAY_AMT` at the 99.9th pctile per feature — tames
  the long right tail.
- 22 engineered features (utilisation, repayment ratios, trend slopes,
  MoM deltas). Full list in `data/processed/feature_metadata.json`.
- Continuous features scaled with a Robust Scaler (median / IQR).
  Categoricals stay raw for embedding lookup (tokeniser design).

## Uses

Permitted: classroom / coursework; methodological research on tabular
DL, calibration, fairness-aware ML, uncertainty quantification.

Prohibited: real-world credit-decision pipelines; inferring sensitive
attributes of named individuals; training any LLM whose outputs
personalise or deny credit for named individuals.

Observed failure modes: extreme disparity on the `EDUCATION`="Other"
subgroup (n=61, not reportable); poor out-of-the-box probability
calibration — post-hoc Platt / isotonic is a prerequisite for any
threshold-based decision.

## Distribution

Source: UCI ML Repository dataset 350. License: CC BY 4.0. Raw `.xls`
tracked under `data/raw/` per UCI redistribution policy.

## Maintenance

Maintained by UCI (dataset 350). Revisions are minor; we pin to the
version fetched on the date in `data/processed/validation_report.json`.
The UCI CSV column `PAY_0` appears as `PAY_1` in Yeh & Lien (2009) — an
upstream rename. We use `PAY_0` to match the CSV.

## Legal and ethical

Public under CC BY 4.0, used in hundreds of papers, so re-use here is
low-risk. No consent renewal is possible for 2005 subjects — don't
attempt contact tracing or re-ID. `SEX`, `MARRIAGE`, `EDUCATION` are
sensitive; see `docs/MODEL_CARD.md` for the project's fairness stance.

## References

Gebru et al. 2021 (Datasheets for Datasets); Yeh & Lien 2009.
