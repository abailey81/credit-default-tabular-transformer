# Data Sheet — UCI Credit Card Default (Taiwan, 2005)

Following Gebru et al. (2021) "Datasheets for Datasets". Describes the
dataset used to train every model in this repository.

## Motivation

- **Why was the dataset created?** To benchmark several classification
  methods (including neural networks) on short-horizon credit-default
  prediction using data from a major Taiwanese bank. Referenced in
  Yeh & Lien (2009), *Expert Systems with Applications*.
- **Who funded its creation?** The original paper was produced in an
  academic setting; funding details are not published with the dataset.
- **Has the dataset been used already?** Widely — it is one of the most
  popular tabular classification benchmarks on the UCI repository.

## Composition

- **Instances**: 30,000 anonymised credit-card holders.
- **Features**: 23 input features + 1 binary target (`DEFAULT`).

| Block | Features | Description |
|---|---|---|
| Demographic | `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | Credit limit, sex, education, marital status, age. |
| Repayment status (monthly, 6 months) | `PAY_0`, `PAY_2..PAY_6` | Ordinal delinquency code. Special values: −2 = no consumption, −1 = paid in full, 0 = revolving, 1..8 = months delayed. |
| Bill amount (monthly, 6 months) | `BILL_AMT1..BILL_AMT6` | Bill statement amount (NT $). Negative values indicate overpayment. |
| Payment amount (monthly, 6 months) | `PAY_AMT1..PAY_AMT6` | Amount paid (NT $). Many zero entries (no payment that month). |
| Target | `DEFAULT` | 1 if customer defaulted in October 2005, else 0. |

- **Is there a label?** Yes — `DEFAULT` (binary). **Base rate 22.1 %**.
- **Missing values?** None at the UCI API level; some categorical
  features carry codes (0/5/6 for EDUCATION, 0 for MARRIAGE) outside
  the documented schema — we merge these into the "Other" bucket during
  preprocessing (see `src/data_preprocessing.py`).
- **Identifiers?** The dataset contains no direct identifiers. `AGE`,
  `SEX`, `EDUCATION`, `MARRIAGE`, `LIMIT_BAL` combined could
  theoretically be used for re-identification against an external
  voter-roll, but no such external table exists in practice.
- **Sensitive features**: `SEX`, `EDUCATION`, `MARRIAGE` are
  quasi-protected attributes under the UK Equality Act and equivalents.
  We audit subgroup performance in `src/fairness.py` (Phase 11A).
- **Split**: 70 % / 15 % / 15 %, stratified on `DEFAULT`, random state
  fixed across all runs.

## Collection process

- **Source**: April – September 2005 transactional records of a single
  Taiwanese commercial bank (name not published).
- **Consent**: Unknown at the individual level; the bank held consent
  to process its own customers' data.
- **Ethical review**: No contemporary ethics-board review is documented.
  2005 predates the current PDPA framework in Taiwan (which took effect
  2012).

## Preprocessing / cleaning / labelling

Applied by `src/data_preprocessing.py` and `src/data_sources.py`:

- Merge EDUCATION values {0, 5, 6} → 4 ("Other").
- Merge MARRIAGE value 0 → 3 ("Other").
- Clip BILL_AMT / PAY_AMT at the 99.9th percentile per feature to
  control the long right tail.
- Derive 22 engineered features (utilisation ratios, repayment ratios,
  trend slopes, month-over-month deltas) — see
  `data/processed/feature_metadata.json` for the full list.
- Scale all continuous features by Robust Scaler (median / IQR) — per
  the tokeniser's design, categorical features are left raw for
  embedding lookup.

## Uses

**Permitted**
- Pedagogical use in a coursework / classroom setting.
- Methodological research into tabular deep learning, calibration,
  fairness-aware ML, uncertainty quantification.

**Prohibited**
- Deployment as part of any real-world credit-decision pipeline.
- Inferring sensitive attributes of individuals.
- Training an LLM whose outputs are intended to personalise or reject
  credit for named individuals.

**Failure modes already observed**
- Extreme disparity on the underpowered EDUCATION = "Other"
  (n = 61) subgroup — results for that subgroup are not reportable.
- A model trained on this dataset is poorly calibrated out-of-the-box —
  post-hoc Platt or isotonic is required before any threshold-based
  decision.

## Distribution

- **Primary source**: UCI ML Repository, dataset ID 350.
- **Licence**: CC BY 4.0.
- **Redistribution**: The raw `.xls` is committed to this repo (under
  `data/raw/`) per UCI policy.

## Maintenance

- **Maintainer**: UCI ML Repository (dataset 350).
- **Versioning**: UCI dataset revisions are minor; we pin to the version
  available at the `data_sources.py` fetch date recorded in
  `data/processed/validation_report.json`.
- **Erratum**: The PAY_0 column in the UCI CSV is documented as PAY_1
  in the original paper (Yeh & Lien 2009) — an upstream rename. We use
  the CSV column name `PAY_0` throughout.

## Legal and ethical considerations

- The dataset is public under CC BY 4.0 and has been used in hundreds
  of published papers; re-use here is low-risk.
- No consent renewal is possible for 2005 subjects; do not attempt
  contact tracing or re-identification.
- The `SEX` / `MARRIAGE` / `EDUCATION` features are sensitive; see
  `docs/MODEL_CARD.md` §7 for our stance on fairness in this project.
