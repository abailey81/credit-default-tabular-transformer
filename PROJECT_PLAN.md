# Project Plan: Transformer-Based Credit Default Prediction

## UCL MSc Coursework — Complete Execution Blueprint

---

## Table of Contents

1. [Project Overview](#1-project-overview) — includes **§1.6 Novelty & Independent Contribution Register**
2. [Dataset Intelligence](#2-dataset-intelligence)
3. [Phase 1: Data Loading & Cleaning](#3-phase-1-data-loading--cleaning)
4. [Phase 2: Exploratory Data Analysis](#4-phase-2-exploratory-data-analysis)
5. [Phase 3: Tokenisation & Embedding Design](#5-phase-3-tokenisation--embedding-design) — incl. §5.4A PLE, §5.4B MLM-compatible masking
6. [Phase 4: Transformer Architecture](#6-phase-4-transformer-architecture) — incl. §6.11 `src/transformer.py` spec, §6.12 novel inductive biases (N2, N3)
7. [Phase 5: Loss Functions & Class Imbalance](#7-phase-5-loss-functions--class-imbalance)
8. [Phase 6: Training Pipeline](#8-phase-6-training-pipeline)
   - 8.5 [Phase 6A: Masked Tabular Language Modelling pretraining (N4)](#85-phase-6a-self-supervised-masked-tabular-language-modelling-novel--n4)
   - 8.6 [Phase 6B: Multi-task auxiliary objective (N5)](#86-phase-6b-multi-task-auxiliary-objective-novel--n5)
9. [Phase 7: Random Forest Benchmark](#9-phase-7-random-forest-benchmark)
10. [Phase 8: Evaluation & Metrics](#10-phase-8-evaluation--metrics)
    - 10.7 [Phase 8A: Business / Cost-Sensitive Evaluation (N9)](#107-phase-8a-business--cost-sensitive-evaluation-novel--n9)
11. [Phase 9: Ablation Studies](#11-phase-9-ablation-studies) — 22 ablations (A1–A22)
12. [Phase 10: Attention Visualisation & Interpretability](#12-phase-10-attention-visualisation--interpretability) — incl. IG, SHAP, probing, CKA, Jain-&-Wallace (N8)
    - 12.5 [Phase 10A: Counterfactual Explanations (N6)](#125-phase-10a-counterfactual-explanations-novel--n6)
13. [Phase 11: Calibration Analysis](#13-phase-11-calibration-analysis) — incl. temperature scaling, Brier decomposition
    - 13.5 [Phase 11A: Fairness & Subgroup Robustness (N10)](#135-phase-11a-fairness--subgroup-robustness-novel--n10)
    - 13.6 [Phase 11B: Uncertainty Quantification (N11)](#136-phase-11b-uncertainty-quantification-novel--n11)
14. [Phase 12: Statistical Significance Testing](#14-phase-12-statistical-significance-testing) — incl. paired bootstrap, BH-FDR, power analysis
15. [Phase 13: Report Writing](#15-phase-13-report-writing)
16. [Phase 14: GitHub Repository Structure](#16-phase-14-github-repository-structure) — incl. §16.4 engineering standards
    - 16.5 [Phase 14A: Reproducibility Guarantees](#165-phase-14a-reproducibility-guarantees)
    - 16.6 [Phase 14B: Model Card & Data Sheet (N12)](#166-phase-14b-model-card--data-sheet-novel--n12)
17. [Marking Alignment Matrix](#17-marking-alignment-matrix)
18. [Risk Register & Mitigation](#18-risk-register--mitigation)
19. [Timeline & Milestones](#19-timeline--milestones)
20. [Reference Library](#20-reference-library)
21. [**Coursework-PDF Strict Relevance Audit**](#21-coursework-pdf-strict-relevance-audit) — verifies every plan item maps to a PDF requirement

---

## 1. Project Overview

### 1.1 Objective

Build and compare two models for predicting credit card default on the UCI Taiwan Credit Card Default dataset:

1. **A small transformer-based language model built from scratch** — the core deliverable, worth 40% of marks
2. **A tuned random forest benchmark** — the comparison baseline

The transformer must use explicit self-attention with queries, keys, and values, implemented manually (no `nn.TransformerEncoder`, no `nn.MultiheadAttention`, no pre-built transformer libraries). A standard feed-forward neural network does not qualify.

### 1.2 What "From Scratch" Means

**Permitted:**
- PyTorch or TensorFlow for tensors, autograd, and optimisers
- `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Dropout`, `nn.GELU`
- `nn.Module` as a base class
- Standard Python libraries (NumPy, pandas, scikit-learn, matplotlib, seaborn, SHAP)

**Prohibited:**
- `nn.TransformerEncoder`, `nn.TransformerEncoderLayer`
- `nn.MultiheadAttention`
- Any pre-built transformer library (HuggingFace, x-transformers, tab-transformer-pytorch, etc.)
- Any LLM API for modelling, training, or inference
- Pre-trained model weights of any kind

### 1.3 Deliverables

| Deliverable | Format | Constraint |
|---|---|---|
| Report | **Single PDF** (one upload) | ≤ 4,000 words main body (code, formulae, figures, tables, appendices excluded) |
| Code | GitHub repository | Link must appear in report |
| Acknowledgements | In report | ≤ 50 words, **must declare any LLM use + how** |
| Contribution table | In Acknowledgements | Breakdown per group member; each member must have a substantive entry |

### 1.3.1 Submission & Viva Logistics

- **Submission**: one PDF, by the **group lead**, one per group. No additional files accepted.
- **Contribution risk**: per coursework PDF, serious contribution concerns must be raised within **11 calendar days** from the start of the project; concerns raised only after submission rarely reverse a shared mark unless there's documentary evidence. Keep a git history, meeting notes, and task-allocation list as contemporaneous evidence.
- **Face-to-face viva**: "A group may be asked to attend a face-to-face meeting to explain the project and the results. In that case, all group members must be present." Every member should be ready to defend every design decision — tokeniser, attention maths, loss choice, hyperparameter rationale, ablation findings. Don't let any one member "own" a concept the others can't explain.
- **LLM use acknowledgement**: using tools like Claude/ChatGPT is permitted, but must be declared in the Acknowledgements (within the 50-word limit). Example wording: "We used [tool] for [specific purpose — e.g. code review, proofreading] only." Failing to declare is a reproducibility/integrity issue; copying verbatim is penalised.

### 1.4 Marking Allocation

| Section | Weight | Target Word Count |
|---|---|---|
| Section 3: Model Build-up | 40% | ~1,400–1,600 words |
| Section 4: Experiments, Results, Discussion | 30% | ~1,000–1,200 words |
| Section 5: Conclusions | 5% | ~150–200 words |
| All other sections + structure + writing + referencing | 25% | ~800–1,000 words |

### 1.5 Target Grade: 85%+ (Distinction)

The UCL HEQF Level 7 criteria for this grade require:
- Authoritative, comprehensive knowledge at the cutting edge
- Sophisticated critical analysis with fluent, persuasive, well-evidenced arguments
- Highly original thought
- Exceptional range of relevant literature used critically
- Independently formulated research questions with sophisticated analysis

This means going substantially beyond minimum requirements: ablation studies, attention interpretability, calibration analysis, statistical significance testing, embedding space analysis, critical engagement with the "trees vs transformers" debate.

### 1.6 Novelty & Independent Contribution Register

Forty per cent of the total mark is allocated to "model design, novelty, independent thought, methodology, and reasoning" (coursework PDF). Markers reward demonstrable independent contributions more than they reward faithful reimplementations of published methods. Every claim in the report will be tagged as either **(standard)** — i.e. a careful implementation of a published technique — or **(novel / independent)** — i.e. an idea that did not exist in the cited literature in the form we use it. The register below declares the independent contributions up front so the marker knows where to look.

| # | Independent Contribution | Verified By | Novelty Type |
|---|---|---|---|
| N1 | **Hybrid PAY tokenisation** — decomposing each PAY value into a 4-state categorical embedding plus an ordinal severity projection, motivated by a specific EDA finding (non-linear default-rate jump between PAY=0 and PAY≥2). No existing tabular transformer in the literature treats ordinal payment statuses this way. | Ablation A1 | Tokenisation design |
| N2 | **Feature-group attention bias** — a learned additive bias matrix that softly encourages attention within the three temporal feature groups (PAY, BILL_AMT, PAY_AMT) while permitting cross-group flow. A credit-risk-specific inductive bias. | Ablation A21 | Architectural |
| N3 | **Temporal-decay positional prior** — a learned scalar decay added to attention scores along the temporal axis, encoding the prior that recent months matter more than distant ones for default risk. | Ablation A22 | Architectural |
| N4 | **Masked Tabular Language Modelling (MTLM) pretraining** — a BERT-style self-supervised pretraining objective where a random subset of feature tokens is masked and predicted from the rest, *before* supervised fine-tuning. Adapts Rubachev et al. (2022) to this specific dataset and hybrid-PAY tokeniser. This is the single most "LLM-flavoured" component of the project and directly addresses the coursework's "language model" framing. | Phase 6A + Ablation A15 | Training objective |
| N5 | **Multi-task auxiliary PAY-forecast head** — during supervised fine-tuning, the model simultaneously predicts DEFAULT *and* PAY_0 from PAY_2..PAY_6, with a weighted joint loss. Regularises the representation via auxiliary supervision without needing additional data. | Phase 6B + Ablation A16 | Training objective |
| N6 | **Counterfactual token substitution** — for interpretability, we substitute individual tokens ("what if this customer had PAY_0=0 instead of PAY_0=3?") and measure the change in predicted probability. Produces customer-level explanations aligned with the EU GDPR "right to explanation" for automated decisions. | Phase 10A | Interpretability method |
| N7 | **Random-attention and linear-probe null baselines** — lower bounds on whether attention is actually doing useful work. A transformer that is not statistically distinguishable from one with uniform attention weights has not learned anything. Forces intellectual honesty. | Ablations A17, A18 | Experimental design |
| N8 | **Jain & Wallace (2019) adversarial attention-perturbation diagnostic on tabular data** — running their attention-as-explanation stress test on our model, which (to our knowledge) has not been done on a tabular transformer before. | Phase 10 | Critical extension |
| N9 | **Cost-sensitive portfolio evaluation** — an Expected Credit Loss (ECL) framing with Loss-Given-Default (LGD) and Exposure-at-Default (EAD) approximations, beyond threshold-independent AUC. Directly addresses the business context of the dataset. | Phase 8A | Domain modelling |
| N10 | **Subgroup fairness audit** across SEX / EDUCATION / MARRIAGE — demographic parity, equalised odds, and subgroup AUC-ROC/calibration. Standard in fair-ML literature, novel for *this* dataset and in comparison between a transformer and RF. | Phase 11A | Responsible ML |
| N11 | **Monte-Carlo-dropout uncertainty quantification** — predictive-entropy-based uncertainty estimates from the transformer, correlated with misclassification rate. Allows "refuse to predict" behaviour critical for regulated credit decisions. | Phase 11B | Uncertainty ML |
| N12 | **Model Card and Data Sheet** — following Mitchell et al. (2019) and Gebru et al. (2021), formal responsible-AI documentation artefacts for the final model. | Phase 14B | Responsible ML |

Everything else in the plan is honest implementation of published work. The reference library in §20 attributes each standard component to its primary source. Markers should look to the 12 items above for the "independent thought" marks.

---

## 2. Dataset Intelligence

### 2.1 Provenance

- **Source**: Yeh, I.C. & Lien, C.H. (2009). Expert Systems with Applications, 36(2), 2473–2480.
- **Origin**: Major Taiwanese bank, April–September 2005, predicting October 2005 default.
- **Context**: Taiwan experienced a severe credit card debt crisis in 2005–2006 due to over-issuance of cards to unqualified applicants. This context matters for the introduction and discussion sections.
- **Size**: 30,000 records, 23 features, 1 binary target.

### 2.2 Complete Feature Schema

**Demographic / Static Features (5):**

| Feature | Type | Values | Semantics | Cleaning Required |
|---|---|---|---|---|
| LIMIT_BAL | Continuous | 10,000–1,000,000 NT$ | Credit limit (individual + supplementary) | None (check for non-positive) |
| SEX | Binary | 1=Male, 2=Female | Gender | None |
| EDUCATION | Categorical | 1=Grad school, 2=University, 3=High school, 4=Others | Education level | Merge 0/5/6 → 4 |
| MARRIAGE | Categorical | 1=Married, 2=Single, 3=Others | Marital status | Merge 0 → 3 |
| AGE | Continuous | 21–79 | Age in years | None |

**Repayment Status Features (6) — Temporal, September → April:**

| Feature | Type | Values | Semantics |
|---|---|---|---|
| PAY_0 | Special ordinal | -2, -1, 0, 1–8 | September repayment status |
| PAY_2 | Special ordinal | Same | August (note: PAY_1 does not exist) |
| PAY_3 | Special ordinal | Same | July |
| PAY_4 | Special ordinal | Same | June |
| PAY_5 | Special ordinal | Same | May |
| PAY_6 | Special ordinal | Same | April |

**PAY value semantics (critical for tokenisation):**
- **-2** = No consumption (no bill existed that month — the customer didn't use the card)
- **-1** = Paid in full and on time
- **0** = Minimum payment made (revolving credit — carrying a balance)
- **1–8** = Months of payment delay (ordinal scale of increasing delinquency)

The values -2, -1, 0 are **qualitatively different states**, not points on a scale. Values 1–8 are an **ordinal delinquency scale** where higher = worse. Treating these as a simple continuous variable destroys this semantic structure. This is the single most important tokenisation design decision in the project.

**Bill Amount Features (6) — Temporal:**

| Feature | Type | Semantics | Notes |
|---|---|---|---|
| BILL_AMT1 | Continuous | September bill statement (NT$) | Can be negative (overpayment — NOT an error) |
| BILL_AMT2 | Continuous | August | Highly right-skewed with long tails |
| BILL_AMT3 | Continuous | July | Strong autocorrelation across months |
| BILL_AMT4 | Continuous | June | |
| BILL_AMT5 | Continuous | May | |
| BILL_AMT6 | Continuous | April | |

**Payment Amount Features (6) — Temporal:**

| Feature | Type | Semantics | Notes |
|---|---|---|---|
| PAY_AMT1 | Continuous | Amount paid in September (NT$) | Many zeros (non-payment) |
| PAY_AMT2 | Continuous | August | Right-skewed |
| PAY_AMT3 | Continuous | July | |
| PAY_AMT4 | Continuous | June | |
| PAY_AMT5 | Continuous | May | |
| PAY_AMT6 | Continuous | April | |

**Target:**

| Feature | Type | Distribution |
|---|---|---|
| DEFAULT | Binary | 0 = No default (23,364 ≈ 77.9%), 1 = Default (6,636 ≈ 22.1%) |

### 2.3 Key Data Structures to Exploit

1. **Three temporal subsequences** (PAY, BILL_AMT, PAY_AMT) each form 6-step time series per customer. Defaulters and non-defaulters show diverging trajectories. Self-attention across time steps can capture these patterns.
2. **Feature interactions**: credit utilisation (BILL_AMT / LIMIT_BAL), repayment ratio (PAY_AMT / BILL_AMT), delinquency × demographics. Self-attention naturally models pairwise interactions.
3. **Class imbalance**: 3.5:1 ratio. Not extreme, but sufficient to bias toward majority class. Must be addressed.
4. **PAY transition dynamics**: month-to-month transitions in payment status differ between defaulters and non-defaulters, forming a Markov-like sequential structure.

---

## 3. Phase 1: Data Loading & Cleaning

**Status: [DONE] COMPLETE**

### 3.1 Tasks

| Task | Detail | Justification |
|---|---|---|
| Load XLS | `pd.read_excel(path, header=1)` — skip metadata row | Raw file has a spurious header row |
| Schema normalisation | Rename PAY_1 → PAY_0 if present, standardise target column name, drop ID | Known naming inconsistency across file versions |
| Clean EDUCATION | Merge values {0, 5, 6} → 4 (Others) | Undocumented in original paper; only codes 1–4 defined |
| Clean MARRIAGE | Merge value 0 → 3 (Others) | Undocumented; only codes 1–3 defined |
| Validate | Check value ranges, missing values, duplicates, target is binary | Data integrity before modelling |
| Feature engineering | Utilisation ratios, repayment ratios, delinquency aggregates, bill slope, temporal summaries | For RF and EDA; transformer will learn these from raw features via attention |
| Stratified split | 70/15/15 train/val/test, stratified by target | Preserves 22.1% default rate in each split; test never seen during model selection |
| Scaling | StandardScaler fitted on train only, applied to val/test | Prevents data leakage; numerical features need normalisation before embedding |
| Metadata export | Category mappings, PAY value vocabularies, feature ordering | Required by tokeniser to build embedding tables |

### 3.2 Outputs

- `data/processed/train_raw.csv`, `val_raw.csv`, `test_raw.csv` — unscaled splits
- `data/processed/train_scaled.csv`, `val_scaled.csv`, `test_scaled.csv` — scaled splits
- `data/processed/train_engineered.csv`, `val_engineered.csv`, `test_engineered.csv` — with derived features
- `data/processed/feature_metadata.json` — tokeniser metadata
- `data/processed/validation_report.json` — data quality report

### 3.3 Key Findings from Cleaning

- 345 undocumented EDUCATION values cleaned (0: 14, 5: 280, 6: 51)
- 54 undocumented MARRIAGE values cleaned (all value 0)
- 35 duplicate rows detected (0.12%) — retained to match original dataset
- No missing values
- BILL_AMT columns contain legitimate negative values (overpayment/credit balance)
- Stratified splits achieved: all three sets have default rate ≈ 0.2211

---

## 4. Phase 2: Exploratory Data Analysis

**Status: [DONE] COMPLETE**

### 4.1 Figures Produced

| Figure | Content | Modelling Implication |
|---|---|---|
| Fig 01 | Class distribution (counts + proportions) | 3.5:1 imbalance → need focal loss or class weighting |
| Fig 02 | Categorical features by default rate + χ² tests | SEX, EDUCATION, MARRIAGE all significantly associated; different default rates per category |
| Fig 03 | LIMIT_BAL and AGE distributions by default + Mann-Whitney tests | Defaulters have lower credit limits; age effect is weaker |
| Fig 04 | PAY_0 semantic analysis: distribution, default rate by value, heatmap across months | **Key figure**: demonstrates non-linear risk profile and categorical-vs-ordinal structure of PAY features → directly motivates hybrid tokenisation |
| Fig 05 | Temporal trajectories (PAY, BILL_AMT, PAY_AMT) with 95% CI | Defaulters show diverging trajectories → motivates sequence modelling / self-attention across time steps |
| Fig 06 | Credit utilisation analysis (distribution + temporal) | Defaulters have higher utilisation → BILL_AMT/LIMIT_BAL interaction is important |
| Fig 07 | Full 24×24 correlation heatmap | BILL_AMT features are highly autocorrelated (0.85–0.97); PAY features most correlated with target |
| Fig 08 | Feature-target association ranking (point-biserial r / Cramér's V) | PAY_0 is the strongest predictor; PAY features dominate; PAY_AMT features are weakest |
| Fig 09 | BILL_AMT autocorrelation analysis + decay by default status | Strong temporal structure; autocorrelation decays faster for defaulters → attention can exploit this |
| Fig 10 | Feature interactions (utilisation scatter, PAY_0 × LIMIT_BAL, age × education × default) | Pairwise interactions exist → self-attention is well-suited to discover them |
| Fig 11 | PAY transition matrices (defaulters vs non-defaulters) | Defaulters have higher transition probabilities toward worse payment states → sequential dependency |
| Fig 13 | Repayment ratio analysis (distribution + temporal) | Defaulters consistently pay a smaller fraction of their bill |

### 4.2 Summary Statistics

Full summary statistics table exported as CSV and LaTeX, including mean/std/median/skewness/kurtosis for all 23 features, split by default status.

### 4.3 Key EDA Insights for the Report

1. **PAY features are by far the strongest predictors** — PAY_0 alone has |r_pb| ≈ 0.32, while most other features are below 0.15.
2. **The PAY value space has dual structure** — categorical states (-2, -1, 0) and ordinal delinquency (1–8). Default rate jumps from ~10% at PAY_0=0 to ~60% at PAY_0=2. This non-linearity demands careful tokenisation.
3. **Clear temporal divergence** — defaulters and non-defaulters separate over the 6-month window. This sequential structure is exactly what self-attention is designed to capture.
4. **Strong BILL_AMT autocorrelation** — adjacent months correlated at 0.95+. But this decays faster for defaulters, creating a signal that attention can exploit.
5. **Feature interactions matter** — credit utilisation, repayment ratio, delinquency × demographics all show interaction effects. Self-attention naturally models pairwise feature interactions.
6. **Class imbalance is moderate** — 22.1% default rate (3.5:1) is enough to bias toward majority class but not extreme enough to require aggressive resampling.

---

## 5. Phase 3: Tokenisation & Embedding Design

**Status: [DONE] COMPLETE** — `src/tokenizer.py` (hybrid PAY state+severity
tokenisation, Novelty N1, plus `MTLMCollator` for Phase 6A); `src/embedding.py`
(`FeatureEmbedding` with per-feature projections, [CLS] token, optional
temporal positional encoding for Ablation A7, optional [MASK] token for MTLM
pretraining — mask content swap preserves both feature positional and temporal
positional signals). PLE numerical encoding variant (§5.4A) remains TODO for
Ablation A14.

This is the intellectual heart of the project. The coursework spec says Section 3 (Model Build-up) is worth 40% of marks and must describe tokenisation, embedding design, and justify every choice.

### 5.1 The Core Challenge

Transformers expect a sequence of token embeddings. NLP tokenises text into subwords. We must tokenise a tabular row (23 heterogeneous features) into a sequence of d-dimensional vectors.

**Approach**: Following the FT-Transformer paradigm (Gorishniy et al., 2021), each feature becomes one token. A row of 23 features → a sequence of 23 tokens + 1 [CLS] token = 24 tokens total.

### 5.2 Feature-Type-Specific Embedding Strategies

#### 5.2.1 Numerical Features (LIMIT_BAL, AGE, BILL_AMT1–6, PAY_AMT1–6 — 14 features)

**Method**: Per-feature linear projection with learnable bias.

$$e_j = x_j \cdot \mathbf{w}_j + \mathbf{b}_j$$

where $x_j \in \mathbb{R}$ is the standardised scalar value, $\mathbf{w}_j \in \mathbb{R}^d$ is a learnable weight vector unique to feature $j$, and $\mathbf{b}_j \in \mathbb{R}^d$ is a learnable bias unique to feature $j$.

**Justification**: The FT-Transformer paper demonstrated that per-feature linear projections with bias are effective for numerical features. The bias is critical — it provides a "default position" in embedding space for each feature even when the value is zero, and the Gorishniy et al. (2021) ablation showed removing biases degrades performance significantly.

**Implementation**: `nn.Linear(1, d, bias=True)` per numerical feature. Each feature gets its own projection (no weight sharing) because the semantic meaning of "1 standard deviation above the mean" differs completely between LIMIT_BAL and AGE.

**Initialisation**: Xavier normal for weights, zeros for bias.

**Advanced option (to implement and ablate)**: Piecewise Linear Encoding (PLE) from Gorishniy et al. (2022). Discretise each numerical feature into Q quantile-based bins, compute the fractional position within the active bin, and project the resulting sparse vector. This captures non-linear relationships that a single linear projection cannot. We will implement PLE as an ablation to compare against simple linear projection.

#### 5.2.2 Categorical Features (SEX, EDUCATION, MARRIAGE — 3 features)

**Method**: Per-feature embedding lookup table.

$$e_j = \text{Embedding}_j(c_j)$$

where $c_j \in \{0, 1, \ldots, C_j - 1\}$ is the integer-encoded category index and $\text{Embedding}_j$ is a learnable lookup table of shape $(C_j, d)$.

| Feature | Vocabulary Size ($C_j$) | Categories |
|---|---|---|
| SEX | 2 | Male, Female |
| EDUCATION | 4 | Grad School, University, High School, Others |
| MARRIAGE | 3 | Married, Single, Others |

**Justification**: Embedding lookup is the natural choice for categorical data — identical to word embeddings in NLP. Each category gets its own freely learnable position in d-dimensional space. No ordinal relationship is assumed between categories.

**Initialisation**: $\mathcal{N}(0, 0.02)$ following BERT convention.

#### 5.2.3 PAY Features (PAY_0, PAY_2–PAY_6 — 6 features) — THE KEY DESIGN DECISION

The PAY features have a hybrid semantic structure that makes them the most interesting tokenisation challenge:

- **-2** (no consumption): a categorical state meaning "inactive"
- **-1** (paid in full): a categorical state meaning "responsible"
- **0** (minimum payment): a categorical state meaning "revolving credit"
- **1–8** (months delay): an ordinal delinquency scale

**Three candidate strategies:**

**Strategy A — Pure Categorical (our primary approach):**
Treat each of the 11 possible PAY values as a distinct category with its own learnable embedding. Vocabulary size = 11 per PAY feature.

$$e_j = \text{Embedding}_j(\text{PAY\_value}_j)$$

- **Pros**: Fully flexible; makes no assumptions about distances; the model learns the optimal geometry from data; naturally captures the categorical nature of -2/-1/0.
- **Cons**: Doesn't encode the prior knowledge that delay=5 is worse than delay=3 in a structured way; must learn this ordering from scratch.
- **Why we prefer this**: With 21,000 training samples, there is sufficient data to learn the ordinal relationship among delay values. And the categorical treatment correctly avoids the false assumption that -2 → -1 → 0 → 1 is a uniform scale.

**Strategy B — Pure Numerical:**
Standardise and project like any other numerical feature.

$$e_j = \text{PAY\_value}_j \cdot \mathbf{w}_j + \mathbf{b}_j$$

- **Pros**: Simple; naturally encodes the ordinal relationship of 1–8.
- **Cons**: Treats the jump from -1 to 0 as equivalent to the jump from 4 to 5, which is semantically incorrect. Destroys the categorical structure of -2/-1/0.

**Strategy C — Hybrid (to implement as ablation):**
Decompose each PAY value into two components:

1. A **state embedding**: categorical embedding for {no_bill, paid, revolving, delinquent} (4 states)
2. A **severity embedding**: numerical projection of the delay count (0 for non-delinquent states, 1–8 for delinquent)

Combine by addition:

$$e_j = \text{StateEmbed}_j(s_j) + \text{severity}_j \cdot \mathbf{w}^{\text{sev}}_j + \mathbf{b}^{\text{sev}}_j$$

- **Pros**: Respects both the categorical structure and the ordinal structure.
- **Cons**: More complex; introduces an assumption about the 4-state decomposition.

**Implementation plan**: Implement Strategy A as the primary model. Implement Strategy B and Strategy C as ablation variants. Compare all three in Section 4 (Experiments).

### 5.3 Feature-Type Embeddings (Column Embeddings)

After computing the value embedding for each feature, we add a **feature-type embedding** that identifies which feature this token represents:

$$\text{token}_j = \text{value\_embed}(x_j) + \text{feature\_type\_embed}(j)$$

**Implementation**: `nn.Embedding(n_features, d)` where `n_features = 23`. Feature 0 always maps to the same learned vector regardless of the customer's data.

**Justification**: Without feature-type embeddings, the transformer cannot distinguish a token from LIMIT_BAL vs a token from AGE — they are just anonymous d-dimensional vectors. Feature-type embeddings are the tabular analogue of positional encodings in NLP. We use learned embeddings rather than sinusoidal because (a) the feature order is arbitrary (unlike word order in NLP), and (b) learned embeddings allow the model to discover which features are semantically related.

**Initialisation**: $\mathcal{N}(0, 0.02)$.

### 5.4 Optional: Temporal Positional Encoding

The three temporal feature groups (PAY, BILL_AMT, PAY_AMT) each span 6 months. While the feature-type embedding already distinguishes PAY_0 from PAY_2, an additional temporal positional encoding could help the model recognise that PAY_0 (September) is more recent than PAY_6 (April).

**Method**: Learned temporal position embedding of size (6, d), added to tokens within each temporal group.

$$\text{token}_j = \text{value\_embed}(x_j) + \text{feature\_type\_embed}(j) + \text{temporal\_pos\_embed}(t_j)$$

where $t_j \in \{0, 1, 2, 3, 4, 5\}$ is the temporal index (0 = most recent).

**Decision**: Implement as an ablation. It may help or may be redundant (the feature-type embedding might already capture temporal position implicitly). Testing this is an interesting experiment.

### 5.4A Advanced Numerical Encoding: Piecewise Linear Encoding (PLE)

Following Gorishniy et al. (2022), we implement PLE as an alternative to the simple linear projection (§5.2.1), tested in Ablation A14.

Per numerical feature $j$:

1. Discretise the training distribution of $x_j$ into $Q$ quantile-based bins with edges $b_0 < b_1 < \cdots < b_Q$.
2. For an input value $x$, produce a sparse PLE vector $\text{ple}_j(x) \in \mathbb{R}^Q$ where:
   - $\text{ple}_j(x)_q = 1$ for all bins $q$ strictly below the active bin (the one containing $x$),
   - $\text{ple}_j(x)_q = (x - b_{q-1}) / (b_q - b_{q-1})$ for the active bin,
   - $\text{ple}_j(x)_q = 0$ for all bins strictly above.
3. Project: $e_j = W_j^{\text{PLE}} \cdot \text{ple}_j(x) + b_j^{\text{PLE}}$ with $W_j^{\text{PLE}} \in \mathbb{R}^{d \times Q}$.

**Why**: PLE captures non-monotone / non-linear relationships between a feature's magnitude and the target that a single linear projection cannot. Gorishniy et al. (2022) showed PLE substantially outperforms linear projection on highly non-linear tabular targets. Use $Q = 10$ quantile bins by default.

### 5.4B MLM-Compatible Tokenisation

For the MLM self-supervised pretraining stage (Phase 6A), the tokeniser must support two additional operations:

1. **`[MASK]` token injection**: with masking probability $p_{\text{mask}} = 0.15$, randomly replace a feature token with a learnable `[MASK]` embedding. To avoid train/eval mismatch, follow BERT: 80% of the time replace with `[MASK]`, 10% with a random valid token from the same feature's vocabulary, 10% keep the original token but flag it for loss computation.
2. **Per-feature prediction heads**: for categorical / PAY features, a classification head over the feature's vocabulary; for numerical features, a regression head (MSE on the scaled value). One head per feature type (shared across positions of the same type).

The tokeniser exposes two modes: `forward()` for supervised training / inference (no masking) and `forward_with_masking()` that additionally returns the mask positions and the original target values for loss computation.

### 5.5 The [CLS] Token

**Purpose**: A learnable aggregation token prepended to the sequence. It has no associated feature value — its role is to collect information from all feature tokens via self-attention and produce the final classification signal.

**Implementation**: `nn.Parameter(torch.randn(1, 1, d) * 0.02)` — broadcast across the batch and concatenated as the first token.

**Justification**: The [CLS] token approach (from BERT) provides a dedicated aggregation point. The alternative — mean pooling over all feature tokens — treats all features equally. [CLS] allows the model to learn a task-specific weighted aggregation via attention.

**Ablation**: Compare [CLS] aggregation vs mean pooling vs max pooling.

### 5.6 Full Token Sequence

For one customer, the input to the transformer is:

```
Position 0:  [CLS]         — learnable parameter, shape (d,)
Position 1:  LIMIT_BAL     — linear projection of scaled value + feature-type embed
Position 2:  SEX           — category embedding lookup + feature-type embed
Position 3:  EDUCATION     — category embedding lookup + feature-type embed
Position 4:  MARRIAGE      — category embedding lookup + feature-type embed
Position 5:  AGE           — linear projection of scaled value + feature-type embed
Position 6:  PAY_0         — PAY embedding lookup + feature-type embed
Position 7:  PAY_2         — PAY embedding lookup + feature-type embed
...
Position 11: PAY_6         — PAY embedding lookup + feature-type embed
Position 12: BILL_AMT1     — linear projection + feature-type embed
...
Position 17: BILL_AMT6     — linear projection + feature-type embed
Position 18: PAY_AMT1      — linear projection + feature-type embed
...
Position 23: PAY_AMT6      — linear projection + feature-type embed
```

**Total sequence length**: 24 tokens (1 CLS + 23 features), each of dimension d.
**Batch shape**: (B, 24, d) where B is the batch size.

### 5.7 Module: `src/tokeniser.py`

```python
class FeatureTokeniser(nn.Module):
    """
    Converts a batch of raw (preprocessed) tabular rows into a
    sequence of token embeddings for the transformer.
    
    Input:  dict with keys for each feature → tensors of shape (B,)
    Output: tensor of shape (B, n_tokens, d_model)
    """
    def __init__(self, metadata, d_model, pay_strategy="categorical"):
        # Numerical embedders: dict of nn.Linear(1, d_model)
        # Categorical embedders: dict of nn.Embedding(n_cats, d_model)
        # PAY embedders: depends on strategy
        # Feature-type embeddings: nn.Embedding(n_features, d_model)
        # [CLS] token: nn.Parameter
```

---

## 6. Phase 4: Transformer Architecture

**Status: [DONE] COMPLETE** — every sub-item of Plan §6 is implemented.
Attention (`src/attention.py`, PR #7; extended with the `attn_bias` hook
in PR #8) + encoder stack (`src/transformer.py`: `FeedForward`,
`TransformerBlock` with independently-ablatable
`attn_dropout`/`ffn_dropout`/`residual_dropout`, `TemporalDecayBias` =
Novelty N3, **`FeatureGroupBias` = Novelty N2** — 5×5 learnable bias
matrix over the {CLS, demographic, PAY, BILL_AMT, PAY_AMT} groups with
scalar / per_head / off modes, `TransformerEncoder` composing both
novelty biases via elementwise sum into a single per-forward
`attn_bias`) + top-level `TabularTransformer` wrapper (`src/model.py`,
§6.7 / §6.10 / §6.11) are all landed. Parameter count at plan defaults:
**28,417** — on target for the ~28K Plan §6.9 budget. Sophisticated
helpers on `TabularTransformer`: `summary()`, `parameter_count_by_module()`,
`get_head_params()`, `get_encoder_params()` (for the §8.5.5 two-stage
optimiser), `load_pretrained_encoder()`.

### 6.1 Scaled Dot-Product Attention

The fundamental operation. Given a sequence of n tokens, each of dimension d:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

where:
- $Q = XW_Q \in \mathbb{R}^{n \times d_k}$ — queries ("what am I looking for?")
- $K = XW_K \in \mathbb{R}^{n \times d_k}$ — keys ("what do I contain?")
- $V = XW_V \in \mathbb{R}^{n \times d_v}$ — values ("what information do I provide?")
- $W_Q, W_K \in \mathbb{R}^{d \times d_k}$, $W_V \in \mathbb{R}^{d \times d_v}$ are learnable projection matrices

**Why divide by $\sqrt{d_k}$?**

If the entries of Q and K are independent random variables with zero mean and unit variance, then each dot product $q_i \cdot k_j = \sum_{l=1}^{d_k} q_{il} k_{jl}$ is a sum of $d_k$ independent products, each with mean 0 and variance 1. By the CLT, the dot product has mean 0 and variance $d_k$. When $d_k$ is large, the dot products become large in magnitude, pushing the softmax into saturated regions where gradients vanish. Dividing by $\sqrt{d_k}$ normalises the variance back to 1, keeping softmax in its gradient-friendly regime.

**Attention weights matrix**: $A = \text{softmax}(QK^\top / \sqrt{d_k}) \in \mathbb{R}^{n \times n}$. Entry $A_{ij}$ is the attention weight from token $i$ to token $j$. Row $i$ sums to 1 (it's a probability distribution over all tokens). **We store and return this matrix for interpretability analysis.**

### 6.2 Multi-Head Attention

Instead of one attention function, we run $h$ parallel attention heads, each with reduced dimensionality $d_k = d_v = d/h$:

$$\text{head}_i = \text{Attention}(XW_{Q_i}, XW_{K_i}, XW_{V_i})$$

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O$$

where $W_O \in \mathbb{R}^{(h \cdot d_v) \times d}$ projects the concatenated heads back to dimension $d$.

**Justification**: Different heads can learn to attend to different types of feature interactions. One head might specialise in temporal payment patterns, another in credit utilisation interactions, another in demographic-delinquency correlations. This is testable — we will visualise per-head attention patterns.

**Hyperparameter choices**:
- $d = 32$ (model dimension — small because we only have 24 tokens and ~21K training samples)
- $h = 4$ (attention heads — each head has $d_k = d_v = 8$)
- Ablation over $h \in \{1, 2, 4, 8\}$

### 6.3 Attention Dropout

Apply dropout to the attention weights matrix before multiplying with V:

$$\text{Attention}(Q, K, V) = \text{Dropout}\left(\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)\right) V$$

This prevents the model from relying too heavily on any single feature-to-feature attention pattern. Rate: 0.1–0.2.

### 6.4 Transformer Block (PreNorm)

Each block consists of two sub-layers with residual connections:

```
Sub-layer 1 (Attention):
    x_norm = LayerNorm(x)
    attn_out, attn_weights = MultiHeadAttention(x_norm)
    x = x + Dropout(attn_out)

Sub-layer 2 (FFN):
    x_norm = LayerNorm(x)
    ffn_out = FFN(x_norm)
    x = x + Dropout(ffn_out)
```

**PreNorm vs PostNorm**:

PostNorm (original Vaswani 2017): `x = LayerNorm(x + SubLayer(x))`
PreNorm (GPT-2, FT-Transformer): `x = x + SubLayer(LayerNorm(x))`

We use **PreNorm** because:
1. The residual path is "clean" — gradients flow directly through the addition without passing through LayerNorm, improving training stability.
2. The FT-Transformer paper demonstrated PreNorm is superior for tabular transformers.
3. PreNorm doesn't require learning rate warmup as critically as PostNorm.

**LayerNorm** (Ba et al., 2016): Normalises across the feature dimension (not the batch dimension like BatchNorm). For each token independently:

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sigma + \epsilon} + \beta$$

where $\mu, \sigma$ are computed across the d dimensions, and $\gamma, \beta$ are learnable scale/shift parameters. Initialised to $\gamma = 1, \beta = 0$.

**FT-Transformer detail**: Gorishniy et al. (2021) found that removing the first LayerNorm in the first block's attention sub-layer (i.e., not normalising the freshly created embeddings before the first attention) improved training. We will implement this and test as an ablation.

### 6.5 Position-wise Feed-Forward Network (FFN)

Applied to each token independently:

$$\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$$

where:
- $W_1 \in \mathbb{R}^{d \times d_{ff}}$, $b_1 \in \mathbb{R}^{d_{ff}}$ — expand to inner dimension
- $W_2 \in \mathbb{R}^{d_{ff} \times d}$, $b_2 \in \mathbb{R}^d$ — project back
- $d_{ff} = 4d$ (standard transformer practice — e.g. $d_{ff} = 128$ when $d = 32$)

With dropout between the two layers:

$$\text{FFN}(x) = W_2 \cdot \text{Dropout}(\text{GELU}(W_1 x + b_1)) + b_2$$

**Why GELU over ReLU?**

$$\text{GELU}(x) = x \cdot \Phi(x)$$

where $\Phi$ is the standard Gaussian CDF. GELU is smoother than ReLU (no hard zero cutoff), which means smoother gradients and easier optimisation. It has become the standard activation in transformers since BERT (Hendrycks & Gimpel, 2016). We will ablate GELU vs ReLU.

**Initialisation**:
- $W_1$: Kaiming normal (He initialisation) — appropriate for GELU activation
- $W_2$: Xavier normal — output feeds into residual connection
- Biases: zeros

### 6.6 Stacking Transformer Blocks

We stack $L$ transformer blocks sequentially. Each block refines the token representations:

$$X^{(l)} = \text{TransformerBlock}^{(l)}(X^{(l-1)})$$

**Hyperparameter**: $L \in \{1, 2, 3, 4\}$, with 2–3 expected to be optimal for this dataset size. More layers increase capacity but also increase overfitting risk on 21,000 training samples.

**Ablation**: Vary $L$ and report validation performance to find the sweet spot.

### 6.7 Classification Head

After the final transformer block, extract the [CLS] token's representation and pass through a classification head:

```
cls_output = X^{(L)}[:, 0, :]          # shape (B, d)
cls_output = LayerNorm(cls_output)       # final normalisation
logit = Linear(d, 1)(GELU(Linear(d, d)(cls_output)))  # 2-layer MLP
probability = sigmoid(logit)
```

Dropout (0.1–0.2) applied between the two linear layers.

**Output**: a single probability $p \in (0, 1)$ representing $P(\text{default} = 1 | x)$.

### 6.8 Weight Initialisation Summary

| Component | Initialisation | Reference |
|---|---|---|
| $W_Q, W_K, W_V, W_O$ | Xavier normal | Glorot & Bengio (2010) |
| FFN $W_1$ | Kaiming normal | He et al. (2015) |
| FFN $W_2$ | Xavier normal | Glorot & Bengio (2010) |
| Embedding tables | $\mathcal{N}(0, 0.02)$ | Devlin et al. (2019) |
| [CLS] token | $\mathcal{N}(0, 0.02)$ | Devlin et al. (2019) |
| LayerNorm $\gamma$ | 1 | Standard |
| LayerNorm $\beta$ | 0 | Standard |
| All biases | 0 | Standard |

### 6.9 Full Model Architecture Summary

```
Input: (B, 23) raw features
    ↓
FeatureTokeniser → (B, 24, d)     [23 feature tokens + 1 CLS]
    ↓
TransformerBlock × L → (B, 24, d)  [self-attention + FFN, with residuals + LayerNorm]
    ↓
Extract CLS → (B, d)
    ↓
ClassificationHead → (B, 1)
    ↓
Sigmoid → P(default)
```

**Total parameter count estimate** (d=32, h=4, d_ff=128, L=2, 23 features):
- Feature embeddings (cat + PAY state + num-feature identity + value projections): ~1,500
- Feature-type / positional embeddings: 24 × 32 = 768
- CLS token: 32
- Per block: attention (W_Q + W_K + W_V + W_O ≈ 4 × 32² = 4,096) + FFN (32×128 + 128×32 = 8,192) + LayerNorms (≈130) ≈ 12,400
- 2 blocks: ~24,800
- Classification head: 32 × 32 + 32 × 1 ≈ 1,060
- LayerNorm (final): ~64
- **Total: ~28,000 parameters**

This is a genuinely tiny model — roughly one parameter per 0.75 training examples — appropriate for 21,000 training samples and designed to resist overfitting.

### 6.10 Modules

```
src/attention.py        — ScaledDotProductAttention, MultiHeadAttention
src/transformer.py      — TransformerBlock, TransformerEncoder
src/model.py            — TabularTransformer (full model: tokeniser + encoder + head)
```

### 6.11 Implementation Plan: `src/transformer.py`

#### What it does

Takes the output of `attention.py` and wraps it into complete transformer blocks with normalisation, feed-forward networks, and residual connections. Then stacks multiple blocks.

#### Two classes

**Class 1: `TransformerBlock`** — one block of the transformer

```
Input x (B, 24, 32)

Sub-layer 1 (Attention):
    x_norm = LayerNorm(x)
    attn_out, attn_weights = MultiHeadAttention(x_norm)
    x = x + Dropout(attn_out)              ← residual connection

Sub-layer 2 (FFN):
    x_norm = LayerNorm(x)
    hidden = Linear(32 → 128)(x_norm)      ← expand
    hidden = GELU(hidden)                   ← activation
    hidden = Dropout(hidden)
    ffn_out = Linear(128 → 32)(hidden)      ← compress back
    x = x + Dropout(ffn_out)               ← residual connection

Output x (B, 24, 32), attn_weights (B, 4, 24, 24)
```

**Components:**
- `nn.LayerNorm(d_model)` × 2 — one before attention, one before FFN
- `MultiHeadAttention` from `attention.py`
- FFN: `nn.Linear(32, 128)` → `nn.GELU()` → `nn.Dropout` → `nn.Linear(128, 32)` → `nn.Dropout`
- `d_ff = 4 × d_model = 128`

PreNorm (not PostNorm) — LayerNorm goes before each sub-layer, keeping the residual path clean for gradient flow.

**Class 2: `TransformerEncoder`** — stacks multiple blocks

```
Input (B, 24, 32)
    → TransformerBlock 1 → attn_weights_1
    → TransformerBlock 2 → attn_weights_2
Output (B, 24, 32), [attn_weights_1, attn_weights_2]
```

- `n_layers = 2` (default, ablate {1, 2, 3, 4})
- Collects attention weights from every layer into a list — needed for attention rollout in Phase 10
- Each block has its own independent parameters (not shared)

#### Weight initialisation

| Component | Init |
|---|---|
| FFN $W_1$ (expand) | Kaiming normal (for GELU) |
| FFN $W_2$ (compress) | Xavier normal |
| FFN biases | Zeros |
| LayerNorm $\gamma$ | 1 (default) |
| LayerNorm $\beta$ | 0 (default) |
| Attention weights | Already initialised in `attention.py` |

#### Parameters (default config)

| Parameter | Value |
|---|---|
| `d_model` | 32 |
| `n_heads` | 4 |
| `d_ff` | 128 (4 × `d_model`) |
| `n_layers` | 2 |
| `dropout` | 0.1 |

All configurable via constructor for ablations.

#### Interface

- **Input from:** `embedding.py` → `(B, 24, d_model)`
- **Uses:** `MultiHeadAttention` from `attention.py`
- **Used by:** `model.py` (not yet built)

#### Verification

```
python src/transformer.py
```

Smoke test checks:
- Output shape `(B, 24, 32)` matches input shape
- Returns list of 2 attention weight tensors
- Gradients flow through all parameters
- Residual connection works (output ≠ 0 even with untrained weights)

### 6.12 Architectural Variants to Explore (independent contributions)

Beyond the standard FT-Transformer block (§6.4–6.7), we implement three **novel inductive biases tailored to this dataset** and ablate each against the standard block. These are items N2 and N3 in the Novelty Register (§1.6).

#### 6.12.1 Feature-Group Attention Bias (N2)

The 23 feature tokens naturally form four groups: demographic (positions 1–5), PAY (6–11), BILL_AMT (12–17), PAY_AMT (18–23). We add a learnable **group-bias matrix** $B \in \mathbb{R}^{4 \times 4}$ to the pre-softmax attention scores:

$$\text{scores}_{ij} = \frac{Q_i K_j^\top}{\sqrt{d_k}} + B[g(i), g(j)]$$

where $g(i)$ is the group index of token $i$ (the [CLS] token gets its own group index 0, with $B$ being $5 \times 5$).

**Initialisation**: $B = 0$ (bias off at start; the model chooses whether to learn group structure). Weight decay excluded from $B$.

**Hypothesis (tested by Ablation A21)**: the model benefits from a soft prior that within-group attention is easier than cross-group, because e.g. the six PAY tokens share semantics the twelve amount tokens do not.

#### 6.12.2 Temporal-Decay Positional Prior (N3)

For the three temporal groups, adjacent months carry more mutual information than distant months (EDA Fig 9: BILL_AMT autocorrelation $\geq$ 0.95 at lag 1, $\approx$ 0.7 at lag 5). Credit-risk intuition: recent delinquency matters much more than six-month-old delinquency for predicting next month's default.

We add a learned scalar decay $\alpha \geq 0$ to attention scores *within* each temporal group, penalising distance along the temporal axis:

$$\text{scores}_{ij} \mathrel{+}= -\alpha \cdot |t(i) - t(j)| \quad \text{if } g(i) = g(j) \in \{\text{PAY, BILL, PAY\_AMT}\}$$

where $t(i) \in \{0, 1, 2, 3, 4, 5\}$ is the month index. A single learnable scalar $\alpha$ is sufficient (one per head is an additional variant to ablate).

**Hypothesis (tested by Ablation A22)**: an explicit recency prior helps the model converge faster and/or generalise better than learning this relationship from scratch via feature-type embeddings alone.

#### 6.12.3 Attention-Dropout-as-Regularisation (standard, tracked here for completeness)

Already covered in §6.3; the rate is treated as a hyperparameter (ablation A12).

#### 6.12.4 Stochastic Depth (DropPath) — optional, not primary

With probability $p_{\text{drop-block}}$ a whole block's residual sub-layer is skipped during training. Huang et al. (2016). We implement as an optional switch; expected gain small at $L=2$.

---

## 7. Phase 5: Loss Functions & Class Imbalance

**Status: [DONE] COMPLETE** — `src/losses.py` provides
`WeightedBCELoss`, `FocalLoss` (γ and α configurable per Ablation A11;
`"balanced"` α fitted from the first training batch with the conventional
caveat) and `LabelSmoothingBCELoss` (ε=0.05 default per §7.3). All three
operate on logits and reduce to `binary_cross_entropy_with_logits` in the
limit (γ=0, ε=0) — verified by the test suite.

### 7.1 Weighted Binary Cross-Entropy (Baseline)

$$\mathcal{L}_{\text{WBCE}} = -\frac{1}{N} \sum_{i=1}^N \left[ \alpha \cdot y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$

where $\alpha = N_0 / N_1 \approx 3.52$ (inverse class frequency ratio).

### 7.2 Focal Loss (Primary)

$$\mathcal{L}_{\text{FL}} = -\frac{1}{N} \sum_{i=1}^N \alpha_t (1 - p_t)^\gamma \log(p_t)$$

where:
- $p_t = p_i$ if $y_i = 1$, else $p_t = 1 - p_i$
- $\alpha_t = \alpha$ if $y_i = 1$, else $\alpha_t = 1 - \alpha$ (class-balanced)
- $\gamma \geq 0$ is the focusing parameter

**Why focal loss?**
1. $(1 - p_t)^\gamma$ is the modulating factor: when the model correctly classifies an example with high confidence ($p_t \to 1$), the factor approaches 0, down-weighting the loss. The model focuses on hard, ambiguous examples.
2. $\gamma = 0$ recovers standard cross-entropy. $\gamma = 2$ is the standard starting point (Lin et al., 2017).
3. Mukhoti et al. (2020) showed focal loss improves **model calibration** — predicted probabilities better match actual default rates. In credit risk, calibration is arguably more important than raw accuracy.

**Hyperparameters to tune**: $\gamma \in \{0, 0.5, 1, 2, 3\}$, $\alpha \in \{0.25, 0.5, 0.75, \text{class-balanced}\}$

### 7.3 Label Smoothing (Optional Enhancement)

Replace hard targets $y \in \{0, 1\}$ with softened targets:

$$y_{\text{smooth}} = y \cdot (1 - \epsilon) + 0.5 \cdot \epsilon$$

with $\epsilon = 0.05$. This prevents the model from becoming overconfident and has a regularising effect.

### 7.4 Module

```
src/losses.py           — FocalLoss, WeightedBCE, label smoothing utilities
```

---

## 8. Phase 6: Training Pipeline

**Status: [DONE] COMPLETE** — training infrastructure + supervised loop
all landed: `src/utils.py` (deterministic seeding per §16.5.1, device
selection, hardened checkpoint save/load with a weights-only default that
closes SECURITY_AUDIT C-1, `EarlyStopping`, parameter accounting,
UTF-8-safe logging), `src/dataset.py` (`StratifiedBatchSampler`,
`make_loader` with supervised / val / test / MTLM modes, reproducible via
a seeded generator), and **`src/train.py`** (~550 LOC) implementing the
full Plan §8 spec: AdamW + linear-warmup-plus-cosine LR schedule + grad
clipping + per-epoch CSV log + `EarlyStopping` on validation AUC-ROC +
best-weight restore + hardened checkpoint save. Two-stage LR for MTLM
fine-tuning (§8.5.5) and multi-task PAY_0 auxiliary objective
(§8.6 / N5 / A16) are wired through the CLI. Every ablation axis is
reachable via argparse flags (A2, A3, A4, A5, A7, A10, A11, A12, A16,
A19, A21, A22). Colab + VS Code Colab extension + local Jupyter all
drive the loop through `notebooks/04_train_transformer.ipynb`.

### 8.1 Optimiser: AdamW

$$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

**AdamW** (Loshchilov & Hutter, 2019) decouples weight decay from the adaptive gradient update, which is critical for transformer training. Standard Adam applies L2 regularisation inside the adaptive step, which doesn't properly regularise when gradients are small.

**Hyperparameters**:
- Learning rate $\eta$: initial value ~1e-4 to 3e-4
- Weight decay $\lambda$: ~1e-5 to 1e-4
- $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$ (standard defaults)

### 8.2 Learning Rate Schedule

**Cosine annealing with linear warmup**:

1. **Warmup** (first 5–10% of training): linearly increase LR from 0 to $\eta_{\text{max}}$
2. **Cosine decay**: smoothly decrease LR from $\eta_{\text{max}}$ to $\eta_{\text{min}} = \eta_{\text{max}} / 100$

$$\eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}})\left(1 + \cos\left(\frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}} \pi\right)\right)$$

**Justification**: Warmup prevents early divergence when the randomly initialised attention weights produce noisy gradients. Cosine decay is smoother than step decay and avoids the need to choose step timing.

### 8.3 Gradient Clipping

$$\|\nabla \theta\| > \text{max\_norm} \implies \nabla \theta \leftarrow \frac{\text{max\_norm}}{\|\nabla \theta\|} \nabla \theta$$

with `max_norm = 1.0`. Prevents exploding gradients, which can occur in attention when attention weights become very sharp.

### 8.4 Regularisation Strategy

| Technique | Location | Rate | Justification |
|---|---|---|---|
| Dropout | After attention weights | 0.1 | Prevents attention from fixating on single features |
| Dropout | Inside FFN (after GELU) | 0.1–0.2 | Standard regularisation |
| Dropout | Classification head | 0.1–0.2 | Prevents overfit on final layer |
| Weight decay | All parameters (via AdamW) | 1e-5 to 1e-4 | L2 regularisation on weights |
| Early stopping | Validation AUC-ROC | Patience: 15–20 epochs | Stop when generalisation plateaus |

### 8.5 Early Stopping

Monitor validation AUC-ROC after each epoch. If no improvement for `patience` consecutive epochs, restore the best model weights and stop training.

**Why AUC-ROC over loss?** Validation loss can fluctuate due to stochastic batch effects, while AUC-ROC is a more stable measure of discrimination ability and is threshold-independent.

### 8.6 Training Configuration

| Parameter | Value | Notes |
|---|---|---|
| Batch size | 256 | Fits in memory; large enough for stable gradients |
| Max epochs | 200 | Early stopping will typically trigger much sooner |
| LR warmup | 10 epochs | ~5% of max epochs |
| Early stopping patience | 20 epochs | |
| Gradient clipping | max_norm = 1.0 | |
| Random seed | 42 | For reproducibility |
| Number of runs | 5 | Report mean ± std for statistical significance |

### 8.7 Hyperparameter Search Strategy

**Method**: Optuna-based Bayesian search OR structured grid search

**Search space**:

| Hyperparameter | Range | Type |
|---|---|---|
| d_model | {16, 32, 64} | Categorical (default: 32) |
| n_heads | {1, 2, 4, 8} | Categorical |
| n_layers | {1, 2, 3, 4} | Categorical |
| d_ff_multiplier | {2, 4} | Categorical |
| dropout | [0.05, 0.3] | Continuous |
| learning_rate | [1e-5, 1e-3] | Log-uniform |
| weight_decay | [1e-6, 1e-3] | Log-uniform |
| focal_gamma | {0, 1, 2, 3} | Categorical |
| focal_alpha | {0.25, 0.5, 0.75} | Categorical |
| batch_size | {128, 256, 512} | Categorical |

**Budget**: 50–100 trials. Validate on validation set. Final evaluation on held-out test set only once.

### 8.8 Data Loading

Use PyTorch `Dataset` and `DataLoader` with:
- Custom collation function that separates numerical, categorical, and PAY features
- Stratified batch sampling (optional — ensures each batch has ~22% default)
- `pin_memory=True` for GPU training
- `num_workers=4` for parallel data loading

### 8.9 Logging

Log per epoch:
- Training loss, validation loss
- Validation AUC-ROC, AUC-PR, F1
- Learning rate
- Gradient norm
- Wall-clock time per epoch

Plot training curves after completion.

### 8.10 Modules

```
src/dataset.py          — CreditDefaultDataset (PyTorch Dataset)
src/train.py            — Training loop, LR schedule, early stopping, logging
src/losses.py           — FocalLoss, WeightedBCE
src/utils.py            — Seed setting, device handling, checkpointing
```

---

## 8.5 Phase 6A: Self-Supervised Masked Tabular Language Modelling (NOVEL — N4)

**Status: [DONE] COMPLETE** — all three MTLM components landed:

* **`src/mtlm.py`** (~420 LOC) — `MTLMHead` with per-feature prediction
  heads (3 categorical CE heads + 6 PAY CE heads + 14 numerical MSE heads,
  drift-safe token slicing from `TOKEN_ORDER`), entropy-normalised CE
  + variance-normalised MSE composite loss (`mtlm_loss`), and the
  `MTLMModel` wrapper whose state-dict prefixes (`embedding.*`,
  `encoder.*`) are drop-in for downstream supervised fine-tuning.
* **`src/train_mtlm.py`** (~440 LOC) — the pretraining loop. Shares the
  cosine-warmup LR schedule + `EarlyStopping` + checkpoint code paths
  with `src/train.py`. Produces a tiny ~130 KB `encoder_pretrained.pt`
  artefact consumed by `train.py --pretrained-encoder PATH`.
* **`model.TabularTransformer.load_pretrained_encoder`** generalised to
  accept either a full checkpoint bundle (via `utils.load_checkpoint`,
  weights-only-safe) or a raw state-dict file — the MTLM handoff
  artefact.

Exercised end-to-end: `results/mtlm/run_42/` contains the full
pretraining artefact set; validation reconstruction loss dropped from
3.81 → 1.46 in 12 epochs (early-stopped at epoch 22 of the 50-epoch
cap). Fine-tuning from the pretrained encoder is then handled by
`src/train.py --pretrained-encoder …` with the §8.5.5 two-stage LR
(`encoder_lr_ratio = 0.2`) — see `results/transformer/seed_42_mtlm_finetune/`
for the fine-tuned supervised model.

This is the single most "language-model-like" component of the project. The coursework brief explicitly frames the requirement as a "small transformer-based **language model**" — we take that framing seriously. Large language models (BERT, GPT) owe their generalisation to a self-supervised pretraining stage on unlabeled data before fine-tuning on the downstream task. We adapt this paradigm to structured credit data via **Masked Tabular Language Modelling** (MTLM).

### 8.5.1 Motivation

Standard supervised training uses 21,000 labelled examples (train split). MTLM pretraining lets us use the same 21,000 rows many times over as a richer signal: instead of predicting one label per row, we predict up to 23 masked feature values per row. This provides a representation-learning objective that is *independent of the default label*, forcing the transformer to learn feature dependencies — exactly the relationships self-attention is meant to capture.

Rubachev et al. (2022) showed pretraining objectives of this form improve downstream tabular performance, particularly in data-scarce regimes. We test this hypothesis directly (Ablation A15).

### 8.5.2 Objective

For each row in the pretraining set (train split only — val and test *never* seen), we mask a random subset of feature tokens and predict their original values from the surviving tokens:

$$\mathcal{L}_{\text{MTLM}} = \sum_{j \in M} \mathcal{L}_j(\hat{y}_j, y_j)$$

where $M$ is the set of masked token positions, and:
- For categorical and PAY tokens: $\mathcal{L}_j$ is **cross-entropy** over the feature's vocabulary.
- For numerical tokens: $\mathcal{L}_j$ is **mean-squared-error** on the scaled value.

**Masking strategy (BERT-style with tabular adaptations):**

| Probability | Action |
|---|---|
| 15% of tokens selected for prediction | base masking rate |
| Of those selected: 80% | replaced with learnable `[MASK]` embedding |
| Of those selected: 10% | replaced with a random valid value from the same feature's training distribution |
| Of those selected: 10% | kept identical (but loss still computed) |

The 10%-random and 10%-keep components mitigate the train/inference mismatch where the model only sees `[MASK]` during training but never at downstream inference time.

**Loss weighting**: numerical MSE and categorical cross-entropy are on different scales. We normalise each feature's loss by the training-set variance (numerical) or entropy (categorical) so no single feature type dominates. Head weights are learnable (GradNorm-style) as an optional enhancement.

### 8.5.3 Architecture Reuse

The transformer encoder (`src/model.py` TabularTransformer) is shared between pretraining and fine-tuning. Only the **prediction head** changes:

- Pretraining: replace the `ClassificationHead` with an `MTLMHead` containing one per-feature prediction head (classification heads for categorical / PAY, regression heads for numerical).
- Fine-tuning: load the pretrained encoder weights, reinstate the `ClassificationHead`, continue training on the supervised task.

This is parameter-efficient: the 28K-parameter encoder is the transferable component.

### 8.5.4 Pretraining Configuration

| Hyperparameter | Value |
|---|---|
| Pretraining epochs | 50–100 (early-stopped on held-out reconstruction loss) |
| Batch size | 256 |
| Optimiser | AdamW with same hyperparameters as supervised training |
| LR schedule | Cosine with 10% warmup |
| LR peak | $3 \times 10^{-4}$ |
| Mask probability | 15% |
| Min tokens masked | 1 per row (so every row contributes to the loss) |
| Max tokens masked | 5 per row (caps the objective's difficulty) |

### 8.5.5 Fine-Tuning Protocol

After pretraining converges:

1. Reinstate the `ClassificationHead` (fresh weights).
2. Continue training with the supervised focal-loss objective.
3. Use a **two-stage learning rate**: peak LR for the classification head, a smaller LR (5× smaller) for the pretrained encoder, to prevent catastrophic forgetting of the pretrained representations.
4. Early-stopping on validation AUC-ROC as in standard supervised training.

### 8.5.6 Module

```
src/mtlm.py             — MTLMHead, MaskingCollator, pretraining loop
src/train_mtlm.py       — Pretraining entry point
```

### 8.5.7 Expected Outcome

We hypothesise (to be tested in Ablation A15) one of three results:
- **H1 (expected)**: pretraining improves downstream AUC by 0.5–1.5 points; converges faster; generalises better to minority class.
- **H2**: pretraining helps calibration more than discrimination (Mukhoti-like effect).
- **H3 (null)**: with 21K rows and the hybrid PAY tokeniser already capturing most structure, pretraining gives little or no lift. **This would itself be an interesting finding worth reporting honestly** — pretraining's value depends on signal-to-noise ratio and data volume (Rubachev et al., 2022).

---

## 8.6 Phase 6B: Multi-Task Auxiliary Objective (NOVEL — N5)

**Status: [TODO] TODO**

During supervised fine-tuning (not pretraining), we optionally add an **auxiliary PAY_0-forecast head** that predicts the most recent repayment status (PAY_0) from the remaining 22 feature tokens, jointly with the default prediction:

$$\mathcal{L}_{\text{joint}} = \mathcal{L}_{\text{focal}}(\hat{y}^{\text{def}}, y^{\text{def}}) + \lambda \cdot \mathcal{L}_{\text{CE}}(\hat{y}^{\text{PAY\_0}}, y^{\text{PAY\_0}})$$

with $\lambda \in \{0.1, 0.3, 0.5\}$ ablated.

**Implementation**: at fine-tuning time, mask the PAY_0 token (using the same `[MASK]` embedding from MTLM) and read the auxiliary prediction from the [CLS] representation via a parallel 11-class classification head. This is structurally identical to MTLM with a forced mask location, but the objective is weighted jointly with the default loss.

**Rationale**: PAY_0 is the single strongest feature–target correlate in the data (|r_pb| ≈ 0.32). Forcing the model to predict it from context (i.e., the other 22 features) is a strong regularisation signal that keeps the representation grounded in credit dynamics rather than memorising spurious patterns.

**Test**: Ablation A16 compares single-task (default only) vs multi-task (default + PAY_0) validation AUC-ROC.

---

## 9. Phase 7: Random Forest Benchmark

**Status: [DONE code / TODO results] IMPLEMENTED, NOT YET EXECUTED** —
`src/random_forest.py` is 870 LOC covering baseline RF, 60-iter
`RandomizedSearchCV`, dual feature importance (Gini MDI + permutation),
5-fold stratified CV, threshold optimisation, and five publication figures.
Running `poetry run python run_pipeline.py --rf-benchmark --source local`
produces `results/rf_*.{csv,json}` and `figures/rf_*.png`. Hyperparameter
grid is a deliberate subset of §9.3 (n_iter=60 vs 200) for time-budget
reasons; the subset is defensible and can be widened for the final runs.

### 9.1 Purpose

The random forest serves as a strong baseline. Tree-based models are known to outperform deep learning on medium-sized tabular data (Grinsztajn et al., 2022). If the transformer underperforms RF, that is an interesting and honest finding — the markers are testing critical thinking, not transformer supremacy.

### 9.2 Feature Preparation

- Use the engineered feature set (all 23 raw features + derived features)
- No standardisation needed (trees are scale-invariant)
- Categorical features can be integer-encoded (trees handle this naturally)
- Same train/val/test split as the transformer

### 9.3 Hyperparameter Tuning

**Method**: `RandomizedSearchCV` with 5-fold stratified cross-validation

**Search space**:

| Hyperparameter | Range |
|---|---|
| n_estimators | [100, 200, 500, 1000, 2000] |
| max_depth | [5, 10, 15, 20, 30, None] |
| min_samples_split | [2, 5, 10, 20] |
| min_samples_leaf | [1, 2, 4, 8] |
| max_features | ["sqrt", "log2", 0.3, 0.5, 0.7] |
| class_weight | ["balanced", "balanced_subsample", None] |
| criterion | ["gini", "entropy"] |

**Budget**: 200 random combinations × 5 folds = 1,000 fits.

### 9.4 Feature Importance Analysis

1. **Gini importance** (MDI): built-in, measures the total decrease in impurity from splits on each feature. Biased toward high-cardinality features.
2. **Permutation importance**: shuffle each feature and measure the drop in AUC-ROC. Unbiased but slower.
3. **SHAP values** (Lundberg & Lee, 2017): game-theoretic feature attributions. Provides per-sample explanations, interaction effects, and summary plots. Compare against attention-based feature importance from the transformer.

### 9.5 OOB Error Analysis

Random forests provide an Out-of-Bag (OOB) error estimate "for free" — each tree is trained on a bootstrap sample, and the OOB samples provide an unbiased estimate of generalisation error. Report OOB score alongside cross-validation score.

### 9.6 Module

```
src/random_forest.py    — RF training, tuning, feature importance, SHAP analysis
```

---

## 10. Phase 8: Evaluation & Metrics

**Status: [DONE] — `src/evaluate.py` + `src/visualise.py` + `src/rf_predictions.py`. Comparison table at `results/comparison_table.{csv,md}`, five figures in `figures/`. Ensemble row + full RF calibration metrics landed on `feature/phase-11-12-14`.**

### 10.1 Comprehensive Metrics Suite

Both models evaluated on the **held-out test set** with the following metrics:

| Metric | Formula / Description | Why |
|---|---|---|
| Accuracy | $(TP + TN) / N$ | Overall correctness (but misleading with imbalance) |
| Precision | $TP / (TP + FP)$ | Of predicted defaults, how many are real? |
| Recall (Sensitivity) | $TP / (TP + FN)$ | Of actual defaults, how many were caught? |
| Specificity | $TN / (TN + FP)$ | Of non-defaults, how many were correctly identified? |
| F1-Score | $2 \cdot \text{Prec} \cdot \text{Rec} / (\text{Prec} + \text{Rec})$ | Harmonic mean of precision and recall |
| AUC-ROC | Area under the ROC curve | Threshold-independent discrimination ability |
| AUC-PR | Area under the Precision-Recall curve | More informative than AUC-ROC under class imbalance |
| ECE | Expected Calibration Error | How well predicted probabilities match actual rates |
| Brier Score | $\frac{1}{N}\sum(p_i - y_i)^2$ | Proper scoring rule combining calibration and discrimination |
| Cohen's Kappa | Agreement correcting for chance | Accounts for class imbalance in accuracy |

### 10.2 Visualisations

1. **Confusion matrices** — side by side for transformer and RF, with annotations
2. **ROC curves** — both models on same plot, with AUC annotations and diagonal reference
3. **Precision-Recall curves** — both models, with no-skill baseline (horizontal line at 0.22)
4. **Reliability diagrams** — calibration curves showing predicted vs actual probability
5. **Training curves** — loss and AUC-ROC over epochs for the transformer
6. **Learning rate schedule** — plotted over training

### 10.3 Threshold Optimisation

The default 0.5 threshold is arbitrary. We optimise the decision threshold on the validation set:

1. **Maximise F1**: find $\tau^* = \arg\max_\tau F1(\tau)$
2. **Maximise Youden's J**: $\tau^* = \arg\max_\tau (\text{Sensitivity}(\tau) + \text{Specificity}(\tau) - 1)$
3. **Business-context optimisation**: weight FP and FN costs differently. In credit risk, a missed default (FN) is typically more costly than a false alarm (FP). If cost(FN) = 5 × cost(FP), optimise accordingly.

Report metrics at both the default 0.5 threshold and the optimised threshold.

### 10.4 Model Comparison Table

```
| Metric          | Transformer (0.5) | Transformer (opt.) | RF (0.5) | RF (opt.) |
|-----------------|--------------------|--------------------|----------|-----------|
| Accuracy        | ...                | ...                | ...      | ...       |
| Precision       | ...                | ...                | ...      | ...       |
| Recall          | ...                | ...                | ...      | ...       |
| F1              | ...                | ...                | ...      | ...       |
| AUC-ROC         | ...                | ...                | ...      | ...       |
| AUC-PR          | ...                | ...                | ...      | ...       |
| ECE             | ...                | ...                | ...      | ...       |
| Brier Score     | ...                | ...                | ...      | ...       |
```

### 10.5 Module

```
src/evaluate.py         — Metric computation, threshold optimisation, comparison tables
src/visualise.py        — All evaluation plots (ROC, PR, confusion, calibration, training curves)
```

### 10.6 Additional Advanced Evaluation Components

Beyond the standard metric suite, we also compute and report:

- **Bootstrap 95% confidence intervals** for every point estimate. Resample the test set with replacement 1,000 times, recompute each metric, report 2.5th and 97.5th percentiles. Without CIs, point estimates on a 4,500-row test set are not trustworthy.
- **Paired bootstrap for model differences**: on each bootstrap resample, compute the *difference* $\Delta = \text{metric}_{\text{transformer}} - \text{metric}_{\text{RF}}$. The 95% CI of $\Delta$ is the proper test for "does the transformer beat the RF?". Unpaired CIs on the two individual metrics are not.
- **McNemar's and DeLong's tests** for binary-prediction agreement and AUC comparison (see Phase 12).
- **Threshold optimisation under three objectives**: $F_1$, Youden's $J$, and the cost-weighted objective from Phase 8A (ECL minimisation).
- **Model decision-surface visualisation** on a 2D projection of the test set (PCA of engineered features) — shows how transformer vs RF partition the space.

---

## 10.7 Phase 8A: Business / Cost-Sensitive Evaluation (NOVEL — N9)

**Status: [TODO] TODO**

Credit default is not a symmetric classification problem. In the Taiwan dataset's economic context, a **missed default** (false negative — we lend to someone who then defaults) costs the bank the entire unpaid balance *plus* collection costs, while a **false alarm** (false positive — we deny credit to someone who would have paid) costs only the foregone interest margin.

Standard AUC-ROC ignores this asymmetry. Marking-wise, the coursework PDF explicitly prompts "is the transformer-based approach the right model for this problem?" — a proper answer needs a business-aware metric, not just accuracy.

### 10.7.1 Cost Matrix

We define a per-decision cost matrix aligned with the credit-risk literature:

| | Predicted: No default | Predicted: Default |
|---|---|---|
| **Actual: No default** (77.9%) | $c_{\text{TN}} = 0$ (normal customer, we earn interest) | $c_{\text{FP}} = \lambda \cdot E[B_i]$ (foregone interest margin on bill $B_i$) |
| **Actual: Default** (22.1%) | $c_{\text{FN}} = \text{LGD} \cdot E[B_i]$ (loss-given-default times exposure) | $c_{\text{TP}} = c_{\text{collect}}$ (small collection / review cost) |

Using industry-standard Basel III parameters as a first-order approximation:
- **LGD** (Loss-Given-Default) = **0.45** — the portion of exposure unrecoverable after default (Basel III foundation-IRB default for unsecured retail).
- **Interest margin** $\lambda$ = **0.05** — 5% net margin on a carried balance is typical for credit cards.
- **Collection cost** $c_{\text{collect}}$ = **fixed 500 NT$** — rough average of soft-collection operations.

The **expected test-set cost** is:

$$C_{\text{total}} = \sum_{i \in D_{\text{test}}} \sum_{j \in \{\text{TP, FP, FN, TN}\}} \mathbb{1}[\text{outcome}_i = j] \cdot c_j(B_i)$$

### 10.7.2 Expected Credit Loss (ECL)

Under IFRS 9 / Basel III, banks hold capital against **Expected Credit Loss** per customer:

$$\text{ECL}_i = P(\text{default}_i) \cdot \text{LGD} \cdot \text{EAD}_i$$

where $\text{EAD}_i$ (Exposure-at-Default) is approximated by BILL_AMT1 (most recent bill) for this dataset.

We compute **portfolio-level ECL** under each model's predictions and compare against the realised default losses. A well-calibrated model produces ECL estimates close to realised losses; a miscalibrated one systematically under- or over-reserves.

### 10.7.3 Cost-Optimal Threshold

The cost-minimising threshold is *not* 0.5:

$$\tau^* = \arg\min_\tau \; C_{\text{total}}(\tau)$$

Given the LGD/margin asymmetry above, $\tau^* \approx 0.1$–$0.15$ is expected (classify as default much more aggressively, because missing one default costs as much as ~40 false alarms).

We report:
1. **Total cost at default threshold (0.5)**: naïve deployment.
2. **Total cost at F1-optimal threshold**: classical ML optimisation.
3. **Total cost at cost-optimal threshold**: business-aware optimisation.
4. **Savings ratio**: $(C_{0.5} - C_{\tau^*}) / C_{0.5}$ — the business value of threshold optimisation.

### 10.7.4 Portfolio Stress Test

Re-evaluate both models under a **stressed scenario** where the true default rate rises from 22.1% to 30%, simulating a recession. This is achieved by bootstrap-oversampling the defaulter rows of the test set. Report:

- Stressed AUC-ROC (should be stable — AUC is threshold-independent).
- Stressed ECL (will rise — by how much per model?).
- Stressed calibration (how does the model's confidence hold up?).

A model that degrades gracefully under stress is more trustworthy than one that collapses. This is a standard requirement for regulatory model validation (Basel III Pillar 3).

### 10.7.5 Module

```
src/business_eval.py    — Cost matrix, ECL, cost-optimal threshold, stress test
```

---

## 11. Phase 9: Ablation Studies

**Status: [TODO] TODO**

### 11.1 Philosophy

Every design choice in the transformer is a hypothesis. Ablation studies test these hypotheses by removing or changing one component at a time, keeping everything else fixed. This demonstrates understanding, not just implementation.

### 11.2 Ablation Matrix

| Experiment | What Changes | Hypothesis Being Tested |
|---|---|---|
| **A1: PAY tokenisation** | Categorical vs Numerical vs Hybrid | Does the PAY tokenisation strategy matter? Does respecting the categorical/ordinal dual structure improve performance? |
| **A2: Number of layers** | L ∈ {1, 2, 3, 4} | How deep does the model need to be? Is there a point of diminishing returns? |
| **A3: Number of heads** | h ∈ {1, 2, 4, 8} | Do multiple heads capture different interaction types? Is single-head sufficient? |
| **A4: Model dimension** | d ∈ {16, 32, 64} | Is 32 the right capacity? Do smaller/larger models under/overfit? |
| **A5: [CLS] vs pooling** | [CLS] token vs mean pooling vs max pooling | Does the [CLS] aggregation mechanism matter? |
| **A6: Feature-type embeddings** | With vs without | How much does feature identity matter? Can the model work without knowing which feature is which? |
| **A7: Temporal positional encoding** | With vs without | Does explicit temporal ordering of the 6-month features help? |
| **A8: PreNorm vs PostNorm** | PreNorm vs PostNorm | Does normalisation placement affect training stability? |
| **A9: GELU vs ReLU** | Activation function | Does GELU's smoothness help? |
| **A10: Focal loss vs WBCE** | Loss function | Does focal loss improve discrimination and/or calibration? |
| **A11: Focal gamma** | γ ∈ {0, 0.5, 1, 2, 3} | How much should the model focus on hard examples? |
| **A12: Dropout rate** | p ∈ {0, 0.05, 0.1, 0.2, 0.3} | What's the right regularisation strength? |
| **A13: First-layer LayerNorm** | Remove first LN vs keep | FT-Transformer finding: removing first LN improves training |
| **A14: Numerical encoding** | Linear projection vs PLE | Does piecewise linear encoding improve numerical feature representation? |
| **A15: MTLM pretraining** *(NOVEL — N4)* | Supervised-only vs MTLM-pretrained then fine-tuned | Does self-supervised pretraining on this dataset improve AUC / calibration / minority-class recall? Directly tests the "language-model" framing. |
| **A16: Multi-task auxiliary** *(NOVEL — N5)* | $\lambda \in \{0, 0.1, 0.3, 0.5, 1.0\}$ on the PAY_0-forecast auxiliary loss | Does auxiliary supervision improve the primary default objective? What is the optimal trade-off? |
| **A17: Random-attention null baseline** *(NOVEL — N7)* | Learned attention vs frozen-uniform attention weights | Is the transformer learning anything attention-specific, or could a "bag-of-feature-embeddings with FFN" match it? A lower bound on the value of attention on this problem. |
| **A18: Linear-probe floor** *(NOVEL — N7)* | Logistic regression on raw features vs transformer | The floor: if a linear model on 23 raw features matches the transformer, then attention is not justified for this task. Forces intellectual honesty. |
| **A19: Training-data-size scaling curve** | Train on $\{25\%, 50\%, 75\%, 100\%\}$ of the training split | How does the transformer's performance scale with training data? A model that plateaus early has capacity or inductive-bias issues. Produces a clean scaling-law plot for the report. |
| **A20: Post-hoc calibration methods** | Raw vs Platt vs Isotonic vs Temperature scaling | Which calibration method best reduces ECE without harming AUC? Standard-practice comparison. |
| **A21: Feature-group attention bias** *(NOVEL — N2)* | With vs without learned group bias | Does the group-structured inductive bias help or is full attention enough at $L=2$? |
| **A22: Temporal-decay prior** *(NOVEL — N3)* | With vs without temporal decay; scalar vs per-head $\alpha$ | Does an explicit recency prior help, or does the model learn this from positional embeddings alone? |

### 11.3 Ablation Reporting

For each ablation:
- Report validation AUC-ROC, AUC-PR, F1 (mean ± std over 3 runs)
- Highlight the default configuration
- Discuss the implication of each finding

Present as a consolidated table and discuss the most interesting findings in the report text.

---

## 12. Phase 10: Attention Visualisation & Interpretability

**Status: [TODO] TODO**

### 12.1 [CLS]-to-Feature Attention

Extract the attention weights from the [CLS] token (row 0 of the attention matrix) to all 23 feature tokens. This provides a natural feature importance ranking:

$$\text{importance}(j) = \frac{1}{|D_{\text{test}}|} \sum_{i \in D_{\text{test}}} A_{0,j}^{(i)}$$

Averaged over all test set samples. Can be computed per-head and per-layer.

### 12.2 Attention Rollout

Single-layer attention weights can be misleading (Jain & Wallace, 2019). Attention Rollout (Abnar & Zuidema, 2020) computes the effective attention flow across multiple layers by multiplying attention matrices:

$$\hat{A}^{(l)} = 0.5 \cdot A^{(l)} + 0.5 \cdot I$$

(adding identity for the residual connection)

$$\hat{A}_{\text{rollout}} = \hat{A}^{(L)} \cdot \hat{A}^{(L-1)} \cdots \hat{A}^{(1)}$$

Then $\hat{A}_{\text{rollout}}[0, :]$ gives the effective attention flow from [CLS] to each feature through the entire network.

### 12.3 Per-Head Specialisation Analysis

For each attention head in each layer:
1. Compute the average attention pattern over the test set
2. Visualise as a heatmap (24×24 or CLS-row only)
3. Look for specialisation: does one head focus on temporal features? Another on demographics?
4. Quantify specialisation using entropy: a specialised head has low-entropy attention (concentrated on a few features), while a diffuse head has high entropy

### 12.4 Attention Patterns: Defaulters vs Non-Defaulters

Compare attention patterns between the two classes:
- Average [CLS]-to-feature attention for defaulters vs non-defaulters
- Identify features that receive differential attention
- This can reveal what the model has learned about default risk

### 12.5 Comparison with RF Feature Importance and SHAP

Create a comparison table / bar chart:

| Feature | Transformer Attention (Rollout) | RF Gini Importance | RF Permutation Importance | SHAP |
|---|---|---|---|---|
| PAY_0 | ... | ... | ... | ... |
| LIMIT_BAL | ... | ... | ... | ... |
| ... | | | | |

**Hypothesis**: All methods should agree that PAY features are most important. Disagreements are interesting and worth discussing.

### 12.6 Embedding Space Analysis

Use t-SNE or UMAP to visualise:
1. The learned feature-type embeddings (23 points in d-dimensional space) — do temporal features cluster? Do demographics cluster?
2. The [CLS] output representations for test samples, coloured by default status — are the classes separable in the learned representation?
3. The PAY embeddings — do they reveal the ordinal structure? Is PAY_value=8 far from PAY_value=-2?

### 12.7 Integrated Gradients (IG) — Sundararajan et al. (2017)

Attention weights have been criticised as unreliable explanations (Jain & Wallace, 2019). Gradient-based attribution methods provide an alternative grounded in the model's actual computation.

For each test sample, Integrated Gradients computes the attribution of each input feature by integrating gradients of the output along a straight path from a chosen baseline to the input:

$$\text{IG}_j(x) = (x_j - x_j^{\text{baseline}}) \cdot \int_0^1 \frac{\partial f(x^{\text{baseline}} + \alpha(x - x^{\text{baseline}}))}{\partial x_j} \, d\alpha$$

approximated by Riemann sum with 50 steps.

**Baselines (three variants, report all three)**: (a) zero vector (scaled features → population mean), (b) median of the training set, (c) mean-of-training-set. Baseline choice affects which "direction" the attribution measures from.

**Deliverable**: per-feature IG attribution averaged over the test set, compared against attention rollout, RF Gini importance, RF permutation importance, and SHAP (from Phase 7). Concordance across methods is evidence of a genuine signal; disagreements are interesting and discussed.

### 12.8 SHAP Analysis for the Transformer — Lundberg & Lee (2017)

KernelSHAP applied to the transformer as a black box. Computationally expensive ($O(2^{23})$ in principle, approximated), but produces locally-exact additive feature attributions with theoretical guarantees (efficiency, symmetry, linearity).

**Deliverable**: global SHAP summary (beeswarm plot), top-10 feature ranking, and direct numerical comparison with the RF-SHAP values from Phase 7. If the transformer's SHAP ranking closely matches the RF's, then both models rely on the same features — which is itself diagnostic.

### 12.9 Probing Classifiers

Train a *linear* classifier on top of the transformer's intermediate representations (each layer's output) to predict specific target quantities. Standard NLP practice (Belinkov, 2022).

**Probes to run**:

1. **Does layer $l$ encode PAY semantics?** Probe: linear classifier on layer-$l$'s PAY_0 token → predict raw PAY_0 value (11-class). A high-accuracy probe means that representation preserves the input.
2. **Does [CLS] encode default risk at each layer?** Probe: linear classifier on layer-$l$'s [CLS] → predict default. Rising accuracy with depth indicates the model is progressively refining a default-relevant representation; flat accuracy means layers after the first are not helping.
3. **Does the model encode utilisation implicitly?** Probe: linear classifier on [CLS] → predict BILL_AMT1/LIMIT_BAL (engineered feature, not fed as input). If the probe succeeds, the model has learned the interaction that we engineered by hand.

Probing results inform the report's "what is the transformer actually doing?" narrative.

### 12.10 Cross-Layer Representation Similarity (CKA)

Centred Kernel Alignment (Kornblith et al., 2019) measures similarity between two neural representations. We compute a $(L+1) \times (L+1)$ CKA matrix across layers (embedding + L transformer blocks) for the [CLS] token:

$$\text{CKA}(X, Y) = \frac{\|Y^\top X\|_F^2}{\|X^\top X\|_F \cdot \|Y^\top Y\|_F}$$

**What this reveals**: if CKA between layer $l$ and layer $l+1$ is very high (>0.95), the deeper block is redundant — we should consider a shallower model. If CKA drops sharply, each block is doing meaningfully different work. Directly informs whether $L=2$ is optimal and connects to Ablation A2.

### 12.11 Attention-Is-Not-Explanation Diagnostics (Jain & Wallace, 2019) — Critical Extension (N8)

Jain & Wallace (2019) showed that attention weights in NLP transformers can be radically perturbed without changing model predictions, undermining their use as explanations. We run their two key diagnostics on our tabular model:

1. **Attention distribution counterfactuals**: swap the attention distribution of a correctly-classified sample with that of a random other sample. If predictions remain stable, attention is not the "reason" for the prediction.
2. **Adversarial attention**: for each test sample, find a different attention distribution that keeps the prediction unchanged, using the constrained-optimisation procedure of Jain & Wallace (Alg. 1). Report the Jensen-Shannon divergence between the original and adversarial attention distributions.

**Importantly**, Wiegreffe & Pinter (2019) argued that attention *can* be a useful explanation in restricted contexts. We engage with both sides — this is the "critical analysis" that distinguishes distinction-level work.

Even if our attention maps survive the Jain-Wallace diagnostic, we discuss whether they *should* be interpreted as feature importance or merely as "what the model is doing computationally."

### 12.12 Attention Entropy Evolution During Training

For each attention head in each layer, compute the mean entropy of the attention distribution across test samples at each training checkpoint:

$$H^{(l,h)}(t) = \mathbb{E}_{x \sim D_{\text{val}}} \left[ -\sum_j A^{(l,h)}_{0,j}(x, t) \log A^{(l,h)}_{0,j}(x, t) \right]$$

**Useful finding if observed**: some heads specialise (entropy decreases) while others stay diffuse (entropy stable or rises). This is evidence of division-of-labour between heads, supporting the "multi-head attention captures diverse interactions" hypothesis.

### 12.13 Modules

```
src/interpret.py        — Attention extraction, rollout, per-head analysis, embedding viz,
                          attention entropy, Jain-Wallace diagnostics
src/ig_shap.py          — Integrated Gradients + KernelSHAP for the transformer
src/probing.py          — Linear probing of intermediate representations
src/cka.py              — Cross-layer representation similarity analysis
```

---

## 12.5 Phase 10A: Counterfactual Explanations (NOVEL — N6)

**Status: [TODO] TODO**

Regulated domains (including retail credit in the EU/UK) increasingly require **actionable, per-customer explanations**: not just "which features mattered globally?" but "what *specifically* would change this customer's decision?" This is the spirit of Article 22 of the GDPR and the forthcoming EU AI Act's transparency requirements for high-risk systems.

### 12.5.1 Method: Token Substitution Counterfactuals

For a customer $x$ classified as *default* with probability $p(x) = 0.78$:

1. For each of the 23 feature tokens, individually substitute it with a counterfactual value drawn from a curated set (e.g., "what if this customer had PAY_0 = 0 instead of 3?", "what if LIMIT_BAL were doubled?", "what if age were 5 years older?").
2. Re-run the transformer forward pass with the substituted token.
3. Record the predicted default probability $p(x')$.
4. Report the *smallest* substitution (in plain English) that flips the prediction to "no default": this is the minimal counterfactual explanation.

**Why this is novel for tabular transformers**: whereas standard tabular counterfactuals operate in raw feature space (Wachter et al., 2017), our method operates in *token-embedding space* via the learned tokeniser, producing counterfactuals that respect the hybrid PAY semantic structure (e.g., cannot propose "PAY_0 = 2.5", only valid integer states).

### 12.5.2 Batch-Level Analysis

- **Most frequent flip-triggers**: across the test set, which features, when substituted, flip predictions most often? This is a data-driven measure of per-feature *decision sensitivity*.
- **Minimum counterfactual distances**: histogram of how much intervention (L0 / L1 / domain-weighted distance) is needed to flip decisions; compare defaulters-predicted-as-non-default vs non-defaulters-predicted-as-default.

### 12.5.3 Module

```
src/counterfactuals.py  — Token-substitution counterfactuals + minimum-intervention search
```

---

## 13. Phase 11: Calibration Analysis

**Status: [DONE] — `src/calibration.py` + `tests/test_calibration.py`: temperature / Platt / isotonic + ECE (equal-width + equal-mass), MCE, Brier decomposition, Brier skill. Raw transformer ECE 0.26 → 0.011 ± 0.003 post-Platt (matches RF's 0.010); AUC unchanged. Artefacts in `results/calibration/` + `figures/calibration_{reliability,ece_bar}.png`.**

### 13.1 Why Calibration Matters

In credit risk, the predicted probability of default is often more valuable than the binary classification (Yeh & Lien, 2009). A well-calibrated model means: when it predicts 30% default probability, approximately 30% of such customers actually default. This enables accurate pricing of credit risk, Basel regulatory compliance, and portfolio-level risk management.

### 13.2 Expected Calibration Error (ECE)

Partition predictions into B equally-spaced bins. For each bin $b$:

$$\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{N} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|$$

where $\text{acc}(B_b)$ is the actual default rate in bin $b$ and $\text{conf}(B_b)$ is the mean predicted probability in bin $b$.

Lower ECE = better calibration. Use B=10 or B=15 bins.

### 13.3 Reliability Diagrams

Plot actual default rate (y-axis) vs predicted probability (x-axis) for each bin. A perfectly calibrated model lies on the diagonal. Overlay both transformer and RF on the same plot.

### 13.4 Post-hoc Calibration (Optional Enhancement)

If the raw model is poorly calibrated, apply:
1. **Platt scaling**: fit a logistic regression on the validation set predictions → calibrated probabilities
2. **Isotonic regression**: non-parametric calibration on validation predictions

Report ECE before and after calibration for both models.

### 13.5 Focal Loss and Calibration

Mukhoti et al. (2020) showed that focal loss inherently improves calibration compared to standard cross-entropy. If our focal loss model shows better ECE than the WBCE model, this supports that finding and is worth highlighting.

### 13.6 Temperature Scaling — Guo et al. (2017)

The simplest effective post-hoc calibration method for neural networks:

$$p_{\text{calibrated}}(x) = \sigma(z(x) / T)$$

where $z(x)$ is the pre-sigmoid logit and $T > 0$ is a learned scalar temperature optimised on the validation set by minimising NLL with LBFGS (2 minutes of compute). $T > 1$ softens over-confident predictions; $T < 1$ sharpens under-confident ones.

Temperature scaling does not change the predicted class (it is rank-preserving on logits) but substantially improves ECE. We report ECE before and after temperature scaling (compared in Ablation A20).

### 13.7 Brier Score Decomposition

The Brier score decomposes into three interpretable components (Murphy, 1973):

$$\text{BS} = \underbrace{\mathbb{E}[(p - \bar{y})^2]}_{\text{Reliability (calibration error)}} - \underbrace{\mathbb{E}[(p - \bar{p})^2]}_{\text{Resolution (discrimination)}} + \underbrace{\bar{y}(1 - \bar{y})}_{\text{Uncertainty (base rate)}}$$

**Why this matters**: a model can have a low Brier score either because it is well-calibrated (low reliability term) or because it is well-discriminating (high resolution term). Reporting all three components separates *accuracy of probabilities* from *usefulness of probabilities*.

### 13.8 Per-Subgroup Calibration

ECE computed within each demographic subgroup (SEX, EDUCATION category, MARRIAGE status) — see Phase 11A for the fairness framing. A model may have low overall ECE but high ECE within a minority subgroup; this is a form of calibration disparity that is a distinct fairness concern (Pleiss et al., 2017).

### 13.9 Module

```
src/calibration.py      — ECE, reliability diagrams, temperature scaling, Brier decomposition
```

---

## 13.5 Phase 11A: Fairness & Subgroup Robustness (NOVEL — N10)

**Status: [DONE] — `src/fairness.py` + `tests/test_fairness.py` audit SEX / EDUCATION / MARRIAGE on demographic parity, equal opportunity, equalised odds, subgroup AUC/ECE. Male/Female AUC gap = 0.011. EDUCATION "Other" (n=61) flagged underpowered (AUC drop 0.19–0.31), not reported. Artefacts in `results/fairness/{subgroup_metrics,disparity_metrics}.csv` + `figures/fairness_disparity.png` + `figures/fairness_reliability_{sex,education,marriage}.png`.**

Credit scoring is a regulated use-case. A model with high aggregate AUC but disparate performance across protected attributes is legally and ethically problematic. Fairness analysis is also one of the clearest places to demonstrate the "independent thought" criterion.

### 13.5.1 Subgroup Definition

The dataset contains three demographic attributes: SEX (2 values), EDUCATION (4 values), MARRIAGE (3 values). We audit performance on each level individually and on common intersections (e.g., female × university, male × high-school).

### 13.5.2 Fairness Metrics

For each subgroup $G$ and each model (transformer, RF):

| Metric | Formula | Interpretation |
|---|---|---|
| **Demographic parity gap** | $P(\hat{y}=1 \mid G=g) - P(\hat{y}=1 \mid G \neq g)$ | Equal positive-prediction rate across groups (the strict, controversial notion) |
| **Equal opportunity** — Hardt et al. (2016) | $\text{TPR}(G=g) - \text{TPR}(G \neq g)$ | Equal true-positive rate among actual defaulters across groups (more defensible in risk contexts) |
| **Equalised odds** | Equal TPR *and* equal FPR across groups | Joint fairness criterion |
| **Subgroup AUC-ROC** | AUC computed on subgroup only | Does discrimination quality differ between groups? |
| **Subgroup ECE** | ECE within subgroup | Is the model equally well-calibrated for each group? |
| **Cost parity** (cost-weighted) | $\Delta C_{\text{per-capita}}(G=g, G \neq g)$ | Does realised business cost differ across groups? |

### 13.5.3 Impossibility Results to Acknowledge

Kleinberg et al. (2017) / Chouldechova (2017) proved that with unequal base rates across groups, demographic parity, equalised odds, and calibration cannot all hold simultaneously. Our report **explicitly acknowledges this trade-off** and justifies our primary fairness criterion (equal opportunity) in the credit-risk context.

### 13.5.4 Mitigation Experiments (if disparities found)

If disparities are observed:
- **Reweight-by-subgroup**: during training, upweight loss on under-performing subgroups.
- **Adversarial fairness**: an adversarial head tries to predict the protected attribute from [CLS]; the encoder is trained to fool it (Zhang et al., 2018). Test as extension.

Report disparity metrics before and after mitigation.

### 13.5.5 Module

```
src/fairness.py         — Subgroup metrics, gap reports, mitigation variants
```

---

## 13.6 Phase 11B: Uncertainty Quantification (NOVEL — N11)

**Status: [DONE] — `src/uncertainty.py` + `tests/test_uncertainty.py`: MC-dropout (T=50) + predictive / aleatoric entropy + mutual info (BALD) + refuse-to-predict at 5% steps across three uncertainty signals. Predictive entropy ↔ misclassification (Spearman 0.175); deferring top 50% most-uncertain lifts retained AUC 0.779 → 0.850. Artefacts in `results/uncertainty/{mc_dropout.npz,refuse_curve.csv}` + `figures/uncertainty_{refuse_curve,entropy_hist}.png`.**

A calibrated probability is not the same as a *reliable* probability on a specific input. A model may be 90%-confident on an easy example but only 55%-confident on a hard one — this is **epistemic uncertainty** and is critical in regulated domains where "refuse to predict and escalate to human review" is a valid decision.

### 13.6.1 Monte-Carlo Dropout — Gal & Ghahramani (2016)

Inference-time uncertainty without retraining:

1. Keep dropout active at inference time.
2. For each test example, run $T = 50$ stochastic forward passes.
3. The mean of predictions is the point estimate; the variance (or predictive entropy) is an uncertainty estimate.

$$\hat{p}(x) = \frac{1}{T} \sum_{t=1}^T p^{(t)}(x), \quad \sigma^2(x) = \frac{1}{T}\sum_{t=1}^T (p^{(t)}(x) - \hat{p}(x))^2$$

### 13.6.2 Predictive Entropy as Uncertainty Signal

$$H(x) = -\hat{p}(x) \log \hat{p}(x) - (1 - \hat{p}(x)) \log (1 - \hat{p}(x))$$

### 13.6.3 Uncertainty–Correctness Correlation

We compute the **Spearman correlation** between $H(x)$ and $\mathbb{1}[\text{wrong}]$ on the test set. A useful uncertainty signal has positive correlation: higher uncertainty ⇒ higher misclassification rate. We report:

- Spearman $\rho(H, \text{error})$ — is there a signal?
- **Selective prediction curves**: accuracy as a function of coverage (fraction of predictions the model is willing to make). If we abstain on the top-20%-uncertain predictions, by how much does accuracy improve on the remaining 80%?

### 13.6.4 Deep Ensembles Alternative (optional)

Train the transformer 5 times with different seeds. Ensemble the predictions. Compare the ensemble-variance-based uncertainty estimate against MC-dropout. Deep ensembles (Lakshminarayanan et al., 2017) are more expensive but often better calibrated than MC-dropout.

### 13.6.5 Module

```
src/uncertainty.py      — MC-dropout inference, predictive entropy, selective prediction curves
```

---

## 14. Phase 12: Statistical Significance Testing

**Status: [DONE] — `src/significance.py` + `tests/test_significance.py`: McNemar (exact + chi-sq), DeLong AUC-diff (Sun & Xu 2014), paired bootstrap on arbitrary metrics, BH-FDR correction, Hanley-McNeil power. DeLong RF-vs-transformer AUC p = 0.023 raw, q = 0.23 after BH over 15 pairs → no significant gap claimed. 4,500-row test split has 80% power only for AUC gaps ≥ 0.02. Artefacts in `results/significance/{pairwise_tests,power_analysis}.csv` + `figures/significance_pvalue_heatmap.png`.**

### 14.1 Multi-Run Analysis

Train the transformer 5 times with different random seeds. Report mean ± standard deviation for all metrics. This demonstrates that results are robust and not artefacts of a lucky initialisation.

### 14.2 McNemar's Test

Compare the binary predictions of the transformer vs RF on the same test set. McNemar's test assesses whether the two models disagree on the same samples in a statistically significant way:

$$\chi^2 = \frac{(b - c)^2}{b + c}$$

where $b$ = samples RF gets right but transformer gets wrong, $c$ = the reverse.

### 14.3 DeLong's Test

Compare AUC-ROC values between the two models using DeLong's test, which accounts for the correlation between paired AUC estimates on the same test data.

### 14.4 Bootstrap Confidence Intervals

Compute 95% bootstrap confidence intervals for all metrics:
1. Resample the test set with replacement (1,000 iterations)
2. Compute metric on each bootstrap sample
3. Report 2.5th and 97.5th percentiles

### 14.5 Paired Bootstrap for Model Differences

Unpaired confidence intervals on two models' AUCs do **not** correctly test whether one model beats the other, because the two AUC estimates are computed on the *same* test set and therefore correlated. The correct procedure is to bootstrap the **difference**:

1. Resample the test set with replacement (1,000 iterations).
2. Compute $\Delta_b = \text{AUC}_{\text{transformer}}(b) - \text{AUC}_{\text{RF}}(b)$ on each resample.
3. The 95% paired-bootstrap CI is [2.5th, 97.5th] percentile of $\{\Delta_b\}$.
4. "Transformer significantly better than RF" iff the lower bound of the CI > 0.

Applied to: AUC-ROC, AUC-PR, F1, accuracy. Reported as a table of $\Delta$ estimates with 95% CIs in the report.

### 14.6 Multiple-Comparison Correction

The ablation study (Phase 9) performs 22+ hypothesis tests. Without correction, the family-wise error rate inflates: at $\alpha = 0.05$, ~5% of *uninformative* comparisons will nominally reach significance.

We apply the **Benjamini-Hochberg procedure** (Benjamini & Hochberg, 1995) at a false-discovery rate of $q = 0.10$:

1. Sort the ablation $p$-values in ascending order: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$.
2. Find the largest $k$ such that $p_{(k)} \leq \frac{k}{m} q$.
3. Reject $H_0$ for all tests with rank $\leq k$.

Ablations whose $p$-values survive the BH correction are reported as "significant at FDR $q=0.10$"; those that do not are reported as suggestive but not confirmed. This is a core statistical-rigour discipline that most student reports omit.

### 14.7 Power Analysis (Sample-Size Diagnostic)

Our test set has 4,500 rows of which 995 are defaults. Given this, what **effect size** in $\Delta$AUC can we reliably detect? Use a bootstrap-based simulation: plant a known $\Delta$ (e.g., 0.01, 0.02, 0.03), resample 1,000 times, measure the fraction that reject $H_0$.

Report the **minimum detectable effect** at 80% power and $\alpha = 0.05$. This contextualises all ablation findings: a non-significant ablation with MDE = 0.02 is consistent either with no effect or with an effect smaller than 0.02 — the data cannot distinguish.

### 14.8 Module

```
src/stat_tests.py       — McNemar, DeLong, paired bootstrap, Benjamini-Hochberg, power analysis
```

---

## 15. Phase 13: Report Writing

**Status: [TODO] TODO**

### 15.1 Report Structure

| Section | Target Words | Content |
|---|---|---|
| **1. Introduction** | 300 | Taiwan credit crisis context; problem definition; why transformers for tabular data is interesting and challenging; brief overview of approach; outline of remaining sections |
| **2. Data Exploration** | 500 | Key EDA findings that motivate modelling decisions; temporal divergence; PAY feature semantic structure; class imbalance; feature interactions; reference figures |
| **3. LLM Model Build-up** | 1,400 | **MOST IMPORTANT SECTION (40% marks)**. Full methodology: tokenisation design with justification for each feature type; embedding strategy; mathematical formulation of attention; transformer block architecture; PreNorm choice; [CLS] token; classification head; loss function; training procedure; hyperparameters. Include mathematical formulations. Every design decision justified with reasoning and/or literature. |
| **4. Experiments, Results, Discussion** | 1,200 | **Required sub-sections** (explicitly mandated by the coursework PDF): (a) experimental setup; (b) all metrics (table); (c) ROC/PR curves; (d) ablation study results (table + discussion); (e) attention visualisation findings; (f) calibration analysis; (g) statistical significance; (h) **head-to-head summary comparing the two models** (PDF: "A clear summary comparing the performance of the two models must be included"); (i) feature importance comparison; (j) **"Is the transformer-based approach the right model for this problem?"** — dedicated sub-section engaging with Grinsztajn et al. (2022), explicitly prompted by the PDF; (k) limitations; (l) suggestions for improvement. |
| **5. Conclusions** | 200 | Summary of key findings; what worked, what didn't; main takeaways; concrete future work suggestions |
| **6. Acknowledgements** | **≤ 50 hard** | Three mandatory contents (per coursework PDF): (a) explicit declaration of any LLM use and *how* it was used (non-negotiable — failure to declare = penalty); (b) contribution-breakdown table listing every group member and what they did; (c) any other acknowledgements (supervisors, data providers). The 50-word ceiling is absolute — the contribution table does not count toward the 50 because it is a table. |
| **7. References** | N/A | All cited works in consistent style |
| **8. Appendices** | N/A | Per coursework PDF: "appendices should be divided into sections corresponding to each phase of your analysis so that they are easy to navigate and reference." Planned structure: **A.** Data loading & cleaning (schema diff, validation report); **B.** Full EDA figure set (figures not shown in main body); **C.** Tokeniser design details; **D.** Transformer architecture (full forward-pass trace, parameter-count breakdown); **E.** Training logs & hyperparameter search results; **F.** Ablation result tables (A1–A14); **G.** Random Forest tuning + SHAP; **H.** Calibration + statistical tests; **I.** Relevant code excerpts. Reference every appendix from the main body by name. |

### 15.2 Writing Quality Standards

- **Precise, economical prose** — every word earns its place. No filler, no repetition.
- **Academic register** — third person, passive voice where appropriate, measured claims.
- **Mathematical rigour** — all formulations rendered properly, consistent notation throughout.
- **Figure quality** — every figure numbered sequentially (Fig 1, Fig 2, …), given a descriptive caption under it, labelled sub-panels (a), (b), (c) where composite, with axis labels, units, and legends. **Every figure must be referenced in the body text at least once** (per coursework PDF: "All figures and tables must be clearly labelled and referenced in the text"). Unreferenced figures cost marks for presentation.
- **Table quality** — numbered sequentially (Table 1, Table 2, …), with a caption above, clear headers, consistent significant figures (4 d.p. for metrics, 2 d.p. for %), referenced at least once in the body text.
- **Referencing style** — pick **one** style (APA-7 is the recommended default for this report) and use it consistently for every citation throughout. In-text: `(Gorishniy et al., 2021)`. Do not mix author-year with numeric styles. Every citation must appear in the References section and vice versa.
- **References integrated naturally** — "Following the FT-Transformer paradigm (Gorishniy et al., 2021), we embed each feature..." not "Gorishniy et al. (2021) proposed the FT-Transformer. We used it."
- **Coherent narrative** — the report tells a story: problem → approach → findings → meaning.
- **Critical honesty** — if the transformer underperforms RF, discuss why honestly. Cite Grinsztajn et al. (2022).

### 15.3 Narrative Arc

1. **Setup the tension**: Transformers revolutionised NLP, but do they work for tabular data? Trees still dominate (Grinsztajn et al., 2022). This coursework tests whether a transformer can compete on a real-world credit risk dataset.
2. **Show the data**: EDA reveals temporal structure, feature interactions, and semantic complexity that could benefit from attention.
3. **Build the solution**: Carefully designed tokenisation respects the data's heterogeneity. From-scratch transformer with principled architecture choices.
4. **Test rigorously**: Comprehensive experiments, ablations, and comparison with a strong RF baseline.
5. **Reflect critically**: What worked? What didn't? What does this tell us about when transformers are/aren't appropriate for tabular data?

### 15.4 Figures for the Report

| Figure | Section | Content |
|---|---|---|
| Fig 1 | §2 | Class distribution |
| Fig 2 | §2 | PAY status semantic analysis (motivates tokenisation) |
| Fig 3 | §2 | Temporal trajectories (motivates attention) |
| Fig 4 | §2 | Feature-target association ranking |
| Fig 5 | §3 | Architecture diagram (model overview) |
| Fig 6 | §3 | Tokenisation scheme (feature → embedding → sequence) |
| Fig 7 | §4 | ROC curves (both models) |
| Fig 8 | §4 | Precision-Recall curves (both models) |
| Fig 9 | §4 | Confusion matrices (side by side) |
| Fig 10 | §4 | Training curves (loss + AUC over epochs) |
| Fig 11 | §4 | Attention heatmap ([CLS] to features) |
| Fig 12 | §4 | Feature importance comparison (attention vs RF Gini vs SHAP) |
| Fig 13 | §4 | Reliability diagram (calibration) |
| Fig 14 | §4 | Embedding space visualisation (t-SNE of [CLS] outputs) |

### 15.5 Tables for the Report

| Table | Section | Content |
|---|---|---|
| Table 1 | §2 | Summary statistics of key features |
| Table 2 | §3 | Model hyperparameters (final configuration) |
| Table 3 | §3 | Parameter count breakdown |
| Table 4 | §4 | Comprehensive metrics comparison (Transformer vs RF) |
| Table 5 | §4 | Ablation study results |
| Table 6 | §4 | Hyperparameter search results (top configurations) |
| Table 7 | §6 | Contribution breakdown (group members) |

---

## 16. Phase 14: GitHub Repository Structure

**Status: [TODO] TODO**

### 16.1 Directory Layout

```
credit-default-transformer/
├── README.md                           # Project overview, setup, usage
├── requirements.txt                    # Pinned dependencies
├── .gitignore                          # Ignore data/, checkpoints/, __pycache__
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py           # [DONE] Data loading, cleaning, splitting, scaling
│   ├── eda.py                          # [DONE] Exploratory data analysis (all figures)
│   ├── tokeniser.py                    # [TODO] Feature tokenisation and embedding
│   ├── attention.py                    # [TODO] ScaledDotProductAttention, MultiHeadAttention
│   ├── transformer.py                  # [TODO] TransformerBlock, TransformerEncoder
│   ├── model.py                        # [TODO] TabularTransformer (full model)
│   ├── losses.py                       # [TODO] FocalLoss, WeightedBCE
│   ├── dataset.py                      # [TODO] PyTorch Dataset and DataLoader
│   ├── train.py                        # [TODO] Training loop, LR schedule, early stopping
│   ├── evaluate.py                     # [TODO] Metrics, threshold optimisation, comparison
│   ├── random_forest.py                # [TODO] RF training, tuning, feature importance, SHAP
│   ├── visualise.py                    # [TODO] All evaluation and analysis plots
│   ├── interpret.py                    # [TODO] Attention analysis, embedding viz
│   ├── calibration.py                  # [TODO] ECE, reliability diagrams, Platt scaling
│   └── utils.py                        # [TODO] Seeds, device, checkpointing, logging
│
├── notebooks/
│   └── full_pipeline.ipynb             # [TODO] End-to-end runnable notebook
│
├── data/
│   ├── raw/                            # Original XLS (gitignored, with download script)
│   └── processed/                      # [DONE] Cleaned CSVs, metadata JSON
│
├── figures/                            # [DONE] (EDA) + [TODO] (evaluation) All output figures
├── results/                            # [TODO] Metrics, ablation tables, checkpoints
├── report/                             # [TODO] Report markdown/LaTeX source
│
└── scripts/
    ├── download_data.py                # [TODO] Download from UCI repository
    ├── run_full_pipeline.sh            # [TODO] End-to-end execution script
    └── run_ablations.sh                # [TODO] Run all ablation experiments
```

### 16.2 README.md Contents

1. Project title and description
2. Repository structure overview
3. Setup instructions (Python version, `pip install -r requirements.txt`)
4. Data download instructions
5. How to reproduce all results (single command)
6. Brief description of the approach
7. Results summary
8. Team contributions
9. License / acknowledgements

### 16.3 requirements.txt

```
torch>=2.0
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
shap>=0.42
scipy>=1.10
xlrd>=2.0
optuna>=3.0
tqdm>=4.65
captum>=0.7               # Integrated Gradients
mlflow>=2.10              # local experiment tracking
hydra-core>=1.3           # configuration management
pytest>=7.4               # testing
hypothesis>=6.100         # property-based tests
ruff>=0.3                 # linting
mypy>=1.8                 # static typing
```

### 16.4 Engineering Standards

Marking criteria include "writing quality" and "presentation" (25% combined). Clean engineering directly earns these marks. We commit to:

- **Static typing**: `mypy --strict` on all `src/` modules. Every public function is type-annotated.
- **Linting**: `ruff check` on every commit. Zero warnings in committed code.
- **Unit tests**: `pytest` target ≥ **70% statement coverage** on `src/` (exclude visualisation code, which is validated visually). Property-based tests (`hypothesis`) for the tokeniser: for any valid row, `tokenise()` must produce shape $(24, d)$ with no NaN / Inf.
- **Pre-commit hooks**: format + lint + typecheck before every commit. No broken commits in history.
- **Continuous integration** (GitHub Actions): on every push, run the test suite and lint on Python 3.10 and 3.11. A passing CI badge on the README signals reviewable-quality code.
- **Config management** (Hydra): hyperparameters live in `configs/*.yaml`, not hardcoded. One config per ablation. Reproducible via `python -m src.train --config=configs/ablation_A15.yaml`.
- **Experiment tracking** (MLflow, local backend): every training run logs hyperparameters, metrics-per-epoch, gradient norms, and the final checkpoint. Runs are browsable via `mlflow ui`. No results are lost to terminal scrollback.
- **Docker / devcontainer**: a `Dockerfile` and `.devcontainer/devcontainer.json` so that any marker on any OS can reproduce the environment in one `docker run` or "Reopen in Container" click.

---

## 16.5 Phase 14A: Reproducibility Guarantees

**Status: [DONE] — `src/repro.py` + `tests/test_repro.py` + `docs/REPRODUCIBILITY.md` + `data/processed/SPLIT_HASHES.md`. Seven checks: artefact presence, transformer-run-file integrity, split-hash SHA-256 (§16.5.3), python/torch pins, git-clean, RF-prediction bit-parity (max |Δp| = 0), `evaluate.py` bit-parity. `python src/repro.py` exits 0 when everything matches. §16.5.4 Dockerfile deferred.**

"Code runs on my machine" is not a guarantee. We commit to stronger guarantees that any marker can verify in a clean environment.

### 16.5.1 Determinism Protocol

Every `train.py` run begins with:

```python
import os, random
import numpy as np
import torch

def set_deterministic(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
```

### 16.5.2 Seed Sweep for All Reported Numbers

For every headline number in the report (main table, ablation table, calibration table), we train with **10 seeds**: $\{0, 1, 42, 100, 123, 256, 512, 1024, 2024, 7777\}$. Report median and IQR (not just mean ± std — more robust to outliers). This directly addresses the "5 runs" suggestion in §8.6 with a stronger commitment.

### 16.5.3 Hash-Stable Data Splits

The train/val/test split is deterministic given the random seed, but we go one step further: the plan commits the SHA-256 hash of each split CSV file so that anyone running the pipeline can verify that they obtained the same split as us:

```
data/processed/SPLIT_HASHES.md
├── train_raw.csv:     sha256:a7f3b9c1...
├── val_raw.csv:       sha256:9d8e2f4a...
├── test_raw.csv:      sha256:2c6a1e5b...
└── feature_metadata.json: sha256:6b1f4d8c...
```

Anyone who does not get these hashes has not reproduced the experiment.

### 16.5.4 Environment Pinning

- `pyproject.toml` pins exact dependency versions (no floating `>=`).
- `poetry.lock` committed.
- Docker image built from a specific `python:3.11.5-slim-bookworm` tag.
- `uname -a` and `nvidia-smi` captured in every MLflow run so that hardware is documented.

### 16.5.5 Reproducibility Statement in the Report

A dedicated appendix (Appendix I) states the exact commands required to reproduce every headline number:

```
git clone https://github.com/abailey81/credit-default-tabular-transformer
cd credit-default-tabular-transformer
git checkout <commit-hash>
docker build -t ccrisk . && docker run -it --gpus=all ccrisk bash
python -m scripts.reproduce_all    # runs every phase end-to-end
```

---

## 16.6 Phase 14B: Model Card & Data Sheet (NOVEL — N12)

**Status: [DONE] — `docs/MODEL_CARD.md` + `docs/DATA_SHEET.md` cover intended use, out-of-scope uses, performance, Phase 11A fairness numbers, the deploy-with-Platt calibration caveat, and the 2005-Taiwan vintage limitation.**

Following Mitchell et al. (2019) and Gebru et al. (2021), we produce two formal responsible-AI artefacts for the final model:

### 16.6.1 Model Card (Mitchell et al., 2019)

A short structured document (≤ 2 pages, in Appendix) covering:

- **Intended use** (credit-default prediction on the UCI Taiwan 2005 dataset — and *only* that; not for any deployment decision)
- **Out-of-scope uses** (e.g., extrapolating to non-Taiwan populations; post-2005 credit policy; any customer not matching the dataset's demographic distribution)
- **Evaluation** (summary of Phase 8 test-set metrics + Phase 11A subgroup breakdown)
- **Training data** (split sizes, class balance, demographic composition)
- **Ethical considerations** (regulatory context, potential for disparate impact, mitigation steps)
- **Caveats and recommendations** (dataset age, sample-size limits, calibration under stress)

### 16.6.2 Data Sheet (Gebru et al., 2021)

A short structured document (≤ 2 pages, in Appendix) covering the Yeh & Lien 2005 dataset itself:

- **Motivation for collection** (Taiwan credit-card crisis)
- **Composition** (30,000 clients, what features, what they mean, missing values, duplicates)
- **Collection process** (major Taiwanese bank records, cross-section)
- **Preprocessing / cleaning done by us** (categorical code merges, our engineered features, splits)
- **Uses** (academic benchmark — Yeh 2009 et al.)
- **Known limitations** (2005 vintage; a single geography; limited demographic breadth; no payment-attempt timing)

### 16.6.3 Why This Earns Marks

The coursework criterion "critical engagement" is explicit. A model card + data sheet demonstrate that the team has thought about **what the model is, what it is not for, and how data provenance shapes limitations** — exactly the reflective posture that distinction-level marking rewards.

---

## 17. Marking Alignment Matrix

This section maps every marking criterion to where it is addressed in the project.

### Section 3: Model Build-up (40%)

| Criterion | Where Addressed |
|---|---|
| i) Tabular data → token sequence | Phase 3: §5.2–5.6 (hybrid PAY tokenisation N1) + §5.4B (MLM-compatible masking) |
| ii) Embedding design | §5.2 (numerical linear / PLE, categorical, PAY hybrid); §5.3 feature-type / positional; §5.4A PLE |
| iii) Q, K, V and attention weights | §6.1–6.2 (full mathematical derivation); §6.12.1 group-bias N2; §6.12.2 temporal-decay N3 |
| iv) Transformer block architecture | §6.4–6.6 (PreNorm, FFN, stacking); §6.11 implementation spec; §6.12 novel inductive biases |
| v) How model predicts default | §6.7 (CLS → classification head → sigmoid) |
| vi) Loss function, optimisation, training | §7 (focal loss), §8 (AdamW, cosine LR, early stop, clipping, 10-seed sweep); **§8.5 MTLM pretraining N4 — directly addresses "language model" framing**; §8.6 multi-task N5 |
| "Deep understanding" | Mathematical derivations for attention scaling (§6.1), PreNorm stability, GELU smoothness; complete engagement with attention-as-kernel perspective; scaling-law analysis (Ablation A19) |
| "Model design" | FT-Transformer-inspired backbone (standard) + three novel inductive biases: hybrid PAY (N1), group-attention bias (N2), temporal-decay prior (N3); MTLM pretraining (N4); multi-task auxiliary (N5) |
| **"Novelty" (direct answer for markers)** | **See §1.6 Novelty Register — 12 independent contributions catalogued, each tied to a verification experiment. The headline novelty is the MTLM pretraining (N4), which is the most "language-model-like" component of the project and was not required by the spec.** |
| "Independent thought" | §1.6 Register + Phase 10 Jain-&-Wallace diagnostic (N8) + Phase 10A counterfactuals (N6) + Ablations A17/A18 null baselines (N7) + Phase 11A fairness impossibility trade-off + Phase 8A cost-sensitive domain framing (N9) |

### Section 4: Experiments and Results (30%)

| Criterion | Where Addressed |
|---|---|
| Model comparison | §10 (comprehensive metrics table + paired bootstrap CIs on $\Delta$); §10.6 decision-surface visualisation |
| Experimental methodology | §11 (22 ablations A1–A22, with hypotheses and 10-seed sweeps); §14.6 Benjamini-Hochberg FDR control; §14.7 power analysis |
| Interpretability | §12 (attention viz, rollout, per-head, entropy-during-training); §12.7 Integrated Gradients; §12.8 SHAP; §12.9 probing; §12.10 CKA; §12.11 Jain-&-Wallace diagnostics (N8); **§12.5 counterfactual explanations (N6)** |
| Calibration | §13 (ECE, reliability diagrams); §13.6 temperature scaling; §13.7 Brier decomposition; §13.8 per-subgroup calibration |
| Statistical validity | §14 (multi-run 10-seed, McNemar, DeLong, **paired bootstrap for $\Delta$ in §14.5**, Benjamini-Hochberg FDR in §14.6, power analysis in §14.7) |
| **Fairness & robustness** | **§13.5 Phase 11A — subgroup metrics across SEX/EDUCATION/MARRIAGE (N10); calibration-parity impossibility discussion; optional mitigation experiments** |
| **Uncertainty quantification** | **§13.6 Phase 11B — MC-Dropout predictive entropy (N11); selective-prediction curves** |
| **Business context** | **§10.7 Phase 8A — cost matrix, Expected Credit Loss, cost-optimal threshold, portfolio stress test (N9)** |
| Limitations | Report §4 (dedicated limitations subsection — sample size, 2005 vintage, single-geography, feature-coverage gaps; Phase 8A stress test reveals degradation) |
| **"Is transformer right for this problem?"** | **Report §4 dedicated sub-section — synthesise Ablation A17 (random attention null) + A18 (linear probe floor) + A19 (scaling curve) with Grinsztajn et al. (2022) to answer directly. If the transformer is no better than a linear probe or a random-attention model, that is the honest finding.** |

### All Other Sections (25%)

| Criterion | Where Addressed |
|---|---|
| Introduction | Report §1 (Taiwan context, problem definition) |
| Data exploration | Report §2 + Phase 2 figures |
| Conclusions | Report §5 |
| Writing quality | §15.2 (writing standards) |
| Referencing | §20 (reference library) |
| Appendices | Well-organised code excerpts |
| Presentation | Publication-quality figures and tables |

---

## 18. Risk Register & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Transformer overfits on 21K samples | High | Medium | Aggressive dropout, early stopping, weight decay, small model (d=32, ~28K params), MTLM pretraining (N4) regularises representation |
| Transformer significantly underperforms RF | Medium | Low | This is an interesting finding — the random-attention null baseline (A17) and linear-probe floor (A18) force us to *quantify* the gap honestly. If the transformer is no better than a linear probe, the report discusses this directly (Grinsztajn et al., 2022) |
| Training instability (NaN loss, divergence) | Medium | High | Gradient clipping, PreNorm, warmup, careful LR tuning, determinism protocol (§16.5.1) |
| Attention weights are uninformative / uniform | Medium | Medium | Try more heads, deeper model, rollout; **run the Jain-&-Wallace diagnostic (§12.11) and engage critically — even if attention survives it, discuss the distinction between "explanatory" and "computational" interpretations** |
| MTLM pretraining does not help (null result) | Medium | Low | **This is itself a valid report finding** — connects to Rubachev 2022 on when pretraining is useful (data scarcity, heterogeneity). Ablation A15 quantifies the null; pretraining code is a feature, not a crutch |
| Multi-task auxiliary hurts primary task | Low | Low | Ablate weight $\lambda \in \{0, 0.1, 0.3, 0.5, 1.0\}$; report whichever works; no deployment risk |
| Hyperparameter search takes too long | Low | Medium | Use Optuna with early-stopping pruner; fixed budget of 50–100 trials; smaller model for search then scale |
| Word count exceeded (4000-word cap) | Medium | Medium | Track continuously via a simple `wc -w` script; move method detail to appendices; use figure/table captions as information-dense substitutes for prose |
| Code doesn't reproduce on marker's machine | Low | High | Full reproducibility protocol (Phase 14A): deterministic training, seeded everything, Docker, SHA-256 split hashes, pinned environment |
| Subgroup analysis reveals embarrassing disparities | Medium | Medium | **Report them honestly** — reporting disparity with critical engagement is *better* for marks than hiding it. Discuss the Kleinberg et al. (2017) impossibility result as context |
| MC-dropout uncertainty is not useful (no correlation with error) | Low | Low | Try deep ensembles as alternative; or report the null correlation and discuss what it means about model behaviour |
| Jain-&-Wallace diagnostic "breaks" our attention explanation | Medium | Low | **This is exactly the kind of critical finding distinction-level markers reward**. Engage with Wiegreffe & Pinter (2019) response paper; distinguish faithful vs plausible explanations |
| Novelty claims (§1.6) challenged as not-truly-novel | Low | High | Use measured language: "to our knowledge, not previously applied to this dataset / setting"; attribute adjacent prior work honestly (Rubachev 2022 for MTLM; FT-Transformer for backbone; Jain & Wallace for diagnostic) |
| Report feels like "LLM output" | Medium | High | Write specific, data-driven claims; quote actual numbers from our experiments; maintain a consistent human voice; cite sources sparingly but specifically; declare LLM usage honestly in Acknowledgements |
| Contribution mark-down due to uneven workload | Medium | High | Keep a dated contribution log from day 1; use distinct git authors; every member owns ≥ 1 phase end-to-end; face-to-face viva readiness ensured by weekly walkthroughs |

---

## 19. Timeline & Milestones

| Phase | Tasks | Dependencies |
|---|---|---|
| **Phase 1–2** [DONE] | Data loading, cleaning, EDA | None |
| **Phase 3** | Tokeniser + embedding module (hybrid PAY, feature-type, [CLS]); MLM-compatible masking (§5.4B) | Phase 1 (metadata) |
| **Phase 4** | Attention (DONE — PR #7), transformer blocks, full model; optional N2 group-bias, N3 temporal-decay variants | Phase 3 |
| **Phase 5** | Loss functions (focal, WBCE, label smoothing) | None |
| **Phase 6** | Supervised training pipeline + dataset class | Phases 3, 4, 5 |
| **Phase 6A** | **MTLM self-supervised pretraining** (N4) — pretraining loop, masking collator, per-feature heads | Phase 3 (§5.4B), Phase 4 |
| **Phase 6B** | **Multi-task auxiliary PAY-forecast head** (N5) during fine-tuning | Phase 6 |
| **Phase 7** | Random forest benchmark (the required comparator) | Phase 1 (engineered features) |
| **Phase 8** | Evaluation framework (metrics, threshold opt, paired bootstrap) | Phases 6, 7 |
| **Phase 8A** | **Business cost-sensitive evaluation** (N9) — cost matrix, ECL, portfolio stress test | Phase 8 |
| **Phase 9** | Ablation studies (A1–A22, 10-seed sweeps, BH correction) | Phases 6A, 6B, 8 |
| **Phase 10** | Attention interpretability: rollout, per-head, entropy, **Jain-Wallace (N8)**, IG, SHAP, probing, CKA | Phase 8 |
| **Phase 10A** | **Counterfactual token-substitution explanations** (N6) | Phase 6 |
| **Phase 11** | Calibration: ECE, reliability diagrams, temperature scaling, Brier decomposition | Phase 8 |
| **Phase 11A** | **Fairness & subgroup robustness** (N10) — disparity metrics, optional mitigation | Phase 8 |
| **Phase 11B** | **Uncertainty quantification** (N11) — MC dropout, predictive entropy, selective prediction | Phase 6 |
| **Phase 12** | Statistical significance: multi-seed, McNemar, DeLong, paired bootstrap, BH-FDR, power analysis | Phase 8 |
| **Phase 13** | Report writing (2–3 days intense) | All previous phases |
| **Phase 14** | GitHub cleanup + README + engineering standards (typing, CI, tests, Docker) | Phase 13 |
| **Phase 14A** | **Reproducibility guarantees** (determinism, 10-seed sweep, hash-stable splits, Docker) | Phase 14 |
| **Phase 14B** | **Model Card + Data Sheet** (N12) — responsible-ML artefacts in Appendix | Phase 8, Phase 11A |

**Critical path (supervised branch)**: Phases 1 → 3 → 4 → 5 → 6 → 7 → 8 → {9, 10, 10A, 11, 11A, 11B, 12, 8A} → 13 → {14, 14A, 14B}

**Critical path (SSL branch)**: Phases 1 → 3 (+§5.4B) → 4 → **6A (MTLM)** → 6 (fine-tune) → 6B → rest as above.

**Phase concurrency**: Phases 8A, 9, 10, 10A, 11, 11A, 11B, 12 can all run in parallel once Phase 8 completes. This is explicitly designed to let 4–5 group members work simultaneously.

---

## 20. Reference Library

### Core Transformer Papers
1. Vaswani, A. et al. (2017). "Attention Is All You Need." *NeurIPS*.
2. Devlin, J. et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL*.

### Tabular Transformers
3. Huang, X. et al. (2020). "TabTransformer: Tabular Data Modeling Using Contextual Embeddings." *arXiv:2012.06678*.
4. Gorishniy, Y. et al. (2021). "Revisiting Deep Learning Models for Tabular Data." *NeurIPS*.
5. Gorishniy, Y. et al. (2022). "On Embeddings for Numerical Features in Tabular Deep Learning." *NeurIPS*.
6. Somepalli, G. et al. (2021). "SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training." *arXiv:2106.01342*.
7. Cholakov, R. & Kolev, T. (2022). "GatedTabTransformer: Enhancing Tabular Data Modeling with Gated Multi-Layer Perceptrons." *arXiv*.
8. Borisov, V. et al. (2022). "Deep Neural Networks and Tabular Data: A Survey." *IEEE TNNLS*.

### The Counter-Argument
9. Grinsztajn, L. et al. (2022). "Why do tree-based models still outperform deep learning on typical tabular data?" *NeurIPS*.

### Dataset
10. Yeh, I.C. & Lien, C.H. (2009). "The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients." *Expert Systems with Applications*, 36(2), 2473–2480.

### Loss Functions & Calibration
11. Lin, T.Y. et al. (2017). "Focal Loss for Dense Object Detection." *ICCV*.
12. Mukhoti, J. et al. (2020). "Calibrating Deep Neural Networks Using Focal Loss." *NeurIPS*.

### Optimisation & Training
13. Loshchilov, I. & Hutter, F. (2019). "Decoupled Weight Decay Regularization." *ICLR*.
14. Kingma, D.P. & Ba, J. (2015). "Adam: A Method for Stochastic Optimization." *ICLR*.

### Regularisation & Normalisation
15. Srivastava, N. et al. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." *JMLR*.
16. Ba, J., Kiros, J.R. & Hinton, G.E. (2016). "Layer Normalization." *arXiv:1607.06450*.

### Activation Functions & Initialisation
17. Hendrycks, D. & Gimpel, K. (2016). "Gaussian Error Linear Units (GELUs)." *arXiv:1606.08415*.
18. Glorot, X. & Bengio, Y. (2010). "Understanding the difficulty of training deep feedforward neural networks." *AISTATS*.
19. He, K. et al. (2015). "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification." *ICCV*.

### Interpretability
20. Lundberg, S.M. & Lee, S.I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.
21. Abnar, S. & Zuidema, W. (2020). "Quantifying Attention Flow in Transformers." *ACL*.
22. Jain, S. & Wallace, B.C. (2019). "Attention is not Explanation." *NAACL*.

### Random Forests
23. Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5–32.

### Self-Supervised Learning for Tabular Data
24. Rubachev, I., Alekberov, A., Gorishniy, Y. & Babenko, A. (2022). "Revisiting Pretraining Objectives for Tabular Deep Learning." *arXiv:2207.03208*.

### Interpretability — Extensions
25. Sundararajan, M., Taly, A. & Yan, Q. (2017). "Axiomatic Attribution for Deep Networks (Integrated Gradients)." *ICML*.
26. Wiegreffe, S. & Pinter, Y. (2019). "Attention is not not Explanation." *EMNLP*.
27. Tsai, Y.-H.H., Bai, S., Yamada, M., Morency, L.-P. & Salakhutdinov, R. (2019). "Transformer Dissection: An Unified Understanding for Transformer's Attention via the Lens of Kernel." *EMNLP*.
28. Belinkov, Y. (2022). "Probing Classifiers: Promises, Shortcomings, and Advances." *Computational Linguistics*, 48(1), 207–219.
29. Kornblith, S., Norouzi, M., Lee, H. & Hinton, G. (2019). "Similarity of Neural Network Representations Revisited (CKA)." *ICML*.
30. Wachter, S., Mittelstadt, B. & Russell, C. (2017). "Counterfactual Explanations Without Opening the Black Box." *Harvard Journal of Law & Technology*, 31(2), 841–887.

### Calibration — Extensions
31. Guo, C., Pleiss, G., Sun, Y. & Weinberger, K.Q. (2017). "On Calibration of Modern Neural Networks." *ICML*.
32. Naeini, M.P., Cooper, G.F. & Hauskrecht, M. (2015). "Obtaining Well Calibrated Probabilities Using Bayesian Binning (ECE)." *AAAI*.
33. Platt, J. (1999). "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods." *Advances in Large Margin Classifiers*.
34. Murphy, A.H. (1973). "A New Vector Partition of the Probability Score (Brier decomposition)." *Journal of Applied Meteorology*, 12(4), 595–600.
35. Pleiss, G., Raghavan, M., Wu, F., Kleinberg, J. & Weinberger, K.Q. (2017). "On Fairness and Calibration." *NeurIPS*.

### Fairness
36. Hardt, M., Price, E. & Srebro, N. (2016). "Equality of Opportunity in Supervised Learning." *NeurIPS*.
37. Kleinberg, J., Mullainathan, S. & Raghavan, M. (2017). "Inherent Trade-Offs in the Fair Determination of Risk Scores." *ITCS*.
38. Chouldechova, A. (2017). "Fair Prediction with Disparate Impact: A Study of Bias in Recidivism Prediction Instruments." *Big Data*, 5(2), 153–163.
39. Dwork, C., Hardt, M., Pitassi, T., Reingold, O. & Zemel, R. (2012). "Fairness Through Awareness." *ITCS*.
40. Zhang, B.H., Lemoine, B. & Mitchell, M. (2018). "Mitigating Unwanted Biases with Adversarial Learning." *AIES*.

### Uncertainty Quantification
41. Gal, Y. & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." *ICML*.
42. Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." *NeurIPS*.

### Statistical Rigour
43. Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing." *JRSS-B*, 57(1), 289–300.
44. DeLong, E.R., DeLong, D.M. & Clarke-Pearson, D.L. (1988). "Comparing the Areas Under Two or More Correlated Receiver Operating Characteristic Curves." *Biometrics*, 44(3), 837–845.
45. Efron, B. & Tibshirani, R.J. (1994). *An Introduction to the Bootstrap*. Chapman & Hall.

### Responsible AI / Documentation
46. Mitchell, M. et al. (2019). "Model Cards for Model Reporting." *FAT*.
47. Gebru, T. et al. (2021). "Datasheets for Datasets." *Communications of the ACM*, 64(12), 86–92.

### Stochastic Depth & Regularisation Extras
48. Huang, G., Sun, Y., Liu, Z., Sedra, D. & Weinberger, K.Q. (2016). "Deep Networks with Stochastic Depth." *ECCV*.

### Risk-Modelling Domain Context
49. Basel Committee on Banking Supervision (2017). "Basel III: Finalising Post-Crisis Reforms." *Bank for International Settlements*.
50. IFRS 9 Financial Instruments — *International Accounting Standards Board*.

---

## 21. Coursework-PDF Strict Relevance Audit

This audit is intentionally paranoid. Every single requirement stated in the coursework PDF (`finance_and_ai_cw___group_project-2.pdf`) is enumerated below, mapped to the exact plan section(s) that satisfy it, and assessed for strict satisfaction. Every plan item that goes *beyond* the minimum spec is separately justified against the marking criteria so that nothing in the plan is "wasted effort".

### 21.1 Hard Task Requirements (must-have or the coursework fails)

| # | Verbatim PDF Requirement | Satisfied Where | Status |
|---:|---|---|:---:|
| 1 | "build a **small transformer-based language model from scratch**, trained and tested only on the credit card default dataset, to predict default" | Phase 4 (§6) + Phase 6 (§8) — full TabularTransformer, trained on this dataset only | ✓ |
| 2 | "takes an ordered, tokenised representation of each record and processes it through **explicit self-attention** using queries, keys, and values in one or more transformer blocks" | §6.1 Scaled Dot-Product Attention + §6.2 Multi-Head Attention implement explicit $Q$, $K$, $V$; §6.4 and §6.11 implement ≥ 1 transformer blocks | ✓ |
| 3 | "A standard feed-forward neural network does **not** satisfy this requirement, even if it is deep" | Our model is not an MLP — §6.1 through §6.11 are attention-centred | ✓ |
| 4 | "simply using a neural network and referring to it as a transformer will **not** receive credit" | We will not claim transformer credit for non-attention components; the report explicitly distinguishes encoder (attention) from the classification head (MLP) | ✓ |
| 5 | "you must **define and justify** how each record is converted into a sequence of tokens" | §5.2 (three strategies analysed); §5.2.3 PAY hybrid with explicit justification from EDA (Fig 04 non-linear risk profile); §5.4A PLE alternative | ✓ |
| 6 | "Your model must use the attention mechanism as a **central component** of the architecture" | §6.1–6.2 is the core of the model; §6.12 introduces additional *attention* biases (not FFN-only tricks) | ✓ |
| 7 | "you may use a standard deep-learning framework ... for tensors, automatic differentiation, and optimisation, but the **core attention mechanism and transformer block must be implemented and explained by you**" | §1.2 explicitly restates the rule; `src/attention.py` (ScaledDotProductAttention + MultiHeadAttention) and `src/transformer.py` (TransformerBlock + Encoder) are our own code; no `nn.TransformerEncoder` / `nn.MultiheadAttention` ever imported | ✓ |
| 8 | "Using a prebuilt transformer model or any LLM API as part of the modelling, training, or inference pipeline is **not allowed**" | We use PyTorch primitives only (`nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Dropout`, `nn.GELU`, `nn.Parameter`). No HuggingFace, no x-transformers, no pre-trained weights, no API calls. | ✓ |
| 9 | "As a benchmark, you will also build a **random forest** model and **tune its hyperparameters**" | Phase 7 (§9) — `RandomizedSearchCV` with 200 combos × 5-fold CV on a 7-dimension search space | ✓ |
| 10 | "develop and **compare** two models: (1) transformer, (2) random forest" | §10.4 comprehensive side-by-side comparison table; §14.5 paired bootstrap for $\Delta$; §13.5.2 subgroup parity of the comparison | ✓ |

### 21.2 Report Structure Requirements

| # | Verbatim PDF Requirement | Satisfied Where | Status |
|---:|---|---|:---:|
| 11 | "submit your report as a **single PDF file**" | §1.3 Deliverables (single PDF) + §1.3.1 submission logistics | ✓ |
| 12 | Exact section list: "1) Introduction, 2) Data Exploration, 3) LLM Model Build-up, 4) Experiments, Results, and Discussions, 5) Conclusions, 6) Acknowledgements, 7) References, 8) Appendices" | §15.1 matches this list verbatim | ✓ |
| 13 | "All code must be submitted through a **GitHub repository**, with the repository link provided in the report" | §1.3 Deliverables; README includes clone URL; link to be inserted in the Introduction | ✓ |
| 14 | "If you need to include code excerpts in the report, place them in the **appendices**" | §15.1 Appendices row — "code excerpts" explicitly in Appendix I | ✓ |
| 15 | "appendices should be divided into sections **corresponding to each phase** of your analysis" | §15.1 Appendices row — structured A–I, one per phase | ✓ |
| 16 | "The total word count for the report must **not exceed 4000 words**" | §1.4 word budget sums to ~3,650 words (buffer ~350); tracked continuously; excess moves to appendix | ✓ |
| 17 | "Programming code, appendix content, formulae, figures, and tables are **not included** in the word count" | §1.4 acknowledged; dense formula & figure use is an information-efficiency lever | ✓ |

### 21.3 Section 3 Detailed Requirements (the 40% section)

The PDF lists six explicit sub-requirements (i–vi) that Section 3 must describe and justify.

| # | Sub-requirement (PDF verbatim) | Satisfied Where | Status |
|---:|---|---|:---:|
| i | "how the tabular data are converted into a sequence of tokens" | §5.2 tokenisation strategies; §5.4A PLE; §5.4B MLM-compatible masking; §5.6 full token sequence layout | ✓ |
| ii | "the embedding design" | §5.2.1 numerical linear/PLE; §5.2.2 categorical lookup; §5.2.3 hybrid PAY (N1); §5.3 feature-type embeddings; §5.5 [CLS] | ✓ |
| iii | "how queries, keys, values, and attention weights are computed" | §6.1 full mathematical derivation of scaled dot-product; §6.2 multi-head split and re-projection; §6.12.1/2 optional additive biases on scores | ✓ |
| iv | "the transformer block architecture" | §6.4 PreNorm structure; §6.5 position-wise FFN; §6.11 concrete `src/transformer.py` spec with shapes | ✓ |
| v | "how the model predicts default" | §6.7 [CLS] → LayerNorm → 2-layer MLP → sigmoid; §6.9 full forward-pass diagram | ✓ |
| vi | "the loss function, optimisation method, and training procedure" | §7 focal loss + WBCE + label smoothing; §8 AdamW, cosine warmup, gradient clipping, early stopping; §8.5 MTLM pretraining (N4); §8.6 multi-task auxiliary (N5) | ✓ |

**"Deep understanding of how transformer-based language models are built"** — this clause is satisfied by the inclusion of §8.5 (MTLM pretraining) which is the single most "language-model-like" piece of the project. Markers can see that we understand transformers are not just attention stacks but pretraining-plus-fine-tuning systems.

### 21.4 Section 4 Detailed Requirements (the 30% section)

| # | PDF requirement | Satisfied Where | Status |
|---:|---|---|:---:|
| 18 | "provide the **details of your experiments**" | §10 metrics; §11 ablation A1–A22; §14 statistical testing | ✓ |
| 19 | "**explain the results**" | §15.3 narrative arc; §15.1 §4 row; multi-run 10-seed reporting with medians and IQRs | ✓ |
| 20 | "**discuss the implications** of your findings" | §15.1 §4 row (all 12 sub-sections); §13.5.3 fairness impossibility discussion; §12.11 Jain-Wallace interpretive caveat | ✓ |
| 21 | "discuss the **limitations** of your analysis" | §15.1 §4 row sub-section (k); Phase 14B Model Card "out-of-scope uses"; §18 Risk Register | ✓ |
| 22 | "provide **suggestions for how it could be improved**" | §15.1 §4 row sub-section (l); extensions discussed: deep ensembles (§13.6.4), adversarial fairness (§13.5.4) | ✓ |
| 23 | "**is the transformer-based approach even the right model for this problem?**" | §15.1 §4 row sub-section (j) — **dedicated named sub-section**, backed by Ablation A17 (random-attention null) + A18 (linear-probe floor) + A19 (scaling curve) + Grinsztajn et al. (2022) | ✓ |
| 24 | "A **clear summary comparing the performance** of the two models must be included" | §10.4 side-by-side model-comparison table; §15.1 §4 row sub-section (h); §15.5 Table 4 (comprehensive metrics) | ✓ |

### 21.5 Acknowledgements Requirements

| # | PDF requirement | Satisfied Where | Status |
|---:|---|---|:---:|
| 25 | "The acknowledgements section must be **no more than 50 words**" | §15.1 Acknowledgements row — 50-word hard cap; contribution table excluded (tables not in word count) | ✓ |
| 26 | "you must **acknowledge this clearly and state how** [LLM usage] was used" | §1.3.1 template provided; §15.1 Acknowledgements row mandatory content (a) | ✓ |
| 27 | "include a **table in the Acknowledgements section listing the group members** and describing how each person contributed to the project" | §15.5 Table 7 — Contribution breakdown; §1.3.1 emphasises substantive per-member entries | ✓ |

### 21.6 Presentation / Marking Criteria Requirements

| # | PDF requirement | Satisfied Where | Status |
|---:|---|---|:---:|
| 28 | "All figures and tables must be clearly **labelled and referenced in the text**" | §15.2 Writing Quality Standards — every figure/table must be numbered, captioned, and cross-referenced ≥ 1× in body text | ✓ |
| 29 | "You may use any standard referencing style, but you should be **consistent** throughout" | §15.2 pins APA-7 as the single style | ✓ |
| 30 | "For the data exploration section, try to use **visualisations that help the reader understand** your overall methodology and findings" | §4 Phase 2 already done — 12 figures produced, each tied to a modelling implication in §4.1 table | ✓ |
| 31 | "these marking criteria are interconnected" — strong methodology lost if not clearly explained | §15.2 writing standards; §15.3 narrative arc; §15.4 figure plan; §15.5 table plan | ✓ |

### 21.7 Submission / Logistics Requirements

| # | PDF requirement | Satisfied Where | Status |
|---:|---|---|:---:|
| 32 | "One submission will be allowed per group ... by the group lead" | §1.3.1 submission logistics | ✓ |
| 33 | "A group may be asked to attend a **face-to-face meeting** to explain the project and the results. In that case, all group members must be present." | §1.3.1 F2F readiness — every member must be able to defend every design decision; weekly walkthroughs planned | ✓ |
| 34 | "all group members are expected to make a meaningful contribution ... concern should be reported to the module lead as early as possible and normally no later than **11 calendar days**" | §1.3.1 — dated contribution log from day 1; distinct git authorship; evidence trail maintained | ✓ |

### 21.8 Strict Verdict: all 34 PDF requirements are explicitly satisfied by specific plan sections.

### 21.9 Extensions Beyond the Minimum — Each Justified Against Marking Criteria

The coursework PDF's main marking criteria for Sections 3 and 4 are explicitly: **"model design, novelty, independent thought, methodology, and reasoning"**. Any plan item that is not strictly required by the PDF must be defensible against these criteria. Below, every extension is audited.

| Extension | Not-strictly-required-by-PDF, but earns marks for … |
|---|---|
| **MTLM pretraining (Phase 6A, N4)** | "Deep understanding of how transformer-based language models are built" — MTLM *is* the defining pretraining method of the LLM paradigm the PDF explicitly frames the task around. Strongest novelty bet. |
| **Multi-task PAY aux head (Phase 6B, N5)** | Methodology / reasoning — demonstrates principled use of auxiliary supervision |
| **Feature-group attention bias (N2)** | Model design / novelty — credit-risk-specific inductive bias |
| **Temporal-decay positional prior (N3)** | Model design / novelty — recency prior tied to EDA finding |
| **Hybrid PAY tokenisation (N1)** | Tokenisation design / novelty — strong justification satisfies PDF sub-req (i) with depth |
| **PLE numerical encoding (§5.4A)** | Ablation rigour — tests whether a more sophisticated numerical encoding helps |
| **Counterfactual explanations (Phase 10A, N6)** | Section 4 interpretability; GDPR/AI-Act relevance; "independent thought" |
| **Null baselines A17/A18 (N7)** | "Methodology" — **forces intellectual honesty**; distinguishes our report from work that uncritically claims attention "helps" |
| **Jain-&-Wallace diagnostic (§12.11, N8)** | "Independent thought" + critical engagement with interpretability literature |
| **Business cost-sensitive (Phase 8A, N9)** | Directly addresses PDF sub-req "is the transformer right for this problem?" — needs a business-aware comparator, not just AUC |
| **Subgroup fairness audit (Phase 11A, N10)** | Section 4 "limitations" + "independent thought"; ethical ML is expected at Level 7 |
| **Uncertainty quantification (Phase 11B, N11)** | Section 4 methodology — predictive uncertainty is a deployment concern |
| **Model Card + Data Sheet (Phase 14B, N12)** | "Writing quality, presentation, referencing" — formal responsible-ML artefacts |
| **Integrated Gradients + SHAP on transformer (§12.7–12.8)** | Section 4 interpretability depth |
| **Probing classifiers (§12.9)** | Section 4 "what is the model actually doing?" — strong evidence |
| **CKA layer analysis (§12.10)** | Section 4 methodology depth |
| **Paired bootstrap for $\Delta$ (§14.5)** | "Methodology" — the correct test for paired-sample model comparison |
| **Benjamini-Hochberg FDR (§14.6)** | "Methodology" — multi-comparison correction is expected at Level 7 |
| **Power analysis (§14.7)** | "Methodology" — contextualises null ablation findings |
| **10-seed sweep (Phase 14A)** | "Methodology" — reproducibility stronger than most student reports |
| **Reproducibility guarantees (Phase 14A)** | "Presentation" + implicit marker concern: can they rerun it? |
| **Engineering standards — typing, CI, tests (§16.4)** | "Writing quality, presentation" — code is referenced from the report, so code quality is marked |

**No extension is unjustified.** Every bullet above maps to a named PDF marking phrase.

### 21.10 Items Explicitly Excluded to Avoid Scope Creep

For strictness: the following *were considered* as possible extensions but **explicitly excluded** from the plan because they either (a) dilute the required two-model comparison, (b) add risk without distinction upside, or (c) fall outside the coursework scope.

| Considered but excluded | Reason |
|---|---|
| Multi-model benchmark zoo (LR + XGBoost + TabNet + FT-Transformer as primary comparators) | PDF specifies exactly two models (transformer + RF). Broadening the benchmark is a *distraction* from the required head-to-head comparison. Any additional baselines stay as null lower-bounds (Ablations A17/A18) or ablation context only. |
| Pre-trained foundation models (TabPFN, transfer from other tabular data) | PDF: "Using a pre-built transformer model or any LLM API ... is **not allowed**". Hard no. |
| Distillation of the transformer into a small tree | Out of scope; interesting side-project but not marked. |
| CUDA / Triton kernel implementations of attention | The PDF permits PyTorch primitives; hand-written CUDA adds risk with zero spec alignment. |
| Neural architecture search over transformer configs | Our 22-ablation study covers the relevant search; NAS is overkill and burns time budget. |
| Publication / arXiv preprint from this work | Out of scope for the marked submission. |

### 21.11 Risk-of-Over-Reach Check — Achievability Against Time Budget

The plan is genuinely large. A rational marker would expect not every element to be polished to publication quality. We prioritise by marginal marks impact:

| Priority tier | Items | If time runs out, drop to … |
|---|---|---|
| **Must complete** (≥ 90% of total marks live here) | Phases 1–9 including Phase 6A (MTLM), 6B, 8A, 9 (all 22 ablations), 10 (standard + Jain-Wallace), 11, 12, 13, 14 | Drop *nothing* from this tier. |
| **High priority** (pushes 78 → 85+) | Phase 10A counterfactuals, Phase 11A fairness, §14.5 paired bootstrap, §14.6 BH, Model Card | Can defer the adversarial-fairness mitigation (§13.5.4) if pressed |
| **Polish** (85 → 90) | Phase 11B UQ (MC dropout is 1 day of work), CKA, Probing, §14.7 power analysis, Data Sheet | Can defer deep-ensemble UQ variant; can defer §13.5.4 adversarial fairness |
| **Extension** (marginal) | Stochastic depth, label smoothing gamma sweep, intersection-subgroup analysis | Drop first if squeezed |

This plan is a **superset**; the report draws from whichever subset of it actually gets executed in time. Scheduling in §19 places the must-complete tier on the critical path.

---

*End of project plan.*
