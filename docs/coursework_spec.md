# Coursework Two – Group Project

**Weighting: 50%**

## Overview

This coursework is a complete group project. The dataset for this project is the credit card default dataset, which we have used several times in this module and which is available at:

[https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

One of the major developments in artificial intelligence has been the attention mechanism, which led to the development of transformer models and, at larger scale, large language models (LLMs) such as OpenAI ChatGPT and Google Gemini. **The main objective of this coursework is to build a small transformer-based language model from scratch, trained and tested only on the credit card default dataset, to predict default.**

For the purpose of this coursework, a small language model means a model that takes an ordered, tokenised representation of each record and processes it through explicit self-attention using queries, keys, and values in one or more transformer blocks. A standard feed-forward neural network does not satisfy this requirement, even if it is deep. Likewise, simply using a neural network and referring to it as a transformer will not receive credit for the language-model component.

Because the dataset is tabular rather than natural-language text, you must define and justify how each record is converted into a sequence of tokens (for example, feature–value tokens or another clearly motivated tokenisation scheme). Your model must use the attention mechanism as a central component of the architecture.

As a benchmark, you will also build a random forest model and tune its hyperparameters. Therefore, you are required to develop and compare two models:

1. a small transformer-based language model built from scratch; and
2. a random forest benchmark model.

## Report Submission and Structure

You must submit your report as a single PDF file. The report should be structured as follows:

1. Introduction
2. Data Exploration
3. LLM Model Build-up
4. Experiments, Results, and Discussions
5. Conclusions
6. Acknowledgements
7. References
8. Appendices

All code must be submitted through a GitHub repository, with the repository link provided in the report. If you need to include code excerpts in the report, place them in the appendices. You will be allowed to upload only one (PDF) file.

In Section 3, you must explain your methodology. **This is the most important section of the report**, and you need to demonstrate a deep understanding of how transformer-based language models are built.

For this coursework, **from scratch** means that you may use a standard deep-learning framework (such as PyTorch or TensorFlow) for tensors, automatic differentiation, and optimisation, but **the core attention mechanism and transformer block must be implemented and explained by you**. Using a prebuilt transformer model or any LLM API as part of the modelling, training, or inference pipeline is not allowed.

At a minimum, Section 3 must clearly describe and justify:

- i) how the tabular data are converted into a sequence of tokens;
- ii) the embedding design;
- iii) how queries, keys, values, and attention weights are computed;
- iv) the transformer block architecture;
- v) how the model predicts default; and
- vi) the loss function, optimisation method, and training procedure.

A model without explicit self-attention will not satisfy the requirement of this section. You need to explain how attention is used in your model. In Section 4, you must provide the details of your experiments, explain the results, and discuss the implications of your findings. You should also discuss the limitations of your analysis and provide suggestions for how it could be improved. For instance, is the transformer-based approach even the right model for this problem? A clear summary comparing the performance of the two models must be included.

To receive full credit, you must show your full working with detailed investigation and analysis. You may support your discussion with diagrams, formulae, figures, and tables. Your models must be valid and should follow the usual rigour of a data analytics project.

## Appendices

Your appendices, which may contain some Python code, must be well organised and easy to follow. Within the main body of the report, you should make appropriate references to the relevant sections of the appendix or the GitHub repository to support your claims and experiments.

The total word count for the report must not exceed **4000 words**. Programming code, appendix content, formulae, figures, and tables are not included in the word count. The appendices should be divided into sections corresponding to each phase of your analysis so that they are easy to navigate and reference.

## Acknowledgements

The acknowledgements section must be no more than **50 words**. You are allowed to use any large language model (LLM), but you must acknowledge this clearly and state how it was used. You will not be penalised for using such tools, provided your submission is not simply copied from an LLM.

## Marking Allocation

| Component | Weight |
|---|---|
| Section 3: Model Build-up | 40% |
| Section 4: Experiments and Results | 30% |
| Section 5: Conclusions | 5% |
| All other sections (including appendices), overall structure, writing quality, referencing, and presentation | 25% |

**The main criteria for Sections 3, 4 are model design, novelty, independent thought, methodology, and reasoning.** For the data exploration section, try to use visualisations that help the reader understand your overall methodology and findings. All figures and tables must be clearly labelled and referenced in the text.

Please note that these marking criteria are interconnected. For example, you may have a strong methodology in Sections 3, but if it is not clearly explained or is difficult for the marker to follow, marks will be deducted.

You may use any standard referencing style, but you should be consistent throughout.

## Further Instructions

One submission will be allowed per group and, unless stated otherwise, the submission should be made by the group lead. Further instructions will be announced in due course.

All group members will receive the same mark. However, your submission must include a breakdown of each member's contribution. Please include a table in the Acknowledgements section listing the group members and describing how each person contributed to the project.

**A group may be asked to attend a face-to-face meeting to explain the project and the results. In that case, all group members must be present.**

## Assessment Criteria

Please refer to the HEQF Level 7 Masters Level criteria in the following document:

[https://www.ucl.ac.uk/teaching-learning/sites/teaching-learning/files/migrated-files/UCL_Assessment_Criteria_Guide.pdf](https://www.ucl.ac.uk/teaching-learning/sites/teaching-learning/files/migrated-files/UCL_Assessment_Criteria_Guide.pdf)

## FAQ

**1. How many words should I allocate to each section?**

As a guideline, you may allocate words roughly in proportion to the weighting of each section. For example, since Section 4 is worth 30%, you might allocate approximately

0.3 × 4000 = 1200

words to that section.

## Possible Contribution Review Meeting

Please note that all group members are expected to make a meaningful contribution to the project. If a group experiences a serious contribution problem, the concern should be reported to the module lead as early as possible and normally no later than 11 calendar days from the start of the project, together with supporting evidence where available (for example, version-control activity, meeting notes, draft contributions, or written task allocations). A concern may be raised by the group lead or by any other group member. In the absence of an earlier reported concern, the normal expectation is that the submitted work will receive a shared group mark. Where a serious concern is raised, the module lead may request individual contribution statements, review the available evidence, and convene a contribution review meeting with some or all group members. Concerns first raised only after submission will normally be considered only where there is clear documentary evidence and a good reason why the issue could not reasonably have been raised earlier. Any decision will be based on the evidence and handled in line with the published assessment arrangements. Where substantial non-contribution is evidenced, the module lead reserves the right to recommend an individual mark adjustment on the basis of documented evidence, student statements, and, where needed, a contribution review meeting. Peer reports may inform the outcome, but will not by themselves determine it.
