"""
Credit Card Default Prediction — EDA, Preprocessing & Random Forest Benchmark.

Modules
-------
* :mod:`data_sources`        — resilient multi-source data loader
                                (UCI ML Repository API → local manual ``.xls`` fallback).
* :mod:`data_preprocessing`  — schema normalisation, cleaning, feature engineering,
                                stratified splitting, leak-free scaling, metadata export.
* :mod:`eda`                 — 12 publication-quality EDA figures with statistical tests.
* :mod:`random_forest`       — hyperparameter-tuned Random Forest benchmark.

Every consumer of the dataset routes through :func:`data_sources.build_default_data_source`,
so the same API → local fallback semantics apply across the entire pipeline.
"""
