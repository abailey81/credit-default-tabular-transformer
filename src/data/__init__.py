"""Data ingestion and preprocessing subpackage.

Two modules, cleanly separated by concern:

* :mod:`src.data.sources`       — raw I/O. Knows about the UCI API, about
                                  local ``.xls`` fallbacks, and about
                                  provenance tracking. Returns a single
                                  ``DataSourceResult`` regardless of which
                                  source won.
* :mod:`src.data.preprocessing` — everything after the raw frame: schema
                                  normalisation, cleaning of undocumented
                                  codes, validation, 22-feature engineering,
                                  stratified 70/15/15 split, leak-free
                                  StandardScaler, metadata export.

The split matters: ``sources`` is where a flaky network or missing file
surfaces; ``preprocessing`` is where modelling-specific invariants (no NaN,
PAY values in [-2, 8], train-only scaler fits) are enforced. Consumers
that only want a frame should call
``src.data.preprocessing.run_preprocessing_pipeline`` — it chains both.
"""
