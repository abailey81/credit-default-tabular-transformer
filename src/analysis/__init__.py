"""Statistical analysis subpackage — exploratory data analysis (§2 of report).

Currently a single module, ``eda``, which produces the twelve figures and
the per-feature summary-statistics table that back the EDA chapter. The
split from ``src.data`` is deliberate: anything here is descriptive and
can be regenerated without touching the modelling pipeline, and anything in
``src.data`` is used during training and therefore must stay lightweight.

Run end-to-end via ``python -m src.analysis.eda`` or programmatically via
:func:`src.analysis.eda.run_eda`.
"""
