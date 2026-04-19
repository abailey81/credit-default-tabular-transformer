"""Classical-ML baselines used as reference points for the transformer.

Currently hosts two modules:

* :mod:`~src.baselines.random_forest` — the full tuned RandomForest benchmark:
  200-iter RandomizedSearchCV, 5-fold CV on the winner, feature-importance
  (Gini + permutation), threshold optimisation, and the diagnostic figures
  committed under ``figures/baseline/``. This is the benchmark the abstract,
  CHANGELOG, and plan all quote.
* :mod:`~src.baselines.rf_predictions` — a lightweight shim that re-reads
  ``results/baseline/rf_config.json``, refits the tuned RF, and dumps
  per-row test probabilities as ``test_predictions.npz`` in the same layout
  ``train.py`` uses. Exists so the calibration / significance / SHAP
  workflows in phases 11-12 can consume the RF outputs through the same
  interface as every transformer variant — without having to re-run the
  whole 200-iter tuning sweep (~minutes vs ~hours).

Design note: no sklearn / pandas symbols are re-exported at the package level.
Downstream modules import directly from the specific sub-module, which keeps
the import side-effects (matplotlib backend, sklearn warnings filter) out of
:mod:`src.baselines` itself."""
