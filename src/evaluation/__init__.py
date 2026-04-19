"""Evaluation subpackage for §4 of the report.

Everything that consumes trained-model artefacts (predictions, attention
weights, training logs) and turns them into publication-ready numbers and
figures lives here. The subpackage is deliberately leaf-node: it depends on
``src.training``, ``src.models``, and ``src.data``, but nothing in this
package depends on it — train.py / random_forest.py finish first, then any
of these modules can run in any order off the same artefacts.

Module map
----------
* ``evaluate``      -- comparison-table builder + ensemble_run + aggregate_runs.
* ``visualise``     -- the five canonical comparison figures (ROC, PR, CM, curves, reliability).
* ``calibration``   -- post-hoc calibrators (identity / temperature / Platt / isotonic) + ECE/MCE/Brier.
* ``fairness``      -- subgroup audit on SEX / EDUCATION / MARRIAGE (N10).
* ``uncertainty``   -- MC-dropout predictive / aleatoric / BALD and refuse-to-predict curves (N11).
* ``significance``  -- McNemar, DeLong, paired bootstrap, BH-FDR, Hanley-McNeil power.
* ``interpret``     -- attention rollout, CLS feature bars, per-head entropy, class-conditional maps.

Each module ships a ``main(argv)`` CLI (invoked via ``python -m
src.evaluation.<module>``) that reads from ``results/`` and writes to
``results/evaluation/<module>/`` + ``figures/evaluation/<module>/``.
"""
