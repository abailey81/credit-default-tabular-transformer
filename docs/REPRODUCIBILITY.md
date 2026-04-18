# Reproducibility Guarantees

What regenerates bit-stably from the committed source, what only
regenerates up to floating-point tolerance, and what is
non-deterministic by design.

Run `poetry run python src/repro.py`. Exit 0 means every derivative
artefact matches its committed copy; exit 1 flags at least one
regeneration failure. CI uses this as a gate.

## Deterministic (bit-stable on POSIX and Windows)

These artefacts come out of pure NumPy, scikit-learn, and pandas, and
regenerate bit-for-bit from committed inputs:

| Artefact | Source script |
|---|---|
| `results/comparison_table.{csv,md}` + `evaluate_summary.json` | `src/evaluate.py` |
| `results/rf/test_predictions.npz` + `test_metrics.json` | `src/rf_predictions.py` |
| `results/calibration/*` | `src/calibration.py` |
| `results/fairness/*` | `src/fairness.py` |
| `results/significance/*` | `src/significance.py` (`--seed 0`) |

`src/repro.py` checks these against committed copies with
`np.allclose(rtol=1e-4, atol=1e-6)` on numeric columns and exact
string match on label columns.

## Approximately deterministic (within float tolerance)

Transformer training (`src/train.py` and `src/train_mtlm.py`) is
deterministic under Python 3.12.10, `torch == 2.2.2+cpu`,
`CUDA_LAUNCH_BLOCKING=1` on GPU, `torch.use_deterministic_algorithms(True)`
(set by `utils.set_deterministic`), and matching `--seed` + config JSON.

Per-seed configs live at `results/transformer/seed_*/config.json`, so
training can be replayed by feeding those values back into `train.py`.
Small cross-machine drift is expected when BLAS differs (Intel MKL vs
OpenBLAS); test metrics stay within ±0.001 AUC.

## Data-split hashes

`data/processed/SPLIT_HASHES.md` commits the SHA-256 digest of every
pre-processed split file (raw, scaled, engineered CSVs plus
`feature_metadata.json`). The `split_hashes_match` check in
`src/repro.py` fails if any committed file drifts from its hash — the
strongest local guarantee short of rerunning
`run_pipeline.py --preprocess-only`.

## Non-deterministic by design

`src/uncertainty.py` runs `--n-samples` stochastic forward passes with
dropout active; the NPZ output is seed-controlled and reproducible at
a fixed `--seed`. Plots under `figures/` come from the same numbers,
but matplotlib renders aren't byte-stable across versions — we check
the source CSV/JSON rather than the PNG bytes.

## Known gap: Docker image

The plan (§16.5.4) promises a `Dockerfile` pinning
`python:3.11.5-slim-bookworm` with a single `docker build && docker
run` command. It has not shipped. The coursework runs locally and on
Colab, and the `pyproject.toml`/`poetry.lock` pins are sufficient for
Python-level reproducibility; Docker matters only for hermetic OS-level
reproduction and is tracked separately.

## Adding a new reproducibility check

Write the regeneration logic as a function in `src/repro.py` that
returns a `Check`. Append it to `run_all()`. Run `python src/repro.py`
and confirm it passes. Commit the check alongside the artefact it
covers.

## Seeds used across the project

| Context | Seed |
|---|---|
| Transformer supervised training | 42, 1, 2 |
| MTLM pretraining + fine-tune | 42 |
| RF hyperparameter search | 42 |
| `src/rf_predictions.py` final refit | 42 |
| Significance-test bootstrap | 0 |
| MC-dropout sampling | 0 |

Every seed is exposed as a `--seed` CLI flag, so replaying with a
different seed is a one-line change.
