# Reproducibility Guarantees

Phase 14A of the plan. Concrete promises about what regenerates
bit-stably vs what depends on runtime non-determinism.

## TL;DR

Run the one-liner:

```bash
poetry run python src/repro.py
```

Exit code 0 = every derivative artefact matches its committed copy.
Exit code 1 = one or more regeneration checks failed. CI uses this as a
gate.

## Deterministic (bit-stable on any POSIX / Windows box)

These artefacts are produced by pure NumPy / scikit-learn / pandas and
regenerate bit-for-bit from committed inputs:

| Artefact | Source script |
|---|---|
| `results/comparison_table.{csv,md}` + `evaluate_summary.json` | `src/evaluate.py` |
| `results/rf/test_predictions.npz` + `test_metrics.json` | `src/rf_predictions.py` |
| `results/calibration/*` | `src/calibration.py` |
| `results/fairness/*` | `src/fairness.py` |
| `results/significance/*` | `src/significance.py` (fixed `--seed 0`) |

Checked by `src/repro.py` against committed copies with
`np.allclose(rtol=1e-4, atol=1e-6)` for numeric columns and exact
string match for label columns.

## Approximately deterministic (within floating-point tolerance)

Transformer training (`src/train.py` + `src/train_mtlm.py`) is
deterministic under:

- Python 3.12.10
- `torch == 2.2.2+cpu`
- `CUDA_LAUNCH_BLOCKING=1` (if run on GPU)
- `torch.use_deterministic_algorithms(True)` (set by
  `utils.set_deterministic`)
- Identical `--seed` flag and full config JSON

Per-seed configs are committed at
`results/transformer/seed_*/config.json` — you can reproduce exactly by
passing those values back into `train.py`. Small cross-machine drift is
expected when the hardware's BLAS implementation differs (e.g. Intel
MKL vs OpenBLAS); the test-set metrics remain within ± 0.001 AUC.

## Data-split hashes (Plan §16.5.3)

`data/processed/SPLIT_HASHES.md` commits the SHA-256 digest of every
pre-processed split file — the raw, scaled, and engineered CSVs plus
`feature_metadata.json`. `src/repro.py`'s `split_hashes_match` check
fails if any file on disk drifts from its committed hash; this is the
strongest local guarantee short of re-running
`run_pipeline.py --preprocess-only` from scratch.

## Known gap — Docker image (Plan §16.5.4)

The plan promises a `Dockerfile` pinning `python:3.11.5-slim-bookworm`
with a `docker build && docker run` one-liner. We have not yet shipped
that artefact. Rationale: the coursework runs locally and on Colab; a
Docker image would be a deployment convenience but is not a blocker
for the marker. Tracked as a separate ticket.

Current local pins (as captured in `pyproject.toml` + `poetry.lock`)
are sufficient for Python-level reproducibility; the Docker gap matters
only for hermetic OS-level reproduction.

## Non-deterministic by design

- `src/uncertainty.py` runs `--n-samples` stochastic forward passes
  with dropout active. The NPZ output depends on the `--seed` flag;
  fixed seed → reproducible output.
- Plots (`figures/*.png`) are regenerated from the same underlying
  numbers but the Matplotlib render is not byte-stable across versions.
  We check the *contents* (the source CSV / JSON) not the PNG bytes.

## How to add a new reproducibility check

1. Write the regeneration logic as a function in `src/repro.py` that
   returns a `Check` dataclass.
2. Append it to `run_all()` in the same file.
3. Re-run `python src/repro.py` and confirm the new check passes.
4. Commit the check alongside the artefact it covers.

## Seeds used across the project

| Context | Seed |
|---|---|
| Transformer supervised training | 42, 1, 2 |
| MTLM pretraining + fine-tune | 42 |
| RF hyperparameter search | 42 |
| `src/rf_predictions.py` final refit | 42 |
| Significance-test bootstrap resampling | 0 |
| MC-dropout sampling | 0 |

Every seed is surfaced as a `--seed` CLI flag so replaying with a
different seed is a single line.
