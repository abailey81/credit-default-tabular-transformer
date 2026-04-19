# results/repro/

Reproducibility harness output — the machine-readable "does
`python -m src.infra.repro` still produce bit-stable artefacts?" report.

## Files

| File | Shows |
|---|---|
| `reproducibility_report.json` | Per-stage PASS/FAIL + hash comparisons: EDA, RF retune, RF predictions, transformer evaluate, calibration, fairness, uncertainty, significance, interpret, visualise. Headline: **7/7 PASS** required before any PR merges. |

## Subfolder

| Folder | Contents |
|---|---|
| `_scratch/` | Side-by-side regenerated copies of the canonical RF and evaluate outputs, kept only to diff against the committed versions. Overwritten each repro run. Contents: `_scratch/rf/{test_metrics.json,test_predictions.npz}` and `_scratch/eval/{comparison_table.csv,comparison_table.md,evaluate_summary.json}`. |

## Produced by

[`src/infra/repro.py`](../../src/infra/repro.py) — the umbrella harness that
re-invokes every `src/<module>` CLI and hashes outputs.

## Consumed by

- CI (`.github/workflows/*`) and local pre-merge checks.
- CHANGELOG entries cite the 7/7 PASS status per phase.

## Regenerate

```bash
poetry run python -m src.infra.repro
```

Deterministic — regenerates bit-stably by construction (that's the point).
If any stage flips to FAIL, diff against `_scratch/` to find the drifting
artefact.
