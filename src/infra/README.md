# `src/infra/` — Reproducibility gate

The "trust-but-verify" layer. Regenerates every committed derivative
artefact and diffs it byte-for-byte (or with a tight numerical
tolerance) against what is on disk. If any check fails, the repo is
not trustworthy for the claims in the report.

## Key modules

| Module    | Purpose |
|---|---|
| `repro.py`| Runs 7 independent checks: `artefacts_exist`, `transformer_run_files`, `split_hashes_match`, `python_pins`, `git_clean`, `rf_predictions_regenerate`, `evaluate_regenerates`. Writes a JSON report under `results/repro/` and an ASCII summary to stdout. |

## Non-obvious dependencies

`repro.py` imports from every other subpackage (it reads everything,
writes only its own JSON). Keep that direction — nothing else should
import `src.infra`. The check for `split_hashes_match` probes
`data/processed/splits/` first and then `data/processed/` (the pre-2026
layout) so it round-trips regardless of which layout is on disk.

## Invocation

```bash
python -m src.infra.repro
# → "All reproducibility checks passed." on a clean repo
```

Also invoked from `scripts/run_all.py` as the final stage.

## Tests

- `tests/infra/test_repro.py` — unit tests for each `Check` helper
  plus a committed-data e2e test that asserts `split_hashes_match`
  passes on a clean `main`.

## Report section

- Appendix (Reproducibility) — `docs/REPRODUCIBILITY.md` is the
  human-readable companion to this module. Every bullet there maps to
  a `Check` name in `repro.py`.
