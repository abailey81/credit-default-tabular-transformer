# results/pipeline/

Per-stage stdout/stderr logs captured by the `scripts/run_all.{sh,ps1}`
driver — a human-readable trail of the most recent end-to-end pipeline run.

## Subfolder

| Folder | Contents |
|---|---|
| `logs/` | One `run_all.<stage>.log` per stage: `data`, `eda`, `rf`, `rf_pred`, `visualise`, `evaluate`, `calibration`, `fairness`, `uncertainty`, `significance`, `interpret`, `repro`. |

Each log captures the merged stdout+stderr of `python -m src.<stage>` for
that run. Overwritten each invocation of `run_all.sh` / `run_all.ps1`.

## Produced by

- [`scripts/run_all.sh`](../../scripts/run_all.sh) (Linux / Git-bash).
- [`scripts/run_all.ps1`](../../scripts/run_all.ps1) (Windows PowerShell).

Both drivers essentially execute the same sequence that
[`src/infra/repro.py`](../../src/infra/repro.py) uses, but without
hashing — they're for "run everything and inspect manually".

## Consumed by

- Developers debugging a stage failure (grep the relevant `.log`).
- CHANGELOG post-mortems when a phase completes.

## Regenerate

```bash
# Linux / macOS / Git-bash
bash scripts/run_all.sh

# Windows PowerShell
pwsh scripts/run_all.ps1
```

Non-deterministic filename timestamps inside the logs; the structured
artefacts these stages produce (under `results/` and `figures/`) remain
bit-stable — see `results/repro/README.md`.
