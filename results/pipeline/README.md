# results/pipeline/

> **Breadcrumb**: [↑ repo root](../../) > [↑ results](../) > **pipeline/**

**Per-stage pipeline run logs** — stdout/stderr captured by the `scripts/run_all.{sh,ps1}` driver. A human-readable trail of the most recent end-to-end run, consumed by developers debugging stage failures and cited in CHANGELOG post-mortems.

These logs are not part of Section 2-4 of the report (they are developer-facing) but are consumed by Appendix 8 (Reproducibility) debugging flows. Structured artefacts under [`../../results/`](../) and [`../../figures/`](../../figures/) remain bit-stable; only the log timestamps inside these files are non-deterministic.

## What's here

| Subfolder | Contents |
|---|---|
| [`logs/`](logs/) | One `run_all.<stage>.log` per stage: `data`, `eda`, `rf`, `rf_pred`, `visualise`, `evaluate`, `calibration`, `fairness`, `uncertainty`, `significance`, `interpret`, `repro`. Merged stdout+stderr of `python -m src.<stage>`. Overwritten each `run_all.sh` / `run_all.ps1` invocation. |

## How it was produced

- [`scripts/run_all.sh`](../../scripts/run_all.sh) (Linux / Git-bash).
- [`scripts/run_all.ps1`](../../scripts/run_all.ps1) (Windows PowerShell).

Both drivers execute the same sequence that [`src/infra/repro.py`](../../src/infra/repro.py) uses, but without hashing — they exist for "run everything and inspect manually".

## How it's consumed

- Developers debugging a stage failure (grep the relevant `.log`).
- CHANGELOG post-mortems when a phase completes.

## How to regenerate

```bash
# Linux / macOS / Git-bash
bash scripts/run_all.sh

# Windows PowerShell
pwsh scripts/run_all.ps1
```

Non-deterministic filename timestamps inside the logs; the structured artefacts these stages produce remain bit-stable — see [`../repro/README.md`](../repro/README.md).

## Neighbours

- **↑ Parent**: [`../`](../) — results/ index
- **↔ Siblings**: [`../analysis/`](../analysis/), [`../baseline/`](../baseline/), [`../transformer/`](../transformer/), [`../mtlm/`](../mtlm/), [`../evaluation/`](../evaluation/), [`../repro/`](../repro/)
- **↓ Children**: none (only `logs/`, an ephemeral log directory with no README)
