# data/raw/

> **Breadcrumb**: [↑ repo root](../../) > [↑ data](../) > **raw/**

**Raw dataset fallback** — tracked local copy of the UCI "Default of Credit Card Clients" `.xls` used when the live UCI API is unreachable. Referenced from Section 2 (Data) of the report and Appendix 8 (Reproducibility).

The `.xls` ships with a banner row at row 0 (`X1 | X2 | ...`); the real column headers sit on row 1. [`src/data/sources.py`](../../src/data/sources.py) (`LocalExcelSource`) passes `header=1` to `pd.read_excel` to skip the banner. Keeping this file under version control means a clean clone can reproduce the full pipeline without external network access.

## What's here

| File | Contents |
|---|---|
| [`default_of_credit_card_clients.xls`](default_of_credit_card_clients.xls) | 5,539,328 bytes; SHA-256 `cf5aefceae81d4409366ca5987898889bd993946836745abf36df028b2470499`. UCI dataset id 350 (Taiwan 2005), 30,000 rows. |

## How it was produced

Downloaded once from the UCI ML Repository and committed verbatim. No transformation.

```bash
# To refresh from the live UCI API and persist back to this path:
poetry run python scripts/run_pipeline.py --preprocess-only --source auto
```

## How it's consumed

- [`src/data/sources.py`](../../src/data/sources.py) (`build_default_data_source`) reads this as fallback on `ConnectionError` / `TimeoutError` / `FileNotFoundError`.
- [`docs/DATA_SHEET.md`](../../docs/DATA_SHEET.md) cites the SHA-256 as provenance.
- [`data/README.md`](../README.md) mirrors the hash.

## How to regenerate

Do not regenerate — the committed bytes are the provenance contract. Only refresh by rerunning the `--source auto` preprocessing pipeline (above), and keep the filename + SHA-256 in sync with [`data/README.md`](../README.md) and [`docs/DATA_SHEET.md`](../../docs/DATA_SHEET.md).

## Neighbours

- **↑ Parent**: [`../`](../) — data/ root
- **↔ Siblings**: [`../processed/`](../processed/)
- **↓ Children**: none
