# `data/raw/` — raw dataset files

The tracked local fallback for the UCI "Default of Credit Card Clients"
dataset.

## Contents

| File | Size | SHA-256 |
| --- | --- | --- |
| `default_of_credit_card_clients.xls` | 5,539,328 bytes | `cf5aefceae81d4409366ca5987898889bd993946836745abf36df028b2470499` |

The `.xls` ships with a banner row at row 0 (`X1 | X2 | …`); the real column
headers sit on row 1. `src/data/sources.py::LocalExcelSource` passes
`header=1` to `pd.read_excel` to skip the banner.

## How to refresh

The canonical loader (`src/data/sources.py::build_default_data_source`)
hits the live UCI repo (id 350) first and only drops to this file on
`ConnectionError` / `TimeoutError` / `FileNotFoundError`. To rebuild the
local copy from the UCI API and persist it back to this path, run the
preprocessing pipeline with `--source auto`:

```bash
python scripts/run_pipeline.py --preprocess-only --source auto
```

Keep the filename and SHA-256 in sync with `data/README.md` and
`docs/DATA_SHEET.md` on any refresh.
