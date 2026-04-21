# Data-Split Hashes

SHA-256 of every committed pre-processed split. Reviewers hash the files
on their disk and compare here to confirm they have the same bytes we
trained and evaluated on. `src/infra/repro.py`'s `split_hashes_match` check
fails if anything drifts.

## Layout

The nine split CSVs live under `data/processed/splits/`; `feature_metadata.json`
sits at `data/processed/` root. `check_split_hashes_match` probes both
locations so the ledger below keys on bare filenames — this row format
(`` | `filename` | `sha256` | ``) is parsed by a regex in `repro.py`, do
not change it.

## SHA-256

| File | SHA-256 |
|---|---|
| `train_raw.csv`        | `132dac42b850e87ff2dd015004076204ac0fd68e15787b5f4de69afcf47e1a58` |
| `val_raw.csv`          | `2894d6650d661ba57799552d201de7c0fd97995982beaa9f5c4d85ee69ca59ae` |
| `test_raw.csv`         | `35cbde5b429804af752e7261b1731c7cc2fe621fb38a3e970219560902fe1e3a` |
| `train_scaled.csv`     | `4ed680350b1384f000a1ebe92f5edc5ca1bb7b38307b2a9e3a751b17a960427e` |
| `val_scaled.csv`       | `e8312a76956bfd06ba099d5d3b2bd15ef1f27bf298fa71edd7aab10f572955f7` |
| `test_scaled.csv`      | `bde32f11ed0146be16814702acfb38b445c90f7e839e624802d091474fd0eed0` |
| `train_engineered.csv` | `5c0bcf633c6f58190051bcefc1efb610929a51c8af9d2a5350b1298d7cccf405` |
| `val_engineered.csv`   | `51c50f7777b93de49668524fef8892a678725a4d59eb88be0373e51ad515456e` |
| `test_engineered.csv`  | `94d79cdf52bd0819f64b7b332905160bac2b0739975394e7340f37018bbd24de` |
| `feature_metadata.json` | `7c2825b389bb5be3adab0e5f8127719fc9f8c2dfcd9e577d981bcfdd6f608017` |

## Verification

```bash
python -m src.infra.repro
# Look for [PASS] split_hashes_match.
```

Or manually:

```bash
sha256sum data/processed/splits/*.csv data/processed/feature_metadata.json
```

A mismatch means the split was regenerated under different
`src/data/preprocessing.py` behaviour, a different raw-data fetch, or the
file was edited locally. In all three cases the published metrics are
not directly comparable; regenerate from `scripts/run_pipeline.py --preprocess-only`.
