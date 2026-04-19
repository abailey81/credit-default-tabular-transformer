# Data-Split Hashes

SHA-256 of every committed pre-processed split. Reviewers hash the files
on their disk and compare here to confirm they have the same bytes we
trained and evaluated on. `src/infra/repro.py`'s `split_hashes_match` check
fails if anything drifts.

## SHA-256

| File | SHA-256 |
|---|---|
| `train_raw.csv`        | `252f945b24d677457f197c4526fa0e3dc5f606b0a5277a9121aa9f35a7d6f2db` |
| `val_raw.csv`          | `1dd2108bd5593104608cb066fba22738fd79ff99e49cd87245ec8b5aa098184f` |
| `test_raw.csv`         | `e5664a149a0fb133037fe912e57bcaf87256fcb1db4c3d228ea2c3f63a671c0d` |
| `train_scaled.csv`     | `c40fe1b74e09818e7460123c2b466a189bd78ad4612bb09e7bc4669687707508` |
| `val_scaled.csv`       | `bc2a679191759757c53efda10d35b21a70dcd37894d136e7d373a201a19568df` |
| `test_scaled.csv`      | `62824d95b797c634570559e492ad31bcdb3fe21b2242eaa5e285c9b59c44c6a1` |
| `train_engineered.csv` | `5203e34749556c9f3b59a3c02e01abd801fdc44998bec990d82269959197ef8b` |
| `val_engineered.csv`   | `2666163bf132899ee173e68906a7024ee09b7b80f44ac97989a55214052bb0e0` |
| `test_engineered.csv`  | `6f36c32e1d507c92646d059fb573c075a2125646f29eaccf5b7810f670656a74` |
| `feature_metadata.json` | `6456f6896ac1b4bbf9814fe5b09579452f12b198da18f1158cab18c44d2a1960` |

## Verification

```bash
python -m src.infra.repro
# Look for [PASS] split_hashes_match.
```

Or manually:

```bash
sha256sum data/processed/*.csv data/processed/feature_metadata.json
```

A mismatch means the split was regenerated under different
`src/data/preprocessing.py` behaviour, a different raw-data fetch, or the
file was edited locally. In all three cases the published metrics are
not directly comparable; regenerate from `scripts/run_pipeline.py --preprocess-only`.
