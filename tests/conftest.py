"""Session fixtures — never mutate the cached preprocessed frame."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DATA_DIR = REPO_ROOT / "data" / "processed"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def metadata() -> Dict:
    path = DATA_DIR / "feature_metadata.json"
    if not path.is_file():
        pytest.skip(
            f"feature_metadata.json missing at {path} — run "
            "`poetry run python run_pipeline.py --preprocess-only` first"
        )
    return json.loads(path.read_text())


@pytest.fixture(scope="session")
def train_df(repo_root) -> pd.DataFrame:
    path = DATA_DIR / "train_scaled.csv"
    if not path.is_file():
        pytest.skip(f"train_scaled.csv missing at {path}")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def train_df_small(train_df) -> pd.DataFrame:
    return train_df.head(128).reset_index(drop=True)


@pytest.fixture(scope="session")
def cat_vocab(metadata) -> Dict[str, Dict[int, int]]:
    from tokenizer import build_categorical_vocab

    return build_categorical_vocab(metadata)


@pytest.fixture(scope="session")
def small_dataset(train_df_small, cat_vocab):
    from tokenizer import CreditDefaultDataset

    return CreditDefaultDataset(train_df_small, cat_vocab, verbose=False)


@pytest.fixture()
def seeded_rng():
    return torch.Generator().manual_seed(42)
