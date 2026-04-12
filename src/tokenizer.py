"""
tokenizer.py — Tabular tokenisation for the Credit Card Default dataset.

Converts each row of the dataset into a sequence of tokens for the transformer.

Hybrid tokenisation scheme:
    1. Categorical features (SEX, EDUCATION, MARRIAGE, PAY_0–PAY_6):
       Feature-value tokens — each unique combination (e.g. SEX_1, PAY_0_-2)
       gets its own learnable embedding.
    2. Numerical features (LIMIT_BAL, AGE, BILL_AMT1–6, PAY_AMT1–6):
       Feature embedding + value projection — the feature name gets an embedding
       and the scaled value is passed through a linear layer.

Result: 23 tokens per client (9 categorical + 14 numerical).
"""

import json
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Categorical: each value becomes a separate token (feature-value tokenisation).
# PAY statuses are included here because values -2/-1/0 are categories (no card
# use, paid in full, revolving credit) with no numerical relationship between them.
CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]
PAY_STATUS_FEATURES = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
ALL_CATEGORICAL = CATEGORICAL_FEATURES + PAY_STATUS_FEATURES  # 3 + 6 = 9

# Numerical: feature embedding + value projection.
# These are true numbers where distance matters (AGE=35 is close to AGE=36).
BILL_AMT_FEATURES = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_FEATURES = [f"PAY_AMT{i}" for i in range(1, 7)]
NUMERICAL_FEATURES = ["LIMIT_BAL", "AGE"] + BILL_AMT_FEATURES + PAY_AMT_FEATURES  # 14

TARGET_COL = "DEFAULT"


# ──────────────────────────────────────────────────────────────────────────────
# Block 1: Categorical token vocabulary
# ──────────────────────────────────────────────────────────────────────────────

def build_categorical_vocab(metadata: Dict) -> Dict[str, int]:
    """
    Build a mapping from token names to integer indices for all categorical features.

    Each unique (feature, value) pair becomes one token. For example:
        SEX=1       → token "SEX_1"       → index 0
        SEX=2       → token "SEX_2"       → index 1
        EDUCATION=1 → token "EDUCATION_1" → index 2
        ...
        PAY_0=-2    → token "PAY_0_-2"    → index 9
        PAY_0=-1    → token "PAY_0_-1"    → index 10
        ...

    The total vocabulary size determines the Embedding table dimensions
    in the transformer (vocab_size × d_model).

    Args:
        metadata: Contents of feature_metadata.json. Contains the list of
                  possible values for each feature (learned from training data).

    Returns:
        Dictionary mapping token name (str) to unique index (int).
    """
    vocab = {}
    idx = 0

    # --- Demographic categoricals (SEX, EDUCATION, MARRIAGE) ---
    for feature in CATEGORICAL_FEATURES:
        # metadata structure: {"categorical_features": {"SEX": {"value_to_idx": {"1": 0, "2": 1}}}}
        value_map = metadata["categorical_features"][feature]["value_to_idx"]
        for value in sorted(value_map.keys(), key=lambda x: int(x)):
            token_name = f"{feature}_{value}"   # e.g. "SEX_1", "EDUCATION_3"
            vocab[token_name] = idx
            idx += 1

    # --- PAY status categoricals (PAY_0, PAY_2, ..., PAY_6) ---
    for feature in PAY_STATUS_FEATURES:
        # metadata structure: {"pay_features": {"PAY_0": {"value_to_idx": {"-2": 0, "-1": 1, ...}}}}
        value_map = metadata["pay_features"][feature]["value_to_idx"]
        for value in sorted(value_map.keys(), key=lambda x: int(x)):
            token_name = f"{feature}_{value}"   # e.g. "PAY_0_-2", "PAY_0_0"
            vocab[token_name] = idx
            idx += 1

    return vocab


# ──────────────────────────────────────────────────────────────────────────────
# Block 2: Numerical feature vocabulary
# ──────────────────────────────────────────────────────────────────────────────

def build_numerical_vocab() -> Dict[str, int]:
    """
    Build a mapping from numerical feature names to integer indices.

    Unlike categorical features, numerical features don't need a token per value.
    Instead, each feature gets ONE index (its identity), and the actual number
    is handled separately by a linear projection layer in the transformer.

    For example:
        "LIMIT_BAL" → index 0    (the model learns: "I am credit limit")
        "AGE"       → index 1    (the model learns: "I am age")
        "BILL_AMT1" → index 2    (the model learns: "I am September bill")
        ...
        "PAY_AMT6"  → index 13

    The actual values (50000, 35, 12000, ...) are passed separately as floats
    and go through a Linear layer: value × weight + bias → vector of size d_model.

    Returns:
        Dictionary mapping feature name (str) to index (int).
    """
    vocab = {}
    for idx, feature in enumerate(NUMERICAL_FEATURES):
        vocab[feature] = idx

    return vocab


# ──────────────────────────────────────────────────────────────────────────────
# Block 3: Tokenize one row
# ──────────────────────────────────────────────────────────────────────────────

def tokenize_row(
    row: pd.Series,
    cat_vocab: Dict[str, int],
) -> Tuple[List[int], List[float], int]:
    """
    Convert a single DataFrame row (one client) into token indices and values.

    Steps:
        1. For each categorical feature: look up its value, build the token name
           (e.g. "SEX_2"), and find the index in cat_vocab.
        2. For each numerical feature: store the raw value. The position in the
           list implicitly identifies the feature (always in NUMERICAL_FEATURES order),
           so per-row feature index lists are unnecessary — the transformer's
           embedding module generates the shared index tensor torch.arange(14)
           internally.
        3. Read the target label (DEFAULT).

    Args:
        row:       One row from a pandas DataFrame (e.g. df.iloc[0]).
        cat_vocab: Categorical vocabulary from build_categorical_vocab().

    Returns:
        cat_token_ids: List of 9 ints   — indices into categorical embedding table.
        num_values:    List of 14 floats — scaled numerical values, in the order
                                            defined by NUMERICAL_FEATURES.
        label:         int (0 or 1)     — default / no default.
    """
    # --- Step 1: Categorical features → token indices ---
    cat_token_ids = []
    for feature in ALL_CATEGORICAL:
        value = row[feature]                          # e.g. SEX → 2
        token_name = f"{feature}_{int(value)}"        # e.g. "SEX_2"
        cat_token_ids.append(cat_vocab[token_name])   # e.g. "SEX_2" → 1

    # --- Step 2: Numerical features → values only ---
    # Order is fixed by NUMERICAL_FEATURES, so position carries feature identity.
    num_values = [float(row[feature]) for feature in NUMERICAL_FEATURES]

    # --- Step 3: Target label ---
    label = int(row[TARGET_COL])                      # 0 or 1

    return cat_token_ids, num_values, label


# ──────────────────────────────────────────────────────────────────────────────
# Block 4: PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────

class CreditDefaultDataset(Dataset):
    """
    PyTorch Dataset that wraps a preprocessed DataFrame.

    On creation (__init__): tokenizes every row using tokenize_row() and stores
    the results in lists. This is done once, not on every access.

    On access (__getitem__): returns one client's data as PyTorch tensors,
    ready for the transformer.

    Usage:
        metadata = json.load(open("data/processed/feature_metadata.json"))
        cat_vocab = build_categorical_vocab(metadata)

        train_df = pd.read_csv("data/processed/train_scaled.csv")
        dataset = CreditDefaultDataset(train_df, cat_vocab)

        # Single client
        sample = dataset[0]

        # With DataLoader (automatic batching and shuffling)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cat_vocab: Dict[str, int],
    ):
        """
        Tokenize all rows in the DataFrame and store results.

        Args:
            df:        Preprocessed DataFrame (e.g. train_scaled.csv loaded as DataFrame).
            cat_vocab: Categorical vocabulary from build_categorical_vocab().
        """
        self.all_cat_ids = []       # will hold N lists of 9 ints each
        self.all_num_vals = []      # will hold N lists of 14 floats each
        self.all_labels = []        # will hold N ints

        # Tokenize every row once at creation time
        for i in range(len(df)):
            row = df.iloc[i]
            cat_ids, num_vals, label = tokenize_row(row, cat_vocab)
            self.all_cat_ids.append(cat_ids)
            self.all_num_vals.append(num_vals)
            self.all_labels.append(label)

        print(f"[TOKENIZER] Tokenized {len(self.all_labels)} rows "
              f"({sum(self.all_labels)} defaults, "
              f"{len(self.all_labels) - sum(self.all_labels)} non-defaults)")

    def __len__(self) -> int:
        """Return the number of clients in this dataset."""
        return len(self.all_labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Return one client's tokenized data as PyTorch tensors.

        Args:
            idx: Client index (0 to len-1).

        Returns:
            Dictionary with three tensors:
                cat_token_ids: [9]  — long tensor, indices for categorical embedding
                num_values:    [14] — float tensor, scaled numerical values
                                       (position implies feature identity)
                label:         []   — float tensor, 0.0 or 1.0
        """
        return {
            "cat_token_ids": torch.tensor(self.all_cat_ids[idx], dtype=torch.long),
            "num_values":    torch.tensor(self.all_num_vals[idx], dtype=torch.float),
            "label":         torch.tensor(self.all_labels[idx], dtype=torch.float),
        }
