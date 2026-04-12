"""
tokenizer.py — Tabular tokenisation for the Credit Card Default dataset.

Converts each row of the dataset into a sequence of tokens for the transformer.

Hybrid tokenisation scheme (three feature types):
    1. Demographic categoricals (SEX, EDUCATION, MARRIAGE):
       Feature-value tokens — each unique combination (e.g. SEX_1, EDUCATION_3)
       gets its own learnable embedding.
    2. PAY status — hybrid state + severity:
       Each PAY value (-2 to 8) is mapped to one of FOUR semantic states:
           -2 → "no_bill"     (no card use that month)
           -1 → "paid_full"   (paid balance in full)
            0 → "minimum"     (revolving credit / minimum payment)
         1..8 → "delinquent"  (delayed by N months)
       The state gets an embedding from a small table. For "delinquent" we
       additionally pass severity = value / 8 through a linear projection.
       This gives the model an inductive bias: PAY=3 and PAY=6 are clearly
       both delinquencies (same state embedding), but PAY=6 is more severe
       (larger severity input). PAY=-2/-1/0 carry severity = 0.
    3. Numerical features (LIMIT_BAL, AGE, BILL_AMT1–6, PAY_AMT1–6):
       Feature embedding + value projection — the feature name gets an embedding
       and the scaled value is passed through a linear layer.

Result: 23 tokens per client (3 demographic + 6 PAY + 14 numerical).
"""

import json
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Demographic categoricals: feature-value tokenisation.
CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]

# PAY status features: hybrid (state + severity).
PAY_STATUS_FEATURES = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

# Numerical features: feature embedding + value projection.
BILL_AMT_FEATURES = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_FEATURES = [f"PAY_AMT{i}" for i in range(1, 7)]
NUMERICAL_FEATURES = ["LIMIT_BAL", "AGE"] + BILL_AMT_FEATURES + PAY_AMT_FEATURES  # 14

TARGET_COL = "DEFAULT"

# PAY value → semantic state index. Used by tokenize_row().
# Four states total — see module docstring for the rationale.
PAY_STATE_NO_BILL = 0     # PAY = -2
PAY_STATE_PAID_FULL = 1   # PAY = -1
PAY_STATE_MINIMUM = 2     # PAY =  0
PAY_STATE_DELINQUENT = 3  # PAY =  1..8
N_PAY_STATES = 4

# Maximum delinquency severity in the dataset (PAY values go up to 8).
# Used to normalise severity into [0, 1] before the linear projection.
MAX_PAY_DELAY = 8


# ──────────────────────────────────────────────────────────────────────────────
# Block 1: Per-feature categorical vocabularies (demographic only)
# ──────────────────────────────────────────────────────────────────────────────

def build_categorical_vocab(metadata: Dict) -> Dict[str, Dict[int, int]]:
    """
    Build a separate vocabulary for each demographic categorical feature
    (SEX, EDUCATION, MARRIAGE).

    Each feature gets its own small mapping from raw value → local index.
    For example:
        SEX:       {1: 0, 2: 1}
        EDUCATION: {1: 0, 2: 1, 3: 2, 4: 3}
        MARRIAGE:  {1: 0, 2: 1, 3: 2}

    This per-feature structure mirrors how FT-Transformer (Gorishniy et al., 2021)
    organises categorical embeddings: instead of one shared lookup table with
    offset tracking, each feature has its own nn.Embedding(n_categories, d_model)
    in an nn.ModuleDict on the model side. It is cleaner, more extensible, and
    each feature's vocabulary size is explicit at construction time.

    Note: PAY statuses are NOT included here — they use a separate hybrid scheme
    (state embedding + severity projection) — see encode_pay_value() below.

    Args:
        metadata: Contents of feature_metadata.json. Contains the list of
                  possible values for each feature (learned from training data).

    Returns:
        Nested dictionary: feature_name → {raw_value: local_index}.
    """
    vocab: Dict[str, Dict[int, int]] = {}

    for feature in CATEGORICAL_FEATURES:
        # metadata structure: {"categorical_features": {"SEX": {"value_to_idx": {"1": 0, "2": 1}}}}
        value_map = metadata["categorical_features"][feature]["value_to_idx"]
        # Sort by integer value so the local indices are deterministic
        sorted_values = sorted(value_map.keys(), key=lambda x: int(x))
        vocab[feature] = {int(value): local_idx for local_idx, value in enumerate(sorted_values)}

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
# Helper: encode one PAY value into (state_id, severity)
# ──────────────────────────────────────────────────────────────────────────────

def encode_pay_value(pay_value: int) -> Tuple[int, float]:
    """
    Map a single PAY value to (state_id, severity) for the hybrid scheme.

    Mapping:
        -2  → (no_bill,      0.0)     no card use
        -1  → (paid_full,    0.0)     paid in full
         0  → (minimum,      0.0)     revolving / minimum payment
       1..8 → (delinquent,   N/8)     N months delayed, severity normalised to [0, 1]

    Severity is normalised by MAX_PAY_DELAY (=8) so that all delinquency values
    fit in a small numeric range — comparable in scale to embedding values.

    Args:
        pay_value: Integer PAY status (range: -2 to 8).

    Returns:
        state_id: int — index into the PAY state embedding table (0-3).
        severity: float — 0.0 for non-delinquent states, value/8 for delinquent.
    """
    if pay_value == -2:
        return PAY_STATE_NO_BILL, 0.0
    if pay_value == -1:
        return PAY_STATE_PAID_FULL, 0.0
    if pay_value == 0:
        return PAY_STATE_MINIMUM, 0.0
    # pay_value in 1..8 → delinquent
    return PAY_STATE_DELINQUENT, pay_value / MAX_PAY_DELAY


# ──────────────────────────────────────────────────────────────────────────────
# Block 3: Tokenize one row
# ──────────────────────────────────────────────────────────────────────────────

def tokenize_row(
    row: pd.Series,
    cat_vocab: Dict[str, Dict[int, int]],
) -> Tuple[Dict[str, int], List[int], List[float], List[float], int]:
    """
    Convert a single DataFrame row (one client) into token indices and values.

    Steps:
        1. Demographic categoricals (SEX, EDUCATION, MARRIAGE) → per-feature local
           indices using each feature's own vocabulary in cat_vocab.
        2. PAY statuses (PAY_0..PAY_6) → (state_id, severity) pairs via the hybrid
           scheme. Two parallel lists of length 6.
        3. Numerical features → values only. Position carries feature identity
           (NUMERICAL_FEATURES order is fixed), so no per-row index list is needed.
        4. Target label (DEFAULT).

    Args:
        row:       One row from a pandas DataFrame (e.g. df.iloc[0]).
        cat_vocab: Per-feature demographic vocabulary from build_categorical_vocab().
                   Structure: {feature_name: {raw_value: local_index}}.

    Returns:
        cat_indices:    Dict mapping feature name → local index, e.g.
                        {"SEX": 1, "EDUCATION": 0, "MARRIAGE": 1}.
                        Each local index goes into that feature's own embedding table.
        pay_state_ids:  List of 6 ints   — PAY state indices (0-3) for each month.
        pay_severities: List of 6 floats — Normalised severity (0.0 if not delinquent).
        num_values:     List of 14 floats — Scaled numerical values, in NUMERICAL_FEATURES order.
        label:          int (0 or 1)     — default / no default.
    """
    # --- Step 1: Demographic categoricals → per-feature local indices ---
    cat_indices: Dict[str, int] = {}
    for feature in CATEGORICAL_FEATURES:
        raw_value = int(row[feature])                  # e.g. SEX → 2
        cat_indices[feature] = cat_vocab[feature][raw_value]  # local index in this feature's table

    # --- Step 2: PAY statuses → state + severity (hybrid) ---
    pay_state_ids = []
    pay_severities = []
    for feature in PAY_STATUS_FEATURES:
        state_id, severity = encode_pay_value(int(row[feature]))
        pay_state_ids.append(state_id)
        pay_severities.append(severity)

    # --- Step 3: Numerical features → values only ---
    # Order is fixed by NUMERICAL_FEATURES, so position carries feature identity.
    num_values = [float(row[feature]) for feature in NUMERICAL_FEATURES]

    # --- Step 4: Target label ---
    label = int(row[TARGET_COL])                      # 0 or 1

    return cat_indices, pay_state_ids, pay_severities, num_values, label


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
        cat_vocab: Dict[str, Dict[int, int]],
    ):
        """
        Tokenize all rows in the DataFrame and store results.

        Args:
            df:        Preprocessed DataFrame (e.g. train_scaled.csv loaded as DataFrame).
            cat_vocab: Per-feature demographic vocabulary from build_categorical_vocab().
        """
        # Store per-feature index lists separately so __getitem__ can build the
        # nested cat_indices dict without re-iterating dictionaries.
        self.all_cat_indices: Dict[str, List[int]] = {f: [] for f in CATEGORICAL_FEATURES}
        self.all_pay_state_ids = []   # will hold N lists of 6 ints each
        self.all_pay_severities = []  # will hold N lists of 6 floats each
        self.all_num_vals = []        # will hold N lists of 14 floats each
        self.all_labels = []          # will hold N ints

        # Tokenize every row once at creation time
        for i in range(len(df)):
            row = df.iloc[i]
            cat_indices, pay_state_ids, pay_severities, num_vals, label = tokenize_row(row, cat_vocab)
            for feature, local_idx in cat_indices.items():
                self.all_cat_indices[feature].append(local_idx)
            self.all_pay_state_ids.append(pay_state_ids)
            self.all_pay_severities.append(pay_severities)
            self.all_num_vals.append(num_vals)
            self.all_labels.append(label)

        print(f"[TOKENIZER] Tokenized {len(self.all_labels)} rows "
              f"({sum(self.all_labels)} defaults, "
              f"{len(self.all_labels) - sum(self.all_labels)} non-defaults)")

    def __len__(self) -> int:
        """Return the number of clients in this dataset."""
        return len(self.all_labels)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        """
        Return one client's tokenized data, ready for the transformer.

        The transformer's nn.ModuleDict of categorical embeddings will look up
        each feature by name, e.g. self.cat_embeddings["SEX"](cat_indices["SEX"]).

        Args:
            idx: Client index (0 to len-1).

        Returns:
            Dictionary with five fields:
                cat_indices:    dict {feature_name: scalar long tensor (local index)}
                                    e.g. {"SEX": tensor(1), "EDUCATION": tensor(0), ...}.
                pay_state_ids:  [6]  — long tensor, PAY state indices (0-3 each).
                pay_severities: [6]  — float tensor, normalised PAY severity (0..1).
                num_values:     [14] — float tensor, scaled numerical values
                                        (position implies feature identity).
                label:          []   — float tensor, 0.0 or 1.0.
        """
        cat_indices = {
            feature: torch.tensor(self.all_cat_indices[feature][idx], dtype=torch.long)
            for feature in CATEGORICAL_FEATURES
        }
        return {
            "cat_indices":    cat_indices,
            "pay_state_ids":  torch.tensor(self.all_pay_state_ids[idx], dtype=torch.long),
            "pay_severities": torch.tensor(self.all_pay_severities[idx], dtype=torch.float),
            "num_values":     torch.tensor(self.all_num_vals[idx], dtype=torch.float),
            "label":          torch.tensor(self.all_labels[idx], dtype=torch.float),
        }
