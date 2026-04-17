"""Tests for src/train_mtlm.py — argparse, model construction, and end-to-end
smoke pretraining.

Closes the 0% pytest coverage gap on ``train_mtlm.py`` flagged during
PR #11 review. The patterns here mirror ``tests/test_train.py`` so the two
training loops share a consistent test surface.

Scope
-----
* Argparse ``--help`` renders without crashing (regression for the ``%``
  escape class of bug that bit ``src/train.py`` on PR #10).
* ``build_mtlm_model`` returns a valid :class:`MTLMModel` and the encoder
  state-dict it produces is drop-in loadable by
  :meth:`TabularTransformer.load_pretrained_encoder` — the pretraining →
  fine-tuning contract.
* ``main(["--smoke-test", ...])`` runs end-to-end and writes every artefact
  that ``train.py --pretrained-encoder`` downstream will consume. This case
  exercises :func:`train_one_epoch` and :func:`evaluate_on_loader` implicitly
  through the full loop; dedicated unit tests for those two functions would
  need a hand-crafted MTLM-collated batch (``mask_positions`` + ``pay_raw``
  etc.), which the smoke test already covers via the real ``MTLMCollator``.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

# Make src/ importable.
REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model import TabularTransformer  # noqa: E402
from mtlm import MTLMModel  # noqa: E402

import train_mtlm as train_mtlm_mod  # noqa: E402 — module under test


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — build the argparse.Namespace that build_mtlm_model expects
# ──────────────────────────────────────────────────────────────────────────────


def _default_args_namespace(**overrides: Any) -> Namespace:
    """Parse the module's own CLI with no arguments — produces a full
    ``Namespace`` with every default populated, which
    :func:`train_mtlm.build_mtlm_model` can consume directly."""
    parser = train_mtlm_mod._build_parser()
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _cat_vocab_sizes() -> Dict[str, int]:
    """Pull the canonical categorical vocab sizes the embedding expects."""
    from embedding import CAT_VOCAB_SIZES

    return dict(CAT_VOCAB_SIZES)


# ──────────────────────────────────────────────────────────────────────────────
# Argparse — --help must render cleanly
# ──────────────────────────────────────────────────────────────────────────────


def test_build_parser_help_renders_without_error(capsys):
    """Every argparse help string must survive ``print_help()``. A single raw
    ``%`` in any help string would crash before any training could start."""
    parser = train_mtlm_mod._build_parser()
    parser.print_help()
    captured = capsys.readouterr()
    assert captured.out.startswith("usage:")
    # Sanity check: a handful of named options must survive the format-string
    # expansion, proving the whole help text rendered.
    assert "--seed" in captured.out
    assert "--smoke-test" in captured.out


# ──────────────────────────────────────────────────────────────────────────────
# build_mtlm_model — structural contract
# ──────────────────────────────────────────────────────────────────────────────


def test_build_mtlm_model_returns_mtlm_model_with_expected_prefixes():
    """``build_mtlm_model`` must return an :class:`MTLMModel` whose
    encoder-only state dict has exactly the key prefixes that
    :meth:`TabularTransformer.load_pretrained_encoder` expects."""
    args = _default_args_namespace()
    model = train_mtlm_mod.build_mtlm_model(args, _cat_vocab_sizes())

    assert isinstance(model, MTLMModel)

    enc_state = model.encoder_state_dict()
    assert len(enc_state) > 0

    # Every key must be prefixed with either embedding.* or encoder.* — the
    # prefixes the downstream model consumes.
    for key in enc_state:
        assert key.startswith("embedding.") or key.startswith("encoder."), (
            f"Unexpected prefix in MTLMModel.encoder_state_dict(): {key!r}"
        )


def test_encoder_state_dict_loadable_by_tabular_transformer(tmp_path: Path):
    """The artefact ``train_mtlm.py`` persists must be consumable by
    :meth:`TabularTransformer.load_pretrained_encoder`. This is the whole
    two-stage fine-tune contract (Plan §8.5.5)."""
    args = _default_args_namespace()
    mtlm = train_mtlm_mod.build_mtlm_model(args, _cat_vocab_sizes())

    # Persist exactly what train_mtlm.py would write at the end of pretraining.
    path = tmp_path / "encoder_pretrained.pt"
    torch.save(mtlm.encoder_state_dict(), path)

    # A downstream TabularTransformer with matching architecture must load
    # the raw state-dict file without raising. ``strict=False`` because the
    # classification head's fresh weights are kept.
    downstream = TabularTransformer(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        use_temporal_pos=args.use_temporal_pos,
    )
    downstream.load_pretrained_encoder(str(path), strict=False)


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end smoke — main --smoke-test produces every expected artefact
# ──────────────────────────────────────────────────────────────────────────────


def test_main_smoke_test_produces_all_expected_artefacts(tmp_path: Path):
    """The headline integration test: ``main(["--smoke-test", ...])`` runs
    MTLM pretraining end-to-end on a tiny slice and writes every artefact
    that ``train.py --pretrained-encoder`` downstream will consume.

    This case exercises :func:`train_one_epoch` and
    :func:`evaluate_on_loader` through the real loop — verifying the full
    call graph end-to-end rather than each function in isolation."""
    if not (REPO / "data/processed/train_scaled.csv").is_file():
        pytest.skip("preprocessing outputs not present; run run_pipeline.py first")

    output_dir = tmp_path / "mtlm_run"
    rc = train_mtlm_mod.main([
        "--seed", "0",
        "--smoke-test",
        "--output-dir", str(output_dir),
    ])
    assert rc == 0

    # config.json — records the resolved args + runtime metadata
    config = json.loads((output_dir / "config.json").read_text())
    assert "seed" in config
    assert "param_count" in config

    # pretrain_log.csv — per-epoch training/validation loss
    import pandas as _pd
    log = _pd.read_csv(output_dir / "pretrain_log.csv")
    assert len(log) > 0
    assert "train_loss" in log.columns
    assert "val_loss" in log.columns

    # Checkpoint bundle + sidecars (the SECURITY_AUDIT C-1 weights-only path)
    assert (output_dir / "best.pt").is_file()
    assert (output_dir / "best.pt.weights").is_file()
    assert (output_dir / "best.pt.meta.json").is_file()

    # encoder_pretrained.pt — the hand-off artefact for
    # ``train.py --pretrained-encoder``. Must load cleanly as a weights-only
    # state dict with the expected prefixes.
    enc_path = output_dir / "encoder_pretrained.pt"
    assert enc_path.is_file()

    state = torch.load(enc_path, map_location="cpu", weights_only=True)
    assert isinstance(state, dict) and len(state) > 0
    for key in state:
        assert key.startswith("embedding.") or key.startswith("encoder."), (
            f"Unexpected prefix in encoder_pretrained.pt: {key!r}"
        )
