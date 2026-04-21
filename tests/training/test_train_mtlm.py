"""train_mtlm.py — argparse, model construction, e2e smoke pretrain."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent.parent

from src.models.model import TabularTransformer  # noqa: E402
from src.models.mtlm import MTLMModel  # noqa: E402
from src.training import train_mtlm as train_mtlm_mod  # noqa: E402


def _default_args_namespace(**overrides: Any) -> Namespace:
    parser = train_mtlm_mod._build_parser()
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _cat_vocab_sizes() -> dict[str, int]:
    from src.tokenization.embedding import CAT_VOCAB_SIZES

    return dict(CAT_VOCAB_SIZES)


def test_build_parser_help_renders_without_error(capsys):
    # regression: raw `%` in argparse help crashed src/train.py on PR #10.
    # every help string must survive print_help()'s format-string pass.
    parser = train_mtlm_mod._build_parser()
    parser.print_help()
    captured = capsys.readouterr()
    assert captured.out.startswith("usage:")
    assert "--seed" in captured.out
    assert "--smoke-test" in captured.out


def test_build_mtlm_model_returns_mtlm_model_with_expected_prefixes():
    args = _default_args_namespace()
    model = train_mtlm_mod.build_mtlm_model(args, _cat_vocab_sizes())

    assert isinstance(model, MTLMModel)

    enc_state = model.encoder_state_dict()
    assert len(enc_state) > 0

    for key in enc_state:
        assert key.startswith("embedding.") or key.startswith(
            "encoder."
        ), f"Unexpected prefix in MTLMModel.encoder_state_dict(): {key!r}"


def test_encoder_state_dict_loadable_by_tabular_transformer(tmp_path: Path):
    # Plan §8.5.5 two-stage contract: what train_mtlm.py writes must load
    # into TabularTransformer.load_pretrained_encoder unchanged
    args = _default_args_namespace()
    mtlm = train_mtlm_mod.build_mtlm_model(args, _cat_vocab_sizes())

    path = tmp_path / "encoder_pretrained.pt"
    torch.save(mtlm.encoder_state_dict(), path)

    downstream = TabularTransformer(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        use_temporal_pos=args.use_temporal_pos,
    )
    downstream.load_pretrained_encoder(str(path), strict=False)


def test_main_smoke_test_produces_all_expected_artefacts(tmp_path: Path):
    if not (REPO / "data/processed/splits/train_scaled.csv").is_file():
        pytest.skip("preprocessing outputs not present; run run_pipeline.py first")

    output_dir = tmp_path / "mtlm_run"
    rc = train_mtlm_mod.main(
        [
            "--seed",
            "0",
            "--smoke-test",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert rc == 0

    config = json.loads((output_dir / "config.json").read_text())
    assert "seed" in config
    assert "param_count" in config

    import pandas as _pd

    log = _pd.read_csv(output_dir / "pretrain_log.csv")
    assert len(log) > 0
    assert "train_loss" in log.columns
    assert "val_loss" in log.columns

    assert (output_dir / "best.pt").is_file()
    assert (output_dir / "best.pt.weights").is_file()
    assert (output_dir / "best.pt.meta.json").is_file()

    enc_path = output_dir / "encoder_pretrained.pt"
    assert enc_path.is_file()

    state = torch.load(enc_path, map_location="cpu", weights_only=True)
    assert isinstance(state, dict) and len(state) > 0
    for key in state:
        assert key.startswith("embedding.") or key.startswith(
            "encoder."
        ), f"Unexpected prefix in encoder_pretrained.pt: {key!r}"
