#!/usr/bin/env python3
"""
run_pipeline.py — Master entry point for the full project pipeline.

Examples
--------
    # Full pipeline (EDA + preprocessing). Tries the UCI API, falls back
    # automatically to the local manual dataset (data/raw/) if the API is
    # unavailable.
    poetry run python run_pipeline.py

    # Individual stages
    poetry run python run_pipeline.py --eda-only
    poetry run python run_pipeline.py --preprocess-only
    poetry run python run_pipeline.py --rf-benchmark

    # Force a specific data source (default: auto = API → local fallback)
    poetry run python run_pipeline.py --source api      # API only, no fallback
    poetry run python run_pipeline.py --source local    # local file only
    poetry run python run_pipeline.py --no-fallback     # auto without fallback

    # Pin to a specific local file
    poetry run python run_pipeline.py \
        --data-path "data/raw/default_of_credit_card_clients.xls"

Data ingestion is handled by ``src/data_sources.py`` which provides a layered,
provenance-aware loader with API → local-file failover.
"""

import sys
import argparse
from pathlib import Path

# Force UTF-8 for stdout/stderr on Windows (default cp1251/cp1252 cannot encode
# characters like "×" or "→" that our provenance loggers emit). Safe no-op on
# platforms where the default encoding is already UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

# Ensure src/ is importable
_src_dir = Path(__file__).parent / "src"
if not _src_dir.is_dir():
    print(f"[ERROR] src/ directory not found at {_src_dir}")
    sys.exit(1)
sys.path.insert(0, str(_src_dir))

from data_preprocessing import run_preprocessing_pipeline  # noqa: E402
from eda import run_eda  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Credit Card Default — EDA, Preprocessing & RF Benchmark Pipeline. "
            "Resilient ingestion: by default the UCI API is tried first, with "
            "automatic fallback to the local manual dataset on failure."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--eda-only", action="store_true",
        help="Run EDA only (generates figures).",
    )
    mode.add_argument(
        "--preprocess-only", action="store_true",
        help="Run preprocessing only (generates data splits).",
    )
    mode.add_argument(
        "--rf-benchmark", action="store_true",
        help="Run Random Forest benchmark only (training + evaluation).",
    )

    parser.add_argument(
        "--data-path",
        default=None,
        help=(
            "Pin loading to a specific .xls/.xlsx file. Bypasses the chained "
            "loader: when set, the network is never contacted."
        ),
    )
    parser.add_argument(
        "--source",
        choices=("auto", "api", "local"),
        default="auto",
        help=(
            "Data source preference. 'auto' (default) tries the UCI API first "
            "and falls back to the local manual dataset on failure. 'api' "
            "uses only the UCI API. 'local' uses only the local manual file."
        ),
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help=(
            "In 'auto' source mode, disable the local fallback so any UCI "
            "API failure becomes a hard error."
        ),
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    data_path = args.data_path
    if data_path is not None:
        resolved = Path(data_path).resolve()
        if not resolved.exists():
            print(f"[ERROR] Dataset not found at: {resolved}")
            sys.exit(1)
        if resolved.suffix.lower() not in (".xls", ".xlsx"):
            print(f"[ERROR] Expected .xls or .xlsx file, got: {resolved.suffix}")
            sys.exit(1)
        data_path = str(resolved)
        print(f"[INFO] Using pinned local dataset: {data_path}")
    else:
        if args.source == "auto":
            mode_desc = (
                "auto (UCI API → local fallback)"
                if not args.no_fallback
                else "auto (UCI API only — fallback disabled)"
            )
        elif args.source == "api":
            mode_desc = "api (UCI API only)"
        else:
            mode_desc = "local (manual dataset only)"
        print(f"[INFO] Data source mode: {mode_desc}")

    loader_kwargs = {
        "mode": args.source,
        "allow_fallback": not args.no_fallback,
    }

    if args.eda_only:
        print("=" * 60)
        print("  RUNNING: Exploratory Data Analysis")
        print("=" * 60)
        run_eda(data_path, save_dir="figures", **loader_kwargs)

    elif args.preprocess_only:
        print("=" * 60)
        print("  RUNNING: Data Preprocessing Pipeline")
        print("=" * 60)
        run_preprocessing_pipeline(data_path, output_dir="data/processed", **loader_kwargs)

    elif args.rf_benchmark:
        from random_forest import run_rf_benchmark
        print("=" * 60)
        print("  RUNNING: Random Forest Benchmark")
        print("=" * 60)
        run_rf_benchmark(
            data_path,
            output_dir="results",
            figure_dir="figures",
            **loader_kwargs,
        )

    else:
        print("=" * 60)
        print("  RUNNING: Full Pipeline (EDA + Preprocessing)")
        print("=" * 60)
        run_eda(data_path, save_dir="figures", **loader_kwargs)
        print()
        run_preprocessing_pipeline(data_path, output_dir="data/processed", **loader_kwargs)

    print("\n[DONE] Pipeline complete.")


if __name__ == "__main__":
    main()
