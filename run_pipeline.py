#!/usr/bin/env python3
"""
run_pipeline.py -- Master entry point for the EDA & Preprocessing pipeline.

Usage:
    poetry run python run_pipeline.py                    # Run both EDA and preprocessing
    poetry run python run_pipeline.py --eda-only         # Run only EDA (figures)
    poetry run python run_pipeline.py --preprocess-only  # Run only preprocessing (data splits)
    poetry run python run_pipeline.py --data-path FILE   # Use local .xls/.xlsx file

If no --data-path is provided, the dataset is fetched automatically from the
UCI Machine Learning Repository via the ucimlrepo package.
"""

import sys
import argparse
from pathlib import Path

# Ensure src/ is importable
_src_dir = Path(__file__).parent / "src"
if not _src_dir.is_dir():
    print(f"[ERROR] src/ directory not found at {_src_dir}")
    sys.exit(1)
sys.path.insert(0, str(_src_dir))

from data_preprocessing import run_preprocessing_pipeline
from eda import run_eda


def main():
    parser = argparse.ArgumentParser(
        description="Credit Card Default -- EDA & Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--eda-only", action="store_true", help="Run EDA only (generates figures)")
    mode.add_argument("--preprocess-only", action="store_true", help="Run preprocessing only (generates data splits)")
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to local .xls/.xlsx file. If omitted, fetches from UCI via ucimlrepo.",
    )
    args = parser.parse_args()

    data_path = args.data_path
    if data_path is not None:
        data_path = str(Path(data_path).resolve())
        p = Path(data_path)
        if not p.exists():
            print(f"[ERROR] Dataset not found at: {data_path}")
            sys.exit(1)
        if p.suffix.lower() not in (".xls", ".xlsx"):
            print(f"[ERROR] Expected .xls or .xlsx file, got: {p.suffix}")
            sys.exit(1)
    else:
        print("[INFO] No --data-path provided; will fetch from UCI via ucimlrepo")

    if args.eda_only:
        print("=" * 60)
        print("  RUNNING: Exploratory Data Analysis")
        print("=" * 60)
        run_eda(data_path, save_dir="figures")

    elif args.preprocess_only:
        print("=" * 60)
        print("  RUNNING: Data Preprocessing Pipeline")
        print("=" * 60)
        run_preprocessing_pipeline(data_path, output_dir="data/processed")

    else:
        print("=" * 60)
        print("  RUNNING: Full Pipeline (EDA + Preprocessing)")
        print("=" * 60)
        run_eda(data_path, save_dir="figures")
        print()
        run_preprocessing_pipeline(data_path, output_dir="data/processed")

    print("\n[DONE] Pipeline complete.")


if __name__ == "__main__":
    main()
