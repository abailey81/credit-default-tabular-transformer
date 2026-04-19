#!/usr/bin/env python3
"""run_all.py — one-shot orchestrator.

TODO (Phase 3B): end-to-end runner that walks train -> evaluate -> calibrate
-> fairness -> uncertainty -> significance -> interpret -> repro with a
single command. For now, see scripts/run_pipeline.py for the preprocessing /
EDA / RF-benchmark entry point and the individual ``python -m src.*``
invocations documented in README.md for everything downstream.
"""
