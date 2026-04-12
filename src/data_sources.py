"""
data_sources.py — Resilient, multi-source data ingestion for the
UCI Credit Card Default dataset.

This module provides a layered abstraction for loading the dataset, with
graceful failover from a remote API source (the UCI ML Repository) to a
locally-shipped Excel spreadsheet. It is designed so that the rest of the
project remains agnostic to *where* the data came from while still being
able to report which source was used for full provenance.

Architecture
------------

        ┌──────────────────────────────────────────────────┐
        │                ChainedDataSource                  │
        │   (tries each child source in order, fails over)  │
        └──────────────┬───────────────────┬────────────────┘
                       ▼                   ▼
              ┌────────────────┐  ┌──────────────────────┐
              │ UCIRepoSource  │  │  LocalExcelSource     │
              │ (network API)  │  │  (offline fallback)   │
              └────────────────┘  └──────────────────────┘

Public API
----------
* :class:`DataSource`            – abstract base for any ingestion source.
* :class:`UCIRepoSource`         – fetches via the ``ucimlrepo`` package, with
                                   bounded exponential-backoff retries.
* :class:`LocalExcelSource`      – reads from a list of candidate ``.xls``/
                                   ``.xlsx`` paths.
* :class:`ChainedDataSource`     – tries sources in order, accumulating errors.
* :class:`DataSourceResult`      – frozen, provenance-bearing success record.
* :class:`DataIngestionError`    – raised when *all* configured sources fail.
* :func:`build_default_data_source` – factory used by the rest of the codebase.

Design notes
------------
*   Sources are side-effect-free until :py:meth:`DataSource.load` is called.
*   The chained source records every failed attempt in
    :py:attr:`DataSourceResult.failed_attempts` so callers can audit the
    fallback path even on success.
*   The factory honours an explicit ``data_path`` argument as a hard pin —
    if the user has named a file, we never silently fall back to the network.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
SourceMode = Literal["auto", "api", "local"]


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

UCI_DATASET_ID = 350

# Mapping for the column codes ucimlrepo returns. Mirrors the mapping that
# previously lived inline in ``data_preprocessing.load_raw_data``.
UCI_COLUMN_MAP: dict[str, str] = {
    "X1": "LIMIT_BAL", "X2": "SEX", "X3": "EDUCATION",
    "X4": "MARRIAGE", "X5": "AGE",
    "X6": "PAY_0", "X7": "PAY_2", "X8": "PAY_3",
    "X9": "PAY_4", "X10": "PAY_5", "X11": "PAY_6",
    "X12": "BILL_AMT1", "X13": "BILL_AMT2", "X14": "BILL_AMT3",
    "X15": "BILL_AMT4", "X16": "BILL_AMT5", "X17": "BILL_AMT6",
    "X18": "PAY_AMT1", "X19": "PAY_AMT2", "X20": "PAY_AMT3",
    "X21": "PAY_AMT4", "X22": "PAY_AMT5", "X23": "PAY_AMT6",
}

# Default search order for the manual fallback file. Resolved at load time
# against both the project root and the current working directory so the
# loader behaves the same whether invoked as a script, a notebook, or an
# installed entry point.
DEFAULT_LOCAL_CANDIDATES: Tuple[str, ...] = (
    "data/raw/default_of_credit_card_clients.xls",
    "data/raw/default_of_credit_card_clients.xlsx",
    "data/raw/default of credit card clients.xls",
    "data/raw/default of credit card clients.xlsx",
    "default_of_credit_card_clients.xls",
    "default of credit card clients.xls",
)

# UCI hosts the original .xls under this URL — recorded as the canonical
# remote origin for provenance reports.
UCI_DATASET_URL = f"https://archive.ics.uci.edu/dataset/{UCI_DATASET_ID}"


# ──────────────────────────────────────────────────────────────────────────────
# Errors and result records
# ──────────────────────────────────────────────────────────────────────────────


class DataIngestionError(RuntimeError):
    """Raised when no configured data source can supply the dataset."""

    def __init__(self, attempts: Sequence[Tuple[str, str]]):
        self.attempts: Tuple[Tuple[str, str], ...] = tuple(attempts)
        lines = ["All configured data sources failed:"]
        for name, err in self.attempts:
            lines.append(f"  - {name}: {err}")
        super().__init__("\n".join(lines))


@dataclass(frozen=True)
class DataSourceResult:
    """Provenance-bearing result from a successful data load."""

    dataframe: pd.DataFrame
    source_name: str
    source_type: Literal["api", "local"]
    origin: str
    duration_s: float
    failed_attempts: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)

    def summary(self) -> str:
        rows, cols = self.dataframe.shape
        line = (
            f"loaded {rows:,} rows × {cols} cols from {self.source_name} "
            f"({self.origin}) in {self.duration_s:.2f}s"
        )
        if self.failed_attempts:
            misses = ", ".join(name for name, _ in self.failed_attempts)
            line += f" [after fallback from: {misses}]"
        return line


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────


class DataSource(ABC):
    """Abstract data source. Implementations are side-effect-free until ``load()``."""

    source_type: Literal["api", "local"]

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for logging and provenance."""

    @abstractmethod
    def load(self) -> DataSourceResult:
        """Materialise the dataset. Raises on any failure."""


# ──────────────────────────────────────────────────────────────────────────────
# UCI repository source (network API)
# ──────────────────────────────────────────────────────────────────────────────


class UCIRepoSource(DataSource):
    """
    Fetch the dataset directly from the UCI ML Repository via ``ucimlrepo``.

    Includes a small retry budget with exponential backoff so that transient
    network glitches do not silently demote the pipeline to the offline
    fallback.
    """

    source_type: Literal["api"] = "api"

    def __init__(
        self,
        dataset_id: int = UCI_DATASET_ID,
        max_retries: int = 3,
        backoff_seconds: float = 1.5,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if backoff_seconds < 0:
            raise ValueError("backoff_seconds must be >= 0")
        self.dataset_id = dataset_id
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    @property
    def name(self) -> str:
        return f"UCI ML Repository (id={self.dataset_id})"

    def load(self) -> DataSourceResult:
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as exc:
            raise ConnectionError(
                "ucimlrepo is not installed. Install with `pip install ucimlrepo` "
                "or rely on the local fallback dataset."
            ) from exc

        last_error: Optional[BaseException] = None
        start = time.perf_counter()
        dataset = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "Fetching dataset %d from UCI repository (attempt %d/%d)",
                    self.dataset_id, attempt, self.max_retries,
                )
                dataset = fetch_ucirepo(id=self.dataset_id)
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001 — third-party may raise anything
                last_error = exc
                logger.warning(
                    "UCI fetch attempt %d/%d failed: %s",
                    attempt, self.max_retries, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))

        if dataset is None:
            raise ConnectionError(
                f"Failed to fetch UCI dataset {self.dataset_id} after "
                f"{self.max_retries} attempt(s): {last_error}"
            ) from last_error

        df = dataset.data.features.copy()
        target = dataset.data.targets

        if "X1" in df.columns:
            df.rename(columns=UCI_COLUMN_MAP, inplace=True)
        df["DEFAULT"] = target.values.ravel()

        elapsed = time.perf_counter() - start
        return DataSourceResult(
            dataframe=df,
            source_name=self.name,
            source_type="api",
            origin=UCI_DATASET_URL,
            duration_s=elapsed,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Local Excel source (offline fallback)
# ──────────────────────────────────────────────────────────────────────────────


class LocalExcelSource(DataSource):
    """
    Read the dataset from a local ``.xls``/``.xlsx`` file.

    A list of candidate paths can be supplied; the first one that resolves
    to an existing file wins. Paths can be absolute or relative; relative
    paths are resolved against (1) the current working directory and (2)
    the repository root, in that order.
    """

    source_type: Literal["local"] = "local"

    def __init__(self, candidates: Sequence[PathLike]) -> None:
        if not candidates:
            raise ValueError("LocalExcelSource requires at least one candidate path")
        # Preserve order, drop duplicates.
        seen: set[str] = set()
        unique: list[Path] = []
        for c in candidates:
            key = str(c)
            if key not in seen:
                seen.add(key)
                unique.append(Path(c))
        self.candidates: Tuple[Path, ...] = tuple(unique)

    @property
    def name(self) -> str:
        return "Local fallback dataset"

    @staticmethod
    def _search_roots() -> list[Path]:
        """Locations to resolve relative candidate paths against."""
        # this file: <repo>/src/data_sources.py → <repo>
        repo_root = Path(__file__).resolve().parent.parent
        roots: list[Path] = []
        for r in (Path.cwd(), repo_root):
            if r not in roots:
                roots.append(r)
        return roots

    def _resolve(self) -> Optional[Path]:
        for cand in self.candidates:
            if cand.is_absolute():
                if cand.is_file():
                    return cand
                continue
            for root in self._search_roots():
                resolved = (root / cand).resolve()
                if resolved.is_file():
                    return resolved
        return None

    def describe_candidates(self) -> str:
        return ", ".join(str(c) for c in self.candidates)

    def load(self) -> DataSourceResult:
        path = self._resolve()
        if path is None:
            raise FileNotFoundError(
                "No local fallback dataset found. Tried: "
                f"{self.describe_candidates()}"
            )
        if path.suffix.lower() not in (".xls", ".xlsx"):
            raise ValueError(
                f"Unsupported extension for local dataset: {path.suffix} "
                "(expected .xls or .xlsx)"
            )

        logger.info("Loading dataset from local file: %s", path)
        start = time.perf_counter()
        # The UCI .xls ships with a section header on row 0; the real header
        # is on row 1. ``header=1`` matches the original loading semantics.
        df = pd.read_excel(path, header=1)
        elapsed = time.perf_counter() - start

        return DataSourceResult(
            dataframe=df,
            source_name=self.name,
            source_type="local",
            origin=str(path),
            duration_s=elapsed,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Chained source
# ──────────────────────────────────────────────────────────────────────────────


class ChainedDataSource(DataSource):
    """Try each child source in order, falling over on failure."""

    # Nominal type — the actual source is reported in the ``DataSourceResult``.
    source_type: Literal["api"] = "api"

    def __init__(self, sources: Sequence[DataSource]) -> None:
        if not sources:
            raise ValueError("ChainedDataSource requires at least one child source")
        self.sources: Tuple[DataSource, ...] = tuple(sources)

    @property
    def name(self) -> str:
        return "Chain[" + " → ".join(s.name for s in self.sources) + "]"

    def load(self) -> DataSourceResult:
        failures: list[Tuple[str, str]] = []
        for source in self.sources:
            try:
                result = source.load()
            except Exception as exc:  # noqa: BLE001 — fall over on any failure
                logger.warning("Source '%s' failed: %s", source.name, exc)
                failures.append((source.name, f"{type(exc).__name__}: {exc}"))
                continue

            if failures:
                logger.info(
                    "Recovered via fallback to '%s' after %d failed source(s)",
                    source.name, len(failures),
                )

            return DataSourceResult(
                dataframe=result.dataframe,
                source_name=result.source_name,
                source_type=result.source_type,
                origin=result.origin,
                duration_s=result.duration_s,
                failed_attempts=tuple(failures),
            )

        raise DataIngestionError(failures)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────


def build_default_data_source(
    data_path: Optional[PathLike] = None,
    *,
    mode: SourceMode = "auto",
    allow_fallback: bool = True,
    extra_local_candidates: Sequence[PathLike] = (),
    max_retries: int = 3,
    backoff_seconds: float = 1.5,
) -> DataSource:
    """
    Build the canonical data source used throughout the project.

    Behaviour
    ---------
    * **Explicit ``data_path``** – a :class:`LocalExcelSource` pinned to that
      file is returned regardless of ``mode``. The user has spoken; we honour
      their choice and never silently fall back to the network.
    * **``mode="auto"`` (default)** – returns a :class:`ChainedDataSource`
      that first tries the UCI API, then falls back to the local file.
      If ``allow_fallback=False`` the local fallback is skipped and API
      failures propagate.
    * **``mode="api"``** – API only; failures propagate.
    * **``mode="local"``** – local file only; tried against the default
      candidates plus any ``extra_local_candidates``.

    Parameters
    ----------
    data_path
        Optional explicit path to a local ``.xls``/``.xlsx`` file.
    mode
        Source preference: ``"auto"``, ``"api"``, or ``"local"``.
    allow_fallback
        Only relevant in ``"auto"`` mode. If ``False``, the API becomes a
        hard requirement.
    extra_local_candidates
        Additional local file candidates appended to the default search list.
    max_retries
        Number of attempts the UCI source will make before giving up.
    backoff_seconds
        Base interval for exponential backoff between UCI retries.
    """
    if mode not in ("auto", "api", "local"):
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'auto', 'api', or 'local'.")

    # Build the local source. We always need at least one candidate to
    # construct it; either the user-supplied path or the defaults.
    if data_path is not None:
        local_candidates: list[PathLike] = [Path(data_path)]
    else:
        local_candidates = list(DEFAULT_LOCAL_CANDIDATES) + list(extra_local_candidates)
    local_source = LocalExcelSource(local_candidates)

    # An explicit path is a hard pin — never silently hit the network.
    if data_path is not None:
        return local_source

    if mode == "local":
        return local_source

    api_source = UCIRepoSource(
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
    )

    if mode == "api":
        return api_source

    # mode == "auto"
    if allow_fallback:
        return ChainedDataSource([api_source, local_source])
    return api_source


__all__ = [
    "DEFAULT_LOCAL_CANDIDATES",
    "UCI_COLUMN_MAP",
    "UCI_DATASET_ID",
    "UCI_DATASET_URL",
    "ChainedDataSource",
    "DataIngestionError",
    "DataSource",
    "DataSourceResult",
    "LocalExcelSource",
    "SourceMode",
    "UCIRepoSource",
    "build_default_data_source",
]
