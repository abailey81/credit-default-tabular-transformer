"""UCI dataset ingestion with a UCI-API → local-``.xls`` fallback chain.

The rest of the pipeline sees a single ``DataSourceResult`` and doesn't care
which source won.
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


# Constants

UCI_DATASET_ID = 350

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

DEFAULT_LOCAL_CANDIDATES: Tuple[str, ...] = (
    "data/raw/default_of_credit_card_clients.xls",
    "data/raw/default_of_credit_card_clients.xlsx",
    "data/raw/default of credit card clients.xls",
    "data/raw/default of credit card clients.xlsx",
    "default_of_credit_card_clients.xls",
    "default of credit card clients.xls",
)

UCI_DATASET_URL = f"https://archive.ics.uci.edu/dataset/{UCI_DATASET_ID}"


# Errors and result records

class DataIngestionError(RuntimeError):
    """Every configured source failed — no data."""

    def __init__(self, attempts: Sequence[Tuple[str, str]]):
        self.attempts: Tuple[Tuple[str, str], ...] = tuple(attempts)
        lines = ["All configured data sources failed:"]
        for name, err in self.attempts:
            lines.append(f"  - {name}: {err}")
        super().__init__("\n".join(lines))


@dataclass(frozen=True)
class DataSourceResult:
    """Successful load + where it came from."""

    dataframe: pd.DataFrame
    source_name: str
    source_type: Literal["api", "local"]
    origin: str
    duration_s: float
    failed_attempts: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)

    def summary(self) -> str:
        rows, cols = self.dataframe.shape
        line = (
            f"loaded {rows:,} rows x {cols} cols from {self.source_name} "
            f"({self.origin}) in {self.duration_s:.2f}s"
        )
        if self.failed_attempts:
            misses = ", ".join(name for name, _ in self.failed_attempts)
            line += f" [after fallback from: {misses}]"
        return line


class DataSource(ABC):
    """Base class. Subclasses stay side-effect-free until ``load()``."""

    source_type: Literal["api", "local"]

    @property
    @abstractmethod
    def name(self) -> str:
        """Log-friendly identifier."""

    @abstractmethod
    def load(self) -> DataSourceResult:
        """Fetch the frame. Raises on failure."""


# UCI API source

class UCIRepoSource(DataSource):
    """UCI ML Repository fetch via ``ucimlrepo``, with exponential backoff
    so a flaky TCP path doesn't silently drop us onto the local fallback."""

    source_type: Literal["api"] = "api"

    def __init__(
        self,
        dataset_id: int = UCI_DATASET_ID,
        max_retries: int = 3,
        backoff_seconds: float = 1.5,
        request_timeout_seconds: float = 30.0,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if backoff_seconds < 0:
            raise ValueError("backoff_seconds must be >= 0")
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be > 0")
        self.dataset_id = dataset_id
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.request_timeout_seconds = request_timeout_seconds

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

        # Two-layer timeout (SECURITY_AUDIT H-2): socket timeout caps TCP-level
        # hangs; ThreadPoolExecutor wraps a wall-clock deadline so a blocked
        # worker thread can't outlive us.
        import socket
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

        last_error: Optional[BaseException] = None
        start = time.perf_counter()
        dataset = None

        def _fetch() -> object:
            previous_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(self.request_timeout_seconds)
            try:
                return fetch_ucirepo(id=self.dataset_id)
            finally:
                socket.setdefaulttimeout(previous_timeout)

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "Fetching dataset %d from UCI repository (attempt %d/%d, timeout %.0fs)",
                    self.dataset_id, attempt, self.max_retries, self.request_timeout_seconds,
                )
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(_fetch)
                    try:
                        dataset = future.result(timeout=self.request_timeout_seconds)
                    except FutureTimeout as exc:
                        future.cancel()
                        raise TimeoutError(
                            f"UCI fetch exceeded {self.request_timeout_seconds:.0f}s deadline"
                        ) from exc
                last_error = None
                break
            except (ConnectionError, TimeoutError, OSError, ValueError) as exc:
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


# Local Excel source

class LocalExcelSource(DataSource):
    """Read from a local ``.xls``/``.xlsx``. Candidates are tried in order;
    relative paths resolve against cwd and repo root."""

    source_type: Literal["local"] = "local"

    def __init__(self, candidates: Sequence[PathLike]) -> None:
        if not candidates:
            raise ValueError("LocalExcelSource requires at least one candidate path")
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
        repo_root = Path(__file__).resolve().parent.parent.parent
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
        # UCI .xls: row 0 is a section banner, real column headers on row 1.
        df = pd.read_excel(path, header=1)
        elapsed = time.perf_counter() - start

        return DataSourceResult(
            dataframe=df,
            source_name=self.name,
            source_type="local",
            origin=str(path),
            duration_s=elapsed,
        )


# Chained source

class ChainedDataSource(DataSource):
    """Walk the child sources in order; fall through on recoverable errors."""

    source_type: Literal["api"] = "api"

    def __init__(self, sources: Sequence[DataSource]) -> None:
        if not sources:
            raise ValueError("ChainedDataSource requires at least one child source")
        self.sources: Tuple[DataSource, ...] = tuple(sources)

    @property
    def name(self) -> str:
        return "Chain[" + " -> ".join(s.name for s in self.sources) + "]"

    # only these errors justify falling through — TypeError / MemoryError /
    # KeyboardInterrupt must propagate so real bugs stay visible.
    _RECOVERABLE_ERRORS: tuple = (
        ConnectionError,
        TimeoutError,
        FileNotFoundError,
        OSError,
        ValueError,
        ImportError,
    )

    def load(self) -> DataSourceResult:
        failures: list[Tuple[str, str]] = []
        for source in self.sources:
            try:
                result = source.load()
            except self._RECOVERABLE_ERRORS as exc:
                logger.warning("Source '%s' failed (recoverable): %s", source.name, exc)
                failures.append((source.name, f"{type(exc).__name__}: {exc}"))
                continue
            except Exception as exc:
                logger.error(
                    "Source '%s' raised unrecoverable %s -- aborting chain",
                    source.name, type(exc).__name__,
                )
                failures.append((source.name, f"{type(exc).__name__}: {exc}"))
                raise DataIngestionError(failures) from exc

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


# Factory

def build_default_data_source(
    data_path: Optional[PathLike] = None,
    *,
    mode: SourceMode = "auto",
    allow_fallback: bool = True,
    extra_local_candidates: Sequence[PathLike] = (),
    max_retries: int = 3,
    backoff_seconds: float = 1.5,
) -> DataSource:
    """Build the project's canonical source.

    ``data_path`` hard-pins a local file (mode ignored). Otherwise:
    ``auto`` → API then local (chained), ``api`` → API only, ``local`` →
    defaults + ``extra_local_candidates``. ``allow_fallback=False`` in auto
    mode makes API failure fatal.
    """
    if mode not in ("auto", "api", "local"):
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'auto', 'api', or 'local'.")

    if data_path is not None:
        local_candidates: list[PathLike] = [Path(data_path)]
    else:
        local_candidates = list(DEFAULT_LOCAL_CANDIDATES) + list(extra_local_candidates)
    local_source = LocalExcelSource(local_candidates)

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
