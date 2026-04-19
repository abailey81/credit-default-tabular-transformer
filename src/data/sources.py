"""UCI credit-card-default dataset ingestion with resilient API -> local fallback.

The rest of the pipeline only ever sees a single ``DataSourceResult`` — it does
not care whether the frame came off the network or a local ``.xls``. That
decoupling is deliberate: downstream code can assume a valid frame and a
populated provenance record without branching on source type.

Key types:

* ``DataSource``         — ABC; subclasses are side-effect-free until
                           ``load()`` is called.
* ``UCIRepoSource``      — hits the UCI ML Repository via ``ucimlrepo``,
                           with exponential backoff + dual-layer timeout.
* ``LocalExcelSource``   — reads a ``.xls``/``.xlsx`` from a candidate list.
* ``ChainedDataSource``  — walks children in order, tolerating only
                           recoverable errors.
* ``DataSourceResult``   — frozen dataclass: frame + where it came from.
* ``build_default_data_source`` — project-canonical factory; every consumer
                                  of the raw dataset goes through it.

Design choices worth calling out:

* Only a narrow error set (``ConnectionError``, ``TimeoutError``,
  ``OSError``, ``ValueError``, ``ImportError``, ``FileNotFoundError``)
  triggers fallback. ``TypeError`` / ``MemoryError`` / ``KeyboardInterrupt``
  must propagate so real bugs and user-initiated cancels stay visible.
* Timeouts are two-layered: a socket default inside the worker and a
  ``ThreadPoolExecutor`` wall-clock deadline wrapping the call. This came
  out of SECURITY_AUDIT H-2 — a single socket timeout was not enough to
  cap a hung TLS handshake.
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# UCI ML Repository dataset id for "default of credit card clients"
UCI_DATASET_ID = 350

# UCI ships the frame with the anonymous ``X1..X23`` naming.
# The preprocessing layer and every downstream model assume the documented
# semantic names, so we rename immediately after fetch to keep a single
# vocabulary across the codebase.
UCI_COLUMN_MAP: dict[str, str] = {
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    "X6": "PAY_0",
    "X7": "PAY_2",
    "X8": "PAY_3",
    "X9": "PAY_4",
    "X10": "PAY_5",
    "X11": "PAY_6",
    "X12": "BILL_AMT1",
    "X13": "BILL_AMT2",
    "X14": "BILL_AMT3",
    "X15": "BILL_AMT4",
    "X16": "BILL_AMT5",
    "X17": "BILL_AMT6",
    "X18": "PAY_AMT1",
    "X19": "PAY_AMT2",
    "X20": "PAY_AMT3",
    "X21": "PAY_AMT4",
    "X22": "PAY_AMT5",
    "X23": "PAY_AMT6",
}

# Filenames we have seen in the wild for the local fallback — UCI has
# shipped both underscore and space-separated variants across years, and
# both .xls and .xlsx exist. Order matters: the most common naming first
# keeps the typical path short.
DEFAULT_LOCAL_CANDIDATES: Tuple[str, ...] = (
    "data/raw/default_of_credit_card_clients.xls",
    "data/raw/default_of_credit_card_clients.xlsx",
    "data/raw/default of credit card clients.xls",
    "data/raw/default of credit card clients.xlsx",
    "default_of_credit_card_clients.xls",
    "default of credit card clients.xls",
)

UCI_DATASET_URL = f"https://archive.ics.uci.edu/dataset/{UCI_DATASET_ID}"


# ---------------------------------------------------------------------------
# Errors and result records
# ---------------------------------------------------------------------------


class DataIngestionError(RuntimeError):
    """Every configured source failed; no data was produced.

    Wraps the per-source failures into a single error the caller can log.
    The ``attempts`` tuple preserves (source_name, human-readable reason)
    for each miss so post-mortems do not need to re-run with DEBUG logging.
    """

    def __init__(self, attempts: Sequence[Tuple[str, str]]):
        self.attempts: Tuple[Tuple[str, str], ...] = tuple(attempts)
        lines = ["All configured data sources failed:"]
        for name, err in self.attempts:
            lines.append(f"  - {name}: {err}")
        super().__init__("\n".join(lines))


@dataclass(frozen=True)
class DataSourceResult:
    """A successful load plus the provenance needed to reproduce it.

    Frozen so callers can pass it around as a cache key or log it without
    worrying about mutation. ``failed_attempts`` is populated only when a
    ``ChainedDataSource`` recovered via fallback — a direct load from the
    first-choice source leaves it empty.
    """

    dataframe: pd.DataFrame
    source_name: str
    source_type: Literal["api", "local"]
    origin: str
    duration_s: float
    failed_attempts: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)

    def summary(self) -> str:
        """Single-line log summary: shape, source, origin, timing, misses."""
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
    """Abstract base: a lazily-evaluated source of the raw frame.

    Subclasses must remain side-effect-free until ``load()`` is called — no
    network, no disk reads in ``__init__``. This lets the factory build a
    chain cheaply in tests and lets callers inspect ``name`` for logging
    without paying ingestion cost.
    """

    source_type: Literal["api", "local"]

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier used in logs + ``failed_attempts``."""

    @abstractmethod
    def load(self) -> DataSourceResult:
        """Fetch the frame. Raises on failure (see ``ChainedDataSource``
        for the error taxonomy that qualifies for fallback)."""


# ---------------------------------------------------------------------------
# UCI API source
# ---------------------------------------------------------------------------


class UCIRepoSource(DataSource):
    """Fetch the dataset from the UCI ML Repository.

    Uses the ``ucimlrepo`` client under an exponential-backoff retry loop,
    so a flaky TCP path doesn't silently drop the pipeline onto the local
    fallback after a single transient DNS blip. The ``ImportError`` for a
    missing ``ucimlrepo`` is mapped to ``ConnectionError`` so the chained
    source treats "no client installed" as a recoverable condition and
    keeps walking.
    """

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

        # Two-layer timeout (SECURITY_AUDIT H-2):
        #   (a) socket default caps TCP-level hangs (connect / read).
        #   (b) ThreadPoolExecutor wall-clock deadline catches the case
        #       where the worker thread is "alive" but stuck in a TLS
        #       handshake or CPython's GIL under an unresponsive peer.
        # Without (b), a blocked worker can outlive the whole retry budget.
        import socket
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

        last_error: Optional[BaseException] = None
        start = time.perf_counter()
        dataset = None

        def _fetch() -> object:
            # Scope the socket timeout change to this call only. Leaking
            # a global socket default into the rest of the process would
            # surprise unrelated code (requests, ucimlrepo subcalls, etc.).
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
                    self.dataset_id,
                    attempt,
                    self.max_retries,
                    self.request_timeout_seconds,
                )
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(_fetch)
                    try:
                        dataset = future.result(timeout=self.request_timeout_seconds)
                    except FutureTimeout as exc:
                        # Best-effort cancel; the worker may still run until
                        # its own socket times out, but we stop waiting.
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
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    # classic exponential backoff: 1x, 2x, 4x, ... of base
                    time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))

        if dataset is None:
            raise ConnectionError(
                f"Failed to fetch UCI dataset {self.dataset_id} after "
                f"{self.max_retries} attempt(s): {last_error}"
            ) from last_error

        df = dataset.data.features.copy()
        target = dataset.data.targets

        # Some UCI payloads have already been renamed upstream; only rename
        # if we still see the anonymous "X*" schema.
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


# ---------------------------------------------------------------------------
# Local Excel source
# ---------------------------------------------------------------------------


class LocalExcelSource(DataSource):
    """Read the dataset from a local ``.xls`` / ``.xlsx``.

    Candidates are tried in order; the first one that exists wins. Relative
    paths resolve against both cwd and the repo root (two roots, not just
    cwd, so ``python -m src.data.preprocessing`` works from any directory
    inside the repo). Absolute paths skip the root search entirely.
    """

    source_type: Literal["local"] = "local"

    def __init__(self, candidates: Sequence[PathLike]) -> None:
        if not candidates:
            raise ValueError("LocalExcelSource requires at least one candidate path")
        # dedup while preserving order — candidates are semantically a priority
        # list and repeats confuse the log output
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
        """cwd + repo root, deduplicated. Repo root is three parents up from
        this file — it survives the subpackage restructure because it always
        points at the git-tracked root, not the package dir."""
        repo_root = Path(__file__).resolve().parent.parent.parent
        roots: list[Path] = []
        for r in (Path.cwd(), repo_root):
            if r not in roots:
                roots.append(r)
        return roots

    def _resolve(self) -> Optional[Path]:
        """Walk candidates; return the first existing file or None."""
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
        """Comma-joined candidate list for error messages."""
        return ", ".join(str(c) for c in self.candidates)

    def load(self) -> DataSourceResult:
        path = self._resolve()
        if path is None:
            raise FileNotFoundError(
                "No local fallback dataset found. Tried: " f"{self.describe_candidates()}"
            )
        if path.suffix.lower() not in (".xls", ".xlsx"):
            raise ValueError(
                f"Unsupported extension for local dataset: {path.suffix} "
                "(expected .xls or .xlsx)"
            )

        logger.info("Loading dataset from local file: %s", path)
        start = time.perf_counter()
        # UCI .xls quirk: row 0 holds the section banner "X1 | X2 | ..." and
        # the real column headers are on row 1. ``header=1`` skips the banner.
        df = pd.read_excel(path, header=1)
        elapsed = time.perf_counter() - start

        return DataSourceResult(
            dataframe=df,
            source_name=self.name,
            source_type="local",
            origin=str(path),
            duration_s=elapsed,
        )


# ---------------------------------------------------------------------------
# Chained source
# ---------------------------------------------------------------------------


class ChainedDataSource(DataSource):
    """Walk child sources in order; fall through on recoverable errors.

    Used for the canonical ``[UCI API, local .xls]`` chain. The error
    taxonomy below is load-bearing: anything classified as *recoverable*
    triggers a log warning and the next child; anything else escapes as
    ``DataIngestionError`` so real bugs (e.g. a ``TypeError`` from a broken
    monkey-patch) are not hidden behind a misleading "network down" message.
    """

    # ChainedDataSource is invoked when the primary is an API source, so
    # advertise "api" — mostly for logging shape; consumers read
    # ``DataSourceResult.source_type`` on the actual winning child.
    source_type: Literal["api"] = "api"

    def __init__(self, sources: Sequence[DataSource]) -> None:
        if not sources:
            raise ValueError("ChainedDataSource requires at least one child source")
        self.sources: Tuple[DataSource, ...] = tuple(sources)

    @property
    def name(self) -> str:
        return "Chain[" + " -> ".join(s.name for s in self.sources) + "]"

    # Recoverable errors: the chain keeps walking. Everything else is
    # re-raised as-is so genuine bugs surface. TypeError / MemoryError /
    # KeyboardInterrupt are deliberately NOT in this list.
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
                # Not a recoverable error — this is almost certainly a bug in
                # the source class itself. Log the class name + message, then
                # surface through DataIngestionError so the caller still gets
                # the full failure list for context.
                logger.error(
                    "Source '%s' raised unrecoverable %s -- aborting chain",
                    source.name,
                    type(exc).__name__,
                )
                failures.append((source.name, f"{type(exc).__name__}: {exc}"))
                raise DataIngestionError(failures) from exc

            if failures:
                logger.info(
                    "Recovered via fallback to '%s' after %d failed source(s)",
                    source.name,
                    len(failures),
                )

            # Re-wrap so ``failed_attempts`` carries the audit trail even
            # though the winning child didn't know about its predecessors.
            return DataSourceResult(
                dataframe=result.dataframe,
                source_name=result.source_name,
                source_type=result.source_type,
                origin=result.origin,
                duration_s=result.duration_s,
                failed_attempts=tuple(failures),
            )

        raise DataIngestionError(failures)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_default_data_source(
    data_path: Optional[PathLike] = None,
    *,
    mode: SourceMode = "auto",
    allow_fallback: bool = True,
    extra_local_candidates: Sequence[PathLike] = (),
    max_retries: int = 3,
    backoff_seconds: float = 1.5,
) -> DataSource:
    """Build the project's canonical data source.

    This is the single entry point the rest of the pipeline uses. Going
    through it guarantees a uniform provenance record and a single place
    to change retry / timeout policy.

    Parameters
    ----------
    data_path
        Pin a specific local file. When set, all other options are ignored
        and the returned source is a one-file ``LocalExcelSource``.
    mode
        ``"auto"``  — API first, then local fallback (chained).
        ``"api"``   — API only.
        ``"local"`` — local only (defaults + ``extra_local_candidates``).
    allow_fallback
        Only meaningful in ``"auto"`` mode. When ``False``, API failure
        is fatal rather than silently dropping to the local file.
    extra_local_candidates
        Appended to ``DEFAULT_LOCAL_CANDIDATES`` for the local source.
    max_retries, backoff_seconds
        Passed through to ``UCIRepoSource``.
    """
    if mode not in ("auto", "api", "local"):
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'auto', 'api', or 'local'.")

    # Pinned file path wins outright — callers that say "use this .xls"
    # should not see surprise network I/O if the file happens to be missing.
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

    # auto mode: chain only when fallback is allowed; otherwise API-only so
    # a CI run can hard-fail on network outage instead of loading stale data.
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
