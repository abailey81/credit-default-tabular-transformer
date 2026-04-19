"""Cross-cutting training utilities: RNG, devices, checkpoints, early-stop, timers.

Public surface
--------------
* :func:`set_deterministic` / :func:`derive_seed` — seed every RNG (Python,
  NumPy, PyTorch CPU+CUDA) and derive collision-safe sub-seeds via SHA-256.
* :func:`get_device` / :func:`describe_device` — auto-pick CUDA/MPS/CPU and
  render a one-line description for logs and checkpoint metadata.
* :func:`save_checkpoint` / :func:`load_checkpoint` — checkpoint I/O with a
  **security boundary**: see the :func:`load_checkpoint` docstring for the
  ``trust_source=True`` pickle-based path and SECURITY_AUDIT C-1.
* :class:`EarlyStopping` — patience-based min/max early stop with best-state
  stashing.
* :func:`count_parameters` / :func:`format_parameter_count` / :class:`Timer` /
  :func:`configure_logging` — small helpers the two training loops share.

Design choice
-------------
Checkpoints always bundle ``torch_version``, ``numpy_version``,
``python_version``, ``platform``, and ``git_sha`` alongside the tensors. This
is deliberate: stale weights produced by a different PyTorch or a pre-bugfix
commit can load cleanly under ``strict=True`` but silently misbehave (e.g. a
fixed init changed between builds). Embedding provenance makes those
mismatches visible in the logs on every load.

Determinism protocol: plan §16.5.1. Briefly, we set
``CUBLAS_WORKSPACE_CONFIG=:4096:8``, disable cuDNN autotune, and call
:func:`torch.use_deterministic_algorithms`. ``warn_only=True`` is the default
so ops without a deterministic kernel fall through with a warning rather than
crashing the training loop — set ``warn_only=False`` to hard-fail instead.

Non-obvious dependency: :func:`_git_sha` shells out to ``git rev-parse``. It's
wrapped in try/except so the module works inside environments without git
(Colab free tier, Docker minimal images) — the sha just falls back to
``"unknown"`` in that case."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Determinism (plan §16.5.1)
# -----------------------------------------------------------------------------


def set_deterministic(seed: int = 42, *, warn_only: bool = True) -> None:
    """Seed every RNG and disable non-deterministic CUDA kernels.

    Call exactly once at the top of a training entry point — before any model
    or loader is constructed — so every subsequent ``torch.rand``, dropout
    draw, and stratified sampler sees the same stream across runs.

    Parameters
    ----------
    seed
        Master seed. Propagated to :mod:`random`, :mod:`numpy.random`,
        :func:`torch.manual_seed`, and :func:`torch.cuda.manual_seed_all`.
        Also exported as ``PYTHONHASHSEED`` so dict iteration order inside
        child processes is stable.
    warn_only
        When True (default), ops without a deterministic CUDA kernel warn and
        fall back to the non-deterministic path. When False, the same ops
        **raise** — use this in CI when you want a hard fail instead of a
        silent non-reproducible result.

    Notes
    -----
    The ``CUBLAS_WORKSPACE_CONFIG`` nudge is required by cuBLAS ≥ 10.2 for
    deterministic matmul; without it ``use_deterministic_algorithms(True)``
    throws at the first GEMM.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # setdefault so an explicit outer env var wins — useful on shared servers.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # benchmark=True would let cuDNN pick the fastest kernel per input shape —
    # great for speed, fatal for reproducibility because the selected kernel
    # can vary across runs on the same hardware.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except TypeError:
        # Older torch versions don't accept warn_only; fall back to hard mode.
        torch.use_deterministic_algorithms(True)

    logger.info("Determinism protocol engaged (seed=%d, warn_only=%s)", seed, warn_only)


def derive_seed(parent_seed: int, *tags: str) -> int:
    """Hash ``(parent_seed, *tags)`` into a collision-safe 32-bit sub-seed.

    SHA-256 gives us plenty of headroom; we truncate to 4 bytes and unpack as
    big-endian unsigned so the output fits :func:`np.random.seed` / torch
    seeders without wrap-around. Use this to spawn per-fold / per-ensemble /
    per-worker RNG streams without collisions.

    Example::

        seed_fold_3 = derive_seed(42, "cv", "fold_3")  # always the same int
    """
    payload = f"{parent_seed}|" + "|".join(tags)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


# -----------------------------------------------------------------------------
# Device selection
# -----------------------------------------------------------------------------


def get_device(prefer: str = "auto") -> torch.device:
    """Resolve a :class:`torch.device` from a user-friendly string.

    Parameters
    ----------
    prefer
        One of:

        * ``"auto"`` — prefer CUDA, then MPS, fall back to CPU.
        * ``"cuda"`` / ``"mps"`` — require the requested backend; raise
          :class:`RuntimeError` if unavailable so misconfigured jobs fail
          loudly instead of silently running on CPU for an hour.
        * ``"cpu"`` — always use CPU (handy for deterministic CI).

    Raises
    ------
    RuntimeError
        If an explicit accelerator is requested but not available.
    ValueError
        On unknown strings.
    """
    prefer = prefer.lower()
    if prefer == "cpu":
        return torch.device("cpu")

    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")

    if prefer == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")

    if prefer != "auto":
        raise ValueError(f"Unknown device preference: {prefer!r}")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """Human-readable device description for logs and checkpoint metadata.

    Examples
    --------
    ``"CUDA:0 NVIDIA RTX A6000 (50.9 GB)"`` / ``"Apple Silicon MPS"`` /
    ``"CPU (AMD64 Family 23 Model 96)"``.
    """
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        mem_gb = torch.cuda.get_device_properties(idx).total_memory / 1e9
        return f"CUDA:{idx} {name} ({mem_gb:.1f} GB)"
    if device.type == "mps":
        return "Apple Silicon MPS"
    return f"CPU ({platform.processor() or platform.machine()})"


# -----------------------------------------------------------------------------
# Checkpoints (SECURITY-SENSITIVE — see load_checkpoint)
# -----------------------------------------------------------------------------


def _git_sha(default: str = "unknown") -> str:
    """Return ``git rev-parse HEAD`` or ``default`` when git is unavailable.

    Wrapped in try/except so importing this module never dies because of a
    missing git binary (Colab, minimal Docker, offline CI runners). The
    2-second timeout prevents pathological hangs in weird VCS states.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return default


@dataclass(frozen=True)
class CheckpointMetadata:
    """Provenance bundle embedded in every checkpoint we produce.

    Frozen because the metadata is a point-in-time snapshot — mutating it
    after ``save_checkpoint`` writes the JSON sidecar would desync the two
    views. Stored both inside the pickle payload *and* in a ``.meta.json``
    sidecar so weights-only loads still surface the provenance.
    """

    timestamp_utc: str
    torch_version: str
    numpy_version: str
    python_version: str
    platform: str
    git_sha: str
    seed: Optional[int]
    step: Optional[int]
    epoch: Optional[int]
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_checkpoint_metadata(
    seed: Optional[int] = None,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> CheckpointMetadata:
    """Capture the current runtime into a :class:`CheckpointMetadata`.

    ``extra`` is a free-form dict the caller can use to record run-specific
    context (role, best val score, config snapshot) without extending the
    dataclass schema.
    """
    return CheckpointMetadata(
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        torch_version=torch.__version__,
        numpy_version=np.__version__,
        python_version=platform.python_version(),
        platform=platform.platform(),
        git_sha=_git_sha(),
        seed=seed,
        step=step,
        epoch=epoch,
        extra=dict(extra or {}),
    )


def save_checkpoint(
    path: os.PathLike[str] | str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    metadata: Optional[CheckpointMetadata] = None,
) -> Path:
    """Persist a three-file checkpoint bundle with separated trust postures.

    This writes three artefacts with deliberately different security profiles,
    so downstream consumers can pick the load path that matches their trust
    in the source:

    ``<path>``
        Full pickle bundle (model + optimizer + scheduler + metadata).
        Requires ``trust_source=True`` to load via :func:`load_checkpoint`
        because :func:`torch.load` with ``weights_only=False`` invokes the
        pickle machinery and is a known RCE vector (SECURITY_AUDIT C-1).
    ``<path>.weights``
        Model state dict only. Loadable under ``weights_only=True``, which
        refuses to materialise arbitrary Python classes. Safe to distribute
        and the default path used by :func:`load_checkpoint`.
    ``<path>.meta.json``
        Plain-text sidecar with the provenance bundle — useful for
        ``ls``-style listing of ``results/`` without importing torch.

    Parameters
    ----------
    path
        Primary checkpoint path (typically ``.pt``). Parent directories are
        created if missing.
    model
        Source of :meth:`state_dict` for both the full payload and the
        ``.weights`` sidecar.
    optimizer, scheduler
        Optional training-state objects. Only written into the full pickle
        bundle; there is no safe way to restore their internal state under
        ``weights_only=True``.
    metadata
        Explicit :class:`CheckpointMetadata`. If ``None``, one is built from
        the current runtime via :func:`build_checkpoint_metadata`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if metadata is None:
        metadata = build_checkpoint_metadata()

    payload: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "metadata": metadata.to_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    # Full-bundle pickle — consumers that want optimiser/scheduler must opt in
    # to trust_source=True when loading.
    torch.save(payload, path)

    # Weights-only sidecar: safe default load path.
    weights_path = path.with_suffix(path.suffix + ".weights")
    torch.save(model.state_dict(), weights_path)

    # JSON provenance: readable without torch, listed by SPLIT_HASHES etc.
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(metadata.to_dict(), indent=2))

    logger.info(
        "Checkpoint saved -> %s (+ %s, %s)",
        path,
        weights_path.name,
        meta_path.name,
    )
    return path


def load_checkpoint(
    path: os.PathLike[str] | str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    *,
    strict: bool = True,
    map_location: Optional[torch.device | str] = None,
    trust_source: bool = False,
) -> Dict[str, Any]:
    """Load a checkpoint produced by :func:`save_checkpoint`.

    Security model (SECURITY_AUDIT C-1)
    -----------------------------------
    :func:`torch.load` with ``weights_only=False`` is **backed by the Python
    pickle machinery**, which is a documented remote-code-execution vector:
    deserialising a malicious checkpoint can invoke arbitrary code on the host
    (see CVE-2023-48022 and the broader family of "pickle RCE" advisories).
    This function therefore exposes **two mutually exclusive load modes**:

    ``trust_source=False`` **(default, safe)**
        Reads the ``<path>.weights`` sidecar under
        :func:`torch.load(..., weights_only=True)`, which refuses to
        materialise anything except tensors + a small allowlist of PyTorch
        types. Optimiser / scheduler state **cannot** be restored this way —
        pass a warning is logged if requested. Use this for any checkpoint
        not produced by your own trusted pipeline.
    ``trust_source=True`` **(dangerous — opt-in only)**
        Loads the full pickle bundle with ``weights_only=False``. Only pass
        this for checkpoints *you personally produced* on this host. Never
        pass this for artefacts downloaded from HuggingFace, model zoos,
        user-uploaded blobs, or any other untrusted origin.

    Parameters
    ----------
    path
        Primary checkpoint path as written by :func:`save_checkpoint`. The
        ``.weights`` and ``.meta.json`` sidecars are looked up alongside.
    model
        Target module. Its :meth:`state_dict` is overwritten in place.
    optimizer, scheduler
        Only restored on the ``trust_source=True`` path; otherwise a warning
        is logged and their state is left untouched.
    strict
        Forwarded to :meth:`nn.Module.load_state_dict`. False allows missing
        keys (used by the MTLM fine-tune path, which loads encoder-only).
    map_location
        Forwarded to :func:`torch.load`; pass a device to remap CUDA→CPU.
    trust_source
        See security model above.

    Returns
    -------
    dict
        A dict with at least ``"metadata"``; on the ``trust_source=True`` path
        also contains ``"model_state"`` / ``"optimizer_state"`` /
        ``"scheduler_state"`` if present.

    Raises
    ------
    FileNotFoundError
        If the primary file or (in weights-only mode) the ``.weights``
        sidecar is missing.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No checkpoint found at: {path}")

    weights_path = path.with_suffix(path.suffix + ".weights")
    meta_path = path.with_suffix(path.suffix + ".meta.json")

    if trust_source:
        # SECURITY_AUDIT C-1: weights_only=False invokes the pickle machinery
        # on the file contents. Only ever reach this branch for checkpoints
        # produced by our own save_checkpoint() on a trusted host — never for
        # externally-sourced artefacts.
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=strict)
        if optimizer is not None and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None and "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
    else:
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"No weights-only sidecar found at {weights_path}. "
                f"Either use trust_source=True (then {path} is loaded with pickle -- "
                "only do this for checkpoints you produced) or ensure the sidecar "
                "was saved by save_checkpoint()."
            )
        # Safe path: weights_only=True rejects anything other than tensors
        # and an allowlist of PyTorch container types.
        state = torch.load(weights_path, map_location=map_location, weights_only=True)
        model.load_state_dict(state, strict=strict)

        # Metadata is in its own JSON sidecar on the safe path — keeps
        # provenance visible without forcing a pickle load.
        if meta_path.is_file():
            checkpoint = {"metadata": json.loads(meta_path.read_text())}
        else:
            checkpoint = {"metadata": {}}

        if optimizer is not None or scheduler is not None:
            # weights-only cannot carry optimiser/scheduler state — warn so
            # the caller doesn't silently resume from a fresh Adam momentum.
            logger.warning(
                "load_checkpoint: optimizer/scheduler requested but trust_source=False. "
                "Their state will NOT be restored. Pass trust_source=True to load "
                "the full pickle bundle (only for checkpoints you produced)."
            )

    meta = checkpoint.get("metadata", {})
    logger.info(
        "Checkpoint loaded <- %s (git=%s, torch=%s, epoch=%s, step=%s, trust_source=%s)",
        path if trust_source else weights_path,
        meta.get("git_sha", "?"),
        meta.get("torch_version", "?"),
        meta.get("epoch", "?"),
        meta.get("step", "?"),
        trust_source,
    )
    return checkpoint


# -----------------------------------------------------------------------------
# Early stopping
# -----------------------------------------------------------------------------


class EarlyStopping:
    """Stop training when a monitored metric plateaus, keeping the best weights.

    The contract is: call :meth:`step` once per epoch with the metric value and
    optionally the current ``model.state_dict()``. It tracks the best score
    seen and, after ``patience`` consecutive non-improvements, signals the
    training loop to stop by returning True. The best-epoch state is stashed
    in :attr:`best_state` (CPU-cloned) so the caller can restore it after the
    loop exits — this is how ``train.py`` and ``train_mtlm.py`` both end up
    evaluating on the best model rather than the last.

    Parameters
    ----------
    patience
        Number of consecutive non-improvements tolerated before stopping.
        Must be ≥ 1.
    mode
        ``"max"`` for metrics where higher is better (AUC, F1) or ``"min"``
        for losses.
    min_delta
        Minimum change in the monitored metric to count as an improvement.
        0.0 means "any strict improvement"; 1e-4 is a common choice that
        ignores epoch-to-epoch noise floors.
    """

    def __init__(
        self,
        patience: int = 20,
        mode: str = "max",
        min_delta: float = 0.0,
    ):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")

        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best_score: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.best_state: Optional[Dict[str, Any]] = None
        self.counter = 0
        self.epoch = 0
        self.stopped = False

    def _improved(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def step(
        self,
        score: float,
        state: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Record a new score; return True when training should stop.

        Parameters
        ----------
        score
            Latest value of the monitored metric.
        state
            Optional ``model.state_dict()``. When provided and this is a new
            best, we CPU-clone it into :attr:`best_state` so the caller can
            restore it after the loop. Cloning decouples the snapshot from
            any further in-place mutation the optimiser does, and moving to
            CPU saves GPU memory across long runs.

        Returns
        -------
        bool
            True if ``patience`` consecutive non-improvements have been
            observed. The caller is expected to break out of the epoch loop.
        """
        self.epoch += 1

        if self._improved(score):
            self.best_score = score
            self.best_epoch = self.epoch
            self.counter = 0
            if state is not None:
                # Detach → clone → CPU: three separate safety properties.
                # detach unhooks from autograd; clone makes a deep copy of the
                # storage (so in-place opt.step() doesn't bleed in); .cpu()
                # keeps the snapshot off the GPU so it survives a device swap.
                self.best_state = {k: v.detach().cpu().clone() for k, v in state.items()}
            return False

        self.counter += 1
        if self.counter >= self.patience:
            self.stopped = True
            return True
        return False


# -----------------------------------------------------------------------------
# Parameter counts and timing
# -----------------------------------------------------------------------------


def count_parameters(
    model: torch.nn.Module,
    trainable_only: bool = True,
) -> int:
    """Sum :attr:`nn.Parameter.numel` across the module tree.

    Parameters
    ----------
    trainable_only
        When True (default), skip parameters with ``requires_grad=False``
        (e.g. the frozen embedding of a pretrained encoder during a warm-up
        phase). Pass False to count every tensor registered as a parameter.
    """
    return sum(p.numel() for p in model.parameters() if (not trainable_only) or p.requires_grad)


def format_parameter_count(n: int) -> str:
    """Render a parameter count as a short human-readable string.

    Examples: ``28_512 → '28.5K'``, ``12_345_678 → '12.35M'``.
    """
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


class Timer:
    """Tiny context manager for wall-clock timing.

    Usage::

        with Timer("epoch") as t:
            ...
        print(t.elapsed)  # seconds

    Uses :func:`time.perf_counter` so it's immune to system clock adjustments.
    """

    def __init__(self, label: str = "block"):
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = time.perf_counter() - self._start
        logger.debug("Timer(%s) = %.3fs", self.label, self.elapsed)


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------


def configure_logging(level: int = logging.INFO) -> None:
    """Attach a formatted stream handler to the root logger, idempotently.

    Idempotency matters because this gets called from both the CLI entry
    points and notebook cells — duplicate handlers would duplicate every log
    line. We tag the handler with a ``_credit_default_handler`` attribute and
    refuse to attach a second one.
    """
    root = logging.getLogger()
    if any(getattr(h, "_credit_default_handler", False) for h in root.handlers):
        root.setLevel(level)
        return

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    handler._credit_default_handler = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level)


# Smoke test

if __name__ == "__main__":
    configure_logging()

    print("-- determinism --")
    set_deterministic(seed=42, warn_only=True)
    a = torch.randn(3)
    set_deterministic(seed=42, warn_only=True)
    b = torch.randn(3)
    assert torch.equal(a, b), "Determinism broken"
    print(f"  two seeds -> identical tensors  ({a.tolist()})")

    sub1 = derive_seed(42, "phase_a", "run_0")
    sub2 = derive_seed(42, "phase_a", "run_0")
    assert sub1 == sub2, "derive_seed is not deterministic"
    assert derive_seed(42, "phase_a", "run_0") != derive_seed(42, "phase_a", "run_1")
    print(f"  derive_seed: {sub1}  (reproducible, collision-safe)")

    print("\n-- device --")
    dev = get_device("auto")
    print(f"  auto device: {dev}  ({describe_device(dev)})")
    assert get_device("cpu") == torch.device("cpu")

    print("\n-- checkpoint --")
    import tempfile

    model = torch.nn.Linear(10, 3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as tmp:
        cp_path = Path(tmp) / "ckpt.pt"
        md = build_checkpoint_metadata(seed=42, epoch=5, step=1234, extra={"note": "smoke"})
        save_checkpoint(cp_path, model, optimizer=opt, metadata=md)

        assert cp_path.is_file(), "checkpoint file missing"
        meta_json = cp_path.with_suffix(".pt.meta.json")
        assert meta_json.is_file(), "sidecar meta.json missing"
        meta_loaded = json.loads(meta_json.read_text())
        assert meta_loaded["seed"] == 42
        assert meta_loaded["epoch"] == 5
        print(f"  saved {cp_path.stat().st_size}B + sidecar meta")

        model2 = torch.nn.Linear(10, 3)
        opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        loaded = load_checkpoint(cp_path, model2, optimizer=opt2, trust_source=True)
        for (k1, v1), (k2, v2) in zip(
            model.state_dict().items(), model2.state_dict().items(), strict=True
        ):
            assert torch.equal(v1, v2), f"Round-trip mismatch on {k1}"
        assert loaded["metadata"]["seed"] == 42
        print("  round-trip equal")

        model3 = torch.nn.Linear(10, 3)
        loaded_safe = load_checkpoint(cp_path, model3)
        for (_, v1), (_, v2) in zip(model.state_dict().items(), model3.state_dict().items()):
            assert torch.equal(v1, v2)
        assert loaded_safe["metadata"]["seed"] == 42
        print("  safe-default load (weights_only=True) works")

    print("\n-- early stopping --")
    es = EarlyStopping(patience=3, mode="max", min_delta=1e-4)
    sequence = [0.70, 0.72, 0.73, 0.729, 0.729, 0.729, 0.729]
    triggered_at = None
    for i, v in enumerate(sequence, start=1):
        if es.step(v, state=model.state_dict()):
            triggered_at = i
            break
    assert triggered_at == 6, f"Expected early-stop at epoch 6, got {triggered_at}"
    assert abs(es.best_score - 0.73) < 1e-9
    assert es.best_epoch == 3
    assert es.best_state is not None, "best_state not stashed"
    print(f"  early-stopped at epoch {triggered_at}, best={es.best_score} @ epoch {es.best_epoch}")

    print("\n-- accounting --")
    n = count_parameters(model)
    assert n == 10 * 3 + 3, f"Unexpected param count {n}"
    assert format_parameter_count(28_512) == "28.5K"
    assert format_parameter_count(12_345_678) == "12.35M"
    with Timer("dummy") as t:
        sum(range(10_000))
    assert t.elapsed >= 0
    print(f"  params: {format_parameter_count(n)}  timer: {t.elapsed*1000:.2f} ms")

    print("\nAll utils smoke tests passed.")
