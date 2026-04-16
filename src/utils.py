"""
utils.py — Foundational training utilities.

Covers:
    1. Deterministic-training setup (seed everything, cuDNN determinism, hash seed).
    2. Device detection (CPU / CUDA / MPS Apple Silicon).
    3. Checkpoint save/load with integrity metadata.
    4. Early stopping on a monitored metric.
    5. Parameter / FLOP / wall-time accounting helpers.

Design aims:
    - Zero ML-domain knowledge — pure infrastructure.
    - Works identically on CPU, CUDA, and MPS.
    - No implicit global state: every function takes its inputs explicitly.
    - Checkpoint files carry metadata (torch version, git SHA, seed, timestamp)
      so a stale checkpoint is never silently loaded into a newer model.

Reference: PROJECT_PLAN.md §16.5.1 (determinism protocol).
"""

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
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Determinism
# ──────────────────────────────────────────────────────────────────────────────


def set_deterministic(seed: int = 42, *, warn_only: bool = True) -> None:
    """
    Seed every RNG in the stack and disable non-deterministic CUDA kernels.

    Follows PROJECT_PLAN.md §16.5.1. Call once at the top of every training
    entry point. Setting ``warn_only=True`` means torch will warn (not error)
    when a non-deterministic op is invoked for which there is no deterministic
    substitute — appropriate for research code where "best-effort determinism"
    is acceptable. Set to ``False`` to error out instead.

    Parameters
    ----------
    seed
        The seed used for every RNG.
    warn_only
        Pass to ``torch.use_deterministic_algorithms(..., warn_only=warn_only)``.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # cuBLAS determinism
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN — disable the auto-tuner + flag deterministic algorithms.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Global torch determinism switch (available since 1.8). May raise on some
    # ops that have no deterministic implementation (e.g. interpolate bilinear).
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except TypeError:
        # Older torch versions do not support warn_only kwarg.
        torch.use_deterministic_algorithms(True)

    logger.info("Determinism protocol engaged (seed=%d, warn_only=%s)", seed, warn_only)


def derive_seed(parent_seed: int, *tags: str) -> int:
    """
    Derive a deterministic sub-seed from a parent seed + a tuple of string tags.

    Useful when multiple stages of a pipeline each need their own seed that is
    reproducible but not the same as the parent's. Uses a hash so collisions
    are improbable.

    Example::

        parent = 42
        split_seed = derive_seed(parent, "stratified_split")
        init_seed  = derive_seed(parent, "model_init", "run_0")
    """
    payload = f"{parent_seed}|" + "|".join(tags)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    # Take first 4 bytes as a non-negative 32-bit int.
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Device selection
# ──────────────────────────────────────────────────────────────────────────────


def get_device(prefer: str = "auto") -> torch.device:
    """
    Resolve the best available compute device.

    ``prefer`` is one of:
        "auto"  — CUDA if available, else MPS if available, else CPU
        "cpu"   — always CPU
        "cuda"  — CUDA or raise
        "mps"   — Apple Silicon MPS or raise
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

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """Human-readable device description for logs and checkpoints."""
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        mem_gb = torch.cuda.get_device_properties(idx).total_memory / 1e9
        return f"CUDA:{idx} {name} ({mem_gb:.1f} GB)"
    if device.type == "mps":
        return "Apple Silicon MPS"
    return f"CPU ({platform.processor() or platform.machine()})"


# ──────────────────────────────────────────────────────────────────────────────
# 3. Git-aware checkpoint save/load
# ──────────────────────────────────────────────────────────────────────────────


def _git_sha(default: str = "unknown") -> str:
    """Return the current git HEAD SHA, or ``default`` if not in a git repo."""
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
    """Metadata bundled inside every checkpoint for integrity auditing."""

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
    """
    Save a training checkpoint — three files, each with a different trust posture.

    1. ``<path>`` — full pickle bundle containing ``model_state``,
       ``optimizer_state`` (if supplied), ``scheduler_state`` (if supplied),
       and ``metadata``. Only safe to load under ``trust_source=True`` (pickle).
    2. ``<path>.weights`` — plain state-dict of model weights ONLY, loadable
       under ``torch.load(..., weights_only=True)``. This is the default path
       for :func:`load_checkpoint`.
    3. ``<path>.meta.json`` — metadata sidecar for checkpoint-listing scripts
       that don't want to materialise tensors.

    The split closes SECURITY_AUDIT C-1: the common load path never touches
    ``weights_only=False``, so an attacker-controlled ``.pt`` cannot execute
    pickle payloads unless the caller explicitly opts in via
    ``load_checkpoint(..., trust_source=True)``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if metadata is None:
        metadata = build_checkpoint_metadata()

    # 1. Full bundle (optimizer + scheduler require pickle).
    payload: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "metadata": metadata.to_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    torch.save(payload, path)

    # 2. Weights-only sidecar for safe loading.
    weights_path = path.with_suffix(path.suffix + ".weights")
    torch.save(model.state_dict(), weights_path)

    # 3. JSON metadata sidecar.
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(metadata.to_dict(), indent=2))

    logger.info(
        "Checkpoint saved → %s (+ %s, %s)",
        path, weights_path.name, meta_path.name,
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
    """
    Load a checkpoint produced by :func:`save_checkpoint`.

    Security-hardened loader (closes SECURITY_AUDIT C-1 / H-1):

    * **Default (``trust_source=False``)**: uses ``weights_only=True``. Only
      tensors / numpy arrays are deserialised; pickle payloads are refused
      at the ``torch.load`` layer. Optimizer / scheduler state cannot be
      restored in this mode (their dicts contain arbitrary Python objects).
      Suitable for any checkpoint from an untrusted source.

    * **Explicit opt-in (``trust_source=True``)**: uses ``weights_only=False``
      to permit optimizer / scheduler state. **Only** pass this for files
      you produced yourself on this machine. A checkpoint downloaded from a
      URL, received from a collaborator, or loaded via a future callsite
      that forwards untrusted paths is an RCE vector in this mode.

    Note: the pinned ``torch==2.2.x`` is affected by PYSEC-2025-41 which shows
    ``weights_only=True`` is also exploitable on torch < 2.6. Upgrade to
    torch ≥ 2.8 when the Intel-Mac wheel situation allows (see SECURITY.md
    in the repo root). Until then, treat *every* checkpoint file as a trust
    boundary and never load one you did not produce.

    Returns the full loaded dict so callers can inspect metadata / extra fields.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No checkpoint found at: {path}")

    # Paths to the three artefacts produced by save_checkpoint.
    weights_path = path.with_suffix(path.suffix + ".weights")
    meta_path = path.with_suffix(path.suffix + ".meta.json")

    if trust_source:
        # Full pickle load: restores optimizer / scheduler state if requested.
        checkpoint = torch.load(
            path, map_location=map_location, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state"], strict=strict)
        if optimizer is not None and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None and "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
    else:
        # Safe default: load ONLY the weights-only sidecar.
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"No weights-only sidecar found at {weights_path}. "
                f"Either use trust_source=True (then {path} is loaded with pickle — "
                "only do this for checkpoints you produced) or ensure the sidecar "
                "was saved by save_checkpoint()."
            )
        state = torch.load(
            weights_path, map_location=map_location, weights_only=True
        )
        model.load_state_dict(state, strict=strict)

        # Metadata comes from the JSON sidecar.
        if meta_path.is_file():
            checkpoint = {"metadata": json.loads(meta_path.read_text())}
        else:
            checkpoint = {"metadata": {}}

        if optimizer is not None or scheduler is not None:
            logger.warning(
                "load_checkpoint: optimizer/scheduler requested but trust_source=False. "
                "Their state will NOT be restored. Pass trust_source=True to load "
                "the full pickle bundle (only for checkpoints you produced)."
            )

    meta = checkpoint.get("metadata", {})
    logger.info(
        "Checkpoint loaded ← %s (git=%s, torch=%s, epoch=%s, step=%s, trust_source=%s)",
        path if trust_source else weights_path,
        meta.get("git_sha", "?"),
        meta.get("torch_version", "?"),
        meta.get("epoch", "?"),
        meta.get("step", "?"),
        trust_source,
    )
    return checkpoint


# ──────────────────────────────────────────────────────────────────────────────
# 4. Early stopping
# ──────────────────────────────────────────────────────────────────────────────


class EarlyStopping:
    """
    Stops training when a monitored metric fails to improve for ``patience`` epochs.

    Supports both minimisation (lower is better, e.g. loss) and maximisation
    (higher is better, e.g. AUC-ROC).

    Example::

        early = EarlyStopping(patience=20, mode="max", min_delta=1e-4)
        for epoch in range(max_epochs):
            val_auc = ...
            if early.step(val_auc, state=model.state_dict()):
                model.load_state_dict(early.best_state)
                break
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
        """
        Record a new score. Returns True when training should stop.

        If ``state`` is passed (typically ``model.state_dict()``), a deep copy
        is stashed on every improvement so ``self.best_state`` holds the best
        weights seen so far — restore them on stop to recover the best model.
        """
        self.epoch += 1

        if self._improved(score):
            self.best_score = score
            self.best_epoch = self.epoch
            self.counter = 0
            if state is not None:
                # Deep-copy via CPU state-dict clone to decouple from live model.
                self.best_state = {k: v.detach().cpu().clone() for k, v in state.items()}
            return False

        self.counter += 1
        if self.counter >= self.patience:
            self.stopped = True
            return True
        return False


# ──────────────────────────────────────────────────────────────────────────────
# 5. Parameter accounting / timing
# ──────────────────────────────────────────────────────────────────────────────


def count_parameters(
    model: torch.nn.Module,
    trainable_only: bool = True,
) -> int:
    """Number of (trainable) parameters in a model."""
    return sum(
        p.numel()
        for p in model.parameters()
        if (not trainable_only) or p.requires_grad
    )


def format_parameter_count(n: int) -> str:
    """Pretty-print a parameter count, e.g. 28_512 → '28.5K', 12_345_678 → '12.3M'."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


class Timer:
    """Tiny context-manager timer that records wall-clock seconds."""

    def __init__(self, label: str = "block"):
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = time.perf_counter() - self._start
        logger.debug("Timer(%s) = %.3fs", self.label, self.elapsed)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Logging setup
# ──────────────────────────────────────────────────────────────────────────────


def configure_logging(level: int = logging.INFO) -> None:
    """
    Idempotent logging config suitable for CLI + notebook use.

    Uses the root logger; every module that does ``logger = logging.getLogger(__name__)``
    automatically inherits the format. Safe to call multiple times — handlers
    are deduplicated.
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


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    configure_logging()

    print("── determinism ──")
    set_deterministic(seed=42, warn_only=True)
    a = torch.randn(3)
    set_deterministic(seed=42, warn_only=True)
    b = torch.randn(3)
    assert torch.equal(a, b), "Determinism broken"
    print(f"  two seeds → identical tensors ✓  ({a.tolist()})")

    sub1 = derive_seed(42, "phase_a", "run_0")
    sub2 = derive_seed(42, "phase_a", "run_0")
    assert sub1 == sub2, "derive_seed is not deterministic"
    assert derive_seed(42, "phase_a", "run_0") != derive_seed(42, "phase_a", "run_1")
    print(f"  derive_seed: {sub1}  (reproducible, collision-safe) ✓")

    print("\n── device ──")
    dev = get_device("auto")
    print(f"  auto device: {dev}  ({describe_device(dev)})")
    assert get_device("cpu") == torch.device("cpu")

    print("\n── checkpoint ──")
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
        print(f"  saved {cp_path.stat().st_size}B + sidecar meta ✓")

        # Round-trip into a fresh model. trust_source=True because we just saved
        # the file ourselves and want optimiser state restored.
        model2 = torch.nn.Linear(10, 3)
        opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        loaded = load_checkpoint(cp_path, model2, optimizer=opt2, trust_source=True)
        for (k1, v1), (k2, v2) in zip(
            model.state_dict().items(), model2.state_dict().items(), strict=True
        ):
            assert torch.equal(v1, v2), f"Round-trip mismatch on {k1}"
        assert loaded["metadata"]["seed"] == 42
        print("  round-trip equal ✓")

        # Safe default (trust_source=False) also works for weight-only loads.
        model3 = torch.nn.Linear(10, 3)
        loaded_safe = load_checkpoint(cp_path, model3)
        for (_, v1), (_, v2) in zip(
            model.state_dict().items(), model3.state_dict().items()
        ):
            assert torch.equal(v1, v2)
        assert loaded_safe["metadata"]["seed"] == 42
        print("  safe-default load (weights_only=True) works ✓")

    print("\n── early stopping ──")
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
    print(f"  early-stopped at epoch {triggered_at}, best={es.best_score} @ epoch {es.best_epoch} ✓")

    print("\n── accounting ──")
    n = count_parameters(model)
    assert n == 10 * 3 + 3, f"Unexpected param count {n}"
    assert format_parameter_count(28_512) == "28.5K"
    assert format_parameter_count(12_345_678) == "12.35M"
    with Timer("dummy") as t:
        sum(range(10_000))
    assert t.elapsed >= 0
    print(f"  params: {format_parameter_count(n)} ✓  timer: {t.elapsed*1000:.2f} ms")

    print("\nAll utils smoke tests passed.")
