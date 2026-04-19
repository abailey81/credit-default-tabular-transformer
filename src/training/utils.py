"""Training glue: seeding, device pick, checkpoint save/load, early stopping, timers.
Checkpoints carry torch/numpy/python versions + git sha so stale weights can't
silently ride along into a new model. Determinism protocol: plan §16.5.1."""

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


# determinism

def set_deterministic(seed: int = 42, *, warn_only: bool = True) -> None:
    """Seed every RNG and turn off non-deterministic CUDA kernels. Call once at
    the top of a training entry point. warn_only=False to hard-fail on ops with
    no deterministic substitute."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except TypeError:
        torch.use_deterministic_algorithms(True)

    logger.info("Determinism protocol engaged (seed=%d, warn_only=%s)", seed, warn_only)


def derive_seed(parent_seed: int, *tags: str) -> int:
    """Hash (parent_seed, *tags) to a reproducible sub-seed."""
    payload = f"{parent_seed}|" + "|".join(tags)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


# device

def get_device(prefer: str = "auto") -> torch.device:
    """auto = CUDA > MPS > CPU. Explicit "cuda"/"mps" raises if unavailable."""
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
    """Pretty string for logs/checkpoints."""
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        mem_gb = torch.cuda.get_device_properties(idx).total_memory / 1e9
        return f"CUDA:{idx} {name} ({mem_gb:.1f} GB)"
    if device.type == "mps":
        return "Apple Silicon MPS"
    return f"CPU ({platform.processor() or platform.machine()})"


# checkpoints

def _git_sha(default: str = "unknown") -> str:
    """git HEAD sha, or default if git isn't around."""
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
    """Provenance bundle stored inside every checkpoint."""

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
    """Write three files with different trust postures.

    <path>: full pickle bundle (model + optimizer + scheduler + metadata),
    needs trust_source=True to load.
    <path>.weights: model state_dict only, loads under weights_only=True. Default.
    <path>.meta.json: JSON sidecar for listing/inspection without touching tensors.
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
    torch.save(payload, path)

    weights_path = path.with_suffix(path.suffix + ".weights")
    torch.save(model.state_dict(), weights_path)

    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(metadata.to_dict(), indent=2))

    logger.info(
        "Checkpoint saved -> %s (+ %s, %s)",
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
    """Load a checkpoint produced by save_checkpoint into model (+ opt/sched).

    trust_source=False (default) reads the .weights sidecar under weights_only=True;
    opt/sched state cannot be restored this way.
    trust_source=True loads the pickle bundle via torch.load. Only do this for
    files you produced yourself — every external checkpoint is a trust boundary.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No checkpoint found at: {path}")

    weights_path = path.with_suffix(path.suffix + ".weights")
    meta_path = path.with_suffix(path.suffix + ".meta.json")

    if trust_source:
        checkpoint = torch.load(
            path, map_location=map_location, weights_only=False
        )
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
        state = torch.load(
            weights_path, map_location=map_location, weights_only=True
        )
        model.load_state_dict(state, strict=strict)

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
        "Checkpoint loaded <- %s (git=%s, torch=%s, epoch=%s, step=%s, trust_source=%s)",
        path if trust_source else weights_path,
        meta.get("git_sha", "?"),
        meta.get("torch_version", "?"),
        meta.get("epoch", "?"),
        meta.get("step", "?"),
        trust_source,
    )
    return checkpoint


# early stopping

class EarlyStopping:
    """Stops when the monitored metric quits improving. mode="min"|"max".
    Pass state=model.state_dict() to step() and best_state keeps the best weights."""

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
        """Record a new score. True = stop."""
        self.epoch += 1

        if self._improved(score):
            self.best_score = score
            self.best_epoch = self.epoch
            self.counter = 0
            if state is not None:
                self.best_state = {k: v.detach().cpu().clone() for k, v in state.items()}
            return False

        self.counter += 1
        if self.counter >= self.patience:
            self.stopped = True
            return True
        return False


# param counts and timing

def count_parameters(
    model: torch.nn.Module,
    trainable_only: bool = True,
) -> int:
    """Count (trainable) params."""
    return sum(
        p.numel()
        for p in model.parameters()
        if (not trainable_only) or p.requires_grad
    )


def format_parameter_count(n: int) -> str:
    """28_512 → '28.5K'."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


class Timer:
    """with Timer("x") as t: ...; t.elapsed in seconds."""

    def __init__(self, label: str = "block"):
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = time.perf_counter() - self._start
        logger.debug("Timer(%s) = %.3fs", self.label, self.elapsed)


# logging

def configure_logging(level: int = logging.INFO) -> None:
    """Idempotent; safe to call repeatedly from CLI and notebooks."""
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
        for (_, v1), (_, v2) in zip(
            model.state_dict().items(), model3.state_dict().items()
        ):
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
