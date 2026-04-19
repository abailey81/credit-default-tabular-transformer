"""Binary-classification losses tuned for the 22.1 %-positive credit-default task.

Exposes three :class:`nn.Module` losses and two helpers:

* :class:`WeightedBCELoss` — the straight ``pos_weight``-scaled BCE baseline.
* :class:`FocalLoss` — Lin+ 2017, with support for scalar / tuple / ``"balanced"``
  class-weight priors (see :func:`balanced_alpha`).
* :class:`LabelSmoothingBCELoss` — ε-smoothed BCE, ε = 0.05 per plan §7.3.
* :func:`compute_pos_weight` / :func:`balanced_alpha` — helpers the training
  entry points use at build time.

All three share the same ``(logits, targets)`` forward signature and route
through :func:`F.binary_cross_entropy_with_logits` rather than
:func:`F.sigmoid` + BCE on probabilities. That choice is load-bearing:
``binary_cross_entropy_with_logits`` uses the log-sum-exp trick internally, so
the loss stays finite for very confident logits (|z| > 30) where a naive
``log(sigmoid(z))`` would underflow to ``-inf``. For focal specifically, we
also use the identity ``exp(-BCE) == p_t`` to recover the true-class probability
without a second :func:`sigmoid` call — slightly cheaper and avoids an extra
rounding step.

Non-obvious dependency: none — this file is leaf-level and imports no sibling
modules, which is why it doubles as the target for quick unit tests of the
gradient flow (bottom-of-file smoke test)."""

from __future__ import annotations

import logging
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

Reduction = Literal["mean", "sum", "none"]


def compute_pos_weight(y: torch.Tensor) -> torch.Tensor:
    """Return ``N_neg / N_pos`` as a scalar tensor for :class:`nn.BCEWithLogitsLoss`.

    PyTorch's ``pos_weight`` multiplies the positive-class term of the BCE;
    ``N_neg / N_pos`` is the "balanced" choice that makes positive and
    negative contributions sum to the same aggregate weight. On our
    22.1 %-positive training split this evaluates to ~3.52.

    Parameters
    ----------
    y
        1-D label tensor with values in ``{0, 1}``. Any shape is accepted —
        it is flattened implicitly via :meth:`Tensor.sum`.

    Raises
    ------
    ValueError
        If there are zero positives in ``y`` (ratio is undefined).
    """
    y = y.float()
    pos = y.sum()
    neg = y.numel() - pos
    if pos <= 0:
        raise ValueError("pos_weight is undefined when the training set has no positives")
    return neg / pos


def balanced_alpha(y: torch.Tensor) -> Tuple[float, float]:
    """Compute class-balancing ``(α_pos, α_neg)`` priors for focal loss.

    Uses the standard inverse-frequency rule::

        α_pos = N_neg / N   ,   α_neg = N_pos / N

    On our split this yields ``(0.779, 0.221)`` — i.e. the positive class is
    up-weighted by ~3.52× so the two classes contribute equally in aggregate.
    Complements :class:`FocalLoss`'s ``α="balanced"`` mode which calls this at
    build time.

    Returns
    -------
    tuple of float
        ``(α_pos, α_neg)``. Python floats, not tensors — the caller
        tensor-ifies when resolving per-sample weights.

    Raises
    ------
    ValueError
        If either class is empty.
    """
    y = y.float()
    n = float(y.numel())
    pos = float(y.sum().item())
    neg = n - pos
    if pos <= 0 or neg <= 0:
        raise ValueError("balanced_alpha requires both classes to be present")
    alpha_pos = neg / n
    alpha_neg = pos / n
    return alpha_pos, alpha_neg


class WeightedBCELoss(nn.Module):
    """Binary cross-entropy with optional positive-class up-weighting.

    Wraps :func:`F.binary_cross_entropy_with_logits` with a ``pos_weight``
    buffer so the module can move between devices with the rest of the model.
    The loss is::

        L = −[pos_weight · y · log σ(z) + (1 − y) · log(1 − σ(z))]

    Parameters
    ----------
    pos_weight
        Scalar up-weight applied to the positive-class term. Pass
        :func:`compute_pos_weight` output for class-balanced BCE, ``None`` to
        disable. Accepts float or tensor; float is wrapped into a tensor so
        ``register_buffer`` can follow ``.to(device)``.
    reduction
        ``"mean"`` (default), ``"sum"``, or ``"none"`` — same semantics as
        :class:`torch.nn.BCEWithLogitsLoss`.

    Notes
    -----
    ``pos_weight`` is stored as a **non-persistent** buffer so checkpoint
    round-trips don't pickle it with the state dict; callers reconstruct the
    weight from the training data at load time.
    """

    def __init__(
        self,
        pos_weight: Optional[float | torch.Tensor] = None,
        reduction: Reduction = "mean",
    ):
        super().__init__()
        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor(float(pos_weight))
        # Buffer (not Parameter) so no grads flow; non-persistent so it stays
        # out of state_dict — we treat pos_weight as a build-time scalar, not
        # something the optimiser should tune.
        self.register_buffer(
            "pos_weight",
            pos_weight if isinstance(pos_weight, torch.Tensor) else None,  # type: ignore[arg-type]
            persistent=False,
        )
        self.reduction: Reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # Flatten + cast: upstream may hand us (B, 1) or (B,), and label dtype
        # varies between long (from DataLoader) and float.
        logits = logits.view(-1).float()
        targets = targets.view(-1).float()
        pos_weight = self.pos_weight if self.pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction=self.reduction
        )


class FocalLoss(nn.Module):
    """Focal loss for imbalanced binary classification (Lin+ 2017).

    Computes::

        FL(p_t) = −α_t · (1 − p_t)^γ · log p_t

    where ``p_t`` is the model's predicted probability of the true class. The
    ``(1 − p_t)^γ`` factor down-weights easy examples (``p_t`` close to 1) so
    the optimiser spends more gradient on the hard minority class, and the
    per-class ``α_t`` prior handles the static class-frequency imbalance.

    Parameters
    ----------
    gamma
        Focusing parameter. ``γ = 0`` collapses to plain weighted BCE (asserted
        in the smoke test). ``γ = 2`` is the Lin+ 2017 default and the value
        we've used in every ablation; higher γ focuses more aggressively on
        hard examples at the cost of early-training stability.
    alpha
        Class-imbalance prior. Four forms are accepted:

        * a **scalar float** ``α_pos`` — resolved to ``(α_pos, 1 − α_pos)``;
        * a **2-tuple** ``(α_pos, α_neg)`` — used verbatim (no constraint that
          the two sum to 1);
        * the literal string ``"balanced"`` — resolved lazily from the first
          forward's targets via :func:`balanced_alpha` (roughly
          ``(0.779, 0.221)`` on our split); cached thereafter;
        * ``None`` — no class weighting (discouraged on this dataset; kept for
          γ=0 / BCE-equivalence tests).
    reduction
        ``"mean"`` / ``"sum"`` / ``"none"``.

    Notes
    -----
    We operate on logits rather than probabilities to keep the loss numerically
    stable for very-confident predictions (|logit| > 30). The key identity is
    ``exp(−BCE(z, y)) == p_t``, which lets us recover the true-class
    probability without a second :func:`sigmoid` call — cheaper and avoids a
    redundant rounding step.

    Raises
    ------
    ValueError
        On negative ``gamma`` or malformed ``alpha`` tuples.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | Tuple[float, float] | Literal["balanced"] | None = None,
        reduction: Reduction = "mean",
    ):
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        self.gamma = float(gamma)
        self.alpha = alpha
        self.reduction: Reduction = reduction
        # Filled on first forward when alpha == "balanced" so we see the real
        # batch distribution before committing. Stays None otherwise.
        self._balanced_fitted: Optional[Tuple[float, float]] = None

    def _resolve_alpha(
        self,
        targets: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Normalise ``self.alpha`` into two device-placed scalar tensors.

        Returns ``None`` when alpha is disabled so the caller can skip the
        multiplication entirely (one fewer op on the hot path).
        """
        alpha = self.alpha
        if alpha is None:
            return None

        if alpha == "balanced":
            # Lazy fit from the first batch we see. This is safe because the
            # sampler guarantees both classes are present and the balanced
            # ratio is a dataset-level property (batches just estimate it).
            if self._balanced_fitted is None:
                self._balanced_fitted = balanced_alpha(targets)
                logger.info(
                    "FocalLoss: balanced alpha fitted from first batch -> (%.3f, %.3f)",
                    *self._balanced_fitted,
                )
            ap, an = self._balanced_fitted
        elif isinstance(alpha, (tuple, list)):
            if len(alpha) != 2:
                raise ValueError(f"alpha tuple must have length 2, got {len(alpha)}")
            ap, an = float(alpha[0]), float(alpha[1])
        else:
            # Scalar — interpret as α_pos, mirror for the negative class.
            ap = float(alpha)
            an = 1.0 - ap

        device = targets.device
        return (
            torch.tensor(ap, device=device, dtype=torch.float32),
            torch.tensor(an, device=device, dtype=torch.float32),
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        logits = logits.view(-1).float()
        targets = targets.view(-1).float()

        # Per-element BCE so we can re-weight it; reduction happens at the end.
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # exp(-BCE) == p_t by construction of binary_cross_entropy_with_logits,
        # which is numerically safer than sigmoid(logits) at |logit| > 30 and
        # exactly matches the autograd graph BCE already built.
        p_t = torch.exp(-bce)
        focal_factor = (1 - p_t) ** self.gamma

        alphas = self._resolve_alpha(targets)
        if alphas is not None:
            alpha_pos, alpha_neg = alphas
            # Element-wise pick: targets is 0/1, so this selects α_pos on
            # positives and α_neg on negatives without a torch.where call.
            alpha_t = alpha_pos * targets + alpha_neg * (1.0 - targets)
        else:
            alpha_t = 1.0

        loss = alpha_t * focal_factor * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothingBCELoss(nn.Module):
    """BCE with ε-smoothed labels (Szegedy+ 2016 adapted to binary).

    Replaces the one-hot target with a soft target::

        y_smooth = y · (1 − ε) + 0.5 · ε

    i.e. positives become ``1 − ε/2`` and negatives become ``ε/2``. At the
    project's default ``ε = 0.05`` (plan §7.3) positives are 0.975 and
    negatives 0.025. This bounds the achievable loss, which (a) prevents
    logits from running off to ±∞ late in training, and (b) empirically nudges
    calibration in the right direction for this dataset — ECE tends to drop
    by a few points vs. plain WBCE.

    Parameters
    ----------
    epsilon
        Smoothing strength in ``[0, 1]``. ``ε = 0`` collapses to plain
        (weighted) BCE — asserted in the smoke test.
    pos_weight
        Optional positive-class up-weight, same semantics as
        :class:`WeightedBCELoss`.
    reduction
        ``"mean"`` / ``"sum"`` / ``"none"``.

    Raises
    ------
    ValueError
        If ``epsilon`` falls outside ``[0, 1]``.
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        pos_weight: Optional[float | torch.Tensor] = None,
        reduction: Reduction = "mean",
    ):
        super().__init__()
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError(f"epsilon must lie in [0, 1], got {epsilon}")
        self.epsilon = float(epsilon)
        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor(float(pos_weight))
        # See WeightedBCELoss for the rationale on non-persistent buffers.
        self.register_buffer(
            "pos_weight",
            pos_weight if isinstance(pos_weight, torch.Tensor) else None,  # type: ignore[arg-type]
            persistent=False,
        )
        self.reduction: Reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        logits = logits.view(-1).float()
        targets = targets.view(-1).float()
        # Soft targets: positives → 1 − ε/2, negatives → ε/2.
        targets_smooth = targets * (1.0 - self.epsilon) + 0.5 * self.epsilon

        pos_weight = self.pos_weight if self.pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(
            logits, targets_smooth, pos_weight=pos_weight, reduction=self.reduction
        )


# Smoke test

if __name__ == "__main__":
    torch.manual_seed(0)

    y = (torch.rand(256) < 0.22).float()
    logits = 1.2 * (y - 0.5) + 0.5 * torch.randn(256)
    pos_weight = compute_pos_weight(y)

    print("-- WBCELoss --")
    wbce = WeightedBCELoss(pos_weight=pos_weight)
    loss_wbce = wbce(logits, y)
    assert loss_wbce.ndim == 0 and torch.isfinite(loss_wbce)
    print(f"  loss: {loss_wbce.item():.4f}  (pos_weight={pos_weight.item():.3f})")

    wbce_none = WeightedBCELoss(pos_weight=pos_weight, reduction="none")
    loss_elem = wbce_none(logits, y)
    assert loss_elem.shape == (256,)
    print(f"  reduction='none' shape: {tuple(loss_elem.shape)}")

    print("\n-- FocalLoss --")
    for gamma in (0.0, 1.0, 2.0, 3.0):
        fl = FocalLoss(gamma=gamma)
        fl_val = fl(logits, y)
        assert torch.isfinite(fl_val)
        print(f"  gamma={gamma}: loss={fl_val.item():.4f}")

    fl_g0 = FocalLoss(gamma=0.0, alpha=None)
    bce = F.binary_cross_entropy_with_logits(logits, y)
    assert torch.allclose(
        fl_g0(logits, y), bce, atol=1e-6
    ), f"gamma=0 focal should equal plain BCE; got {fl_g0(logits, y).item()} vs {bce.item()}"
    print("  gamma=0 matches plain BCE (delta < 1e-6)")

    fl_bal = FocalLoss(gamma=2.0, alpha="balanced")
    fl_bal(logits, y)
    fl_bal(logits, y)
    assert fl_bal._balanced_fitted is not None
    ap, an = fl_bal._balanced_fitted
    assert 0.7 < ap < 0.9, f"balanced alpha_pos ~ 0.78 expected for ~22% positives; got {ap}"
    print(f"  balanced alpha = ({ap:.3f}, {an:.3f})")

    fl_tup = FocalLoss(gamma=2.0, alpha=(0.75, 0.25))
    fl_tup(logits, y)
    print("  alpha=(0.75, 0.25) runs")

    print("\n-- LabelSmoothingBCELoss --")
    lsm = LabelSmoothingBCELoss(epsilon=0.05, pos_weight=pos_weight)
    loss_lsm = lsm(logits, y)
    assert torch.isfinite(loss_lsm)
    print(f"  epsilon=0.05: loss={loss_lsm.item():.4f}")

    lsm_eps0 = LabelSmoothingBCELoss(epsilon=0.0, pos_weight=pos_weight)
    assert torch.allclose(
        lsm_eps0(logits, y), wbce(logits, y), atol=1e-6
    ), "epsilon=0 should collapse to plain WBCE"
    print("  epsilon=0 matches WBCE")

    print("\n-- gradient flow --")
    for name, loss_fn in [
        ("WBCE", WeightedBCELoss(pos_weight=pos_weight)),
        ("Focal gamma=2", FocalLoss(gamma=2.0, alpha=0.75)),
        ("LabelSmoothing", LabelSmoothingBCELoss(epsilon=0.05)),
    ]:
        x = logits.detach().clone().requires_grad_(True)
        loss = loss_fn(x, y)
        loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        print(f"  {name:16s} -> grad norm {x.grad.norm().item():.4f}")

    print("\nAll loss smoke tests passed.")
