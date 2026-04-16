"""
losses.py — Loss functions for class-imbalanced binary classification.

Contains:
    1. WeightedBCELoss  — baseline, inverse-frequency class weighting.
    2. FocalLoss        — Lin et al. (2017) with α-balancing.
    3. LabelSmoothingBCELoss — Müller et al. (2019) adapted to binary targets.

All losses operate on **logits** (pre-sigmoid) for numerical stability. They
do *not* call ``torch.sigmoid`` internally except via the log-sum-exp
``binary_cross_entropy_with_logits`` primitive.

References
----------
* Lin, T.Y. et al. (2017). "Focal Loss for Dense Object Detection." ICCV.
* Mukhoti, J. et al. (2020). "Calibrating Deep Neural Networks Using Focal Loss." NeurIPS.
* Müller, R. et al. (2019). "When Does Label Smoothing Help?" NeurIPS.

Project plan alignment: §7.1 (WBCE), §7.2 (Focal), §7.3 (Label Smoothing).
"""

from __future__ import annotations

import logging
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

Reduction = Literal["mean", "sum", "none"]


# ──────────────────────────────────────────────────────────────────────────────
# Class-weight utilities
# ──────────────────────────────────────────────────────────────────────────────


def compute_pos_weight(y: torch.Tensor) -> torch.Tensor:
    """
    Inverse-frequency positive-class weight, suitable for
    ``F.binary_cross_entropy_with_logits(..., pos_weight=...)``.

    Returns ``N_neg / N_pos`` as a scalar tensor.
    On this dataset (22.1% positive) this yields ≈ 3.52.
    """
    y = y.float()
    pos = y.sum()
    neg = y.numel() - pos
    if pos <= 0:
        raise ValueError("pos_weight is undefined when the training set has no positives")
    return neg / pos


def balanced_alpha(y: torch.Tensor) -> Tuple[float, float]:
    """
    Class-balanced α weights for Focal loss, as used in Lin et al. (2017).

    Returns ``(alpha_pos, alpha_neg)`` such that each class contributes equally
    to the loss in expectation. For a dataset with 22.1% positives this gives
    (α₊, α₋) ≈ (0.779, 0.221).
    """
    y = y.float()
    n = float(y.numel())
    pos = float(y.sum().item())
    neg = n - pos
    if pos <= 0 or neg <= 0:
        raise ValueError("balanced_alpha requires both classes to be present")
    # inverse class prior → flipped to put heavier weight on the minority
    alpha_pos = neg / n
    alpha_neg = pos / n
    return alpha_pos, alpha_neg


# ──────────────────────────────────────────────────────────────────────────────
# 1. Weighted Binary Cross-Entropy
# ──────────────────────────────────────────────────────────────────────────────


class WeightedBCELoss(nn.Module):
    """
    Binary cross-entropy with an optional positive-class weight.

    Equivalent to ``torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)`` but
    exposes the same (logits, targets) contract as the other losses in this
    module and accepts the class weights at call time or at construction.

    Parameters
    ----------
    pos_weight : float | torch.Tensor | None
        If provided, multiplies the positive-class BCE term. Pass
        ``compute_pos_weight(y_train)`` for the standard WBCE baseline.
    reduction : {"mean", "sum", "none"}
    """

    def __init__(
        self,
        pos_weight: Optional[float | torch.Tensor] = None,
        reduction: Reduction = "mean",
    ):
        super().__init__()
        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor(float(pos_weight))
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
        pos_weight = self.pos_weight if self.pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction=self.reduction
        )


# ──────────────────────────────────────────────────────────────────────────────
# 2. Focal Loss
# ──────────────────────────────────────────────────────────────────────────────


class FocalLoss(nn.Module):
    """
    Binary focal loss (Lin et al., 2017):

        FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    where ``p_t = p`` if ``y = 1`` else ``1 - p``, ``p = sigmoid(logit)``, and
    ``α_t = α_pos`` if ``y = 1`` else ``α_neg``.

    The ``(1 - p_t)^γ`` modulating factor down-weights correctly-classified
    easy examples so the model focuses gradient on hard / misclassified ones.
    At ``γ = 0`` this collapses to plain weighted BCE. Plan §7.2 recommends
    ablating over γ ∈ {0, 0.5, 1, 2, 3}.

    Parameters
    ----------
    gamma : float
        Focusing parameter ≥ 0. 0 recovers class-weighted BCE.
    alpha : float or tuple[float, float] or "balanced" or None
        * float — shared positive-class α, with α_neg := 1 - α_pos (Lin's form)
        * tuple — explicit (α_pos, α_neg)
        * "balanced" — α_pos = N_neg/N, α_neg = N_pos/N, computed from the *first
          batch seen*. Rarely what you want at long-run: prefer to compute
          ``balanced_alpha(y_train)`` once and pass the tuple explicitly.
        * None — no α weighting.
    reduction : {"mean", "sum", "none"}

    Numerically stable: uses ``F.binary_cross_entropy_with_logits`` under the
    hood so we never call ``log(sigmoid(x))`` directly.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: (
            float | Tuple[float, float] | Literal["balanced"] | None
        ) = None,
        reduction: Reduction = "mean",
    ):
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        self.gamma = float(gamma)
        self.alpha = alpha
        self.reduction: Reduction = reduction
        self._balanced_fitted: Optional[Tuple[float, float]] = None

    def _resolve_alpha(
        self,
        targets: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return (alpha_pos, alpha_neg) as 0-d tensors on the right device, or None."""
        alpha = self.alpha
        if alpha is None:
            return None

        if alpha == "balanced":
            if self._balanced_fitted is None:
                self._balanced_fitted = balanced_alpha(targets)
                logger.info(
                    "FocalLoss: balanced alpha fitted from first batch → (%.3f, %.3f)",
                    *self._balanced_fitted,
                )
            ap, an = self._balanced_fitted
        elif isinstance(alpha, (tuple, list)):
            if len(alpha) != 2:
                raise ValueError(f"alpha tuple must have length 2, got {len(alpha)}")
            ap, an = float(alpha[0]), float(alpha[1])
        else:
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

        # BCE per-example (with no reduction) — numerically stable.
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # p_t and the (1 - p_t)^gamma focusing factor.
        #   p_t = sigmoid(logit) · y + (1 - sigmoid(logit)) · (1 - y)
        # Computed via exp(-BCE) which avoids a separate sigmoid call and is
        # exactly equivalent by construction.
        p_t = torch.exp(-bce)
        focal_factor = (1 - p_t) ** self.gamma

        # α_t weighting, optional.
        alphas = self._resolve_alpha(targets)
        if alphas is not None:
            alpha_pos, alpha_neg = alphas
            alpha_t = alpha_pos * targets + alpha_neg * (1.0 - targets)
        else:
            alpha_t = 1.0

        loss = alpha_t * focal_factor * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss  # none


# ──────────────────────────────────────────────────────────────────────────────
# 3. Label-smoothing BCE
# ──────────────────────────────────────────────────────────────────────────────


class LabelSmoothingBCELoss(nn.Module):
    """
    BCE with label smoothing (Müller et al., 2019) adapted to binary targets:

        y_smooth = y · (1 - ε) + 0.5 · ε

    Plan §7.3 recommends ε = 0.05. Label smoothing acts as a regularisation
    technique — the model cannot drive its logits to ±∞ without suffering a
    smoothing penalty, so it generalises marginally better and calibrates
    somewhat better.

    Parameters
    ----------
    epsilon : float in [0, 1]
    pos_weight : Optional[float | Tensor]  -- combines with BCE in the same way as WBCE
    reduction : {"mean", "sum", "none"}
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
        targets_smooth = targets * (1.0 - self.epsilon) + 0.5 * self.epsilon

        pos_weight = self.pos_weight if self.pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(
            logits, targets_smooth, pos_weight=pos_weight, reduction=self.reduction
        )


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)

    # Simulate a credit-risk-style minibatch: 256 samples, 22% positive rate.
    y = (torch.rand(256) < 0.22).float()
    # A "weak but informative" model: logits correlate with labels.
    logits = 1.2 * (y - 0.5) + 0.5 * torch.randn(256)
    pos_weight = compute_pos_weight(y)

    print("── WBCELoss ──")
    wbce = WeightedBCELoss(pos_weight=pos_weight)
    loss_wbce = wbce(logits, y)
    assert loss_wbce.ndim == 0 and torch.isfinite(loss_wbce)
    print(f"  loss: {loss_wbce.item():.4f}  (pos_weight={pos_weight.item():.3f})")

    # "none" reduction shape check
    wbce_none = WeightedBCELoss(pos_weight=pos_weight, reduction="none")
    loss_elem = wbce_none(logits, y)
    assert loss_elem.shape == (256,)
    print(f"  reduction='none' shape: {tuple(loss_elem.shape)} ✓")

    print("\n── FocalLoss ──")
    for gamma in (0.0, 1.0, 2.0, 3.0):
        fl = FocalLoss(gamma=gamma)
        fl_val = fl(logits, y)
        assert torch.isfinite(fl_val)
        print(f"  γ={gamma}: loss={fl_val.item():.4f}")

    # At γ=0 with no α, focal should equal plain BCE (to machine precision).
    fl_g0 = FocalLoss(gamma=0.0, alpha=None)
    bce = F.binary_cross_entropy_with_logits(logits, y)
    assert torch.allclose(fl_g0(logits, y), bce, atol=1e-6), \
        f"γ=0 focal should equal plain BCE; got {fl_g0(logits, y).item()} vs {bce.item()}"
    print(f"  γ=0 ↔ plain BCE (verified, Δ < 1e-6) ✓")

    # α="balanced" should fit once from the data.
    fl_bal = FocalLoss(gamma=2.0, alpha="balanced")
    fl_bal(logits, y)  # fit
    fl_bal(logits, y)  # should reuse
    assert fl_bal._balanced_fitted is not None
    ap, an = fl_bal._balanced_fitted
    assert 0.7 < ap < 0.9, f"balanced α_pos ≈ 0.78 expected for ~22% positives; got {ap}"
    print(f"  balanced α = ({ap:.3f}, {an:.3f}) ✓")

    # Explicit tuple α.
    fl_tup = FocalLoss(gamma=2.0, alpha=(0.75, 0.25))
    fl_tup(logits, y)  # must not raise
    print(f"  α=(0.75, 0.25) runs ✓")

    print("\n── LabelSmoothingBCELoss ──")
    lsm = LabelSmoothingBCELoss(epsilon=0.05, pos_weight=pos_weight)
    loss_lsm = lsm(logits, y)
    assert torch.isfinite(loss_lsm)
    print(f"  ε=0.05: loss={loss_lsm.item():.4f}")

    # ε=0 should equal plain WBCE.
    lsm_eps0 = LabelSmoothingBCELoss(epsilon=0.0, pos_weight=pos_weight)
    assert torch.allclose(lsm_eps0(logits, y), wbce(logits, y), atol=1e-6), \
        "ε=0 should collapse to plain WBCE"
    print(f"  ε=0 ↔ WBCE (verified) ✓")

    # Gradient flow check on each loss.
    print("\n── gradient flow ──")
    for name, loss_fn in [
        ("WBCE",          WeightedBCELoss(pos_weight=pos_weight)),
        ("Focal γ=2",     FocalLoss(gamma=2.0, alpha=0.75)),
        ("LabelSmoothing", LabelSmoothingBCELoss(epsilon=0.05)),
    ]:
        x = logits.detach().clone().requires_grad_(True)
        loss = loss_fn(x, y)
        loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        print(f"  {name:16s} → grad norm {x.grad.norm().item():.4f} ✓")

    print("\nAll loss smoke tests passed.")
