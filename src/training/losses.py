"""Three losses for the imbalanced binary task: weighted BCE, focal, label-smoothed BCE.
All take logits and go through bce_with_logits so we don't blow up at the extremes."""

from __future__ import annotations

import logging
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

Reduction = Literal["mean", "sum", "none"]


def compute_pos_weight(y: torch.Tensor) -> torch.Tensor:
    """N_neg / N_pos as a scalar tensor. ~3.52 on our 22.1%-positive split."""
    y = y.float()
    pos = y.sum()
    neg = y.numel() - pos
    if pos <= 0:
        raise ValueError("pos_weight is undefined when the training set has no positives")
    return neg / pos


def balanced_alpha(y: torch.Tensor) -> Tuple[float, float]:
    """(α_pos, α_neg) so each class contributes equally. ~(0.779, 0.221) here."""
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
    """BCEWithLogits with optional pos_weight. Same (logits, targets) signature as the others."""

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


class FocalLoss(nn.Module):
    """-α·(1-p_t)^γ·log(p_t). γ downweights easy examples, α handles imbalance.
    γ=0 collapses to weighted BCE. α can be float, (α_pos, α_neg), "balanced", or None."""

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
        alpha = self.alpha
        if alpha is None:
            return None

        if alpha == "balanced":
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

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # exp(-bce) == p_t exactly, skips the sigmoid
        p_t = torch.exp(-bce)
        focal_factor = (1 - p_t) ** self.gamma

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
        return loss


class LabelSmoothingBCELoss(nn.Module):
    """BCE with y → y(1-ε) + 0.5ε. ε=0.05 per plan §7.3. Keeps logits from running
    to infinity and usually nudges calibration in the right direction."""

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
    assert torch.allclose(fl_g0(logits, y), bce, atol=1e-6), \
        f"gamma=0 focal should equal plain BCE; got {fl_g0(logits, y).item()} vs {bce.item()}"
    print(f"  gamma=0 matches plain BCE (delta < 1e-6)")

    fl_bal = FocalLoss(gamma=2.0, alpha="balanced")
    fl_bal(logits, y)
    fl_bal(logits, y)
    assert fl_bal._balanced_fitted is not None
    ap, an = fl_bal._balanced_fitted
    assert 0.7 < ap < 0.9, f"balanced alpha_pos ~ 0.78 expected for ~22% positives; got {ap}"
    print(f"  balanced alpha = ({ap:.3f}, {an:.3f})")

    fl_tup = FocalLoss(gamma=2.0, alpha=(0.75, 0.25))
    fl_tup(logits, y)
    print(f"  alpha=(0.75, 0.25) runs")

    print("\n-- LabelSmoothingBCELoss --")
    lsm = LabelSmoothingBCELoss(epsilon=0.05, pos_weight=pos_weight)
    loss_lsm = lsm(logits, y)
    assert torch.isfinite(loss_lsm)
    print(f"  epsilon=0.05: loss={loss_lsm.item():.4f}")

    lsm_eps0 = LabelSmoothingBCELoss(epsilon=0.0, pos_weight=pos_weight)
    assert torch.allclose(lsm_eps0(logits, y), wbce(logits, y), atol=1e-6), \
        "epsilon=0 should collapse to plain WBCE"
    print(f"  epsilon=0 matches WBCE")

    print("\n-- gradient flow --")
    for name, loss_fn in [
        ("WBCE",          WeightedBCELoss(pos_weight=pos_weight)),
        ("Focal gamma=2", FocalLoss(gamma=2.0, alpha=0.75)),
        ("LabelSmoothing", LabelSmoothingBCELoss(epsilon=0.05)),
    ]:
        x = logits.detach().clone().requires_grad_(True)
        loss = loss_fn(x, y)
        loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        print(f"  {name:16s} -> grad norm {x.grad.norm().item():.4f}")

    print("\nAll loss smoke tests passed.")
