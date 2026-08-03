"""Pixel-wise loss functions.

Provides standard pixel-wise reconstruction losses for image restoration tasks.
"""

from typing import Any

import torch
import torch.nn.functional as F

from clearview.losses.base import BaseLoss


class L1Loss(BaseLoss):
    """L1 (Mean Absolute Error) loss.

    Computes the mean absolute error between predicted and target images.
    Less sensitive to outliers compared to L2 loss.

    Args:
        reduction: Reduction method. Default: 'mean'
        weight: Loss weight. Default: 1.0

    Example:
        >>> loss_fn = L1Loss()
        >>> pred = torch.randn(4, 3, 256, 256)
        >>> target = torch.randn(4, 3, 256, 256)
        >>> loss = loss_fn(pred, target)
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute L1 loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            L1 loss value
        """
        loss = F.l1_loss(pred, target, reduction=self.reduction)
        return self.apply_weight(loss)


class MAELoss(L1Loss):
    """Mean Absolute Error loss (alias for L1Loss).

    This is functionally identical to L1Loss, provided for convenience
    and naming consistency.
    """

    pass


class L2Loss(BaseLoss):
    """L2 (Mean Squared Error) loss.

    Computes the mean squared error between predicted and target images.
    More sensitive to large errors compared to L1 loss.

    Args:
        reduction: Reduction method. Default: 'mean'
        weight: Loss weight. Default: 1.0

    Example:
        >>> loss_fn = L2Loss()
        >>> pred = torch.randn(4, 3, 256, 256)
        >>> target = torch.randn(4, 3, 256, 256)
        >>> loss = loss_fn(pred, target)
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute L2 loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            L2 loss value
        """
        loss = F.mse_loss(pred, target, reduction=self.reduction)
        return self.apply_weight(loss)


class MSELoss(L2Loss):
    """Mean Squared Error loss (alias for L2Loss).

    This is functionally identical to L2Loss, provided for convenience
    and naming consistency.
    """

    pass


class CharbonnierLoss(BaseLoss):
    """Charbonnier loss (smooth L1 loss).

    A differentiable variant of L1 loss that is smooth around zero.
    Often used in image restoration as a robust alternative to L2 loss.

    Loss = sqrt((pred - target)^2 + epsilon^2)

    Args:
        epsilon: Small constant for numerical stability. Default: 1e-6
        reduction: Reduction method. Default: 'mean'
        weight: Loss weight. Default: 1.0

    Reference:
        Charbonnier et al. "Two deterministic half-quadratic regularization
        algorithms for computed imaging." ICIP 1994.

    Example:
        >>> loss_fn = CharbonnierLoss(epsilon=1e-3)
        >>> pred = torch.randn(4, 3, 256, 256)
        >>> target = torch.randn(4, 3, 256, 256)
        >>> loss = loss_fn(pred, target)
    """

    def __init__(
        self,
        epsilon: float = 1e-6,
        reduction: str = "mean",
        weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize Charbonnier loss.

        Args:
            epsilon: Smoothing parameter
            reduction: Reduction method
            weight: Loss weight
            **kwargs: Additional arguments
        """
        super().__init__(reduction=reduction, weight=weight, **kwargs)
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Charbonnier loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            Charbonnier loss value
        """
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        loss = self.apply_reduction(loss)
        return self.apply_weight(loss)

    def get_config(self) -> dict:
        """Get configuration dictionary."""
        config = super().get_config()
        config["epsilon"] = self.epsilon
        return config


class ColorConsistencyLoss(BaseLoss):
    """Color consistency loss.

    Penalizes discrepancies in each image's global per-channel color
    statistics (mean, and optionally standard deviation) between the
    prediction and target. Pixel/structural losses are computed locally
    and can let small, spatially-distributed color/exposure errors cancel
    out in aggregate; this loss explicitly discourages global hue,
    saturation, or brightness drift that restoration networks can
    otherwise introduce (e.g. a slight overall color cast left behind
    after removing rain streaks).

    Args:
        use_std: Whether to also penalize per-channel standard deviation
            mismatch (contrast/saturation), in addition to the mean
            (brightness/color cast). Default: True
        reduction: Reduction method. Default: 'mean'
        weight: Loss weight. Default: 1.0

    Example:
        >>> loss_fn = ColorConsistencyLoss()
        >>> pred = torch.rand(4, 3, 256, 256)
        >>> target = torch.rand(4, 3, 256, 256)
        >>> loss = loss_fn(pred, target)
    """

    def __init__(
        self,
        use_std: bool = True,
        reduction: str = "mean",
        weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize color consistency loss.

        Args:
            use_std: Whether to also penalize per-channel std mismatch
            reduction: Reduction method
            weight: Loss weight
            **kwargs: Additional arguments
        """
        super().__init__(reduction=reduction, weight=weight, **kwargs)
        self.use_std = use_std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute color consistency loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            Color consistency loss value
        """
        pred_mean = pred.mean(dim=(-2, -1))
        target_mean = target.mean(dim=(-2, -1))
        loss = F.l1_loss(pred_mean, target_mean, reduction=self.reduction)

        if self.use_std:
            pred_std = pred.std(dim=(-2, -1))
            target_std = target.std(dim=(-2, -1))
            loss = loss + F.l1_loss(pred_std, target_std, reduction=self.reduction)

        return self.apply_weight(loss)

    def get_config(self) -> dict:
        """Get configuration dictionary."""
        config = super().get_config()
        config["use_std"] = self.use_std
        return config
