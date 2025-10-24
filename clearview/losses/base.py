"""Base loss class for all loss functions.

Provides a common interface and utilities for implementing custom loss functions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn


class BaseLoss(nn.Module, ABC):
    """Abstract base class for all loss functions.

    All custom loss functions should inherit from this class and implement
    the forward method. This ensures a consistent interface across all losses.

    Args:
        reduction: Specifies the reduction to apply to the output.
            Options: 'none' | 'mean' | 'sum'. Default: 'mean'
        weight: Manual rescaling weight. Default: 1.0

    Example:
        >>> class MyCustomLoss(BaseLoss):
        ...     def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ...         return torch.mean((pred - target) ** 2)
        >>> loss_fn = MyCustomLoss()
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(
        self, reduction: str = "mean", weight: float = 1.0, **kwargs: Any
    ) -> None:
        """Initialize base loss.

        Args:
            reduction: Reduction method for the loss
            weight: Weighting factor for this loss component
            **kwargs: Additional arguments for subclasses
        """
        super().__init__()

        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Invalid reduction mode: {reduction}. "
                "Choose from 'none', 'mean', 'sum'."
            )

        self.reduction = reduction
        self.weight = weight

    @abstractmethod
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            pred: Predicted tensor of shape (B, C, H, W)
            target: Target tensor of shape (B, C, H, W)

        Returns:
            Computed loss value

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def apply_reduction(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply the specified reduction to the loss.

        Args:
            loss: Loss tensor before reduction

        Returns:
            Reduced loss tensor
        """
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

    def apply_weight(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply the weight to the loss.

        Args:
            loss: Loss tensor

        Returns:
            Weighted loss tensor
        """
        return self.weight * loss

    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for serialization.

        Returns:
            Dictionary containing loss configuration
        """
        return {
            "type": self.__class__.__name__,
            "reduction": self.reduction,
            "weight": self.weight,
        }

    def __repr__(self) -> str:
        """String representation of the loss."""
        return (
            f"{self.__class__.__name__}("
            f"reduction={self.reduction}, "
            f"weight={self.weight})"
        )
