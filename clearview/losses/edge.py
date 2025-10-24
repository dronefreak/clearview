"""Edge-aware loss functions.

Implements losses that preserve edge information during image restoration.
"""

from typing import Any, Tuple

import torch
import torch.nn.functional as F

from clearview.losses.base import BaseLoss


class SobelEdgeLoss(BaseLoss):
    """Sobel edge-aware loss.

    Computes the difference in edge maps between predicted and target images
    using Sobel filters. Helps preserve sharp edges during restoration.

    Args:
        reduction: Reduction method. Default: 'mean'
        weight: Loss weight. Default: 1.0
        convert_to_grayscale: Whether to convert RGB to grayscale. Default: True

    Example:
        >>> loss_fn = SobelEdgeLoss()
        >>> pred = torch.randn(4, 3, 256, 256)
        >>> target = torch.randn(4, 3, 256, 256)
        >>> loss = loss_fn(pred, target)
    """

    def __init__(
        self,
        reduction: str = "mean",
        weight: float = 1.0,
        convert_to_grayscale: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize Sobel edge loss.

        Args:
            reduction: Reduction method
            weight: Loss weight
            convert_to_grayscale: Convert to grayscale before edge detection
            **kwargs: Additional arguments
        """
        super().__init__(reduction=reduction, weight=weight, **kwargs)
        self.convert_to_grayscale = convert_to_grayscale

        # Sobel filters for edge detection
        sobel_x = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        sobel_y = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _rgb_to_grayscale(self, img: torch.Tensor) -> torch.Tensor:
        """Convert RGB image to grayscale.

        Uses standard RGB to grayscale conversion weights:
        Y = 0.299*R + 0.587*G + 0.114*B

        Args:
            img: RGB image (B, 3, H, W)

        Returns:
            Grayscale image (B, 1, H, W)
        """
        if img.size(1) == 1:
            return img

        # RGB to grayscale weights
        weights = torch.tensor([0.299, 0.587, 0.114], device=img.device)
        weights = weights.view(1, 3, 1, 1)

        gray = (img * weights).sum(dim=1, keepdim=True)
        return gray

    def _compute_edges(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute edge maps using Sobel filters.

        Args:
            img: Input image (B, C, H, W)

        Returns:
            Tuple of (dx, dy) edge maps
        """
        if self.convert_to_grayscale:
            img = self._rgb_to_grayscale(img)

        # Apply Sobel filters
        sobel_x_tensor: torch.Tensor = self.sobel_x
        sobel_y_tensor: torch.Tensor = self.sobel_y
        dx = F.conv2d(img, sobel_x_tensor, padding=1)
        dy = F.conv2d(img, sobel_y_tensor, padding=1)

        return dx, dy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute edge loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            Edge loss value
        """
        # Compute edge maps
        pred_dx, pred_dy = self._compute_edges(pred)
        target_dx, target_dy = self._compute_edges(target)

        # Compute L1 loss on edge maps
        loss_dx = F.l1_loss(pred_dx, target_dx, reduction=self.reduction)
        loss_dy = F.l1_loss(pred_dy, target_dy, reduction=self.reduction)

        loss = loss_dx + loss_dy
        return self.apply_weight(loss)

    def get_config(self) -> dict:
        """Get configuration dictionary."""
        config = super().get_config()
        config["convert_to_grayscale"] = self.convert_to_grayscale
        return config


class EdgeLoss(SobelEdgeLoss):
    """Alias for SobelEdgeLoss for backward compatibility."""

    pass


class LaplacianEdgeLoss(BaseLoss):
    """Laplacian edge-aware loss.

    Uses Laplacian filter to detect edges. Alternative to Sobel filter
    that is rotation-invariant.

    Args:
        reduction: Reduction method. Default: 'mean'
        weight: Loss weight. Default: 1.0
        convert_to_grayscale: Whether to convert RGB to grayscale. Default: True

    Example:
        >>> loss_fn = LaplacianEdgeLoss()
        >>> pred = torch.randn(4, 3, 256, 256)
        >>> target = torch.randn(4, 3, 256, 256)
        >>> loss = loss_fn(pred, target)
    """

    def __init__(
        self,
        reduction: str = "mean",
        weight: float = 1.0,
        convert_to_grayscale: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize Laplacian edge loss.

        Args:
            reduction: Reduction method
            weight: Loss weight
            convert_to_grayscale: Convert to grayscale before edge detection
            **kwargs: Additional arguments
        """
        super().__init__(reduction=reduction, weight=weight, **kwargs)
        self.convert_to_grayscale = convert_to_grayscale

        # Laplacian filter
        laplacian = (
            torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        self.register_buffer("laplacian", laplacian)

    def _rgb_to_grayscale(self, img: torch.Tensor) -> torch.Tensor:
        """Convert RGB image to grayscale."""
        if img.size(1) == 1:
            return img

        weights = torch.tensor([0.299, 0.587, 0.114], device=img.device)
        weights = weights.view(1, 3, 1, 1)
        gray = (img * weights).sum(dim=1, keepdim=True)
        return gray

    def _compute_edges(self, img: torch.Tensor) -> torch.Tensor:
        """Compute edge map using Laplacian filter.

        Args:
            img: Input image (B, C, H, W)

        Returns:
            Edge map tensor
        """
        if self.convert_to_grayscale:
            img = self._rgb_to_grayscale(img)

        laplacian_tensor: torch.Tensor = self.laplacian
        edges = F.conv2d(img, laplacian_tensor, padding=1)
        return edges

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian edge loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            Edge loss value
        """
        pred_edges = self._compute_edges(pred)
        target_edges = self._compute_edges(target)

        loss = F.l1_loss(pred_edges, target_edges, reduction=self.reduction)
        return self.apply_weight(loss)

    def get_config(self) -> dict:
        """Get configuration dictionary."""
        config = super().get_config()
        config["convert_to_grayscale"] = self.convert_to_grayscale
        return config
