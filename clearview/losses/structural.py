"""Structural similarity loss functions.

Implements SSIM-based losses for preserving structural information in images.
"""

from typing import Any, List, Optional

import torch
import torch.nn.functional as F

from clearview.losses.base import BaseLoss


def _gaussian_kernel(kernel_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Create a 2D Gaussian kernel.

    Args:
        kernel_size: Size of the Gaussian kernel
        sigma: Standard deviation of the Gaussian

    Returns:
        2D Gaussian kernel tensor
    """
    coords = torch.arange(kernel_size, dtype=torch.float32)
    coords -= (kernel_size - 1) / 2.0

    g = coords**2
    g = (-(g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma**2)).exp()
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def _ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channel: int,
    size_average: bool = True,
) -> torch.Tensor:
    """Compute SSIM between two images.

    Args:
        img1: First image (B, C, H, W)
        img2: Second image (B, C, H, W)
        window: Gaussian window for filtering
        window_size: Size of the window
        channel: Number of channels
        size_average: Whether to average the result

    Returns:
        SSIM value
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(BaseLoss):
    """Structural Similarity Index (SSIM) loss.

    Measures the structural similarity between two images. Better correlates
    with human perception than pixel-wise losses.

    Args:
        window_size: Size of the Gaussian window. Default: 11
        sigma: Standard deviation of Gaussian. Default: 1.5
        channel: Number of image channels. Default: 3
        reduction: Reduction method. Default: 'mean'
        weight: Loss weight. Default: 1.0

    Reference:
        Wang et al. "Image quality assessment: from error visibility to
        structural similarity." IEEE TIP 2004.

    Example:
        >>> loss_fn = SSIMLoss(window_size=11)
        >>> pred = torch.randn(4, 3, 256, 256)
        >>> target = torch.randn(4, 3, 256, 256)
        >>> loss = loss_fn(pred, target)
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        channel: int = 3,
        reduction: str = "mean",
        weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize SSIM loss.

        Args:
            window_size: Gaussian window size
            sigma: Gaussian standard deviation
            channel: Number of channels
            reduction: Reduction method
            weight: Loss weight
            **kwargs: Additional arguments
        """
        super().__init__(reduction=reduction, weight=weight, **kwargs)
        self.window_size = window_size
        self.sigma = sigma
        self.channel = channel

        # Create Gaussian window
        window = _gaussian_kernel(window_size, sigma)
        self.register_buffer("window", window.repeat(channel, 1, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            SSIM loss (1 - SSIM for minimization)
        """
        window_tensor: torch.Tensor = self.window
        ssim_value = _ssim(
            pred,
            target,
            window_tensor,
            self.window_size,
            self.channel,
            size_average=self.reduction == "mean",
        )

        # Return 1 - SSIM so we minimize the loss
        loss = 1.0 - ssim_value
        return self.apply_weight(loss)

    def get_config(self) -> dict:
        """Get configuration dictionary."""
        config = super().get_config()
        config.update(
            {
                "window_size": self.window_size,
                "sigma": self.sigma,
                "channel": self.channel,
            }
        )
        return config


class MultiScaleSSIMLoss(BaseLoss):
    """Multi-Scale Structural Similarity Index (MS-SSIM) loss.

    Computes SSIM at multiple scales to capture both fine and coarse
    structural similarities.

    Args:
        window_size: Size of the Gaussian window. Default: 11
        sigma: Standard deviation of Gaussian. Default: 1.5
        channel: Number of image channels. Default: 3
        scales: Number of scales to compute SSIM. Default: 5
        weights: Weights for each scale. If None, uses default weights.
        reduction: Reduction method. Default: 'mean'
        weight: Loss weight. Default: 1.0

    Reference:
        Wang et al. "Multi-scale structural similarity for image quality
        assessment." Asilomar Conference 2003.

    Example:
        >>> loss_fn = MultiScaleSSIMLoss(scales=5)
        >>> pred = torch.randn(4, 3, 256, 256)
        >>> target = torch.randn(4, 3, 256, 256)
        >>> loss = loss_fn(pred, target)
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        channel: int = 3,
        scales: int = 5,
        weights: Optional[List[float]] = None,
        reduction: str = "mean",
        weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize MS-SSIM loss.

        Args:
            window_size: Gaussian window size
            sigma: Gaussian standard deviation
            channel: Number of channels
            scales: Number of scales
            weights: Scale weights (if None, uses default)
            reduction: Reduction method
            weight: Loss weight
            **kwargs: Additional arguments
        """
        super().__init__(reduction=reduction, weight=weight, **kwargs)
        self.window_size = window_size
        self.sigma = sigma
        self.channel = channel
        self.scales = scales

        # Default weights from paper
        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333][:scales]
        self.register_buffer("weights", torch.FloatTensor(weights))

        # Create Gaussian window
        window = _gaussian_kernel(window_size, sigma)
        self.register_buffer("window", window.repeat(channel, 1, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute MS-SSIM loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            MS-SSIM loss (1 - MS-SSIM for minimization)
        """
        ms_ssim_list: List[torch.Tensor] = []

        window_tensor: torch.Tensor = self.window
        weights_tensor: torch.Tensor = self.weights

        for i in range(self.scales):
            ssim_value = _ssim(
                pred,
                target,
                window_tensor,
                self.window_size,
                self.channel,
                size_average=False,
            )
            ms_ssim_list.append(ssim_value)

            # Downsample for next scale
            if i < self.scales - 1:
                pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
                target = F.avg_pool2d(target, kernel_size=2, stride=2)

        # Weighted combination
        ms_ssim_tensor = torch.stack(ms_ssim_list, dim=0)
        ms_ssim = (ms_ssim_tensor ** weights_tensor.unsqueeze(1)).prod(dim=0)

        if self.reduction == "mean":
            ms_ssim = ms_ssim.mean()

        # Return 1 - MS-SSIM for minimization
        loss = 1.0 - ms_ssim
        return self.apply_weight(loss)

    def get_config(self) -> dict:
        """Get configuration dictionary."""
        config = super().get_config()
        weights_tensor: torch.Tensor = self.weights
        config.update(
            {
                "window_size": self.window_size,
                "sigma": self.sigma,
                "channel": self.channel,
                "scales": self.scales,
                "weights": weights_tensor.tolist(),
            }
        )
        return config
