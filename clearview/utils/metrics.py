"""Evaluation metrics for image restoration.

Provides standard metrics like PSNR, SSIM, MAE, and MSE for evaluating
image deraining and restoration quality.
"""

from typing import Dict, List, Optional, cast, Union
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np


def compute_psnr(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    max_val: float = 1.0,
    reduction: str = "mean",
) -> Union[float, torch.Tensor, np.ndarray]:
    """Compute Peak Signal-to-Noise Ratio (PSNR).

    PSNR = 20 * log10(max_val / sqrt(MSE))

    Args:
        pred: Predicted image (B, C, H, W) or (H, W, C)
        target: Target image (same shape as pred)
        max_val: Maximum possible pixel value (1.0 for normalized images)
        reduction: 'mean' | 'none'. If 'none', returns per-image PSNR

    Returns:
        PSNR value(s) in dB

    Example:
        >>> pred = torch.randn(4, 3, 256, 256).clamp(0, 1)
        >>> target = torch.randn(4, 3, 256, 256).clamp(0, 1)
        >>> psnr = compute_psnr(pred, target)
        >>> print(f"PSNR: {psnr:.2f} dB")
    """
    if isinstance(pred, torch.Tensor):
        if not isinstance(target, np.ndarray):
            raise TypeError(f"Expected np.npdarray target, got {type(target)}")
        mse = F.mse_loss(pred, target, reduction="none")
        mse = mse.mean(dim=(1, 2, 3))  # Mean over channels and spatial dims

        # Avoid log(0)
        mse = torch.clamp(mse, min=1e-10)

        psnr = 20 * torch.log10(max_val / torch.sqrt(mse))

        if reduction == "mean":
            return psnr.mean().item()
        return psnr
    else:
        if not isinstance(target, np.ndarray):
            raise TypeError(f"Expected np.npdarray target, got {type(target)}")
        # NumPy implementation
        axis_tuple = (1, 2, 3) if pred.ndim == 4 else (0, 1, 2)
        mse_np = np.mean((pred - target) ** 2, axis=axis_tuple)
        mse_np = np.clip(mse_np, 1e-10, None)

        psnr_np: np.ndarray = 20 * np.log10(max_val / np.sqrt(mse_np))

        if reduction == "mean":
            return float(np.mean(psnr_np))
        return psnr_np


def compute_ssim(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    max_val: float = 1.0,
    window_size: int = 11,
    reduction: str = "mean",
) -> Union[float, torch.Tensor, np.ndarray]:
    """Compute Structural Similarity Index (SSIM).

    Measures structural similarity between images. Better correlation
    with human perception than MSE/PSNR.

    Args:
        pred: Predicted image (B, C, H, W) or (H, W, C)
        target: Target image (same shape as pred)
        max_val: Maximum possible pixel value
        window_size: Size of Gaussian window (must be odd)
        reduction: 'mean' | 'none'. If 'none', returns per-image SSIM

    Returns:
        SSIM value(s) in range [0, 1]

    Reference:
        Wang et al. "Image quality assessment: from error visibility to
        structural similarity." IEEE TIP 2004.

    Example:
        >>> pred = torch.randn(4, 3, 256, 256).clamp(0, 1)
        >>> target = torch.randn(4, 3, 256, 256).clamp(0, 1)
        >>> ssim = compute_ssim(pred, target)
        >>> print(f"SSIM: {ssim:.4f}")
    """
    if isinstance(pred, torch.Tensor):
        if not isinstance(target, np.ndarray):
            raise TypeError(f"Expected np.npdarray target, got {type(target)}")
        # Use PyTorch implementation
        from clearview.losses.structural import _ssim, _gaussian_kernel

        # Create Gaussian window
        window = _gaussian_kernel(window_size, 1.5)
        channel_count = pred.size(1)
        window = window.repeat(channel_count, 1, 1, 1).to(pred.device)

        ssim_val = _ssim(
            pred,
            target,
            window,
            window_size,
            channel_count,
            size_average=(reduction == "mean"),
        )

        if reduction == "mean":
            return ssim_val.item()
        return ssim_val
    else:
        if not isinstance(target, np.ndarray):
            raise TypeError(f"Expected np.npdarray target, got {type(target)}")
        # NumPy implementation (simplified)
        try:
            from skimage.metrics import structural_similarity

            if pred.ndim == 4:
                # Batch processing
                ssim_values = []
                for i in range(pred.shape[0]):
                    pred_hwc = pred[i].transpose(1, 2, 0)
                    target_hwc = target[i].transpose(1, 2, 0)
                    ssim_val = structural_similarity(
                        pred_hwc, target_hwc, data_range=max_val, channel_axis=2
                    )
                    ssim_values.append(ssim_val)

                if reduction == "mean":
                    return float(np.mean(ssim_values))
                return cast(np.ndarray, np.array(ssim_values))
            else:
                ssim_val = structural_similarity(
                    pred,
                    target,
                    data_range=max_val,
                    channel_axis=2 if pred.ndim == 3 else None,
                )
                return float(ssim_val)
        except ImportError:
            raise ImportError(
                "scikit-image is required for SSIM computation with NumPy arrays"
            )


def compute_mae(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    reduction: str = "mean",
) -> Union[float, torch.Tensor, np.ndarray]:
    """Compute Mean Absolute Error (MAE).

    Args:
        pred: Predicted image
        target: Target image
        reduction: 'mean' | 'none'

    Returns:
        MAE value(s)

    Example:
        >>> pred = torch.randn(4, 3, 256, 256)
        >>> target = torch.randn(4, 3, 256, 256)
        >>> mae = compute_mae(pred, target)
    """
    if isinstance(pred, torch.Tensor):
        if not isinstance(target, np.ndarray):
            raise TypeError(f"Expected np.npdarray target, got {type(target)}")
        mae = F.l1_loss(pred, target, reduction="none")
        mae = mae.mean(dim=(1, 2, 3))

        if reduction == "mean":
            return mae.mean().item()
        return mae
    else:
        if not isinstance(target, np.ndarray):
            raise TypeError(f"Expected np.npdarray target, got {type(target)}")
        axis_tuple = (1, 2, 3) if pred.ndim == 4 else (0, 1, 2)
        mae_np: np.ndarray = np.mean(np.abs(pred - target), axis=axis_tuple)

        if reduction == "mean":
            return float(np.mean(mae_np))
        return mae_np


def compute_mse(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    reduction: str = "mean",
) -> Union[float, torch.Tensor, np.ndarray]:
    """Compute Mean Squared Error (MSE).

    Args:
        pred: Predicted image
        target: Target image
        reduction: 'mean' | 'none'

    Returns:
        MSE value(s)

    Example:
        >>> pred = torch.randn(4, 3, 256, 256)
        >>> target = torch.randn(4, 3, 256, 256)
        >>> mse = compute_mse(pred, target)
    """
    if isinstance(pred, torch.Tensor):
        if not isinstance(target, np.ndarray):
            raise TypeError(f"Expected np.npdarray target, got {type(target)}")
        mse = F.mse_loss(pred, target, reduction="none")
        mse = mse.mean(dim=(1, 2, 3))

        if reduction == "mean":
            return mse.mean().item()
        return mse
    else:
        if not isinstance(target, np.ndarray):
            raise TypeError(f"Expected np.npdarray target, got {type(target)}")
        axis_tuple = (1, 2, 3) if pred.ndim == 4 else (0, 1, 2)
        mse_np: np.ndarray = np.mean((pred - target) ** 2, axis=axis_tuple)

        if reduction == "mean":
            return float(np.mean(mse_np))
        return mse_np


def compute_metrics(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    metrics: Optional[List[str]] = None,
    max_val: float = 1.0,
) -> Dict[str, float]:
    """Compute multiple metrics at once.

    Args:
        pred: Predicted image
        target: Target image
        metrics: List of metrics to compute. If None, computes all.
            Options: 'psnr', 'ssim', 'mae', 'mse'
        max_val: Maximum pixel value

    Returns:
        Dictionary mapping metric names to values

    Example:
        >>> pred = torch.randn(4, 3, 256, 256).clamp(0, 1)
        >>> target = torch.randn(4, 3, 256, 256).clamp(0, 1)
        >>> metrics = compute_metrics(pred, target)
        >>> print(metrics)
        {'psnr': 25.3, 'ssim': 0.85, 'mae': 0.12, 'mse': 0.015}
    """
    if metrics is None:
        metrics = ["psnr", "ssim", "mae", "mse"]

    results = {}

    for metric in metrics:
        metric_lower = metric.lower()

        if metric_lower == "psnr":
            results["psnr"] = float(compute_psnr(pred, target, max_val=max_val))
        elif metric_lower == "ssim":
            results["ssim"] = float(compute_ssim(pred, target, max_val=max_val))
        elif metric_lower == "mae":
            results["mae"] = float(compute_mae(pred, target))
        elif metric_lower == "mse":
            results["mse"] = float(compute_mse(pred, target))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return results


class MetricsTracker:
    """Track and aggregate metrics over multiple batches.

    Useful for tracking metrics during training/evaluation epochs.

    Example:
        >>> tracker = MetricsTracker()
        >>>
        >>> for batch in dataloader:
        ...     pred, target = model(batch), batch['target']
        ...     metrics = compute_metrics(pred, target)
        ...     tracker.update(metrics)
        >>>
        >>> avg_metrics = tracker.average()
        >>> print(f"Average PSNR: {avg_metrics['psnr']:.2f} dB")
        >>> tracker.reset()
    """

    def __init__(self) -> None:
        """Initialize metrics tracker."""
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.count = 0

    def update(self, metrics: Dict[str, float], batch_size: int = 1) -> None:
        """Update tracker with new metrics.

        Args:
            metrics: Dictionary of metric values
            batch_size: Number of samples in batch (for weighted averaging)
        """
        for name, value in metrics.items():
            self.metrics[name].append(value)
        self.count += batch_size

    def average(self) -> Dict[str, float]:
        """Compute average of all tracked metrics.

        Returns:
            Dictionary of averaged metrics
        """
        return {name: float(np.mean(values)) for name, values in self.metrics.items()}

    def std(self) -> Dict[str, float]:
        """Compute standard deviation of all tracked metrics.

        Returns:
            Dictionary of metric standard deviations
        """
        return {name: float(np.std(values)) for name, values in self.metrics.items()}

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive summary statistics.

        Returns:
            Dictionary with 'mean', 'std', 'min', 'max' for each metric
        """
        summary = {}
        for name, values in self.metrics.items():
            summary[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        return summary

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.metrics.clear()
        self.count = 0

    def __repr__(self) -> str:
        """String representation."""
        avg = self.average()
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in avg.items()])
        return f"MetricsTracker({metrics_str}, count={self.count})"


__all__ = [
    "compute_psnr",
    "compute_ssim",
    "compute_mae",
    "compute_mse",
    "compute_metrics",
    "MetricsTracker",
]
