"""Utility functions for training, evaluation, and inference.

This module provides various utilities including metrics computation,
checkpoint management, visualization tools, and helper functions.
"""

from clearview.utils.checkpoint import (
    load_checkpoint,
    load_model,
    save_checkpoint,
    save_model,
)
from clearview.utils.image import (
    denormalize_image,
    normalize_image,
    numpy_to_tensor,
    rgb_to_grayscale,
    tensor_to_numpy,
)
from clearview.utils.logger import (
    get_logger,
    setup_logging,
)
from clearview.utils.metrics import (
    ALL_METRICS,
    MetricsTracker,
    compute_brisque,
    compute_dists,
    compute_fid,
    compute_lpips,
    compute_mae,
    compute_metrics,
    compute_mse,
    compute_psnr,
    compute_rain_removal_rate,
    compute_ssim,
)
from clearview.utils.niqe import NIQEModel, compute_niqe, fit_niqe_model
from clearview.utils.visualization import (
    create_comparison_grid,
    plot_metric_histogram,
    plot_training_curves,
    save_comparison,
    visualize_results,
)

__all__ = [
    # Metrics
    "compute_psnr",
    "compute_ssim",
    "compute_mae",
    "compute_mse",
    "compute_lpips",
    "compute_dists",
    "compute_brisque",
    "compute_rain_removal_rate",
    "compute_fid",
    "compute_metrics",
    "ALL_METRICS",
    "MetricsTracker",
    # No-reference NIQE
    "NIQEModel",
    "fit_niqe_model",
    "compute_niqe",
    # Checkpointing
    "save_checkpoint",
    "load_checkpoint",
    "save_model",
    "load_model",
    # Visualization
    "visualize_results",
    "plot_training_curves",
    "create_comparison_grid",
    "save_comparison",
    "plot_metric_histogram",
    # Image utilities
    "normalize_image",
    "denormalize_image",
    "rgb_to_grayscale",
    "tensor_to_numpy",
    "numpy_to_tensor",
    # Logging
    "get_logger",
    "setup_logging",
]
