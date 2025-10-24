"""Utility functions for training, evaluation, and inference.

This module provides various utilities including metrics computation,
checkpoint management, visualization tools, and helper functions.
"""

from clearview.utils.metrics import (
    compute_psnr,
    compute_ssim,
    compute_mae,
    compute_mse,
    compute_metrics,
    MetricsTracker,
)
from clearview.utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    save_model,
    load_model,
)
from clearview.utils.visualization import (
    visualize_results,
    plot_training_curves,
    create_comparison_grid,
    save_comparison,
)
from clearview.utils.image import (
    normalize_image,
    denormalize_image,
    rgb_to_grayscale,
    tensor_to_numpy,
    numpy_to_tensor,
)
from clearview.utils.logger import (
    get_logger,
    setup_logging,
)

__all__ = [
    # Metrics
    "compute_psnr",
    "compute_ssim",
    "compute_mae",
    "compute_mse",
    "compute_metrics",
    "MetricsTracker",
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
