"""Visualization utilities for results and training progress.

Provides functions for creating comparison grids, plotting training curves,
and visualizing deraining results.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to displayable numpy image.

    Args:
        tensor: Image tensor (C, H, W) or (B, C, H, W)

    Returns:
        NumPy array (H, W, C) in range [0, 255]
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image from batch

    # Move to CPU and convert to numpy
    img: np.ndarray = tensor.detach().cpu().numpy()

    # Transpose from (C, H, W) to (H, W, C)
    img = np.transpose(img, (1, 2, 0))

    # Clip to [0, 1] and convert to [0, 255]
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    # Handle grayscale
    if img.shape[2] == 1:
        img = img.squeeze(2)

    return img


def visualize_results(
    rainy: Union[torch.Tensor, np.ndarray],
    derained: Union[torch.Tensor, np.ndarray],
    clean: Optional[Union[torch.Tensor, np.ndarray]] = None,
    titles: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 5),
) -> Figure:
    """Visualize deraining results side-by-side.

    Args:
        rainy: Rainy input image
        derained: Model output
        clean: Ground truth (optional)
        titles: Custom titles for each image
        figsize: Figure size

    Returns:
        Matplotlib figure

    Example:
        >>> fig = visualize_results(rainy_img, derained_img, clean_img)
        >>> plt.show()
        >>> # Or save
        >>> fig.savefig('results.png', dpi=150, bbox_inches='tight')
    """
    # Convert tensors to images
    if isinstance(rainy, torch.Tensor):
        rainy = tensor_to_image(rainy)
    if isinstance(derained, torch.Tensor):
        derained = tensor_to_image(derained)
    if clean is not None and isinstance(clean, torch.Tensor):
        clean = tensor_to_image(clean)

    # Setup subplot layout
    num_images = 3 if clean is not None else 2
    fig, axes = plt.subplots(1, num_images, figsize=figsize)

    if num_images == 2:
        axes = [axes[0], axes[1]]

    # Default titles
    if titles is None:
        titles = ["Rainy Input", "Derained Output"]
        if clean is not None:
            titles.append("Ground Truth")

    # Plot images
    axes[0].imshow(rainy)
    axes[0].set_title(titles[0], fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(derained)
    axes[1].set_title(titles[1], fontsize=12, fontweight="bold")
    axes[1].axis("off")

    if clean is not None:
        axes[2].imshow(clean)
        axes[2].set_title(titles[2], fontsize=12, fontweight="bold")
        axes[2].axis("off")

    plt.tight_layout()
    return fig


def create_comparison_grid(
    images: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
    max_images: int = 8,
    figsize: Optional[Tuple[int, int]] = None,
) -> Figure:
    """Create a grid comparing multiple deraining results.

    Args:
        images: List of (rainy, derained, clean) tuples
        max_images: Maximum number of image sets to show
        figsize: Figure size (auto-calculated if None)

    Returns:
        Matplotlib figure with comparison grid

    Example:
        >>> # Collect results from test set
        >>> results = []
        >>> for rainy, clean in test_loader:
        ...     derained = model(rainy)
        ...     results.append((rainy[0], derained[0], clean[0]))
        >>>
        >>> fig = create_comparison_grid(results, max_images=4)
        >>> fig.savefig('comparison_grid.png', dpi=150)
    """
    images = images[:max_images]
    num_images = len(images)

    # Check if we have ground truth
    has_clean = images[0][2] is not None
    num_cols = 3 if has_clean else 2

    if figsize is None:
        figsize = (num_cols * 4, num_images * 3)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(num_images, num_cols, hspace=0.3, wspace=0.1)

    titles = (
        ["Rainy", "Derained", "Ground Truth"] if has_clean else ["Rainy", "Derained"]
    )

    for i, (rainy, derained, clean) in enumerate(images):
        # Convert to displayable format
        rainy_img = tensor_to_image(rainy)
        derained_img = tensor_to_image(derained)

        # Rainy image
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(rainy_img)
        if i == 0:
            ax.set_title(titles[0], fontsize=12, fontweight="bold")
        ax.axis("off")

        # Derained image
        ax = fig.add_subplot(gs[i, 1])
        ax.imshow(derained_img)
        if i == 0:
            ax.set_title(titles[1], fontsize=12, fontweight="bold")
        ax.axis("off")

        # Ground truth (if available)
        if has_clean and clean is not None:
            clean_img = tensor_to_image(clean)
            ax = fig.add_subplot(gs[i, 2])
            ax.imshow(clean_img)
            if i == 0:
                ax.set_title(titles[2], fontsize=12, fontweight="bold")
            ax.axis("off")

    return fig


def plot_training_curves(
    train_history: dict,
    val_history: Optional[dict] = None,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Plot training and validation curves.

    Args:
        train_history: Dict with metric lists, e.g. {'loss': [...], 'psnr': [...]}
        val_history: Validation metrics (optional)
        metrics: Specific metrics to plot (if None, plots all)
        figsize: Figure size
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure

    Example:
        >>> history = {
        ...     'loss': [0.5, 0.4, 0.3, ...],
        ...     'psnr': [25.0, 26.5, 28.0, ...],
        ...     'ssim': [0.80, 0.85, 0.88, ...]
        ... }
        >>> fig = plot_training_curves(history, metrics=['loss', 'psnr'])
        >>> plt.show()
    """
    if metrics is None:
        metrics = list(train_history.keys())

    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)

    if num_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Plot training curve
        if metric in train_history:
            epochs = range(1, len(train_history[metric]) + 1)
            ax.plot(
                epochs,
                train_history[metric],
                label="Train",
                marker="o",
                markersize=3,
                linewidth=2,
            )

        # Plot validation curve
        if val_history and metric in val_history:
            epochs = range(1, len(val_history[metric]) + 1)
            ax.plot(
                epochs,
                val_history[metric],
                label="Validation",
                marker="s",
                markersize=3,
                linewidth=2,
            )

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(metric.upper(), fontsize=11)
        ax.set_title(f"{metric.upper()} over Epochs", fontsize=12, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def save_comparison(
    rainy: Union[torch.Tensor, np.ndarray],
    derained: Union[torch.Tensor, np.ndarray],
    clean: Optional[Union[torch.Tensor, np.ndarray]] = None,
    save_path: Union[str, Path] = "comparison.png",
    dpi: int = 150,
) -> None:
    """Save a comparison visualization to file.

    Convenience function that creates and saves in one call.

    Args:
        rainy: Rainy input
        derained: Derained output
        clean: Ground truth (optional)
        save_path: Where to save the image
        dpi: Resolution for saved image

    Example:
        >>> save_comparison(
        ...     rainy_img, derained_img, clean_img,
        ...     save_path='results/test_001.png'
        ... )
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = visualize_results(rainy, derained, clean)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_metric_histogram(
    metrics: dict,
    bins: int = 30,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Plot histogram distribution of metrics across test set.

    Args:
        metrics: Dict with metric name -> list of values
        bins: Number of histogram bins
        figsize: Figure size
        save_path: Optional save path

    Returns:
        Matplotlib figure

    Example:
        >>> # Collect metrics from test set
        >>> psnr_values = []
        >>> ssim_values = []
        >>> for batch in test_loader:
        ...     # ... compute metrics ...
        ...     psnr_values.append(psnr)
        ...     ssim_values.append(ssim)
        >>>
        >>> fig = plot_metric_histogram({
        ...     'psnr': psnr_values,
        ...     'ssim': ssim_values
        ... })
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)

    if num_metrics == 1:
        axes = [axes]

    for idx, (name, values) in enumerate(metrics.items()):
        ax = axes[idx]

        # Plot histogram
        ax.hist(values, bins=bins, alpha=0.7, edgecolor="black")

        # Add mean line
        mean_val = np.mean(values)
        ax.axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:.2f}",
        )

        ax.set_xlabel(name.upper(), fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"{name.upper()} Distribution", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


__all__ = [
    "tensor_to_image",
    "visualize_results",
    "create_comparison_grid",
    "plot_training_curves",
    "save_comparison",
    "plot_metric_histogram",
]
