"""Checkpoint management utilities.

Provides functions for saving and loading model checkpoints, including
optimizer state, training progress, and configuration.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    filepath: Union[str, Path] = "checkpoint.pth",
    **kwargs: Any,
) -> None:
    """Save model checkpoint with full training state.

    Saves model weights, optimizer state, epoch, metrics, and any additional
    training information for resuming training.

    Args:
        model: PyTorch model to save
        optimizer: Optimizer (optional, for resuming training)
        epoch: Current epoch number
        metrics: Dictionary of metrics (e.g., {'loss': 0.5, 'psnr': 25.3})
        config: Model/training configuration
        filepath: Path to save checkpoint
        **kwargs: Additional items to save (e.g., scheduler state, best_metric)

    Example:
        >>> model = UNet()
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> save_checkpoint(
        ...     model, optimizer,
        ...     epoch=50,
        ...     metrics={'psnr': 28.5, 'ssim': 0.92},
        ...     filepath='checkpoints/epoch_50.pth'
        ... )
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "config": config,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    # Add any extra items
    checkpoint.update(kwargs)

    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[Optimizer] = None,
    map_location: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load model checkpoint and restore training state.

    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into (optional)
        optimizer: Optimizer to load state into (optional)
        map_location: Device to map tensors to (e.g., 'cpu', 'cuda:0')
        strict: Whether to strictly enforce state_dict keys match

    Returns:
        Dictionary containing checkpoint data

    Example:
        >>> model = UNet()
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>>
        >>> checkpoint = load_checkpoint(
        ...     'checkpoints/epoch_50.pth',
        ...     model=model,
        ...     optimizer=optimizer
        ... )
        >>>
        >>> start_epoch = checkpoint['epoch'] + 1
        >>> best_metric = checkpoint.get('best_metric', 0)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=map_location)

    # Load model weights
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected dict checkpoint, got {type(checkpoint)}")
    return checkpoint


def save_model(
    model: nn.Module,
    filepath: Union[str, Path] = "model.pth",
    save_weights_only: bool = True,
) -> None:
    """Save model weights or entire model.

    Simplified interface for saving just the model (no training state).

    Args:
        model: PyTorch model
        filepath: Save path
        save_weights_only: If True, saves only state_dict. If False, saves entire model.

    Example:
        >>> model = UNet()
        >>> save_model(model, 'models/unet_trained.pth')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if save_weights_only:
        torch.save(model.state_dict(), filepath)
    else:
        torch.save(model, filepath)


def load_model(
    filepath: Union[str, Path],
    model: Optional[nn.Module] = None,
    map_location: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
) -> nn.Module:
    """Load model weights or entire model.

    Args:
        filepath: Path to model file
        model: Model architecture (if loading weights only)
        map_location: Device to map tensors to
        strict: Whether to strictly enforce state_dict keys match

    Returns:
        Loaded model

    Example:
        >>> # Load weights into existing model
        >>> model = UNet()
        >>> model = load_model('models/unet_trained.pth', model=model)
        >>>
        >>> # Or load entire model (if saved with save_weights_only=False)
        >>> model = load_model('models/unet_complete.pth')
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    if model is None:
        # Load entire model
        model = torch.load(filepath, map_location=map_location)
    else:
        # Load weights into existing model
        state_dict = torch.load(filepath, map_location=map_location)
        model.load_state_dict(state_dict, strict=strict)

    return model


class CheckpointManager:
    """Manage model checkpoints with automatic cleanup.

    Keeps only the best N checkpoints based on a metric, automatically
    deleting older/worse checkpoints.

    Args:
        checkpoint_dir: Directory to save checkpoints
        max_checkpoints: Maximum number of checkpoints to keep
        mode: 'min' or 'max' (minimize or maximize metric)

    Example:
        >>> manager = CheckpointManager('checkpoints/', max_checkpoints=5, mode='max')
        >>>
        >>> for epoch in range(100):
        ...     # ... training ...
        ...     psnr = evaluate(model)
        ...
        ...     manager.save_checkpoint(
        ...         model, optimizer,
        ...         epoch=epoch,
        ...         metric_value=psnr,
        ...         metric_name='psnr'
        ...     )
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        mode: str = "max",
    ) -> None:
        """Initialize checkpoint manager."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.mode = mode

        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        # Track saved checkpoints: {filepath: metric_value}
        self.checkpoints: Dict[Path, float] = {}

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        epoch: Optional[int] = None,
        metric_value: Optional[float] = None,
        metric_name: str = "metric",
        **kwargs: Any,
    ) -> Optional[Path]:
        """Save checkpoint and manage cleanup.

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Epoch number
            metric_value: Metric value for ranking
            metric_name: Name of the metric
            **kwargs: Additional checkpoint data

        Returns:
            Path to saved checkpoint, or None if not saved
        """
        if metric_value is None:
            # If no metric, just save with epoch number
            filename = f"checkpoint_epoch_{epoch}.pth"
        else:
            # Include metric in filename for easy identification
            filename = f"checkpoint_epoch_{epoch}_{metric_name}_{metric_value:.4f}.pth"

        filepath = self.checkpoint_dir / filename

        # Save checkpoint
        save_checkpoint(
            model,
            optimizer,
            epoch=epoch,
            metrics={metric_name: metric_value} if metric_value is not None else None,
            filepath=filepath,
            **kwargs,
        )

        if metric_value is not None:
            self.checkpoints[filepath] = metric_value
            self._cleanup_checkpoints()

        return filepath

    def _cleanup_checkpoints(self) -> None:
        """Remove worst checkpoints to maintain max_checkpoints limit."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return

        # Sort checkpoints by metric
        sorted_checkpoints = sorted(
            self.checkpoints.items(), key=lambda x: x[1], reverse=(self.mode == "max")
        )

        # Keep best checkpoints, remove others
        _ = sorted_checkpoints[: self.max_checkpoints]
        to_remove = sorted_checkpoints[self.max_checkpoints :]

        for filepath, _ in to_remove:
            if filepath.exists():
                filepath.unlink()
            del self.checkpoints[filepath]

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint.

        Returns:
            Path to best checkpoint, or None if no checkpoints saved
        """
        if not self.checkpoints:
            return None

        best = (
            max(self.checkpoints.items(), key=lambda x: x[1])
            if self.mode == "max"
            else min(self.checkpoints.items(), key=lambda x: x[1])
        )

        return best[0]

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to most recently saved checkpoint.

        Returns:
            Path to latest checkpoint, or None if no checkpoints saved
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pth"))
        if not checkpoints:
            return None

        return max(checkpoints, key=lambda p: p.stat().st_mtime)


__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "save_model",
    "load_model",
    "CheckpointManager",
]
