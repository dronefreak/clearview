"""Training callbacks for monitoring and controlling training.

Provides extensible callback system for model checkpointing, early stopping,
learning rate scheduling, and custom training behavior.
"""

from abc import ABC
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
import logging

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


logger = logging.getLogger(__name__)


class Callback(ABC):
    """Base class for training callbacks.

    Callbacks provide hooks into the training loop for custom behavior
    like checkpointing, early stopping, logging, etc.

    Example:
        >>> class MyCallback(Callback):
        ...     def on_epoch_end(self, epoch, logs=None):
        ...         print(f"Epoch {epoch} finished!")
    """

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of each batch."""
        pass


class CallbackList:
    """Container for managing multiple callbacks.

    Args:
        callbacks: List of callback instances

    Example:
        >>> callbacks = CallbackList([
        ...     ModelCheckpoint('checkpoints/'),
        ...     EarlyStopping(patience=10)
        ... ])
    """

    def __init__(self, callbacks: List[Callback]) -> None:
        """Initialize callback list."""
        self.callbacks = callbacks or []

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_train_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_epoch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_epoch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_batch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)


class ModelCheckpoint(Callback):
    """Save model checkpoints during training.

    Saves checkpoints based on monitoring a metric. Can save best model only
    or save at every epoch.

    Args:
        filepath: Path pattern for saving checkpoints.
            Can include formatting like 'checkpoint_{epoch:03d}_{val_psnr:.2f}.pth'
        monitor: Metric to monitor (e.g., 'val_loss', 'val_psnr')
        mode: 'min' or 'max' (minimize or maximize the monitored metric)
        save_best_only: If True, only saves when monitored metric improves
        save_weights_only: If True, saves only model weights (not optimizer state)
        verbose: Verbosity level

    Example:
        >>> checkpoint = ModelCheckpoint(
        ...     filepath='checkpoints/best_model.pth',
        ...     monitor='val_psnr',
        ...     mode='max',
        ...     save_best_only=True
        ... )
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_weights_only: bool = False,
        verbose: int = 1,
    ) -> None:
        """Initialize model checkpoint callback."""
        super().__init__()

        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose

        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.model = None
        self.optimizer = None

    def set_model(self, model: nn.Module) -> None:
        """Set the model to checkpoint."""
        self.model = model

    def set_optimizer(self, optimizer: Optimizer) -> None:
        """Set the optimizer to checkpoint."""
        self.optimizer = optimizer

    def _is_improvement(self, current: float) -> bool:
        """Check if current metric is an improvement."""
        if self.mode == "min":
            return current < self.best_metric
        else:
            return current > self.best_metric

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Save checkpoint at epoch end."""
        logs = logs or {}

        current = logs.get(self.monitor)

        if current is None:
            if self.verbose > 0:
                logger.warning(
                    f"ModelCheckpoint: {self.monitor} not found in logs. "
                    f"Available metrics: {list(logs.keys())}"
                )
            return

        # Check if we should save
        should_save = not self.save_best_only or self._is_improvement(current)

        if should_save:
            # Format filepath
            filepath_str = str(self.filepath)
            if "{epoch" in filepath_str:
                filepath = Path(filepath_str.format(epoch=epoch, **logs))
            else:
                filepath = self.filepath

            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Save checkpoint
            if self.model is not None:
                state_dict: Dict[str, torch.Tensor] = self.model.state_dict()
            else:
                state_dict = {}
            if self.save_weights_only:
                torch.save(state_dict, filepath)
            else:
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": state_dict,
                    "metrics": logs,
                }
                if self.optimizer is not None:
                    checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()

                torch.save(checkpoint, filepath)

            if self._is_improvement(current):
                self.best_metric = current
                if self.verbose > 0:
                    logger.info(
                        f"Epoch {epoch}: {self.monitor} improved to {current:.4f}, "
                        f"saving model to {filepath}"
                    )
            elif self.verbose > 0:
                logger.info(f"Epoch {epoch}: saving model to {filepath}")


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving.

    Args:
        monitor: Metric to monitor
        patience: Number of epochs with no improvement to wait before stopping
        mode: 'min' or 'max'
        min_delta: Minimum change to qualify as an improvement
        verbose: Verbosity level
        restore_best_weights: Whether to restore model weights from best epoch

    Example:
        >>> early_stop = EarlyStopping(
        ...     monitor='val_loss',
        ...     patience=10,
        ...     min_delta=0.001,
        ...     restore_best_weights=True
        ... )
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 0.0,
        verbose: int = 1,
        restore_best_weights: bool = False,
    ) -> None:
        """Initialize early stopping callback."""
        super().__init__()

        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = abs(min_delta)
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights

        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        self.wait = 0
        self.stopped_epoch = 0
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.best_weights: Optional[Dict[str, torch.Tensor]] = None
        self.model = None
        self.stop_training = False

    def set_model(self, model: nn.Module) -> None:
        """Set the model."""
        self.model = model

    def _is_improvement(self, current: float) -> bool:
        """Check if current metric is an improvement."""
        if self.mode == "min":
            return current < (self.best_metric - self.min_delta)
        else:
            return current > (self.best_metric + self.min_delta)

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Reset state at training start."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_metric = float("inf") if self.mode == "min" else float("-inf")
        self.best_weights = None
        self.stop_training = False

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Check for early stopping at epoch end."""
        logs = logs or {}

        current = logs.get(self.monitor)

        if current is None:
            if self.verbose > 0:
                logger.warning(
                    f"EarlyStopping: {self.monitor} not found in logs. "
                    f"Available metrics: {list(logs.keys())}"
                )
            return

        if self._is_improvement(current):
            self.best_metric = current
            self.wait = 0

            if self.model is not None:
                state_dict: Dict[str, torch.Tensor] = self.model.state_dict()
            else:
                state_dict = {}

            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in state_dict.items()}

            if self.verbose > 0:
                logger.info(f"Epoch {epoch}: {self.monitor} improved to {current:.4f}")

        else:
            self.wait += 1
            if self.verbose > 0:
                logger.info(
                    f"Epoch {epoch}: {self.monitor} did not improve from {self.best_metric:.4f} "
                    f"({self.wait}/{self.patience})"
                )

            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True

                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose > 0:
                        logger.info("Restoring model weights from best epoch")
                    if self.model is not None:
                        self.model.load_state_dict(self.best_weights)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Print early stopping message."""
        if self.stopped_epoch > 0 and self.verbose > 0:
            logger.info(
                f"Early stopping triggered after epoch {self.stopped_epoch}. "
                f"Best {self.monitor}: {self.best_metric:.4f}"
            )


class LearningRateScheduler(Callback):
    """Learning rate scheduler callback.

    Wraps PyTorch learning rate schedulers for use in training loop.

    Args:
        scheduler: PyTorch LR scheduler instance
        verbose: Verbosity level

    Example:
        >>> from torch.optim.lr_scheduler import ReduceLROnPlateau
        >>>
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)
        >>> lr_callback = LearningRateScheduler(scheduler)
    """

    def __init__(
        self,
        scheduler: _LRScheduler,
        monitor: Optional[str] = None,
        verbose: int = 1,
    ) -> None:
        """Initialize LR scheduler callback."""
        super().__init__()

        self.scheduler = scheduler
        self.monitor = monitor
        self.verbose = verbose

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Step the scheduler at epoch end."""
        logs = logs or {}

        # Get current LR
        current_lr = self.scheduler.get_last_lr()[0]

        # Step scheduler
        if self.monitor is not None:
            # ReduceLROnPlateau needs metric value
            metric = logs.get(self.monitor)
            if metric is not None:
                self.scheduler.step(metric)
            else:
                if self.verbose > 0:
                    logger.warning(
                        f"LearningRateScheduler: {self.monitor} not found in logs"
                    )
        else:
            # Other schedulers just need step()
            self.scheduler.step()

        # Check if LR changed
        new_lr = self.scheduler.get_last_lr()[0]
        if new_lr != current_lr and self.verbose > 0:
            logger.info(
                f"Epoch {epoch}: Learning rate changed from {current_lr:.6f} to {new_lr:.6f}"
            )


class ProgressCallback(Callback):
    """Simple progress logging callback.

    Logs training progress at regular intervals.

    Args:
        log_every_n_epochs: Log every N epochs
        metrics_to_log: List of metric names to log (if None, logs all)

    Example:
        >>> progress = ProgressCallback(log_every_n_epochs=5)
    """

    def __init__(
        self,
        log_every_n_epochs: int = 1,
        metrics_to_log: Optional[List[str]] = None,
    ) -> None:
        """Initialize progress callback."""
        super().__init__()

        self.log_every_n_epochs = log_every_n_epochs
        self.metrics_to_log = metrics_to_log

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log progress at epoch end."""
        if epoch % self.log_every_n_epochs != 0:
            return

        logs = logs or {}

        # Filter metrics
        if self.metrics_to_log is not None:
            metrics = {k: v for k, v in logs.items() if k in self.metrics_to_log}
        else:
            metrics = logs

        # Format metrics string
        metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        logger.info(f"Epoch {epoch}: {metrics_str}")


__all__ = [
    "Callback",
    "CallbackList",
    "ModelCheckpoint",
    "EarlyStopping",
    "LearningRateScheduler",
    "ProgressCallback",
]
