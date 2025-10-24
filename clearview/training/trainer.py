"""Main training orchestrator.

Provides a high-level Trainer class that handles the training loop,
validation, callbacks, and logging.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from clearview.training.callbacks import Callback, CallbackList
from clearview.utils.metrics import compute_metrics, MetricsTracker
from clearview.utils.logger import get_logger


logger = get_logger(__name__)


class Trainer:
    """High-level trainer for image deraining models.

    Handles training loop, validation, callbacks, and metric tracking
    with a clean, extensible interface.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        loss_fn: Loss function
        device: Device to train on ('cpu' or 'cuda')
        callbacks: List of callbacks
        metrics: List of metrics to compute (e.g., ['psnr', 'ssim'])
        gradient_clip_val: Max gradient norm for clipping (None = no clipping)
        mixed_precision: Use automatic mixed precision (AMP)

    Example:
        >>> from clearview import UNet
        >>> from clearview.losses import CombinedLoss
        >>> from clearview.training import Trainer, ModelCheckpoint, EarlyStopping
        >>>
        >>> model = UNet()
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> loss_fn = CombinedLoss.from_config({
        ...     'l1': {'weight': 1.0},
        ...     'ssim': {'weight': 1.0},
        ... })
        >>>
        >>> trainer = Trainer(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     loss_fn=loss_fn,
        ...     callbacks=[
        ...         ModelCheckpoint('checkpoints/', monitor='val_psnr', mode='max'),
        ...         EarlyStopping(monitor='val_loss', patience=10)
        ...     ],
        ...     metrics=['psnr', 'ssim']
        ... )
        >>>
        >>> history = trainer.fit(
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     epochs=100
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        callbacks: Optional[List[Callback]] = None,
        metrics: Optional[List[str]] = None,
        gradient_clip_val: Optional[float] = None,
        mixed_precision: bool = False,
    ) -> None:
        """Initialize trainer."""
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn.to(device)
        self.device = device
        self.gradient_clip_val = gradient_clip_val
        self.mixed_precision = mixed_precision

        # Metrics
        self.metrics = metrics or ["psnr", "ssim"]

        # Callbacks
        self.callbacks = CallbackList(callbacks or [])

        # Set model/optimizer in callbacks that need it
        for callback in callbacks or []:
            if hasattr(callback, "set_model"):
                callback.set_model(self.model)
            if hasattr(callback, "set_optimizer"):
                callback.set_optimizer(self.optimizer)

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

        # Training state
        self.epoch: int = 0
        self.history: Dict[str, List] = {
            "train_loss": [],
            "val_loss": [],
        }
        for metric in self.metrics:
            self.history[f"train_{metric}"] = []
            self.history[f"val_{metric}"] = []

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of average training metrics
        """
        self.model.train()

        loss_tracker = MetricsTracker()
        metrics_tracker = MetricsTracker()

        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch} [Train]")

        for batch_idx, batch in enumerate(pbar):
            # Get data
            if isinstance(batch, (tuple, list)):
                rainy, clean = batch[0].to(self.device), batch[1].to(self.device)
            else:
                rainy = batch["rainy"].to(self.device)
                clean = batch["clean"].to(self.device)

            # Callback
            self.callbacks.on_batch_begin(batch_idx)

            # Forward pass
            self.optimizer.zero_grad()

            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    output = self.model(rainy)
                    loss = self.loss_fn(output, clean)

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.gradient_clip_val is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_val
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(rainy)
                loss = self.loss_fn(output, clean)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_val
                    )

                self.optimizer.step()

            # Track loss
            loss_tracker.update({"loss": loss.item()}, batch_size=rainy.size(0))

            # Compute metrics
            with torch.no_grad():
                batch_metrics = compute_metrics(
                    output.detach(), clean.detach(), metrics=self.metrics
                )
                metrics_tracker.update(batch_metrics, batch_size=rainy.size(0))

            # Update progress bar
            avg_loss = loss_tracker.average()["loss"]
            avg_metrics = metrics_tracker.average()
            pbar.set_postfix(
                {
                    "loss": f"{avg_loss:.4f}",
                    **{k: f"{v:.2f}" for k, v in avg_metrics.items()},
                }
            )

            # Callback
            self.callbacks.on_batch_end(batch_idx, logs={"loss": loss.item()})

        # Get epoch averages
        epoch_loss = loss_tracker.average()["loss"]
        epoch_metrics = metrics_tracker.average()

        return {"loss": epoch_loss, **epoch_metrics}

    @torch.no_grad()
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of average validation metrics
        """
        self.model.eval()

        loss_tracker = MetricsTracker()
        metrics_tracker = MetricsTracker()

        pbar = tqdm(val_loader, desc=f"Epoch {self.epoch} [Val]")

        for batch in pbar:
            # Get data
            if isinstance(batch, (tuple, list)):
                rainy, clean = batch[0].to(self.device), batch[1].to(self.device)
            else:
                rainy = batch["rainy"].to(self.device)
                clean = batch["clean"].to(self.device)

            # Forward pass
            output = self.model(rainy)
            loss = self.loss_fn(output, clean)

            # Track loss
            loss_tracker.update({"loss": loss.item()}, batch_size=rainy.size(0))

            # Compute metrics
            batch_metrics = compute_metrics(output, clean, metrics=self.metrics)
            metrics_tracker.update(batch_metrics, batch_size=rainy.size(0))

            # Update progress bar
            avg_loss = loss_tracker.average()["loss"]
            avg_metrics = metrics_tracker.average()
            pbar.set_postfix(
                {
                    "loss": f"{avg_loss:.4f}",
                    **{k: f"{v:.2f}" for k, v in avg_metrics.items()},
                }
            )

        # Get epoch averages
        epoch_loss = loss_tracker.average()["loss"]
        epoch_metrics = metrics_tracker.average()

        return {"loss": epoch_loss, **epoch_metrics}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        start_epoch: int = 0,
    ) -> Dict[str, List[float]]:
        """Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs to train
            start_epoch: Starting epoch (for resuming training)

        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )

        self.callbacks.on_train_begin()

        try:
            for epoch in range(start_epoch, epochs):
                self.epoch = epoch

                # Callback
                self.callbacks.on_epoch_begin(epoch)

                # Train
                train_metrics = self.train_epoch(train_loader)

                # Validate
                val_metrics = None
                if val_loader is not None:
                    val_metrics = self.validate_epoch(val_loader)

                # Update history
                self.history["train_loss"].append(train_metrics["loss"])
                for metric in self.metrics:
                    self.history[f"train_{metric}"].append(train_metrics[metric])

                if val_metrics is not None:
                    self.history["val_loss"].append(val_metrics["loss"])
                    for metric in self.metrics:
                        self.history[f"val_{metric}"].append(val_metrics[metric])

                # Prepare logs for callbacks
                logs = {
                    "train_loss": train_metrics["loss"],
                }
                for metric in self.metrics:
                    logs[f"train_{metric}"] = train_metrics[metric]

                if val_metrics is not None:
                    logs["val_loss"] = val_metrics["loss"]
                    for metric in self.metrics:
                        logs[f"val_{metric}"] = val_metrics[metric]

                # Callback
                self.callbacks.on_epoch_end(epoch, logs)

                # Check for early stopping
                if self._check_early_stop():
                    logger.info("Early stopping triggered")
                    break

                # Log epoch summary
                self._log_epoch_summary(epoch, train_metrics, val_metrics)

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        finally:
            self.callbacks.on_train_end()

        return self.history

    def _check_early_stop(self) -> bool:
        """Check if any callback triggered early stopping."""
        for callback in self.callbacks.callbacks:
            if hasattr(callback, "stop_training") and callback.stop_training:
                return True
        return False

    def _log_epoch_summary(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log epoch summary."""
        train_str = " | ".join([f"train_{k}={v:.4f}" for k, v in train_metrics.items()])

        if val_metrics is not None:
            val_str = " | ".join([f"val_{k}={v:.4f}" for k, v in val_metrics.items()])
            logger.info(f"Epoch {epoch}: {train_str} | {val_str}")
        else:
            logger.info(f"Epoch {epoch}: {train_str}")

    def save_checkpoint(self, filepath: Union[str, Path], **kwargs: Any) -> None:
        """Save training checkpoint.

        Args:
            filepath: Path to save checkpoint
            **kwargs: Additional items to save
        """
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }
        checkpoint.update(kwargs)

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(
        self, filepath: Union[str, Path], load_optimizer: bool = True
    ) -> Dict[str, Any]:
        """Load training checkpoint.

        Args:
            filepath: Path to checkpoint
            load_optimizer: Whether to load optimizer state

        Returns:
            Checkpoint dictionary
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"]

        if "history" in checkpoint:
            self.history = checkpoint["history"]

        logger.info(f"Checkpoint loaded from {filepath}")

        if not isinstance(checkpoint, dict):
            raise TypeError(f"Expected dict checkpoint, got {type(checkpoint)}")
        return checkpoint


__all__ = ["Trainer"]
