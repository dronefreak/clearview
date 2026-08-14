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
from clearview.training.ema import ExponentialMovingAverage
from clearview.utils.logger import get_logger
from clearview.utils.metrics import MetricsTracker, compute_metrics

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
        accumulation_steps: Number of batches to accumulate gradients over
            before an optimizer step (simulates a larger batch size)
        use_ema: Track an exponential moving average (EMA) of model weights,
            typically yielding better-generalizing weights for validation
            and final checkpoints
        ema_decay: EMA decay rate (higher = smoother/longer averaging window)
        ema_update_after_step: Number of optimizer steps to skip before EMA
            tracking begins
        ema_use_warmup: Ramp the effective EMA decay up gradually during
            early training instead of using the full decay from step 1
        validate_with_ema: If True (default) and ``use_ema`` is enabled,
            validation runs with the EMA weights instead of the raw
            training weights
        compile_model: Wrap the model with ``torch.compile()`` for
            potentially faster training/inference (PyTorch 2.x). Falls
            back to eager mode with a warning if compilation fails.
        compile_kwargs: Extra keyword arguments forwarded to
            ``torch.compile()`` (e.g. ``{'mode': 'reduce-overhead'}``)
        channels_last: Use ``torch.channels_last`` (NHWC) memory format for
            the model and input batches instead of the default contiguous
            (NCHW) format. Can noticeably speed up convolutional models on
            modern GPUs with Tensor Cores when combined with mixed
            precision. Only affects 4D (image) tensors.
        amp_dtype: Autocast dtype to use when ``mixed_precision=True``.
            Default: ``torch.float16``. Use ``torch.bfloat16`` on
            Ampere+ GPUs (or CPUs) to avoid gradient-scaling overhead and
            reduce the risk of overflow, at a small cost in precision.
            When ``amp_dtype`` is ``torch.bfloat16``, the internal
            ``GradScaler`` is disabled since bfloat16 does not need loss
            scaling.

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
        accumulation_steps: int = 1,
        use_ema: bool = False,
        ema_decay: float = 0.999,
        ema_update_after_step: int = 0,
        ema_use_warmup: bool = True,
        validate_with_ema: bool = True,
        compile_model: bool = False,
        compile_kwargs: Optional[Dict[str, Any]] = None,
        channels_last: bool = False,
        amp_dtype: torch.dtype = torch.float16,
    ) -> None:
        """Initialize trainer."""
        self.channels_last = channels_last
        self._memory_format = (
            torch.channels_last if channels_last else torch.contiguous_format
        )
        self.model = model.to(device, memory_format=self._memory_format)
        self.optimizer = optimizer
        self.loss_fn = loss_fn.to(device)
        self.device = device
        self.gradient_clip_val = gradient_clip_val
        self.mixed_precision = mixed_precision
        self.amp_dtype = amp_dtype
        self.accumulation_steps = max(1, accumulation_steps)

        # Optionally compile the model for faster training/inference
        # (PyTorch 2.x). Falls back gracefully with a warning if
        # torch.compile is unavailable or fails on this platform/model.
        self.compiled = False
        if compile_model:
            try:
                self.model = torch.compile(self.model, **(compile_kwargs or {}))
                self.compiled = True
            except Exception as e:
                logger.warning(
                    f"torch.compile() failed, falling back to eager mode: {e}"
                )

        # Exponential moving average of model weights
        self.use_ema = use_ema
        self.validate_with_ema = validate_with_ema
        self.ema: Optional[ExponentialMovingAverage] = (
            ExponentialMovingAverage(
                self.raw_model,
                decay=ema_decay,
                update_after_step=ema_update_after_step,
                use_warmup=ema_use_warmup,
            )
            if use_ema
            else None
        )

        # Metrics
        self.metrics = metrics or ["psnr", "ssim"]

        # Callbacks
        self.callbacks = CallbackList(callbacks or [])

        # Set model/optimizer in callbacks that need it. Callbacks that
        # checkpoint the model (e.g. ModelCheckpoint) must see raw_model, not
        # self.model directly — otherwise, under torch.compile(), they save
        # OptimizedModule's `_orig_mod.`-prefixed state_dict, which is not
        # loadable by from_pretrained() without --compile.
        for callback in callbacks or []:
            if hasattr(callback, "set_model"):
                try:
                    callback.set_model(self.raw_model)
                except Exception as e:
                    logger.warning(
                        f"Failed to set model on {callback.__class__.__name__}: {e}"
                    )
            if hasattr(callback, "set_optimizer"):
                try:
                    callback.set_optimizer(self.optimizer)
                except Exception as e:
                    logger.warning(
                        f"Failed to set optimizer on {callback.__class__.__name__}: {e}"
                    )

        # Mixed precision scaler. bfloat16 does not need loss scaling (it
        # has the same exponent range as float32), so disable the scaler
        # in that case — GradScaler(enabled=False) makes scale()/step()/
        # update() transparent pass-throughs, so the rest of the training
        # loop doesn't need a separate code path.
        self.scaler = torch.amp.GradScaler(
            device=self.device.split(":")[0],
            enabled=mixed_precision and self.amp_dtype == torch.float16,
        )

        # Training state
        self.epoch: int = 0
        self.history: Dict[str, List] = {
            "train_loss": [],
            "val_loss": [],
        }
        for metric in self.metrics:
            self.history[f"train_{metric}"] = []
            self.history[f"val_{metric}"] = []

    @property
    def raw_model(self) -> nn.Module:
        """Return the underlying model, unwrapped from ``torch.compile()``.

        ``torch.compile()`` wraps the model in an ``OptimizedModule`` whose
        ``state_dict()`` keys are prefixed with ``_orig_mod.``. Using the raw
        module for EMA and checkpointing keeps checkpoints portable across
        compiled/eager runs.
        """
        return getattr(self.model, "_orig_mod", self.model)

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move a tensor to the trainer's device and memory format.

        Applies ``torch.channels_last`` (NHWC) memory format when
        ``channels_last=True`` was passed to the constructor; this only
        makes sense for 4D (N, C, H, W) image tensors, so lower-dimensional
        tensors are moved without a memory-format change.
        """
        if self.channels_last and tensor.dim() == 4:
            return tensor.to(self.device, memory_format=torch.channels_last)
        return tensor.to(self.device)

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
        num_batches = len(train_loader)

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Get data
            if isinstance(batch, (tuple, list)):
                rainy, clean = self._to_device(batch[0]), self._to_device(batch[1])
            else:
                rainy = self._to_device(batch["rainy"])
                clean = self._to_device(batch["clean"])

            # Callback
            self.callbacks.on_batch_begin(batch_idx)

            is_last_batch = (batch_idx + 1) == num_batches
            should_step = (
                batch_idx + 1
            ) % self.accumulation_steps == 0 or is_last_batch

            # Forward pass
            if self.mixed_precision:
                with torch.autocast(
                    device_type=self.device.split(":")[0], dtype=self.amp_dtype
                ):
                    output = self.model(rainy)
                    loss = self.loss_fn(output, clean)

                # Scale loss for accumulation, then backpropagate
                scaled_loss = loss / self.accumulation_steps
                self.scaler.scale(scaled_loss).backward()

                if should_step:
                    if self.gradient_clip_val is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip_val
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    if self.ema is not None:
                        self.ema.update(self.raw_model)
            else:
                output = self.model(rainy)
                loss = self.loss_fn(output, clean)

                # Scale loss for accumulation, then backpropagate
                (loss / self.accumulation_steps).backward()

                if should_step:
                    if self.gradient_clip_val is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip_val
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if self.ema is not None:
                        self.ema.update(self.raw_model)

            # Track unscaled loss
            loss_tracker.update({"loss": loss.item()}, batch_size=rainy.size(0))

            # Compute metrics
            with torch.no_grad():
                batch_metrics = compute_metrics(
                    output.detach().float(),
                    clean.detach().float(),
                    metrics=self.metrics,
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
                rainy, clean = self._to_device(batch[0]), self._to_device(batch[1])
            else:
                rainy = self._to_device(batch["rainy"])
                clean = self._to_device(batch["clean"])

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

                # Validate. When validating with EMA, keep the shadow weights
                # applied through the checkpoint callbacks below (not just
                # validate_epoch) — otherwise a callback like ModelCheckpoint
                # saves the raw (non-EMA) weights even though the val_psnr it
                # is keying off of was computed on the EMA weights.
                val_metrics = None
                ema_backup = None
                if val_loader is not None:
                    if self.ema is not None and self.validate_with_ema:
                        ema_backup = self.ema.apply_shadow(self.raw_model)
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
                try:
                    self.callbacks.on_epoch_end(epoch, logs)
                finally:
                    if ema_backup is not None and self.ema is not None:
                        self.ema.restore(self.raw_model, ema_backup)

                # Check for early stopping
                if self._check_early_stop():
                    logger.info("Early stopping triggered")
                    break

                # Log epoch summary
                self._log_epoch_summary(epoch, train_metrics, val_metrics)

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        finally:
            if self.scaler is not None:
                logger.debug(
                    f"GradScaler final state — scale: {self.scaler.get_scale():.1f}"
                )
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
            "model_state_dict": self.raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }
        if self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.state_dict()
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

        self.raw_model.load_state_dict(checkpoint["model_state_dict"])

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.ema is not None and "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])

        if "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"]

        if "history" in checkpoint:
            self.history = checkpoint["history"]

        logger.info(f"Checkpoint loaded from {filepath}")

        if not isinstance(checkpoint, dict):
            raise TypeError(f"Expected dict checkpoint, got {type(checkpoint)}")
        return checkpoint


__all__ = ["Trainer"]
