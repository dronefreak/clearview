"""Training utilities for image deraining models.

This module provides the training infrastructure including the main Trainer class,
callbacks for training control, and utilities for managing the training loop.
"""

from clearview.training.callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    ProgressCallback,
)
from clearview.training.ema import ExponentialMovingAverage
from clearview.training.schedulers import WarmupCosineScheduler
from clearview.training.trainer import Trainer

__all__ = [
    # Trainer
    "Trainer",
    # Callbacks
    "Callback",
    "CallbackList",
    "ModelCheckpoint",
    "EarlyStopping",
    "LearningRateScheduler",
    "ProgressCallback",
    # EMA
    "ExponentialMovingAverage",
    # Schedulers
    "WarmupCosineScheduler",
]
