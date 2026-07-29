"""Learning rate schedulers.

Provides convenience schedulers not available directly in
``torch.optim.lr_scheduler``, notably a cosine-annealing schedule with a
linear warmup phase — a widely used recipe for training restoration
transformers/CNNs (e.g. Restormer, NAFNet) from scratch.
"""

import math
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """Linear warmup followed by cosine annealing decay.

    During the first ``warmup_epochs`` steps, the learning rate increases
    linearly from ``warmup_start_lr`` to each parameter group's base LR.
    Afterwards, it follows a cosine decay down to ``eta_min`` over the
    remaining ``total_epochs - warmup_epochs`` steps.

    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of warmup steps (epochs or iterations,
            depending on how often ``.step()`` is called)
        total_epochs: Total number of steps the schedule spans
        warmup_start_lr: Initial learning rate at the start of warmup.
            Default: 0.0
        eta_min: Minimum learning rate at the end of the cosine decay.
            Default: 0.0
        last_epoch: The index of the last epoch. Default: -1

    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        >>> scheduler = WarmupCosineScheduler(
        ...     optimizer, warmup_epochs=5, total_epochs=100
        ... )
        >>> for epoch in range(100):
        ...     train_one_epoch(...)
        ...     scheduler.step()

        Can also be used with the existing ``LearningRateScheduler`` callback:

        >>> from clearview.training import LearningRateScheduler
        >>> callback = LearningRateScheduler(scheduler)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """Initialize the warmup + cosine scheduler."""
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        if total_epochs <= warmup_epochs:
            raise ValueError(
                f"total_epochs ({total_epochs}) must be greater than "
                f"warmup_epochs ({warmup_epochs})"
            )

        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute the learning rate for the current step."""
        step = self.last_epoch

        if self.warmup_epochs > 0 and step < self.warmup_epochs:
            progress = step / self.warmup_epochs
            return [
                self.warmup_start_lr + progress * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]

        cosine_progress = (step - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs
        )
        cosine_progress = min(cosine_progress, 1.0)

        return [
            self.eta_min
            + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * cosine_progress))
            for base_lr in self.base_lrs
        ]


__all__ = ["WarmupCosineScheduler"]
