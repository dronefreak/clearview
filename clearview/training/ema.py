"""Exponential Moving Average (EMA) of model weights.

Maintains a shadow copy of model parameters that is updated after every
optimizer step using an exponential moving average. EMA weights typically
generalize better than the raw training weights and are commonly used for
validation/checkpointing and final model export in SOTA restoration models.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from clearview.utils.logger import get_logger

logger = get_logger(__name__)


class ExponentialMovingAverage:
    """Exponential moving average of model parameters and buffers.

    Args:
        model: Model whose weights should be tracked
        decay: EMA decay rate. Higher values (e.g. 0.999, 0.9999) average
            over a longer window and are recommended for longer training runs.
        update_after_step: Number of updates to skip before EMA starts
            tracking (useful to avoid averaging in noisy early weights).
        use_warmup: If True, ramps the effective decay up from 0 towards
            ``decay`` following ``min(decay, (1 + n) / (10 + n))`` where
            ``n`` is the number of updates. This prevents the shadow weights
            from being dominated by the (near-random) initial weights.
        device: Device to store the shadow weights on. Defaults to the
            model's parameters' device.

    Example:
        >>> model = UNet()
        >>> ema = ExponentialMovingAverage(model, decay=0.999)
        >>> for batch in train_loader:
        ...     loss = train_step(model, batch)
        ...     optimizer.step()
        ...     ema.update(model)
        >>> backup = ema.apply_shadow(model)  # swap in EMA weights
        ...  # run validation/inference with model
        >>> ema.restore(model, backup)  # restore raw training weights
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        update_after_step: int = 0,
        use_warmup: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize EMA tracker with a shadow copy of the model's state dict."""
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1], got {decay}")

        self.decay = decay
        self.update_after_step = update_after_step
        self.use_warmup = use_warmup
        self.num_updates = 0

        self.shadow: Dict[str, torch.Tensor] = {
            name: param.detach().clone().to(device or param.device)
            for name, param in model.state_dict().items()
        }

    def _current_decay(self) -> float:
        """Compute the effective decay for the current update step."""
        if not self.use_warmup:
            return self.decay
        return min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow weights with the current model weights.

        Args:
            model: Model with the latest (post-optimizer-step) weights
        """
        self.num_updates += 1

        if self.num_updates <= self.update_after_step:
            return

        decay = self._current_decay()
        model_state = model.state_dict()

        for name, shadow_param in self.shadow.items():
            model_param = model_state[name].detach().to(shadow_param.device)

            if shadow_param.dtype.is_floating_point:
                shadow_param.mul_(decay).add_(model_param, alpha=1.0 - decay)
            else:
                # Non-floating point buffers (e.g. BatchNorm num_batches_tracked)
                # are not meaningfully averaged — just track the latest value.
                shadow_param.copy_(model_param)

    def apply_shadow(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Load the EMA shadow weights into the model in-place.

        Args:
            model: Model to load shadow weights into

        Returns:
            Backup of the model's original state dict, for use with
            :meth:`restore`.
        """
        backup = {
            name: param.detach().clone() for name, param in model.state_dict().items()
        }
        device_matched_shadow = {
            name: tensor.to(backup[name].device) for name, tensor in self.shadow.items()
        }
        model.load_state_dict(device_matched_shadow, strict=True)
        return backup

    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        """Restore the model's original (non-EMA) weights.

        Args:
            model: Model to restore
            backup: Backup dict returned by :meth:`apply_shadow`
        """
        model.load_state_dict(backup, strict=True)

    def state_dict(self) -> Dict[str, Any]:
        """Get a serializable state dict for checkpointing."""
        return {
            "shadow": self.shadow,
            "num_updates": self.num_updates,
            "decay": self.decay,
            "update_after_step": self.update_after_step,
            "use_warmup": self.use_warmup,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load EMA state from a checkpoint dict."""
        self.shadow = state["shadow"]
        self.num_updates = state.get("num_updates", 0)
        self.decay = state.get("decay", self.decay)
        self.update_after_step = state.get("update_after_step", self.update_after_step)
        self.use_warmup = state.get("use_warmup", self.use_warmup)


__all__ = ["ExponentialMovingAverage"]
