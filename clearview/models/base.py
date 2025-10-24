"""Base model class for all architectures.

Provides common interface and utilities for model implementation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Abstract base class for all deraining models.

    All custom models should inherit from this class and implement
    the forward method. This ensures a consistent interface across models.

    Args:
        in_channels: Number of input channels (typically 3 for RGB)
        out_channels: Number of output channels (typically 3 for RGB)

    Example:
        >>> class MyModel(BaseModel):
        ...     def __init__(self, in_channels=3, out_channels=3):
        ...         super().__init__(in_channels, out_channels)
        ...         self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        ...
        ...     def forward(self, x):
        ...         return self.conv(x)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
    ) -> None:
        """Initialize base model.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Get number of parameters in the model.

        Args:
            trainable_only: If True, count only trainable parameters

        Returns:
            Number of parameters

        Example:
            >>> model = UNet()
            >>> print(f"Trainable params: {model.get_num_params():,}")
            >>> print(f"Total params: {model.get_num_params(trainable_only=False):,}")
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_model_size_mb(self) -> float:
        """Get approximate model size in megabytes.

        Returns:
            Model size in MB

        Example:
            >>> model = UNet()
            >>> print(f"Model size: {model.get_model_size_mb():.2f} MB")
        """
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers())
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return float(size_mb)

    def summary(self, input_size: Optional[Tuple[int, ...]] = None) -> Dict[str, Any]:
        """Get model summary with parameter counts and size.

        Args:
            input_size: Optional input size (B, C, H, W) for computing output shape

        Returns:
            Dictionary containing model statistics

        Example:
            >>> model = UNet()
            >>> summary = model.summary(input_size=(1, 3, 256, 256))
            >>> print(summary)
        """
        summary_dict = {
            "model_name": self.__class__.__name__,
            "total_params": self.get_num_params(trainable_only=False),
            "trainable_params": self.get_num_params(trainable_only=True),
            "model_size_mb": self.get_model_size_mb(),
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
        }

        if input_size is not None:
            # Compute output shape
            device = next(self.parameters()).device
            dummy_input = torch.randn(*input_size).to(device)

            with torch.no_grad():
                output = self(dummy_input)

            summary_dict["input_shape"] = tuple(dummy_input.shape)
            summary_dict["output_shape"] = tuple(output.shape)

        return summary_dict

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        Returns:
            Dictionary containing model configuration

        Example:
            >>> model = UNet(in_channels=3, out_channels=3)
            >>> config = model.get_config()
            >>> # Can be used to recreate model later
        """
        return {
            "model_type": self.__class__.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
        }

    def freeze(self) -> None:
        """Freeze all model parameters (disable gradient computation).

        Useful for fine-tuning or using as a feature extractor.

        Example:
            >>> model = UNet()
            >>> model.freeze()
            >>> # Now model parameters won't be updated during training
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all model parameters (enable gradient computation).

        Example:
            >>> model = UNet()
            >>> model.freeze()
            >>> # ... do something ...
            >>> model.unfreeze()
            >>> # Now parameters can be updated again
        """
        for param in self.parameters():
            param.requires_grad = True

    def __repr__(self) -> str:
        """String representation of the model."""
        num_params = self.get_num_params()
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"params={num_params:,})"
        )
