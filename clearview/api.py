"""High-level API for easy model usage.

Provides a simple interface for loading models, running inference,
and processing images without dealing with low-level details.
"""

from pathlib import Path
from typing import Optional, Union, Any

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from clearview.models import get_model
from clearview.utils.image import numpy_to_tensor, tensor_to_numpy, pad_to_multiple


class DerainingModel:
    """High-level interface for image deraining.

    Simplifies model loading, inference, and image processing.

    Example:
        >>> # Load pretrained model
        >>> model = DerainingModel.from_pretrained('unet', weights='path/to/weights.pth')
        >>>
        >>> # Process single image
        >>> clean = model.process('rainy_image.png', output_path='derained.png')
        >>>
        >>> # Process batch
        >>> clean_batch = model.process_batch(rainy_images)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """Initialize deraining model.

        Args:
            model: PyTorch model
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        weights: Optional[Union[str, Path]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **model_kwargs: Any,
    ) -> "DerainingModel":
        """Load model with optional pretrained weights.

        Args:
            model_name: Model architecture name (e.g., 'unet', 'attention_unet')
            weights: Path to model weights (optional)
            device: Device to load model on
            **model_kwargs: Additional arguments for model initialization

        Returns:
            Initialized DerainingModel

        Example:
            >>> model = DerainingModel.from_pretrained(
            ...     'attention_unet',
            ...     weights='checkpoints/best_model.pth'
            ... )
        """
        # Create model
        model = get_model(model_name, **model_kwargs)

        # Load weights if provided
        if weights is not None:
            weights = Path(weights)
            if weights.exists():
                checkpoint = torch.load(weights, map_location=device)

                # Handle different checkpoint formats
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
            else:
                raise FileNotFoundError(f"Weights file not found: {weights}")

        return cls(model=model, device=device)

    @torch.no_grad()
    def process(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """Process a single image.

        Args:
            image: Input image (filepath, numpy array, or PIL Image)
            output_path: Optional path to save output

        Returns:
            Derained image as numpy array (H, W, 3) in range [0, 255]

        Example:
            >>> model = DerainingModel.from_pretrained('unet')
            >>> clean = model.process('rainy.png', output_path='derained.png')
        """
        # Load image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
            image = np.array(image)
        elif isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))

        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Convert to tensor
        image_tensor = numpy_to_tensor(image, device=self.device)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        # Pad to multiple of 32 (for U-Net compatibility)
        padded, pad_vals = pad_to_multiple(image_tensor, multiple=32)

        # Forward pass
        output = self.model(padded)

        # Unpad
        left, right, top, bottom = pad_vals
        if right > 0:
            output = output[:, :, top : -bottom if bottom > 0 else None, left:-right]
        else:
            output = output[:, :, top : -bottom if bottom > 0 else None, left:]

        # Convert to numpy
        output_np = tensor_to_numpy(output[0])  # Remove batch dimension

        # Clip and convert to uint8
        output_np = np.clip(output_np, 0, 1)
        output_uint8 = (output_np * 255).astype(np.uint8)

        # Save if requested
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(output_uint8).save(output_path)

        return output_uint8

    @torch.no_grad()
    def process_batch(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Process a batch of images (tensor format).

        Args:
            images: Batch of images (B, C, H, W) in range [0, 1]

        Returns:
            Derained batch (B, C, H, W)

        Example:
            >>> images = torch.randn(4, 3, 256, 256).clamp(0, 1)
            >>> clean_images = model.process_batch(images)
        """
        images = images.to(self.device)

        # Pad if needed
        padded, pad_vals = pad_to_multiple(images, multiple=32)

        # Forward pass
        output = self.model(padded)

        # Unpad
        left, right, top, bottom = pad_vals
        if right > 0:
            output = output[:, :, top : -bottom if bottom > 0 else None, left:-right]
        else:
            output = output[:, :, top : -bottom if bottom > 0 else None, left:]

        return output.clamp(0, 1)

    def to(self, device: str) -> "DerainingModel":
        """Move model to device.

        Args:
            device: Target device

        Returns:
            Self for chaining
        """
        self.model = self.model.to(device)
        self.device = device
        return self

    def eval(self) -> "DerainingModel":
        """Set model to evaluation mode.

        Returns:
            Self for chaining
        """
        self.model.eval()
        return self

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DerainingModel(\n"
            f"  model={self.model.__class__.__name__},\n"
            f"  device={self.device},\n"
            f"  params={sum(p.numel() for p in self.model.parameters()):,}\n"
            f")"
        )


__all__ = ["DerainingModel"]
