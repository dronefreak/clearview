"""Perceptual loss functions using pretrained networks.

Implements perceptual losses that compare high-level features extracted
from pretrained networks, better capturing semantic similarity.
"""

from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from clearview.losses.base import BaseLoss


class VGGPerceptualLoss(BaseLoss):
    """Perceptual loss using VGG16 features.

    Compares feature representations from a pretrained VGG16 network rather
    than raw pixels. This better captures perceptual similarity and helps
    generate more natural-looking textures.

    Args:
        layers: List of VGG layer names to use for loss computation.
            Default: ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        layer_weights: Weights for each layer. If None, uses equal weights.
        normalize: Whether to normalize input images. Default: True
        reduction: Reduction method. Default: 'mean'
        weight: Loss weight. Default: 1.0

    Reference:
        Johnson et al. "Perceptual Losses for Real-Time Style Transfer
        and Super-Resolution." ECCV 2016.

    Example:
        >>> loss_fn = VGGPerceptualLoss()
        >>> pred = torch.randn(4, 3, 256, 256)
        >>> target = torch.randn(4, 3, 256, 256)
        >>> loss = loss_fn(pred, target)
    """

    # VGG layer name to index mapping
    LAYER_MAP = {
        "relu1_1": 1,
        "relu1_2": 3,
        "relu2_1": 6,
        "relu2_2": 8,
        "relu3_1": 11,
        "relu3_2": 13,
        "relu3_3": 15,
        "relu4_1": 18,
        "relu4_2": 20,
        "relu4_3": 22,
        "relu5_1": 25,
        "relu5_2": 27,
        "relu5_3": 29,
    }

    def __init__(
        self,
        layers: Optional[List[str]] = None,
        layer_weights: Optional[List[float]] = None,
        normalize: bool = True,
        reduction: str = "mean",
        weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize VGG perceptual loss.

        Args:
            layers: VGG layers to use
            layer_weights: Weights for each layer
            normalize: Normalize inputs with ImageNet stats
            reduction: Reduction method
            weight: Loss weight
            **kwargs: Additional arguments
        """
        super().__init__(reduction=reduction, weight=weight, **kwargs)

        if layers is None:
            layers = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]

        if layer_weights is None:
            layer_weights = [1.0] * len(layers)

        if len(layers) != len(layer_weights):
            raise ValueError(
                f"Number of layers ({len(layers)}) must match "
                f"number of weights ({len(layer_weights)})"
            )

        self.layers = layers
        self.layer_weights = layer_weights
        self.normalize = normalize

        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=True).features
        vgg.eval()

        # Freeze VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False

        # Extract relevant layers
        self.feature_extractors = nn.ModuleList()
        prev_idx = 0

        for layer_name in layers:
            if layer_name not in self.LAYER_MAP:
                raise ValueError(
                    f"Invalid layer name: {layer_name}. "
                    f"Choose from {list(self.LAYER_MAP.keys())}"
                )

            layer_idx = self.LAYER_MAP[layer_name]
            self.feature_extractors.append(
                nn.Sequential(*list(vgg.children())[prev_idx : layer_idx + 1])
            )
            prev_idx = layer_idx + 1

        # ImageNet normalization values
        if normalize:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)

    def _normalize(self, img: torch.Tensor) -> torch.Tensor:
        """Normalize image using ImageNet statistics.

        Args:
            img: Input image in range [0, 1]

        Returns:
            Normalized image
        """
        if self.normalize:
            mean_tensor: torch.Tensor = self.mean
            std_tensor: torch.Tensor = self.std
            return (img - mean_tensor) / std_tensor
        return img

    def _extract_features(self, img: torch.Tensor) -> List[torch.Tensor]:
        """Extract features from VGG layers.

        Args:
            img: Input image (B, 3, H, W)

        Returns:
            List of feature tensors from each layer
        """
        img = self._normalize(img)
        features = []

        x = img
        for extractor in self.feature_extractors:
            x = extractor(x)
            features.append(x)

        return features

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss.

        Args:
            pred: Predicted image (B, 3, H, W) in range [0, 1]
            target: Target image (B, 3, H, W) in range [0, 1]

        Returns:
            Perceptual loss value
        """
        # Extract features
        pred_features = self._extract_features(pred)
        target_features = self._extract_features(target)

        # Compute weighted loss across layers
        loss_tensor = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        for pred_feat, target_feat, weight in zip(
            pred_features, target_features, self.layer_weights
        ):
            loss_tensor = loss_tensor + weight * F.mse_loss(
                pred_feat, target_feat, reduction=self.reduction
            )

        return self.apply_weight(loss_tensor)

    def get_config(self) -> dict:
        """Get configuration dictionary."""
        config = super().get_config()
        config.update(
            {
                "layers": self.layers,
                "layer_weights": self.layer_weights,
                "normalize": self.normalize,
            }
        )
        return config


class PerceptualLoss(VGGPerceptualLoss):
    """Alias for VGGPerceptualLoss for backward compatibility."""

    pass
