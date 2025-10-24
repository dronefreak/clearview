"""Combined loss functions.

Provides flexible combination of multiple loss components for comprehensive
image restoration objectives.
"""

from typing import Any, Dict, Optional, cast, Type

import torch
import torch.nn as nn

from clearview.losses.base import BaseLoss


class CombinedLoss(BaseLoss):
    """Combined loss with multiple components.

    Combines pixel-wise, structural, edge, and perceptual losses with
    configurable weights for comprehensive image restoration.

    Args:
        losses: Dictionary mapping loss names to loss instances or configs
        weights: Dictionary mapping loss names to weights
        reduction: Reduction method. Default: 'mean'

    Example:
        >>> # Simple combination
        >>> loss_fn = CombinedLoss(
        ...     losses={
        ...         'l1': L1Loss(),
        ...         'ssim': SSIMLoss(),
        ...         'edge': SobelEdgeLoss(),
        ...     },
        ...     weights={'l1': 1.0, 'ssim': 1.0, 'edge': 0.5}
        ... )
        >>>
        >>> # Using string configs (for easy serialization)
        >>> loss_fn = CombinedLoss.from_config({
        ...     'l1': {'weight': 1.0},
        ...     'l2': {'weight': 1.0},
        ...     'ssim': {'weight': 1.0},
        ...     'edge': {'weight': 0.5},
        ... })
    """

    def __init__(
        self,
        losses: Dict[str, nn.Module],
        weights: Optional[Dict[str, float]] = None,
        reduction: str = "mean",
        **kwargs: Any,
    ) -> None:
        """Initialize combined loss.

        Args:
            losses: Dictionary of loss name to loss instance
            weights: Optional weights for each loss (uses 1.0 if not specified)
            reduction: Reduction method
            **kwargs: Additional arguments
        """
        super().__init__(reduction=reduction, weight=1.0, **kwargs)

        self.loss_components: nn.ModuleDict = nn.ModuleDict(losses)

        # Set weights
        if weights is None:
            weights = {name: 1.0 for name in losses.keys()}

        self.loss_weights: Dict[str, float] = {}
        for name in losses.keys():
            self.loss_weights[name] = weights.get(name, 1.0)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, return_dict: bool = False
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)
            return_dict: If True, returns dict with individual losses (stored in instance variable)

        Returns:
            Combined loss value
        """
        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        loss_dict: Dict[str, torch.Tensor] = {}

        for name, loss_fn in self.loss_components.items():
            weight = self.loss_weights[name]
            component_loss = loss_fn(pred, target)
            weighted_loss = weight * component_loss

            total_loss = total_loss + weighted_loss
            loss_dict[name] = component_loss.detach()

        if return_dict:
            loss_dict["total"] = total_loss.detach()
            self._last_loss_dict = loss_dict

        return total_loss

    def get_last_loss_dict(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get the loss dictionary from the last forward call with return_dict=True.

        Returns:
            Dictionary of losses or None if forward was not called with return_dict=True
        """
        return getattr(self, "_last_loss_dict", None)

    @classmethod
    def from_config(cls, config: Dict[str, Dict[str, Any]]) -> "CombinedLoss":
        """Create CombinedLoss from configuration dict.

        Args:
            config: Dict mapping loss names to their configs
                Example: {
                    'l1': {'weight': 1.0},
                    'ssim': {'weight': 1.0, 'window_size': 11},
                    'edge': {'weight': 0.5},
                }

        Returns:
            Configured CombinedLoss instance

        Raises:
            ValueError: If loss name is not recognized
        """
        from clearview.losses.edge import LaplacianEdgeLoss, SobelEdgeLoss
        from clearview.losses.perceptual import VGGPerceptualLoss
        from clearview.losses.pixel import CharbonnierLoss, L1Loss, L2Loss
        from clearview.losses.structural import MultiScaleSSIMLoss, SSIMLoss

        loss_registry: Dict[str, Type[Any]] = {
            "l1": L1Loss,
            "l2": L2Loss,
            "mse": L2Loss,
            "mae": L1Loss,
            "charbonnier": CharbonnierLoss,
            "ssim": SSIMLoss,
            "ms_ssim": MultiScaleSSIMLoss,
            "multi_scale_ssim": MultiScaleSSIMLoss,
            "edge": SobelEdgeLoss,
            "sobel": SobelEdgeLoss,
            "laplacian": LaplacianEdgeLoss,
            "perceptual": VGGPerceptualLoss,
            "vgg": VGGPerceptualLoss,
        }

        losses: Dict[str, nn.Module] = {}
        weights: Dict[str, float] = {}

        for name, loss_config in config.items():
            name_lower = name.lower()

            if name_lower not in loss_registry:
                raise ValueError(
                    f"Unknown loss type: {name}. "
                    f"Available losses: {list(loss_registry.keys())}"
                )

            # Extract weight
            loss_weight = loss_config.pop("weight", 1.0)
            weights[name] = loss_weight

            # Create loss instance
            loss_class = loss_registry[name_lower]
            losses[name] = loss_class(**loss_config)

        return cls(losses=losses, weights=weights)

    def get_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        for name, loss_fn in self.loss_components.items():
            lf = cast(BaseLoss, loss_fn)
            loss_config = lf.get_config() if hasattr(lf, "get_config") else {}
            loss_config["weight"] = float(self.loss_weights[name])
            config[name] = loss_config
        return config

    def __repr__(self) -> str:
        """String representation of combined loss."""
        components = [
            f"{name}(weight={self.loss_weights[name]})"
            for name in self.loss_components.keys()
        ]
        return f"CombinedLoss({', '.join(components)})"


# Predefined combinations matching common use cases
class L1L2SSIMLoss(CombinedLoss):
    """L1 + L2 + SSIM loss (matching your original TensorFlow implementation).

    Example:
        >>> loss_fn = L1L2SSIMLoss()
        >>> loss = loss_fn(pred, target)
    """

    def __init__(self, reduction: str = "mean", **kwargs: Any) -> None:
        """Initialize L1+L2+SSIM loss."""
        from clearview.losses.pixel import L1Loss, L2Loss
        from clearview.losses.structural import SSIMLoss

        losses: Dict[str, nn.Module] = {
            "l1": L1Loss(reduction=reduction),
            "l2": L2Loss(reduction=reduction),
            "ssim": SSIMLoss(reduction=reduction),
        }
        weights = {"l1": 1.0, "l2": 1.0, "ssim": 1.0}
        super().__init__(losses=losses, weights=weights, reduction=reduction, **kwargs)


class L1L2SSIMEdgeLoss(CombinedLoss):
    """L1 + L2 + SSIM + Edge loss (matching your TensorFlow implementation).

    Example:
        >>> loss_fn = L1L2SSIMEdgeLoss()
        >>> loss = loss_fn(pred, target)
    """

    def __init__(self, reduction: str = "mean", **kwargs: Any) -> None:
        """Initialize L1+L2+SSIM+Edge loss."""
        from clearview.losses.edge import SobelEdgeLoss
        from clearview.losses.pixel import L1Loss, L2Loss
        from clearview.losses.structural import SSIMLoss

        losses: Dict[str, nn.Module] = {
            "l1": L1Loss(reduction=reduction),
            "l2": L2Loss(reduction=reduction),
            "ssim": SSIMLoss(reduction=reduction),
            "edge": SobelEdgeLoss(reduction=reduction),
        }
        weights = {"l1": 1.0, "l2": 1.0, "ssim": 1.0, "edge": 1.0}
        super().__init__(losses=losses, weights=weights, reduction=reduction, **kwargs)


class L1L2MSSSIMEdgeLoss(CombinedLoss):
    """L1 + L2 + MS-SSIM + Edge loss (your TensorFlow multi-scale variant).

    Example:
        >>> loss_fn = L1L2MSSSIMEdgeLoss()
        >>> loss = loss_fn(pred, target)
    """

    def __init__(self, reduction: str = "mean", **kwargs: Any) -> None:
        """Initialize L1+L2+MS-SSIM+Edge loss."""
        from clearview.losses.edge import SobelEdgeLoss
        from clearview.losses.pixel import L1Loss, L2Loss
        from clearview.losses.structural import MultiScaleSSIMLoss

        losses: Dict[str, nn.Module] = {
            "l1": L1Loss(reduction=reduction),
            "l2": L2Loss(reduction=reduction),
            "ms_ssim": MultiScaleSSIMLoss(reduction=reduction),
            "edge": SobelEdgeLoss(reduction=reduction),
        }
        weights = {"l1": 1.0, "l2": 1.0, "ms_ssim": 1.0, "edge": 1.0}
        super().__init__(losses=losses, weights=weights, reduction=reduction, **kwargs)


class L1SSIMEdgePerceptualLoss(CombinedLoss):
    """Modern combination: L1 + SSIM + Edge + Perceptual.

    Recommended for high-quality results with natural textures.

    Example:
        >>> loss_fn = L1SSIMEdgePerceptualLoss()
        >>> loss = loss_fn(pred, target)
    """

    def __init__(self, reduction: str = "mean", **kwargs: Any) -> None:
        """Initialize L1+SSIM+Edge+Perceptual loss."""
        from clearview.losses.edge import SobelEdgeLoss
        from clearview.losses.perceptual import VGGPerceptualLoss
        from clearview.losses.pixel import L1Loss
        from clearview.losses.structural import SSIMLoss

        losses: Dict[str, nn.Module] = {
            "l1": L1Loss(reduction=reduction),
            "ssim": SSIMLoss(reduction=reduction),
            "edge": SobelEdgeLoss(reduction=reduction),
            "perceptual": VGGPerceptualLoss(reduction=reduction),
        }
        weights = {"l1": 1.0, "ssim": 1.0, "edge": 0.5, "perceptual": 0.1}
        super().__init__(losses=losses, weights=weights, reduction=reduction, **kwargs)
