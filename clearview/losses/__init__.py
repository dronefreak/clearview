"""Loss functions for image deraining.

This module provides various loss functions optimized for image restoration tasks,
including pixel-wise losses, structural similarity losses, and perceptual losses.
"""

from clearview.losses.base import BaseLoss
from clearview.losses.combined import CombinedLoss
from clearview.losses.edge import EdgeLoss, SobelEdgeLoss
from clearview.losses.perceptual import PerceptualLoss, VGGPerceptualLoss
from clearview.losses.pixel import L1Loss, L2Loss, MAELoss, MSELoss
from clearview.losses.structural import MultiScaleSSIMLoss, SSIMLoss

__all__ = [
    # Base
    "BaseLoss",
    # Pixel losses
    "L1Loss",
    "L2Loss",
    "MSELoss",
    "MAELoss",
    # Structural losses
    "SSIMLoss",
    "MultiScaleSSIMLoss",
    # Edge losses
    "EdgeLoss",
    "SobelEdgeLoss",
    # Perceptual losses
    "PerceptualLoss",
    "VGGPerceptualLoss",
    # Combined losses
    "CombinedLoss",
]
