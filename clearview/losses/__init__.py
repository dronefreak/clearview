"""Loss functions for image deraining.

This module provides various loss functions optimized for image restoration tasks,
including pixel-wise losses, structural similarity losses, and perceptual losses.
"""

from clearview.losses.adversarial import AdversarialLoss, GANLoss, PatchDiscriminator
from clearview.losses.base import BaseLoss
from clearview.losses.combined import CombinedLoss
from clearview.losses.edge import EdgeLoss, SobelEdgeLoss
from clearview.losses.frequency import FFTLoss, FocalFrequencyLoss, WaveletLoss
from clearview.losses.perceptual import DISTSLoss, PerceptualLoss, VGGPerceptualLoss
from clearview.losses.pixel import (
    ColorConsistencyLoss,
    L1Loss,
    L2Loss,
    MAELoss,
    MSELoss,
)
from clearview.losses.structural import MultiScaleSSIMLoss, SSIMLoss

__all__ = [
    # Base
    "BaseLoss",
    # Pixel losses
    "L1Loss",
    "L2Loss",
    "MSELoss",
    "MAELoss",
    "ColorConsistencyLoss",
    # Structural losses
    "SSIMLoss",
    "MultiScaleSSIMLoss",
    # Edge losses
    "EdgeLoss",
    "SobelEdgeLoss",
    # Frequency losses
    "FFTLoss",
    "FocalFrequencyLoss",
    "WaveletLoss",
    # Perceptual losses
    "PerceptualLoss",
    "VGGPerceptualLoss",
    "DISTSLoss",
    # Adversarial losses
    "PatchDiscriminator",
    "GANLoss",
    "AdversarialLoss",
    # Combined losses
    "CombinedLoss",
]
