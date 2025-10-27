"""Common building blocks for neural network architectures.

Provides reusable components like convolution blocks, attention modules,
and encoder/decoder blocks.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block with normalization and activation.

    Standard building block: Conv -> BatchNorm -> Activation

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolution kernel. Default: 3
        stride: Stride of convolution. Default: 1
        padding: Padding for convolution. Default: 1
        use_batchnorm: Whether to use batch normalization. Default: True
        activation: Activation function ('relu', 'leaky_relu', 'gelu'). Default: 'relu'

    Example:
        >>> block = ConvBlock(64, 128, kernel_size=3)
        >>> x = torch.randn(4, 64, 256, 256)
        >>> y = block(x)  # (4, 128, 256, 256)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batchnorm: bool = True,
        activation: str = "relu",
    ) -> None:
        """Initialize convolution block."""
        super().__init__()

        layers: List[nn.Module] = []

        # Convolution
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not use_batchnorm,
            )
        )

        # Batch normalization
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        # Activation
        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "leaky_relu":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == "gelu":
            layers.append(nn.GELU())
        elif activation == "none":
            pass
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        result: torch.Tensor = self.block(x)
        return result


class DoubleConv(nn.Module):
    """Double convolution block: Conv -> Conv.

    Standard U-Net building block with two consecutive convolutions.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        mid_channels: Number of intermediate channels. If None, uses out_channels.
        use_batchnorm: Whether to use batch normalization. Default: True
        activation: Activation function. Default: 'relu'

    Example:
        >>> block = DoubleConv(64, 128)
        >>> x = torch.randn(4, 64, 256, 256)
        >>> y = block(x)  # (4, 128, 256, 256)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        use_batchnorm: bool = True,
        activation: str = "relu",
    ) -> None:
        """Initialize double convolution block."""
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            ConvBlock(
                in_channels,
                mid_channels,
                use_batchnorm=use_batchnorm,
                activation=activation,
            ),
            ConvBlock(
                mid_channels,
                out_channels,
                use_batchnorm=use_batchnorm,
                activation=activation,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        result: torch.Tensor = self.double_conv(x)
        return result


class DownBlock(nn.Module):
    """Downsampling block for encoder.

    MaxPool -> DoubleConv

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        use_batchnorm: Whether to use batch normalization. Default: True
        activation: Activation function. Default: 'relu'

    Example:
        >>> block = DownBlock(64, 128)
        >>> x = torch.randn(4, 64, 256, 256)
        >>> y = block(x)  # (4, 128, 128, 128)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
        activation: str = "relu",
    ) -> None:
        """Initialize downsampling block."""
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(
                in_channels,
                out_channels,
                use_batchnorm=use_batchnorm,
                activation=activation,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        result: torch.Tensor = self.maxpool_conv(x)
        return result


class UpBlock(nn.Module):
    """Upsampling block for decoder.

    Upsample -> Conv -> Concat -> DoubleConv

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        use_transpose_conv: Use transposed convolution instead of upsampling. Default: True
        use_batchnorm: Whether to use batch normalization. Default: True
        activation: Activation function. Default: 'relu'

    Example:
        >>> block = UpBlock(256, 128)
        >>> x = torch.randn(4, 256, 64, 64)
        >>> skip = torch.randn(4, 128, 128, 128)
        >>> y = block(x, skip)  # (4, 128, 128, 128)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_transpose_conv: bool = True,
        use_batchnorm: bool = True,
        activation: str = "relu",
    ) -> None:
        """Initialize upsampling block."""
        super().__init__()

        # Upsampling
        if use_transpose_conv:
            self.up: nn.Module = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                ConvBlock(in_channels, out_channels, kernel_size=1, padding=0),
            )

        # Convolution after concatenation
        self.conv = DoubleConv(
            out_channels * 2,  # because of concatenation
            out_channels,
            use_batchnorm=use_batchnorm,
            activation=activation,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection.

        Args:
            x: Input tensor from previous layer
            skip: Skip connection tensor from encoder

        Returns:
            Output tensor after upsampling and concatenation
        """
        x = self.up(x)

        # Handle size mismatch between x and skip
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)

        if diff_h > 0 or diff_w > 0:
            x = F.pad(
                x,
                [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2],
            )

        # Concatenate
        x = torch.cat([skip, x], dim=1)

        result: torch.Tensor = self.conv(x)
        return result


class AttentionGate(nn.Module):
    """Attention gate for highlighting relevant features.

    Uses attention mechanism to focus on important spatial regions
    before concatenation with encoder features.

    Args:
        gate_channels: Number of channels in gating signal
        skip_channels: Number of channels in skip connection
        inter_channels: Number of intermediate channels. If None, uses skip_channels // 2

    Reference:
        Oktay et al. "Attention U-Net: Learning Where to Look for the Pancreas."
        MIDL 2018.

    Example:
        >>> attn = AttentionGate(gate_channels=256, skip_channels=128)
        >>> g = torch.randn(4, 256, 32, 32)  # Gating signal
        >>> x = torch.randn(4, 128, 64, 64)  # Skip connection
        >>> out = attn(g, x)  # (4, 128, 64, 64)
    """

    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        inter_channels: Optional[int] = None,
    ) -> None:
        """Initialize attention gate."""
        super().__init__()

        if inter_channels is None:
            inter_channels = skip_channels // 2

        # Gating signal transform
        self.W_g = nn.Sequential(
            nn.Conv2d(
                gate_channels,
                inter_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm2d(inter_channels),
        )

        # Skip connection transform
        self.W_x = nn.Sequential(
            nn.Conv2d(
                skip_channels,
                inter_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm2d(inter_channels),
        )

        # Attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

        # Store latest attention map
        self.latest_psi: Optional[torch.Tensor] = None

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            g: Gating signal from coarser scale
            x: Skip connection from encoder

        Returns:
            Attention-weighted features
        """
        # Transform gating signal
        g1 = self.W_g(g)

        # Transform skip connection
        x1 = self.W_x(x)

        # Upsample gating signal if necessary
        if g1.size(2) != x1.size(2) or g1.size(3) != x1.size(3):
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=True)

        # Attention coefficients
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        # Keep for visualization
        self.latest_psi = psi.detach().cpu()

        # Apply attention
        result: torch.Tensor = x * psi

        return result


__all__ = [
    "ConvBlock",
    "DoubleConv",
    "DownBlock",
    "UpBlock",
    "AttentionGate",
]
