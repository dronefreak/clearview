"""U-Net architecture for image deraining.

Classic encoder-decoder architecture with skip connections.
"""

from typing import Any, List, Optional

import torch
import torch.nn as nn

from clearview.models.base import BaseModel
from clearview.models.blocks import DoubleConv, DownBlock, UpBlock


class UNet(BaseModel):
    """U-Net architecture for image restoration.

    Classic encoder-decoder with skip connections. Proven effective for
    image-to-image translation tasks including deraining.

    Args:
        in_channels: Number of input channels. Default: 3 (RGB)
        out_channels: Number of output channels. Default: 3 (RGB)
        features: List of feature dimensions for each level. Default: [64, 128, 256, 512]
        use_transpose_conv: Use transposed convolution for upsampling. Default: True
        use_batchnorm: Use batch normalization. Default: True
        activation: Activation function ('relu', 'leaky_relu'). Default: 'relu'

    Reference:
        Ronneberger et al. "U-Net: Convolutional Networks for Biomedical
        Image Segmentation." MICCAI 2015.

    Example:
        >>> model = UNet(in_channels=3, out_channels=3)
        >>> x = torch.randn(4, 3, 256, 256)
        >>> y = model(x)  # (4, 3, 256, 256)
        >>> print(f"Params: {model.get_num_params():,}")
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: Optional[List[int]] = None,
        use_transpose_conv: bool = True,
        use_batchnorm: bool = True,
        activation: str = "relu",
    ) -> None:
        """Initialize U-Net."""
        super().__init__(in_channels=in_channels, out_channels=out_channels)

        if features is None:
            features = [64, 128, 256, 512]

        self.features = features
        self.use_transpose_conv = use_transpose_conv
        self.use_batchnorm = use_batchnorm
        self.activation = activation

        # Encoder (downsampling path)
        self.encoder = nn.ModuleList()

        # Initial convolution
        self.encoder.append(
            DoubleConv(
                in_channels,
                features[0],
                use_batchnorm=use_batchnorm,
                activation=activation,
            )
        )

        # Downsampling blocks
        for i in range(len(features) - 1):
            self.encoder.append(
                DownBlock(
                    features[i],
                    features[i + 1],
                    use_batchnorm=use_batchnorm,
                    activation=activation,
                )
            )

        # Bottleneck
        self.bottleneck = DoubleConv(
            features[-1],
            features[-1] * 2,
            use_batchnorm=use_batchnorm,
            activation=activation,
        )

        # Decoder (upsampling path)
        self.decoder = nn.ModuleList()

        for i in reversed(range(len(features) - 1)):
            in_ch = features[i + 1] * 2 if i == len(features) - 2 else features[i + 1]
            self.decoder.append(
                UpBlock(
                    in_ch,
                    features[i],
                    use_transpose_conv=use_transpose_conv,
                    use_batchnorm=use_batchnorm,
                    activation=activation,
                )
            )

        # Output layer
        self.output = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.final_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        # Encoder with skip connections
        skip_connections = []

        for i, down in enumerate(self.encoder):
            x = down(x)
            if i < len(self.encoder) - 1:  # Don't save last encoder output
                skip_connections.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse skip connections for decoder
        skip_connections = skip_connections[::-1]

        # Decoder with skip connections
        for i, up in enumerate(self.decoder):
            x = up(x, skip_connections[i])

        # Output
        result: torch.Tensor = self.final_activation(self.output(x))
        return result

    def get_config(self) -> dict:
        """Get model configuration."""
        config = super().get_config()
        config.update(
            {
                "features": self.features,
                "use_transpose_conv": self.use_transpose_conv,
                "use_batchnorm": self.use_batchnorm,
                "activation": self.activation,
            }
        )
        return config


class UNetSmall(UNet):
    """Smaller U-Net variant for faster inference.

    Uses fewer features at each level: [32, 64, 128, 256]

    Example:
        >>> model = UNetSmall()
        >>> print(f"Params: {model.get_num_params():,}")
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: Optional[List[int]] = [32, 64, 128, 256],
        **kwargs: Any,
    ) -> None:
        """Initialize small U-Net."""
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            **kwargs,
        )


class UNetLarge(UNet):
    """Larger U-Net variant for higher quality.

    Uses more features at each level: [64, 128, 256, 512, 1024]

    Example:
        >>> model = UNetLarge()
        >>> print(f"Params: {model.get_num_params():,}")
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: Optional[List[int]] = [64, 128, 256, 512, 1024],
        **kwargs: Any,
    ) -> None:
        """Initialize large U-Net."""
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            **kwargs,
        )


__all__ = [
    "UNet",
    "UNetSmall",
    "UNetLarge",
]
