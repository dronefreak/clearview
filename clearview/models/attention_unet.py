"""Attention U-Net architecture for image deraining.

U-Net with attention gates that learn to focus on relevant features.
"""

from typing import List, Optional, Any

import torch
import torch.nn as nn

from clearview.models.base import BaseModel
from clearview.models.blocks import DoubleConv, DownBlock, UpBlock, AttentionGate


class AttentionUNet(BaseModel):
    """Attention U-Net with attention gates.

    Enhanced U-Net that uses attention mechanisms to focus on relevant
    spatial regions. Particularly effective for complex restoration tasks.

    Args:
        in_channels: Number of input channels. Default: 3 (RGB)
        out_channels: Number of output channels. Default: 3 (RGB)
        features: List of feature dimensions for each level. Default: [64, 128, 256, 512]
        use_transpose_conv: Use transposed convolution for upsampling. Default: True
        use_batchnorm: Use batch normalization. Default: True
        activation: Activation function ('relu', 'leaky_relu'). Default: 'relu'

    Reference:
        Oktay et al. "Attention U-Net: Learning Where to Look for the Pancreas."
        MIDL 2018.

    Example:
        >>> model = AttentionUNet(in_channels=3, out_channels=3)
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
        """Initialize Attention U-Net."""
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

        # Attention gates
        self.attention_gates = nn.ModuleList()

        for i in reversed(range(len(features))):
            gate_ch = features[i] * 2 if i == len(features) - 1 else features[i + 1]
            skip_ch = features[i]

            self.attention_gates.append(
                AttentionGate(gate_channels=gate_ch, skip_channels=skip_ch)
            )

        # Decoder (upsampling path)
        self.decoder = nn.ModuleList()

        for i in reversed(range(len(features))):
            in_ch = features[i] * 2 if i == len(features) - 1 else features[i + 1]
            self.decoder.append(
                UpBlock(
                    in_ch * 2,  # *2 because of skip connection
                    features[i],
                    use_transpose_conv=use_transpose_conv,
                    use_batchnorm=use_batchnorm,
                    activation=activation,
                )
            )

        # Output layer
        self.output = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        # Encoder with skip connections
        skip_connections = []

        for i, down in enumerate(self.encoder):
            x = down(x)
            if i < len(self.encoder) - 1:
                skip_connections.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse skip connections for decoder
        skip_connections = skip_connections[::-1]

        # Decoder with attention-weighted skip connections
        for i, (up, attn_gate) in enumerate(zip(self.decoder, self.attention_gates)):
            # Apply attention gate to skip connection
            skip_attended = attn_gate(x, skip_connections[i])

            # Upsample and concatenate
            x = up(x, skip_attended)

        # Output
        result: torch.Tensor = self.output(x)
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


class AttentionUNetSmall(AttentionUNet):
    """Smaller Attention U-Net variant.

    Uses fewer features: [32, 64, 128, 256]

    Example:
        >>> model = AttentionUNetSmall()
        >>> print(f"Params: {model.get_num_params():,}")
    """

    def __init__(
        self, in_channels: int = 3, out_channels: int = 3, **kwargs: Any
    ) -> None:
        """Initialize small Attention U-Net."""
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            features=[32, 64, 128, 256],
            **kwargs,
        )


class AttentionUNetLarge(AttentionUNet):
    """Larger Attention U-Net variant.

    Uses more features: [64, 128, 256, 512, 1024]

    Example:
        >>> model = AttentionUNetLarge()
        >>> print(f"Params: {model.get_num_params():,}")
    """

    def __init__(
        self, in_channels: int = 3, out_channels: int = 3, **kwargs: Any
    ) -> None:
        """Initialize large Attention U-Net."""
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            features=[64, 128, 256, 512, 1024],
            **kwargs,
        )


__all__ = [
    "AttentionUNet",
    "AttentionUNetSmall",
    "AttentionUNetLarge",
]
