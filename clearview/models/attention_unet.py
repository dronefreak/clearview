"""Attention U-Net architecture for image deraining.

U-Net with attention gates that learn to focus on relevant features.
"""

from typing import Any, List, Optional

import torch
import torch.nn as nn

from clearview.models.base import BaseModel
from clearview.models.blocks import AttentionGate, DoubleConv, DownBlock, UpBlock


class AttentionUNet(BaseModel):
    """Attention U-Net with attention gates."""

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

        # We'll create one attention gate + one UpBlock per skip (deepest -> shallowest).
        num_skips = len(features) - 1  # number of encoder outputs we will use as skips

        self.attention_gates = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Gating channels start at bottleneck output channels
        cur_channels = features[-1] * 2  # e.g., 512 * 2 = 1024

        # Build for each skip level, deepest -> shallowest
        for j in reversed(range(num_skips)):
            skip_ch = features[j]  # skip channels at this level (j runs  num_skips-1 .. 0)
            # Attention gate: gating is cur_channels, skip is skip_ch
            self.attention_gates.append(AttentionGate(gate_channels=cur_channels, skip_channels=skip_ch))
            # UpBlock upsamples cur_channels -> features[j+?], but we want output = features[j]
            # For U-Net, after upsampling we produce features[j] (matching encoder level j)
            self.decoder.append(
                UpBlock(
                    in_channels=cur_channels,
                    out_channels=features[j],
                    use_transpose_conv=use_transpose_conv,
                    use_batchnorm=use_batchnorm,
                    activation=activation,
                )
            )
            # After this up-step, the decoder output channels become features[j]
            cur_channels = features[j]

        # Note: attention_gates and decoder are both length `num_skips` and ordered deepest -> shallowest.

        # Output layer
        self.output = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention."""
        # Encoder with skip connections: collect encoder outputs EXCEPT the last one (which goes to bottleneck)
        skip_connections = []

        for i, down in enumerate(self.encoder):
            x = down(x)
            # Collect all encoder outputs except the deepest (last) which feeds the bottleneck
            if i < len(self.encoder) - 1:
                skip_connections.append(x)

        # Bottleneck consumes the last encoder output
        x = self.bottleneck(x)

        # Reverse skip connections so that index 0 is the deepest usable skip (matches decoder order)
        skip_connections = skip_connections[::-1]

        # Decoder with attention-weighted skip connections
        # Both self.decoder and self.attention_gates are ordered deepest -> shallowest
        for i, (up, attn_gate) in enumerate(zip(self.decoder, self.attention_gates)):
            skip = skip_connections[i]  # this skip now has the same spatial size expected by up(x) after upsampling
            skip_attended = attn_gate(x, skip)
            x = up(x, skip_attended)

        # Output
        result: torch.Tensor = self.output(x)
        return result

    def get_config(self) -> dict:
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
    """Smaller Attention U-Net variant."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, **kwargs: Any) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            features=[32, 64, 128, 256],
            **kwargs,
        )


class AttentionUNetLarge(AttentionUNet):
    """Larger Attention U-Net variant."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, **kwargs: Any) -> None:
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
