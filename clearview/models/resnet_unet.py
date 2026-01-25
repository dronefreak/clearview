"""ResNet-based U-Net for image deraining.

Modular architecture supporting ResNet18/34/50/101 backbones with skip connections.
Uses bilinear upsampling + conv instead of ConvTranspose2d to avoid checkerboard artifacts.
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DecoderBlock(nn.Module):
    """Decoder block with upsampling + conv (no ConvTranspose2d to avoid artifacts)."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
    ):
        """Initialize decoder block.

        Args:
            in_channels: Channels from previous decoder layer
            skip_channels: Channels from skip connection
            out_channels: Output channels
            use_batchnorm: Whether to use batch normalization
        """
        super().__init__()

        # Bilinear upsampling (no learnable params, no artifacts)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Convolution after concatenation with skip connection
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=not use_batchnorm,
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)

        # Second convolution for refinement
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input from previous decoder layer (B, in_channels, H, W)
            skip: Skip connection from encoder (B, skip_channels, H*2, W*2)

        Returns:
            Output tensor (B, out_channels, H*2, W*2)
        """
        # Upsample
        x = self.upsample(x)

        # Handle size mismatch (if skip is slightly different size)
        if x.shape != skip.shape:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=True
            )

        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)

        # Two convolutions
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        return x


class ResNetUNet(nn.Module):
    """U-Net with ResNet encoder.

    Supports ResNet18/34/50/101/152 backbones with pretrained ImageNet weights.
    Uses skip connections from encoder to decoder for detail preservation.

    Example:
        >>> model = ResNetUNet(backbone='resnet34', pretrained=True)
        >>> x = torch.randn(2, 3, 256, 256)
        >>> out = model(x)  # (2, 3, 256, 256)
    """

    # Channel dimensions for different ResNet architectures
    RESNET_CHANNELS = {
        "resnet18": [64, 64, 128, 256, 512],
        "resnet34": [64, 64, 128, 256, 512],
        "resnet50": [64, 256, 512, 1024, 2048],
        "resnet101": [64, 256, 512, 1024, 2048],
        "resnet152": [64, 256, 512, 1024, 2048],
    }

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        backbone: Literal[
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
        ] = "resnet34",
        pretrained: bool = True,
        use_batchnorm: bool = True,
    ):
        """Initialize ResNet U-Net.

        Args:
            in_channels: Number of input channels (default: 3 for RGB)
            out_channels: Number of output channels (default: 3 for RGB)
            backbone: ResNet backbone to use
            pretrained: Whether to use ImageNet pretrained weights
            use_batchnorm: Whether to use batch normalization in decoder
        """
        super().__init__()

        # ImageNet normalization parameters
        self.register_buffer(
            "imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.backbone_name = backbone

        # Get encoder channels for this backbone
        if backbone not in self.RESNET_CHANNELS:
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                f"Choose from {list(self.RESNET_CHANNELS.keys())}"
            )

        encoder_channels = self.RESNET_CHANNELS[backbone]

        # Create a dict of weights from ImageNet
        weights_dict = {
            "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
            "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
            "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
            "resnet101": models.ResNet101_Weights.IMAGENET1K_V1,
            "resnet152": models.ResNet152_Weights.IMAGENET1K_V1,
        }

        # Load pretrained ResNet encoder
        resnet = getattr(models, backbone)(
            weights=weights_dict[backbone] if pretrained else None
        )

        # ====================================================================
        # ENCODER (from ResNet)
        # ====================================================================

        # Initial convolution (conv1 + bn1 + relu + maxpool)
        self.encoder0 = nn.Sequential(
            resnet.conv1,  # 7x7 conv, stride 2 → H/2, W/2
            resnet.bn1,
            resnet.relu,
        )
        self.encoder0_pool = resnet.maxpool  # 3x3 maxpool, stride 2 → H/4, W/4

        # ResNet stages (layer1, layer2, layer3, layer4)
        self.encoder1 = resnet.layer1  # stride 1 → H/4, W/4
        self.encoder2 = resnet.layer2  # stride 2 → H/8, W/8
        self.encoder3 = resnet.layer3  # stride 2 → H/16, W/16
        self.encoder4 = resnet.layer4  # stride 2 → H/32, W/32

        # ====================================================================
        # DECODER (with skip connections)
        # ====================================================================

        # Decoder blocks (upsample + skip connection + convs)
        # decoder4: H/32 → H/16, skip from encoder3
        self.decoder4 = DecoderBlock(
            in_channels=encoder_channels[4],  # From encoder4
            skip_channels=encoder_channels[3],  # From encoder3
            out_channels=encoder_channels[3],
            use_batchnorm=use_batchnorm,
        )

        # decoder3: H/16 → H/8, skip from encoder2
        self.decoder3 = DecoderBlock(
            in_channels=encoder_channels[3],
            skip_channels=encoder_channels[2],
            out_channels=encoder_channels[2],
            use_batchnorm=use_batchnorm,
        )

        # decoder2: H/8 → H/4, skip from encoder1
        self.decoder2 = DecoderBlock(
            in_channels=encoder_channels[2],
            skip_channels=encoder_channels[1],
            out_channels=encoder_channels[1],
            use_batchnorm=use_batchnorm,
        )

        # decoder1: H/4 → H/2, skip from encoder0 (before pooling)
        self.decoder1 = DecoderBlock(
            in_channels=encoder_channels[1],
            skip_channels=encoder_channels[0],
            out_channels=encoder_channels[0],
            use_batchnorm=use_batchnorm,
        )

        # Final upsampling: H/2 → H (no skip connection here)
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(encoder_channels[0], 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        # Output layer
        self.output = nn.Conv2d(32, out_channels, kernel_size=1)
        self.final_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, in_channels, H, W)

        Returns:
            Output tensor (B, out_channels, H, W) in range [0, 1]
        """
        # ====================================================================
        # ENCODER
        # ====================================================================

        # Normalize from [0,1] to ImageNet distribution
        x = (x - self.imagenet_mean) / self.imagenet_std

        # Initial conv + pool
        e0 = self.encoder0(x)  # H/2, W/2
        e0_pooled = self.encoder0_pool(e0)  # H/4, W/4

        # ResNet stages
        e1 = self.encoder1(e0_pooled)  # H/4, W/4
        e2 = self.encoder2(e1)  # H/8, W/8
        e3 = self.encoder3(e2)  # H/16, W/16
        e4 = self.encoder4(e3)  # H/32, W/32 (bottleneck)

        # ====================================================================
        # DECODER (with skip connections)
        # ====================================================================

        d4 = self.decoder4(e4, e3)  # H/16, W/16
        d3 = self.decoder3(d4, e2)  # H/8, W/8
        d2 = self.decoder2(d3, e1)  # H/4, W/4
        d1 = self.decoder1(d2, e0)  # H/2, W/2

        # Final upsampling and output
        d0 = self.final_upsample(d1)  # H, W
        out = self.output(d0)  # H, W
        out = self.final_activation(out)

        ## Debug shapes
        # print(f"e0 (before pool): {e0.shape}")
        # print(f"e0_pooled: {e0_pooled.shape}")
        # print(f"e1: {e1.shape}")
        # print(f"e2: {e2.shape}")
        # print(f"e3: {e3.shape}")
        # print(f"e4 (bottleneck): {e4.shape}")
        # print(f"d4: {d4.shape}")

        return out

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_model_size_mb(self) -> float:
        """Get model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return float((param_size + buffer_size) / (1024**2))

    def freeze_encoder(self) -> None:
        """Freeze encoder weights (useful for fine-tuning)."""
        for param in self.encoder0.parameters():
            param.requires_grad = False
        for param in self.encoder1.parameters():
            param.requires_grad = False
        for param in self.encoder2.parameters():
            param.requires_grad = False
        for param in self.encoder3.parameters():
            param.requires_grad = False
        for param in self.encoder4.parameters():
            param.requires_grad = False
        print("Encoder frozen!.")

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder weights."""
        for param in self.encoder0.parameters():
            param.requires_grad = True
        for param in self.encoder1.parameters():
            param.requires_grad = True
        for param in self.encoder2.parameters():
            param.requires_grad = True
        for param in self.encoder3.parameters():
            param.requires_grad = True
        for param in self.encoder4.parameters():
            param.requires_grad = True
        print("Encoder unfrozen!.")


# ============================================================================
# Factory function for easy model creation
# ============================================================================


def create_resnet_unet(
    backbone: str = "resnet34",
    pretrained: bool = True,
    in_channels: int = 3,
    out_channels: int = 3,
) -> ResNetUNet:
    """Create a ResNet U-Net model.

    Args:
        backbone: ResNet backbone ('resnet18', 'resnet34', 'resnet50', etc.)
        pretrained: Use ImageNet pretrained weights
        in_channels: Number of input channels
        out_channels: Number of output channels

    Returns:
        ResNetUNet model

    Example:
        >>> model = create_resnet_unet('resnet34', pretrained=True)
        >>> print(f"Parameters: {model.get_num_params():,}")
        >>> print(f"Size: {model.get_model_size_mb():.2f} MB")
    """
    return ResNetUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        backbone=backbone,  # type: ignore[arg-type]
        pretrained=pretrained,
    )


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ResNet U-Net Architecture Test")
    print("=" * 80)

    # Test all backbones
    backbones = ["resnet18", "resnet34", "resnet50"]

    for backbone in backbones:
        print(f"\n{backbone.upper()}:")
        print("-" * 40)

        model = create_resnet_unet(backbone=backbone, pretrained=False)
        if backbone == "resnet18":
            weights = "experiments/resnet18_unet_rain1400/checkpoints/best_val_psnr.pth"
            checkpoint = torch.load(weights, map_location="cpu")
            # Handle different checkpoint formats
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                print("Loaded weights from checkpoint.")
            else:
                model.load_state_dict(checkpoint)
                print("Loaded weights from raw checkpoint.")

        # Test forward pass
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            out = model(x)

        print(f"Input shape:  {x.shape}")
        print(f"Output shape: {out.shape}")
        print(f"Parameters:   {model.get_num_params():,}")
        print(f"Size:         {model.get_model_size_mb():.2f} MB")
        print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")

        # Verify output is bounded [0, 1]
        assert out.min() >= 0 and out.max() <= 1, "Output not in [0, 1]!"
        assert out.shape == x.shape, "Output shape mismatch!"

        print("✅ Test passed!")

    print("\n" + "=" * 80)
    print("All tests passed! Model is ready to use.")
    print("=" * 80)
