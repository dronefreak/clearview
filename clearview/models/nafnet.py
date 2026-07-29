"""NAFNet: Nonlinear Activation Free Network for image restoration.

A simple, efficient baseline that removes nonlinear activations (ReLU/GELU/Sigmoid)
in favor of SimpleGate and a parameter-free channel attention mechanism, achieving
state-of-the-art restoration quality with lower computational cost.

Reference:
    Chen et al. "Simple Baselines for Image Restoration." ECCV 2022.
    https://github.com/megvii-research/NAFNet
"""

from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from clearview.models.base import BaseModel


class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm operating directly on (B, C, H, W) tensors.

    Normalizes across the channel dimension for each spatial location,
    avoiding the permute/reshape overhead of standard nn.LayerNorm.

    Args:
        channels: Number of channels to normalize over
        eps: Small value for numerical stability

    Example:
        >>> norm = LayerNorm2d(64)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> y = norm(x)  # (2, 64, 32, 32)
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        """Initialize channel-first layer norm."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        result: torch.Tensor = x * self.weight.view(1, -1, 1, 1) + self.bias.view(
            1, -1, 1, 1
        )
        return result


class SimpleGate(nn.Module):
    """Splits channels in half and multiplies the two halves element-wise.

    Replaces standard nonlinear activations (ReLU/GELU) with a simple,
    parameter-free gating mechanism.

    Example:
        >>> gate = SimpleGate()
        >>> x = torch.randn(2, 64, 32, 32)
        >>> y = gate(x)  # (2, 32, 32, 32)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """Parameter-light channel attention without nonlinearities.

    Global average pooling followed by a 1x1 convolution, with no
    sigmoid/softmax gating (hence "simplified").

    Args:
        channels: Number of input/output channels

    Example:
        >>> sca = SimplifiedChannelAttention(64)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> y = sca(x)  # (2, 64, 32, 32)
    """

    def __init__(self, channels: int) -> None:
        """Initialize simplified channel attention."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x * self.conv(self.pool(x))


class NAFBlock(nn.Module):
    """NAFNet building block.

    Two residual sub-blocks: a spatial-gating conv branch (1x1 -> depthwise 3x3
    -> SimpleGate -> channel attention -> 1x1) and a gated feed-forward branch
    (1x1 -> SimpleGate -> 1x1), each scaled by a learnable per-channel factor
    initialized to zero for stable training from scratch.

    Args:
        channels: Number of input/output channels
        ffn_expansion: Channel expansion factor for the feed-forward branch
        drop_out_rate: Dropout probability applied after each branch

    Example:
        >>> block = NAFBlock(64)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> y = block(x)  # (2, 64, 32, 32)
    """

    def __init__(
        self,
        channels: int,
        ffn_expansion: int = 2,
        drop_out_rate: float = 0.0,
    ) -> None:
        """Initialize NAFNet block."""
        super().__init__()
        dw_channels = channels * 2

        # Conv branch: 1x1 -> depthwise 3x3 -> SimpleGate -> SCA -> 1x1
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(
            dw_channels,
            dw_channels,
            kernel_size=3,
            padding=1,
            groups=dw_channels,
            bias=True,
        )
        self.sg1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_channels // 2)
        self.conv2 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1, bias=True)

        # Feed-forward branch: 1x1 -> SimpleGate -> 1x1
        ffn_channels = channels * ffn_expansion
        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, ffn_channels, kernel_size=1, bias=True)
        self.sg2 = SimpleGate()
        self.conv4 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1, bias=True)

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.dropout = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        residual = x
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.sg1(y)
        y = self.sca(y)
        y = self.conv2(y)
        y = self.dropout(y)
        x = residual + y * self.beta

        residual = x
        y = self.norm2(x)
        y = self.conv3(y)
        y = self.sg2(y)
        y = self.conv4(y)
        y = self.dropout(y)
        x = residual + y * self.gamma

        return x


class NAFNet(BaseModel):
    """NAFNet for image restoration and deraining.

    An efficient encoder-decoder architecture built entirely from NAFBlocks
    (no ReLU/GELU/Sigmoid), predicting a residual that is added back to the
    (padded) input image.

    Args:
        in_channels: Number of input channels. Default: 3 (RGB)
        out_channels: Number of output channels. Default: 3 (RGB)
        width: Base channel width. Default: 32
        enc_blk_nums: Number of NAFBlocks per encoder stage. Default: [2, 2, 4, 8]
        middle_blk_num: Number of NAFBlocks in the bottleneck. Default: 4
        dec_blk_nums: Number of NAFBlocks per decoder stage. Default: [2, 2, 2, 2]

    Reference:
        Chen et al. "Simple Baselines for Image Restoration." ECCV 2022.

    Example:
        >>> model = NAFNet(width=32)
        >>> x = torch.randn(2, 3, 256, 256)
        >>> y = model(x)  # (2, 3, 256, 256)
        >>> print(f"Params: {model.get_num_params():,}")
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        width: int = 32,
        enc_blk_nums: Optional[List[int]] = None,
        middle_blk_num: int = 4,
        dec_blk_nums: Optional[List[int]] = None,
    ) -> None:
        """Initialize NAFNet."""
        super().__init__(in_channels=in_channels, out_channels=out_channels)

        if enc_blk_nums is None:
            enc_blk_nums = [2, 2, 4, 8]
        if dec_blk_nums is None:
            dec_blk_nums = [2, 2, 2, 2]

        self.width = width
        self.enc_blk_nums = enc_blk_nums
        self.middle_blk_num = middle_blk_num
        self.dec_blk_nums = dec_blk_nums

        self.intro = nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=True)
        self.ending = nn.Conv2d(
            width, out_channels, kernel_size=3, padding=1, bias=True
        )

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, kernel_size=2, stride=2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.enc_blk_nums)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        _, _, h, w = x.shape
        x_padded = self._pad(x)

        feat = self.intro(x_padded)

        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            feat = encoder(feat)
            skips.append(feat)
            feat = down(feat)

        feat = self.middle_blks(feat)

        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            feat = up(feat)
            feat = feat + skip
            feat = decoder(feat)

        feat = self.ending(feat)

        if self.in_channels == self.out_channels:
            feat = feat + x_padded

        result: torch.Tensor = feat[:, :, :h, :w]
        return result

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        """Pad input so height/width are divisible by the network's stride."""
        _, _, h, w = x.shape
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, pad_w, 0, pad_h))

    def get_config(self) -> dict:
        """Get model configuration."""
        config = super().get_config()
        config.update(
            {
                "width": self.width,
                "enc_blk_nums": self.enc_blk_nums,
                "middle_blk_num": self.middle_blk_num,
                "dec_blk_nums": self.dec_blk_nums,
            }
        )
        return config


class NAFNetSmall(NAFNet):
    """Smaller, faster NAFNet variant for quick experimentation.

    Uses a reduced width (16) and a single NAFBlock per stage.

    Example:
        >>> model = NAFNetSmall()
        >>> print(f"Params: {model.get_num_params():,}")
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        **kwargs: Any,
    ) -> None:
        """Initialize small NAFNet."""
        kwargs.setdefault("width", 16)
        kwargs.setdefault("enc_blk_nums", [1, 1, 1, 1])
        kwargs.setdefault("middle_blk_num", 1)
        kwargs.setdefault("dec_blk_nums", [1, 1, 1, 1])
        super().__init__(in_channels=in_channels, out_channels=out_channels, **kwargs)


class NAFNetLarge(NAFNet):
    """Larger NAFNet variant for higher quality restoration.

    Uses the full-size configuration from the original paper (width=64).

    Example:
        >>> model = NAFNetLarge()
        >>> print(f"Params: {model.get_num_params():,}")
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        **kwargs: Any,
    ) -> None:
        """Initialize large NAFNet."""
        kwargs.setdefault("width", 64)
        kwargs.setdefault("enc_blk_nums", [2, 2, 4, 8])
        kwargs.setdefault("middle_blk_num", 12)
        kwargs.setdefault("dec_blk_nums", [2, 2, 2, 2])
        super().__init__(in_channels=in_channels, out_channels=out_channels, **kwargs)


__all__ = [
    "NAFNet",
    "NAFNetSmall",
    "NAFNetLarge",
    "NAFBlock",
    "SimpleGate",
    "SimplifiedChannelAttention",
    "LayerNorm2d",
]
