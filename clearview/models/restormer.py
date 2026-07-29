"""Restormer: Efficient Transformer for High-Resolution Image Restoration.

A Transformer-based encoder-decoder architecture using Multi-Dconv Head
Transposed Attention (MDTA) — self-attention computed across the channel
dimension instead of the spatial dimension — making it linear-complexity
with respect to image resolution, plus a Gated-Dconv Feed-Forward Network
(GDFN) for controlled feature transformation.

Reference:
    Zamir et al. "Restormer: Efficient Transformer for High-Resolution
    Image Restoration." CVPR 2022.
    https://github.com/swz30/Restormer
"""

from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from clearview.models.base import BaseModel


class BiasFreeLayerNorm(nn.Module):
    """Bias-free LayerNorm operating over the channel dimension.

    Normalizes each spatial location across channels without mean-centering
    or a bias term, following the Restormer formulation.

    Args:
        channels: Number of channels to normalize over

    Example:
        >>> norm = BiasFreeLayerNorm(64)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> y = norm(x)  # (2, 64, 32, 32)
    """

    def __init__(self, channels: int) -> None:
        """Initialize bias-free layer norm."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        sigma = x.var(dim=1, keepdim=True, unbiased=False)
        result: torch.Tensor = (
            x / torch.sqrt(sigma + 1e-5) * self.weight.view(1, -1, 1, 1)
        )
        return result


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network.

    Expands channels, applies a depthwise convolution, then gates one half
    of the resulting features with a GELU-activated copy of the other half
    before projecting back down.

    Args:
        channels: Number of input/output channels
        ffn_expansion_factor: Channel expansion factor for the hidden layer

    Example:
        >>> ffn = GDFN(64)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> y = ffn(x)  # (2, 64, 32, 32)
    """

    def __init__(self, channels: int, ffn_expansion_factor: float = 2.66) -> None:
        """Initialize gated-Dconv feed-forward network."""
        super().__init__()
        hidden = int(channels * ffn_expansion_factor)

        self.project_in = nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            kernel_size=3,
            padding=1,
            groups=hidden * 2,
            bias=True,
        )
        self.project_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        result: torch.Tensor = self.project_out(x)
        return result


class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention.

    Computes self-attention across the channel dimension (rather than the
    spatial dimension), yielding linear complexity with respect to image
    resolution while still capturing global context.

    Args:
        channels: Number of input/output channels
        num_heads: Number of attention heads

    Example:
        >>> attn = MDTA(64, num_heads=4)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> y = attn(x)  # (2, 64, 32, 32)
    """

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        """Initialize multi-Dconv head transposed attention."""
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=True)
        self.qkv_dwconv = nn.Conv2d(
            channels * 3,
            channels * 3,
            kernel_size=3,
            padding=1,
            groups=channels * 3,
            bias=True,
        )
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        head_dim = c // self.num_heads
        q = q.reshape(b, self.num_heads, head_dim, h * w)
        k = k.reshape(b, self.num_heads, head_dim, h * w)
        v = v.reshape(b, self.num_heads, head_dim, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.reshape(b, c, h, w)
        result: torch.Tensor = self.project_out(out)
        return result


class TransformerBlock(nn.Module):
    """Restormer Transformer block: MDTA + GDFN with residual connections.

    Args:
        channels: Number of input/output channels
        num_heads: Number of attention heads for MDTA
        ffn_expansion_factor: Channel expansion factor for GDFN

    Example:
        >>> block = TransformerBlock(64, num_heads=4)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> y = block(x)  # (2, 64, 32, 32)
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        ffn_expansion_factor: float = 2.66,
    ) -> None:
        """Initialize transformer block."""
        super().__init__()
        self.norm1 = BiasFreeLayerNorm(channels)
        self.attn = MDTA(channels, num_heads=num_heads)
        self.norm2 = BiasFreeLayerNorm(channels)
        self.ffn = GDFN(channels, ffn_expansion_factor=ffn_expansion_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Downsample(nn.Module):
    """Downsample by 2x via pixel-unshuffle (channels x2, spatial /2).

    Args:
        channels: Number of input channels

    Example:
        >>> down = Downsample(64)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> y = down(x)  # (2, 128, 16, 16)
    """

    def __init__(self, channels: int) -> None:
        """Initialize downsampling module."""
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        result: torch.Tensor = self.body(x)
        return result


class Upsample(nn.Module):
    """Upsample by 2x via pixel-shuffle (channels /2, spatial x2).

    Args:
        channels: Number of input channels

    Example:
        >>> up = Upsample(64)
        >>> x = torch.randn(2, 64, 16, 16)
        >>> y = up(x)  # (2, 32, 32, 32)
    """

    def __init__(self, channels: int) -> None:
        """Initialize upsampling module."""
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        result: torch.Tensor = self.body(x)
        return result


class Restormer(BaseModel):
    """Restormer for image restoration and deraining.

    A 4-level Transformer encoder-decoder with skip connections, using
    Multi-Dconv Head Transposed Attention (linear complexity channel-wise
    self-attention) and Gated-Dconv Feed-Forward Networks throughout. The
    architecture predicts a residual added back to the (padded) input.

    Args:
        in_channels: Number of input channels. Default: 3 (RGB)
        out_channels: Number of output channels. Default: 3 (RGB)
        dim: Base embedding dimension. Default: 48
        num_blocks: Number of Transformer blocks per encoder/decoder level.
            Default: [2, 3, 3, 4]
        num_refinement_blocks: Number of Transformer blocks in the final
            refinement stage. Default: 4
        heads: Number of attention heads per level. Default: [1, 2, 4, 8]
        ffn_expansion_factor: Channel expansion factor for GDFN. Default: 2.66

    Reference:
        Zamir et al. "Restormer: Efficient Transformer for High-Resolution
        Image Restoration." CVPR 2022.

    Example:
        >>> model = Restormer(dim=48)
        >>> x = torch.randn(2, 3, 256, 256)
        >>> y = model(x)  # (2, 3, 256, 256)
        >>> print(f"Params: {model.get_num_params():,}")
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: Optional[List[int]] = None,
        num_refinement_blocks: int = 4,
        heads: Optional[List[int]] = None,
        ffn_expansion_factor: float = 2.66,
    ) -> None:
        """Initialize Restormer."""
        super().__init__(in_channels=in_channels, out_channels=out_channels)

        if num_blocks is None:
            num_blocks = [2, 3, 3, 4]
        if heads is None:
            heads = [1, 2, 4, 8]

        self.dim = dim
        self.num_blocks = num_blocks
        self.num_refinement_blocks = num_refinement_blocks
        self.heads = heads
        self.ffn_expansion_factor = ffn_expansion_factor

        self.patch_embed = nn.Conv2d(
            in_channels, dim, kernel_size=3, padding=1, bias=True
        )

        # Encoder
        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(dim, heads[0], ffn_expansion_factor)
                for _ in range(num_blocks[0])
            ]
        )
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(dim * 2, heads[1], ffn_expansion_factor)
                for _ in range(num_blocks[1])
            ]
        )
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = nn.Sequential(
            *[
                TransformerBlock(dim * 4, heads[2], ffn_expansion_factor)
                for _ in range(num_blocks[2])
            ]
        )
        self.down3_4 = Downsample(dim * 4)
        self.latent = nn.Sequential(
            *[
                TransformerBlock(dim * 8, heads[3], ffn_expansion_factor)
                for _ in range(num_blocks[3])
            ]
        )

        # Decoder
        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=True)
        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(dim * 4, heads[2], ffn_expansion_factor)
                for _ in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=True)
        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(dim * 2, heads[1], ffn_expansion_factor)
                for _ in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(dim * 2)
        # Level-1 decoder keeps concatenated channels (dim * 2), unreduced,
        # following the original Restormer design.
        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(dim * 2, heads[0], ffn_expansion_factor)
                for _ in range(num_blocks[0])
            ]
        )

        self.refinement = nn.Sequential(
            *[
                TransformerBlock(dim * 2, heads[0], ffn_expansion_factor)
                for _ in range(num_refinement_blocks)
            ]
        )

        self.output = nn.Conv2d(
            dim * 2, out_channels, kernel_size=3, padding=1, bias=True
        )

        # 3 downsampling stages => input must be divisible by 2^3
        self.padder_size = 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        _, _, h, w = x.shape
        x_padded = self._pad(x)

        shallow = self.patch_embed(x_padded)

        enc1 = self.encoder_level1(shallow)
        enc2_in = self.down1_2(enc1)
        enc2 = self.encoder_level2(enc2_in)
        enc3_in = self.down2_3(enc2)
        enc3 = self.encoder_level3(enc3_in)
        enc4_in = self.down3_4(enc3)
        latent = self.latent(enc4_in)

        dec3_in = self.up4_3(latent)
        dec3_in = torch.cat([dec3_in, enc3], dim=1)
        dec3_in = self.reduce_chan_level3(dec3_in)
        dec3 = self.decoder_level3(dec3_in)

        dec2_in = self.up3_2(dec3)
        dec2_in = torch.cat([dec2_in, enc2], dim=1)
        dec2_in = self.reduce_chan_level2(dec2_in)
        dec2 = self.decoder_level2(dec2_in)

        dec1_in = self.up2_1(dec2)
        dec1_in = torch.cat([dec1_in, enc1], dim=1)
        dec1 = self.decoder_level1(dec1_in)

        refined = self.refinement(dec1)

        out = self.output(refined)

        if self.in_channels == self.out_channels:
            out = out + x_padded

        result: torch.Tensor = out[:, :, :h, :w]
        return result

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        """Pad input so height/width are divisible by the network's stride."""
        _, _, h, w = x.shape
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    def get_config(self) -> dict:
        """Get model configuration."""
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_blocks": self.num_blocks,
                "num_refinement_blocks": self.num_refinement_blocks,
                "heads": self.heads,
                "ffn_expansion_factor": self.ffn_expansion_factor,
            }
        )
        return config


class RestormerSmall(Restormer):
    """Smaller, faster Restormer variant for quick experimentation.

    Uses a reduced embedding dimension (24) and fewer blocks per level.

    Example:
        >>> model = RestormerSmall()
        >>> print(f"Params: {model.get_num_params():,}")
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        **kwargs: Any,
    ) -> None:
        """Initialize small Restormer."""
        kwargs.setdefault("dim", 24)
        kwargs.setdefault("num_blocks", [1, 1, 1, 2])
        kwargs.setdefault("num_refinement_blocks", 2)
        kwargs.setdefault("heads", [1, 2, 4, 8])
        super().__init__(in_channels=in_channels, out_channels=out_channels, **kwargs)


class RestormerLarge(Restormer):
    """Larger Restormer variant matching the original paper's configuration.

    Uses the full-size configuration (dim=48, num_blocks=[4,6,6,8]) from
    the original CVPR 2022 paper for maximum quality.

    Example:
        >>> model = RestormerLarge()
        >>> print(f"Params: {model.get_num_params():,}")
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        **kwargs: Any,
    ) -> None:
        """Initialize large Restormer."""
        kwargs.setdefault("dim", 48)
        kwargs.setdefault("num_blocks", [4, 6, 6, 8])
        kwargs.setdefault("num_refinement_blocks", 4)
        kwargs.setdefault("heads", [1, 2, 4, 8])
        super().__init__(in_channels=in_channels, out_channels=out_channels, **kwargs)


__all__ = [
    "Restormer",
    "RestormerSmall",
    "RestormerLarge",
    "TransformerBlock",
    "MDTA",
    "GDFN",
    "BiasFreeLayerNorm",
    "Downsample",
    "Upsample",
]
