"""Histoformer: histogram self-attention transformer for image restoration.

A transformer that replaces standard self-attention with Dynamic-range
Histogram Self-Attention (DHSA), sorting and grouping features by intensity
to attend across similar regions regardless of spatial distance.

Reference:
    Sun et al. "Restoring Images in Adverse Weather Conditions via
    Histogram Transformer." ECCV 2024.
    https://github.com/sunshangquan/Histoformer
"""

import numbers
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from clearview.models.base import BaseModel


def _to_3d(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, "b c h w -> b (h w) c")


def _to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class _BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape: Any) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        assert len(self.normalized_shape) == 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5)


class _WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape: Any) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        assert len(self.normalized_shape) == 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5)


class _LayerNorm(nn.Module):
    def __init__(self, dim: int, layer_norm_type: str = "WithBias") -> None:
        super().__init__()
        self.body: nn.Module = (
            _BiasFreeLayerNorm(dim)
            if layer_norm_type == "BiasFree"
            else _WithBiasLayerNorm(dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return _to_4d(self.body(_to_3d(x)), h, w)


class _FeedForward(nn.Module):
    """Dual-scale Gated Feed-Forward Network (DGFF)."""

    def __init__(self, dim: int, ffn_expansion_factor: float, bias: bool) -> None:
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv_5 = nn.Conv2d(
            hidden_features // 4,
            hidden_features // 4,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=hidden_features // 4,
            bias=bias,
        )
        self.dwconv_dilated2_1 = nn.Conv2d(
            hidden_features // 4,
            hidden_features // 4,
            kernel_size=3,
            stride=1,
            padding=2,
            groups=hidden_features // 4,
            bias=bias,
            dilation=2,
        )
        self.p_unshuffle = nn.PixelUnshuffle(2)
        self.p_shuffle = nn.PixelShuffle(2)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x = self.p_shuffle(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.dwconv_5(x1)
        x2 = self.dwconv_dilated2_1(x2)
        x = F.mish(x2) * x1
        x = self.p_unshuffle(x)
        x = self.project_out(x)
        return x


class _AttentionHistogram(nn.Module):
    """Dynamic-range Histogram Self-Attention (DHSA)."""

    def __init__(self, dim: int, num_heads: int, bias: bool) -> None:
        super().__init__()
        self.factor = num_heads
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 5,
            dim * 5,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 5,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def _pad(self, x: torch.Tensor, factor: int) -> Any:
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, "constant", 0)
        return x, t_pad

    def _unpad(self, x: torch.Tensor, t_pad: List[int]) -> torch.Tensor:
        _, _, hw = x.shape
        return x[:, :, t_pad[0] : hw - t_pad[1]]

    def _softmax_1(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        logit = x.exp()
        return logit / (logit.sum(dim, keepdim=True) + 1)

    def _reshape_attn(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, if_box: bool
    ) -> torch.Tensor:
        b = q.shape[0]
        q, t_pad = self._pad(q, self.factor)
        k, t_pad = self._pad(k, self.factor)
        v, t_pad = self._pad(v, self.factor)
        hw = q.shape[-1] // self.factor
        shape_ori = "b (head c) (factor hw)" if if_box else "b (head c) (hw factor)"
        shape_tar = "b head (c factor) hw"
        q = rearrange(
            q,
            f"{shape_ori} -> {shape_tar}",
            factor=self.factor,
            hw=hw,
            head=self.num_heads,
        )
        k = rearrange(
            k,
            f"{shape_ori} -> {shape_tar}",
            factor=self.factor,
            hw=hw,
            head=self.num_heads,
        )
        v = rearrange(
            v,
            f"{shape_ori} -> {shape_tar}",
            factor=self.factor,
            hw=hw,
            head=self.num_heads,
        )
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self._softmax_1(attn, dim=-1)
        out = attn @ v
        out = rearrange(
            out,
            f"{shape_tar} -> {shape_ori}",
            factor=self.factor,
            hw=hw,
            b=b,
            head=self.num_heads,
        )
        out = self._unpad(out, t_pad)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_sort, idx_h = x[:, : c // 2].sort(-2)
        x_sort, idx_w = x_sort.sort(-1)
        x[:, : c // 2] = x_sort
        qkv = self.qkv_dwconv(self.qkv(x))
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)

        v, idx = v.view(b, c, -1).sort(dim=-1)
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx)

        out1 = self._reshape_attn(q1, k1, v, True)
        out2 = self._reshape_attn(q2, k2, v, False)

        out1 = torch.scatter(out1, 2, idx, out1).view(b, c, h, w)
        out2 = torch.scatter(out2, 2, idx, out2).view(b, c, h, w)
        out = out1 * out2
        out = self.project_out(out)
        out_replace = out[:, : c // 2]
        out_replace = torch.scatter(out_replace, -1, idx_w, out_replace)
        out_replace = torch.scatter(out_replace, -2, idx_h, out_replace)
        out[:, : c // 2] = out_replace
        return out


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion_factor: float,
        bias: bool,
        layer_norm_type: str,
    ) -> None:
        super().__init__()
        self.attn_g = _AttentionHistogram(dim, num_heads, bias)
        self.norm_g = _LayerNorm(dim, layer_norm_type)
        self.ffn = _FeedForward(dim, ffn_expansion_factor, bias)
        self.norm_ff1 = _LayerNorm(dim, layer_norm_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_g(self.norm_g(x))
        return x + self.ffn(self.norm_ff1(x))


class _OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c: int = 3, embed_dim: int = 48, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.proj(x)
        return result


class _SkipPatchEmbed(nn.Module):
    def __init__(self, in_c: int = 3, dim: int = 3, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.AvgPool2d(
                2, stride=2, padding=0, ceil_mode=False, count_include_pad=True
            ),
            nn.Conv2d(in_c, dim, kernel_size=1, bias=bias),
            nn.Conv2d(
                dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.proj(x)
        return result


class _Downsample(nn.Module):
    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.body(x)
        return result


class _Upsample(nn.Module):
    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.body(x)
        return result


class Histoformer(BaseModel):
    """Histoformer for image restoration.

    Encoder-decoder transformer using histogram self-attention. Inference
    only; defaults match the released checkpoint config. The skip-connection
    path is hardcoded to 3 channels, so in_channels must stay at 3.

    Args:
        in_channels: Number of input channels. Default: 3 (RGB)
        out_channels: Number of output channels. Default: 3 (RGB)
        dim: Base channel dimension. Default: 36
        num_blocks: Number of transformer blocks per encoder/decoder stage. Default: [4, 4, 6, 8]
        num_refinement_blocks: Number of refinement blocks. Default: 4
        heads: Number of attention heads per stage. Default: [1, 2, 4, 8]
        ffn_expansion_factor: Feed-forward expansion ratio. Default: 2.667

    Reference:
        Sun et al. "Restoring Images in Adverse Weather Conditions via
        Histogram Transformer." ECCV 2024.
        https://github.com/sunshangquan/Histoformer
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 36,
        num_blocks: Optional[List[int]] = None,
        num_refinement_blocks: int = 4,
        heads: Optional[List[int]] = None,
        ffn_expansion_factor: float = 2.667,
        bias: bool = False,
        layer_norm_type: str = "WithBias",
        dual_pixel_task: bool = False,
    ) -> None:
        """Initialize Histoformer."""
        super().__init__(in_channels=in_channels, out_channels=out_channels)

        if num_blocks is None:
            num_blocks = [4, 4, 6, 8]
        if heads is None:
            heads = [1, 2, 4, 8]

        self.dim = dim
        self.num_blocks = num_blocks
        self.num_refinement_blocks = num_refinement_blocks
        self.heads = heads
        self.ffn_expansion_factor = ffn_expansion_factor
        self.bias = bias
        self.layer_norm_type = layer_norm_type
        self.dual_pixel_task = dual_pixel_task
        # 3 downsampling stages (down1_2, down2_3, down3_4), each /2.
        self.padder_size = 8

        def _block(d: int, h: int, n: int) -> nn.Sequential:
            return nn.Sequential(
                *[
                    _TransformerBlock(d, h, ffn_expansion_factor, bias, layer_norm_type)
                    for _ in range(n)
                ]
            )

        self.patch_embed = _OverlapPatchEmbed(in_channels, dim)

        self.encoder_level1 = _block(dim, heads[0], num_blocks[0])
        self.down1_2 = _Downsample(dim)
        self.encoder_level2 = _block(dim * 2, heads[1], num_blocks[1])
        self.down2_3 = _Downsample(dim * 2)
        self.encoder_level3 = _block(dim * 4, heads[2], num_blocks[2])
        self.down3_4 = _Downsample(dim * 4)
        self.latent = _block(dim * 8, heads[3], num_blocks[3])

        self.up4_3 = _Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=bias)
        self.decoder_level3 = _block(dim * 4, heads[2], num_blocks[2])

        self.up3_2 = _Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.decoder_level2 = _block(dim * 2, heads[1], num_blocks[1])

        self.up2_1 = _Upsample(dim * 2)
        self.decoder_level1 = _block(dim * 2, heads[0], num_blocks[0])

        self.refinement = _block(dim * 2, heads[0], num_refinement_blocks)

        self.skip_patch_embed1 = _SkipPatchEmbed(3, 3)
        self.skip_patch_embed2 = _SkipPatchEmbed(3, 3)
        self.skip_patch_embed3 = _SkipPatchEmbed(3, 3)
        self.reduce_chan_level_1 = nn.Conv2d(
            dim * 2 + 3, dim * 2, kernel_size=1, bias=bias
        )
        self.reduce_chan_level_2 = nn.Conv2d(
            dim * 4 + 3, dim * 4, kernel_size=1, bias=bias
        )
        self.reduce_chan_level_3 = nn.Conv2d(
            dim * 8 + 3, dim * 8, kernel_size=1, bias=bias
        )

        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)

        self.output = nn.Conv2d(
            dim * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        """Pad input so height/width are divisible by the network's stride."""
        _, _, h, w = x.shape
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        _, _, h, w = x.shape
        inp_img = self._pad(x)

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        skip_enc_level1 = self.skip_patch_embed1(inp_img)
        inp_enc_level2 = self.reduce_chan_level_1(
            torch.cat([inp_enc_level2, skip_enc_level1], 1)
        )
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        skip_enc_level2 = self.skip_patch_embed2(skip_enc_level1)
        inp_enc_level3 = self.reduce_chan_level_2(
            torch.cat([inp_enc_level3, skip_enc_level2], 1)
        )
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        skip_enc_level3 = self.skip_patch_embed3(skip_enc_level2)
        inp_enc_level4 = self.reduce_chan_level_3(
            torch.cat([inp_enc_level4, skip_enc_level3], 1)
        )
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1)
        out = out_dec_level1 + inp_img

        result: torch.Tensor = out[:, :, :h, :w]
        return result

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_blocks": self.num_blocks,
                "num_refinement_blocks": self.num_refinement_blocks,
                "heads": self.heads,
                "ffn_expansion_factor": self.ffn_expansion_factor,
                "bias": self.bias,
                "layer_norm_type": self.layer_norm_type,
                "dual_pixel_task": self.dual_pixel_task,
            }
        )
        return config


__all__ = ["Histoformer"]
