"""Tests for NAFNet architecture.

Comprehensive tests for NAFNet building blocks (LayerNorm2d, SimpleGate,
SimplifiedChannelAttention, NAFBlock) and the full NAFNet model including
architecture validation, forward pass, padding behavior, and utilities.
"""

import pytest
import torch

from clearview.models import get_model
from clearview.models.nafnet import (
    LayerNorm2d,
    NAFBlock,
    NAFNet,
    NAFNetLarge,
    NAFNetSmall,
    SimpleGate,
    SimplifiedChannelAttention,
)


class TestLayerNorm2d:
    """Tests for LayerNorm2d component."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        norm = LayerNorm2d(64)
        x = torch.randn(2, 64, 32, 32)
        y = norm(x)
        assert y.shape == x.shape

    def test_normalizes_channel_dim(self):
        """Test that normalization occurs across the channel dimension."""
        norm = LayerNorm2d(64)
        x = torch.randn(2, 64, 8, 8) * 10 + 5
        y = norm(x)

        # Mean across channels per-pixel should be ~0 before affine transform
        mean_per_pixel = y.mean(dim=1)
        assert torch.allclose(
            mean_per_pixel, torch.zeros_like(mean_per_pixel), atol=1e-1
        )

    def test_learnable_params(self):
        """Test that weight and bias are learnable parameters."""
        norm = LayerNorm2d(32)
        assert norm.weight.requires_grad
        assert norm.bias.requires_grad
        assert norm.weight.shape == (32,)
        assert norm.bias.shape == (32,)


class TestSimpleGate:
    """Tests for SimpleGate component."""

    def test_halves_channels(self):
        """Test that SimpleGate halves the channel dimension."""
        gate = SimpleGate()
        x = torch.randn(2, 64, 16, 16)
        y = gate(x)
        assert y.shape == (2, 32, 16, 16)

    def test_multiplies_halves(self):
        """Test that SimpleGate multiplies the two channel halves."""
        gate = SimpleGate()
        x1 = torch.randn(2, 4, 8, 8)
        x2 = torch.randn(2, 4, 8, 8)
        x = torch.cat([x1, x2], dim=1)
        y = gate(x)
        assert torch.allclose(y, x1 * x2)


class TestSimplifiedChannelAttention:
    """Tests for SimplifiedChannelAttention component."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        sca = SimplifiedChannelAttention(64)
        x = torch.randn(2, 64, 32, 32)
        y = sca(x)
        assert y.shape == x.shape

    def test_gradient_flow(self):
        """Test that gradients flow through SCA."""
        sca = SimplifiedChannelAttention(32)
        x = torch.randn(1, 32, 16, 16, requires_grad=True)
        y = sca(x)
        y.sum().backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestNAFBlock:
    """Tests for NAFBlock component."""

    def test_output_shape(self):
        """Test that NAFBlock preserves input shape."""
        block = NAFBlock(64)
        x = torch.randn(2, 64, 32, 32)
        y = block(x)
        assert y.shape == x.shape

    def test_zero_init_scales(self):
        """Test that beta/gamma scaling parameters are zero-initialized."""
        block = NAFBlock(64)
        assert torch.all(block.beta == 0)
        assert torch.all(block.gamma == 0)

    def test_gradient_flow(self):
        """Test that gradients flow through the block."""
        block = NAFBlock(32)
        x = torch.randn(1, 32, 16, 16, requires_grad=True)
        y = block(x)
        y.sum().backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_dropout_variant(self):
        """Test NAFBlock with dropout enabled."""
        block = NAFBlock(32, drop_out_rate=0.5)
        x = torch.randn(2, 32, 16, 16)
        y = block(x)
        assert y.shape == x.shape


class TestNAFNet:
    """Tests for the full NAFNet model."""

    def test_init_defaults(self):
        """Test default initialization."""
        model = NAFNet()
        assert model.in_channels == 3
        assert model.out_channels == 3
        assert model.width == 32
        assert model.enc_blk_nums == [2, 2, 4, 8]
        assert model.dec_blk_nums == [2, 2, 2, 2]

    def test_forward_pass_basic(self):
        """Test basic forward pass with aligned input size."""
        model = NAFNetSmall()
        model.eval()
        x = torch.randn(2, 3, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 3, 64, 64)

    @pytest.mark.parametrize("height,width", [(64, 64), (100, 130), (33, 65)])
    def test_forward_non_aligned_sizes(self, height, width):
        """Test forward pass with sizes not divisible by the padder size."""
        model = NAFNetSmall()
        model.eval()
        x = torch.randn(1, 3, height, width)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 3, height, width)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_forward_different_batch_sizes(self, batch_size):
        """Test forward pass with different batch sizes."""
        model = NAFNetSmall()
        model.eval()
        x = torch.randn(batch_size, 3, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 3, 64, 64)

    def test_gradient_flow(self):
        """Test that gradients flow through the entire model."""
        model = NAFNetSmall()
        x = torch.randn(1, 3, 32, 32, requires_grad=True)

        output = model(x)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert all(p.grad is not None for p in model.parameters())

    def test_mismatched_in_out_channels(self):
        """Test that model handles differing in/out channels without residual add."""
        model = NAFNetSmall(in_channels=1, out_channels=3)
        x = torch.randn(1, 1, 32, 32)

        output = model(x)

        assert output.shape == (1, 3, 32, 32)

    def test_get_config(self):
        """Test model configuration serialization."""
        model = NAFNetSmall()
        config = model.get_config()

        assert config["model_type"] == "NAFNetSmall"
        assert config["in_channels"] == 3
        assert config["out_channels"] == 3
        assert "width" in config
        assert "enc_blk_nums" in config
        assert "middle_blk_num" in config
        assert "dec_blk_nums" in config

    def test_small_variant_smaller_than_default(self):
        """Test that NAFNetSmall has fewer parameters than the default NAFNet."""
        small = NAFNetSmall()
        default = NAFNet()

        assert small.get_num_params() < default.get_num_params()

    def test_large_variant_larger_than_default(self):
        """Test that NAFNetLarge has more parameters than the default NAFNet."""
        large = NAFNetLarge()
        default = NAFNet()

        assert large.get_num_params() > default.get_num_params()

    def test_get_model_factory(self):
        """Test that NAFNet variants are accessible via the model factory."""
        for name in ["nafnet", "nafnet_small", "nafnet_large"]:
            model = get_model(name)
            assert isinstance(model, NAFNet)
