"""Tests for Restormer architecture.

Comprehensive tests for Restormer building blocks (BiasFreeLayerNorm, GDFN,
MDTA, TransformerBlock, Downsample, Upsample) and the full Restormer model
including architecture validation, forward pass, padding behavior, and
utilities.
"""

import pytest
import torch

from clearview.models import get_model
from clearview.models.restormer import (
    GDFN,
    MDTA,
    BiasFreeLayerNorm,
    Downsample,
    Restormer,
    RestormerLarge,
    RestormerSmall,
    TransformerBlock,
    Upsample,
)


class TestBiasFreeLayerNorm:
    """Tests for BiasFreeLayerNorm component."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        norm = BiasFreeLayerNorm(64)
        x = torch.randn(2, 64, 16, 16)
        y = norm(x)
        assert y.shape == x.shape

    def test_no_bias_param(self):
        """Test that the layer has no bias parameter, only weight."""
        norm = BiasFreeLayerNorm(32)
        assert hasattr(norm, "weight")
        assert not hasattr(norm, "bias")


class TestGDFN:
    """Tests for Gated-Dconv Feed-Forward Network."""

    def test_output_shape(self):
        """Test that GDFN preserves channel dimension."""
        ffn = GDFN(64)
        x = torch.randn(2, 64, 16, 16)
        y = ffn(x)
        assert y.shape == x.shape

    def test_gradient_flow(self):
        """Test that gradients flow through GDFN."""
        ffn = GDFN(32)
        x = torch.randn(1, 32, 8, 8, requires_grad=True)
        y = ffn(x)
        y.sum().backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestMDTA:
    """Tests for Multi-Dconv Head Transposed Attention."""

    def test_output_shape(self):
        """Test that MDTA preserves input shape."""
        attn = MDTA(64, num_heads=4)
        x = torch.randn(2, 64, 16, 16)
        y = attn(x)
        assert y.shape == x.shape

    @pytest.mark.parametrize("num_heads", [1, 2, 4, 8])
    def test_different_head_counts(self, num_heads):
        """Test MDTA with different numbers of attention heads."""
        attn = MDTA(64, num_heads=num_heads)
        x = torch.randn(1, 64, 8, 8)
        y = attn(x)
        assert y.shape == x.shape

    def test_gradient_flow(self):
        """Test that gradients flow through MDTA."""
        attn = MDTA(32, num_heads=4)
        x = torch.randn(1, 32, 8, 8, requires_grad=True)
        y = attn(x)
        y.sum().backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestTransformerBlock:
    """Tests for the full Transformer block (MDTA + GDFN)."""

    def test_output_shape(self):
        """Test that TransformerBlock preserves input shape."""
        block = TransformerBlock(64, num_heads=4)
        x = torch.randn(2, 64, 16, 16)
        y = block(x)
        assert y.shape == x.shape

    def test_gradient_flow(self):
        """Test that gradients flow through the block."""
        block = TransformerBlock(32, num_heads=2)
        x = torch.randn(1, 32, 8, 8, requires_grad=True)
        y = block(x)
        y.sum().backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestDownsampleUpsample:
    """Tests for Downsample/Upsample modules."""

    def test_downsample_shape(self):
        """Test that Downsample halves spatial dims and doubles channels."""
        down = Downsample(64)
        x = torch.randn(2, 64, 32, 32)
        y = down(x)
        assert y.shape == (2, 128, 16, 16)

    def test_upsample_shape(self):
        """Test that Upsample doubles spatial dims and halves channels."""
        up = Upsample(64)
        x = torch.randn(2, 64, 16, 16)
        y = up(x)
        assert y.shape == (2, 32, 32, 32)

    def test_down_up_roundtrip(self):
        """Test that Downsample followed by Upsample restores original shape."""
        down = Downsample(64)
        up = Upsample(128)
        x = torch.randn(2, 64, 32, 32)
        y = up(down(x))
        assert y.shape == x.shape


class TestRestormer:
    """Tests for the full Restormer model."""

    def test_init_defaults(self):
        """Test default initialization."""
        model = Restormer()
        assert model.in_channels == 3
        assert model.out_channels == 3
        assert model.dim == 48
        assert model.num_blocks == [2, 3, 3, 4]
        assert model.heads == [1, 2, 4, 8]

    def test_forward_pass_basic(self):
        """Test basic forward pass with aligned input size."""
        model = RestormerSmall()
        model.eval()
        x = torch.randn(2, 3, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 3, 64, 64)

    @pytest.mark.parametrize("height,width", [(64, 64), (100, 130), (33, 65)])
    def test_forward_non_aligned_sizes(self, height, width):
        """Test forward pass with sizes not divisible by the padder size."""
        model = RestormerSmall()
        model.eval()
        x = torch.randn(1, 3, height, width)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 3, height, width)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_forward_different_batch_sizes(self, batch_size):
        """Test forward pass with different batch sizes."""
        model = RestormerSmall()
        model.eval()
        x = torch.randn(batch_size, 3, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 3, 64, 64)

    def test_gradient_flow(self):
        """Test that gradients flow through the entire model."""
        model = RestormerSmall()
        x = torch.randn(1, 3, 32, 32, requires_grad=True)

        output = model(x)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert all(p.grad is not None for p in model.parameters())

    def test_mismatched_in_out_channels(self):
        """Test that model handles differing in/out channels without residual add."""
        model = RestormerSmall(in_channels=1, out_channels=3)
        x = torch.randn(1, 1, 32, 32)

        output = model(x)

        assert output.shape == (1, 3, 32, 32)

    def test_get_config(self):
        """Test model configuration serialization."""
        model = RestormerSmall()
        config = model.get_config()

        assert config["model_type"] == "RestormerSmall"
        assert config["in_channels"] == 3
        assert config["out_channels"] == 3
        assert "dim" in config
        assert "num_blocks" in config
        assert "num_refinement_blocks" in config
        assert "heads" in config
        assert "ffn_expansion_factor" in config

    def test_small_variant_smaller_than_default(self):
        """Test that RestormerSmall has fewer parameters than the default."""
        small = RestormerSmall()
        default = Restormer()

        assert small.get_num_params() < default.get_num_params()

    def test_large_variant_larger_than_default(self):
        """Test that RestormerLarge has more parameters than the default."""
        large = RestormerLarge()
        default = Restormer()

        assert large.get_num_params() > default.get_num_params()

    def test_get_model_factory(self):
        """Test that Restormer variants are accessible via the model factory."""
        for name in ["restormer", "restormer_small", "restormer_large"]:
            model = get_model(name)
            assert isinstance(model, Restormer)
