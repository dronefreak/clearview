"""Histoformer: histogram self-attention transformer for image restoration.

A transformer that replaces standard self-attention with Dynamic-range
Histogram Self-Attention (DHSA), sorting and grouping features by intensity
to attend across similar regions regardless of spatial distance.

Reference:
    Sun et al. "Restoring Images in Adverse Weather Conditions via
    Histogram Transformer." ECCV 2024.
    https://github.com/sunshangquan/Histoformer
"""

import pytest
import torch

from clearview.models import get_model
from clearview.models.histoformer import Histoformer


class TestHistoformer:
    """Tests for the Histoformer model."""

    def test_init_defaults(self):
        """Test default initialization matches the released checkpoint's config."""
        model = Histoformer()
        assert model.in_channels == 3
        assert model.out_channels == 3
        assert model.dim == 36
        assert model.num_blocks == [4, 4, 6, 8]
        assert model.heads == [1, 2, 4, 8]
        assert model.ffn_expansion_factor == 2.667

    def test_default_param_count_matches_released_checkpoint(self):
        """Test that the default config produces exactly 16,615,100 params,
        matching the officially released net_g_real/net_g_best checkpoint."""
        model = Histoformer()
        assert sum(p.numel() for p in model.parameters()) == 16_615_100

    def test_forward_pass_basic(self):
        """Test basic forward pass with an already-aligned input size."""
        model = Histoformer(
            dim=16,
            ffn_expansion_factor=2.0,
            num_blocks=[1, 1, 1, 1],
            num_refinement_blocks=1,
        )
        model.eval()
        x = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 3, 64, 64)

    def test_forward_non_aligned_sizes(self):
        """Test forward pass with sizes not divisible by the padder size (8)."""
        model = Histoformer(
            dim=16,
            ffn_expansion_factor=2.0,
            num_blocks=[1, 1, 1, 1],
            num_refinement_blocks=1,
        )
        model.eval()
        x = torch.randn(1, 3, 37, 51)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 3, 37, 51)

    def test_gradient_flow(self):
        """Test that gradients flow through the entire model."""
        model = Histoformer(
            dim=16,
            ffn_expansion_factor=2.0,
            num_blocks=[1, 1, 1, 1],
            num_refinement_blocks=1,
        )
        x = torch.randn(1, 3, 32, 32, requires_grad=True)

        output = model(x)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert all(p.grad is not None for p in model.parameters())

    def test_requires_3_input_channels(self):
        """Test the (inherited from the original architecture, not a porting
        bug) constraint that input must be 3-channel RGB: the skip-connection
        path (skip_patch_embed1/2/3) operates directly on the raw input and
        is hardcoded to 3 channels regardless of the in_channels kwarg,
        unlike Restormer/UNet/NAFNet which handle arbitrary in_channels."""
        model = Histoformer(
            in_channels=1,
            dim=16,
            ffn_expansion_factor=2.0,
            num_blocks=[1, 1, 1, 1],
            num_refinement_blocks=1,
        )
        x = torch.randn(1, 1, 32, 32)

        with pytest.raises(RuntimeError):
            model(x)

    def test_get_config(self):
        """Test model configuration serialization."""
        model = Histoformer()
        config = model.get_config()

        assert config["model_type"] == "Histoformer"
        assert config["in_channels"] == 3
        assert config["out_channels"] == 3
        assert config["dim"] == 36
        assert "num_blocks" in config
        assert "heads" in config
        assert "ffn_expansion_factor" in config
        assert "layer_norm_type" in config

    def test_get_model_factory(self):
        """Test that Histoformer is accessible via the model factory."""
        model = get_model(
            "histoformer",
            dim=16,
            ffn_expansion_factor=2.0,
            num_blocks=[1, 1, 1, 1],
            num_refinement_blocks=1,
        )
        assert isinstance(model, Histoformer)
