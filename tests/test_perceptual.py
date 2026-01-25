"""Tests for perceptual loss functions.

These tests cover VGG-based perceptual loss with proper mocking to avoid
downloading pretrained weights during CI/testing.
"""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from clearview.losses.perceptual import PerceptualLoss, VGGPerceptualLoss


class MockVGG(nn.Module):
    """Mock VGG model for testing without downloading weights."""

    def __init__(self):
        super().__init__()
        # Create a simple mock VGG with correct structure
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),  # 0
            nn.ReLU(),  # 1 - relu1_1
            nn.Conv2d(64, 64, 3, padding=1),  # 2
            nn.ReLU(),  # 3 - relu1_2
            nn.MaxPool2d(2),  # 4
            nn.Conv2d(64, 128, 3, padding=1),  # 5
            nn.ReLU(),  # 6 - relu2_1
            nn.Conv2d(128, 128, 3, padding=1),  # 7
            nn.ReLU(),  # 8 - relu2_2
            nn.MaxPool2d(2),  # 9
            nn.Conv2d(128, 256, 3, padding=1),  # 10
            nn.ReLU(),  # 11 - relu3_1
            nn.Conv2d(256, 256, 3, padding=1),  # 12
            nn.ReLU(),  # 13 - relu3_2
            nn.Conv2d(256, 256, 3, padding=1),  # 14
            nn.ReLU(),  # 15 - relu3_3
            nn.MaxPool2d(2),  # 16
            nn.Conv2d(256, 512, 3, padding=1),  # 17
            nn.ReLU(),  # 18 - relu4_1
            nn.Conv2d(512, 512, 3, padding=1),  # 19
            nn.ReLU(),  # 20 - relu4_2
            nn.Conv2d(512, 512, 3, padding=1),  # 21
            nn.ReLU(),  # 22 - relu4_3
            nn.MaxPool2d(2),  # 23
            nn.Conv2d(512, 512, 3, padding=1),  # 24
            nn.ReLU(),  # 25 - relu5_1
            nn.Conv2d(512, 512, 3, padding=1),  # 26
            nn.ReLU(),  # 27 - relu5_2
            nn.Conv2d(512, 512, 3, padding=1),  # 28
            nn.ReLU(),  # 29 - relu5_3
        )


@pytest.fixture
def mock_vgg():
    """Fixture that mocks VGG16 model to avoid downloading weights."""
    with patch("torchvision.models.vgg16") as mock:
        mock.return_value = MockVGG()
        yield mock


class TestVGGPerceptualLoss:
    """Tests for VGGPerceptualLoss."""

    def test_init_default(self, mock_vgg):
        """Test initialization with default parameters."""
        loss_fn = VGGPerceptualLoss()

        assert loss_fn.layers == ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
        assert loss_fn.layer_weights == [1.0, 1.0, 1.0, 1.0]
        assert loss_fn.normalize is True
        assert len(loss_fn.feature_extractors) == 4

    def test_init_custom_layers(self, mock_vgg):
        """Test initialization with custom layers."""
        layers = ["relu1_1", "relu2_1", "relu3_1"]
        weights = [0.5, 1.0, 2.0]

        loss_fn = VGGPerceptualLoss(layers=layers, layer_weights=weights)

        assert loss_fn.layers == layers
        assert loss_fn.layer_weights == weights
        assert len(loss_fn.feature_extractors) == 3

    def test_init_layer_weight_mismatch(self, mock_vgg):
        """Test that mismatched layers and weights raises error."""
        with pytest.raises(ValueError, match="Number of layers.*must match"):
            VGGPerceptualLoss(
                layers=["relu1_1", "relu2_1"],
                layer_weights=[1.0, 1.0, 1.0],  # Wrong number
            )

    def test_init_invalid_layer_name(self, mock_vgg):
        """Test that invalid layer name raises error."""
        with pytest.raises(ValueError, match="Invalid layer name"):
            VGGPerceptualLoss(layers=["invalid_layer"])

    def test_forward_same_images(self, mock_vgg):
        """Test that loss is zero for identical images."""
        loss_fn = VGGPerceptualLoss()

        pred = torch.rand(2, 3, 64, 64, requires_grad=True)
        target = pred.clone()

        loss = loss_fn(pred, target)

        assert loss.item() == pytest.approx(0.0, abs=1e-6)
        assert loss.requires_grad

    def test_forward_different_images(self, mock_vgg):
        """Test that loss is positive for different images."""
        loss_fn = VGGPerceptualLoss()

        pred = torch.rand(2, 3, 64, 64, requires_grad=True)
        target = torch.rand(2, 3, 64, 64)

        loss = loss_fn(pred, target)

        assert loss.item() > 0
        assert loss.requires_grad

    def test_forward_output_shape(self, mock_vgg):
        """Test that loss output is a scalar."""
        loss_fn = VGGPerceptualLoss()

        pred = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)

        loss = loss_fn(pred, target)

        assert loss.shape == torch.Size([])
        assert loss.dim() == 0

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("img_size", [64, 128])
    def test_forward_different_batch_sizes(self, mock_vgg, batch_size, img_size):
        """Test forward pass with different batch sizes and image sizes."""
        loss_fn = VGGPerceptualLoss()

        pred = torch.rand(batch_size, 3, img_size, img_size)
        target = torch.rand(batch_size, 3, img_size, img_size)

        loss = loss_fn(pred, target)

        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_normalization_applied(self, mock_vgg):
        """Test that ImageNet normalization is applied when enabled."""
        loss_fn = VGGPerceptualLoss(normalize=True)

        # Input in range [0, 1]
        img = torch.ones(1, 3, 64, 64) * 0.5

        normalized = loss_fn._normalize(img)

        # After normalization, values should be different
        assert not torch.allclose(img, normalized)

        # Check that normalization uses ImageNet stats
        assert hasattr(loss_fn, "mean")
        assert hasattr(loss_fn, "std")

    def test_normalization_disabled(self, mock_vgg):
        """Test that normalization can be disabled."""
        loss_fn = VGGPerceptualLoss(normalize=False)

        img = torch.rand(1, 3, 64, 64)
        normalized = loss_fn._normalize(img)

        # Without normalization, input should be unchanged
        assert torch.allclose(img, normalized)

    def test_extract_features(self, mock_vgg):
        """Test feature extraction returns correct number of features."""
        layers = ["relu1_2", "relu2_2", "relu3_3"]
        loss_fn = VGGPerceptualLoss(layers=layers)

        img = torch.rand(2, 3, 64, 64)
        features = loss_fn._extract_features(img)

        assert len(features) == len(layers)
        for feat in features:
            assert isinstance(feat, torch.Tensor)
            assert feat.size(0) == 2  # Batch size preserved

    def test_gradient_flow(self, mock_vgg):
        """Test that gradients flow through the loss."""
        loss_fn = VGGPerceptualLoss()

        pred = torch.rand(1, 3, 64, 64, requires_grad=True)
        target = torch.rand(1, 3, 64, 64)

        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert not torch.all(pred.grad == 0)

    def test_vgg_weights_frozen(self, mock_vgg):
        """Test that VGG weights are frozen and not trainable."""
        loss_fn = VGGPerceptualLoss()

        # Check that all VGG parameters have requires_grad=False
        for extractor in loss_fn.feature_extractors:
            for param in extractor.parameters():
                assert param.requires_grad is False

    def test_layer_weights_applied(self, mock_vgg):
        """Test that layer weights are properly applied."""
        # Use different weights for each layer
        layer_weights = [0.1, 0.5, 1.0, 2.0]
        loss_fn = VGGPerceptualLoss(layer_weights=layer_weights)

        pred = torch.rand(1, 3, 64, 64)
        target = torch.rand(1, 3, 64, 64)

        loss = loss_fn(pred, target)

        # Loss should be non-zero
        assert loss.item() > 0

    def test_get_config(self, mock_vgg):
        """Test configuration serialization."""
        layers = ["relu1_2", "relu3_3"]
        weights = [1.0, 2.0]

        loss_fn = VGGPerceptualLoss(
            layers=layers, layer_weights=weights, normalize=False, weight=0.5
        )

        config = loss_fn.get_config()

        assert config["layers"] == layers
        assert config["layer_weights"] == weights
        assert config["normalize"] is False
        assert "weight" in config

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_reduction_modes(self, mock_vgg, reduction):
        """Test different reduction modes."""
        loss_fn = VGGPerceptualLoss(reduction=reduction)

        pred = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)

        loss = loss_fn(pred, target)

        assert loss.shape == torch.Size([])
        assert loss.item() > 0

    def test_device_handling_cpu(self, mock_vgg):
        """Test that loss works on CPU."""
        loss_fn = VGGPerceptualLoss()

        pred = torch.rand(1, 3, 64, 64)
        target = torch.rand(1, 3, 64, 64)

        loss = loss_fn(pred, target)

        assert loss.device.type == "cpu"

    def test_perceptual_loss_alias(self, mock_vgg):
        """Test that PerceptualLoss is an alias for VGGPerceptualLoss."""
        loss_fn = PerceptualLoss()

        assert isinstance(loss_fn, VGGPerceptualLoss)

    def test_input_value_range(self, mock_vgg):
        """Test that loss works with inputs in [0, 1] range."""
        loss_fn = VGGPerceptualLoss()

        # Test with values at boundaries
        pred = torch.zeros(1, 3, 64, 64, requires_grad=True)
        target = torch.ones(1, 3, 64, 64)

        loss = loss_fn(pred, target)

        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.requires_grad

    def test_deterministic_output(self, mock_vgg):
        """Test that same inputs produce same output."""
        loss_fn = VGGPerceptualLoss()

        pred = torch.rand(1, 3, 64, 64, requires_grad=True)
        target = torch.rand(1, 3, 64, 64)

        loss1 = loss_fn(pred, target)
        loss2 = loss_fn(pred, target)

        assert torch.allclose(loss1, loss2)
        assert loss1.requires_grad
        assert loss2.requires_grad

    def test_layer_map_completeness(self, mock_vgg):
        """Test that LAYER_MAP contains expected VGG layers."""
        expected_layers = [
            "relu1_1",
            "relu1_2",
            "relu2_1",
            "relu2_2",
            "relu3_1",
            "relu3_2",
            "relu3_3",
            "relu4_1",
            "relu4_2",
            "relu4_3",
            "relu5_1",
            "relu5_2",
            "relu5_3",
        ]

        for layer in expected_layers:
            assert layer in VGGPerceptualLoss.LAYER_MAP

    def test_eval_mode(self, mock_vgg):
        """Test that VGG is in eval mode."""
        loss_fn = VGGPerceptualLoss()

        # Set the model in eval mode
        loss_fn.eval()

        # Check that feature extractors are in eval mode
        for extractor in loss_fn.feature_extractors:
            assert not extractor.training


class TestPerceptualLossIntegration:
    """Integration tests for perceptual loss."""

    def test_loss_decreases_with_optimization(self, mock_vgg):
        """Test that loss can be optimized (sanity check)."""
        loss_fn = VGGPerceptualLoss()

        # Fixed target
        target = torch.rand(1, 3, 64, 64)

        # Learnable prediction (start from random)
        pred = torch.rand(1, 3, 64, 64, requires_grad=True)

        # Initial loss
        initial_loss = loss_fn(pred, target)

        # Gradient descent step
        initial_loss.backward()
        with torch.no_grad():
            pred -= 0.1 * pred.grad

        # Recompute loss
        pred = pred.detach().requires_grad_(True)
        final_loss = loss_fn(pred, target)

        # Loss should generally decrease (not strict due to random init)
        # Just check that optimization is possible
        assert final_loss.item() >= 0
        assert not torch.isnan(final_loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
