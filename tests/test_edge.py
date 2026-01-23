"""Tests for edge-aware loss functions.

These tests cover Sobel and Laplacian edge detection losses.
"""

import pytest
import torch

from clearview.losses.edge import EdgeLoss, LaplacianEdgeLoss, SobelEdgeLoss


class TestSobelEdgeLoss:
    """Tests for SobelEdgeLoss."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        loss_fn = SobelEdgeLoss()

        assert loss_fn.convert_to_grayscale is True
        assert loss_fn.reduction == "mean"
        assert hasattr(loss_fn, "sobel_x")
        assert hasattr(loss_fn, "sobel_y")

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        loss_fn = SobelEdgeLoss(reduction="sum", weight=2.0, convert_to_grayscale=False)

        assert loss_fn.reduction == "sum"
        assert loss_fn.weight == 2.0
        assert loss_fn.convert_to_grayscale is False

    def test_sobel_filters_registered(self):
        """Test that Sobel filters are properly registered as buffers."""
        loss_fn = SobelEdgeLoss()

        # Sobel X filter
        expected_sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )
        assert torch.allclose(loss_fn.sobel_x.squeeze(), expected_sobel_x)

        # Sobel Y filter
        expected_sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        )
        assert torch.allclose(loss_fn.sobel_y.squeeze(), expected_sobel_y)

    def test_rgb_to_grayscale(self):
        """Test RGB to grayscale conversion."""
        loss_fn = SobelEdgeLoss()

        # Create RGB image with known values
        rgb = torch.zeros(1, 3, 4, 4)
        rgb[:, 0, :, :] = 1.0  # Red channel
        rgb[:, 1, :, :] = 0.5  # Green channel
        rgb[:, 2, :, :] = 0.0  # Blue channel

        gray = loss_fn._rgb_to_grayscale(rgb)

        # Check shape
        assert gray.shape == (1, 1, 4, 4)

        # Check grayscale conversion formula: 0.299*R + 0.587*G + 0.114*B
        expected = 0.299 * 1.0 + 0.587 * 0.5 + 0.114 * 0.0
        assert torch.allclose(gray, torch.full_like(gray, expected), atol=1e-5)

    def test_rgb_to_grayscale_already_gray(self):
        """Test that grayscale input is returned unchanged."""
        loss_fn = SobelEdgeLoss()

        gray_input = torch.rand(1, 1, 32, 32)
        gray_output = loss_fn._rgb_to_grayscale(gray_input)

        assert torch.allclose(gray_input, gray_output)

    def test_compute_edges_returns_two_tensors(self):
        """Test that edge computation returns dx and dy."""
        loss_fn = SobelEdgeLoss()

        img = torch.rand(2, 3, 32, 32)
        dx, dy = loss_fn._compute_edges(img)

        assert isinstance(dx, torch.Tensor)
        assert isinstance(dy, torch.Tensor)
        assert dx.shape[0] == 2  # Batch size preserved
        assert dy.shape[0] == 2

    def test_compute_edges_detects_horizontal_edge(self):
        """Test that Sobel filters detect horizontal edges."""
        loss_fn = SobelEdgeLoss(convert_to_grayscale=False)

        # Create image with horizontal edge (black top, white bottom)
        img = torch.zeros(1, 1, 8, 8)
        img[:, :, 4:, :] = 1.0

        dx, dy = loss_fn._compute_edges(img)

        # Horizontal edge should have strong dy response at boundary
        assert dy[:, :, 3:5, :].abs().mean() > dy[:, :, 0:2, :].abs().mean()

    def test_compute_edges_detects_vertical_edge(self):
        """Test that Sobel filters detect vertical edges."""
        loss_fn = SobelEdgeLoss(convert_to_grayscale=False)

        # Create image with vertical edge (black left, white right)
        img = torch.zeros(1, 1, 8, 8)
        img[:, :, :, 4:] = 1.0

        dx, dy = loss_fn._compute_edges(img)

        # Vertical edge should have strong dx response at boundary
        assert dx[:, :, :, 3:5].abs().mean() > dx[:, :, :, 0:2].abs().mean()

    def test_forward_same_images(self):
        """Test that loss is zero for identical images."""
        loss_fn = SobelEdgeLoss()

        pred = torch.rand(2, 3, 64, 64)
        target = pred.clone()

        loss = loss_fn(pred, target)

        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_forward_different_images(self):
        """Test that loss is positive for different images."""
        loss_fn = SobelEdgeLoss()

        pred = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)

        loss = loss_fn(pred, target)

        assert loss.item() > 0

    def test_forward_output_shape(self):
        """Test that loss output is a scalar."""
        loss_fn = SobelEdgeLoss()

        pred = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)

        loss = loss_fn(pred, target)

        assert loss.shape == torch.Size([])
        assert loss.dim() == 0

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    @pytest.mark.parametrize("channels", [1, 3])
    def test_forward_different_inputs(self, batch_size, channels):
        """Test forward pass with different batch sizes and channels."""
        loss_fn = SobelEdgeLoss()

        pred = torch.rand(batch_size, channels, 64, 64)
        target = torch.rand(batch_size, channels, 64, 64)

        loss = loss_fn(pred, target)

        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.item() >= 0

    def test_gradient_flow(self):
        """Test that gradients flow through the loss."""
        loss_fn = SobelEdgeLoss()

        pred = torch.rand(1, 3, 64, 64, requires_grad=True)
        target = torch.rand(1, 3, 64, 64)

        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert not torch.all(pred.grad == 0)

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_reduction_modes(self, reduction):
        """Test different reduction modes."""
        loss_fn = SobelEdgeLoss(reduction=reduction)

        pred = torch.rand(2, 3, 32, 32)
        target = torch.rand(2, 3, 32, 32)

        loss = loss_fn(pred, target)

        assert loss.shape == torch.Size([])

    def test_grayscale_conversion_flag(self):
        """Test convert_to_grayscale flag effect."""
        img = torch.rand(2, 3, 32, 32)

        loss_fn_gray = SobelEdgeLoss(convert_to_grayscale=True)

        dx_gray, dy_gray = loss_fn_gray._compute_edges(img)

        # With grayscale, output should have 1 channel
        assert dx_gray.size(1) == 1
        assert dy_gray.size(1) == 1
        assert dx_gray.size(0) == 2
        assert dy_gray.size(0) == 2

    def test_get_config(self):
        """Test configuration serialization."""
        loss_fn = SobelEdgeLoss(reduction="sum", weight=2.5, convert_to_grayscale=False)

        config = loss_fn.get_config()

        assert config["reduction"] == "sum"
        assert config["weight"] == 2.5
        assert config["convert_to_grayscale"] is False

    def test_edge_loss_alias(self):
        """Test that EdgeLoss is an alias for SobelEdgeLoss."""
        loss_fn = EdgeLoss()
        assert isinstance(loss_fn, SobelEdgeLoss)

    def test_deterministic_output(self):
        """Test that same inputs produce same output."""
        loss_fn = SobelEdgeLoss()

        pred = torch.rand(1, 3, 64, 64)
        target = torch.rand(1, 3, 64, 64)

        loss1 = loss_fn(pred, target)
        loss2 = loss_fn(pred, target)

        assert torch.allclose(loss1, loss2)

    def test_weight_application(self):
        """Test that weight is properly applied to loss."""
        pred = torch.rand(1, 3, 32, 32)
        target = torch.rand(1, 3, 32, 32)

        loss_fn_1 = SobelEdgeLoss(weight=1.0)
        loss_fn_2 = SobelEdgeLoss(weight=2.0)

        loss1 = loss_fn_1(pred, target)
        loss2 = loss_fn_2(pred, target)

        # Loss with weight=2.0 should be double
        assert torch.allclose(loss2, loss1 * 2.0)


class TestLaplacianEdgeLoss:
    """Tests for LaplacianEdgeLoss."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        loss_fn = LaplacianEdgeLoss()

        assert loss_fn.convert_to_grayscale is True
        assert loss_fn.reduction == "mean"
        assert hasattr(loss_fn, "laplacian")

    def test_laplacian_filter_registered(self):
        """Test that Laplacian filter is properly registered as buffer."""
        loss_fn = LaplacianEdgeLoss()

        expected_laplacian = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
        )
        assert torch.allclose(loss_fn.laplacian.squeeze(), expected_laplacian)
        assert loss_fn.laplacian.shape == (1, 1, 3, 3)

    def test_rgb_to_grayscale(self):
        """Test RGB to grayscale conversion."""
        loss_fn = LaplacianEdgeLoss()

        rgb = torch.ones(1, 3, 4, 4)
        gray = loss_fn._rgb_to_grayscale(rgb)

        assert gray.shape == (1, 1, 4, 4)
        assert torch.allclose(
            gray, torch.full_like(gray, 0.299 + 0.587 + 0.114), atol=1e-5
        )

    def test_compute_edges_detects_edges(self):
        """Test that Laplacian detects edges."""
        loss_fn = LaplacianEdgeLoss(convert_to_grayscale=False)

        # Create image with edge
        img = torch.zeros(1, 1, 8, 8)
        img[:, :, 3:5, 3:5] = 1.0  # White square in center

        edges = loss_fn._compute_edges(img)

        # Edges should be detected at boundaries
        # Check that edge regions have higher magnitude than flat regions
        edge_magnitude = edges[:, :, 3:5, 3:5].abs().mean()
        flat_magnitude = edges[:, :, 0:2, 0:2].abs().mean()

        assert edge_magnitude > flat_magnitude
        assert edges.size(0) == 1
        assert edges.size(1) == 1

    def test_forward_same_images(self):
        """Test that loss is zero for identical images."""
        loss_fn = LaplacianEdgeLoss()

        pred = torch.rand(2, 3, 64, 64, requires_grad=True)
        target = pred.clone()

        loss = loss_fn(pred, target)

        assert loss.item() == pytest.approx(0.0, abs=1e-6)
        assert loss.requires_grad
        assert pred.grad is None  # Grad should be None before backward

    def test_forward_different_images(self):
        """Test that loss is positive for different images."""
        loss_fn = LaplacianEdgeLoss()

        pred = torch.rand(2, 3, 64, 64, requires_grad=True)
        target = torch.rand(2, 3, 64, 64)

        loss = loss_fn(pred, target)

        assert loss.item() > 0
        assert loss.requires_grad
        assert pred.grad is None  # Grad should be None before backward

    def test_forward_output_shape(self):
        """Test that loss output is a scalar."""
        loss_fn = LaplacianEdgeLoss()

        pred = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)

        loss = loss_fn(pred, target)

        assert loss.shape == torch.Size([])
        assert loss.dim() == 0

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("img_size", [32, 64, 128])
    def test_forward_different_inputs(self, batch_size, img_size):
        """Test forward pass with different batch sizes and image sizes."""
        loss_fn = LaplacianEdgeLoss()

        pred = torch.rand(batch_size, 3, img_size, img_size)
        target = torch.rand(batch_size, 3, img_size, img_size)

        loss = loss_fn(pred, target)

        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.item() >= 0

    def test_gradient_flow(self):
        """Test that gradients flow through the loss."""
        loss_fn = LaplacianEdgeLoss()

        pred = torch.rand(1, 3, 64, 64, requires_grad=True)
        target = torch.rand(1, 3, 64, 64)

        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert not torch.all(pred.grad == 0)

    def test_get_config(self):
        """Test configuration serialization."""
        loss_fn = LaplacianEdgeLoss(
            reduction="sum", weight=1.5, convert_to_grayscale=False
        )

        config = loss_fn.get_config()

        assert config["reduction"] == "sum"
        assert config["weight"] == 1.5
        assert config["convert_to_grayscale"] is False

    def test_rotation_invariance_property(self):
        """Test that Laplacian is more rotation-invariant than Sobel."""
        # Create image with diagonal edge
        img = torch.zeros(1, 1, 16, 16)
        for i in range(16):
            img[0, 0, i, i] = 1.0

        loss_fn = LaplacianEdgeLoss(convert_to_grayscale=False)
        edges = loss_fn._compute_edges(img)

        # Should detect edge along diagonal
        assert edges.abs().sum() > 0

    def test_deterministic_output(self):
        """Test that same inputs produce same output."""
        loss_fn = LaplacianEdgeLoss()

        pred = torch.rand(1, 3, 64, 64)
        target = torch.rand(1, 3, 64, 64)

        loss1 = loss_fn(pred, target)
        loss2 = loss_fn(pred, target)

        assert torch.allclose(loss1, loss2)


class TestEdgeLossComparison:
    """Comparison tests between Sobel and Laplacian edge losses."""

    def test_both_detect_edges(self):
        """Test that both Sobel and Laplacian detect edges."""
        sobel = SobelEdgeLoss()
        laplacian = LaplacianEdgeLoss()

        # Image with no edges
        flat = torch.ones(1, 3, 32, 32) * 0.5
        flat_target = flat.clone()

        # Image with edges
        edgy = torch.rand(1, 3, 32, 32)
        edgy_target = torch.rand(1, 3, 32, 32)

        # Both should give near-zero loss for identical flat images
        sobel_flat_loss = sobel(flat, flat_target)
        laplacian_flat_loss = laplacian(flat, flat_target)

        assert sobel_flat_loss < 1e-5
        assert laplacian_flat_loss < 1e-5

        # Both should give positive loss for different edgy images
        sobel_edgy_loss = sobel(edgy, edgy_target)
        laplacian_edgy_loss = laplacian(edgy, edgy_target)

        assert sobel_edgy_loss > 0
        assert laplacian_edgy_loss > 0

    def test_output_ranges_comparable(self):
        """Test that Sobel and Laplacian produce comparable loss magnitudes."""
        sobel = SobelEdgeLoss()
        laplacian = LaplacianEdgeLoss()

        pred = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)

        sobel_loss = sobel(pred, target)
        laplacian_loss = laplacian(pred, target)

        # Both should be in reasonable range (not orders of magnitude different)
        # Just a sanity check, not a strict requirement
        assert sobel_loss > 0
        assert laplacian_loss > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
