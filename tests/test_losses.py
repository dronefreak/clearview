"""Unit tests for loss functions."""

import pytest
import torch

from clearview.losses.pixel import CharbonnierLoss, L1Loss, L2Loss, MAELoss, MSELoss
from clearview.losses.structural import MultiScaleSSIMLoss, SSIMLoss


class TestL1Loss:
    """Tests for L1Loss."""

    def test_initialization(self) -> None:
        """Test L1Loss initialization."""
        loss_fn = L1Loss()
        assert loss_fn.weight == 1.0
        assert loss_fn.reduction == "mean"

    def test_forward_pass(self) -> None:
        """Test L1Loss forward pass."""
        loss_fn = L1Loss()
        pred = torch.randn(4, 3, 256, 256)
        target = torch.randn(4, 3, 256, 256)
        loss = loss_fn(pred, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar loss

    def test_identical_inputs(self) -> None:
        """Test L1Loss with identical inputs."""
        loss_fn = L1Loss()
        x = torch.randn(4, 3, 256, 256)
        loss = loss_fn(x, x)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_loss_weight(self) -> None:
        """Test L1Loss with custom weight."""
        loss_fn = L1Loss(weight=2.0)
        pred = torch.randn(4, 3, 256, 256)
        target = torch.randn(4, 3, 256, 256)
        loss = loss_fn(pred, target)

        # Compare with unweighted loss
        loss_fn_unweighted = L1Loss(weight=1.0)
        loss_unweighted = loss_fn_unweighted(pred, target)
        assert torch.isclose(loss, loss_unweighted * 2.0)

    def test_reduction_none(self) -> None:
        """Test L1Loss with reduction='none'."""
        loss_fn = L1Loss(reduction="none")
        pred = torch.randn(4, 3, 256, 256)
        target = torch.randn(4, 3, 256, 256)
        loss = loss_fn(pred, target)
        assert loss.shape == pred.shape

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through L1Loss."""
        loss_fn = L1Loss()
        pred = torch.randn(4, 3, 256, 256, requires_grad=True)
        target = torch.randn(4, 3, 256, 256)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None


class TestMAELoss:
    """Tests for MAELoss (alias for L1Loss)."""

    def test_initialization(self) -> None:
        """Test MAELoss initialization."""
        loss_fn = MAELoss()
        assert isinstance(loss_fn, L1Loss)

    def test_identical_to_l1(self) -> None:
        """Test that MAELoss produces same result as L1Loss."""
        mae_loss = MAELoss()
        l1_loss = L1Loss()

        pred = torch.randn(4, 3, 256, 256)
        target = torch.randn(4, 3, 256, 256)

        mae_result = mae_loss(pred, target)
        l1_result = l1_loss(pred, target)

        assert torch.isclose(mae_result, l1_result)


class TestL2Loss:
    """Tests for L2Loss."""

    def test_initialization(self) -> None:
        """Test L2Loss initialization."""
        loss_fn = L2Loss()
        assert loss_fn.weight == 1.0
        assert loss_fn.reduction == "mean"

    def test_forward_pass(self) -> None:
        """Test L2Loss forward pass."""
        loss_fn = L2Loss()
        pred = torch.randn(4, 3, 256, 256)
        target = torch.randn(4, 3, 256, 256)
        loss = loss_fn(pred, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_identical_inputs(self) -> None:
        """Test L2Loss with identical inputs."""
        loss_fn = L2Loss()
        x = torch.randn(4, 3, 256, 256)
        loss = loss_fn(x, x)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_greater_than_l1_for_large_errors(self) -> None:
        """Test that L2 penalizes large errors more than L1."""
        l1_loss = L1Loss()
        l2_loss = L2Loss()

        # Create predictions with large error
        pred = torch.tensor([[0.0]])
        target = torch.tensor([[10.0]])

        l1_result = l1_loss(pred, target)
        l2_result = l2_loss(pred, target)

        # L2 = 100, L1 = 10
        assert l2_result > l1_result


class TestMSELoss:
    """Tests for MSELoss (alias for L2Loss)."""

    def test_initialization(self) -> None:
        """Test MSELoss initialization."""
        loss_fn = MSELoss()
        assert isinstance(loss_fn, L2Loss)

    def test_identical_to_l2(self) -> None:
        """Test that MSELoss produces same result as L2Loss."""
        mse_loss = MSELoss()
        l2_loss = L2Loss()

        pred = torch.randn(4, 3, 256, 256)
        target = torch.randn(4, 3, 256, 256)

        mse_result = mse_loss(pred, target)
        l2_result = l2_loss(pred, target)

        assert torch.isclose(mse_result, l2_result)


class TestCharbonnierLoss:
    """Tests for CharbonnierLoss."""

    def test_initialization(self) -> None:
        """Test CharbonnierLoss initialization."""
        loss_fn = CharbonnierLoss(epsilon=1e-3)
        assert loss_fn.epsilon == 1e-3
        assert loss_fn.weight == 1.0

    def test_forward_pass(self) -> None:
        """Test CharbonnierLoss forward pass."""
        loss_fn = CharbonnierLoss()
        pred = torch.randn(4, 3, 256, 256)
        target = torch.randn(4, 3, 256, 256)
        loss = loss_fn(pred, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_identical_inputs(self) -> None:
        """Test CharbonnierLoss with identical inputs."""
        loss_fn = CharbonnierLoss(epsilon=1e-6)
        x = torch.randn(4, 3, 256, 256)
        loss = loss_fn(x, x)
        # Should be close to epsilon due to smoothing
        assert loss < 1e-4

    def test_smooth_around_zero(self) -> None:
        """Test that CharbonnierLoss is smooth around zero."""
        loss_fn = CharbonnierLoss(epsilon=1e-3)
        pred = torch.tensor([[0.0]], requires_grad=True)
        target = torch.tensor([[0.0]])
        loss = loss_fn(pred, target)
        loss.backward()

        # Gradient should exist and be finite
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()

    def test_different_epsilon_values(self) -> None:
        """Test CharbonnierLoss with different epsilon values."""
        pred = torch.randn(4, 3, 256, 256)
        target = torch.randn(4, 3, 256, 256)

        loss_fn_small = CharbonnierLoss(epsilon=1e-6)
        loss_fn_large = CharbonnierLoss(epsilon=1e-3)

        loss_small = loss_fn_small(pred, target)
        loss_large = loss_fn_large(pred, target)

        # Larger epsilon should result in larger minimum loss
        assert isinstance(loss_small, torch.Tensor)
        assert isinstance(loss_large, torch.Tensor)

    def test_get_config(self) -> None:
        """Test CharbonnierLoss get_config method."""
        loss_fn = CharbonnierLoss(epsilon=1e-3, weight=2.0)
        config = loss_fn.get_config()
        assert config["epsilon"] == 1e-3
        assert config["weight"] == 2.0


class TestSSIMLoss:
    """Tests for SSIMLoss."""

    def test_initialization(self) -> None:
        """Test SSIMLoss initialization."""
        loss_fn = SSIMLoss(window_size=11, channel=3)
        assert loss_fn.window_size == 11
        assert loss_fn.channel == 3

    def test_forward_pass(self) -> None:
        """Test SSIMLoss forward pass."""
        loss_fn = SSIMLoss(channel=3)
        pred = torch.randn(2, 3, 256, 256)
        target = torch.randn(2, 3, 256, 256)
        loss = loss_fn(pred, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_identical_inputs(self) -> None:
        """Test SSIMLoss with identical inputs."""
        loss_fn = SSIMLoss(channel=3)
        x = torch.randn(2, 3, 256, 256)
        loss = loss_fn(x, x)
        # SSIM of identical images should be 1, so loss should be ~0
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_loss_range(self) -> None:
        """Test that SSIMLoss is in valid range [0, 2]."""
        loss_fn = SSIMLoss(channel=3)
        pred = torch.randn(2, 3, 256, 256)
        target = torch.randn(2, 3, 256, 256)
        loss = loss_fn(pred, target)
        # Loss = 1 - SSIM, where SSIM in [-1, 1], so loss in [0, 2]
        assert loss >= 0
        assert loss <= 2

    def test_different_window_sizes(self) -> None:
        """Test SSIMLoss with different window sizes."""
        pred = torch.randn(2, 3, 256, 256)
        target = torch.randn(2, 3, 256, 256)

        for window_size in [5, 7, 11]:
            loss_fn = SSIMLoss(window_size=window_size, channel=3)
            loss = loss_fn(pred, target)
            assert isinstance(loss, torch.Tensor)

    def test_single_channel(self) -> None:
        """Test SSIMLoss with single channel images."""
        loss_fn = SSIMLoss(channel=1)
        pred = torch.randn(2, 1, 256, 256)
        target = torch.randn(2, 1, 256, 256)
        loss = loss_fn(pred, target)
        assert isinstance(loss, torch.Tensor)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through SSIMLoss."""
        loss_fn = SSIMLoss(channel=3)
        pred = torch.randn(2, 3, 256, 256, requires_grad=True)
        target = torch.randn(2, 3, 256, 256)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None

    def test_get_config(self) -> None:
        """Test SSIMLoss get_config method."""
        loss_fn = SSIMLoss(window_size=11, sigma=1.5, channel=3, weight=2.0)
        config = loss_fn.get_config()
        assert config["window_size"] == 11
        assert config["sigma"] == 1.5
        assert config["channel"] == 3
        assert config["weight"] == 2.0


class TestMultiScaleSSIMLoss:
    """Tests for MultiScaleSSIMLoss."""

    def test_initialization(self) -> None:
        """Test MultiScaleSSIMLoss initialization."""
        loss_fn = MultiScaleSSIMLoss(scales=3, channel=3)
        assert loss_fn.scales == 3
        assert loss_fn.channel == 3

    def test_initialization_custom_weights(self) -> None:
        """Test MultiScaleSSIMLoss with custom weights."""
        weights = [0.2, 0.3, 0.5]
        loss_fn = MultiScaleSSIMLoss(scales=3, weights=weights, channel=3)
        assert torch.allclose(loss_fn.weights, torch.tensor(weights))

    def test_forward_pass(self) -> None:
        """Test MultiScaleSSIMLoss forward pass."""
        loss_fn = MultiScaleSSIMLoss(scales=3, channel=3)
        # Need larger images for multi-scale
        pred = torch.randn(2, 3, 256, 256)
        target = torch.randn(2, 3, 256, 256)
        loss = loss_fn(pred, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_identical_inputs(self) -> None:
        """Test MultiScaleSSIMLoss with identical inputs."""
        loss_fn = MultiScaleSSIMLoss(scales=3, channel=3)
        x = torch.randn(2, 3, 256, 256)
        loss = loss_fn(x, x)
        # MS-SSIM of identical images should be 1, so loss should be ~0
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-4)

    def test_different_scales(self) -> None:
        """Test MultiScaleSSIMLoss with different number of scales."""
        pred = torch.randn(2, 3, 256, 256)
        target = torch.randn(2, 3, 256, 256)

        for scales in [2, 3, 4]:
            loss_fn = MultiScaleSSIMLoss(scales=scales, channel=3)
            loss = loss_fn(pred, target)
            assert isinstance(loss, torch.Tensor)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through MultiScaleSSIMLoss."""
        loss_fn = MultiScaleSSIMLoss(scales=3, channel=3)
        pred = torch.randn(2, 3, 256, 256, requires_grad=True)
        target = torch.randn(2, 3, 256, 256)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None

    def test_get_config(self) -> None:
        """Test MultiScaleSSIMLoss get_config method."""
        weights = [0.2, 0.3, 0.5]
        loss_fn = MultiScaleSSIMLoss(
            window_size=11, sigma=1.5, channel=3, scales=3, weights=weights
        )
        config = loss_fn.get_config()
        assert config["window_size"] == 11
        assert config["sigma"] == 1.5
        assert config["channel"] == 3
        assert config["scales"] == 3
        assert weights == pytest.approx(config["weights"], rel=1e-6)
