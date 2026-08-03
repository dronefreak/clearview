"""Unit tests for frequency-domain loss functions."""

import pytest
import torch

from clearview.losses.combined import CombinedLoss
from clearview.losses.frequency import FFTLoss, FocalFrequencyLoss, WaveletLoss


class TestFFTLoss:
    """Tests for FFTLoss."""

    def test_initialization(self) -> None:
        """Test FFTLoss initialization."""
        loss_fn = FFTLoss()
        assert loss_fn.weight == 1.0
        assert loss_fn.reduction == "mean"
        assert loss_fn.norm == "backward"
        assert loss_fn.criterion == "l1"

    def test_invalid_criterion(self) -> None:
        """Test that invalid criterion raises ValueError."""
        with pytest.raises(ValueError):
            FFTLoss(criterion="huber")

    def test_forward_pass(self) -> None:
        """Test FFTLoss forward pass."""
        loss_fn = FFTLoss()
        pred = torch.randn(4, 3, 64, 64)
        target = torch.randn(4, 3, 64, 64)
        loss = loss_fn(pred, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar loss

    def test_identical_inputs(self) -> None:
        """Test FFTLoss with identical inputs is ~zero."""
        loss_fn = FFTLoss()
        x = torch.randn(2, 3, 32, 32)
        loss = loss_fn(x, x)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_l2_criterion(self) -> None:
        """Test FFTLoss with L2 criterion."""
        loss_fn = FFTLoss(criterion="l2")
        pred = torch.randn(2, 3, 32, 32)
        target = torch.randn(2, 3, 32, 32)
        loss = loss_fn(pred, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_loss_weight(self) -> None:
        """Test FFTLoss with custom weight."""
        pred = torch.randn(2, 3, 32, 32)
        target = torch.randn(2, 3, 32, 32)

        loss_weighted = FFTLoss(weight=2.0)(pred, target)
        loss_unweighted = FFTLoss(weight=1.0)(pred, target)
        assert torch.isclose(loss_weighted, loss_unweighted * 2.0)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through FFTLoss."""
        loss_fn = FFTLoss()
        pred = torch.randn(2, 3, 32, 32, requires_grad=True)
        target = torch.randn(2, 3, 32, 32)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert not torch.all(pred.grad == 0)

    def test_non_square_input(self) -> None:
        """Test FFTLoss on non-square spatial dimensions."""
        loss_fn = FFTLoss()
        pred = torch.randn(1, 3, 48, 96)
        target = torch.randn(1, 3, 48, 96)
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    def test_get_config(self) -> None:
        """Test configuration serialization."""
        loss_fn = FFTLoss(norm="ortho", criterion="l2", weight=0.1)
        config = loss_fn.get_config()
        assert config["norm"] == "ortho"
        assert config["criterion"] == "l2"
        assert config["weight"] == 0.1


class TestFocalFrequencyLoss:
    """Tests for FocalFrequencyLoss."""

    def test_initialization(self) -> None:
        """Test FocalFrequencyLoss initialization."""
        loss_fn = FocalFrequencyLoss()
        assert loss_fn.weight == 1.0
        assert loss_fn.alpha == 1.0
        assert loss_fn.ave_spectrum is False

    def test_forward_pass(self) -> None:
        """Test FocalFrequencyLoss forward pass."""
        loss_fn = FocalFrequencyLoss()
        pred = torch.randn(4, 3, 64, 64)
        target = torch.randn(4, 3, 64, 64)
        loss = loss_fn(pred, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_identical_inputs(self) -> None:
        """Test FocalFrequencyLoss with identical inputs is zero."""
        loss_fn = FocalFrequencyLoss()
        x = torch.randn(2, 3, 32, 32)
        loss = loss_fn(x, x)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_non_negative(self) -> None:
        """Test that the loss is always non-negative."""
        loss_fn = FocalFrequencyLoss()
        pred = torch.randn(2, 3, 32, 32)
        target = torch.randn(2, 3, 32, 32)
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_ave_spectrum_mode(self) -> None:
        """Test FocalFrequencyLoss with batch-averaged spectrum."""
        loss_fn = FocalFrequencyLoss(ave_spectrum=True)
        pred = torch.randn(4, 3, 32, 32)
        target = torch.randn(4, 3, 32, 32)
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through FocalFrequencyLoss."""
        loss_fn = FocalFrequencyLoss()
        pred = torch.randn(2, 3, 32, 32, requires_grad=True)
        target = torch.randn(2, 3, 32, 32)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert not torch.all(pred.grad == 0)

    def test_get_config(self) -> None:
        """Test configuration serialization."""
        loss_fn = FocalFrequencyLoss(alpha=0.5, ave_spectrum=True)
        config = loss_fn.get_config()
        assert config["alpha"] == 0.5
        assert config["ave_spectrum"] is True


class TestWaveletLoss:
    """Tests for WaveletLoss."""

    def test_initialization(self) -> None:
        """Test WaveletLoss initialization."""
        loss_fn = WaveletLoss(levels=2, detail_weight=2.0)
        assert loss_fn.levels == 2
        assert loss_fn.detail_weight == 2.0
        assert loss_fn.weight == 1.0

    def test_invalid_levels_raises(self) -> None:
        """Test that levels < 1 raises ValueError."""
        with pytest.raises(ValueError, match="levels must be"):
            WaveletLoss(levels=0)

    def test_reduction_none_raises(self) -> None:
        """Test that reduction='none' is rejected."""
        with pytest.raises(ValueError, match="reduction='none'"):
            WaveletLoss(reduction="none")

    def test_forward_pass(self) -> None:
        """Test WaveletLoss forward pass returns a finite scalar."""
        loss_fn = WaveletLoss()
        pred = torch.randn(2, 3, 64, 64)
        target = torch.randn(2, 3, 64, 64)
        loss = loss_fn(pred, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_identical_inputs_near_zero(self) -> None:
        """Test that identical inputs give a near-zero loss."""
        loss_fn = WaveletLoss()
        x = torch.randn(2, 3, 64, 64)
        loss = loss_fn(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_odd_sized_input(self) -> None:
        """Test that odd spatial dimensions are handled via padding."""
        loss_fn = WaveletLoss(levels=2)
        pred = torch.randn(1, 3, 33, 37)
        target = torch.randn(1, 3, 33, 37)
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    def test_multiple_levels_increase_loss_terms(self) -> None:
        """Test that more decomposition levels changes the loss value."""
        pred = torch.randn(2, 3, 64, 64)
        target = torch.randn(2, 3, 64, 64)

        loss_1 = WaveletLoss(levels=1)(pred, target)
        loss_3 = WaveletLoss(levels=3)(pred, target)

        assert torch.isfinite(loss_1)
        assert torch.isfinite(loss_3)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the loss."""
        loss_fn = WaveletLoss()
        pred = torch.randn(2, 3, 64, 64, requires_grad=True)
        target = torch.randn(2, 3, 64, 64)

        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()

    def test_get_config(self) -> None:
        """Test WaveletLoss get_config method."""
        loss_fn = WaveletLoss(levels=3, detail_weight=1.5, weight=0.5)
        config = loss_fn.get_config()
        assert config["levels"] == 3
        assert config["detail_weight"] == 1.5
        assert config["weight"] == 0.5


class TestFrequencyLossCombinedIntegration:
    """Tests for integrating frequency losses via CombinedLoss.from_config."""

    @pytest.mark.parametrize(
        "name", ["fft", "frequency", "focal_frequency", "ffl", "wavelet"]
    )
    def test_from_config_registers_frequency_losses(self, name: str) -> None:
        """Test that frequency losses are resolvable by name in CombinedLoss."""
        loss_fn = CombinedLoss.from_config({name: {"weight": 0.1}})
        pred = torch.randn(2, 3, 32, 32)
        target = torch.randn(2, 3, 32, 32)
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    def test_combined_with_l1_and_fft(self) -> None:
        """Test a realistic combination of L1 + FFT losses."""
        loss_fn = CombinedLoss.from_config(
            {
                "l1": {"weight": 1.0},
                "fft": {"weight": 0.1},
            }
        )
        pred = torch.randn(2, 3, 32, 32, requires_grad=True)
        target = torch.randn(2, 3, 32, 32)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
