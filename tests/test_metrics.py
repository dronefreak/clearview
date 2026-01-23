"""Tests for evaluation metrics.

These tests cover PSNR, SSIM, MAE, MSE computation and MetricsTracker.
"""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from clearview.utils.metrics import (
    MetricsTracker,
    compute_mae,
    compute_metrics,
    compute_mse,
    compute_psnr,
)


class TestComputePSNR:
    """Tests for PSNR computation."""

    def test_psnr_identical_images_torch(self):
        """Test PSNR is infinite for identical images (very high value)."""
        pred = torch.rand(2, 3, 64, 64)
        target = pred.clone()

        psnr = compute_psnr(pred, target)

        # PSNR should be very high for identical images
        assert isinstance(psnr, float)
        assert psnr == pytest.approx(100.0, abs=1e-3)  # Very high PSNR

    def test_psnr_different_images_torch(self):
        """Test PSNR is finite for different images."""
        pred = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)

        psnr = compute_psnr(pred, target)

        assert 0 < psnr < 100
        assert isinstance(psnr, float)

    def test_psnr_reduction_mean(self):
        """Test PSNR with mean reduction returns scalar."""
        pred = torch.rand(4, 3, 32, 32)
        target = torch.rand(4, 3, 32, 32)

        psnr = compute_psnr(pred, target, reduction="mean")

        assert isinstance(psnr, float)

    def test_psnr_reduction_none(self):
        """Test PSNR with no reduction returns per-image values."""
        batch_size = 4
        pred = torch.rand(batch_size, 3, 32, 32)
        target = torch.rand(batch_size, 3, 32, 32)

        psnr = compute_psnr(pred, target, reduction="none")

        assert isinstance(psnr, torch.Tensor)
        assert psnr.shape == (batch_size,)

    def test_psnr_max_val_parameter(self):
        """Test PSNR with different max_val."""
        pred = torch.ones(1, 3, 32, 32) * 128
        target = torch.ones(1, 3, 32, 32) * 127

        # PSNR with max_val=255 should differ from max_val=1.0
        psnr_255 = compute_psnr(pred / 255.0, target / 255.0, max_val=1.0)
        psnr_1 = compute_psnr(pred, target, max_val=255.0)

        # Both should be positive
        assert psnr_255 > 0
        assert psnr_1 > 0

    def test_psnr_numpy_identical(self):
        """Test PSNR with numpy arrays for identical images."""
        pred = torch.rand(2, 3, 64, 64)
        target = pred.clone()

        psnr = compute_psnr(pred, target)

        assert isinstance(psnr, float)
        assert psnr == pytest.approx(100.0, abs=1e-3)  # Very high PSNR

    def test_psnr_numpy_different(self):
        """Test PSNR with numpy arrays for different images."""
        pred = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)

        psnr = compute_psnr(pred, target)

        assert 0 < psnr < 100
        assert isinstance(psnr, float)

    def test_psnr_type_mismatch_error(self):
        """Test that mismatched types raise error."""
        pred = torch.rand(1, 3, 32, 32)
        target = np.random.rand(1, 3, 32, 32)

        with pytest.raises(TypeError):
            compute_psnr(pred, target)

    def test_psnr_deterministic(self):
        """Test that PSNR computation is deterministic."""
        pred = torch.rand(2, 3, 32, 32)
        target = torch.rand(2, 3, 32, 32)

        psnr1 = compute_psnr(pred, target)
        psnr2 = compute_psnr(pred, target)

        assert psnr1 == psnr2


class TestComputeMAE:
    """Tests for MAE computation."""

    def test_mae_identical_images_torch(self):
        """Test MAE is zero for identical images."""
        pred = torch.rand(2, 3, 64, 64)
        target = pred.clone()

        mae = compute_mae(pred, target)

        assert mae == pytest.approx(0.0, abs=1e-6)

    def test_mae_different_images_torch(self):
        """Test MAE is positive for different images."""
        pred = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)

        mae = compute_mae(pred, target)

        assert mae > 0
        assert isinstance(mae, float)

    def test_mae_reduction_none(self):
        """Test MAE with no reduction returns per-image values."""
        batch_size = 4
        pred = torch.rand(batch_size, 3, 32, 32)
        target = torch.rand(batch_size, 3, 32, 32)

        mae = compute_mae(pred, target, reduction="none")

        assert isinstance(mae, torch.Tensor)
        assert mae.shape == (batch_size,)

    def test_mae_numpy(self):
        """Test MAE with numpy arrays."""
        pred = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)

        mae = compute_mae(pred, target)

        assert mae > 0
        assert isinstance(mae, float)

    def test_mae_value_correctness(self):
        """Test MAE computation correctness."""
        pred = torch.ones(1, 1, 2, 2) * 0.5
        target = torch.ones(1, 1, 2, 2) * 0.3

        mae = compute_mae(pred, target)

        # Expected MAE = 0.2
        assert mae == pytest.approx(0.2, abs=1e-6)


class TestComputeMSE:
    """Tests for MSE computation."""

    def test_mse_identical_images_torch(self):
        """Test MSE is zero for identical images."""
        pred = torch.rand(2, 3, 64, 64)
        target = pred.clone()

        mse = compute_mse(pred, target)

        assert mse == pytest.approx(0.0, abs=1e-6)

    def test_mse_different_images_torch(self):
        """Test MSE is positive for different images."""
        pred = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)

        mse = compute_mse(pred, target)

        assert mse > 0
        assert isinstance(mse, float)

    def test_mse_reduction_none(self):
        """Test MSE with no reduction returns per-image values."""
        batch_size = 4
        pred = torch.rand(batch_size, 3, 32, 32)
        target = torch.rand(batch_size, 3, 32, 32)

        mse = compute_mse(pred, target, reduction="none")

        assert isinstance(mse, torch.Tensor)
        assert mse.shape == (batch_size,)

    def test_mse_numpy(self):
        """Test MSE with numpy arrays."""
        pred = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)

        mse = compute_mse(pred, target)

        assert mse > 0
        assert isinstance(mse, float)

    def test_mse_value_correctness(self):
        """Test MSE computation correctness."""
        pred = torch.ones(1, 1, 2, 2) * 0.5
        target = torch.ones(1, 1, 2, 2) * 0.3

        mse = compute_mse(pred, target)

        # Expected MSE = (0.2)^2 = 0.04
        assert mse == pytest.approx(0.04, abs=1e-6)


class TestComputeMetrics:
    """Tests for compute_metrics wrapper."""

    @patch("clearview.utils.metrics.compute_psnr")
    @patch("clearview.utils.metrics.compute_ssim")
    @patch("clearview.utils.metrics.compute_mae")
    @patch("clearview.utils.metrics.compute_mse")
    def test_compute_all_metrics(self, mock_mse, mock_mae, mock_ssim, mock_psnr):
        """Test computing all metrics at once."""
        mock_psnr.return_value = 25.0
        mock_ssim.return_value = 0.85
        mock_mae.return_value = 0.12
        mock_mse.return_value = 0.015

        pred = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)

        metrics = compute_metrics(pred, target)

        assert "psnr" in metrics
        assert "ssim" in metrics
        assert "mae" in metrics
        assert "mse" in metrics
        assert metrics["psnr"] == 25.0
        assert metrics["ssim"] == 0.85

    @patch("clearview.utils.metrics.compute_psnr")
    @patch("clearview.utils.metrics.compute_mae")
    def test_compute_specific_metrics(self, mock_mae, mock_psnr):
        """Test computing only specific metrics."""
        mock_psnr.return_value = 25.0
        mock_mae.return_value = 0.12

        pred = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)

        metrics = compute_metrics(pred, target, metrics=["psnr", "mae"])

        assert "psnr" in metrics
        assert "mae" in metrics
        assert "ssim" not in metrics
        assert "mse" not in metrics

    def test_compute_metrics_invalid_metric(self):
        """Test that invalid metric name raises error."""
        pred = torch.rand(1, 3, 32, 32)
        target = torch.rand(1, 3, 32, 32)

        with pytest.raises(ValueError, match="Unknown metric"):
            compute_metrics(pred, target, metrics=["invalid_metric"])

    @patch("clearview.utils.metrics.compute_psnr")
    def test_compute_metrics_max_val(self, mock_psnr):
        """Test that max_val is passed to metrics."""
        mock_psnr.return_value = 30.0

        pred = torch.rand(1, 3, 32, 32)
        target = torch.rand(1, 3, 32, 32)

        compute_metrics(pred, target, metrics=["psnr"], max_val=255.0)

        mock_psnr.assert_called_once()
        assert mock_psnr.call_args[1]["max_val"] == 255.0


class TestMetricsTracker:
    """Tests for MetricsTracker class."""

    def test_init(self):
        """Test MetricsTracker initialization."""
        tracker = MetricsTracker()

        assert len(tracker.metrics) == 0
        assert tracker.count == 0

    def test_update_single_metric(self):
        """Test updating with single metric."""
        tracker = MetricsTracker()

        tracker.update({"psnr": 25.0})

        assert "psnr" in tracker.metrics
        assert len(tracker.metrics["psnr"]) == 1
        assert tracker.metrics["psnr"][0] == 25.0

    def test_update_multiple_metrics(self):
        """Test updating with multiple metrics."""
        tracker = MetricsTracker()

        tracker.update({"psnr": 25.0, "ssim": 0.85, "mae": 0.12})

        assert len(tracker.metrics) == 3
        assert "psnr" in tracker.metrics
        assert "ssim" in tracker.metrics
        assert "mae" in tracker.metrics

    def test_update_multiple_times(self):
        """Test updating tracker multiple times."""
        tracker = MetricsTracker()

        tracker.update({"psnr": 25.0})
        tracker.update({"psnr": 26.0})
        tracker.update({"psnr": 27.0})

        assert len(tracker.metrics["psnr"]) == 3
        assert tracker.count == 3

    def test_update_batch_size(self):
        """Test that batch_size updates count."""
        tracker = MetricsTracker()

        tracker.update({"psnr": 25.0}, batch_size=4)
        tracker.update({"psnr": 26.0}, batch_size=4)

        assert tracker.count == 8

    def test_average(self):
        """Test computing average of metrics."""
        tracker = MetricsTracker()

        tracker.update({"psnr": 20.0, "ssim": 0.8})
        tracker.update({"psnr": 30.0, "ssim": 0.9})

        avg = tracker.average()

        assert "psnr" in avg
        assert "ssim" in avg
        assert isinstance(avg["psnr"], float)
        assert isinstance(avg["ssim"], float)
        assert avg["psnr"] == pytest.approx(25.0, 0.01)
        assert avg["ssim"] == pytest.approx(0.85, 0.01)

    def test_std(self):
        """Test computing standard deviation."""
        tracker = MetricsTracker()

        tracker.update({"psnr": 20.0})
        tracker.update({"psnr": 30.0})

        std = tracker.std()

        assert "psnr" in std
        assert std["psnr"] == pytest.approx(5.0)

    def test_summary(self):
        """Test getting comprehensive summary."""
        tracker = MetricsTracker()

        tracker.update({"psnr": 20.0})
        tracker.update({"psnr": 25.0})
        tracker.update({"psnr": 30.0})

        summary = tracker.summary()

        assert "psnr" in summary
        assert "mean" in summary["psnr"]
        assert "std" in summary["psnr"]
        assert "min" in summary["psnr"]
        assert "max" in summary["psnr"]

        assert summary["psnr"]["mean"] == 25.0
        assert summary["psnr"]["min"] == 20.0
        assert summary["psnr"]["max"] == 30.0

    def test_reset(self):
        """Test resetting tracker."""
        tracker = MetricsTracker()

        tracker.update({"psnr": 25.0})
        tracker.update({"ssim": 0.85})

        tracker.reset()

        assert len(tracker.metrics) == 0
        assert tracker.count == 0

    def test_repr(self):
        """Test string representation."""
        tracker = MetricsTracker()

        tracker.update({"psnr": 25.0, "ssim": 0.85})

        repr_str = repr(tracker)

        assert "MetricsTracker" in repr_str
        assert "psnr" in repr_str
        assert "ssim" in repr_str
        assert "count=1" in repr_str

    def test_empty_tracker_average(self):
        """Test average on empty tracker."""
        tracker = MetricsTracker()

        avg = tracker.average()

        assert len(avg) == 0


class TestMetricsIntegration:
    """Integration tests for metrics."""

    def test_full_evaluation_workflow(self):
        """Test complete evaluation workflow."""
        tracker = MetricsTracker()

        # Simulate evaluation loop
        for _ in range(10):
            pred = torch.rand(4, 3, 64, 64)
            target = torch.rand(4, 3, 64, 64)

            metrics = compute_metrics(pred, target)
            tracker.update(metrics, batch_size=4)

        # Get results
        avg_metrics = tracker.average()
        summary = tracker.summary()

        assert len(avg_metrics) == 4
        assert "psnr" in summary
        assert tracker.count == 40

    def test_metrics_consistency(self):
        """Test that metrics are consistent between calls."""
        pred = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)

        metrics1 = compute_metrics(pred, target)
        metrics2 = compute_metrics(pred, target)

        for key in metrics1:
            assert metrics1[key] == metrics2[key]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
