"""Tests for visualization utilities.

These tests cover plotting and visualization functions with matplotlib mocking.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from clearview.utils.visualization import (
    create_comparison_grid,
    plot_metric_histogram,
    plot_training_curves,
    save_comparison,
    tensor_to_image,
    visualize_results,
)


class TestTensorToImage:
    """Tests for tensor_to_image conversion."""

    def test_convert_3d_tensor(self):
        """Test converting 3D tensor (C, H, W)."""
        tensor = torch.rand(3, 64, 64)

        img = tensor_to_image(tensor)

        assert isinstance(img, np.ndarray)
        assert img.shape == (64, 64, 3)
        assert img.dtype == np.uint8
        assert img.min() >= 0
        assert img.max() <= 255

    def test_convert_4d_tensor(self):
        """Test converting 4D tensor batch (takes first image)."""
        tensor = torch.rand(4, 3, 64, 64)

        img = tensor_to_image(tensor)

        assert img.shape == (64, 64, 3)
        assert img.dtype == np.uint8

    def test_convert_grayscale(self):
        """Test converting grayscale image."""
        tensor = torch.rand(1, 64, 64)

        img = tensor_to_image(tensor)

        assert img.shape == (64, 64)  # Squeezed
        assert img.dtype == np.uint8

    def test_clipping_applied(self):
        """Test that values are clipped to [0, 1]."""
        # Create tensor with values outside [0, 1]
        tensor = torch.tensor([[[2.0, -1.0], [0.5, 0.8]]])

        img = tensor_to_image(tensor)

        # After clipping and scaling, values should be valid
        assert img.min() >= 0
        assert img.max() <= 255

    def test_value_scaling(self):
        """Test that values are correctly scaled to [0, 255]."""
        # All zeros
        tensor_zeros = torch.zeros(1, 8, 8)
        img_zeros = tensor_to_image(tensor_zeros)
        assert np.all(img_zeros == 0)

        # All ones
        tensor_ones = torch.ones(1, 8, 8)
        img_ones = tensor_to_image(tensor_ones)
        assert np.all(img_ones == 255)

    def test_gradient_not_tracked(self):
        """Test that gradient is detached."""
        tensor = torch.rand(3, 32, 32, requires_grad=True)

        img = tensor_to_image(tensor)

        # Should not raise error
        assert isinstance(img, np.ndarray)


@patch("clearview.utils.visualization.plt")
class TestVisualizeResults:
    """Tests for visualize_results function."""

    def test_visualize_two_images(self, mock_plt):
        """Test visualizing rainy and derained images."""
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        rainy = torch.rand(3, 64, 64)
        derained = torch.rand(3, 64, 64)

        _ = visualize_results(rainy, derained)

        mock_plt.subplots.assert_called_once_with(1, 2, figsize=(15, 5))
        assert mock_axes[0].imshow.called
        assert mock_axes[1].imshow.called
        assert mock_axes[0].set_title.called
        assert mock_axes[1].set_title.called

    def test_visualize_three_images(self, mock_plt):
        """Test visualizing rainy, derained, and clean images."""
        mock_fig = Mock()
        mock_axes = [Mock(), Mock(), Mock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        rainy = torch.rand(3, 64, 64)
        derained = torch.rand(3, 64, 64)
        clean = torch.rand(3, 64, 64)

        _ = visualize_results(rainy, derained, clean)

        mock_plt.subplots.assert_called_once_with(1, 3, figsize=(15, 5))
        assert mock_axes[0].imshow.called
        assert mock_axes[1].imshow.called
        assert mock_axes[2].imshow.called

    def test_custom_titles(self, mock_plt):
        """Test with custom titles."""
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        rainy = torch.rand(3, 64, 64)
        derained = torch.rand(3, 64, 64)
        titles = ["Input", "Output"]

        visualize_results(rainy, derained, titles=titles)

        mock_axes[0].set_title.assert_called_with(
            "Input", fontsize=12, fontweight="bold"
        )
        mock_axes[1].set_title.assert_called_with(
            "Output", fontsize=12, fontweight="bold"
        )

    def test_custom_figsize(self, mock_plt):
        """Test with custom figure size."""
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        rainy = torch.rand(3, 64, 64)
        derained = torch.rand(3, 64, 64)

        visualize_results(rainy, derained, figsize=(20, 8))

        mock_plt.subplots.assert_called_once_with(1, 2, figsize=(20, 8))

    def test_numpy_input(self, mock_plt):
        """Test with numpy array input."""
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        rainy = np.random.rand(64, 64, 3)
        derained = np.random.rand(64, 64, 3)

        _ = visualize_results(rainy, derained)

        assert mock_axes[0].imshow.called
        assert mock_axes[1].imshow.called

    def test_axis_off(self, mock_plt):
        """Test that axes are turned off."""
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        rainy = torch.rand(3, 64, 64)
        derained = torch.rand(3, 64, 64)

        visualize_results(rainy, derained)

        mock_axes[0].axis.assert_called_with("off")
        mock_axes[1].axis.assert_called_with("off")


@patch("clearview.utils.visualization.plt")
@patch("clearview.utils.visualization.gridspec")
class TestCreateComparisonGrid:
    """Tests for create_comparison_grid function."""

    def test_create_grid_without_clean(self, mock_gridspec, mock_plt):
        """Test creating grid without ground truth."""
        mock_fig = Mock()
        mock_plt.figure.return_value = mock_fig

        # Make gridspec subscriptable by returning a Mock for any index access
        mock_gs = Mock()
        mock_gs.__getitem__ = Mock(return_value=Mock())
        mock_gridspec.GridSpec.return_value = mock_gs

        images = [
            (torch.rand(3, 64, 64), torch.rand(3, 64, 64), None),
            (torch.rand(3, 64, 64), torch.rand(3, 64, 64), None),
        ]

        _ = create_comparison_grid(images)

        mock_gridspec.GridSpec.assert_called_once_with(2, 2, hspace=0.3, wspace=0.1)
        assert mock_fig.add_subplot.call_count == 4  # 2 images × 2 columns

    def test_create_grid_with_clean(self, mock_gridspec, mock_plt):
        """Test creating grid with ground truth."""
        mock_fig = Mock()
        mock_plt.figure.return_value = mock_fig

        # Make gridspec subscriptable
        mock_gs = Mock()
        mock_gs.__getitem__ = Mock(return_value=Mock())
        mock_gridspec.GridSpec.return_value = mock_gs

        images = [
            (torch.rand(3, 64, 64), torch.rand(3, 64, 64), torch.rand(3, 64, 64)),
            (torch.rand(3, 64, 64), torch.rand(3, 64, 64), torch.rand(3, 64, 64)),
        ]

        _ = create_comparison_grid(images)

        mock_gridspec.GridSpec.assert_called_once_with(2, 3, hspace=0.3, wspace=0.1)
        assert mock_fig.add_subplot.call_count == 6  # 2 images × 3 columns

    def test_max_images_limit(self, mock_gridspec, mock_plt):
        """Test that max_images limits number of images shown."""
        mock_fig = Mock()
        mock_plt.figure.return_value = mock_fig

        # Make gridspec subscriptable
        mock_gs = Mock()
        mock_gs.__getitem__ = Mock(return_value=Mock())
        mock_gridspec.GridSpec.return_value = mock_gs

        # Create 10 images but limit to 4
        images = [
            (torch.rand(3, 32, 32), torch.rand(3, 32, 32), None) for _ in range(10)
        ]

        _ = create_comparison_grid(images, max_images=4)

        # Should only create grid for 4 images
        mock_gridspec.GridSpec.assert_called_once_with(4, 2, hspace=0.3, wspace=0.1)

    def test_auto_figsize(self, mock_gridspec, mock_plt):
        """Test automatic figure size calculation."""
        mock_fig = Mock()
        mock_plt.figure.return_value = mock_fig

        # Make gridspec subscriptable
        mock_gs = Mock()
        mock_gs.__getitem__ = Mock(return_value=Mock())
        mock_gridspec.GridSpec.return_value = mock_gs

        images = [
            (torch.rand(3, 32, 32), torch.rand(3, 32, 32), None) for _ in range(3)
        ]

        _ = create_comparison_grid(images, figsize=None)

        # Figsize should be auto-calculated: (2 cols * 4, 3 images * 3)
        mock_plt.figure.assert_called_once_with(figsize=(8, 9))


@patch("clearview.utils.visualization.plt")
class TestPlotTrainingCurves:
    """Tests for plot_training_curves function."""

    def test_plot_single_metric(self, mock_plt):
        """Test plotting single metric."""
        mock_fig = Mock()
        mock_ax = Mock()
        # For single metric, return single ax (code will wrap in list)
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        train_history = {"loss": [0.5, 0.4, 0.3, 0.2]}

        _ = plot_training_curves(train_history, metrics=["loss"])

        mock_plt.subplots.assert_called_once_with(1, 1, figsize=(15, 5))
        assert mock_ax.plot.called
        assert mock_ax.set_xlabel.called
        assert mock_ax.set_ylabel.called

    def test_plot_multiple_metrics(self, mock_plt):
        """Test plotting multiple metrics."""
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        train_history = {"loss": [0.5, 0.4], "psnr": [25.0, 26.0]}

        _ = plot_training_curves(train_history, metrics=["loss", "psnr"])

        mock_plt.subplots.assert_called_once_with(1, 2, figsize=(15, 5))
        assert mock_axes[0].plot.called
        assert mock_axes[1].plot.called

    def test_plot_with_validation(self, mock_plt):
        """Test plotting with validation curves."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        train_history = {"loss": [0.5, 0.4, 0.3]}
        val_history = {"loss": [0.6, 0.5, 0.4]}

        _ = plot_training_curves(train_history, val_history, metrics=["loss"])

        # Should plot both train and val curves
        assert mock_ax.plot.call_count == 2

    def test_plot_all_metrics_default(self, mock_plt):
        """Test plotting all metrics when none specified."""
        mock_fig = Mock()
        mock_axes = [Mock(), Mock(), Mock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        train_history = {"loss": [0.5], "psnr": [25.0], "ssim": [0.85]}

        _ = plot_training_curves(train_history)

        # Should plot all 3 metrics
        mock_plt.subplots.assert_called_once_with(1, 3, figsize=(15, 5))

    def test_save_to_file(self, mock_plt):
        """Test saving plot to file."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        train_history = {"loss": [0.5, 0.4]}
        save_path = "/tmp/test_plot.png"

        _ = plot_training_curves(train_history, save_path=save_path)

        mock_fig.savefig.assert_called_once()
        # Check that path is in the call
        assert str(save_path) in str(mock_fig.savefig.call_args)

    def test_grid_enabled(self, mock_plt):
        """Test that grid is enabled on plots."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        train_history = {"loss": [0.5, 0.4]}

        plot_training_curves(train_history)

        mock_ax.grid.assert_called_with(True, alpha=0.3)


@patch("clearview.utils.visualization.plt")
class TestSaveComparison:
    """Tests for save_comparison function."""

    @patch("clearview.utils.visualization.visualize_results")
    def test_save_comparison_calls_visualize(self, mock_visualize, mock_plt):
        """Test that save_comparison calls visualize_results."""
        mock_fig = Mock()
        mock_visualize.return_value = mock_fig

        rainy = torch.rand(3, 64, 64)
        derained = torch.rand(3, 64, 64)

        save_comparison(rainy, derained, save_path="/tmp/test.png")

        mock_visualize.assert_called_once()
        mock_fig.savefig.assert_called_once()

    @patch("clearview.utils.visualization.visualize_results")
    def test_save_creates_parent_dir(self, mock_visualize, mock_plt):
        """Test that parent directory is created."""
        mock_fig = Mock()
        mock_visualize.return_value = mock_fig

        rainy = torch.rand(3, 64, 64)
        derained = torch.rand(3, 64, 64)

        # Use a path that would require directory creation
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            save_comparison(rainy, derained, save_path="/tmp/subdir/test.png")
            mock_mkdir.assert_called()

    @patch("clearview.utils.visualization.visualize_results")
    def test_save_custom_dpi(self, mock_visualize, mock_plt):
        """Test saving with custom DPI."""
        mock_fig = Mock()
        mock_visualize.return_value = mock_fig

        rainy = torch.rand(3, 64, 64)
        derained = torch.rand(3, 64, 64)

        save_comparison(rainy, derained, save_path="/tmp/test.png", dpi=300)

        # Check DPI in savefig call
        assert mock_fig.savefig.call_args[1]["dpi"] == 300

    @patch("clearview.utils.visualization.visualize_results")
    def test_save_closes_figure(self, mock_visualize, mock_plt):
        """Test that figure is closed after saving."""
        mock_fig = Mock()
        mock_visualize.return_value = mock_fig

        rainy = torch.rand(3, 64, 64)
        derained = torch.rand(3, 64, 64)

        save_comparison(rainy, derained, save_path="/tmp/test.png")

        mock_plt.close.assert_called_once_with(mock_fig)


@patch("clearview.utils.visualization.plt")
class TestPlotMetricHistogram:
    """Tests for plot_metric_histogram function."""

    def test_plot_single_metric_histogram(self, mock_plt):
        """Test plotting histogram for single metric."""
        mock_fig = Mock()
        mock_ax = Mock()
        # For single metric, return single ax
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        metrics = {"psnr": [25.0, 26.0, 27.0, 28.0, 29.0]}

        _ = plot_metric_histogram(metrics)

        mock_plt.subplots.assert_called_once()
        mock_ax.hist.assert_called_once()
        mock_ax.axvline.assert_called_once()  # Mean line

    def test_plot_multiple_metric_histograms(self, mock_plt):
        """Test plotting histograms for multiple metrics."""
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        metrics = {"psnr": [25.0, 26.0, 27.0], "ssim": [0.8, 0.85, 0.9]}

        _ = plot_metric_histogram(metrics)

        assert mock_axes[0].hist.called
        assert mock_axes[1].hist.called

    def test_custom_bins(self, mock_plt):
        """Test with custom number of bins."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        metrics = {"psnr": list(range(50))}

        _ = plot_metric_histogram(metrics, bins=20)

        # Check bins parameter in hist call
        call_args = mock_ax.hist.call_args
        assert call_args[1]["bins"] == 20

    def test_mean_line_drawn(self, mock_plt):
        """Test that mean line is drawn."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        values = [20.0, 25.0, 30.0]
        metrics = {"psnr": values}

        _ = plot_metric_histogram(metrics)

        # Check that axvline was called with mean
        expected_mean = np.mean(values)
        mock_ax.axvline.assert_called_once()
        assert mock_ax.axvline.call_args[0][0] == expected_mean

    def test_save_histogram(self, mock_plt):
        """Test saving histogram to file."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        metrics = {"psnr": [25.0, 26.0, 27.0]}
        save_path = "/tmp/histogram.png"

        _ = plot_metric_histogram(metrics, save_path=save_path)

        mock_fig.savefig.assert_called_once()


class TestVisualizationIntegration:
    """Integration tests for visualization utilities."""

    @patch("clearview.utils.visualization.plt")
    def test_complete_visualization_workflow(self, mock_plt):
        """Test complete visualization workflow."""
        mock_fig = Mock()
        mock_axes = [Mock(), Mock(), Mock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        # Create test data
        rainy = torch.rand(3, 128, 128)
        derained = torch.rand(3, 128, 128)
        clean = torch.rand(3, 128, 128)

        # Visualize
        _ = visualize_results(rainy, derained, clean)

        # Should create 3-column visualization
        mock_plt.subplots.assert_called_with(1, 3, figsize=(15, 5))

        # All axes should be used
        assert all(ax.imshow.called for ax in mock_axes)
        assert all(ax.axis.called for ax in mock_axes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
