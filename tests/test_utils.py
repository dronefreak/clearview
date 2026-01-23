"""Unit tests for utility functions."""

import numpy as np
import torch

from clearview.utils.image import (
    denormalize_image,
    normalize_image,
    numpy_to_tensor,
    rgb_to_grayscale,
    tensor_to_numpy,
)
from clearview.utils.metrics import compute_mae, compute_mse, compute_psnr, compute_ssim


class TestNormalization:
    """Tests for image normalization functions."""

    def test_normalize_image_tensor_3d(self) -> None:
        """Test normalize_image with 3D tensor."""
        img = torch.rand(3, 256, 256)
        normalized = normalize_image(img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        assert isinstance(normalized, torch.Tensor)
        assert normalized.shape == img.shape
        # With mean=0.5 and std=0.5, range [0,1] becomes [-1,1]
        assert normalized.min() >= -1.5  # Allow some margin
        assert normalized.max() <= 1.5

    def test_normalize_image_tensor_4d(self) -> None:
        """Test normalize_image with 4D tensor (batch)."""
        img = torch.rand(4, 3, 256, 256)
        normalized = normalize_image(img)

        assert isinstance(normalized, torch.Tensor)
        assert normalized.shape == img.shape

    def test_normalize_image_numpy(self) -> None:
        """Test normalize_image with numpy array."""
        img = np.random.rand(3, 256, 256).astype(np.float32)
        normalized = normalize_image(img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        assert isinstance(normalized, np.ndarray)
        assert normalized.shape == img.shape

    def test_denormalize_image_tensor(self) -> None:
        """Test denormalize_image with tensor."""
        img = torch.rand(3, 256, 256)
        normalized = normalize_image(img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        denormalized = denormalize_image(
            normalized, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        )

        assert torch.allclose(img, denormalized, atol=1e-6)

    def test_denormalize_image_numpy(self) -> None:
        """Test denormalize_image with numpy array."""
        img = np.random.rand(3, 256, 256).astype(np.float32)
        normalized = normalize_image(img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        denormalized = denormalize_image(
            normalized, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        )

        assert np.allclose(img, denormalized, atol=1e-6)

    def test_normalize_custom_mean_std(self) -> None:
        """Test normalize_image with custom mean and std."""
        img = torch.rand(3, 256, 256)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        normalized = normalize_image(img, mean=mean, std=std)
        denormalized = denormalize_image(normalized, mean=mean, std=std)

        assert torch.allclose(img, denormalized, atol=1e-5)


class TestRGBToGrayscale:
    """Tests for RGB to grayscale conversion."""

    def test_rgb_to_grayscale_tensor_3d(self) -> None:
        """Test rgb_to_grayscale with 3D tensor."""
        rgb = torch.rand(3, 256, 256)
        gray = rgb_to_grayscale(rgb)

        assert isinstance(gray, torch.Tensor)
        assert gray.shape == (1, 256, 256)

    def test_rgb_to_grayscale_tensor_4d(self) -> None:
        """Test rgb_to_grayscale with 4D tensor (batch)."""
        rgb = torch.rand(4, 3, 256, 256)
        gray = rgb_to_grayscale(rgb)

        assert isinstance(gray, torch.Tensor)
        assert gray.shape == (4, 1, 256, 256)

    def test_rgb_to_grayscale_numpy(self) -> None:
        """Test rgb_to_grayscale with numpy array."""
        rgb = np.random.rand(3, 256, 256).astype(np.float32)
        gray = rgb_to_grayscale(rgb)

        assert isinstance(gray, np.ndarray)
        assert gray.shape == (1, 256, 256)

    def test_rgb_to_grayscale_already_grayscale(self) -> None:
        """Test rgb_to_grayscale with already grayscale image."""
        gray_in = torch.rand(1, 256, 256)
        gray_out = rgb_to_grayscale(gray_in)

        assert gray_out.shape == gray_in.shape
        assert torch.equal(gray_out, gray_in)

    def test_rgb_to_grayscale_value_range(self) -> None:
        """Test that grayscale values are in valid range."""
        rgb = torch.rand(3, 256, 256)
        gray = rgb_to_grayscale(rgb)

        assert gray.min() >= 0.0
        assert gray.max() <= 1.0


class TestTensorNumpyConversion:
    """Tests for tensor-numpy conversion functions."""

    def test_tensor_to_numpy(self) -> None:
        """Test tensor_to_numpy conversion."""
        tensor = torch.rand(3, 256, 256)
        array = tensor_to_numpy(tensor)

        assert isinstance(array, np.ndarray)
        assert array.shape == (256, 256, 3)  # H, W, C format

    def test_tensor_to_numpy_batch(self) -> None:
        """Test tensor_to_numpy with batch."""
        tensor = torch.rand(4, 3, 256, 256)
        array = tensor_to_numpy(tensor)

        assert isinstance(array, np.ndarray)
        assert array.shape == (4, 256, 256, 3)

    def test_numpy_to_tensor(self) -> None:
        """Test numpy_to_tensor conversion."""
        array = np.random.rand(256, 256, 3).astype(np.float32)
        tensor = numpy_to_tensor(array)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 256, 256)  # C, H, W format

    def test_numpy_to_tensor_batch(self) -> None:
        """Test numpy_to_tensor with batch."""
        array = np.random.rand(4, 256, 256, 3).astype(np.float32)
        tensor = numpy_to_tensor(array)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (4, 3, 256, 256)

    def test_roundtrip_conversion(self) -> None:
        """Test roundtrip tensor -> numpy -> tensor conversion."""
        original = torch.rand(3, 256, 256)
        array = tensor_to_numpy(original)
        recovered = numpy_to_tensor(array)

        assert torch.allclose(original, recovered, atol=1e-6)


class TestMetricsPSNR:
    """Tests for PSNR metric."""

    def test_compute_psnr_identical_images(self) -> None:
        """Test PSNR with identical images."""
        img = torch.rand(4, 3, 256, 256)
        psnr = compute_psnr(img, img)

        # PSNR should be very high for identical images
        assert isinstance(psnr, float)
        assert psnr > 50  # Typically > 50 dB for identical images

    def test_compute_psnr_different_images(self) -> None:
        """Test PSNR with different images."""
        pred = torch.rand(4, 3, 256, 256)
        target = torch.rand(4, 3, 256, 256)
        psnr = compute_psnr(pred, target)

        assert isinstance(psnr, float)
        assert psnr > 0  # PSNR should be positive

    def test_compute_psnr_reduction_none(self) -> None:
        """Test PSNR with reduction='none'."""
        pred = torch.rand(4, 3, 256, 256)
        target = torch.rand(4, 3, 256, 256)
        psnr = compute_psnr(pred, target, reduction="none")

        assert isinstance(psnr, torch.Tensor)
        assert psnr.shape == (4,)  # One PSNR per image in batch

    def test_compute_psnr_numpy(self) -> None:
        """Test PSNR with numpy arrays."""
        pred = torch.rand(4, 3, 256, 256)
        target = torch.rand(4, 3, 256, 256)
        psnr = compute_psnr(pred, target)

        assert isinstance(psnr, float)
        assert psnr > 0

    def test_compute_psnr_max_val(self) -> None:
        """Test PSNR with different max_val."""
        # Images in range [0, 255]
        pred = torch.rand(4, 3, 256, 256) * 255
        target = torch.rand(4, 3, 256, 256) * 255
        psnr = compute_psnr(pred, target, max_val=255.0)

        assert isinstance(psnr, float)
        assert psnr > 0


class TestMetricsSSIM:
    """Tests for SSIM metric."""

    def test_compute_ssim_identical_images(self) -> None:
        """Test SSIM with identical images."""
        img = torch.rand(2, 3, 256, 256)
        ssim = compute_ssim(img, img)

        # SSIM should be 1.0 for identical images
        assert isinstance(ssim, float)
        assert ssim > 0.99  # Close to 1.0

    def test_compute_ssim_different_images(self) -> None:
        """Test SSIM with different images."""
        pred = torch.rand(2, 3, 256, 256)
        target = torch.rand(2, 3, 256, 256)
        ssim = compute_ssim(pred, target)

        assert isinstance(ssim, float)
        assert 0.0 <= ssim <= 1.0  # SSIM in [0, 1]

    def test_compute_ssim_reduction_none(self) -> None:
        """Test SSIM with reduction='none'."""
        pred = torch.rand(2, 3, 256, 256)
        target = torch.rand(2, 3, 256, 256)
        ssim = compute_ssim(pred, target, reduction="none")

        assert isinstance(ssim, torch.Tensor)
        assert ssim.shape == (2,)  # One SSIM per image

    def test_compute_ssim_window_size(self) -> None:
        """Test SSIM with different window sizes."""
        pred = torch.rand(2, 3, 256, 256)
        target = torch.rand(2, 3, 256, 256)

        for window_size in [5, 7, 11]:
            ssim = compute_ssim(pred, target, window_size=window_size)
            assert isinstance(ssim, float)
            assert 0.0 <= ssim <= 1.0


class TestMetricsMAE:
    """Tests for MAE metric."""

    def test_compute_mae_identical_images(self) -> None:
        """Test MAE with identical images."""
        img = torch.rand(4, 3, 256, 256)
        mae = compute_mae(img, img)

        # MAE should be 0 for identical images
        assert isinstance(mae, float)
        assert mae < 1e-6

    def test_compute_mae_different_images(self) -> None:
        """Test MAE with different images."""
        pred = torch.rand(4, 3, 256, 256)
        target = torch.rand(4, 3, 256, 256)
        mae = compute_mae(pred, target)

        assert isinstance(mae, float)
        assert mae >= 0.0

    def test_compute_mae_reduction_none(self) -> None:
        """Test MAE with reduction='none'."""
        pred = torch.rand(4, 3, 256, 256)
        target = torch.rand(4, 3, 256, 256)
        mae = compute_mae(pred, target, reduction="none")

        assert isinstance(mae, torch.Tensor)
        assert mae.shape == (4,)

    def test_compute_mae_numpy(self) -> None:
        """Test MAE with numpy arrays."""
        pred = torch.rand(4, 3, 256, 256)
        target = torch.rand(4, 3, 256, 256)
        mae = compute_mae(pred, target)

        assert isinstance(mae, float)
        assert mae >= 0.0


class TestMetricsMSE:
    """Tests for MSE metric."""

    def test_compute_mse_identical_images(self) -> None:
        """Test MSE with identical images."""
        img = torch.rand(4, 3, 256, 256)
        mse = compute_mse(img, img)

        # MSE should be 0 for identical images
        assert isinstance(mse, float)
        assert mse < 1e-6

    def test_compute_mse_different_images(self) -> None:
        """Test MSE with different images."""
        pred = torch.rand(4, 3, 256, 256)
        target = torch.rand(4, 3, 256, 256)
        mse = compute_mse(pred, target)

        assert isinstance(mse, float)
        assert mse >= 0.0

    def test_compute_mse_reduction_none(self) -> None:
        """Test MSE with reduction='none'."""
        pred = torch.rand(4, 3, 256, 256)
        target = torch.rand(4, 3, 256, 256)
        mse = compute_mse(pred, target, reduction="none")

        assert isinstance(mse, torch.Tensor)
        assert mse.shape == (4,)

    def test_mse_greater_than_mae_for_large_errors(self) -> None:
        """Test that MSE penalizes large errors more than MAE."""
        pred = torch.tensor([[[[0.0]]]])
        target = torch.tensor([[[[10.0]]]])

        mae = compute_mae(pred, target)
        mse = compute_mse(pred, target)

        # MSE = 100, MAE = 10
        assert mse > mae
