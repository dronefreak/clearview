"""Unit tests for the native NIQE (no-reference) quality metric."""

import numpy as np
import pytest
import torch

from clearview.utils.niqe import NIQEModel, compute_niqe, fit_niqe_model


def _random_images(count: int = 12, size: int = 128, seed: int = 0) -> list:
    """Create a list of random torch image tensors for fitting/scoring."""
    generator = torch.Generator().manual_seed(seed)
    return [torch.rand(3, size, size, generator=generator) for _ in range(count)]


class TestFitNIQEModel:
    """Tests for fit_niqe_model()."""

    def test_returns_niqe_model(self) -> None:
        """Test that fitting returns a NIQEModel with mu/cov arrays."""
        images = _random_images()
        model = fit_niqe_model(images, patch_size=32)

        assert isinstance(model, NIQEModel)
        assert isinstance(model.mu, np.ndarray)
        assert isinstance(model.cov, np.ndarray)

    def test_feature_dimensionality(self) -> None:
        """Test that mu/cov have the expected 36-dim NIQE feature size."""
        images = _random_images()
        model = fit_niqe_model(images, patch_size=32)

        assert model.mu.shape == (36,)
        assert model.cov.shape == (36, 36)

    def test_accepts_numpy_images(self) -> None:
        """Test fitting works with (H, W, 3) numpy uint8-range images."""
        images = [
            (np.random.rand(128, 128, 3) * 255).astype(np.float32) for _ in range(8)
        ]
        model = fit_niqe_model(images, patch_size=32)

        assert model.mu.shape == (36,)

    def test_raises_on_no_valid_patches(self) -> None:
        """Test that fitting raises when images are too small for patch_size."""
        tiny_images = [torch.rand(3, 8, 8) for _ in range(4)]

        with pytest.raises(ValueError):
            fit_niqe_model(tiny_images, patch_size=32)


class TestComputeNIQE:
    """Tests for compute_niqe()."""

    def test_same_distribution_scores_lower_than_degenerate(self) -> None:
        """Test that an in-distribution image scores lower than a constant image."""
        images = _random_images(count=16, size=128, seed=1)
        model = fit_niqe_model(images, patch_size=32)

        in_distribution = torch.rand(
            3, 128, 128, generator=torch.Generator().manual_seed(2)
        )
        degenerate = torch.full((3, 128, 128), 0.5)

        score_in_distribution = compute_niqe(in_distribution, model, patch_size=32)
        score_degenerate = compute_niqe(degenerate, model, patch_size=32)

        assert score_in_distribution < score_degenerate

    def test_returns_float(self) -> None:
        """Test that compute_niqe returns a plain float."""
        images = _random_images()
        model = fit_niqe_model(images, patch_size=32)
        image = torch.rand(3, 128, 128)

        score = compute_niqe(image, model, patch_size=32)

        assert isinstance(score, float)
        assert score >= 0.0

    def test_score_is_non_negative_and_finite(self) -> None:
        """Test that the NIQE distance is a finite, non-negative number."""
        images = _random_images()
        model = fit_niqe_model(images, patch_size=32)
        image = torch.rand(1, 128, 128)  # single-channel grayscale input

        score = compute_niqe(image, model, patch_size=32)

        assert score >= 0.0
        assert np.isfinite(score)

    def test_accepts_grayscale_numpy_image(self) -> None:
        """Test scoring works with a (H, W) grayscale numpy array."""
        images = _random_images()
        model = fit_niqe_model(images, patch_size=32)
        image = np.random.rand(128, 128).astype(np.float32)

        score = compute_niqe(image, model, patch_size=32)

        assert isinstance(score, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
