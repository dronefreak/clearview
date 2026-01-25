"""Shared pytest fixtures for ClearView test suite."""

import tempfile
from pathlib import Path
from typing import Generator, Tuple

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def device() -> torch.device:
    """Get the device for testing (CPU or CUDA if available).

    Returns:
        torch.device: Device for testing.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_image_tensor() -> torch.Tensor:
    """Create a sample image tensor for testing.

    Returns:
        torch.Tensor: Sample image tensor with shape (1, 3, 256, 256).
    """
    return torch.randn(1, 3, 256, 256)


@pytest.fixture
def sample_image_pair() -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a pair of sample image tensors (rainy and clean).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Rainy and clean image tensors.
    """
    rainy = torch.randn(1, 3, 256, 256)
    clean = torch.randn(1, 3, 256, 256)
    return rainy, clean


@pytest.fixture
def sample_batch() -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a batch of sample image pairs.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Batch of rainy and clean images.
    """
    batch_size = 4
    rainy = torch.randn(batch_size, 3, 256, 256)
    clean = torch.randn(batch_size, 3, 256, 256)
    return rainy, clean


@pytest.fixture
def sample_numpy_image() -> np.ndarray:
    """Create a sample numpy image array.

    Returns:
        np.ndarray: Sample image array with shape (256, 256, 3) in range [0, 255].
    """
    return np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)


@pytest.fixture
def sample_pil_image(sample_numpy_image: np.ndarray) -> Image.Image:
    """Create a sample PIL Image.

    Args:
        sample_numpy_image: Sample numpy image array.

    Returns:
        Image.Image: Sample PIL Image.
    """
    return Image.fromarray(sample_numpy_image)


@pytest.fixture
def temp_image_file(
    sample_pil_image: Image.Image,
) -> Generator[Path, None, None]:
    """Create a temporary image file.

    Args:
        sample_pil_image: Sample PIL Image to save.

    Yields:
        Path: Path to temporary image file.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_path = Path(f.name)
        sample_pil_image.save(temp_path)
        yield temp_path
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing.

    Yields:
        Path: Path to temporary directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_dataset_dir(temp_dir: Path) -> Path:
    """Create a sample dataset directory with image pairs.

    Args:
        temp_dir: Temporary directory path.

    Returns:
        Path: Path to dataset directory.
    """
    rainy_dir = temp_dir / "rainy"
    clean_dir = temp_dir / "clean"
    rainy_dir.mkdir()
    clean_dir.mkdir()

    # Create sample image pairs
    for i in range(5):
        # Create rainy image
        rainy_img = Image.fromarray(
            np.random.randint(0, 256, size=(128, 128, 3), dtype=np.uint8)
        )
        rainy_img.save(rainy_dir / f"image_{i:03d}.png")

        # Create clean image
        clean_img = Image.fromarray(
            np.random.randint(0, 256, size=(128, 128, 3), dtype=np.uint8)
        )
        clean_img.save(clean_dir / f"image_{i:03d}.png")

    return temp_dir


@pytest.fixture
def model_config() -> dict:
    """Get default model configuration for testing.

    Returns:
        dict: Model configuration dictionary.
    """
    return {
        "in_channels": 3,
        "out_channels": 3,
        "init_features": 32,
        "depth": 3,
    }


@pytest.fixture
def training_config() -> dict:
    """Get default training configuration for testing.

    Returns:
        dict: Training configuration dictionary.
    """
    return {
        "learning_rate": 1e-4,
        "batch_size": 2,
        "epochs": 2,
        "device": "cpu",
    }
