"""Unit tests for data loading and datasets."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from clearview.data.datasets import (
    ImagePairDataset,
    Rain100Dataset,
    SingleFolderDataset,
)


class TestImagePairDataset:
    """Tests for ImagePairDataset."""

    def test_initialization(self, sample_dataset_dir: Path) -> None:
        """Test ImagePairDataset initialization."""
        rainy_dir = sample_dataset_dir / "rainy"
        clean_dir = sample_dataset_dir / "clean"

        dataset = ImagePairDataset(rainy_dir=rainy_dir, clean_dir=clean_dir)
        assert len(dataset) == 5

    def test_initialization_mismatched_counts(
        self, sample_dataset_dir: Path, temp_dir: Path
    ) -> None:
        """Test ImagePairDataset raises error for mismatched image counts."""
        rainy_dir = sample_dataset_dir / "rainy"

        # Create clean dir with different number of images
        clean_dir = temp_dir / "clean_mismatch"
        clean_dir.mkdir()
        img = Image.fromarray(
            torch.randint(0, 256, (128, 128, 3), dtype=torch.uint8).numpy()
        )
        img.save(clean_dir / "image_000.png")

        with pytest.raises(ValueError, match="Unpaired images found"):
            ImagePairDataset(rainy_dir=rainy_dir, clean_dir=clean_dir)

    def test_length(self, sample_dataset_dir: Path) -> None:
        """Test ImagePairDataset length."""
        dataset = ImagePairDataset(
            rainy_dir=sample_dataset_dir / "rainy",
            clean_dir=sample_dataset_dir / "clean",
        )
        assert len(dataset) == 5

    def test_getitem(self, sample_dataset_dir: Path) -> None:
        """Test ImagePairDataset __getitem__."""
        dataset = ImagePairDataset(
            rainy_dir=sample_dataset_dir / "rainy",
            clean_dir=sample_dataset_dir / "clean",
        )

        rainy, clean = dataset[0]

        # Check types
        assert isinstance(rainy, torch.Tensor)
        assert isinstance(clean, torch.Tensor)

        # Check shapes (C, H, W) format
        assert rainy.ndim == 3
        assert clean.ndim == 3
        assert rainy.shape[0] == 3  # RGB channels
        assert clean.shape[0] == 3

    def test_getitem_value_range(self, sample_dataset_dir: Path) -> None:
        """Test that loaded images are in [0, 1] range."""
        dataset = ImagePairDataset(
            rainy_dir=sample_dataset_dir / "rainy",
            clean_dir=sample_dataset_dir / "clean",
        )

        rainy, clean = dataset[0]

        # Check value ranges
        assert rainy.min() >= 0.0
        assert rainy.max() <= 1.0
        assert clean.min() >= 0.0
        assert clean.max() <= 1.0

    def test_getitem_all_indices(self, sample_dataset_dir: Path) -> None:
        """Test accessing all indices in dataset."""
        dataset = ImagePairDataset(
            rainy_dir=sample_dataset_dir / "rainy",
            clean_dir=sample_dataset_dir / "clean",
        )

        for i in range(len(dataset)):
            rainy, clean = dataset[i]
            assert isinstance(rainy, torch.Tensor)
            assert isinstance(clean, torch.Tensor)

    def test_with_transform(self, sample_dataset_dir: Path) -> None:
        """Test ImagePairDataset with transform."""

        def dummy_transform(image: torch.Tensor, target: torch.Tensor) -> dict:
            """Dummy transform that returns dict."""
            return {"image": image, "target": target}

        dataset = ImagePairDataset(
            rainy_dir=sample_dataset_dir / "rainy",
            clean_dir=sample_dataset_dir / "clean",
            transform=dummy_transform,
        )

        rainy, clean = dataset[0]
        assert isinstance(rainy, torch.Tensor)
        assert isinstance(clean, torch.Tensor)

    def test_different_extensions(self, temp_dir: Path) -> None:
        """Test ImagePairDataset with different file extensions."""
        rainy_dir = temp_dir / "rainy_jpg"
        clean_dir = temp_dir / "clean_jpg"
        rainy_dir.mkdir()
        clean_dir.mkdir()

        # Create JPG images
        for i in range(3):
            img = Image.fromarray(
                torch.randint(0, 256, (128, 128, 3), dtype=torch.uint8).numpy()
            )
            img.save(rainy_dir / f"image_{i:03d}.jpg")
            img.save(clean_dir / f"image_{i:03d}.jpg")

        dataset = ImagePairDataset(
            rainy_dir=rainy_dir, clean_dir=clean_dir, extensions=(".jpg", ".jpeg")
        )
        assert len(dataset) == 3


class TestSingleFolderDataset:
    """Tests for SingleFolderDataset."""

    def test_initialization(self, sample_dataset_dir: Path) -> None:
        """Test SingleFolderDataset initialization."""
        rainy_dir = sample_dataset_dir / "rainy"
        dataset = SingleFolderDataset(image_dir=rainy_dir)
        assert len(dataset) == 5

    def test_length(self, sample_dataset_dir: Path) -> None:
        """Test SingleFolderDataset length."""
        dataset = SingleFolderDataset(image_dir=sample_dataset_dir / "rainy")
        assert len(dataset) == 5

    def test_getitem(self, sample_dataset_dir: Path) -> None:
        """Test SingleFolderDataset __getitem__."""
        dataset = SingleFolderDataset(image_dir=sample_dataset_dir / "rainy")
        img = dataset[0]

        # Check type and shape
        assert isinstance(img, torch.Tensor)
        assert img.ndim == 3
        assert img.shape[0] == 3  # RGB channels

    def test_getitem_value_range(self, sample_dataset_dir: Path) -> None:
        """Test that loaded images are in [0, 1] range."""
        dataset = SingleFolderDataset(image_dir=sample_dataset_dir / "rainy")
        img = dataset[0]

        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_get_filename(self, sample_dataset_dir: Path) -> None:
        """Test SingleFolderDataset get_filename method."""
        dataset = SingleFolderDataset(image_dir=sample_dataset_dir / "rainy")
        filename = dataset.get_filename(0)

        assert isinstance(filename, str)
        assert filename.endswith(".png")

    def test_with_transform(self, sample_dataset_dir: Path) -> None:
        """Test SingleFolderDataset with transform."""

        def dummy_transform(image: torch.Tensor) -> dict:
            """Dummy transform."""
            return {"image": image}

        dataset = SingleFolderDataset(
            image_dir=sample_dataset_dir / "rainy", transform=dummy_transform
        )

        img = dataset[0]
        assert isinstance(img, torch.Tensor)


class TestRain100Dataset:
    """Tests for Rain100Dataset."""

    def test_initialization(self, temp_dir: Path) -> None:
        """Test Rain100Dataset initialization."""
        # Create directory structure
        root_dir = temp_dir / "Rain100L"
        rainy_dir = root_dir / "rainy"
        norain_dir = root_dir / "norain"
        rainy_dir.mkdir(parents=True)
        norain_dir.mkdir(parents=True)

        # Create sample images
        for i in range(3):
            img = Image.fromarray(
                torch.randint(0, 256, (128, 128, 3), dtype=torch.uint8).numpy()
            )
            img.save(rainy_dir / f"image_{i:03d}.png")
            img.save(norain_dir / f"image_{i:03d}.png")

        dataset = Rain100Dataset(root_dir=root_dir)
        assert len(dataset) == 3

    def test_inheritance(self, temp_dir: Path) -> None:
        """Test that Rain100Dataset inherits from ImagePairDataset."""
        # Create directory structure
        root_dir = temp_dir / "Rain100L"
        rainy_dir = root_dir / "rainy"
        norain_dir = root_dir / "norain"
        rainy_dir.mkdir(parents=True)
        norain_dir.mkdir(parents=True)

        # Create sample images
        img = Image.fromarray(
            torch.randint(0, 256, (128, 128, 3), dtype=torch.uint8).numpy()
        )
        img.save(rainy_dir / "image_000.png")
        img.save(norain_dir / "image_000.png")

        dataset = Rain100Dataset(root_dir=root_dir)
        assert isinstance(dataset, ImagePairDataset)

    def test_getitem(self, temp_dir: Path) -> None:
        """Test Rain100Dataset __getitem__."""
        # Create directory structure
        root_dir = temp_dir / "Rain100L"
        rainy_dir = root_dir / "rainy"
        norain_dir = root_dir / "norain"
        rainy_dir.mkdir(parents=True)
        norain_dir.mkdir(parents=True)

        # Create sample images
        img = Image.fromarray(
            torch.randint(0, 256, (128, 128, 3), dtype=torch.uint8).numpy()
        )
        img.save(rainy_dir / "image_000.png")
        img.save(norain_dir / "image_000.png")

        dataset = Rain100Dataset(root_dir=root_dir)
        rainy, clean = dataset[0]

        assert isinstance(rainy, torch.Tensor)
        assert isinstance(clean, torch.Tensor)
        assert rainy.shape[0] == 3
        assert clean.shape[0] == 3


class TestDataLoaderIntegration:
    """Integration tests with PyTorch DataLoader."""

    def test_dataloader_with_imagepair(self, sample_dataset_dir: Path) -> None:
        """Test ImagePairDataset with DataLoader."""
        from torch.utils.data import DataLoader

        dataset = ImagePairDataset(
            rainy_dir=sample_dataset_dir / "rainy",
            clean_dir=sample_dataset_dir / "clean",
        )

        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # Get a batch
        for rainy_batch, clean_batch in dataloader:
            assert rainy_batch.shape[0] == 2  # Batch size
            assert rainy_batch.shape[1] == 3  # RGB channels
            assert clean_batch.shape[0] == 2
            assert clean_batch.shape[1] == 3
            break

    def test_dataloader_with_singlefolder(self, sample_dataset_dir: Path) -> None:
        """Test SingleFolderDataset with DataLoader."""
        from torch.utils.data import DataLoader

        dataset = SingleFolderDataset(image_dir=sample_dataset_dir / "rainy")
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        # Get a batch
        for img_batch in dataloader:
            assert img_batch.shape[0] == 2  # Batch size
            assert img_batch.shape[1] == 3  # RGB channels
            break

    def test_dataloader_iterations(self, sample_dataset_dir: Path) -> None:
        """Test full iteration through DataLoader."""
        from torch.utils.data import DataLoader

        dataset = ImagePairDataset(
            rainy_dir=sample_dataset_dir / "rainy",
            clean_dir=sample_dataset_dir / "clean",
        )

        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        total_samples = 0
        for rainy_batch, clean_batch in dataloader:
            total_samples += rainy_batch.shape[0]
            assert rainy_batch.shape[1] == 3  # RGB channels
            assert clean_batch.shape[1] == 3  # RGB channels

        assert total_samples == len(dataset)
