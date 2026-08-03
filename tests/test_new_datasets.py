"""Unit tests for the Rain13K, DDN-Data, and DID-Data dataset classes."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from clearview.data.datasets import (
    DDNDataDataset,
    DIDDataDataset,
    ImagePairDataset,
    Rain13KDataset,
    SPADataDataset,
)


def _save_random_image(path: Path, size: int = 64) -> None:
    """Save a random RGB image to disk."""
    img = Image.fromarray(
        torch.randint(0, 256, (size, size, 3), dtype=torch.uint8).numpy()
    )
    img.save(path)


class TestRain13KDataset:
    """Tests for Rain13KDataset."""

    def test_initialization_with_split(self, temp_dir: Path) -> None:
        """Test Rain13KDataset initialization with a split subdirectory."""
        root_dir = temp_dir / "Rain13K"
        input_dir = root_dir / "train" / "input"
        target_dir = root_dir / "train" / "target"
        input_dir.mkdir(parents=True)
        target_dir.mkdir(parents=True)

        for i in range(3):
            _save_random_image(input_dir / f"{i:03d}.png")
            _save_random_image(target_dir / f"{i:03d}.png")

        dataset = Rain13KDataset(root_dir=root_dir, split="train")
        assert len(dataset) == 3

    def test_initialization_without_split(self, temp_dir: Path) -> None:
        """Test Rain13KDataset initialization when root_dir is already the split dir."""
        root_dir = temp_dir / "Rain13K" / "train"
        input_dir = root_dir / "input"
        target_dir = root_dir / "target"
        input_dir.mkdir(parents=True)
        target_dir.mkdir(parents=True)

        _save_random_image(input_dir / "000.png")
        _save_random_image(target_dir / "000.png")

        dataset = Rain13KDataset(root_dir=root_dir)
        assert len(dataset) == 1

    def test_alternate_directory_names(self, temp_dir: Path) -> None:
        """Test Rain13KDataset resolves alternate rainy/clean directory names."""
        root_dir = temp_dir / "Rain13K" / "test"
        rainy_dir = root_dir / "rainy"
        clean_dir = root_dir / "gt"
        rainy_dir.mkdir(parents=True)
        clean_dir.mkdir(parents=True)

        _save_random_image(rainy_dir / "000.png")
        _save_random_image(clean_dir / "000.png")

        dataset = Rain13KDataset(root_dir=root_dir)
        assert len(dataset) == 1

    def test_missing_directories_raises(self, temp_dir: Path) -> None:
        """Test that missing rainy/clean directories raise FileNotFoundError."""
        root_dir = temp_dir / "Rain13K" / "train"
        root_dir.mkdir(parents=True)

        with pytest.raises(FileNotFoundError):
            Rain13KDataset(root_dir=root_dir)

    def test_inheritance(self, temp_dir: Path) -> None:
        """Test that Rain13KDataset inherits from ImagePairDataset."""
        root_dir = temp_dir / "Rain13K" / "train"
        input_dir = root_dir / "input"
        target_dir = root_dir / "target"
        input_dir.mkdir(parents=True)
        target_dir.mkdir(parents=True)
        _save_random_image(input_dir / "000.png")
        _save_random_image(target_dir / "000.png")

        dataset = Rain13KDataset(root_dir=root_dir)
        assert isinstance(dataset, ImagePairDataset)

    def test_getitem(self, temp_dir: Path) -> None:
        """Test Rain13KDataset __getitem__ returns valid tensors."""
        root_dir = temp_dir / "Rain13K" / "train"
        input_dir = root_dir / "input"
        target_dir = root_dir / "target"
        input_dir.mkdir(parents=True)
        target_dir.mkdir(parents=True)
        _save_random_image(input_dir / "000.png")
        _save_random_image(target_dir / "000.png")

        dataset = Rain13KDataset(root_dir=root_dir)
        rainy, clean = dataset[0]

        assert isinstance(rainy, torch.Tensor)
        assert isinstance(clean, torch.Tensor)
        assert rainy.shape[0] == 3
        assert clean.shape[0] == 3


class TestDDNDataDataset:
    """Tests for DDNDataDataset."""

    def test_initialization_with_split(self, temp_dir: Path) -> None:
        """Test DDNDataDataset initialization with multi-variant rainy images."""
        root_dir = temp_dir / "DDN-Data"
        rainy_dir = root_dir / "train" / "rainy_image"
        clean_dir = root_dir / "train" / "ground_truth"
        rainy_dir.mkdir(parents=True)
        clean_dir.mkdir(parents=True)

        # 2 clean images, each with 3 rainy variants
        for clean_id in (1, 2):
            _save_random_image(clean_dir / f"{clean_id}.jpg")
            for variant in (1, 2, 3):
                _save_random_image(rainy_dir / f"{clean_id}_{variant}.jpg")

        dataset = DDNDataDataset(root_dir=root_dir, split="train")
        assert len(dataset) == 6

    def test_getitem(self, temp_dir: Path) -> None:
        """Test DDNDataDataset __getitem__ pairs rainy variants with the right clean image."""
        root_dir = temp_dir / "DDN-Data" / "train"
        rainy_dir = root_dir / "rainy_image"
        clean_dir = root_dir / "ground_truth"
        rainy_dir.mkdir(parents=True)
        clean_dir.mkdir(parents=True)

        _save_random_image(clean_dir / "1.jpg")
        _save_random_image(rainy_dir / "1_1.jpg")

        dataset = DDNDataDataset(root_dir=root_dir)
        rainy, clean = dataset[0]

        assert isinstance(rainy, torch.Tensor)
        assert isinstance(clean, torch.Tensor)

    def test_missing_directories_raises(self, temp_dir: Path) -> None:
        """Test that missing directories raise FileNotFoundError."""
        root_dir = temp_dir / "DDN-Data" / "train"
        root_dir.mkdir(parents=True)

        with pytest.raises(FileNotFoundError):
            DDNDataDataset(root_dir=root_dir)


class TestDIDDataDataset:
    """Tests for DIDDataDataset."""

    def test_flat_layout(self, temp_dir: Path) -> None:
        """Test DIDDataDataset with a flat rainy/clean directory layout."""
        root_dir = temp_dir / "DID-Data" / "train"
        rainy_dir = root_dir / "rainy"
        clean_dir = root_dir / "clean"
        rainy_dir.mkdir(parents=True)
        clean_dir.mkdir(parents=True)

        for i in range(4):
            _save_random_image(rainy_dir / f"{i:03d}.png")
            _save_random_image(clean_dir / f"{i:03d}.png")

        dataset = DIDDataDataset(root_dir=root_dir)
        assert len(dataset) == 4
        assert dataset.get_density_label(0) is None

    def test_density_subfolder_layout(self, temp_dir: Path) -> None:
        """Test DIDDataDataset with density-aware subfolders."""
        root_dir = temp_dir / "DID-Data" / "train"

        for density in ("Rain_Light", "Rain_Medium", "Rain_Heavy"):
            rainy_dir = root_dir / density / "rainy"
            clean_dir = root_dir / density / "clean"
            rainy_dir.mkdir(parents=True)
            clean_dir.mkdir(parents=True)
            for i in range(2):
                _save_random_image(rainy_dir / f"{i:03d}.png")
                _save_random_image(clean_dir / f"{i:03d}.png")

        dataset = DIDDataDataset(root_dir=root_dir)
        assert len(dataset) == 6

        labels = {dataset.get_density_label(i) for i in range(len(dataset))}
        assert labels == {"light", "medium", "heavy"}

    def test_getitem_shapes(self, temp_dir: Path) -> None:
        """Test DIDDataDataset __getitem__ returns valid image tensors."""
        root_dir = temp_dir / "DID-Data" / "train"
        rainy_dir = root_dir / "rainy"
        clean_dir = root_dir / "clean"
        rainy_dir.mkdir(parents=True)
        clean_dir.mkdir(parents=True)
        _save_random_image(rainy_dir / "000.png")
        _save_random_image(clean_dir / "000.png")

        dataset = DIDDataDataset(root_dir=root_dir)
        rainy, clean = dataset[0]

        assert isinstance(rainy, torch.Tensor)
        assert isinstance(clean, torch.Tensor)
        assert rainy.shape[0] == 3
        assert clean.shape[0] == 3

    def test_missing_layout_raises(self, temp_dir: Path) -> None:
        """Test that an unsupported directory layout raises FileNotFoundError."""
        root_dir = temp_dir / "DID-Data" / "train"
        root_dir.mkdir(parents=True)

        with pytest.raises(FileNotFoundError):
            DIDDataDataset(root_dir=root_dir)

    def test_unpaired_images_raises(self, temp_dir: Path) -> None:
        """Test that mismatched rainy/clean filenames raise ValueError."""
        root_dir = temp_dir / "DID-Data" / "train"
        rainy_dir = root_dir / "rainy"
        clean_dir = root_dir / "clean"
        rainy_dir.mkdir(parents=True)
        clean_dir.mkdir(parents=True)

        _save_random_image(rainy_dir / "000.png")
        _save_random_image(clean_dir / "001.png")

        with pytest.raises(ValueError):
            DIDDataDataset(root_dir=root_dir)


class TestSPADataDataset:
    """Tests for SPADataDataset."""

    def test_flat_rain_norain_layout(self, temp_dir: Path) -> None:
        """Test SPADataDataset with a flat rain/norain directory layout."""
        root_dir = temp_dir / "SPA-Data" / "train"
        rainy_dir = root_dir / "rain"
        clean_dir = root_dir / "norain"
        rainy_dir.mkdir(parents=True)
        clean_dir.mkdir(parents=True)

        for i in range(4):
            _save_random_image(rainy_dir / f"rain-{i}.png")
            _save_random_image(clean_dir / f"norain-{i}.png")

        dataset = SPADataDataset(root_dir=root_dir)
        assert len(dataset) == 4

    def test_rgb_reconstruction_layout(self, temp_dir: Path) -> None:
        """Test SPADataDataset with the nested rgb_reconstruction layout."""
        root_dir = temp_dir / "SPA-Data" / "train"
        rainy_dir = root_dir / "rgb_reconstruction" / "rain"
        clean_dir = root_dir / "rgb_reconstruction" / "norain"
        rainy_dir.mkdir(parents=True)
        clean_dir.mkdir(parents=True)

        for i in range(3):
            _save_random_image(rainy_dir / f"rain-{i}.png")
            _save_random_image(clean_dir / f"norain-{i}.png")

        dataset = SPADataDataset(root_dir=root_dir)
        assert len(dataset) == 3

    def test_split_argument(self, temp_dir: Path) -> None:
        """Test SPADataDataset with an explicit split subdirectory."""
        root_dir = temp_dir / "SPA-Data"
        for split in ("train", "val"):
            rainy_dir = root_dir / split / "rain"
            clean_dir = root_dir / split / "norain"
            rainy_dir.mkdir(parents=True)
            clean_dir.mkdir(parents=True)
            _save_random_image(rainy_dir / "rain-0.png")
            _save_random_image(clean_dir / "norain-0.png")

        train_dataset = SPADataDataset(root_dir=root_dir, split="train")
        val_dataset = SPADataDataset(root_dir=root_dir, split="val")
        assert len(train_dataset) == 1
        assert len(val_dataset) == 1

    def test_getitem_shapes(self, temp_dir: Path) -> None:
        """Test SPADataDataset __getitem__ returns valid image tensors."""
        root_dir = temp_dir / "SPA-Data" / "train"
        rainy_dir = root_dir / "rain"
        clean_dir = root_dir / "norain"
        rainy_dir.mkdir(parents=True)
        clean_dir.mkdir(parents=True)
        _save_random_image(rainy_dir / "rain-0.png")
        _save_random_image(clean_dir / "norain-0.png")

        dataset = SPADataDataset(root_dir=root_dir)
        rainy, clean = dataset[0]

        assert isinstance(rainy, torch.Tensor)
        assert isinstance(clean, torch.Tensor)
        assert rainy.shape[0] == 3
        assert clean.shape[0] == 3

    def test_numeric_id_ordering(self, temp_dir: Path) -> None:
        """Test that numeric IDs are ordered numerically, not lexically."""
        root_dir = temp_dir / "SPA-Data" / "train"
        rainy_dir = root_dir / "rain"
        clean_dir = root_dir / "norain"
        rainy_dir.mkdir(parents=True)
        clean_dir.mkdir(parents=True)

        for i in [2, 10, 1]:
            _save_random_image(rainy_dir / f"rain-{i}.png")
            _save_random_image(clean_dir / f"norain-{i}.png")

        dataset = SPADataDataset(root_dir=root_dir)
        ids = [f.stem[len("rain-") :] for f in dataset.rainy_files]
        assert ids == ["1", "2", "10"]

    def test_missing_layout_raises(self, temp_dir: Path) -> None:
        """Test that an unsupported directory layout raises FileNotFoundError."""
        root_dir = temp_dir / "SPA-Data" / "train"
        root_dir.mkdir(parents=True)

        with pytest.raises(FileNotFoundError):
            SPADataDataset(root_dir=root_dir)

    def test_unpaired_images_raises(self, temp_dir: Path) -> None:
        """Test that mismatched rain/norain IDs raise ValueError."""
        root_dir = temp_dir / "SPA-Data" / "train"
        rainy_dir = root_dir / "rain"
        clean_dir = root_dir / "norain"
        rainy_dir.mkdir(parents=True)
        clean_dir.mkdir(parents=True)

        _save_random_image(rainy_dir / "rain-0.png")
        _save_random_image(clean_dir / "norain-1.png")

        with pytest.raises(ValueError):
            SPADataDataset(root_dir=root_dir)
