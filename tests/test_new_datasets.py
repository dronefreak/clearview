"""Unit tests for the Rain13K, DDN-Data, and DID-Data dataset classes."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from clearview.data.datasets import (
    DDNDataDataset,
    DIDDataDataset,
    GTRainDataset,
    ImagePairDataset,
    MixedDataset,
    NTIREHazeDataset,
    Rain13KDataset,
    RainDropDataset,
    RainDSDataset,
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


class TestGTRainDataset:
    """Tests for GTRainDataset."""

    def test_one_clean_shared_across_rainy(self, temp_dir: Path) -> None:
        """Test the common case: one clean frame broadcast to many rainy frames."""
        root_dir = temp_dir / "GT-RAIN_train"
        scene_dir = root_dir / "Scene_0-0"
        scene_dir.mkdir(parents=True)

        _save_random_image(scene_dir / "Scene_0-0-Webcam-C-000.png")
        for i in range(5):
            _save_random_image(scene_dir / f"Scene_0-0-Webcam-R-{i:03d}.png")

        dataset = GTRainDataset(root_dir=root_dir)
        assert len(dataset) == 5
        assert all(c.name == "Scene_0-0-Webcam-C-000.png" for c in dataset.clean_files)

    def test_one_clean_per_rainy_matched_by_index(self, temp_dir: Path) -> None:
        """Test the 'Gurutto_1-2'-style case: distinct clean frame per rainy frame."""
        root_dir = temp_dir / "GT-RAIN_val"
        scene_dir = root_dir / "Gurutto_1-2"
        scene_dir.mkdir(parents=True)

        for i in range(4):
            _save_random_image(scene_dir / f"Gurutto_1-2-Webcam-C-{i:03d}.png")
            _save_random_image(scene_dir / f"Gurutto_1-2-Webcam-R-{i:03d}.png")

        dataset = GTRainDataset(root_dir=root_dir)
        assert len(dataset) == 4
        for rainy, clean in zip(dataset.rainy_files, dataset.clean_files):
            assert rainy.name.split("-R-")[-1] == clean.name.split("-C-")[-1]

    def test_multiple_scenes(self, temp_dir: Path) -> None:
        """Test that pairs from multiple scene directories are all collected."""
        root_dir = temp_dir / "GT-RAIN_test"
        for scene in ("Scene_A", "Scene_B"):
            scene_dir = root_dir / scene
            scene_dir.mkdir(parents=True)
            _save_random_image(scene_dir / f"{scene}-Webcam-C-000.png")
            for i in range(3):
                _save_random_image(scene_dir / f"{scene}-Webcam-R-{i:03d}.png")

        dataset = GTRainDataset(root_dir=root_dir)
        assert len(dataset) == 6

    def test_getitem_shapes(self, temp_dir: Path) -> None:
        """Test GTRainDataset __getitem__ returns valid image tensors."""
        root_dir = temp_dir / "GT-RAIN_train"
        scene_dir = root_dir / "Scene_0-0"
        scene_dir.mkdir(parents=True)
        _save_random_image(scene_dir / "Scene_0-0-Webcam-C-000.png")
        _save_random_image(scene_dir / "Scene_0-0-Webcam-R-000.png")

        dataset = GTRainDataset(root_dir=root_dir)
        rainy, clean = dataset[0]

        assert isinstance(rainy, torch.Tensor)
        assert isinstance(clean, torch.Tensor)
        assert rainy.shape[0] == 3
        assert clean.shape[0] == 3

    def test_no_scene_dirs_raises(self, temp_dir: Path) -> None:
        """Test that an empty root directory raises FileNotFoundError."""
        root_dir = temp_dir / "GT-RAIN_train"
        root_dir.mkdir(parents=True)

        with pytest.raises(FileNotFoundError):
            GTRainDataset(root_dir=root_dir)

    def test_no_pairs_found_raises(self, temp_dir: Path) -> None:
        """Test that scenes with no matching -R-/-C- files raise ValueError."""
        root_dir = temp_dir / "GT-RAIN_train"
        scene_dir = root_dir / "Scene_0-0"
        scene_dir.mkdir(parents=True)
        _save_random_image(scene_dir / "unrelated.png")

        with pytest.raises(ValueError):
            GTRainDataset(root_dir=root_dir)


class TestRainDropDataset:
    """Tests for RainDropDataset."""

    def test_basic_layout(self, temp_dir: Path) -> None:
        """Test RainDropDataset with a flat data/gt directory layout."""
        root_dir = temp_dir / "RainDrop" / "train"
        rainy_dir = root_dir / "data"
        clean_dir = root_dir / "gt"
        rainy_dir.mkdir(parents=True)
        clean_dir.mkdir(parents=True)

        for i in range(4):
            _save_random_image(rainy_dir / f"{i}_rain.png")
            _save_random_image(clean_dir / f"{i}_clean.png")

        dataset = RainDropDataset(root_dir=root_dir)
        assert len(dataset) == 4

    def test_split_argument(self, temp_dir: Path) -> None:
        """Test RainDropDataset with an explicit split subdirectory."""
        root_dir = temp_dir / "RainDrop"
        for split in ("train", "test_a", "test_b"):
            rainy_dir = root_dir / split / "data"
            clean_dir = root_dir / split / "gt"
            rainy_dir.mkdir(parents=True)
            clean_dir.mkdir(parents=True)
            _save_random_image(rainy_dir / "0_rain.png")
            _save_random_image(clean_dir / "0_clean.png")

        train_dataset = RainDropDataset(root_dir=root_dir, split="train")
        test_a_dataset = RainDropDataset(root_dir=root_dir, split="test_a")
        assert len(train_dataset) == 1
        assert len(test_a_dataset) == 1

    def test_mixed_extensions_per_split(self, temp_dir: Path) -> None:
        """Test that a split using .jpg (like the official test_b) works too."""
        root_dir = temp_dir / "RainDrop" / "test_b"
        rainy_dir = root_dir / "data"
        clean_dir = root_dir / "gt"
        rainy_dir.mkdir(parents=True)
        clean_dir.mkdir(parents=True)

        for i in range(3):
            _save_random_image(rainy_dir / f"{i}_rain.jpg")
            _save_random_image(clean_dir / f"{i}_clean.jpg")

        dataset = RainDropDataset(root_dir=root_dir)
        assert len(dataset) == 3

    def test_getitem_shapes(self, temp_dir: Path) -> None:
        """Test RainDropDataset __getitem__ returns valid image tensors."""
        root_dir = temp_dir / "RainDrop" / "train"
        rainy_dir = root_dir / "data"
        clean_dir = root_dir / "gt"
        rainy_dir.mkdir(parents=True)
        clean_dir.mkdir(parents=True)
        _save_random_image(rainy_dir / "0_rain.png")
        _save_random_image(clean_dir / "0_clean.png")

        dataset = RainDropDataset(root_dir=root_dir)
        rainy, clean = dataset[0]

        assert isinstance(rainy, torch.Tensor)
        assert isinstance(clean, torch.Tensor)
        assert rainy.shape[0] == 3
        assert clean.shape[0] == 3

    def test_numeric_id_ordering(self, temp_dir: Path) -> None:
        """Test that numeric IDs are ordered numerically, not lexically."""
        root_dir = temp_dir / "RainDrop" / "train"
        rainy_dir = root_dir / "data"
        clean_dir = root_dir / "gt"
        rainy_dir.mkdir(parents=True)
        clean_dir.mkdir(parents=True)

        for i in [2, 10, 1]:
            _save_random_image(rainy_dir / f"{i}_rain.png")
            _save_random_image(clean_dir / f"{i}_clean.png")

        dataset = RainDropDataset(root_dir=root_dir)
        ids = [f.stem[: -len("_rain")] for f in dataset.rainy_files]
        assert ids == ["1", "2", "10"]

    def test_missing_layout_raises(self, temp_dir: Path) -> None:
        """Test that an unsupported directory layout raises FileNotFoundError."""
        root_dir = temp_dir / "RainDrop" / "train"
        root_dir.mkdir(parents=True)

        with pytest.raises(FileNotFoundError):
            RainDropDataset(root_dir=root_dir)

    def test_unpaired_images_raises(self, temp_dir: Path) -> None:
        """Test that mismatched data/gt IDs raise ValueError."""
        root_dir = temp_dir / "RainDrop" / "train"
        rainy_dir = root_dir / "data"
        clean_dir = root_dir / "gt"
        rainy_dir.mkdir(parents=True)
        clean_dir.mkdir(parents=True)

        _save_random_image(rainy_dir / "0_rain.png")
        _save_random_image(clean_dir / "1_clean.png")

        with pytest.raises(ValueError):
            RainDropDataset(root_dir=root_dir)


class TestNTIREHazeDataset:
    """Tests for NTIREHazeDataset."""

    def test_gt_hazy_subdirs(self, temp_dir: Path) -> None:
        """Test with separate GT/hazy subdirectories (I-Haze/O-Haze/Dense-Haze style)."""
        root_dir = temp_dir / "I-HAZE"
        gt_dir = root_dir / "GT"
        hazy_dir = root_dir / "hazy"
        gt_dir.mkdir(parents=True)
        hazy_dir.mkdir(parents=True)

        for i in range(3):
            _save_random_image(gt_dir / f"{i:02d}_indoor_GT.jpg")
            _save_random_image(hazy_dir / f"{i:02d}_indoor_hazy.jpg")

        dataset = NTIREHazeDataset(root_dir=root_dir)
        assert len(dataset) == 3

    def test_flat_layout(self, temp_dir: Path) -> None:
        """Test with a flat directory mixing GT and hazy files (NH-Haze style)."""
        root_dir = temp_dir / "NH-HAZE"
        root_dir.mkdir(parents=True)

        for i in range(4):
            _save_random_image(root_dir / f"{i:02d}_GT.png")
            _save_random_image(root_dir / f"{i:02d}_hazy.png")

        dataset = NTIREHazeDataset(root_dir=root_dir)
        assert len(dataset) == 4

    def test_mixed_extension_case(self, temp_dir: Path) -> None:
        """Test that mixed-case extensions (.jpg vs .JPG, as in O-Haze) both match."""
        root_dir = temp_dir / "O-HAZE"
        gt_dir = root_dir / "GT"
        hazy_dir = root_dir / "hazy"
        gt_dir.mkdir(parents=True)
        hazy_dir.mkdir(parents=True)

        _save_random_image(gt_dir / "03_outdoor_GT.JPG")
        _save_random_image(hazy_dir / "03_outdoor_hazy.JPG")

        dataset = NTIREHazeDataset(root_dir=root_dir)
        assert len(dataset) == 1

    def test_getitem_shapes(self, temp_dir: Path) -> None:
        """Test NTIREHazeDataset __getitem__ returns valid image tensors."""
        root_dir = temp_dir / "Dense-Haze"
        gt_dir = root_dir / "GT"
        hazy_dir = root_dir / "hazy"
        gt_dir.mkdir(parents=True)
        hazy_dir.mkdir(parents=True)
        _save_random_image(gt_dir / "01_GT.png")
        _save_random_image(hazy_dir / "01_hazy.png")

        dataset = NTIREHazeDataset(root_dir=root_dir)
        hazy, clean = dataset[0]

        assert isinstance(hazy, torch.Tensor)
        assert isinstance(clean, torch.Tensor)
        assert hazy.shape[0] == 3
        assert clean.shape[0] == 3

    def test_no_pairs_found_raises(self, temp_dir: Path) -> None:
        """Test that a directory with no _GT/_hazy markers raises ValueError."""
        root_dir = temp_dir / "Empty-Haze"
        root_dir.mkdir(parents=True)
        _save_random_image(root_dir / "unrelated.png")

        with pytest.raises(ValueError):
            NTIREHazeDataset(root_dir=root_dir)


class TestRainDSDataset:
    """Tests for RainDSDataset."""

    def test_syn_style_markers(self, temp_dir: Path) -> None:
        """Test the RainDS_syn marker-prefixed naming convention."""
        root_dir = temp_dir / "RainDS_syn" / "train"
        gt_dir = root_dir / "gt"
        both_dir = root_dir / "rainstreak_raindrop"
        gt_dir.mkdir(parents=True)
        both_dir.mkdir(parents=True)

        for i in range(4):
            _save_random_image(gt_dir / f"norain-{i}.png")
            _save_random_image(both_dir / f"rd-rain-{i}.png")

        dataset = RainDSDataset(root_dir=root_dir, degradation="rainstreak_raindrop")
        assert len(dataset) == 4

    def test_pie_subcollection_does_not_collide(self, temp_dir: Path) -> None:
        """Test that 'pie-' and plain sub-collections with overlapping IDs don't collide."""
        root_dir = temp_dir / "RainDS_syn" / "train"
        gt_dir = root_dir / "gt"
        rd_dir = root_dir / "raindrop"
        gt_dir.mkdir(parents=True)
        rd_dir.mkdir(parents=True)

        # Same numeric ID, but one plain and one "pie-" prefixed: 2 distinct pairs.
        _save_random_image(gt_dir / "norain-101.png")
        _save_random_image(rd_dir / "rd-101.png")
        _save_random_image(gt_dir / "pie-norain-101.png")
        _save_random_image(rd_dir / "pie-rd-101.png")

        dataset = RainDSDataset(root_dir=root_dir, degradation="raindrop")
        assert len(dataset) == 2

    def test_real_style_plain_stems(self, temp_dir: Path) -> None:
        """Test the RainDS_real plain identical-stem naming convention."""
        root_dir = temp_dir / "RainDS_real" / "test_set"
        gt_dir = root_dir / "gt"
        rain_dir = root_dir / "rainstreak"
        gt_dir.mkdir(parents=True)
        rain_dir.mkdir(parents=True)

        for i in range(3):
            _save_random_image(gt_dir / f"{i}.png")
            _save_random_image(rain_dir / f"{i}.png")

        dataset = RainDSDataset(root_dir=root_dir, degradation="rainstreak")
        assert len(dataset) == 3

    def test_unmatched_files_excluded_not_fatal(self, temp_dir: Path) -> None:
        """Test that an unmatched filename (e.g. IMG_7435.png) is excluded, not an error."""
        root_dir = temp_dir / "RainDS_real" / "test_set"
        gt_dir = root_dir / "gt"
        rain_dir = root_dir / "rainstreak"
        gt_dir.mkdir(parents=True)
        rain_dir.mkdir(parents=True)

        _save_random_image(gt_dir / "218.png")
        _save_random_image(rain_dir / "IMG_7435.png")
        _save_random_image(gt_dir / "101.png")
        _save_random_image(rain_dir / "101.png")

        dataset = RainDSDataset(root_dir=root_dir, degradation="rainstreak")
        assert len(dataset) == 1

    def test_getitem_shapes(self, temp_dir: Path) -> None:
        """Test RainDSDataset __getitem__ returns valid image tensors."""
        root_dir = temp_dir / "RainDS_syn" / "train"
        gt_dir = root_dir / "gt"
        rd_dir = root_dir / "raindrop"
        gt_dir.mkdir(parents=True)
        rd_dir.mkdir(parents=True)
        _save_random_image(gt_dir / "norain-0.png")
        _save_random_image(rd_dir / "rd-0.png")

        dataset = RainDSDataset(root_dir=root_dir, degradation="raindrop")
        degraded, clean = dataset[0]

        assert isinstance(degraded, torch.Tensor)
        assert isinstance(clean, torch.Tensor)
        assert degraded.shape[0] == 3
        assert clean.shape[0] == 3

    def test_invalid_degradation_raises(self, temp_dir: Path) -> None:
        """Test that an unrecognized degradation category raises ValueError."""
        root_dir = temp_dir / "RainDS_syn" / "train"
        root_dir.mkdir(parents=True)

        with pytest.raises(ValueError):
            RainDSDataset(root_dir=root_dir, degradation="not-a-real-category")

    def test_missing_dir_raises(self, temp_dir: Path) -> None:
        """Test that a missing gt directory raises FileNotFoundError."""
        root_dir = temp_dir / "RainDS_syn" / "train"
        root_dir.mkdir(parents=True)

        with pytest.raises(FileNotFoundError):
            RainDSDataset(root_dir=root_dir, degradation="raindrop")


def _make_pair_dataset(temp_dir: Path, name: str, n: int) -> ImagePairDataset:
    """Build a tiny ImagePairDataset with n synthetic rainy/clean pairs."""
    rainy_dir = temp_dir / name / "rainy"
    clean_dir = temp_dir / name / "clean"
    rainy_dir.mkdir(parents=True)
    clean_dir.mkdir(parents=True)
    for i in range(n):
        _save_random_image(rainy_dir / f"{i:03d}.png")
        _save_random_image(clean_dir / f"{i:03d}.png")
    return ImagePairDataset(rainy_dir=rainy_dir, clean_dir=clean_dir)


class TestMixedDataset:
    """Tests for MixedDataset."""

    def test_length_is_sum_of_sources(self, temp_dir: Path) -> None:
        """Test that length is the sum of all source dataset lengths."""
        a = _make_pair_dataset(temp_dir, "a", 3)
        b = _make_pair_dataset(temp_dir, "b", 5)

        mixed = MixedDataset([a, b])

        assert len(mixed) == 8

    def test_indexing_spans_source_boundaries(self, temp_dir: Path) -> None:
        """Test that indexing past the first source reaches the second."""
        a = _make_pair_dataset(temp_dir, "a", 2)
        b = _make_pair_dataset(temp_dir, "b", 2)

        mixed = MixedDataset([a, b])

        for idx in range(len(mixed)):
            rainy, clean = mixed[idx]
            assert rainy.shape == a[0][0].shape
            assert clean.shape == a[0][1].shape

    def test_empty_datasets_raises(self) -> None:
        """Test that an empty datasets list raises ValueError."""
        with pytest.raises(ValueError):
            MixedDataset([])

    def test_mismatched_weights_length_raises(self, temp_dir: Path) -> None:
        """Test that a weights list of the wrong length raises ValueError."""
        a = _make_pair_dataset(temp_dir, "a", 2)
        b = _make_pair_dataset(temp_dir, "b", 2)

        with pytest.raises(ValueError):
            MixedDataset([a, b], weights=[1.0])

    def test_default_weights_are_uniform(self, temp_dir: Path) -> None:
        """Test that sample_weights() defaults to 1.0 per example."""
        a = _make_pair_dataset(temp_dir, "a", 3)
        b = _make_pair_dataset(temp_dir, "b", 2)

        mixed = MixedDataset([a, b])

        assert mixed.sample_weights() == [1.0] * 5

    def test_sample_weights_reflect_per_source_weight(self, temp_dir: Path) -> None:
        """Test that each example inherits its source dataset's weight."""
        a = _make_pair_dataset(temp_dir, "a", 3)  # e.g. a large synthetic source
        b = _make_pair_dataset(temp_dir, "b", 2)  # e.g. a small real-world source

        mixed = MixedDataset([a, b], weights=[1.0, 2.0])

        assert mixed.sample_weights() == [1.0, 1.0, 1.0, 2.0, 2.0]

    def test_weighted_sampler_oversamples_small_source(self, temp_dir: Path) -> None:
        """Test that WeightedRandomSampler actually draws the up-weighted
        (smaller) source more often than its natural frequency would give."""
        big = _make_pair_dataset(temp_dir, "big", 90)
        small = _make_pair_dataset(temp_dir, "small", 10)

        mixed = MixedDataset([big, small], weights=[1.0, 9.0])
        sampler = torch.utils.data.WeightedRandomSampler(
            mixed.sample_weights(), num_samples=2000, replacement=True
        )

        drawn = list(sampler)
        from_small = sum(1 for idx in drawn if idx >= len(big))

        # Natural frequency would be ~10%; weight=9 on a 9x-smaller source
        # targets ~50/50, so this should land well above natural frequency.
        assert from_small / len(drawn) > 0.35
