"""Dataset implementations for image deraining.

Provides dataset classes for loading paired rainy/clean images from
various formats (image pairs, directories, etc.).
"""

from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import logging

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from clearview.utils.image import numpy_to_tensor


logger = logging.getLogger(__name__)


class ImagePairDataset(Dataset):
    """Dataset for paired rainy/clean images.

    Loads images from parallel directory structures:
    ```
    data/
    ├── rainy/
    │   ├── img001.png
    │   └── img002.png
    └── clean/
        ├── img001.png
        └── img002.png
    ```

    Args:
        rainy_dir: Directory containing rainy images
        clean_dir: Directory containing clean images
        transform: Optional transform to apply to both images
        extensions: Valid image extensions

    Example:
        >>> dataset = ImagePairDataset(
        ...     rainy_dir='data/train/rainy',
        ...     clean_dir='data/train/clean'
        ... )
        >>> rainy, clean = dataset[0]
    """

    def __init__(
        self,
        rainy_dir: Union[str, Path],
        clean_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ) -> None:
        """Initialize dataset."""
        self.rainy_dir = Path(rainy_dir)
        self.clean_dir = Path(clean_dir)
        self.transform = transform
        self.extensions = extensions

        # Get image file lists
        self.rainy_files = sorted(
            [f for f in self.rainy_dir.iterdir() if f.suffix.lower() in extensions]
        )

        self.clean_files = sorted(
            [f for f in self.clean_dir.iterdir() if f.suffix.lower() in extensions]
        )

        # Validate
        if len(self.rainy_files) != len(self.clean_files):
            raise ValueError(
                f"Mismatch in number of images: "
                f"{len(self.rainy_files)} rainy vs {len(self.clean_files)} clean"
            )

        logger.info(f"Loaded {len(self.rainy_files)} image pairs")

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.rainy_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a rainy/clean image pair.

        Args:
            idx: Index

        Returns:
            Tuple of (rainy_tensor, clean_tensor)
        """
        # Load images
        rainy_img = Image.open(self.rainy_files[idx]).convert("RGB")
        clean_img = Image.open(self.clean_files[idx]).convert("RGB")

        # Convert to numpy
        rainy_np = np.array(rainy_img).astype(np.float32) / 255.0
        clean_np = np.array(clean_img).astype(np.float32) / 255.0

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=rainy_np, target=clean_np)
            rainy_np = transformed["image"]
            clean_np = transformed["target"]

        # Convert to tensors
        rainy_tensor = numpy_to_tensor(rainy_np)
        clean_tensor = numpy_to_tensor(clean_np)

        return rainy_tensor, clean_tensor


class SingleFolderDataset(Dataset):
    """Dataset for images in a single folder (no pairs).

    Used for inference on rainy images without ground truth.

    Args:
        image_dir: Directory containing images
        transform: Optional transform
        extensions: Valid image extensions

    Example:
        >>> dataset = SingleFolderDataset('data/test/rainy')
        >>> rainy_img = dataset[0]
    """

    def __init__(
        self,
        image_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ) -> None:
        """Initialize dataset."""
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.extensions = extensions

        # Get image files
        self.image_files = sorted(
            [f for f in self.image_dir.iterdir() if f.suffix.lower() in extensions]
        )

        logger.info(f"Loaded {len(self.image_files)} images")

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get an image.

        Args:
            idx: Index

        Returns:
            Image tensor
        """
        # Load image
        img = Image.open(self.image_files[idx]).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0

        # Apply transform
        if self.transform is not None:
            transformed = self.transform(image=img_np)
            img_np = transformed["image"]

        # Convert to tensor
        img_tensor = numpy_to_tensor(img_np)

        return img_tensor

    def get_filename(self, idx: int) -> str:
        """Get filename for an index."""
        return self.image_files[idx].name


class Rain100Dataset(ImagePairDataset):
    """Dataset for Rain100L/Rain100H benchmarks.

    Convenience class for common rain datasets.

    Example:
        >>> train_dataset = Rain100Dataset('data/Rain100L/train')
        >>> test_dataset = Rain100Dataset('data/Rain100L/test')
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialize Rain100 dataset.

        Args:
            root_dir: Root directory containing 'rainy' and 'norain' folders
            transform: Optional transform
        """
        root_dir = Path(root_dir)

        # Try common naming conventions
        rainy_dirs = ["rainy", "rain", "input", "rainy_image"]
        clean_dirs = ["norain", "clean", "ground_truth", "gt", "target"]

        rainy_dir = None
        clean_dir = None

        for name in rainy_dirs:
            if (root_dir / name).exists():
                rainy_dir = root_dir / name
                break

        for name in clean_dirs:
            if (root_dir / name).exists():
                clean_dir = root_dir / name
                break

        if rainy_dir is None or clean_dir is None:
            raise FileNotFoundError(
                f"Could not find rainy/clean directories in {root_dir}. "
                f"Expected one of: rainy={rainy_dirs}, clean={clean_dirs}"
            )

        super().__init__(rainy_dir=rainy_dir, clean_dir=clean_dir, transform=transform)


class SyntheticRainDataset(Dataset):
    """Dataset with on-the-fly synthetic rain generation.

    Generates rainy images by adding synthetic rain to clean images.
    Useful for data augmentation.

    Args:
        clean_dir: Directory with clean images
        rain_generator: Function that adds rain to images
        transform: Optional transform

    Example:
        >>> def add_rain(img):
        ...     # Add synthetic rain streaks
        ...     return img_with_rain
        >>>
        >>> dataset = SyntheticRainDataset(
        ...     clean_dir='data/clean',
        ...     rain_generator=add_rain
        ... )
    """

    def __init__(
        self,
        clean_dir: Union[str, Path],
        rain_generator: Callable[[np.ndarray], np.ndarray],
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ) -> None:
        """Initialize synthetic rain dataset."""
        self.clean_dir = Path(clean_dir)
        self.rain_generator = rain_generator
        self.transform = transform
        self.extensions = extensions

        self.clean_files = sorted(
            [f for f in self.clean_dir.iterdir() if f.suffix.lower() in extensions]
        )

        logger.info(f"Loaded {len(self.clean_files)} clean images for synthetic rain")

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.clean_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a synthetic rainy/clean pair.

        Args:
            idx: Index

        Returns:
            Tuple of (rainy_tensor, clean_tensor)
        """
        # Load clean image
        clean_img = Image.open(self.clean_files[idx]).convert("RGB")
        clean_np = np.array(clean_img).astype(np.float32) / 255.0

        # Generate rainy version
        rainy_np = self.rain_generator(clean_np.copy())

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=rainy_np, target=clean_np)
            rainy_np = transformed["image"]
            clean_np = transformed["target"]

        # Convert to tensors
        rainy_tensor = numpy_to_tensor(rainy_np)
        clean_tensor = numpy_to_tensor(clean_np)

        return rainy_tensor, clean_tensor


class Rain1400Dataset(Dataset):
    """Dataset for Rain1400 rainy/clean images.

    Loads images from parallel directory structures:
    ```
    data/
    ├── train/
        ├── rainy_image/
            ├──1_1.png
            ├──1_2.png
            └── ...
        └── ground_truth/
            ├──1.png
            ├──2.png
            └── ...
    └── test/
        ├── rainy_image/
            ├──1_1.png
            ├──1_2.png
            └── ...
        └── ground_truth/
            ├──1.png
            ├──2.png
            └── ...
    ```

    Args:
        rainy_dir: Directory containing rainy images
        clean_dir: Directory containing clean images
        transform: Optional transform to apply to both images
        extensions: Valid image extensions

    Example:
        >>> dataset = Rain1400Dataset(
        ...     rainy_dir='data/train/rainy_image',
        ...     clean_dir='data/train/ground_truth'
        ... )
        >>> rainy, clean = dataset[0]
    """

    def __init__(
        self,
        rainy_dir: Union[str, Path],
        clean_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ) -> None:
        """Initialize dataset."""
        self.rainy_dir = Path(rainy_dir)
        self.clean_dir = Path(clean_dir)
        self.transform = transform
        self.extensions = extensions

        # Get image file lists
        self.rainy_files = sorted(
            [f for f in self.rainy_dir.iterdir() if f.suffix.lower() in extensions]
        )

        self.clean_files = sorted(
            [f for f in self.clean_dir.iterdir() if f.suffix.lower() in extensions]
        )
        self.clean_files = []
        for item in self.rainy_files:
            filename = item.name.split("_")[0] + ".jpg"
            self.clean_files.append(self.clean_dir / filename)

        # Validate
        if len(self.rainy_files) != len(self.clean_files):
            raise ValueError(
                f"Mismatch in number of images: "
                f"{len(self.rainy_files)} rainy vs {len(self.clean_files)} clean"
            )

        logger.info(f"Loaded {len(self.rainy_files)} image pairs")

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.rainy_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a rainy/clean image pair.

        Args:
            idx: Index

        Returns:
            Tuple of (rainy_tensor, clean_tensor)
        """
        # Load images
        rainy_img = Image.open(self.rainy_files[idx]).convert("RGB")
        clean_img = Image.open(self.clean_files[idx]).convert("RGB")

        # Convert to numpy
        rainy_np = np.array(rainy_img).astype(np.float32) / 255.0
        clean_np = np.array(clean_img).astype(np.float32) / 255.0

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=rainy_np, target=clean_np)
            rainy_np = transformed["image"]
            clean_np = transformed["target"]

        # Convert to tensors
        rainy_tensor = numpy_to_tensor(rainy_np)
        clean_tensor = numpy_to_tensor(clean_np)

        return rainy_tensor, clean_tensor

__all__ = [
    "ImagePairDataset",
    "SingleFolderDataset",
    "Rain100Dataset",
    "SyntheticRainDataset",
    "Rain1400Dataset",
]