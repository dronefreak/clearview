"""Dataset implementations for image deraining.

Provides dataset classes for loading paired rainy/clean images from
various formats (image pairs, directories, etc.).
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

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

        # Get image file lists. Scanned recursively (rglob) rather than a flat
        # iterdir() so that a rainy_dir/clean_dir sharded into subfolders (e.g.
        # to stay under a hosting platform's per-directory file-count limit) is
        # found transparently, with no change needed for a plain flat directory.
        try:
            rainy_entries = list(self.rainy_dir.rglob("*"))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Rainy directory not found: {self.rainy_dir}"
            ) from e
        except PermissionError as e:
            raise PermissionError(
                f"Cannot read rainy directory '{self.rainy_dir}': {e}"
            ) from e

        try:
            clean_entries = list(self.clean_dir.rglob("*"))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Clean directory not found: {self.clean_dir}"
            ) from e
        except PermissionError as e:
            raise PermissionError(
                f"Cannot read clean directory '{self.clean_dir}': {e}"
            ) from e

        self.rainy_files = sorted(
            [f for f in rainy_entries if f.suffix.lower() in extensions]
        )

        self.clean_files = sorted(
            [f for f in clean_entries if f.suffix.lower() in extensions]
        )

        # Validate that filenames match between rainy and clean directories
        rainy_names = {f.stem for f in self.rainy_files}
        clean_names = {f.stem for f in self.clean_files}
        only_in_rainy = rainy_names - clean_names
        only_in_clean = clean_names - rainy_names

        if only_in_rainy or only_in_clean:
            details = []
            if only_in_rainy:
                details.append(f"only in rainy: {sorted(only_in_rainy)}")
            if only_in_clean:
                details.append(f"only in clean: {sorted(only_in_clean)}")
            raise ValueError(
                f"Unpaired images found ({'; '.join(details)}). "
                "Rainy and clean directories must contain matching filenames."
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

        try:
            clean_entries = list(self.clean_dir.iterdir())
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Clean directory not found: {self.clean_dir}"
            ) from e
        except PermissionError as e:
            raise PermissionError(
                f"Cannot read clean directory '{self.clean_dir}': {e}"
            ) from e

        self.clean_files = sorted(
            [f for f in clean_entries if f.suffix.lower() in extensions]
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

        # Get image file lists. Scanned recursively (rglob) rather than a flat
        # iterdir() so that a rainy_dir sharded into subfolders (e.g. to stay
        # under a hosting platform's per-directory file-count limit) is found
        # transparently, with no change needed for a plain flat directory.
        try:
            rainy_entries = list(self.rainy_dir.rglob("*"))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Rainy directory not found: {self.rainy_dir}"
            ) from e
        except PermissionError as e:
            raise PermissionError(
                f"Cannot read rainy directory '{self.rainy_dir}': {e}"
            ) from e

        self.rainy_files = sorted(
            [f for f in rainy_entries if f.suffix.lower() in extensions]
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


class Rain13KDataset(ImagePairDataset):
    """Dataset for the Rain13K composite benchmark.

    Rain13K is a widely used composite training set (aggregating Rain1400,
    Rain12000, Rain800, Rain1200, etc., ~13,712 pairs total) reported on by
    nearly every recent deraining paper (MPRNet, Restormer, MAXIM, ...).
    It is conventionally distributed with an ``input``/``target`` (or
    ``rainy``/``clean``) directory layout, optionally nested under a
    ``train``/``test`` split directory:

    ```
    Rain13K/
    ├── train/
    │   ├── input/
    │   └── target/
    └── test/
        ├── input/
        └── target/
    ```

    Args:
        root_dir: Root directory containing the dataset (or split subdir)
        split: Optional split name ('train' | 'test' | 'val') to look for
            under ``root_dir``. If ``None``, ``root_dir`` is treated as
            already pointing at the split directory.
        transform: Optional transform to apply to both images

    Example:
        >>> train_dataset = Rain13KDataset('data/Rain13K', split='train')
        >>> test_dataset = Rain13KDataset('data/Rain13K', split='test')
    """

    #: Directory name candidates tried (in order) for locating rainy/clean images.
    RAINY_DIR_CANDIDATES: Tuple[str, ...] = ("input", "rainy", "rain")
    CLEAN_DIR_CANDIDATES: Tuple[str, ...] = ("target", "clean", "gt", "ground_truth")

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialize Rain13K dataset.

        Args:
            root_dir: Root directory containing the dataset (or split subdir)
            split: Optional split subdirectory name under ``root_dir``
            transform: Optional transform

        Raises:
            FileNotFoundError: If rainy/clean directories cannot be located
        """
        root_dir = Path(root_dir)
        base_dir = root_dir / split if split is not None else root_dir

        rainy_dir = self._find_dir(base_dir, self.RAINY_DIR_CANDIDATES)
        clean_dir = self._find_dir(base_dir, self.CLEAN_DIR_CANDIDATES)

        if rainy_dir is None or clean_dir is None:
            raise FileNotFoundError(
                f"Could not find rainy/clean directories in {base_dir}. "
                f"Expected one of: rainy={self.RAINY_DIR_CANDIDATES}, "
                f"clean={self.CLEAN_DIR_CANDIDATES}"
            )

        super().__init__(rainy_dir=rainy_dir, clean_dir=clean_dir, transform=transform)

    @staticmethod
    def _find_dir(base_dir: Path, candidates: Tuple[str, ...]) -> Optional[Path]:
        """Find the first existing directory among a list of name candidates."""
        for name in candidates:
            candidate = base_dir / name
            if candidate.exists():
                return candidate
        return None


class DDNDataDataset(Rain1400Dataset):
    """Dataset for DDN-Data, also distributed as "Rain1400".

    From Fu et al., "Removing Rain from Single Images via a Deep Detail
    Network", CVPR 2017.

    Each clean image has multiple (commonly 14) synthetically rained
    versions, named ``{clean_id}_{variant}.jpg`` in the rainy directory and
    ``{clean_id}.jpg`` in the ground-truth directory:

    ```
    DDN-Data/
    ├── train/
    │   ├── rainy_image/
    │   │   ├── 1_1.jpg
    │   │   ├── 1_2.jpg
    │   │   └── ...
    │   └── ground_truth/
    │       ├── 1.jpg
    │       └── ...
    └── test/
        └── ...
    ```

    Args:
        root_dir: Root directory containing the dataset (or split subdir)
        split: Optional split name ('train' | 'test') to look for under
            ``root_dir``. If ``None``, ``root_dir`` is treated as already
            pointing at the split directory.
        transform: Optional transform to apply to both images

    Example:
        >>> train_dataset = DDNDataDataset('data/DDN-Data', split='train')
        >>> test_dataset = DDNDataDataset('data/DDN-Data', split='test')
    """

    RAINY_DIR_CANDIDATES: Tuple[str, ...] = ("rainy_image", "rain_image", "input")
    CLEAN_DIR_CANDIDATES: Tuple[str, ...] = ("ground_truth", "gt", "target", "clean")

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialize DDN-Data dataset.

        Args:
            root_dir: Root directory containing the dataset (or split subdir)
            split: Optional split subdirectory name under ``root_dir``
            transform: Optional transform

        Raises:
            FileNotFoundError: If rainy/ground-truth directories cannot be located
        """
        root_dir = Path(root_dir)
        base_dir = root_dir / split if split is not None else root_dir

        rainy_dir = Rain13KDataset._find_dir(base_dir, self.RAINY_DIR_CANDIDATES)
        clean_dir = Rain13KDataset._find_dir(base_dir, self.CLEAN_DIR_CANDIDATES)

        if rainy_dir is None or clean_dir is None:
            raise FileNotFoundError(
                f"Could not find rainy/ground-truth directories in {base_dir}. "
                f"Expected one of: rainy={self.RAINY_DIR_CANDIDATES}, "
                f"clean={self.CLEAN_DIR_CANDIDATES}"
            )

        super().__init__(rainy_dir=rainy_dir, clean_dir=clean_dir, transform=transform)


class DIDDataDataset(Dataset):
    """Dataset for DID-Data.

    From Zhang & Patel, "Density-aware Single Image De-raining using a
    Multi-stream Dense Network", CVPR 2018.

    DID-Data provides rain images at three density levels (light/medium/
    heavy) and is commonly distributed with either:

    1. Separate density subfolders, each with paired rainy/clean images::

        DID-Data/
        ├── train/
        │   ├── Rain_Light/{rainy,clean}/
        │   ├── Rain_Medium/{rainy,clean}/
        │   └── Rain_Heavy/{rainy,clean}/
        └── test/
            └── ...

    2. A flat rainy/clean directory pair (density information discarded)::

        DID-Data/
        ├── train/{rainy,clean}/
        └── test/{rainy,clean}/

    This class auto-detects which layout is present and, for the
    density-subfolder layout, concatenates all density levels into a single
    dataset while exposing per-sample density labels via
    :meth:`get_density_label`.

    Args:
        root_dir: Root directory containing the dataset (or split subdir)
        split: Optional split name ('train' | 'test') to look for under
            ``root_dir``. If ``None``, ``root_dir`` is treated as already
            pointing at the split directory.
        transform: Optional transform to apply to both images

    Example:
        >>> train_dataset = DIDDataDataset('data/DID-Data', split='train')
        >>> rainy, clean = train_dataset[0]
        >>> density = train_dataset.get_density_label(0)  # 'light' | 'medium' | 'heavy'
    """

    DENSITY_DIR_CANDIDATES: Tuple[str, ...] = (
        "Rain_Light",
        "Rain_Medium",
        "Rain_Heavy",
    )
    RAINY_DIR_CANDIDATES: Tuple[str, ...] = ("rainy", "rain", "input")
    CLEAN_DIR_CANDIDATES: Tuple[str, ...] = ("clean", "gt", "ground_truth", "target")

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ) -> None:
        """Initialize DID-Data dataset.

        Args:
            root_dir: Root directory containing the dataset (or split subdir)
            split: Optional split subdirectory name under ``root_dir``
            transform: Optional transform
            extensions: Valid image extensions

        Raises:
            FileNotFoundError: If no supported directory layout is found
        """
        root_dir = Path(root_dir)
        self.base_dir = root_dir / split if split is not None else root_dir
        self.transform = transform
        self.extensions = extensions

        self.rainy_files: List[Path] = []
        self.clean_files: List[Path] = []
        self.density_labels: List[Optional[str]] = []

        density_dirs_found = [
            self.base_dir / name
            for name in self.DENSITY_DIR_CANDIDATES
            if (self.base_dir / name).exists()
        ]

        if density_dirs_found:
            for density_dir in density_dirs_found:
                density_label = density_dir.name.split("_")[-1].lower()
                rainy_dir = Rain13KDataset._find_dir(
                    density_dir, self.RAINY_DIR_CANDIDATES
                )
                clean_dir = Rain13KDataset._find_dir(
                    density_dir, self.CLEAN_DIR_CANDIDATES
                )
                if rainy_dir is None or clean_dir is None:
                    logger.warning(
                        f"Skipping density folder '{density_dir}': "
                        "could not locate rainy/clean subdirectories"
                    )
                    continue

                rainy_files, clean_files = self._collect_pairs(rainy_dir, clean_dir)
                self.rainy_files.extend(rainy_files)
                self.clean_files.extend(clean_files)
                self.density_labels.extend([density_label] * len(rainy_files))
        else:
            rainy_dir = Rain13KDataset._find_dir(
                self.base_dir, self.RAINY_DIR_CANDIDATES
            )
            clean_dir = Rain13KDataset._find_dir(
                self.base_dir, self.CLEAN_DIR_CANDIDATES
            )

            if rainy_dir is None or clean_dir is None:
                raise FileNotFoundError(
                    f"Could not find DID-Data directory layout in {self.base_dir}. "
                    f"Expected density subfolders {self.DENSITY_DIR_CANDIDATES} or "
                    f"a flat rainy={self.RAINY_DIR_CANDIDATES}/"
                    f"clean={self.CLEAN_DIR_CANDIDATES} pair."
                )

            rainy_files, clean_files = self._collect_pairs(rainy_dir, clean_dir)
            self.rainy_files.extend(rainy_files)
            self.clean_files.extend(clean_files)
            self.density_labels.extend([None] * len(rainy_files))

        logger.info(f"Loaded {len(self.rainy_files)} image pairs from DID-Data")

    def _collect_pairs(
        self, rainy_dir: Path, clean_dir: Path
    ) -> Tuple[List[Path], List[Path]]:
        """Collect matching rainy/clean file pairs from a directory pair.

        Args:
            rainy_dir: Directory containing rainy images
            clean_dir: Directory containing clean images

        Returns:
            Tuple of (rainy_files, clean_files) sorted lists with matching stems

        Raises:
            FileNotFoundError: If either directory cannot be read
            PermissionError: If either directory cannot be read
        """
        try:
            rainy_entries = list(rainy_dir.iterdir())
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Rainy directory not found: {rainy_dir}") from e
        except PermissionError as e:
            raise PermissionError(
                f"Cannot read rainy directory '{rainy_dir}': {e}"
            ) from e

        try:
            clean_entries = list(clean_dir.iterdir())
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Clean directory not found: {clean_dir}") from e
        except PermissionError as e:
            raise PermissionError(
                f"Cannot read clean directory '{clean_dir}': {e}"
            ) from e

        rainy_files = sorted(
            [f for f in rainy_entries if f.suffix.lower() in self.extensions]
        )
        clean_files = sorted(
            [f for f in clean_entries if f.suffix.lower() in self.extensions]
        )

        rainy_names = {f.stem for f in rainy_files}
        clean_names = {f.stem for f in clean_files}
        only_in_rainy = rainy_names - clean_names
        only_in_clean = clean_names - rainy_names

        if only_in_rainy or only_in_clean:
            details = []
            if only_in_rainy:
                details.append(f"only in rainy: {sorted(only_in_rainy)}")
            if only_in_clean:
                details.append(f"only in clean: {sorted(only_in_clean)}")
            raise ValueError(
                f"Unpaired images found in {rainy_dir}/{clean_dir} "
                f"({'; '.join(details)}). Rainy and clean directories must "
                "contain matching filenames."
            )

        return rainy_files, clean_files

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
        rainy_img = Image.open(self.rainy_files[idx]).convert("RGB")
        clean_img = Image.open(self.clean_files[idx]).convert("RGB")

        rainy_np = np.array(rainy_img).astype(np.float32) / 255.0
        clean_np = np.array(clean_img).astype(np.float32) / 255.0

        if self.transform is not None:
            transformed = self.transform(image=rainy_np, target=clean_np)
            rainy_np = transformed["image"]
            clean_np = transformed["target"]

        rainy_tensor = numpy_to_tensor(rainy_np)
        clean_tensor = numpy_to_tensor(clean_np)

        return rainy_tensor, clean_tensor

    def get_density_label(self, idx: int) -> Optional[str]:
        """Get the rain density label ('light' | 'medium' | 'heavy') for a sample.

        Args:
            idx: Index

        Returns:
            Density label string, or None if the dataset was loaded from a
            flat (non-density-aware) directory layout
        """
        return self.density_labels[idx]


class SPADataDataset(Dataset):
    """Dataset for SPA-Data, a real-world rain/clean image dataset.

    From Wang et al., "Spatial Attentive Single-Image Deraining with a High
    Quality Real Rain Dataset" (SPANet), CVPR 2019.

    Unlike most other deraining datasets, SPA-Data's rainy and clean images
    do **not** share identical filenames — they share a common numeric ID but
    different prefixes (``rain-{id}.png`` / ``norain-{id}.png``), so the
    exact-filename matching used by :class:`ImagePairDataset` does not apply
    directly. This class matches pairs by stripping the known prefixes
    before comparing IDs.

    Commonly distributed with a ``train``/``val`` split, each containing a
    ``rain``/``norain`` directory pair (optionally nested under an
    ``rgb_reconstruction`` folder, as in some repackaged mirrors)::

        SPA-Data/
        ├── train/
        │   └── rgb_reconstruction/
        │       ├── rain/
        │       │   ├── rain-0.png
        │       │   └── ...
        │       └── norain/
        │           ├── norain-0.png
        │           └── ...
        └── val/
            └── rgb_reconstruction/
                ├── rain/
                └── norain/

    Args:
        root_dir: Root directory containing the dataset (or split subdir)
        split: Optional split name ('train' | 'val' | 'test') to look for
            under ``root_dir``. If ``None``, ``root_dir`` is treated as
            already pointing at the split directory.
        transform: Optional transform to apply to both images
        extensions: Valid image extensions

    Example:
        >>> train_dataset = SPADataDataset('data/SPA-Data', split='train')
        >>> val_dataset = SPADataDataset('data/SPA-Data', split='val')
        >>> rainy, clean = train_dataset[0]
    """

    #: Subdirectory candidates tried (in order), relative to the split dir,
    #: for locating the rain/norain image directories.
    RAINY_DIR_CANDIDATES: Tuple[str, ...] = (
        "rgb_reconstruction/rain",
        "rain",
        "rainy",
    )
    CLEAN_DIR_CANDIDATES: Tuple[str, ...] = (
        "rgb_reconstruction/norain",
        "norain",
        "clean",
    )

    #: Filename prefixes stripped before matching rainy/clean image IDs.
    RAINY_PREFIX = "rain-"
    CLEAN_PREFIX = "norain-"

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ) -> None:
        """Initialize SPA-Data dataset.

        Args:
            root_dir: Root directory containing the dataset (or split subdir)
            split: Optional split subdirectory name under ``root_dir``
            transform: Optional transform
            extensions: Valid image extensions

        Raises:
            FileNotFoundError: If rain/norain directories cannot be located
            ValueError: If rainy and clean directories contain mismatched IDs
        """
        root_dir = Path(root_dir)
        base_dir = root_dir / split if split is not None else root_dir

        rainy_dir = Rain13KDataset._find_dir(base_dir, self.RAINY_DIR_CANDIDATES)
        clean_dir = Rain13KDataset._find_dir(base_dir, self.CLEAN_DIR_CANDIDATES)

        if rainy_dir is None or clean_dir is None:
            raise FileNotFoundError(
                f"Could not find rain/norain directories in {base_dir}. "
                f"Expected one of: rain={self.RAINY_DIR_CANDIDATES}, "
                f"norain={self.CLEAN_DIR_CANDIDATES}"
            )

        self.rainy_dir = rainy_dir
        self.clean_dir = clean_dir
        self.transform = transform
        self.extensions = extensions

        try:
            rainy_entries = list(self.rainy_dir.iterdir())
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Rainy directory not found: {self.rainy_dir}"
            ) from e
        except PermissionError as e:
            raise PermissionError(
                f"Cannot read rainy directory '{self.rainy_dir}': {e}"
            ) from e

        try:
            clean_entries = list(self.clean_dir.iterdir())
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Clean directory not found: {self.clean_dir}"
            ) from e
        except PermissionError as e:
            raise PermissionError(
                f"Cannot read clean directory '{self.clean_dir}': {e}"
            ) from e

        rainy_files = sorted(f for f in rainy_entries if f.suffix.lower() in extensions)
        clean_files = sorted(f for f in clean_entries if f.suffix.lower() in extensions)

        def _strip_prefix(stem: str, prefix: str) -> str:
            return stem[len(prefix) :] if stem.startswith(prefix) else stem

        rainy_by_id = {_strip_prefix(f.stem, self.RAINY_PREFIX): f for f in rainy_files}
        clean_by_id = {_strip_prefix(f.stem, self.CLEAN_PREFIX): f for f in clean_files}

        rainy_ids = set(rainy_by_id)
        clean_ids = set(clean_by_id)
        only_in_rainy = rainy_ids - clean_ids
        only_in_clean = clean_ids - rainy_ids

        if only_in_rainy or only_in_clean:
            details = []
            if only_in_rainy:
                details.append(f"only in rain: {sorted(only_in_rainy)[:5]}")
            if only_in_clean:
                details.append(f"only in norain: {sorted(only_in_clean)[:5]}")
            raise ValueError(
                f"Unpaired images found ({'; '.join(details)}). rain/norain "
                "directories must contain matching IDs after stripping "
                f"'{self.RAINY_PREFIX}'/'{self.CLEAN_PREFIX}' prefixes."
            )

        def _sort_key(image_id: str) -> Tuple[int, Union[int, str]]:
            # Sort numerically when possible so ids like '2' come before '10'.
            return (
                (0, cast(Union[int, str], int(image_id)))
                if image_id.isdigit()
                else (1, image_id)
            )

        common_ids = sorted(rainy_ids, key=_sort_key)
        self.rainy_files = [rainy_by_id[i] for i in common_ids]
        self.clean_files = [clean_by_id[i] for i in common_ids]

        logger.info(f"Loaded {len(self.rainy_files)} image pairs from SPA-Data")

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
        rainy_img = Image.open(self.rainy_files[idx]).convert("RGB")
        clean_img = Image.open(self.clean_files[idx]).convert("RGB")

        rainy_np = np.array(rainy_img).astype(np.float32) / 255.0
        clean_np = np.array(clean_img).astype(np.float32) / 255.0

        if self.transform is not None:
            transformed = self.transform(image=rainy_np, target=clean_np)
            rainy_np = transformed["image"]
            clean_np = transformed["target"]

        rainy_tensor = numpy_to_tensor(rainy_np)
        clean_tensor = numpy_to_tensor(clean_np)

        return rainy_tensor, clean_tensor


class GTRainDataset(Dataset):
    """Dataset for GT-RAIN, a real-world paired-frame rain dataset.

    From Ba et al., "Not Just Streaks: Towards Ground Truth for Single Image
    Deraining" (GT-RAIN), ECCV 2022. Pairs are real photographs, not
    synthetically rendered: each scene was filmed continuously across the
    moment rain starts or stops, rather than compositing rain onto a clean
    reference shot.

    Distributed as one directory per scene, each holding a single clean
    reference frame shared across roughly 300 rainy frames of that same
    scene, distinguished by a ``-C-``/``-R-`` marker in the filename::

        GT-RAIN_train/
        ├── scene_name_1/
        │   ├── scene_name_1-Webcam-C-000.png   # clean reference
        │   ├── scene_name_1-Webcam-R-000.png   # rainy frame
        │   ├── scene_name_1-Webcam-R-001.png
        │   └── ...
        └── scene_name_2/
            └── ...

    One scene in the official validation split ("Gurutto_1-2") instead has a
    distinct clean frame per rainy frame rather than one shared clean frame.
    This class detects that automatically by counting clean files per scene
    and matching by trailing index when there's more than one, rather than
    hardcoding the scene name.

    Args:
        root_dir: Directory containing one subdirectory per scene (e.g.
            pointed at ``GT-RAIN_train``, ``GT-RAIN_val``, or ``GT-RAIN_test``
            after extracting the official split archives)
        transform: Optional transform to apply to both images
        extensions: Valid image extensions

    Example:
        >>> dataset = GTRainDataset('data/GT-RAIN_train')
        >>> rainy, clean = dataset[0]
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ) -> None:
        """Initialize GT-RAIN dataset.

        Args:
            root_dir: Directory containing one subdirectory per scene
            transform: Optional transform
            extensions: Valid image extensions

        Raises:
            FileNotFoundError: If root_dir has no scene subdirectories
            ValueError: If no rainy/clean pairs could be matched
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.extensions = extensions

        try:
            scene_dirs = sorted(d for d in self.root_dir.iterdir() if d.is_dir())
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}") from e
        except PermissionError as e:
            raise PermissionError(
                f"Cannot read root directory '{self.root_dir}': {e}"
            ) from e

        if not scene_dirs:
            raise FileNotFoundError(f"No scene subdirectories found in {self.root_dir}")

        self.rainy_files: List[Path] = []
        self.clean_files: List[Path] = []

        for scene_dir in scene_dirs:
            files = sorted(
                f for f in scene_dir.iterdir() if f.suffix.lower() in extensions
            )
            rainy = [f for f in files if "-R-" in f.name]
            clean = [f for f in files if "-C-" in f.name]

            if not rainy or not clean:
                continue

            if len(clean) == 1:
                self.rainy_files.extend(rainy)
                self.clean_files.extend([clean[0]] * len(rainy))
            else:
                # More than one clean frame in this scene (e.g. the official
                # "Gurutto_1-2" validation scene): match by shared trailing
                # index after the -R-/-C- marker instead of broadcasting.
                clean_by_idx = {f.name.split("-C-")[-1]: f for f in clean}
                for r in rainy:
                    idx = r.name.split("-R-")[-1]
                    c = clean_by_idx.get(idx)
                    if c is not None:
                        self.rainy_files.append(r)
                        self.clean_files.append(c)

        if not self.rainy_files:
            raise ValueError(f"No rainy/clean pairs found under {self.root_dir}")

        logger.info(f"Loaded {len(self.rainy_files)} image pairs from GT-RAIN")

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
        rainy_img = Image.open(self.rainy_files[idx]).convert("RGB")
        clean_img = Image.open(self.clean_files[idx]).convert("RGB")

        rainy_np = np.array(rainy_img).astype(np.float32) / 255.0
        clean_np = np.array(clean_img).astype(np.float32) / 255.0

        if self.transform is not None:
            transformed = self.transform(image=rainy_np, target=clean_np)
            rainy_np = transformed["image"]
            clean_np = transformed["target"]

        rainy_tensor = numpy_to_tensor(rainy_np)
        clean_tensor = numpy_to_tensor(clean_np)

        return rainy_tensor, clean_tensor


class RainDropDataset(Dataset):
    """Dataset for RainDrop / DeRaindrop, a lens-adherent raindrop removal dataset.

    From Qian et al., "Attentive Generative Adversarial Network for Raindrop
    Removal from a Single Image" (DeRaindrop), CVPR 2018. This is a
    physically distinct degradation from rain streaks or general rain/haze
    mixes: droplets adhered to a glass window or camera lens, captured with
    two panes of glass side by side (one clean, one sprayed with water) so
    the background scene stays spatially aligned between pairs.

    Rainy and clean images share a common numeric ID but different filename
    **suffixes** (``{id}_rain.png`` / ``{id}_clean.png``), not identical
    stems, so the exact-filename matching used by :class:`ImagePairDataset`
    does not apply directly. This class matches pairs by stripping the known
    suffixes before comparing IDs::

        train/
        ├── data/    # {id}_rain.png   (861 images)
        └── gt/      # {id}_clean.png  (861 images)
        test_a/
        ├── data/    # {id}_rain.png   (58 images)
        └── gt/      # {id}_clean.png  (58 images)
        test_b/
        ├── data/    # {id}_rain.jpg   (249 images)
        └── gt/      # {id}_clean.jpg  (249 images)

    Args:
        root_dir: Directory containing the dataset (or split subdir)
        split: Optional split name (e.g. 'train' | 'test_a' | 'test_b') to
            look for under ``root_dir``. If ``None``, ``root_dir`` is
            treated as already pointing at the split directory.
        transform: Optional transform to apply to both images
        extensions: Valid image extensions

    Example:
        >>> train_dataset = RainDropDataset('data/RainDrop', split='train')
        >>> test_dataset = RainDropDataset('data/RainDrop', split='test_a')
        >>> rainy, clean = train_dataset[0]
    """

    #: Subdirectory candidates tried (in order), relative to the split dir.
    RAINY_DIR_CANDIDATES: Tuple[str, ...] = ("data",)
    CLEAN_DIR_CANDIDATES: Tuple[str, ...] = ("gt",)

    #: Filename suffixes stripped before matching rainy/clean image IDs.
    RAINY_SUFFIX = "_rain"
    CLEAN_SUFFIX = "_clean"

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ) -> None:
        """Initialize RainDrop dataset.

        Args:
            root_dir: Root directory containing the dataset (or split subdir)
            split: Optional split subdirectory name under ``root_dir``
            transform: Optional transform
            extensions: Valid image extensions

        Raises:
            FileNotFoundError: If data/gt directories cannot be located
            ValueError: If rainy and clean directories contain mismatched IDs
        """
        root_dir = Path(root_dir)
        base_dir = root_dir / split if split is not None else root_dir

        rainy_dir = Rain13KDataset._find_dir(base_dir, self.RAINY_DIR_CANDIDATES)
        clean_dir = Rain13KDataset._find_dir(base_dir, self.CLEAN_DIR_CANDIDATES)

        if rainy_dir is None or clean_dir is None:
            raise FileNotFoundError(
                f"Could not find data/gt directories in {base_dir}. "
                f"Expected one of: data={self.RAINY_DIR_CANDIDATES}, "
                f"gt={self.CLEAN_DIR_CANDIDATES}"
            )

        self.rainy_dir = rainy_dir
        self.clean_dir = clean_dir
        self.transform = transform
        self.extensions = extensions

        try:
            rainy_entries = list(self.rainy_dir.iterdir())
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Rainy directory not found: {self.rainy_dir}"
            ) from e
        except PermissionError as e:
            raise PermissionError(
                f"Cannot read rainy directory '{self.rainy_dir}': {e}"
            ) from e

        try:
            clean_entries = list(self.clean_dir.iterdir())
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Clean directory not found: {self.clean_dir}"
            ) from e
        except PermissionError as e:
            raise PermissionError(
                f"Cannot read clean directory '{self.clean_dir}': {e}"
            ) from e

        rainy_files = sorted(f for f in rainy_entries if f.suffix.lower() in extensions)
        clean_files = sorted(f for f in clean_entries if f.suffix.lower() in extensions)

        def _strip_suffix(stem: str, suffix: str) -> str:
            return stem[: -len(suffix)] if stem.endswith(suffix) else stem

        rainy_by_id = {_strip_suffix(f.stem, self.RAINY_SUFFIX): f for f in rainy_files}
        clean_by_id = {_strip_suffix(f.stem, self.CLEAN_SUFFIX): f for f in clean_files}

        rainy_ids = set(rainy_by_id)
        clean_ids = set(clean_by_id)
        only_in_rainy = rainy_ids - clean_ids
        only_in_clean = clean_ids - rainy_ids

        if only_in_rainy or only_in_clean:
            details = []
            if only_in_rainy:
                details.append(f"only in data: {sorted(only_in_rainy)[:5]}")
            if only_in_clean:
                details.append(f"only in gt: {sorted(only_in_clean)[:5]}")
            raise ValueError(
                f"Unpaired images found ({'; '.join(details)}). data/gt "
                "directories must contain matching IDs after stripping "
                f"'{self.RAINY_SUFFIX}'/'{self.CLEAN_SUFFIX}' suffixes."
            )

        def _sort_key(image_id: str) -> Tuple[int, Union[int, str]]:
            return (
                (0, cast(Union[int, str], int(image_id)))
                if image_id.isdigit()
                else (1, image_id)
            )

        common_ids = sorted(rainy_ids, key=_sort_key)
        self.rainy_files = [rainy_by_id[i] for i in common_ids]
        self.clean_files = [clean_by_id[i] for i in common_ids]

        logger.info(f"Loaded {len(self.rainy_files)} image pairs from RainDrop")

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
        rainy_img = Image.open(self.rainy_files[idx]).convert("RGB")
        clean_img = Image.open(self.clean_files[idx]).convert("RGB")

        rainy_np = np.array(rainy_img).astype(np.float32) / 255.0
        clean_np = np.array(clean_img).astype(np.float32) / 255.0

        if self.transform is not None:
            transformed = self.transform(image=rainy_np, target=clean_np)
            rainy_np = transformed["image"]
            clean_np = transformed["target"]

        rainy_tensor = numpy_to_tensor(rainy_np)
        clean_tensor = numpy_to_tensor(clean_np)

        return rainy_tensor, clean_tensor


class NTIREHazeDataset(Dataset):
    """Dataset for the NTIRE real-haze dehazing benchmarks (I-Haze, O-Haze, Dense-Haze, NH-Haze).

    All four NTIRE challenge dehazing sets (Ancuti et al., NTIRE 2018-2020)
    share the same real haze-machine capture methodology and near-identical
    naming convention, differing only in a few surface details this class
    normalizes away:

    - Some ship a ``GT``/``hazy`` subdirectory pair, others (NH-Haze) ship a
      single flat directory with both image types mixed together.
    - Filenames mark the image type with a ``_GT``/``_hazy`` suffix,
      sometimes with an extra domain tag in between (I-Haze/O-Haze's
      ``_indoor``/``_outdoor``), matched here by stripping the marker
      case-insensitively rather than assuming a fixed filename shape.
    - File extension casing is inconsistent even within a single set
      (O-Haze ships both ``.jpg`` and ``.JPG``).

    Point ``root_dir`` at any one set's extracted top level (containing
    either ``GT``/``hazy`` subdirectories, or the images directly)::

        I-HAZE/
        ├── GT/    # {id}_indoor_GT.jpg
        └── hazy/  # {id}_indoor_hazy.jpg

        NH-HAZE/
        ├── 01_GT.png
        ├── 01_hazy.png
        └── ...

    Args:
        root_dir: Directory for one NTIRE haze set (with or without
            GT/hazy subdirectories)
        transform: Optional transform to apply to both images
        extensions: Valid image extensions

    Example:
        >>> dataset = NTIREHazeDataset('data/I-HAZE')
        >>> hazy, clean = dataset[0]
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ) -> None:
        """Initialize NTIRE haze dataset.

        Args:
            root_dir: Directory for one NTIRE haze set
            transform: Optional transform
            extensions: Valid image extensions

        Raises:
            FileNotFoundError: If root_dir cannot be read
            ValueError: If no rainy/clean pairs could be matched
        """
        root_dir = Path(root_dir)
        gt_dir = root_dir / "GT" if (root_dir / "GT").is_dir() else root_dir
        hazy_dir = root_dir / "hazy" if (root_dir / "hazy").is_dir() else root_dir

        self.transform = transform
        self.extensions = extensions

        try:
            gt_entries = list(gt_dir.iterdir())
            hazy_entries = list(hazy_dir.iterdir())
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Root directory not found: {root_dir}") from e
        except PermissionError as e:
            raise PermissionError(
                f"Cannot read root directory '{root_dir}': {e}"
            ) from e

        def _match_marker(stem: str, marker: str) -> Optional[str]:
            low = stem.lower()
            idx = low.rfind(marker)
            return stem[:idx] if idx != -1 else None

        gt_by_id = {}
        for f in gt_entries:
            if f.suffix.lower() not in extensions:
                continue
            image_id = _match_marker(f.stem, "_gt")
            if image_id is not None:
                gt_by_id[image_id] = f

        hazy_by_id = {}
        for f in hazy_entries:
            if f.suffix.lower() not in extensions:
                continue
            image_id = _match_marker(f.stem, "_hazy")
            if image_id is not None:
                hazy_by_id[image_id] = f

        common_ids = sorted(set(gt_by_id) & set(hazy_by_id))
        if not common_ids:
            raise ValueError(
                f"No hazy/GT pairs found under {root_dir}. Expected filenames "
                "ending in '_GT'/'_hazy' (case-insensitive), optionally under "
                "GT/hazy subdirectories."
            )

        self.rainy_files = [hazy_by_id[i] for i in common_ids]
        self.clean_files = [gt_by_id[i] for i in common_ids]

        logger.info(f"Loaded {len(self.rainy_files)} image pairs from NTIRE haze set")

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.rainy_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a hazy/clean image pair.

        Args:
            idx: Index

        Returns:
            Tuple of (hazy_tensor, clean_tensor)
        """
        rainy_img = Image.open(self.rainy_files[idx]).convert("RGB")
        clean_img = Image.open(self.clean_files[idx]).convert("RGB")

        rainy_np = np.array(rainy_img).astype(np.float32) / 255.0
        clean_np = np.array(clean_img).astype(np.float32) / 255.0

        if self.transform is not None:
            transformed = self.transform(image=rainy_np, target=clean_np)
            rainy_np = transformed["image"]
            clean_np = transformed["target"]

        rainy_tensor = numpy_to_tensor(rainy_np)
        clean_tensor = numpy_to_tensor(clean_np)

        return rainy_tensor, clean_tensor


class RainDSDataset(Dataset):
    """Dataset for RainDS, a combined rain-streak + raindrop removal benchmark.

    From Quan et al., "Removing Raindrops and Rain Streaks in One Go"
    (RainDS), CVPR 2021. Each clean (``gt``) image has up to three separate
    degraded variants captured/rendered against it: raindrop-only,
    rainstreak-only, and both combined, so a single dataset instance covers
    raindrop removal, streak removal, or the joint task depending on which
    ``degradation`` is selected.

    Distributed as two collections with different splits and a different
    filename convention:

        RainDS_syn/{train,test}/{gt,raindrop,rainstreak,rainstreak_raindrop}/
            gt/norain-{id}.png                    (or pie-norain-{id}.png)
            raindrop/rd-{id}.png                  (or pie-rd-{id}.png)
            rainstreak/rain-{id}.png               (or pie-rain-{id}.png)
            rainstreak_raindrop/rd-rain-{id}.png   (or pie-rd-rain-{id}.png)

        RainDS_real/{train_set,test_set}/{gt,raindrop,rainstreak,rainstreak_raindrop}/
            all four subdirectories use identical plain ``{id}.png`` filenames

    The ``pie-`` prefix marks a second sub-collection folded into RainDS_syn
    with its own, otherwise-overlapping numeric IDs; this class preserves it
    as part of the matching key (rather than stripping it) so the two
    sub-collections don't collide with each other.

    Args:
        root_dir: Directory for one split (e.g. pointed at
            ``RainDS_syn/train`` or ``RainDS_real/test_set``), containing a
            ``gt/`` subdirectory and at least one degraded-category
            subdirectory
        degradation: Which degraded category to pair against ``gt``:
            ``'raindrop'``, ``'rainstreak'``, or ``'rainstreak_raindrop'``
        transform: Optional transform to apply to both images
        extensions: Valid image extensions

    Example:
        >>> dataset = RainDSDataset('data/RainDS_syn/train', degradation='rainstreak_raindrop')
        >>> degraded, clean = dataset[0]

    Note:
        RainDS_real's official ``test_set/rainstreak`` folder ships one file
        (``IMG_7435.png``) with no corresponding numeric ID in ``gt/``, a
        known inconsistency in the upstream release, not something specific
        to this mirror. This class matches on the intersection of available
        IDs rather than raising on mismatch, and logs a warning naming how
        many images were excluded.
    """

    GT_MARKER = "norain-"
    DEGRADATION_MARKERS: Dict[str, str] = {
        "raindrop": "rd-",
        "rainstreak": "rain-",
        "rainstreak_raindrop": "rd-rain-",
    }

    def __init__(
        self,
        root_dir: Union[str, Path],
        degradation: str = "rainstreak_raindrop",
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ) -> None:
        """Initialize RainDS dataset.

        Args:
            root_dir: Directory for one split
            degradation: Which degraded category to pair against gt
            transform: Optional transform
            extensions: Valid image extensions

        Raises:
            ValueError: If degradation is not a recognized category, or no
                matching gt/degraded pairs could be found
            FileNotFoundError: If the gt or degradation directory is missing
        """
        if degradation not in self.DEGRADATION_MARKERS:
            raise ValueError(
                f"degradation must be one of {sorted(self.DEGRADATION_MARKERS)}, "
                f"got {degradation!r}"
            )

        root_dir = Path(root_dir)
        gt_dir = root_dir / "gt"
        deg_dir = root_dir / degradation

        self.transform = transform
        self.extensions = extensions

        try:
            gt_entries = list(gt_dir.iterdir())
        except FileNotFoundError as e:
            raise FileNotFoundError(f"gt directory not found: {gt_dir}") from e
        except PermissionError as e:
            raise PermissionError(f"Cannot read gt directory '{gt_dir}': {e}") from e

        try:
            deg_entries = list(deg_dir.iterdir())
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"'{degradation}' directory not found: {deg_dir}"
            ) from e
        except PermissionError as e:
            raise PermissionError(
                f"Cannot read '{degradation}' directory '{deg_dir}': {e}"
            ) from e

        def _strip(stem: str, marker: str) -> str:
            for pie_prefix in ("pie-", ""):
                full = pie_prefix + marker
                if stem.startswith(full):
                    return pie_prefix + stem[len(full) :]
            return stem

        gt_by_id = {
            _strip(f.stem, self.GT_MARKER): f
            for f in gt_entries
            if f.suffix.lower() in extensions
        }
        deg_by_id = {
            _strip(f.stem, self.DEGRADATION_MARKERS[degradation]): f
            for f in deg_entries
            if f.suffix.lower() in extensions
        }

        common_ids_set = set(gt_by_id) & set(deg_by_id)
        if not common_ids_set:
            raise ValueError(
                f"No matching gt/{degradation} pairs found under {root_dir}"
            )

        skipped = (set(gt_by_id) | set(deg_by_id)) - common_ids_set
        if skipped:
            logger.warning(
                f"RainDSDataset: {len(skipped)} unmatched filename(s) excluded "
                f"under {root_dir} ({degradation}), a known upstream naming "
                "inconsistency, not specific to this mirror."
            )

        def _sort_key(image_id: str) -> Tuple[int, Union[int, str]]:
            return (
                (0, cast(Union[int, str], int(image_id)))
                if image_id.isdigit()
                else (1, image_id)
            )

        common_ids = sorted(common_ids_set, key=_sort_key)
        self.rainy_files = [deg_by_id[i] for i in common_ids]
        self.clean_files = [gt_by_id[i] for i in common_ids]

        logger.info(
            f"Loaded {len(self.rainy_files)} image pairs from RainDS ({degradation})"
        )

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.rainy_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a degraded/clean image pair.

        Args:
            idx: Index

        Returns:
            Tuple of (degraded_tensor, clean_tensor)
        """
        rainy_img = Image.open(self.rainy_files[idx]).convert("RGB")
        clean_img = Image.open(self.clean_files[idx]).convert("RGB")

        rainy_np = np.array(rainy_img).astype(np.float32) / 255.0
        clean_np = np.array(clean_img).astype(np.float32) / 255.0

        if self.transform is not None:
            transformed = self.transform(image=rainy_np, target=clean_np)
            rainy_np = transformed["image"]
            clean_np = transformed["target"]

        rainy_tensor = numpy_to_tensor(rainy_np)
        clean_tensor = numpy_to_tensor(clean_np)

        return rainy_tensor, clean_tensor


class MixedDataset(ConcatDataset):
    """Combines multiple paired deraining datasets (e.g. synthetic + real).

    Concatenation itself (via ``ConcatDataset``) preserves each source's
    natural size -- a source is not automatically up- or down-weighted just
    because it's larger or smaller than the others. Oversampling small
    sources (e.g. a real-world dataset dwarfed by a large synthetic one) is
    opt-in via ``weights`` + :meth:`sample_weights`, consumed by a
    ``torch.utils.data.WeightedRandomSampler`` -- ``MixedDataset`` itself
    stays a plain, shuffle-compatible map-style dataset either way.

    Args:
        datasets: Paired-image datasets to combine (e.g. ``Rain13KDataset``,
            ``SPADataDataset``, ``ImagePairDataset``). Each must already be
            transformed to a common output shape (typically via a shared
            ``RandomCrop`` in each dataset's own ``transform``).
        weights: Optional per-dataset oversampling weight, same length and
            order as ``datasets``. A weight of 2.0 means that dataset's
            pairs are, on average, drawn twice as often per epoch by a
            sampler built from :meth:`sample_weights`, relative to a
            weight-1.0 dataset. Defaults to 1.0 (natural frequency) for
            every dataset if not given.

    Example:
        >>> from torch.utils.data import DataLoader, WeightedRandomSampler
        >>> mixed = MixedDataset(
        ...     [rain13k_train, ddn_train, spa_train, rr1k_h_train, rr1k_l_train],
        ...     weights=[1.0, 1.0, 2.0, 2.0, 2.0],  # mild oversampling of real data
        ... )
        >>> sampler = WeightedRandomSampler(
        ...     mixed.sample_weights(), num_samples=len(mixed), replacement=True
        ... )
        >>> loader = DataLoader(mixed, batch_size=8, sampler=sampler)
    """

    def __init__(
        self,
        datasets: Sequence[Dataset],
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        """Initialize the mixed dataset."""
        if not datasets:
            raise ValueError("MixedDataset requires at least one dataset")

        super().__init__(list(datasets))

        if weights is None:
            weights = [1.0] * len(datasets)
        elif len(weights) != len(datasets):
            raise ValueError(
                f"weights must have the same length as datasets "
                f"({len(weights)} != {len(datasets)})"
            )

        self.source_weights: List[float] = list(weights)

    def sample_weights(self) -> List[float]:
        """Return one weight per example, for use with ``WeightedRandomSampler``.

        Each example inherits its source dataset's configured weight, so a
        source with weight 2.0 is (on average, with ``replacement=True``)
        sampled twice as often per epoch as a same-size weight-1.0 source,
        regardless of how many examples it actually contains.
        """
        per_example_weights: List[float] = []
        for dataset, weight in zip(self.datasets, self.source_weights):
            per_example_weights.extend([weight] * len(dataset))
        return per_example_weights


__all__ = [
    "ImagePairDataset",
    "SingleFolderDataset",
    "Rain100Dataset",
    "SyntheticRainDataset",
    "Rain1400Dataset",
    "Rain13KDataset",
    "DDNDataDataset",
    "DIDDataDataset",
    "SPADataDataset",
    "MixedDataset",
]
