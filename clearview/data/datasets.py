"""Dataset implementations for image deraining.

Provides dataset classes for loading paired rainy/clean images from
various formats (image pairs, directories, etc.).
"""

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union, cast

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

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

        # Get image file lists
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
                (0, cast(Union[int, str], image_id))
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
]
