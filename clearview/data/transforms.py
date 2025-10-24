"""Data augmentation and transformation utilities.

Provides augmentation pipelines for training image deraining models.
"""

from typing import Dict, Optional, Tuple, List, Any
import random

import numpy as np


class PairedTransform:
    """Base class for transforms that apply to both rainy and clean images.

    Ensures geometric transforms are applied consistently to paired images.
    """

    def __call__(
        self, image: np.ndarray, target: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Apply transform.

        Args:
            image: Rainy image (H, W, C)
            target: Clean image (H, W, C), optional

        Returns:
            Dictionary with 'image' and optionally 'target'
        """
        raise NotImplementedError


class RandomCrop(PairedTransform):
    """Random crop for paired images.

    Args:
        crop_size: Size of crop (height, width)

    Example:
        >>> transform = RandomCrop(crop_size=(256, 256))
        >>> result = transform(image=rainy, target=clean)
    """

    def __init__(self, crop_size: Tuple[int, int]) -> None:
        """Initialize random crop."""
        self.crop_size = crop_size

    def __call__(
        self, image: np.ndarray, target: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Apply random crop."""
        h, w = image.shape[:2]
        crop_h, crop_w = self.crop_size

        if h < crop_h or w < crop_w:
            raise ValueError(
                f"Image size ({h}, {w}) is smaller than crop size {self.crop_size}"
            )

        # Random top-left corner
        top = random.randint(0, h - crop_h)  # nosec B311
        left = random.randint(0, w - crop_w)  # nosec B311

        # Crop
        image_cropped = image[top : top + crop_h, left : left + crop_w]

        result = {"image": image_cropped}

        if target is not None:
            target_cropped = target[top : top + crop_h, left : left + crop_w]
            result["target"] = target_cropped

        return result


class CenterCrop(PairedTransform):
    """Center crop for paired images.

    Args:
        crop_size: Size of crop (height, width)
    """

    def __init__(self, crop_size: Tuple[int, int]) -> None:
        """Initialize center crop."""
        self.crop_size = crop_size

    def __call__(
        self, image: np.ndarray, target: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Apply center crop."""
        h, w = image.shape[:2]
        crop_h, crop_w = self.crop_size

        # Center coordinates
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2

        # Crop
        image_cropped = image[top : top + crop_h, left : left + crop_w]

        result = {"image": image_cropped}

        if target is not None:
            target_cropped = target[top : top + crop_h, left : left + crop_w]
            result["target"] = target_cropped

        return result


class RandomHorizontalFlip(PairedTransform):
    """Random horizontal flip for paired images.

    Args:
        p: Probability of flipping
    """

    def __init__(self, p: float = 0.5) -> None:
        """Initialize random flip."""
        self.p = p

    def __call__(
        self, image: np.ndarray, target: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Apply random horizontal flip."""
        if random.random() < self.p:  # nosec B311
            image = np.fliplr(image).copy()
            result = {"image": image}

            if target is not None:
                target = np.fliplr(target).copy()
                result["target"] = target
        else:
            result = {"image": image}
            if target is not None:
                result["target"] = target

        return result


class RandomVerticalFlip(PairedTransform):
    """Random vertical flip for paired images.

    Args:
        p: Probability of flipping
    """

    def __init__(self, p: float = 0.5) -> None:
        """Initialize random flip."""
        self.p = p

    def __call__(
        self, image: np.ndarray, target: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Apply random vertical flip."""
        if random.random() < self.p:  # nosec B311
            image = np.flipud(image).copy()
            result = {"image": image}

            if target is not None:
                target = np.flipud(target).copy()
                result["target"] = target
        else:
            result = {"image": image}
            if target is not None:
                result["target"] = target

        return result


class RandomRotation(PairedTransform):
    """Random 90-degree rotations for paired images.

    Args:
        rotations: List of rotation angles (0, 90, 180, 270)
    """

    def __init__(self, rotations: Tuple[int, ...] = (0, 90, 180, 270)) -> None:
        """Initialize random rotation."""
        self.rotations = rotations
        self.k_values = {0: 0, 90: 1, 180: 2, 270: 3}

    def __call__(
        self, image: np.ndarray, target: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Apply random rotation."""
        angle = random.choice(self.rotations)  # nosec B311
        k = self.k_values[angle]

        image = np.rot90(image, k=k, axes=(0, 1)).copy()
        result = {"image": image}

        if target is not None:
            target = np.rot90(target, k=k, axes=(0, 1)).copy()
            result["target"] = target

        return result


class Resize(PairedTransform):
    """Resize images to target size.

    Args:
        size: Target size (height, width)
    """

    def __init__(self, size: Tuple[int, int]) -> None:
        """Initialize resize."""
        self.size = size

    def __call__(
        self, image: np.ndarray, target: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Apply resize."""
        try:
            import cv2

            image = cv2.resize(
                image, (self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR
            )
            result = {"image": image}

            if target is not None:
                target = cv2.resize(
                    target, (self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR
                )
                result["target"] = target
        except ImportError:
            raise ImportError("OpenCV (cv2) is required for resize transform")

        return result


class Compose:
    """Compose multiple transforms.

    Args:
        transforms: List of transform instances

    Example:
        >>> transform = Compose([
        ...     RandomCrop((256, 256)),
        ...     RandomHorizontalFlip(p=0.5),
        ...     RandomRotation(),
        ... ])
    """

    def __init__(self, transforms: List[PairedTransform]) -> None:
        """Initialize composed transforms."""
        self.transforms = transforms

    def __call__(
        self, image: np.ndarray, target: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Apply all transforms sequentially."""
        result = {"image": image}
        if target is not None:
            result["target"] = target

        for transform in self.transforms:
            result = transform(**result)

        return result


def get_train_transforms(
    crop_size: Tuple[int, int] = (256, 256),
    flip_prob: float = 0.5,
    rotate: bool = True,
) -> Compose:
    """Get standard training augmentation pipeline.

    Args:
        crop_size: Random crop size
        flip_prob: Probability of flipping
        rotate: Whether to apply random rotations

    Returns:
        Composed transforms

    Example:
        >>> transforms = get_train_transforms(crop_size=(256, 256))
        >>> dataset = ImagePairDataset('data/train', transform=transforms)
    """
    transforms = [
        RandomCrop(crop_size),
        RandomHorizontalFlip(p=flip_prob),
        RandomVerticalFlip(p=flip_prob),
    ]

    if rotate:
        transforms.append(RandomRotation())

    return Compose(transforms)


def get_val_transforms(
    crop_size: Optional[Tuple[int, int]] = None,
    resize: Optional[Tuple[int, int]] = None,
) -> Compose:
    """Get standard validation transform pipeline.

    Args:
        crop_size: Center crop size (optional)
        resize: Resize target (optional)

    Returns:
        Composed transforms

    Example:
        >>> transforms = get_val_transforms(crop_size=(256, 256))
        >>> dataset = ImagePairDataset('data/val', transform=transforms)
    """
    transforms: List[PairedTransform] = []

    if resize is not None:
        transforms.append(Resize(resize))

    if crop_size is not None:
        transforms.append(CenterCrop(crop_size))

    # If no transforms specified, return identity
    if not transforms:

        class Identity(PairedTransform):
            def __call__(self, image: np.ndarray, target: Any = None) -> Dict[str, Any]:
                result = {"image": image}
                if target is not None:
                    result["target"] = target
                return result

        transforms.append(Identity())

    return Compose(transforms)


__all__ = [
    "PairedTransform",
    "RandomCrop",
    "CenterCrop",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomRotation",
    "Resize",
    "Compose",
    "get_train_transforms",
    "get_val_transforms",
]
