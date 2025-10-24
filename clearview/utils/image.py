"""Image processing utilities.

Provides common image operations like normalization, format conversion,
and preprocessing functions.
"""

from typing import Optional, Tuple, Union, cast

import numpy as np
import torch
import torch.nn.functional as F


def normalize_image(
    image: Union[torch.Tensor, np.ndarray],
    mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
    std: Tuple[float, ...] = (0.5, 0.5, 0.5),
) -> Union[torch.Tensor, np.ndarray]:
    """Normalize image with mean and standard deviation.

    Args:
        image: Image tensor/array (C, H, W) or (B, C, H, W)
        mean: Mean for each channel
        std: Standard deviation for each channel

    Returns:
        Normalized image

    Example:
        >>> img = torch.rand(3, 256, 256)  # Range [0, 1]
        >>> normalized = normalize_image(img)  # Range [-1, 1]
    """
    if isinstance(image, torch.Tensor):
        device = image.device
        mean_t = torch.tensor(mean, device=device).view(-1, 1, 1)
        std_t = torch.tensor(std, device=device).view(-1, 1, 1)

        if image.dim() == 4:
            mean_t = mean_t.unsqueeze(0)
            std_t = std_t.unsqueeze(0)

        result: torch.Tensor = (image - mean_t) / std_t
        return result
    else:
        mean_np = np.array(mean).reshape(-1, 1, 1)
        std_np = np.array(std).reshape(-1, 1, 1)

        if image.ndim == 4:
            mean_np = mean_np[np.newaxis, ...]
            std_np = std_np[np.newaxis, ...]

        result_np: np.ndarray = (image - mean_np) / std_np
        return result_np


def denormalize_image(
    image: Union[torch.Tensor, np.ndarray],
    mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
    std: Tuple[float, ...] = (0.5, 0.5, 0.5),
) -> Union[torch.Tensor, np.ndarray]:
    """Denormalize image (reverse of normalize_image).

    Args:
        image: Normalized image
        mean: Mean used for normalization
        std: Std used for normalization

    Returns:
        Denormalized image

    Example:
        >>> normalized = normalize_image(img)
        >>> original = denormalize_image(normalized)
    """
    if isinstance(image, torch.Tensor):
        device = image.device
        mean_t = torch.tensor(mean, device=device).view(-1, 1, 1)
        std_t = torch.tensor(std, device=device).view(-1, 1, 1)

        if image.dim() == 4:
            mean_t = mean_t.unsqueeze(0)
            std_t = std_t.unsqueeze(0)

        result: torch.Tensor = image * std_t + mean_t
        return result
    else:
        mean_np = np.array(mean).reshape(-1, 1, 1)
        std_np = np.array(std).reshape(-1, 1, 1)

        if image.ndim == 4:
            mean_np = mean_np[np.newaxis, ...]
            std_np = std_np[np.newaxis, ...]

        result_np: np.ndarray = image * std_np + mean_np
        return result_np


def rgb_to_grayscale(
    image: Union[torch.Tensor, np.ndarray],
    weights: Tuple[float, float, float] = (0.299, 0.587, 0.114),
) -> Union[torch.Tensor, np.ndarray]:
    """Convert RGB image to grayscale.

    Uses standard RGB to grayscale conversion weights.

    Args:
        image: RGB image (3, H, W) or (B, 3, H, W)
        weights: RGB weights for grayscale conversion

    Returns:
        Grayscale image (1, H, W) or (B, 1, H, W)

    Example:
        >>> rgb = torch.rand(3, 256, 256)
        >>> gray = rgb_to_grayscale(rgb)  # (1, 256, 256)
    """
    if isinstance(image, torch.Tensor):
        channel_idx = 1 if image.dim() == 4 else 0
        if image.size(channel_idx) == 1:
            return image

        device = image.device
        weights_t = torch.tensor(weights, device=device)

        if image.dim() == 4:
            weights_t = weights_t.view(1, 3, 1, 1)
            gray: torch.Tensor = (image * weights_t).sum(dim=1, keepdim=True)
        else:
            weights_t = weights_t.view(3, 1, 1)
            gray = (image * weights_t).sum(dim=0, keepdim=True)

        return gray
    else:
        channel_idx = 1 if image.ndim == 4 else 0
        if image.shape[channel_idx] == 1:
            return image

        weights_np = np.array(weights)

        if image.ndim == 4:
            weights_np = weights_np.reshape(1, 3, 1, 1)
            gray_np: np.ndarray = (image * weights_np).sum(axis=1, keepdims=True)
        else:
            weights_np = weights_np.reshape(3, 1, 1)
            gray_np = (image * weights_np).sum(axis=0, keepdims=True)

        return gray_np


def tensor_to_numpy(
    tensor: torch.Tensor,
    denormalize: bool = False,
    mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
    std: Tuple[float, ...] = (0.5, 0.5, 0.5),
) -> np.ndarray:
    """Convert PyTorch tensor to NumPy array.

    Args:
        tensor: Image tensor (C, H, W) or (B, C, H, W)
        denormalize: Whether to denormalize before conversion
        mean: Mean for denormalization
        std: Std for denormalization

    Returns:
        NumPy array (H, W, C) or (B, H, W, C)

    Example:
        >>> tensor = torch.rand(3, 256, 256)
        >>> array = tensor_to_numpy(tensor)  # (256, 256, 3)
    """
    tensor_to_convert = tensor
    if denormalize:
        denorm_result = denormalize_image(tensor, mean, std)
        if not isinstance(denorm_result, torch.Tensor):
            raise TypeError(
                f"Expected torch.Tensor denorm_result, got {type(denorm_result)}"
            )
        tensor_to_convert = denorm_result

    # Move to CPU and convert
    array: np.ndarray = tensor_to_convert.detach().cpu().numpy()

    # Transpose to (H, W, C) format
    if array.ndim == 4:
        array = np.transpose(array, (0, 2, 3, 1))
    else:
        array = np.transpose(array, (1, 2, 0))

    return array


def numpy_to_tensor(
    array: np.ndarray,
    normalize: bool = False,
    mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
    std: Tuple[float, ...] = (0.5, 0.5, 0.5),
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert NumPy array to PyTorch tensor.

    Args:
        array: Image array (H, W, C) or (B, H, W, C)
        normalize: Whether to normalize after conversion
        mean: Mean for normalization
        std: Std for normalization
        device: Target device

    Returns:
        Tensor (C, H, W) or (B, C, H, W)

    Example:
        >>> array = np.random.rand(256, 256, 3)
        >>> tensor = numpy_to_tensor(array)  # (3, 256, 256)
    """
    # Transpose to (C, H, W) format
    if array.ndim == 4:
        tensor = torch.from_numpy(array.transpose(0, 3, 1, 2).copy())
    else:
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())

    # Convert to float
    if tensor.dtype != torch.float32:
        tensor = tensor.float()

    # Normalize
    if normalize:
        norm_result = normalize_image(tensor, mean, std)
        if not isinstance(norm_result, torch.Tensor):
            raise TypeError(
                f"Expected torch.Tensor denorm_result, got {type(norm_result)}"
            )
        tensor = norm_result

    # Move to device
    if device is not None:
        tensor = tensor.to(device)

    return tensor


def resize_image(
    image: Union[torch.Tensor, np.ndarray],
    size: Tuple[int, int],
    mode: str = "bilinear",
) -> Union[torch.Tensor, np.ndarray]:
    """Resize image to target size.

    Args:
        image: Image (C, H, W) or (B, C, H, W)
        size: Target (height, width)
        mode: Interpolation mode ('bilinear', 'nearest', 'bicubic')

    Returns:
        Resized image

    Example:
        >>> img = torch.rand(3, 512, 512)
        >>> resized = resize_image(img, (256, 256))  # (3, 256, 256)
    """
    if isinstance(image, torch.Tensor):
        squeeze = False
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze = True

        resized: torch.Tensor = F.interpolate(
            image,
            size=size,
            mode=mode,
            align_corners=False if mode != "nearest" else None,
        )

        if squeeze:
            resized = resized.squeeze(0)

        return resized
    else:
        import cv2

        # OpenCV interpolation modes
        cv2_modes = {
            "bilinear": cv2.INTER_LINEAR,
            "nearest": cv2.INTER_NEAREST,
            "bicubic": cv2.INTER_CUBIC,
        }

        interp = cv2_modes.get(mode, cv2.INTER_LINEAR)

        if image.ndim == 4:
            resized_np = np.stack(
                [
                    cv2.resize(
                        img.transpose(1, 2, 0), (size[1], size[0]), interpolation=interp
                    ).transpose(2, 0, 1)
                    for img in image
                ]
            )
        else:
            resized_np = cv2.resize(
                image.transpose(1, 2, 0), (size[1], size[0]), interpolation=interp
            ).transpose(2, 0, 1)

        return cast(np.ndarray, resized_np)


def pad_to_multiple(
    image: torch.Tensor, multiple: int = 32, mode: str = "reflect"
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """Pad image so dimensions are multiples of a value.

    Useful for models that require specific input sizes (e.g., U-Net with pooling).

    Args:
        image: Image tensor (B, C, H, W)
        multiple: Pad to multiple of this value
        mode: Padding mode ('reflect', 'replicate', 'constant')

    Returns:
        Tuple of (padded_image, padding_values)
        padding_values = (left, right, top, bottom) for unpadding

    Example:
        >>> img = torch.rand(1, 3, 250, 250)
        >>> padded, pad_vals = pad_to_multiple(img, multiple=32)
        >>> print(padded.shape)  # (1, 3, 256, 256)
        >>>
        >>> # Unpad after processing
        >>> l, r, t, b = pad_vals
        >>> unpadded = padded[:, :, t:padded.size(2)-b, l:padded.size(3)-r]
    """
    _, _, h, w = image.shape

    # Calculate padding needed
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    # Split padding between sides
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Apply padding (left, right, top, bottom)
    padded = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode=mode)

    padding_vals = (pad_left, pad_right, pad_top, pad_bottom)

    return padded, padding_vals


__all__ = [
    "normalize_image",
    "denormalize_image",
    "rgb_to_grayscale",
    "tensor_to_numpy",
    "numpy_to_tensor",
    "resize_image",
    "pad_to_multiple",
]
