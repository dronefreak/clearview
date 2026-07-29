"""NIQE (Natural Image Quality Evaluator) — a no-reference image quality metric.

Implements the classic Mittal et al. (2013) NIQE algorithm: natural scene
statistics (mean-subtracted-contrast-normalized coefficients fit to an
Asymmetric Generalized Gaussian Distribution) are extracted from
non-overlapping patches at two scales, and an image's quality is scored as
the Mahalanobis-like distance between its feature distribution and a
"pristine" reference distribution.

Unlike BRISQUE (which ships a fixed pretrained SVR model), NIQE's reference
statistics are inherently corpus-dependent: the original paper fits them
from a curated set of high-quality natural images. To avoid silently
depending on unverified bundled parameters, this module requires the
pristine reference model to be fit explicitly via :func:`fit_niqe_model`
from a set of clean/high-quality images (e.g. a dataset's ground-truth
images), then reused for scoring with :func:`compute_niqe`.

Reference:
    Mittal, Soundararajan, Bovik. "Making a 'Completely Blind' Image
    Quality Analyzer." IEEE Signal Processing Letters, 2013.
"""

from dataclasses import dataclass
from typing import List, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

from clearview.utils.image import rgb_to_grayscale

_PATCH_SIZE = 96
_GAUSSIAN_WINDOW_SIZE = 7
_GAUSSIAN_SIGMA = 7.0 / 6.0
_C = 1.0  # MSCN normalization constant


@dataclass
class NIQEModel:
    """Fitted pristine reference statistics for NIQE scoring.

    Attributes:
        mu: Mean feature vector (36,) fit from pristine images
        cov: Covariance matrix (36, 36) fit from pristine images
    """

    mu: np.ndarray
    cov: np.ndarray


def _gaussian_window(size: int, sigma: float) -> torch.Tensor:
    """Create a normalized 2D Gaussian window."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window = g.outer(g)
    return (window / window.sum()).view(1, 1, size, size)


def _mscn_coefficients(gray: torch.Tensor) -> torch.Tensor:
    """Compute mean-subtracted contrast-normalized (MSCN) coefficients.

    Args:
        gray: Grayscale image (B, 1, H, W) in range [0, 255]

    Returns:
        MSCN coefficients (B, 1, H, W)
    """
    window = _gaussian_window(_GAUSSIAN_WINDOW_SIZE, _GAUSSIAN_SIGMA).to(gray.device)
    pad = _GAUSSIAN_WINDOW_SIZE // 2

    mu = F.conv2d(gray, window, padding=pad)
    sigma_sq = F.conv2d(gray**2, window, padding=pad) - mu**2
    sigma = torch.sqrt(torch.clamp(sigma_sq, min=0.0))

    return (gray - mu) / (sigma + _C)


def _aggd_fit(block: np.ndarray) -> np.ndarray:
    """Fit an Asymmetric Generalized Gaussian Distribution to a 1D array.

    Args:
        block: Flattened array of coefficients

    Returns:
        Array of [alpha, left_std, right_std, mean] parameters
    """
    block = block.flatten()
    left = block[block < 0]
    right = block[block >= 0]

    left_std = np.sqrt(np.mean(left**2)) if left.size > 0 else 1e-8
    right_std = np.sqrt(np.mean(right**2)) if right.size > 0 else 1e-8
    left_std = max(left_std, 1e-8)
    right_std = max(right_std, 1e-8)

    gamma_hat = left_std / right_std
    abs_mean = np.mean(np.abs(block))
    r_hat = (abs_mean**2) / np.mean(block**2) if np.mean(block**2) > 0 else 0.0
    rhat_norm = r_hat * (gamma_hat**3 + 1) * (gamma_hat + 1) / ((gamma_hat**2 + 1) ** 2)

    # Solve for alpha via a precomputed lookup table (as in the original
    # NIQE/BRISQUE implementations), matching gamma function ratios.
    try:
        from scipy.special import gamma as _gamma_fn
    except ImportError as e:
        raise ImportError(
            "The 'scipy' package is required for NIQE computation. "
            "Install it with `pip install scipy`."
        ) from e

    alpha_candidates = np.arange(0.2, 10.001, 0.001)

    r_gam = (_gamma_fn(2.0 / alpha_candidates) ** 2) / (
        _gamma_fn(1.0 / alpha_candidates) * _gamma_fn(3.0 / alpha_candidates)
    )
    alpha = alpha_candidates[np.argmin(np.abs(r_gam - rhat_norm))]

    const = np.sqrt(_gamma_fn(1.0 / alpha) / _gamma_fn(3.0 / alpha))
    mean = (
        (right_std - left_std)
        * const
        * (_gamma_fn(2.0 / alpha) / _gamma_fn(1.0 / alpha))
    )

    return np.array([alpha, left_std, right_std, mean])


def _patch_features(patch: np.ndarray) -> np.ndarray:
    """Compute the 36-dim NIQE feature vector for a single MSCN patch.

    Args:
        patch: MSCN coefficient patch (H, W)

    Returns:
        36-dimensional feature vector
    """
    features: List[float] = []

    # Feature 1-2: MSCN AGGD fit (alpha, variance)
    aggd = _aggd_fit(patch)
    alpha, left_std, right_std = aggd[0], aggd[1], aggd[2]
    features.extend([alpha, (left_std**2 + right_std**2) / 2.0])

    # Features from pairwise products in 4 directions (horizontal, vertical,
    # main-diagonal, anti-diagonal), each contributing 4 AGGD params.
    shifts = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dy, dx in shifts:
        shifted = np.roll(np.roll(patch, dy, axis=0), dx, axis=1)
        product = patch * shifted
        aggd_shift = _aggd_fit(product)
        alpha_s, left_s, right_s, mean_s = aggd_shift
        features.extend([alpha_s, mean_s, left_s**2, right_s**2])

    return np.array(features, dtype=np.float64)


def _extract_features(image_gray_np: np.ndarray, patch_size: int) -> np.ndarray:
    """Extract per-patch NIQE features (2-scale) from a grayscale image.

    Args:
        image_gray_np: Grayscale image array (H, W) in range [0, 255]
        patch_size: Size of non-overlapping patches at the original scale

    Returns:
        Array of shape (num_patches, 36) — one feature vector per patch
        (concatenating scale-1 [18-dim] and scale-2 [18-dim] features)
    """
    gray_tensor = torch.from_numpy(image_gray_np).float().unsqueeze(0).unsqueeze(0)

    scale_features = []
    current = gray_tensor
    for _scale in range(2):
        mscn = _mscn_coefficients(current)[0, 0].numpy()

        h, w = mscn.shape
        n_h, n_w = h // patch_size, w // patch_size

        patch_feats = []
        for i in range(n_h):
            for j in range(n_w):
                patch = mscn[
                    i * patch_size : (i + 1) * patch_size,
                    j * patch_size : (j + 1) * patch_size,
                ]
                feats = _patch_features(patch)
                # First 2 entries -> scale global feature; the rest 16 form
                # the direction-product features (2 + 4*4 = 18 total).
                patch_feats.append(feats)

        scale_features.append(
            np.stack(patch_feats) if patch_feats else np.zeros((0, 18))
        )

        # Downsample by 2 for the next scale
        current = F.avg_pool2d(current, kernel_size=2)

    n_patches = min(scale_features[0].shape[0], scale_features[1].shape[0])
    if n_patches == 0:
        raise ValueError(
            f"Image too small for NIQE patch size {patch_size}: "
            f"got grayscale shape {image_gray_np.shape}"
        )

    combined = np.concatenate(
        [scale_features[0][:n_patches], scale_features[1][:n_patches]], axis=1
    )
    return combined


def _to_grayscale_numpy(image: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert a single image (any supported format) to an (H, W) grayscale array in [0, 255]."""
    if isinstance(image, torch.Tensor):
        img = image
        if img.dim() == 4:
            if img.size(0) != 1:
                raise ValueError(
                    "_to_grayscale_numpy expects a single image; "
                    "iterate over the batch dimension for batched input."
                )
            img = img[0]
        if img.size(0) == 3:
            img = rgb_to_grayscale(img)
        gray_np = img[0].detach().cpu().numpy()
    else:
        gray_np = image
        if gray_np.ndim == 3:
            gray_np = (
                np.mean(gray_np, axis=-1) if gray_np.shape[-1] == 3 else gray_np[..., 0]
            )

    if gray_np.max() <= 1.0 + 1e-6:
        gray_np = gray_np * 255.0

    return gray_np.astype(np.float64)


def fit_niqe_model(
    images: Sequence[Union[torch.Tensor, np.ndarray]],
    patch_size: int = _PATCH_SIZE,
) -> NIQEModel:
    """Fit NIQE's pristine reference statistics from a set of clean images.

    Args:
        images: Sequence of high-quality/clean reference images. Each can be
            a (3, H, W)/(1, H, W) torch.Tensor or (H, W, 3)/(H, W) np.ndarray,
            with pixel values in [0, 1] or [0, 255]
        patch_size: Non-overlapping patch size at the original scale

    Returns:
        Fitted :class:`NIQEModel` with mean vector and covariance matrix

    Raises:
        ValueError: If no valid patches could be extracted from any image

    Example:
        >>> clean_images = [torch.rand(3, 256, 256) for _ in range(50)]
        >>> model = fit_niqe_model(clean_images)
        >>> score = compute_niqe(distorted_image, model)
    """
    all_features = []

    for image in images:
        gray_np = _to_grayscale_numpy(image)
        try:
            feats = _extract_features(gray_np, patch_size)
        except ValueError:
            continue
        if feats.shape[0] > 0:
            all_features.append(feats)

    if not all_features:
        raise ValueError(
            "Could not extract any NIQE features from the provided images. "
            f"Ensure images are at least {patch_size}x{patch_size} pixels."
        )

    features = np.concatenate(all_features, axis=0)
    mu = np.mean(features, axis=0)
    cov = np.cov(features, rowvar=False)

    return NIQEModel(mu=mu, cov=cov)


def compute_niqe(
    image: Union[torch.Tensor, np.ndarray],
    model: NIQEModel,
    patch_size: int = _PATCH_SIZE,
) -> float:
    """Score an image's quality against a fitted NIQE pristine model.

    Args:
        image: Single image to score — (3, H, W)/(1, H, W) torch.Tensor or
            (H, W, 3)/(H, W) np.ndarray, pixel values in [0, 1] or [0, 255]
        model: Pristine reference statistics from :func:`fit_niqe_model`
        patch_size: Non-overlapping patch size at the original scale
            (must match the value used to fit ``model``)

    Returns:
        NIQE score — lower indicates better (more "natural") quality

    Raises:
        ValueError: If the image is smaller than ``patch_size``

    Example:
        >>> model = fit_niqe_model(clean_reference_images)
        >>> score = compute_niqe(restored_image, model)
    """
    gray_np = _to_grayscale_numpy(image)
    features = _extract_features(gray_np, patch_size)

    distorted_mu = np.mean(features, axis=0)
    distorted_cov = (
        np.cov(features, rowvar=False) if features.shape[0] > 1 else model.cov
    )

    cov_avg = (model.cov + distorted_cov) / 2.0
    # Add a small epsilon for numerical stability before inversion.
    cov_avg += np.eye(cov_avg.shape[0]) * 1e-8
    inv_cov = np.linalg.pinv(cov_avg)

    diff = (model.mu - distorted_mu).reshape(1, -1)
    distance = np.sqrt(diff @ inv_cov @ diff.T)

    return float(distance.item())


__all__ = ["NIQEModel", "fit_niqe_model", "compute_niqe"]
