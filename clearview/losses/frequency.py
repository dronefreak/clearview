"""Frequency-domain loss functions.

Implements losses that operate in the frequency domain via the Fast Fourier
Transform (FFT). Rain streaks and other restoration artifacts have distinctive
spectral signatures that are not always well captured by pixel-space losses
(L1/L2/SSIM), making frequency-domain supervision a useful complement.
"""

from typing import Any, Dict

import torch
import torch.nn.functional as F

from clearview.losses.base import BaseLoss


class FFTLoss(BaseLoss):
    """FFT-based frequency loss.

    Computes an L1 (or L2) loss between the 2D FFT representations of the
    predicted and target images. Operating on the amplitude/phase-preserving
    complex spectrum penalizes frequency-domain discrepancies (e.g. residual
    rain streaks with periodic/high-frequency structure) that pixel-space
    losses may under-penalize.

    Args:
        reduction: Reduction method. Default: 'mean'
        weight: Loss weight. Default: 1.0
        norm: FFT normalization mode passed to ``torch.fft.rfft2``.
            Default: 'backward'
        criterion: Distance metric applied to the (stacked real/imaginary)
            spectra. One of 'l1' | 'l2'. Default: 'l1'

    Example:
        >>> loss_fn = FFTLoss()
        >>> pred = torch.randn(4, 3, 256, 256)
        >>> target = torch.randn(4, 3, 256, 256)
        >>> loss = loss_fn(pred, target)
    """

    def __init__(
        self,
        reduction: str = "mean",
        weight: float = 1.0,
        norm: str = "backward",
        criterion: str = "l1",
        **kwargs: Any,
    ) -> None:
        """Initialize FFT loss.

        Args:
            reduction: Reduction method
            weight: Loss weight
            norm: FFT normalization mode ('backward' | 'forward' | 'ortho')
            criterion: Distance metric on the spectra ('l1' | 'l2')
            **kwargs: Additional arguments

        Raises:
            ValueError: If criterion is not 'l1' or 'l2'
        """
        super().__init__(reduction=reduction, weight=weight, **kwargs)

        if criterion not in ("l1", "l2"):
            raise ValueError(f"Invalid criterion: {criterion}. Choose from 'l1', 'l2'.")

        self.norm = norm
        self.criterion = criterion

    def _compute_spectrum(self, img: torch.Tensor) -> torch.Tensor:
        """Compute the 2D real FFT spectrum of an image, stacked as real channels.

        Args:
            img: Input image (B, C, H, W)

        Returns:
            Stacked real/imaginary spectrum of shape (B, C, H, W_freq, 2)
        """
        # rfft2 assumes a real-valued input and only computes the
        # non-redundant half of the spectrum along the last dimension.
        spectrum = torch.fft.rfft2(img.float(), norm=self.norm)
        return torch.stack([spectrum.real, spectrum.imag], dim=-1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute FFT loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            FFT loss value
        """
        pred_spectrum = self._compute_spectrum(pred)
        target_spectrum = self._compute_spectrum(target)

        if self.criterion == "l1":
            loss = F.l1_loss(pred_spectrum, target_spectrum, reduction=self.reduction)
        else:
            loss = F.mse_loss(pred_spectrum, target_spectrum, reduction=self.reduction)

        return self.apply_weight(loss)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        config = super().get_config()
        config["norm"] = self.norm
        config["criterion"] = self.criterion
        return config


class FocalFrequencyLoss(BaseLoss):
    """Focal Frequency Loss (Jiang et al., ICCV 2021).

    Adaptively down-weights easy frequency components (already well
    reconstructed) and focuses training on hard frequencies, using a
    per-frequency weighting matrix derived from the current spectral
    distance itself. This is more effective at closing frequency-domain
    gaps than a plain FFT L1/L2 loss, particularly for high-frequency
    detail such as rain streaks.

    Args:
        reduction: Reduction method. Default: 'mean'
        weight: Loss weight. Default: 1.0
        alpha: Scaling exponent for the frequency weighting matrix.
            Default: 1.0
        ave_spectrum: Whether to average the spectrum over the batch
            before computing the loss (as in the original paper's batch
            statistics mode). Default: False

    Example:
        >>> loss_fn = FocalFrequencyLoss()
        >>> pred = torch.randn(4, 3, 256, 256)
        >>> target = torch.randn(4, 3, 256, 256)
        >>> loss = loss_fn(pred, target)
    """

    def __init__(
        self,
        reduction: str = "mean",
        weight: float = 1.0,
        alpha: float = 1.0,
        ave_spectrum: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize focal frequency loss.

        Args:
            reduction: Reduction method
            weight: Loss weight
            alpha: Scaling exponent applied to the frequency weight matrix
            ave_spectrum: Whether to average spectra across the batch
            **kwargs: Additional arguments
        """
        super().__init__(reduction=reduction, weight=weight, **kwargs)
        self.alpha = alpha
        self.ave_spectrum = ave_spectrum

    def _compute_spectrum(self, img: torch.Tensor) -> torch.Tensor:
        """Compute the 2D FFT spectrum stacked as real channels.

        Args:
            img: Input image (B, C, H, W)

        Returns:
            Stacked real/imaginary spectrum of shape (B, C, H, W_freq, 2)
        """
        spectrum = torch.fft.rfft2(img.float(), norm="ortho")
        return torch.stack([spectrum.real, spectrum.imag], dim=-1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal frequency loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            Focal frequency loss value
        """
        pred_spectrum = self._compute_spectrum(pred)
        target_spectrum = self._compute_spectrum(target)

        if self.ave_spectrum:
            pred_spectrum = pred_spectrum.mean(dim=0, keepdim=True)
            target_spectrum = target_spectrum.mean(dim=0, keepdim=True)

        # Per-frequency squared distance
        freq_distance = (pred_spectrum - target_spectrum) ** 2
        freq_distance = freq_distance.sum(
            dim=-1
        )  # combine real/imag -> squared magnitude of diff

        # Dynamic spectrum weight matrix: harder (larger error) frequencies
        # get higher weight. Detached so it purely re-weights the loss
        # landscape without contributing its own gradient.
        weight_matrix = freq_distance.detach().clone() ** self.alpha
        max_val = weight_matrix.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        weight_matrix = weight_matrix / max_val
        weight_matrix = torch.nan_to_num(weight_matrix, nan=0.0)

        loss = weight_matrix * freq_distance

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        # else: 'none' -> leave per-element

        return self.apply_weight(loss)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        config = super().get_config()
        config["alpha"] = self.alpha
        config["ave_spectrum"] = self.ave_spectrum
        return config
