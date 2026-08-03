"""Frequency-domain loss functions.

Implements losses that operate in the frequency domain via the Fast Fourier
Transform (FFT). Rain streaks and other restoration artifacts have distinctive
spectral signatures that are not always well captured by pixel-space losses
(L1/L2/SSIM), making frequency-domain supervision a useful complement.
"""

from typing import Any, Dict, Tuple

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


class WaveletLoss(BaseLoss):
    """Multi-scale Haar wavelet loss.

    Decomposes both prediction and target into an approximation (LL) and
    three detail (LH/HL/HH) sub-bands using the 2D Haar wavelet transform,
    then penalizes discrepancies at each sub-band across ``levels``
    recursive decompositions (each level operates on the previous level's
    LL sub-band, halving resolution each time — similar to an image
    pyramid). Unlike a single global FFT, the wavelet transform preserves
    *spatial* localization of frequency content, so the loss can still
    tell "this patch has a high-frequency error" rather than only "this
    frequency has an error somewhere in the image" — useful for rain
    streaks, which are spatially localized high-frequency structures.

    Args:
        levels: Number of wavelet decomposition levels. Default: 2
        detail_weight: Extra multiplier applied to the detail sub-bands
            (LH/HL/HH) relative to the approximation (LL) sub-band, since
            rain streaks and fine texture live in the detail bands.
            Default: 1.0 (no extra weighting)
        reduction: Reduction method. One of 'mean' | 'sum' (``'none'`` is
            not supported since sub-bands across levels have different
            spatial shapes and cannot be combined without a reduction).
            Default: 'mean'
        weight: Loss weight. Default: 1.0

    Raises:
        ValueError: If ``levels < 1`` or ``reduction == 'none'``

    Example:
        >>> loss_fn = WaveletLoss(levels=2)
        >>> pred = torch.rand(4, 3, 256, 256)
        >>> target = torch.rand(4, 3, 256, 256)
        >>> loss = loss_fn(pred, target)
    """

    def __init__(
        self,
        levels: int = 2,
        detail_weight: float = 1.0,
        reduction: str = "mean",
        weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize wavelet loss.

        Args:
            levels: Number of wavelet decomposition levels
            detail_weight: Extra multiplier for detail sub-bands
            reduction: Reduction method ('mean' | 'sum')
            weight: Loss weight
            **kwargs: Additional arguments

        Raises:
            ValueError: If levels < 1 or reduction is 'none'
        """
        super().__init__(reduction=reduction, weight=weight, **kwargs)

        if levels < 1:
            raise ValueError(f"levels must be >= 1, got {levels}")
        if reduction == "none":
            raise ValueError(
                "WaveletLoss does not support reduction='none' — sub-bands "
                "from different decomposition levels have different spatial "
                "shapes and cannot be combined without a reduction."
            )

        self.levels = levels
        self.detail_weight = detail_weight

        # 2x2 Haar analysis filters (approximation + 3 detail sub-bands),
        # each normalized by 1/2. Shape (4, 1, 2, 2) so they can be applied
        # depthwise (per input channel) via a grouped conv2d.
        ll = torch.tensor([[1.0, 1.0], [1.0, 1.0]]) * 0.5
        lh = torch.tensor([[1.0, 1.0], [-1.0, -1.0]]) * 0.5
        hl = torch.tensor([[1.0, -1.0], [1.0, -1.0]]) * 0.5
        hh = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) * 0.5
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("filters", filters)

    def _dwt(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-level 2D Haar DWT.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Tuple of (ll, lh, hl, hh) sub-bands, each (B, C, ceil(H/2), ceil(W/2))
        """
        b, c, h, w = x.shape
        pad_h, pad_w = h % 2, w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        filters: torch.Tensor = self.filters
        depthwise_filters = filters.repeat(c, 1, 1, 1).to(dtype=x.dtype)
        out = F.conv2d(x, depthwise_filters, stride=2, groups=c)

        _, _, h2, w2 = out.shape
        out = out.view(b, c, 4, h2, w2)
        return out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale wavelet loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            Wavelet loss value
        """
        loss_tensor = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        pred_ll, target_ll = pred, target

        for _ in range(self.levels):
            p_ll, p_lh, p_hl, p_hh = self._dwt(pred_ll)
            t_ll, t_lh, t_hl, t_hh = self._dwt(target_ll)

            loss_tensor = loss_tensor + F.l1_loss(p_ll, t_ll, reduction=self.reduction)
            detail_loss = (
                F.l1_loss(p_lh, t_lh, reduction=self.reduction)
                + F.l1_loss(p_hl, t_hl, reduction=self.reduction)
                + F.l1_loss(p_hh, t_hh, reduction=self.reduction)
            )
            loss_tensor = loss_tensor + self.detail_weight * detail_loss

            pred_ll, target_ll = p_ll, t_ll

        return self.apply_weight(loss_tensor)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        config = super().get_config()
        config["levels"] = self.levels
        config["detail_weight"] = self.detail_weight
        return config
