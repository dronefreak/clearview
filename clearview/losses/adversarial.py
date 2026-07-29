"""Adversarial (GAN) loss functions for photorealistic image restoration.

Provides a PatchGAN-style discriminator and the associated generator/
discriminator adversarial losses. Adversarial training removes the
"regression-to-the-mean" over-smoothing that pure pixel/perceptual losses
(L1, L2, SSIM) tend to introduce, at the cost of a more involved (two
optimizer) training loop.

Since :class:`~clearview.training.trainer.Trainer` only manages a single
optimizer/model pair, GAN training requires a custom training loop. A
typical pattern looks like::

    >>> from clearview.losses.adversarial import GANLoss, PatchDiscriminator
    >>> discriminator = PatchDiscriminator(in_channels=3)
    >>> gan_loss = GANLoss(gan_mode="lsgan")
    >>> g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
    >>> d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    >>>
    >>> for rainy, clean in train_loader:
    ...     fake_clean = generator(rainy)
    ...
    ...     # --- Discriminator step ---
    ...     d_optimizer.zero_grad()
    ...     real_logits = discriminator(clean)
    ...     fake_logits = discriminator(fake_clean.detach())
    ...     d_loss = gan_loss.discriminator_loss(real_logits, fake_logits)
    ...     d_loss.backward()
    ...     d_optimizer.step()
    ...
    ...     # --- Generator step (pixel/perceptual + adversarial) ---
    ...     g_optimizer.zero_grad()
    ...     recon_loss = pixel_loss_fn(fake_clean, clean)
    ...     adv_loss = adversarial_loss_fn(fake_clean, clean)  # AdversarialLoss
    ...     g_loss = recon_loss + adv_loss
    ...     g_loss.backward()
    ...     g_optimizer.step()

For convenience, :class:`AdversarialLoss` wraps the generator-side term as a
``BaseLoss`` so it can be combined with other losses via
:class:`~clearview.losses.combined.CombinedLoss` (the discriminator itself
still needs to be trained separately, as shown above).
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from clearview.losses.base import BaseLoss

_SUPPORTED_GAN_MODES = frozenset({"vanilla", "lsgan", "hinge"})


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator (pix2pix-style).

    Classifies overlapping N x N patches of the input as real/fake rather
    than the whole image at once, encouraging high-frequency (texture-level)
    realism — well suited as an adversarial signal for restoration tasks.

    Args:
        in_channels: Number of input image channels. Default: 3
        base_channels: Number of channels in the first conv layer. Default: 64
        num_layers: Number of downsampling conv layers. Default: 3
        use_spectral_norm: Apply spectral normalization to conv layers for
            training stability. Default: True

    Example:
        >>> discriminator = PatchDiscriminator(in_channels=3)
        >>> x = torch.randn(2, 3, 256, 256)
        >>> patch_logits = discriminator(x)  # (2, 1, H', W')
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 3,
        use_spectral_norm: bool = True,
    ) -> None:
        """Initialize the PatchGAN discriminator."""
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_layers = num_layers
        self.use_spectral_norm = use_spectral_norm

        def norm(module: nn.Module) -> nn.Module:
            return nn.utils.spectral_norm(module) if use_spectral_norm else module

        layers: List[nn.Module] = [
            norm(nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        channels = base_channels
        for i in range(1, num_layers):
            next_channels = min(channels * 2, base_channels * 8)
            stride = 2 if i < num_layers - 1 else 1
            layers.append(
                norm(nn.Conv2d(channels, next_channels, 4, stride=stride, padding=1))
            )
            layers.append(nn.InstanceNorm2d(next_channels, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            channels = next_channels

        # Final layer maps to a single-channel real/fake patch map
        layers.append(norm(nn.Conv2d(channels, 1, 4, stride=1, padding=1)))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute patch-wise real/fake logits.

        Args:
            x: Input image (B, C, H, W)

        Returns:
            Patch logits (B, 1, H', W') — NOT passed through sigmoid
        """
        return self.model(x)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for serialization."""
        return {
            "in_channels": self.in_channels,
            "base_channels": self.base_channels,
            "num_layers": self.num_layers,
            "use_spectral_norm": self.use_spectral_norm,
        }


class GANLoss:
    """Generator and discriminator adversarial loss terms.

    Supports three common GAN objectives:

    - ``'vanilla'``: original minimax GAN loss (binary cross-entropy)
    - ``'lsgan'``: least-squares GAN (MSE against real/fake labels),
      generally more stable to train. Default.
    - ``'hinge'``: hinge loss, commonly used in modern high-fidelity GANs

    Args:
        gan_mode: One of 'vanilla' | 'lsgan' | 'hinge'. Default: 'lsgan'
        real_label: Target value for real samples (vanilla/lsgan only).
            Default: 1.0
        fake_label: Target value for fake samples (vanilla/lsgan only).
            Default: 0.0

    Example:
        >>> gan_loss = GANLoss(gan_mode="lsgan")
        >>> d_loss = gan_loss.discriminator_loss(real_logits, fake_logits)
        >>> g_loss = gan_loss.generator_loss(fake_logits)
    """

    def __init__(
        self,
        gan_mode: str = "lsgan",
        real_label: float = 1.0,
        fake_label: float = 0.0,
    ) -> None:
        """Initialize GAN loss."""
        if gan_mode not in _SUPPORTED_GAN_MODES:
            raise ValueError(
                f"Unknown gan_mode '{gan_mode}'. "
                f"Supported: {sorted(_SUPPORTED_GAN_MODES)}"
            )

        self.gan_mode = gan_mode
        self.real_label = real_label
        self.fake_label = fake_label

    def _target_tensor(self, logits: torch.Tensor, is_real: bool) -> torch.Tensor:
        """Build a constant target tensor matching `logits`'s shape."""
        value = self.real_label if is_real else self.fake_label
        return torch.full_like(logits, value)

    def generator_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        """Compute the generator's adversarial loss (wants D(fake) -> real).

        Args:
            fake_logits: Discriminator output on generator predictions

        Returns:
            Scalar generator adversarial loss
        """
        if self.gan_mode == "hinge":
            return -fake_logits.mean()
        elif self.gan_mode == "lsgan":
            return F.mse_loss(fake_logits, self._target_tensor(fake_logits, True))
        else:  # vanilla
            return F.binary_cross_entropy_with_logits(
                fake_logits, self._target_tensor(fake_logits, True)
            )

    def discriminator_loss(
        self, real_logits: torch.Tensor, fake_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute the discriminator's loss (real -> real, fake -> fake).

        Args:
            real_logits: Discriminator output on real (ground-truth) images
            fake_logits: Discriminator output on generator predictions
                (should be detached from the generator's graph)

        Returns:
            Scalar discriminator loss
        """
        if self.gan_mode == "hinge":
            loss_real = F.relu(1.0 - real_logits).mean()
            loss_fake = F.relu(1.0 + fake_logits).mean()
        elif self.gan_mode == "lsgan":
            loss_real = F.mse_loss(real_logits, self._target_tensor(real_logits, True))
            loss_fake = F.mse_loss(fake_logits, self._target_tensor(fake_logits, False))
        else:  # vanilla
            loss_real = F.binary_cross_entropy_with_logits(
                real_logits, self._target_tensor(real_logits, True)
            )
            loss_fake = F.binary_cross_entropy_with_logits(
                fake_logits, self._target_tensor(fake_logits, False)
            )
        return (loss_real + loss_fake) * 0.5


class AdversarialLoss(BaseLoss):
    """Generator-side adversarial loss, usable inside ``CombinedLoss``.

    Wraps :class:`GANLoss`'s ``generator_loss`` as a standard ``BaseLoss``
    so it can be combined with pixel/perceptual losses. The discriminator
    passed in must be trained separately (see module docstring for the
    typical dual-optimizer training loop pattern).

    Args:
        discriminator: A discriminator module (e.g. :class:`PatchDiscriminator`)
            mapping images to patch-wise real/fake logits
        gan_mode: One of 'vanilla' | 'lsgan' | 'hinge'. Default: 'lsgan'
        reduction: Reduction method (unused directly — GANLoss always
            reduces internally — kept for ``BaseLoss`` interface consistency)
        weight: Loss weight. Default: 1.0

    Example:
        >>> discriminator = PatchDiscriminator()
        >>> adv_loss = AdversarialLoss(discriminator, gan_mode="lsgan", weight=0.01)
        >>> fake_clean = generator(rainy)
        >>> loss = adv_loss(fake_clean, clean)  # `clean` is unused, kept for BaseLoss API
    """

    def __init__(
        self,
        discriminator: nn.Module,
        gan_mode: str = "lsgan",
        reduction: str = "mean",
        weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize adversarial loss wrapping a discriminator."""
        super().__init__(reduction=reduction, weight=weight, **kwargs)
        self.discriminator = discriminator
        self.gan_mode = gan_mode
        self._gan_loss = GANLoss(gan_mode=gan_mode)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the generator's adversarial loss on `pred`.

        Args:
            pred: Generator output (fake image), (B, C, H, W)
            target: Unused — present only to satisfy the ``BaseLoss`` interface

        Returns:
            Weighted generator adversarial loss
        """
        del (
            target
        )  # unused: adversarial loss only needs the discriminator's verdict on `pred`
        fake_logits = self.discriminator(pred)
        loss = self._gan_loss.generator_loss(fake_logits)
        return self.apply_weight(loss)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        config = super().get_config()
        config["gan_mode"] = self.gan_mode
        return config


__all__ = ["PatchDiscriminator", "GANLoss", "AdversarialLoss"]
