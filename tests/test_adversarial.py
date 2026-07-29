"""Unit tests for adversarial (GAN) loss functions."""

import pytest
import torch

from clearview.losses import CombinedLoss
from clearview.losses.adversarial import AdversarialLoss, GANLoss, PatchDiscriminator


class TestPatchDiscriminator:
    """Tests for PatchDiscriminator."""

    def test_initialization(self) -> None:
        """Test default initialization."""
        discriminator = PatchDiscriminator()
        assert discriminator.in_channels == 3
        assert discriminator.base_channels == 64
        assert discriminator.num_layers == 3
        assert discriminator.use_spectral_norm is True

    def test_forward_pass_shape(self) -> None:
        """Test that forward pass produces a patch-wise logit map."""
        discriminator = PatchDiscriminator(in_channels=3)
        x = torch.randn(2, 3, 64, 64)
        logits = discriminator(x)

        assert logits.ndim == 4
        assert logits.shape[0] == 2
        assert logits.shape[1] == 1
        # Patch map should be smaller than the input spatial size.
        assert logits.shape[2] < x.shape[2]
        assert logits.shape[3] < x.shape[3]

    def test_forward_pass_not_sigmoid_bounded(self) -> None:
        """Test that output logits are raw (not squashed to [0, 1])."""
        discriminator = PatchDiscriminator()
        x = torch.randn(2, 3, 64, 64) * 10
        logits = discriminator(x)

        assert (logits.min() < 0) or (logits.max() > 1)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow back to the input image."""
        discriminator = PatchDiscriminator()
        x = torch.randn(2, 3, 64, 64, requires_grad=True)
        logits = discriminator(x)
        logits.sum().backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_gradient_flow_to_parameters(self) -> None:
        """Test that gradients flow to all discriminator parameters."""
        discriminator = PatchDiscriminator()
        x = torch.randn(2, 3, 64, 64)
        logits = discriminator(x)
        logits.sum().backward()

        for name, param in discriminator.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_without_spectral_norm(self) -> None:
        """Test that the discriminator works with spectral norm disabled."""
        discriminator = PatchDiscriminator(use_spectral_norm=False)
        x = torch.randn(2, 3, 64, 64)
        logits = discriminator(x)

        assert logits.shape[0] == 2

    def test_custom_num_layers(self) -> None:
        """Test that num_layers controls network depth/downsampling."""
        shallow = PatchDiscriminator(num_layers=1)
        deep = PatchDiscriminator(num_layers=4)

        x = torch.randn(2, 3, 128, 128)
        shallow_out = shallow(x)
        deep_out = deep(x)

        # A deeper network should downsample more aggressively.
        assert deep_out.shape[2] < shallow_out.shape[2]

    def test_get_config(self) -> None:
        """Test configuration serialization."""
        discriminator = PatchDiscriminator(
            in_channels=1, base_channels=32, num_layers=2, use_spectral_norm=False
        )
        config = discriminator.get_config()

        assert config == {
            "in_channels": 1,
            "base_channels": 32,
            "num_layers": 2,
            "use_spectral_norm": False,
        }


class TestGANLoss:
    """Tests for GANLoss."""

    def test_invalid_gan_mode_raises(self) -> None:
        """Test that an unsupported gan_mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown gan_mode"):
            GANLoss(gan_mode="not_a_real_mode")

    @pytest.mark.parametrize("gan_mode", ["vanilla", "lsgan", "hinge"])
    def test_generator_loss_scalar(self, gan_mode: str) -> None:
        """Test that generator_loss returns a scalar for all supported modes."""
        gan_loss = GANLoss(gan_mode=gan_mode)
        fake_logits = torch.randn(2, 1, 8, 8)

        loss = gan_loss.generator_loss(fake_logits)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    @pytest.mark.parametrize("gan_mode", ["vanilla", "lsgan", "hinge"])
    def test_discriminator_loss_scalar(self, gan_mode: str) -> None:
        """Test that discriminator_loss returns a scalar for all supported modes."""
        gan_loss = GANLoss(gan_mode=gan_mode)
        real_logits = torch.randn(2, 1, 8, 8)
        fake_logits = torch.randn(2, 1, 8, 8)

        loss = gan_loss.discriminator_loss(real_logits, fake_logits)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    @pytest.mark.parametrize("gan_mode", ["vanilla", "lsgan"])
    def test_discriminator_loss_low_for_well_separated_logits(
        self, gan_mode: str
    ) -> None:
        """Test discriminator_loss is low when real/fake logits are ideal."""
        gan_loss = GANLoss(gan_mode=gan_mode)
        # Strongly "real" and strongly "fake" logits -- an (almost) perfect
        # discriminator should incur very low loss.
        real_logits = torch.full((2, 1, 4, 4), 10.0)
        fake_logits = torch.full((2, 1, 4, 4), -10.0)

        good_loss = gan_loss.discriminator_loss(real_logits, fake_logits)
        bad_loss = gan_loss.discriminator_loss(fake_logits, real_logits)

        assert good_loss.item() < bad_loss.item()

    def test_hinge_generator_loss_formula(self) -> None:
        """Test hinge generator loss matches -mean(fake_logits)."""
        gan_loss = GANLoss(gan_mode="hinge")
        fake_logits = torch.tensor([[1.0, -2.0, 3.0]])

        loss = gan_loss.generator_loss(fake_logits)

        assert torch.isclose(loss, -fake_logits.mean())

    def test_lsgan_generator_loss_zero_when_fake_looks_real(self) -> None:
        """Test lsgan generator loss is (near) zero when D(fake) == real_label."""
        gan_loss = GANLoss(gan_mode="lsgan", real_label=1.0)
        fake_logits = torch.ones(2, 1, 4, 4)

        loss = gan_loss.generator_loss(fake_logits)

        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


class TestAdversarialLoss:
    """Tests for AdversarialLoss."""

    def test_initialization(self) -> None:
        """Test default initialization."""
        discriminator = PatchDiscriminator()
        loss_fn = AdversarialLoss(discriminator=discriminator)

        assert loss_fn.gan_mode == "lsgan"
        assert loss_fn.weight == 1.0

    def test_forward_pass_scalar(self) -> None:
        """Test that forward returns a scalar loss."""
        discriminator = PatchDiscriminator()
        loss_fn = AdversarialLoss(discriminator=discriminator)

        pred = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)
        loss = loss_fn(pred, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_forward_ignores_target(self) -> None:
        """Test that the loss value is unaffected by the `target` argument."""
        discriminator = PatchDiscriminator()
        discriminator.eval()  # freeze spectral-norm power-iteration buffers
        loss_fn = AdversarialLoss(discriminator=discriminator)

        pred = torch.rand(2, 3, 64, 64)
        target_a = torch.rand(2, 3, 64, 64)
        target_b = torch.rand(2, 3, 64, 64)

        loss_a = loss_fn(pred, target_a)
        loss_b = loss_fn(pred, target_b)

        assert torch.isclose(loss_a, loss_b)

    def test_gradient_flow_to_generator_output(self) -> None:
        """Test that gradients flow from the loss back to `pred`."""
        discriminator = PatchDiscriminator()
        loss_fn = AdversarialLoss(discriminator=discriminator)

        pred = torch.rand(2, 3, 64, 64, requires_grad=True)
        target = torch.rand(2, 3, 64, 64)
        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert not torch.all(pred.grad == 0)

    def test_loss_weight(self) -> None:
        """Test that weight scales the loss value."""
        torch.manual_seed(0)
        discriminator = PatchDiscriminator()
        discriminator.eval()  # freeze spectral-norm power-iteration buffers
        pred = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)

        loss_fn = AdversarialLoss(discriminator=discriminator, weight=1.0)
        loss_fn_scaled = AdversarialLoss(discriminator=discriminator, weight=2.0)

        loss = loss_fn(pred, target)
        loss_scaled = loss_fn_scaled(pred, target)

        assert torch.isclose(loss_scaled, loss * 2.0)

    def test_get_config(self) -> None:
        """Test configuration serialization includes gan_mode."""
        discriminator = PatchDiscriminator()
        loss_fn = AdversarialLoss(
            discriminator=discriminator, gan_mode="hinge", weight=0.5
        )

        config = loss_fn.get_config()

        assert config["gan_mode"] == "hinge"
        assert config["weight"] == 0.5

    def test_combined_loss_from_config(self) -> None:
        """Test that AdversarialLoss can be registered via CombinedLoss.from_config."""
        discriminator = PatchDiscriminator()
        combined = CombinedLoss.from_config(
            {
                "l1": {"weight": 1.0},
                "adversarial": {"weight": 0.01, "discriminator": discriminator},
            }
        )

        pred = torch.rand(2, 3, 64, 64, requires_grad=True)
        target = torch.rand(2, 3, 64, 64)
        loss = combined(pred, target)
        loss.backward()

        assert isinstance(loss, torch.Tensor)
        assert pred.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
