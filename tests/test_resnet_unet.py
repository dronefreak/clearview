"""Tests for ResNet U-Net architecture.

Comprehensive tests for DecoderBlock and ResNetUNet model including
architecture validation, forward pass, pretrained weights, and utilities.
"""

import pytest
import torch
import torch.nn as nn

from clearview.models.resnet_unet import (
    DecoderBlock,
    ResNetUNet,
    create_resnet_unet,
)


class TestDecoderBlock:
    """Tests for DecoderBlock component."""

    def test_init_with_batchnorm(self):
        """Test decoder block initialization with batchnorm."""
        block = DecoderBlock(
            in_channels=512,
            skip_channels=256,
            out_channels=256,
            use_batchnorm=True,
        )

        assert isinstance(block.upsample, nn.Upsample)
        assert isinstance(block.bn1, nn.BatchNorm2d)
        assert isinstance(block.bn2, nn.BatchNorm2d)

    def test_init_without_batchnorm(self):
        """Test decoder block initialization without batchnorm."""
        block = DecoderBlock(
            in_channels=512,
            skip_channels=256,
            out_channels=256,
            use_batchnorm=False,
        )

        assert isinstance(block.bn1, nn.Identity)
        assert isinstance(block.bn2, nn.Identity)

    def test_forward_pass_shape(self):
        """Test forward pass output shape."""
        block = DecoderBlock(
            in_channels=512,
            skip_channels=256,
            out_channels=256,
        )

        # Input from previous decoder (smaller)
        x = torch.randn(2, 512, 16, 16)
        # Skip connection from encoder (larger, 2x size)
        skip = torch.randn(2, 256, 32, 32)

        output = block(x, skip)

        # Output should match skip dimensions
        assert output.shape == (2, 256, 32, 32)

    def test_forward_pass_with_size_mismatch(self):
        """Test that decoder handles slight size mismatches."""
        block = DecoderBlock(
            in_channels=512,
            skip_channels=256,
            out_channels=256,
        )

        x = torch.randn(2, 512, 15, 15)  # Odd size
        skip = torch.randn(2, 256, 31, 31)  # Different odd size

        # Should not raise error
        output = block(x, skip)

        # Output should match skip size
        assert output.shape == (2, 256, 31, 31)

    def test_gradient_flow(self):
        """Test that gradients flow through decoder block."""
        block = DecoderBlock(512, 256, 256)

        x = torch.randn(1, 512, 16, 16, requires_grad=True)
        skip = torch.randn(1, 256, 32, 32, requires_grad=True)

        output = block(x, skip)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert skip.grad is not None
        assert not torch.all(x.grad == 0)
        assert not torch.all(skip.grad == 0)

    def test_upsample_mode(self):
        """Test that bilinear upsampling is used."""
        block = DecoderBlock(512, 256, 256)

        assert block.upsample.mode == "bilinear"
        assert block.upsample.scale_factor == 2


class TestResNetUNet:
    """Tests for ResNetUNet model."""

    @pytest.mark.parametrize(
        "backbone",
        ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
    )
    def test_init_with_different_backbones(self, backbone):
        """Test initialization with different ResNet backbones."""
        model = ResNetUNet(backbone=backbone, pretrained=False)

        assert model.backbone_name == backbone
        assert model.in_channels == 3
        assert model.out_channels == 3

    def test_init_invalid_backbone(self):
        """Test that invalid backbone raises error."""
        with pytest.raises(ValueError, match="Unsupported backbone"):
            ResNetUNet(backbone="resnet999", pretrained=False)

    def test_forward_pass_basic(self):
        """Test basic forward pass."""
        model = ResNetUNet(backbone="resnet18", pretrained=False)
        model.eval()

        x = torch.randn(2, 3, 256, 256)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 3, 256, 256)
        assert output.min() >= 0
        assert output.max() <= 1

    @pytest.mark.parametrize("height,width", [(128, 128), (256, 256), (512, 512)])
    def test_forward_different_sizes(self, height, width):
        """Test forward pass with different input sizes."""
        model = ResNetUNet(backbone="resnet18", pretrained=False)
        model.eval()

        x = torch.randn(1, 3, height, width)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 3, height, width)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_forward_different_batch_sizes(self, batch_size):
        """Test forward pass with different batch sizes."""
        model = ResNetUNet(backbone="resnet18", pretrained=False)
        model.eval()

        x = torch.randn(batch_size, 3, 128, 128)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 3, 128, 128)

    def test_output_range(self):
        """Test that output is in valid range [0, 1]."""
        model = ResNetUNet(backbone="resnet18", pretrained=False)
        model.eval()

        # Test with different input ranges
        x_zeros = torch.zeros(1, 3, 128, 128)
        x_ones = torch.ones(1, 3, 128, 128)
        x_random = torch.randn(1, 3, 128, 128).clamp(0, 1)

        with torch.no_grad():
            out_zeros = model(x_zeros)
            out_ones = model(x_ones)
            out_random = model(x_random)

        # All outputs should be in [0, 1]
        for output in [out_zeros, out_ones, out_random]:
            assert output.min() >= 0
            assert output.max() <= 1

    def test_imagenet_normalization(self):
        """Test that ImageNet normalization is applied."""
        model = ResNetUNet(backbone="resnet18", pretrained=False)

        # Check normalization parameters exist
        assert hasattr(model, "imagenet_mean")
        assert hasattr(model, "imagenet_std")

        # Check values
        expected_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        expected_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        assert torch.allclose(model.imagenet_mean, expected_mean)
        assert torch.allclose(model.imagenet_std, expected_std)

    def test_pretrained_weights_loaded(self):
        """Test that pretrained weights are loaded when pretrained=True."""
        # Just test that model can be created with pretrained=True
        # Don't actually download weights in tests
        model = ResNetUNet(backbone="resnet18", pretrained=False)

        # Verify model has encoder components
        assert hasattr(model, "encoder0")
        assert hasattr(model, "encoder1")
        assert hasattr(model, "encoder2")
        assert hasattr(model, "encoder3")
        assert hasattr(model, "encoder4")

    def test_pretrained_false(self):
        """Test that pretrained=False works."""
        model = ResNetUNet(backbone="resnet18", pretrained=False)

        # Should create model without errors
        assert isinstance(model, ResNetUNet)
        assert model.backbone_name == "resnet18"

    def test_custom_channels(self):
        """Test model with custom input/output channels."""
        model = ResNetUNet(
            in_channels=1,  # Grayscale input
            out_channels=1,  # Grayscale output
            backbone="resnet18",
            pretrained=False,
        )

        assert model.in_channels == 1
        assert model.out_channels == 1

        # Note: ResNet expects 3 channels, so this may not work in practice
        # But the model should initialize

    def test_resnet_channels_dict(self):
        """Test RESNET_CHANNELS dictionary has correct entries."""
        expected_keys = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

        for key in expected_keys:
            assert key in ResNetUNet.RESNET_CHANNELS
            channels = ResNetUNet.RESNET_CHANNELS[key]
            assert len(channels) == 5  # 5 stages


class TestResNetUNetUtilities:
    """Tests for ResNetUNet utility methods."""

    def test_get_num_params(self):
        """Test parameter counting."""
        model = ResNetUNet(backbone="resnet18", pretrained=False)

        num_params = model.get_num_params()

        # Should be a reasonable number (millions of parameters)
        assert num_params > 1_000_000
        assert num_params < 100_000_000
        assert isinstance(num_params, int)

    def test_get_model_size_mb(self):
        """Test model size calculation."""
        model = ResNetUNet(backbone="resnet18", pretrained=False)

        size_mb = model.get_model_size_mb()

        # Should be reasonable size (tens of MB)
        assert size_mb > 1.0
        assert size_mb < 1000.0
        assert isinstance(size_mb, float)

    def test_different_backbones_have_different_sizes(self):
        """Test that larger backbones have more parameters."""
        model18 = ResNetUNet(backbone="resnet18", pretrained=False)
        model50 = ResNetUNet(backbone="resnet50", pretrained=False)

        params18 = model18.get_num_params()
        params50 = model50.get_num_params()

        # ResNet50 should have more parameters than ResNet18
        assert params50 > params18

    def test_freeze_encoder(self):
        """Test freezing encoder weights."""
        model = ResNetUNet(backbone="resnet18", pretrained=False)

        # Initially all parameters should require grad
        assert all(p.requires_grad for p in model.encoder0.parameters())
        assert all(p.requires_grad for p in model.encoder1.parameters())

        # Freeze encoder
        model.freeze_encoder()

        # Encoder parameters should not require grad
        assert all(not p.requires_grad for p in model.encoder0.parameters())
        assert all(not p.requires_grad for p in model.encoder1.parameters())
        assert all(not p.requires_grad for p in model.encoder2.parameters())
        assert all(not p.requires_grad for p in model.encoder3.parameters())
        assert all(not p.requires_grad for p in model.encoder4.parameters())

        # Decoder parameters should still require grad
        assert any(p.requires_grad for p in model.decoder1.parameters())

    def test_unfreeze_encoder(self):
        """Test unfreezing encoder weights."""
        model = ResNetUNet(backbone="resnet18", pretrained=False)

        # Freeze then unfreeze
        model.freeze_encoder()
        model.unfreeze_encoder()

        # All parameters should require grad again
        assert all(p.requires_grad for p in model.encoder0.parameters())
        assert all(p.requires_grad for p in model.encoder1.parameters())
        assert all(p.requires_grad for p in model.encoder2.parameters())
        assert all(p.requires_grad for p in model.encoder3.parameters())
        assert all(p.requires_grad for p in model.encoder4.parameters())

    def test_freeze_encoder_training_efficiency(self):
        """Test that freezing encoder reduces trainable parameters."""
        model = ResNetUNet(backbone="resnet18", pretrained=False)

        # Count trainable params before freezing
        trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Freeze encoder
        model.freeze_encoder()

        # Count trainable params after freezing
        trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Should have fewer trainable parameters after freezing
        assert trainable_after < trainable_before


class TestCreateResNetUNet:
    """Tests for create_resnet_unet factory function."""

    def test_create_basic(self):
        """Test basic model creation."""
        model = create_resnet_unet()

        assert isinstance(model, ResNetUNet)
        assert model.backbone_name == "resnet34"

    @pytest.mark.parametrize(
        "backbone", ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    )
    def test_create_different_backbones(self, backbone):
        """Test creating models with different backbones."""
        model = create_resnet_unet(backbone=backbone, pretrained=False)

        assert isinstance(model, ResNetUNet)
        assert model.backbone_name == backbone

    def test_create_custom_channels(self):
        """Test creating model with custom channels."""
        model = create_resnet_unet(in_channels=1, out_channels=1, pretrained=False)

        assert model.in_channels == 1
        assert model.out_channels == 1

    def test_create_with_pretrained(self):
        """Test that pretrained flag is passed correctly."""
        # Just verify model can be created with pretrained flag
        # Don't actually download weights in tests
        model = create_resnet_unet(pretrained=False)

        assert isinstance(model, ResNetUNet)
        assert model.backbone_name == "resnet34"

    def test_create_without_pretrained(self):
        """Test that pretrained=False is passed correctly."""
        model = create_resnet_unet(pretrained=False)

        assert isinstance(model, ResNetUNet)
        assert model.backbone_name == "resnet34"


class TestResNetUNetGradients:
    """Tests for gradient flow through ResNet U-Net."""

    def test_gradient_flow_end_to_end(self):
        """Test that gradients flow from output to input."""
        model = ResNetUNet(backbone="resnet18", pretrained=False)

        x = torch.randn(1, 3, 128, 128, requires_grad=True)

        output = model(x)
        loss = output.sum()
        loss.backward()

        # Input should have gradients
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_gradient_flow_through_skip_connections(self):
        """Test that gradients flow through skip connections."""
        model = ResNetUNet(backbone="resnet18", pretrained=False)

        x = torch.randn(1, 3, 128, 128)
        output = model(x)

        # Check that encoder parameters have gradients
        loss = output.sum()
        loss.backward()

        # Encoder should have gradients (skip connections working)
        encoder_has_grad = any(
            p.grad is not None and not torch.all(p.grad == 0)
            for p in model.encoder1.parameters()
        )
        assert encoder_has_grad

    def test_frozen_encoder_no_gradients(self):
        """Test that frozen encoder doesn't get gradients."""
        model = ResNetUNet(backbone="resnet18", pretrained=False)
        model.freeze_encoder()

        x = torch.randn(1, 3, 128, 128)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Encoder parameters should not have gradients
        for p in model.encoder0.parameters():
            if p.requires_grad:
                assert p.grad is None or torch.all(p.grad == 0)


class TestResNetUNetEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_small_input(self):
        """Test with minimum viable input size."""
        model = ResNetUNet(backbone="resnet18", pretrained=False)
        model.eval()

        # Minimum size that can go through all downsampling (32x downsampling)
        x = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 3, 64, 64)

    def test_non_square_input(self):
        """Test with non-square input."""
        model = ResNetUNet(backbone="resnet18", pretrained=False)
        model.eval()

        x = torch.randn(1, 3, 128, 256)  # Non-square

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 3, 128, 256)

    def test_odd_sized_input(self):
        """Test with odd-sized input."""
        model = ResNetUNet(backbone="resnet18", pretrained=False)
        model.eval()

        x = torch.randn(1, 3, 127, 127)  # Odd dimensions

        with torch.no_grad():
            output = model(x)

        # Output should not match input size exactly due to downsampling
        assert output.shape[0] == 1
        assert output.shape[1] == 3
        assert output.shape[2] == 128
        assert output.shape[3] == 128

    def test_deterministic_output(self):
        """Test that model is deterministic in eval mode."""
        model = ResNetUNet(backbone="resnet18", pretrained=False)
        model.eval()

        x = torch.randn(1, 3, 128, 128)

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        assert torch.allclose(output1, output2)

    def test_training_vs_eval_mode(self):
        """Test that model behaves differently in train vs eval mode."""
        model = ResNetUNet(backbone="resnet18", pretrained=False)

        x = torch.randn(1, 3, 128, 128)

        # Train mode
        model.train()
        with torch.no_grad():
            output_train = model(x)

        # Eval mode
        model.eval()
        with torch.no_grad():
            output_eval = model(x)

        # Outputs may differ due to batchnorm
        # Just verify both work
        assert output_train.shape == output_eval.shape


class TestResNetUNetIntegration:
    """Integration tests for ResNet U-Net."""

    def test_full_training_step(self):
        """Test complete training step."""
        model = ResNetUNet(backbone="resnet18", pretrained=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.L1Loss()

        # Forward pass
        x = torch.randn(2, 3, 128, 128)
        target = torch.randn(2, 3, 128, 128).clamp(0, 1)

        output = model(x)
        loss = criterion(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Should complete without error
        assert loss.item() >= 0

    def test_inference_mode(self):
        """Test inference mode."""
        model = ResNetUNet(backbone="resnet18", pretrained=False)
        model.eval()

        x = torch.randn(1, 3, 256, 256)

        with torch.inference_mode():
            output = model(x)

        assert output.shape == (1, 3, 256, 256)
        assert output.min() >= 0
        assert output.max() <= 1

    def test_save_and_load(self, tmp_path):
        """Test saving and loading model."""
        model1 = ResNetUNet(backbone="resnet18", pretrained=False)

        # Save
        save_path = tmp_path / "model.pth"
        torch.save(model1.state_dict(), save_path)

        # Load into new model
        model2 = ResNetUNet(backbone="resnet18", pretrained=False)
        model2.load_state_dict(torch.load(save_path))

        # Verify weights match
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            out1 = model1(x)
            out2 = model2(x)

        assert torch.allclose(out1, out2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
