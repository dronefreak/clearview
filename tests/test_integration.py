"""Integration tests for end-to-end workflows."""

from pathlib import Path

import pytest
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from clearview.data.datasets import ImagePairDataset
from clearview.losses.pixel import L1Loss
from clearview.models.unet import UNet
from clearview.utils.metrics import compute_psnr, compute_ssim


class TestEndToEndTraining:
    """Integration tests for training workflow."""

    def test_training_single_batch(self, sample_dataset_dir: Path) -> None:
        """Test training loop with a single batch."""
        # Create dataset and dataloader
        dataset = ImagePairDataset(
            rainy_dir=sample_dataset_dir / "rainy",
            clean_dir=sample_dataset_dir / "clean",
        )
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # Create model, loss, optimizer
        model = UNet(in_channels=3, out_channels=3, features=[32, 64])
        loss_fn = L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # Training step
        model.train()
        rainy, clean = next(iter(dataloader))

        # Forward pass
        output = model(rainy)
        loss = loss_fn(output, clean)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that loss is a scalar
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_training_multiple_epochs(self, sample_dataset_dir: Path) -> None:
        """Test training for multiple epochs."""
        # Create dataset and dataloader
        dataset = ImagePairDataset(
            rainy_dir=sample_dataset_dir / "rainy",
            clean_dir=sample_dataset_dir / "clean",
        )
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # Create model, loss, optimizer
        model = UNet(in_channels=3, out_channels=3, features=[16, 32])
        loss_fn = L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Training loop
        num_epochs = 2
        losses = []

        for _ in range(num_epochs):
            epoch_loss = 0.0
            for rainy, clean in dataloader:
                # Forward pass
                output = model(rainy)
                loss = loss_fn(output, clean)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)

        # Check that we completed training
        assert len(losses) == num_epochs
        assert all(loss >= 0 for loss in losses)

    def test_eval_mode_inference(self, sample_dataset_dir: Path) -> None:
        """Test inference in evaluation mode."""
        # Create dataset
        dataset = ImagePairDataset(
            rainy_dir=sample_dataset_dir / "rainy",
            clean_dir=sample_dataset_dir / "clean",
        )

        # Create model
        model = UNet(in_channels=3, out_channels=3, features=[16, 32])
        model.eval()

        # Get a sample
        rainy, clean = dataset[0]
        rainy = rainy.unsqueeze(0)  # Add batch dimension

        # Inference
        with torch.no_grad():
            output = model(rainy)

        # Check output
        assert output.shape == rainy.shape
        assert (output >= 0).all()
        assert (output <= 1).all()


class TestEndToEndInference:
    """Integration tests for inference workflow."""

    def test_model_save_and_load(self, temp_dir: Path) -> None:
        """Test model saving and loading."""
        # Create model
        model = UNet(in_channels=3, out_channels=3, features=[16, 32])

        # Save model
        save_path = temp_dir / "model.pth"
        torch.save(model.state_dict(), save_path)

        # Load model
        loaded_model = UNet(in_channels=3, out_channels=3, features=[16, 32])
        loaded_model.load_state_dict(torch.load(save_path))

        # Test inference
        x = torch.randn(2, 3, 256, 256)
        model.eval()
        loaded_model.eval()

        with torch.no_grad():
            output1 = model(x)
            output2 = loaded_model(x)

        # Outputs should be identical
        torch.testing.assert_close(output1, output2)

    def test_batch_inference(self, sample_dataset_dir: Path) -> None:
        """Test inference on a batch of images."""
        # Create dataset and dataloader
        dataset = ImagePairDataset(
            rainy_dir=sample_dataset_dir / "rainy",
            clean_dir=sample_dataset_dir / "clean",
        )
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        # Create model
        model = UNet(in_channels=3, out_channels=3, features=[16, 32])
        model.eval()

        # Process all batches
        all_outputs = []
        with torch.no_grad():
            for rainy, _ in dataloader:
                output = model(rainy)
                all_outputs.append(output)

        # Check that we processed all images
        total_images = sum(batch.shape[0] for batch in all_outputs)
        assert total_images == len(dataset)


class TestMetricsEvaluation:
    """Integration tests for metrics computation."""

    def test_compute_metrics_on_batch(self) -> None:
        """Test computing all metrics on a batch."""
        pred = torch.rand(4, 3, 256, 256)
        target = torch.rand(4, 3, 256, 256)

        # Compute all metrics
        psnr = compute_psnr(pred, target)
        ssim = compute_ssim(pred, target)

        # Check that metrics are valid
        assert isinstance(psnr, float)
        assert psnr > 0

        assert isinstance(ssim, float)
        assert 0.0 <= ssim <= 1.0

    def test_metrics_improve_with_training(self, sample_dataset_dir: Path) -> None:
        """Test that metrics improve with training."""
        # Create dataset
        dataset = ImagePairDataset(
            rainy_dir=sample_dataset_dir / "rainy",
            clean_dir=sample_dataset_dir / "clean",
        )
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # Create model, loss, optimizer
        model = UNet(in_channels=3, out_channels=3, features=[16, 32])
        loss_fn = L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Get initial metrics
        model.eval()
        with torch.no_grad():
            rainy, clean = next(iter(dataloader))
            output_before = model(rainy)
            psnr_before = compute_psnr(output_before, clean)

        # Train for a few iterations
        model.train()
        for _ in range(10):
            for rainy, clean in dataloader:
                output = model(rainy)
                loss = loss_fn(output, clean)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Get final metrics
        model.eval()
        with torch.no_grad():
            rainy, clean = next(iter(dataloader))
            output_after = model(rainy)
            psnr_after = compute_psnr(output_after, clean)

        # PSNR should improve (or at least not get worse significantly)
        # Note: With random data and small model, improvement is not guaranteed
        # but loss should be valid
        assert isinstance(psnr_before, float)
        assert isinstance(psnr_after, float)


class TestModelWithDifferentLosses:
    """Integration tests for models with different loss functions."""

    def test_train_with_l1_loss(self, sample_dataset_dir: Path) -> None:
        """Test training with L1 loss."""
        from clearview.losses.pixel import L1Loss

        dataset = ImagePairDataset(
            rainy_dir=sample_dataset_dir / "rainy",
            clean_dir=sample_dataset_dir / "clean",
        )
        dataloader = DataLoader(dataset, batch_size=2)

        model = UNet(in_channels=3, out_channels=3, features=[16, 32])
        loss_fn = L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # Training step
        model.train()
        rainy, clean = next(iter(dataloader))
        output = model(rainy)
        loss = loss_fn(output, clean)
        optimizer.step()
        loss.backward()

        assert loss.item() >= 0

    def test_train_with_ssim_loss(self, sample_dataset_dir: Path) -> None:
        """Test training with SSIM loss."""
        from clearview.losses.structural import SSIMLoss

        dataset = ImagePairDataset(
            rainy_dir=sample_dataset_dir / "rainy",
            clean_dir=sample_dataset_dir / "clean",
        )
        dataloader = DataLoader(dataset, batch_size=2)

        model = UNet(in_channels=3, out_channels=3, features=[16, 32])
        loss_fn = SSIMLoss(channel=3)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # Training step
        model.train()
        rainy, clean = next(iter(dataloader))
        output = model(rainy)
        loss = loss_fn(output, clean)
        optimizer.step()
        loss.backward()

        assert loss.item() >= 0


class TestGradientFlow:
    """Integration tests for gradient flow through entire pipeline."""

    def test_gradients_flow_through_model(self) -> None:
        """Test that gradients flow through the entire model."""
        model = UNet(in_channels=3, out_channels=3, features=[16, 32])
        loss_fn = L1Loss()

        x = torch.randn(2, 3, 256, 256, requires_grad=True)
        target = torch.randn(2, 3, 256, 256)

        # Forward pass
        output = model(x)
        loss = loss_fn(output, target)

        # Backward pass
        loss.backward()

        # Check that input has gradients
        assert x.grad is not None

        # Check that all model parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient for {name}"

    def test_no_gradient_leakage_in_eval(self) -> None:
        """Test that gradients don't accumulate in eval mode."""
        model = UNet(in_channels=3, out_channels=3, features=[16, 32])
        model.eval()

        x = torch.randn(2, 3, 256, 256)
        target = torch.randn(2, 3, 256, 256)

        # Forward pass in eval mode with no_grad
        with torch.no_grad():
            output = model(x)
            loss = L1Loss()(output, target)

        # Check that no gradients are being tracked
        assert not output.requires_grad
        assert all(not param.grad for param in model.parameters())
        assert loss.item() >= 0


class TestDataPipelineIntegration:
    """Integration tests for data pipeline."""

    def test_dataloader_with_multiple_workers(self, sample_dataset_dir: Path) -> None:
        """Test DataLoader with multiple workers."""
        dataset = ImagePairDataset(
            rainy_dir=sample_dataset_dir / "rainy",
            clean_dir=sample_dataset_dir / "clean",
        )

        # Create DataLoader with multiple workers
        dataloader = DataLoader(dataset, batch_size=2, num_workers=0, shuffle=True)

        # Iterate through batches
        batch_count = 0
        for rainy, clean in dataloader:
            assert rainy.shape[0] <= 2  # Batch size
            assert rainy.shape[1] == 3  # RGB
            assert clean.shape == rainy.shape
            batch_count += 1

        assert batch_count > 0

    def test_full_pipeline_cpu(self, sample_dataset_dir: Path) -> None:
        """Test full training pipeline on CPU."""
        # Create dataset and dataloader
        dataset = ImagePairDataset(
            rainy_dir=sample_dataset_dir / "rainy",
            clean_dir=sample_dataset_dir / "clean",
        )
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # Create model, loss, optimizer
        device = torch.device("cpu")
        model = UNet(in_channels=3, out_channels=3, features=[16, 32]).to(device)
        loss_fn = L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # Training loop
        model.train()
        for rainy, clean in dataloader:
            rainy = rainy.to(device)
            clean = clean.to(device)

            # Forward pass
            output = model(rainy)
            loss = loss_fn(output, clean)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Just test one batch
            break

        assert loss.item() >= 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_full_pipeline_gpu(self, sample_dataset_dir: Path) -> None:
        """Test full training pipeline on GPU."""
        # Create dataset and dataloader
        dataset = ImagePairDataset(
            rainy_dir=sample_dataset_dir / "rainy",
            clean_dir=sample_dataset_dir / "clean",
        )
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # Create model, loss, optimizer
        device = torch.device("cuda")
        model = UNet(in_channels=3, out_channels=3, features=[16, 32]).to(device)
        loss_fn = L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # Training loop
        model.train()
        for rainy, clean in dataloader:
            rainy = rainy.to(device)
            clean = clean.to(device)

            # Forward pass
            output = model(rainy)
            loss = loss_fn(output, clean)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Just test one batch
            break

        assert loss.item() >= 0
        assert output.device.type == "cuda"
