"""Tests for checkpoint management utilities.

These tests cover model and checkpoint saving/loading functionality.
"""

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam

from clearview.utils.checkpoint import (
    CheckpointManager,
    load_checkpoint,
    load_model,
    save_checkpoint,
    save_model,
)


# Simple model for testing
class DummyModel(nn.Module):
    """Dummy model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


class TestSaveCheckpoint:
    """Tests for save_checkpoint function."""

    def test_save_basic_checkpoint(self, tmp_path):
        """Test saving basic checkpoint with model only."""
        model = DummyModel()
        filepath = tmp_path / "checkpoint.pth"

        save_checkpoint(model, filepath=filepath)

        assert filepath.exists()

        # Load and verify
        checkpoint = torch.load(filepath)
        assert "model_state_dict" in checkpoint
        assert isinstance(checkpoint["model_state_dict"], dict)

    def test_save_checkpoint_with_optimizer(self, tmp_path):
        """Test saving checkpoint with optimizer state."""
        model = DummyModel()
        optimizer = Adam(model.parameters())
        filepath = tmp_path / "checkpoint.pth"

        save_checkpoint(model, optimizer=optimizer, filepath=filepath)

        checkpoint = torch.load(filepath)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint

    def test_save_checkpoint_with_epoch(self, tmp_path):
        """Test saving checkpoint with epoch number."""
        model = DummyModel()
        filepath = tmp_path / "checkpoint.pth"

        save_checkpoint(model, epoch=50, filepath=filepath)

        checkpoint = torch.load(filepath)
        assert checkpoint["epoch"] == 50

    def test_save_checkpoint_with_metrics(self, tmp_path):
        """Test saving checkpoint with metrics."""
        model = DummyModel()
        metrics = {"psnr": 28.5, "ssim": 0.92}
        filepath = tmp_path / "checkpoint.pth"

        save_checkpoint(model, metrics=metrics, filepath=filepath)

        checkpoint = torch.load(filepath)
        assert checkpoint["metrics"] == metrics

    def test_save_checkpoint_with_config(self, tmp_path):
        """Test saving checkpoint with configuration."""
        model = DummyModel()
        config = {"model": "unet", "lr": 0.001}
        filepath = tmp_path / "checkpoint.pth"

        save_checkpoint(model, config=config, filepath=filepath)

        checkpoint = torch.load(filepath)
        assert checkpoint["config"] == config

    def test_save_checkpoint_with_kwargs(self, tmp_path):
        """Test saving checkpoint with additional kwargs."""
        model = DummyModel()
        filepath = tmp_path / "checkpoint.pth"

        save_checkpoint(
            model,
            filepath=filepath,
            best_metric=30.0,
            scheduler_state={"step": 100},
        )

        checkpoint = torch.load(filepath)
        assert "best_metric" in checkpoint
        assert checkpoint["best_metric"] == 30.0
        assert "scheduler_state" in checkpoint

    def test_save_creates_parent_directory(self, tmp_path):
        """Test that parent directory is created if it doesn't exist."""
        model = DummyModel()
        filepath = tmp_path / "subdir" / "checkpoint.pth"

        save_checkpoint(model, filepath=filepath)

        assert filepath.exists()
        assert filepath.parent.exists()

    def test_save_checkpoint_complete(self, tmp_path):
        """Test saving complete checkpoint with all components."""
        model = DummyModel()
        optimizer = Adam(model.parameters())
        filepath = tmp_path / "checkpoint.pth"

        save_checkpoint(
            model,
            optimizer=optimizer,
            epoch=50,
            metrics={"psnr": 28.5},
            config={"model": "unet"},
            filepath=filepath,
            best_metric=30.0,
        )

        checkpoint = torch.load(filepath)
        assert all(
            key in checkpoint
            for key in [
                "model_state_dict",
                "optimizer_state_dict",
                "epoch",
                "metrics",
                "config",
                "best_metric",
            ]
        )


class TestLoadCheckpoint:
    """Tests for load_checkpoint function."""

    def test_load_checkpoint_basic(self, tmp_path):
        """Test loading basic checkpoint."""
        model = DummyModel()
        filepath = tmp_path / "checkpoint.pth"

        # Save first
        save_checkpoint(model, epoch=10, filepath=filepath)

        # Load
        checkpoint = load_checkpoint(filepath)

        assert isinstance(checkpoint, dict)
        assert "model_state_dict" in checkpoint
        assert checkpoint["epoch"] == 10

    def test_load_checkpoint_into_model(self, tmp_path):
        """Test loading checkpoint and restoring model weights."""
        model1 = DummyModel()
        model2 = DummyModel()

        # Modify model1 weights
        with torch.no_grad():
            model1.fc.weight.fill_(1.0)

        filepath = tmp_path / "checkpoint.pth"
        save_checkpoint(model1, filepath=filepath)

        # Load into model2
        load_checkpoint(filepath, model=model2)

        # Weights should match
        assert torch.allclose(model1.fc.weight, model2.fc.weight)

    def test_load_checkpoint_into_optimizer(self, tmp_path):
        """Test loading checkpoint and restoring optimizer state."""
        model = DummyModel()
        optimizer1 = Adam(model.parameters(), lr=0.001)
        optimizer2 = Adam(model.parameters(), lr=0.01)

        # Do a step to create state
        loss = model(torch.randn(1, 10)).sum()
        loss.backward()
        optimizer1.step()

        filepath = tmp_path / "checkpoint.pth"
        save_checkpoint(model, optimizer=optimizer1, filepath=filepath)

        # Load into optimizer2
        load_checkpoint(filepath, model=model, optimizer=optimizer2)

        # Optimizer states should match
        assert (
            optimizer1.state_dict()["param_groups"][0]["lr"]
            == optimizer2.state_dict()["param_groups"][0]["lr"]
        )

    def test_load_checkpoint_map_location(self, tmp_path):
        """Test loading checkpoint with map_location."""
        model = DummyModel()
        filepath = tmp_path / "checkpoint.pth"

        save_checkpoint(model, filepath=filepath)

        # Load to CPU explicitly
        checkpoint = load_checkpoint(filepath, map_location="cpu")

        assert isinstance(checkpoint, dict)

    def test_load_checkpoint_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for missing file."""
        filepath = tmp_path / "nonexistent.pth"

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            load_checkpoint(filepath)

    def test_load_checkpoint_strict_mode(self, tmp_path):
        """Test loading with strict mode."""
        model = DummyModel()
        filepath = tmp_path / "checkpoint.pth"

        save_checkpoint(model, filepath=filepath)

        # Load with strict=True (should succeed)
        load_checkpoint(filepath, model=model, strict=True)

        # Create model with different architecture
        class DifferentModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 3)  # Different output size

        different_model = DifferentModel()

        # Load with strict=True (should fail)
        with pytest.raises(RuntimeError):
            load_checkpoint(filepath, model=different_model, strict=True)

        # Load with strict=False (should work but with warnings)
        different_model = DummyModel()
        load_checkpoint(filepath, model=different_model, strict=False)


class TestSaveModel:
    """Tests for save_model function."""

    def test_save_model_weights_only(self, tmp_path):
        """Test saving model weights only."""
        model = DummyModel()
        filepath = tmp_path / "model.pth"

        save_model(model, filepath=filepath, save_weights_only=True)

        assert filepath.exists()

        # Load and verify it's a state dict
        state_dict = torch.load(filepath)
        assert isinstance(state_dict, dict)
        assert "fc.weight" in state_dict

    def test_save_entire_model(self, tmp_path):
        """Test saving entire model."""
        model = DummyModel()
        filepath = tmp_path / "model.pth"

        save_model(model, filepath=filepath, save_weights_only=False)

        # Load and verify it's a model instance
        loaded = torch.load(filepath, map_location="cpu", weights_only=False)
        assert isinstance(loaded, nn.Module)

    def test_save_model_creates_directory(self, tmp_path):
        """Test that directory is created."""
        model = DummyModel()
        filepath = tmp_path / "models" / "model.pth"

        save_model(model, filepath=filepath)

        assert filepath.exists()
        assert filepath.parent.exists()


class TestLoadModel:
    """Tests for load_model function."""

    def test_load_model_weights(self, tmp_path):
        """Test loading model weights into existing model."""
        model1 = DummyModel()
        model2 = DummyModel()

        # Modify model1
        with torch.no_grad():
            model1.fc.weight.fill_(2.0)

        filepath = tmp_path / "model.pth"
        save_model(model1, filepath=filepath)

        # Load into model2
        loaded_model = load_model(filepath, model=model2)

        assert torch.allclose(loaded_model.fc.weight, model1.fc.weight)

    def test_load_entire_model(self, tmp_path):
        """Test loading entire model."""
        model = DummyModel()

        filepath = tmp_path / "model.pth"
        save_model(model, filepath=filepath, save_weights_only=False)

        # Load without providing model
        loaded_model = load_model(filepath)

        assert isinstance(loaded_model, DummyModel)

    def test_load_model_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised."""
        filepath = tmp_path / "nonexistent.pth"

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            load_model(filepath)

    def test_load_model_map_location(self, tmp_path):
        """Test loading with map_location."""
        model = DummyModel()
        filepath = tmp_path / "model.pth"

        save_model(model, filepath=filepath)

        # Load to CPU
        loaded_model = load_model(filepath, model=DummyModel(), map_location="cpu")

        assert isinstance(loaded_model, nn.Module)


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    def test_init(self, tmp_path):
        """Test CheckpointManager initialization."""
        manager = CheckpointManager(tmp_path, max_checkpoints=5, mode="max")

        assert manager.checkpoint_dir == tmp_path
        assert manager.max_checkpoints == 5
        assert manager.mode == "max"
        assert len(manager.checkpoints) == 0

    def test_init_invalid_mode(self, tmp_path):
        """Test that invalid mode raises error."""
        with pytest.raises(ValueError, match="mode must be 'min' or 'max'"):
            CheckpointManager(tmp_path, mode="invalid")

    def test_save_checkpoint_without_metric(self, tmp_path):
        """Test saving checkpoint without metric value."""
        manager = CheckpointManager(tmp_path)
        model = DummyModel()

        filepath = manager.save_checkpoint(model, epoch=10)

        assert filepath is not None
        assert filepath.exists()
        assert "epoch_10" in filepath.name

    def test_save_checkpoint_with_metric(self, tmp_path):
        """Test saving checkpoint with metric value."""
        manager = CheckpointManager(tmp_path)
        model = DummyModel()

        filepath = manager.save_checkpoint(
            model, epoch=10, metric_value=28.5, metric_name="psnr"
        )

        assert filepath is not None
        assert "epoch_10" in filepath.name
        assert "psnr" in filepath.name
        assert "28.5" in filepath.name

    def test_cleanup_max_checkpoints(self, tmp_path):
        """Test that manager keeps only max_checkpoints."""
        manager = CheckpointManager(tmp_path, max_checkpoints=3, mode="max")
        model = DummyModel()

        # Save 5 checkpoints with increasing metrics
        for i in range(5):
            manager.save_checkpoint(
                model, epoch=i, metric_value=float(i), metric_name="psnr"
            )

        # Should only have 3 checkpoints (best ones)
        assert len(manager.checkpoints) == 3

        # Check that best checkpoints are kept
        metrics = list(manager.checkpoints.values())
        assert min(metrics) >= 2.0  # Worst kept checkpoint should be >= 2.0

    def test_cleanup_mode_min(self, tmp_path):
        """Test cleanup with mode='min' (keep lowest metrics)."""
        manager = CheckpointManager(tmp_path, max_checkpoints=2, mode="min")
        model = DummyModel()

        # Save checkpoints with different metrics
        manager.save_checkpoint(model, epoch=0, metric_value=5.0, metric_name="loss")
        manager.save_checkpoint(model, epoch=1, metric_value=3.0, metric_name="loss")
        manager.save_checkpoint(model, epoch=2, metric_value=4.0, metric_name="loss")

        # Should keep the 2 lowest (3.0 and 4.0)
        metrics = list(manager.checkpoints.values())
        assert len(metrics) == 2
        assert 5.0 not in metrics
        assert 3.0 in metrics
        assert 4.0 in metrics

    def test_get_best_checkpoint_max_mode(self, tmp_path):
        """Test getting best checkpoint in max mode."""
        manager = CheckpointManager(tmp_path, mode="max")
        model = DummyModel()

        manager.save_checkpoint(model, epoch=0, metric_value=25.0, metric_name="psnr")
        manager.save_checkpoint(model, epoch=1, metric_value=28.0, metric_name="psnr")
        manager.save_checkpoint(model, epoch=2, metric_value=26.0, metric_name="psnr")

        best = manager.get_best_checkpoint()

        assert best is not None
        assert "28.0" in best.name

    def test_get_best_checkpoint_min_mode(self, tmp_path):
        """Test getting best checkpoint in min mode."""
        manager = CheckpointManager(tmp_path, mode="min")
        model = DummyModel()

        manager.save_checkpoint(model, epoch=0, metric_value=0.5, metric_name="loss")
        manager.save_checkpoint(model, epoch=1, metric_value=0.3, metric_name="loss")
        manager.save_checkpoint(model, epoch=2, metric_value=0.4, metric_name="loss")

        best = manager.get_best_checkpoint()

        assert best is not None
        assert "0.3" in best.name

    def test_get_best_checkpoint_empty(self, tmp_path):
        """Test getting best checkpoint when none saved."""
        manager = CheckpointManager(tmp_path)

        best = manager.get_best_checkpoint()

        assert best is None

    def test_get_latest_checkpoint(self, tmp_path):
        """Test getting latest checkpoint."""
        manager = CheckpointManager(tmp_path)
        model = DummyModel()

        # Save multiple checkpoints
        manager.save_checkpoint(model, epoch=0, metric_value=25.0)
        import time

        time.sleep(0.1)  # Ensure different timestamps
        manager.save_checkpoint(model, epoch=1, metric_value=26.0)
        time.sleep(0.1)
        filepath3 = manager.save_checkpoint(model, epoch=2, metric_value=27.0)

        latest = manager.get_latest_checkpoint()

        assert latest is not None
        # Latest should be the most recently saved (epoch 2)
        assert latest == filepath3

    def test_get_latest_checkpoint_empty(self, tmp_path):
        """Test getting latest checkpoint when none saved."""
        manager = CheckpointManager(tmp_path)

        latest = manager.get_latest_checkpoint()

        assert latest is None


class TestCheckpointIntegration:
    """Integration tests for checkpoint utilities."""

    def test_save_and_load_workflow(self, tmp_path):
        """Test complete save and load workflow."""
        # Create and train model
        model = DummyModel()
        optimizer = Adam(model.parameters())

        # Simulate training step
        x = torch.randn(1, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        # Save checkpoint
        filepath = tmp_path / "checkpoint.pth"
        save_checkpoint(
            model,
            optimizer=optimizer,
            epoch=50,
            metrics={"psnr": 28.5},
            filepath=filepath,
        )

        # Create new model and optimizer
        new_model = DummyModel()
        new_optimizer = Adam(new_model.parameters())

        # Load checkpoint
        checkpoint = load_checkpoint(filepath, model=new_model, optimizer=new_optimizer)

        # Verify state is restored
        assert checkpoint["epoch"] == 50
        assert checkpoint["metrics"]["psnr"] == 28.5
        assert torch.allclose(model.fc.weight, new_model.fc.weight)

    def test_checkpoint_manager_workflow(self, tmp_path):
        """Test CheckpointManager workflow."""
        manager = CheckpointManager(tmp_path, max_checkpoints=3, mode="max")
        model = DummyModel()
        optimizer = Adam(model.parameters())

        # Simulate training loop
        best_metric = 0.0
        for epoch in range(10):
            metric = 20.0 + epoch  # Increasing metric

            filepath = manager.save_checkpoint(
                model,
                optimizer=optimizer,
                epoch=epoch,
                metric_value=metric,
                metric_name="psnr",
            )

            if metric > best_metric:
                best_metric = metric

        # Manager should have kept 3 best checkpoints
        assert len(manager.checkpoints) == 3
        assert best_metric in manager.checkpoints.values()
        assert sorted(manager.checkpoints.values()) == [27.0, 28.0, 29.0]
        assert filepath.exists()

        # Get best checkpoint
        best_checkpoint_path = manager.get_best_checkpoint()
        checkpoint = load_checkpoint(best_checkpoint_path)

        # Best checkpoint should have epoch 9 (metric 29.0)
        assert checkpoint["epoch"] == 9
        assert checkpoint["metrics"]["psnr"] == 29.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
