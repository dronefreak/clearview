"""Integration tests for the Trainer class, focused on newer features:
EMA weight tracking, gradient accumulation, and LR scheduler integration.

These tests use a tiny synthetic dataset and a minimal model to keep runtime
low while still exercising the real training loop end-to-end.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from clearview.training.callbacks import LearningRateScheduler
from clearview.training.ema import ExponentialMovingAverage
from clearview.training.schedulers import WarmupCosineScheduler
from clearview.training.trainer import Trainer


def _tiny_model() -> nn.Module:
    """Create a tiny convolutional model for fast training tests."""
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 3, 3, padding=1),
        nn.Sigmoid(),
    )


def _tiny_loader(num_samples: int = 8, batch_size: int = 2) -> DataLoader:
    """Create a tiny synthetic rainy/clean DataLoader."""
    rainy = torch.rand(num_samples, 3, 16, 16)
    clean = torch.rand(num_samples, 3, 16, 16)
    dataset = TensorDataset(rainy, clean)
    return DataLoader(dataset, batch_size=batch_size)


class TestTrainerEMA:
    """Tests for EMA integration in Trainer."""

    def test_ema_initialized_when_enabled(self) -> None:
        """Test that trainer.ema is created when use_ema=True."""
        model = _tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.L1Loss()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            use_ema=True,
        )

        assert isinstance(trainer.ema, ExponentialMovingAverage)

    def test_ema_not_initialized_by_default(self) -> None:
        """Test that trainer.ema is None when use_ema=False (default)."""
        model = _tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.L1Loss()

        trainer = Trainer(
            model=model, optimizer=optimizer, loss_fn=loss_fn, device="cpu"
        )

        assert trainer.ema is None

    def test_ema_updates_after_training(self) -> None:
        """Test that EMA shadow weights change after training epochs."""
        model = _tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = nn.L1Loss()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            use_ema=True,
            ema_decay=0.9,
        )

        initial_shadow = {k: v.clone() for k, v in trainer.ema.shadow.items()}
        train_loader = _tiny_loader()
        trainer.fit(train_loader, epochs=2)

        changed = any(
            not torch.allclose(initial_shadow[k], trainer.ema.shadow[k])
            for k in initial_shadow
        )
        assert changed

    def test_fit_with_ema_and_validation(self) -> None:
        """Test a full fit() call with EMA + validation runs without error."""
        model = _tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.L1Loss()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            use_ema=True,
        )

        train_loader = _tiny_loader()
        val_loader = _tiny_loader()
        history = trainer.fit(train_loader, val_loader, epochs=2)

        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2

    def test_checkpoint_roundtrip_with_ema(self, tmp_path: Path) -> None:
        """Test that EMA state survives a save/load checkpoint roundtrip."""
        model = _tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.L1Loss()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            use_ema=True,
        )
        trainer.fit(_tiny_loader(), epochs=1)

        ckpt_path = tmp_path / "ckpt.pt"
        trainer.save_checkpoint(ckpt_path)

        new_model = _tiny_model()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
        new_trainer = Trainer(
            model=new_model,
            optimizer=new_optimizer,
            loss_fn=loss_fn,
            device="cpu",
            use_ema=True,
        )
        new_trainer.load_checkpoint(ckpt_path)

        for name in trainer.ema.shadow:
            assert torch.allclose(
                trainer.ema.shadow[name], new_trainer.ema.shadow[name]
            )


class TestTrainerSchedulerIntegration:
    """Tests for WarmupCosineScheduler integration via LearningRateScheduler callback."""

    def test_scheduler_steps_during_fit(self) -> None:
        """Test that the LR scheduler is stepped once per epoch during fit()."""
        model = _tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        loss_fn = nn.L1Loss()

        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=1, total_epochs=5)
        callback = LearningRateScheduler(scheduler, verbose=0)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            callbacks=[callback],
        )

        train_loader = _tiny_loader()
        trainer.fit(train_loader, epochs=3)

        # After 3 epoch-end steps, scheduler.last_epoch should equal 3
        assert scheduler.last_epoch == 3


class TestTrainerGradientAccumulation:
    """Tests for gradient accumulation combined with EMA (regression guard)."""

    def test_accumulation_with_ema_only_updates_on_step(self) -> None:
        """Test that EMA updates only occur on optimizer steps, not every batch."""
        model = _tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.L1Loss()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            use_ema=True,
            accumulation_steps=4,
        )

        # 8 samples, batch_size=2 -> 4 batches; with accumulation_steps=4,
        # optimizer.step() (and thus ema.update()) should fire once.
        train_loader = _tiny_loader(num_samples=8, batch_size=2)
        trainer.train_epoch(train_loader)

        assert trainer.ema.num_updates == 1


class TestTrainerCompile:
    """Tests for the torch.compile() Trainer option."""

    def test_compile_disabled_by_default(self) -> None:
        """Test that compile_model=False (default) leaves model uncompiled."""
        model = _tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.L1Loss()

        trainer = Trainer(
            model=model, optimizer=optimizer, loss_fn=loss_fn, device="cpu"
        )

        assert trainer.compiled is False
        assert trainer.model is model

    def test_compile_enabled_wraps_model(self) -> None:
        """Test that compile_model=True wraps the model and sets trainer.compiled."""
        model = _tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.L1Loss()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            compile_model=True,
        )

        assert trainer.compiled is True
        # raw_model should unwrap back to the original module
        assert trainer.raw_model is model

    def test_compile_trains_without_error(self) -> None:
        """Test that a compiled model can still complete a training epoch."""
        model = _tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.L1Loss()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            compile_model=True,
        )

        history = trainer.train_epoch(_tiny_loader())

        assert "loss" in history

    def test_compile_checkpoint_keys_are_uncompiled(self, tmp_path: Path) -> None:
        """Test that checkpoints saved from a compiled model use plain keys.

        This guards against torch.compile()'s ``_orig_mod.`` state_dict key
        prefix leaking into checkpoints, which would break loading into a
        non-compiled model.
        """
        model = _tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.L1Loss()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            compile_model=True,
        )
        trainer.train_epoch(_tiny_loader())

        ckpt_path = tmp_path / "compiled_ckpt.pt"
        trainer.save_checkpoint(ckpt_path)

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        assert not any(
            key.startswith("_orig_mod.") for key in checkpoint["model_state_dict"]
        )

        # Loading into a fresh, non-compiled trainer should work without error.
        new_model = _tiny_model()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
        new_trainer = Trainer(
            model=new_model, optimizer=new_optimizer, loss_fn=loss_fn, device="cpu"
        )
        new_trainer.load_checkpoint(ckpt_path)
