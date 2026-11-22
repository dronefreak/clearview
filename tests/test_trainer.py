"""Integration tests for the Trainer class, focused on newer features:
EMA weight tracking, gradient accumulation, and LR scheduler integration.

These tests use a tiny synthetic dataset and a minimal model to keep runtime
low while still exercising the real training loop end-to-end.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from clearview.training.callbacks import Callback, LearningRateScheduler
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


class TestTrainerChannelsLast:
    """Tests for the channels_last memory format Trainer option."""

    def test_channels_last_disabled_by_default(self) -> None:
        """Test that channels_last=False (default) keeps contiguous format."""
        model = _tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.L1Loss()

        trainer = Trainer(
            model=model, optimizer=optimizer, loss_fn=loss_fn, device="cpu"
        )

        assert trainer.channels_last is False
        first_conv = next(trainer.model.parameters())
        assert first_conv.is_contiguous(memory_format=torch.contiguous_format)

    def test_channels_last_converts_model_weights(self) -> None:
        """Test that channels_last=True moves model params to NHWC layout."""
        model = _tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.L1Loss()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            channels_last=True,
        )

        assert trainer.channels_last is True
        conv_weight = trainer.model[0].weight
        assert conv_weight.is_contiguous(memory_format=torch.channels_last)

    def test_channels_last_trains_without_error(self) -> None:
        """Test that a channels_last trainer completes a training epoch."""
        model = _tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.L1Loss()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            channels_last=True,
        )

        history = trainer.train_epoch(_tiny_loader())
        val_history = trainer.validate_epoch(_tiny_loader())

        assert "loss" in history
        assert "loss" in val_history

    def test_to_device_helper_converts_4d_tensor(self) -> None:
        """Test the internal _to_device() helper applies channels_last to 4D tensors."""
        model = _tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.L1Loss()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            channels_last=True,
        )

        image = torch.rand(2, 3, 16, 16)
        converted = trainer._to_device(image)
        assert converted.is_contiguous(memory_format=torch.channels_last)

        # Non-4D tensors are unaffected (no memory-format concept for them).
        vector = torch.rand(8)
        converted_vector = trainer._to_device(vector)
        assert converted_vector.shape == vector.shape


class TestTrainerAMPDtype:
    """Tests for the configurable AMP dtype Trainer option."""

    def test_default_amp_dtype_is_float16(self) -> None:
        """Test that amp_dtype defaults to torch.float16."""
        model = _tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.L1Loss()

        trainer = Trainer(
            model=model, optimizer=optimizer, loss_fn=loss_fn, device="cpu"
        )

        assert trainer.amp_dtype == torch.float16

    def test_bfloat16_disables_grad_scaler(self) -> None:
        """Test that requesting bfloat16 AMP disables loss scaling."""
        model = _tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.L1Loss()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            mixed_precision=True,
            amp_dtype=torch.bfloat16,
        )

        assert trainer.amp_dtype == torch.bfloat16
        assert trainer.scaler.is_enabled() is False

    def test_float16_mixed_precision_enables_grad_scaler(self) -> None:
        """Test that float16 AMP keeps loss scaling enabled."""
        model = _tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.L1Loss()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            mixed_precision=True,
            amp_dtype=torch.float16,
        )

        assert trainer.scaler.is_enabled() is True

    def test_bfloat16_trains_without_error_on_cpu(self) -> None:
        """Test that a bfloat16 mixed-precision trainer completes an epoch on CPU."""
        model = _tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.L1Loss()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            mixed_precision=True,
            amp_dtype=torch.bfloat16,
        )

        history = trainer.train_epoch(_tiny_loader())

        assert "loss" in history
        assert not torch.isnan(torch.tensor(history["loss"]))


class _RaiseOnEpochEnd(Callback):
    """Callback that snapshots the model's weights, then simulates Ctrl+C."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.seen_state: dict = {}

    def on_epoch_end(self, epoch, logs=None) -> None:
        """Record the weights the model holds right now, then interrupt."""
        self.seen_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        raise KeyboardInterrupt


def _warmup_trainer(model: nn.Module, **trainer_kwargs) -> Trainer:
    """Trainer with a warmup+cosine schedule and best-PSNR checkpointing."""
    from clearview.training.callbacks import ModelCheckpoint

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=2, total_epochs=6)
    callbacks = [
        LearningRateScheduler(scheduler, verbose=0),
        ModelCheckpoint(
            trainer_kwargs.pop("best_path"), monitor="val_psnr", mode="max", verbose=0
        ),
    ]
    return Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.L1Loss(),
        device="cpu",
        callbacks=callbacks,
        **trainer_kwargs,
    )


class TestTrainerResume:
    """Resuming must continue the run, not restart its schedule/trackers."""

    def test_resume_restores_lr_schedule_and_best_metric(self, tmp_path: Path) -> None:
        """LR scheduler position and ModelCheckpoint's best value survive a resume."""
        trainer = _warmup_trainer(_tiny_model(), best_path=tmp_path / "best.pth")
        trainer.fit(_tiny_loader(), val_loader=_tiny_loader(), epochs=3)
        trainer.save_checkpoint(tmp_path / "last.pth")

        old_sched = trainer.callbacks.callbacks[0].scheduler
        old_best = trainer.callbacks.callbacks[1].best_metric
        old_lr = trainer.optimizer.param_groups[0]["lr"]

        resumed = _warmup_trainer(_tiny_model(), best_path=tmp_path / "best.pth")
        ckpt = resumed.load_checkpoint(tmp_path / "last.pth")

        assert ckpt["epoch"] == 2  # last completed epoch, resume at 3
        new_sched = resumed.callbacks.callbacks[0].scheduler
        assert new_sched.last_epoch == old_sched.last_epoch == 3
        assert resumed.optimizer.param_groups[0]["lr"] == old_lr
        assert old_lr > 0.0  # past warmup start, not reset to warmup_start_lr
        assert resumed.callbacks.callbacks[1].best_metric == old_best

    def test_early_stopping_state_survives_resume(self) -> None:
        """EarlyStopping keeps its patience counter instead of resetting it."""
        from clearview.training.callbacks import CallbackList, EarlyStopping

        stopper = EarlyStopping(monitor="val_psnr", mode="max", patience=5)
        stopper.best_metric, stopper.wait = 30.0, 3
        saved = CallbackList([stopper]).state_dict()

        fresh = EarlyStopping(monitor="val_psnr", mode="max", patience=5)
        CallbackList([fresh]).load_state_dict(saved)
        fresh.on_train_begin()

        assert fresh.best_metric == 30.0
        assert fresh.wait == 3

    def test_mismatched_callback_state_is_skipped(self) -> None:
        """Resuming with a different callback list warns instead of crashing."""
        from clearview.training.callbacks import CallbackList, EarlyStopping

        stopper = EarlyStopping(monitor="val_psnr", mode="max")
        CallbackList([stopper]).load_state_dict(
            [{"name": "ModelCheckpoint", "state": {"best_metric": 1.0}}]
        )
        assert stopper.best_metric == float("-inf")

    def test_save_every_writes_resumable_checkpoint(self, tmp_path: Path) -> None:
        """fit(save_every=N) writes a full checkpoint every N epochs."""
        trainer = _warmup_trainer(_tiny_model(), best_path=tmp_path / "best.pth")
        last = tmp_path / "last.pth"
        trainer.fit(
            _tiny_loader(),
            val_loader=_tiny_loader(),
            epochs=2,
            save_every=2,
            last_checkpoint_path=last,
        )
        ckpt = torch.load(last)
        assert ckpt["epoch"] == 1
        assert "callbacks_state" in ckpt


class TestTrainerInterrupt:
    """Ctrl+C handling: propagate to the caller, never leave EMA weights in."""

    def test_keyboard_interrupt_propagates_and_restores_raw_weights(self) -> None:
        """An interrupt during callbacks re-raises with raw weights swapped back."""
        import pytest

        model = _tiny_model()
        interrupter = _RaiseOnEpochEnd(model)
        trainer = Trainer(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
            loss_fn=nn.L1Loss(),
            device="cpu",
            callbacks=[interrupter],
            use_ema=True,
            ema_decay=0.999,
        )

        with pytest.raises(KeyboardInterrupt):
            trainer.fit(_tiny_loader(), val_loader=_tiny_loader(), epochs=3)

        # During on_epoch_end the model held the EMA shadow weights. After
        # the interrupt it must hold the raw weights again, which differ.
        after = model.state_dict()
        assert any(
            not torch.allclose(after[k], interrupter.seen_state[k]) for k in after
        )
        assert trainer.last_completed_epoch == -1  # epoch 0 never completed

    def test_early_stopping_restores_best_weights_with_ema(self) -> None:
        """restore_best_weights is not undone by the trainer's EMA swap-back."""
        from clearview.training.callbacks import EarlyStopping

        model = _tiny_model()
        stopper = EarlyStopping(
            monitor="val_loss", mode="min", patience=1, restore_best_weights=True
        )
        # min_delta so large nothing after epoch 0 counts as improvement
        stopper.min_delta = 1e9
        trainer = Trainer(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
            loss_fn=nn.L1Loss(),
            device="cpu",
            callbacks=[stopper],
            use_ema=True,
        )
        trainer.fit(_tiny_loader(), val_loader=_tiny_loader(), epochs=5)

        assert stopper.stop_training
        assert stopper.best_weights is not None
        for k, v in model.state_dict().items():
            assert torch.allclose(v, stopper.best_weights[k])
