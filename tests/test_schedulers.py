"""Unit tests for the WarmupCosineScheduler learning rate scheduler."""

import pytest
import torch

from clearview.training.schedulers import WarmupCosineScheduler


def _make_optimizer(lr: float = 0.1) -> torch.optim.Optimizer:
    """Create a simple optimizer for scheduler testing."""
    model = torch.nn.Linear(4, 4)
    return torch.optim.SGD(model.parameters(), lr=lr)


class TestWarmupCosineScheduler:
    """Tests for WarmupCosineScheduler."""

    def test_invalid_total_epochs_raises(self) -> None:
        """Test that total_epochs <= warmup_epochs raises ValueError."""
        optimizer = _make_optimizer()
        with pytest.raises(ValueError):
            WarmupCosineScheduler(optimizer, warmup_epochs=10, total_epochs=5)

    def test_negative_warmup_raises(self) -> None:
        """Test that a negative warmup_epochs raises ValueError."""
        optimizer = _make_optimizer()
        with pytest.raises(ValueError):
            WarmupCosineScheduler(optimizer, warmup_epochs=-1, total_epochs=10)

    def test_warmup_start_lr(self) -> None:
        """Test that LR starts at warmup_start_lr on step 0."""
        optimizer = _make_optimizer(lr=0.1)
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=5, total_epochs=50, warmup_start_lr=0.0
        )
        assert scheduler.get_last_lr()[0] == pytest.approx(0.0, abs=1e-6)

    def test_warmup_reaches_base_lr(self) -> None:
        """Test that LR reaches the base LR at the end of warmup."""
        base_lr = 0.1
        optimizer = _make_optimizer(lr=base_lr)
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=5, total_epochs=50, warmup_start_lr=0.0
        )

        for _ in range(5):
            scheduler.step()

        assert scheduler.get_last_lr()[0] == pytest.approx(base_lr, abs=1e-6)

    def test_lr_increases_monotonically_during_warmup(self) -> None:
        """Test that LR increases monotonically during the warmup phase."""
        optimizer = _make_optimizer(lr=0.1)
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=10, total_epochs=100)

        lrs = [scheduler.get_last_lr()[0]]
        for _ in range(10):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        assert all(lrs[i] <= lrs[i + 1] for i in range(len(lrs) - 1))

    def test_lr_decreases_after_warmup(self) -> None:
        """Test that LR follows a cosine decay after warmup ends."""
        optimizer = _make_optimizer(lr=0.1)
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5, total_epochs=55)

        for _ in range(5):
            scheduler.step()
        post_warmup_lr = scheduler.get_last_lr()[0]

        lrs = [post_warmup_lr]
        for _ in range(50):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        assert all(lrs[i] >= lrs[i + 1] - 1e-9 for i in range(len(lrs) - 1))

    def test_lr_reaches_eta_min_at_end(self) -> None:
        """Test that LR reaches eta_min at the end of the schedule."""
        eta_min = 1e-5
        optimizer = _make_optimizer(lr=0.1)
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=2, total_epochs=12, eta_min=eta_min
        )

        for _ in range(12):
            scheduler.step()

        assert scheduler.get_last_lr()[0] == pytest.approx(eta_min, abs=1e-6)

    def test_no_warmup(self) -> None:
        """Test scheduler behavior with zero warmup epochs (pure cosine decay)."""
        base_lr = 0.1
        optimizer = _make_optimizer(lr=base_lr)
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=0, total_epochs=10)

        assert scheduler.get_last_lr()[0] == pytest.approx(base_lr, abs=1e-6)

    def test_lr_stays_at_eta_min_beyond_total_epochs(self) -> None:
        """Test that LR clamps at eta_min if stepped beyond total_epochs."""
        eta_min = 0.0
        optimizer = _make_optimizer(lr=0.1)
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=2, total_epochs=10, eta_min=eta_min
        )

        for _ in range(20):
            scheduler.step()

        assert scheduler.get_last_lr()[0] == pytest.approx(eta_min, abs=1e-6)

    def test_multiple_param_groups(self) -> None:
        """Test scheduler with multiple optimizer parameter groups."""
        model1 = torch.nn.Linear(4, 4)
        model2 = torch.nn.Linear(4, 4)
        optimizer = torch.optim.SGD(
            [
                {"params": model1.parameters(), "lr": 0.1},
                {"params": model2.parameters(), "lr": 0.01},
            ]
        )
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5, total_epochs=50)

        for _ in range(5):
            scheduler.step()

        last_lrs = scheduler.get_last_lr()
        assert last_lrs[0] == pytest.approx(0.1, abs=1e-6)
        assert last_lrs[1] == pytest.approx(0.01, abs=1e-6)
