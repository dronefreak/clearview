"""Unit tests for exponential moving average (EMA) of model weights."""

import pytest
import torch
import torch.nn as nn

from clearview.training.ema import ExponentialMovingAverage


def _tiny_model() -> nn.Module:
    """Create a tiny model for EMA testing."""
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4))


class TestExponentialMovingAverage:
    """Tests for ExponentialMovingAverage."""

    def test_initialization_copies_weights(self) -> None:
        """Test that shadow weights are initialized as a copy of the model."""
        model = _tiny_model()
        ema = ExponentialMovingAverage(model, decay=0.99)

        for name, param in model.state_dict().items():
            assert torch.allclose(ema.shadow[name], param)

    def test_invalid_decay_raises(self) -> None:
        """Test that an out-of-range decay raises ValueError."""
        model = _tiny_model()
        with pytest.raises(ValueError):
            ExponentialMovingAverage(model, decay=1.5)

    def test_shadow_is_independent_copy(self) -> None:
        """Test that mutating the model doesn't affect the shadow weights."""
        model = _tiny_model()
        ema = ExponentialMovingAverage(model, decay=0.99)

        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)

        for name, param in model.named_parameters():
            assert not torch.allclose(ema.shadow[name], param)

    def test_update_moves_toward_model_weights(self) -> None:
        """Test that update() moves shadow weights toward the model's weights."""
        model = _tiny_model()
        ema = ExponentialMovingAverage(model, decay=0.9, use_warmup=False)
        initial_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)

        ema.update(model)

        for name, param in model.named_parameters():
            # Shadow should have moved from its initial value toward the
            # (now perturbed) model weights, but not fully reached them.
            assert not torch.allclose(ema.shadow[name], initial_shadow[name])
            assert not torch.allclose(ema.shadow[name], param)

    def test_update_after_step_skips_early_updates(self) -> None:
        """Test that updates before update_after_step are ignored."""
        model = _tiny_model()
        ema = ExponentialMovingAverage(model, decay=0.5, update_after_step=3)
        initial_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        for _ in range(3):
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(1.0)
            ema.update(model)

        for name in ema.shadow:
            assert torch.allclose(ema.shadow[name], initial_shadow[name])

    def test_apply_shadow_and_restore_roundtrip(self) -> None:
        """Test that apply_shadow + restore returns the model to its original state."""
        model = _tiny_model()
        ema = ExponentialMovingAverage(model, decay=0.9)

        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)
        ema.update(model)

        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        backup = ema.apply_shadow(model)
        for name, param in model.state_dict().items():
            assert torch.allclose(param, ema.shadow[name])

        ema.restore(model, backup)
        for name, param in model.state_dict().items():
            assert torch.allclose(param, original_state[name])

    def test_state_dict_roundtrip(self) -> None:
        """Test saving and loading EMA state."""
        model = _tiny_model()
        ema = ExponentialMovingAverage(model, decay=0.9)

        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)
        ema.update(model)

        state = ema.state_dict()

        new_model = _tiny_model()
        new_ema = ExponentialMovingAverage(new_model, decay=0.99)
        new_ema.load_state_dict(state)

        assert new_ema.num_updates == ema.num_updates
        assert new_ema.decay == ema.decay
        for name in ema.shadow:
            assert torch.allclose(new_ema.shadow[name], ema.shadow[name])

    def test_warmup_ramps_decay(self) -> None:
        """Test that use_warmup produces a smaller effective decay early on."""
        model = _tiny_model()
        ema = ExponentialMovingAverage(model, decay=0.999, use_warmup=True)

        ema.num_updates = 1
        early_decay = ema._current_decay()
        ema.num_updates = 10_000
        late_decay = ema._current_decay()

        assert early_decay < late_decay
        assert late_decay == pytest.approx(0.999, abs=1e-3)
