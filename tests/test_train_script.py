"""Tests for the training script's flexible loss-selection support.

These tests exercise ``setup_loss()`` and ``_load_loss_config_file()`` from
``clearview.scripts.train`` directly (via a minimal ``argparse.Namespace``)
without invoking the full CLI or training loop.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
import torch

from clearview.losses import CombinedLoss
from clearview.scripts.train import _load_loss_config_file, setup_loss


def _make_args(
    loss: str = "l1_l2_ssim_edge",
    loss_config: Optional[str] = None,
    loss_config_file: Optional[str] = None,
    **weight_overrides: float,
) -> argparse.Namespace:
    """Build a minimal Namespace with the fields setup_loss() reads."""
    defaults: Dict[str, Any] = {
        "loss": loss,
        "loss_config": loss_config,
        "loss_config_file": loss_config_file,
        "l1_weight": 1.0,
        "l2_weight": 1.0,
        "ssim_weight": 1.0,
        "edge_weight": 0.5,
        "vgg_weight": 0.5,
    }
    defaults.update(weight_overrides)
    return argparse.Namespace(**defaults)


class TestSetupLossPresets:
    """Tests for the existing named-preset behavior (backward compatibility)."""

    @pytest.mark.parametrize(
        "preset",
        [
            "l1",
            "l2",
            "l1_l2_ssim",
            "l1_l2_ssim_edge",
        ],
    )
    def test_presets_build_working_loss(self, preset: str) -> None:
        """Test that each preset builds a CombinedLoss that runs forward."""
        args = _make_args(loss=preset)
        loss_fn = setup_loss(args)

        assert isinstance(loss_fn, CombinedLoss)
        pred = torch.rand(2, 3, 32, 32)
        target = torch.rand(2, 3, 32, 32)
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    def test_custom_without_config_raises(self) -> None:
        """Test that --loss custom without a config source raises ValueError."""
        args = _make_args(loss="custom")

        with pytest.raises(ValueError, match="custom"):
            setup_loss(args)

    def test_unknown_preset_raises(self) -> None:
        """Test that an unrecognized --loss value raises ValueError."""
        args = _make_args(loss="not_a_real_preset")

        with pytest.raises(ValueError, match="Unknown loss"):
            setup_loss(args)


class TestSetupLossInlineConfig:
    """Tests for --loss-config (inline JSON)."""

    def test_inline_json_config_used(self) -> None:
        """Test that --loss-config builds the exact requested combination."""
        config = {"l1": {"weight": 1.0}, "ssim": {"weight": 0.5}}
        args = _make_args(loss="custom", loss_config=json.dumps(config))

        loss_fn = setup_loss(args)

        assert isinstance(loss_fn, CombinedLoss)
        assert set(loss_fn.loss_components.keys()) == {"l1", "ssim"}

    def test_inline_json_config_overrides_preset(self) -> None:
        """Test that --loss-config takes precedence even if --loss is a preset."""
        config = {"l2": {"weight": 2.0}}
        args = _make_args(loss="l1_l2_ssim_edge", loss_config=json.dumps(config))

        loss_fn = setup_loss(args)

        assert set(loss_fn.loss_components.keys()) == {"l2"}

    def test_inline_json_supports_full_registry(self) -> None:
        """Test a combination using newly-added loss names (wavelet, color)."""
        config = {
            "l1": {"weight": 1.0},
            "wavelet": {"weight": 0.1, "levels": 1},
            "color": {"weight": 0.05},
        }
        args = _make_args(loss="custom", loss_config=json.dumps(config))

        loss_fn = setup_loss(args)
        pred = torch.rand(1, 3, 32, 32, requires_grad=True)
        target = torch.rand(1, 3, 32, 32)
        loss = loss_fn(pred, target)
        loss.backward()

        assert torch.isfinite(loss)
        assert pred.grad is not None

    def test_invalid_loss_name_raises(self) -> None:
        """Test that an unknown loss name in the config raises ValueError."""
        config = {"not_a_real_loss": {"weight": 1.0}}
        args = _make_args(loss="custom", loss_config=json.dumps(config))

        with pytest.raises(ValueError, match="Unknown loss type"):
            setup_loss(args)


class TestSetupLossConfigFile:
    """Tests for --loss-config-file (JSON/YAML file)."""

    def test_json_file_used(self, tmp_path: Path) -> None:
        """Test loading a loss combination from a .json file."""
        config = {"l1": {"weight": 1.0}, "edge": {"weight": 0.5}}
        config_path = tmp_path / "loss.json"
        config_path.write_text(json.dumps(config))

        args = _make_args(loss="custom", loss_config_file=str(config_path))
        loss_fn = setup_loss(args)

        assert set(loss_fn.loss_components.keys()) == {"l1", "edge"}

    def test_yaml_file_used(self, tmp_path: Path) -> None:
        """Test loading a loss combination from a .yaml file."""
        config_path = tmp_path / "loss.yaml"
        config_path.write_text("l1:\n  weight: 1.0\nssim:\n  weight: 0.3\n")

        args = _make_args(loss="custom", loss_config_file=str(config_path))
        loss_fn = setup_loss(args)

        assert set(loss_fn.loss_components.keys()) == {"l1", "ssim"}

    def test_config_file_overrides_inline_config(self, tmp_path: Path) -> None:
        """Test that --loss-config-file takes precedence over --loss-config."""
        file_config = {"l2": {"weight": 1.0}}
        config_path = tmp_path / "loss.json"
        config_path.write_text(json.dumps(file_config))

        inline_config = {"l1": {"weight": 1.0}}
        args = _make_args(
            loss="custom",
            loss_config=json.dumps(inline_config),
            loss_config_file=str(config_path),
        )

        loss_fn = setup_loss(args)

        assert set(loss_fn.loss_components.keys()) == {"l2"}

    def test_non_mapping_file_raises(self, tmp_path: Path) -> None:
        """Test that a file whose top level isn't a mapping raises ValueError."""
        config_path = tmp_path / "loss.json"
        config_path.write_text(json.dumps(["l1", "l2"]))

        with pytest.raises(ValueError, match="top-level mapping"):
            _load_loss_config_file(str(config_path))
