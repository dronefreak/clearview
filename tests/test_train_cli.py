"""Tests for train.py's argument handling: --config layering, scheduler setup,
and wiring of model-construction flags (--backbone/--pretrained).

These call parse_args()/setup_scheduler()/setup_model() directly, without
running a training loop.
"""

from pathlib import Path
from typing import List

import pytest
import torch

from clearview.scripts import train as train_script
from clearview.scripts.train import parse_args, setup_model, setup_scheduler

REQUIRED: List[str] = ["--data-dir", "data", "--output-dir", "out"]


class TestConfigFile:
    """Precedence: built-in defaults < --config file < explicit CLI flags."""

    def test_config_overrides_defaults(self, tmp_path: Path) -> None:
        """Values in the YAML file replace argparse defaults."""
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("batch_size: 32\nlr: 0.0002\nscheduler: cosine\n")
        args = parse_args([*REQUIRED, "--config", str(cfg)])
        assert args.batch_size == 32
        assert args.lr == pytest.approx(2e-4)
        assert args.scheduler == "cosine"

    def test_cli_overrides_config(self, tmp_path: Path) -> None:
        """An option given on the command line wins over the file."""
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("batch_size: 32\n")
        args = parse_args([*REQUIRED, "--config", str(cfg), "--batch-size", "4"])
        assert args.batch_size == 4

    def test_dashed_keys_and_required_from_config(self, tmp_path: Path) -> None:
        """Dashed keys work, and required options can come from the file."""
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("data-dir: /d\noutput-dir: /o\nval-batch-size: 3\n")
        args = parse_args(["--config", str(cfg)])
        assert (args.data_dir, args.output_dir, args.val_batch_size) == ("/d", "/o", 3)

    def test_unknown_key_is_an_error(self, tmp_path: Path) -> None:
        """A typo in the config file fails loudly instead of being ignored."""
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("batch_sise: 32\n")
        with pytest.raises(SystemExit):
            parse_args([*REQUIRED, "--config", str(cfg)])


class TestSchedulers:
    """Every --scheduler choice must construct on the installed torch."""

    @pytest.mark.parametrize("name", ["plateau", "cosine", "step", "multistep"])
    @pytest.mark.parametrize("warmup", [0, 2])
    def test_scheduler_constructs(self, name: str, warmup: int) -> None:
        """setup_scheduler() builds each choice without raising."""
        args = parse_args(
            [*REQUIRED, "--scheduler", name, "--warmup-epochs", str(warmup)]
        )
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
        scheduler, monitor = setup_scheduler(args, optimizer)
        assert scheduler is not None
        assert monitor == ("val_loss" if name == "plateau" else None)


class TestModelFlags:
    """--backbone/--pretrained reach the ResNet-UNet constructor."""

    @pytest.fixture
    def captured(self, monkeypatch: pytest.MonkeyPatch) -> dict:
        """Replace get_model with a recorder returning a tiny real model."""
        calls: dict = {}
        real_get_model = train_script.get_model

        def fake_get_model(name: str, **kwargs):
            calls["name"], calls["kwargs"] = name, kwargs
            return real_get_model("unet", in_channels=3, out_channels=3)

        monkeypatch.setattr(train_script, "get_model", fake_get_model)
        return calls

    def test_resnet_unet_gets_backbone_and_pretrained(self, captured: dict) -> None:
        """The generic resnet_unet receives both flags."""
        args = parse_args(
            [
                *REQUIRED,
                "--model",
                "resnet_unet",
                "--backbone",
                "resnet50",
                "--no-pretrained",
            ]
        )
        setup_model(args)
        assert captured["kwargs"]["backbone"] == "resnet50"
        assert captured["kwargs"]["pretrained"] is False

    def test_named_resnet_gets_pretrained_only(self, captured: dict) -> None:
        """resnet18_unet fixes its backbone by name, so only pretrained is passed."""
        args = parse_args([*REQUIRED, "--model", "resnet18_unet"])
        setup_model(args)
        assert captured["kwargs"]["pretrained"] is True
        assert "backbone" not in captured["kwargs"]

    def test_non_resnet_gets_neither(self, captured: dict) -> None:
        """Other architectures don't receive ResNet-only kwargs."""
        args = parse_args([*REQUIRED, "--model", "nafnet_small"])
        setup_model(args)
        assert "pretrained" not in captured["kwargs"]
        assert "backbone" not in captured["kwargs"]
