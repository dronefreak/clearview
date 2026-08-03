"""Tests for the evaluation script's metric handling.

Covers ``_filter_available_metrics()`` (graceful skipping of metrics whose
optional dependency is missing) and the ``evaluate()`` loop itself, using a
tiny synthetic model/dataset so the tests run quickly without any real
pretrained weights.
"""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from clearview.api import DerainingModel
from clearview.scripts.evaluate import _filter_available_metrics, evaluate


def _tiny_model() -> nn.Module:
    """Create a tiny convolutional model for fast evaluation tests."""
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 3, 3, padding=1),
        nn.Sigmoid(),
    )


def _tiny_loader(num_samples: int = 4, batch_size: int = 2) -> DataLoader:
    """Create a tiny synthetic rainy/clean DataLoader."""
    rainy = torch.rand(num_samples, 3, 64, 64)
    clean = torch.rand(num_samples, 3, 64, 64)
    dataset = TensorDataset(rainy, clean)
    return DataLoader(dataset, batch_size=batch_size)


class TestFilterAvailableMetrics:
    """Tests for _filter_available_metrics()."""

    def test_always_available_metrics_pass_through(self) -> None:
        """Test that dependency-free metrics are never filtered out."""
        metrics = ["psnr", "ssim", "mae", "mse", "rain_removal_rate"]
        result = _filter_available_metrics(metrics)
        assert result == metrics

    def test_optional_metrics_kept_when_dependency_installed(self) -> None:
        """Test lpips/dists/brisque survive filtering when piq/lpips exist."""
        pytest.importorskip("piq")
        pytest.importorskip("lpips")
        metrics = ["psnr", "lpips", "dists", "brisque"]
        result = _filter_available_metrics(metrics)
        assert result == metrics

    def test_unknown_dependency_module_is_skipped(self, monkeypatch) -> None:
        """Test that a missing optional dependency drops just that metric."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "piq":
                raise ImportError("simulated missing piq")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        result = _filter_available_metrics(["psnr", "dists", "mae"])

        assert result == ["psnr", "mae"]


class TestEvaluateLoop:
    """Tests for the evaluate() function using a tiny model/dataset."""

    def test_evaluate_basic_metrics(self, tmp_path: Path) -> None:
        """Test evaluate() computes requested reference-based metrics."""
        model = DerainingModel(_tiny_model(), device="cpu")
        loader = _tiny_loader()

        final_metrics, summary, metric_values, vis_samples, fid_score = evaluate(
            model=model,
            dataloader=loader,
            metrics=["psnr", "ssim", "mae", "mse"],
            output_dir=tmp_path,
            num_vis=2,
        )

        assert set(final_metrics.keys()) == {"psnr", "ssim", "mae", "mse"}
        for metric in final_metrics:
            assert metric in summary
        assert len(vis_samples) == 2
        assert fid_score is None

    def test_evaluate_rain_removal_rate_uses_rainy(self, tmp_path: Path) -> None:
        """Test evaluate() wires the rainy tensor through for rain_removal_rate."""
        model = DerainingModel(_tiny_model(), device="cpu")
        loader = _tiny_loader()

        final_metrics, summary, metric_values, vis_samples, fid_score = evaluate(
            model=model,
            dataloader=loader,
            metrics=["psnr", "rain_removal_rate"],
            output_dir=tmp_path,
            num_vis=2,
        )

        assert "rain_removal_rate" in final_metrics
        assert isinstance(final_metrics["rain_removal_rate"], float)

    def test_evaluate_with_fid(self, tmp_path: Path) -> None:
        """Test evaluate() computes an overall FID score when requested."""
        pytest.importorskip("piq")
        model = DerainingModel(_tiny_model(), device="cpu")
        loader = _tiny_loader()

        final_metrics, summary, metric_values, vis_samples, fid_score = evaluate(
            model=model,
            dataloader=loader,
            metrics=["psnr"],
            output_dir=tmp_path,
            num_vis=1,
            compute_fid_score=True,
        )

        assert fid_score is not None
        assert isinstance(fid_score, float)
