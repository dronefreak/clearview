"""Tests for the inference script's no-reference quality metrics support.

Since inference has no ground-truth clean image, only no-reference metrics
(BRISQUE) can be reported. These tests exercise
``compute_output_quality_metrics()`` and ``process_single_image()`` with a
tiny synthetic model and on-disk image, without invoking the full CLI.
"""

from pathlib import Path

import numpy as np
import pytest
import torch.nn as nn
from PIL import Image

from clearview.api import DerainingModel
from clearview.scripts.inference import (
    compute_output_quality_metrics,
    process_single_image,
)


def _tiny_model() -> nn.Module:
    """Create a tiny convolutional model for fast inference tests."""
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 3, 3, padding=1),
        nn.Sigmoid(),
    )


def _write_dummy_image(path: Path, size: int = 96) -> None:
    """Write a random RGB image to disk for testing."""
    arr = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


class TestComputeOutputQualityMetrics:
    """Tests for compute_output_quality_metrics()."""

    def test_returns_brisque_when_piq_available(self, tmp_path: Path) -> None:
        """Test that BRISQUE is computed for a valid image on disk."""
        pytest.importorskip("piq")
        img_path = tmp_path / "out.png"
        _write_dummy_image(img_path)

        metrics = compute_output_quality_metrics(img_path)

        assert "brisque" in metrics
        assert isinstance(metrics["brisque"], float)

    def test_returns_empty_dict_without_piq(self, tmp_path: Path, monkeypatch) -> None:
        """Test graceful fallback (empty dict) when piq isn't installed."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "piq":
                raise ImportError("simulated missing piq")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        img_path = tmp_path / "out.png"
        _write_dummy_image(img_path)

        metrics = compute_output_quality_metrics(img_path)

        assert metrics == {}


class TestProcessSingleImageQualityMetrics:
    """Tests for process_single_image()'s report_quality_metrics option."""

    def test_quality_metrics_none_by_default(self, tmp_path: Path) -> None:
        """Test that quality metrics are not computed unless requested."""
        model = DerainingModel(_tiny_model(), device="cpu")
        input_path = tmp_path / "rainy.png"
        _write_dummy_image(input_path)
        output_path = tmp_path / "derained.png"

        _, quality_metrics = process_single_image(
            model=model, input_path=input_path, output_path=output_path
        )

        assert quality_metrics is None
        assert output_path.exists()

    def test_quality_metrics_computed_when_requested(self, tmp_path: Path) -> None:
        """Test that report_quality_metrics=True returns a metrics dict."""
        pytest.importorskip("piq")
        model = DerainingModel(_tiny_model(), device="cpu")
        input_path = tmp_path / "rainy.png"
        _write_dummy_image(input_path)
        output_path = tmp_path / "derained.png"

        _, quality_metrics = process_single_image(
            model=model,
            input_path=input_path,
            output_path=output_path,
            report_quality_metrics=True,
        )

        assert quality_metrics is not None
        assert "brisque" in quality_metrics
