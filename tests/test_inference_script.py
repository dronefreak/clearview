"""Tests for the inference script's no-reference quality metrics support.

Since inference has no ground-truth clean image, only no-reference metrics
(BRISQUE) can be reported. These tests exercise
``compute_output_quality_metrics()`` and ``process_single_image()`` with a
tiny synthetic model and on-disk image, without invoking the full CLI.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch.nn as nn
from PIL import Image

from clearview.api import DerainingModel
from clearview.scripts.inference import (
    compute_output_quality_metrics,
    is_video_file,
    process_single_image,
    process_video,
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


def _write_dummy_video(
    path: Path, num_frames: int = 5, size: int = 96, fps: float = 10.0
) -> None:
    """Write a small random-noise video to disk for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (size, size))
    for _ in range(num_frames):
        frame = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
        writer.write(frame)
    writer.release()


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


class TestIsVideoFile:
    """Tests for the video/image extension dispatch helper."""

    @pytest.mark.parametrize(
        "name", ["clip.mp4", "clip.MOV", "clip.avi", "clip.mkv", "clip.webm"]
    )
    def test_recognizes_video_extensions(self, name: str) -> None:
        assert is_video_file(Path(name)) is True

    @pytest.mark.parametrize("name", ["photo.png", "photo.jpg", "photo.bmp"])
    def test_rejects_image_extensions(self, name: str) -> None:
        assert is_video_file(Path(name)) is False


class TestProcessVideo:
    """Tests for process_video()."""

    def test_writes_output_video_with_matching_frame_count(
        self, tmp_path: Path
    ) -> None:
        """Test that every input frame produces a derained output frame."""
        model = DerainingModel(_tiny_model(), device="cpu")
        input_path = tmp_path / "rainy.mp4"
        _write_dummy_video(input_path, num_frames=5)
        output_path = tmp_path / "derained.mp4"

        inference_time, quality_metrics = process_video(
            model=model, input_path=input_path, output_path=output_path
        )

        assert output_path.exists()
        assert inference_time > 0
        assert quality_metrics is None

        cap = cv2.VideoCapture(str(output_path))
        out_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert out_frame_count == 5

    def test_save_comparison_writes_side_by_side_video(self, tmp_path: Path) -> None:
        """Test that save_comparison=True writes a second, double-width video."""
        model = DerainingModel(_tiny_model(), device="cpu")
        input_path = tmp_path / "rainy.mp4"
        _write_dummy_video(input_path, num_frames=3, size=64)
        output_path = tmp_path / "derained.mp4"

        process_video(
            model=model,
            input_path=input_path,
            output_path=output_path,
            save_comparison=True,
        )

        comparison_path = tmp_path / "derained_comparison.mp4"
        assert comparison_path.exists()

        cap = cv2.VideoCapture(str(comparison_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        assert width == 64 * 2

    def test_quality_metrics_computed_when_requested(self, tmp_path: Path) -> None:
        """Test that report_quality_metrics=True returns an averaged BRISQUE."""
        pytest.importorskip("piq")
        model = DerainingModel(_tiny_model(), device="cpu")
        input_path = tmp_path / "rainy.mp4"
        _write_dummy_video(input_path, num_frames=3)
        output_path = tmp_path / "derained.mp4"

        _, quality_metrics = process_video(
            model=model,
            input_path=input_path,
            output_path=output_path,
            report_quality_metrics=True,
        )

        assert quality_metrics is not None
        assert "brisque" in quality_metrics

    def test_raises_on_unopenable_input(self, tmp_path: Path) -> None:
        """Test that a non-video file raises a clean ValueError."""
        model = DerainingModel(_tiny_model(), device="cpu")
        bogus_path = tmp_path / "not_a_video.mp4"
        bogus_path.write_text("not a video")
        output_path = tmp_path / "derained.mp4"

        with pytest.raises(ValueError, match="Cannot open video file"):
            process_video(model=model, input_path=bogus_path, output_path=output_path)
