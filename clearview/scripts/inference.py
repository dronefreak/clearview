#!/usr/bin/env python
"""Inference script for image and video deraining."""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import torch
from tqdm import tqdm

from clearview.api import DerainingModel
from clearview.models import list_models
from clearview.utils import compute_brisque
from clearview.utils.image import numpy_to_tensor

logger = logging.getLogger(__name__)

# Recognized video extensions for --input. Anything else is treated as an
# image (matching the existing --extensions handling for directory mode).
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def is_video_file(path: Path) -> bool:
    """Return True if path's extension is a recognized video format."""
    return path.suffix.lower() in VIDEO_EXTENSIONS


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference on rainy images or video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=list_models(),
        help="Model architecture",
    )
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to model weights"
    )

    # Input/Output arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=str,
        help="Input image or video path. Video is detected by extension "
        f"({sorted(VIDEO_EXTENSIONS)}) and derained frame by frame -- there "
        "is no temporal-consistency term, so some flicker is expected.",
    )
    input_group.add_argument(
        "--input-dir", type=str, help="Input directory containing images"
    )

    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output", type=str, help="Output image or video path")
    output_group.add_argument(
        "--output-dir", type=str, help="Output directory for processed images"
    )

    # Processing arguments
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".bmp"],
        help="Valid image extensions",
    )
    parser.add_argument(
        "--save-comparison",
        action="store_true",
        help="Save side-by-side comparison (input + output)",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Process subdirectories recursively"
    )

    # Performance arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Benchmark inference speed"
    )
    parser.add_argument(
        "--report-quality-metrics",
        action="store_true",
        help="Compute and log no-reference image quality metrics (BRISQUE) "
        "on each derained output, since inference has no ground-truth "
        "clean image to compare against. Requires the optional 'piq' "
        "package. In directory mode, per-image and aggregate scores are "
        "also saved to '<output-dir>/quality_metrics.json'.",
    )

    return parser.parse_args()


def get_image_files(
    input_dir: Path, extensions: List[str], recursive: bool = False
) -> List[Path]:
    """Get list of image files from directory."""
    image_files: List[Path] = []

    if recursive:
        for ext in extensions:
            image_files.extend(input_dir.rglob(f"*{ext}"))
            image_files.extend(input_dir.rglob(f"*{ext.upper()}"))
    else:
        for ext in extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))

    return sorted(image_files)


def compute_output_quality_metrics(image_path: Path) -> Dict[str, float]:
    """Compute no-reference quality metrics for a derained output image.

    Only metrics that don't require a ground-truth clean image or a fitted
    reference model are used here (e.g. BRISQUE), since inference has no
    access to either.

    Args:
        image_path: Path to the derained output image

    Returns:
        Dictionary mapping metric name to value. Empty if the required
        optional dependency ('piq') is not installed.
    """
    import numpy as np
    from PIL import Image

    try:
        import piq  # noqa: F401
    except ImportError:
        logger.warning(
            "Skipping quality metrics: requires the optional 'piq' package "
            "(pip install piq)"
        )
        return {}

    img = np.array(Image.open(image_path).convert("RGB")).astype(np.float32) / 255.0
    img_tensor = numpy_to_tensor(img).unsqueeze(0)

    return {"brisque": compute_brisque(img_tensor)}


def process_single_image(
    model: DerainingModel,
    input_path: Path,
    output_path: Path,
    save_comparison: bool = False,
    benchmark: bool = False,
    report_quality_metrics: bool = False,
) -> Tuple[float, Optional[Dict[str, float]]]:
    """Process a single image."""
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Measure time
    start_time = time.time()

    # Process image
    if save_comparison:
        # Load input for comparison
        import numpy as np
        from PIL import Image

        rainy_img = np.array(Image.open(input_path).convert("RGB"))

        # Process
        derained_img = model.process(input_path, output_path=output_path)

        # Save comparison
        comparison_path = (
            output_path.parent / f"{output_path.stem}_comparison{output_path.suffix}"
        )

        from clearview.utils.visualization import (
            save_comparison as save_comparison_util,
        )

        rainy_tensor = numpy_to_tensor(rainy_img.astype(np.float32) / 255.0)
        derained_tensor = numpy_to_tensor(derained_img.astype(np.float32) / 255.0)

        save_comparison_util(
            rainy_tensor, derained_tensor, clean=None, save_path=comparison_path
        )
    else:
        # Just process
        model.process(input_path, output_path=output_path)

    inference_time = time.time() - start_time

    quality_metrics: Optional[Dict[str, float]] = None
    if report_quality_metrics:
        quality_metrics = compute_output_quality_metrics(output_path)

    return inference_time, quality_metrics


def process_video(
    model: DerainingModel,
    input_path: Path,
    output_path: Path,
    save_comparison: bool = False,
    benchmark: bool = False,
    report_quality_metrics: bool = False,
) -> Tuple[float, Optional[Dict[str, float]]]:
    """Process a video file, deraining it frame by frame.

    Every frame is derained independently: there is no temporal-consistency
    term (no optical-flow warping, no recurrent state), so some frame-to-frame
    flicker is expected on video input, not a bug. Output is re-encoded with
    the source's original fps/resolution.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file '{input_path}'")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # may be 0/unreliable

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video writer for '{output_path}'")

    comparison_path: Optional[Path] = None
    comparison_writer = None
    if save_comparison:
        comparison_path = (
            output_path.parent / f"{output_path.stem}_comparison{output_path.suffix}"
        )
        comparison_writer = cv2.VideoWriter(
            str(comparison_path), fourcc, fps, (width * 2, height)
        )

    # Quality metrics require the optional 'piq' package (see
    # compute_output_quality_metrics). Check once up front rather than
    # letting every frame raise the same ImportError.
    if report_quality_metrics:
        try:
            import piq  # noqa: F401
        except ImportError:
            logger.warning(
                "Skipping quality metrics: requires the optional 'piq' "
                "package (pip install piq)"
            )
            report_quality_metrics = False

    frame_brisque_scores: List[float] = []
    start_time = time.time()
    pbar = tqdm(
        total=frame_count if frame_count > 0 else None,
        desc=f"Deraining {input_path.name}",
    )

    frame_idx = 0
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Long-running video processing accumulates CUDA memory
            # fragmentation over hundreds of frames even at a constant
            # resolution (observed in practice: a multi-minute run OOM'd on
            # a completely ordinary frame with ~9.7GB already "in use" on a
            # 12GB card). Periodic cache clearing bounds that; a per-frame
            # CPU fallback (same pattern as clearview.scripts.evaluate)
            # covers the case where a frame is still too large even so,
            # rather than aborting the whole video.
            if frame_idx > 0 and frame_idx % 50 == 0:
                torch.cuda.empty_cache()
            try:
                derained_rgb = model.process(frame_rgb)
            except torch.OutOfMemoryError:
                logger.warning(
                    f"CUDA OOM on frame {frame_idx} (shape {frame_rgb.shape}); "
                    "retrying this frame on CPU"
                )
                torch.cuda.empty_cache()
                original_device = model.device
                model.to("cpu")
                try:
                    derained_rgb = model.process(frame_rgb)
                finally:
                    model.to(original_device)
                    torch.cuda.empty_cache()

            if derained_rgb.shape[:2] != (height, width):
                derained_rgb = cv2.resize(derained_rgb, (width, height))

            derained_bgr = cv2.cvtColor(derained_rgb, cv2.COLOR_RGB2BGR)
            writer.write(derained_bgr)

            if comparison_writer is not None:
                comparison_writer.write(cv2.hconcat([frame_bgr, derained_bgr]))

            if report_quality_metrics:
                frame_tensor = numpy_to_tensor(
                    derained_rgb.astype("float32") / 255.0
                ).unsqueeze(0)
                frame_brisque_scores.append(compute_brisque(frame_tensor))

            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        writer.release()
        if comparison_writer is not None:
            comparison_writer.release()

    inference_time = time.time() - start_time

    if benchmark and frame_idx > 0:
        avg_time = inference_time / frame_idx
        logger.info(
            f"  Frames: {frame_idx}, total: {inference_time:.2f}s, "
            f"avg: {avg_time:.3f}s/frame ({1 / avg_time:.1f} fps)"
        )

    quality_metrics: Optional[Dict[str, float]] = None
    if frame_brisque_scores:
        quality_metrics = {
            "brisque": sum(frame_brisque_scores) / len(frame_brisque_scores)
        }

    return inference_time, quality_metrics


def main() -> None:
    """Main inference function."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    logger.info("=" * 80)
    logger.info("Image/Video Deraining Inference")
    logger.info("=" * 80)

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Load model
    logger.info("\nLoading model...")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Weights: {args.weights}")
    logger.info(f"  Device: {args.device}")

    model = DerainingModel.from_pretrained(
        model_name=args.model, weights=args.weights, device=args.device
    )

    logger.info("Model loaded successfully!")

    # Process single image, single video, or a directory of images
    if args.input is not None:
        input_path = Path(args.input)
        output_path = Path(args.output)

        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return

        logger.info(f"\nProcessing: {input_path}")

        if is_video_file(input_path):
            if not is_video_file(output_path):
                logger.warning(
                    f"--output '{output_path}' doesn't look like a video file "
                    f"({sorted(VIDEO_EXTENSIONS)}); writing it anyway with "
                    "an mp4-compatible codec."
                )
            inference_time, quality_metrics = process_video(
                model=model,
                input_path=input_path,
                output_path=output_path,
                save_comparison=args.save_comparison,
                benchmark=args.benchmark,
                report_quality_metrics=args.report_quality_metrics,
            )
        else:
            inference_time, quality_metrics = process_single_image(
                model=model,
                input_path=input_path,
                output_path=output_path,
                save_comparison=args.save_comparison,
                benchmark=args.benchmark,
                report_quality_metrics=args.report_quality_metrics,
            )

        logger.info(f"Output saved to: {output_path}")

        if args.benchmark:
            logger.info(f"Inference time: {inference_time:.3f}s")

        if quality_metrics:
            for name, value in quality_metrics.items():
                logger.info(f"Quality metric {name.upper()}: {value:.4f}")

        if args.save_comparison:
            comparison_path = (
                output_path.parent
                / f"{output_path.stem}_comparison{output_path.suffix}"
            )
            logger.info(f"Comparison saved to: {comparison_path}")

    else:
        # Directory
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)

        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return

        # Get image files
        logger.info(f"\nScanning directory: {input_dir}")
        image_files = get_image_files(
            input_dir, args.extensions, recursive=args.recursive
        )

        if not image_files:
            logger.error(f"No images found in {input_dir}")
            logger.error(f"Looking for extensions: {args.extensions}")
            return

        logger.info(f"Found {len(image_files)} images")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process images
        total_time = 0.0
        per_image_quality: Dict[str, Dict[str, float]] = {}

        pbar = tqdm(image_files, desc="Processing images")

        for img_path in pbar:
            # Compute relative path for directory structure preservation
            if args.recursive:
                rel_path = img_path.relative_to(input_dir)
                output_path = output_dir / rel_path
            else:
                output_path = output_dir / img_path.name

            # Process
            try:
                inference_time, quality_metrics = process_single_image(
                    model=model,
                    input_path=img_path,
                    output_path=output_path,
                    save_comparison=args.save_comparison,
                    benchmark=args.benchmark,
                    report_quality_metrics=args.report_quality_metrics,
                )

                total_time += inference_time
                if quality_metrics:
                    per_image_quality[str(output_path)] = quality_metrics

                # Update progress bar
                if args.benchmark:
                    avg_time = total_time / (pbar.n + 1)
                    pbar.set_postfix(
                        {"avg_time": f"{avg_time:.3f}s", "fps": f"{1 / avg_time:.1f}"}
                    )

            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")

        logger.info(f"\nProcessed {len(image_files)} images")
        logger.info(f"Output directory: {output_dir}")

        if args.benchmark:
            avg_time = total_time / len(image_files)
            logger.info("\nBenchmark Results:")
            logger.info(f"  Total time: {total_time:.2f}s")
            logger.info(f"  Average time per image: {avg_time:.3f}s")
            logger.info(f"  Average FPS: {1 / avg_time:.1f}")

        if per_image_quality:
            aggregate: Dict[str, float] = {}
            metric_names = next(iter(per_image_quality.values())).keys()
            for name in metric_names:
                values = [m[name] for m in per_image_quality.values()]
                aggregate[name] = sum(values) / len(values)

            logger.info("\nQuality Metrics (no-reference):")
            for name, value in aggregate.items():
                logger.info(f"  Average {name.upper()}: {value:.4f}")

            quality_results: Dict[str, Any] = {
                "aggregate": aggregate,
                "per_image": per_image_quality,
            }
            quality_file = output_dir / "quality_metrics.json"
            with open(quality_file, "w") as f:
                json.dump(quality_results, f, indent=2)
            logger.info(f"  Quality metrics saved to {quality_file}")

    logger.info("\n" + "=" * 80)
    logger.info("Inference completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
