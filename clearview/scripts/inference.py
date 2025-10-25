#!/usr/bin/env python
"""Inference script for image deraining."""

import argparse
import logging
import time
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm


from clearview.api import DerainingModel

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference on rainy images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["unet", "attention_unet"],
        help="Model architecture",
    )
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to model weights"
    )

    # Input/Output arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=str, help="Input image path")
    input_group.add_argument(
        "--input-dir", type=str, help="Input directory containing images"
    )

    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output", type=str, help="Output image path")
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


def process_single_image(
    model: DerainingModel,
    input_path: Path,
    output_path: Path,
    save_comparison: bool = False,
    benchmark: bool = False,
) -> float:
    """Process a single image."""
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Measure time
    start_time = time.time()

    # Process image
    if save_comparison:
        # Load input for comparison
        from PIL import Image

        import numpy as np

        rainy_img = np.array(Image.open(input_path).convert("RGB"))

        # Process
        derained_img = model.process(input_path, output_path=output_path)

        # Save comparison
        comparison_path = (
            output_path.parent / f"{output_path.stem}_comparison{output_path.suffix}"
        )

        from clearview.utils.image import numpy_to_tensor
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

    return inference_time


def main() -> None:
    """Main inference function."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    logger.info("=" * 80)
    logger.info("Image Deraining Inference")
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

    # Process single image or directory
    if args.input is not None:
        # Single image
        input_path = Path(args.input)
        output_path = Path(args.output)

        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return

        logger.info(f"\nProcessing: {input_path}")

        inference_time = process_single_image(
            model=model,
            input_path=input_path,
            output_path=output_path,
            save_comparison=args.save_comparison,
            benchmark=args.benchmark,
        )

        logger.info(f"Output saved to: {output_path}")

        if args.benchmark:
            logger.info(f"Inference time: {inference_time:.3f}s")

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
                inference_time = process_single_image(
                    model=model,
                    input_path=img_path,
                    output_path=output_path,
                    save_comparison=args.save_comparison,
                    benchmark=args.benchmark,
                )

                total_time += inference_time

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

    logger.info("\n" + "=" * 80)
    logger.info("Inference completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
