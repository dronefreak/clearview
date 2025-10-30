#!/usr/bin/env python
"""Evaluation script for image deraining models."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from clearview.api import DerainingModel
from clearview.data import ImagePairDataset, Rain100Dataset, get_val_transforms
from clearview.utils import (
    MetricsTracker,
    compute_metrics,
    create_comparison_grid,
    plot_metric_histogram,
    save_comparison,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate image deraining models",
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
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")

    # Data arguments
    parser.add_argument("--data-dir", type=str, required=True, help="Test data directory")
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="pair",
        choices=["pair", "rain100", "rain1400"],
        help="Dataset type",
    )
    parser.add_argument("--rainy-dir", type=str, default="rainy", help="Rainy images subdirectory")
    parser.add_argument("--clean-dir", type=str, default="clean", help="Clean images subdirectory")

    # Evaluation arguments
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["psnr", "ssim", "mae", "mse"],
        help="Metrics to compute",
    )

    # Output arguments
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--save-images", action="store_true", help="Save output images")
    parser.add_argument("--num-vis", type=int, default=10, help="Number of images to visualize")
    parser.add_argument("--save-comparison-grid", action="store_true", help="Save comparison grid")

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run evaluation on",
    )

    return parser.parse_args()


def evaluate(
    model: DerainingModel,
    dataloader: DataLoader,
    metrics: List[str],
    output_dir: Path,
    save_images: bool = False,
    num_vis: int = 10,
) -> Tuple[
    Dict[str, float],
    Dict[str, Dict[str, float]],
    Dict[str, List[float]],
    List[Tuple[Any, Any, Any]],
]:
    """Evaluate model on dataset."""
    model.eval()

    metrics_tracker = MetricsTracker()

    # Storage for visualizations
    vis_samples: List[Tuple[Any, Any, Any]] = []

    # Metric lists for histogram
    metric_values: Dict[str, List[float]] = {m: [] for m in metrics}

    logger.info("Starting evaluation...")

    pbar = tqdm(dataloader, desc="Evaluating")

    for idx, batch in enumerate(pbar):
        # Get data
        if isinstance(batch, (tuple, list)):
            rainy, clean = batch[0], batch[1]
        else:
            rainy = batch["rainy"]
            clean = batch["clean"]

        # Process
        derained = model.process_batch(rainy)

        # Compute metrics
        batch_metrics = compute_metrics(derained.cpu(), clean.cpu(), metrics=metrics)

        metrics_tracker.update(batch_metrics, batch_size=rainy.size(0))

        # Store per-image metrics
        for metric in metrics:
            if metric in batch_metrics:
                metric_values[metric].append(batch_metrics[metric])

        # Update progress bar
        avg_metrics = metrics_tracker.average()
        pbar.set_postfix({k: f"{v:.2f}" for k, v in avg_metrics.items()})

        # Save images
        if save_images and idx < num_vis:
            img_dir = output_dir / "images"
            img_dir.mkdir(parents=True, exist_ok=True)

            for i in range(rainy.size(0)):
                save_comparison(
                    rainy[i].cpu(),
                    derained[i].cpu(),
                    clean[i].cpu(),
                    save_path=img_dir / f"result_{idx:04d}_{i:02d}.png",
                )

        # Store for comparison grid
        if len(vis_samples) < num_vis:
            for i in range(min(rainy.size(0), num_vis - len(vis_samples))):
                vis_samples.append((rainy[i].cpu(), derained[i].cpu(), clean[i].cpu()))

    # Get final metrics
    final_metrics = metrics_tracker.average()
    summary = metrics_tracker.summary()

    return final_metrics, summary, metric_values, vis_samples


def main() -> None:
    """Main evaluation function."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(log_file=output_dir / "evaluation.log", level=logging.INFO)

    logger.info("=" * 80)
    logger.info("Starting evaluation")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Weights: {args.weights}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Load model
    logger.info("\n" + "=" * 80)
    logger.info("Loading model")
    logger.info("=" * 80)

    model = DerainingModel.from_pretrained(model_name=args.model, weights=args.weights, device=args.device)

    logger.info("Model loaded successfully")
    logger.info(f"Device: {args.device}")

    # Setup data
    logger.info("\n" + "=" * 80)
    logger.info("Loading dataset")
    logger.info("=" * 80)

    data_dir = Path(args.data_dir)

    # Validation transform (no augmentation)
    transform = get_val_transforms()

    # Create dataset
    if args.dataset_type == "rain100":
        dataset = Rain100Dataset(root_dir=data_dir, transform=transform)
    else:  # pair
        dataset = ImagePairDataset(
            rainy_dir=data_dir / args.rainy_dir,
            clean_dir=data_dir / args.clean_dir,
            transform=transform,
        )

    logger.info(f"Test samples: {len(dataset)}")

    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Evaluate
    logger.info("\n" + "=" * 80)
    logger.info("Running evaluation")
    logger.info("=" * 80)

    final_metrics, summary, metric_values, vis_samples = evaluate(
        model=model,
        dataloader=dataloader,
        metrics=args.metrics,
        output_dir=output_dir,
        save_images=args.save_images,
        num_vis=args.num_vis,
    )

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Results")
    logger.info("=" * 80)

    for metric, value in final_metrics.items():
        metric_summary = summary[metric]
        logger.info(
            f"{metric.upper()}: {value:.4f} "
            f"(±{metric_summary['std']:.4f}, "
            f"min={metric_summary['min']:.4f}, "
            f"max={metric_summary['max']:.4f})"
        )

    # Save results to JSON
    results = {
        "model": args.model,
        "weights": str(args.weights),
        "dataset": str(args.data_dir),
        "num_samples": len(dataset),
        "metrics": final_metrics,
        "summary": summary,
    }

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {results_file}")

    # Create comparison grid
    if args.save_comparison_grid and vis_samples:
        logger.info("\nCreating comparison grid...")
        fig = create_comparison_grid(vis_samples, max_images=min(len(vis_samples), args.num_vis))
        grid_path = output_dir / "comparison_grid.png"
        fig.savefig(grid_path, dpi=150, bbox_inches="tight")
        logger.info(f"Comparison grid saved to {grid_path}")

    # Plot metric histograms
    logger.info("\nPlotting metric distributions...")
    plot_metric_histogram(metric_values, save_path=output_dir / "metric_distributions.png")
    logger.info(f"Metric distributions saved to {output_dir / 'metric_distributions.png'}")

    logger.info("\n" + "=" * 80)
    logger.info("Evaluation completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
