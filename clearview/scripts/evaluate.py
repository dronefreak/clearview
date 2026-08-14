#!/usr/bin/env python
"""Evaluation script for image deraining models."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from clearview.api import DerainingModel
from clearview.data import (
    ImagePairDataset,
    Rain100Dataset,
    Rain1400Dataset,
    SPADataDataset,
    get_val_transforms,
)
from clearview.models import list_models
from clearview.utils import (
    ALL_METRICS,
    MetricsTracker,
    compute_fid,
    compute_metrics,
    create_comparison_grid,
    plot_metric_histogram,
    save_comparison,
    setup_logging,
)
from clearview.utils.niqe import fit_niqe_model

logger = logging.getLogger(__name__)


def _filter_available_metrics(metrics: List[str]) -> List[str]:
    """Drop requested metrics whose optional dependency isn't installed.

    Warns (rather than raising) so evaluation can still run and report
    every metric it *can* compute even if e.g. ``lpips``/``piq`` are
    missing from the environment.
    """
    optional_deps = {
        "lpips": "lpips",
        "dists": "piq",
        "brisque": "piq",
        "niqe": None,  # pure-numpy, no optional dependency
    }

    available = []
    for metric in metrics:
        dep = optional_deps.get(metric.lower())
        if dep is not None:
            try:
                __import__(dep)
            except ImportError:
                logger.warning(
                    f"Skipping metric '{metric}': requires the optional "
                    f"'{dep}' package (pip install {dep})"
                )
                continue
        available.append(metric)

    return available


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
        default="unet",
        choices=list_models(),
        help="Model architecture",
    )
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to model weights"
    )

    # Data arguments
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Test data directory"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="pair",
        choices=["pair", "rain100", "rain1400", "spa-data"],
        help="Dataset type. 'spa-data' expects --data-dir to point directly "
        "at a split directory (e.g. .../SPA-Data/val) and auto-detects the "
        "rain/norain subdirectory names within it.",
    )
    parser.add_argument(
        "--rainy-dir", type=str, default="rainy", help="Rainy images subdirectory"
    )
    parser.add_argument(
        "--clean-dir", type=str, default="clean", help="Clean images subdirectory"
    )

    # Evaluation arguments
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=list(ALL_METRICS),
        help="Metrics to compute and log. Defaults to every metric "
        f"implemented in clearview.utils.metrics (currently: {ALL_METRICS}). "
        "'lpips'/'dists'/'brisque' require the optional 'lpips'/'piq' "
        "packages and are skipped with a warning if unavailable. 'niqe' "
        "fits a reference model from a sample of this dataset's clean "
        "images (see --niqe-fit-samples). 'rain_removal_rate' uses the "
        "rainy input image automatically.",
    )
    parser.add_argument(
        "--niqe-fit-samples",
        type=int,
        default=50,
        help="Number of clean images used to fit the NIQE reference model "
        "when 'niqe' is included in --metrics",
    )
    parser.add_argument(
        "--compute-fid",
        action="store_true",
        help="Also compute FID (Frechet Inception Distance) between all "
        "derained and clean images across the full dataset. Requires the "
        "optional 'piq' package. This is a distribution-level metric "
        "(not per-image) and is reported separately from --metrics.",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for results"
    )
    parser.add_argument("--save-images", action="store_true", help="Save output images")
    parser.add_argument(
        "--num-vis", type=int, default=10, help="Number of images to visualize"
    )
    parser.add_argument(
        "--save-comparison-grid", action="store_true", help="Save comparison grid"
    )

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
    niqe_fit_samples: int = 50,
    compute_fid_score: bool = False,
) -> Tuple[
    Dict[str, float],
    Dict[str, Dict[str, float]],
    Dict[str, List[float]],
    List[Tuple[Any, Any, Any]],
    Optional[float],
]:
    """Evaluate model on dataset."""
    model.eval()

    metrics_tracker = MetricsTracker()

    # Storage for visualizations
    vis_samples: List[Tuple[Any, Any, Any]] = []

    # Metric lists for histogram
    metric_values: Dict[str, List[float]] = {m: [] for m in metrics}

    # NIQE requires a reference model fitted on pristine (clean) images.
    niqe_model = None
    if "niqe" in metrics:
        logger.info(
            f"Fitting NIQE reference model from up to {niqe_fit_samples} "
            "clean images..."
        )
        reference_images = []
        for batch in dataloader:
            clean = batch[1] if isinstance(batch, (tuple, list)) else batch["clean"]
            for i in range(clean.size(0)):
                reference_images.append(clean[i])
                if len(reference_images) >= niqe_fit_samples:
                    break
            if len(reference_images) >= niqe_fit_samples:
                break
        niqe_model = fit_niqe_model(reference_images)

    # Accumulators for the optional (distribution-level) FID score
    fid_real_images: List[torch.Tensor] = []
    fid_fake_images: List[torch.Tensor] = []

    logger.info("Starting evaluation...")

    pbar = tqdm(dataloader, desc="Evaluating")

    for idx, batch in enumerate(pbar):
        # Get data
        if isinstance(batch, (tuple, list)):
            rainy, clean = batch[0], batch[1]
        else:
            rainy = batch["rainy"]
            clean = batch["clean"]

        # Process. Native-resolution evaluation sets can contain outlier
        # images large enough to OOM the GPU even at batch size 1 (seen in
        # practice on Rain13K's Test2800, DDN-Data, and RealRain-1k). Fall
        # back to CPU for just that one sample rather than aborting the run.
        try:
            derained = model.process_batch(rainy)
        except torch.OutOfMemoryError:
            logger.warning(
                f"CUDA OOM on sample {idx} (shape {tuple(rainy.shape)}); "
                "retrying this sample on CPU"
            )
            torch.cuda.empty_cache()
            original_device = model.device
            model.to("cpu")
            try:
                derained = model.process_batch(rainy)
            finally:
                model.to(original_device)
                torch.cuda.empty_cache()

        # Compute metrics
        batch_metrics = compute_metrics(
            derained.cpu(),
            clean.cpu(),
            metrics=metrics,
            rainy=rainy.cpu(),
            niqe_model=niqe_model,
        )

        metrics_tracker.update(batch_metrics, batch_size=rainy.size(0))

        # Store per-image metrics
        for metric in metrics:
            if metric in batch_metrics:
                metric_values[metric].append(batch_metrics[metric])

        # Update progress bar
        avg_metrics = metrics_tracker.average()
        pbar.set_postfix({k: f"{v:.2f}" for k, v in avg_metrics.items()})

        if compute_fid_score:
            fid_real_images.append(clean.cpu())
            fid_fake_images.append(derained.cpu())

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

    fid_score: Optional[float] = None
    if compute_fid_score:
        logger.info("Computing FID over the full dataset...")
        fid_score = compute_fid(
            torch.cat(fid_real_images, dim=0), torch.cat(fid_fake_images, dim=0)
        )

    return final_metrics, summary, metric_values, vis_samples, fid_score


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

    model = DerainingModel.from_pretrained(
        model_name=args.model, weights=args.weights, device=args.device
    )

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
    elif args.dataset_type == "rain1400":
        dataset = Rain1400Dataset(
            rainy_dir=data_dir / args.rainy_dir,
            clean_dir=data_dir / args.clean_dir,
            transform=transform,
        )
    elif args.dataset_type == "spa-data":
        dataset = SPADataDataset(root_dir=data_dir, transform=transform)
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

    requested_metrics = _filter_available_metrics(args.metrics)
    logger.info(f"Metrics: {requested_metrics}")

    final_metrics, summary, metric_values, vis_samples, fid_score = evaluate(
        model=model,
        dataloader=dataloader,
        metrics=requested_metrics,
        output_dir=output_dir,
        save_images=args.save_images,
        num_vis=args.num_vis,
        niqe_fit_samples=args.niqe_fit_samples,
        compute_fid_score=args.compute_fid,
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

    if fid_score is not None:
        logger.info(f"FID: {fid_score:.4f}")

    # Save results to JSON
    results = {
        "model": args.model,
        "weights": str(args.weights),
        "dataset": str(args.data_dir),
        "num_samples": len(dataset),
        "metrics": final_metrics,
        "summary": summary,
        "fid": fid_score,
    }

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {results_file}")

    # Create comparison grid
    if args.save_comparison_grid and vis_samples:
        logger.info("\nCreating comparison grid...")
        fig = create_comparison_grid(
            vis_samples, max_images=min(len(vis_samples), args.num_vis)
        )
        grid_path = output_dir / "comparison_grid.png"
        fig.savefig(grid_path, dpi=150, bbox_inches="tight")
        logger.info(f"Comparison grid saved to {grid_path}")

    # Plot metric histograms
    logger.info("\nPlotting metric distributions...")
    plot_metric_histogram(
        metric_values, save_path=output_dir / "metric_distributions.png"
    )
    logger.info(
        f"Metric distributions saved to {output_dir / 'metric_distributions.png'}"
    )

    logger.info("\n" + "=" * 80)
    logger.info("Evaluation completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
