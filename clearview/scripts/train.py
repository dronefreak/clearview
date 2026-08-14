#!/usr/bin/env python
"""Training script for image deraining models."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn
import yaml
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    MultiStepLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from clearview.data import (
    ImagePairDataset,
    MixedDataset,
    Rain100Dataset,
    Rain1400Dataset,
    SPADataDataset,
    get_train_transforms,
    get_val_transforms,
)
from clearview.losses import CombinedLoss
from clearview.models import get_model, list_models
from clearview.training import (
    Callback,
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    Trainer,
    WarmupCosineScheduler,
)
from clearview.utils import plot_training_curves, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train image deraining models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory containing train/val data",
    )
    data_group.add_argument(
        "--dataset-type",
        type=str,
        default="pair",
        choices=["pair", "rain100", "rain1400", "spa-data"],
        help="Dataset type. 'pair'/'rain100'/'rain1400' use the "
        "--train-rainy/--train-clean/--val-rainy/--val-clean subdirectory "
        "overrides. 'spa-data' uses --train-split/--val-split instead (see "
        "below) since SPADataDataset auto-detects the rain/norain "
        "subdirectory names within each split.",
    )
    data_group.add_argument(
        "--train-rainy",
        type=str,
        default="train/rainy",
        help="Training rainy images subdirectory",
    )
    data_group.add_argument(
        "--train-clean",
        type=str,
        default="train/clean",
        help="Training clean images subdirectory",
    )
    data_group.add_argument(
        "--val-rainy",
        type=str,
        default="val/rainy",
        help="Validation rainy images subdirectory",
    )
    data_group.add_argument(
        "--val-clean",
        type=str,
        default="val/clean",
        help="Validation clean images subdirectory",
    )
    data_group.add_argument(
        "--train-split",
        type=str,
        default="train",
        help="Training split name, looked up under --data-dir (only used "
        "with --dataset-type spa-data)",
    )
    data_group.add_argument(
        "--val-split",
        type=str,
        default="val",
        help="Validation split name, looked up under --data-dir (only used "
        "with --dataset-type spa-data)",
    )
    data_group.add_argument(
        "--mix-config",
        type=str,
        default=None,
        help="Path to a YAML file listing multiple training sources to "
        "combine into one MixedDataset (e.g. synthetic + real-world "
        "datasets), each with its own dataset_type/data_dir/weight. When "
        "given, this replaces how the *training* dataset is built; "
        "--data-dir/--dataset-type/--val-split etc. are still used as-is "
        "for the *validation* dataset (typically a single, real-world "
        "source). See clearview/scripts/README.md for the file format.",
    )
    data_group.add_argument(
        "--mix-sampler",
        action="store_true",
        help="When --mix-config is given, draw training batches with a "
        "WeightedRandomSampler using each source's configured weight "
        "(oversampling small sources), instead of a single shuffled pass "
        "over the concatenated data.",
    )

    # Model arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=list_models(),
        help="Model architecture",
    )
    model_group.add_argument(
        "--in-channels", type=int, default=3, help="Number of input channels"
    )
    model_group.add_argument(
        "--out-channels", type=int, default=3, help="Number of output channels"
    )

    model_group.add_argument(
        "--backbone",
        type=str,
        default="resnet34",
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        help="ResNet backbone (only used with resnet_unet model)",
    )

    model_group.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use ImageNet pretrained weights for ResNet backbone",
    )

    model_group.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze encoder (train decoder only)",
    )
    model_group.add_argument(
        "--unfreeze-encoder",
        action="store_true",
        help="Unfreeze encoder (for stage 2 fine-tuning)",
    )

    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    train_group.add_argument(
        "--batch-size", type=int, default=8, help="Training batch size"
    )
    train_group.add_argument(
        "--val-batch-size", type=int, default=8, help="Validation batch size"
    )
    train_group.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading workers"
    )
    train_group.add_argument("--seed", type=int, default=42, help="Random seed")

    # Optimizer arguments
    optim_group = parser.add_argument_group("Optimizer")
    optim_group.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "adamw", "sgd"],
        help="Optimizer type",
    )
    optim_group.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    optim_group.add_argument(
        "--weight-decay", type=float, default=0.0, help="Weight decay"
    )
    optim_group.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum (for SGD)"
    )

    # Loss arguments
    loss_group = parser.add_argument_group("Loss")
    loss_group.add_argument(
        "--loss",
        type=str,
        default="l1_l2_ssim_edge",
        choices=[
            "l1",
            "l2",
            "l1_l2_ssim",
            "l1_l2_ssim_edge",
            "l1_l2_ssim_edge_perceptual",
            "custom",
        ],
        help="Loss function preset. Ignored (with a warning) if "
        "--loss-config or --loss-config-file is also given; use 'custom' "
        "explicitly when relying solely on one of those.",
    )
    loss_group.add_argument(
        "--loss-config",
        type=str,
        default=None,
        help="Arbitrary loss combination as a JSON object mapping loss "
        "names (from clearview.losses.CombinedLoss's registry — e.g. "
        "l1, l2, charbonnier, ssim, ms_ssim, edge, sobel, laplacian, "
        "perceptual, dists, fft, focal_frequency, wavelet, color, "
        "adversarial) to their keyword arguments, each including a "
        "'weight'. Example: "
        '\'{"l1": {"weight": 1.0}, "ssim": {"weight": 0.5}, '
        '"dists": {"weight": 0.1}}\'. '
        "Takes precedence over --loss when provided. See also "
        "--loss-config-file for loading this from a JSON/YAML file.",
    )
    loss_group.add_argument(
        "--loss-config-file",
        type=str,
        default=None,
        help="Path to a JSON or YAML file with the same structure as "
        "--loss-config (a mapping of loss name -> kwargs incl. 'weight'). "
        "Takes precedence over both --loss and --loss-config when given.",
    )
    loss_group.add_argument(
        "--l1-weight", type=float, default=1.0, help="L1 loss weight"
    )
    loss_group.add_argument(
        "--l2-weight", type=float, default=1.0, help="L2 loss weight"
    )
    loss_group.add_argument(
        "--ssim-weight", type=float, default=1.0, help="SSIM loss weight"
    )
    loss_group.add_argument(
        "--edge-weight", type=float, default=0.5, help="Edge loss weight"
    )
    loss_group.add_argument(
        "--vgg-weight", type=float, default=0.5, help="VGG loss weight"
    )

    # Augmentation arguments
    aug_group = parser.add_argument_group("Augmentation")
    aug_group.add_argument(
        "--crop-size", type=int, default=256, help="Random crop size"
    )
    aug_group.add_argument(
        "--flip-prob", type=float, default=0.5, help="Probability of flipping"
    )
    aug_group.add_argument(
        "--no-rotation", action="store_true", help="Disable random rotation"
    )

    # Scheduler arguments
    sched_group = parser.add_argument_group("LR Scheduler")
    sched_group.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        choices=["plateau", "cosine", "step", "multistep", "none"],
        help="Learning rate scheduler",
    )
    sched_group.add_argument(
        "--scheduler-patience",
        type=int,
        default=10,
        help="Patience for ReduceLROnPlateau",
    )
    sched_group.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.5,
        help="Factor for ReduceLROnPlateau",
    )
    sched_group.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Number of linear warmup epochs before cosine decay begins "
        "(only used with --scheduler cosine; 0 = no warmup, uses plain "
        "CosineAnnealingLR)",
    )
    sched_group.add_argument(
        "--warmup-start-lr",
        type=float,
        default=0.0,
        help="Learning rate at the start of warmup (only used when "
        "--warmup-epochs > 0)",
    )

    # Callbacks arguments
    callback_group = parser.add_argument_group("Callbacks")
    callback_group.add_argument(
        "--early-stopping", action="store_true", help="Enable early stopping"
    )
    callback_group.add_argument(
        "--patience", type=int, default=15, help="Early stopping patience"
    )
    callback_group.add_argument(
        "--checkpoint-monitor",
        type=str,
        default="val_psnr",
        help="Metric to monitor for checkpointing",
    )
    callback_group.add_argument(
        "--checkpoint-mode",
        type=str,
        default="max",
        choices=["min", "max"],
        help="Mode for checkpoint monitoring",
    )

    # Mixed precision & optimization
    opt_group = parser.add_argument_group("Optimization")
    opt_group.add_argument(
        "--mixed-precision", action="store_true", help="Use automatic mixed precision"
    )
    opt_group.add_argument(
        "--gradient-clip", type=float, default=None, help="Gradient clipping value"
    )
    opt_group.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients over before an "
        "optimizer step (simulates a larger batch size)",
    )
    opt_group.add_argument(
        "--compile",
        action="store_true",
        help="Wrap the model with torch.compile() for potentially faster "
        "training (PyTorch 2.x). Falls back to eager mode with a warning "
        "if compilation fails.",
    )
    opt_group.add_argument(
        "--compile-mode",
        type=str,
        default=None,
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile() mode (only used with --compile)",
    )

    # EMA (exponential moving average of model weights)
    ema_group = parser.add_argument_group("EMA")
    ema_group.add_argument(
        "--ema",
        action="store_true",
        help="Track an exponential moving average of model weights, "
        "typically yielding better-generalizing weights for validation and "
        "final checkpoints",
    )
    ema_group.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="EMA decay rate (higher = smoother/longer averaging window, "
        "only used with --ema)",
    )
    ema_group.add_argument(
        "--ema-update-after-step",
        type=int,
        default=0,
        help="Number of optimizer steps to skip before EMA tracking begins "
        "(only used with --ema)",
    )
    ema_group.add_argument(
        "--no-ema-warmup",
        action="store_true",
        help="Disable EMA decay warmup ramp-up during early training "
        "(only used with --ema)",
    )
    ema_group.add_argument(
        "--no-validate-with-ema",
        action="store_true",
        help="Validate using raw training weights instead of EMA weights "
        "(only used with --ema)",
    )

    # Output arguments
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and logs",
    )
    output_group.add_argument(
        "--save-every", type=int, default=10, help="Save checkpoint every N epochs"
    )

    # Resuming
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides CLI args)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to train on",
    )

    args = parser.parse_args()

    # Load config file if provided
    if args.config is not None:
        import yaml

        with open(args.config) as f:
            config = yaml.safe_load(f)

        # Update args with config (CLI args take precedence)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key.replace("-", "_"), value)

    return args


def _build_dataset(
    dataset_type: str,
    data_dir: Path,
    transform: Any,
    *,
    rainy_dir: Optional[str] = None,
    clean_dir: Optional[str] = None,
    split: Optional[str] = None,
) -> Dataset:
    """Build a single dataset of the given type.

    Shared by both the single-source path (--dataset-type/--data-dir) and
    each source listed under --mix-config, so the dataset_type -> class
    mapping only lives in one place.
    """
    if dataset_type == "rain100":
        return Rain100Dataset(root_dir=data_dir / (split or ""), transform=transform)
    elif dataset_type == "rain1400":
        return Rain1400Dataset(
            rainy_dir=data_dir / (rainy_dir or "rainy"),
            clean_dir=data_dir / (clean_dir or "clean"),
            transform=transform,
        )
    elif dataset_type == "spa-data":
        return SPADataDataset(root_dir=data_dir, split=split, transform=transform)
    else:  # pair
        return ImagePairDataset(
            rainy_dir=data_dir / (rainy_dir or "rainy"),
            clean_dir=data_dir / (clean_dir or "clean"),
            transform=transform,
        )


def _load_mix_config(path: str) -> List[Dict[str, Any]]:
    """Load and validate a --mix-config YAML file.

    Expected format::

        sources:
          - dataset_type: pair       # pair | rain100 | rain1400 | spa-data
            data_dir: /path/to/source
            rainy_dir: input         # pair/rain1400 only
            clean_dir: target        # pair/rain1400 only
            split: train             # spa-data only
            weight: 1.0              # optional, default 1.0

    Returns:
        The list of source dicts under the top-level "sources" key.

    Raises:
        ValueError: If the file has no top-level "sources" list, or any
            source is missing "data_dir".
    """
    with open(path) as f:
        config = yaml.safe_load(f)

    sources = config.get("sources") if isinstance(config, dict) else None
    if not sources:
        raise ValueError(
            f"--mix-config file {path!r} must contain a top-level "
            "'sources' list with at least one entry"
        )
    for i, source in enumerate(sources):
        if "data_dir" not in source:
            raise ValueError(f"--mix-config source #{i} is missing 'data_dir'")

    return cast(List[Dict[str, Any]], sources)


def _build_mixed_train_dataset(mix_config_path: str, transform: Any) -> MixedDataset:
    """Build a MixedDataset from a --mix-config YAML file."""
    sources = _load_mix_config(mix_config_path)

    datasets = []
    weights = []
    for source in sources:
        dataset = _build_dataset(
            source.get("dataset_type", "pair"),
            Path(source["data_dir"]),
            transform,
            rainy_dir=source.get("rainy_dir"),
            clean_dir=source.get("clean_dir"),
            split=source.get("split"),
        )
        datasets.append(dataset)
        weights.append(float(source.get("weight", 1.0)))
        logger.info(
            f"  Mix source: {source['data_dir']} "
            f"(type={source.get('dataset_type', 'pair')}, "
            f"weight={weights[-1]}) -> {len(dataset)} pairs"
        )

    return MixedDataset(datasets, weights=weights)


def setup_data(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    """Setup data loaders."""
    data_dir = Path(args.data_dir)

    # Training transforms
    train_transform = get_train_transforms(
        crop_size=(args.crop_size, args.crop_size),
        flip_prob=args.flip_prob,
        rotate=not args.no_rotation,
    )

    # Validation transforms
    val_transform = get_val_transforms(crop_size=(args.crop_size, args.crop_size))

    # Training dataset: either a single source (default) or a MixedDataset
    # combining multiple sources (--mix-config).
    train_sampler: Optional[WeightedRandomSampler] = None
    if args.mix_config is not None:
        logger.info(f"Building mixed training dataset from {args.mix_config}")
        train_dataset: Dataset = _build_mixed_train_dataset(
            args.mix_config, train_transform
        )
        if args.mix_sampler:
            train_sampler = WeightedRandomSampler(
                cast(MixedDataset, train_dataset).sample_weights(),
                num_samples=len(train_dataset),
                replacement=True,
            )
    else:
        train_dataset = _build_dataset(
            args.dataset_type,
            data_dir,
            train_transform,
            rainy_dir=args.train_rainy,
            clean_dir=args.train_clean,
            split=args.train_split,
        )

    # Validation dataset always comes from --data-dir/--dataset-type, even
    # when --mix-config is used for training (typically a single,
    # real-world source you want to monitor generalization against).
    val_dataset = _build_dataset(
        args.dataset_type,
        data_dir,
        val_transform,
        rainy_dir=args.val_rainy,
        clean_dir=args.val_clean,
        split=args.val_split,
    )

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def setup_model(args: argparse.Namespace) -> nn.Module:
    """Setup model."""
    model = get_model(
        args.model, in_channels=args.in_channels, out_channels=args.out_channels
    )

    if args.freeze_encoder and hasattr(model, "freeze_encoder"):
        model.freeze_encoder()
        logger.info("Encoder frozen - training decoder only")

    if args.unfreeze_encoder and hasattr(model, "unfreeze_encoder"):
        model.unfreeze_encoder()
        logger.info("Encoder unfrozen - training full network")

    logger.info(f"Model: {args.model}")
    logger.info(f"Parameters: {model.get_num_params():,}")
    logger.info(f"Model size: {model.get_model_size_mb():.2f} MB")

    return model


def setup_optimizer(
    args: argparse.Namespace, model: nn.Module
) -> torch.optim.Optimizer:
    """Setup optimizer."""
    if args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    logger.info(f"Optimizer: {args.optimizer}")
    logger.info(f"Learning rate: {args.lr}")

    return optimizer


def _load_loss_config_file(path: str) -> Dict[str, Dict[str, Any]]:
    """Load a loss configuration dict from a JSON or YAML file.

    Args:
        path: Path to a ``.json``, ``.yml``, or ``.yaml`` file containing a
            mapping of loss name -> keyword arguments (including ``weight``)

    Returns:
        The parsed loss configuration dict

    Raises:
        ValueError: If the file's top-level content is not a mapping
    """
    file_path = Path(path)
    text = file_path.read_text()

    if file_path.suffix.lower() == ".json":
        config = json.loads(text)
    else:
        config = yaml.safe_load(text)

    if not isinstance(config, dict):
        raise ValueError(
            f"Loss config file {path!r} must contain a top-level mapping of "
            f"loss name -> kwargs, got {type(config).__name__}"
        )

    return config


def setup_loss(args: argparse.Namespace) -> nn.Module:
    """Setup loss function.

    Supports two ways to configure the loss:

    1. A named preset via ``--loss`` (e.g. ``l1_l2_ssim_edge``), tunable
       with the individual ``--*-weight`` flags.
    2. An arbitrary combination of any loss registered in
       ``CombinedLoss.from_config()`` via ``--loss-config`` (inline JSON)
       or ``--loss-config-file`` (JSON/YAML file) — this is required (and
       takes precedence over ``--loss``) when ``--loss custom`` is chosen,
       but can also be used to override any preset directly.
    """
    loss_config: Dict[str, Dict[str, Any]]

    if args.loss_config_file is not None:
        loss_config = _load_loss_config_file(args.loss_config_file)
        if args.loss != "custom":
            logger.warning(
                f"--loss-config-file was given; ignoring --loss={args.loss!r} preset"
            )
    elif args.loss_config is not None:
        loss_config = json.loads(args.loss_config)
        if args.loss != "custom":
            logger.warning(
                f"--loss-config was given; ignoring --loss={args.loss!r} preset"
            )
    elif args.loss == "custom":
        raise ValueError(
            "--loss custom requires --loss-config or --loss-config-file "
            "to specify the loss combination"
        )
    elif args.loss == "l1":
        loss_config = {"l1": {"weight": 1.0}}
    elif args.loss == "l2":
        loss_config = {"l2": {"weight": 1.0}}
    elif args.loss == "l1_l2_ssim":
        loss_config = {
            "l1": {"weight": args.l1_weight},
            "l2": {"weight": args.l2_weight},
            "ssim": {"weight": args.ssim_weight},
        }
    elif args.loss == "l1_l2_ssim_edge":
        loss_config = {
            "l1": {"weight": args.l1_weight},
            "l2": {"weight": args.l2_weight},
            "ssim": {"weight": args.ssim_weight},
            "edge": {"weight": args.edge_weight},
        }
    elif args.loss == "l1_l2_ssim_edge_perceptual":
        loss_config = {
            "l1": {"weight": args.l1_weight},
            "l2": {"weight": args.l2_weight},
            "ssim": {"weight": args.ssim_weight},
            "edge": {"weight": args.edge_weight},
            "perceptual": {"weight": args.vgg_weight},
        }
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    loss_fn = CombinedLoss.from_config(loss_config)

    logger.info(f"Loss function: {loss_fn}")

    return loss_fn


def setup_scheduler(
    args: argparse.Namespace, optimizer: torch.optim.Optimizer
) -> Tuple[Optional[Any], Optional[str]]:
    """Setup learning rate scheduler."""
    if args.scheduler == "none":
        return None, None

    monitor: Optional[str]
    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
        )
        monitor = "val_loss"
    elif args.scheduler == "cosine":
        if args.warmup_epochs > 0:
            scheduler = WarmupCosineScheduler(
                optimizer,
                warmup_epochs=args.warmup_epochs,
                total_epochs=args.epochs,
                warmup_start_lr=args.warmup_start_lr,
            )
            logger.info(
                f"Using cosine annealing with {args.warmup_epochs}-epoch linear "
                f"warmup (warmup_start_lr={args.warmup_start_lr})"
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, verbose=True)
        monitor = None
    elif args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1, verbose=True)
        monitor = None
    elif args.scheduler == "multistep":
        scheduler = MultiStepLR(
            optimizer, milestones=[30, 60, 90], gamma=0.1, verbose=True
        )
        monitor = None
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")

    logger.info(f"Scheduler: {args.scheduler}")

    return scheduler, monitor


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(log_file=output_dir / "training.log", level=logging.INFO)

    logger.info("=" * 80)
    logger.info("Starting training")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Arguments: {vars(args)}")

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Setup components
    logger.info("\n" + "=" * 80)
    logger.info("Setting up data loaders")
    logger.info("=" * 80)
    train_loader, val_loader = setup_data(args)

    logger.info("\n" + "=" * 80)
    logger.info("Setting up model")
    logger.info("=" * 80)
    model = setup_model(args)

    logger.info("\n" + "=" * 80)
    logger.info("Setting up optimizer")
    logger.info("=" * 80)
    optimizer = setup_optimizer(args, model)

    logger.info("\n" + "=" * 80)
    logger.info("Setting up loss function")
    logger.info("=" * 80)
    loss_fn = setup_loss(args)

    # Setup callbacks
    callbacks: List[Callback] = []

    # Model checkpoint
    checkpoint_path = output_dir / "checkpoints" / f"best_{args.checkpoint_monitor}.pth"
    callbacks.append(
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=args.checkpoint_monitor,
            mode=args.checkpoint_mode,
            save_best_only=True,
            verbose=1,
        )
    )

    # Early stopping
    if args.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=args.checkpoint_monitor,
                patience=args.patience,
                mode=args.checkpoint_mode,
                restore_best_weights=True,
                verbose=1,
            )
        )
        logger.info(f"Early stopping enabled (patience={args.patience})")

    # Learning rate scheduler
    if args.scheduler != "none":
        scheduler, monitor = setup_scheduler(args, optimizer)
        if scheduler is not None:
            callbacks.append(
                LearningRateScheduler(scheduler, monitor=monitor, verbose=1)
            )

    # Setup trainer
    logger.info("\n" + "=" * 80)
    logger.info("Setting up trainer")
    logger.info("=" * 80)

    compile_kwargs = {"mode": args.compile_mode} if args.compile_mode else None

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=args.device,
        callbacks=callbacks,
        metrics=["psnr", "ssim"],
        gradient_clip_val=args.gradient_clip,
        mixed_precision=args.mixed_precision,
        accumulation_steps=args.accumulation_steps,
        use_ema=args.ema,
        ema_decay=args.ema_decay,
        ema_update_after_step=args.ema_update_after_step,
        ema_use_warmup=not args.no_ema_warmup,
        validate_with_ema=not args.no_validate_with_ema,
        compile_model=args.compile,
        compile_kwargs=compile_kwargs,
    )

    if args.ema:
        logger.info(
            f"EMA enabled (decay={args.ema_decay}, "
            f"update_after_step={args.ema_update_after_step}, "
            f"warmup={not args.no_ema_warmup}, "
            f"validate_with_ema={not args.no_validate_with_ema})"
        )
    if args.compile:
        logger.info(f"torch.compile() enabled (mode={args.compile_mode or 'default'})")

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume is not None:
        logger.info(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = trainer.load_checkpoint(args.resume)
        start_epoch = checkpoint.get("epoch", 0) + 1
        logger.info(f"Resuming from epoch {start_epoch}")

    # Train
    logger.info("\n" + "=" * 80)
    logger.info("Starting training loop")
    logger.info("=" * 80)

    try:
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            start_epoch=start_epoch,
        )

        # Save final checkpoint
        final_checkpoint = output_dir / "checkpoints" / "final.pth"
        trainer.save_checkpoint(final_checkpoint)
        logger.info(f"\nFinal checkpoint saved to {final_checkpoint}")

        # Plot training curves
        logger.info("\nPlotting training curves")
        plot_training_curves(
            train_history=history, save_path=output_dir / "training_curves.png"
        )
        logger.info(f"Training curves saved to {output_dir / 'training_curves.png'}")

        logger.info("\n" + "=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        trainer.save_checkpoint(output_dir / "checkpoints" / "interrupted.pth")
        logger.info(
            f"Checkpoint saved to {output_dir / 'checkpoints' / 'interrupted.pth'}"
        )

    except Exception as e:
        logger.error(f"\nTraining failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
