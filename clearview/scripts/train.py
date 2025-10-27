#!/usr/bin/env python
"""Training script for image deraining models."""

import argparse
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    MultiStepLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.data import DataLoader

from clearview.data import (
    ImagePairDataset,
    Rain100Dataset,
    get_train_transforms,
    get_val_transforms,
)
from clearview.losses import CombinedLoss
from clearview.models import get_model
from clearview.training import (
    Callback,
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    Trainer,
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
        choices=["pair", "rain100"],
        help="Dataset type",
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

    # Model arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=["unet", "attention_unet"],
        help="Model architecture",
    )
    model_group.add_argument("--in-channels", type=int, default=3, help="Number of input channels")
    model_group.add_argument("--out-channels", type=int, default=3, help="Number of output channels")

    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    train_group.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    train_group.add_argument("--val-batch-size", type=int, default=8, help="Validation batch size")
    train_group.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
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
    optim_group.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    optim_group.add_argument("--momentum", type=float, default=0.9, help="Momentum (for SGD)")

    # Loss arguments
    loss_group = parser.add_argument_group("Loss")
    loss_group.add_argument(
        "--loss",
        type=str,
        default="l1_l2_ssim_edge",
        choices=["l1", "l2", "l1_l2_ssim", "l1_l2_ssim_edge", "l1_l2_ssim_edge_perceptual", "custom"],
        help="Loss function",
    )
    loss_group.add_argument("--l1-weight", type=float, default=1.0, help="L1 loss weight")
    loss_group.add_argument("--l2-weight", type=float, default=1.0, help="L2 loss weight")
    loss_group.add_argument("--ssim-weight", type=float, default=1.0, help="SSIM loss weight")
    loss_group.add_argument("--edge-weight", type=float, default=0.5, help="Edge loss weight")
    loss_group.add_argument("--vgg-weight", type=float, default=0.5, help="VGG loss weight")

    # Augmentation arguments
    aug_group = parser.add_argument_group("Augmentation")
    aug_group.add_argument("--crop-size", type=int, default=256, help="Random crop size")
    aug_group.add_argument("--flip-prob", type=float, default=0.5, help="Probability of flipping")
    aug_group.add_argument("--no-rotation", action="store_true", help="Disable random rotation")

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

    # Callbacks arguments
    callback_group = parser.add_argument_group("Callbacks")
    callback_group.add_argument("--early-stopping", action="store_true", help="Enable early stopping")
    callback_group.add_argument("--patience", type=int, default=15, help="Early stopping patience")
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
    opt_group.add_argument("--mixed-precision", action="store_true", help="Use automatic mixed precision")
    opt_group.add_argument("--gradient-clip", type=float, default=None, help="Gradient clipping value")

    # Output arguments
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and logs",
    )
    output_group.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs")

    # Resuming
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

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

    # Create datasets
    if args.dataset_type == "rain100":
        train_dataset = Rain100Dataset(root_dir=data_dir / "train", transform=train_transform)
        val_dataset = Rain100Dataset(root_dir=data_dir / "val", transform=val_transform)
    else:  # pair
        train_dataset = ImagePairDataset(
            rainy_dir=data_dir / args.train_rainy,
            clean_dir=data_dir / args.train_clean,
            transform=train_transform,
        )
        val_dataset = ImagePairDataset(
            rainy_dir=data_dir / args.val_rainy,
            clean_dir=data_dir / args.val_clean,
            transform=val_transform,
        )

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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
    model = get_model(args.model, in_channels=args.in_channels, out_channels=args.out_channels)

    logger.info(f"Model: {args.model}")
    logger.info(f"Parameters: {model.get_num_params():,}")
    logger.info(f"Model size: {model.get_model_size_mb():.2f} MB")

    return model


def setup_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    """Setup optimizer."""
    if args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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


def setup_loss(args: argparse.Namespace) -> nn.Module:
    """Setup loss function."""
    if args.loss == "l1":
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


def setup_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer) -> Tuple[Optional[Any], Optional[str]]:
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
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, verbose=True)
        monitor = None
    elif args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1, verbose=True)
        monitor = None
    elif args.scheduler == "multistep":
        scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1, verbose=True)
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

    # Regular checkpoints
    callbacks.append(
        ModelCheckpoint(
            filepath=output_dir / "checkpoints" / "epoch_{epoch:03d}.pth",
            save_best_only=False,
            verbose=0,
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
            callbacks.append(LearningRateScheduler(scheduler, monitor=monitor, verbose=1))

    # Setup trainer
    logger.info("\n" + "=" * 80)
    logger.info("Setting up trainer")
    logger.info("=" * 80)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=args.device,
        callbacks=callbacks,
        metrics=["psnr", "ssim"],
        gradient_clip_val=args.gradient_clip,
        mixed_precision=args.mixed_precision,
    )

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
        plot_training_curves(train_history=history, save_path=output_dir / "training_curves.png")
        logger.info(f"Training curves saved to {output_dir / 'training_curves.png'}")

        logger.info("\n" + "=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        trainer.save_checkpoint(output_dir / "checkpoints" / "interrupted.pth")
        logger.info(f"Checkpoint saved to {output_dir / 'checkpoints' / 'interrupted.pth'}")

    except Exception as e:
        logger.error(f"\nTraining failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
