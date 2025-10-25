# ClearView Scripts

Command-line tools for training, evaluation, and inference with ClearView models.

## Overview

- **`train.py`** - Train image deraining models
- **`evaluate.py`** - Evaluate trained models on test sets
- **`inference.py`** - Run inference on new images

---

## Installation

Make sure ClearView is installed:

```bash
pip install -e .  # From project root
```

---

## Training (`train.py`)

Train a model from scratch or resume training.

### Basic Usage

```bash
python scripts/train.py \
    --data-dir data/Rain100L \
    --output-dir experiments/unet_base \
    --model unet \
    --epochs 100 \
    --batch-size 8
```

### Full Example

```bash
python scripts/train.py \
    --data-dir data/Rain100L \
    --output-dir experiments/unet_attention \
    --model attention_unet \
    --loss l1_l2_ssim_edge \
    --optimizer adam \
    --lr 1e-3 \
    --epochs 100 \
    --batch-size 8 \
    --crop-size 256 \
    --early-stopping \
    --patience 15 \
    --mixed-precision \
    --scheduler plateau
```

### Resume Training

```bash
python scripts/train.py \
    --data-dir data/Rain100L \
    --output-dir experiments/unet_base \
    --model unet \
    --resume experiments/unet_base/checkpoints/last.pth
```

### Key Arguments

**Data:**

- `--data-dir`: Root directory with train/val splits
- `--dataset-type`: Dataset format (`pair`, `rain100`)

**Model:**

- `--model`: Architecture (`unet`, `attention_unet`)
- `--in-channels`, `--out-channels`: Channel configuration

**Training:**

- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--optimizer`: Optimizer type (`adam`, `adamw`, `sgd`)
- `--lr`: Learning rate
- `--loss`: Loss function preset

**Callbacks:**

- `--early-stopping`: Enable early stopping
- `--patience`: Early stopping patience
- `--checkpoint-monitor`: Metric to monitor for best checkpoint

**Optimization:**

- `--mixed-precision`: Use AMP for faster training
- `--gradient-clip`: Gradient clipping value

---

## Evaluation (`evaluate.py`)

Evaluate a trained model on a test set.

### Basic Usage

```bash
python scripts/evaluate.py \
    --model unet \
    --weights checkpoints/best_model.pth \
    --data-dir data/Rain100L/test \
    --output-dir results/test
```

### With Visualizations

```bash
python scripts/evaluate.py \
    --model attention_unet \
    --weights experiments/unet_attention/checkpoints/best_val_psnr.pth \
    --data-dir data/test \
    --output-dir results/evaluation \
    --save-images \
    --save-comparison-grid \
    --num-vis 20 \
    --metrics psnr ssim mae mse
```

### Key Arguments

**Model:**

- `--model`: Model architecture
- `--weights`: Path to trained weights

**Data:**

- `--data-dir`: Test data directory
- `--dataset-type`: Dataset format

**Metrics:**

- `--metrics`: List of metrics to compute (default: psnr ssim mae mse)

**Output:**

- `--save-images`: Save individual derained images
- `--save-comparison-grid`: Create comparison grid
- `--num-vis`: Number of images to visualize

### Output Files

- `results.json` - Quantitative metrics
- `evaluation.log` - Detailed log
- `comparison_grid.png` - Visual comparison
- `metric_distributions.png` - Metric histograms
- `images/` - Individual result images (if `--save-images`)

---

## Inference (`inference.py`)

Run inference on new rainy images.

### Single Image

```bash
python scripts/inference.py \
    --model unet \
    --weights checkpoints/best_model.pth \
    --input rainy_image.png \
    --output derained_image.png
```

### Directory (Batch Processing)

```bash
python scripts/inference.py \
    --model attention_unet \
    --weights checkpoints/best.pth \
    --input-dir data/rainy_images \
    --output-dir results/derained
```

### With Comparison Visualization

```bash
python scripts/inference.py \
    --model unet \
    --weights checkpoints/best_model.pth \
    --input rainy.png \
    --output derained.png \
    --save-comparison
```

### With Benchmarking

```bash
python scripts/inference.py \
    --model attention_unet \
    --weights checkpoints/best.pth \
    --input-dir test_images \
    --output-dir results \
    --benchmark
```

### Key Arguments

**Model:**

- `--model`: Model architecture
- `--weights`: Path to trained weights

**Input/Output:**

- `--input`: Single image path
- `--input-dir`: Directory of images
- `--output`: Output image path
- `--output-dir`: Output directory

**Options:**

- `--save-comparison`: Save side-by-side comparison
- `--benchmark`: Measure and report inference speed
- `--recursive`: Process subdirectories recursively
- `--device`: Device to use (`cuda`, `cpu`)

---

## Example Workflows

### Complete Training Pipeline

```bash
# 1. Train model
python scripts/train.py \
    --data-dir data/Rain100L \
    --output-dir experiments/exp1 \
    --model unet \
    --epochs 100 \
    --early-stopping

# 2. Evaluate on test set
python scripts/evaluate.py \
    --model unet \
    --weights experiments/exp1/checkpoints/best_val_psnr.pth \
    --data-dir data/Rain100L/test \
    --output-dir experiments/exp1/evaluation \
    --save-images \
    --save-comparison-grid

# 3. Run inference on new images
python scripts/inference.py \
    --model unet \
    --weights experiments/exp1/checkpoints/best_val_psnr.pth \
    --input-dir data/new_rainy_images \
    --output-dir experiments/exp1/inference \
    --benchmark
```

### Quick Test Run

```bash
# Small test run (10 epochs)
python scripts/train.py \
    --data-dir data/Rain100L \
    --output-dir experiments/test_run \
    --model unet \
    --epochs 10 \
    --batch-size 4

# Quick evaluation
python scripts/evaluate.py \
    --model unet \
    --weights experiments/test_run/checkpoints/best_val_psnr.pth \
    --data-dir data/Rain100L/test \
    --output-dir experiments/test_run/results \
    --num-vis 5
```

---

## Tips

### Performance

- Use `--mixed-precision` for 2-3x speedup on modern GPUs
- Increase `--num-workers` for faster data loading
- Use `--batch-size` as large as GPU memory allows

### Best Results

- Train for at least 50-100 epochs
- Use `--early-stopping` to prevent overfitting
- Try both `unet` and `attention_unet` models
- Experiment with different loss combinations

### Debugging

- Check logs in `<output-dir>/training.log`
- Monitor training curves in `<output-dir>/training_curves.png`
- Start with a small `--crop-size` (128) for faster iteration

---

## Common Issues

**CUDA Out of Memory**

```bash
# Reduce batch size
--batch-size 4

# Or reduce crop size
--crop-size 128
```

**Slow Training**

```bash
# Enable mixed precision
--mixed-precision

# Increase workers
--num-workers 8
```

**Poor Results**

```bash
# Train longer
--epochs 150

# Try different loss
--loss l1_l2_ssim_edge

# Use attention model
--model attention_unet
```

---

## Getting Help

For detailed argument descriptions:

```bash
python scripts/train.py --help
python scripts/evaluate.py --help
python scripts/inference.py --help
```
