# ClearView command-line tools

Installing the package (`pip install -e ".[metrics,hub]"` from the repo
root) provides three commands:

| Command               | Module                           | Purpose                                  |
| --------------------- | -------------------------------- | ---------------------------------------- |
| `clearview-train`     | `clearview/scripts/train.py`     | Train a model, or resume a run           |
| `clearview-evaluate`  | `clearview/scripts/evaluate.py`  | Score a checkpoint on a labeled test set |
| `clearview-inference` | `clearview/scripts/inference.py` | Derain images, a folder, or a video      |

Each also runs as `python -m clearview.scripts.<name>`. `--help` lists every
flag with its default. This page covers the ones you'll use most.

**Models** (`--model`): `unet`, `attention_unet`, `resnet_unet` (with
`--backbone`), `resnet18_unet` / `resnet34_unet` / `resnet50_unet` /
`resnet101_unet` / `resnet152_unet`, `nafnet_small` / `nafnet` /
`nafnet_large`, `restormer_small` / `restormer` / `restormer_large`, and
`histoformer` (inference and evaluation only).

**Dataset types** (`--dataset-type`): `pair` (any rainy/clean folders with
matching filenames), `rain100`, `rain1400` (DDN-Data), `spa-data`.

---

## Training

```bash
clearview-train \
  --model nafnet --data-dir data/Rain100L --output-dir runs/nafnet_rain100l \
  --train-rainy train/rainy --train-clean train/clean \
  --val-rainy val/rainy --val-clean val/clean \
  --loss custom --loss-config '{"charbonnier": {"weight": 1.0}}' \
  --optimizer adamw --lr 1e-4 --scheduler cosine --warmup-epochs 5 \
  --epochs 100 --batch-size 8 --crop-size 256 \
  --mixed-precision --ema --early-stopping --patience 15
```

To train on several datasets at once, use `--mix-config` / `--val-mix-config`.
The recipe behind the published checkpoints is in
[`configs/mix/README.md`](../../configs/mix/README.md).

| Group        | Flags                                                                                                                                                                                                                             |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Data         | `--data-dir`, `--dataset-type`, `--train-rainy/--train-clean/--val-rainy/--val-clean`, `--train-split/--val-split` (spa-data), `--mix-config`, `--mix-sampler`, `--val-mix-config`                                                |
| Model        | `--model`, `--backbone` (`resnet_unet` only), `--pretrained/--no-pretrained` (ResNet-UNets), `--freeze-encoder`                                                                                                                   |
| Loss         | `--loss` preset (`l1`, `l2`, `l1_l2_ssim`, `l1_l2_ssim_edge`, `l1_l2_ssim_edge_perceptual`), or `--loss custom` with `--loss-config` (JSON) / `--loss-config-file` (JSON/YAML)                                                    |
| Optimization | `--optimizer`, `--lr`, `--weight-decay`, `--scheduler` (`plateau`, `cosine`, `step`, `multistep`, `none`), `--warmup-epochs`, `--accumulation-steps`, `--gradient-clip`, `--mixed-precision`, `--compile`, `--ema`, `--ema-decay` |
| Checkpoints  | `--checkpoint-monitor`, `--checkpoint-mode`, `--early-stopping`, `--patience`, `--save-every`, `--resume`                                                                                                                         |
| Other        | `--seed`, `--config` (YAML of flag values), `--device`                                                                                                                                                                            |

`--loss-config` accepts any loss in `CombinedLoss`: `l1`/`mae`, `l2`/`mse`,
`charbonnier`, `ssim`, `ms_ssim`, `edge`, `sobel`, `laplacian`,
`perceptual`/`vgg`, `dists`, `fft`, `focal_frequency`, `wavelet`, `color`,
each with its own `weight`. (`adversarial` needs a discriminator module, so
it is only usable from Python, not from the CLI.)

**Config files.** `--config run.yaml` sets any flag (`batch_size: 16` or
`batch-size: 16`). Precedence is built-in defaults, then the file, then flags
given on the command line. Unknown keys are an error.

**Outputs** under `--output-dir`:

- `checkpoints/best_<monitor>.pth`: the best epoch so far (EMA weights if `--ema`)
- `checkpoints/last.pth`: full resumable state, every `--save-every` epochs
- `checkpoints/final.pth`: end of training (best weights if early stopping restored them)
- `checkpoints/interrupted.pth`: written on Ctrl+C
- `training.log`, `training_curves.png`

**Resuming.** `--resume runs/x/checkpoints/last.pth` (or `interrupted.pth`)
with the same arguments continues from the next epoch. The optimizer, LR
schedule, AMP scaler, EMA, best-checkpoint value and early-stopping patience
all carry over.

---

## Evaluation

```bash
clearview-evaluate \
  --model nafnet --weights runs/nafnet_rain100l/checkpoints/best_val_psnr.pth \
  --data-dir data/Rain100L/test --dataset-type pair \
  --rainy-dir rainy --clean-dir clean \
  --output-dir runs/nafnet_rain100l/eval --save-comparison-grid
```

- `--metrics` defaults to all of `psnr ssim mae mse lpips dists brisque rain_removal_rate niqe`.
  `lpips` needs the `lpips` package, `dists`/`brisque` need `piq`, and `niqe`
  needs `scipy` (all in the `metrics` extra). Missing ones are skipped with a warning.
- `--compute-fid` adds a dataset-level FID score (needs `piq`).
- Images are evaluated at native resolution, so keep `--batch-size 1` when
  sizes vary. A sample that runs out of GPU memory is retried on the CPU.
- PSNR and SSIM are computed in RGB on the [0, 1]-clipped output, not on
  the Y channel.

Outputs: `results.json` (mean, std, min, max per metric), `evaluation.log`,
`metric_distributions.png`, plus `comparison_grid.png` and `images/` when
`--save-comparison-grid` / `--save-images` are given.

---

## Inference

```bash
# one image (or a video: .mp4 .avi .mov .mkv .webm, derained frame by frame)
clearview-inference --model nafnet --weights best_val_psnr.pth \
  --input rainy.jpg --output derained.jpg

# a folder of images
clearview-inference --model nafnet --weights best_val_psnr.pth \
  --input-dir rainy_photos/ --output-dir derained_photos/ --recursive
```

`--input` pairs with `--output`, and `--input-dir` with `--output-dir`.
Other flags: `--save-comparison` (side-by-side input and output),
`--benchmark` (timing), `--report-quality-metrics` (no-reference BRISQUE,
needs `piq`), `--extensions`, `--device`.

Video output has no temporal-consistency term, so some frame-to-frame
flicker is expected.

---

## Troubleshooting

- **CUDA out of memory in training**: lower `--batch-size` and raise
  `--accumulation-steps` to keep the effective batch size, or lower
  `--crop-size`.
- **Slow data loading**: raise `--num-workers`.
- **`--compile` fails**: training falls back to eager mode with a warning.
  Drop the flag to silence it.
