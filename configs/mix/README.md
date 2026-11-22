# Mixed-domain training recipe

This directory holds the exact recipe behind the published ClearView
checkpoints:

| File                             | Purpose                                                                   |
| -------------------------------- | ------------------------------------------------------------------------- |
| `rain_mixed_synthetic_real.yaml` | `--mix-config`: 5-source training mix, real-world sources weighted 2x     |
| `rain_mixed_val.yaml`            | `--val-mix-config`: 4-source blended validation set for checkpoint choice |
| `eval_mixed.sh`                  | Evaluates one checkpoint on every benchmark in the README's results table |

`data_dir` entries in both YAML files are relative to the dataset root you
pass as `--data-dir`, so the same files work on any machine.

## 1. Build the dataset roots

Every training and validation source is available from the ClearView
Hugging Face mirrors. Download them once, then symlink them into the layout
the configs expect:

```bash
DL=/path/to/hf_downloads          # where the raw mirrors go
MIX=/path/to/mixed_datasets       # --data-dir for training
EVAL=/path/to/eval_datasets       # EVAL_ROOT for eval_mixed.sh

for repo in Rain13K DDN-Data SPA-Data RealRain-1k; do
  hf download "dronefreak/$repo" --repo-type dataset --local-dir "$DL/$repo"
done

mkdir -p "$MIX/rain13k" "$MIX/ddn_data" "$EVAL"
ln -s "$DL/Rain13K/train"                  "$MIX/rain13k/train"
ln -s "$DL/Rain13K/test/Rain100L"          "$MIX/rain100l_test"
ln -s "$DL/DDN-Data/training"              "$MIX/ddn_data/training"
ln -s "$DL/SPA-Data"                       "$MIX/spa_data"
ln -s "$DL/RealRain-1k/RealRain-1k-H"      "$MIX/realrain1k_h"
ln -s "$DL/RealRain-1k/RealRain-1k-L"      "$MIX/realrain1k_l"

for set in Rain100H Test100 Test1200 Test2800; do
  ln -s "$DL/Rain13K/test/$set" "$EVAL/$(echo "$set" | tr 'A-Z' 'a-z')_test"
done
ln -s "$DL/DDN-Data/testing"               "$EVAL/ddn_data_test"
ln -s "$DL/RealRain-1k/RealRain-1k-H/test" "$EVAL/realrain1k_h_test"
ln -s "$DL/RealRain-1k/RealRain-1k-L/test" "$EVAL/realrain1k_l_test"
```

The AllWeather rain+fog test set (Outdoor-Rain, Li et al. CVPR 2019) is not
mirrored. Place it at `$EVAL/allweather_rain_test/{input,gt}` yourself, or
`eval_mixed.sh` skips it.

## 2. Train

All published models were trained with the command below. Only the
per-model values in the table differ:

```bash
clearview-train \
  --model MODEL \
  --mix-config configs/mix/rain_mixed_synthetic_real.yaml --mix-sampler \
  --val-mix-config configs/mix/rain_mixed_val.yaml \
  --data-dir "$MIX" \
  --loss custom --loss-config '{"charbonnier": {"weight": 1.0}}' \
  --crop-size 256 --batch-size BATCH --val-batch-size BATCH --num-workers 8 \
  --optimizer adamw --lr 1e-4 --scheduler cosine --warmup-epochs 5 \
  --epochs EPOCHS --early-stopping --patience PATIENCE \
  --checkpoint-monitor val_psnr --checkpoint-mode max \
  --mixed-precision --ema --ema-decay 0.999 --compile \
  --seed 42 --output-dir ./runs/rain_mixed_MODEL --device cuda
```

| `--model`         | `BATCH` | `EPOCHS` | `PATIENCE` |
| ----------------- | ------- | -------- | ---------- |
| `restormer`       | 4       | 100      | 15         |
| `restormer_small` | 32      | 100      | 15         |
| `unet`            | 24      | 300      | 50         |
| `nafnet_small`    | 64      | 300      | 15         |
| `nafnet`          | 6       | 100      | 15         |
| `nafnet_large`    | 16      | 100      | 15         |
| `resnet18_unet`   | 24      | 100      | 50         |
| `resnet34_unet`   | 64      | 100      | 15         |
| `resnet50_unet`   | 32      | 100      | 15         |

ResNet-UNet encoders start from ImageNet weights (`--pretrained`, the
default). Batch sizes were chosen for 24GB to 48GB GPUs. On smaller cards,
lower `BATCH` and add `--accumulation-steps` to keep the effective batch size.

The best checkpoint (by blended validation PSNR) is written to
`checkpoints/best_val_psnr.pth`. A full resumable checkpoint is written to
`checkpoints/last.pth` every `--save-every` epochs (default 10), and to
`checkpoints/interrupted.pth` on Ctrl+C. Resume with `--resume <path>` and
the same arguments.

## 3. Evaluate

```bash
MIXED_ROOT="$MIX" EVAL_ROOT="$EVAL" \
  ./configs/mix/eval_mixed.sh ./runs/rain_mixed_restormer/checkpoints/best_val_psnr.pth restormer
```

Results land in `./runs/rain_mixed_<model>/eval/<dataset>/results.json`.
PSNR and SSIM are computed in RGB on the model's full-resolution output
(clipped to [0, 1], not quantized to 8 bit), not on the Y channel that many
deraining papers report, so expect them to differ from Y-channel numbers.

The blended metric used for checkpoint selection averages SPA-Data val
capped to its first 150 pairs. `eval_mixed.sh` reports SPA-Data val on all
1,000 pairs, so averaging its per-source numbers won't exactly reproduce the
training-time blended value.
