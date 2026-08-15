#!/usr/bin/env bash
# Evaluate the mixed-training UNet checkpoint across all local test sets:
# SPA-Data val (the checkpoint's own monitored metric), Rain13K's 5
# benchmarks, DDN-Data's test split, RealRain-1k-H/L test splits, and
# AllWeather rain+fog (cross-domain stress test). Run from the clearview
# repo root, with the clearview venv active.
#
# Paths match the unzipped contents of mixed_datasets.zip and
# eval_datasets.zip exactly (see configs/mix/trainer.md §1 and §5b) -- same
# structure this script would use on the server, just with local base dirs.
set -euo pipefail

WEIGHTS="${1:-./runs/rain_mixed_unet/checkpoints/best_val_psnr.pth}"
OUT="./runs/rain_mixed_unet/eval"
M="/home/neural_debugger/Downloads/datasets/clearview/mixed_datasets"
E="/home/neural_debugger/Downloads/datasets/clearview/eval_datasets"

if [ ! -f "$WEIGHTS" ]; then
  echo "Checkpoint not found: $WEIGHTS" >&2
  echo "Usage: $0 [path/to/checkpoint.pth]" >&2
  exit 1
fi

echo "=== Evaluating UNet checkpoint: $WEIGHTS ==="

# SPA-Data val (the checkpoint metric itself -- sanity check against the
# training log's number, not new information)
clearview-evaluate --model unet --weights "$WEIGHTS" \
  --data-dir "$M/spa_data/val" --dataset-type spa-data \
  --output-dir "$OUT/SPA-Data_val" --device cuda

# Rain100L (from mixed_datasets.zip, the synthetic sanity-check val used
# during training)
clearview-evaluate --model unet --weights "$WEIGHTS" \
  --data-dir "$M/rain100l_test" \
  --dataset-type pair --rainy-dir input --clean-dir target \
  --output-dir "$OUT/Rain100L" --device cuda

# Rain13K's other 4 benchmarks (from eval_datasets.zip)
for SET in rain100h test100 test1200 test2800; do
  clearview-evaluate --model unet --weights "$WEIGHTS" \
    --data-dir "$E/${SET}_test" \
    --dataset-type pair --rainy-dir input --clean-dir target \
    --output-dir "$OUT/$SET" --device cuda
done

# DDN-Data test split (1,400 pairs, 14 rain variants per clean image)
clearview-evaluate --model unet --weights "$WEIGHTS" \
  --data-dir "$E/ddn_data_test" \
  --dataset-type rain1400 --rainy-dir rainy_image --clean-dir ground_truth \
  --output-dir "$OUT/DDN_Data" --device cuda

# RealRain-1k-H / -L held-out test splits (224 pairs each)
clearview-evaluate --model unet --weights "$WEIGHTS" \
  --data-dir "$E/realrain1k_h_test" \
  --dataset-type pair --rainy-dir input --clean-dir target \
  --output-dir "$OUT/RealRain1k_H_test" --device cuda

clearview-evaluate --model unet --weights "$WEIGHTS" \
  --data-dir "$E/realrain1k_l_test" \
  --dataset-type pair --rainy-dir input --clean-dir target \
  --output-dir "$OUT/RealRain1k_L_test" --device cuda

# AllWeather rain+fog -- the real stress test, this is what broke the
# single-dataset Restormer badly
clearview-evaluate --model unet --weights "$WEIGHTS" \
  --data-dir "$E/allweather_rain_test" \
  --dataset-type pair --rainy-dir input --clean-dir gt \
  --output-dir "$OUT/AllWeather_rain" --device cuda

echo "=== Done. Results under $OUT/*/results.json ==="
