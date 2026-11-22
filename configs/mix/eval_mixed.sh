#!/usr/bin/env bash
# Evaluate one checkpoint (any architecture) on every benchmark in the
# README's results table: Rain100L, SPA-Data val, Rain13K's other four test
# sets, DDN-Data test, RealRain-1k-H/L test, and AllWeather rain+fog (if
# present). Run from the repo root with ClearView installed.
#
# Usage:
#   MIXED_ROOT=/path/to/mixed_datasets EVAL_ROOT=/path/to/eval_datasets \
#     ./configs/mix/eval_mixed.sh path/to/checkpoint.pth model-name
#
# See configs/mix/README.md for how to build both dataset roots from the
# ClearView Hugging Face mirrors. Results go to
# ./runs/rain_mixed_<model-name>/eval/<dataset>/results.json.
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: MIXED_ROOT=... EVAL_ROOT=... $0 path/to/checkpoint.pth model-name" >&2
  echo "  model-name is any clearview-evaluate --model choice," >&2
  echo "  e.g. unet, restormer, nafnet, nafnet_small, nafnet_large" >&2
  exit 1
fi

: "${MIXED_ROOT:?set MIXED_ROOT to the mixed training dataset root}"
: "${EVAL_ROOT:?set EVAL_ROOT to the evaluation dataset root}"

WEIGHTS="$1"
MODEL="$2"
OUT="./runs/rain_mixed_${MODEL}/eval"
DEVICE="${DEVICE:-cuda}"

if [ ! -f "$WEIGHTS" ]; then
  echo "Checkpoint not found: $WEIGHTS" >&2
  exit 1
fi

run_eval() {
  # run_eval <output-name> <clearview-evaluate args...>
  local name="$1"
  shift
  clearview-evaluate --model "$MODEL" --weights "$WEIGHTS" \
    --output-dir "$OUT/$name" --device "$DEVICE" "$@"
}

echo "=== Evaluating $MODEL checkpoint: $WEIGHTS ==="

# SPA-Data val, all 1,000 pairs
run_eval SPA-Data_val --data-dir "$MIXED_ROOT/spa_data/val" --dataset-type spa-data

# Rain100L test
run_eval Rain100L --data-dir "$MIXED_ROOT/rain100l_test" \
  --dataset-type pair --rainy-dir input --clean-dir target

# Rain13K's other four test sets
for SET in rain100h test100 test1200 test2800; do
  run_eval "$SET" --data-dir "$EVAL_ROOT/${SET}_test" \
    --dataset-type pair --rainy-dir input --clean-dir target
done

# DDN-Data test split (1,400 pairs, 14 rain variants per clean image)
run_eval DDN_Data --data-dir "$EVAL_ROOT/ddn_data_test" \
  --dataset-type rain1400 --rainy-dir rainy_image --clean-dir ground_truth

# RealRain-1k-H / -L held-out test splits (224 pairs each)
run_eval RealRain1k_H_test --data-dir "$EVAL_ROOT/realrain1k_h_test" \
  --dataset-type pair --rainy-dir input --clean-dir target
run_eval RealRain1k_L_test --data-dir "$EVAL_ROOT/realrain1k_l_test" \
  --dataset-type pair --rainy-dir input --clean-dir target

# AllWeather rain+fog: cross-domain stress test, not mirrored on HF
if [ -d "$EVAL_ROOT/allweather_rain_test" ]; then
  run_eval AllWeather_rain --data-dir "$EVAL_ROOT/allweather_rain_test" \
    --dataset-type pair --rainy-dir input --clean-dir gt
else
  echo "Skipping AllWeather: $EVAL_ROOT/allweather_rain_test not found" >&2
fi

echo "=== Done. Results under $OUT/*/results.json ==="
