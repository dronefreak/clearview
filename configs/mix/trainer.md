# Mixed synthetic + real rain training

Commands for unzipping `mixed_datasets.zip` on the server and launching the
mixed-data training run across the three architectures we're comparing with
the same methodology: same 5-source training mix (`configs/mix/rain_mixed_synthetic_real.yaml`),
same mild real-data oversampling (`--mix-sampler`), same blended 4-source
validation set for checkpoint selection (`configs/mix/rain_mixed_val.yaml`,
via `--val-mix-config`), same Charbonnier-only loss, same crop
size/schedule/EMA/compile settings. Only `--batch-size`/`--accumulation-steps`
vary per architecture, sized for a 24GB card (RTX A5000).

`--val-mix-config` blends SPA-Data val (capped to 150 of its 1,000 pairs),
RealRain-1k-H/L validation (112 each), and Rain100L (100) into one
checkpoint-selection metric, so "best" means "doesn't fail badly anywhere,"
not "maxes out SPA-Data specifically." Smoke-tested end-to-end (real data,
real `Trainer`, real GPU) before this doc was updated to use it.

**Before running any of these for real**: time one epoch first (see the note
at the bottom) rather than trusting the batch sizes below blind, they're
sized from a Restormer memory measurement at a smaller crop, extrapolated,
not independently verified on this exact hardware.

---

## 1. Unzip the dataset

```bash
cd /home/saumya.saksena/projects
unzip mixed_datasets.zip -d mixed_datasets
```

This expects `mixed_datasets.zip` to already be at
`/home/saumya.saksena/projects/mixed_datasets.zip`, and unpacks to
`/home/saumya.saksena/projects/mixed_datasets/{rain13k,ddn_data,spa_data,realrain1k_h,realrain1k_l}`
matching the exact layout `configs/mix/rain_mixed_synthetic_real.yaml` expects.

Also make sure the `clearview` checkout on the server has the `--mix-config`/
`--mix-sampler`/`--val-mix-config` flags (this branch's code), they don't
exist in any published version yet.

---

## 2. Restormer -- resuming with `--val-mix-config`

This run was already going (SPA-Data-only val, epoch 14 at 40.3 PSNR when
last checked). Switching to the blended val requires a resume, not a
relaunch, to keep the epoch count/optimizer/scheduler/EMA state continuous.

**Stop it gracefully first** -- Ctrl+C in the terminal it's running in, or
`kill -SIGINT <pid>` if it's backgrounded (`kill -9` will _not_ save a
resumable checkpoint, don't use it here). This triggers the
`except KeyboardInterrupt` handler in `train.py`, which saves
`checkpoints/interrupted.pth` with full state (model, optimizer, epoch, EMA)
-- unlike `checkpoints/best_val_psnr.pth`, which is just the best single
epoch snapshot, `interrupted.pth` is what actually resumes cleanly.

**Back up the current best checkpoint before resuming** -- once training
starts using the new blended metric, `ModelCheckpoint`'s "best so far"
tracking resets to `-inf` (it's per-process state, not saved/restored by
`--resume`), so the very next epoch will unconditionally overwrite
`best_val_psnr.pth`, even before the blended metric has actually beaten
anything. If you want to keep the SPA-Data-only-optimized snapshot as a
reference point, copy it aside first:

```bash
cp ./runs/rain_mixed_restormer/checkpoints/best_val_psnr.pth \
   ./runs/rain_mixed_restormer/checkpoints/best_val_psnr_spa_only.pth
```

```bash
clearview-train \
  --model restormer \
  --resume ./runs/rain_mixed_restormer/checkpoints/interrupted.pth \
  --mix-config configs/mix/rain_mixed_synthetic_real.yaml --mix-sampler \
  --val-mix-config configs/mix/rain_mixed_val.yaml \
  --data-dir /home/saumya.saksena/projects/mixed_datasets \
  --loss custom --loss-config '{"charbonnier": {"weight": 1.0}}' \
  --crop-size 256 --batch-size 4 --accumulation-steps 1 --val-batch-size 4 --num-workers 8 \
  --optimizer adamw --lr 1e-4 --scheduler cosine --warmup-epochs 5 \
  --epochs 100 --early-stopping --patience 15 \
  --checkpoint-monitor val_psnr --checkpoint-mode max \
  --mixed-precision --ema --ema-decay 0.999 --compile \
  --output-dir ./runs/rain_mixed_restormer \
  --device cuda
```

`--batch-size 4 --accumulation-steps 1` (true batch 4, no accumulation crutch),
the 24GB A5000 should comfortably beat the 12GB card's batch=2+accum=2
setup. 15.3M params. `EarlyStopping`'s patience counter also resets fresh on
resume (same reason as `ModelCheckpoint` above) -- it gets a full new
15-epoch budget from wherever this resumes, not the leftover count from the
SPA-only run.

---

## 3. UNet -- fresh run, 300 epochs

The single-dataset UNet was still visibly converging at 100 epochs (38 PSNR
on SPA-Data val, not yet plateaued), so this one launches from scratch with
a longer budget rather than resuming anything.

```bash
clearview-train \
  --model unet \
  --mix-config configs/mix/rain_mixed_synthetic_real.yaml --mix-sampler \
  --val-mix-config configs/mix/rain_mixed_val.yaml \
  --data-dir /home/saumya.saksena/projects/mixed_datasets \
  --loss custom --loss-config '{"charbonnier": {"weight": 1.0}}' \
  --crop-size 256 --batch-size 24 --val-batch-size 24 --num-workers 8 \
  --optimizer adamw --lr 1e-4 --scheduler cosine --warmup-epochs 5 \
  --epochs 300 --early-stopping --patience 15 \
  --checkpoint-monitor val_psnr --checkpoint-mode max \
  --mixed-precision --ema --ema-decay 0.999 --compile \
  --output-dir ./runs/rain_mixed_unet \
  --device cuda
```

`--batch-size 24` matches the batch size the original single-dataset UNet
recipe already ran successfully on a 12GB card (21.5M params, but plain
convolutions are far cheaper than Restormer's attention at the same param
count), the A5000 has ample headroom here, this one is low-risk.
`--patience 15` still applies against the full 300-epoch budget, so this
stops itself if it plateaus well before 300 -- it's a ceiling, not a target.

Note: this uses `use_transpose_conv=False` (UNet's current default,
bilinear upsampling) since it's a fresh training run, not loading the older
`clearview-derain-unet` checkpoint that needed the transpose-conv override.

---

## 4. NAFNet -- resuming with `--val-mix-config`

Same situation as Restormer: already running (SPA-Data-only val, epoch 49 at
41.6 PSNR when last checked, ~10 min/epoch). `--batch-size 6` turned out fine
in practice over those 49 real epochs -- the earlier "not smoke-tested"
caveat no longer applies.

**Stop it gracefully first** (Ctrl+C / `kill -SIGINT <pid>`, not `kill -9`)
so `checkpoints/interrupted.pth` gets saved with full resumable state --
same reasoning as Restormer above.

**Back up the current best checkpoint before resuming**, for the same
reason as Restormer (`ModelCheckpoint`'s "best so far" resets to `-inf` on
resume, so the next epoch unconditionally overwrites `best_val_psnr.pth`):

```bash
cp ./runs/rain_mixed_nafnet/checkpoints/best_val_psnr.pth \
   ./runs/rain_mixed_nafnet/checkpoints/best_val_psnr_spa_only.pth
```

```bash
clearview-train \
  --model nafnet \
  --resume ./runs/rain_mixed_nafnet/checkpoints/interrupted.pth \
  --mix-config configs/mix/rain_mixed_synthetic_real.yaml --mix-sampler \
  --val-mix-config configs/mix/rain_mixed_val.yaml \
  --data-dir /home/saumya.saksena/projects/mixed_datasets \
  --loss custom --loss-config '{"charbonnier": {"weight": 1.0}}' \
  --crop-size 256 --batch-size 6 --accumulation-steps 1 --val-batch-size 6 --num-workers 8 \
  --optimizer adamw --lr 1e-4 --scheduler cosine --warmup-epochs 5 \
  --epochs 100 --early-stopping --patience 15 \
  --checkpoint-monitor val_psnr --checkpoint-mode max \
  --mixed-precision --ema --ema-decay 0.999 --compile \
  --output-dir ./runs/rain_mixed_nafnet \
  --device cuda
```

`--model nafnet` is the mid-size variant (14.3M params, comparable to
Restormer's 15.3M), not `nafnet_small` (1.1M) or `nafnet_large` (116M).
`EarlyStopping`'s patience counter resets fresh on resume too, same as
Restormer -- full new 15-epoch budget from wherever this picks back up.

---

## 4a. NAFNet (Small) -- new run, 48GB (RTX A6000)

`nafnet_small` is the smallest NAFNet variant, 1.1M params, well under a
tenth the size of the mid-size `nafnet` already running. This is a fresh
run, not a resume, and its own separate output directory so it doesn't
collide with the mid-size run above.

**Sized for the 48GB A6000, not the 24GB A5000 the rest of this doc
targets.** Everything else in this section 4a/4b pair assumes double the
VRAM budget the other entries were sized for.

```bash
clearview-train \
  --model nafnet_small \
  --mix-config configs/mix/rain_mixed_synthetic_real.yaml --mix-sampler \
  --val-mix-config configs/mix/rain_mixed_val.yaml \
  --data-dir /home/saumya.saksena/projects/mixed_datasets \
  --loss custom --loss-config '{"charbonnier": {"weight": 1.0}}' \
  --crop-size 256 --batch-size 64 --val-batch-size 64 --num-workers 8 \
  --optimizer adamw --lr 1e-4 --scheduler cosine --warmup-epochs 5 \
  --epochs 100 --early-stopping --patience 15 \
  --checkpoint-monitor val_psnr --checkpoint-mode max \
  --mixed-precision --ema --ema-decay 0.999 --compile \
  --output-dir ./runs/rain_mixed_nafnet_small \
  --device cuda
```

`--batch-size 64` is an estimate, not a measurement, doubled from the
24GB-sized estimate (32) rather than independently derived, no smoke test
has been run against this specific variant on either card. 1.1M params is
small enough that it should comfortably fit, but "should" is still doing
real work in that sentence. Time one epoch before trusting the full
100-epoch budget, same as every other unverified number in this doc.

---

## 4b. NAFNet (Large) -- new run, 48GB (RTX A6000)

`nafnet_large` is the biggest NAFNet variant, 116M params, about 8x the
mid-size `nafnet` and roughly 7.5x Restormer. This is the one entry in this
whole document I'd treat as genuinely uncertain rather than a reasonable
estimate: nothing at this parameter scale has been smoke-tested or run at
all this session, and the batch size below is a conservative guess sized to
avoid an obvious OOM, not a number backed by any measurement.

```bash
clearview-train \
  --model nafnet_large \
  --mix-config configs/mix/rain_mixed_synthetic_real.yaml --mix-sampler \
  --val-mix-config configs/mix/rain_mixed_val.yaml \
  --data-dir /home/saumya.saksena/projects/mixed_datasets \
  --loss custom --loss-config '{"charbonnier": {"weight": 1.0}}' \
  --crop-size 256 --batch-size 4 --accumulation-steps 1 --val-batch-size 4 --num-workers 8 \
  --optimizer adamw --lr 1e-4 --scheduler cosine --warmup-epochs 5 \
  --epochs 100 --early-stopping --patience 15 \
  --checkpoint-monitor val_psnr --checkpoint-mode max \
  --mixed-precision --ema --ema-decay 0.999 --compile \
  --output-dir ./runs/rain_mixed_nafnet_large \
  --device cuda
```

`--batch-size 4 --accumulation-steps 1` doubles the true batch size from
the 24GB-sized version (2, with accumulation-steps 2 to reach an effective
batch of 4) rather than just doubling the accumulation count, same "drop
the accumulation crutch once there's real headroom" pattern used for
Restormer and UNet earlier in this doc. Strongly recommend a real smoke
test (small subsets, 1 epoch, same pattern used for Restormer/UNet earlier)
before launching this one for real, not just a 1-epoch timing check. At
116M params, an OOM here wastes more time to discover than at any smaller
variant, and there's no measurement yet to say `--batch-size 4` is even
safe on 48GB, only that it's conservative relative to what 48GB could
plausibly support.

---

## Before committing the weekend to any of these

Time one real epoch on the actual server first, a 4070 Super and an A5000
are close enough on paper that guessing wastes more time than measuring:

```bash
# same command as above, with --epochs 1 (drop --early-stopping/--patience,
# irrelevant for a 1-epoch timing check)
```

Multiply the wall time by 100 for a real ETA before letting any of these run
unattended.

---

## 5. Evaluation, all test sets, all three models

Checkpoints (matching each `--output-dir` above):

```bash
RESTORMER_WEIGHTS=./runs/rain_mixed_restormer/checkpoints/best_val_psnr.pth
UNET_WEIGHTS=./runs/rain_mixed_unet/checkpoints/best_val_psnr.pth
NAFNET_WEIGHTS=./runs/rain_mixed_nafnet/checkpoints/best_val_psnr.pth
```

**Data availability note**: `mixed_datasets.zip` only shipped train + validation
splits (it was scoped for training, not a full benchmark sweep). Rain100L,
SPA-Data val, and RealRain-1k-H/L _validation_ are covered by that zip (§5a).
Everything else -- Rain100H/Test100/Test1200/Test2800, DDN-Data's actual test
split, RealRain-1k-H/L's _test_ splits (224 pairs each, distinct from
validation), and AllWeather rain+fog -- is now in a second archive,
`eval_datasets.zip` (§5b), built the same way (hardlink staging, junk-file
swept, verified pair counts) and sitting at
`/home/neural_debugger/Downloads/datasets/clearview/eval_datasets.zip` ready
to transfer to the server alongside it.

### 5a. Available now (already unzipped on the server)

**Note on the checkpoint metric**: since training now uses `--val-mix-config`
(§2-4), the number that actually picked "best" is a _blended_ PSNR across
SPA-Data val (capped 150) + RealRain-1k-H/L validation + Rain100L -- there's
no single `clearview-evaluate` call that reproduces that exact blended number
(evaluate.py doesn't support multi-source blending the way training now
does). The four commands below reproduce the _per-source_ numbers that went
into it; averaging them yourself will be close but not identical, since the
training-time blend capped SPA-Data to 150 of its 1,000 val pairs and these
run against the full 1,000.

```bash
for MODEL in restormer unet nafnet; do
  case $MODEL in
    restormer) WEIGHTS=$RESTORMER_WEIGHTS ;;
    unet)      WEIGHTS=$UNET_WEIGHTS ;;
    nafnet)    WEIGHTS=$NAFNET_WEIGHTS ;;
  esac
  OUT=./runs/rain_mixed_${MODEL}/eval

  # Rain100L (synthetic sanity check, and one of the 4 sources in the
  # blended checkpoint metric)
  clearview-evaluate --model $MODEL --weights $WEIGHTS \
    --data-dir /home/saumya.saksena/projects/mixed_datasets/rain100l_test \
    --dataset-type pair --rainy-dir input --clean-dir target \
    --output-dir $OUT/Rain100L --device cuda

  # SPA-Data val, full 1,000 pairs (the blended checkpoint metric only used
  # 150 of these -- see the note above)
  clearview-evaluate --model $MODEL --weights $WEIGHTS \
    --data-dir /home/saumya.saksena/projects/mixed_datasets/spa_data/val \
    --dataset-type spa-data \
    --output-dir $OUT/SPA-Data_val --device cuda

  # RealRain-1k-H / -L validation (112 pairs each -- not the held-out test
  # split, see the note above)
  clearview-evaluate --model $MODEL --weights $WEIGHTS \
    --data-dir /home/saumya.saksena/projects/mixed_datasets/realrain1k_h/validation \
    --dataset-type pair --rainy-dir input --clean-dir target \
    --output-dir $OUT/RealRain1k_H_val --device cuda

  clearview-evaluate --model $MODEL --weights $WEIGHTS \
    --data-dir /home/saumya.saksena/projects/mixed_datasets/realrain1k_l/validation \
    --dataset-type pair --rainy-dir input --clean-dir target \
    --output-dir $OUT/RealRain1k_L_val --device cuda
done
```

### 5b. From `eval_datasets.zip`

```bash
cd /home/saumya.saksena/projects
unzip eval_datasets.zip -d eval_datasets
```

Unpacks to
`/home/saumya.saksena/projects/eval_datasets/{rain100h_test,test100_test,test1200_test,test2800_test,ddn_data_test,realrain1k_h_test,realrain1k_l_test,allweather_rain_test}`.
Verified pair counts while building it (DDN-Data's 1400:100 rainy:clean ratio
is expected -- 14 rain variants per clean image, not a mismatch); swept for
sync-conflict/`.DS_Store` junk, none found this time.

```bash
for MODEL in restormer unet nafnet; do
  case $MODEL in
    restormer) WEIGHTS=$RESTORMER_WEIGHTS ;;
    unet)      WEIGHTS=$UNET_WEIGHTS ;;
    nafnet)    WEIGHTS=$NAFNET_WEIGHTS ;;
  esac
  OUT=./runs/rain_mixed_${MODEL}/eval
  E=/home/saumya.saksena/projects/eval_datasets

  # Rain13K's other 4 benchmarks
  for SET in rain100h test100 test1200 test2800; do
    clearview-evaluate --model $MODEL --weights $WEIGHTS \
      --data-dir "$E/${SET}_test" \
      --dataset-type pair --rainy-dir input --clean-dir target \
      --output-dir $OUT/$SET --device cuda
  done

  # DDN-Data test split (1,400 pairs)
  clearview-evaluate --model $MODEL --weights $WEIGHTS \
    --data-dir "$E/ddn_data_test" \
    --dataset-type rain1400 --rainy-dir rainy_image --clean-dir ground_truth \
    --output-dir $OUT/DDN_Data --device cuda

  # RealRain-1k-H / -L held-out test splits (224 pairs each)
  clearview-evaluate --model $MODEL --weights $WEIGHTS \
    --data-dir "$E/realrain1k_h_test" \
    --dataset-type pair --rainy-dir input --clean-dir target \
    --output-dir $OUT/RealRain1k_H_test --device cuda

  clearview-evaluate --model $MODEL --weights $WEIGHTS \
    --data-dir "$E/realrain1k_l_test" \
    --dataset-type pair --rainy-dir input --clean-dir target \
    --output-dir $OUT/RealRain1k_L_test --device cuda

  # AllWeather rain+fog -- cross-domain generalization stress test (this is
  # the one that broke the single-dataset Restormer badly; the whole point
  # of this run is checking whether mixed training fixes it)
  clearview-evaluate --model $MODEL --weights $WEIGHTS \
    --data-dir "$E/allweather_rain_test" \
    --dataset-type pair --rainy-dir input --clean-dir gt \
    --output-dir $OUT/AllWeather_rain --device cuda
done
```

Native-resolution sets (Test2800, DDN-Data, RealRain-1k) can OOM on large
outlier images at `--batch-size 1` (the evaluate.py default) -- this is
already handled with an automatic per-sample CPU fallback (see
`clearview/scripts/evaluate.py`), not something to work around manually.
