# Mixed synthetic + real rain training

Commands for unzipping `mixed_datasets.zip` on the server and launching the
mixed-data training run across the three architectures we're comparing with
the same methodology: same 5-source mix (`configs/mix/rain_mixed_synthetic_real.yaml`),
same mild real-data oversampling (`--mix-sampler`), same Charbonnier-only
loss, same crop size/schedule/EMA/compile settings. Only `--batch-size`/
`--accumulation-steps` vary per architecture, sized for a 24GB card (RTX A5000).

**Before running any of these for real**: time one epoch first (see the note
at the bottom) rather than trusting the batch sizes below blind — they're
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
— the exact layout `configs/mix/rain_mixed_synthetic_real.yaml` expects.

Also make sure the `clearview` checkout on the server has the `--mix-config`/
`--mix-sampler` flags (this branch's code) — they don't exist in any
published version yet.

---

## 2. Restormer

```bash
clearview-train \
  --model restormer \
  --mix-config configs/mix/rain_mixed_synthetic_real.yaml --mix-sampler \
  --dataset-type spa-data --data-dir /home/saumya.saksena/projects/mixed_datasets/spa_data --val-split val \
  --loss custom --loss-config '{"charbonnier": {"weight": 1.0}}' \
  --crop-size 256 --batch-size 4 --accumulation-steps 1 --val-batch-size 4 --num-workers 8 \
  --optimizer adamw --lr 1e-4 --scheduler cosine --warmup-epochs 5 \
  --epochs 100 --early-stopping --patience 15 \
  --checkpoint-monitor val_psnr --checkpoint-mode max \
  --mixed-precision --ema --ema-decay 0.999 --compile \
  --output-dir ./runs/rain_mixed_restormer \
  --device cuda
```

`--batch-size 4 --accumulation-steps 1` (true batch 4, no accumulation crutch)
— the 24GB A5000 should comfortably beat the 12GB card's batch=2+accum=2
setup. 15.3M params.

---

## 3. UNet

```bash
clearview-train \
  --model unet \
  --mix-config configs/mix/rain_mixed_synthetic_real.yaml --mix-sampler \
  --dataset-type spa-data --data-dir /home/saumya.saksena/projects/mixed_datasets/spa_data --val-split val \
  --loss custom --loss-config '{"charbonnier": {"weight": 1.0}}' \
  --crop-size 256 --batch-size 24 --val-batch-size 24 --num-workers 8 \
  --optimizer adamw --lr 1e-4 --scheduler cosine --warmup-epochs 5 \
  --epochs 100 --early-stopping --patience 15 \
  --checkpoint-monitor val_psnr --checkpoint-mode max \
  --mixed-precision --ema --ema-decay 0.999 --compile \
  --output-dir ./runs/rain_mixed_unet \
  --device cuda
```

`--batch-size 24` matches the batch size the original single-dataset UNet
recipe already ran successfully on a 12GB card (21.5M params, but plain
convolutions are far cheaper than Restormer's attention at the same param
count) — the A5000 has ample headroom here, this one is low-risk.

Note: this uses `use_transpose_conv=False` (UNet's current default,
bilinear upsampling) since it's a fresh training run, not loading the older
`clearview-derain-unet` checkpoint that needed the transpose-conv override.

---

## 4. NAFNet

```bash
clearview-train \
  --model nafnet \
  --mix-config configs/mix/rain_mixed_synthetic_real.yaml --mix-sampler \
  --dataset-type spa-data --data-dir /home/saumya.saksena/projects/mixed_datasets/spa_data --val-split val \
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
Restormer's 15.3M) — not `nafnet_small` (1.1M) or `nafnet_large` (116M).
**Caveat, unlike the other two: this exact pipeline (MixedDataset +
Charbonnier + this crop/schedule) has not been smoke-tested with NAFNet at
all this session** — only Restormer and a small UNet were. `--batch-size 6`
is an estimate based on NAFNet's generally lower memory footprint than
transformer-attention models at a similar param count, not a measurement.
I'd run the same kind of 1-epoch smoke test we did for Restormer/UNet
before trusting this one on the full 100 epochs.

---

## Before committing the weekend to any of these

Time one real epoch on the actual server first — a 4070 Super and an A5000
are close enough on paper that guessing wastes more time than measuring:

```bash
# same command as above, with --epochs 1 (drop --early-stopping/--patience,
# irrelevant for a 1-epoch timing check)
```

Multiply the wall time by 100 for a real ETA before letting any of these run
unattended.
