# 🌧️ ClearView: Neural Image Deraining

<p align="center">
  <img src="https://github.com/dronefreak/clearview/raw/main/assets/demo_showcase.jpg" alt="ClearView demo showcase: rainy input vs. derained output across four scenes"/>
</p>

<!-- ROW 1: Core Identity (What this project is) -->
<div style="display: flex; justify-content: center; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 24px;">
  <!-- Project Identity -->
  <img src="https://img.shields.io/badge/Models-10%20architectures-0aa1a7?style=flat-square" alt="Models">

  <!-- Tech Stack & Quality -->
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.8+-blue?style=flat-square" alt="Python">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square" alt="PyTorch">
  </a>
  <a href="https://github.com/dronefreak/clearview/actions/workflows/ci.yml">
    <img src="https://github.com/dronefreak/clearview/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square" alt="Ruff">
  </a>

  <!-- Metadata -->
  <a href="https://huggingface.co/spaces/dronefreak/clearview-derain-demo">
    <img src="https://img.shields.io/badge/🤗%20HuggingFace-Demo-FFD21E?style=flat-square" alt="Demo">
  </a>
  <img src="https://img.shields.io/badge/License-Apache--2.0-lightgrey?style=flat-square" alt="License">
</div>

---

## 🚀 Quick Start

### Try on Hugging Face

👉 [Live Demo on HuggingFace](https://huggingface.co/spaces/dronefreak/clearview-derain-demo)

### Install & Run Locally

```bash
git clone https://github.com/dronefreak/clearview.git
cd clearview
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

### Inference

`clearview-inference` handles a single image, a directory of images, or a video, matching input/output type automatically.

**Pull the latest weights from HF**

```python
from huggingface_hub import hf_hub_download

weights = hf_hub_download(
    repo_id="dronefreak/clearview-derain-unet", filename="clearview-derain-unet.pth"
)
```

**Single image**

```bash
clearview-inference --model unet --weights clearview-derain-unet.pth \
  --input rainy.jpg --output derained.jpg
```

**Directory of images**

```bash
clearview-inference --model unet --weights clearview-derain-unet.pth \
  --input-dir ./rainy_photos --output-dir ./derained_photos
```

**Video** (deraining runs frame by frame, with no temporal-consistency term, so some frame-to-frame flicker could be expected)

```bash
clearview-inference --model unet --weights clearview-derain-unet.pth \
  --input rainy_clip.mp4 --output derained_clip.mp4
```

---

## 🌍 Mixed-Domain Training

Training on a blended synthetic + real-world rain set (Rain13K, DDN-Data,
SPA-Data, RealRain-1k-H/L, mildly oversampling the real-world sources) and
selecting checkpoints against a blended validation metric across four of
those sources, rather than optimizing for one benchmark, for a model that
holds up reasonably across domains instead of maxing out a single dataset's
quirks. See [`configs/mix/`](configs/mix/) for the exact recipe.

### Training Mix

`--mix-config` combines 5 sources into one training set, oversampling the
real-world sources 2x so the small RealRain-1k tracks aren't drowned out by
the much larger synthetic sets:

| Source              | Type       | Weight | Pairs      |
| ------------------- | ---------- | ------ | ---------- |
| Rain13K             | Synthetic  | 1.0    | 13,711     |
| DDN-Data / Rain1400 | Synthetic  | 1.0    | 12,600     |
| SPA-Data            | Real-world | 2.0    | 6,385      |
| RealRain-1k-H       | Real-world | 2.0    | 784        |
| RealRain-1k-L       | Real-world | 2.0    | 784        |
| **Total**           |            |        | **34,264** |

~77% synthetic / ~23% real by raw pair count. After the 2x real-world
oversampling weight is applied (i.e. what the sampler actually draws from per epoch), that shifts to **~62% synthetic / ~38% real**.

### Validation / Checkpoint Selection

`--val-mix-config` blends 4 validation sources into one checkpoint-selection
metric, so "best" means "doesn't fail badly anywhere" rather than "maxes out
one dataset's quirks." A single deterministic pass, no oversampling, and
SPA-Data val is capped well below its full size so it can't dominate the
blended average on its own:

| Source                           | Pairs used | Notes                   |
| -------------------------------- | ---------- | ----------------------- |
| SPA-Data (val split)             | 150        | Capped from 1,000 pairs |
| RealRain-1k-H (validation split) | 112        | Full split              |
| RealRain-1k-L (validation split) | 112        | Full split              |
| Rain100L (test split)            | 100        | Synthetic sanity anchor |
| **Total**                        | **474**    |                         |

### Models

10 architectures trained or wired in so far. All 9 ClearView-trained models below share the identical mixed-domain recipe, training data (5-source mix, see above) and checkpoint selection (blended validation set, see above), only batch size/accumulation steps and architecture-specific hyperparameters vary by size. Histoformer is the one exception, see the note below the table.

| Model                              | Params | HF Model Card                                                                                                    |
| ---------------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------- |
| Restormer [[9]](#references)       | 15.3M  | 🤗 [dronefreak/clearview-derain-restormer](https://huggingface.co/dronefreak/clearview-derain-restormer)         |
| Restormer-Small [[9]](#references) | 2.3M   | 🚧 Coming soon                                                                                                   |
| UNet (Vanilla) [[10]](#references) | 21.5M  | 🤗 [dronefreak/clearview-derain-unet](https://huggingface.co/dronefreak/clearview-derain-unet)                   |
| NAFNet (Small) [[11]](#references) | 1.1M   | 🤗 [dronefreak/clearview-derain-nafnet-small](https://huggingface.co/dronefreak/clearview-derain-nafnet-small)   |
| NAFNet (Mid) [[11]](#references)   | 14.3M  | 🤗 [dronefreak/clearview-derain-nafnet](https://huggingface.co/dronefreak/clearview-derain-nafnet)               |
| NAFNet (Large) [[11]](#references) | 116M   | 🤗 [dronefreak/clearview-derain-nafnet-large](https://huggingface.co/dronefreak/clearview-derain-nafnet-large)   |
| ResNet18-UNet [[12]](#references)  | 14.4M  | 🚧 Coming soon                                                                                                   |
| ResNet34-UNet [[12]](#references)  | 24.5M  | 🤗 [dronefreak/clearview-derain-resnet34-unet](https://huggingface.co/dronefreak/clearview-derain-resnet34-unet) |
| ResNet50-UNet [[12]](#references)  | 73.3M  | 🚧 Coming soon                                                                                                   |
| Histoformer [[8]](#references)     | 16.6M  | 🤗 [dronefreak/Histoformer](https://huggingface.co/dronefreak/Histoformer)                                       |

Histoformer is wired in for inference and cross-domain comparison only. Its weights are the original authors' own all-weather (rain/raindrop/snow) checkpoint, not a ClearView training run, trained on different data, selected against a different validation set, so there is no `clearview-train` recipe for it here. To train Histoformer from scratch, use the [official repository](https://github.com/sunshangquan/Histoformer).

In our own spot checks (both `net_g_real` and `net_g_best`), its visible effect leaned much closer to dehazing/contrast correction than streak removal, on images with a haze or veiling component it cleaned up dramatically, but on images with clear rain streaks and no haze, including a genuine rain photograph, streaks were left largely untouched. Results can vary a lot by input, treat it as a reference point rather than a strong deraining baseline.

### Benchmark Results (PSNR / SSIM)

| Test Set                                 | Domain                | Restormer     | NAFNet (Large) | NAFNet (Mid)  | UNet (Vanilla) | ResNet50-UNet |
| ---------------------------------------- | --------------------- | ------------- | -------------- | ------------- | -------------- | ------------- |
| Rain100L [[1]](#references)              | Synthetic             | 36.04 / 0.969 | 34.59 / 0.961  | 34.14 / 0.957 | 30.96 / 0.932  | 29.79 / 0.906 |
| Rain100H [[1]](#references)              | Synthetic             | 27.78 / 0.868 | 27.65 / 0.856  | 27.72 / 0.849 | 26.41 / 0.823  | 25.37 / 0.794 |
| Test100 [[2]](#references)               | Synthetic             | 28.65 / 0.881 | 27.71 / 0.865  | 27.96 / 0.873 | 24.91 / 0.836  | 26.16 / 0.839 |
| Test1200 [[3]](#references)              | Synthetic             | 31.91 / 0.906 | 31.37 / 0.898  | 31.28 / 0.898 | 29.08 / 0.868  | 28.44 / 0.856 |
| Test2800 [[4]](#references)              | Synthetic             | 32.05 / 0.928 | 31.75 / 0.924  | 31.66 / 0.923 | 30.61 / 0.909  | 28.67 / 0.883 |
| DDN-Data [[4]](#references)              | Synthetic             | 32.21 / 0.931 | 31.90 / 0.927  | 31.84 / 0.926 | 30.67 / 0.912  | 28.72 / 0.886 |
| SPA-Data [[5]](#references)              | Real-world            | 44.67 / 0.989 | 41.99 / 0.986  | 41.77 / 0.986 | 39.01 / 0.980  | 37.07 / 0.973 |
| RealRain-1k-H [[6]](#references)         | Real-world            | 40.28 / 0.985 | 39.34 / 0.982  | 38.68 / 0.980 | 35.98 / 0.971  | 34.94 / 0.970 |
| RealRain-1k-L [[6]](#references)         | Real-world            | 42.35 / 0.989 | 41.17 / 0.987  | 40.64 / 0.986 | 38.04 / 0.980  | 36.52 / 0.978 |
| AllWeather (rain+fog) [[7]](#references) | Cross-domain (stress) | 13.72 / 0.584 | 13.53 / 0.576  | 13.64 / 0.579 | 13.66 / 0.570  | 13.61 / 0.555 |

Columns sorted highest to lowest by average PSNR across the 9 rain-only test sets (Restormer 35.10, NAFNet Large 34.16, NAFNet Mid 33.97, UNet 31.74, ResNet50-UNet 30.63). This is a curated subset, not the full model zoo, NAFNet (Small), ResNet18/34-UNet, and Histoformer are trained/evaluated too but omitted here for brevity, see their individual model cards (above) for full numbers. Metrics computed on each source's own held-out test/eval split (not the blended validation set used for checkpoint selection), full per-image distributions and logs live under `runs/<model>/eval/<dataset>/`.

---

## 📚 Supported Datasets

- **[Rain13K](https://huggingface.co/datasets/dronefreak/Rain13K)** (composite synthetic set; includes Rain100H/L, Test100, Test1200, Test2800): 13.7K train pairs
- **[DDN-Data / Rain1400](https://huggingface.co/datasets/dronefreak/DDN-Data)**: 12.6K train / 1.4K test
- **[SPA-Data](https://huggingface.co/datasets/dronefreak/SPA-Data)**: real-world, video-derived rain/clean pairs
- **[RealRain-1k-H/L](https://huggingface.co/datasets/dronefreak/RealRain-1k)**: real-world, heavy/light density tracks
- **Custom**: Organize as `train/{rainy_image,ground_truth}`, or combine any of the above via [`--mix-config`](configs/mix/)

---

## 🏋️ Training and Evaluation

### Training

Launch a single-GPU training run with an example command like the one below:

```bash
clearview-train \
  --model nafnet_small \
  --mix-config configs/mix/rain_mixed_synthetic_real.yaml --mix-sampler \
  --val-mix-config configs/mix/rain_mixed_val.yaml \
  --data-dir /path/to/mixed_datasets \
  --loss custom --loss-config '{"charbonnier": {"weight": 1.0}}' \
  --crop-size 256 --batch-size 64 --val-batch-size 64 --num-workers 8 \
  --optimizer adamw --lr 1e-4 --scheduler cosine --warmup-epochs 5 \
  --epochs 100 --early-stopping --patience 15 \
  --checkpoint-monitor val_psnr --checkpoint-mode max \
  --mixed-precision --ema --ema-decay 0.999 --compile \
  --output-dir ./runs/rain_mixed_nafnet_small \
  --device cuda
```

### Evaluation

Once training finishes, evaluate a checkpoint against a labeled test set:

```bash
clearview-evaluate \
  --model unet --weights ./runs/rain_mixed_unet/checkpoints/best_val_psnr.pth \
  --data-dir /path/to/rainy_image_dataset \
  --dataset-type rain1400 \
  --rainy-dir testing/rainy_image --clean-dir testing/ground_truth \
  --batch-size 16 \
  --output-dir ./runs/rain_mixed_unet/test_eval
```

---

## 🔮 Roadmap

- [x] Real-world rain dataset (SPA-Data, RealRain-1k-H/L, blended with synthetic sources via `--mix-config`)
- [x] Add UResNet model support
- [x] Add Restormer model support
- [x] Add NAFNet model support
- [x] Add Histoformer model support (inference-only)
- [x] Mixed-domain training across the full model zoo (Restormer, UNet, NAFNet Small/Mid/Large)
- [x] Release trained checkpoints on Hugging Face
- [ ] Temporal consistency for video
- [ ] Mobile deployment (ONNX/TensorRT)
- [ ] Snow/fog/haze removal

---

## 🤝 Contribute

PRs welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).
Need help? Open an [Issue](https://github.com/dronefreak/clearview/issues).

---

## 📖 Citation

```bibtex
@software{saksena2025clearview,
  author = {Saksena, Saumya Kumaar},
  title = {ClearView: Practical Image Deraining with PyTorch},
  year = {2025},
  url = {https://github.com/dronefreak/clearview}
}
```

**License**: [Apache 2.0](LICENSE)
**Author**: [Saumya Kumaar Saksena (@dronefreak)](https://github.com/dronefreak)

### References

1. Yang et al., _Deep Joint Rain Detection and Removal from a Single Image_, CVPR 2017 (Rain100H/L).
2. Zhang & Patel, _Density-aware Single Image De-raining using a Multi-stream Dense Network_, CVPR 2018 (Test100).
3. Zhang, Sindagi & Patel, _Image De-raining Using a Conditional Generative Adversarial Network_, IEEE TCSVT 2019 (Test1200).
4. Fu et al., _Removing Rain from Single Images via a Deep Detail Network_, CVPR 2017 (Test2800 / DDN-Data / Rain1400).
5. Wang et al., _Spatial Attentive Single-Image Deraining with a High Quality Real Rain Dataset_, CVPR 2019 (SPA-Data).
6. Li et al., _RealRain-1k: A Large-Scale Dataset for Real-World Single Image Deraining_, arXiv:2206.05514, 2022.
7. Li et al., _Heavy Rain Image Restoration: Integrating Physics Model and Conditional Adversarial Learning_, CVPR 2019 (AllWeather rain+fog / Outdoor-Rain).
8. Sun, Ren, Gao, Wang & Cao, _Restoring Images in Adverse Weather Conditions via Histogram Transformer_, ECCV 2024, [arXiv:2407.10172](https://arxiv.org/abs/2407.10172) (Histoformer, inference-only baseline).
9. Zamir et al., _Restormer: Efficient Transformer for High-Resolution Image Restoration_, CVPR 2022, [arXiv:2111.09881](https://arxiv.org/abs/2111.09881).
10. Ronneberger, Fischer & Brox, _U-Net: Convolutional Networks for Biomedical Image Segmentation_, MICCAI 2015, [arXiv:1505.04597](https://arxiv.org/abs/1505.04597).
11. Chen, Chu, Zhang & Sun, _Simple Baselines for Image Restoration_, ECCV 2022, [arXiv:2204.04676](https://arxiv.org/abs/2204.04676) (NAFNet).
12. He, Zhang, Ren & Sun, _Deep Residual Learning for Image Recognition_, CVPR 2016, [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) (ResNet, used as the encoder backbone for ResNet34-UNet).

---
