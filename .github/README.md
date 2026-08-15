# 🌧️ ClearView: Neural Image Deraining

<p align="center">
  <img src="https://github.com/dronefreak/clearview/raw/main/assets/demo_showcase.jpg" alt="ClearView demo showcase: rainy input vs. derained output across four scenes"/>
</p>

<!-- ROW 1: Core Identity (What this project is) -->
<div style="display: flex; justify-content: center; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 24px;">
  <!-- Project Identity -->
  <img src="https://img.shields.io/badge/Models-5%20architectures-0aa1a7?style=flat-square" alt="Models">

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

## 🌍 Mixed-Domain Training (In Progress)

Training on a blended synthetic + real-world rain set (Rain13K, DDN-Data,
SPA-Data, RealRain-1k-H/L, mildly oversampling the real-world sources) and
selecting checkpoints against a blended validation metric across four of
those sources, rather than optimizing for one benchmark, for a model that
holds up reasonably across domains instead of maxing out a single dataset's
quirks. See [`configs/mix/`](configs/mix/) for the exact recipe.

**Status: training in progress. Numbers below are placeholders, not results.**

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

| Model          | Training Data            | Validation Checkpoint                           | HF Model Card  |
| -------------- | ------------------------ | ----------------------------------------------- | -------------- |
| Restormer      | Mixed (synthetic + real) | Blended (SPA-Data + RealRain-1k-H/L + Rain100L) | 🚧 Coming soon |
| UNet (Vanilla) | Mixed (synthetic + real) | Blended (SPA-Data + RealRain-1k-H/L + Rain100L) | 🚧 Coming soon |
| NAFNet         | Mixed (synthetic + real) | Blended (SPA-Data + RealRain-1k-H/L + Rain100L) | 🚧 Coming soon |

### Benchmark Results (PSNR / SSIM)

| Test Set                                 | Domain                | Restormer (mixed) | UNet (Vanilla, mixed) | NAFNet (mixed) |
| ---------------------------------------- | --------------------- | ----------------- | --------------------- | -------------- |
| Rain100L [[1]](#references)              | Synthetic             | _pending_         | _pending_             | _pending_      |
| Rain100H [[1]](#references)              | Synthetic             | _pending_         | _pending_             | _pending_      |
| Test100 [[2]](#references)               | Synthetic             | _pending_         | _pending_             | _pending_      |
| Test1200 [[3]](#references)              | Synthetic             | _pending_         | _pending_             | _pending_      |
| Test2800 [[4]](#references)              | Synthetic             | _pending_         | _pending_             | _pending_      |
| DDN-Data [[4]](#references)              | Synthetic             | _pending_         | _pending_             | _pending_      |
| SPA-Data [[5]](#references)              | Real-world            | _pending_         | _pending_             | _pending_      |
| RealRain-1k-H [[6]](#references)         | Real-world            | _pending_         | _pending_             | _pending_      |
| RealRain-1k-L [[6]](#references)         | Real-world            | _pending_         | _pending_             | _pending_      |
| AllWeather (rain+fog) [[7]](#references) | Cross-domain (stress) | _pending_         | _pending_             | _pending_      |

---

## 📚 Supported Datasets

- **[Rain13K](https://huggingface.co/datasets/dronefreak/Rain13K)** (composite synthetic set; includes Rain100H/L, Test100, Test1200, Test2800): 13.7K train pairs
- **[DDN-Data / Rain1400](https://huggingface.co/datasets/dronefreak/DDN-Data)**: 12.6K train / 1.4K test
- **[SPA-Data](https://huggingface.co/datasets/dronefreak/SPA-Data)**: real-world, video-derived rain/clean pairs
- **[RealRain-1k-H/L](https://huggingface.co/datasets/dronefreak/RealRain-1k)**: real-world, heavy/light density tracks
- **Custom**: Organize as `train/{rainy_image,ground_truth}`, or combine any of the above via [`--mix-config`](configs/mix/)

---

## 🔮 Roadmap

- [ ] Temporal consistency for video
- [x] Real-world rain dataset (SPA-Data, RealRain-1k-H/L, blended with synthetic sources via `--mix-config`)
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

---
