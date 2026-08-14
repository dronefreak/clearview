# 🌧️ ClearView: Neural Image Deraining

[![🐍 Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white&style=for-the-badge)](https://www.python.org/downloads/)
[![🔥 PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white&style=for-the-badge)](https://pytorch.org/)
[![⚖️ License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green?logo=osi&logoColor=white&style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
[![🤗 Demo](https://img.shields.io/badge/HuggingFace-Demo-FFD21E?logo=huggingface&logoColor=yellow&style=for-the-badge)](https://huggingface.co/spaces/dronefreak/clearview-derain-demo)

> **Fast, deployement-ready static image deraining model** for autonomous driving, surveillance, and photo restoration.  
> **30.9 PSNR / 0.914 SSIM** on Rain1400 • **~15ms inference** (RTX 4070) • **L1 loss + vanilla UNet = best results**

<p align="center">
  <img src="https://github.com/dronefreak/clearview/raw/main/assets/demo_showcase.jpg" alt="ClearView demo showcase: rainy input vs. derained output across four scenes"/>
</p>

---

## 🚀 Quick Start

### Try Online

👉 [Live Demo on HuggingFace](https://huggingface.co/spaces/dronefreak/clearview-derain-demo)

### Install & Run Locally

```bash
git clone https://github.com/dronefreak/clearview.git
cd clearview
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

### Pretrained Weights

| Model     | Status           | Weights                                                                                                       |
| --------- | ---------------- | ------------------------------------------------------------------------------------------------------------- |
| UNet      | ✅ Available now | [`dronefreak/clearview-derain-unet`](https://huggingface.co/dronefreak/clearview-derain-unet) on Hugging Face |
| Restormer | 🚧 Coming soon   | —                                                                                                             |

```python
from huggingface_hub import hf_hub_download

weights = hf_hub_download(
    repo_id="dronefreak/clearview-derain-unet", filename="clearview-derain-unet.pth"
)
```

### Inference

`clearview-inference` handles a single image, a directory of images, or a video — input/output type is matched automatically.

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

**Video** (deraining runs frame by frame — there's no temporal-consistency term, so some frame-to-frame flicker is expected)

```bash
clearview-inference --model unet --weights clearview-derain-unet.pth \
  --input rainy_clip.mp4 --output derained_clip.mp4
```

### Train

```bash
clearview-train \
  --data-dir /path/to/Rain1400 \
  --model unet --loss l1 --optimizer adamw --lr 1e-4 \
  --batch-size 24 --epochs 200 --dataset-type rain1400
```

---

## 📊 Performance

| Model          | PSNR  | SSIM  | Params | Speed |
| -------------- | ----- | ----- | ------ | ----- |
| **UNet (L1)**  | 30.91 | 0.914 | 7.8M   | ~15ms |
| Attention UNet | 30.04 | 0.910 | 8.9M   | ~20ms |

✅ **Key insight**: L1 loss alone outperforms complex multi-loss combos.  
⚠️ **Limitations**: Trained on synthetic rain; slight texture smoothing.

---

## 🏗️ Architecture Highlights

- **Backbone**: Vanilla UNet (4 encoder/decoder blocks + skip connections)
- **Output**: Sigmoid-bounded to [0,1]
- **Loss**: Pixel-wise L1 (`loss = L1(pred, target)`)
- **Why not attention?** No measurable gain—adds latency and params.

---

## 📦 Pretrained Model

Download from Hugging Face:

```python
from huggingface_hub import hf_hub_download
weights = hf_hub_download("dronefreak/clearview-unet", "clearview-unet.pth")
```

🔗 [Model Card](https://huggingface.co/dronefreak/clearview-derain-unet)

---

## 📚 Supported Datasets

- **Rain1400** (recommended): 12.6K train / 1.4K test
- **Rain100H/L**: Heavy/light rain variants
- **Custom**: Organize as `train/{rainy_image,ground_truth}`

---

## 🛠️ Advanced Usage

- **Video**: Frame-by-frame processing via `scripts/video_demo.py` _(no temporal consistency yet)_
- **Metrics**: `clearview-eval` reports PSNR, SSIM, MAE, MSE
- **Training Tips**:
  - Use mixed precision + gradient clipping
  - Early stopping (patience=50)
  - Avoid multi-component losses

---

## 🔮 Roadmap

- [ ] Temporal consistency for video
- [ ] Real-world rain dataset
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

---
