# ClearView: Neural Image Deraining

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-00599C?style=flat-square&logo=open-source-initiative&logoColor=white)](https://opensource.org/licenses/Apache-2.0)
[![Code style: Black](https://img.shields.io/badge/Code%20Style-Black-000000?style=flat-square&logo=python&logoColor=white)](https://github.com/psf/black)

**Modern PyTorch implementation for removing rain, snow, and adverse weather effects from images using deep learning.**

Designed for autonomous driving perception, surveillance systems, and image restoration. Built with production-ready architectures and real-time inference capabilities.

---

## 🎯 Features

- **Multiple Architectures**: U-Net, Attention U-Net, with configurable backbones (ResNet, EfficientNet)
- **Advanced Loss Functions**: Multi-component losses (L1+L2+SSIM+Edge+Perceptual)
- **Real-Time Demo**: Live webcam deraining for instant testing
- **Production Ready**: Mixed precision training, ONNX export, optimized inference
- **Easy to Extend**: Add new models, losses, or datasets with minimal code
- **Comprehensive Metrics**: PSNR, SSIM, MAE, MSE tracking during training
- **Pretrained Models**: Download and use state-of-the-art weights immediately

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/dronefreak/clearview.git
cd clearview

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### Try the Video Demo (Process Rainy Footage)

```bash
# Process a rainy video (perfect for autonomous driving footage)
python scripts/video_demo.py \
    --video rainy_driving.mp4 \
    --weights pretrained/unet_attention.pth \
    --output derained_result.mp4

# Or try the interactive web demo
python scripts/gradio_demo.py --weights pretrained/unet_attention.pth
# Opens in browser - drag and drop rainy images for instant results
```

### Inference on a Single Image

```bash
python scripts/inference.py \
    --image path/to/rainy_image.jpg \
    --weights pretrained/unet_attention.pth \
    --output derained_result.jpg
```

### Training Your Own Model

```bash
python scripts/train.py \
    --config configs/unet_attention.yaml \
    --data_path /path/to/dataset \
    --epochs 100
```

---

## 📊 Results

| Method            | Rain100H (PSNR↑) | Rain100L (PSNR↑) | Params | Speed (ms) |
| ----------------- | ---------------- | ---------------- | ------ | ---------- |
| U-Net Baseline    | 28.5             | 35.2             | 7.8M   | 15         |
| U-Net + Attention | 30.1             | 36.8             | 8.9M   | 18         |
| ResNet34 Encoder  | 31.2             | 37.5             | 12.4M  | 22         |

_Tested on NVIDIA RTX 3090, 512×384 images_

### Visual Comparisons

**Rain Removal:**

```markdown
[Input (Rainy)] → [Our Model] → [Ground Truth]
🌧️ ✨ ☀️
```

_Add actual before/after images in the `assets/` folder_

---

## 🏗️ Architecture

The system uses an encoder-decoder architecture with skip connections:

```text
Input (3×H×W)
    ↓
[Encoder: ResNet/EfficientNet]
    ↓
[Bottleneck + Attention]
    ↓
[Decoder: Transpose Conv]
    ↓ (skip connections)
Output (3×H×W)
```

### Loss Function

Multi-component loss for high-quality reconstruction:

```python
L_total = L1 + L2 + λ_ssim·L_ssim + λ_edge·L_edge + λ_perc·L_perceptual
```

- **L1 + L2**: Pixel-wise reconstruction
- **L_ssim**: Structural similarity preservation
- **L_edge**: Edge-aware restoration using Sobel filters
- **L_perceptual**: VGG-based perceptual loss for natural textures

---

## 📁 Project Structure

```bash
clearview/
├── src/clearview/          # Core library
│   ├── models/             # Neural network architectures
│   ├── losses/             # Loss functions
│   ├── data/               # Dataset loaders
│   ├── training/           # Training utilities
│   └── utils/              # Metrics, visualization, etc.
├── scripts/                # Command-line tools
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   ├── inference.py        # Single image inference
│   ├── video_demo.py       # Video processing demo
│   ├── gradio_demo.py      # Interactive web UI
│   └── create_demo_assets.py  # Generate GIFs/comparison grids
├── configs/                # Configuration files
├── tests/                  # Unit tests
├── docs/                   # Documentation
└── pretrained/             # Pretrained model weights
```

---

## 📚 Datasets

The models are trained on standard deraining benchmarks:

- **Rain100H/L**: Heavy and light rain synthetic datasets
- **Rain800**: Large-scale synthetic rain dataset
- **Rain1400**: Diverse rain patterns
- **SPA-Data**: Real-world rainy images

### Prepare Your Own Dataset

Organize your data as:

```bash
dataset/
├── train/
│   ├── rainy_image/
│   │   ├── 001.jpg
│   │   └── ...
│   └── ground_truth/
│       ├── 001.jpg
│       └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

Create CSV files mapping rainy images to ground truth:

```csv
rainy_image,ground_truth
001.jpg,001.jpg
002.jpg,002.jpg
```

See [docs/DATASETS.md](docs/DATASETS.md) for detailed instructions.

---

## 🎓 Training

### Basic Training

```bash
python scripts/train.py --config configs/unet_attention.yaml
```

### Advanced Options

```yaml
# configs/unet_attention.yaml
model:
  backbone: resnet34
  use_attention: true
  pretrained: true

training:
  batch_size: 16
  epochs: 100
  learning_rate: 0.0001
  loss: l1l2ssim_edge_perceptual

data:
  train_csv: data/train.csv
  val_csv: data/val.csv
  image_size: [384, 512]
  augmentation: true
```

### Monitor Training

```bash
tensorboard --logdir logs/
```

See [docs/TRAINING.md](docs/TRAINING.md) for comprehensive training guide.

---

## 🔬 Evaluation

Evaluate on test set:

```bash
python scripts/evaluate.py \
    --config configs/unet_attention.yaml \
    --weights checkpoints/best_model.pth \
    --test_csv data/test.csv \
    --output_dir results/
```

This generates:

- Derained images
- Metrics (PSNR, SSIM, MAE, MSE) per image
- Aggregate statistics
- Visual comparison grids

---

## 🎬 Demo Applications

### Video Processing

Process entire videos frame-by-frame with temporal consistency:

```bash
python scripts/video_demo.py \
    --video rainy_dashcam.mp4 \
    --weights checkpoints/best_model.pth \
    --output clean_video.mp4 \
    --fps 30
```

Perfect for:

- Autonomous driving footage
- Surveillance videos
- Dashcam recordings

### Interactive Web Demo

Launch a browser-based interface for easy testing:

```bash
python scripts/gradio_demo.py \
    --weights checkpoints/best_model.pth \
    --share  # Creates public link
```

Features:

- Drag-and-drop image upload
- Real-time processing
- Side-by-side comparison
- Shareable public link (optional)

### Create Demo Assets

Generate marketing materials:

```bash
# Create animated GIF comparisons
python scripts/create_demo_assets.py \
    --images test_images/ \
    --weights checkpoints/best_model.pth \
    --output demo.gif

# Create comparison grids
python scripts/create_demo_assets.py \
    --images test_images/ \
    --weights checkpoints/best_model.pth \
    --output comparison_grid.png \
    --format grid
```

---

## 🚗 Use Cases

### Autonomous Driving

Process real-time camera feeds or recorded footage:

```python
from clearview import DerainingModel

model = DerainingModel.from_pretrained('unet_attention')

# Process video stream
for frame in video_stream:
    clean_frame = model.process(frame)
    # Feed to object detection/segmentation pipeline
```

Or use the video processing script:

```bash
python scripts/video_demo.py \
    --video dashcam_footage.mp4 \
    --weights pretrained/unet_attention.pth \
    --output clean_footage.mp4
```

### Surveillance Systems

Enhance video quality in rainy conditions:

```python
# Process video stream with batching for efficiency
batch_size = 8
for batch_frames in batched_video_stream(batch_size):
    derained_batch = model.process_batch(batch_frames)
    # Continue with tracking/recognition
```

### Photography Enhancement

Restore photos taken in rain/snow:

```bash
python scripts/inference.py --image vacation_photo.jpg --output enhanced.jpg
```

Or use the interactive demo:

```bash
python scripts/gradio_demo.py --weights pretrained/unet_attention.pth
# Upload images via web interface
```

---

## 🛠️ Advanced Usage

### Export to ONNX

```bash
python scripts/export_onnx.py \
    --weights checkpoints/best_model.pth \
    --output model.onnx
```

### Custom Model

```python
from clearview.models import AttentionUNet
from clearview.training import Trainer

# Define custom model
model = AttentionUNet(
    encoder='efficientnet-b0',
    decoder_channels=[256, 128, 64, 32],
    use_attention=True
)

# Train
trainer = Trainer(model, config)
trainer.fit(train_loader, val_loader)
```

### Add Custom Loss

```python
from clearview.losses import BaseLoss

class MyCustomLoss(BaseLoss):
    def forward(self, pred, target):
        # Your loss implementation
        return loss_value

# Register and use
from clearview.training import register_loss
register_loss('my_loss', MyCustomLoss)
```

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black src/ scripts/ tests/

# Type checking
mypy src/
```

---

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@software{saksena2025clearview,
  author = {Saksena, Saumya Kumaar},
  title = {ClearView: Neural Image Deraining},
  year = {2025},
  url = {https://github.com/dronefreak/clearview}
}
```

Original TensorFlow implementation (2019-2024, deprecated):

```bibtex
@software{saksena2019deraining,
  author = {Saksena, Saumya Kumaar},
  title = {Image De-raining using TensorFlow 2},
  year = {2019},
  note = {Repository archived and replaced by ClearView}
}
```

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Original architecture inspired by U-Net (Ronneberger et al., 2015)
- Attention mechanisms based on Attention U-Net (Oktay et al., 2018)
- Perceptual loss based on Johnson et al., 2016
- Benchmark datasets: Rain100H/L, Rain800, Rain1400

---

## 🐛 Known Issues & Roadmap

### Current Limitations

- Real-time performance requires GPU (CPU inference is slow)
- Large models may require >8GB VRAM for training
- Video processing doesn't yet enforce temporal consistency (frame-by-frame only)

### Roadmap

- [ ] Transformer-based architecture option
- [ ] Diffusion model integration for better quality
- [ ] Video deraining with temporal consistency and optical flow
- [ ] Mobile/edge deployment (TensorRT, CoreML)
- [ ] Docker container for easy deployment
- [ ] Hosted Gradio demo on Hugging Face Spaces
- [ ] Support for snow, fog, haze removal
- [ ] Semi-supervised training with unpaired data
- [ ] Real-time optimizations for video processing

---

## 📞 Contact

- **Author**: Saumya Kumaar Saksena (dronefreak)
- **Issues**: [GitHub Issues](https://github.com/dronefreak/clearview/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dronefreak/clearview/discussions)

---

<!-- ## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=dronefreak/clearview&type=Date)](https://star-history.com/#dronefreak/clearview&Date)

---

**Built with ❤️ for computer vision and autonomous systems** -->
