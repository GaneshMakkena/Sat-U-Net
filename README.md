# Satellite Image Segmentation with U-Net (M4 Optimized)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MLX](https://img.shields.io/badge/MLX-0.4+-green.svg)](https://github.com/ml-explore/mlx)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**High-resolution satellite image segmentation optimized for Apple Silicon (M4 MacBook Air)**

Train and deploy semantic segmentation models at 1024×1024 resolution with efficient memory usage (<9GB on 16GB RAM) and fast inference (50-100ms using Neural Engine).

---

## 🌟 Features

- ✅ **1024×1024 Resolution** - 4× better than baseline (512×512)
- ✅ **Memory Efficient** - Runs on M4 MacBook Air 16GB RAM
- ✅ **Fast Training** - Mixed precision (FP16) with gradient checkpointing
- ✅ **Neural Engine Deployment** - Core ML optimized for M-series chips
- ✅ **Multiple Backends** - MLX, Core ML, or PyTorch MPS
- ✅ **Production Ready** - Complete pipeline from training to deployment

---

## 📋 Table of Contents

- [Requirements](#-requirements)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Inference](#-inference)
- [Deployment](#-deployment)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)

---

## 💻 Requirements

### Hardware
- **Minimum**: M1/M2/M3 MacBook with 16GB RAM
- **Recommended**: M4 MacBook Air/Pro with 16GB+ RAM
- **Storage**: 10GB free space (5GB for dataset, 5GB for models)

### Software
- **OS**: macOS 14.0+ (Sonoma or later)
- **Python**: 3.11 or higher
- **GPU**: Metal-compatible (built into M-series chips)

---

## 🚀 Installation

### 1. Clone Repository

```bash
git clone https://github.com/GaneshMakkena/Sat-U-Net.git
cd Sat-U-Net
```

### 2. Create Virtual Environment

```bash
# Create environment
python3 -m venv sat-mlx-env

# Activate environment
source sat-mlx-env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Verify MLX installation
python -c "import mlx.core as mx; print(f'MLX: {mx.metal.is_available()}')"
```

### 4. Set Up Jupyter Kernel (Optional)

```bash
# Install kernel for notebooks
python -m ipykernel install --user --name=sat-mlx-m4

# Launch Jupyter
jupyter notebook
```

**Expected output**: `MLX: True` ✅

---

## 📁 Project Structure

```
Sat-U-Net/
├── README.md                          # This file
├── README_M4_UPGRADE.md               # Detailed technical documentation
├── requirements.txt                   # Python dependencies
│
├── # Training Files
├── Sat_IMG_UNet_MLX_1024_M4.ipynb    # Main training notebook (1024×1024)
├── memory_utils.py                    # Memory optimization utilities
├── efficient_unet_mlx.py              # Efficient U-Net architecture
│
├── # Inference Files
├── mlx_to_coreml.py                   # Model conversion script
├── coreml_inference.py                # Core ML inference wrapper
├── hybrid_inference.py                # Multi-backend inference
│
├── # Dataset Directories
├── train/                             # Training images
│   ├── *_sat.jpg                      # Satellite images
│   └── *_mask.png                     # Segmentation masks
├── valid/                             # Validation images
│   ├── *_sat.jpg
│   └── *_mask.png
└── test/                              # Test images
    ├── *_sat.jpg
    └── *_mask.png
```

### File Descriptions

| File | Purpose | Size |
|------|---------|------|
| `memory_utils.py` | Mixed precision, gradient checkpointing, profiling | 13 KB |
| `efficient_unet_mlx.py` | Memory-efficient U-Net (3.5M parameters) | 12 KB |
| `Sat_IMG_UNet_MLX_1024_M4.ipynb` | Complete training pipeline | 39 KB |
| `mlx_to_coreml.py` | MLX→PyTorch→Core ML conversion | 17 KB |
| `coreml_inference.py` | Production inference API | 12 KB |
| `hybrid_inference.py` | Multi-backend inference | 18 KB |

---

## ⚡ Quick Start

### Option 1: Training from Scratch

```bash
# 1. Ensure dataset is in place
ls train/*.jpg | wc -l   # Should show training images

# 2. Open training notebook
jupyter notebook Sat_IMG_UNet_MLX_1024_M4.ipynb

# 3. Select kernel: sat-mlx-m4
# 4. Run all cells (Runtime > Run All)
```

**Expected time**: ~12 hours for 50 epochs

### Option 2: Inference with Pre-trained Model

```bash
# 1. Ensure you have a trained model
ls best_model_mlx_1024_m4.safetensors

# 2. Convert to Core ML (one-time)
python mlx_to_coreml.py \
    --mlx-weights best_model_mlx_1024_m4.safetensors \
    --output satellite_unet_1024_m4.mlpackage

# 3. Run inference
python hybrid_inference.py --image test/example_sat.jpg
```

**Expected time**: <1 second per image

---

## 🎓 Training

### Step 1: Prepare Dataset

Your dataset should follow this structure:

```
train/
├── 001_sat.jpg      # Satellite image
├── 001_mask.png     # Corresponding mask
├── 002_sat.jpg
├── 002_mask.png
└── ...

valid/
├── 100_sat.jpg
├── 100_mask.png
└── ...
```

**Mask colors** (RGB format):
- Urban: Cyan (0, 255, 255)
- Agriculture: Yellow (255, 255, 0)
- Rangeland: Magenta (255, 0, 255)
- Forest: Green (0, 255, 0)
- Water: Blue (0, 0, 255)
- Barren: White (255, 255, 255)
- Unknown: Black (0, 0, 0)

### Step 2: Configure Training

Open `Sat_IMG_UNet_MLX_1024_M4.ipynb` and adjust if needed:

```python
@dataclass
class Config:
    IMG_SIZE = 1024              # Image resolution
    BATCH_SIZE = 1               # Keep at 1 for 16GB RAM
    GRADIENT_ACCUMULATION = 4    # Effective batch size
    EPOCHS = 50                  # Total epochs
    MIXED_PRECISION = True       # Enable FP16
    CHECKPOINT_DECODER = True    # Enable memory savings
    BASE_FILTERS = 32            # Model capacity
```

### Step 3: Run Training

```bash
# Method 1: Jupyter Notebook (Recommended)
jupyter notebook Sat_IMG_UNet_MLX_1024_M4.ipynb

# Method 2: Convert to script and run
jupyter nbconvert --to script Sat_IMG_UNet_MLX_1024_M4.ipynb
python Sat_IMG_UNet_MLX_1024_M4.py
```

### Step 4: Monitor Progress

Training will show:
- Current epoch and stage (512/768/1024)
- Loss, accuracy, and IoU metrics
- Memory usage
- Estimated time remaining

**Expected metrics** (after 50 epochs):
- Validation Accuracy: 85-90%
- Mean IoU: 70-75%
- Training time: ~12 hours on M4

### Step 5: Output

After training completes:
- `best_model_mlx_1024_m4.safetensors` - Best model weights
- `training_history_1024_m4.png` - Training curves
- Memory profiling statistics

---

## 🔮 Inference

### Using Core ML (Fastest - Recommended)

```bash
# 1. Convert model (one-time)
python mlx_to_coreml.py \
    --mlx-weights best_model_mlx_1024_m4.safetensors \
    --output satellite_unet_1024_m4.mlpackage

# 2. Single image inference
python coreml_inference.py \
    --model satellite_unet_1024_m4.mlpackage \
    --image test/example_sat.jpg

# 3. Batch processing
python coreml_inference.py \
    --model satellite_unet_1024_m4.mlpackage \
    --batch "test/*.jpg" \
    --output-dir predictions/

# 4. Benchmark performance
python coreml_inference.py \
    --model satellite_unet_1024_m4.mlpackage \
    --benchmark
```

### Using Hybrid Pipeline (Auto-Select Backend)

```bash
# Single image
python hybrid_inference.py --image test/example_sat.jpg

# Compare all backends
python hybrid_inference.py --compare

# Specific backend
python hybrid_inference.py --backend mlx --image test/example_sat.jpg
```

### Programmatic Usage

```python
from hybrid_inference import HybridPredictor

# Initialize (auto-selects best backend)
predictor = HybridPredictor(backend='auto')

# Predict
mask, composition = predictor.predict('test_image.jpg')

# Visualize
predictor.visualize_prediction('test_image.jpg', 'output.png')

# Print composition
for class_name, percentage in composition.items():
    print(f"{class_name}: {percentage:.2f}%")
```

---

## 🚢 Deployment

### Option 1: Core ML (Recommended for Production)

**Best for**: macOS/iOS apps, fastest inference

```bash
# Convert model
python mlx_to_coreml.py \
    --mlx-weights best_model_mlx_1024_m4.safetensors \
    --output satellite_unet_1024_m4.mlpackage

# Package includes:
# - Neural Engine optimization
# - FP16 quantization
# - Ready for Xcode integration
```

**Performance**: 50-100ms per image on M4

### Option 2: MLX (Recommended for Development)

**Best for**: Research, experimentation, training

```python
from efficient_unet_mlx import create_efficient_unet_m4

# Load model
model = create_efficient_unet_m4(1024, 7, 32)
model.load_weights('best_model_mlx_1024_m4.safetensors')

# Inference
import mlx.core as mx
output = model(mx.array(image_tensor))
```

**Performance**: 100-150ms per image on M4

### Option 3: PyTorch MPS (Fallback)

**Best for**: Compatibility, older macOS

```bash
# Use hybrid pipeline
python hybrid_inference.py --backend pytorch_mps --image test.jpg
```

**Performance**: 150-200ms per image on M4

---

## 📊 Performance

### Training Performance (M4 MacBook Air 16GB)

| Metric | Value |
|--------|-------|
| Resolution | 1024×1024 |
| Batch Size | 1 (effective: 4) |
| Memory Usage | 8.9 GB peak |
| Speed | 18 min/epoch |
| GPU Utilization | 90% |
| Total Time | ~12 hours (50 epochs) |

### Inference Performance (M4)

| Backend | Speed (ms) | Throughput (FPS) | Memory (GB) |
|---------|-----------|------------------|-------------|
| **Core ML** | **75** | **13.3** | **2.1** |
| MLX | 118 | 8.4 | 3.2 |
| PyTorch MPS | 182 | 5.6 | 4.5 |

### Model Statistics

| Metric | Value |
|--------|-------|
| Parameters | 3.52M |
| Model Size (FP16) | 7.04 MB |
| Input Size | 1024×1024×3 |
| Output Size | 1024×1024×7 |
| Classes | 7 |

---

## 🐛 Troubleshooting

### Installation Issues

**Problem**: `import mlx` fails

```bash
# Solution: Reinstall MLX
pip install --upgrade mlx mlx-nn

# Verify
python -c "import mlx.core as mx; print(mx.metal.is_available())"
```

**Problem**: Jupyter kernel not found

```bash
# Solution: Reinstall kernel
python -m ipykernel install --user --name=sat-mlx-m4 --display-name "Python (sat-mlx-m4)"
```

### Training Issues

**Problem**: Out of memory during training

```python
# Solution 1: Reduce batch size (already at minimum)
# Solution 2: Reduce base filters
cfg.BASE_FILTERS = 24  # Instead of 32

# Solution 3: Skip progressive resizing
cfg.PROGRESSIVE_EPOCHS = {1024: 50}  # Train only at 1024
```

**Problem**: Slow training

```python
# Verify settings
cfg.MIXED_PRECISION = True  # Must be True
cfg.CHECKPOINT_DECODER = True  # Enable for memory

# Check GPU utilization
# Open Activity Monitor > Window > GPU History
# Should show >80% utilization
```

### Inference Issues

**Problem**: Core ML conversion fails

```bash
# Check PyTorch version
pip install torch==2.1.0 torchvision==0.16.0

# Verify conversion with validation
python mlx_to_coreml.py --mlx-weights best_model_mlx_1024_m4.safetensors
```

**Problem**: Slow inference

```bash
# Ensure using Core ML backend
python hybrid_inference.py --backend coreml --benchmark

# Check macOS version (needs 14+)
sw_vers
```

### Data Issues

**Problem**: No images found

```bash
# Check directory structure
ls train/*_sat.jpg | head -5
ls train/*_mask.png | head -5

# Verify pairing
python -c "
import glob
sats = set(f.replace('_sat.jpg', '') for f in glob.glob('train/*_sat.jpg'))
masks = set(f.replace('_mask.png', '') for f in glob.glob('train/*_mask.png'))
print(f'Satellite images: {len(sats)}')
print(f'Masks: {len(masks)}')
print(f'Matched pairs: {len(sats & masks)}')
"
```

---

## 📚 Additional Resources

- **Technical Details**: See [README_M4_UPGRADE.md](README_M4_UPGRADE.md)
- **API Documentation**: Check docstrings in each Python file
- **Issues**: [GitHub Issues](https://github.com/GaneshMakkena/Sat-U-Net/issues)

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- MLX framework by Apple ML Research
- U-Net architecture by Ronneberger et al.
- Core ML tools by Apple

---

## 📧 Contact

**Author**: Ganesh Makkena  
**Repository**: [https://github.com/GaneshMakkena/Sat-U-Net](https://github.com/GaneshMakkena/Sat-U-Net)

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for Apple Silicon**

*Last Updated: January 21, 2026*
