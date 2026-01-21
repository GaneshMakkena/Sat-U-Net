# Setup Guide for New Mac

This guide helps you set up the Satellite Segmentation project on a new M-series Mac.

---

## Prerequisites Checklist

Before starting, ensure you have:

- [ ] M1/M2/M3/M4 MacBook with **16GB+ RAM**
- [ ] macOS **14.0 or later** (Sonoma+)
- [ ] **10GB free disk space**
- [ ] **Internet connection** for downloading dependencies
- [ ] **Git** installed (`git --version`)
- [ ] **Python 3.11+** installed (`python3 --version`)

---

## Step-by-Step Installation

### 1. Install Homebrew (if not installed)

```bash
# Check if Homebrew is installed
which brew

# If not installed, run:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install Python 3.11+

```bash
# Check Python version
python3 --version

# If < 3.11, install via Homebrew
brew install python@3.11

# Verify
python3 --version  # Should show 3.11.x or higher
```

### 3. Clone Repository

```bash
# Navigate to desired location
cd ~/Documents  # or wherever you want

# Clone repository
git clone https://github.com/GaneshMakkena/Sat-U-Net.git

# Enter directory
cd Sat-U-Net

# Verify files
ls -la
```

**Expected output**: You should see all project files including `requirements.txt`, notebooks, and Python scripts.

### 4. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv sat-mlx-env

# Activate environment
source sat-mlx-env/bin/activate

# Verify activation (prompt should show (sat-mlx-env))
```

**✅ Success indicator**: Your terminal prompt should now start with `(sat-mlx-env)`.

### 5. Upgrade pip and Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# This will take 5-10 minutes
```

**Note**: If you see any warnings about dependencies, that's normal. As long as no errors occur, continue.

### 6. Verify Installation

```bash
# Test MLX
python3 << EOF
import mlx.core as mx
print(f"MLX Version: {mx.__version__}")
print(f"Metal GPU Available: {mx.metal.is_available()}")
print(f"Default Device: {mx.default_device()}")
EOF
```

**Expected output**:
```
MLX Version: 0.x.x
Metal GPU Available: True
Default Device: gpu
```

```bash
# Test Core ML Tools
python3 -c "import coremltools; print(f'CoreMLTools: {coremltools.__version__}')"

# Test PyTorch
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Test other dependencies
python3 -c "import cv2, numpy, pandas, matplotlib; print('✅ All dependencies installed')"
```

### 7. Set Up Jupyter (Optional but Recommended)

```bash
# Install Jupyter kernel
python -m ipykernel install --user --name=sat-mlx-m4 --display-name "Python (sat-mlx-m4)"

# Verify kernel installation
jupyter kernelspec list
```

**Expected output**: Should show `sat-mlx-m4` in the list.

### 8. Verify Dataset

```bash
# Check training data
echo "Training images: $(ls train/*_sat.jpg 2>/dev/null | wc -l)"
echo "Training masks: $(ls train/*_mask.png 2>/dev/null | wc -l)"

# Check validation data
echo "Validation images: $(ls valid/*_sat.jpg 2>/dev/null | wc -l)"
echo "Validation masks: $(ls valid/*_mask.png 2>/dev/null | wc -l)"

# Check test data
echo "Test images: $(ls test/*_sat.jpg 2>/dev/null | wc -l)"
echo "Test masks: $(ls test/*_mask.png 2>/dev/null | wc -l)"
```

**Expected output**: Non-zero counts for each category.

**If dataset is missing**: The repository includes the dataset. If it's not there, you may need to download it separately or use Git LFS (Large File Storage).

---

## Quick Test

### Test 1: Memory Estimation

```bash
python3 << EOF
from memory_utils import estimate_memory_usage

# Estimate memory for 1024x1024 training
mem = estimate_memory_usage(1024, 1, 32, 7, True)
print(f"\nMemory Estimate for 1024×1024:")
print(f"  Total: {mem['total_gb']:.2f} GB")
print(f"  Safe for 16GB: {'✅ Yes' if mem['safe_for_16gb'] else '❌ No'}")
EOF
```

**Expected output**: Total ~8.7 GB, Safe: ✅ Yes

### Test 2: Model Creation

```bash
python3 << EOF
from efficient_unet_mlx import create_efficient_unet_m4

# Create model
model = create_efficient_unet_m4(1024, 7, 32)
print("\n✅ Model created successfully!")
model.print_architecture_summary()
EOF
```

**Expected output**: Model architecture summary with ~3.5M parameters.

### Test 3: Backend Detection

```bash
python3 << EOF
from hybrid_inference import HybridPredictor

# Initialize predictor
predictor = HybridPredictor(backend='auto', img_size=1024)
predictor.print_backend_info()
EOF
```

**Expected output**: List of available backends (should include at least MLX and Core ML).

---

## Next Steps

### For Training

1. **Open Jupyter Notebook**:
   ```bash
   jupyter notebook Sat_IMG_UNet_MLX_1024_M4.ipynb
   ```

2. **Select Kernel**: Choose `Python (sat-mlx-m4)` from the kernel dropdown

3. **Run Training**: Execute cells sequentially or all at once

4. **Monitor**: Watch Activity Monitor for memory and GPU usage

**Expected time**: ~12 hours for full training (50 epochs)

### For Inference Only

If you already have a trained model:

1. **Convert to Core ML**:
   ```bash
   python mlx_to_coreml.py \
       --mlx-weights best_model_mlx_1024_m4.safetensors \
       --output satellite_unet_1024_m4.mlpackage
   ```

2. **Run Inference**:
   ```bash
   python hybrid_inference.py --image test/example_sat.jpg
   ```

---

## Troubleshooting Setup Issues

### Issue: "MLX not available" or "Metal GPU not found"

**Solution**:
```bash
# Reinstall MLX
pip uninstall mlx mlx-nn
pip install mlx>=0.4.0

# Verify
python3 -c "import mlx.core as mx; print(mx.metal.is_available())"
```

### Issue: "No module named 'torch'"

**Solution**:
```bash
# Install PyTorch
pip install torch>=2.1.0 torchvision>=0.16.0

# Verify
python3 -c "import torch; print(torch.__version__)"
```

### Issue: "coremltools not found"

**Solution**:
```bash
# Install Core ML Tools
pip install coremltools>=7.1

# Verify
python3 -c "import coremltools; print(coremltools.__version__)"
```

### Issue: Jupyter kernel not found

**Solution**:
```bash
# Remove old kernel
jupyter kernelspec uninstall sat-mlx-m4

# Reinstall
python -m ipykernel install --user --name=sat-mlx-m4

# Restart Jupyter
```

### Issue: Permission denied when installing packages

**Solution**:
```bash
# Ensure you're in virtual environment
source sat-mlx-env/bin/activate

# If still issues, try without sudo:
pip install --user -r requirements.txt
```

### Issue: Dataset missing or incomplete

**Solution**:
```bash
# Check if Git LFS is needed
git lfs install
git lfs pull

# Or download dataset separately (if provided externally)
```

---

## Environment Variables (Optional)

For better performance, you can set:

```bash
# Add to ~/.zshrc or ~/.bash_profile
export PYTORCH_ENABLE_MPS_FALLBACK=1
export MLXBACKEND=metal

# Apply changes
source ~/.zshrc  # or ~/.bash_profile
```

---

## System Requirements Summary

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Chip** | M1 | M4 |
| **RAM** | 16GB | 16GB+ |
| **Storage** | 10GB free | 20GB+ free |
| **macOS** | 14.0 (Sonoma) | 14.5+ |
| **Python** | 3.11 | 3.11+ |

---

## Verification Checklist

Before starting training or inference, verify:

- [ ] Python 3.11+ installed
- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip list | grep mlx`)
- [ ] MLX Metal GPU available
- [ ] Jupyter kernel installed (if using notebooks)
- [ ] Dataset present and properly structured
- [ ] Memory estimation shows <14GB for training
- [ ] At least 10GB free disk space

---

## Getting Help

If you encounter issues:

1. **Check README.md** - Main documentation
2. **Check README_M4_UPGRADE.md** - Technical details
3. **Review error messages** - Often self-explanatory
4. **Search GitHub Issues** - Someone may have had the same problem
5. **Open a new issue** - Describe your problem with:
   - macOS version (`sw_vers`)
   - Python version (`python3 --version`)
   - Error message (full traceback)
   - Steps to reproduce

---

## Success Indicators

You've successfully set up the environment when:

✅ MLX reports Metal GPU as available  
✅ All dependencies install without errors  
✅ Memory estimation shows safe memory usage  
✅ Model creation works without errors  
✅ Backend detection finds at least MLX  
✅ Jupyter kernel appears in kernel list  

**You're ready to start training or inference!**

---

*Setup guide last updated: January 21, 2026*
