# Production-Grade Workflow (NVIDIA L4, 1024x1024)

This document describes the production-grade pipeline created in `Sat_U_Net_Production.ipynb` for the DeepGlobe land-cover dataset.

## What This Fixes

1. Uses class-index masks instead of RGB reconstruction.
2. Uses mIoU and per-class IoU instead of pixel accuracy.
3. Uses AMP (mixed precision) and gradient accumulation to fit 1024x1024 on L4.
4. Uses deterministic splits when validation masks are missing.
5. Uses mask-safe transforms (nearest-neighbor for labels).
6. Supports tiled inference for 2048x2048 input.

## Environment Setup (Windows + L4)

Install Python 3.10 or 3.11, then create a fresh virtual environment.

1. Create and activate venv
   - `python -m venv .venv`
   - `.\venv\Scripts\activate`
2. Install PyTorch with CUDA
   - Use the official PyTorch selector for your CUDA build.
   - For CUDA 12.1 as an example:
   - `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
3. Install the rest
   - `pip install -r requirements.txt`

Note: CUDA 13.1 is installed on your system, but PyTorch ships with its own CUDA runtime. The recommended approach is to install the matching PyTorch wheel for CUDA 12.1 or 12.4, then rely on the bundled runtime.

## Data Layout

Your dataset root must include:

1. `metadata.csv`
2. `class_dict.csv`
3. `train/` with both `*_sat.jpg` and `*_mask.png`
4. `valid/` and `test/` images (masks optional)

If `valid/` masks are missing, the notebook will split the training set into train and validation automatically.

## Run the Notebook

1. Open `Sat_U_Net_Production.ipynb`.
2. Update `CFG.data_root` to your dataset root.
3. Set `CFG.image_size = 1024` and tune `CFG.batch_size`.
4. Choose the model type:
   - `unet_resnet34` (default, fast and stable)
   - `segformer_b2` (transformers v5, heavier but strong)
5. Run the training loop.

Artifacts are saved under `runs/l4_1024/`.

## Performance Tips for L4 (24 GB)

1. Keep `batch_size` at 1 or 2 for 1024x1024.
2. Use `grad_accum_steps` to simulate a larger batch.
3. Keep `amp=True` for mixed precision.
4. Use tiled inference for 2048x2048 images.

## Troubleshooting

1. CUDA out of memory
   - Reduce `batch_size` or increase `grad_accum_steps`.
2. Slow data loading
   - Increase `num_workers` and keep `pin_memory=True`.
3. Validation masks missing
   - The notebook auto-splits training into train and val.
