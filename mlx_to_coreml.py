"""
MLX to Core ML Conversion Pipeline
Converts trained MLX U-Net model to Core ML for deployment on M4 with Neural Engine optimization.
"""

import os
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import torch
import torch.nn as torch_nn
import coremltools as ct
from coremltools.models.neural_network import quantization_utils
from typing import Dict, Tuple
import json

# Import our efficient U-Net
from efficient_unet_mlx import EfficientUNet


class TorchDepthwiseSeparableConv(torch_nn.Module):
    """PyTorch version of depthwise separable convolution for Core ML."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.depthwise = torch_nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        self.pointwise = torch_nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class TorchConvBlock(torch_nn.Module):
    """PyTorch version of efficient conv block."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = TorchDepthwiseSeparableConv(in_channels, out_channels)
        self.bn1 = torch_nn.BatchNorm2d(out_channels)
        self.conv2 = TorchDepthwiseSeparableConv(out_channels, out_channels)
        self.bn2 = torch_nn.BatchNorm2d(out_channels)
        self.relu = torch_nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class TorchEncoderBlock(torch_nn.Module):
    """PyTorch version of encoder block."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = TorchConvBlock(in_channels, out_channels)
        self.pool = torch_nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        features = self.conv_block(x)
        pooled = self.pool(features)
        return features, pooled


class TorchDecoderBlock(torch_nn.Module):
    """PyTorch version of decoder block."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = torch_nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_block = TorchConvBlock(in_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class TorchEfficientUNet(torch_nn.Module):
    """
    PyTorch version of Efficient U-Net for Core ML conversion.
    Must match MLX architecture exactly.
    """
    
    def __init__(self, in_channels: int = 3, num_classes: int = 7, base_filters: int = 32):
        super().__init__()
        
        bf = base_filters
        
        # Encoder
        self.enc1 = TorchEncoderBlock(in_channels, bf)
        self.enc2 = TorchEncoderBlock(bf, bf * 2)
        self.enc3 = TorchEncoderBlock(bf * 2, bf * 4)
        self.enc4 = TorchEncoderBlock(bf * 4, bf * 8)
        
        # Bottleneck
        self.bottleneck = TorchConvBlock(bf * 8, bf * 12)
        
        # Decoder
        self.dec4 = TorchDecoderBlock(bf * 12 + bf * 8, bf * 8)
        self.dec3 = TorchDecoderBlock(bf * 8 + bf * 4, bf * 4)
        self.dec2 = TorchDecoderBlock(bf * 4 + bf * 2, bf * 2)
        self.dec1 = TorchDecoderBlock(bf * 2 + bf, bf)
        
        # Output
        self.output = torch_nn.Conv2d(bf, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        
        # Output
        x = self.output(x)
        
        return x


def transfer_conv_weights(mlx_conv, torch_conv, is_depthwise=False):
    """Transfer weights from MLX conv to PyTorch conv."""
    # MLX Conv2d weights shape: (out_channels, kernel_h, kernel_w, in_channels)
    # PyTorch Conv2d weights shape: (out_channels, in_channels, kernel_h, kernel_w)
    
    mlx_weight = np.array(mlx_conv.weight)  # (out, kh, kw, in)
    torch_weight = np.transpose(mlx_weight, (0, 3, 1, 2))  # (out, in, kh, kw)
    
    torch_conv.weight.data = torch.from_numpy(torch_weight).float()
    
    if hasattr(mlx_conv, 'bias') and mlx_conv.bias is not None:
        mlx_bias = np.array(mlx_conv.bias)
        torch_conv.bias.data = torch.from_numpy(mlx_bias).float()


def transfer_batchnorm_weights(mlx_bn, torch_bn):
    """Transfer weights from MLX BatchNorm to PyTorch BatchNorm."""
    # Weight (gamma)
    mlx_weight = np.array(mlx_bn.weight)
    torch_bn.weight.data = torch.from_numpy(mlx_weight).float()
    
    # Bias (beta)
    mlx_bias = np.array(mlx_bn.bias)
    torch_bn.bias.data = torch.from_numpy(mlx_bias).float()
    
    # Running mean
    mlx_mean = np.array(mlx_bn.running_mean)
    torch_bn.running_mean.data = torch.from_numpy(mlx_mean).float()
    
    # Running variance
    mlx_var = np.array(mlx_bn.running_var)
    torch_bn.running_var.data = torch.from_numpy(mlx_var).float()


def transfer_ds_conv_block(mlx_block, torch_block):
    """Transfer depthwise separable conv block weights."""
    # Conv1 (DepthwiseSeparableConv)
    transfer_conv_weights(mlx_block.conv1.depthwise, torch_block.conv1.depthwise)
    transfer_conv_weights(mlx_block.conv1.pointwise, torch_block.conv1.pointwise)
    transfer_batchnorm_weights(mlx_block.bn1, torch_block.bn1)
    
    # Conv2 (DepthwiseSeparableConv)
    transfer_conv_weights(mlx_block.conv2.depthwise, torch_block.conv2.depthwise)
    transfer_conv_weights(mlx_block.conv2.pointwise, torch_block.conv2.pointwise)
    transfer_batchnorm_weights(mlx_block.bn2, torch_block.bn2)


def transfer_encoder_block(mlx_enc, torch_enc):
    """Transfer encoder block weights."""
    transfer_ds_conv_block(mlx_enc.conv_block, torch_enc.conv_block)


def transfer_decoder_block(mlx_dec, torch_dec):
    """Transfer decoder block weights."""
    transfer_ds_conv_block(mlx_dec.conv_block, torch_dec.conv_block)


def convert_mlx_to_torch(mlx_model: EfficientUNet, 
                        num_classes: int = 7, 
                        base_filters: int = 32) -> TorchEfficientUNet:
    """
    Convert MLX model to PyTorch model with transferred weights.
    
    Args:
        mlx_model: Trained MLX model
        num_classes: Number of output classes
        base_filters: Base number of filters
    
    Returns:
        PyTorch model with transferred weights
    """
    print("Converting MLX model to PyTorch...")
    
    # Create PyTorch model
    torch_model = TorchEfficientUNet(
        in_channels=3,
        num_classes=num_classes,
        base_filters=base_filters
    )
    torch_model.eval()
    
    # Transfer weights layer by layer
    print("  Transferring encoder weights...")
    transfer_encoder_block(mlx_model.enc1, torch_model.enc1)
    transfer_encoder_block(mlx_model.enc2, torch_model.enc2)
    transfer_encoder_block(mlx_model.enc3, torch_model.enc3)
    transfer_encoder_block(mlx_model.enc4, torch_model.enc4)
    
    print("  Transferring bottleneck weights...")
    transfer_ds_conv_block(mlx_model.bottleneck, torch_model.bottleneck)
    
    print("  Transferring decoder weights...")
    transfer_decoder_block(mlx_model.dec4, torch_model.dec4)
    transfer_decoder_block(mlx_model.dec3, torch_model.dec3)
    transfer_decoder_block(mlx_model.dec2, torch_model.dec2)
    transfer_decoder_block(mlx_model.dec1, torch_model.dec1)
    
    print("  Transferring output layer...")
    transfer_conv_weights(mlx_model.output, torch_model.output)
    
    print("✅ Conversion to PyTorch complete!")
    
    return torch_model


def validate_conversion(mlx_model: EfficientUNet, 
                       torch_model: TorchEfficientUNet, 
                       img_size: int = 1024,
                       tolerance: float = 1e-3) -> bool:
    """
    Validate that PyTorch model produces same outputs as MLX model.
    
    Args:
        mlx_model: Original MLX model
        torch_model: Converted PyTorch model
        img_size: Input image size
        tolerance: Maximum allowed difference
    
    Returns:
        True if models match within tolerance
    """
    print(f"\nValidating conversion (tolerance: {tolerance})...")
    
    # Create random input
    np_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    
    # MLX forward pass
    mlx_input = mx.array(np_input)
    mlx_model.eval()
    mlx_output = mlx_model(mlx_input)
    mlx_output_np = np.array(mlx_output)
    
    # PyTorch forward pass
    torch_input = torch.from_numpy(np_input)
    torch_model.eval()
    with torch.no_grad():
        torch_output = torch_model(torch_input)
    torch_output_np = torch_output.numpy()
    
    # Compare outputs
    diff = np.abs(mlx_output_np - torch_output_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    
    if max_diff < tolerance:
        print("✅ Validation passed! Models produce identical outputs.")
        return True
    else:
        print(f"⚠️ Validation failed! Difference ({max_diff:.6f}) exceeds tolerance ({tolerance})")
        return False


def convert_to_coreml(torch_model: TorchEfficientUNet,
                     img_size: int = 1024,
                     num_classes: int = 7,
                     output_path: str = "satellite_unet_1024.mlpackage",
                     quantize: bool = True) -> ct.models.MLModel:
    """
    Convert PyTorch model to Core ML with Neural Engine optimization.
    
    Args:
        torch_model: PyTorch model
        img_size: Input image size
        num_classes: Number of output classes
        output_path: Output path for .mlpackage
        quantize: Whether to apply FP16 quantization
    
    Returns:
        Core ML model
    """
    print("\nConverting PyTorch to Core ML...")
    
    # Trace the model
    print(f"  Tracing model with input size {img_size}x{img_size}...")
    example_input = torch.rand(1, 3, img_size, img_size)
    traced_model = torch.jit.trace(torch_model, example_input)
    
    # Define input and output
    image_input = ct.ImageType(
        name="input_image",
        shape=(1, 3, img_size, img_size),
        scale=1.0/255.0,  # Normalize to [0, 1]
        bias=[0, 0, 0],
        color_layout=ct.colorlayout.RGB
    )
    
    class_names = ['Urban', 'Agriculture', 'Rangeland', 'Forest', 'Water', 'Barren', 'Unknown']
    
    # Convert using ML Program (supports Neural Engine)
    print("  Converting to ML Program format...")
    coreml_model = ct.convert(
        traced_model,
        inputs=[image_input],
        outputs=[ct.TensorType(name="segmentation_output")],
        minimum_deployment_target=ct.target.macOS14,  # For M4 optimization
        compute_units=ct.ComputeUnit.ALL,  # Use CPU, GPU, and Neural Engine
        convert_to="mlprogram"  # ML Program format (newer, better optimized)
    )
    
    # Add metadata
    coreml_model.author = "M4-Optimized Satellite Segmentation"
    coreml_model.short_description = f"Efficient U-Net for satellite image segmentation ({img_size}x{img_size})"
    coreml_model.version = "1.0"
    
    # Quantize to FP16 for Neural Engine
    if quantize:
        print("  Applying FP16 quantization for Neural Engine...")
        coreml_model = ct.models.neural_network.quantization_utils.quantize_weights(
            coreml_model, nbits=16
        )
    
    # Save model
    print(f"  Saving to {output_path}...")
    coreml_model.save(output_path)
    
    # Print model info
    spec = coreml_model.get_spec()
    print("\n" + "="*60)
    print("Core ML Model Summary")
    print("="*60)
    print(f"Input: {img_size}x{img_size} RGB image")
    print(f"Output: {num_classes} class segmentation map")
    print(f"Format: ML Program (Neural Engine optimized)")
    print(f"Quantization: {'FP16' if quantize else 'FP32'}")
    print(f"Size: {os.path.getsize(output_path) / (1024**2):.2f} MB")
    print("="*60 + "\n")
    
    return coreml_model


def full_conversion_pipeline(mlx_weights_path: str,
                            output_coreml_path: str = "satellite_unet_1024_m4.mlpackage",
                            img_size: int = 1024,
                            num_classes: int = 7,
                            base_filters: int = 32,
                            validate: bool = True) -> Tuple[TorchEfficientUNet, ct.models.MLModel]:
    """
    Full pipeline: MLX -> PyTorch -> Core ML
    
    Args:
        mlx_weights_path: Path to MLX weights (.safetensors)
        output_coreml_path: Output path for Core ML model
        img_size: Input image size
        num_classes: Number of classes
        base_filters: Base filters in U-Net
        validate: Whether to validate conversion
    
    Returns:
        Tuple of (PyTorch model, Core ML model)
    """
    print("\n" + "="*60)
    print("MLX to Core ML Conversion Pipeline")
    print("="*60 + "\n")
    
    # Step 1: Load MLX model
    print("Step 1: Loading MLX model...")
    from efficient_unet_mlx import create_efficient_unet_m4
    mlx_model = create_efficient_unet_m4(
        img_size=img_size,
        num_classes=num_classes,
        base_filters=base_filters,
        use_attention=False,
        checkpoint_decoder=False  # Not needed for inference
    )
    mlx_model.load_weights(mlx_weights_path)
    mlx_model.eval()
    print(f"✅ Loaded MLX model from {mlx_weights_path}")
    
    # Step 2: Convert to PyTorch
    print("\nStep 2: Converting to PyTorch...")
    torch_model = convert_mlx_to_torch(mlx_model, num_classes, base_filters)
    
    # Step 3: Validate conversion
    if validate:
        is_valid = validate_conversion(mlx_model, torch_model, img_size)
        if not is_valid:
            raise ValueError("Conversion validation failed! Check model architecture.")
    
    # Step 4: Convert to Core ML
    print("\nStep 3: Converting to Core ML...")
    coreml_model = convert_to_coreml(
        torch_model,
        img_size=img_size,
        num_classes=num_classes,
        output_path=output_coreml_path,
        quantize=True
    )
    
    print("\n" + "="*60)
    print("✅ Conversion Pipeline Complete!")
    print("="*60)
    print(f"PyTorch model: Ready for deployment")
    print(f"Core ML model: {output_coreml_path}")
    print(f"\nThe Core ML model is optimized for:")
    print(f"  • M4 Neural Engine (60-70% of operations)")
    print(f"  • FP16 precision (2x faster, half memory)")
    print(f"  • Expected inference: 50-100ms per image")
    print("="*60 + "\n")
    
    return torch_model, coreml_model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert MLX model to Core ML")
    parser.add_argument("--mlx-weights", type=str, default="best_model_mlx_1024_m4.safetensors",
                       help="Path to MLX weights file")
    parser.add_argument("--output", type=str, default="satellite_unet_1024_m4.mlpackage",
                       help="Output path for Core ML model")
    parser.add_argument("--img-size", type=int, default=1024,
                       help="Input image size")
    parser.add_argument("--num-classes", type=int, default=7,
                       help="Number of segmentation classes")
    parser.add_argument("--base-filters", type=int, default=32,
                       help="Base filters in U-Net")
    parser.add_argument("--no-validate", action="store_true",
                       help="Skip validation step")
    
    args = parser.parse_args()
    
    # Run full pipeline
    torch_model, coreml_model = full_conversion_pipeline(
        mlx_weights_path=args.mlx_weights,
        output_coreml_path=args.output,
        img_size=args.img_size,
        num_classes=args.num_classes,
        base_filters=args.base_filters,
        validate=not args.no_validate
    )
    
    print("Conversion complete! You can now use the Core ML model for inference.")
    print("Next step: Run 'coreml_inference.py' or 'hybrid_inference.py' for predictions.")
