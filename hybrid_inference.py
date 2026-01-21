"""
Hybrid Inference Pipeline
Unified API supporting MLX, Core ML, and PyTorch MPS backends with automatic selection.
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Optional, Literal
import time
from dataclasses import dataclass


@dataclass
class BackendInfo:
    """Information about an inference backend."""
    name: str
    available: bool
    speed_rank: int  # 1 = fastest
    memory_mb: float
    description: str


class HybridPredictor:
    """
    Unified predictor supporting multiple backends.
    Automatically selects the best available backend or allows manual selection.
    """
    
    BACKENDS = ['coreml', 'mlx', 'pytorch_mps']
    
    def __init__(self, 
                 backend: Literal['auto', 'coreml', 'mlx', 'pytorch_mps'] = 'auto',
                 model_path: Optional[str] = None,
                 img_size: int = 1024,
                 num_classes: int = 7,
                 base_filters: int = 32):
        """
        Initialize hybrid predictor.
        
        Args:
            backend: Backend to use ('auto' for automatic selection)
            model_path: Path to model (backend-specific)
            img_size: Input image size
            num_classes: Number of classes
            base_filters: Base filters in U-Net
        """
        self.img_size = img_size
        self.num_classes = num_classes
        self.base_filters = base_filters
        
        # Class names and colors
        self.class_names = ['Urban', 'Agriculture', 'Rangeland', 'Forest', 
                           'Water', 'Barren', 'Unknown']
        self.colors = {
            0: (0, 255, 255),    # Urban - Cyan
            1: (255, 255, 0),    # Agriculture - Yellow
            2: (255, 0, 255),    # Rangeland - Magenta
            3: (0, 255, 0),      # Forest - Green
            4: (0, 0, 255),      # Water - Blue
            5: (255, 255, 255),  # Barren - White
            6: (0, 0, 0)         # Unknown - Black
        }
        
        # Detect available backends
        self.available_backends = self._detect_backends()
        
        # Select backend
        if backend == 'auto':
            self.backend = self._select_best_backend()
        else:
            if backend not in self.available_backends:
                raise ValueError(f"Backend '{backend}' not available. Available: {list(self.available_backends.keys())}")
            self.backend = backend
        
        print(f"\n{'='*60}")
        print(f"Initializing Hybrid Predictor")
        print(f"{'='*60}")
        print(f"Selected Backend: {self.backend.upper()}")
        print(f"Available Backends: {', '.join(self.available_backends.keys())}")
        print(f"{'='*60}\n")
        
        # Initialize the selected backend
        self.model = self._init_backend(self.backend, model_path)
    
    def _detect_backends(self) -> Dict[str, BackendInfo]:
        """Detect which backends are available on this system."""
        backends = {}
        
        # Check Core ML
        try:
            import coremltools
            backends['coreml'] = BackendInfo(
                name='Core ML',
                available=True,
                speed_rank=1,
                memory_mb=2048,
                description='Fastest, Neural Engine optimized, inference only'
            )
        except ImportError:
            pass
        
        # Check MLX
        try:
            import mlx.core as mx
            if mx.metal.is_available():
                backends['mlx'] = BackendInfo(
                    name='MLX',
                    available=True,
                    speed_rank=2,
                    memory_mb=3072,
                    description='Fast, Metal GPU optimized, training + inference'
                )
        except (ImportError, Exception):
            pass
        
        # Check PyTorch MPS
        try:
            import torch
            if torch.backends.mps.is_available():
                backends['pytorch_mps'] = BackendInfo(
                    name='PyTorch MPS',
                    available=True,
                    speed_rank=3,
                    memory_mb=4096,
                    description='Compatible, Metal GPU, fallback option'
                )
        except (ImportError, AttributeError):
            pass
        
        return backends
    
    def _select_best_backend(self) -> str:
        """Select the best available backend based on speed ranking."""
        if not self.available_backends:
            raise RuntimeError("No backends available! Install at least one: coremltools, mlx, or pytorch")
        
        # Sort by speed rank
        sorted_backends = sorted(
            self.available_backends.items(),
            key=lambda x: x[1].speed_rank
        )
        
        return sorted_backends[0][0]
    
    def _init_backend(self, backend: str, model_path: Optional[str]):
        """Initialize the selected backend."""
        if backend == 'coreml':
            return self._init_coreml(model_path or 'satellite_unet_1024_m4.mlpackage')
        elif backend == 'mlx':
            return self._init_mlx(model_path or 'best_model_mlx_1024_m4.safetensors')
        elif backend == 'pytorch_mps':
            return self._init_pytorch(model_path or 'satellite_unet_1024_pytorch.pth')
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _init_coreml(self, model_path: str):
        """Initialize Core ML backend."""
        import coremltools as ct
        print(f"Loading Core ML model from {model_path}...")
        model = ct.models.MLModel(model_path)
        print("✅ Core ML model loaded")
        return model
    
    def _init_mlx(self, model_path: str):
        """Initialize MLX backend."""
        import mlx.core as mx
        from efficient_unet_mlx import create_efficient_unet_m4
        
        print(f"Loading MLX model from {model_path}...")
        model = create_efficient_unet_m4(
            img_size=self.img_size,
            num_classes=self.num_classes,
            base_filters=self.base_filters,
            use_attention=False,
            checkpoint_decoder=False
        )
        model.load_weights(model_path)
        model.eval()
        print("✅ MLX model loaded")
        return model
    
    def _init_pytorch(self, model_path: str):
        """Initialize PyTorch MPS backend."""
        import torch
        from mlx_to_coreml import TorchEfficientUNet
        
        print(f"Loading PyTorch model from {model_path}...")
        model = TorchEfficientUNet(
            in_channels=3,
            num_classes=self.num_classes,
            base_filters=self.base_filters
        )
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
        else:
            print(f"⚠️ Model file not found: {model_path}")
            print("Using randomly initialized model for demonstration")
        
        model.eval()
        model.to('mps')
        print("✅ PyTorch model loaded on MPS")
        return model
    
    def preprocess(self, image_path: str) -> np.ndarray:
        """Preprocess image for inference."""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        return img
    
    def predict(self, image_path: str) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Run inference using the selected backend.
        
        Args:
            image_path: Path to input image
        
        Returns:
            Tuple of (segmentation_mask, class_composition)
        """
        # Preprocess
        img = self.preprocess(image_path)
        
        # Run inference based on backend
        start_time = time.time()
        
        if self.backend == 'coreml':
            mask = self._predict_coreml(img)
        elif self.backend == 'mlx':
            mask = self._predict_mlx(img)
        elif self.backend == 'pytorch_mps':
            mask = self._predict_pytorch(img)
        
        inference_time = time.time() - start_time
        
        # Calculate composition
        composition = self.calculate_composition(mask)
        
        print(f"✅ [{self.backend.upper()}] Inference: {inference_time*1000:.1f}ms")
        
        return mask, composition
    
    def _predict_coreml(self, img: np.ndarray) -> np.ndarray:
        """Core ML inference."""
        pil_img = Image.fromarray(img)
        output = self.model.predict({'input_image': pil_img})
        seg_output = output['segmentation_output']
        mask = np.argmax(seg_output[0], axis=0)
        return mask
    
    def _predict_mlx(self, img: np.ndarray) -> np.ndarray:
        """MLX inference."""
        import mlx.core as mx
        
        # Normalize and convert to CHW
        img_norm = img.astype(np.float32) / 255.0
        img_chw = np.transpose(img_norm, (2, 0, 1))
        img_tensor = mx.array(img_chw[np.newaxis, ...])
        
        # Forward pass
        output = self.model(img_tensor)
        mask = np.array(mx.argmax(output, axis=1).squeeze())
        
        return mask
    
    def _predict_pytorch(self, img: np.ndarray) -> np.ndarray:
        """PyTorch MPS inference."""
        import torch
        
        # Normalize and convert to CHW
        img_norm = img.astype(np.float32) / 255.0
        img_chw = np.transpose(img_norm, (2, 0, 1))
        img_tensor = torch.from_numpy(img_chw[np.newaxis, ...]).to('mps')
        
        # Forward pass
        with torch.no_grad():
            output = self.model(img_tensor)
        
        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        return mask
    
    def calculate_composition(self, mask: np.ndarray) -> Dict[str, float]:
        """Calculate class composition."""
        composition = {}
        total_pixels = mask.size
        
        for cls_id, cls_name in enumerate(self.class_names):
            count = np.sum(mask == cls_id)
            percentage = (count / total_pixels) * 100
            composition[cls_name] = round(percentage, 2)
        
        return composition
    
    def mask_to_rgb(self, mask: np.ndarray) -> np.ndarray:
        """Convert class mask to RGB."""
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        
        for cls_id, color in self.colors.items():
            rgb[mask == cls_id] = color
        
        return rgb
    
    def visualize_prediction(self, image_path: str, output_path: Optional[str] = None):
        """Visualize prediction."""
        import matplotlib.pyplot as plt
        
        # Predict
        mask, composition = self.predict(image_path)
        
        # Load original
        img = self.preprocess(image_path)
        mask_rgb = self.mask_to_rgb(mask)
        overlay = cv2.addWeighted(img, 0.6, mask_rgb, 0.4, 0)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(img)
        axes[0].set_title('Original', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(mask_rgb)
        axes[1].set_title(f'Segmentation ({self.backend.upper()})', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved to {output_path}")
        
        plt.show()
        
        # Print composition
        print("\nClass Composition:")
        print("-" * 40)
        for cls_name, pct in composition.items():
            if pct > 0.1:
                print(f"{cls_name:15s}: {pct:6.2f}%")
        print("-" * 40)
    
    def benchmark(self, num_iterations: int = 50) -> Dict[str, float]:
        """
        Benchmark the current backend.
        
        Args:
            num_iterations: Number of iterations
        
        Returns:
            Dictionary with benchmark results
        """
        print(f"\nBenchmarking {self.backend.upper()} ({num_iterations} iterations)...")
        
        # Create dummy image
        dummy_img = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Warm-up
        for _ in range(5):
            if self.backend == 'coreml':
                self._predict_coreml(dummy_img)
            elif self.backend == 'mlx':
                self._predict_mlx(dummy_img)
            elif self.backend == 'pytorch_mps':
                self._predict_pytorch(dummy_img)
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.time()
            if self.backend == 'coreml':
                self._predict_coreml(dummy_img)
            elif self.backend == 'mlx':
                self._predict_mlx(dummy_img)
            elif self.backend == 'pytorch_mps':
                self._predict_pytorch(dummy_img)
            times.append(time.time() - start)
        
        times = np.array(times)
        
        results = {
            'backend': self.backend,
            'mean_ms': float(np.mean(times) * 1000),
            'std_ms': float(np.std(times) * 1000),
            'min_ms': float(np.min(times) * 1000),
            'max_ms': float(np.max(times) * 1000),
            'median_ms': float(np.median(times) * 1000),
            'throughput_fps': float(1.0 / np.mean(times))
        }
        
        print(f"\n{'='*60}")
        print(f"Benchmark Results - {self.backend.upper()}")
        print(f"{'='*60}")
        print(f"Mean:       {results['mean_ms']:.1f} ± {results['std_ms']:.1f} ms")
        print(f"Median:     {results['median_ms']:.1f} ms")
        print(f"Min/Max:    {results['min_ms']:.1f} / {results['max_ms']:.1f} ms")
        print(f"Throughput: {results['throughput_fps']:.1f} FPS")
        print(f"{'='*60}\n")
        
        return results
    
    def compare_backends(self, num_iterations: int = 30):
        """
        Compare all available backends.
        
        Args:
            num_iterations: Iterations per backend
        """
        print(f"\n{'='*60}")
        print("Comparing All Available Backends")
        print(f"{'='*60}\n")
        
        results = {}
        
        for backend in self.available_backends.keys():
            print(f"\nTesting {backend.upper()}...")
            
            try:
                # Create temporary predictor for this backend
                temp_predictor = HybridPredictor(
                    backend=backend,
                    img_size=self.img_size,
                    num_classes=self.num_classes,
                    base_filters=self.base_filters
                )
                
                # Benchmark
                bench_results = temp_predictor.benchmark(num_iterations)
                results[backend] = bench_results
            
            except Exception as e:
                print(f"❌ Error benchmarking {backend}: {e}")
                results[backend] = None
        
        # Print comparison
        print(f"\n{'='*60}")
        print("Backend Comparison Summary")
        print(f"{'='*60}\n")
        
        print(f"{'Backend':<15} {'Mean (ms)':<12} {'Median (ms)':<12} {'FPS':<8}")
        print("-" *60)
        
        for backend, res in results.items():
            if res:
                print(f"{backend.upper():<15} {res['mean_ms']:<12.1f} {res['median_ms']:<12.1f} {res['throughput_fps']:<8.1f}")
            else:
                print(f"{backend.upper():<15} {'Failed':<12} {'Failed':<12} {'N/A':<8}")
        
        print(f"{'='*60}\n")
        
        # Recommend best backend
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            best = min(valid_results.items(), key=lambda x: x[1]['mean_ms'])
            print(f"🏆 Recommended: {best[0].upper()} ({best[1]['mean_ms']:.1f}ms)")
        
        return results
    
    def print_backend_info(self):
        """Print information about all backends."""
        print(f"\n{'='*60}")
        print("Available Backends")
        print(f"{'='*60}\n")
        
        for backend, info in self.available_backends.items():
            status = "✅" if info.available else "❌"
            print(f"{status} {info.name} ({backend})")
            print(f"   Speed Rank: #{info.speed_rank}")
            print(f"   Memory: ~{info.memory_mb/1024:.1f} GB")
            print(f"   {info.description}")
            print()
        
        print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid inference for satellite segmentation")
    parser.add_argument("--backend", type=str, default="auto",
                       choices=['auto', 'coreml', 'mlx', 'pytorch_mps'],
                       help="Backend to use")
    parser.add_argument("--image", type=str, help="Image to predict")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark current backend")
    parser.add_argument("--compare", action="store_true", help="Compare all backends")
    parser.add_argument("--info", action="store_true", help="Show backend info")
    parser.add_argument("--img-size", type=int, default=1024, help="Image size")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = HybridPredictor(backend=args.backend, img_size=args.img_size)
    
    if args.info:
        predictor.print_backend_info()
    
    elif args.compare:
        predictor.compare_backends(num_iterations=30)
    
    elif args.benchmark:
        predictor.benchmark(num_iterations=50)
    
    elif args.image:
        predictor.visualize_prediction(args.image, output_path="hybrid_prediction.png")
    
    else:
        print("Please specify --image, --benchmark, --compare, or --info")
        print("\nUsage examples:")
        print("  python hybrid_inference.py --info")
        print("  python hybrid_inference.py --backend coreml --image test.jpg")
        print("  python hybrid_inference.py --compare")
        print("  python hybrid_inference.py --benchmark")
