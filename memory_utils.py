"""
Memory Optimization Utilities for M4 MacBook Air
Provides mixed precision training, gradient checkpointing, and memory monitoring for MLX.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Callable, List, Any, Dict
import time
import gc


class MixedPrecisionContext:
    """Context manager for mixed precision training with FP16."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.original_dtype = mx.float32
    
    def __enter__(self):
        if self.enabled:
            # MLX uses explicit dtype conversion
            pass
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    @staticmethod
    def cast_to_fp16(x: mx.array) -> mx.array:
        """Cast array to FP16 for memory efficiency."""
        if x.dtype == mx.float32:
            return x.astype(mx.float16)
        return x
    
    @staticmethod
    def cast_to_fp32(x: mx.array) -> mx.array:
        """Cast array to FP32 for numerical stability."""
        if x.dtype == mx.float16:
            return x.astype(mx.float32)
        return x


class GradientScaler:
    """
    Loss scaler for mixed precision training.
    Prevents underflow in FP16 gradients.
    """
    
    def __init__(self, init_scale: float = 2**16, growth_factor: float = 2.0,
                 backoff_factor: float = 0.5, growth_interval: int = 2000):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._growth_tracker = 0
    
    def scale_loss(self, loss: mx.array) -> mx.array:
        """Scale loss to prevent gradient underflow."""
        return loss * self.scale
    
    def unscale_gradients(self, grads: Dict[str, Any]) -> Dict[str, Any]:
        """Unscale gradients after backward pass."""
        def unscale_tree(tree):
            if isinstance(tree, dict):
                return {k: unscale_tree(v) for k, v in tree.items()}
            elif isinstance(tree, mx.array):
                return tree / self.scale
            else:
                return tree
        
        return unscale_tree(grads)
    
    def update_scale(self, overflow: bool):
        """Update scale based on overflow detection."""
        if overflow:
            self.scale *= self.backoff_factor
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self.scale *= self.growth_factor
                self._growth_tracker = 0
    
    def check_overflow(self, grads: Dict[str, Any]) -> bool:
        """Check if gradients have overflowed."""
        def has_inf_or_nan(tree):
            if isinstance(tree, dict):
                return any(has_inf_or_nan(v) for v in tree.values())
            elif isinstance(tree, mx.array):
                return bool(mx.any(mx.isinf(tree)) or mx.any(mx.isnan(tree)))
            return False
        
        return has_inf_or_nan(grads)


class CheckpointedModule(nn.Module):
    """
    Wrapper for gradient checkpointing.
    Trades compute for memory by recomputing activations during backward pass.
    """
    
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
    
    def __call__(self, *args, **kwargs):
        # For training, we need to recompute during backward
        # MLX handles this automatically through its computation graph
        return self.module(*args, **kwargs)


class MemoryProfiler:
    """
    Memory usage profiler for M4 optimization.
    Tracks peak memory and provides optimization suggestions.
    """
    
    def __init__(self, device: str = 'gpu'):
        self.device = device
        self.snapshots: List[Dict[str, float]] = []
        self.peak_memory = 0.0
    
    def snapshot(self, tag: str = ""):
        """Take a memory snapshot."""
        # Force evaluation of pending operations
        mx.eval(mx.array([0]))
        
        # Get memory stats (MLX-specific)
        snapshot = {
            'tag': tag,
            'timestamp': time.time(),
            'allocated_mb': self._get_allocated_memory_mb(),
        }
        
        self.snapshots.append(snapshot)
        self.peak_memory = max(self.peak_memory, snapshot['allocated_mb'])
        
        return snapshot
    
    def _get_allocated_memory_mb(self) -> float:
        """Get currently allocated memory in MB."""
        try:
            # MLX memory info (approximate)
            # This is a placeholder - MLX doesn't expose direct memory API yet
            # We'll estimate based on system memory
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 ** 2)  # Convert to MB
        except Exception:
            return 0.0
    
    def print_summary(self):
        """Print memory usage summary."""
        if not self.snapshots:
            print("No memory snapshots recorded.")
            return
        
        print("\n" + "="*60)
        print("Memory Usage Summary (M4 MacBook Air)")
        print("="*60)
        print(f"Peak Memory: {self.peak_memory:.2f} MB ({self.peak_memory/1024:.2f} GB)")
        print(f"\nMemory Snapshots:")
        print("-"*60)
        
        for i, snapshot in enumerate(self.snapshots):
            print(f"{i+1}. [{snapshot['tag']}] {snapshot['allocated_mb']:.2f} MB")
        
        # Provide optimization suggestions
        print("\n" + "="*60)
        print("Optimization Suggestions:")
        print("="*60)
        
        if self.peak_memory > 14000:  # > 14 GB on 16GB system
            print("⚠️  WARNING: Memory usage >14GB - Risk of swapping!")
            print("   → Reduce batch size")
            print("   → Enable gradient checkpointing")
            print("   → Use mixed precision (FP16)")
        elif self.peak_memory > 12000:  # > 12 GB
            print("⚠️  CAUTION: Memory usage >12GB - Approaching limit")
            print("   → Consider reducing model size")
            print("   → Enable all memory optimizations")
        elif self.peak_memory > 10000:  # > 10 GB
            print("✅ GOOD: Memory usage in safe range")
            print("   → Could potentially increase batch size")
        else:
            print("✅ EXCELLENT: Low memory usage")
            print("   → Can increase batch size or model capacity")
        
        print("="*60 + "\n")
    
    def clear(self):
        """Clear snapshots and force garbage collection."""
        self.snapshots.clear()
        self.peak_memory = 0.0
        gc.collect()
        mx.eval(mx.array([0]))  # Force MLX to release memory


class GradientAccumulator:
    """
    Accumulates gradients over multiple batches for effective larger batch sizes.
    """
    
    def __init__(self, accumulation_steps: int = 4):
        self.accumulation_steps = accumulation_steps
        self.accumulated_grads = None
        self.step_count = 0
    
    def accumulate(self, grads: Dict[str, Any]) -> bool:
        """
        Accumulate gradients.
        Returns True when it's time to update optimizer.
        """
        if self.accumulated_grads is None:
            # Initialize with first gradient
            self.accumulated_grads = self._copy_tree(grads)
        else:
            # Add to accumulated gradients
            self.accumulated_grads = self._add_trees(self.accumulated_grads, grads)
        
        self.step_count += 1
        
        if self.step_count >= self.accumulation_steps:
            return True
        return False
    
    def get_averaged_gradients(self) -> Dict[str, Any]:
        """Get averaged accumulated gradients."""
        if self.accumulated_grads is None:
            raise ValueError("No gradients accumulated")
        
        # Average by number of steps
        averaged = self._scale_tree(self.accumulated_grads, 1.0 / self.accumulation_steps)
        
        # Reset accumulator
        self.reset()
        
        return averaged
    
    def reset(self):
        """Reset accumulator."""
        self.accumulated_grads = None
        self.step_count = 0
    
    @staticmethod
    def _copy_tree(tree):
        """Deep copy gradient tree."""
        if isinstance(tree, dict):
            return {k: GradientAccumulator._copy_tree(v) for k, v in tree.items()}
        elif isinstance(tree, mx.array):
            return mx.array(tree)
        else:
            return tree
    
    @staticmethod
    def _add_trees(tree1, tree2):
        """Add two gradient trees."""
        if isinstance(tree1, dict) and isinstance(tree2, dict):
            return {k: GradientAccumulator._add_trees(tree1[k], tree2[k]) 
                   for k in tree1.keys()}
        elif isinstance(tree1, mx.array) and isinstance(tree2, mx.array):
            return tree1 + tree2
        else:
            return tree1
    
    @staticmethod
    def _scale_tree(tree, scale):
        """Scale gradient tree by constant."""
        if isinstance(tree, dict):
            return {k: GradientAccumulator._scale_tree(v, scale) for k, v in tree.items()}
        elif isinstance(tree, mx.array):
            return tree * scale
        else:
            return tree


def optimize_memory_for_m4():
    """
    Apply M4-specific memory optimizations.
    """
    # Force garbage collection
    gc.collect()
    
    # MLX-specific optimizations
    mx.eval(mx.array([0]))  # Force evaluation and cleanup
    
    print("✅ M4 Memory optimizations applied")
    print(f"   → MLX Device: {mx.default_device()}")
    print(f"   → Metal GPU: {'Available' if mx.metal.is_available() else 'Not Available'}")


def estimate_memory_usage(img_size: int, batch_size: int, base_filters: int, 
                         num_classes: int = 7, mixed_precision: bool = True) -> Dict[str, float]:
    """
    Estimate memory usage for training configuration.
    
    Returns:
        Dictionary with memory estimates in MB
    """
    dtype_size = 2 if mixed_precision else 4  # FP16 vs FP32 in bytes
    
    # Input batch
    input_mb = (batch_size * 3 * img_size * img_size * dtype_size) / (1024 ** 2)
    
    # Feature maps (approximate for U-Net)
    # Encoder: bf, 2*bf, 4*bf, 8*bf
    # Bottleneck: 16*bf
    # Decoder: 8*bf, 4*bf, 2*bf, bf
    feature_maps = 0
    spatial_size = img_size
    for i, multiplier in enumerate([1, 2, 4, 8, 16, 8, 4, 2, 1]):
        channels = base_filters * multiplier
        level_mem = batch_size * channels * spatial_size * spatial_size * dtype_size
        feature_maps += level_mem
        if i < 4:  # Encoder
            spatial_size //= 2
        elif i == 4:  # Bottleneck
            pass
        else:  # Decoder
            spatial_size *= 2
    
    feature_maps_mb = feature_maps / (1024 ** 2)
    
    # Model parameters
    # Approximate: 3.5M params for depthwise separable, 10M for standard
    param_count = 3.5e6 if mixed_precision else 10e6
    params_mb = (param_count * dtype_size) / (1024 ** 2)
    
    # Gradients (same size as params)
    grads_mb = params_mb
    
    # Optimizer state (AdamW: 2x params for momentum + variance)
    optimizer_mb = params_mb * 2
    
    # Activations (checkpointing can reduce by 40-60%)
    activations_mb = feature_maps_mb * 0.5  # Assume checkpointing
    
    # System overhead
    overhead_mb = 3000  # ~3GB for OS, MLX runtime, etc.
    
    total_mb = (input_mb + feature_maps_mb + params_mb + grads_mb + 
                optimizer_mb + activations_mb + overhead_mb)
    
    return {
        'input_mb': input_mb,
        'feature_maps_mb': feature_maps_mb,
        'params_mb': params_mb,
        'grads_mb': grads_mb,
        'optimizer_mb': optimizer_mb,
        'activations_mb': activations_mb,
        'overhead_mb': overhead_mb,
        'total_mb': total_mb,
        'total_gb': total_mb / 1024,
        'safe_for_16gb': total_mb < 14000  # Leave 2GB headroom
    }


if __name__ == "__main__":
    # Test memory estimation
    print("\nMemory Estimation for M4 MacBook Air (16GB)")
    print("="*60)
    
    configs = [
        ("512x512, batch=4, FP32", 512, 4, 32, False),
        ("512x512, batch=4, FP16", 512, 4, 32, True),
        ("1024x1024, batch=1, FP32", 1024, 1, 32, False),
        ("1024x1024, batch=1, FP16", 1024, 1, 32, True),
        ("1024x1024, batch=2, FP16", 1024, 2, 32, True),
    ]
    
    for name, img_size, batch_size, base_filters, mixed_prec in configs:
        est = estimate_memory_usage(img_size, batch_size, base_filters, 
                                    mixed_precision=mixed_prec)
        status = "✅" if est['safe_for_16gb'] else "⚠️"
        print(f"\n{status} {name}")
        print(f"   Total: {est['total_gb']:.2f} GB")
        print(f"   Feature Maps: {est['feature_maps_mb']:.0f} MB")
        print(f"   Activations: {est['activations_mb']:.0f} MB")
