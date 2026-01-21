"""
Core ML Inference Wrapper
High-level API for satellite image segmentation using optimized Core ML model.
"""

import os
import cv2
import numpy as np
from PIL import Image
import coremltools as ct
from typing import Tuple, Dict, List, Optional
import time


class CoreMLPredictor:
    """
    Fast inference using Core ML with Neural Engine optimization.
    Optimized for M4 MacBook Air.
    """
    
    def __init__(self, model_path: str, img_size: int = 1024):
        """
        Initialize Core ML predictor.
        
        Args:
            model_path: Path to .mlpackage file
            img_size: Expected input size
        """
        self.img_size = img_size
        self.model_path = model_path
        
        print(f"Loading Core ML model from {model_path}...")
        self.model = ct.models.MLModel(model_path)
        print("✅ Core ML model loaded successfully!")
        
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
    
    def preprocess(self, image_path: str) -> Image.Image:
        """
        Load and preprocess image for Core ML.
        
        Args:
            image_path: Path to input image
        
        Returns:
            PIL Image ready for Core ML
        """
        # Load image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Convert to PIL Image (Core ML expects PIL)
        pil_img = Image.fromarray(img)
        
        return pil_img
    
    def predict(self, image_path: str, return_probs: bool = False) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to input image
            return_probs: Whether to return class probabilities
        
        Returns:
            Tuple of (segmentation_mask, class_composition)
        """
        # Preprocess
        pil_img = self.preprocess(image_path)
        
        # Run inference
        start_time = time.time()
        output = self.model.predict({'input_image': pil_img})
        inference_time = time.time() - start_time
        
        # Extract output (Core ML returns dict)
        seg_output = output['segmentation_output']  # Shape: (1, num_classes, H, W)
        
        # Get class predictions
        seg_mask = np.argmax(seg_output[0], axis=0)  # (H, W)
        
        # Calculate class composition
        composition = self.calculate_composition(seg_mask)
        
        print(f"✅ Inference complete in {inference_time*1000:.1f}ms")
        
        if return_probs:
            probs = seg_output[0]  # (num_classes, H, W)
            return seg_mask, composition, probs
        
        return seg_mask, composition
    
    def calculate_composition(self, mask: np.ndarray) -> Dict[str, float]:
        """
        Calculate percentage of each class in the mask.
        
        Args:
            mask: Segmentation mask (H, W)
        
        Returns:
            Dictionary mapping class names to percentages
        """
        composition = {}
        total_pixels = mask.size
        
        for cls_id, cls_name in enumerate(self.class_names):
            count = np.sum(mask == cls_id)
            percentage = (count / total_pixels) * 100
            composition[cls_name] = round(percentage, 2)
        
        return composition
    
    def mask_to_rgb(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert class mask to RGB visualization.
        
        Args:
            mask: Class mask (H, W)
        
        Returns:
            RGB image (H, W, 3)
        """
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        
        for cls_id, color in self.colors.items():
            rgb[mask == cls_id] = color
        
        return rgb
    
    def visualize_prediction(self, image_path: str, output_path: Optional[str] = None):
        """
        Visualize prediction with side-by-side comparison.
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save visualization
        """
        import matplotlib.pyplot as plt
        
        # Run prediction
        mask, composition = self.predict(image_path)
        
        # Load original image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Convert mask to RGB
        mask_rgb = self.mask_to_rgb(mask)
        
        # Create overlay
        overlay = cv2.addWeighted(img, 0.6, mask_rgb, 0.4, 0)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(img)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(mask_rgb)
        axes[1].set_title('Segmentation', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Visualization saved to {output_path}")
        
        plt.show()
        
        # Print composition
        print("\nClass Composition:")
        print("-" * 40)
        for cls_name, percentage in composition.items():
            if percentage > 0.1:  # Only show classes with >0.1%
                print(f"{cls_name:15s}: {percentage:6.2f}%")
        print("-" * 40)
    
    def batch_predict(self, image_paths: List[str], save_dir: Optional[str] = None) -> List[Dict]:
        """
        Run inference on multiple images.
        
        Args:
            image_paths: List of image paths
            save_dir: Optional directory to save results
        
        Returns:
            List of dictionaries with results
        """
        results = []
        
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        print(f"\nProcessing {len(image_paths)} images...")
        start_time = time.time()
        
        for i, img_path in enumerate(image_paths):
            print(f"\n[{i+1}/{len(image_paths)}] {os.path.basename(img_path)}")
            
            try:
                # Predict
                mask, composition = self.predict(img_path)
                
                # Save if requested
                if save_dir:
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    mask_rgb = self.mask_to_rgb(mask)
                    mask_path = os.path.join(save_dir, f"{base_name}_seg.png")
                    cv2.imwrite(mask_path, cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))
                
                results.append({
                    'image_path': img_path,
                    'composition': composition,
                    'success': True
                })
            
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")
                results.append({
                    'image_path': img_path,
                    'error': str(e),
                    'success': False
                })
        
        total_time = time.time() - start_time
        avg_time = total_time / len(image_paths)
        throughput = len(image_paths) / total_time
        
        print(f"\n{'='*60}")
        print("Batch Processing Complete")
        print(f"{'='*60}")
        print(f"Total images: {len(image_paths)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time: {avg_time*1000:.1f}ms per image")
        print(f"Throughput: {throughput:.1f} images/sec")
        print(f"{'='*60}\n")
        
        return results
    
    def benchmark(self, num_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark inference speed on M4.
        
        Args:
            num_iterations: Number of iterations for benchmarking
        
        Returns:
            Dictionary with benchmark results
        """
        print(f"\nBenchmarking Core ML model ({num_iterations} iterations)...")
        
        # Create random input
        dummy_img = Image.fromarray(
            np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
        )
        
        # Warm-up
        for _ in range(10):
            _ = self.model.predict({'input_image': dummy_img})
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.time()
            _ = self.model.predict({'input_image': dummy_img})
            times.append(time.time() - start)
        
        times = np.array(times)
        
        results = {
            'mean_ms': float(np.mean(times) * 1000),
            'std_ms': float(np.std(times) * 1000),
            'min_ms': float(np.min(times) * 1000),
            'max_ms': float(np.max(times) * 1000),
            'median_ms': float(np.median(times) * 1000),
            'throughput_fps': float(1.0 / np.mean(times))
        }
        
        print(f"\n{'='*60}")
        print("Benchmark Results (M4 MacBook Air)")
        print(f"{'='*60}")
        print(f"Mean:       {results['mean_ms']:.1f} ± {results['std_ms']:.1f} ms")
        print(f"Median:     {results['median_ms']:.1f} ms")
        print(f"Min/Max:    {results['min_ms']:.1f} / {results['max_ms']:.1f} ms")
        print(f"Throughput: {results['throughput_fps']:.1f} FPS")
        print(f"{'='*60}\n")
        
        return results


if __name__ == "__main__":
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description="Core ML inference for satellite segmentation")
    parser.add_argument("--model", type=str, default="satellite_unet_1024_m4.mlpackage",
                       help="Path to Core ML model")
    parser.add_argument("--image", type=str, help="Path to single image for prediction")
    parser.add_argument("--batch", type=str, help="Directory or pattern for batch processing")
    parser.add_argument("--output-dir", type=str, default="./predictions",
                       help="Output directory for batch processing")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--img-size", type=int, default=1024, help="Input image size")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = CoreMLPredictor(args.model, args.img_size)
    
    if args.benchmark:
        # Run benchmark
        predictor.benchmark(num_iterations=100)
    
    elif args.image:
        # Single image prediction
        predictor.visualize_prediction(args.image, output_path="prediction_viz.png")
    
    elif args.batch:
        # Batch processing
        if os.path.isdir(args.batch):
            pattern = os.path.join(args.batch, "*.jpg")
        else:
            pattern = args.batch
        
        image_paths = glob.glob(pattern)
        
        if not image_paths:
            print(f"No images found matching pattern: {pattern}")
        else:
            results = predictor.batch_predict(image_paths, args.output_dir)
            print(f"\nProcessed {len(results)} images")
            print(f"Results saved to {args.output_dir}")
    
    else:
        print("Please specify --image, --batch, or --benchmark")
        print("Usage examples:")
        print("  python coreml_inference.py --image test.jpg")
        print("  python coreml_inference.py --batch './images/*.jpg' --output-dir ./results")
        print("  python coreml_inference.py --benchmark")
