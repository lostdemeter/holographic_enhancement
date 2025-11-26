#!/usr/bin/env python3
"""
Example usage of the Holographic Enhancement algorithm.

This script demonstrates various ways to use the enhancement:
1. Basic image enhancement
2. Parameter tuning
3. Batch processing
4. Video processing
"""

import sys
sys.path.insert(0, '..')

import cv2
import numpy as np
from pathlib import Path

# Import the enhancement function
from src.enhance import holographic_enhance, enhance_video


def example_basic():
    """Basic image enhancement example."""
    print("=" * 50)
    print("Example 1: Basic Image Enhancement")
    print("=" * 50)
    
    # Load image
    image = cv2.imread('sample.jpg')
    if image is None:
        print("Creating a sample image...")
        # Create a sample gradient image with some detail
        h, w = 480, 640
        image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Gradient background
        for y in range(h):
            for x in range(w):
                image[y, x] = [
                    int(255 * x / w),
                    int(255 * y / h),
                    int(128 + 64 * np.sin(x/20) * np.sin(y/20))
                ]
        
        # Add some texture
        noise = np.random.randint(0, 30, (h, w, 3), dtype=np.uint8)
        image = cv2.add(image, noise)
        
        cv2.imwrite('sample.jpg', image)
        print("  Created sample.jpg")
    
    # Enhance with default parameters
    enhanced = holographic_enhance(image)
    cv2.imwrite('sample_enhanced.jpg', enhanced)
    print("  Saved: sample_enhanced.jpg")
    
    # Create side-by-side comparison
    comparison = np.hstack([image, enhanced])
    cv2.imwrite('comparison.jpg', comparison)
    print("  Saved: comparison.jpg (side-by-side)")


def example_parameters():
    """Demonstrate different parameter settings."""
    print("\n" + "=" * 50)
    print("Example 2: Parameter Tuning")
    print("=" * 50)
    
    image = cv2.imread('sample.jpg')
    if image is None:
        example_basic()
        image = cv2.imread('sample.jpg')
    
    # Different boost levels
    boosts = [1.0, 1.3, 1.5, 1.8]
    results = []
    
    for boost in boosts:
        enhanced = holographic_enhance(image, boost=boost)
        results.append(enhanced)
        cv2.imwrite(f'boost_{boost:.1f}.jpg', enhanced)
        print(f"  Saved: boost_{boost:.1f}.jpg")
    
    # Create comparison grid
    top = np.hstack(results[:2])
    bottom = np.hstack(results[2:])
    grid = np.vstack([top, bottom])
    cv2.imwrite('boost_comparison.jpg', grid)
    print("  Saved: boost_comparison.jpg (2x2 grid)")
    
    # Different sigma levels
    sigmas = [1.0, 2.0, 3.0, 5.0]
    results = []
    
    for sigma in sigmas:
        enhanced = holographic_enhance(image, sigma=sigma)
        results.append(enhanced)
        cv2.imwrite(f'sigma_{sigma:.1f}.jpg', enhanced)
        print(f"  Saved: sigma_{sigma:.1f}.jpg")


def example_batch():
    """Batch process multiple images."""
    print("\n" + "=" * 50)
    print("Example 3: Batch Processing")
    print("=" * 50)
    
    # Find all images in current directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [f for f in Path('.').iterdir() 
              if f.suffix.lower() in image_extensions
              and not f.stem.endswith('_enhanced')]
    
    if not images:
        print("  No images found to process")
        return
    
    print(f"  Found {len(images)} images")
    
    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        enhanced = holographic_enhance(image)
        output_path = img_path.stem + '_enhanced' + img_path.suffix
        cv2.imwrite(output_path, enhanced)
        print(f"  {img_path.name} -> {output_path}")


def example_analysis():
    """Analyze the enhancement effect."""
    print("\n" + "=" * 50)
    print("Example 4: Enhancement Analysis")
    print("=" * 50)
    
    image = cv2.imread('sample.jpg')
    if image is None:
        example_basic()
        image = cv2.imread('sample.jpg')
    
    enhanced = holographic_enhance(image)
    
    # Convert to grayscale for analysis
    gray_orig = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
    gray_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY).astype(float)
    
    # Compute statistics
    print(f"  Original - Mean: {gray_orig.mean():.1f}, Std: {gray_orig.std():.1f}")
    print(f"  Enhanced - Mean: {gray_enh.mean():.1f}, Std: {gray_enh.std():.1f}")
    
    # Compute local contrast (Laplacian variance)
    lap_orig = cv2.Laplacian(gray_orig, cv2.CV_64F).var()
    lap_enh = cv2.Laplacian(gray_enh, cv2.CV_64F).var()
    
    print(f"  Original Laplacian variance: {lap_orig:.1f}")
    print(f"  Enhanced Laplacian variance: {lap_enh:.1f}")
    print(f"  Sharpness improvement: {100*(lap_enh/lap_orig - 1):.1f}%")


def main():
    """Run all examples."""
    print("\nHolographic Enhancement - Example Usage")
    print("=" * 50)
    
    example_basic()
    example_parameters()
    example_batch()
    example_analysis()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("=" * 50)


if __name__ == '__main__':
    main()
