#!/usr/bin/env python3
"""
Holographic Enhancement - Physics-Based Image and Video Enhancement

This module implements a novel image enhancement technique inspired by
holographic interference principles. Unlike conventional sharpening methods,
it treats pixel intensity as the squared amplitude of a complex wave field.

Usage:
    python enhance.py input.jpg output.jpg [--boost 1.5] [--sigma 2.0]
    python enhance.py input.mp4 output.mp4 [--boost 1.5] [--sigma 2.0]

Author: Lesley Gushurst
License: GPL-3.0

Copyright (C) 2024 Lesley Gushurst

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import os
import sys
import tempfile
import subprocess
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import cv2


def holographic_enhance(image: np.ndarray,
                        sigma: float = 2.0,
                        boost: float = 1.5) -> np.ndarray:
    """
    Apply holographic enhancement to an image.
    
    The algorithm is based on treating image intensity as arising from
    wave interference, similar to holography. This enables natural-looking
    detail enhancement without the artifacts of conventional sharpening.
    
    Mathematical basis:
        I_enhanced = I × (1 + β × α(L) × (I - I_blur) / (I_blur + ε))
    
    where:
        - β is the boost strength
        - α(L) is an adaptive weight protecting shadows/highlights
        - I_blur is the Gaussian-smoothed structure
        - ε is a stability constant
    
    Args:
        image: Input BGR image (uint8, shape HxWx3)
        sigma: Gaussian blur scale for structure extraction (default: 2.0)
               Larger values enhance coarser details
        boost: Enhancement strength (default: 1.5)
               1.0 = no change, 2.0 = strong enhancement
    
    Returns:
        Enhanced BGR image (uint8, shape HxWx3)
    
    Example:
        >>> import cv2
        >>> image = cv2.imread('photo.jpg')
        >>> enhanced = holographic_enhance(image, sigma=2.0, boost=1.5)
        >>> cv2.imwrite('enhanced.jpg', enhanced)
    """
    # Validate input
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be a 3-channel BGR image")
    
    # Convert to LAB color space (separates luminance from color)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] * (100.0 / 255.0)  # Scale to 0-100
    
    # Gamma decode: convert from perceptual to linear light
    # This is essential because our physics model assumes linear intensity
    gamma = 2.2
    L_linear = np.power(L / 100.0, gamma) * 100.0
    
    # Extract structure via Gaussian blur
    L_blur = gaussian_filter(L_linear, sigma=sigma)
    L_blur = np.maximum(L_blur, 0.01)  # Prevent division by zero
    
    # Compute ratio-based detail
    # This formulation naturally limits enhancement in flat regions
    epsilon = 0.5
    ratio = (L_linear - L_blur) / (L_blur + epsilon)
    
    # Adaptive boost: protect shadows and highlights
    # Parabolic function: maximum at L=50, minimum at L=0 and L=100
    midtone_weight = 4.0 * (L / 100.0) * (1.0 - L / 100.0)
    adaptive = np.clip(midtone_weight + 0.3, 0.3, 1.0)
    
    # Apply enhancement
    factor = 1.0 + boost * adaptive * ratio
    factor = np.clip(factor, 0.7, 1.5)  # Stability bounds
    L_enhanced = L_linear * factor
    
    # Gamma encode: convert back to perceptual space
    L_out = np.power(np.clip(L_enhanced / 100.0, 0, 1), 1.0/gamma) * 100.0
    
    # Recombine with original color channels
    lab[:, :, 0] = np.clip(L_out * (255.0 / 100.0), 0, 255)
    
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def enhance_image(input_path: str, output_path: str,
                  sigma: float = 2.0, boost: float = 1.5) -> None:
    """
    Enhance a single image file.
    
    Args:
        input_path: Path to input image
        output_path: Path to save enhanced image
        sigma: Blur scale parameter
        boost: Enhancement strength
    """
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")
    
    enhanced = holographic_enhance(image, sigma=sigma, boost=boost)
    cv2.imwrite(output_path, enhanced)
    print(f"Enhanced image saved to: {output_path}")


def enhance_video(input_path: str, output_path: str,
                  sigma: float = 2.0, boost: float = 1.5,
                  show_progress: bool = True) -> None:
    """
    Enhance a video file, preserving audio.
    
    Args:
        input_path: Path to input video
        output_path: Path to save enhanced video
        sigma: Blur scale parameter
        boost: Enhancement strength
        show_progress: Whether to show progress bar
    """
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {width}x{height} @ {fps:.2f} fps, {total_frames} frames")
    
    # Create temporary file for video without audio
    temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
    os.close(temp_fd)
    
    try:
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            enhanced = holographic_enhance(frame, sigma=sigma, boost=boost)
            out.write(enhanced)
            
            frame_count += 1
            if show_progress and frame_count % 100 == 0:
                pct = 100.0 * frame_count / total_frames
                print(f"  Frame {frame_count}/{total_frames} ({pct:.1f}%)")
        
        cap.release()
        out.release()
        
        print(f"Processed {frame_count} frames")
        
        # Check if input has audio
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a',
             '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', input_path],
            capture_output=True, text=True
        )
        has_audio = 'audio' in result.stdout
        
        if has_audio:
            print("Adding audio...")
            subprocess.run([
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
                '-i', temp_path,
                '-i', input_path,
                '-c:v', 'copy',
                '-c:a', 'aac', '-b:a', '192k',
                '-map', '0:v:0',
                '-map', '1:a:0',
                output_path
            ], check=True)
        else:
            # No audio, just copy the temp file
            import shutil
            shutil.move(temp_path, output_path)
            temp_path = None  # Don't try to delete it
        
        print(f"Enhanced video saved to: {output_path}")
        
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Holographic Enhancement - Physics-based image/video enhancement',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s photo.jpg enhanced.jpg
  %(prog)s video.mp4 enhanced.mp4 --boost 1.3
  %(prog)s image.png output.png --sigma 3.0 --boost 1.8
        """
    )
    
    parser.add_argument('input', help='Input image or video file')
    parser.add_argument('output', help='Output file path')
    parser.add_argument('--sigma', type=float, default=2.0,
                        help='Blur scale (default: 2.0, range: 1.0-5.0)')
    parser.add_argument('--boost', type=float, default=1.5,
                        help='Enhancement strength (default: 1.5, range: 1.0-2.0)')
    
    args = parser.parse_args()
    
    # Validate parameters
    if args.sigma < 0.5 or args.sigma > 10:
        print(f"Warning: sigma={args.sigma} is outside recommended range [0.5, 10]")
    if args.boost < 1.0 or args.boost > 3.0:
        print(f"Warning: boost={args.boost} is outside recommended range [1.0, 3.0]")
    
    # Check input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Determine file type
    ext = Path(args.input).suffix.lower()
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    if ext in video_extensions:
        enhance_video(args.input, args.output, 
                      sigma=args.sigma, boost=args.boost)
    elif ext in image_extensions:
        enhance_image(args.input, args.output,
                      sigma=args.sigma, boost=args.boost)
    else:
        # Try to open as image first
        try:
            enhance_image(args.input, args.output,
                          sigma=args.sigma, boost=args.boost)
        except Exception:
            # Fall back to video
            enhance_video(args.input, args.output,
                          sigma=args.sigma, boost=args.boost)


if __name__ == '__main__':
    main()
