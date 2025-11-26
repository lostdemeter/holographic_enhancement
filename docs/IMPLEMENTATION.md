# Implementation Guide

This document details the implementation of the holographic enhancement algorithm.

## Algorithm Pipeline

```
┌─────────┐    ┌─────────┐    ┌──────────┐    ┌───────────┐
│  Input  │───▶│ BGR→LAB │───▶│  Gamma   │───▶│  Gaussian │
│  Image  │    │         │    │  Decode  │    │   Blur    │
└─────────┘    └─────────┘    └──────────┘    └─────┬─────┘
                                                    │
                    ┌───────────────────────────────┘
                    ▼
              ┌───────────┐    ┌──────────┐    ┌─────────┐
              │   Ratio   │───▶│ Adaptive │───▶│  Gamma  │
              │Enhancement│    │  Boost   │    │ Encode  │
              └───────────┘    └──────────┘    └────┬────┘
                                                    │
                    ┌───────────────────────────────┘
                    ▼
              ┌───────────┐    ┌──────────┐
              │ Recombine │───▶│  Output  │
              │  L, a, b  │    │  Image   │
              └───────────┘    └──────────┘
```

## Step-by-Step Implementation

### Step 1: Color Space Conversion

Convert from BGR (OpenCV default) to CIE LAB:

```python
import cv2
import numpy as np

# Input: BGR uint8 image
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

# Extract channels
L = lab[:, :, 0] * (100.0 / 255.0)  # Scale to 0-100
a = lab[:, :, 1]  # Keep as-is
b = lab[:, :, 2]  # Keep as-is
```

### Step 2: Gamma Decoding

Convert from perceptual (gamma-encoded) to linear light:

```python
gamma = 2.2
L_linear = np.power(L / 100.0, gamma) * 100.0
```

### Step 3: Structure Extraction

Apply Gaussian blur to extract low-frequency structure:

```python
from scipy.ndimage import gaussian_filter

sigma = 2.0  # Blur scale
L_blur = gaussian_filter(L_linear, sigma=sigma)
L_blur = np.maximum(L_blur, 0.01)  # Prevent division by zero
```

### Step 4: Ratio-Based Enhancement

Compute the enhancement factor:

```python
epsilon = 0.5
ratio = (L_linear - L_blur) / (L_blur + epsilon)
```

### Step 5: Adaptive Boost

Compute luminance-dependent weight:

```python
# Parabolic function: max at L=50, min at L=0 and L=100
midtone_weight = 4.0 * (L / 100.0) * (1.0 - L / 100.0)
adaptive = np.clip(midtone_weight + 0.3, 0.3, 1.0)
```

### Step 6: Apply Enhancement

Combine ratio and adaptive weight:

```python
boost = 1.5  # Enhancement strength
factor = 1.0 + boost * adaptive * ratio
factor = np.clip(factor, 0.7, 1.5)  # Stability bounds

L_enhanced = L_linear * factor
```

### Step 7: Gamma Encoding

Convert back to perceptual space:

```python
L_out = np.power(np.clip(L_enhanced / 100.0, 0, 1), 1.0/gamma) * 100.0
```

### Step 8: Recombine and Convert

Merge enhanced L with original a, b and convert back:

```python
lab[:, :, 0] = np.clip(L_out * (255.0 / 100.0), 0, 255)
# a and b channels unchanged

result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
```

## Complete Python Function

```python
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2

def holographic_enhance(image: np.ndarray,
                        sigma: float = 2.0,
                        boost: float = 1.5) -> np.ndarray:
    """
    Apply holographic enhancement to an image.
    
    Args:
        image: Input BGR image (uint8, shape HxWx3)
        sigma: Gaussian blur scale (default: 2.0)
        boost: Enhancement strength (default: 1.5)
    
    Returns:
        Enhanced BGR image (uint8, shape HxWx3)
    """
    # Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] * (100.0 / 255.0)
    
    # Gamma decode
    gamma = 2.2
    L_linear = np.power(L / 100.0, gamma) * 100.0
    
    # Extract structure
    L_blur = gaussian_filter(L_linear, sigma=sigma)
    L_blur = np.maximum(L_blur, 0.01)
    
    # Ratio-based detail
    epsilon = 0.5
    ratio = (L_linear - L_blur) / (L_blur + epsilon)
    
    # Adaptive boost
    midtone_weight = 4.0 * (L / 100.0) * (1.0 - L / 100.0)
    adaptive = np.clip(midtone_weight + 0.3, 0.3, 1.0)
    
    # Apply enhancement
    factor = 1.0 + boost * adaptive * ratio
    factor = np.clip(factor, 0.7, 1.5)
    L_enhanced = L_linear * factor
    
    # Gamma encode
    L_out = np.power(np.clip(L_enhanced / 100.0, 0, 1), 1.0/gamma) * 100.0
    
    # Recombine
    lab[:, :, 0] = np.clip(L_out * (255.0 / 100.0), 0, 255)
    
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
```

## Video Processing

For video, process each frame and preserve audio:

```python
import cv2
import subprocess

def enhance_video(input_path: str, output_path: str, 
                  sigma: float = 2.0, boost: float = 1.5):
    """Enhance a video file, preserving audio."""
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create temp output (no audio)
    temp_path = output_path + '.temp.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        enhanced = holographic_enhance(frame, sigma, boost)
        out.write(enhanced)
    
    cap.release()
    out.release()
    
    # Add audio from original
    subprocess.run([
        'ffmpeg', '-y',
        '-i', temp_path,
        '-i', input_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0?',
        output_path
    ], check=True)
    
    os.remove(temp_path)
```

## Parameter Tuning

### Sigma (Blur Scale)

| Value | Effect | Use Case |
|-------|--------|----------|
| 1.0 | Fine detail only | Sharpening text |
| 2.0 | Balanced (default) | General photos |
| 3.0-5.0 | Coarse detail | Landscapes, architecture |

### Boost (Enhancement Strength)

| Value | Effect | Use Case |
|-------|--------|----------|
| 1.0 | No change | - |
| 1.3 | Subtle | Professional photos |
| 1.5 | Moderate (default) | General use |
| 1.8-2.0 | Strong | Low-quality sources |

## Performance Optimization

### NumPy Vectorization

All operations are vectorized—no Python loops over pixels:

```python
# Good: Vectorized
L_enhanced = L_linear * factor

# Bad: Loop (1000x slower)
for y in range(height):
    for x in range(width):
        L_enhanced[y, x] = L_linear[y, x] * factor[y, x]
```

### Memory Layout

Use contiguous arrays for cache efficiency:

```python
# Ensure contiguous memory
image = np.ascontiguousarray(image)
```

### Batch Processing

For multiple images, reuse allocated buffers:

```python
# Pre-allocate
lab_buffer = np.empty((height, width, 3), dtype=np.float32)

for image in images:
    cv2.cvtColor(image, cv2.COLOR_BGR2LAB, dst=lab_buffer)
    # ... process ...
```
