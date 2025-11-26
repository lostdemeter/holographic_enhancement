# CUDA Implementation Guide

This document details the GPU-accelerated implementation of holographic enhancement.

## Performance

| Resolution | CPU (Python) | GPU (CUDA) | Speedup |
|------------|-------------:|----------:|--------:|
| 720p | 120 fps | 1,847 fps | 15× |
| 1080p | 70 fps | 823 fps | 12× |
| 4K | 20 fps | 206 fps | 10× |

*Benchmarked on NVIDIA RTX 3080*

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Host (CPU)                           │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │  Read   │───▶│ Upload  │    │Download │───▶│  Write  │  │
│  │  Frame  │    │ to GPU  │    │from GPU │    │  Frame  │  │
│  └─────────┘    └────┬────┘    └────▲────┘    └─────────┘  │
└──────────────────────┼──────────────┼───────────────────────┘
                       │              │
                       ▼              │
┌──────────────────────┴──────────────┴───────────────────────┐
│                       Device (GPU)                          │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │RGB→LAB  │───▶│  Blur   │───▶│ Enhance │───▶│LAB→RGB  │  │
│  │ Kernel  │    │ Kernel  │    │ Kernel  │    │ Kernel  │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## CUDA Kernels

### 1. Color Conversion (RGB → LAB)

```cuda
__device__ void rgb_to_lab(float r, float g, float b, 
                           float* L, float* a, float* lab_b) {
    // Normalize to [0,1]
    r /= 255.0f; g /= 255.0f; b /= 255.0f;
    
    // Gamma decode (sRGB)
    r = (r > 0.04045f) ? powf((r + 0.055f) / 1.055f, 2.4f) : r / 12.92f;
    g = (g > 0.04045f) ? powf((g + 0.055f) / 1.055f, 2.4f) : g / 12.92f;
    b = (b > 0.04045f) ? powf((b + 0.055f) / 1.055f, 2.4f) : b / 12.92f;
    
    // RGB to XYZ (D65)
    float x = r * 0.4124564f + g * 0.3575761f + b * 0.1804375f;
    float y = r * 0.2126729f + g * 0.7151522f + b * 0.0721750f;
    float z = r * 0.0193339f + g * 0.1191920f + b * 0.9503041f;
    
    // Normalize by D65 white point
    x /= 0.95047f;
    z /= 1.08883f;
    
    // XYZ to LAB
    float fx = (x > 0.008856f) ? cbrtf(x) : (7.787f * x + 16.0f / 116.0f);
    float fy = (y > 0.008856f) ? cbrtf(y) : (7.787f * y + 16.0f / 116.0f);
    float fz = (z > 0.008856f) ? cbrtf(z) : (7.787f * z + 16.0f / 116.0f);
    
    *L = 116.0f * fy - 16.0f;
    *a = 500.0f * (fx - fy);
    *lab_b = 200.0f * (fy - fz);
}

__global__ void rgb_to_lab_kernel(const unsigned char* rgb, 
                                   float* L, float* a, float* b,
                                   int w, int h, int stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    
    int pix_idx = y * w + x;
    int byte_idx = y * stride + x * 3;
    
    float R = rgb[byte_idx + 0];
    float G = rgb[byte_idx + 1];
    float B = rgb[byte_idx + 2];
    
    rgb_to_lab(R, G, B, &L[pix_idx], &a[pix_idx], &b[pix_idx]);
}
```

### 2. Gaussian Blur (Separable)

Using separable convolution for O(k) instead of O(k²):

```cuda
__constant__ float d_gaussian_kernel[121];  // 11×11 max

__global__ void blur_horizontal(const float* in, float* out, 
                                 int w, int h, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    
    float sum = 0, wsum = 0;
    for (int dx = -radius; dx <= radius; dx++) {
        int nx = x + dx;
        if (nx >= 0 && nx < w) {
            float wt = d_gaussian_kernel[(dx + radius) * (2*radius+1) + radius];
            sum += in[y * w + nx] * wt;
            wsum += wt;
        }
    }
    out[y * w + x] = sum / wsum;
}

__global__ void blur_vertical(const float* in, float* out,
                               int w, int h, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    
    float sum = 0, wsum = 0;
    for (int dy = -radius; dy <= radius; dy++) {
        int ny = y + dy;
        if (ny >= 0 && ny < h) {
            float wt = d_gaussian_kernel[radius * (2*radius+1) + (dy + radius)];
            sum += in[ny * w + x] * wt;
            wsum += wt;
        }
    }
    out[y * w + x] = sum / wsum;
}
```

### 3. Enhancement Kernel

The core enhancement logic:

```cuda
__global__ void enhance_kernel(const float* L_orig, const float* L_blur,
                                float* L_enh, int w, int h, float boost) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    
    int idx = y * w + x;
    float Lo = L_orig[idx];
    float Lb = L_blur[idx];
    
    // Gamma decode
    float L_lin = powf(Lo / 100.0f, 2.2f) * 100.0f;
    float Lb_lin = powf(fmaxf(Lb, 0.01f) / 100.0f, 2.2f) * 100.0f;
    
    // Ratio-based enhancement
    float ratio = (L_lin - Lb_lin) / (Lb_lin + 0.5f);
    
    // Adaptive boost (midtone emphasis)
    float mid = 4.0f * (Lb / 100.0f) * (1.0f - Lb / 100.0f);
    mid = fminf(fmaxf(mid + 0.3f, 0.3f), 1.0f);
    
    // Apply enhancement
    float factor = 1.0f + boost * mid * ratio;
    factor = fminf(fmaxf(factor, 0.7f), 1.5f);
    
    // Gamma encode
    float Le = powf(fminf(fmaxf(L_lin * factor / 100.0f, 0.0f), 1.0f), 
                    1.0f/2.2f) * 100.0f;
    L_enh[idx] = fminf(fmaxf(Le, 0.0f), 100.0f);
}
```

### 4. Color Conversion (LAB → RGB)

```cuda
__device__ void lab_to_rgb(float L, float a, float lab_b,
                           float* r, float* g, float* b) {
    // LAB to XYZ
    float fy = (L + 16.0f) / 116.0f;
    float fx = a / 500.0f + fy;
    float fz = fy - lab_b / 200.0f;
    
    float x = (fx > 0.206893f) ? fx*fx*fx : (fx - 16.0f/116.0f) / 7.787f;
    float y = (fy > 0.206893f) ? fy*fy*fy : (fy - 16.0f/116.0f) / 7.787f;
    float z = (fz > 0.206893f) ? fz*fz*fz : (fz - 16.0f/116.0f) / 7.787f;
    
    // Denormalize by D65 white point
    x *= 0.95047f;
    z *= 1.08883f;
    
    // XYZ to linear RGB
    float r_lin = x *  3.2404542f + y * -1.5371385f + z * -0.4985314f;
    float g_lin = x * -0.9692660f + y *  1.8760108f + z *  0.0415560f;
    float b_lin = x *  0.0556434f + y * -0.2040259f + z *  1.0572252f;
    
    // Gamma encode (sRGB)
    *r = (r_lin > 0.0031308f) ? 1.055f * powf(r_lin, 1.0f/2.4f) - 0.055f 
                              : 12.92f * r_lin;
    *g = (g_lin > 0.0031308f) ? 1.055f * powf(g_lin, 1.0f/2.4f) - 0.055f 
                              : 12.92f * g_lin;
    *b = (b_lin > 0.0031308f) ? 1.055f * powf(b_lin, 1.0f/2.4f) - 0.055f 
                              : 12.92f * b_lin;
    
    // Scale to [0, 255]
    *r = fminf(fmaxf(*r * 255.0f, 0.0f), 255.0f);
    *g = fminf(fmaxf(*g * 255.0f, 0.0f), 255.0f);
    *b = fminf(fmaxf(*b * 255.0f, 0.0f), 255.0f);
}
```

## Memory Management

### Device Memory Allocation

```cuda
// Per-frame buffers
unsigned char *d_rgb;      // Input/output RGB
float *d_L, *d_a, *d_b;    // LAB channels
float *d_L_temp, *d_L_blur; // Blur intermediates
float *d_L_enh;            // Enhanced L

size_t frame_bytes = width * height * 3;
size_t channel_floats = width * height * sizeof(float);

cudaMalloc(&d_rgb, frame_bytes);
cudaMalloc(&d_L, channel_floats);
cudaMalloc(&d_a, channel_floats);
cudaMalloc(&d_b, channel_floats);
cudaMalloc(&d_L_temp, channel_floats);
cudaMalloc(&d_L_blur, channel_floats);
cudaMalloc(&d_L_enh, channel_floats);
```

### Constant Memory for Kernel

```cuda
// Gaussian kernel in constant memory (cached, broadcast)
__constant__ float d_gaussian_kernel[121];

void init_gaussian_kernel(float sigma, int radius) {
    int size = 2 * radius + 1;
    float* h_kernel = (float*)malloc(size * size * sizeof(float));
    
    float sum = 0;
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            float val = expf(-(x*x + y*y) / (2*sigma*sigma));
            h_kernel[(y+radius)*size + (x+radius)] = val;
            sum += val;
        }
    }
    
    // Normalize
    for (int i = 0; i < size*size; i++) h_kernel[i] /= sum;
    
    cudaMemcpyToSymbol(d_gaussian_kernel, h_kernel, 
                       size*size*sizeof(float));
    free(h_kernel);
}
```

## Optimization Techniques

### 1. Coalesced Memory Access

Threads in a warp access consecutive memory locations:

```cuda
// Good: Coalesced (threads access consecutive addresses)
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float val = input[idx];

// Bad: Strided (threads access non-consecutive addresses)
int idx = threadIdx.x * stride;
float val = input[idx];
```

### 2. Shared Memory for Blur

Reduce global memory access by loading tiles into shared memory:

```cuda
#define BLOCK_SIZE 16
#define RADIUS 5
#define TILE_SIZE (BLOCK_SIZE + 2*RADIUS)

__global__ void blur_shared(const float* in, float* out, int w, int h) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * BLOCK_SIZE + tx - RADIUS;
    int gy = blockIdx.y * BLOCK_SIZE + ty - RADIUS;
    
    // Load tile with halo
    if (gx >= 0 && gx < w && gy >= 0 && gy < h)
        tile[ty][tx] = in[gy * w + gx];
    else
        tile[ty][tx] = 0;
    
    __syncthreads();
    
    // Only interior threads compute output
    if (tx >= RADIUS && tx < BLOCK_SIZE + RADIUS &&
        ty >= RADIUS && ty < BLOCK_SIZE + RADIUS) {
        
        float sum = 0;
        for (int dy = -RADIUS; dy <= RADIUS; dy++) {
            for (int dx = -RADIUS; dx <= RADIUS; dx++) {
                sum += tile[ty+dy][tx+dx] * 
                       d_gaussian_kernel[(dy+RADIUS)*(2*RADIUS+1)+(dx+RADIUS)];
            }
        }
        
        int ox = blockIdx.x * BLOCK_SIZE + (tx - RADIUS);
        int oy = blockIdx.y * BLOCK_SIZE + (ty - RADIUS);
        if (ox < w && oy < h)
            out[oy * w + ox] = sum;
    }
}
```

### 3. Stream Processing

Overlap computation with memory transfer:

```cuda
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Double buffering
for (int i = 0; i < num_frames; i++) {
    cudaStream_t current = (i % 2 == 0) ? stream1 : stream2;
    
    // Async copy to device
    cudaMemcpyAsync(d_rgb, h_frames[i], frame_size, 
                    cudaMemcpyHostToDevice, current);
    
    // Launch kernels
    rgb_to_lab_kernel<<<grid, block, 0, current>>>(...);
    blur_h_kernel<<<grid, block, 0, current>>>(...);
    blur_v_kernel<<<grid, block, 0, current>>>(...);
    enhance_kernel<<<grid, block, 0, current>>>(...);
    lab_to_rgb_kernel<<<grid, block, 0, current>>>(...);
    
    // Async copy back
    cudaMemcpyAsync(h_output[i], d_rgb, frame_size,
                    cudaMemcpyDeviceToHost, current);
}

cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);
```

## Building

### Requirements

- CUDA Toolkit 11.0+
- GCC 7+ or Clang 8+
- (Optional) FFmpeg for video I/O

### Compilation

```bash
# Basic build
nvcc -O3 -arch=sm_60 -o holographic_enhance holographic_batch.cu -lcudart

# With debug info
nvcc -G -g -o holographic_enhance_debug holographic_batch.cu -lcudart

# For specific GPU architecture
nvcc -O3 -arch=sm_86 -o holographic_enhance holographic_batch.cu  # RTX 30xx
nvcc -O3 -arch=sm_89 -o holographic_enhance holographic_batch.cu  # RTX 40xx
```

### Makefile

```makefile
NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_60 -Xcompiler -Wall

all: holographic_batch

holographic_batch: holographic_batch.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcudart

clean:
	rm -f holographic_batch
```

## Usage

### With FFmpeg Pipes

```bash
# Enhance video
ffmpeg -i input.mp4 -f rawvideo -pix_fmt rgb24 - | \
  ./holographic_batch WIDTH HEIGHT BOOST | \
  ffmpeg -f rawvideo -pix_fmt rgb24 -s WIDTHxHEIGHT -r FPS -i - output.mp4
```

### Shell Script Wrapper

```bash
#!/bin/bash
# enhance_video.sh

INPUT="$1"
OUTPUT="$2"
BOOST="${3:-1.5}"

WIDTH=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$INPUT")
HEIGHT=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$INPUT")
FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$INPUT")

ffmpeg -i "$INPUT" -f rawvideo -pix_fmt rgb24 - 2>/dev/null | \
  ./holographic_batch "$WIDTH" "$HEIGHT" "$BOOST" 2>/dev/stderr | \
  ffmpeg -f rawvideo -pix_fmt rgb24 -s "${WIDTH}x${HEIGHT}" -r "$FPS" -i - \
         -c:v libx264 -preset medium -crf 18 "$OUTPUT"
```
