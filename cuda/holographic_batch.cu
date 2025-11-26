/**
 * Holographic Batch Enhancement - CUDA
 * =====================================
 * 
 * Process multiple frames from stdin, output to stdout.
 * Designed to work with FFmpeg pipes.
 * 
 * Author: Lesley Gushurst
 * License: GPL-3.0
 * 
 * Copyright (C) 2024 Lesley Gushurst
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Compile:
 *   nvcc -O3 -o holographic_batch holographic_batch.cu -lcudart
 * 
 * Usage with FFmpeg:
 *   ffmpeg -i input.mp4 -f rawvideo -pix_fmt rgb24 - | \
 *     ./holographic_batch WIDTH HEIGHT BOOST | \
 *     ffmpeg -f rawvideo -pix_fmt rgb24 -s WIDTHxHEIGHT -r FPS -i - output.mp4
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define BLOCK_SIZE 16
#define GAUSSIAN_RADIUS 5

__constant__ float d_gaussian_kernel[121]; // 11x11

// Color conversion functions
__device__ void rgb_to_lab(float r, float g, float b, float* L, float* a, float* lab_b) {
    r /= 255.0f; g /= 255.0f; b /= 255.0f;
    r = (r > 0.04045f) ? powf((r + 0.055f) / 1.055f, 2.4f) : r / 12.92f;
    g = (g > 0.04045f) ? powf((g + 0.055f) / 1.055f, 2.4f) : g / 12.92f;
    b = (b > 0.04045f) ? powf((b + 0.055f) / 1.055f, 2.4f) : b / 12.92f;
    
    float x = r * 0.4124564f + g * 0.3575761f + b * 0.1804375f;
    float y = r * 0.2126729f + g * 0.7151522f + b * 0.0721750f;
    float z = r * 0.0193339f + g * 0.1191920f + b * 0.9503041f;
    
    x /= 0.95047f; z /= 1.08883f;
    
    float fx = (x > 0.008856f) ? cbrtf(x) : (7.787f * x + 16.0f / 116.0f);
    float fy = (y > 0.008856f) ? cbrtf(y) : (7.787f * y + 16.0f / 116.0f);
    float fz = (z > 0.008856f) ? cbrtf(z) : (7.787f * z + 16.0f / 116.0f);
    
    *L = 116.0f * fy - 16.0f;
    *a = 500.0f * (fx - fy);
    *lab_b = 200.0f * (fy - fz);
}

__device__ void lab_to_rgb(float L, float a, float lab_b, float* r, float* g, float* b) {
    float fy = (L + 16.0f) / 116.0f;
    float fx = a / 500.0f + fy;
    float fz = fy - lab_b / 200.0f;
    
    float x = (fx > 0.206893f) ? fx * fx * fx : (fx - 16.0f / 116.0f) / 7.787f;
    float y = (fy > 0.206893f) ? fy * fy * fy : (fy - 16.0f / 116.0f) / 7.787f;
    float z = (fz > 0.206893f) ? fz * fz * fz : (fz - 16.0f / 116.0f) / 7.787f;
    
    x *= 0.95047f; z *= 1.08883f;
    
    float r_lin = x *  3.2404542f + y * -1.5371385f + z * -0.4985314f;
    float g_lin = x * -0.9692660f + y *  1.8760108f + z *  0.0415560f;
    float b_lin = x *  0.0556434f + y * -0.2040259f + z *  1.0572252f;
    
    *r = (r_lin > 0.0031308f) ? 1.055f * powf(r_lin, 1.0f/2.4f) - 0.055f : 12.92f * r_lin;
    *g = (g_lin > 0.0031308f) ? 1.055f * powf(g_lin, 1.0f/2.4f) - 0.055f : 12.92f * g_lin;
    *b = (b_lin > 0.0031308f) ? 1.055f * powf(b_lin, 1.0f/2.4f) - 0.055f : 12.92f * b_lin;
    
    *r = fminf(fmaxf(*r * 255.0f, 0.0f), 255.0f);
    *g = fminf(fmaxf(*g * 255.0f, 0.0f), 255.0f);
    *b = fminf(fmaxf(*b * 255.0f, 0.0f), 255.0f);
}

// Kernels
__global__ void rgb_to_lab_kernel(const unsigned char* rgb, float* L, float* a, float* b, int w, int h, int stride) {
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

__global__ void blur_h_kernel(const float* in, float* out, int w, int h, int r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    
    float sum = 0, wsum = 0;
    for (int dx = -r; dx <= r; dx++) {
        int nx = x + dx;
        if (nx >= 0 && nx < w) {
            float wt = d_gaussian_kernel[(dx + r) * (2*r+1) + r];
            sum += in[y * w + nx] * wt;
            wsum += wt;
        }
    }
    out[y * w + x] = sum / wsum;
}

__global__ void blur_v_kernel(const float* in, float* out, int w, int h, int r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    
    float sum = 0, wsum = 0;
    for (int dy = -r; dy <= r; dy++) {
        int ny = y + dy;
        if (ny >= 0 && ny < h) {
            float wt = d_gaussian_kernel[r * (2*r+1) + (dy + r)];
            sum += in[ny * w + x] * wt;
            wsum += wt;
        }
    }
    out[y * w + x] = sum / wsum;
}

__global__ void enhance_kernel(const float* L_orig, const float* L_blur, float* L_enh, int w, int h, float boost) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    
    int idx = y * w + x;
    float Lo = L_orig[idx];
    float Lb = L_blur[idx];
    
    float L_lin = powf(Lo / 100.0f, 2.2f) * 100.0f;
    float Lb_lin = powf(fmaxf(Lb, 0.01f) / 100.0f, 2.2f) * 100.0f;
    
    float ratio = (L_lin - Lb_lin) / (Lb_lin + 0.5f);
    float mid = 4.0f * (Lb / 100.0f) * (1.0f - Lb / 100.0f);
    mid = fminf(fmaxf(mid + 0.3f, 0.3f), 1.0f);
    
    float factor = 1.0f + boost * mid * ratio;
    factor = fminf(fmaxf(factor, 0.7f), 1.5f);
    
    float Le = powf(fminf(fmaxf(L_lin * factor / 100.0f, 0.0f), 1.0f), 1.0f/2.2f) * 100.0f;
    L_enh[idx] = fminf(fmaxf(Le, 0.0f), 100.0f);
}

__global__ void lab_to_rgb_kernel(const float* L, const float* a, const float* b, unsigned char* rgb, int w, int h, int stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    
    int pix_idx = y * w + x;
    int byte_idx = y * stride + x * 3;
    
    float R, G, B;
    lab_to_rgb(L[pix_idx], a[pix_idx], b[pix_idx], &R, &G, &B);
    
    rgb[byte_idx + 0] = (unsigned char)R;
    rgb[byte_idx + 1] = (unsigned char)G;
    rgb[byte_idx + 2] = (unsigned char)B;
}

void init_kernel(float sigma, int r) {
    int sz = 2 * r + 1;
    float* k = (float*)malloc(sz * sz * sizeof(float));
    float sum = 0;
    for (int y = -r; y <= r; y++) {
        for (int x = -r; x <= r; x++) {
            float v = expf(-(x*x + y*y) / (2*sigma*sigma));
            k[(y+r)*sz + (x+r)] = v;
            sum += v;
        }
    }
    for (int i = 0; i < sz*sz; i++) k[i] /= sum;
    CUDA_CHECK(cudaMemcpyToSymbol(d_gaussian_kernel, k, sz*sz*sizeof(float)));
    free(k);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s WIDTH HEIGHT BOOST < input.raw > output.raw\n", argv[0]);
        return 1;
    }
    
    int w = atoi(argv[1]);
    int h = atoi(argv[2]);
    float boost = atof(argv[3]);
    int npix = w * h;
    int stride = w * 3;  // Row stride in bytes
    size_t frame_size = stride * h;
    
    fprintf(stderr, "Holographic CUDA: %dx%d, stride=%d, boost=%.2f\n", w, h, stride, boost);
    
    // Set binary mode and disable buffering for pipes
    setvbuf(stdin, NULL, _IONBF, 0);
    setvbuf(stdout, NULL, _IONBF, 0);
    
    init_kernel(2.0f, GAUSSIAN_RADIUS);
    
    // Allocate
    unsigned char* h_rgb = (unsigned char*)malloc(frame_size);
    unsigned char *d_rgb;
    float *d_L, *d_a, *d_b, *d_Lt, *d_Lb, *d_Le;
    
    CUDA_CHECK(cudaMalloc(&d_rgb, frame_size));
    CUDA_CHECK(cudaMalloc(&d_L, npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a, npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Lt, npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Lb, npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Le, npix * sizeof(float)));
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((w + BLOCK_SIZE - 1) / BLOCK_SIZE, (h + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    int frame = 0;
    while (fread(h_rgb, 1, frame_size, stdin) == frame_size) {
        CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb, frame_size, cudaMemcpyHostToDevice));
        
        rgb_to_lab_kernel<<<grid, block>>>(d_rgb, d_L, d_a, d_b, w, h, stride);
        blur_h_kernel<<<grid, block>>>(d_L, d_Lt, w, h, GAUSSIAN_RADIUS);
        blur_v_kernel<<<grid, block>>>(d_Lt, d_Lb, w, h, GAUSSIAN_RADIUS);
        enhance_kernel<<<grid, block>>>(d_L, d_Lb, d_Le, w, h, boost);
        lab_to_rgb_kernel<<<grid, block>>>(d_Le, d_a, d_b, d_rgb, w, h, stride);
        
        // Wait for GPU to finish before copying back
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(h_rgb, d_rgb, frame_size, cudaMemcpyDeviceToHost));
        
        // Write exactly frame_size bytes
        size_t written = fwrite(h_rgb, 1, frame_size, stdout);
        if (written != frame_size) {
            fprintf(stderr, "Write error: wrote %zu of %zu bytes\n", written, frame_size);
        }
        fflush(stdout);
        
        frame++;
        if (frame % 100 == 0) fprintf(stderr, "  Frame %d\n", frame);
    }
    
    fprintf(stderr, "Done: %d frames\n", frame);
    
    free(h_rgb);
    cudaFree(d_rgb);
    cudaFree(d_L);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_Lt);
    cudaFree(d_Lb);
    cudaFree(d_Le);
    
    return 0;
}
