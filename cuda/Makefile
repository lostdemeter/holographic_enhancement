# Holographic Enhancement - CUDA Build
# =====================================

NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_60 -Xcompiler -Wall

# Detect GPU architecture (optional)
# Run: nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# Then set: NVCC_FLAGS = -O3 -arch=sm_XX

.PHONY: all clean

all: holographic_batch

holographic_batch: holographic_batch.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcudart

# Debug build with symbols
debug: holographic_batch.cu
	$(NVCC) -G -g -o holographic_batch_debug $< -lcudart

clean:
	rm -f holographic_batch holographic_batch_debug

# Installation (optional)
install: holographic_batch
	cp holographic_batch /usr/local/bin/
	cp enhance_video.sh /usr/local/bin/

# Test with sample
test: holographic_batch
	@echo "Testing with a sample frame..."
	@echo "Create a test with: ffmpeg -i input.mp4 -vframes 1 -f rawvideo -pix_fmt rgb24 test.raw"
