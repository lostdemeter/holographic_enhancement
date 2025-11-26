#!/bin/bash
# Holographic Video Enhancement using CUDA + FFmpeg
# ==================================================
#
# Author: Lesley Gushurst
# License: GPL-3.0
# Copyright (C) 2024 Lesley Gushurst
#
# Usage: ./enhance_video.sh input.mp4 output.mp4 [boost]

set -e

INPUT="$1"
OUTPUT="$2"
BOOST="${3:-1.5}"

if [ -z "$INPUT" ] || [ -z "$OUTPUT" ]; then
    echo "Holographic Video Enhancement (CUDA + FFmpeg)"
    echo "=============================================="
    echo ""
    echo "Usage: $0 input.mp4 output.mp4 [boost]"
    echo ""
    echo "Arguments:"
    echo "  input.mp4   Input video file"
    echo "  output.mp4  Output video file"
    echo "  boost       Enhancement strength (default: 1.5)"
    echo ""
    echo "Example:"
    echo "  $0 movie.mp4 movie_enhanced.mp4 1.5"
    exit 1
fi

# Check dependencies
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg not found"
    exit 1
fi

if [ ! -f "./holographic_batch" ]; then
    echo "Error: holographic_batch not found. Run 'make' first."
    exit 1
fi

# Get video info
WIDTH=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$INPUT")
HEIGHT=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$INPUT")
FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$INPUT")

echo "=============================================="
echo "Holographic Video Enhancement (CUDA)"
echo "=============================================="
echo "  Input:  $INPUT"
echo "  Output: $OUTPUT"
echo "  Size:   ${WIDTH}x${HEIGHT} @ ${FPS} fps"
echo "  Boost:  $BOOST"
echo ""
echo "Processing..."

# Create temp file for video without audio
TEMP_VIDEO=$(mktemp --suffix=.mp4)
trap "rm -f $TEMP_VIDEO" EXIT

# Process: decode -> CUDA enhance -> encode
# Use rgb24 for consistent color order
# IMPORTANT: stderr from holographic_batch must go to stderr, not mixed with stdout
ffmpeg -hide_banner -loglevel warning -i "$INPUT" -f rawvideo -pix_fmt rgb24 - 2>/dev/null | \
    ./holographic_batch "$WIDTH" "$HEIGHT" "$BOOST" 2>/dev/stderr | \
    ffmpeg -hide_banner -loglevel warning -y \
           -f rawvideo -pix_fmt rgb24 -s "${WIDTH}x${HEIGHT}" -r "$FPS" -i - \
           -c:v libx264 -preset medium -crf 18 \
           "$TEMP_VIDEO"

# Check if input has audio
HAS_AUDIO=$(ffprobe -v error -select_streams a -show_entries stream=codec_type -of csv=p=0 "$INPUT" 2>/dev/null | head -1)

if [ "$HAS_AUDIO" = "audio" ]; then
    echo "Adding audio..."
    ffmpeg -hide_banner -loglevel warning -y \
           -i "$TEMP_VIDEO" -i "$INPUT" \
           -c:v copy -c:a aac -b:a 192k \
           -map 0:v:0 -map 1:a:0 \
           "$OUTPUT"
else
    mv "$TEMP_VIDEO" "$OUTPUT"
fi

echo ""
echo "Done! Output: $OUTPUT"
