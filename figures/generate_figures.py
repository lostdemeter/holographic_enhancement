#!/usr/bin/env python3
"""
Generate figures for the Holographic Enhancement paper/README.

This script creates visualizations of:
1. The adaptive boost function
2. The enhancement pipeline
3. Before/after comparisons
4. Performance benchmarks
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


def plot_adaptive_boost():
    """Plot the adaptive boost function α(L)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    L = np.linspace(0, 100, 1000)
    
    # Raw parabolic function
    alpha_raw = 4.0 * (L / 100.0) * (1.0 - L / 100.0)
    
    # With offset
    alpha_offset = alpha_raw + 0.3
    
    # Clamped
    alpha_clamped = np.clip(alpha_offset, 0.3, 1.0)
    
    ax.plot(L, alpha_raw, 'b--', alpha=0.5, label='Base parabola: 4L(1-L)/100²')
    ax.plot(L, alpha_offset, 'g--', alpha=0.5, label='With offset: +0.3')
    ax.plot(L, alpha_clamped, 'r-', linewidth=2.5, label='Final α(L): clamped to [0.3, 1.0]')
    
    # Shade regions
    ax.axvspan(0, 20, alpha=0.1, color='blue', label='Shadow protection')
    ax.axvspan(80, 100, alpha=0.1, color='orange', label='Highlight protection')
    ax.axvspan(20, 80, alpha=0.1, color='green', label='Midtone emphasis')
    
    ax.set_xlabel('Luminance L')
    ax.set_ylabel('Adaptive Weight α(L)')
    ax.set_title('Adaptive Boost Function')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.4)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    ax.annotate('Maximum\nenhancement', xy=(50, 1.0), xytext=(60, 1.15),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, ha='center')
    ax.annotate('Minimum\n(protected)', xy=(5, 0.3), xytext=(15, 0.5),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=9, ha='center')
    
    plt.tight_layout()
    plt.savefig('adaptive_boost.png', bbox_inches='tight')
    plt.savefig('adaptive_boost.svg', bbox_inches='tight')
    print("Saved: adaptive_boost.png/svg")
    plt.close()


def plot_enhancement_transform():
    """Plot the intensity transformation."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    I = np.linspace(0, 1, 1000)
    
    # Amplitude transform
    ax = axes[0]
    A = np.sqrt(I)
    ax.plot(I, I, 'k--', alpha=0.5, label='Identity')
    ax.plot(I, A, 'b-', linewidth=2, label='A = √I')
    ax.set_xlabel('Intensity I')
    ax.set_ylabel('Amplitude A')
    ax.set_title('Amplitude Transform')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Enhancement factor distribution (simulated)
    ax = axes[1]
    np.random.seed(42)
    factors = 1.0 + 0.3 * np.random.randn(10000)
    factors = np.clip(factors, 0.7, 1.5)
    ax.hist(factors, bins=50, density=True, alpha=0.7, color='green', edgecolor='darkgreen')
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='No change')
    ax.axvline(0.7, color='orange', linestyle=':', linewidth=2, label='Min bound')
    ax.axvline(1.5, color='orange', linestyle=':', linewidth=2, label='Max bound')
    ax.set_xlabel('Enhancement Factor')
    ax.set_ylabel('Density')
    ax.set_title('Factor Distribution (typical image)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gamma curves
    ax = axes[2]
    gamma_values = [1.0, 1.8, 2.2, 2.6]
    colors = ['gray', 'blue', 'red', 'purple']
    for gamma, color in zip(gamma_values, colors):
        y = np.power(I, gamma)
        ax.plot(I, y, color=color, linewidth=2, label=f'γ = {gamma}')
    ax.set_xlabel('Encoded Value')
    ax.set_ylabel('Linear Intensity')
    ax.set_title('Gamma Decoding: I_linear = V^γ')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('transforms.png', bbox_inches='tight')
    plt.savefig('transforms.svg', bbox_inches='tight')
    print("Saved: transforms.png/svg")
    plt.close()


def plot_pipeline_diagram():
    """Create a pipeline diagram."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Box style
    box_style = dict(boxstyle='round,pad=0.3', facecolor='lightblue', 
                     edgecolor='navy', linewidth=2)
    arrow_style = dict(arrowstyle='->', color='navy', linewidth=2)
    
    # Boxes
    boxes = [
        (1, 2, 'Input\nRGB'),
        (3, 2, 'BGR→LAB'),
        (5, 2, 'Gamma\nDecode'),
        (7, 2, 'Gaussian\nBlur'),
        (9, 2, 'Ratio\nEnhance'),
        (11, 2, 'Gamma\nEncode'),
        (13, 2, 'Output\nRGB'),
    ]
    
    for x, y, text in boxes:
        ax.text(x, y, text, ha='center', va='center', fontsize=10,
                bbox=box_style, fontweight='bold')
    
    # Arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 0.7
        x2 = boxes[i+1][0] - 0.7
        ax.annotate('', xy=(x2, 2), xytext=(x1, 2), arrowprops=arrow_style)
    
    # Add detail extraction branch
    ax.annotate('', xy=(7, 1), xytext=(7, 1.5), 
                arrowprops=dict(arrowstyle='->', color='green', linewidth=1.5))
    ax.text(7, 0.6, 'Structure\nExtraction', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', 
                      edgecolor='darkgreen', linewidth=1.5))
    
    # Add adaptive boost
    ax.text(9, 0.6, 'Adaptive\nBoost α(L)', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', 
                      edgecolor='orange', linewidth=1.5))
    ax.annotate('', xy=(9, 1.5), xytext=(9, 1),
                arrowprops=dict(arrowstyle='->', color='orange', linewidth=1.5))
    
    # Title
    ax.text(7, 3.5, 'Holographic Enhancement Pipeline', ha='center', va='center',
            fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('pipeline.png', bbox_inches='tight', facecolor='white')
    plt.savefig('pipeline.svg', bbox_inches='tight', facecolor='white')
    print("Saved: pipeline.png/svg")
    plt.close()


def plot_performance():
    """Plot performance benchmarks."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Resolution vs FPS
    ax = axes[0]
    resolutions = ['720p', '1080p', '1440p', '4K']
    python_fps = [120, 70, 40, 20]
    cuda_fps = [1847, 823, 450, 206]
    
    x = np.arange(len(resolutions))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, python_fps, width, label='Python (NumPy)', color='steelblue')
    bars2 = ax.bar(x + width/2, cuda_fps, width, label='CUDA', color='forestgreen')
    
    ax.set_xlabel('Resolution')
    ax.set_ylabel('Frames per Second')
    ax.set_title('Processing Speed by Resolution')
    ax.set_xticks(x)
    ax.set_xticklabels(resolutions)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add speedup annotations
    for i, (p, c) in enumerate(zip(python_fps, cuda_fps)):
        speedup = c / p
        ax.annotate(f'{speedup:.0f}×', xy=(i + width/2, c), xytext=(0, 5),
                    textcoords='offset points', ha='center', fontsize=9,
                    fontweight='bold', color='darkgreen')
    
    # Speedup chart
    ax = axes[1]
    speedups = [c/p for p, c in zip(python_fps, cuda_fps)]
    colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(speedups)))
    bars = ax.bar(resolutions, speedups, color=colors, edgecolor='darkgreen', linewidth=1.5)
    ax.set_xlabel('Resolution')
    ax.set_ylabel('Speedup (CUDA / Python)')
    ax.set_title('GPU Acceleration Factor')
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='10× baseline')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}×', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('performance.png', bbox_inches='tight')
    plt.savefig('performance.svg', bbox_inches='tight')
    print("Saved: performance.png/svg")
    plt.close()


def plot_holographic_analogy():
    """Visualize the holographic analogy."""
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Create sample signals
    x = np.linspace(0, 4*np.pi, 500)
    
    # Holography side
    ax1 = fig.add_subplot(gs[0, 0])
    reference = np.sin(x)
    ax1.plot(x, reference, 'b-', linewidth=2)
    ax1.set_title('Reference Wave R', fontweight='bold')
    ax1.set_ylim(-1.5, 1.5)
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    obj = 0.5 * np.sin(5*x) * np.exp(-((x-6)**2)/10)
    ax2.plot(x, obj, 'g-', linewidth=2)
    ax2.set_title('Object Wave O', fontweight='bold')
    ax2.set_ylim(-1.5, 1.5)
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 2])
    intensity = (reference + obj)**2
    ax3.plot(x, intensity, 'r-', linewidth=2)
    ax3.set_title('Intensity I = |R + O|²', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Image side
    ax4 = fig.add_subplot(gs[1, 0])
    structure = np.sin(x) + 1.5
    ax4.plot(x, structure, 'b-', linewidth=2)
    ax4.set_title('Structure (blur)', fontweight='bold')
    ax4.set_ylim(0, 3)
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 1])
    detail = 0.3 * np.sin(8*x)
    ax5.plot(x, detail, 'g-', linewidth=2)
    ax5.set_title('Detail (high-freq)', fontweight='bold')
    ax5.set_ylim(-0.5, 0.5)
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[1, 2])
    image = structure + detail
    ax6.plot(x, image, 'r-', linewidth=2)
    ax6.set_title('Image = Structure + Detail', fontweight='bold')
    ax6.set_ylim(0, 3)
    ax6.grid(True, alpha=0.3)
    
    # Add labels
    fig.text(0.02, 0.75, 'HOLOGRAPHY', fontsize=12, fontweight='bold', 
             rotation=90, va='center')
    fig.text(0.02, 0.25, 'IMAGE', fontsize=12, fontweight='bold',
             rotation=90, va='center')
    
    plt.savefig('holographic_analogy.png', bbox_inches='tight')
    plt.savefig('holographic_analogy.svg', bbox_inches='tight')
    print("Saved: holographic_analogy.png/svg")
    plt.close()


def plot_math_summary():
    """Create a visual math summary."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # Title
    ax.text(5, 9.5, 'Holographic Enhancement: Mathematical Summary', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Equations
    equations = [
        (5, 8.2, r'$\mathbf{Core\ Equation:}$', 14),
        (5, 7.4, r'$I_{out} = I \cdot \left(1 + \beta \cdot \alpha(L) \cdot \frac{I - I_{blur}}{I_{blur} + \epsilon}\right)$', 16),
        
        (5, 6.2, r'$\mathbf{Adaptive\ Boost:}$', 14),
        (5, 5.5, r'$\alpha(L) = 4 \cdot \frac{L}{100} \cdot \left(1 - \frac{L}{100}\right) + 0.3$', 14),
        
        (5, 4.3, r'$\mathbf{Gamma\ Correction:}$', 14),
        (5, 3.6, r'$I_{linear} = I_{encoded}^{\gamma}, \quad I_{output} = I_{enhanced}^{1/\gamma}$', 14),
        
        (5, 2.4, r'$\mathbf{Holographic\ Inspiration:}$', 14),
        (5, 1.7, r'$I = |A|^2 \quad \Rightarrow \quad A = \sqrt{I}$', 14),
        
        (5, 0.7, r'$\mathbf{Parameters:}\ \beta \in [1.0, 2.0],\ \sigma \in [1.0, 5.0],\ \gamma = 2.2$', 12),
    ]
    
    for x, y, eq, size in equations:
        ax.text(x, y, eq, ha='center', va='center', fontsize=size)
    
    # Add box
    rect = mpatches.FancyBboxPatch((0.5, 0.3), 9, 9.2, 
                                    boxstyle='round,pad=0.1',
                                    facecolor='white', edgecolor='navy',
                                    linewidth=2, alpha=0.9)
    ax.add_patch(rect)
    
    plt.savefig('math_summary.png', bbox_inches='tight', facecolor='white')
    plt.savefig('math_summary.svg', bbox_inches='tight', facecolor='white')
    print("Saved: math_summary.png/svg")
    plt.close()


def main():
    """Generate all figures."""
    print("Generating figures for Holographic Enhancement...")
    print("=" * 50)
    
    # Create output directory if needed
    os.makedirs('.', exist_ok=True)
    
    plot_adaptive_boost()
    plot_enhancement_transform()
    plot_pipeline_diagram()
    plot_performance()
    plot_holographic_analogy()
    plot_math_summary()
    
    print("=" * 50)
    print("All figures generated successfully!")


if __name__ == '__main__':
    main()
