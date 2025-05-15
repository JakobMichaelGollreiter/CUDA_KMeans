# CUDA Accelerated K-Means Clustering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance CUDA implementation of Lloyd's K-Means algorithm optimized for GPU acceleration, outperforming scikit-learn's CPU implementation by up to 12×. Ideal for large-scale, high-dimensional datasets.

## Key Features

🚀 **GPU Acceleration**  
- Leverages NVIDIA GPUs for massive parallelism
- Optimized memory hierarchy usage (constant/shared memory)
- Two-stage reduction to minimize global memory atomics

⚡ **Algorithmic Optimizations**  
- Triangle inequality pruning (Elkan/Hamerly)
- Structure-of-Arrays (SoA) memory layout
- Dynamic mini-batching for out-of-core datasets

📊 **Benchmarking Tools**  
- Direct comparison with scikit-learn CPU implementation
- Performance analysis scripts for scaling tests
- Detailed timing breakdowns (GPU-CPU transfers, kernel execution)

## Installation

### Prerequisites
- NVIDIA GPU (Compute Capability ≥ 6.0)
- CUDA Toolkit ≥ 11.0
- CMake ≥ 3.12
- Python 3.8+ (for scikit-learn comparisons)

### Build Instructions
```bash
git clone https://github.com/JakobMichaelGollreiter/CUDA_KMeans.git
cd CUDA_KMeans
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80  # Set to your GPU arch
make -j
