# Native Kernels

High-performance kernels for Orbit Inference Engine.

This directory will contain optional C++ / CUDA implementations that can be
called from Rust via FFI (`cxx`, `bindgen`, or custom).

Planned kernels:

- Fused IterAdaLN + residual
- Sparse expert GEMM (Top-K only)
- Zone compress / decompress (GZ Cache offload)
- Custom attention paths for looped execution

Build system: CMake + optional CUDA toolkit.

Status: **skeleton only**.
