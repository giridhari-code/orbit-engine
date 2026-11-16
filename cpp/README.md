# C++ Kernels for Orbit Engine

This directory will contain optional high-performance kernels (CPU + CUDA)
that can be called from Rust via `cxx` or `bindgen` / FFI.

Planned kernels:

- Fused IterAdaLN + residual
- Sparse expert GEMM (Top-K only)
- Zone compress / decompress (for GZ Cache offload)
- Custom attention for looped paths

Build system: CMake + optional CUDA toolkit.

Status: **skeleton only** — no sources yet.
