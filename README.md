# 🌌 Orbit Engine

**Tagline:** *Loop the Future. Compute only what matters. Scale beyond boundaries.*

**Status:** Pre-Alpha / Architectural Design  
**Language:** Rust (Powered by Hugging Face `candle`) + C++ kernels  
**Target Architecture:** Looped Mixture-of-Experts (Looped MoE)

---

## Abstract

The rapid scaling of Large Language Models (LLMs) has led to unsustainable inference costs. **Orbit** is a next-generation, memory-safe LLM inference engine built in Rust, designed specifically for **Looped Mixture-of-Experts (Looped MoE)** architectures.

Unlike traditional engines that treat LLMs as static dense models, Orbit embraces the orthogonal scaling axes of **Recurrence** (looping shared layers for effective depth) and **Sparsity** (activating only relevant experts for parameter capacity). By combining Rust's fearless concurrency with intelligent CPU-GPU orchestration and C++ performance kernels, Orbit delivers enterprise-grade throughput, low latency, and zero-copy memory management on any hardware—from edge devices to multi-GPU clusters.

---

## The Orbit Philosophy (10 Pillars)

1. **Recurrence over Width:** Loop the parameters, don't just stack them. Maximize effective depth with minimal VRAM.
2. **Sparsity by Default:** Compute only what matters. Per-loop routers ensure zero FLOPs on irrelevant weights.
3. **Memory is Sacred:** Fearless concurrency and zero garbage collection overhead. Paged KV-cache + **GZ Cache** ensures no fragmentation.
4. **Micro-kernel Architecture:** Separation of concerns between Core (math), Runtime (memory/scheduling), and Server (API).
5. **CPU-GPU Synergy:** *"Compute on GPU, Park on CPU."* Asynchronous offloading of inactive experts to CPU RAM.
6. **Distributed Orbit:** Expert Parallelism by default.
7. **Intelligent Scheduling:** Continuous Batching and Prefill-Decode Disaggregation.
8. **Zero-Redundancy Caching:** RadixAttention and Prefix Caching.
9. **Lossless Acceleration:** Speculative Decoding.
10. **Enterprise-Ready & Edge-Native:** Multi-tenancy, RBAC/SSO, offline edge profile.

---

## GZ Cache (Grouped-Zonal Cache)

Traditional KV-Cache explodes under Looped MoE (R loops × experts). **GZ Cache** replaces it:

- **Route over Store** — memory allocated only for active expert paths
- **Zone the Memory** — zones keyed by `(Loop_Index, Expert_ID)`
- **Reuse over Recompute** — shared base tensors modulated by IterAdaLN
- **Zip the Inactive** — compress + offload inactive zones to CPU

---

## Workspace Structure

```
orbit-engine/
├── Cargo.toml              # Workspace root
├── orbit-core/             # Core tensors, Looped MoE layers, GZ Cache
├── orbit-runtime/          # Scheduler, continuous batching, KV/GZ management
├── orbit-server/           # HTTP/gRPC serving, auth, multi-tenancy
├── cpp/                    # High-performance C++ kernels (optional CUDA/CPU)
├── docs/                   # Research survey + design docs
└── examples/
```

---

## Quick Start (when ready)

```bash
cargo build --release
cargo run -p orbit-server -- --model path/to/looped-moe
```

---

## Research Background

See `docs/` for the full Looped MoE survey (LoopMoE, LOOM, Loopie, scaling laws).

**Key ideas implemented:**
- IterAdaLN (iteration-aware adaptive layer norm)
- Per-loop routers
- Residual scaling + input re-injection
- Load balancing (aux loss + z-loss)

---

## License

Apache-2.0 (planned)

---

*Loop the Future. Compute only what matters.*
