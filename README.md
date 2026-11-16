# 🌌 Orbit Engine

**Tagline:** *Loop the Future. Compute only what matters. Scale beyond boundaries.*

**Status:** Pre-Alpha  
**Language:** Rust + native kernels  
**Target:** Looped Mixture-of-Experts (Looped MoE) Inference Engine

---

## What is Orbit?

Orbit is a **memory-safe LLM inference engine** specialized for **Looped MoE** architectures.

It treats **Recurrence** (looping shared layers) and **Sparsity** (activating only relevant experts) as first-class scaling axes, and provides a production-oriented memory model (**GZ Cache**) + scheduling layer for CPU-GPU offloading and multi-GPU execution.

---

## Where is the LLM Inference Engine?

| Layer | Crate | Responsibility |
|-------|-------|----------------|
| **Primitives** | `crates/orbit-core` | GZ Cache, ZoneBuffer, LoopedMoE, routers, IterAdaLN |
| **Inference Engine** | `crates/orbit-runtime` | **Main engine** — Scheduler, request lifecycle, continuous batching, cache orchestration |
| **Serving** | `crates/orbit-server` | HTTP/gRPC, auth, multi-tenancy |

**Short answer:** The actual inference engine lives in **`orbit-runtime`** (built on top of `orbit-core`).

---

## Workspace Structure

```
orbit-engine/
├── Cargo.toml
├── LICENSE
├── README.md
├── crates/
│   ├── orbit-core/          # memory + model primitives
│   ├── orbit-runtime/       # ★ main LLM inference engine
│   └── orbit-server/        # serving layer
├── native/                 # C++/CUDA kernels (FFI)
├── docs/
└── examples/               # (coming)
```

---

## GZ Cache (Grouped-Zonal Cache)

Traditional KV-Cache explodes under Looped MoE. GZ Cache replaces it:

- **Route over Store** — allocate only for active expert paths
- **Zone the Memory** — zones keyed by `(loop_idx, expert_id)`
- **Reuse over Recompute** — shared base + IterAdaLN modulation
- **Zip the Inactive** — compress + offload to CPU

---

## Quick Start (when ready)

```bash
cargo build --release
cargo run -p orbit-server
```

---

## License

Apache-2.0

---

*Loop the Future. Compute only what matters.*
