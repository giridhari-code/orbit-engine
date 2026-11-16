# 🧠 The GZ Cache (Grouped-Zonal Cache) Philosophy

**Tagline:** *Route the Compute. Zone the Memory. Zip the Rest.*

**Document Type:** Architectural Design Document  
**Target System:** Orbit Engine (Looped Mixture-of-Experts)  
**Status:** Core Specification

---

## 1. Abstract

Traditional LLM inference engines rely on the **KV-Cache (Key-Value Cache)** to avoid recomputing past tokens during autoregressive generation. However, the KV-Cache was designed for *dense, static* Transformers. It stores memory linearly based on token position. 

In a **Looped Mixture-of-Experts (Looped MoE)** architecture, a token passes through the same layer R times (recurrence), and at each loop, it is routed to different experts (sparsity). Storing KV-cache per token, per loop, and per layer causes a catastrophic memory explosion (VRAM bottleneck).

The **GZ Cache (Grouped-Zonal Cache)** is a ground-up replacement for the traditional KV-Cache. It shifts the memory paradigm from *Token-Position* to *Routing-Path*. It groups tokens by their destination experts and manages memory in isolated zones.

---

## 2. The 4 Pillars of GZ Cache

### 🏛️ Pillar 1: Route over Store (Sparsity)
*   **The Old Way:** Allocate a contiguous block of memory for every token in the sequence.
*   **The GZ Way:** Allocate memory based on the Router's decision. If an expert is not selected by a token, it receives zero memory for that token. Memory is only reserved for the *active computation paths*.

### 🏛️ Pillar 2: Zone the Memory (Locality)
*   **The Old Way:** Memory is fragmented across layers and sequence positions.
*   **The GZ Way:** Memory is divided into distinct **Zones**. A Zone is defined by the tuple `(Loop_Index, Expert_ID)`. All tokens that route to Expert 3 during Loop 2 share the `(2, 3)` Zone. This ensures high cache locality and minimizes pointer chasing during GPU inference.

### 🏛️ Pillar 3: Reuse over Recompute (Recurrence)
*   **The Old Way:** Recompute the KV-cache or store R copies of it for every loop.
*   **The GZ Way:** Because all loops share the exact same weights (the shared MoE layer), the underlying semantic representation is similar. GZ Cache stores the base tensor and reuses it across loops, simply *modulating* it using the `IterAdaLN` parameters (γ, β) of that specific loop.

### 🏛️ Pillar 4: Zip the Inactive (Compression & Offloading)
*   **The Old Way:** The entire model and cache must fit inside the GPU VRAM.
*   **The GZ Way:** At any given time, only Top-K experts are active. The Zones of inactive experts are **zipped (compressed)** and offloaded to CPU RAM. When the Router requests an inactive expert, its Zone is asynchronously decompressed and transferred back to GPU VRAM.

---

## 3. Zonal Memory Architecture

Unlike PagedAttention (used in vLLM) which uses logical blocks mapped to physical blocks, GZ Cache uses a 2D hash-map-like structure in VRAM/DRAM.

### The Zone Data Structure (Rust):

```rust
struct Zone {
    loop_idx: usize,       // Which iteration (0 to R)
    expert_id: usize,      // Which expert (0 to E)
    keys: Tensor,          // Shape: [Seq_Len, Head_Dim]
    values: Tensor,        // Shape: [Seq_Len, Head_Dim]
    is_active: bool,       // Is this zone currently in GPU VRAM?
}
```
