//! Model and runtime configuration for Orbit.
//!
//! **Only supported semantics: Parallel Residual LoopMoE**
//!
//!   h₀ = Embed(x)
//!   for k in 1..R:          # all loops in parallel
//!       r_k = λ_k * Loop_k(h₀)
//!   final = h₀ + Σ r_k
//!
//! Sequential recurrence has been removed from the design.

use serde::{Deserialize, Serialize};

/// Top-level configuration for an Orbit model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbitConfig {
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Intermediate size of each expert FFN.
    pub intermediate_size: usize,
    /// Total number of experts (E).
    pub num_experts: usize,
    /// Top-K experts activated per token per loop.
    pub top_k: usize,
    /// Number of parallel residual loops (R).
    pub num_loops: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length supported by the cache.
    pub max_seq_len: usize,
    /// Residual scale initial value (λ_k).
    pub residual_scale_init: f32,
    /// Whether to use per-loop routers.
    pub per_loop_routers: bool,
    /// Capacity factor for expert load balancing.
    pub capacity_factor: f32,
}

impl Default for OrbitConfig {
    fn default() -> Self {
        Self {
            hidden_size: 2048,
            num_heads: 16,
            intermediate_size: 8192,
            num_experts: 8,
            top_k: 2,
            num_loops: 4,
            vocab_size: 32000,
            max_seq_len: 8192,
            residual_scale_init: 1.0,
            per_loop_routers: true,
            capacity_factor: 1.5,
        }
    }
}

impl OrbitConfig {
    /// Validate basic invariants.
    pub fn validate(&self) -> Result<(), String> {
        if self.top_k == 0 || self.top_k > self.num_experts {
            return Err(format!(
                "top_k ({}) must be in 1..={}",
                self.top_k, self.num_experts
            ));
        }
        if self.num_loops == 0 {
            return Err("num_loops must be >= 1".into());
        }
        if self.hidden_size % self.num_heads != 0 {
            return Err("hidden_size must be divisible by num_heads".into());
        }
        Ok(())
    }
}
