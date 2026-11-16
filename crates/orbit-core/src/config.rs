//! Model and runtime configuration for Orbit.

use serde::{Deserialize, Serialize};

/// Top-level configuration for a Looped MoE model.
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
    /// Top-K experts activated per token.
    pub top_k: usize,
    /// Number of loops / recurrence steps (R).
    pub num_loops: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length supported by the cache.
    pub max_seq_len: usize,
    /// Residual scale initial value (λ).
    pub residual_scale_init: f32,
    /// Input re-injection strength (η).
    pub reinjection_eta: f32,
    /// Whether to use per-loop routers (LOOM-style).
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
            reinjection_eta: 0.1,
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
