//! Parallel Residual LoopMoE layer.
//!
//! Semantics (only mode):
//!   h0 = input
//!   for k in 0..R:          # parallel
//!       r_k = λ_k * ExpertPath_k(h0)
//!   out = h0 + sum(r_k)

use crate::config::OrbitConfig;
use crate::iter_adln::IterAdaLN;
use crate::router::{top_k, RoutingDecision};
use crate::{OrbitError, Result};

/// A single expert (placeholder).
#[derive(Debug)]
pub struct Expert {
    pub id: usize,
}

/// Shared experts + per-loop routers + residual scales.
#[derive(Debug)]
pub struct LoopedMoELayer {
    pub config: OrbitConfig,
    pub experts: Vec<Expert>,
    pub routers: Vec<()>,
    pub iter_adln_attn: IterAdaLN,
    pub iter_adln_moe: IterAdaLN,
    pub lambda: Vec<f32>,
}

impl LoopedMoELayer {
    pub fn new(config: OrbitConfig) -> Result<Self> {
        config.validate().map_err(OrbitError::Config)?;

        let experts = (0..config.num_experts)
            .map(|id| Expert { id })
            .collect();

        let num_routers = if config.per_loop_routers {
            config.num_loops
        } else {
            1
        };
        let routers = vec![(); num_routers];
        let lambda = vec![config.residual_scale_init; config.num_loops];

        Ok(Self {
            iter_adln_attn: IterAdaLN::new(config.num_loops, config.hidden_size),
            iter_adln_moe: IterAdaLN::new(config.num_loops, config.hidden_size),
            config,
            experts,
            routers,
            lambda,
        })
    }

    /// Parallel residual forward over **all** loops from the same base `h0`.
    ///
    /// Real implementation will:
    /// 1. For each loop k in parallel:
    ///    - IterAdaLN(h0, k)
    ///    - Attention / MoE with Router_k
    ///    - scale by λ_k
    /// 2. Sum residuals into h0
    pub fn forward(&self, h0: &[f32]) -> Result<Vec<f32>> {
        let mut acc = h0.to_vec();

        for k in 0..self.config.num_loops {
            let _ada = self.iter_adln_moe.forward(k, h0)?;

            let dummy_logits = vec![0.0f32; self.config.num_experts];
            let decision: RoutingDecision = top_k(&dummy_logits, self.config.top_k)?;
            let _ = decision;

            let lambda = self.lambda[k];
            // Real: acc[i] += lambda * expert_out[i]
            let _ = lambda;
        }

        Ok(acc)
    }
}
