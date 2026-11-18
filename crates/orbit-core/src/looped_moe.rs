//! Parallel Residual LoopMoE layer.
//!
//! Semantics:
//!   h0 = input
//!   for k in 0..R:          # parallel
//!       r_k = λ_k * ExpertPath_k(h0)
//!   out = h0 + sum(r_k)
//!
//! Sequential recurrence is no longer the default.

use crate::config::{LoopSemantics, OrbitConfig};
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
    pub fn forward_parallel_residual(&self, h0: &[f32]) -> Result<Vec<f32>> {
        if !matches!(self.config.loop_semantics, LoopSemantics::ParallelResidual) {
            return Err(OrbitError::Config(
                "forward_parallel_residual requires ParallelResidual semantics".into(),
            ));
        }

        let mut acc = h0.to_vec();

        for k in 0..self.config.num_loops {
            let _ada = self.iter_adln_moe.forward(k, h0)?;

            // Placeholder routing + expert (identity path for now)
            let dummy_logits = vec![0.0f32; self.config.num_experts];
            let decision: RoutingDecision = top_k(&dummy_logits, self.config.top_k)?;
            let _ = decision;

            let lambda = self.lambda[k];
            // Real: acc[i] += lambda * expert_out[i]
            // Scaffold: no-op residual (keeps numerical shape)
            let _ = lambda;
        }

        Ok(acc)
    }

    /// Legacy single-loop path (only valid under Sequential semantics).
    pub fn forward_loop(
        &self,
        loop_idx: usize,
        hidden: &[f32],
        _input_embedding: &[f32],
    ) -> Result<Vec<f32>> {
        if loop_idx >= self.config.num_loops {
            return Err(OrbitError::Config(format!(
                "loop_idx {} out of range 0..{}",
                loop_idx, self.config.num_loops
            )));
        }

        let _ada = self.iter_adln_moe.forward(loop_idx, hidden)?;
        let dummy_logits = vec![0.0f32; self.config.num_experts];
        let _decision = top_k(&dummy_logits, self.config.top_k)?;

        Ok(hidden.to_vec())
    }
}
