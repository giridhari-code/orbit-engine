//! Looped Mixture-of-Experts layer.
//!
//! Shared experts + optional per-loop routers + residual scaling (λ_k)
//! + input re-injection (η).

use crate::config::OrbitConfig;
use crate::iter_adln::IterAdaLN;
use crate::router::{top_k, RoutingDecision};
use crate::{OrbitError, Result};

/// A single expert (placeholder).
#[derive(Debug)]
pub struct Expert {
    pub id: usize,
}

/// The shared set of experts plus routing & modulation state.
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

        let _ada_attn = self.iter_adln_attn.forward(loop_idx, hidden)?;
        let m = hidden.to_vec();
        let _ada_moe = self.iter_adln_moe.forward(loop_idx, &m)?;

        let dummy_logits = vec![0.0f32; self.config.num_experts];
        let decision: RoutingDecision = top_k(&dummy_logits, self.config.top_k)?;

        let mut moe_out = vec![0.0f32; m.len()];
        for (eid, w) in decision.expert_ids.iter().zip(decision.weights.iter()) {
            let _ = (eid, w);
        }

        let lambda = self.lambda[loop_idx];
        let out: Vec<f32> = m
            .iter()
            .zip(moe_out.iter())
            .map(|(m_i, e_i)| m_i + lambda * e_i)
            .collect();

        Ok(out)
    }
}
