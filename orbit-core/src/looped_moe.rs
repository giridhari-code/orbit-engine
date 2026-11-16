//! Looped Mixture-of-Experts layer.
//!
//! Shared experts + optional per-loop routers + residual scaling (λ_k)
//! + input re-injection (η).
//!
//! This is still a structural skeleton; real candle tensors and
//! Expert FFN implementations will be filled in next.

use crate::config::OrbitConfig;
use crate::iter_adln::IterAdaLN;
use crate::router::{top_k, RoutingDecision};
use crate::{OrbitError, Result};

/// A single expert (placeholder).
#[derive(Debug)]
pub struct Expert {
    pub id: usize,
    // weights would live here
}

/// The shared set of experts plus routing & modulation state.
#[derive(Debug)]
pub struct LoopedMoELayer {
    pub config: OrbitConfig,
    pub experts: Vec<Expert>,
    /// One router per loop when `config.per_loop_routers` is true.
    /// Otherwise a single shared router is used (index 0).
    pub routers: Vec<()>, // placeholder for router weights
    pub iter_adln_attn: IterAdaLN,
    pub iter_adln_moe: IterAdaLN,
    /// Per-loop residual scales λ_k.
    pub lambda: Vec<f32>,
}

impl LoopedMoELayer {
    /// Construct a new layer from configuration.
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

    /// Run one full loop (attention + MoE) for a single token hidden state.
    ///
    /// Real implementation will operate on batched tensors.
    pub fn forward_loop(
        &self,
        loop_idx: usize,
        hidden: &[f32],
        _input_embedding: &[f32], // for re-injection
    ) -> Result<Vec<f32>> {
        if loop_idx >= self.config.num_loops {
            return Err(OrbitError::Config(format!(
                "loop_idx {} out of range 0..{}",
                loop_idx, self.config.num_loops
            )));
        }

        // 1. IterAdaLN for attention (placeholder)
        let _ada_attn = self.iter_adln_attn.forward(loop_idx, hidden)?;

        // 2. Attention would happen here → intermediate m
        let m = hidden.to_vec(); // identity for now

        // 3. IterAdaLN for MoE
        let _ada_moe = self.iter_adln_moe.forward(loop_idx, &m)?;

        // 4. Route (dummy logits for scaffolding)
        let dummy_logits = vec![0.0f32; self.config.num_experts];
        let decision: RoutingDecision = top_k(&dummy_logits, self.config.top_k)?;

        // 5. Sparse expert aggregation (identity placeholder)
        let mut moe_out = vec![0.0f32; m.len()];
        for (eid, w) in decision.expert_ids.iter().zip(decision.weights.iter()) {
            // real: moe_out += w * Expert[*eid](modulated_m)
            let _ = (eid, w);
        }

        // 6. Residual with per-loop scale
        let lambda = self.lambda[loop_idx];
        let mut out: Vec<f32> = m
            .iter()
            .zip(moe_out.iter())
            .map(|(m_i, e_i)| m_i + lambda * e_i)
            .collect();

        // 7. Input re-injection is applied by the caller *between* loops
        //    (skipped after the final loop).

        Ok(out)
    }
}
