//! IterAdaLN — Iteration-Aware Adaptive Layer Normalization
//!
//! From LoopMoE: the same shared weights cannot distinguish loop k from k+1.
//! Solution: generate (γ, β, α) from a conditioning vector that includes
//! a sinusoidal positional encoding of the loop index PE(k) concatenated
//! with the current hidden state.
//!
//!   (γ, β, α) = MLP(concat(PE(k), h))
//!
//! The MLP weights themselves are shared across loops; only the *outputs*
//! differ because PE(k) and h change.

use crate::Result;

/// Placeholder for the IterAdaLN modulation parameters.
#[derive(Debug, Clone)]
pub struct AdaLNParams {
    /// Scale (γ).
    pub gamma: f32,
    /// Shift (β).
    pub beta: f32,
    /// Residual gate (α) — often initialized near zero (AdaLN-Zero).
    pub alpha: f32,
}

impl Default for AdaLNParams {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            beta: 0.0,
            alpha: 0.0,
        }
    }
}

/// Generates loop- and token-specific modulation parameters.
#[derive(Debug)]
pub struct IterAdaLN {
    pub num_loops: usize,
    pub hidden_size: usize,
}

impl IterAdaLN {
    pub fn new(num_loops: usize, hidden_size: usize) -> Self {
        Self {
            num_loops,
            hidden_size,
        }
    }

    pub fn forward(&self, loop_idx: usize, _hidden: &[f32]) -> Result<AdaLNParams> {
        if loop_idx >= self.num_loops {
            // soft clamp for safety during scaffolding
        }
        // TODO: sinusoidal PE(k) + MLP
        Ok(AdaLNParams::default())
    }
}
