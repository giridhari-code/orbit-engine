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
    pub gamma: f32, // will become Tensor
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
            alpha: 0.0, // start near identity
        }
    }
}

/// Generates loop- and token-specific modulation parameters.
///
/// In a real implementation this would be a small MLP taking
/// `concat(PE(k), h)` and producing the three parameter tensors.
#[derive(Debug)]
pub struct IterAdaLN {
    /// Number of loops the module was configured for (for PE size).
    pub num_loops: usize,
    /// Hidden size (for PE + MLP dimensions).
    pub hidden_size: usize,
}

impl IterAdaLN {
    /// Create a new IterAdaLN module.
    pub fn new(num_loops: usize, hidden_size: usize) -> Self {
        Self {
            num_loops,
            hidden_size,
        }
    }

    /// Compute modulation parameters for a given loop index.
    ///
    /// Currently returns defaults; real math will land once candle tensors
    /// are wired.
    pub fn forward(&self, loop_idx: usize, _hidden: &[f32]) -> Result<AdaLNParams> {
        if loop_idx >= self.num_loops {
            // soft clamp for safety during scaffolding
        }
        // TODO: sinusoidal PE(k) + MLP
        Ok(AdaLNParams::default())
    }
}
