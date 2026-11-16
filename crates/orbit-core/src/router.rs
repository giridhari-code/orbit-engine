//! Routing utilities for sparse expert selection.
//!
//! Supports both shared routers and per-loop routers (LOOM style).

use crate::{OrbitError, Result};

/// Result of a Top-K routing decision for a single token.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub expert_ids: Vec<usize>,
    pub weights: Vec<f32>,
}

/// Simple Top-K selection over a vector of logits.
pub fn top_k(logits: &[f32], k: usize) -> Result<RoutingDecision> {
    if logits.is_empty() {
        return Err(OrbitError::Routing("empty logits".into()));
    }
    let k = k.min(logits.len());

    let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let selected: Vec<(usize, f32)> = pairs.into_iter().take(k).collect();

    let max_logit = selected
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = selected.iter().map(|(_, v)| (v - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let weights: Vec<f32> = exps.into_iter().map(|e| e / sum).collect();
    let expert_ids: Vec<usize> = selected.into_iter().map(|(i, _)| i).collect();

    Ok(RoutingDecision {
        expert_ids,
        weights,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_basic() {
        let logits = vec![0.1, 2.0, 0.5, 1.5];
        let decision = top_k(&logits, 2).unwrap();
        assert_eq!(decision.expert_ids, vec![1, 3]);
        assert!((decision.weights.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }
}
