//! Step executor for the Parallel Residual Token Factory.
//!
//! This is a **control-flow complete** but **numerically fake** executor.
//! It walks every ExpertBucket, accumulates per-request residual energy,
//! turns that into fake logits, and samples a token (greedy over a tiny
//! vocab). Real candle/CUDA kernels replace the body of `run_bucket` later
//! without changing the TokenFactory loop.

use std::collections::HashMap;

use crate::parallel::{ExpertBucket, ParallelStepPlan};
use crate::RequestId;

/// Result of one full parallel step for a single request.
#[derive(Debug, Clone)]
pub struct StepOutput {
    pub request_id: RequestId,
    /// Sampled next token id (dummy vocab).
    pub token_id: u32,
    /// Whether this request should stop (EOS / max length handled by factory).
    pub hit_eos: bool,
}

/// Accumulated residual signal per request (placeholder for hidden state).
#[derive(Debug, Default, Clone)]
struct ResidualAcc {
    /// Sum of gate_weight * (loop+1) * (expert+1) — deterministic fake energy.
    energy: f32,
    /// Number of work items that touched this request.
    hits: u32,
}

/// Run all buckets of a plan (conceptually in parallel).
///
/// Returns one [`StepOutput`] per request that appeared in the plan.
pub fn execute_parallel_step(
    plan: &ParallelStepPlan,
    vocab_size: u32,
    eos_token_id: u32,
) -> Vec<StepOutput> {
    let mut acc: HashMap<RequestId, ResidualAcc> = HashMap::new();

    // Parallelism point: each bucket is independent.
    for bucket in plan.buckets_sorted() {
        run_bucket(bucket, &mut acc);
    }

    let mut outputs = Vec::with_capacity(acc.len());
    for (request_id, r) in acc {
        let token_id = sample_greedy(&r, vocab_size);
        let hit_eos = token_id == eos_token_id;
        outputs.push(StepOutput {
            request_id,
            token_id,
            hit_eos,
        });
    }

    // Stable order for tests / determinism
    outputs.sort_by_key(|o| o.request_id.0);
    outputs
}

fn run_bucket(bucket: &ExpertBucket, acc: &mut HashMap<RequestId, ResidualAcc>) {
    // Real system: one fused kernel over all items in this zone.
    // Dummy: fold gate weights into per-request energy.
    let loop_scale = (bucket.loop_idx + 1) as f32;
    let expert_scale = (bucket.expert_id + 1) as f32;

    for item in &bucket.items {
        let entry = acc.entry(item.request_id).or_default();
        entry.energy += item.gate_weight * loop_scale * expert_scale;
        entry.hits += 1;
    }
}

/// Deterministic greedy sample from residual energy.
fn sample_greedy(r: &ResidualAcc, vocab_size: u32) -> u32 {
    if vocab_size == 0 {
        return 0;
    }
    // Mix energy + hits into a stable token id in [0, vocab_size).
    let raw = (r.energy * 1000.0) as u32 ^ (r.hits.wrapping_mul(2654435761));
    raw % vocab_size
}

/// Build uniform dummy router decisions for testing:
/// every loop selects experts 0..top_k-1 with equal weight.
pub fn dummy_decisions(
    num_requests: usize,
    num_loops: usize,
    top_k: usize,
    num_experts: usize,
) -> Vec<Vec<Vec<(u32, f32)>>> {
    let k = top_k.min(num_experts).max(1);
    let w = 1.0 / k as f32;
    let experts: Vec<(u32, f32)> = (0..k as u32).map(|e| (e, w)).collect();

    (0..num_requests)
        .map(|_| (0..num_loops).map(|_| experts.clone()).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel::build_parallel_plan;
    use crate::RequestId;

    #[test]
    fn executor_emits_one_token_per_request() {
        let requests = vec![(RequestId(1), 0), (RequestId(2), 0)];
        let decisions = dummy_decisions(2, 2, 2, 4);
        let plan = build_parallel_plan(&requests, 2, &decisions);
        let outs = execute_parallel_step(&plan, 100, 0);
        assert_eq!(outs.len(), 2);
        assert_eq!(outs[0].request_id, RequestId(1));
        assert_eq!(outs[1].request_id, RequestId(2));
    }
}
