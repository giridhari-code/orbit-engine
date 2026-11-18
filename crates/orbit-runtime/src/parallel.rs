//! Parallel Residual execution primitives for the Token Factory.
//!
//! Model semantics (Option A):
//!
//!   h0 = Embed(x)
//!   for k in 1..R:          # all k parallel
//!       r_k = λ_k * Loop_k(h0)
//!   final = h0 + sum(r_k)
//!
//! The runtime never walks loops sequentially for a single token.
//! It builds a flat set of work items, groups them by (loop, expert),
//! and launches those groups in parallel.

use std::collections::HashMap;

use orbit_core::ZoneKey;

use crate::RequestId;

/// One unit of work: a specific token position going through one loop
/// and one expert path.
#[derive(Debug, Clone)]
pub struct WorkItem {
    pub request_id: RequestId,
    /// Position in the sequence (for prefill many positions; decode usually last).
    pub token_pos: u32,
    /// Which parallel residual loop (0..R-1).
    pub loop_idx: u32,
    /// Which expert inside that loop.
    pub expert_id: u32,
    /// Gate weight from the router for this expert.
    pub gate_weight: f32,
}

/// All work that shares the same (loop, expert) and can therefore
/// run as one dense kernel launch against one zone.
#[derive(Debug, Clone)]
pub struct ExpertBucket {
    pub loop_idx: u32,
    pub expert_id: u32,
    pub items: Vec<WorkItem>,
}

impl ExpertBucket {
    pub fn zone_key(&self) -> ZoneKey {
        ZoneKey::new(self.loop_idx, self.expert_id)
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// A full parallel step: every residual loop × selected experts for the
/// current active batch.
#[derive(Debug, Default)]
pub struct ParallelStepPlan {
    /// Buckets keyed by (loop_idx, expert_id).
    pub buckets: HashMap<(u32, u32), ExpertBucket>,
    /// All zone keys that must be pinned before launch.
    pub required_zones: Vec<ZoneKey>,
}

impl ParallelStepPlan {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a work item; automatically buckets by (loop, expert).
    pub fn push(&mut self, item: WorkItem) {
        let key = (item.loop_idx, item.expert_id);
        let bucket = self.buckets.entry(key).or_insert_with(|| ExpertBucket {
            loop_idx: item.loop_idx,
            expert_id: item.expert_id,
            items: Vec::new(),
        });
        bucket.items.push(item);
    }

    /// Finalize: compute the set of zones that must be resident.
    pub fn finalize(&mut self) {
        self.required_zones = self
            .buckets
            .values()
            .map(|b| b.zone_key())
            .collect();
        // Stable order helps debugging / deterministic pinning.
        self.required_zones.sort_by_key(|z| (z.loop_idx, z.expert_id));
        self.required_zones.dedup();
    }

    pub fn num_buckets(&self) -> usize {
        self.buckets.len()
    }

    pub fn num_work_items(&self) -> usize {
        self.buckets.values().map(|b| b.len()).sum()
    }

    pub fn buckets_sorted(&self) -> Vec<&ExpertBucket> {
        let mut v: Vec<_> = self.buckets.values().collect();
        v.sort_by_key(|b| (b.loop_idx, b.expert_id));
        v
    }
}

/// Build a parallel step plan for Parallel Residual semantics.
///
/// For every active request and every loop k in 0..R-1, the caller supplies
/// the Top-K expert decisions (expert_id + gate weight). This function
/// flattens them into independent work items.
pub fn build_parallel_plan(
    requests: &[(RequestId, u32)], // (request_id, token_pos)
    num_loops: u32,
    // router decisions: request_idx -> loop_idx -> Vec<(expert_id, gate)>
    decisions: &[Vec<Vec<(u32, f32)>>],
) -> ParallelStepPlan {
    let mut plan = ParallelStepPlan::new();

    for (req_idx, &(request_id, token_pos)) in requests.iter().enumerate() {
        if req_idx >= decisions.len() {
            continue;
        }
        let per_loop = &decisions[req_idx];
        for loop_idx in 0..num_loops {
            let experts = per_loop.get(loop_idx as usize);
            if let Some(experts) = experts {
                for &(expert_id, gate_weight) in experts {
                    plan.push(WorkItem {
                        request_id,
                        token_pos,
                        loop_idx,
                        expert_id,
                        gate_weight,
                    });
                }
            }
        }
    }

    plan.finalize();
    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RequestId;

    #[test]
    fn parallel_plan_buckets_correctly() {
        // 2 requests, 2 loops, top-2 experts each
        let requests = vec![(RequestId(1), 0), (RequestId(2), 0)];
        let decisions = vec![
            // request 1
            vec![vec![(0, 0.6), (1, 0.4)], vec![(1, 0.7), (2, 0.3)]],
            // request 2
            vec![vec![(0, 0.5), (2, 0.5)], vec![(0, 0.9), (1, 0.1)]],
        ];

        let plan = build_parallel_plan(&requests, 2, &decisions);

        // Expected unique (loop, expert) pairs:
        // (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
        assert_eq!(plan.num_buckets(), 6);
        assert_eq!(plan.num_work_items(), 8); // 2 req * 2 loops * 2 experts
        assert_eq!(plan.required_zones.len(), 6);
    }
}
