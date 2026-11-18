//! Orbit Runtime — **the main LLM Inference Engine**
//!
//! Parallel Residual LoopMoE Token Factory.
//!
//! Model semantics (only mode):
//!   h0 = Embed(x)
//!   for k in 1..R (parallel):
//!       r_k = λ_k * Loop_k(h0)
//!   final = h0 + sum(r_k)
//!
//! Sequential recurrence is not supported.

#![deny(missing_docs)]

mod parallel;

pub use parallel::{
    build_parallel_plan, ExpertBucket, ParallelStepPlan, WorkItem,
};

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use orbit_core::{
    GzCache, GzCacheConfig, MemoryStats, OrbitConfig, ZoneKey,
};

/// Unique identifier for an in-flight generation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestId(pub u64);

/// Lightweight per-request state.
///
/// All parallel residual loops run together each step — no current_loop.
#[derive(Debug)]
pub struct Sequence {
    pub id: RequestId,
    pub generated_len: usize,
    pub max_new_tokens: usize,
    pub touched_zones: Vec<ZoneKey>,
    pub finished: bool,
}

impl Sequence {
    pub fn new(id: RequestId, max_new_tokens: usize) -> Self {
        Self {
            id,
            generated_len: 0,
            max_new_tokens,
            touched_zones: Vec::new(),
            finished: false,
        }
    }

    pub fn is_finished(&self) -> bool {
        self.finished || self.generated_len >= self.max_new_tokens
    }
}

/// The Parallel Residual Token Factory.
#[derive(Debug)]
pub struct TokenFactory {
    pub model_config: OrbitConfig,
    cache: Arc<Mutex<GzCache>>,
    sequences: HashMap<RequestId, Sequence>,
    next_request_id: u64,
}

impl TokenFactory {
    pub fn new(model_config: OrbitConfig, cache_config: GzCacheConfig) -> Self {
        Self {
            model_config,
            cache: Arc::new(Mutex::new(GzCache::new(cache_config))),
            sequences: HashMap::new(),
            next_request_id: 1,
        }
    }

    pub fn with_defaults(model_config: OrbitConfig) -> Self {
        Self::new(model_config, GzCacheConfig::default())
    }

    /// Admit a new generation request.
    pub fn add_request(&mut self, max_new_tokens: usize) -> RequestId {
        let id = RequestId(self.next_request_id);
        self.next_request_id += 1;
        self.sequences
            .insert(id, Sequence::new(id, max_new_tokens));
        id
    }

    pub fn active_count(&self) -> usize {
        self.sequences.values().filter(|s| !s.is_finished()).count()
    }

    /// Build + pin a parallel step for all active requests.
    ///
    /// `decisions[request_index][loop_idx] = Vec<(expert_id, gate_weight)>`
    pub fn prepare_parallel_step(
        &mut self,
        decisions: &[Vec<Vec<(u32, f32)>>],
    ) -> Result<ParallelStepPlan, String> {
        let active: Vec<(RequestId, u32)> = self
            .sequences
            .values()
            .filter(|s| !s.is_finished())
            .map(|s| (s.id, s.generated_len as u32))
            .collect();

        if active.is_empty() {
            return Ok(ParallelStepPlan::new());
        }

        let num_loops = self.model_config.num_loops as u32;
        let plan = build_parallel_plan(&active, num_loops, decisions);

        let mut cache = self.cache.lock().map_err(|e| e.to_string())?;
        for zone in &plan.required_zones {
            cache
                .pin_for_compute(*zone)
                .map_err(|e| e.to_string())?;
        }

        for item in plan.buckets.values().flat_map(|b| b.items.iter()) {
            if let Some(seq) = self.sequences.get_mut(&item.request_id) {
                let key = ZoneKey::new(item.loop_idx, item.expert_id);
                if !seq.touched_zones.contains(&key) {
                    seq.touched_zones.push(key);
                }
            }
        }

        Ok(plan)
    }

    /// After compute + sampling: advance token counts.
    pub fn advance_tokens(&mut self, finished_ids: &[RequestId]) -> Result<(), String> {
        for id in self.sequences.keys().copied().collect::<Vec<_>>() {
            if let Some(seq) = self.sequences.get_mut(&id) {
                if !seq.is_finished() {
                    seq.generated_len += 1;
                    if seq.generated_len >= seq.max_new_tokens {
                        seq.finished = true;
                    }
                }
            }
        }
        for &id in finished_ids {
            if let Some(seq) = self.sequences.get_mut(&id) {
                seq.finished = true;
            }
        }
        Ok(())
    }

    /// Release zones and drop a finished request.
    pub fn finish_request(&mut self, request_id: RequestId) -> Result<(), String> {
        let seq = self
            .sequences
            .remove(&request_id)
            .ok_or_else(|| format!("unknown request {:?}", request_id))?;

        let mut cache = self.cache.lock().map_err(|e| e.to_string())?;
        for key in seq.touched_zones {
            let _ = cache.release(key);
        }
        Ok(())
    }

    pub fn memory_stats(&self) -> Result<MemoryStats, String> {
        let cache = self.cache.lock().map_err(|e| e.to_string())?;
        Ok(cache.memory_stats())
    }

    pub fn cache(&self) -> Arc<Mutex<GzCache>> {
        Arc::clone(&self.cache)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use orbit_core::OrbitConfig;

    #[test]
    fn parallel_step_pins_all_loop_zones() {
        let mut cfg = OrbitConfig::default();
        cfg.num_loops = 2;
        let mut factory = TokenFactory::with_defaults(cfg);
        let _id1 = factory.add_request(8);
        let _id2 = factory.add_request(8);

        let decisions = vec![
            vec![vec![(0, 0.6), (1, 0.4)], vec![(1, 0.7), (2, 0.3)]],
            vec![vec![(0, 0.5), (2, 0.5)], vec![(0, 0.9), (1, 0.1)]],
        ];

        let plan = factory.prepare_parallel_step(&decisions).unwrap();
        assert_eq!(plan.num_buckets(), 6);
        assert!(factory.memory_stats().unwrap().num_gpu_resident >= 6);
    }
}
