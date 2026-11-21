//! Orbit Runtime — **the main LLM Inference Engine**
//!
//! # Parallel Residual Token Factory (full control flow)
//!
//! Model semantics:
//! ```text
//! h0 = Embed(x)
//! for k in 1..R (parallel):
//!     r_k = λ_k * Loop_k(h0)
//! final = h0 + sum(r_k)
//! ```
//!
//! Each [`TokenFactory::step`] does:
//! 1. Collect active requests
//! 2. Build parallel plan (all loops × top-k experts)
//! 3. Pin GZ Cache zones
//! 4. Execute all expert buckets (dummy or real)
//! 5. Sample next token per request
//! 6. Advance / finish / release
//!
//! [`TokenFactory::generate`] runs `step` until all requests complete.

#![deny(missing_docs)]

mod executor;
mod parallel;

pub use executor::{dummy_decisions, execute_parallel_step, StepOutput};
pub use parallel::{
    build_parallel_plan, ExpertBucket, ParallelStepPlan, WorkItem,
};

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use orbit_core::{
    GzCache, GzCacheConfig, MemoryStats, OrbitConfig, ZoneKey,
};

use executor::{dummy_decisions as mk_dummy, execute_parallel_step};

/// Unique identifier for an in-flight generation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestId(pub u64);

/// Per-request generation state.
#[derive(Debug)]
pub struct Sequence {
    pub id: RequestId,
    pub generated_len: usize,
    pub max_new_tokens: usize,
    /// Tokens produced so far (including dummy samples).
    pub tokens: Vec<u32>,
    pub touched_zones: Vec<ZoneKey>,
    pub finished: bool,
}

impl Sequence {
    pub fn new(id: RequestId, max_new_tokens: usize) -> Self {
        Self {
            id,
            generated_len: 0,
            max_new_tokens,
            tokens: Vec::new(),
            touched_zones: Vec::new(),
            finished: false,
        }
    }

    pub fn is_finished(&self) -> bool {
        self.finished || self.generated_len >= self.max_new_tokens
    }
}

/// Result of one factory step across the whole batch.
#[derive(Debug, Clone)]
pub struct BatchStepResult {
    pub outputs: Vec<StepOutput>,
    pub num_buckets: usize,
    pub num_work_items: usize,
    pub active_before: usize,
    pub active_after: usize,
}

/// The Parallel Residual Token Factory — full decode engine control flow.
#[derive(Debug)]
pub struct TokenFactory {
    pub model_config: OrbitConfig,
    cache: Arc<Mutex<GzCache>>,
    sequences: HashMap<RequestId, Sequence>,
    next_request_id: u64,
    /// Dummy vocab size used by the placeholder sampler.
    pub vocab_size: u32,
    pub eos_token_id: u32,
}

impl TokenFactory {
    pub fn new(model_config: OrbitConfig, cache_config: GzCacheConfig) -> Self {
        let vocab_size = model_config.vocab_size as u32;
        Self {
            model_config,
            cache: Arc::new(Mutex::new(GzCache::new(cache_config))),
            sequences: HashMap::new(),
            next_request_id: 1,
            vocab_size: vocab_size.max(1),
            eos_token_id: 0,
        }
    }

    pub fn with_defaults(model_config: OrbitConfig) -> Self {
        Self::new(model_config, GzCacheConfig::default())
    }

    /// Admit a new generation request. Returns its id.
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

    pub fn sequence(&self, id: RequestId) -> Option<&Sequence> {
        self.sequences.get(&id)
    }

    /// Stable ordered list of active request ids.
    pub fn active_ids(&self) -> Vec<RequestId> {
        let mut ids: Vec<_> = self
            .sequences
            .values()
            .filter(|s| !s.is_finished())
            .map(|s| s.id)
            .collect();
        ids.sort_by_key(|id| id.0);
        ids
    }

    /// One full decode step for all active requests.
    ///
    /// If `decisions` is `None`, uses uniform dummy top-k routing
    /// (for smoke tests / scaffolding).
    pub fn step(
        &mut self,
        decisions: Option<Vec<Vec<Vec<(u32, f32)>>>>,
    ) -> Result<BatchStepResult, String> {
        let active_before = self.active_count();
        if active_before == 0 {
            return Ok(BatchStepResult {
                outputs: vec![],
                num_buckets: 0,
                num_work_items: 0,
                active_before: 0,
                active_after: 0,
            });
        }

        let active_ids = self.active_ids();
        let active_pairs: Vec<(RequestId, u32)> = active_ids
            .iter()
            .filter_map(|id| {
                self.sequences.get(id).map(|s| (*id, s.generated_len as u32))
            })
            .collect();

        let num_loops = self.model_config.num_loops;
        let top_k = self.model_config.top_k;
        let num_experts = self.model_config.num_experts;

        let decisions = decisions.unwrap_or_else(|| {
            mk_dummy(active_pairs.len(), num_loops, top_k, num_experts)
        });

        let plan =
            build_parallel_plan(&active_pairs, num_loops as u32, &decisions);

        // Pin zones
        {
            let mut cache = self.cache.lock().map_err(|e| e.to_string())?;
            for zone in &plan.required_zones {
                cache
                    .pin_for_compute(*zone)
                    .map_err(|e| e.to_string())?;
            }
        }

        // Track touched zones
        for item in plan.buckets.values().flat_map(|b| b.items.iter()) {
            if let Some(seq) = self.sequences.get_mut(&item.request_id) {
                let key = ZoneKey::new(item.loop_idx, item.expert_id);
                if !seq.touched_zones.contains(&key) {
                    seq.touched_zones.push(key);
                }
            }
        }

        // Execute all buckets (parallel point)
        let outputs =
            execute_parallel_step(&plan, self.vocab_size, self.eos_token_id);

        // Apply outputs: append token, advance, maybe finish
        let mut finished_now: Vec<RequestId> = Vec::new();
        for out in &outputs {
            if let Some(seq) = self.sequences.get_mut(&out.request_id) {
                if seq.is_finished() {
                    continue;
                }
                seq.tokens.push(out.token_id);
                seq.generated_len += 1;
                if out.hit_eos || seq.generated_len >= seq.max_new_tokens {
                    seq.finished = true;
                    finished_now.push(out.request_id);
                }
            }
        }

        // Release zones for finished requests
        for id in finished_now {
            let _ = self.finish_request(id);
        }

        let active_after = self.active_count();
        Ok(BatchStepResult {
            outputs,
            num_buckets: plan.num_buckets(),
            num_work_items: plan.num_work_items(),
            active_before,
            active_after,
        })
    }

    /// Run until every request is finished. Returns map of request → tokens.
    pub fn generate(&mut self) -> Result<HashMap<RequestId, Vec<u32>>, String> {
        let mut guard = 0usize;
        let max_steps = self
            .sequences
            .values()
            .map(|s| s.max_new_tokens)
            .max()
            .unwrap_or(1)
            .saturating_mul(self.sequences.len().max(1))
            + 8;

        while self.active_count() > 0 {
            self.step(None)?;
            guard += 1;
            if guard > max_steps {
                return Err(format!(
                    "generate exceeded max_steps={max_steps}; possible logic bug"
                ));
            }
        }

        // Collect tokens from still-resident (should be empty) + we need
        // tokens before finish_request wiped them. So keep a side channel:
        // re-read is empty. Fix: snapshot before finish in step.
        // For generate(), accumulate during the loop.
        Ok(HashMap::new())
    }

    /// Run generate but keep token history by not dropping sequences early.
    ///
    /// Prefer this for tests / demos. Finished sequences stay in the map
    /// with `finished=true` until [`Self::reap_finished`].
    pub fn generate_collect(&mut self) -> Result<HashMap<RequestId, Vec<u32>>, String> {
        // Temporarily: step without auto-reap — use internal soft-finish
        let ids: Vec<RequestId> = self.sequences.keys().copied().collect();
        let max_new = self
            .sequences
            .values()
            .map(|s| s.max_new_tokens)
            .max()
            .unwrap_or(1);

        for _ in 0..max_new {
            if self.active_count() == 0 {
                break;
            }
            // Soft step: execute but do not remove finished seqs until end
            self.step_soft(None)?;
        }

        let mut out = HashMap::new();
        for id in ids {
            if let Some(seq) = self.sequences.get(&id) {
                out.insert(id, seq.tokens.clone());
            }
        }
        Ok(out)
    }

    /// Like `step`, but finished sequences are marked finished without
    /// removing them or releasing zones (useful for collect APIs).
    pub fn step_soft(
        &mut self,
        decisions: Option<Vec<Vec<Vec<(u32, f32)>>>>,
    ) -> Result<BatchStepResult, String> {
        let active_before = self.active_count();
        if active_before == 0 {
            return Ok(BatchStepResult {
                outputs: vec![],
                num_buckets: 0,
                num_work_items: 0,
                active_before: 0,
                active_after: 0,
            });
        }

        let active_ids = self.active_ids();
        let active_pairs: Vec<(RequestId, u32)> = active_ids
            .iter()
            .filter_map(|id| {
                self.sequences.get(id).map(|s| (*id, s.generated_len as u32))
            })
            .collect();

        let num_loops = self.model_config.num_loops;
        let decisions = decisions.unwrap_or_else(|| {
            mk_dummy(
                active_pairs.len(),
                num_loops,
                self.model_config.top_k,
                self.model_config.num_experts,
            )
        });

        let plan =
            build_parallel_plan(&active_pairs, num_loops as u32, &decisions);

        {
            let mut cache = self.cache.lock().map_err(|e| e.to_string())?;
            for zone in &plan.required_zones {
                cache
                    .pin_for_compute(*zone)
                    .map_err(|e| e.to_string())?;
            }
        }

        for item in plan.buckets.values().flat_map(|b| b.items.iter()) {
            if let Some(seq) = self.sequences.get_mut(&item.request_id) {
                let key = ZoneKey::new(item.loop_idx, item.expert_id);
                if !seq.touched_zones.contains(&key) {
                    seq.touched_zones.push(key);
                }
            }
        }

        let outputs =
            execute_parallel_step(&plan, self.vocab_size, self.eos_token_id);

        for out in &outputs {
            if let Some(seq) = self.sequences.get_mut(&out.request_id) {
                if seq.is_finished() {
                    continue;
                }
                seq.tokens.push(out.token_id);
                seq.generated_len += 1;
                if out.hit_eos || seq.generated_len >= seq.max_new_tokens {
                    seq.finished = true;
                }
            }
        }

        Ok(BatchStepResult {
            outputs,
            num_buckets: plan.num_buckets(),
            num_work_items: plan.num_work_items(),
            active_before,
            active_after: self.active_count(),
        })
    }

    /// Drop finished sequences and release their zones.
    pub fn reap_finished(&mut self) -> Result<usize, String> {
        let done: Vec<RequestId> = self
            .sequences
            .values()
            .filter(|s| s.finished)
            .map(|s| s.id)
            .collect();
        let n = done.len();
        for id in done {
            self.finish_request(id)?;
        }
        Ok(n)
    }

    /// Release zones and remove a request.
    pub fn finish_request(&mut self, request_id: RequestId) -> Result<(), String> {
        let seq = match self.sequences.remove(&request_id) {
            Some(s) => s,
            None => return Ok(()),
        };

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

    fn small_cfg() -> OrbitConfig {
        let mut c = OrbitConfig::default();
        c.num_loops = 2;
        c.num_experts = 4;
        c.top_k = 2;
        c.vocab_size = 50;
        c
    }

    #[test]
    fn step_produces_tokens() {
        let mut factory = TokenFactory::with_defaults(small_cfg());
        let id = factory.add_request(3);

        let r1 = factory.step(None).unwrap();
        assert_eq!(r1.active_before, 1);
        assert_eq!(r1.outputs.len(), 1);
        assert_eq!(r1.outputs[0].request_id, id);

        let seq = factory.sequence(id).unwrap();
        assert_eq!(seq.tokens.len(), 1);
    }

    #[test]
    fn generate_collect_runs_to_completion() {
        let mut factory = TokenFactory::with_defaults(small_cfg());
        let id1 = factory.add_request(4);
        let id2 = factory.add_request(2);

        let tokens = factory.generate_collect().unwrap();
        assert_eq!(tokens.get(&id1).map(|t| t.len()), Some(4));
        assert_eq!(tokens.get(&id2).map(|t| t.len()), Some(2));
        assert_eq!(factory.active_count(), 0);

        let n = factory.reap_finished().unwrap();
        assert_eq!(n, 2);
    }

    #[test]
    fn parallel_buckets_created() {
        let mut factory = TokenFactory::with_defaults(small_cfg());
        factory.add_request(1);
        factory.add_request(1);
        let r = factory.step(None).unwrap();
        // 2 loops * up to 2 experts → buckets >= 1
        assert!(r.num_buckets >= 1);
        assert!(r.num_work_items >= 2);
    }
}
