//! Orbit Runtime — scheduling, continuous batching, and cache orchestration.
//!
//! This layer sits between the server and the core compute.
//! Its job is to decide *when* zones should be pinned, released, or prefetched,
//! and to keep per-request state coherent across loops.

#![deny(missing_docs)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use orbit_core::{
    GzCache, GzCacheConfig, MemoryStats, OrbitConfig, ZoneKey,
};

/// Unique identifier for an in-flight generation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestId(pub u64);

/// State of a single sequence / request.
#[derive(Debug)]
pub struct Sequence {
    pub id: RequestId,
    /// How many tokens have been generated so far.
    pub generated_len: usize,
    /// Maximum tokens this request is allowed to generate.
    pub max_new_tokens: usize,
    /// Current loop index we are executing (0..R).
    pub current_loop: u32,
    /// Zones this sequence has touched (for cleanup).
    pub touched_zones: Vec<ZoneKey>,
    /// Whether the sequence is finished.
    pub finished: bool,
}

impl Sequence {
    pub fn new(id: RequestId, max_new_tokens: usize) -> Self {
        Self {
            id,
            generated_len: 0,
            max_new_tokens,
            current_loop: 0,
            touched_zones: Vec::new(),
            finished: false,
        }
    }

    pub fn is_finished(&self) -> bool {
        self.finished || self.generated_len >= self.max_new_tokens
    }
}

/// Simple scheduler that owns the GZ Cache and a set of active sequences.
///
/// This is still a single-threaded, synchronous sketch. Continuous batching
/// and async offload will be layered on top later.
#[derive(Debug)]
pub struct Scheduler {
    pub model_config: OrbitConfig,
    cache: Arc<Mutex<GzCache>>,
    sequences: HashMap<RequestId, Sequence>,
    next_request_id: u64,
}

impl Scheduler {
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

    /// Number of currently active (not finished) sequences.
    pub fn active_count(&self) -> usize {
        self.sequences.values().filter(|s| !s.is_finished()).count()
    }

    /// Prepare zones for a given loop of a request.
    ///
    /// In a real system the router would tell us which experts were selected.
    /// Here we accept the list of expert ids explicitly so the call site stays clear.
    pub fn prepare_loop(
        &mut self,
        request_id: RequestId,
        loop_idx: u32,
        expert_ids: &[u32],
    ) -> Result<(), String> {
        let seq = self
            .sequences
            .get_mut(&request_id)
            .ok_or_else(|| format!("unknown request {:?}", request_id))?;

        if seq.is_finished() {
            return Err("sequence already finished".into());
        }

        seq.current_loop = loop_idx;

        let mut cache = self.cache.lock().map_err(|e| e.to_string())?;

        for &expert_id in expert_ids {
            let key = ZoneKey::new(loop_idx, expert_id);
            cache
                .pin_for_compute(key)
                .map_err(|e| e.to_string())?;

            if !seq.touched_zones.contains(&key) {
                seq.touched_zones.push(key);
            }
        }

        Ok(())
    }

    /// Mark that one token has been generated for this request.
    pub fn advance_token(&mut self, request_id: RequestId) -> Result<(), String> {
        let seq = self
            .sequences
            .get_mut(&request_id)
            .ok_or_else(|| format!("unknown request {:?}", request_id))?;

        seq.generated_len += 1;
        if seq.generated_len >= seq.max_new_tokens {
            seq.finished = true;
        }
        Ok(())
    }

    /// Release all zones touched by a finished request.
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

    /// Current memory statistics from the underlying cache.
    pub fn memory_stats(&self) -> Result<MemoryStats, String> {
        let cache = self.cache.lock().map_err(|e| e.to_string())?;
        Ok(cache.memory_stats())
    }

    /// Access the shared cache (for advanced use / testing).
    pub fn cache(&self) -> Arc<Mutex<GzCache>> {
        Arc::clone(&self.cache)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use orbit_core::OrbitConfig;

    #[test]
    fn request_lifecycle() {
        let mut sched = Scheduler::with_defaults(OrbitConfig::default());
        let id = sched.add_request(10);

        assert_eq!(sched.active_count(), 1);

        // Simulate one loop with two experts
        sched.prepare_loop(id, 0, &[1, 4]).unwrap();

        let stats = sched.memory_stats().unwrap();
        assert_eq!(stats.num_gpu_resident, 2);

        sched.advance_token(id).unwrap();
        sched.finish_request(id).unwrap();

        assert_eq!(sched.active_count(), 0);
    }
}
