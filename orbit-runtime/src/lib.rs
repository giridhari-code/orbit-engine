//! Orbit Runtime — scheduling, continuous batching, and cache orchestration.

#![deny(missing_docs)]

pub use orbit_core as core;

/// Placeholder for the continuous-batching scheduler.
#[derive(Debug, Default)]
pub struct Scheduler {
    /// Maximum concurrent sequences.
    pub max_batch_size: usize,
}

impl Scheduler {
    /// Create a new scheduler.
    pub fn new(max_batch_size: usize) -> Self {
        Self { max_batch_size }
    }
}
