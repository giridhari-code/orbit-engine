//! Grouped-Zonal Cache (GZ Cache)
//!
//! Memory is organized by *routing path* rather than pure token position.
//! A Zone is identified by the tuple `(loop_idx, expert_id)`.
//!
//! Four pillars:
//! 1. Route over Store
//! 2. Zone the Memory
//! 3. Reuse over Recompute (modulate shared base with IterAdaLN)
//! 4. Zip the Inactive (compress + offload)

use std::collections::HashMap;

use crate::{OrbitError, Result};

/// Unique key for a memory zone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ZoneKey {
    /// Loop / iteration index (0-based or 1-based depending on convention).
    pub loop_idx: usize,
    /// Expert identifier.
    pub expert_id: usize,
}

impl ZoneKey {
    /// Construct a new zone key.
    pub fn new(loop_idx: usize, expert_id: usize) -> Self {
        Self {
            loop_idx,
            expert_id,
        }
    }
}

/// A single zone holding keys and values for tokens routed to a particular
/// (loop, expert) pair.
///
/// In a full implementation the tensors would be `candle::Tensor`.
/// Here we keep a lightweight placeholder so the structure is clear.
#[derive(Debug)]
pub struct Zone {
    /// Which loop this zone belongs to.
    pub loop_idx: usize,
    /// Which expert.
    pub expert_id: usize,
    /// Whether the zone currently resides in GPU VRAM.
    pub is_active: bool,
    /// Logical sequence length currently stored.
    pub seq_len: usize,
    // keys: Tensor,   // [seq_len, head_dim]  (placeholder)
    // values: Tensor, // [seq_len, head_dim]
}

impl Zone {
    /// Create an empty inactive zone.
    pub fn empty(loop_idx: usize, expert_id: usize) -> Self {
        Self {
            loop_idx,
            expert_id,
            is_active: false,
            seq_len: 0,
        }
    }
}

/// The Grouped-Zonal Cache manager.
#[derive(Debug, Default)]
pub struct GzCache {
    /// Active zones keyed by (loop, expert).
    zones: HashMap<ZoneKey, Zone>,
    /// Maximum number of concurrent active zones (VRAM budget).
    max_active_zones: usize,
}

impl GzCache {
    /// Create a new cache with a soft limit on simultaneously active zones.
    pub fn new(max_active_zones: usize) -> Self {
        Self {
            zones: HashMap::new(),
            max_active_zones,
        }
    }

    /// Obtain (or create) a zone for the given routing path.
    pub fn get_or_create(&mut self, key: ZoneKey) -> &mut Zone {
        self.zones
            .entry(key)
            .or_insert_with(|| Zone::empty(key.loop_idx, key.expert_id))
    }

    /// Mark a zone as active (bring into VRAM).
    /// Returns true if it was already active.
    pub fn activate(&mut self, key: ZoneKey) -> Result<bool> {
        let zone = self.get_or_create(key);
        if zone.is_active {
            return Ok(true);
        }
        // TODO: real eviction / zip logic when max_active_zones is exceeded
        zone.is_active = true;
        Ok(false)
    }

    /// Mark a zone as inactive (eligible for zip + offload).
    pub fn deactivate(&mut self, key: ZoneKey) -> Result<()> {
        if let Some(zone) = self.zones.get_mut(&key) {
            zone.is_active = false;
        }
        Ok(())
    }

    /// Number of currently active zones.
    pub fn active_count(&self) -> usize {
        self.zones.values().filter(|z| z.is_active).count()
    }

    /// Clear all zones (end of sequence / request).
    pub fn clear(&mut self) {
        self.zones.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zone_lifecycle() {
        let mut cache = GzCache::new(4);
        let key = ZoneKey::new(0, 3);
        assert!(!cache.activate(key).unwrap()); // was inactive
        assert!(cache.activate(key).unwrap()); // now active
        assert_eq!(cache.active_count(), 1);
        cache.deactivate(key).unwrap();
        assert_eq!(cache.active_count(), 0);
    }
}
