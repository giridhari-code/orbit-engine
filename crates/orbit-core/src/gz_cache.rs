//! Grouped-Zonal Cache (GZ Cache)
//!
//! Memory is organized by *routing path* rather than pure token position.
//! A Zone is identified by the tuple `(loop_idx, expert_id)`.
//!
//! Four pillars:
//! 1. Route over Store     — only allocate for active expert paths
//! 2. Zone the Memory      — zones keyed by (loop, expert)
//! 3. Reuse over Recompute — base tensors modulated by IterAdaLN (future)
//! 4. Zip the Inactive     — compress + offload to CPU
//!
//! This module focuses on the *memory management and lifecycle* layer.
//! Actual tensor storage is abstracted behind the [`ZoneBuffer`] trait.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::buffer::{DeviceKind, PlaceholderBuffer, ZoneBuffer};
use crate::{OrbitError, Result};

/// Unique key for a memory zone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ZoneKey {
    pub loop_idx: u32,
    pub expert_id: u32,
}

impl ZoneKey {
    #[inline]
    pub fn new(loop_idx: u32, expert_id: u32) -> Self {
        Self { loop_idx, expert_id }
    }
}

impl fmt::Display for ZoneKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(loop={}, expert={})", self.loop_idx, self.expert_id)
    }
}

/// Lifecycle state of a zone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZoneState {
    GpuResident,
    CpuResident,
    Compressed,
    InTransit { to_gpu: bool },
}

impl fmt::Display for ZoneState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZoneState::GpuResident => write!(f, "GpuResident"),
            ZoneState::CpuResident => write!(f, "CpuResident"),
            ZoneState::Compressed => write!(f, "Compressed"),
            ZoneState::InTransit { to_gpu } => write!(f, "InTransit(to_gpu={})", to_gpu),
        }
    }
}

/// A single zone holding KV data for tokens that routed to a particular (loop, expert) pair.
#[derive(Debug)]
pub struct Zone {
    pub key: ZoneKey,
    pub state: ZoneState,
    pub buffer: Arc<dyn ZoneBuffer>,
    pub last_used: u64,
}

impl Zone {
    fn new(key: ZoneKey) -> Self {
        Self {
            key,
            state: ZoneState::CpuResident,
            buffer: Arc::new(PlaceholderBuffer::empty(DeviceKind::Cpu)),
            last_used: 0,
        }
    }

    #[inline]
    pub fn is_gpu_resident(&self) -> bool {
        matches!(self.state, ZoneState::GpuResident)
    }

    #[inline]
    pub fn is_ready_for_compute(&self) -> bool {
        matches!(self.state, ZoneState::GpuResident)
    }

    #[inline]
    pub fn seq_len(&self) -> usize {
        self.buffer.seq_len()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.buffer.capacity()
    }

    #[inline]
    pub fn bytes_used(&self) -> u64 {
        self.buffer.bytes_used()
    }
}

/// Snapshot of memory usage.
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub gpu_bytes: u64,
    pub cpu_bytes: u64,
    pub compressed_bytes: u64,
    pub num_zones: usize,
    pub num_gpu_resident: usize,
    pub num_cpu_resident: usize,
    pub num_compressed: usize,
    pub num_in_transit: usize,
}

/// Configuration for the cache manager.
#[derive(Debug, Clone)]
pub struct GzCacheConfig {
    pub max_gpu_zones: usize,
    pub max_gpu_bytes: u64,
    pub compress_on_offload: bool,
}

impl Default for GzCacheConfig {
    fn default() -> Self {
        Self {
            max_gpu_zones: 32,
            max_gpu_bytes: 0,
            compress_on_offload: false,
        }
    }
}

/// The Grouped-Zonal Cache manager.
#[derive(Debug)]
pub struct GzCache {
    config: GzCacheConfig,
    zones: HashMap<ZoneKey, Zone>,
    clock: u64,
}

impl GzCache {
    pub fn new(config: GzCacheConfig) -> Self {
        Self {
            config,
            zones: HashMap::new(),
            clock: 0,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(GzCacheConfig::default())
    }

    pub fn get(&self, key: ZoneKey) -> Option<&Zone> {
        self.zones.get(&key)
    }

    pub fn get_mut(&mut self, key: ZoneKey) -> Option<&mut Zone> {
        self.zones.get_mut(&key)
    }

    pub fn get_or_create(&mut self, key: ZoneKey) -> &mut Zone {
        self.zones.entry(key).or_insert_with(|| Zone::new(key))
    }

    pub fn contains(&self, key: ZoneKey) -> bool {
        self.zones.contains_key(&key)
    }

    pub fn len(&self) -> usize {
        self.zones.len()
    }

    pub fn is_empty(&self) -> bool {
        self.zones.is_empty()
    }

    pub fn pin_for_compute(&mut self, key: ZoneKey) -> Result<bool> {
        self.clock = self.clock.wrapping_add(1);

        let already_ready = {
            let zone = self.get_or_create(key);
            zone.last_used = self.clock;
            zone.is_ready_for_compute()
        };

        if already_ready {
            return Ok(true);
        }

        self.ensure_gpu_capacity_for(key)?;

        let zone = self.get_or_create(key);
        match zone.state {
            ZoneState::GpuResident => Ok(true),
            ZoneState::CpuResident | ZoneState::Compressed => {
                zone.state = ZoneState::GpuResident;
                Ok(false)
            }
            ZoneState::InTransit { to_gpu: true } => Ok(false),
            ZoneState::InTransit { to_gpu: false } => Err(OrbitError::Cache(format!(
                "zone {} is currently being offloaded; cannot pin",
                key
            ))),
        }
    }

    pub fn release(&mut self, key: ZoneKey) -> Result<()> {
        if let Some(zone) = self.zones.get_mut(&key) {
            if matches!(zone.state, ZoneState::GpuResident) {
                zone.state = ZoneState::CpuResident;
            }
        }
        Ok(())
    }

    pub fn offload(&mut self, key: ZoneKey) -> Result<()> {
        let compress = self.config.compress_on_offload;
        if let Some(zone) = self.zones.get_mut(&key) {
            match zone.state {
                ZoneState::GpuResident | ZoneState::CpuResident => {
                    zone.state = if compress {
                        ZoneState::Compressed
                    } else {
                        ZoneState::CpuResident
                    };
                    Ok(())
                }
                ZoneState::Compressed => Ok(()),
                ZoneState::InTransit { .. } => Err(OrbitError::Cache(format!(
                    "zone {} is in transit; cannot offload",
                    key
                ))),
            }
        } else {
            Ok(())
        }
    }

    pub fn prefetch(&mut self, key: ZoneKey) -> Result<()> {
        let _ = self.get_or_create(key);
        Ok(())
    }

    pub fn drop_zone(&mut self, key: ZoneKey) -> Result<()> {
        self.zones.remove(&key);
        Ok(())
    }

    pub fn clear(&mut self) {
        self.zones.clear();
        self.clock = 0;
    }

    fn ensure_gpu_capacity_for(&mut self, _needed: ZoneKey) -> Result<()> {
        let gpu_count = self.zones.values().filter(|z| z.is_gpu_resident()).count();

        if gpu_count < self.config.max_gpu_zones {
            return Ok(());
        }

        let victim = self
            .zones
            .iter()
            .filter(|(_, z)| z.is_gpu_resident())
            .min_by_key(|(_, z)| z.last_used)
            .map(|(k, _)| *k);

        if let Some(victim_key) = victim {
            self.offload(victim_key)?;
        }

        Ok(())
    }

    pub fn memory_stats(&self) -> MemoryStats {
        let mut stats = MemoryStats {
            num_zones: self.zones.len(),
            ..Default::default()
        };

        for zone in self.zones.values() {
            let bytes = zone.bytes_used();
            match zone.state {
                ZoneState::GpuResident => {
                    stats.gpu_bytes += bytes;
                    stats.num_gpu_resident += 1;
                }
                ZoneState::CpuResident => {
                    stats.cpu_bytes += bytes;
                    stats.num_cpu_resident += 1;
                }
                ZoneState::Compressed => {
                    stats.compressed_bytes += bytes;
                    stats.num_compressed += 1;
                }
                ZoneState::InTransit { .. } => {
                    stats.num_in_transit += 1;
                }
            }
        }

        stats
    }

    pub fn gpu_resident_keys(&self) -> Vec<ZoneKey> {
        self.zones
            .iter()
            .filter(|(_, z)| z.is_gpu_resident())
            .map(|(k, _)| *k)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_lifecycle() {
        let mut cache = GzCache::with_defaults();
        let key = ZoneKey::new(0, 3);

        assert!(!cache.contains(key));
        let was_ready = cache.pin_for_compute(key).unwrap();
        assert!(!was_ready);
        assert!(cache.get(key).unwrap().is_gpu_resident());

        let was_ready = cache.pin_for_compute(key).unwrap();
        assert!(was_ready);

        cache.release(key).unwrap();
        assert!(!cache.get(key).unwrap().is_gpu_resident());
    }

    #[test]
    fn lru_eviction() {
        let config = GzCacheConfig {
            max_gpu_zones: 2,
            ..Default::default()
        };
        let mut cache = GzCache::new(config);

        let k0 = ZoneKey::new(0, 0);
        let k1 = ZoneKey::new(0, 1);
        let k2 = ZoneKey::new(0, 2);

        cache.pin_for_compute(k0).unwrap();
        cache.pin_for_compute(k1).unwrap();
        assert_eq!(cache.memory_stats().num_gpu_resident, 2);

        cache.pin_for_compute(k2).unwrap();
        let stats = cache.memory_stats();
        assert_eq!(stats.num_gpu_resident, 2);
        assert!(stats.num_cpu_resident + stats.num_compressed >= 1);
    }
}
