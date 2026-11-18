//! Orbit Core — Parallel Residual LoopMoE primitives + Grouped-Zonal Cache
//!
//! Model semantics (default):
//!   h0 = Embed(x)
//!   for k in 1..R (parallel):
//!       r_k = λ_k * Loop_k(h0)
//!   final = h0 + sum(r_k)
//!
//! Modules:
//! - [`config`]     — OrbitConfig + LoopSemantics
//! - [`buffer`]     — ZoneBuffer trait
//! - [`gz_cache`]   — Grouped-Zonal Cache
//! - [`looped_moe`] — parallel residual layer skeleton
//! - [`iter_adln`]  — iteration-aware adaptive LN
//! - [`router`]     — Top-K routing

#![deny(missing_docs)]
#![allow(dead_code)]

pub mod buffer;
pub mod config;
pub mod gz_cache;
pub mod iter_adln;
pub mod looped_moe;
pub mod router;

pub use buffer::{DeviceKind, PlaceholderBuffer, ZoneBuffer};
pub use config::{LoopSemantics, OrbitConfig};
pub use gz_cache::{
    GzCache, GzCacheConfig, MemoryStats, Zone, ZoneKey, ZoneState,
};
pub use looped_moe::LoopedMoELayer;

/// Crate-level error type.
#[derive(Debug, thiserror::Error)]
pub enum OrbitError {
    #[error("configuration error: {0}")]
    Config(String),
    #[error("cache error: {0}")]
    Cache(String),
    #[error("routing error: {0}")]
    Routing(String),
    #[error("tensor error: {0}")]
    Tensor(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, OrbitError>;
