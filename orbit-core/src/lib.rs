//! Orbit Core — Looped Mixture-of-Experts primitives + Grouped-Zonal Cache
//!
//! This crate provides the mathematical and memory building blocks for
//! the Orbit inference engine:
//!
//! - [`config`]        — model & runtime configuration
//! - [`gz_cache`]      — Grouped-Zonal Cache (route / zone / reuse / zip)
//! - [`looped_moe`]    — shared experts + per-loop routers + residual scaling
//! - [`iter_adln`]     — IterAdaLN (iteration-aware adaptive LayerNorm)
//! - [`router`]        — Top-K routing utilities

#![deny(missing_docs)]
#![allow(dead_code)] // scaffolding

pub mod config;
pub mod gz_cache;
pub mod iter_adln;
pub mod looped_moe;
pub mod router;

pub use config::OrbitConfig;
pub use gz_cache::{GzCache, Zone, ZoneKey};
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
