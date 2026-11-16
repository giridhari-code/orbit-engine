//! Abstract buffer interface for GZ Cache zones.
//!
//! The goal is to keep the cache lifecycle logic independent of the concrete
//! tensor / memory backend (candle, custom CUDA, CPU Vec, compressed blob, …).
//!
//! Later we will provide real implementations:
//! - `CandleZoneBuffer`
//! - `CudaZoneBuffer`
//! - `CompressedZoneBuffer`
//!
//! For now we only ship a `PlaceholderBuffer` so the rest of the system can
//! compile and be tested.

use std::fmt;

/// Device on which a buffer currently resides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Gpu { device_id: u32 },
}

impl fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceKind::Cpu => write!(f, "CPU"),
            DeviceKind::Gpu { device_id } => write!(f, "GPU:{}", device_id),
        }
    }
}

/// Minimal interface every zone buffer must satisfy.
///
/// This trait is intentionally small. We will grow it only when a real
/// backend forces us to.
pub trait ZoneBuffer: Send + Sync + fmt::Debug {
    /// Approximate number of bytes currently occupied.
    fn bytes_used(&self) -> u64;

    /// Current logical sequence length (number of tokens stored).
    fn seq_len(&self) -> usize;

    /// Allocated capacity in tokens.
    fn capacity(&self) -> usize;

    /// Where the data currently lives.
    fn device(&self) -> DeviceKind;

    /// Whether the buffer is compressed.
    fn is_compressed(&self) -> bool {
        false
    }
}

/// Simple placeholder used while we do not yet have real tensors.
///
/// It only tracks metadata so that memory accounting and lifecycle
/// logic can be exercised in tests.
#[derive(Debug, Clone)]
pub struct PlaceholderBuffer {
    pub seq_len: usize,
    pub capacity: usize,
    pub bytes_used: u64,
    pub device: DeviceKind,
    pub compressed: bool,
}

impl PlaceholderBuffer {
    pub fn empty(device: DeviceKind) -> Self {
        Self {
            seq_len: 0,
            capacity: 0,
            bytes_used: 0,
            device,
            compressed: false,
        }
    }

    pub fn with_size(seq_len: usize, bytes_per_token: u64, device: DeviceKind) -> Self {
        Self {
            seq_len,
            capacity: seq_len,
            bytes_used: seq_len as u64 * bytes_per_token,
            device,
            compressed: false,
        }
    }
}

impl ZoneBuffer for PlaceholderBuffer {
    fn bytes_used(&self) -> u64 {
        self.bytes_used
    }

    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn device(&self) -> DeviceKind {
        self.device
    }

    fn is_compressed(&self) -> bool {
        self.compressed
    }
}
