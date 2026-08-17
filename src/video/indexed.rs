//! Bounded source-time indexing and reverse-frame caching for video transport.
//!
//! Live callers build the keyframe index on the decoder worker.  The index is
//! deliberately sparse and bounded: FFmpeg remains the authority for the
//! actual seek, while the index supplies a known preceding keyframe rather
//! than making reverse playback reopen the source or scan from frame zero.

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ffmpeg_next::format::context::Input;

use crate::media_safety::{ReverseCacheLease, ReverseCacheLedger};

use super::codec_motion::CodecFrameIdentity;
use super::codec_motion_sequence::CodecMotionProduct;

/// Hard entry bound for one source's in-memory keyframe index.
pub const MAX_KEYFRAME_INDEX_ENTRIES: usize = 4_096;
/// A corrupt or keyframe-free input cannot make an index pass unbounded.
pub const MAX_INDEX_PACKETS: usize = 1_000_000;
/// Indexing is opportunistic; playback can seek with the entries found so far.
pub const MAX_INDEX_BUILD_TIME: Duration = Duration::from_secs(3);
/// Per-decoder reverse cache ceiling from the host performance budget.
pub const MAX_REVERSE_CACHE_BYTES: usize = 32 * 1024 * 1024;
/// A second bound protects tiny-frame sources from retaining huge frame counts.
pub const MAX_REVERSE_CACHE_FRAMES: usize = 512;

/// Source-time metadata attached to every decoded result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameMetadata {
    /// Generation of the selected clip/source. Frames from older generations
    /// are never eligible for publication.
    pub source_generation: u64,
    /// Best-effort presentation timestamp in the source stream's time base.
    pub pts: Option<i64>,
    /// Presentation time relative to the start of the source.
    pub source_seconds: f64,
    /// Total source duration, or zero when the container cannot report one.
    pub duration_seconds: f64,
    /// Exact identity is present only for decoded video frames with an integer
    /// source PTS. It is the sole predecessor token for codec-motion proof.
    pub codec_identity: Option<CodecFrameIdentity>,
}

impl FrameMetadata {
    pub const fn still() -> Self {
        Self {
            source_generation: 0,
            pts: None,
            source_seconds: 0.0,
            duration_seconds: 0.0,
            codec_identity: None,
        }
    }

    pub fn sanitized(
        source_generation: u64,
        pts: Option<i64>,
        source_seconds: f64,
        duration_seconds: f64,
    ) -> Self {
        Self {
            source_generation,
            pts,
            source_seconds: finite_nonnegative(source_seconds),
            duration_seconds: finite_nonnegative(duration_seconds),
            codec_identity: None,
        }
    }

    pub fn with_codec_identity(mut self, identity: Option<CodecFrameIdentity>) -> Self {
        self.codec_identity = identity;
        self
    }
}

impl Default for FrameMetadata {
    fn default() -> Self {
        Self::still()
    }
}

/// Metadata-rich result returned by indexed decoder entry points.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedVideoFrame {
    pub rgba: Vec<u8>,
    pub metadata: FrameMetadata,
    /// Sparse codec records paired with this exact image/generation. Still
    /// sources and reverse-cache hits are explicitly absent.
    pub codec_motion: Option<CodecMotionProduct>,
}

/// Recoverable decoder-work outcome used by generation-aware workers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeWorkError {
    /// A newer absolute selection invalidated this work item.
    Superseded,
    /// FFmpeg, validation, cancellation, or allocation failed.
    Failed(String),
}

impl fmt::Display for DecodeWorkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Superseded => formatter.write_str("decode work was superseded"),
            Self::Failed(error) => formatter.write_str(error),
        }
    }
}

impl std::error::Error for DecodeWorkError {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeyframeEntry {
    pub pts: i64,
    pub source_seconds: f64,
}

/// Sparse, monotonically ordered keyframe table.
#[derive(Debug, Clone)]
pub struct KeyframeIndex {
    entries: Vec<KeyframeEntry>,
}

impl KeyframeIndex {
    pub fn fallback(start_pts: i64) -> Result<Self, String> {
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(1)
            .map_err(|error| format!("could not reserve fallback keyframe index: {error}"))?;
        entries.push(KeyframeEntry {
            pts: start_pts,
            source_seconds: 0.0,
        });
        Ok(Self { entries })
    }

    /// Scan packet headers only. Call this on a media worker, never in a
    /// render callback. Allocation failure returns a normal error and leaves
    /// the decoder's fallback zero entry usable.
    pub fn scan(
        input: &mut Input,
        stream_index: usize,
        time_base_seconds: f64,
        start_pts: i64,
        cancel: &AtomicBool,
    ) -> Result<Self, String> {
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(MAX_KEYFRAME_INDEX_ENTRIES)
            .map_err(|error| format!("could not reserve bounded keyframe index: {error}"))?;
        entries.push(KeyframeEntry {
            pts: start_pts,
            source_seconds: 0.0,
        });

        let deadline = Instant::now() + MAX_INDEX_BUILD_TIME;
        let mut packets_scanned = 0usize;
        let mut keyframe_ordinal = 0usize;
        let mut stride = 1usize;
        for (stream, packet) in input.packets() {
            if cancel.load(Ordering::Acquire)
                || packets_scanned >= MAX_INDEX_PACKETS
                || Instant::now() >= deadline
            {
                break;
            }
            packets_scanned = packets_scanned.saturating_add(1);
            if stream.index() != stream_index || !packet.is_key() {
                continue;
            }
            let Some(pts) = packet.pts().or_else(|| packet.dts()) else {
                continue;
            };
            let source_seconds = timestamp_to_source_seconds(pts, start_pts, time_base_seconds);
            keyframe_ordinal = keyframe_ordinal.saturating_add(1);
            if !keyframe_ordinal.is_multiple_of(stride) {
                continue;
            }
            if entries
                .last()
                .is_some_and(|entry| pts <= entry.pts || source_seconds <= entry.source_seconds)
            {
                continue;
            }

            if entries.len() == MAX_KEYFRAME_INDEX_ENTRIES {
                // Retain zero plus every second indexed keyframe, then sample
                // future keyframes at the matching wider stride. This keeps
                // source-wide coverage without ever growing the allocation.
                let mut write = 1usize;
                for read in (2..entries.len()).step_by(2) {
                    entries[write] = entries[read];
                    write += 1;
                }
                entries.truncate(write);
                stride = stride.saturating_mul(2).max(2);
            }
            entries.push(KeyframeEntry {
                pts,
                source_seconds,
            });
        }

        Ok(Self { entries })
    }

    pub fn preceding(&self, target_seconds: f64) -> KeyframeEntry {
        let target_seconds = finite_nonnegative(target_seconds);
        let index = self
            .entries
            .partition_point(|entry| entry.source_seconds <= target_seconds)
            .saturating_sub(1);
        self.entries[index]
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

#[derive(Debug)]
struct CachedFrame {
    rgba: Vec<u8>,
    metadata: FrameMetadata,
    _lease: ReverseCacheLease,
}

/// One decoder's bounded decoded-frame cache. It retains a GOP-sized recent
/// source-time window when practical and evicts oldest frames first. Codec
/// motion is deliberately noncached: entries lease/account only RGBA bytes,
/// and retrieval returns `codec_motion: None`. The live decoder and depth-one
/// threaded mailbox remain the only bounded owners of codec records.
#[derive(Debug)]
pub struct ReverseFrameCache {
    frames: VecDeque<CachedFrame>,
    bytes: usize,
    max_bytes: usize,
    max_frames: usize,
    ledger: Arc<ReverseCacheLedger>,
}

impl Default for ReverseFrameCache {
    fn default() -> Self {
        Self::new(MAX_REVERSE_CACHE_BYTES, MAX_REVERSE_CACHE_FRAMES)
    }
}

impl ReverseFrameCache {
    pub fn new(max_bytes: usize, max_frames: usize) -> Self {
        let aggregate_bytes = u64::try_from(max_bytes).unwrap_or(u64::MAX);
        Self::with_ledger(
            max_bytes,
            max_frames,
            ReverseCacheLedger::new(aggregate_bytes),
        )
    }

    pub(crate) fn with_ledger(
        max_bytes: usize,
        max_frames: usize,
        ledger: Arc<ReverseCacheLedger>,
    ) -> Self {
        Self {
            frames: VecDeque::new(),
            bytes: 0,
            max_bytes,
            max_frames,
            ledger,
        }
    }

    pub fn insert(&mut self, frame: &DecodedVideoFrame) -> Result<(), String> {
        let frame_bytes = frame.rgba.len();
        if frame_bytes > self.max_bytes || self.max_frames == 0 {
            self.clear();
            return Ok(());
        }
        self.frames
            .try_reserve(1)
            .map_err(|error| format!("could not reserve reverse-frame cache slot: {error}"))?;
        while self.frames.len() >= self.max_frames
            || self
                .bytes
                .checked_add(frame_bytes)
                .is_none_or(|next| next > self.max_bytes)
        {
            let Some(evicted) = self.frames.pop_front() else {
                break;
            };
            self.bytes = self.bytes.saturating_sub(evicted.rgba.len());
        }
        let lease = match u64::try_from(frame_bytes)
            .map_err(|_| "reverse-frame cache entry size does not fit u64".to_string())
            .and_then(|bytes| self.ledger.try_reserve(bytes).map_err(str::to_string))
        {
            Ok(lease) => lease,
            // Reverse caching is an optimization. Another decoder may own the
            // remaining aggregate budget; declining this entry must never turn
            // successful source decoding into an audience-visible failure.
            Err(error) if error == "aggregate reverse cache budget exceeded" => return Ok(()),
            Err(error) => return Err(error),
        };
        let mut rgba = Vec::new();
        rgba.try_reserve_exact(frame_bytes)
            .map_err(|error| format!("could not reserve reverse-frame cache entry: {error}"))?;
        rgba.extend_from_slice(&frame.rgba);
        self.bytes = self.bytes.checked_add(frame_bytes).ok_or_else(|| {
            "reverse-frame cache byte accounting overflowed unexpectedly".to_string()
        })?;
        let metadata = FrameMetadata {
            // A cache may retag an exact decoded identity for a newer request
            // generation, but it must not launder a mismatched/forged token
            // into an upload predecessor first.
            codec_identity: frame.metadata.codec_identity.filter(|identity| {
                identity.source_generation == frame.metadata.source_generation
                    && Some(identity.pts) == frame.metadata.pts
            }),
            ..frame.metadata
        };
        self.frames.push_back(CachedFrame {
            rgba,
            metadata,
            _lease: lease,
        });
        Ok(())
    }

    /// Return the nearest cached frame at or before the requested source time.
    pub fn near_at_or_before(
        &self,
        target_seconds: f64,
        source_generation: u64,
        max_age_seconds: f64,
    ) -> Result<Option<DecodedVideoFrame>, String> {
        let target_seconds = finite_nonnegative(target_seconds);
        let max_age_seconds = finite_nonnegative(max_age_seconds);
        let Some(frame) = self.frames.iter().rev().find(|frame| {
            frame.metadata.source_seconds <= target_seconds + f64::EPSILON
                && target_seconds - frame.metadata.source_seconds <= max_age_seconds
        }) else {
            return Ok(None);
        };
        let mut rgba = Vec::new();
        rgba.try_reserve_exact(frame.rgba.len()).map_err(|error| {
            format!(
                "could not reserve {} bytes for reverse-frame cache retrieval: {error}",
                frame.rgba.len()
            )
        })?;
        rgba.extend_from_slice(&frame.rgba);
        Ok(Some(DecodedVideoFrame {
            rgba,
            metadata: FrameMetadata {
                source_generation,
                codec_identity: frame.metadata.codec_identity.map(|mut identity| {
                    identity.source_generation = source_generation;
                    identity
                }),
                ..frame.metadata
            },
            codec_motion: None,
        }))
    }

    pub fn clear(&mut self) {
        self.frames.clear();
        self.bytes = 0;
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

pub(crate) fn timestamp_to_source_seconds(pts: i64, start_pts: i64, time_base_seconds: f64) -> f64 {
    let relative = pts.saturating_sub(start_pts) as f64 * time_base_seconds;
    finite_nonnegative(relative)
}

pub(crate) fn finite_nonnegative(value: f64) -> f64 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(id: u8, seconds: f64, bytes: usize) -> DecodedVideoFrame {
        DecodedVideoFrame {
            rgba: vec![id; bytes],
            metadata: FrameMetadata::sanitized(7, Some(i64::from(id)), seconds, 10.0),
            codec_motion: None,
        }
    }

    fn frame_with_motion(id: u8, seconds: f64, bytes: usize) -> DecodedVideoFrame {
        let mut frame = frame(id, seconds, bytes);
        frame.metadata.codec_identity = Some(CodecFrameIdentity {
            source_generation: frame.metadata.source_generation,
            pts: i64::from(id),
            presentation_ordinal: u64::from(id),
        });
        frame.codec_motion = Some(
            crate::video::CodecMotionFrame {
                source_dimensions: [64, 64],
                frame_delta_seconds: 1.0 / 30.0,
                source_generation: 7,
                frame_ordinal: u64::from(id),
                algorithm_version: crate::motion::MOTION_ALGORITHM_VERSION,
                provenance: crate::video::CodecMotionProvenance::FfmpegExportMvs,
                frame_type: crate::video::CodecMotionFrameType::Predictive,
                status: crate::video::CodecMotionStatus::Available,
                past_reference_proof: None,
                vectors: vec![crate::motion::CodecMotionVector {
                    destination: [16, 16],
                    block: [8, 8],
                    motion: [-1, 0],
                    motion_scale: 1,
                    seconds_from_reference: 1.0 / 30.0,
                    reference: crate::motion::CodecReferenceDirection::Past,
                    visibility: 1.0,
                }],
            }
            .into(),
        );
        frame
    }

    #[test]
    fn reverse_cache_leases_only_rgba_and_never_retains_codec_motion() {
        let ledger = ReverseCacheLedger::new(16);
        let mut cache = ReverseFrameCache::with_ledger(16, 2, ledger.clone());
        cache
            .insert(&frame_with_motion(1, 1.0, 12))
            .expect("RGBA cache admission");
        assert_eq!(cache.bytes(), 12);
        assert_eq!(ledger.reserved_bytes(), 12);
        let retrieved = cache.near_at_or_before(1.0, 99, 0.1).unwrap().unwrap();
        assert_eq!(retrieved.metadata.source_generation, 99);
        assert_eq!(
            retrieved.metadata.codec_identity,
            Some(CodecFrameIdentity {
                source_generation: 99,
                pts: 1,
                presentation_ordinal: 1,
            })
        );
        assert!(retrieved.codec_motion.is_none());
        assert_eq!(cache.bytes(), 12, "motion bytes entered the RGBA lease");
    }

    #[test]
    fn reverse_cache_never_launders_a_hostile_upload_identity() {
        for hostile_identity in [
            CodecFrameIdentity {
                source_generation: 8,
                pts: 1,
                presentation_ordinal: 1,
            },
            CodecFrameIdentity {
                source_generation: 7,
                pts: 2,
                presentation_ordinal: 1,
            },
        ] {
            let mut hostile = frame(1, 1.0, 4);
            hostile.metadata.codec_identity = Some(hostile_identity);
            let mut cache = ReverseFrameCache::new(8, 1);
            cache.insert(&hostile).unwrap();
            let retrieved = cache.near_at_or_before(1.0, 99, 0.1).unwrap().unwrap();
            assert_eq!(retrieved.metadata.source_generation, 99);
            assert_eq!(retrieved.metadata.codec_identity, None);
        }
    }

    #[test]
    fn reverse_cache_enforces_byte_and_frame_bounds_without_overflow() {
        let mut cache = ReverseFrameCache::new(12, 2);
        cache.insert(&frame(1, 1.0, 6)).unwrap();
        cache.insert(&frame(2, 2.0, 6)).unwrap();
        cache.insert(&frame(3, 3.0, 6)).unwrap();
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.bytes(), 12);
        assert_eq!(
            cache.near_at_or_before(2.1, 9, 0.2).unwrap().unwrap().rgba[0],
            2
        );
        assert_eq!(
            cache
                .near_at_or_before(2.1, 9, 0.2)
                .unwrap()
                .unwrap()
                .metadata
                .source_generation,
            9
        );

        cache.insert(&frame(4, 4.0, 13)).unwrap();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.bytes(), 0);
    }

    #[test]
    fn six_caches_share_one_aggregate_budget_and_release_for_reuse() {
        let ledger = ReverseCacheLedger::new(128);
        let mut caches: Vec<_> = (0..6)
            .map(|_| ReverseFrameCache::with_ledger(32, 512, ledger.clone()))
            .collect();

        for (index, cache) in caches.iter_mut().enumerate() {
            cache
                .insert(&frame(index as u8 + 1, index as f64, 32))
                .unwrap();
        }

        assert_eq!(ledger.max_bytes(), 128);
        assert_eq!(ledger.reserved_bytes(), 128);
        assert!(caches[..4].iter().all(|cache| cache.len() == 1));
        assert!(caches[4..].iter().all(|cache| cache.len() == 0));

        caches[1].clear();
        assert_eq!(ledger.reserved_bytes(), 96);
        caches[4].insert(&frame(9, 9.0, 32)).unwrap();
        assert_eq!(caches[4].len(), 1);
        assert_eq!(ledger.reserved_bytes(), 128);

        drop(caches);
        assert_eq!(ledger.reserved_bytes(), 0);
    }

    #[test]
    fn cache_move_transfers_leases_and_eviction_releases_exact_bytes() {
        let ledger = ReverseCacheLedger::new(64);
        let mut cache = ReverseFrameCache::with_ledger(24, 2, ledger.clone());
        cache.insert(&frame(1, 1.0, 12)).unwrap();
        cache.insert(&frame(2, 2.0, 12)).unwrap();
        assert_eq!(ledger.reserved_bytes(), 24);

        cache.insert(&frame(3, 3.0, 12)).unwrap();
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.bytes(), 24);
        assert_eq!(ledger.reserved_bytes(), 24);

        let transferred = cache;
        assert_eq!(transferred.bytes(), 24);
        assert_eq!(ledger.reserved_bytes(), 24);
        drop(transferred);
        assert_eq!(ledger.reserved_bytes(), 0);
    }

    #[test]
    fn aggregate_ledger_rejects_checked_add_overflow_and_recovers_on_drop() {
        let ledger = ReverseCacheLedger::new(u64::MAX);
        let all = ledger.try_reserve(u64::MAX).unwrap();
        assert_eq!(ledger.reserved_bytes(), u64::MAX);
        assert_eq!(
            ledger.try_reserve(1).unwrap_err(),
            "reverse cache byte accounting overflow"
        );
        drop(all);
        assert_eq!(ledger.reserved_bytes(), 0);
        let one = ledger.try_reserve(1).unwrap();
        assert_eq!(ledger.reserved_bytes(), 1);
        drop(one);
        assert_eq!(ledger.reserved_bytes(), 0);
    }

    #[test]
    fn fallback_index_clamps_hostile_time_and_returns_preceding_zero() {
        let index = KeyframeIndex::fallback(42).unwrap();
        assert_eq!(index.preceding(f64::NAN).pts, 42);
        assert_eq!(index.preceding(-1.0).source_seconds, 0.0);
        assert!(index.len() <= MAX_KEYFRAME_INDEX_ENTRIES);
    }

    #[test]
    fn timestamp_math_saturates_before_float_conversion() {
        assert_eq!(timestamp_to_source_seconds(i64::MIN, i64::MAX, 1.0), 0.0);
        assert!(timestamp_to_source_seconds(i64::MAX, i64::MIN, 1.0).is_finite());
        assert_eq!(finite_nonnegative(f64::INFINITY), 0.0);
    }
}
