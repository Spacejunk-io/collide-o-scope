//! Bounded, non-blocking live capture and safe artifact publication.
//!
//! The render thread owns only fixed-capacity channels, atomics, and frame
//! leases allocated by the worker before it reports `Recording`. It never
//! waits for FFmpeg or filesystem I/O. A successful terminal event is the
//! sole authority for importing or publishing a new clip slot.

use std::collections::VecDeque;
use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use image::codecs::png::PngEncoder;
use image::{ExtendedColorType, ImageEncoder as _};
use serde::{Deserialize, Serialize};

use crate::image_routing::StableLayerId;
use crate::media_safety::SAFE_MEDIA_MAX_RGBA_BYTES;
use crate::visual_rack::GroupId;

pub const RECORDER_FRAME_POOL_CAPACITY: usize = 4;
pub const RECORDER_FRAME_QUEUE_CAPACITY: usize = 2;
pub const RECORDER_MAX_POOL_BYTES: u64 = 128 * 1024 * 1024;
pub const RECORDER_REPORT_SCHEMA_VERSION: u16 = 3;
pub const RECORDER_MAX_REPORT_BYTES: usize = 64 * 1024;

/// Seconds of device audio the bounded program-PCM ring retains before the
/// overflow law discards the oldest frames as an explicit, counted gap.
pub const RECORDER_AUDIO_RING_SECONDS: u32 = 4;
/// Absolute sample ceiling for the program-PCM ring (8 MiB of f32 samples),
/// intersected with the duration-derived capacity above.
pub const RECORDER_AUDIO_RING_MAX_SAMPLES: usize = 2_097_152;
/// Hard byte cap for the staged raw-PCM capture temp. Reaching it marks the
/// audio capture truncated; the remainder of the artifact is padded silence.
pub const RECORDER_MAX_AUDIO_TEMP_BYTES: u64 = 2 * 1024 * 1024 * 1024;
/// One bounded drift correction fires when the device clock has slipped this
/// far (as a fraction of one second) against the program capture cadence.
pub const RECORDER_AUDIO_DRIFT_THRESHOLD_SECONDS: f64 = 0.25;

const RECORDER_EVENT_QUEUE_CAPACITY: usize = 1;
const RECORDER_POLL_INTERVAL: Duration = Duration::from_millis(10);
const RECORDER_FINISH_TIMEOUT: Duration = Duration::from_secs(30);
/// Bounded wait at finish for the device to deliver the tail of the audio
/// timeline before the mux pads the exact remainder with silence.
const RECORDER_AUDIO_FINISH_GRACE: Duration = Duration::from_millis(500);
/// Silence is written in bounded chunks so one large gap cannot allocate a
/// proportional buffer.
const RECORDER_AUDIO_SILENCE_CHUNK_SAMPLES: usize = 65_536;
const RECORDER_WORKER_EXIT_GRACE: Duration = Duration::from_secs(2);
const MAX_DUPLICATION_GAP_FRAMES: u64 = 2_400;
const MAX_ERROR_CHARS: usize = 1_024;

const PHASE_STARTING: u8 = 0;
const PHASE_RECORDING: u8 = 1;
const PHASE_FINISHING: u8 = 2;
const PHASE_SUCCEEDED: u8 = 3;
const PHASE_FAILED: u8 = 4;
const PHASE_CANCELLED: u8 = 5;

const PUBLICATION_OPEN: u8 = 0;
const PUBLICATION_CANCELLED: u8 = 1;
const PUBLICATION_COMMITTING: u8 = 2;
const PUBLICATION_PUBLISHED: u8 = 3;

#[cfg(test)]
type PublicationTestHook = Arc<dyn Fn() + Send + Sync + 'static>;

/// One lock-free linearization point shared by Cancel and final publication.
///
/// Cancel wins only while publication is open. Once the worker claims the
/// commit, Cancel is intentionally a no-op: the worker already owns the
/// bounded no-replace transaction and must report its actual terminal result.
#[derive(Default)]
struct PublicationGate {
    state: AtomicU8,
    #[cfg(test)]
    before_claim_hook: std::sync::Mutex<Option<PublicationTestHook>>,
    #[cfg(test)]
    after_claim_hook: std::sync::Mutex<Option<PublicationTestHook>>,
}

impl PublicationGate {
    /// Nonblocking Cancel linearization. Repeated Cancel calls remain true
    /// after the first one wins, while a worker-owned commit is never revoked.
    fn request_cancel(&self) -> bool {
        match self.state.compare_exchange(
            PUBLICATION_OPEN,
            PUBLICATION_CANCELLED,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) | Err(PUBLICATION_CANCELLED) => true,
            Err(PUBLICATION_COMMITTING | PUBLICATION_PUBLISHED) => false,
            Err(_) => false,
        }
    }

    fn cancelled(&self) -> bool {
        self.state.load(Ordering::Acquire) == PUBLICATION_CANCELLED
    }

    fn claim(&self) -> Result<(), String> {
        self.run_before_claim_hook();
        self.state
            .compare_exchange(
                PUBLICATION_OPEN,
                PUBLICATION_COMMITTING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map_err(|state| match state {
                PUBLICATION_CANCELLED => "capture cancelled before final publication".to_string(),
                _ => "capture final publication was already claimed".to_string(),
            })?;
        self.run_after_claim_hook();
        Ok(())
    }

    fn mark_published(&self) {
        self.state.store(PUBLICATION_PUBLISHED, Ordering::Release);
    }

    #[cfg(test)]
    fn set_before_claim_hook(&self, hook: PublicationTestHook) {
        *self.before_claim_hook.lock().unwrap() = Some(hook);
    }

    #[cfg(test)]
    fn set_after_claim_hook(&self, hook: PublicationTestHook) {
        *self.after_claim_hook.lock().unwrap() = Some(hook);
    }

    #[cfg(test)]
    fn run_before_claim_hook(&self) {
        let hook = self.before_claim_hook.lock().unwrap().clone();
        if let Some(hook) = hook {
            hook();
        }
    }

    #[cfg(not(test))]
    fn run_before_claim_hook(&self) {}

    #[cfg(test)]
    fn run_after_claim_hook(&self) {
        let hook = self.after_claim_hook.lock().unwrap().clone();
        if let Some(hook) = hook {
            hook();
        }
    }

    #[cfg(not(test))]
    fn run_after_claim_hook(&self) {}
}

/// Exact rational capture cadence. NTSC rates retain their 1001 denominator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecorderFrameRate {
    pub numerator: u32,
    pub denominator: u32,
}

impl RecorderFrameRate {
    #[allow(
        dead_code,
        reason = "supported native picker cadence retained by the recorder API"
    )]
    pub const FPS_24: Self = Self::new_unchecked(24, 1);
    pub const FPS_30: Self = Self::new_unchecked(30, 1);
    #[allow(
        dead_code,
        reason = "supported native picker cadence retained by the recorder API"
    )]
    pub const FPS_60: Self = Self::new_unchecked(60, 1);
    #[allow(
        dead_code,
        reason = "supported native picker cadence retained by the recorder API"
    )]
    pub const NTSC_30: Self = Self::new_unchecked(30_000, 1_001);
    #[allow(
        dead_code,
        reason = "supported native picker cadence retained by the recorder API"
    )]
    pub const NTSC_60: Self = Self::new_unchecked(60_000, 1_001);

    pub const fn new_unchecked(numerator: u32, denominator: u32) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    #[allow(
        dead_code,
        reason = "validated rational cadence constructor is a native adapter seam"
    )]
    pub fn new(numerator: u32, denominator: u32) -> Result<Self, String> {
        let rate = Self {
            numerator,
            denominator,
        };
        rate.validate()?;
        Ok(rate)
    }

    fn validate(self) -> Result<(), String> {
        if self.numerator == 0 || self.denominator == 0 {
            return Err("recording frame rate numerator and denominator must be non-zero".into());
        }
        let scaled = u64::from(self.numerator);
        let denominator = u64::from(self.denominator);
        if scaled < denominator || scaled > denominator.saturating_mul(240) {
            return Err(format!(
                "recording frame rate {}/{} is outside 1..=240 fps",
                self.numerator, self.denominator
            ));
        }
        Ok(())
    }

    fn ffmpeg_arg(self) -> String {
        format!("{}/{}", self.numerator, self.denominator)
    }
}

impl Default for RecorderFrameRate {
    fn default() -> Self {
        Self::FPS_30
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecorderDimensions {
    pub width: u32,
    pub height: u32,
}

impl RecorderDimensions {
    pub fn new(width: u32, height: u32) -> Result<Self, String> {
        let dimensions = Self { width, height };
        dimensions.frame_bytes()?;
        Ok(dimensions)
    }

    pub fn frame_bytes(self) -> Result<usize, String> {
        let bytes = u64::from(self.width)
            .checked_mul(u64::from(self.height))
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| "recording RGBA dimensions overflow".to_string())?;
        if bytes == 0 {
            return Err("recording dimensions must be non-zero".into());
        }
        if bytes > SAFE_MEDIA_MAX_RGBA_BYTES {
            return Err(format!(
                "recording frame needs {bytes} bytes; per-frame limit is {SAFE_MEDIA_MAX_RGBA_BYTES}"
            ));
        }
        let aggregate = bytes
            .checked_mul(RECORDER_FRAME_POOL_CAPACITY as u64)
            .ok_or_else(|| "recording frame-pool byte accounting overflow".to_string())?;
        if aggregate > RECORDER_MAX_POOL_BYTES {
            return Err(format!(
                "recording frame pool needs {aggregate} bytes; aggregate limit is {RECORDER_MAX_POOL_BYTES}"
            ));
        }
        usize::try_from(bytes).map_err(|_| "recording frame does not fit host usize".into())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureTarget {
    Program,
    Layer(StableLayerId),
    Group(GroupId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapturePurpose {
    External,
    AutoImport,
    Resample {
        destination_layer: StableLayerId,
        activate: bool,
    },
}

#[derive(Debug, Clone)]
pub struct RecorderConfig {
    pub dimensions: RecorderDimensions,
    pub frame_rate: RecorderFrameRate,
    pub output_path: PathBuf,
    pub target: CaptureTarget,
    pub purpose: CapturePurpose,
    /// Armed program-PCM source. `None` keeps the exact video-only path and
    /// the truthful `audio_not_muxed` report.
    pub audio_tap: Option<Arc<ProgramAudioTap>>,
}

impl RecorderConfig {
    fn validate(&self) -> Result<usize, String> {
        let frame_bytes = self.dimensions.frame_bytes()?;
        self.frame_rate.validate()?;
        validate_capture_pair_destinations(&self.output_path)?;
        Ok(frame_bytes)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioClockStamp {
    pub sample_position: u64,
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioClockStamp {
    fn validate(self) -> bool {
        (8_000..=384_000).contains(&self.sample_rate) && (1..=32).contains(&self.channels)
    }
}

struct AudioTapRing {
    /// Interleaved device samples, always a whole number of frames.
    samples: VecDeque<f32>,
    /// Absolute frame index of the oldest retained sample frame.
    head_frame: u64,
    /// Total frames the device has delivered since arm: the PCM clock.
    delivered_frames: u64,
    /// Single-reader cursor; `None` until the recorder anchors it.
    reader_frame: Option<u64>,
}

/// The recorder-owned Program PCM clock and bounded ring.
///
/// The audio callback thread pushes interleaved device samples through
/// [`ProgramAudioTap::push_interleaved`]; the recorder worker is the single
/// reader. `delivered_frames` advances only when the device actually delivers
/// samples — it is the clock every captured video frame is stamped with — and
/// a span the bounded ring had to discard on overflow is recovered by the
/// reader as an explicit, counted silence gap rather than a silent timeline
/// shift. Analysis audio is deliberately not this clock: the analyzer's FFT
/// ring keeps its own law and neither can starve the other.
pub struct ProgramAudioTap {
    ring: std::sync::Mutex<AudioTapRing>,
    sample_rate: u32,
    channels: u16,
    device_name: String,
    capacity_samples: usize,
    device_lost: AtomicBool,
}

impl std::fmt::Debug for ProgramAudioTap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProgramAudioTap")
            .field("sample_rate", &self.sample_rate)
            .field("channels", &self.channels)
            .field("device_name", &self.device_name)
            .field("capacity_samples", &self.capacity_samples)
            .field("device_lost", &self.device_lost.load(Ordering::Acquire))
            .finish()
    }
}

impl ProgramAudioTap {
    pub fn new(sample_rate: u32, channels: u16, device_name: String) -> Result<Arc<Self>, String> {
        let probe = AudioClockStamp {
            sample_position: 0,
            sample_rate,
            channels,
        };
        if !probe.validate() {
            return Err(format!(
                "program audio tap rate {sample_rate} Hz / {channels} channels is unsupported"
            ));
        }
        let duration_capacity = (sample_rate as usize)
            .saturating_mul(RECORDER_AUDIO_RING_SECONDS as usize)
            .saturating_mul(channels as usize);
        let mut capacity_samples = duration_capacity.min(RECORDER_AUDIO_RING_MAX_SAMPLES);
        // Whole frames only, and never a ring too small to hold one frame.
        capacity_samples -= capacity_samples % channels as usize;
        if capacity_samples < channels as usize {
            return Err("program audio tap ring capacity is below one frame".into());
        }
        Ok(Arc::new(Self {
            ring: std::sync::Mutex::new(AudioTapRing {
                samples: VecDeque::with_capacity(capacity_samples),
                head_frame: 0,
                delivered_frames: 0,
                reader_frame: None,
            }),
            sample_rate,
            channels,
            device_name,
            capacity_samples,
            device_lost: AtomicBool::new(false),
        }))
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> u16 {
        self.channels
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    fn lock_ring(&self) -> std::sync::MutexGuard<'_, AudioTapRing> {
        match self.ring.lock() {
            Ok(ring) => ring,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    /// Callback-side push of interleaved device samples. A trailing partial
    /// frame is discarded; a channel-layout mismatch marks the tap lost
    /// instead of writing frames on the wrong lattice.
    pub fn push_interleaved(&self, samples: &mut dyn Iterator<Item = f32>, channel_count: u16) {
        if channel_count != self.channels {
            self.mark_device_lost();
            return;
        }
        let channels = self.channels as usize;
        let mut ring = self.lock_ring();
        let mut frame = [0.0f32; 32];
        loop {
            let mut filled = 0;
            for slot in frame.iter_mut().take(channels) {
                match samples.next() {
                    Some(sample) => {
                        *slot = if sample.is_finite() {
                            sample.clamp(-4.0, 4.0)
                        } else {
                            0.0
                        };
                        filled += 1;
                    }
                    None => break,
                }
            }
            if filled < channels {
                break;
            }
            while ring.samples.len() + channels > self.capacity_samples {
                for _ in 0..channels {
                    ring.samples.pop_front();
                }
                ring.head_frame += 1;
            }
            ring.samples.extend(frame.iter().copied().take(channels));
            ring.delivered_frames += 1;
        }
    }

    pub fn mark_device_lost(&self) {
        self.device_lost.store(true, Ordering::Release);
    }

    pub fn device_lost(&self) -> bool {
        self.device_lost.load(Ordering::Acquire)
    }

    /// The Program PCM clock: what the device has delivered so far.
    pub fn clock_stamp(&self) -> AudioClockStamp {
        let ring = self.lock_ring();
        AudioClockStamp {
            sample_position: ring.delivered_frames,
            sample_rate: self.sample_rate,
            channels: self.channels,
        }
    }

    fn delivered_frames(&self) -> u64 {
        self.lock_ring().delivered_frames
    }

    /// Anchor the single reader at an absolute frame index, discarding any
    /// retained audio recorded before the first accepted video frame.
    fn begin_read_at(&self, anchor_frame: u64) {
        let channels = self.channels as usize;
        let mut ring = self.lock_ring();
        while ring.head_frame < anchor_frame && ring.samples.len() >= channels {
            for _ in 0..channels {
                ring.samples.pop_front();
            }
            ring.head_frame += 1;
        }
        if ring.head_frame < anchor_frame && ring.samples.is_empty() {
            // Nothing delivered past the anchor yet; the cursor still starts
            // there so the first delivered frame is not misread as a gap.
            ring.head_frame = ring.head_frame.max(ring.delivered_frames);
        }
        ring.reader_frame = Some(anchor_frame);
    }

    /// Drain everything available. The returned count is the exact number of
    /// frames the bounded ring discarded before the reader arrived — the
    /// underrun law's explicit silence gap.
    fn take_available(&self, out: &mut Vec<f32>) -> u64 {
        let channels = self.channels as usize;
        let mut ring = self.lock_ring();
        let Some(reader) = ring.reader_frame else {
            return 0;
        };
        let gap_frames = ring.head_frame.saturating_sub(reader);
        let popped_frames = (ring.samples.len() / channels) as u64;
        out.extend(ring.samples.drain(..));
        ring.head_frame += popped_frames;
        ring.reader_frame = Some(ring.head_frame);
        gap_frames
    }
}

/// Capture-time identity retained even though this foundation intentionally
/// does not claim to mux audio without a bounded program-PCM source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecorderFrameMetadata {
    pub capture_index: u64,
    pub capture_time_ns: u64,
    pub program_time_ns: u64,
    pub visual_epoch: u64,
    pub program_frozen: bool,
    pub media_frozen: bool,
    pub blackout: bool,
    pub audio_clock: Option<AudioClockStamp>,
}

impl RecorderFrameMetadata {
    fn validate(self) -> Result<(), String> {
        if self.audio_clock.is_some_and(|clock| !clock.validate()) {
            return Err("recording audio clock has unsupported rate or channel count".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecorderCounters {
    pub attempted: u64,
    pub accepted: u64,
    pub encoded: u64,
    pub duplicated: u64,
    pub dropped_not_ready: u64,
    pub dropped_source_unavailable: u64,
    pub dropped_pool_empty: u64,
    pub dropped_queue_full: u64,
    pub rejected_metadata: u64,
    pub encoder_failures: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecorderStatus {
    Starting,
    Recording,
    Finishing,
    Succeeded,
    Failed,
    Cancelled,
}

impl RecorderStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Starting => "starting",
            Self::Recording => "recording",
            Self::Finishing => "finishing",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecorderSnapshot {
    pub status: RecorderStatus,
    pub counters: RecorderCounters,
    pub error: String,
}

#[derive(Default)]
struct RecorderAtomicCounters {
    attempted: AtomicU64,
    accepted: AtomicU64,
    encoded: AtomicU64,
    duplicated: AtomicU64,
    dropped_not_ready: AtomicU64,
    dropped_source_unavailable: AtomicU64,
    dropped_pool_empty: AtomicU64,
    dropped_queue_full: AtomicU64,
    rejected_metadata: AtomicU64,
    encoder_failures: AtomicU64,
}

impl RecorderAtomicCounters {
    fn snapshot(&self) -> RecorderCounters {
        RecorderCounters {
            attempted: self.attempted.load(Ordering::Acquire),
            accepted: self.accepted.load(Ordering::Acquire),
            encoded: self.encoded.load(Ordering::Acquire),
            duplicated: self.duplicated.load(Ordering::Acquire),
            dropped_not_ready: self.dropped_not_ready.load(Ordering::Acquire),
            dropped_source_unavailable: self.dropped_source_unavailable.load(Ordering::Acquire),
            dropped_pool_empty: self.dropped_pool_empty.load(Ordering::Acquire),
            dropped_queue_full: self.dropped_queue_full.load(Ordering::Acquire),
            rejected_metadata: self.rejected_metadata.load(Ordering::Acquire),
            encoder_failures: self.encoder_failures.load(Ordering::Acquire),
        }
    }
}

struct RecorderShared {
    phase: AtomicU8,
    cancel: AtomicBool,
    publication: PublicationGate,
    finish_requested: AtomicBool,
    finish_capture_index: AtomicU64,
    counters: RecorderAtomicCounters,
    error: std::sync::Mutex<String>,
}

impl RecorderShared {
    fn new() -> Self {
        Self {
            phase: AtomicU8::new(PHASE_STARTING),
            cancel: AtomicBool::new(false),
            publication: PublicationGate::default(),
            finish_requested: AtomicBool::new(false),
            finish_capture_index: AtomicU64::new(0),
            counters: RecorderAtomicCounters::default(),
            error: std::sync::Mutex::new(String::new()),
        }
    }

    fn status(&self) -> RecorderStatus {
        match self.phase.load(Ordering::Acquire) {
            PHASE_RECORDING => RecorderStatus::Recording,
            PHASE_FINISHING => RecorderStatus::Finishing,
            PHASE_SUCCEEDED => RecorderStatus::Succeeded,
            PHASE_FAILED => RecorderStatus::Failed,
            PHASE_CANCELLED => RecorderStatus::Cancelled,
            _ => RecorderStatus::Starting,
        }
    }

    fn set_error(&self, error: impl AsRef<str>) {
        let error = bounded_text(error.as_ref(), MAX_ERROR_CHARS);
        match self.error.lock() {
            Ok(mut slot) => *slot = error,
            Err(poisoned) => *poisoned.into_inner() = error,
        }
    }

    fn error(&self) -> String {
        match self.error.try_lock() {
            Ok(error) => error.clone(),
            Err(std::sync::TryLockError::Poisoned(poisoned)) => poisoned.into_inner().clone(),
            // Snapshot publication is allowed to miss one in-flight error
            // update; the terminal event carries the same bounded detail.
            Err(std::sync::TryLockError::WouldBlock) => String::new(),
        }
    }
}

pub enum RecorderAcquire {
    Lease(RecorderFrameLease),
    NotReady,
    DroppedPoolEmpty,
}

pub struct RecorderFrameLease {
    pixels: Option<Vec<u8>>,
    return_tx: SyncSender<Vec<u8>>,
}

impl RecorderFrameLease {
    pub fn pixels_mut(&mut self) -> &mut [u8] {
        self.pixels
            .as_mut()
            .expect("live recorder lease always owns its buffer")
    }

    #[allow(
        dead_code,
        reason = "bounded copy adapter is retained for non-mapped capture sources"
    )]
    pub fn copy_from_slice(&mut self, pixels: &[u8]) -> Result<(), String> {
        if pixels.len() != self.pixels_mut().len() {
            return Err(format!(
                "recording source has {} bytes; expected {}",
                pixels.len(),
                self.pixels_mut().len()
            ));
        }
        self.pixels_mut().copy_from_slice(pixels);
        Ok(())
    }

    fn take_pixels(&mut self) -> Vec<u8> {
        self.pixels
            .take()
            .expect("submitted recorder lease owns its buffer")
    }
}

impl Drop for RecorderFrameLease {
    fn drop(&mut self) {
        if let Some(pixels) = self.pixels.take() {
            let _ = self.return_tx.try_send(pixels);
        }
    }
}

struct RecorderWorkItem {
    pixels: Vec<u8>,
    metadata: RecorderFrameMetadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecorderSubmit {
    Accepted,
    DroppedQueueFull,
    Rejected,
}

#[derive(Debug, Clone)]
#[allow(
    dead_code,
    reason = "durable terminal report retains complete artifact provenance"
)]
pub struct CommittedCapture {
    pub media_path: PathBuf,
    pub report_path: PathBuf,
    pub target: CaptureTarget,
    pub purpose: CapturePurpose,
    pub dimensions: RecorderDimensions,
    pub frame_rate: RecorderFrameRate,
    pub counters: RecorderCounters,
    /// True when the committed artifact carries a muxed AAC program-audio
    /// stream; false is the exact video-only publication.
    pub audio_muxed: bool,
}

/// Post-commit instruction for Main. It deliberately contains no ClipSlotId:
/// a new stable slot is allocated only after the file is durably visible and
/// then enters the existing prepared-source transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommittedCaptureIntent {
    None,
    AutoImport {
        committed_path: PathBuf,
    },
    NewClipSlot {
        committed_path: PathBuf,
        destination_layer: StableLayerId,
        activate: bool,
    },
}

impl CommittedCapture {
    pub fn publication_intent(&self) -> CommittedCaptureIntent {
        match self.purpose {
            CapturePurpose::External => CommittedCaptureIntent::None,
            CapturePurpose::AutoImport => CommittedCaptureIntent::AutoImport {
                committed_path: self.media_path.clone(),
            },
            CapturePurpose::Resample {
                destination_layer,
                activate,
            } => CommittedCaptureIntent::NewClipSlot {
                committed_path: self.media_path.clone(),
                destination_layer,
                activate,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum RecorderTerminalEvent {
    Succeeded(CommittedCapture),
    Failed(String),
    Cancelled,
}

pub struct ProgramRecorder {
    #[allow(dead_code, reason = "retained for native recorder inspection adapters")]
    config: RecorderConfig,
    #[allow(dead_code, reason = "retained for native recorder inspection adapters")]
    frame_bytes: usize,
    pool_rx: Receiver<Vec<u8>>,
    pool_tx: SyncSender<Vec<u8>>,
    work_tx: SyncSender<RecorderWorkItem>,
    event_rx: Receiver<RecorderTerminalEvent>,
    shared: Arc<RecorderShared>,
    worker: Option<JoinHandle<()>>,
}

impl ProgramRecorder {
    pub fn spawn(config: RecorderConfig) -> Result<Self, String> {
        Self::spawn_with_hooks(
            config,
            Box::new(spawn_ffmpeg_sink),
            Box::new(run_capture_mux),
        )
    }

    #[cfg(test)]
    fn spawn_with_factory(config: RecorderConfig, factory: SinkFactory) -> Result<Self, String> {
        Self::spawn_with_hooks(config, factory, Box::new(run_capture_mux))
    }

    fn spawn_with_hooks(
        config: RecorderConfig,
        factory: SinkFactory,
        mux_runner: MuxRunner,
    ) -> Result<Self, String> {
        let frame_bytes = config.validate()?;
        let (pool_tx, pool_rx) = mpsc::sync_channel(RECORDER_FRAME_POOL_CAPACITY);
        let (work_tx, work_rx) = mpsc::sync_channel(RECORDER_FRAME_QUEUE_CAPACITY);
        let (event_tx, event_rx) = mpsc::sync_channel(RECORDER_EVENT_QUEUE_CAPACITY);
        let shared = Arc::new(RecorderShared::new());
        let worker_shared = shared.clone();
        let worker_pool = pool_tx.clone();
        let worker_config = config.clone();
        let worker = std::thread::Builder::new()
            .name("program-recorder".into())
            .spawn(move || {
                recorder_worker(
                    worker_config,
                    frame_bytes,
                    worker_pool,
                    work_rx,
                    event_tx,
                    worker_shared,
                    factory,
                    mux_runner,
                );
            })
            .map_err(|error| format!("start recorder worker: {error}"))?;
        Ok(Self {
            config,
            frame_bytes,
            pool_rx,
            pool_tx,
            work_tx,
            event_rx,
            shared,
            worker: Some(worker),
        })
    }

    #[allow(dead_code, reason = "retained for native recorder inspection adapters")]
    pub fn dimensions(&self) -> RecorderDimensions {
        self.config.dimensions
    }

    #[allow(dead_code, reason = "retained for native recorder inspection adapters")]
    pub fn frame_bytes(&self) -> usize {
        self.frame_bytes
    }

    /// Nonblocking acquisition from the preallocated pool.
    pub fn try_acquire_frame(&self) -> RecorderAcquire {
        self.shared
            .counters
            .attempted
            .fetch_add(1, Ordering::Relaxed);
        if self.shared.phase.load(Ordering::Acquire) != PHASE_RECORDING {
            self.shared
                .counters
                .dropped_not_ready
                .fetch_add(1, Ordering::Relaxed);
            return RecorderAcquire::NotReady;
        }
        match self.pool_rx.try_recv() {
            Ok(pixels) => RecorderAcquire::Lease(RecorderFrameLease {
                pixels: Some(pixels),
                return_tx: self.pool_tx.clone(),
            }),
            Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => {
                self.shared
                    .counters
                    .dropped_pool_empty
                    .fetch_add(1, Ordering::Relaxed);
                RecorderAcquire::DroppedPoolEmpty
            }
        }
    }

    /// Account one accepted program-cadence frame whose exact source pixels
    /// could not be harvested (no GPU readback slot, failed map, or vanished
    /// stable layer/group target). Main still advances its capture index, so a
    /// later admitted frame creates an explicit CFR duplicate gap.
    pub fn note_source_frame_dropped(&self) {
        self.shared
            .counters
            .attempted
            .fetch_add(1, Ordering::Relaxed);
        if self.shared.phase.load(Ordering::Acquire) == PHASE_RECORDING {
            self.shared
                .counters
                .dropped_source_unavailable
                .fetch_add(1, Ordering::Relaxed);
        } else {
            self.shared
                .counters
                .dropped_not_ready
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Drop-newest submission. A full worker queue never delays presentation.
    pub fn try_submit(
        &self,
        mut lease: RecorderFrameLease,
        metadata: RecorderFrameMetadata,
    ) -> RecorderSubmit {
        if metadata.validate().is_err()
            || self.shared.phase.load(Ordering::Acquire) != PHASE_RECORDING
        {
            self.shared
                .counters
                .rejected_metadata
                .fetch_add(1, Ordering::Relaxed);
            return RecorderSubmit::Rejected;
        }
        let item = RecorderWorkItem {
            pixels: lease.take_pixels(),
            metadata,
        };
        match self.work_tx.try_send(item) {
            Ok(()) => {
                self.shared
                    .counters
                    .accepted
                    .fetch_add(1, Ordering::Relaxed);
                RecorderSubmit::Accepted
            }
            Err(TrySendError::Full(item)) => {
                let _ = self.pool_tx.try_send(item.pixels);
                self.shared
                    .counters
                    .dropped_queue_full
                    .fetch_add(1, Ordering::Relaxed);
                RecorderSubmit::DroppedQueueFull
            }
            Err(TrySendError::Disconnected(item)) => {
                let _ = self.pool_tx.try_send(item.pixels);
                self.shared
                    .counters
                    .rejected_metadata
                    .fetch_add(1, Ordering::Relaxed);
                RecorderSubmit::Rejected
            }
        }
    }

    /// Finish after the requested capture index. Missing accepted frames are
    /// represented by CFR duplicates of the newest successfully encoded frame.
    pub fn request_finish(&self, final_capture_index: u64) {
        self.shared
            .finish_capture_index
            .store(final_capture_index, Ordering::Release);
        self.shared.finish_requested.store(true, Ordering::Release);
        let _ = self.shared.phase.compare_exchange(
            PHASE_RECORDING,
            PHASE_FINISHING,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    pub fn cancel(&self) {
        if self.shared.publication.request_cancel() {
            self.shared.cancel.store(true, Ordering::Release);
        }
    }

    pub fn snapshot(&self) -> RecorderSnapshot {
        RecorderSnapshot {
            status: self.shared.status(),
            counters: self.shared.counters.snapshot(),
            error: self.shared.error(),
        }
    }

    pub fn poll_terminal(&mut self) -> Option<RecorderTerminalEvent> {
        match self.event_rx.try_recv() {
            Ok(event) => {
                if self.worker.as_ref().is_some_and(JoinHandle::is_finished) {
                    if let Some(worker) = self.worker.take() {
                        let _ = worker.join();
                    }
                }
                Some(event)
            }
            Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => None,
        }
    }
}

impl Drop for ProgramRecorder {
    fn drop(&mut self) {
        // Never join from the render thread. The worker and process supervisor
        // share the cancel flag and retain sole ownership of temporary files.
        self.cancel();
        self.worker.take();
    }
}

type SinkFactory = Box<
    dyn FnOnce(
            &RecorderConfig,
            &Path,
            Arc<RecorderShared>,
        ) -> Result<Box<dyn RecorderVideoSink>, String>
        + Send,
>;

trait RecorderVideoSink: Send {
    fn write_frame(&mut self, pixels: &[u8]) -> Result<(), String>;
    fn finish(self: Box<Self>) -> Result<(), String>;
}

struct FfmpegVideoSink {
    stdin: Option<ChildStdin>,
    finish_requested: Arc<AtomicBool>,
    completion: Receiver<Result<(), String>>,
    supervisor: Option<JoinHandle<()>>,
    shared: Arc<RecorderShared>,
}

impl Drop for FfmpegVideoSink {
    fn drop(&mut self) {
        // Any abnormal worker exit still closes stdin and starts the bounded
        // supervisor deadline. This prevents a helper that ignores EOF from
        // surviving after the sink/worker has gone away.
        self.stdin.take();
        self.finish_requested.store(true, Ordering::Release);
    }
}

impl RecorderVideoSink for FfmpegVideoSink {
    fn write_frame(&mut self, pixels: &[u8]) -> Result<(), String> {
        self.stdin
            .as_mut()
            .ok_or_else(|| "recording encoder stdin is closed".to_string())?
            .write_all(pixels)
            .map_err(|error| format!("write recording frame to encoder: {error}"))
    }

    fn finish(mut self: Box<Self>) -> Result<(), String> {
        self.stdin.take();
        self.finish_requested.store(true, Ordering::Release);
        let result = match self
            .completion
            .recv_timeout(RECORDER_FINISH_TIMEOUT + RECORDER_WORKER_EXIT_GRACE)
        {
            Ok(result) => result,
            Err(_) => {
                self.shared.cancel.store(true, Ordering::Release);
                let _ = self.completion.recv_timeout(RECORDER_WORKER_EXIT_GRACE);
                Err("recording encoder supervisor did not reap within its bound".to_string())
            }
        };
        if let Some(supervisor) = self.supervisor.take() {
            // Receiving completion proves the child has already been reaped.
            // Join only when the tiny supervisor closure has visibly exited;
            // a scheduler-stalled helper thread must not turn the bounded
            // worker timeout into an unbounded join.
            if supervisor.is_finished() {
                let _ = supervisor.join();
            }
        }
        result
    }
}

fn spawn_ffmpeg_sink(
    config: &RecorderConfig,
    temp_path: &Path,
    shared: Arc<RecorderShared>,
) -> Result<Box<dyn RecorderVideoSink>, String> {
    let stderr_path = sibling_temp_path(temp_path, "encoder-log")?;
    let stderr_file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&stderr_path)
        .map_err(|error| format!("reserve recording encoder log: {error}"))?;
    let size = format!("{}x{}", config.dimensions.width, config.dimensions.height);
    let child = match Command::new(crate::host_paths::ffmpeg())
        .arg("-nostdin")
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-y")
        .arg("-f")
        .arg("rawvideo")
        .arg("-pixel_format")
        .arg("rgba")
        .arg("-video_size")
        .arg(size)
        .arg("-framerate")
        .arg(config.frame_rate.ffmpeg_arg())
        .arg("-i")
        .arg("pipe:0")
        .arg("-an")
        .arg("-c:v")
        .arg("libx264")
        .arg("-preset")
        .arg("veryfast")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg("-movflags")
        .arg("+faststart")
        .arg("-f")
        .arg("mp4")
        .arg(temp_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::from(stderr_file))
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            let _ = std::fs::remove_file(&stderr_path);
            return Err(format!("start recording encoder: {error}"));
        }
    };
    let (stdin, finish_requested, completion, supervisor) =
        supervise_encoder_process(child, stderr_path, shared.clone(), RECORDER_FINISH_TIMEOUT)?;
    Ok(Box::new(FfmpegVideoSink {
        stdin: Some(stdin),
        finish_requested,
        completion,
        supervisor: Some(supervisor),
        shared,
    }))
}

type SupervisedEncoderProcess = (
    ChildStdin,
    Arc<AtomicBool>,
    Receiver<Result<(), String>>,
    JoinHandle<()>,
);

fn supervise_encoder_process(
    mut child: Child,
    stderr_path: PathBuf,
    shared: Arc<RecorderShared>,
    finish_timeout: Duration,
) -> Result<SupervisedEncoderProcess, String> {
    let stdin = match child.stdin.take() {
        Some(stdin) => stdin,
        None => {
            let _ = child.kill();
            let _ = child.wait();
            let _ = std::fs::remove_file(&stderr_path);
            return Err("recording encoder did not expose stdin".to_string());
        }
    };
    let child = Arc::new(std::sync::Mutex::new(child));
    let supervisor_child = child.clone();
    let finish_requested = Arc::new(AtomicBool::new(false));
    let supervisor_finish = finish_requested.clone();
    let (completion_tx, completion) = mpsc::sync_channel(1);
    let supervisor_shared = shared.clone();
    let supervisor_stderr_path = stderr_path.clone();
    let supervisor = std::thread::Builder::new()
        .name("program-recorder-ffmpeg".into())
        .spawn(move || {
            let mut finish_started = None;
            let result = loop {
                let mut child = match supervisor_child.lock() {
                    Ok(child) => child,
                    Err(poisoned) => poisoned.into_inner(),
                };
                if supervisor_shared.cancel.load(Ordering::Acquire) {
                    let _ = child.kill();
                    let _ = child.wait();
                    break Err("recording cancelled".to_string());
                }
                match child.try_wait() {
                    Ok(Some(status)) if status.success() => break Ok(()),
                    Ok(Some(status)) => {
                        let detail = read_bounded_file(&supervisor_stderr_path, MAX_ERROR_CHARS)
                            .unwrap_or_else(|_| String::new());
                        break Err(if detail.is_empty() {
                            format!("recording encoder exited with {status}")
                        } else {
                            format!("recording encoder exited with {status}: {detail}")
                        });
                    }
                    Ok(None) => {}
                    Err(error) => break Err(format!("poll recording encoder: {error}")),
                }
                if supervisor_finish.load(Ordering::Acquire) {
                    let started = *finish_started.get_or_insert_with(Instant::now);
                    if started.elapsed() >= finish_timeout {
                        let _ = child.kill();
                        let _ = child.wait();
                        break Err("recording encoder finish timed out and was reaped".into());
                    }
                }
                drop(child);
                std::thread::sleep(RECORDER_POLL_INTERVAL);
            };
            let _ = std::fs::remove_file(&supervisor_stderr_path);
            let _ = completion_tx.try_send(result);
        })
        .map_err(|error| {
            let mut child = match child.lock() {
                Ok(child) => child,
                Err(poisoned) => poisoned.into_inner(),
            };
            let _ = child.kill();
            let _ = child.wait();
            let _ = std::fs::remove_file(&stderr_path);
            format!("start recording encoder supervisor: {error}")
        })?;
    Ok((stdin, finish_requested, completion, supervisor))
}

/// Everything the finish-time audio mux needs, resolved before any process
/// is spawned. The fixture mux runner asserts against this exact plan.
#[derive(Debug, Clone, PartialEq)]
pub struct CaptureMuxPlan {
    pub video_temp: PathBuf,
    pub audio_temp: PathBuf,
    pub output_temp: PathBuf,
    pub sample_rate: u32,
    pub channels: u16,
    /// Exact rational CFR duration of the encoded video, in seconds.
    pub duration_secs: f64,
}

/// The finish-time mux invocation: video stream copied, raw program PCM
/// encoded to AAC, padded and trimmed to the exact CFR video duration —
/// the offline exporter's own audio law.
fn capture_mux_args(plan: &CaptureMuxPlan) -> Vec<OsString> {
    let duration = format!("{:.6}", plan.duration_secs);
    let mut args: Vec<OsString> = ["-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i"]
        .into_iter()
        .map(OsString::from)
        .collect();
    args.push(plan.video_temp.as_os_str().to_owned());
    args.extend(
        [
            "-f".to_owned(),
            "f32le".to_owned(),
            "-ar".to_owned(),
            plan.sample_rate.to_string(),
            "-ac".to_owned(),
            plan.channels.to_string(),
            "-i".to_owned(),
        ]
        .into_iter()
        .map(OsString::from),
    );
    args.push(plan.audio_temp.as_os_str().to_owned());
    args.extend(
        [
            "-map".to_owned(),
            "0:v:0".to_owned(),
            "-map".to_owned(),
            "1:a:0".to_owned(),
            "-c:v".to_owned(),
            "copy".to_owned(),
            "-filter:a".to_owned(),
            format!("asetpts=PTS-STARTPTS,apad,atrim=end={duration}"),
            "-c:a".to_owned(),
            "aac".to_owned(),
            "-b:a".to_owned(),
            "192k".to_owned(),
            "-map_metadata".to_owned(),
            "-1".to_owned(),
            "-t".to_owned(),
            duration,
            "-movflags".to_owned(),
            "+faststart".to_owned(),
            "-f".to_owned(),
            "mp4".to_owned(),
        ]
        .into_iter()
        .map(OsString::from),
    );
    args.push(plan.output_temp.as_os_str().to_owned());
    args
}

type MuxRunner =
    Box<dyn FnOnce(&CaptureMuxPlan, &Arc<RecorderShared>) -> Result<(), String> + Send>;

/// Spawn the bounded finish-time mux helper and babysit it: cancel kills it,
/// the absolute finish deadline kills it, and a failure surfaces the bounded
/// captured stderr. There is no stdin; the inputs are complete staged temps.
fn run_capture_mux(plan: &CaptureMuxPlan, shared: &Arc<RecorderShared>) -> Result<(), String> {
    let stderr_path = sibling_temp_path(&plan.output_temp, "mux-log")?;
    let stderr_file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&stderr_path)
        .map_err(|error| format!("reserve recording mux log: {error}"))?;
    let mut child = match Command::new(crate::host_paths::ffmpeg())
        .args(capture_mux_args(plan))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(stderr_file))
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            let _ = std::fs::remove_file(&stderr_path);
            return Err(format!("start recording audio mux: {error}"));
        }
    };
    let started = Instant::now();
    let result = loop {
        if shared.cancel.load(Ordering::Acquire) {
            let _ = child.kill();
            let _ = child.wait();
            break Err("recording cancelled".to_string());
        }
        match child.try_wait() {
            Ok(Some(status)) if status.success() => break Ok(()),
            Ok(Some(status)) => {
                let detail = read_bounded_file(&stderr_path, MAX_ERROR_CHARS).unwrap_or_default();
                break Err(if detail.is_empty() {
                    format!("recording audio mux exited with {status}")
                } else {
                    format!("recording audio mux exited with {status}: {detail}")
                });
            }
            Ok(None) => {}
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                break Err(format!("poll recording audio mux: {error}"));
            }
        }
        if started.elapsed() >= RECORDER_FINISH_TIMEOUT {
            let _ = child.kill();
            let _ = child.wait();
            break Err("recording audio mux timed out and was reaped".to_string());
        }
        std::thread::sleep(RECORDER_POLL_INTERVAL);
    };
    let _ = std::fs::remove_file(&stderr_path);
    result
}

/// Per-artifact audio truth, serialized into the durable recording report.
#[derive(Debug, Clone, Serialize)]
struct RecorderAudioReport {
    device: String,
    sample_rate: u32,
    channels: u16,
    /// Program-PCM clock position anchored to the first accepted video frame.
    anchor_sample_position: u64,
    /// Real device frames written to the artifact's audio timeline.
    captured_frames: u64,
    /// Explicit silence written for ring-overflow gaps and for an armed
    /// source that never delivered.
    silence_gap_frames: u64,
    /// Bounded drift-correction insertions (device clock slow).
    drift_inserted_frames: u64,
    /// Bounded drift-correction drops (device clock fast).
    drift_dropped_frames: u64,
    device_lost: bool,
    capture_truncated: bool,
    muxed_duration_secs: f64,
}

/// Worker-side program-PCM capture: drains the tap ring into a staged raw
/// f32le temp, fills every discarded span with explicit counted silence, and
/// applies the bounded drift-correction law against the frame clock stamps.
struct RecorderAudioCapture {
    tap: Arc<ProgramAudioTap>,
    temp_path: PathBuf,
    mux_temp_path: PathBuf,
    writer: Option<BufWriter<File>>,
    anchor: Option<(u64, u64)>,
    captured_frames: u64,
    silence_gap_frames: u64,
    drift_inserted_frames: u64,
    drift_dropped_frames: u64,
    pending_drop_frames: u64,
    bytes_written: u64,
    truncated: bool,
    device_lost: bool,
    scratch: Vec<f32>,
}

impl RecorderAudioCapture {
    fn open(
        tap: Arc<ProgramAudioTap>,
        temp_path: PathBuf,
        mux_temp_path: PathBuf,
    ) -> Result<Self, String> {
        let file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&temp_path)
            .map_err(|error| format!("open recording audio temp: {error}"))?;
        Ok(Self {
            tap,
            temp_path,
            mux_temp_path,
            writer: Some(BufWriter::new(file)),
            anchor: None,
            captured_frames: 0,
            silence_gap_frames: 0,
            drift_inserted_frames: 0,
            drift_dropped_frames: 0,
            pending_drop_frames: 0,
            bytes_written: 0,
            truncated: false,
            device_lost: false,
            scratch: Vec::new(),
        })
    }

    fn total_file_frames(&self) -> u64 {
        self.captured_frames + self.silence_gap_frames + self.drift_inserted_frames
    }

    /// Anchor audio file position zero to the first accepted video frame's
    /// clock stamp. Audio armed without a stamp is a contract violation, not
    /// a guessable alignment.
    fn anchor_at(&mut self, metadata: RecorderFrameMetadata) -> Result<(), String> {
        if self.anchor.is_some() {
            return Ok(());
        }
        let Some(stamp) = metadata.audio_clock else {
            return Err(
                "recording audio is armed but the accepted frame carries no audio clock"
                    .to_string(),
            );
        };
        if stamp.sample_rate != self.tap.sample_rate() || stamp.channels != self.tap.channels() {
            return Err(format!(
                "recording audio clock {} Hz / {} ch does not match the armed tap ({} Hz / {} ch)",
                stamp.sample_rate,
                stamp.channels,
                self.tap.sample_rate(),
                self.tap.channels()
            ));
        }
        self.tap.begin_read_at(stamp.sample_position);
        self.anchor = Some((stamp.sample_position, metadata.capture_index));
        Ok(())
    }

    fn write_raw_samples(&mut self, samples: &[f32]) -> Result<(), String> {
        if samples.is_empty() || self.truncated {
            return Ok(());
        }
        let bytes = samples.len() as u64 * 4;
        if self.bytes_written.saturating_add(bytes) > RECORDER_MAX_AUDIO_TEMP_BYTES {
            self.truncated = true;
            return Ok(());
        }
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| "recording audio temp writer is closed".to_string())?;
        for sample in samples {
            writer
                .write_all(&sample.to_le_bytes())
                .map_err(|error| format!("write recording audio temp: {error}"))?;
        }
        self.bytes_written += bytes;
        Ok(())
    }

    fn write_silence_frames(&mut self, frames: u64) -> Result<u64, String> {
        if frames == 0 || self.truncated {
            return Ok(0);
        }
        let channels = self.tap.channels() as u64;
        let mut remaining_samples = frames.saturating_mul(channels);
        let chunk =
            vec![0.0f32; RECORDER_AUDIO_SILENCE_CHUNK_SAMPLES.min(remaining_samples as usize)];
        let mut written_samples = 0u64;
        while remaining_samples > 0 && !self.truncated {
            let take = (chunk.len() as u64).min(remaining_samples) as usize;
            self.write_raw_samples(&chunk[..take])?;
            if self.truncated {
                break;
            }
            written_samples += take as u64;
            remaining_samples -= take as u64;
        }
        Ok(written_samples / channels)
    }

    /// Drain the ring: recovered gaps become explicit silence, a pending
    /// drift drop consumes real frames, and the remainder lands in the temp.
    fn pump(&mut self) -> Result<(), String> {
        if self.anchor.is_none() || self.truncated {
            return Ok(());
        }
        if self.tap.device_lost() {
            self.device_lost = true;
        }
        self.scratch.clear();
        let gap_frames = self.tap.take_available(&mut self.scratch);
        if gap_frames > 0 {
            let written = self.write_silence_frames(gap_frames)?;
            self.silence_gap_frames += written;
        }
        let channels = self.tap.channels() as usize;
        let mut start_frame = 0usize;
        if self.pending_drop_frames > 0 {
            let available = (self.scratch.len() / channels) as u64;
            let dropped = available.min(self.pending_drop_frames);
            self.pending_drop_frames -= dropped;
            self.drift_dropped_frames += dropped;
            start_frame = dropped as usize;
        }
        let scratch = std::mem::take(&mut self.scratch);
        let result = self.write_raw_samples(&scratch[start_frame * channels..]);
        let written_frames = (scratch.len() / channels).saturating_sub(start_frame) as u64;
        self.scratch = scratch;
        result?;
        if !self.truncated {
            self.captured_frames += written_frames;
        }
        Ok(())
    }

    /// The bounded drift-correction law. The stamp was taken on the render
    /// thread at the frame's capture intent, so the measurement compares the
    /// device clock against the program capture cadence directly and is
    /// immune to worker or encoder lag.
    fn correct_drift(
        &mut self,
        metadata: RecorderFrameMetadata,
        frame_rate: RecorderFrameRate,
    ) -> Result<(), String> {
        let (Some((anchor_position, anchor_index)), Some(stamp)) =
            (self.anchor, metadata.audio_clock)
        else {
            return Ok(());
        };
        if self.truncated || self.device_lost {
            return Ok(());
        }
        let rate = self.tap.sample_rate() as u128;
        let expected = u128::from(metadata.capture_index.saturating_sub(anchor_index))
            * rate
            * u128::from(frame_rate.denominator)
            / u128::from(frame_rate.numerator);
        let raw =
            i128::from(stamp.sample_position.saturating_sub(anchor_position)) - expected as i128;
        let net_correction =
            i128::from(self.drift_dropped_frames) - i128::from(self.drift_inserted_frames);
        let drift = raw - net_correction + i128::from(self.pending_drop_frames);
        let threshold =
            (self.tap.sample_rate() as f64 * RECORDER_AUDIO_DRIFT_THRESHOLD_SECONDS) as i128;
        if drift >= threshold {
            self.pending_drop_frames += drift as u64;
        } else if drift <= -threshold {
            let inserted = self.write_silence_frames((-drift) as u64)?;
            self.drift_inserted_frames += inserted;
        }
        Ok(())
    }

    /// Finish the audio timeline: bounded grace for the device to deliver the
    /// tail, one final drain, explicit full silence for an armed source that
    /// never delivered, then flush and durably sync the staged temp.
    fn finish(
        &mut self,
        encoded_frames: u64,
        frame_rate: RecorderFrameRate,
        shared: &Arc<RecorderShared>,
    ) -> Result<(), String> {
        let target_frames =
            mux_target_audio_frames(encoded_frames, frame_rate, self.tap.sample_rate());
        if let Some((anchor_position, _)) = self.anchor {
            let grace_started = Instant::now();
            while !self.tap.device_lost()
                && !self.truncated
                && self.tap.delivered_frames().saturating_sub(anchor_position) < target_frames
                && grace_started.elapsed() < RECORDER_AUDIO_FINISH_GRACE
            {
                if shared.cancel.load(Ordering::Acquire) {
                    return Err("recording cancelled".into());
                }
                std::thread::sleep(RECORDER_POLL_INTERVAL);
            }
        }
        self.pump()?;
        if self.total_file_frames() == 0 {
            // An armed source that never delivered a single frame still
            // publishes a fully explicit silent timeline: the mux must never
            // see an empty raw input.
            let written = self.write_silence_frames(target_frames.max(1))?;
            self.silence_gap_frames += written;
        }
        if self.tap.device_lost() {
            self.device_lost = true;
        }
        let writer = self
            .writer
            .take()
            .ok_or_else(|| "recording audio temp writer is closed".to_string())?;
        let file = writer
            .into_inner()
            .map_err(|error| format!("flush recording audio temp: {error}"))?;
        file.sync_all()
            .map_err(|error| format!("sync recording audio temp: {error}"))?;
        Ok(())
    }

    fn report(&self, muxed_duration_secs: f64) -> RecorderAudioReport {
        RecorderAudioReport {
            device: self.tap.device_name().to_string(),
            sample_rate: self.tap.sample_rate(),
            channels: self.tap.channels(),
            anchor_sample_position: self.anchor.map(|(position, _)| position).unwrap_or(0),
            captured_frames: self.captured_frames,
            silence_gap_frames: self.silence_gap_frames,
            drift_inserted_frames: self.drift_inserted_frames,
            drift_dropped_frames: self.drift_dropped_frames,
            device_lost: self.device_lost,
            capture_truncated: self.truncated,
            muxed_duration_secs,
        }
    }
}

/// Exact rational CFR duration of `encoded_frames` at `frame_rate`.
fn capture_video_duration_secs(encoded_frames: u64, frame_rate: RecorderFrameRate) -> f64 {
    encoded_frames as f64 * frame_rate.denominator as f64 / frame_rate.numerator as f64
}

/// Audio frames required to cover the encoded video timeline, rounded up.
fn mux_target_audio_frames(
    encoded_frames: u64,
    frame_rate: RecorderFrameRate,
    sample_rate: u32,
) -> u64 {
    let numerator =
        u128::from(encoded_frames) * u128::from(frame_rate.denominator) * u128::from(sample_rate);
    let denominator = u128::from(frame_rate.numerator);
    (numerator.div_ceil(denominator)).min(u128::from(u64::MAX)) as u64
}

struct TempArtifactGuard {
    paths: Vec<PathBuf>,
    armed: bool,
}

impl TempArtifactGuard {
    fn new() -> Self {
        Self {
            paths: Vec::new(),
            armed: true,
        }
    }

    fn track(&mut self, path: PathBuf) {
        self.paths.push(path);
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TempArtifactGuard {
    fn drop(&mut self) {
        if self.armed {
            cleanup_paths(self.paths.iter().map(PathBuf::as_path));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn recorder_worker(
    config: RecorderConfig,
    frame_bytes: usize,
    pool_tx: SyncSender<Vec<u8>>,
    work_rx: Receiver<RecorderWorkItem>,
    event_tx: SyncSender<RecorderTerminalEvent>,
    shared: Arc<RecorderShared>,
    factory: SinkFactory,
    mux_runner: MuxRunner,
) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_recorder_worker(
            &config,
            frame_bytes,
            &pool_tx,
            &work_rx,
            &shared,
            factory,
            mux_runner,
        )
    }));
    let terminal = match result {
        Ok(Ok(capture)) => {
            shared.phase.store(PHASE_SUCCEEDED, Ordering::Release);
            RecorderTerminalEvent::Succeeded(capture)
        }
        Ok(Err(error))
            if shared.cancel.load(Ordering::Acquire) || shared.publication.cancelled() =>
        {
            shared.set_error(error);
            shared.phase.store(PHASE_CANCELLED, Ordering::Release);
            RecorderTerminalEvent::Cancelled
        }
        Ok(Err(error)) => {
            shared.set_error(&error);
            shared.phase.store(PHASE_FAILED, Ordering::Release);
            RecorderTerminalEvent::Failed(error)
        }
        Err(_) => {
            let error = "recording worker panicked".to_string();
            shared.set_error(&error);
            shared.phase.store(PHASE_FAILED, Ordering::Release);
            RecorderTerminalEvent::Failed(error)
        }
    };
    let _ = event_tx.try_send(terminal);
}

fn run_recorder_worker(
    config: &RecorderConfig,
    frame_bytes: usize,
    pool_tx: &SyncSender<Vec<u8>>,
    work_rx: &Receiver<RecorderWorkItem>,
    shared: &Arc<RecorderShared>,
    factory: SinkFactory,
    mux_runner: MuxRunner,
) -> Result<CommittedCapture, String> {
    for _ in 0..RECORDER_FRAME_POOL_CAPACITY {
        pool_tx
            .send(vec![0; frame_bytes])
            .map_err(|_| "recording frame-pool receiver disconnected".to_string())?;
    }
    let temp_path = sibling_temp_path(&config.output_path, "recording")?;
    let report_path = recorder_report_path(&config.output_path);
    let report_temp = sibling_temp_path(&report_path, "recording-report")?;
    reserve_temp(&temp_path)?;
    // Arm cleanup as soon as the first file exists. In particular, a failure
    // to reserve the report temp must not strand the already-reserved media
    // temp beside the requested artifact.
    let mut temp_guard = TempArtifactGuard::new();
    temp_guard.track(temp_path.clone());
    reserve_temp(&report_temp)?;
    temp_guard.track(report_temp.clone());
    let mut audio = match config.audio_tap.as_ref() {
        Some(tap) => {
            let audio_temp = sibling_temp_path(&config.output_path, "recording-audio")?;
            reserve_temp(&audio_temp)?;
            temp_guard.track(audio_temp.clone());
            let mux_temp = sibling_temp_path(&config.output_path, "recording-mux")?;
            reserve_temp(&mux_temp)?;
            temp_guard.track(mux_temp.clone());
            Some(RecorderAudioCapture::open(
                tap.clone(),
                audio_temp,
                mux_temp,
            )?)
        }
        None => None,
    };
    let mut sink = factory(config, &temp_path, shared.clone())?;
    shared.phase.store(PHASE_RECORDING, Ordering::Release);

    let mut last_frame: Option<Vec<u8>> = None;
    let mut first_metadata = None;
    let mut last_metadata = None;
    let mut finish_idle_since = None;
    loop {
        if shared.cancel.load(Ordering::Acquire) {
            drop(sink);
            return Err("recording cancelled".into());
        }
        match work_rx.recv_timeout(RECORDER_POLL_INTERVAL) {
            Ok(item) => {
                finish_idle_since = None;
                if item.pixels.len() != frame_bytes
                    || !metadata_follows(last_metadata, item.metadata)
                {
                    shared
                        .counters
                        .rejected_metadata
                        .fetch_add(1, Ordering::Relaxed);
                    let _ = pool_tx.try_send(item.pixels);
                    continue;
                }
                if let (Some(last), Some(previous)) = (&last_frame, last_metadata) {
                    let gap = item
                        .metadata
                        .capture_index
                        .saturating_sub(previous.capture_index)
                        .saturating_sub(1);
                    if gap > MAX_DUPLICATION_GAP_FRAMES {
                        shared
                            .counters
                            .rejected_metadata
                            .fetch_add(1, Ordering::Relaxed);
                        let _ = pool_tx.try_send(item.pixels);
                        continue;
                    }
                    for _ in 0..gap {
                        sink.write_frame(last)?;
                        shared.counters.duplicated.fetch_add(1, Ordering::Relaxed);
                        shared.counters.encoded.fetch_add(1, Ordering::Relaxed);
                    }
                }
                sink.write_frame(&item.pixels)?;
                shared.counters.encoded.fetch_add(1, Ordering::Relaxed);
                first_metadata.get_or_insert(item.metadata);
                last_metadata = Some(item.metadata);
                if let Some(previous) = last_frame.replace(item.pixels) {
                    let _ = pool_tx.try_send(previous);
                }
                if let Some(audio) = audio.as_mut() {
                    audio.anchor_at(item.metadata)?;
                    audio.pump()?;
                    audio.correct_drift(item.metadata, config.frame_rate)?;
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if let Some(audio) = audio.as_mut() {
                    audio.pump()?;
                }
                if shared.finish_requested.load(Ordering::Acquire) {
                    let started = *finish_idle_since.get_or_insert_with(Instant::now);
                    // One empty poll after the finish flag proves the bounded
                    // queue is drained. The duration guard protects a future
                    // channel implementation from spinning forever.
                    if started.elapsed() >= RECORDER_POLL_INTERVAL {
                        break;
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    if let (Some(last), Some(previous)) = (&last_frame, last_metadata) {
        let requested = shared.finish_capture_index.load(Ordering::Acquire);
        let gap = requested
            .checked_sub(previous.capture_index)
            .ok_or_else(|| {
                format!(
                    "recording finish index {requested} predates last accepted frame {}",
                    previous.capture_index
                )
            })?;
        if gap > MAX_DUPLICATION_GAP_FRAMES {
            return Err(format!(
                "recording finish gap {gap} exceeds {MAX_DUPLICATION_GAP_FRAMES} frames"
            ));
        }
        for _ in 0..gap {
            sink.write_frame(last)?;
            shared.counters.duplicated.fetch_add(1, Ordering::Relaxed);
            shared.counters.encoded.fetch_add(1, Ordering::Relaxed);
        }
    }
    if first_metadata.is_none() {
        return Err("recording ended before any frame was accepted".into());
    }
    let encoded_frames = shared.counters.encoded.load(Ordering::Acquire);
    if let Some(audio) = audio.as_mut() {
        audio.finish(encoded_frames, config.frame_rate, shared)?;
    }
    if let Err(error) = sink.finish() {
        shared
            .counters
            .encoder_failures
            .fetch_add(1, Ordering::Relaxed);
        return Err(error);
    }
    sync_path(&temp_path)?;
    let (commit_media_temp, audio_report) = match audio.as_ref() {
        Some(audio) => {
            let duration_secs = capture_video_duration_secs(encoded_frames, config.frame_rate);
            let plan = CaptureMuxPlan {
                video_temp: temp_path.clone(),
                audio_temp: audio.temp_path.clone(),
                output_temp: audio.mux_temp_path.clone(),
                sample_rate: audio.tap.sample_rate(),
                channels: audio.tap.channels(),
                duration_secs,
            };
            if let Err(error) = mux_runner(&plan, shared) {
                shared
                    .counters
                    .encoder_failures
                    .fetch_add(1, Ordering::Relaxed);
                return Err(error);
            }
            sync_path(&audio.mux_temp_path)?;
            (
                audio.mux_temp_path.clone(),
                Some(audio.report(duration_secs)),
            )
        }
        None => (temp_path.clone(), None),
    };
    let counters = shared.counters.snapshot();
    let requested_final_capture_index = shared.finish_capture_index.load(Ordering::Acquire);
    let audio_muxed = audio_report.is_some();
    let report = RecorderReport::new(
        config,
        counters,
        first_metadata.expect("guarded metadata"),
        last_metadata.expect("guarded metadata"),
        requested_final_capture_index,
        audio_report,
    );
    write_report_temp(&report_temp, &report)?;
    commit_artifact_pair_linearized(
        &shared.publication,
        &commit_media_temp,
        &config.output_path,
        &report_temp,
        &report_path,
    )?;
    temp_guard.disarm();
    if let Some(audio) = audio.as_ref() {
        // The staged intermediates behind a committed mux are no longer part
        // of any transaction; remove them now that the guard is disarmed.
        cleanup_paths([temp_path.as_path(), audio.temp_path.as_path()]);
    }
    Ok(CommittedCapture {
        media_path: config.output_path.clone(),
        report_path,
        target: config.target,
        purpose: config.purpose,
        dimensions: config.dimensions,
        frame_rate: config.frame_rate,
        counters,
        audio_muxed,
    })
}

fn metadata_follows(previous: Option<RecorderFrameMetadata>, next: RecorderFrameMetadata) -> bool {
    if next.validate().is_err() {
        return false;
    }
    previous.is_none_or(|previous| {
        next.capture_index > previous.capture_index
            && next.capture_time_ns >= previous.capture_time_ns
            && next.program_time_ns >= previous.program_time_ns
            && match (previous.audio_clock, next.audio_clock) {
                (Some(a), Some(b)) => {
                    a.sample_rate == b.sample_rate
                        && a.channels == b.channels
                        && b.sample_position >= a.sample_position
                }
                _ => true,
            }
    })
}

#[derive(Debug, Serialize)]
struct RecorderReport {
    schema_version: u16,
    build_identity: crate::build_identity::BuildIdentitySnapshot,
    media_file_name: String,
    width: u32,
    height: u32,
    frame_rate: RecorderFrameRate,
    drop_policy: &'static str,
    pool_capacity: usize,
    queue_capacity: usize,
    counters: RecorderCounters,
    first_frame: RecorderFrameMetadata,
    last_accepted_frame: RecorderFrameMetadata,
    /// Inclusive CFR frame index requested by the render thread at finish.
    /// It can exceed `last_accepted_frame.capture_index` when drop-newest was
    /// recovered by duplicating the last encoded picture.
    requested_final_capture_index: u64,
    audio_not_muxed: bool,
    /// Present exactly when a program-PCM source was armed and muxed.
    #[serde(skip_serializing_if = "Option::is_none")]
    audio: Option<RecorderAudioReport>,
    target: ReportTarget,
    purpose: ReportPurpose,
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ReportTarget {
    Program,
    Layer { stable_id: String },
    Group { stable_id: String },
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ReportPurpose {
    External,
    AutoImport,
    Resample {
        destination_layer_id: String,
        activate: bool,
    },
}

impl RecorderReport {
    fn new(
        config: &RecorderConfig,
        counters: RecorderCounters,
        first_frame: RecorderFrameMetadata,
        last_accepted_frame: RecorderFrameMetadata,
        requested_final_capture_index: u64,
        audio: Option<RecorderAudioReport>,
    ) -> Self {
        Self::for_capture(
            &config.output_path,
            config.dimensions,
            config.frame_rate,
            config.target,
            config.purpose,
            "drop_newest_duplicate_previous_cfr",
            RECORDER_FRAME_POOL_CAPACITY,
            RECORDER_FRAME_QUEUE_CAPACITY,
            counters,
            first_frame,
            last_accepted_frame,
            requested_final_capture_index,
            audio,
        )
    }

    fn new_still(
        config: &StillSnapshotConfig,
        metadata: RecorderFrameMetadata,
        counters: RecorderCounters,
    ) -> Self {
        Self::for_capture(
            &config.output_path,
            config.dimensions,
            RecorderFrameRate::FPS_30,
            config.target,
            config.purpose,
            "single_frame_no_drop",
            1,
            0,
            counters,
            metadata,
            metadata,
            metadata.capture_index,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn for_capture(
        output_path: &Path,
        dimensions: RecorderDimensions,
        frame_rate: RecorderFrameRate,
        capture_target: CaptureTarget,
        capture_purpose: CapturePurpose,
        drop_policy: &'static str,
        pool_capacity: usize,
        queue_capacity: usize,
        counters: RecorderCounters,
        first_frame: RecorderFrameMetadata,
        last_accepted_frame: RecorderFrameMetadata,
        requested_final_capture_index: u64,
        audio: Option<RecorderAudioReport>,
    ) -> Self {
        let target = match capture_target {
            CaptureTarget::Program => ReportTarget::Program,
            CaptureTarget::Layer(id) => ReportTarget::Layer {
                stable_id: id.get().to_string(),
            },
            CaptureTarget::Group(id) => ReportTarget::Group {
                stable_id: id.get().to_string(),
            },
        };
        let purpose = match capture_purpose {
            CapturePurpose::External => ReportPurpose::External,
            CapturePurpose::AutoImport => ReportPurpose::AutoImport,
            CapturePurpose::Resample {
                destination_layer,
                activate,
            } => ReportPurpose::Resample {
                destination_layer_id: destination_layer.get().to_string(),
                activate,
            },
        };
        Self {
            schema_version: RECORDER_REPORT_SCHEMA_VERSION,
            build_identity: crate::build_identity::current().snapshot(),
            media_file_name: output_path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default(),
            width: dimensions.width,
            height: dimensions.height,
            frame_rate,
            drop_policy,
            pool_capacity,
            queue_capacity,
            counters,
            first_frame,
            last_accepted_frame,
            requested_final_capture_index,
            audio_not_muxed: audio.is_none(),
            audio,
            target,
            purpose,
        }
    }
}

fn write_report_temp(path: &Path, report: &RecorderReport) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("serialize recording report: {error}"))?;
    if bytes.len() > RECORDER_MAX_REPORT_BYTES {
        return Err(format!(
            "recording report has {} bytes; limit is {RECORDER_MAX_REPORT_BYTES}",
            bytes.len()
        ));
    }
    let mut file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(path)
        .map_err(|error| format!("open recording report temp: {error}"))?;
    file.write_all(&bytes)
        .map_err(|error| format!("write recording report temp: {error}"))?;
    file.sync_all()
        .map_err(|error| format!("sync recording report temp: {error}"))
}

#[derive(Debug, Clone)]
pub struct StillSnapshotConfig {
    pub dimensions: RecorderDimensions,
    pub output_path: PathBuf,
    pub target: CaptureTarget,
    pub purpose: CapturePurpose,
}

pub struct StillSnapshotJob {
    cancel: Arc<AtomicBool>,
    publication: Arc<PublicationGate>,
    event_rx: Receiver<RecorderTerminalEvent>,
    worker: Option<JoinHandle<()>>,
}

impl StillSnapshotJob {
    /// Transfer one already-harvested RGBA frame to a helper. No filesystem or
    /// PNG work occurs on the caller after the bounded validation below.
    pub fn spawn(
        config: StillSnapshotConfig,
        pixels: Vec<u8>,
        metadata: RecorderFrameMetadata,
    ) -> Result<Self, String> {
        Self::spawn_with_publication(
            config,
            pixels,
            metadata,
            Arc::new(PublicationGate::default()),
        )
    }

    fn spawn_with_publication(
        config: StillSnapshotConfig,
        pixels: Vec<u8>,
        metadata: RecorderFrameMetadata,
        publication: Arc<PublicationGate>,
    ) -> Result<Self, String> {
        let expected = config.dimensions.frame_bytes()?;
        if pixels.len() != expected {
            return Err(format!(
                "still snapshot has {} RGBA bytes; expected {expected}",
                pixels.len()
            ));
        }
        metadata.validate()?;
        validate_capture_pair_destinations(&config.output_path)?;
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        let worker_publication = publication.clone();
        let (event_tx, event_rx) = mpsc::sync_channel(1);
        let worker = std::thread::Builder::new()
            .name("still-snapshot".into())
            .spawn(move || {
                let event = match publish_still(
                    &config,
                    &pixels,
                    metadata,
                    &worker_cancel,
                    &worker_publication,
                ) {
                    Ok(committed) => RecorderTerminalEvent::Succeeded(committed),
                    Err(_)
                        if worker_cancel.load(Ordering::Acquire)
                            || worker_publication.cancelled() =>
                    {
                        RecorderTerminalEvent::Cancelled
                    }
                    Err(error) => RecorderTerminalEvent::Failed(error),
                };
                let _ = event_tx.try_send(event);
            })
            .map_err(|error| format!("start still snapshot worker: {error}"))?;
        Ok(Self {
            cancel,
            publication,
            event_rx,
            worker: Some(worker),
        })
    }

    pub fn cancel(&self) {
        if self.publication.request_cancel() {
            self.cancel.store(true, Ordering::Release);
        }
    }

    pub fn poll_terminal(&mut self) -> Option<RecorderTerminalEvent> {
        match self.event_rx.try_recv() {
            Ok(event) => {
                if self.worker.as_ref().is_some_and(JoinHandle::is_finished) {
                    if let Some(worker) = self.worker.take() {
                        let _ = worker.join();
                    }
                }
                Some(event)
            }
            Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => None,
        }
    }
}

impl Drop for StillSnapshotJob {
    fn drop(&mut self) {
        self.cancel();
        self.worker.take();
    }
}

fn publish_still(
    config: &StillSnapshotConfig,
    pixels: &[u8],
    metadata: RecorderFrameMetadata,
    cancel: &AtomicBool,
    publication: &PublicationGate,
) -> Result<CommittedCapture, String> {
    let media_temp = sibling_temp_path(&config.output_path, "still")?;
    let report_path = recorder_report_path(&config.output_path);
    let report_temp = sibling_temp_path(&report_path, "still-report")?;
    reserve_temp(&media_temp)?;
    let mut temp_guard = TempArtifactGuard::new();
    temp_guard.track(media_temp.clone());
    reserve_temp(&report_temp)?;
    temp_guard.track(report_temp.clone());
    if cancel.load(Ordering::Acquire) || publication.cancelled() {
        return Err("still snapshot cancelled".into());
    }

    let file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(&media_temp)
        .map_err(|error| format!("open still snapshot temp: {error}"))?;
    let mut writer = BufWriter::new(file);
    PngEncoder::new(&mut writer)
        .write_image(
            pixels,
            config.dimensions.width,
            config.dimensions.height,
            ExtendedColorType::Rgba8,
        )
        .map_err(|error| format!("encode still snapshot PNG: {error}"))?;
    writer
        .flush()
        .map_err(|error| format!("flush still snapshot PNG: {error}"))?;
    let file = writer
        .into_inner()
        .map_err(|error| format!("finish still snapshot PNG: {error}"))?;
    file.sync_all()
        .map_err(|error| format!("sync still snapshot PNG: {error}"))?;

    let counters = RecorderCounters {
        attempted: 1,
        accepted: 1,
        encoded: 1,
        ..RecorderCounters::default()
    };
    write_report_temp(
        &report_temp,
        &RecorderReport::new_still(config, metadata, counters),
    )?;
    commit_artifact_pair_linearized(
        publication,
        &media_temp,
        &config.output_path,
        &report_temp,
        &report_path,
    )?;
    temp_guard.disarm();
    Ok(CommittedCapture {
        media_path: config.output_path.clone(),
        report_path,
        target: config.target,
        purpose: config.purpose,
        dimensions: config.dimensions,
        frame_rate: RecorderFrameRate::FPS_30,
        counters,
        audio_muxed: false,
    })
}

pub fn recorder_report_path(media_path: &Path) -> PathBuf {
    let mut name = media_path
        .file_name()
        .map(OsString::from)
        .unwrap_or_else(|| OsString::from("recording"));
    name.push(".recording.json");
    media_path.with_file_name(name)
}

fn validate_capture_pair_destinations(media_path: &Path) -> Result<PathBuf, String> {
    validate_final_path(media_path)?;
    let report_path = recorder_report_path(media_path);
    if report_path == media_path {
        return Err("recording report path collides with the media path".into());
    }
    if report_path.exists() {
        return Err(format!(
            "recording report destination already exists: {}",
            report_path.display()
        ));
    }
    Ok(report_path)
}

fn validate_final_path(path: &Path) -> Result<(), String> {
    if path.file_name().is_none() {
        return Err("capture destination must name a file".into());
    }
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(format!(
            "capture destination directory is unavailable: {}",
            parent.display()
        ));
    }
    if path.exists() {
        return Err(format!(
            "capture destination already exists: {}",
            path.display()
        ));
    }
    Ok(())
}

fn sibling_temp_path(path: &Path, label: &str) -> Result<PathBuf, String> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .ok_or_else(|| "capture destination must name a file".to_string())?;
    for _ in 0..8 {
        let mut random = [0_u8; 12];
        getrandom::fill(&mut random).map_err(|error| format!("capture temp entropy: {error}"))?;
        let nonce = random
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let mut candidate = OsString::from(".");
        candidate.push(name);
        candidate.push(format!(".{label}-{nonce}.tmp"));
        let path = parent.join(candidate);
        if !path.exists() {
            return Ok(path);
        }
    }
    Err("could not reserve a unique same-directory capture temp name".into())
}

fn reserve_temp(path: &Path) -> Result<(), String> {
    OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .and_then(|file| file.sync_all())
        .map_err(|error| format!("reserve capture temp '{}': {error}", path.display()))
}

fn sync_path(path: &Path) -> Result<(), String> {
    // Windows FlushFileBuffers requires a write-capable handle even though no
    // bytes are changed here. A read-only File::open yields ERROR_ACCESS_DENIED.
    OpenOptions::new()
        .write(true)
        .open(path)
        .and_then(|file| file.sync_all())
        .map_err(|error| format!("sync capture temp '{}': {error}", path.display()))
}

fn commit_artifact_pair_no_replace(
    media_temp: &Path,
    media_final: &Path,
    report_temp: &Path,
    report_final: &Path,
) -> Result<(), String> {
    // Report first: once the media name appears, its durable clock/drop truth
    // already exists. If media publication loses a no-replace race, roll back
    // only the report name created by this worker.
    atomic_commit_no_replace(report_temp, report_final)?;
    if let Err(error) = atomic_commit_no_replace(media_temp, media_final) {
        let _ = std::fs::remove_file(report_final);
        sync_parent(report_final);
        let _ = std::fs::remove_file(media_temp);
        return Err(error);
    }
    sync_parent(media_final);
    Ok(())
}

fn commit_artifact_pair_linearized(
    publication: &PublicationGate,
    media_temp: &Path,
    media_final: &Path,
    report_temp: &Path,
    report_final: &Path,
) -> Result<(), String> {
    publication.claim()?;
    commit_artifact_pair_no_replace(media_temp, media_final, report_temp, report_final)?;
    publication.mark_published();
    Ok(())
}

#[cfg(windows)]
fn atomic_commit_no_replace(temp: &Path, final_path: &Path) -> Result<(), String> {
    use std::os::windows::ffi::OsStrExt as _;
    use windows_sys::Win32::Storage::FileSystem::{MoveFileExW, MOVEFILE_WRITE_THROUGH};

    let from: Vec<u16> = temp.as_os_str().encode_wide().chain(Some(0)).collect();
    let to: Vec<u16> = final_path
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect();
    // Deliberately omit MOVEFILE_REPLACE_EXISTING.
    let moved = unsafe { MoveFileExW(from.as_ptr(), to.as_ptr(), MOVEFILE_WRITE_THROUGH) };
    if moved == 0 {
        return Err(format!(
            "atomic no-replace commit '{}' failed: {}",
            final_path.display(),
            std::io::Error::last_os_error()
        ));
    }
    Ok(())
}

#[cfg(not(windows))]
fn atomic_commit_no_replace(temp: &Path, final_path: &Path) -> Result<(), String> {
    std::fs::hard_link(temp, final_path).map_err(|error| {
        format!(
            "atomic no-replace commit '{}' failed: {error}",
            final_path.display()
        )
    })?;
    std::fs::remove_file(temp)
        .map_err(|error| format!("remove committed capture temp: {error}"))?;
    Ok(())
}

fn sync_parent(path: &Path) {
    #[cfg(unix)]
    if let Some(parent) = path.parent() {
        if let Ok(directory) = File::open(parent) {
            let _ = directory.sync_all();
        }
    }
    #[cfg(not(unix))]
    let _ = path;
}

fn cleanup_paths<'a>(paths: impl IntoIterator<Item = &'a Path>) {
    for path in paths {
        let _ = std::fs::remove_file(path);
    }
}

fn read_bounded_file(path: &Path, max_chars: usize) -> Result<String, String> {
    // UTF-8 needs at most four bytes per scalar. Reading only this bounded
    // prefix prevents a hostile helper log from causing proportional memory.
    let max_bytes = max_chars.saturating_mul(4).saturating_add(1);
    let mut bytes = Vec::with_capacity(max_bytes);
    File::open(path)
        .map_err(|error| format!("open encoder log: {error}"))?
        .take(max_bytes as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read encoder log: {error}"))?;
    Ok(bounded_text(&String::from_utf8_lossy(&bytes), max_chars))
}

fn bounded_text(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut bounded = text
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    bounded.push('…');
    bounded
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Condvar, Mutex};

    fn temp_path(name: &str) -> PathBuf {
        let mut random = [0_u8; 8];
        getrandom::fill(&mut random).unwrap();
        let nonce = u64::from_le_bytes(random);
        let unique = format!("collide-recorder-{}-{nonce}-{name}", std::process::id());
        std::env::temp_dir().join(unique)
    }

    fn config(path: PathBuf) -> RecorderConfig {
        RecorderConfig {
            dimensions: RecorderDimensions::new(2, 2).unwrap(),
            frame_rate: RecorderFrameRate::FPS_30,
            output_path: path,
            target: CaptureTarget::Program,
            purpose: CapturePurpose::External,
            audio_tap: None,
        }
    }

    fn metadata(index: u64) -> RecorderFrameMetadata {
        RecorderFrameMetadata {
            capture_index: index,
            capture_time_ns: index * 33_333_333,
            program_time_ns: index * 33_333_333,
            visual_epoch: 1,
            program_frozen: false,
            media_frozen: false,
            blackout: false,
            audio_clock: Some(AudioClockStamp {
                sample_position: index * 1_600,
                sample_rate: 48_000,
                channels: 2,
            }),
        }
    }

    fn publication_pause() -> (
        PublicationTestHook,
        std::sync::mpsc::Receiver<()>,
        std::sync::mpsc::SyncSender<()>,
    ) {
        let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
        let release_rx = Mutex::new(release_rx);
        let hook: PublicationTestHook = Arc::new(move || {
            entered_tx
                .try_send(())
                .expect("publication hook must run exactly once");
            release_rx
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(2))
                .expect("publication hook release timed out");
        });
        (hook, entered_rx, release_tx)
    }

    struct FixtureSink {
        temp: PathBuf,
    }

    impl RecorderVideoSink for FixtureSink {
        fn write_frame(&mut self, pixels: &[u8]) -> Result<(), String> {
            OpenOptions::new()
                .append(true)
                .open(&self.temp)
                .map_err(|error| error.to_string())?
                .write_all(pixels)
                .map_err(|error| error.to_string())
        }

        fn finish(self: Box<Self>) -> Result<(), String> {
            Ok(())
        }
    }

    fn fixture_sink_factory() -> SinkFactory {
        Box::new(|_, temp, _| {
            Ok(Box::new(FixtureSink {
                temp: temp.to_owned(),
            }))
        })
    }

    fn wait_until_recording(recorder: &ProgramRecorder) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while recorder.snapshot().status != RecorderStatus::Recording {
            assert!(Instant::now() < deadline, "recorder did not become ready");
            std::thread::yield_now();
        }
    }

    fn submit_fixture_frame(recorder: &ProgramRecorder, index: u64) {
        let RecorderAcquire::Lease(mut lease) = recorder.try_acquire_frame() else {
            panic!("preallocated recorder lease unavailable");
        };
        lease.pixels_mut().fill(index as u8);
        assert_eq!(
            recorder.try_submit(lease, metadata(index)),
            RecorderSubmit::Accepted
        );
    }

    fn wait_recorder_terminal(recorder: &mut ProgramRecorder) -> RecorderTerminalEvent {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if let Some(event) = recorder.poll_terminal() {
                return event;
            }
            assert!(
                Instant::now() < deadline,
                "recorder terminal event timed out"
            );
            std::thread::yield_now();
        }
    }

    fn wait_still_terminal(job: &mut StillSnapshotJob) -> RecorderTerminalEvent {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if let Some(event) = job.poll_terminal() {
                return event;
            }
            assert!(
                Instant::now() < deadline,
                "still snapshot terminal event timed out"
            );
            std::thread::yield_now();
        }
    }

    #[test]
    fn pool_is_bounded_by_per_frame_and_aggregate_limits() {
        assert!(RecorderDimensions::new(3_840, 2_160).is_ok());
        assert!(RecorderDimensions::new(3_841, 2_160).is_err());
        assert!(
            u64::from(3_840_u32) * 2_160 * 4 * RECORDER_FRAME_POOL_CAPACITY as u64
                <= RECORDER_MAX_POOL_BYTES
        );
    }

    #[test]
    fn no_replace_pair_never_overwrites_and_rolls_back_report() {
        let media = temp_path("capture.mp4");
        let report = recorder_report_path(&media);
        let media_temp = sibling_temp_path(&media, "media").unwrap();
        let report_temp = sibling_temp_path(&report, "report").unwrap();
        std::fs::write(&media_temp, b"new media").unwrap();
        std::fs::write(&report_temp, b"new report").unwrap();
        std::fs::write(&media, b"existing").unwrap();
        assert!(
            commit_artifact_pair_no_replace(&media_temp, &media, &report_temp, &report).is_err()
        );
        assert_eq!(std::fs::read(&media).unwrap(), b"existing");
        assert!(!report.exists());
        assert!(!media_temp.exists());
        let _ = std::fs::remove_file(media);
    }

    struct SlowGate {
        entered: Mutex<bool>,
        release: Condvar,
    }

    struct SlowSink {
        gate: Arc<SlowGate>,
        temp: PathBuf,
        first: bool,
    }

    impl RecorderVideoSink for SlowSink {
        fn write_frame(&mut self, pixels: &[u8]) -> Result<(), String> {
            if self.first {
                self.first = false;
                let mut entered = self.gate.entered.lock().unwrap();
                *entered = true;
                self.gate.release.notify_all();
                while *entered {
                    entered = self.gate.release.wait(entered).unwrap();
                }
            }
            std::fs::OpenOptions::new()
                .append(true)
                .open(&self.temp)
                .unwrap()
                .write_all(pixels)
                .unwrap();
            Ok(())
        }

        fn finish(self: Box<Self>) -> Result<(), String> {
            Ok(())
        }
    }

    #[test]
    fn slow_encoder_sheds_newest_without_blocking_or_growing_queue() {
        let output = temp_path("slow.mp4");
        let gate = Arc::new(SlowGate {
            entered: Mutex::new(false),
            release: Condvar::new(),
        });
        let factory_gate = gate.clone();
        let factory: SinkFactory = Box::new(move |_, temp, _| {
            Ok(Box::new(SlowSink {
                gate: factory_gate,
                temp: temp.to_owned(),
                first: true,
            }))
        });
        let mut recorder = ProgramRecorder::spawn_with_factory(config(output.clone()), factory)
            .expect("start recorder");
        let deadline = Instant::now() + Duration::from_secs(2);
        while recorder.snapshot().status != RecorderStatus::Recording {
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
        let submit = |recorder: &ProgramRecorder, index| {
            let RecorderAcquire::Lease(mut lease) = recorder.try_acquire_frame() else {
                return RecorderSubmit::DroppedQueueFull;
            };
            lease.pixels_mut().fill(index as u8);
            recorder.try_submit(lease, metadata(index))
        };
        assert_eq!(submit(&recorder, 0), RecorderSubmit::Accepted);
        let mut entered = gate.entered.lock().unwrap();
        while !*entered {
            entered = gate.release.wait(entered).unwrap();
        }
        drop(entered);
        assert_eq!(submit(&recorder, 1), RecorderSubmit::Accepted);
        assert_eq!(submit(&recorder, 2), RecorderSubmit::Accepted);
        assert_eq!(submit(&recorder, 3), RecorderSubmit::DroppedQueueFull);
        assert_eq!(recorder.snapshot().counters.dropped_queue_full, 1);
        assert!(recorder.poll_terminal().is_none());
        assert!(
            !output.exists(),
            "uncommitted recording leaked its final name"
        );
        let mut entered = gate.entered.lock().unwrap();
        *entered = false;
        gate.release.notify_all();
        drop(entered);
        // Frame 3 was dropped, so finishing through index 3 must duplicate the
        // last accepted picture once to preserve the exact CFR clock.
        recorder.request_finish(3);
        let deadline = Instant::now() + Duration::from_secs(2);
        let event = loop {
            if let Some(event) = recorder.poll_terminal() {
                break event;
            }
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        };
        assert!(
            matches!(&event, RecorderTerminalEvent::Succeeded(_)),
            "unexpected recorder terminal event: {event:?}"
        );
        let report = recorder_report_path(&output);
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&report).unwrap()).unwrap();
        assert_eq!(value["audio_not_muxed"], true);
        assert_eq!(value["schema_version"], RECORDER_REPORT_SCHEMA_VERSION);
        assert_eq!(value["last_accepted_frame"]["capture_index"], 2);
        assert_eq!(value["requested_final_capture_index"], 3);
        assert_eq!(value["counters"]["duplicated"], 1);
        assert_eq!(value["counters"]["encoded"], 4);
        let RecorderTerminalEvent::Succeeded(committed) = event else {
            unreachable!()
        };
        assert_eq!(committed.publication_intent(), CommittedCaptureIntent::None);
        let _ = std::fs::remove_file(output);
        let _ = std::fs::remove_file(report);
    }

    #[test]
    fn stubborn_helper_finish_timeout_kills_reaps_and_removes_log() {
        let stderr_path = temp_path("stubborn-helper.log");
        let stderr = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&stderr_path)
            .unwrap();
        #[cfg(windows)]
        let mut command = {
            let mut command = Command::new("cmd.exe");
            command.args(["/d", "/s", "/c", "ping -n 30 127.0.0.1 >nul"]);
            command
        };
        #[cfg(not(windows))]
        let mut command = {
            let mut command = Command::new("sh");
            command.args(["-c", "sleep 30"]);
            command
        };
        let child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::from(stderr))
            .spawn()
            .unwrap();
        let shared = Arc::new(RecorderShared::new());
        let (_stdin, finish_requested, completion, supervisor) = supervise_encoder_process(
            child,
            stderr_path.clone(),
            shared,
            Duration::from_millis(40),
        )
        .unwrap();
        let started = Instant::now();
        finish_requested.store(true, Ordering::Release);
        let error = completion
            .recv_timeout(Duration::from_secs(2))
            .expect("supervisor must enforce its timeout")
            .expect_err("stubborn helper cannot report success");
        assert!(error.contains("timed out and was reaped"));
        assert!(started.elapsed() < Duration::from_secs(2));
        let join_deadline = Instant::now() + Duration::from_secs(1);
        while !supervisor.is_finished() {
            assert!(Instant::now() < join_deadline);
            std::thread::yield_now();
        }
        supervisor.join().unwrap();
        assert!(!stderr_path.exists());
    }

    #[test]
    fn hostile_encoder_log_is_read_through_a_fixed_prefix() {
        let path = temp_path("hostile-encoder.log");
        std::fs::write(&path, vec![b'x'; 64 * 1024]).unwrap();
        let detail = read_bounded_file(&path, 17).unwrap();
        assert_eq!(detail.chars().count(), 17);
        assert!(detail.ends_with('…'));
        std::fs::remove_file(path).unwrap();
    }

    struct FailOnWriteSink;

    impl RecorderVideoSink for FailOnWriteSink {
        fn write_frame(&mut self, _pixels: &[u8]) -> Result<(), String> {
            Err("synthetic encoder failure".into())
        }

        fn finish(self: Box<Self>) -> Result<(), String> {
            Ok(())
        }
    }

    #[test]
    fn encoder_failure_publishes_nothing_and_cleans_all_temps() {
        let directory = temp_path("failure-cleanup");
        std::fs::create_dir(&directory).unwrap();
        let output = directory.join("capture.mp4");
        let factory: SinkFactory = Box::new(|_, _, _| Ok(Box::new(FailOnWriteSink)));
        let mut recorder =
            ProgramRecorder::spawn_with_factory(config(output.clone()), factory).unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        while recorder.snapshot().status != RecorderStatus::Recording {
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
        let RecorderAcquire::Lease(mut lease) = recorder.try_acquire_frame() else {
            panic!("preallocated recorder lease unavailable");
        };
        lease.pixels_mut().fill(7);
        assert_eq!(
            recorder.try_submit(lease, metadata(0)),
            RecorderSubmit::Accepted
        );
        recorder.request_finish(0);
        let event = loop {
            if let Some(event) = recorder.poll_terminal() {
                break event;
            }
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        };
        assert!(matches!(event, RecorderTerminalEvent::Failed(_)));
        assert!(!output.exists());
        assert!(!recorder_report_path(&output).exists());
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 0);
        std::fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn video_cancel_wins_before_publication_claim_without_blocking() {
        let output = temp_path("video-cancel-wins.mp4");
        let mut recorder =
            ProgramRecorder::spawn_with_factory(config(output.clone()), fixture_sink_factory())
                .unwrap();
        let (hook, entered, release) = publication_pause();
        recorder.shared.publication.set_before_claim_hook(hook);
        wait_until_recording(&recorder);
        submit_fixture_frame(&recorder, 0);
        recorder.request_finish(0);
        entered
            .recv_timeout(Duration::from_secs(2))
            .expect("video worker did not reach the publication barrier");

        let started = Instant::now();
        recorder.cancel();
        assert!(
            started.elapsed() < Duration::from_millis(250),
            "Cancel must remain a nonblocking render-thread operation"
        );
        release.send(()).unwrap();

        assert!(matches!(
            wait_recorder_terminal(&mut recorder),
            RecorderTerminalEvent::Cancelled
        ));
        assert!(!output.exists());
        assert!(!recorder_report_path(&output).exists());
    }

    #[test]
    fn video_publication_claim_wins_over_late_cancel() {
        let output = temp_path("video-publication-wins.mp4");
        let mut recorder =
            ProgramRecorder::spawn_with_factory(config(output.clone()), fixture_sink_factory())
                .unwrap();
        let (hook, entered, release) = publication_pause();
        recorder.shared.publication.set_after_claim_hook(hook);
        wait_until_recording(&recorder);
        submit_fixture_frame(&recorder, 0);
        recorder.request_finish(0);
        entered
            .recv_timeout(Duration::from_secs(2))
            .expect("video worker did not claim publication");

        recorder.cancel();
        release.send(()).unwrap();

        let RecorderTerminalEvent::Succeeded(committed) = wait_recorder_terminal(&mut recorder)
        else {
            panic!("a worker-owned publication must report its actual commit result");
        };
        let report = recorder_report_path(&output);
        assert_eq!(committed.report_path, report);
        assert!(output.exists());
        assert!(report.exists());
        std::fs::remove_file(output).unwrap();
        std::fs::remove_file(report).unwrap();
    }

    #[test]
    fn still_cancel_wins_before_publication_claim_and_cleans_both_temps() {
        let output = temp_path("still-cancel-wins.png");
        let config = StillSnapshotConfig {
            dimensions: RecorderDimensions::new(2, 1).unwrap(),
            output_path: output.clone(),
            target: CaptureTarget::Program,
            purpose: CapturePurpose::External,
        };
        let publication = Arc::new(PublicationGate::default());
        let (hook, entered, release) = publication_pause();
        publication.set_before_claim_hook(hook);
        let mut job = StillSnapshotJob::spawn_with_publication(
            config,
            vec![255, 0, 0, 255, 0, 255, 0, 255],
            metadata(7),
            publication,
        )
        .unwrap();
        entered
            .recv_timeout(Duration::from_secs(2))
            .expect("still worker did not reach the publication barrier");

        let started = Instant::now();
        job.cancel();
        assert!(
            started.elapsed() < Duration::from_millis(250),
            "still Cancel must remain nonblocking"
        );
        release.send(()).unwrap();

        assert!(matches!(
            wait_still_terminal(&mut job),
            RecorderTerminalEvent::Cancelled
        ));
        assert!(!output.exists());
        assert!(!recorder_report_path(&output).exists());
    }

    #[test]
    fn still_publication_claim_wins_over_late_cancel_and_commits_pair() {
        let output = temp_path("still-publication-wins.png");
        let config = StillSnapshotConfig {
            dimensions: RecorderDimensions::new(2, 1).unwrap(),
            output_path: output.clone(),
            target: CaptureTarget::Program,
            purpose: CapturePurpose::External,
        };
        let publication = Arc::new(PublicationGate::default());
        let (hook, entered, release) = publication_pause();
        publication.set_after_claim_hook(hook);
        let mut job = StillSnapshotJob::spawn_with_publication(
            config,
            vec![255, 0, 0, 255, 0, 255, 0, 255],
            metadata(8),
            publication,
        )
        .unwrap();
        entered
            .recv_timeout(Duration::from_secs(2))
            .expect("still worker did not claim publication");

        job.cancel();
        release.send(()).unwrap();

        let RecorderTerminalEvent::Succeeded(committed) = wait_still_terminal(&mut job) else {
            panic!("a worker-owned still publication must report its actual commit result");
        };
        let report = recorder_report_path(&output);
        assert_eq!(committed.report_path, report);
        assert!(image::open(&output).is_ok());
        assert!(report.exists());
        std::fs::remove_file(output).unwrap();
        std::fs::remove_file(report).unwrap();
    }

    #[test]
    fn still_snapshot_commits_only_after_complete_png_and_never_overwrites() {
        let output = temp_path("still.png");
        let config = StillSnapshotConfig {
            dimensions: RecorderDimensions::new(2, 1).unwrap(),
            output_path: output.clone(),
            target: CaptureTarget::Program,
            purpose: CapturePurpose::AutoImport,
        };
        let mut job = StillSnapshotJob::spawn(
            config.clone(),
            vec![255, 0, 0, 255, 0, 255, 0, 255],
            metadata(0),
        )
        .unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        let committed = loop {
            if let Some(event) = job.poll_terminal() {
                let RecorderTerminalEvent::Succeeded(committed) = event else {
                    panic!("unexpected still terminal event: {event:?}");
                };
                break committed;
            }
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        };
        let report = recorder_report_path(&output);
        assert_eq!(committed.report_path, report);
        assert!(image::open(&output).is_ok());
        let report_value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&report).unwrap()).unwrap();
        assert_eq!(
            report_value["media_file_name"].as_str(),
            output.file_name().and_then(|name| name.to_str())
        );
        assert_eq!(report_value["drop_policy"], "single_frame_no_drop");
        assert_eq!(report_value["pool_capacity"], 1);
        assert_eq!(report_value["queue_capacity"], 0);
        assert_eq!(report_value["first_frame"]["capture_index"], 0);
        assert_eq!(report_value["last_accepted_frame"]["visual_epoch"], 1);
        assert_eq!(report_value["counters"]["attempted"], 1);
        assert_eq!(report_value["counters"]["accepted"], 1);
        assert_eq!(report_value["counters"]["encoded"], 1);
        assert!(StillSnapshotJob::spawn(config, vec![0; 8], metadata(1)).is_err());
        std::fs::remove_file(output).unwrap();
        std::fs::remove_file(report).unwrap();
    }

    #[test]
    fn malformed_clock_metadata_is_rejected_without_unbounded_duplication() {
        assert!(!metadata_follows(Some(metadata(10)), metadata(9)));
        let mut hostile = metadata(11);
        hostile.audio_clock = Some(AudioClockStamp {
            sample_position: 0,
            sample_rate: u32::MAX,
            channels: u16::MAX,
        });
        assert!(!metadata_follows(Some(metadata(10)), hostile));
        assert!(
            metadata(10 + MAX_DUPLICATION_GAP_FRAMES + 2)
                .capture_index
                .saturating_sub(metadata(10).capture_index)
                .saturating_sub(1)
                > MAX_DUPLICATION_GAP_FRAMES
        );
    }

    #[test]
    fn publication_intent_allocates_no_slot_before_a_committed_capture_exists() {
        let destination = StableLayerId::new(9).unwrap();
        let capture = CommittedCapture {
            media_path: PathBuf::from("committed.mp4"),
            report_path: PathBuf::from("committed.mp4.recording.json"),
            target: CaptureTarget::Group(GroupId::new(4).unwrap()),
            purpose: CapturePurpose::Resample {
                destination_layer: destination,
                activate: true,
            },
            dimensions: RecorderDimensions::new(2, 2).unwrap(),
            frame_rate: RecorderFrameRate::FPS_30,
            counters: RecorderCounters::default(),
            audio_muxed: false,
        };
        assert_eq!(
            capture.publication_intent(),
            CommittedCaptureIntent::NewClipSlot {
                committed_path: PathBuf::from("committed.mp4"),
                destination_layer: destination,
                activate: true,
            }
        );
    }

    fn audio_tap(rate: u32, channels: u16) -> Arc<ProgramAudioTap> {
        ProgramAudioTap::new(rate, channels, "fixture-device".into()).unwrap()
    }

    fn config_with_audio(path: PathBuf, tap: Arc<ProgramAudioTap>) -> RecorderConfig {
        RecorderConfig {
            audio_tap: Some(tap),
            ..config(path)
        }
    }

    fn ramp_value(frame: u64) -> f32 {
        (frame % 1_000) as f32 / 1_000.0
    }

    fn push_ramp_frames(tap: &ProgramAudioTap, start_frame: u64, frames: u64) {
        let channels = tap.channels() as u64;
        let mut samples = Vec::with_capacity((frames * channels) as usize);
        for frame in start_frame..start_frame + frames {
            for _ in 0..channels {
                samples.push(ramp_value(frame));
            }
        }
        tap.push_interleaved(&mut samples.into_iter(), tap.channels());
    }

    type CapturedMux = Arc<Mutex<Option<(CaptureMuxPlan, Vec<u8>)>>>;

    /// Fixture mux: records the exact plan, snapshots the staged PCM before
    /// the worker removes it, and stands in for the committed media by
    /// copying the video temp into the mux output temp.
    fn capturing_mux_runner(captured: CapturedMux) -> MuxRunner {
        Box::new(move |plan, _shared| {
            let audio = std::fs::read(&plan.audio_temp).map_err(|error| error.to_string())?;
            std::fs::copy(&plan.video_temp, &plan.output_temp)
                .map_err(|error| error.to_string())?;
            *captured.lock().unwrap() = Some((plan.clone(), audio));
            Ok(())
        })
    }

    fn submit_frame_with_clock(recorder: &ProgramRecorder, index: u64, tap: &ProgramAudioTap) {
        let RecorderAcquire::Lease(mut lease) = recorder.try_acquire_frame() else {
            panic!("preallocated recorder lease unavailable");
        };
        lease.pixels_mut().fill(index as u8);
        let mut meta = metadata(index);
        meta.audio_clock = Some(tap.clock_stamp());
        assert_eq!(recorder.try_submit(lease, meta), RecorderSubmit::Accepted);
    }

    fn staged_temp_residue(output: &Path) -> Vec<PathBuf> {
        let parent = output
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let needle = output
            .file_name()
            .expect("test output names a file")
            .to_string_lossy()
            .into_owned();
        std::fs::read_dir(parent)
            .expect("test temp directory is listable")
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                let name = path
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default();
                name.contains(&needle) && name.ends_with(".tmp")
            })
            .collect()
    }

    fn report_json(output: &Path) -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(recorder_report_path(output)).unwrap()).unwrap()
    }

    fn cleanup_capture_outputs(output: &Path) {
        let _ = std::fs::remove_file(recorder_report_path(output));
        let _ = std::fs::remove_file(output);
    }

    #[test]
    fn program_audio_tap_clock_and_bounded_ring_overflow_law() {
        // 8 kHz mono: the duration bound (4 s = 32,000 frames) governs.
        let tap = audio_tap(8_000, 1);
        push_ramp_frames(&tap, 0, 40_000);
        assert_eq!(tap.clock_stamp().sample_position, 40_000);
        tap.begin_read_at(0);
        let mut out = Vec::new();
        let gap = tap.take_available(&mut out);
        assert_eq!(
            gap, 8_000,
            "overflow drops oldest and the reader observes the exact discarded span"
        );
        assert_eq!(out.len(), 32_000);
        assert_eq!(out[0], ramp_value(8_000));
        // Later delivery continues contiguously with no phantom gap.
        push_ramp_frames(&tap, 40_000, 100);
        out.clear();
        assert_eq!(tap.take_available(&mut out), 0);
        assert_eq!(out.len(), 100);
        assert_eq!(out[0], ramp_value(40_000));
    }

    #[test]
    fn program_audio_tap_anchor_discards_only_pre_anchor_audio() {
        let tap = audio_tap(48_000, 2);
        push_ramp_frames(&tap, 0, 4_800);
        tap.begin_read_at(1_600);
        let mut out = Vec::new();
        assert_eq!(tap.take_available(&mut out), 0);
        assert_eq!(out.len(), (4_800 - 1_600) * 2);
        assert_eq!(out[0], ramp_value(1_600));
    }

    #[test]
    fn program_audio_tap_sanitizes_input_and_refuses_the_wrong_layout() {
        let tap = audio_tap(48_000, 2);
        // A non-finite sample lands as neutral silence and the trailing
        // partial frame is discarded rather than shifting the lattice.
        tap.push_interleaved(&mut [f32::NAN, 0.25, 0.5].iter().copied(), 2);
        assert_eq!(tap.clock_stamp().sample_position, 1);
        tap.begin_read_at(0);
        let mut out = Vec::new();
        assert_eq!(tap.take_available(&mut out), 0);
        assert_eq!(out, vec![0.0, 0.25]);
        assert!(!tap.device_lost());
        tap.push_interleaved(&mut [0.0f32; 4].iter().copied(), 4);
        assert!(
            tap.device_lost(),
            "a channel-layout mismatch is a device loss, never a relattice"
        );
        assert!(ProgramAudioTap::new(4_000, 2, String::new()).is_err());
        assert!(ProgramAudioTap::new(48_000, 0, String::new()).is_err());
        assert!(ProgramAudioTap::new(48_000, 33, String::new()).is_err());
    }

    #[test]
    fn capture_mux_args_follow_the_export_audio_law() {
        let plan = CaptureMuxPlan {
            video_temp: PathBuf::from("v.tmp"),
            audio_temp: PathBuf::from("a.tmp"),
            output_temp: PathBuf::from("o.tmp"),
            sample_rate: 48_000,
            channels: 2,
            duration_secs: 0.1,
        };
        let args = capture_mux_args(&plan)
            .into_iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            args,
            vec![
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                "v.tmp",
                "-f",
                "f32le",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-i",
                "a.tmp",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-filter:a",
                "asetpts=PTS-STARTPTS,apad,atrim=end=0.100000",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-map_metadata",
                "-1",
                "-t",
                "0.100000",
                "-movflags",
                "+faststart",
                "-f",
                "mp4",
                "o.tmp",
            ]
        );
    }

    #[test]
    fn audio_mux_receives_anchored_pcm_and_the_exact_rational_duration() {
        let output = temp_path("audio-mux.mp4");
        let tap = audio_tap(48_000, 2);
        let captured: CapturedMux = Arc::new(Mutex::new(None));
        let mut recorder = ProgramRecorder::spawn_with_hooks(
            config_with_audio(output.clone(), tap.clone()),
            fixture_sink_factory(),
            capturing_mux_runner(captured.clone()),
        )
        .expect("start recorder");
        wait_until_recording(&recorder);
        // Real-time-equivalent audio for the three video frames plus one frame
        // of slack, so the finish drain never waits out its grace period.
        push_ramp_frames(&tap, 0, 6_400);
        for index in 0..3 {
            submit_fixture_frame(&recorder, index);
            std::thread::sleep(Duration::from_millis(5));
        }
        recorder.request_finish(2);
        let event = wait_recorder_terminal(&mut recorder);
        let RecorderTerminalEvent::Succeeded(committed) = event else {
            panic!("unexpected recorder terminal event: {event:?}");
        };
        assert!(committed.audio_muxed);
        let (plan, audio_bytes) = captured
            .lock()
            .unwrap()
            .take()
            .expect("the audio mux must run for an armed recording");
        assert_eq!(plan.sample_rate, 48_000);
        assert_eq!(plan.channels, 2);
        assert!((plan.duration_secs - 0.1).abs() < 1e-9);
        // The staged PCM starts at the first accepted frame's clock stamp.
        assert_eq!(audio_bytes.len(), 6_400 * 2 * 4);
        let first = f32::from_le_bytes(audio_bytes[0..4].try_into().unwrap());
        assert_eq!(first, ramp_value(0));
        let value = report_json(&output);
        assert_eq!(value["schema_version"], RECORDER_REPORT_SCHEMA_VERSION);
        assert_eq!(value["audio_not_muxed"], false);
        assert_eq!(value["audio"]["captured_frames"], 6_400);
        assert_eq!(value["audio"]["silence_gap_frames"], 0);
        assert_eq!(value["audio"]["drift_inserted_frames"], 0);
        assert_eq!(value["audio"]["drift_dropped_frames"], 0);
        assert_eq!(value["audio"]["device_lost"], false);
        assert_eq!(value["audio"]["device"], "fixture-device");
        assert!(output.exists());
        assert!(
            staged_temp_residue(&output).is_empty(),
            "a committed mux leaves no staged intermediates behind"
        );
        cleanup_capture_outputs(&output);
    }

    #[test]
    fn a_recording_without_audio_keeps_the_exact_video_only_path() {
        let output = temp_path("no-audio.mp4");
        let captured: CapturedMux = Arc::new(Mutex::new(None));
        let mut recorder = ProgramRecorder::spawn_with_hooks(
            config(output.clone()),
            fixture_sink_factory(),
            capturing_mux_runner(captured.clone()),
        )
        .expect("start recorder");
        wait_until_recording(&recorder);
        submit_fixture_frame(&recorder, 0);
        recorder.request_finish(0);
        let event = wait_recorder_terminal(&mut recorder);
        let RecorderTerminalEvent::Succeeded(committed) = event else {
            panic!("unexpected recorder terminal event: {event:?}");
        };
        assert!(!committed.audio_muxed);
        assert!(
            captured.lock().unwrap().is_none(),
            "no mux may run for a video-only recording"
        );
        let value = report_json(&output);
        assert_eq!(value["audio_not_muxed"], true);
        assert!(value.get("audio").is_none());
        cleanup_capture_outputs(&output);
    }

    #[test]
    fn operator_docs_state_the_conditional_live_audio_mux_without_contradiction() {
        const LAW: &str = "If a live capture stream is running when recording starts, the recorder muxes its bounded Program PCM tap;";
        let docs = [
            ("README", include_str!("../README.md")),
            ("remote control", include_str!("../docs/remote-control.md")),
            (
                "professional console",
                include_str!("../docs/professional-console-and-stage.md"),
            ),
        ];
        let forbidden = [
            "Recording is currently video-only",
            "starts a video-only capture",
            "The current live recorder does not mux audio",
            "The current recorder is video-only",
        ];

        for (name, document) in docs {
            let normalized = document.split_whitespace().collect::<Vec<_>>().join(" ");
            assert_eq!(
                normalized.matches(LAW).count(),
                1,
                "{name} must state the conditional mux law exactly once"
            );
            assert!(
                normalized.contains("`audio_not_muxed=true`"),
                "{name} must name video-only sidecar truth"
            );
            for contradiction in forbidden {
                assert!(
                    !normalized.contains(contradiction),
                    "{name} retains stale recorder claim: {contradiction}"
                );
            }
        }
    }

    #[test]
    fn device_loss_publishes_with_honest_padding_truth() {
        let output = temp_path("audio-loss.mp4");
        let tap = audio_tap(48_000, 2);
        let captured: CapturedMux = Arc::new(Mutex::new(None));
        let mut recorder = ProgramRecorder::spawn_with_hooks(
            config_with_audio(output.clone(), tap.clone()),
            fixture_sink_factory(),
            capturing_mux_runner(captured.clone()),
        )
        .expect("start recorder");
        wait_until_recording(&recorder);
        push_ramp_frames(&tap, 0, 1_600);
        submit_fixture_frame(&recorder, 0);
        std::thread::sleep(Duration::from_millis(5));
        submit_fixture_frame(&recorder, 1);
        tap.mark_device_lost();
        recorder.request_finish(1);
        let event = wait_recorder_terminal(&mut recorder);
        let RecorderTerminalEvent::Succeeded(committed) = event else {
            panic!("unexpected recorder terminal event: {event:?}");
        };
        assert!(committed.audio_muxed, "a lost device still publishes audio");
        let (plan, _) = captured.lock().unwrap().take().expect("mux ran");
        assert!((plan.duration_secs - 2.0 / 30.0).abs() < 1e-9);
        let value = report_json(&output);
        assert_eq!(value["audio"]["device_lost"], true);
        assert_eq!(value["audio"]["captured_frames"], 1_600);
        assert_eq!(value["audio_not_muxed"], false);
        cleanup_capture_outputs(&output);
    }

    #[test]
    fn armed_audio_that_never_delivers_publishes_explicit_full_silence() {
        let output = temp_path("audio-silent.mp4");
        let tap = audio_tap(48_000, 2);
        tap.mark_device_lost();
        let captured: CapturedMux = Arc::new(Mutex::new(None));
        let mut recorder = ProgramRecorder::spawn_with_hooks(
            config_with_audio(output.clone(), tap.clone()),
            fixture_sink_factory(),
            capturing_mux_runner(captured.clone()),
        )
        .expect("start recorder");
        wait_until_recording(&recorder);
        submit_frame_with_clock(&recorder, 0, &tap);
        std::thread::sleep(Duration::from_millis(5));
        submit_frame_with_clock(&recorder, 1, &tap);
        recorder.request_finish(1);
        let event = wait_recorder_terminal(&mut recorder);
        let RecorderTerminalEvent::Succeeded(_) = event else {
            panic!("unexpected recorder terminal event: {event:?}");
        };
        let (_, audio_bytes) = captured.lock().unwrap().take().expect("mux ran");
        // Two encoded frames at 30 fps demand 3,200 stereo frames of audio;
        // the never-delivering source becomes explicit zeroed PCM, never an
        // empty raw input the mux could misread.
        assert_eq!(audio_bytes.len(), 3_200 * 2 * 4);
        assert!(audio_bytes.iter().all(|byte| *byte == 0));
        let value = report_json(&output);
        assert_eq!(value["audio"]["captured_frames"], 0);
        assert_eq!(value["audio"]["silence_gap_frames"], 3_200);
        assert_eq!(value["audio"]["device_lost"], true);
        cleanup_capture_outputs(&output);
    }

    #[test]
    fn cancel_removes_every_staged_temp_including_audio_and_mux() {
        let output = temp_path("audio-cancel.mp4");
        let tap = audio_tap(48_000, 2);
        let captured: CapturedMux = Arc::new(Mutex::new(None));
        let mut recorder = ProgramRecorder::spawn_with_hooks(
            config_with_audio(output.clone(), tap.clone()),
            fixture_sink_factory(),
            capturing_mux_runner(captured.clone()),
        )
        .expect("start recorder");
        wait_until_recording(&recorder);
        push_ramp_frames(&tap, 0, 1_600);
        submit_fixture_frame(&recorder, 0);
        recorder.cancel();
        let event = wait_recorder_terminal(&mut recorder);
        assert!(matches!(event, RecorderTerminalEvent::Cancelled));
        assert!(!output.exists());
        assert!(!recorder_report_path(&output).exists());
        assert!(
            staged_temp_residue(&output).is_empty(),
            "cancel must remove the video, report, audio, and mux temps"
        );
    }

    #[test]
    fn mux_failure_fails_the_recording_and_removes_the_temps() {
        let output = temp_path("audio-mux-fail.mp4");
        let tap = audio_tap(48_000, 2);
        tap.mark_device_lost();
        let mut recorder = ProgramRecorder::spawn_with_hooks(
            config_with_audio(output.clone(), tap.clone()),
            fixture_sink_factory(),
            Box::new(|_, _| Err("fixture mux refused".to_string())),
        )
        .expect("start recorder");
        wait_until_recording(&recorder);
        submit_frame_with_clock(&recorder, 0, &tap);
        recorder.request_finish(0);
        let event = wait_recorder_terminal(&mut recorder);
        let RecorderTerminalEvent::Failed(error) = event else {
            panic!("unexpected recorder terminal event: {event:?}");
        };
        assert!(error.contains("fixture mux refused"));
        assert_eq!(recorder.snapshot().counters.encoder_failures, 1);
        assert!(!output.exists());
        assert!(!recorder_report_path(&output).exists());
        assert!(staged_temp_residue(&output).is_empty());
    }

    #[test]
    fn drift_correction_is_bounded_counted_and_direction_correct() {
        let output = temp_path("drift.mp4");
        let audio_temp = sibling_temp_path(&output, "audio").unwrap();
        let mux_temp = sibling_temp_path(&output, "mux").unwrap();
        reserve_temp(&audio_temp).unwrap();
        reserve_temp(&mux_temp).unwrap();
        let tap = audio_tap(48_000, 1);
        let mut capture =
            RecorderAudioCapture::open(tap.clone(), audio_temp.clone(), mux_temp.clone()).unwrap();
        let stamp = |index: u64, position: u64| RecorderFrameMetadata {
            capture_index: index,
            capture_time_ns: index,
            program_time_ns: index,
            visual_epoch: 1,
            program_frozen: false,
            media_frozen: false,
            blackout: false,
            audio_clock: Some(AudioClockStamp {
                sample_position: position,
                sample_rate: 48_000,
                channels: 1,
            }),
        };
        // A mismatched clock layout is refused at the anchor, never guessed.
        let mut hostile = stamp(0, 0);
        hostile.audio_clock = Some(AudioClockStamp {
            sample_position: 0,
            sample_rate: 44_100,
            channels: 1,
        });
        assert!(capture.anchor_at(hostile).is_err());
        capture.anchor_at(stamp(0, 0)).unwrap();
        // In sync at exactly one program second: nothing to correct.
        capture
            .correct_drift(stamp(30, 48_000), RecorderFrameRate::FPS_30)
            .unwrap();
        assert_eq!(capture.pending_drop_frames, 0);
        assert_eq!(capture.drift_inserted_frames, 0);
        // Fast device clock: +20,000 frames beyond two program seconds trips
        // the quarter-second threshold and schedules one bounded drop.
        capture
            .correct_drift(stamp(60, 96_000 + 20_000), RecorderFrameRate::FPS_30)
            .unwrap();
        assert_eq!(capture.pending_drop_frames, 20_000);
        // The pending drop consumes real frames at the next pump.
        push_ramp_frames(&tap, 0, 30_000);
        capture.pump().unwrap();
        assert_eq!(capture.drift_dropped_frames, 20_000);
        assert_eq!(capture.captured_frames, 10_000);
        // With the correction accounted, the same steady offset is no drift.
        capture
            .correct_drift(stamp(61, 97_600 + 20_000), RecorderFrameRate::FPS_30)
            .unwrap();
        assert_eq!(capture.pending_drop_frames, 0);
        // Slow device clock: −20,000 frames inserts explicit counted silence.
        let mut slow =
            RecorderAudioCapture::open(tap.clone(), audio_temp.clone(), mux_temp.clone()).unwrap();
        slow.anchor_at(stamp(0, 0)).unwrap();
        slow.correct_drift(stamp(30, 48_000 - 20_000), RecorderFrameRate::FPS_30)
            .unwrap();
        assert_eq!(slow.drift_inserted_frames, 20_000);
        assert_eq!(slow.pending_drop_frames, 0);
        drop(capture);
        drop(slow);
        cleanup_paths([audio_temp.as_path(), mux_temp.as_path()]);
    }

    /// The tranche's gate fixture: a real recording of a clocked test signal
    /// whose audio stream, duration, and A/V offset are verified by ffprobe.
    /// Opt-in like `effects_audit`: it requires the ffmpeg and ffprobe CLIs.
    #[test]
    #[ignore = "requires the ffmpeg and ffprobe CLIs on this host"]
    fn recorder_audio_mux_end_to_end_duration_and_offset_verified_by_ffprobe() {
        let output = temp_path("audio-e2e.mp4");
        let tap = audio_tap(48_000, 2);
        let mut recorder = ProgramRecorder::spawn(RecorderConfig {
            dimensions: RecorderDimensions::new(64, 64).unwrap(),
            frame_rate: RecorderFrameRate::FPS_30,
            output_path: output.clone(),
            target: CaptureTarget::Program,
            purpose: CapturePurpose::External,
            audio_tap: Some(tap.clone()),
        })
        .expect("start recorder");
        wait_until_recording(&recorder);
        // Three seconds: 90 CFR frames, each paired with its exact 1,600
        // stereo frames of clocked ramp signal.
        for index in 0..90u64 {
            push_ramp_frames(&tap, index * 1_600, 1_600);
            let deadline = Instant::now() + Duration::from_secs(2);
            loop {
                if let RecorderAcquire::Lease(mut lease) = recorder.try_acquire_frame() {
                    lease.pixels_mut().fill((index % 255) as u8);
                    if recorder.try_submit(lease, metadata(index)) == RecorderSubmit::Accepted {
                        break;
                    }
                }
                assert!(Instant::now() < deadline, "recorder never accepted frame");
                std::thread::sleep(Duration::from_millis(2));
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        recorder.request_finish(89);
        let deadline = Instant::now() + Duration::from_secs(60);
        let event = loop {
            if let Some(event) = recorder.poll_terminal() {
                break event;
            }
            assert!(Instant::now() < deadline, "recorder terminal timed out");
            std::thread::sleep(Duration::from_millis(10));
        };
        let RecorderTerminalEvent::Succeeded(committed) = event else {
            panic!("unexpected recorder terminal event: {event:?}");
        };
        assert!(committed.audio_muxed);
        let probe = Command::new(crate::host_paths::ffprobe())
            .args([
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,codec_name,start_time,duration",
                "-of",
                "json",
            ])
            .arg(&output)
            .output()
            .expect("run ffprobe");
        assert!(probe.status.success(), "ffprobe failed: {probe:?}");
        let value: serde_json::Value = serde_json::from_slice(&probe.stdout).unwrap();
        let streams = value["streams"].as_array().expect("ffprobe streams");
        let field = |stream: &serde_json::Value, key: &str| -> f64 {
            stream[key]
                .as_str()
                .and_then(|text| text.parse::<f64>().ok())
                .unwrap_or(f64::NAN)
        };
        let video = streams
            .iter()
            .find(|stream| stream["codec_type"] == "video")
            .expect("video stream present");
        let audio = streams
            .iter()
            .find(|stream| stream["codec_type"] == "audio")
            .expect("audio stream present");
        assert_eq!(audio["codec_name"], "aac");
        let video_duration = field(video, "duration");
        let audio_duration = field(audio, "duration");
        assert!(
            (video_duration - 3.0).abs() < 0.05,
            "video duration {video_duration} strayed from 3.0"
        );
        assert!(
            (audio_duration - 3.0).abs() < 0.2,
            "audio duration {audio_duration} strayed from 3.0"
        );
        let offset = field(audio, "start_time") - field(video, "start_time");
        assert!(
            offset.abs() < 0.06,
            "A/V start offset {offset} exceeds the container tolerance"
        );
        let value = report_json(&output);
        assert_eq!(value["audio_not_muxed"], false);
        assert_eq!(value["audio"]["device_lost"], false);
        cleanup_capture_outputs(&output);
    }
}
