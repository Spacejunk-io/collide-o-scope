//! Bounded, non-blocking live capture and safe artifact publication.
//!
//! The render thread owns only fixed-capacity channels, atomics, and frame
//! leases allocated by the worker before it reports `Recording`. It never
//! waits for FFmpeg or filesystem I/O. A successful terminal event is the
//! sole authority for importing or publishing a new clip slot.

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
pub const RECORDER_REPORT_SCHEMA_VERSION: u16 = 1;
pub const RECORDER_MAX_REPORT_BYTES: usize = 64 * 1024;

const RECORDER_EVENT_QUEUE_CAPACITY: usize = 1;
const RECORDER_POLL_INTERVAL: Duration = Duration::from_millis(10);
const RECORDER_FINISH_TIMEOUT: Duration = Duration::from_secs(30);
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
        Self::spawn_with_factory(config, Box::new(spawn_ffmpeg_sink))
    }

    fn spawn_with_factory(config: RecorderConfig, factory: SinkFactory) -> Result<Self, String> {
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

struct TempArtifactGuard {
    media: PathBuf,
    report: PathBuf,
    armed: bool,
}

impl TempArtifactGuard {
    fn new(media: PathBuf, report: PathBuf) -> Self {
        Self {
            media,
            report,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TempArtifactGuard {
    fn drop(&mut self) {
        if self.armed {
            cleanup_paths([self.media.as_path(), self.report.as_path()]);
        }
    }
}

fn recorder_worker(
    config: RecorderConfig,
    frame_bytes: usize,
    pool_tx: SyncSender<Vec<u8>>,
    work_rx: Receiver<RecorderWorkItem>,
    event_tx: SyncSender<RecorderTerminalEvent>,
    shared: Arc<RecorderShared>,
    factory: SinkFactory,
) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_recorder_worker(&config, frame_bytes, &pool_tx, &work_rx, &shared, factory)
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
    let mut temp_guard = TempArtifactGuard::new(temp_path.clone(), report_temp.clone());
    reserve_temp(&report_temp)?;
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
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
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
    if let Err(error) = sink.finish() {
        shared
            .counters
            .encoder_failures
            .fetch_add(1, Ordering::Relaxed);
        return Err(error);
    }
    sync_path(&temp_path)?;
    let counters = shared.counters.snapshot();
    let requested_final_capture_index = shared.finish_capture_index.load(Ordering::Acquire);
    let report = RecorderReport::new(
        config,
        counters,
        first_metadata.expect("guarded metadata"),
        last_metadata.expect("guarded metadata"),
        requested_final_capture_index,
    );
    write_report_temp(&report_temp, &report)?;
    commit_artifact_pair_linearized(
        &shared.publication,
        &temp_path,
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
        frame_rate: config.frame_rate,
        counters,
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
            audio_not_muxed: true,
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
    let mut temp_guard = TempArtifactGuard::new(media_temp.clone(), report_temp.clone());
    reserve_temp(&report_temp)?;
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
}
