//! Bounded, privacy-safe evidence for the seconds immediately before a fault.
//!
//! The live side of this recorder can only enqueue fixed-shape facts. It has no
//! API for logging a message, a path, an HTTP header, authored text, or media
//! bytes. JSON serialization and filesystem I/O happen on a bounded helper
//! thread. The active file is never treated as durable evidence: a rotation is
//! published only after the complete marker and file contents have reached the
//! filesystem. Consequently, a forced termination can damage only the active
//! rotation; the preceding completed rotations remain independent and readable.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

const FORMAT_SCHEMA_VERSION: u16 = 1;
const DIRECTORY_NAME: &str = "flight-recorder-v1";
const ACTIVE_PREFIX: &str = "active-";
const COMPLETED_PREFIX: &str = "completed-";
const FILE_SUFFIX: &str = ".ndjson";
const QUEUE_CAPACITY: usize = 512;
const RETAINED_COMPLETED_ROTATIONS: usize = 2;
const MIN_ROTATION_PERIOD: Duration = Duration::from_secs(30);
const MAX_ROTATION_PERIOD: Duration = Duration::from_secs(60);
const DEFAULT_ROTATION_PERIOD: Duration = Duration::from_secs(45);
const DEFAULT_TOTAL_BYTE_CAP: u64 = 3 * 1024 * 1024;
const MIN_TOTAL_BYTE_CAP: u64 = 3 * 64 * 1024;
const COMPLETE_RECORD_RESERVE: u64 = 512;

/// A fixed SHA-256 value. Callers may supply only the digest, never the source
/// bytes from which it was calculated.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Digest32(pub [u8; 32]);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostOs {
    Windows,
    MacOs,
    Linux,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpuArchitecture {
    X86_64,
    Aarch64,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuBackend {
    Vulkan,
    Metal,
    Dx12,
    Gl,
    BrowserWebGpu,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PresentModeFact {
    Fifo,
    FifoRelaxed,
    Immediate,
    Mailbox,
    AutoVsync,
    AutoNoVsync,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrecisionFact {
    Rgba8,
    Rgba16Float,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuFact {
    pub backend: GpuBackend,
    pub pci_vendor_id: u32,
    pub pci_device_id: u32,
    pub driver_version: [u32; 4],
    pub timestamp_query_supported: bool,
    pub precision: PrecisionFact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DisplayFact {
    pub raster_width: u32,
    pub raster_height: u32,
    pub refresh_millihertz: u32,
    pub present_mode: PresentModeFact,
    pub fullscreen: bool,
}

/// Hardware and display facts use numeric identifiers instead of adapter,
/// monitor, user, or machine names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostFact {
    pub os: HostOs,
    pub os_build: u32,
    pub architecture: CpuArchitecture,
    pub logical_cpu_count: u16,
    pub gpu: GpuFact,
    pub display: DisplayFact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageKind {
    SourcePrepare,
    CreativeComposition,
    TemporalMotion,
    MoshVhs,
    AudienceResolve,
    Submission,
    Decode,
    Upload,
    WholeFrame,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimingDomain {
    Cpu,
    Gpu,
    EngineAction,
    PhysicalFixture,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageFact {
    pub stage: StageKind,
    pub domain: TimingDomain,
    pub sample_count: u32,
    pub p50_nanoseconds: u64,
    pub p95_nanoseconds: u64,
    pub p99_nanoseconds: u64,
    pub dropped_samples: u64,
    pub submission_generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerKind {
    VideoDecode,
    StillDecode,
    Proxy,
    Mosh,
    Ntsc,
    Export,
    Audio,
    WebControl,
    FlightRecorder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerState {
    Starting,
    Ready,
    Busy,
    Backpressured,
    Recovering,
    Stopped,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerFact {
    pub worker: WorkerKind,
    pub state: WorkerState,
    pub queue_depth: u16,
    pub queue_capacity: u16,
    pub completed_jobs: u64,
    pub dropped_jobs: u64,
    pub restart_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionSource {
    Browser,
    Phone,
    Native,
    Midi,
    Osc,
    Automation,
    Replay,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionDisposition {
    Presented,
    Coalesced,
    Refused,
    Superseded,
    Quantized,
    NotYetPresented,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionFact {
    pub sequence: u64,
    pub source: ActionSource,
    pub disposition: ActionDisposition,
    pub ingress_to_apply_nanoseconds: Option<u64>,
    pub apply_to_submit_nanoseconds: Option<u64>,
    pub submission_generation: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorDomain {
    Renderer,
    Surface,
    Device,
    Decode,
    Upload,
    Audio,
    Proxy,
    Mosh,
    Ntsc,
    WebControl,
    TlsIdentity,
    Export,
    Recovery,
    FlightRecorder,
}

/// Stable categories replace arbitrary `Display`/`Debug` error messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    Unavailable,
    InvalidInput,
    Timeout,
    Backpressure,
    ResourceExhausted,
    DeviceLost,
    SurfaceLost,
    PermissionDenied,
    IntegrityFailure,
    Unsupported,
    Cancelled,
    Internal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorFact {
    pub domain: ErrorDomain,
    pub code: ErrorCode,
    pub retryable: bool,
    pub occurrence_count: u32,
}

/// Untrusted diagnostic material may cross this one-way privacy boundary, but
/// none of it is retained in the resulting [`ErrorFact`]. The wrapper has no
/// formatting implementation, which also keeps it out of accidental logs.
pub struct SensitiveDiagnostic<'a> {
    _filesystem_path: Option<&'a Path>,
    _access_token: Option<&'a str>,
    _cookie_header: Option<&'a str>,
    _media_bytes: Option<&'a [u8]>,
    _controller_secret: Option<&'a str>,
    _authored_text: Option<&'a str>,
}

impl<'a> SensitiveDiagnostic<'a> {
    pub const fn new(
        filesystem_path: Option<&'a Path>,
        access_token: Option<&'a str>,
        cookie_header: Option<&'a str>,
        media_bytes: Option<&'a [u8]>,
        controller_secret: Option<&'a str>,
        authored_text: Option<&'a str>,
    ) -> Self {
        Self {
            _filesystem_path: filesystem_path,
            _access_token: access_token,
            _cookie_header: cookie_header,
            _media_bytes: media_bytes,
            _controller_secret: controller_secret,
            _authored_text: authored_text,
        }
    }
}

impl ErrorFact {
    /// Classifies an error while intentionally destroying all sensitive source
    /// detail. Only the closed domain/code vocabulary crosses the boundary.
    pub fn redact(
        domain: ErrorDomain,
        code: ErrorCode,
        retryable: bool,
        occurrence_count: u32,
        _sensitive: SensitiveDiagnostic<'_>,
    ) -> Self {
        Self {
            domain,
            code,
            retryable,
            occurrence_count,
        }
    }
}

/// Aggregate, already-redacted source identity. Individual file names and
/// paths are deliberately unrepresentable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentIdentityFact {
    /// Legacy wire name for the canonical authored-world/history fingerprint.
    /// It conservatively partitions identical patch plans when other authored
    /// state (for example StageMap or controller profile) differs.
    pub patch_plan_digest: Digest32,
    pub source_set_digest: Digest32,
    pub source_count: u16,
}

/// The last valid bounded resource ledger. It is repeated in each subsequent
/// rotation header so a quiet failure still retains the latest ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceLedgerFact {
    pub plan_digest: Digest32,
    pub full_frame_passes: u16,
    pub texture_lookups_per_pixel: u32,
    pub live_textures: u16,
    pub live_buffers: u16,
    pub live_bind_groups: u16,
    pub gpu_bytes: u64,
    pub cpu_bytes: u64,
    pub budget_gpu_bytes: u64,
    pub budget_cpu_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationKeyFact {
    pub backend: GpuBackend,
    pub pci_vendor_id: u32,
    pub pci_device_id: u32,
    pub driver_version: [u32; 4],
    pub raster_width: u32,
    pub raster_height: u32,
    pub refresh_millihertz: u32,
    pub present_mode: PresentModeFact,
    pub precision: PrecisionFact,
    /// Same conservative authored-world/history fingerprint carried by
    /// `ContentIdentityFact::patch_plan_digest`; not a patch-only digest.
    pub patch_plan_digest: Digest32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterCalibrationFact {
    pub key: CalibrationKeyFact,
    pub sample_count: u32,
    pub source_prepare_p95_nanoseconds: u64,
    pub creative_composition_p95_nanoseconds: u64,
    pub temporal_motion_p95_nanoseconds: u64,
    pub mosh_vhs_p95_nanoseconds: u64,
    pub audience_resolve_p95_nanoseconds: u64,
    pub submission_p95_nanoseconds: u64,
    pub critical_stage: StageKind,
    pub critical_stage_p95_nanoseconds: u64,
    pub full_frame_passes: u16,
    pub texture_lookups_per_pixel: u32,
    pub gpu_bytes: u64,
    /// This receipt may explain a measured critical path. It never authorizes
    /// an automatic quality, precision, resolution, or effect change.
    pub advisory_only: bool,
}

impl AdapterCalibrationFact {
    pub fn from_p95(
        key: CalibrationKeyFact,
        sample_count: u32,
        stages: [(StageKind, u64); 6],
        full_frame_passes: u16,
        texture_lookups_per_pixel: u32,
        gpu_bytes: u64,
    ) -> Self {
        let (critical_stage, critical_stage_p95_nanoseconds) = stages
            .iter()
            .copied()
            .max_by_key(|(_, duration)| *duration)
            .unwrap_or((StageKind::Submission, 0));
        Self {
            key,
            sample_count,
            source_prepare_p95_nanoseconds: stages[0].1,
            creative_composition_p95_nanoseconds: stages[1].1,
            temporal_motion_p95_nanoseconds: stages[2].1,
            mosh_vhs_p95_nanoseconds: stages[3].1,
            audience_resolve_p95_nanoseconds: stages[4].1,
            submission_p95_nanoseconds: stages[5].1,
            critical_stage,
            critical_stage_p95_nanoseconds,
            full_frame_passes,
            texture_lookups_per_pixel,
            gpu_bytes,
            advisory_only: true,
        }
    }
}

/// Every writable payload is a fixed, typed fact. This enum intentionally has
/// no `String`, `PathBuf`, `Vec<u8>`, catch-all JSON, or arbitrary code field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "facts", rename_all = "snake_case")]
pub enum FlightEvent {
    Host(HostFact),
    Stage(StageFact),
    Worker(WorkerFact),
    Action(ActionFact),
    Error(ErrorFact),
    ContentIdentity(ContentIdentityFact),
    ResourceLedger(ResourceLedgerFact),
    AdapterCalibration(AdapterCalibrationFact),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlightRecorderConfig {
    rotation_period: Duration,
    total_byte_cap: u64,
}

impl Default for FlightRecorderConfig {
    fn default() -> Self {
        Self {
            rotation_period: DEFAULT_ROTATION_PERIOD,
            total_byte_cap: DEFAULT_TOTAL_BYTE_CAP,
        }
    }
}

impl FlightRecorderConfig {
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "host embedders may choose bounded recorder limits"
        )
    )]
    pub fn new(rotation_period: Duration, total_byte_cap: u64) -> io::Result<Self> {
        let config = Self {
            rotation_period,
            total_byte_cap,
        };
        config.validate()?;
        Ok(config)
    }

    pub const fn rotation_period(self) -> Duration {
        self.rotation_period
    }

    pub const fn total_byte_cap(self) -> u64 {
        self.total_byte_cap
    }

    fn validate(self) -> io::Result<()> {
        if !(MIN_ROTATION_PERIOD..=MAX_ROTATION_PERIOD).contains(&self.rotation_period) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "flight-recorder rotation must be between 30 and 60 seconds",
            ));
        }
        if self.total_byte_cap < MIN_TOTAL_BYTE_CAP {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "flight-recorder byte cap is too small for three bounded rotations",
            ));
        }
        Ok(())
    }

    fn per_rotation_byte_cap(self) -> u64 {
        self.total_byte_cap / (RETAINED_COMPLETED_ROTATIONS as u64 + 1)
    }

    #[cfg(test)]
    fn for_test(rotation_period: Duration, total_byte_cap: u64) -> Self {
        Self {
            rotation_period,
            total_byte_cap,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordDisposition {
    Queued { sequence: u64 },
    DroppedFull { sequence: u64 },
    WorkerUnavailable { sequence: u64 },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FlightRecorderStats {
    pub queued: u64,
    pub dropped_full: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FlightRecorderShutdown {
    pub completed_rotations: u64,
    pub events_written: u64,
    pub queued: u64,
    pub dropped_full: u64,
}

#[derive(Debug, Clone, Copy)]
struct QueuedEvent {
    sequence: u64,
    elapsed_microseconds: u64,
    event: FlightEvent,
}

/// Nonblocking producer handle. Dropping or explicitly shutting it down drains
/// the bounded queue and publishes the active rotation.
pub struct FlightRecorder {
    sender: Option<SyncSender<QueuedEvent>>,
    worker: Option<JoinHandle<io::Result<FlightRecorderShutdown>>>,
    started_at: Instant,
    next_sequence: AtomicU64,
    queued: Arc<AtomicU64>,
    dropped_full: Arc<AtomicU64>,
}

impl FlightRecorder {
    /// Starts the recorder below the application's per-user state root.
    #[cfg_attr(
        test,
        allow(
            dead_code,
            reason = "App tests deliberately disable the operator's real per-user recorder"
        )
    )]
    pub fn start() -> io::Result<Self> {
        Self::start_at(
            &crate::host_paths::state_root(),
            FlightRecorderConfig::default(),
        )
    }

    /// Alternate state root for tests and host embedding. The recorder always
    /// adds its own versioned directory below this root.
    pub fn start_at(state_root: &Path, config: FlightRecorderConfig) -> io::Result<Self> {
        #[cfg(not(test))]
        config.validate()?;
        #[cfg(test)]
        if config.rotation_period.is_zero() || config.total_byte_cap < 3 * 4096 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid flight-recorder test bounds",
            ));
        }

        let directory = recorder_directory_at(state_root);
        prepare_private_directory(&directory)?;
        let first_rotation = prepare_artifacts(&directory)?;
        let build_identity = crate::build_identity::current().snapshot();
        let cache = SnapshotCache::default();
        let active = RotationFile::open(
            &directory,
            first_rotation,
            config.per_rotation_byte_cap(),
            &build_identity,
            cache,
        )?;

        let (sender, receiver) = mpsc::sync_channel(QUEUE_CAPACITY);
        let queued = Arc::new(AtomicU64::new(0));
        let dropped_full = Arc::new(AtomicU64::new(0));
        let writer_queued = queued.clone();
        let writer_dropped = dropped_full.clone();
        let worker = thread::Builder::new()
            .name("flight-recorder".to_owned())
            .spawn(move || {
                writer_loop(
                    receiver,
                    directory,
                    config,
                    build_identity,
                    active,
                    writer_queued,
                    writer_dropped,
                )
            })?;

        Ok(Self {
            sender: Some(sender),
            worker: Some(worker),
            started_at: Instant::now(),
            next_sequence: AtomicU64::new(1),
            queued,
            dropped_full,
        })
    }

    /// Attempts one bounded enqueue and returns immediately. Queue pressure is
    /// explicit evidence, never an invitation to block the render thread.
    pub fn try_record(&self, event: FlightEvent) -> RecordDisposition {
        let sequence = self.next_sequence.fetch_add(1, Ordering::Relaxed);
        let elapsed_microseconds = self
            .started_at
            .elapsed()
            .as_micros()
            .min(u128::from(u64::MAX)) as u64;
        let queued = QueuedEvent {
            sequence,
            elapsed_microseconds,
            event,
        };
        let Some(sender) = self.sender.as_ref() else {
            return RecordDisposition::WorkerUnavailable { sequence };
        };
        match sender.try_send(queued) {
            Ok(()) => {
                self.queued.fetch_add(1, Ordering::Relaxed);
                RecordDisposition::Queued { sequence }
            }
            Err(TrySendError::Full(_)) => {
                self.dropped_full.fetch_add(1, Ordering::Relaxed);
                RecordDisposition::DroppedFull { sequence }
            }
            Err(TrySendError::Disconnected(_)) => RecordDisposition::WorkerUnavailable { sequence },
        }
    }

    pub fn stats(&self) -> FlightRecorderStats {
        FlightRecorderStats {
            queued: self.queued.load(Ordering::Relaxed),
            dropped_full: self.dropped_full.load(Ordering::Relaxed),
        }
    }

    /// Drains queued facts and durably publishes the final active rotation.
    /// This may wait for filesystem I/O and is therefore a shutdown operation,
    /// not a live-frame operation.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "explicit shutdown is a host-embedding seam; Drop uses the same drain law"
        )
    )]
    pub fn shutdown(mut self) -> io::Result<FlightRecorderShutdown> {
        self.finish()
    }

    fn finish(&mut self) -> io::Result<FlightRecorderShutdown> {
        self.sender.take();
        let Some(worker) = self.worker.take() else {
            return Ok(FlightRecorderShutdown {
                queued: self.queued.load(Ordering::Relaxed),
                dropped_full: self.dropped_full.load(Ordering::Relaxed),
                ..FlightRecorderShutdown::default()
            });
        };
        worker
            .join()
            .map_err(|_| io::Error::other("flight-recorder worker panicked"))?
    }
}

impl Drop for FlightRecorder {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
struct SnapshotCache {
    host: Option<HostFact>,
    content_identity: Option<ContentIdentityFact>,
    resource_ledger: Option<ResourceLedgerFact>,
    adapter_calibration: Option<AdapterCalibrationFact>,
}

impl SnapshotCache {
    fn observe(&mut self, event: FlightEvent) {
        match event {
            FlightEvent::Host(fact) => self.host = Some(fact),
            FlightEvent::ContentIdentity(fact) => self.content_identity = Some(fact),
            FlightEvent::ResourceLedger(fact) => self.resource_ledger = Some(fact),
            FlightEvent::AdapterCalibration(fact) => self.adapter_calibration = Some(fact),
            FlightEvent::Stage(_)
            | FlightEvent::Worker(_)
            | FlightEvent::Action(_)
            | FlightEvent::Error(_) => {}
        }
    }
}

#[derive(Serialize)]
#[serde(tag = "record", rename_all = "snake_case")]
#[allow(
    clippy::large_enum_variant,
    reason = "a WireRecord is serialized immediately and never stored in a collection; boxing SnapshotCache would add allocation to every rotation header without reducing retained memory"
)]
enum WireRecord<'a> {
    Header {
        schema_version: u16,
        rotation: u64,
        build_identity: &'a crate::build_identity::BuildIdentitySnapshot,
        snapshots: SnapshotCache,
    },
    Event {
        schema_version: u16,
        sequence: u64,
        elapsed_microseconds: u64,
        event: &'a FlightEvent,
    },
    Complete {
        schema_version: u16,
        event_count: u64,
        dropped_full: u64,
    },
}

struct RotationFile {
    directory: PathBuf,
    active_path: PathBuf,
    rotation: u64,
    file: Option<File>,
    bytes_written: u64,
    event_count: u64,
    byte_cap: u64,
}

impl RotationFile {
    fn open(
        directory: &Path,
        rotation: u64,
        byte_cap: u64,
        build_identity: &crate::build_identity::BuildIdentitySnapshot,
        snapshots: SnapshotCache,
    ) -> io::Result<Self> {
        let active_path = directory.join(format!("{ACTIVE_PREFIX}{rotation:020}{FILE_SUFFIX}"));
        let mut file = open_private_new(&active_path)?;
        let header = encode_line(&WireRecord::Header {
            schema_version: FORMAT_SCHEMA_VERSION,
            rotation,
            build_identity,
            snapshots,
        })?;
        if header.len() as u64 + COMPLETE_RECORD_RESERVE > byte_cap {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "flight-recorder build header exceeds its rotation byte cap",
            ));
        }
        file.write_all(&header)?;
        file.sync_all()?;
        Ok(Self {
            directory: directory.to_path_buf(),
            active_path,
            rotation,
            file: Some(file),
            bytes_written: header.len() as u64,
            event_count: 0,
            byte_cap,
        })
    }

    fn can_append(&self, encoded_len: usize) -> bool {
        self.bytes_written
            .saturating_add(encoded_len as u64)
            .saturating_add(COMPLETE_RECORD_RESERVE)
            <= self.byte_cap
    }

    fn append_encoded(&mut self, encoded: &[u8]) -> io::Result<()> {
        if !self.can_append(encoded.len()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "flight-recorder event exceeds an empty bounded rotation",
            ));
        }
        self.file_mut()?.write_all(encoded)?;
        self.bytes_written += encoded.len() as u64;
        self.event_count += 1;
        Ok(())
    }

    fn seal(mut self, dropped_full: u64) -> io::Result<PathBuf> {
        let complete = encode_line(&WireRecord::Complete {
            schema_version: FORMAT_SCHEMA_VERSION,
            event_count: self.event_count,
            dropped_full,
        })?;
        if self.bytes_written.saturating_add(complete.len() as u64) > self.byte_cap {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "flight-recorder completion record exceeded its reserved space",
            ));
        }
        let mut file = self.file.take().ok_or_else(|| {
            io::Error::new(io::ErrorKind::BrokenPipe, "flight-recorder file is closed")
        })?;
        file.write_all(&complete)?;
        file.sync_all()?;
        drop(file);

        let completed_path = self.directory.join(format!(
            "{COMPLETED_PREFIX}{:020}{FILE_SUFFIX}",
            self.rotation
        ));
        fs::rename(&self.active_path, &completed_path)?;
        sync_directory(&self.directory)?;
        Ok(completed_path)
    }

    fn file_mut(&mut self) -> io::Result<&mut File> {
        self.file.as_mut().ok_or_else(|| {
            io::Error::new(io::ErrorKind::BrokenPipe, "flight-recorder file is closed")
        })
    }
}

fn writer_loop(
    receiver: Receiver<QueuedEvent>,
    directory: PathBuf,
    config: FlightRecorderConfig,
    build_identity: crate::build_identity::BuildIdentitySnapshot,
    mut active: RotationFile,
    queued: Arc<AtomicU64>,
    dropped_full: Arc<AtomicU64>,
) -> io::Result<FlightRecorderShutdown> {
    let mut cache = SnapshotCache::default();
    let mut deadline = Instant::now() + config.rotation_period;
    let mut rotations = 0_u64;
    let mut events_written = 0_u64;

    loop {
        let wait = deadline.saturating_duration_since(Instant::now());
        match receiver.recv_timeout(wait) {
            Ok(queued_event) => {
                if Instant::now() >= deadline {
                    active.seal(dropped_full.load(Ordering::Relaxed))?;
                    rotations += 1;
                    retain_latest_completed(&directory)?;
                    active = RotationFile::open(
                        &directory,
                        active_rotation_after(&directory)?,
                        config.per_rotation_byte_cap(),
                        &build_identity,
                        cache,
                    )?;
                    deadline = Instant::now() + config.rotation_period;
                }

                let encoded = encode_line(&WireRecord::Event {
                    schema_version: FORMAT_SCHEMA_VERSION,
                    sequence: queued_event.sequence,
                    elapsed_microseconds: queued_event.elapsed_microseconds,
                    event: &queued_event.event,
                })?;
                if !active.can_append(encoded.len()) {
                    active.seal(dropped_full.load(Ordering::Relaxed))?;
                    rotations += 1;
                    retain_latest_completed(&directory)?;
                    active = RotationFile::open(
                        &directory,
                        active_rotation_after(&directory)?,
                        config.per_rotation_byte_cap(),
                        &build_identity,
                        cache,
                    )?;
                    deadline = Instant::now() + config.rotation_period;
                }
                active.append_encoded(&encoded)?;
                cache.observe(queued_event.event);
                events_written += 1;
            }
            Err(RecvTimeoutError::Timeout) => {
                active.seal(dropped_full.load(Ordering::Relaxed))?;
                rotations += 1;
                retain_latest_completed(&directory)?;
                active = RotationFile::open(
                    &directory,
                    active_rotation_after(&directory)?,
                    config.per_rotation_byte_cap(),
                    &build_identity,
                    cache,
                )?;
                deadline = Instant::now() + config.rotation_period;
            }
            Err(RecvTimeoutError::Disconnected) => {
                active.seal(dropped_full.load(Ordering::Relaxed))?;
                rotations += 1;
                retain_latest_completed(&directory)?;
                return Ok(FlightRecorderShutdown {
                    completed_rotations: rotations,
                    events_written,
                    queued: queued.load(Ordering::Relaxed),
                    dropped_full: dropped_full.load(Ordering::Relaxed),
                });
            }
        }
    }
}

fn encode_line(record: &WireRecord<'_>) -> io::Result<Vec<u8>> {
    let mut encoded = serde_json::to_vec(record)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    encoded.push(b'\n');
    Ok(encoded)
}

/// Versioned directory containing only recorder-owned artifacts.
#[allow(
    dead_code,
    reason = "support tooling discovers completed rotations through this path-free accessor"
)]
pub fn recorder_directory() -> PathBuf {
    recorder_directory_at(&crate::host_paths::state_root())
}

pub fn recorder_directory_at(state_root: &Path) -> PathBuf {
    state_root.join(DIRECTORY_NAME)
}

/// Completed rotations, oldest first. Active or truncated files are excluded.
#[allow(
    dead_code,
    reason = "support-bundle integration consumes this bounded inventory"
)]
pub fn completed_rotation_paths() -> io::Result<Vec<PathBuf>> {
    completed_rotation_paths_at(&crate::host_paths::state_root())
}

#[allow(
    dead_code,
    reason = "host and fault fixtures use an alternate state root"
)]
pub fn completed_rotation_paths_at(state_root: &Path) -> io::Result<Vec<PathBuf>> {
    let directory = recorder_directory_at(state_root);
    let mut files = completed_artifacts(&directory)?;
    files.retain(|artifact| completed_marker_is_valid(&artifact.path));
    Ok(files.into_iter().map(|artifact| artifact.path).collect())
}

#[derive(Debug)]
struct Artifact {
    rotation: u64,
    path: PathBuf,
}

fn prepare_artifacts(directory: &Path) -> io::Result<u64> {
    let mut highest = 0_u64;
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if let Some(rotation) = artifact_rotation(name, COMPLETED_PREFIX) {
            highest = highest.max(rotation);
        } else if let Some(rotation) = artifact_rotation(name, ACTIVE_PREFIX) {
            highest = highest.max(rotation);
            // An active file has no completion guarantee. Removing it cannot
            // damage any independently published completed rotation.
            let _ = fs::remove_file(path);
        }
    }
    retain_latest_completed(directory)?;
    Ok(highest.saturating_add(1))
}

fn active_rotation_after(directory: &Path) -> io::Result<u64> {
    Ok(completed_artifacts(directory)?
        .last()
        .map_or(1, |artifact| artifact.rotation.saturating_add(1)))
}

fn completed_artifacts(directory: &Path) -> io::Result<Vec<Artifact>> {
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error),
    };
    let mut artifacts = Vec::new();
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if let Some(rotation) = artifact_rotation(name, COMPLETED_PREFIX) {
            artifacts.push(Artifact { rotation, path });
        }
    }
    artifacts.sort_by_key(|artifact| artifact.rotation);
    Ok(artifacts)
}

fn artifact_rotation(name: &str, prefix: &str) -> Option<u64> {
    name.strip_prefix(prefix)?
        .strip_suffix(FILE_SUFFIX)?
        .parse()
        .ok()
}

fn retain_latest_completed(directory: &Path) -> io::Result<()> {
    let artifacts = completed_artifacts(directory)?;
    let remove_count = artifacts.len().saturating_sub(RETAINED_COMPLETED_ROTATIONS);
    for artifact in artifacts.into_iter().take(remove_count) {
        fs::remove_file(artifact.path)?;
    }
    if remove_count > 0 {
        sync_directory(directory)?;
    }
    Ok(())
}

#[allow(
    dead_code,
    reason = "called by the support inventory seam when that optional consumer is linked"
)]
fn completed_marker_is_valid(path: &Path) -> bool {
    let Ok(bytes) = fs::read(path) else {
        return false;
    };
    let Some(last) = bytes
        .split(|byte| *byte == b'\n')
        .rev()
        .find(|line| !line.is_empty())
    else {
        return false;
    };
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(last) else {
        return false;
    };
    value.get("record").and_then(|value| value.as_str()) == Some("complete")
        && value.get("schema_version").and_then(|value| value.as_u64())
            == Some(u64::from(FORMAT_SCHEMA_VERSION))
}

fn prepare_private_directory(directory: &Path) -> io::Result<()> {
    fs::create_dir_all(directory)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(directory, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

fn open_private_new(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)
}

fn sync_directory(directory: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        File::open(directory)?.sync_all()
    }
    #[cfg(windows)]
    {
        // The completed file itself is synced before rename. Opening a
        // directory for `sync_all` is not portable on Windows std, and the
        // destination is unique rather than a replacement.
        let _ = directory;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;
    use std::path::Path;
    use std::process::{Child, Command, Stdio};
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_DIRECTORY: AtomicU64 = AtomicU64::new(1);
    const FORCED_TERMINATION_CHILD_ROOT: &str =
        "COLLIDE_O_SCOPE_FLIGHT_RECORDER_FORCED_TERMINATION_ROOT";

    struct TestRoot(PathBuf);

    impl TestRoot {
        fn new(label: &str) -> Self {
            let ordinal = TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "collide-flight-recorder-{label}-{}-{ordinal}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    struct KillChildOnDrop(Option<Child>);

    impl KillChildOnDrop {
        fn child_mut(&mut self) -> &mut Child {
            self.0.as_mut().expect("child process still owned")
        }

        fn kill_and_wait(&mut self) {
            if let Some(mut child) = self.0.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }

    impl Drop for KillChildOnDrop {
        fn drop(&mut self) {
            self.kill_and_wait();
        }
    }

    fn error_event(count: u32) -> FlightEvent {
        FlightEvent::Error(ErrorFact {
            domain: ErrorDomain::Renderer,
            code: ErrorCode::DeviceLost,
            retryable: true,
            occurrence_count: count,
        })
    }

    fn all_artifact_bytes(root: &Path) -> Vec<u8> {
        let directory = recorder_directory_at(root);
        let mut bytes = Vec::new();
        for entry in fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_file() {
                bytes.extend(fs::read(path).unwrap());
            }
        }
        bytes
    }

    #[test]
    fn production_rotation_contract_is_thirty_to_sixty_seconds() {
        assert!(FlightRecorderConfig::new(Duration::from_secs(29), MIN_TOTAL_BYTE_CAP).is_err());
        assert!(FlightRecorderConfig::new(Duration::from_secs(30), MIN_TOTAL_BYTE_CAP).is_ok());
        assert!(FlightRecorderConfig::new(Duration::from_secs(60), MIN_TOTAL_BYTE_CAP).is_ok());
        assert!(FlightRecorderConfig::new(Duration::from_secs(61), MIN_TOTAL_BYTE_CAP).is_err());
        assert!(
            FlightRecorderConfig::new(Duration::from_secs(45), MIN_TOTAL_BYTE_CAP - 1).is_err()
        );
    }

    #[test]
    fn live_queue_is_fixed_and_try_send_never_waits() {
        let (sender, _receiver) = mpsc::sync_channel(QUEUE_CAPACITY);
        let queued = QueuedEvent {
            sequence: 1,
            elapsed_microseconds: 0,
            event: error_event(1),
        };
        for _ in 0..QUEUE_CAPACITY {
            sender.try_send(queued).unwrap();
        }
        assert!(matches!(
            sender.try_send(queued),
            Err(TrySendError::Full(_))
        ));
    }

    #[test]
    fn adapter_calibration_is_keyed_and_advisory_only() {
        let key = CalibrationKeyFact {
            backend: GpuBackend::Dx12,
            pci_vendor_id: 0x1002,
            pci_device_id: 0x73bf,
            driver_version: [32, 0, 21023, 1015],
            raster_width: 1280,
            raster_height: 720,
            refresh_millihertz: 60_000,
            present_mode: PresentModeFact::Fifo,
            precision: PrecisionFact::Rgba16Float,
            patch_plan_digest: Digest32([0xA7; 32]),
        };
        let receipt = AdapterCalibrationFact::from_p95(
            key,
            300,
            [
                (StageKind::SourcePrepare, 100),
                (StageKind::CreativeComposition, 900),
                (StageKind::TemporalMotion, 300),
                (StageKind::MoshVhs, 400),
                (StageKind::AudienceResolve, 200),
                (StageKind::Submission, 800),
            ],
            9,
            48,
            64 * 1024 * 1024,
        );
        assert_eq!(receipt.key, key);
        assert_eq!(receipt.critical_stage, StageKind::CreativeComposition);
        assert_eq!(receipt.critical_stage_p95_nanoseconds, 900);
        assert!(receipt.advisory_only);
        let wire = serde_json::to_string(&FlightEvent::AdapterCalibration(receipt)).unwrap();
        assert!(wire.contains("adapter_calibration"));
        assert!(!wire.contains("adapter_name"));
        assert!(!wire.contains("driver_text"));
        assert!(!wire.contains("path"));
    }

    #[test]
    fn hostile_paths_tokens_cookies_media_secrets_and_text_cannot_enter_events() {
        let root = TestRoot::new("privacy");
        let forbidden = [
            r#"C:\Users\Eve\private\show.cos"#,
            "lan-token-A7m!seed",
            "Cookie: collide_session=seeded-cookie",
            "raw-media-byte-sentinel-89ab",
            "controller-secret-sentinel-3e1c",
            "authored text: meet me behind the venue",
        ];

        // Pass every hostile class through the explicit one-way boundary. The
        // returned value has no field capable of retaining any of them.
        let sanitized = ErrorFact::redact(
            ErrorDomain::WebControl,
            ErrorCode::PermissionDenied,
            false,
            1,
            SensitiveDiagnostic::new(
                Some(Path::new(forbidden[0])),
                Some(forbidden[1]),
                Some(forbidden[2]),
                Some(forbidden[3].as_bytes()),
                Some(forbidden[4]),
                Some(forbidden[5]),
            ),
        );

        let config = FlightRecorderConfig::for_test(Duration::from_secs(5), 3 * 64 * 1024);
        let recorder = FlightRecorder::start_at(&root.0, config).unwrap();
        for event in [
            FlightEvent::Error(sanitized),
            FlightEvent::ContentIdentity(ContentIdentityFact {
                patch_plan_digest: Digest32([0xA5; 32]),
                source_set_digest: Digest32([0x5A; 32]),
                source_count: 2,
            }),
            FlightEvent::ResourceLedger(ResourceLedgerFact {
                plan_digest: Digest32([0x33; 32]),
                full_frame_passes: 9,
                texture_lookups_per_pixel: 48,
                live_textures: 21,
                live_buffers: 14,
                live_bind_groups: 12,
                gpu_bytes: 4_000_000,
                cpu_bytes: 2_000_000,
                budget_gpu_bytes: 8_000_000,
                budget_cpu_bytes: 8_000_000,
            }),
        ] {
            assert!(matches!(
                recorder.try_record(event),
                RecordDisposition::Queued { .. }
            ));
        }
        recorder.shutdown().unwrap();

        let recorded = String::from_utf8(all_artifact_bytes(&root.0)).unwrap();
        for seed in forbidden {
            assert!(
                !recorded.contains(seed),
                "privacy fixture entered recorder bytes: {seed}"
            );
        }
        assert!(recorded.contains("content_identity"));
        assert!(recorded.contains("resource_ledger"));
    }

    #[test]
    fn previous_completed_rotation_survives_a_truncated_active_file() {
        let root = TestRoot::new("forced-termination");
        let config = FlightRecorderConfig::for_test(Duration::from_secs(5), 3 * 64 * 1024);
        let recorder = FlightRecorder::start_at(&root.0, config).unwrap();
        recorder.try_record(error_event(7));
        recorder.shutdown().unwrap();

        let before = completed_rotation_paths_at(&root.0).unwrap();
        assert_eq!(before.len(), 1);
        let completed_bytes = fs::read(&before[0]).unwrap();
        assert!(completed_marker_is_valid(&before[0]));

        let directory = recorder_directory_at(&root.0);
        let truncated = directory.join(format!("{ACTIVE_PREFIX}{:020}{FILE_SUFFIX}", 999));
        fs::write(&truncated, br#"{"record":"event","torn":"#).unwrap();

        let readable = completed_rotation_paths_at(&root.0).unwrap();
        assert_eq!(readable, before);
        assert_eq!(fs::read(&readable[0]).unwrap(), completed_bytes);

        // Restart cleanup may discard only the uncommitted active tail.
        let next = prepare_artifacts(&directory).unwrap();
        assert!(next > 999);
        assert!(!truncated.exists());
        assert_eq!(fs::read(&readable[0]).unwrap(), completed_bytes);
    }

    #[test]
    #[ignore = "subprocess helper; invoked by the forced-termination parent test"]
    fn forced_termination_child_publishes_then_keeps_an_active_tail() {
        let Some(root) = std::env::var_os(FORCED_TERMINATION_CHILD_ROOT) else {
            panic!("forced-termination child root was not provided");
        };
        let root = PathBuf::from(root);
        let config = FlightRecorderConfig::for_test(Duration::from_secs(30), 3 * 12 * 1024);
        let recorder = FlightRecorder::start_at(&root, config).unwrap();

        let deadline = Instant::now() + Duration::from_secs(5);
        let mut count = 1_u32;
        loop {
            let completed = completed_rotation_paths_at(&root).unwrap();
            if !completed.is_empty() {
                break;
            }
            assert!(Instant::now() < deadline, "first rotation did not publish");
            let _ = recorder.try_record(error_event(count));
            count = count.saturating_add(1);
            thread::sleep(Duration::from_millis(1));
        }

        // A timed sleep cannot prove that the writer drained every previously
        // admitted fact. Queue a unique sentinel instead and wait until its
        // bytes are observable in the active tail. Once observed, every older
        // fact has been handled and no later event can race the parent's kill.
        const ACTIVE_TAIL_SENTINEL: u32 = u32::MAX;
        let sentinel_deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match recorder.try_record(error_event(ACTIVE_TAIL_SENTINEL)) {
                RecordDisposition::Queued { .. } => break,
                RecordDisposition::DroppedFull { .. } => {
                    assert!(
                        Instant::now() < sentinel_deadline,
                        "active-tail sentinel could not enter the bounded queue"
                    );
                    thread::sleep(Duration::from_millis(5));
                }
                RecordDisposition::WorkerUnavailable { .. } => {
                    panic!("flight-recorder worker stopped before the kill sentinel")
                }
            }
        }
        let directory = recorder_directory_at(&root);
        let active_deadline = Instant::now() + Duration::from_secs(5);
        let sentinel = format!(r#""occurrence_count":{ACTIVE_TAIL_SENTINEL}"#);
        loop {
            let sentinel_is_active = fs::read_dir(&directory).unwrap().any(|entry| {
                let Ok(entry) = entry else {
                    return false;
                };
                let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                    return false;
                };
                name.starts_with(ACTIVE_PREFIX)
                    && fs::read(entry.path()).ok().is_some_and(|bytes| {
                        bytes
                            .windows(sentinel.len())
                            .any(|part| part == sentinel.as_bytes())
                    })
            });
            if sentinel_is_active {
                break;
            }
            assert!(
                Instant::now() < active_deadline,
                "active-tail sentinel did not reach the uncommitted rotation"
            );
            thread::sleep(Duration::from_millis(5));
        }

        fs::write(root.join("ready-to-kill"), b"ready").unwrap();
        loop {
            thread::sleep(Duration::from_secs(1));
        }
    }

    #[test]
    fn killed_recorder_process_preserves_the_previous_durable_rotation() {
        let root = TestRoot::new("actual-process-kill");
        let ready = root.0.join("ready-to-kill");
        let mut child = KillChildOnDrop(Some(
            Command::new(std::env::current_exe().unwrap())
                .arg("flight_recorder::tests::forced_termination_child_publishes_then_keeps_an_active_tail")
                .arg("--exact")
                .arg("--ignored")
                .env(FORCED_TERMINATION_CHILD_ROOT, &root.0)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap(),
        ));

        let deadline = Instant::now() + Duration::from_secs(10);
        while !ready.exists() {
            if let Some(status) = child.child_mut().try_wait().unwrap() {
                panic!("forced-termination child exited before readiness: {status}");
            }
            assert!(
                Instant::now() < deadline,
                "forced-termination child did not become ready"
            );
            thread::sleep(Duration::from_millis(10));
        }

        let before = completed_rotation_paths_at(&root.0).unwrap();
        assert!(!before.is_empty());
        let durable_bytes: Vec<Vec<u8>> = before
            .iter()
            .map(|path| {
                assert!(completed_marker_is_valid(path));
                fs::read(path).unwrap()
            })
            .collect();

        child.kill_and_wait();

        let after_kill = completed_rotation_paths_at(&root.0).unwrap();
        assert_eq!(after_kill, before);
        for (path, expected) in after_kill.iter().zip(&durable_bytes) {
            assert!(completed_marker_is_valid(path));
            assert_eq!(&fs::read(path).unwrap(), expected);
        }

        let directory = recorder_directory_at(&root.0);
        prepare_artifacts(&directory).unwrap();
        let after_restart = completed_rotation_paths_at(&root.0).unwrap();
        assert_eq!(after_restart, before);
        for (path, expected) in after_restart.iter().zip(&durable_bytes) {
            assert_eq!(&fs::read(path).unwrap(), expected);
        }
        assert!(!fs::read_dir(&directory).unwrap().any(|entry| {
            entry
                .ok()
                .and_then(|entry| entry.file_name().into_string().ok())
                .is_some_and(|name| name.starts_with(ACTIVE_PREFIX))
        }));
    }

    #[test]
    fn time_rotation_repeats_cached_host_content_and_last_ledger() {
        let root = TestRoot::new("time-rotation");
        let config = FlightRecorderConfig::for_test(Duration::from_millis(15), 3 * 64 * 1024);
        let recorder = FlightRecorder::start_at(&root.0, config).unwrap();
        recorder.try_record(FlightEvent::Host(HostFact {
            os: HostOs::Windows,
            os_build: 26_200,
            architecture: CpuArchitecture::X86_64,
            logical_cpu_count: 32,
            gpu: GpuFact {
                backend: GpuBackend::Vulkan,
                pci_vendor_id: 0x1002,
                pci_device_id: 0x73A5,
                driver_version: [32, 0, 21_045, 1000],
                timestamp_query_supported: true,
                precision: PrecisionFact::Rgba16Float,
            },
            display: DisplayFact {
                raster_width: 3840,
                raster_height: 2160,
                refresh_millihertz: 120_000,
                present_mode: PresentModeFact::Fifo,
                fullscreen: true,
            },
        }));
        recorder.try_record(FlightEvent::ContentIdentity(ContentIdentityFact {
            patch_plan_digest: Digest32([1; 32]),
            source_set_digest: Digest32([2; 32]),
            source_count: 3,
        }));
        recorder.try_record(FlightEvent::ResourceLedger(ResourceLedgerFact {
            plan_digest: Digest32([3; 32]),
            full_frame_passes: 4,
            texture_lookups_per_pixel: 12,
            live_textures: 8,
            live_buffers: 5,
            live_bind_groups: 6,
            gpu_bytes: 100,
            cpu_bytes: 200,
            budget_gpu_bytes: 300,
            budget_cpu_bytes: 400,
        }));
        thread::sleep(Duration::from_millis(35));
        recorder.try_record(error_event(2));
        let shutdown = recorder.shutdown().unwrap();
        assert!(shutdown.completed_rotations >= 2);

        let files = completed_rotation_paths_at(&root.0).unwrap();
        assert_eq!(files.len(), RETAINED_COMPLETED_ROTATIONS);
        let latest = String::from_utf8(fs::read(files.last().unwrap()).unwrap()).unwrap();
        assert!(latest.contains("\"host\":{"));
        assert!(latest.contains("\"content_identity\":{"));
        assert!(latest.contains("\"resource_ledger\":{"));
    }

    #[test]
    fn byte_rotation_and_retention_never_exceed_the_fixed_total_cap() {
        let root = TestRoot::new("byte-cap");
        let total_cap = 3 * 12 * 1024;
        let config = FlightRecorderConfig::for_test(Duration::from_secs(5), total_cap);
        let recorder = FlightRecorder::start_at(&root.0, config).unwrap();
        for count in 0..20_000 {
            let _ = recorder.try_record(FlightEvent::Stage(StageFact {
                stage: StageKind::CreativeComposition,
                domain: TimingDomain::Gpu,
                sample_count: count,
                p50_nanoseconds: 1_000_000,
                p95_nanoseconds: 2_000_000,
                p99_nanoseconds: 3_000_000,
                dropped_samples: u64::from(count),
                submission_generation: u64::from(count),
            }));
        }
        recorder.shutdown().unwrap();

        let directory = recorder_directory_at(&root.0);
        let files = completed_artifacts(&directory).unwrap();
        assert!(files.len() <= RETAINED_COMPLETED_ROTATIONS);
        let total: u64 = files
            .iter()
            .map(|artifact| fs::metadata(&artifact.path).unwrap().len())
            .sum();
        assert!(total <= total_cap);
        assert!(files.iter().all(|artifact| {
            fs::metadata(&artifact.path).unwrap().len() <= config.per_rotation_byte_cap()
                && completed_marker_is_valid(&artifact.path)
        }));
    }

    #[test]
    fn unrelated_files_are_never_pruned_as_recorder_artifacts() {
        let root = TestRoot::new("ownership");
        let directory = recorder_directory_at(&root.0);
        prepare_private_directory(&directory).unwrap();
        let unrelated = directory.join(OsStr::new("operator-note.txt"));
        fs::write(&unrelated, b"owned by somebody else").unwrap();
        prepare_artifacts(&directory).unwrap();
        assert_eq!(fs::read(unrelated).unwrap(), b"owned by somebody else");
    }
}
