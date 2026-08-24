//! Checksummed, bounded recovery checkpoints for durable patch state.
//!
//! Records are append-only between compactions. A torn/corrupt tail never
//! invalidates a preceding checkpoint and opening the journal is read-only.
//! Compaction is permitted only explicitly or while publishing a newly
//! successful checkpoint, and uses a same-directory atomic replacement.

use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use crate::patch::PatchState;

pub const RECOVERY_JOURNAL_VERSION: u16 = 1;
pub const RECOVERY_MAX_ENTRIES: usize = 256;
pub const RECOVERY_MAX_FILE_BYTES: u64 = 64 * 1024 * 1024;
pub const RECOVERY_MAX_PAYLOAD_BYTES: usize = 8 * 1024 * 1024;
/// A clean exit may wait for strong durability, but it may never wait without
/// a named bound. A timed-out worker is detached; process exit does not join it.
pub const RECOVERY_WRITER_SHUTDOWN_DEADLINE: Duration = Duration::from_millis(1_500);
pub const RECOVERY_WRITER_MAX_JOBS: usize = 2;
pub const RECOVERY_WRITER_STATUS_MAX_BYTES: usize = 512;
#[cfg_attr(
    test,
    allow(
        dead_code,
        reason = "App tests use explicit temporary recovery paths instead of the operator default"
    )
)]
pub const RECOVERY_JOURNAL_FILENAME: &str = "recovery-v1.journal";

/// One host-local recovery stream is used before any user patch path is
/// known. It is deliberately outside the patch corpus so a corrupt journal
/// can never overwrite or masquerade as the user's saved patch.
#[cfg_attr(
    test,
    allow(
        dead_code,
        reason = "App tests use explicit temporary recovery paths instead of the operator default"
    )
)]
pub fn default_recovery_journal_path() -> PathBuf {
    if let Some(base) = std::env::var_os("LOCALAPPDATA") {
        return PathBuf::from(base)
            .join("collide-o-scope")
            .join("recovery")
            .join(RECOVERY_JOURNAL_FILENAME);
    }
    if let Some(base) = std::env::var_os("XDG_STATE_HOME") {
        return PathBuf::from(base)
            .join("collide-o-scope")
            .join(RECOVERY_JOURNAL_FILENAME);
    }
    if let Some(base) = std::env::var_os("HOME") {
        return PathBuf::from(base)
            .join(".local")
            .join("state")
            .join("collide-o-scope")
            .join(RECOVERY_JOURNAL_FILENAME);
    }
    PathBuf::from(".collide-o-scope").join(RECOVERY_JOURNAL_FILENAME)
}

const RECORD_MAGIC: [u8; 8] = *b"COSRECJ1";
const RECORD_FLAGS: u16 = 0;
const RECORD_RESERVED: u32 = 0;
const RECORD_HEADER_BYTES: usize = 8 + 2 + 2 + 8 + 4 + 4 + 32;
const CHECKSUM_DOMAIN: &[u8] = b"collide-o-scope recovery journal record v1\0";
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecoveryLimits {
    pub max_entries: usize,
    pub max_file_bytes: u64,
    pub max_payload_bytes: usize,
}

impl Default for RecoveryLimits {
    fn default() -> Self {
        Self {
            max_entries: RECOVERY_MAX_ENTRIES,
            max_file_bytes: RECOVERY_MAX_FILE_BYTES,
            max_payload_bytes: RECOVERY_MAX_PAYLOAD_BYTES,
        }
    }
}

impl RecoveryLimits {
    fn bounded(self) -> Result<Self, RecoveryJournalError> {
        let minimum_record = u64::try_from(RECORD_HEADER_BYTES + 1).unwrap_or(u64::MAX);
        if self.max_entries == 0
            || self.max_entries > RECOVERY_MAX_ENTRIES
            || self.max_file_bytes < minimum_record
            || self.max_file_bytes > RECOVERY_MAX_FILE_BYTES
            || self.max_payload_bytes == 0
            || self.max_payload_bytes > RECOVERY_MAX_PAYLOAD_BYTES
            || u64::try_from(RECORD_HEADER_BYTES + self.max_payload_bytes).unwrap_or(u64::MAX)
                > self.max_file_bytes
        {
            return Err(RecoveryJournalError::InvalidLimits);
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RecoveryTailStatus {
    #[default]
    Clean,
    Truncated,
    Corrupt,
    UnsupportedVersion,
    LimitExceeded,
}

impl RecoveryTailStatus {
    pub const fn has_bad_tail(self) -> bool {
        !matches!(self, Self::Clean)
    }
}

#[derive(Clone)]
pub struct RecoveryCheckpoint {
    pub sequence: u64,
    pub patch: PatchState,
    payload: Vec<u8>,
}

impl fmt::Debug for RecoveryCheckpoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RecoveryCheckpoint")
            .field("sequence", &self.sequence)
            .field("payload_bytes", &self.payload.len())
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone)]
pub struct RecoveryScan {
    pub latest: Option<RecoveryCheckpoint>,
    pub valid_entries: usize,
    pub valid_bytes: u64,
    pub tail_status: RecoveryTailStatus,
    pub warning: Option<String>,
}

impl Default for RecoveryScan {
    fn default() -> Self {
        Self {
            latest: None,
            valid_entries: 0,
            valid_bytes: 0,
            tail_status: RecoveryTailStatus::Clean,
            warning: None,
        }
    }
}

impl RecoveryScan {
    pub const fn recovery_available(&self) -> bool {
        self.latest.is_some()
    }

    pub const fn has_bad_tail(&self) -> bool {
        self.tail_status.has_bad_tail()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JournalAppendReceipt {
    pub sequence: u64,
    pub payload_bytes: usize,
    pub journal_bytes: u64,
    pub compacted: bool,
}

#[derive(Debug)]
pub enum RecoveryJournalError {
    InvalidLimits,
    SequenceExhausted,
    PayloadTooLarge {
        bytes: usize,
        limit: usize,
    },
    RecordTooLarge {
        bytes: u64,
        limit: u64,
    },
    Serialize(String),
    Io {
        operation: &'static str,
        error: io::Error,
    },
    DestinationRace(PathBuf),
}

impl fmt::Display for RecoveryJournalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimits => {
                formatter.write_str("recovery journal limits exceed the hard bounds")
            }
            Self::SequenceExhausted => {
                formatter.write_str("recovery journal sequence is exhausted")
            }
            Self::PayloadTooLarge { bytes, limit } => write!(
                formatter,
                "recovery checkpoint is {bytes} bytes; limit is {limit}"
            ),
            Self::RecordTooLarge { bytes, limit } => write!(
                formatter,
                "recovery journal record is {bytes} bytes; file limit is {limit}"
            ),
            Self::Serialize(error) => write!(formatter, "serialize recovery checkpoint: {error}"),
            Self::Io { operation, error } => write!(formatter, "{operation}: {error}"),
            Self::DestinationRace(path) => write!(
                formatter,
                "recovery journal destination appeared during atomic publication: {}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for RecoveryJournalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { error, .. } => Some(error),
            _ => None,
        }
    }
}

fn io_error(operation: &'static str, error: io::Error) -> RecoveryJournalError {
    RecoveryJournalError::Io { operation, error }
}

pub struct RecoveryJournal {
    path: PathBuf,
    limits: RecoveryLimits,
    scan: RecoveryScan,
    next_sequence: u64,
}

impl RecoveryJournal {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, RecoveryJournalError> {
        Self::open_with_limits(path, RecoveryLimits::default())
    }

    pub fn open_with_limits(
        path: impl Into<PathBuf>,
        limits: RecoveryLimits,
    ) -> Result<Self, RecoveryJournalError> {
        let path = path.into();
        let limits = limits.bounded()?;
        let scan = scan_path(&path, limits)?;
        let next_sequence = scan.latest.as_ref().map_or(1, |checkpoint| {
            checkpoint.sequence.checked_add(1).unwrap_or(0)
        });
        Ok(Self {
            path,
            limits,
            scan,
            next_sequence,
        })
    }

    pub fn latest_valid(&self) -> &RecoveryScan {
        &self.scan
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "explicit rescan is retained for crash-tail recovery goldens"
        )
    )]
    pub fn rescan(&mut self) -> Result<&RecoveryScan, RecoveryJournalError> {
        self.scan = scan_path(&self.path, self.limits)?;
        self.next_sequence = self.scan.latest.as_ref().map_or(1, |checkpoint| {
            checkpoint.sequence.checked_add(1).unwrap_or(0)
        });
        Ok(&self.scan)
    }

    pub fn append_patch(
        &mut self,
        patch: &PatchState,
    ) -> Result<JournalAppendReceipt, RecoveryJournalError> {
        let sequence = self.next_sequence;
        if sequence == 0 {
            return Err(RecoveryJournalError::SequenceExhausted);
        }
        let payload = serde_yaml::to_string(patch)
            .map_err(|error| RecoveryJournalError::Serialize(error.to_string()))?
            .into_bytes();
        self.validate_payload(&payload)?;
        // Round-trip through the hostile PatchState boundary before any bytes
        // are published. This catches accidental non-loadable checkpoints.
        serde_yaml::from_slice::<PatchState>(&payload)
            .map_err(|error| RecoveryJournalError::Serialize(error.to_string()))?;
        let record = encode_record(sequence, &payload)?;
        let current_file_bytes = fs::metadata(&self.path).map_or(0, |metadata| metadata.len());
        let projected =
            current_file_bytes.saturating_add(u64::try_from(record.len()).unwrap_or(u64::MAX));
        let compact = !self.path.exists()
            || self.scan.has_bad_tail()
            || self.scan.valid_entries >= self.limits.max_entries
            || projected > self.limits.max_file_bytes;

        if compact {
            write_atomic_replacement(&self.path, &record)?;
        } else {
            append_record(&self.path, &record)?;
        }
        sync_parent(&self.path).map_err(|error| io_error("sync recovery directory", error))?;

        self.scan = scan_path(&self.path, self.limits)?;
        let accepted = self
            .scan
            .latest
            .as_ref()
            .is_some_and(|checkpoint| checkpoint.sequence == sequence)
            && !self.scan.has_bad_tail();
        if !accepted {
            return Err(RecoveryJournalError::Io {
                operation: "verify committed recovery checkpoint",
                error: io::Error::new(io::ErrorKind::InvalidData, "checkpoint was not readable"),
            });
        }
        self.next_sequence = sequence.checked_add(1).unwrap_or(0);
        Ok(JournalAppendReceipt {
            sequence,
            payload_bytes: payload.len(),
            journal_bytes: fs::metadata(&self.path).map_or(0, |metadata| metadata.len()),
            compacted: compact,
        })
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "explicit compaction is retained for bounded journal goldens"
        )
    )]
    pub fn compact_to_latest(&mut self) -> Result<(), RecoveryJournalError> {
        let bytes = self
            .scan
            .latest
            .as_ref()
            .map(|checkpoint| encode_record(checkpoint.sequence, &checkpoint.payload))
            .transpose()?
            .unwrap_or_default();
        write_atomic_replacement(&self.path, &bytes)?;
        sync_parent(&self.path).map_err(|error| io_error("sync recovery directory", error))?;
        self.rescan()?;
        Ok(())
    }

    /// Explicit discard publishes an empty valid journal atomically. It never
    /// touches the user's patch file and leaves no recoverable checkpoint.
    pub fn discard(&mut self) -> Result<(), RecoveryJournalError> {
        write_atomic_replacement(&self.path, &[])?;
        sync_parent(&self.path).map_err(|error| io_error("sync recovery directory", error))?;
        self.scan = RecoveryScan::default();
        self.next_sequence = 1;
        Ok(())
    }

    fn validate_payload(&self, payload: &[u8]) -> Result<(), RecoveryJournalError> {
        if payload.len() > self.limits.max_payload_bytes {
            return Err(RecoveryJournalError::PayloadTooLarge {
                bytes: payload.len(),
                limit: self.limits.max_payload_bytes,
            });
        }
        let record_bytes = u64::try_from(RECORD_HEADER_BYTES + payload.len()).unwrap_or(u64::MAX);
        if record_bytes > self.limits.max_file_bytes {
            return Err(RecoveryJournalError::RecordTooLarge {
                bytes: record_bytes,
                limit: self.limits.max_file_bytes,
            });
        }
        Ok(())
    }
}

/// Process-lifetime identifier for a request accepted by the recovery writer.
/// It is deliberately not a journal sequence: coalesced and failed requests
/// have IDs, but only a verified append receipt earns a durable sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RecoveryRequestId(u64);

impl RecoveryRequestId {
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for RecoveryRequestId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryWriterLifecycle {
    Starting,
    Idle,
    Pending,
    Writing,
    Flushing,
    Failed,
    Unavailable,
    Stopped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecoveryEnqueueReceipt {
    pub request_id: RecoveryRequestId,
    /// A request displaced from the one newest-wins pending slot. It never
    /// receives a journal sequence.
    pub superseded_request_id: Option<RecoveryRequestId>,
    /// The last worker-verified sequence at enqueue time, not a prediction of
    /// the sequence this request might eventually receive.
    pub durable_sequence_at_request: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryRequestError {
    RequestIdExhausted,
    ShuttingDown,
    Unavailable,
}

impl fmt::Display for RecoveryRequestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RequestIdExhausted => {
                formatter.write_str("recovery request identifier is exhausted")
            }
            Self::ShuttingDown => formatter.write_str("recovery writer is shutting down"),
            Self::Unavailable => formatter.write_str("recovery writer is unavailable"),
        }
    }
}

impl std::error::Error for RecoveryRequestError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryWriterStatus {
    pub revision: u64,
    pub lifecycle: RecoveryWriterLifecycle,
    pub in_flight_request_id: Option<RecoveryRequestId>,
    pub pending_request_id: Option<RecoveryRequestId>,
    pub last_durable_request_id: Option<RecoveryRequestId>,
    pub durable_sequence: Option<u64>,
    pub last_failed_request_id: Option<RecoveryRequestId>,
    pub submitted_requests: u64,
    pub superseded_requests: u64,
    pub failed_requests: u64,
    pub queue_depth: usize,
    pub max_observed_jobs: usize,
    pub error: Option<String>,
    pub last_receipt: Option<JournalAppendReceipt>,
}

impl RecoveryWriterStatus {
    pub fn line(&self) -> String {
        let durable = self.durable_sequence.map_or_else(
            || "no durable checkpoint".to_string(),
            |sequence| format!("durable checkpoint {sequence}"),
        );
        let pending = self
            .pending_request_id
            .map(|request| format!("; newest pending request {request}"))
            .unwrap_or_default();
        let coalesced = if self.superseded_requests > 0 {
            format!("; {} coalesced", self.superseded_requests)
        } else {
            String::new()
        };
        let notice = self
            .error
            .as_deref()
            .map(|error| format!("; {error}"))
            .unwrap_or_default();
        let line = match self.lifecycle {
            RecoveryWriterLifecycle::Starting => {
                format!("Recovery writer is scanning on its worker; {durable}{pending}")
            }
            RecoveryWriterLifecycle::Writing => format!(
                "Recovery request {} is writing; {durable} remains authoritative{pending}{coalesced}{notice}",
                self.in_flight_request_id
                    .map_or_else(|| "?".to_string(), |request| request.to_string())
            ),
            RecoveryWriterLifecycle::Pending => format!(
                "Recovery request {} is pending; {durable} remains authoritative{coalesced}{notice}",
                self.pending_request_id
                    .map_or_else(|| "?".to_string(), |request| request.to_string())
            ),
            RecoveryWriterLifecycle::Flushing => format!(
                "Recovery writer is flushing with a fixed deadline; {durable}{pending}{coalesced}{notice}"
            ),
            RecoveryWriterLifecycle::Failed => format!(
                "Recovery request {} failed; {durable} remains authoritative: {}{pending}{coalesced}",
                self.last_failed_request_id
                    .map_or_else(|| "?".to_string(), |request| request.to_string()),
                self.error.as_deref().unwrap_or("bounded worker failure")
            ),
            RecoveryWriterLifecycle::Unavailable => format!(
                "Recovery writer is unavailable; {durable}: {}",
                self.error.as_deref().unwrap_or("bounded worker failure")
            ),
            RecoveryWriterLifecycle::Stopped => {
                format!("Recovery writer stopped; {durable}{coalesced}{notice}")
            }
            RecoveryWriterLifecycle::Idle => {
                if let (Some(request), Some(receipt)) =
                    (self.last_durable_request_id, self.last_receipt)
                {
                    format!(
                        "Recovery request {request} verified as durable checkpoint {} ({} bytes{}){coalesced}{notice}",
                        receipt.sequence,
                        receipt.payload_bytes,
                        if receipt.compacted { ", compacted" } else { "" }
                    )
                } else {
                    format!("Recovery writer ready; {durable}{coalesced}{notice}")
                }
            }
        };
        bounded_status(line)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecoveryFlushReceipt {
    pub completed: bool,
    pub durable_sequence: Option<u64>,
    pub in_flight_request_id: Option<RecoveryRequestId>,
    pub pending_request_id: Option<RecoveryRequestId>,
}

enum RecoveryWriterOperation {
    Checkpoint(Box<PatchState>),
    Discard,
}

struct RecoveryWriterRequest {
    id: RecoveryRequestId,
    operation: RecoveryWriterOperation,
}

trait RecoveryWriterBackend: Send {
    fn latest_valid(&self) -> &RecoveryScan;
    fn append_patch(
        &mut self,
        patch: &PatchState,
    ) -> Result<JournalAppendReceipt, RecoveryJournalError>;
    fn discard(&mut self) -> Result<(), RecoveryJournalError>;
    /// An append can fail after touching the file (for example, directory
    /// sync failure). Reopening on the worker repairs its internal sequence
    /// and bad-tail view before a newer request is attempted. The published
    /// acknowledged scan is intentionally not advanced by this rescan.
    fn recover_after_failure(&mut self) -> Result<(), RecoveryJournalError>;
}

impl RecoveryWriterBackend for RecoveryJournal {
    fn latest_valid(&self) -> &RecoveryScan {
        self.latest_valid()
    }

    fn append_patch(
        &mut self,
        patch: &PatchState,
    ) -> Result<JournalAppendReceipt, RecoveryJournalError> {
        self.append_patch(patch)
    }

    fn discard(&mut self) -> Result<(), RecoveryJournalError> {
        self.discard()
    }

    fn recover_after_failure(&mut self) -> Result<(), RecoveryJournalError> {
        let replacement = RecoveryJournal::open_with_limits(self.path.clone(), self.limits)?;
        *self = replacement;
        Ok(())
    }
}

type RecoveryBackendFactory = Box<
    dyn FnOnce() -> Result<Box<dyn RecoveryWriterBackend>, RecoveryJournalError> + Send + 'static,
>;

struct RecoveryWriterShared {
    pending: Option<RecoveryWriterRequest>,
    in_flight_request_id: Option<RecoveryRequestId>,
    durable_scan: Arc<RecoveryScan>,
    accepting: bool,
    shutdown_requested: bool,
    initialized: bool,
    worker_stopped: bool,
    revision: u64,
    last_durable_request_id: Option<RecoveryRequestId>,
    last_failed_request_id: Option<RecoveryRequestId>,
    last_receipt: Option<JournalAppendReceipt>,
    submitted_requests: u64,
    superseded_requests: u64,
    failed_requests: u64,
    max_observed_jobs: usize,
    error: Option<String>,
}

impl Default for RecoveryWriterShared {
    fn default() -> Self {
        Self {
            pending: None,
            in_flight_request_id: None,
            durable_scan: Arc::new(RecoveryScan::default()),
            accepting: true,
            shutdown_requested: false,
            initialized: false,
            worker_stopped: false,
            revision: 1,
            last_durable_request_id: None,
            last_failed_request_id: None,
            last_receipt: None,
            submitted_requests: 0,
            superseded_requests: 0,
            failed_requests: 0,
            max_observed_jobs: 0,
            error: None,
        }
    }
}

struct RecoveryWriterSync {
    state: Mutex<RecoveryWriterShared>,
    changed: Condvar,
}

/// Dedicated single-writer ownership for the durable journal. The producer
/// performs only immutable PatchState capture plus one mutex-protected slot
/// replacement; serialization, validation, file I/O, fsync, compaction, and
/// verification remain on the named worker.
pub struct RecoveryWriter {
    path: PathBuf,
    sync: Arc<RecoveryWriterSync>,
    next_request_id: AtomicU64,
    /// Dropping a JoinHandle detaches. This handle is never joined from the
    /// render/event-loop thread, including after a flush receipt.
    worker: Option<JoinHandle<()>>,
}

impl RecoveryWriter {
    #[cfg_attr(
        test,
        allow(
            dead_code,
            reason = "App tests isolate startup from the operator's real recovery journal"
        )
    )]
    pub fn start_default() -> Result<Self, RecoveryJournalError> {
        Self::start(default_recovery_journal_path())
    }

    pub fn start(path: impl Into<PathBuf>) -> Result<Self, RecoveryJournalError> {
        let path = path.into();
        let worker_path = path.clone();
        Self::start_with_factory(
            path,
            Box::new(move || {
                RecoveryJournal::open(worker_path)
                    .map(|journal| Box::new(journal) as Box<dyn RecoveryWriterBackend>)
            }),
        )
    }

    fn start_with_factory(
        path: PathBuf,
        factory: RecoveryBackendFactory,
    ) -> Result<Self, RecoveryJournalError> {
        let sync = Arc::new(RecoveryWriterSync {
            state: Mutex::new(RecoveryWriterShared::default()),
            changed: Condvar::new(),
        });
        let worker_sync = Arc::clone(&sync);
        let worker = thread::Builder::new()
            .name("recovery-writer".to_string())
            .spawn(move || run_recovery_writer_guarded(worker_sync, factory))
            .map_err(|error| io_error("spawn recovery writer", error))?;
        Ok(Self {
            path,
            sync,
            next_request_id: AtomicU64::new(1),
            worker: Some(worker),
        })
    }

    pub fn request_checkpoint(
        &self,
        patch: PatchState,
    ) -> Result<RecoveryEnqueueReceipt, RecoveryRequestError> {
        self.request(RecoveryWriterOperation::Checkpoint(Box::new(patch)))
    }

    pub fn request_discard(&self) -> Result<RecoveryEnqueueReceipt, RecoveryRequestError> {
        self.request(RecoveryWriterOperation::Discard)
    }

    fn request(
        &self,
        operation: RecoveryWriterOperation,
    ) -> Result<RecoveryEnqueueReceipt, RecoveryRequestError> {
        let request_id = self
            .next_request_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1).filter(|next| *next != 0)
            })
            .map(RecoveryRequestId)
            .map_err(|_| RecoveryRequestError::RequestIdExhausted)?;
        let mut state = lock_state(&self.sync);
        if !state.accepting {
            return Err(if state.shutdown_requested {
                RecoveryRequestError::ShuttingDown
            } else {
                RecoveryRequestError::Unavailable
            });
        }
        let superseded_request_id = state.pending.take().map(|pending| pending.id);
        if superseded_request_id.is_some() {
            state.superseded_requests = state.superseded_requests.saturating_add(1);
        }
        state.pending = Some(RecoveryWriterRequest {
            id: request_id,
            operation,
        });
        state.submitted_requests = state.submitted_requests.saturating_add(1);
        let jobs = usize::from(state.in_flight_request_id.is_some()) + 1;
        debug_assert!(jobs <= RECOVERY_WRITER_MAX_JOBS);
        state.max_observed_jobs = state.max_observed_jobs.max(jobs);
        state.revision = state.revision.wrapping_add(1).max(1);
        let durable_sequence_at_request = state
            .durable_scan
            .latest
            .as_ref()
            .map(|checkpoint| checkpoint.sequence);
        drop(state);
        self.sync.changed.notify_one();
        Ok(RecoveryEnqueueReceipt {
            request_id,
            superseded_request_id,
            durable_sequence_at_request,
        })
    }

    pub fn status(&self) -> RecoveryWriterStatus {
        let state = lock_state(&self.sync);
        snapshot_status(&state)
    }

    pub fn durable_scan(&self) -> Arc<RecoveryScan> {
        Arc::clone(&lock_state(&self.sync).durable_scan)
    }

    pub fn recovery_available(&self) -> bool {
        lock_state(&self.sync).durable_scan.recovery_available()
    }

    /// Clone the authored patch only for an explicit operator restore. Normal
    /// status publication reads the small Arc-backed scan facts instead.
    pub fn latest_checkpoint(&self) -> Option<(PatchState, PathBuf, u64)> {
        let scan = self.durable_scan();
        scan.latest.as_ref().map(|checkpoint| {
            (
                checkpoint.patch.clone(),
                self.path.clone(),
                checkpoint.sequence,
            )
        })
    }

    pub fn flush_with_deadline(&self, timeout: Duration) -> RecoveryFlushReceipt {
        let started = Instant::now();
        let mut state = lock_state(&self.sync);
        state.accepting = false;
        state.shutdown_requested = true;
        state.revision = state.revision.wrapping_add(1).max(1);
        self.sync.changed.notify_all();
        while !state.worker_stopped {
            let Some(remaining) = timeout.checked_sub(started.elapsed()) else {
                break;
            };
            if remaining.is_zero() {
                break;
            }
            let (next, wait) = self
                .sync
                .changed
                .wait_timeout(state, remaining)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state = next;
            if wait.timed_out() && !state.worker_stopped {
                break;
            }
        }
        RecoveryFlushReceipt {
            completed: state.worker_stopped,
            durable_sequence: state
                .durable_scan
                .latest
                .as_ref()
                .map(|checkpoint| checkpoint.sequence),
            in_flight_request_id: state.in_flight_request_id,
            pending_request_id: state.pending.as_ref().map(|pending| pending.id),
        }
    }
}

impl Drop for RecoveryWriter {
    fn drop(&mut self) {
        {
            let mut state = lock_state(&self.sync);
            state.accepting = false;
            state.shutdown_requested = true;
            state.revision = state.revision.wrapping_add(1).max(1);
        }
        self.sync.changed.notify_all();
        // Explicitly detach. Strong-durability waiting is done only by the
        // bounded clean-shutdown API above, never by Drop.
        let _ = self.worker.take();
    }
}

fn lock_state(sync: &RecoveryWriterSync) -> MutexGuard<'_, RecoveryWriterShared> {
    sync.state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn snapshot_status(state: &RecoveryWriterShared) -> RecoveryWriterStatus {
    let lifecycle = if state.worker_stopped {
        if state.error.is_some() {
            RecoveryWriterLifecycle::Unavailable
        } else if state.initialized {
            RecoveryWriterLifecycle::Stopped
        } else {
            RecoveryWriterLifecycle::Unavailable
        }
    } else if !state.initialized {
        RecoveryWriterLifecycle::Starting
    } else if state.in_flight_request_id.is_some() {
        if state.shutdown_requested {
            RecoveryWriterLifecycle::Flushing
        } else {
            RecoveryWriterLifecycle::Writing
        }
    } else if state.pending.is_some() {
        if state.shutdown_requested {
            RecoveryWriterLifecycle::Flushing
        } else {
            RecoveryWriterLifecycle::Pending
        }
    } else if state.last_failed_request_id.is_some() && state.error.is_some() {
        RecoveryWriterLifecycle::Failed
    } else if state.shutdown_requested {
        RecoveryWriterLifecycle::Flushing
    } else {
        RecoveryWriterLifecycle::Idle
    };
    RecoveryWriterStatus {
        revision: state.revision,
        lifecycle,
        in_flight_request_id: state.in_flight_request_id,
        pending_request_id: state.pending.as_ref().map(|pending| pending.id),
        last_durable_request_id: state.last_durable_request_id,
        durable_sequence: state
            .durable_scan
            .latest
            .as_ref()
            .map(|checkpoint| checkpoint.sequence),
        last_failed_request_id: state.last_failed_request_id,
        submitted_requests: state.submitted_requests,
        superseded_requests: state.superseded_requests,
        failed_requests: state.failed_requests,
        queue_depth: usize::from(state.in_flight_request_id.is_some())
            + usize::from(state.pending.is_some()),
        max_observed_jobs: state.max_observed_jobs,
        error: state.error.clone(),
        last_receipt: state.last_receipt,
    }
}

fn run_recovery_writer_guarded(sync: Arc<RecoveryWriterSync>, factory: RecoveryBackendFactory) {
    let worker_sync = Arc::clone(&sync);
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_recovery_writer(worker_sync, factory);
    }))
    .is_err()
    {
        let mut state = lock_state(&sync);
        state.accepting = false;
        state.shutdown_requested = true;
        state.worker_stopped = true;
        let in_flight = state.in_flight_request_id.take();
        let pending = state.pending.take().map(|request| request.id);
        if let Some(request) = pending.or(in_flight) {
            state.last_failed_request_id = Some(request);
            state.failed_requests = state
                .failed_requests
                .saturating_add(u64::from(in_flight.is_some()) + u64::from(pending.is_some()));
        }
        state.error = Some("recovery worker stopped after an internal panic".to_string());
        state.revision = state.revision.wrapping_add(1).max(1);
        drop(state);
        sync.changed.notify_all();
    }
}

fn run_recovery_writer(sync: Arc<RecoveryWriterSync>, factory: RecoveryBackendFactory) {
    let mut backend = match factory() {
        Ok(backend) => backend,
        Err(error) => {
            let mut state = lock_state(&sync);
            state.accepting = false;
            state.worker_stopped = true;
            state.error = Some(bounded_status(error.to_string()));
            if let Some(request) = state.pending.take() {
                state.last_failed_request_id = Some(request.id);
                state.failed_requests = state.failed_requests.saturating_add(1);
            }
            state.revision = state.revision.wrapping_add(1).max(1);
            drop(state);
            sync.changed.notify_all();
            return;
        }
    };
    {
        let mut state = lock_state(&sync);
        state.durable_scan = Arc::new(backend.latest_valid().clone());
        state.error = backend.latest_valid().warning.clone().map(bounded_status);
        state.initialized = true;
        state.revision = state.revision.wrapping_add(1).max(1);
    }
    sync.changed.notify_all();

    loop {
        let request = {
            let mut state = lock_state(&sync);
            while state.pending.is_none() && !state.shutdown_requested {
                state = sync
                    .changed
                    .wait(state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
            if state.pending.is_none() && state.shutdown_requested {
                state.accepting = false;
                state.worker_stopped = true;
                state.revision = state.revision.wrapping_add(1).max(1);
                drop(state);
                sync.changed.notify_all();
                return;
            }
            let request = state
                .pending
                .take()
                .expect("the recovery-writer wake predicate guarantees a request");
            state.in_flight_request_id = Some(request.id);
            state.revision = state.revision.wrapping_add(1).max(1);
            request
        };
        sync.changed.notify_all();

        let outcome = match &request.operation {
            RecoveryWriterOperation::Checkpoint(patch) => backend
                .append_patch(patch.as_ref())
                .map(RecoveryWorkerSuccess::Checkpoint),
            RecoveryWriterOperation::Discard => {
                backend.discard().map(|()| RecoveryWorkerSuccess::Discard)
            }
        };

        let mut state = lock_state(&sync);
        debug_assert_eq!(state.in_flight_request_id, Some(request.id));
        state.in_flight_request_id = None;
        match outcome {
            Ok(RecoveryWorkerSuccess::Checkpoint(receipt)) => {
                // The receipt was produced only after append/replace, file and
                // directory sync, hostile rescan, and exact sequence proof.
                state.durable_scan = Arc::new(backend.latest_valid().clone());
                state.last_durable_request_id = Some(request.id);
                state.last_receipt = Some(receipt);
                state.last_failed_request_id = None;
                state.error = None;
            }
            Ok(RecoveryWorkerSuccess::Discard) => {
                state.durable_scan = Arc::new(backend.latest_valid().clone());
                state.last_durable_request_id = None;
                state.last_receipt = None;
                state.last_failed_request_id = None;
                state.error = None;
            }
            Err(error) => {
                state.last_failed_request_id = Some(request.id);
                state.failed_requests = state.failed_requests.saturating_add(1);
                state.error = Some(bounded_status(error.to_string()));
                // Do not update durable_scan: the prior acknowledged
                // checkpoint remains the only in-process recovery claim.
                drop(state);
                let reopen_error = backend.recover_after_failure().err();
                state = lock_state(&sync);
                if let Some(reopen_error) = reopen_error {
                    state.error = Some(bounded_status(format!(
                        "{}; worker rescan failed: {reopen_error}",
                        state.error.as_deref().unwrap_or("append failed")
                    )));
                }
            }
        }
        state.revision = state.revision.wrapping_add(1).max(1);
        drop(state);
        sync.changed.notify_all();
    }
}

enum RecoveryWorkerSuccess {
    Checkpoint(JournalAppendReceipt),
    Discard,
}

fn bounded_status(mut status: String) -> String {
    if status.len() <= RECOVERY_WRITER_STATUS_MAX_BYTES {
        return status;
    }
    let mut end = RECOVERY_WRITER_STATUS_MAX_BYTES.saturating_sub(3);
    while !status.is_char_boundary(end) {
        end = end.saturating_sub(1);
    }
    status.truncate(end);
    status.push_str("...");
    status
}

fn record_checksum(
    version: u16,
    flags: u16,
    sequence: u64,
    payload_len: u32,
    reserved: u32,
    payload: &[u8],
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(CHECKSUM_DOMAIN);
    digest.update(RECORD_MAGIC);
    digest.update(version.to_le_bytes());
    digest.update(flags.to_le_bytes());
    digest.update(sequence.to_le_bytes());
    digest.update(payload_len.to_le_bytes());
    digest.update(reserved.to_le_bytes());
    digest.update(payload);
    digest.finalize().into()
}

pub(crate) fn encode_record(
    sequence: u64,
    payload: &[u8],
) -> Result<Vec<u8>, RecoveryJournalError> {
    if sequence == 0 {
        return Err(RecoveryJournalError::SequenceExhausted);
    }
    let payload_len =
        u32::try_from(payload.len()).map_err(|_| RecoveryJournalError::PayloadTooLarge {
            bytes: payload.len(),
            limit: u32::MAX as usize,
        })?;
    let checksum = record_checksum(
        RECOVERY_JOURNAL_VERSION,
        RECORD_FLAGS,
        sequence,
        payload_len,
        RECORD_RESERVED,
        payload,
    );
    let mut record = Vec::with_capacity(RECORD_HEADER_BYTES + payload.len());
    record.extend_from_slice(&RECORD_MAGIC);
    record.extend_from_slice(&RECOVERY_JOURNAL_VERSION.to_le_bytes());
    record.extend_from_slice(&RECORD_FLAGS.to_le_bytes());
    record.extend_from_slice(&sequence.to_le_bytes());
    record.extend_from_slice(&payload_len.to_le_bytes());
    record.extend_from_slice(&RECORD_RESERVED.to_le_bytes());
    record.extend_from_slice(&checksum);
    record.extend_from_slice(payload);
    Ok(record)
}

fn scan_path(path: &Path, limits: RecoveryLimits) -> Result<RecoveryScan, RecoveryJournalError> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(RecoveryScan::default()),
        Err(error) => return Err(io_error("open recovery journal", error)),
    };
    let metadata = file
        .metadata()
        .map_err(|error| io_error("inspect recovery journal", error))?;
    let oversized = metadata.len() > limits.max_file_bytes;
    let read_limit = limits.max_file_bytes.saturating_add(1);
    let mut bytes = Vec::with_capacity(
        usize::try_from(metadata.len().min(read_limit)).unwrap_or(RECOVERY_MAX_FILE_BYTES as usize),
    );
    file.take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|error| io_error("read recovery journal", error))?;
    scan_bytes(&bytes, limits, oversized)
}

pub(crate) fn scan_bytes(
    bytes: &[u8],
    limits: RecoveryLimits,
    oversized_file: bool,
) -> Result<RecoveryScan, RecoveryJournalError> {
    let mut scan = RecoveryScan::default();
    let mut offset = 0_usize;
    let mut previous_sequence = 0_u64;
    while offset < bytes.len() {
        if scan.valid_entries == limits.max_entries {
            set_bad_tail(
                &mut scan,
                RecoveryTailStatus::LimitExceeded,
                format!(
                    "Recovery journal has more than {} entries; ignored the excess tail.",
                    limits.max_entries
                ),
            );
            break;
        }
        let remaining = bytes.len() - offset;
        if remaining < RECORD_HEADER_BYTES {
            set_bad_tail(
                &mut scan,
                if oversized_file {
                    RecoveryTailStatus::LimitExceeded
                } else {
                    RecoveryTailStatus::Truncated
                },
                "Recovery journal has an incomplete crash tail; using the latest valid checkpoint."
                    .to_string(),
            );
            break;
        }
        let header = &bytes[offset..offset + RECORD_HEADER_BYTES];
        if header[..8] != RECORD_MAGIC {
            set_bad_tail(
                &mut scan,
                RecoveryTailStatus::Corrupt,
                "Recovery journal checksum/header is corrupt; using the latest valid checkpoint."
                    .to_string(),
            );
            break;
        }
        let version = u16::from_le_bytes(header[8..10].try_into().expect("fixed header"));
        let flags = u16::from_le_bytes(header[10..12].try_into().expect("fixed header"));
        let sequence = u64::from_le_bytes(header[12..20].try_into().expect("fixed header"));
        let payload_len = u32::from_le_bytes(header[20..24].try_into().expect("fixed header"));
        let reserved = u32::from_le_bytes(header[24..28].try_into().expect("fixed header"));
        let expected_checksum: [u8; 32] = header[28..60].try_into().expect("fixed checksum header");
        if version != RECOVERY_JOURNAL_VERSION {
            set_bad_tail(
                &mut scan,
                RecoveryTailStatus::UnsupportedVersion,
                format!(
                    "Recovery journal version {version} is unsupported; using the latest compatible checkpoint."
                ),
            );
            break;
        }
        let payload_len_usize = payload_len as usize;
        if flags != RECORD_FLAGS
            || reserved != RECORD_RESERVED
            || sequence == 0
            || sequence <= previous_sequence
        {
            set_bad_tail(
                &mut scan,
                RecoveryTailStatus::Corrupt,
                "Recovery journal record order/header is corrupt; using the latest valid checkpoint."
                    .to_string(),
            );
            break;
        }
        if payload_len_usize > limits.max_payload_bytes {
            set_bad_tail(
                &mut scan,
                RecoveryTailStatus::LimitExceeded,
                format!(
                    "Recovery checkpoint is {payload_len_usize} bytes; limit is {}. Ignored the tail.",
                    limits.max_payload_bytes
                ),
            );
            break;
        }
        let end = offset
            .checked_add(RECORD_HEADER_BYTES)
            .and_then(|value| value.checked_add(payload_len_usize));
        let Some(end) = end.filter(|end| *end <= bytes.len()) else {
            set_bad_tail(
                &mut scan,
                if oversized_file {
                    RecoveryTailStatus::LimitExceeded
                } else {
                    RecoveryTailStatus::Truncated
                },
                "Recovery journal has an incomplete crash tail; using the latest valid checkpoint."
                    .to_string(),
            );
            break;
        };
        let payload = &bytes[offset + RECORD_HEADER_BYTES..end];
        let actual_checksum =
            record_checksum(version, flags, sequence, payload_len, reserved, payload);
        if actual_checksum != expected_checksum {
            set_bad_tail(
                &mut scan,
                RecoveryTailStatus::Corrupt,
                "Recovery journal checksum is corrupt; using the latest valid checkpoint."
                    .to_string(),
            );
            break;
        }
        let patch = match serde_yaml::from_slice::<PatchState>(payload) {
            Ok(patch) => patch,
            Err(_) => {
                set_bad_tail(
                    &mut scan,
                    RecoveryTailStatus::Corrupt,
                    "Recovery checkpoint payload is invalid; using the latest valid checkpoint."
                        .to_string(),
                );
                break;
            }
        };
        scan.valid_entries += 1;
        scan.valid_bytes = u64::try_from(end).unwrap_or(u64::MAX);
        scan.latest = Some(RecoveryCheckpoint {
            sequence,
            patch,
            payload: payload.to_vec(),
        });
        previous_sequence = sequence;
        offset = end;
    }
    if oversized_file && !scan.has_bad_tail() {
        set_bad_tail(
            &mut scan,
            RecoveryTailStatus::LimitExceeded,
            format!(
                "Recovery journal exceeds {} bytes; ignored the excess tail.",
                limits.max_file_bytes
            ),
        );
    }
    Ok(scan)
}

fn set_bad_tail(scan: &mut RecoveryScan, status: RecoveryTailStatus, warning: String) {
    scan.tail_status = status;
    scan.warning = Some(warning);
}

fn append_record(path: &Path, record: &[u8]) -> Result<(), RecoveryJournalError> {
    let mut file = OpenOptions::new()
        .append(true)
        .open(path)
        .map_err(|error| io_error("open recovery journal for append", error))?;
    file.write_all(record)
        .map_err(|error| io_error("append recovery checkpoint", error))?;
    file.sync_all()
        .map_err(|error| io_error("sync recovery checkpoint", error))
}

fn temporary_path(destination: &Path) -> PathBuf {
    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    let mut file_name = destination
        .file_name()
        .unwrap_or_else(|| std::ffi::OsStr::new("recovery.journal"))
        .to_os_string();
    let ordinal = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    file_name.push(format!(".tmp-{}-{ordinal}", std::process::id()));
    parent.join(file_name)
}

fn write_atomic_replacement(destination: &Path, bytes: &[u8]) -> Result<(), RecoveryJournalError> {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| io_error("create recovery journal directory", error))?;
    }
    let temp = temporary_path(destination);
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .map_err(|error| io_error("create temporary recovery journal", error))?;
        file.write_all(bytes)
            .map_err(|error| io_error("write temporary recovery journal", error))?;
        file.sync_all()
            .map_err(|error| io_error("sync temporary recovery journal", error))?;
        drop(file);
        if destination.exists() {
            atomic_replace(&temp, destination)
                .map_err(|error| io_error("replace recovery journal", error))
        } else {
            match rename_noreplace(&temp, destination) {
                Ok(()) => Ok(()),
                Err(_error) if destination.exists() => Err(RecoveryJournalError::DestinationRace(
                    destination.to_path_buf(),
                )),
                Err(error) => Err(io_error("publish new recovery journal", error)),
            }
        }
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

fn sync_parent(path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        if let Ok(directory) = File::open(parent) {
            directory.sync_all()?;
        }
    }
    Ok(())
}

#[cfg(windows)]
fn atomic_replace(source: &Path, destination: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    let source = source
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    let destination = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    let moved = unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if moved == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(not(windows))]
fn atomic_replace(source: &Path, destination: &Path) -> io::Result<()> {
    fs::rename(source, destination)
}

#[cfg(windows)]
fn rename_noreplace(source: &Path, destination: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::MoveFileExW;

    let source = source
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    let destination = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    let moved = unsafe { MoveFileExW(source.as_ptr(), destination.as_ptr(), 0) };
    if moved == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn rename_noreplace(source: &Path, destination: &Path) -> io::Result<()> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let source = CString::new(source.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "source path contains NUL"))?;
    let destination = CString::new(destination.as_os_str().as_bytes()).map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidInput, "destination path contains NUL")
    })?;
    let moved = unsafe {
        libc::syscall(
            libc::SYS_renameat2,
            libc::AT_FDCWD,
            source.as_ptr(),
            libc::AT_FDCWD,
            destination.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    if moved == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(target_vendor = "apple")]
fn rename_noreplace(source: &Path, destination: &Path) -> io::Result<()> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let source = CString::new(source.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "source path contains NUL"))?;
    let destination = CString::new(destination.as_os_str().as_bytes()).map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidInput, "destination path contains NUL")
    })?;
    let moved =
        unsafe { libc::renamex_np(source.as_ptr(), destination.as_ptr(), libc::RENAME_EXCL) };
    if moved == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(all(
    unix,
    not(any(target_os = "linux", target_os = "android", target_vendor = "apple"))
))]
fn rename_noreplace(_source: &Path, _destination: &Path) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "atomic no-replace rename is unavailable on this target",
    ))
}

#[cfg(not(any(windows, unix)))]
fn rename_noreplace(_source: &Path, _destination: &Path) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "atomic no-replace rename is unavailable on this target",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempJournal(PathBuf);

    impl TempJournal {
        fn new(label: &str) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            Self(std::env::temp_dir().join(format!(
                "collide-o-scope-{label}-{}-{nonce}.journal",
                std::process::id()
            )))
        }
    }

    impl Drop for TempJournal {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
            if let Some(parent) = self.0.parent() {
                let prefix = self
                    .0
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                if let Ok(entries) = fs::read_dir(parent) {
                    for entry in entries.flatten() {
                        if entry.file_name().to_string_lossy().starts_with(&prefix)
                            && entry.file_name().to_string_lossy().contains(".tmp-")
                        {
                            let _ = fs::remove_file(entry.path());
                        }
                    }
                }
            }
        }
    }

    fn patch(seed: u32) -> PatchState {
        serde_yaml::from_str(&format!("master:\n  random_seed: {seed}\nlayers: []\n")).unwrap()
    }

    fn seed(checkpoint: &RecoveryCheckpoint) -> u32 {
        checkpoint.patch.master.random_seed
    }

    #[derive(Default)]
    struct MockBackendState {
        blocked: bool,
        started_seeds: Vec<u32>,
        fail_seeds: BTreeSet<u32>,
        recoveries: usize,
        worker_names: Vec<String>,
    }

    #[derive(Default)]
    struct MockBackendControl {
        state: Mutex<MockBackendState>,
        changed: Condvar,
    }

    impl MockBackendControl {
        fn set_blocked(&self, blocked: bool) {
            self.state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .blocked = blocked;
            self.changed.notify_all();
        }

        fn fail_seed(&self, seed: u32) {
            self.state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .fail_seeds
                .insert(seed);
        }

        fn wait_for_started(&self, count: usize) {
            let deadline = Instant::now() + Duration::from_secs(2);
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            while state.started_seeds.len() < count {
                let remaining = deadline
                    .checked_duration_since(Instant::now())
                    .expect("mock recovery writer did not start in time");
                let (next, wait) = self
                    .changed
                    .wait_timeout(state, remaining)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                state = next;
                assert!(
                    !wait.timed_out(),
                    "mock recovery writer did not start in time"
                );
            }
        }
    }

    struct MockBackend {
        scan: RecoveryScan,
        control: Arc<MockBackendControl>,
    }

    impl MockBackend {
        fn new(initial: Option<(u64, PatchState)>, control: Arc<MockBackendControl>) -> Self {
            let scan = initial.map_or_else(RecoveryScan::default, |(sequence, patch)| {
                let payload = serde_yaml::to_string(&patch).unwrap().into_bytes();
                RecoveryScan {
                    latest: Some(RecoveryCheckpoint {
                        sequence,
                        patch,
                        payload: payload.clone(),
                    }),
                    valid_entries: 1,
                    valid_bytes: u64::try_from(RECORD_HEADER_BYTES + payload.len()).unwrap(),
                    tail_status: RecoveryTailStatus::Clean,
                    warning: None,
                }
            });
            Self { scan, control }
        }
    }

    impl RecoveryWriterBackend for MockBackend {
        fn latest_valid(&self) -> &RecoveryScan {
            &self.scan
        }

        fn append_patch(
            &mut self,
            patch: &PatchState,
        ) -> Result<JournalAppendReceipt, RecoveryJournalError> {
            let requested_seed = patch.master.random_seed;
            let fail = {
                let mut state = self
                    .control
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                state.started_seeds.push(requested_seed);
                state
                    .worker_names
                    .push(thread::current().name().unwrap_or("unnamed").to_string());
                self.control.changed.notify_all();
                while state.blocked {
                    state = self
                        .control
                        .changed
                        .wait(state)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
                state.fail_seeds.remove(&requested_seed)
            };
            if fail {
                return Err(RecoveryJournalError::Serialize(
                    "x".repeat(RECOVERY_WRITER_STATUS_MAX_BYTES * 4),
                ));
            }
            let payload = serde_yaml::to_string(patch).unwrap().into_bytes();
            let hostile_round_trip: PatchState = serde_yaml::from_slice(&payload).unwrap();
            let sequence = self
                .scan
                .latest
                .as_ref()
                .map_or(1, |checkpoint| checkpoint.sequence + 1);
            self.scan = RecoveryScan {
                latest: Some(RecoveryCheckpoint {
                    sequence,
                    patch: hostile_round_trip,
                    payload: payload.clone(),
                }),
                valid_entries: self.scan.valid_entries.saturating_add(1),
                valid_bytes: self
                    .scan
                    .valid_bytes
                    .saturating_add(u64::try_from(RECORD_HEADER_BYTES + payload.len()).unwrap()),
                tail_status: RecoveryTailStatus::Clean,
                warning: None,
            };
            Ok(JournalAppendReceipt {
                sequence,
                payload_bytes: payload.len(),
                journal_bytes: self.scan.valid_bytes,
                compacted: sequence == 1,
            })
        }

        fn discard(&mut self) -> Result<(), RecoveryJournalError> {
            self.scan = RecoveryScan::default();
            Ok(())
        }

        fn recover_after_failure(&mut self) -> Result<(), RecoveryJournalError> {
            self.control
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .recoveries += 1;
            Ok(())
        }
    }

    fn mock_writer(
        label: &str,
        initial: Option<(u64, PatchState)>,
        control: Arc<MockBackendControl>,
    ) -> RecoveryWriter {
        let temporary = TempJournal::new(label);
        let path = temporary.0.clone();
        let backend = MockBackend::new(initial, control);
        RecoveryWriter::start_with_factory(
            path,
            Box::new(move || Ok(Box::new(backend) as Box<dyn RecoveryWriterBackend>)),
        )
        .unwrap()
    }

    fn wait_for_writer(
        writer: &RecoveryWriter,
        predicate: impl Fn(&RecoveryWriterStatus) -> bool,
    ) -> RecoveryWriterStatus {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let status = writer.status();
            if predicate(&status) {
                return status;
            }
            assert!(
                Instant::now() < deadline,
                "recovery writer status did not converge: {status:?}"
            );
            thread::yield_now();
        }
    }

    #[test]
    fn writer_runs_real_yaml_sync_and_rescan_work_off_the_caller_thread() {
        let path = TempJournal::new("writer-real");
        let writer = RecoveryWriter::start(path.0.clone()).unwrap();
        wait_for_writer(&writer, |status| {
            status.lifecycle == RecoveryWriterLifecycle::Idle
        });
        let receipt = writer.request_checkpoint(patch(101)).unwrap();
        assert_eq!(receipt.durable_sequence_at_request, None);
        let status = wait_for_writer(&writer, |status| {
            status.last_durable_request_id == Some(receipt.request_id)
        });
        assert_eq!(status.durable_sequence, Some(1));
        assert_eq!(status.queue_depth, 0);
        let reopened = RecoveryJournal::open(&path.0).unwrap();
        assert_eq!(seed(reopened.latest_valid().latest.as_ref().unwrap()), 101);
        assert_eq!(
            writer.latest_checkpoint().map(|(_, _, sequence)| sequence),
            Some(1)
        );
        assert!(writer.flush_with_deadline(Duration::from_secs(1)).completed);
    }

    #[test]
    fn writer_surfaces_a_bad_tail_then_compacts_it_only_with_a_verified_request() {
        let path = TempJournal::new("writer-bad-tail");
        let mut journal = RecoveryJournal::open(&path.0).unwrap();
        journal.append_patch(&patch(40)).unwrap();
        OpenOptions::new()
            .append(true)
            .open(&path.0)
            .unwrap()
            .write_all(&RECORD_MAGIC[..3])
            .unwrap();

        let writer = RecoveryWriter::start(path.0.clone()).unwrap();
        let damaged = wait_for_writer(&writer, |status| {
            status.lifecycle == RecoveryWriterLifecycle::Idle
        });
        assert_eq!(damaged.durable_sequence, Some(1));
        assert!(damaged
            .error
            .as_deref()
            .is_some_and(|warning| warning.contains("incomplete crash tail")));
        assert_eq!(seed(writer.durable_scan().latest.as_ref().unwrap()), 40);

        let request = writer.request_checkpoint(patch(41)).unwrap();
        let repaired = wait_for_writer(&writer, |status| {
            status.last_durable_request_id == Some(request.request_id)
        });
        assert_eq!(repaired.durable_sequence, Some(2));
        assert!(repaired.error.is_none());
        assert!(repaired.last_receipt.unwrap().compacted);
        assert_eq!(
            RecoveryJournal::open(&path.0)
                .unwrap()
                .latest_valid()
                .tail_status,
            RecoveryTailStatus::Clean
        );
        assert!(writer.flush_with_deadline(Duration::from_secs(1)).completed);
    }

    #[test]
    fn slow_disk_has_one_in_flight_and_one_newest_wins_pending_slot() {
        let control = Arc::new(MockBackendControl::default());
        control.set_blocked(true);
        let writer = mock_writer(
            "writer-coalesce",
            Some((7, patch(70))),
            Arc::clone(&control),
        );
        wait_for_writer(&writer, |status| {
            status.lifecycle == RecoveryWriterLifecycle::Idle
        });

        let first = writer.request_checkpoint(patch(1)).unwrap();
        control.wait_for_started(1);
        let enqueue_started = Instant::now();
        let stale = writer.request_checkpoint(patch(2)).unwrap();
        let newest = writer.request_checkpoint(patch(3)).unwrap();
        assert!(enqueue_started.elapsed() < Duration::from_millis(250));
        assert_eq!(newest.superseded_request_id, Some(stale.request_id));
        assert_eq!(newest.durable_sequence_at_request, Some(7));

        let blocked = writer.status();
        assert_eq!(blocked.in_flight_request_id, Some(first.request_id));
        assert_eq!(blocked.pending_request_id, Some(newest.request_id));
        assert_eq!(blocked.durable_sequence, Some(7));
        assert_eq!(blocked.queue_depth, RECOVERY_WRITER_MAX_JOBS);
        assert_eq!(blocked.max_observed_jobs, RECOVERY_WRITER_MAX_JOBS);
        assert_eq!(blocked.superseded_requests, 1);
        assert!(blocked.line().contains("remains authoritative"));
        assert!(blocked.line().contains("newest pending request"));

        control.set_blocked(false);
        let finished = wait_for_writer(&writer, |status| {
            status.last_durable_request_id == Some(newest.request_id)
        });
        assert_eq!(finished.durable_sequence, Some(9));
        assert_eq!(finished.queue_depth, 0);
        let state = control
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(state.started_seeds, vec![1, 3]);
        assert!(state
            .worker_names
            .iter()
            .all(|name| name == "recovery-writer"));
        drop(state);
        assert_eq!(seed(writer.durable_scan().latest.as_ref().unwrap()), 3);
        assert!(writer.flush_with_deadline(Duration::from_secs(1)).completed);
    }

    #[test]
    fn failed_request_never_advances_the_prior_acknowledged_checkpoint() {
        let control = Arc::new(MockBackendControl::default());
        control.fail_seed(20);
        let writer = mock_writer("writer-fault", Some((11, patch(10))), Arc::clone(&control));
        wait_for_writer(&writer, |status| {
            status.lifecycle == RecoveryWriterLifecycle::Idle
        });

        let failed = writer.request_checkpoint(patch(20)).unwrap();
        let status = wait_for_writer(&writer, |status| {
            status.last_failed_request_id == Some(failed.request_id)
        });
        assert_eq!(status.lifecycle, RecoveryWriterLifecycle::Failed);
        assert_eq!(status.durable_sequence, Some(11));
        assert_ne!(status.last_durable_request_id, Some(failed.request_id));
        assert_eq!(status.failed_requests, 1);
        assert!(status.error.as_ref().unwrap().len() <= RECOVERY_WRITER_STATUS_MAX_BYTES);
        assert!(status.line().len() <= RECOVERY_WRITER_STATUS_MAX_BYTES);
        assert_eq!(seed(writer.durable_scan().latest.as_ref().unwrap()), 10);
        assert_eq!(
            control
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .recoveries,
            1
        );

        let good = writer.request_checkpoint(patch(30)).unwrap();
        let recovered = wait_for_writer(&writer, |status| {
            status.last_durable_request_id == Some(good.request_id)
        });
        assert_eq!(recovered.durable_sequence, Some(12));
        assert_eq!(seed(writer.durable_scan().latest.as_ref().unwrap()), 30);
        assert!(writer.flush_with_deadline(Duration::from_secs(1)).completed);
    }

    #[test]
    fn clean_shutdown_drains_the_newest_pending_checkpoint() {
        let control = Arc::new(MockBackendControl::default());
        let writer = mock_writer("writer-flush", None, Arc::clone(&control));
        wait_for_writer(&writer, |status| {
            status.lifecycle == RecoveryWriterLifecycle::Idle
        });
        let first = writer.request_checkpoint(patch(1)).unwrap();
        let newest = writer.request_checkpoint(patch(2)).unwrap();
        let flush = writer.flush_with_deadline(Duration::from_secs(1));
        assert!(flush.completed);
        assert!(matches!(flush.durable_sequence, Some(1 | 2)));
        let scan = writer.durable_scan();
        let durable = scan.latest.as_ref().unwrap();
        // Depending on whether the worker acquired the first request before
        // the second enqueue, seed 1 is either in-flight then seed 2 is durable
        // sequence 2, or it is coalesced and seed 2 is durable sequence 1.
        assert_eq!(seed(durable), 2);
        assert!(matches!(durable.sequence, 1 | 2));
        assert_ne!(first.request_id, newest.request_id);
        assert!(writer.request_checkpoint(patch(3)).is_err());
    }

    #[test]
    fn shutdown_timeout_is_visible_and_never_joins_the_stalled_worker() {
        let control = Arc::new(MockBackendControl::default());
        control.set_blocked(true);
        let writer = mock_writer("writer-timeout", None, Arc::clone(&control));
        wait_for_writer(&writer, |status| {
            status.lifecycle == RecoveryWriterLifecycle::Idle
        });
        let request = writer.request_checkpoint(patch(4)).unwrap();
        control.wait_for_started(1);

        let started = Instant::now();
        let flush = writer.flush_with_deadline(Duration::from_millis(20));
        assert!(!flush.completed);
        assert_eq!(flush.in_flight_request_id, Some(request.request_id));
        assert!(started.elapsed() < Duration::from_millis(500));
        assert_eq!(writer.status().lifecycle, RecoveryWriterLifecycle::Flushing);

        // Let the detached-capable worker retire so this test itself leaves no
        // process-lifetime work behind. The first flush already proved the
        // caller returned without joining it.
        control.set_blocked(false);
        assert!(writer.flush_with_deadline(Duration::from_secs(1)).completed);
    }

    #[test]
    fn append_and_reopen_returns_latest_valid_patch_without_runtime_pixels() {
        let path = TempJournal::new("roundtrip");
        let mut journal = RecoveryJournal::open(&path.0).unwrap();
        let first = journal.append_patch(&patch(11)).unwrap();
        assert!(first.compacted);
        let second = journal.append_patch(&patch(22)).unwrap();
        assert!(!second.compacted);
        let reopened = RecoveryJournal::open(&path.0).unwrap();
        assert_eq!(reopened.latest_valid().valid_entries, 2);
        assert_eq!(
            reopened.latest_valid().tail_status,
            RecoveryTailStatus::Clean
        );
        let latest = reopened.latest_valid().latest.as_ref().unwrap();
        assert_eq!(latest.sequence, 2);
        assert_eq!(seed(latest), 22);
        let payload = String::from_utf8(latest.payload.clone()).unwrap();
        assert!(!payload.contains("texture"));
        assert!(!payload.contains("carrier_pixels"));
    }

    #[test]
    fn crash_tail_is_ignored_read_only_and_reported_separately() {
        let path = TempJournal::new("crash-tail");
        let mut journal = RecoveryJournal::open(&path.0).unwrap();
        journal.append_patch(&patch(1)).unwrap();
        journal.append_patch(&patch(2)).unwrap();
        OpenOptions::new()
            .append(true)
            .open(&path.0)
            .unwrap()
            .write_all(&RECORD_MAGIC[..3])
            .unwrap();
        let damaged_len = fs::metadata(&path.0).unwrap().len();
        let mut reopened = RecoveryJournal::open(&path.0).unwrap();
        assert_eq!(
            reopened.latest_valid().tail_status,
            RecoveryTailStatus::Truncated
        );
        assert_eq!(seed(reopened.latest_valid().latest.as_ref().unwrap()), 2);
        assert!(reopened.latest_valid().warning.is_some());
        assert_eq!(fs::metadata(&path.0).unwrap().len(), damaged_len);

        // A newly successful checkpoint is the first operation authorized to
        // compact away that crash tail.
        let receipt = reopened.append_patch(&patch(3)).unwrap();
        assert!(receipt.compacted);
        assert_eq!(reopened.latest_valid().valid_entries, 1);
        assert_eq!(
            reopened.latest_valid().tail_status,
            RecoveryTailStatus::Clean
        );
        assert_eq!(seed(reopened.latest_valid().latest.as_ref().unwrap()), 3);
    }

    #[test]
    fn a_field_collider_block_round_trips_through_the_checksummed_journal() {
        use crate::patch::{
            FaradayConfig, FieldColliderConfig, FieldColliderModeConfig, MotionBoundaryModeConfig,
            MotionConfig, MotionDonorConfig,
        };
        use crate::performance::SavedLayerPosition;

        let collider = FieldColliderConfig {
            enabled: true,
            mode: FieldColliderModeConfig::Projection,
            boundary: MotionBoundaryModeConfig::Wrap,
            input_a: MotionDonorConfig::Selected {
                saved_position: SavedLayerPosition::new(1).unwrap(),
            },
            // A retained tombstone must survive the journal as a tombstone.
            input_b: MotionDonorConfig::Missing {
                saved_position: SavedLayerPosition::new(4).unwrap(),
            },
            ..FieldColliderConfig::default()
        };
        let motion = MotionConfig {
            transplant: FaradayConfig {
                amount: 0.6,
                ..FaradayConfig::default()
            },
            collider,
            ..MotionConfig::default()
        };

        let mut carrier: PatchState = serde_yaml::from_str(
            "master:
  random_seed: 83
layers:
  - filename: collider.mov
",
        )
        .unwrap();
        carrier.layers[0].motion = Some(motion);

        let path = TempJournal::new("field-collider");
        let mut journal = RecoveryJournal::open(&path.0).unwrap();
        let receipt = journal.append_patch(&carrier).unwrap();
        assert!(receipt.payload_bytes > 0);

        let reopened = RecoveryJournal::open(&path.0).unwrap();
        assert_eq!(
            reopened.latest_valid().tail_status,
            RecoveryTailStatus::Clean
        );
        assert!(!reopened.latest_valid().has_bad_tail());
        let latest = reopened.latest_valid().latest.as_ref().unwrap();
        assert_eq!(seed(latest), 83);

        let restored = latest.patch.layers[0]
            .motion
            .expect("the journal payload carries the collider block");
        assert_eq!(restored.collider, collider);
        assert!(restored.collider.enabled);
        assert_eq!(restored.collider.mode, FieldColliderModeConfig::Projection);
        assert_eq!(restored.collider.boundary, MotionBoundaryModeConfig::Wrap);
        // The tombstone is still a tombstone and still names its saved position.
        assert_eq!(
            restored.collider.input_b,
            MotionDonorConfig::Missing {
                saved_position: SavedLayerPosition::new(4).unwrap()
            }
        );

        // Authored topology only. No derived vector, no transient pair, no gate
        // parity, no process-lifetime identity, and no filesystem metadata.
        let payload = String::from_utf8(latest.payload.clone()).unwrap();
        assert!(payload.contains("collider"));
        assert!(!payload.contains("layer_id"));
        assert!(!payload.contains("derived"));
        assert!(!payload.contains("transient"));
        assert!(!payload.contains("velocity"));
        assert!(!payload.contains("texture"));
        assert!(!payload.contains("modified"));
        assert!(!payload.contains(std::env::temp_dir().to_string_lossy().as_ref()));

        // A patch with no collider section at all still loads: absent is
        // exactly the pre-collider path.
        let mut legacy: PatchState = serde_yaml::from_str(
            "master:
  random_seed: 84
layers:
  - filename: legacy.mov
",
        )
        .unwrap();
        legacy.layers[0].motion = Some(MotionConfig::default());
        let legacy_path = TempJournal::new("field-collider-legacy");
        let mut legacy_journal = RecoveryJournal::open(&legacy_path.0).unwrap();
        legacy_journal.append_patch(&legacy).unwrap();
        let legacy_payload = String::from_utf8(
            RecoveryJournal::open(&legacy_path.0)
                .unwrap()
                .latest_valid()
                .latest
                .as_ref()
                .unwrap()
                .payload
                .clone(),
        )
        .unwrap();
        assert!(!legacy_payload.contains("collider"));
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn a_recorded_gesture_track_round_trips_through_the_journal_and_still_validates() {
        use crate::gesture::{
            GestureEvent, GestureMode, GesturePhase, GestureTrack, GestureTrackDocument,
        };

        let mut track = GestureTrack::default();
        for (tick, phase) in [
            (700_u64, GesturePhase::Begin),
            (701, GesturePhase::Move),
            (703, GesturePhase::Move),
            (709, GesturePhase::End),
        ] {
            assert!(track
                .record_accepted(
                    tick,
                    GestureEvent::quantized(
                        2,
                        phase,
                        GestureMode::Curl,
                        [0.25, 0.75],
                        0.5,
                        [0.0, 1.0],
                    ),
                )
                .unwrap());
        }
        let document = GestureTrackDocument::capture(&track);
        let expected_checksum = track.checksum_hex();
        assert_eq!(document.checksum, expected_checksum);

        let mut carrier = patch(77);
        carrier.gesture_track = Some(document.clone());

        let path = TempJournal::new("gesture-track");
        let mut journal = RecoveryJournal::open(&path.0).unwrap();
        let receipt = journal.append_patch(&carrier).unwrap();
        assert!(receipt.payload_bytes > 0);

        // The record's own checksum still validates with the track present:
        // a clean tail is exactly what `scan_bytes` reports when every
        // `record_checksum` matched.
        let reopened = RecoveryJournal::open(&path.0).unwrap();
        assert_eq!(
            reopened.latest_valid().tail_status,
            RecoveryTailStatus::Clean
        );
        assert!(!reopened.latest_valid().has_bad_tail());
        let latest = reopened.latest_valid().latest.as_ref().unwrap();
        assert_eq!(seed(latest), 77);

        // And the track itself round-tripped byte for byte, through the same
        // acceptance path live ingest uses.
        let restored = latest
            .patch
            .gesture_track
            .as_ref()
            .expect("the journal payload carries the recorded gesture track");
        assert_eq!(restored, &document);
        let decoded = restored.decode().unwrap();
        assert_eq!(decoded, track);
        assert_eq!(decoded.checksum_hex(), expected_checksum);
        assert_eq!(decoded.events().len(), 4);
        assert_eq!(decoded.origin_tick(), Some(700));
        assert!(decoded.is_complete());

        // Operational paths and filesystem metadata never enter the payload.
        let payload = String::from_utf8(latest.payload.clone()).unwrap();
        assert!(payload.contains(&expected_checksum));
        assert!(!payload.contains("modified"));
        assert!(!payload.contains(std::env::temp_dir().to_string_lossy().as_ref()));

        // A payload whose declared digest no longer describes its events is
        // refused by the patch boundary rather than accepted as a recording
        // the operator never made.
        let mut tampered = carrier.clone();
        let mut broken = document.clone();
        broken.events[1].x = broken.events[1].x.wrapping_add(1);
        tampered.gesture_track = Some(broken);
        let serialized = serde_yaml::to_string(&tampered).unwrap();
        let error = match serde_yaml::from_str::<PatchState>(&serialized) {
            Ok(_) => panic!("a rewritten event must fail the re-derived digest"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("checksum"),
            "a rewritten event must fail the re-derived digest: {error}"
        );

        // An absent section is exactly the pre-gesture path.
        let plain = serde_yaml::to_string(&patch(78)).unwrap();
        assert!(!plain.contains("gesture_track"));
        assert!(serde_yaml::from_str::<PatchState>(&plain)
            .expect("a legacy patch with no gesture section still loads")
            .gesture_track
            .is_none());
    }

    #[test]
    fn checksum_corruption_preserves_the_valid_prefix_and_never_overwrites_patch() {
        let path = TempJournal::new("checksum");
        let mut journal = RecoveryJournal::open(&path.0).unwrap();
        journal.append_patch(&patch(7)).unwrap();
        journal.append_patch(&patch(8)).unwrap();
        let first_len = encode_record(1, &journal_scan_payload(&path.0, 1))
            .unwrap()
            .len();
        let mut bytes = fs::read(&path.0).unwrap();
        let corrupt_at = first_len + RECORD_HEADER_BYTES;
        bytes[corrupt_at] ^= 0x40;
        fs::write(&path.0, &bytes).unwrap();
        let reopened = RecoveryJournal::open(&path.0).unwrap();
        assert_eq!(
            reopened.latest_valid().tail_status,
            RecoveryTailStatus::Corrupt
        );
        assert_eq!(reopened.latest_valid().valid_entries, 1);
        assert_eq!(seed(reopened.latest_valid().latest.as_ref().unwrap()), 7);
    }

    fn journal_scan_payload(path: &Path, wanted_sequence: u64) -> Vec<u8> {
        // Used only for the first record in a two-record fixture; rescan raw
        // bytes to avoid exposing every retained payload in the public API.
        let bytes = fs::read(path).unwrap();
        let payload_len = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;
        assert_eq!(wanted_sequence, 1);
        bytes[RECORD_HEADER_BYTES..RECORD_HEADER_BYTES + payload_len].to_vec()
    }

    #[test]
    fn count_cap_compacts_only_while_publishing_a_new_checkpoint() {
        let path = TempJournal::new("cap");
        let limits = RecoveryLimits {
            max_entries: 2,
            max_file_bytes: 64 * 1024,
            max_payload_bytes: 16 * 1024,
        };
        let mut journal = RecoveryJournal::open_with_limits(&path.0, limits).unwrap();
        assert!(journal.append_patch(&patch(1)).unwrap().compacted);
        assert!(!journal.append_patch(&patch(2)).unwrap().compacted);
        let receipt = journal.append_patch(&patch(3)).unwrap();
        assert!(receipt.compacted);
        assert_eq!(journal.latest_valid().valid_entries, 1);
        assert_eq!(seed(journal.latest_valid().latest.as_ref().unwrap()), 3);
    }

    #[test]
    fn explicit_compaction_and_discard_are_atomic_and_bounded() {
        let path = TempJournal::new("discard");
        let mut journal = RecoveryJournal::open(&path.0).unwrap();
        journal.append_patch(&patch(4)).unwrap();
        journal.append_patch(&patch(5)).unwrap();
        journal.compact_to_latest().unwrap();
        assert_eq!(journal.latest_valid().valid_entries, 1);
        assert_eq!(seed(journal.latest_valid().latest.as_ref().unwrap()), 5);
        journal.discard().unwrap();
        assert!(!journal.latest_valid().recovery_available());
        assert_eq!(fs::metadata(&path.0).unwrap().len(), 0);
    }

    #[test]
    fn hostile_headers_and_oversized_payloads_fail_closed_without_allocation() {
        let path = TempJournal::new("hostile");
        let mut header = vec![0_u8; RECORD_HEADER_BYTES];
        header[..8].copy_from_slice(&RECORD_MAGIC);
        header[8..10].copy_from_slice(&RECOVERY_JOURNAL_VERSION.to_le_bytes());
        header[12..20].copy_from_slice(&1_u64.to_le_bytes());
        header[20..24].copy_from_slice(&u32::MAX.to_le_bytes());
        fs::write(&path.0, header).unwrap();
        let journal = RecoveryJournal::open(&path.0).unwrap();
        assert_eq!(
            journal.latest_valid().tail_status,
            RecoveryTailStatus::LimitExceeded
        );
        assert!(journal.latest_valid().latest.is_none());
    }
}
