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

use sha2::{Digest, Sha256};

use crate::patch::PatchState;

pub const RECOVERY_JOURNAL_VERSION: u16 = 1;
pub const RECOVERY_MAX_ENTRIES: usize = 256;
pub const RECOVERY_MAX_FILE_BYTES: u64 = 64 * 1024 * 1024;
pub const RECOVERY_MAX_PAYLOAD_BYTES: usize = 8 * 1024 * 1024;
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
    #[cfg_attr(
        test,
        allow(
            dead_code,
            reason = "App tests use explicit temporary recovery paths instead of the operator default"
        )
    )]
    pub fn open_default() -> Result<Self, RecoveryJournalError> {
        Self::open(default_recovery_journal_path())
    }

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

    pub fn path(&self) -> &Path {
        &self.path
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

fn encode_record(sequence: u64, payload: &[u8]) -> Result<Vec<u8>, RecoveryJournalError> {
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

fn scan_bytes(
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
