//! One crash-safe publication law for user-authored files.
//!
//! Both native patch saves and browser uploads stage a newly created file in
//! the destination directory, flush and sync that file, publish with an
//! explicit replace/no-replace operation, and finally sync the parent.  The
//! staging guard removes an unpublished artifact on every ordinary error or
//! cancellation path.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Whether the atomic publication is allowed to replace an existing name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PublishMode {
    Replace,
    NoReplace,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AdmissionLimits {
    pub max_concurrent: usize,
    pub max_reserved_bytes: u64,
    pub min_free_after_reservations: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AdmissionError {
    Concurrency,
    AggregateBytes,
    DiskHeadroom,
    CleanupBusy,
}

#[derive(Debug, Default)]
struct AdmissionState {
    active: usize,
    reserved_bytes: u64,
    cleaned_generation: u64,
    cleanup_in_progress: Option<u64>,
}

/// Process-wide upload admission shared by all three listener roles. A lease
/// is released by Drop, including when Hyper cancels a handler on disconnect.
#[derive(Debug, Clone, Default)]
pub(crate) struct UploadAdmission {
    inner: Arc<Mutex<AdmissionState>>,
}

impl UploadAdmission {
    /// Claim the one orphan-cleanup pass for a library generation. A folder
    /// change waits for old leases to cancel before scanning, and other new
    /// uploads are refused while the bounded scan is running.
    pub(crate) fn begin_cleanup(
        &self,
        generation: u64,
    ) -> Result<Option<UploadCleanupLease>, AdmissionError> {
        let mut state = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if generation != 0 && state.cleaned_generation == generation {
            return Ok(None);
        }
        if state.cleanup_in_progress.is_some() || state.active != 0 {
            return Err(AdmissionError::CleanupBusy);
        }
        state.cleanup_in_progress = Some(generation);
        Ok(Some(UploadCleanupLease {
            inner: self.inner.clone(),
            generation,
            completed: false,
        }))
    }

    pub(crate) fn mark_cleanup_complete(&self, generation: u64) {
        let mut state = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.active == 0 && state.cleanup_in_progress.is_none() && generation != 0 {
            state.cleaned_generation = generation;
        }
    }

    pub(crate) fn try_reserve(
        &self,
        requested_bytes: u64,
        available_bytes: u64,
        limits: AdmissionLimits,
    ) -> Result<UploadLease, AdmissionError> {
        let mut state = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.cleanup_in_progress.is_some() {
            return Err(AdmissionError::CleanupBusy);
        }
        if state.active >= limits.max_concurrent {
            return Err(AdmissionError::Concurrency);
        }
        let next_reserved = state
            .reserved_bytes
            .checked_add(requested_bytes)
            .ok_or(AdmissionError::AggregateBytes)?;
        if next_reserved > limits.max_reserved_bytes {
            return Err(AdmissionError::AggregateBytes);
        }
        let required_free = next_reserved
            .checked_add(limits.min_free_after_reservations)
            .ok_or(AdmissionError::DiskHeadroom)?;
        if available_bytes < required_free {
            return Err(AdmissionError::DiskHeadroom);
        }
        state.active += 1;
        state.reserved_bytes = next_reserved;
        Ok(UploadLease {
            inner: self.inner.clone(),
            reserved_bytes: requested_bytes,
        })
    }

    #[cfg(test)]
    fn snapshot(&self) -> (usize, u64) {
        let state = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        (state.active, state.reserved_bytes)
    }
}

#[derive(Debug)]
pub(crate) struct UploadLease {
    inner: Arc<Mutex<AdmissionState>>,
    reserved_bytes: u64,
}

#[derive(Debug)]
pub(crate) struct UploadCleanupLease {
    inner: Arc<Mutex<AdmissionState>>,
    generation: u64,
    completed: bool,
}

impl UploadCleanupLease {
    pub(crate) fn complete(mut self) {
        let mut state = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.cleanup_in_progress == Some(self.generation) {
            state.cleanup_in_progress = None;
            state.cleaned_generation = self.generation;
        }
        self.completed = true;
    }
}

impl Drop for UploadCleanupLease {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        let mut state = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.cleanup_in_progress == Some(self.generation) {
            state.cleanup_in_progress = None;
        }
    }
}

impl Drop for UploadLease {
    fn drop(&mut self) {
        let mut state = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.active = state.active.saturating_sub(1);
        state.reserved_bytes = state.reserved_bytes.saturating_sub(self.reserved_bytes);
    }
}

/// An unpublished same-directory file. Dropping the guard is cancellation:
/// the staging name is removed and the destination is never touched.
#[derive(Debug)]
pub(crate) struct StagedPublication {
    destination: PathBuf,
    staging: Option<PathBuf>,
}

impl StagedPublication {
    /// Reserve a cryptographically unpredictable staging name with
    /// `create_new`. `prefix` is a fixed internal label, never user input.
    pub(crate) fn create(destination: &Path, prefix: &str) -> io::Result<(Self, File)> {
        let file_name = destination.file_name().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "destination has no file name")
        })?;
        let parent = destination
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        if !parent.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("destination directory {} does not exist", parent.display()),
            ));
        }

        for _ in 0..16 {
            let mut nonce = [0_u8; 16];
            getrandom::fill(&mut nonce)
                .map_err(|error| io::Error::other(format!("staging entropy: {error}")))?;
            let suffix = nonce
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            let staging = parent.join(format!(".{prefix}-{}-{suffix}.part", std::process::id()));
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&staging)
            {
                Ok(file) => {
                    return Ok((
                        Self {
                            destination: parent.join(file_name),
                            staging: Some(staging),
                        },
                        file,
                    ));
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not reserve a unique staging file",
        ))
    }

    pub(crate) fn staging_path(&self) -> &Path {
        self.staging
            .as_deref()
            .expect("an unpublished transaction always owns its staging path")
    }

    /// Commit after the caller's final validation barrier. The callback runs
    /// after the file sync and immediately before the atomic name operation;
    /// upload callers use it to re-check generation, shutdown, and deadline.
    pub(crate) fn commit_if<F>(
        mut self,
        mut file: File,
        mode: PublishMode,
        before_publish: F,
    ) -> io::Result<()>
    where
        F: FnOnce() -> io::Result<()>,
    {
        file.flush()?;
        file.sync_all()?;
        drop(file);
        before_publish()?;

        let staging = self
            .staging
            .as_deref()
            .expect("publication staging path disappeared before commit");
        match mode {
            PublishMode::Replace => atomic_replace(staging, &self.destination)?,
            PublishMode::NoReplace => rename_noreplace(staging, &self.destination)?,
        }
        // The staging name no longer exists after either successful atomic
        // operation. Disarm before directory sync so Drop never treats a
        // post-publication sync error as an unpublished file.
        self.staging = None;
        sync_parent(&self.destination)
    }

    pub(crate) fn commit(self, file: File, mode: PublishMode) -> io::Result<()> {
        self.commit_if(file, mode, || Ok(()))
    }

    /// Commit a file that the caller has already flushed and `sync_all`'d.
    /// Uploads use this after their media probe so the short generation gate
    /// encloses only the atomic name operation and parent sync, never body IO
    /// or decoding.
    pub(crate) fn commit_presynced(mut self, file: File, mode: PublishMode) -> io::Result<()> {
        drop(file);
        let staging = self
            .staging
            .as_deref()
            .expect("publication staging path disappeared before commit");
        match mode {
            PublishMode::Replace => atomic_replace(staging, &self.destination)?,
            PublishMode::NoReplace => rename_noreplace(staging, &self.destination)?,
        }
        self.staging = None;
        sync_parent(&self.destination)
    }
}

impl Drop for StagedPublication {
    fn drop(&mut self) {
        if let Some(staging) = self.staging.take() {
            let _ = fs::remove_file(staging);
        }
    }
}

/// Publish an already validated byte document using the shared transaction.
pub(crate) fn publish_bytes(
    destination: &Path,
    bytes: &[u8],
    prefix: &str,
    mode: PublishMode,
) -> io::Result<()> {
    let (publication, mut file) = StagedPublication::create(destination, prefix)?;
    file.write_all(bytes)?;
    publication.commit(file, mode)
}

/// Flush one directory using the same platform law as a published file's
/// parent-directory barrier. Directory-generation publishers use this after
/// every staged child is complete and before the generation rename.
pub(crate) fn sync_directory(directory: &Path) -> io::Result<()> {
    sync_parent(&directory.join(".durable-directory-sync"))
}

/// Atomically make one complete staged directory visible without replacing an
/// external winner. This is the multi-file sibling of `PublishMode::NoReplace`:
/// callers fully populate and sync the directory before invoking it.
pub(crate) fn publish_directory_noreplace(staging: &Path, destination: &Path) -> io::Result<()> {
    let metadata = fs::symlink_metadata(staging)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "staged publication is not a regular directory",
        ));
    }
    rename_noreplace(staging, destination)?;
    sync_parent(destination)
}

/// Best-effort, bounded cleanup of files left by a process that died before a
/// staging/reservation guard could run. Only fixed hidden prefixes are ever
/// removed, and a directory with more entries is deliberately left partly
/// scanned for a later startup rather than monopolizing the server runtime.
pub(crate) fn cleanup_orphans(
    directory: &Path,
    prefixes: &[&str],
    max_entries: usize,
) -> io::Result<usize> {
    let mut removed = 0;
    for entry in fs::read_dir(directory)?.take(max_entries) {
        let entry = entry?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if !prefixes.iter().any(|prefix| name.starts_with(prefix)) {
            continue;
        }
        let metadata = entry.metadata()?;
        if metadata.is_file() {
            match fs::remove_file(entry.path()) {
                Ok(()) => removed += 1,
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
        }
    }
    Ok(removed)
}

/// Free bytes available to this process on the destination volume.
#[cfg(windows)]
pub(crate) fn available_space(path: &Path) -> io::Result<u64> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;

    let wide = path
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    let mut available = 0_u64;
    let result = unsafe {
        GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &mut available,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    if result == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(available)
    }
}

#[cfg(unix)]
fn statvfs_field_u64<T: Into<u64>>(value: T) -> u64 {
    value.into()
}

#[cfg(unix)]
pub(crate) fn available_space(path: &Path) -> io::Result<u64> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "path contains NUL"))?;
    let mut stats = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    if unsafe { libc::statvfs(path.as_ptr(), stats.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    let stats = unsafe { stats.assume_init() };
    Ok(statvfs_field_u64(stats.f_bavail).saturating_mul(statvfs_field_u64(stats.f_frsize)))
}

#[cfg(not(any(windows, unix)))]
pub(crate) fn available_space(_path: &Path) -> io::Result<u64> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "free-space inspection is unavailable on this platform",
    ))
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
    if unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    } == 0
    {
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
    use windows_sys::Win32::Storage::FileSystem::{MoveFileExW, MOVEFILE_WRITE_THROUGH};

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
    // Omitting MOVEFILE_REPLACE_EXISTING is the atomic no-overwrite law.
    if unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_WRITE_THROUGH,
        )
    } == 0
    {
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
    let result = unsafe {
        libc::syscall(
            libc::SYS_renameat2,
            libc::AT_FDCWD,
            source.as_ptr(),
            libc::AT_FDCWD,
            destination.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    if result == 0 {
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
    if unsafe { libc::renamex_np(source.as_ptr(), destination.as_ptr(), libc::RENAME_EXCL) } == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(all(
    unix,
    not(any(target_os = "linux", target_os = "android", target_vendor = "apple"))
))]
fn rename_noreplace(source: &Path, destination: &Path) -> io::Result<()> {
    // A same-filesystem hard link is an atomic create-if-absent operation.
    fs::hard_link(source, destination)?;
    fs::remove_file(source)
}

#[cfg(not(any(windows, unix)))]
fn rename_noreplace(source: &Path, destination: &Path) -> io::Result<()> {
    fs::hard_link(source, destination)?;
    fs::remove_file(source)
}

#[cfg(unix)]
fn sync_parent(destination: &Path) -> io::Result<()> {
    let parent = destination
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    File::open(parent)?.sync_all()
}

#[cfg(windows)]
fn sync_parent(destination: &Path) -> io::Result<()> {
    use std::os::windows::fs::OpenOptionsExt;
    use windows_sys::Win32::Storage::FileSystem::FILE_FLAG_BACKUP_SEMANTICS;

    let parent = destination
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    // MoveFileExW carries WRITE_THROUGH above; also flush a directory handle
    // where the filesystem supports it. Windows commonly rejects
    // FlushFileBuffers on directory handles with ACCESS_DENIED (5) or
    // INVALID_HANDLE (6); WRITE_THROUGH is the documented durability barrier
    // on those volumes.
    let directory = match OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
        .open(parent)
    {
        Ok(directory) => directory,
        Err(error) if matches!(error.raw_os_error(), Some(5 | 6)) => return Ok(()),
        Err(error) => return Err(error),
    };
    match directory.sync_all() {
        Ok(()) => Ok(()),
        Err(error) if matches!(error.raw_os_error(), Some(5 | 6)) => Ok(()),
        Err(error) => Err(error),
    }
}

#[cfg(not(any(windows, unix)))]
fn sync_parent(_destination: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(label: &str) -> Self {
            let ordinal = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "collide-o-scope-publication-{label}-{}-{ordinal}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn assert_no_staging(directory: &Path) {
        assert!(fs::read_dir(directory).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains("test-stage")
        }));
    }

    #[test]
    fn injected_prepublication_fault_retains_acknowledged_destination() {
        let directory = TempDir::new("fault");
        let destination = directory.0.join("patch.yaml");
        fs::write(&destination, b"acknowledged").unwrap();
        let (publication, mut file) =
            StagedPublication::create(&destination, "test-stage").unwrap();
        file.write_all(b"unacknowledged replacement").unwrap();
        let error = publication
            .commit_if(file, PublishMode::Replace, || {
                Err(io::Error::other("injected before atomic publication"))
            })
            .unwrap_err();
        assert!(error.to_string().contains("injected"));
        assert_eq!(fs::read(&destination).unwrap(), b"acknowledged");
        assert_no_staging(&directory.0);
    }

    #[test]
    fn no_replace_is_atomic_against_a_final_name_race() {
        let directory = TempDir::new("race");
        let destination = directory.0.join("clip.png");
        let (publication, mut file) =
            StagedPublication::create(&destination, "test-stage").unwrap();
        file.write_all(b"upload").unwrap();
        fs::write(&destination, b"external winner").unwrap();
        let error = publication
            .commit(file, PublishMode::NoReplace)
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(fs::read(&destination).unwrap(), b"external winner");
        assert_no_staging(&directory.0);
    }

    proptest! {
        #[test]
        fn replacement_is_whole_and_never_a_prefix(
            prior in proptest::collection::vec(any::<u8>(), 0..4096),
            next in proptest::collection::vec(any::<u8>(), 0..4096),
        ) {
            let directory = TempDir::new("property");
            let destination = directory.0.join("document.bin");
            fs::write(&destination, &prior).unwrap();
            publish_bytes(&destination, &next, "test-stage", PublishMode::Replace).unwrap();
            prop_assert_eq!(fs::read(&destination).unwrap(), next);
            assert_no_staging(&directory.0);
        }
    }

    #[test]
    fn orphan_cleanup_is_prefix_scoped_and_entry_bounded() {
        let directory = TempDir::new("cleanup");
        fs::write(directory.0.join(".upload-stage-dead.part"), b"partial").unwrap();
        fs::write(directory.0.join(".upload-reserve-dead"), b"").unwrap();
        fs::write(directory.0.join("keep.png"), b"operator media").unwrap();
        assert_eq!(
            cleanup_orphans(&directory.0, &[".upload-stage-", ".upload-reserve-"], 16,).unwrap(),
            2
        );
        assert_eq!(
            fs::read(directory.0.join("keep.png")).unwrap(),
            b"operator media"
        );
    }

    #[test]
    fn upload_admission_bounds_concurrency_aggregate_and_disk_headroom() {
        let admission = UploadAdmission::default();
        let limits = AdmissionLimits {
            max_concurrent: 2,
            max_reserved_bytes: 100,
            min_free_after_reservations: 25,
        };
        let first = admission.try_reserve(40, 1_000, limits).unwrap();
        let second = admission.try_reserve(50, 1_000, limits).unwrap();
        assert_eq!(admission.snapshot(), (2, 90));
        assert_eq!(
            admission.try_reserve(1, 1_000, limits).unwrap_err(),
            AdmissionError::Concurrency
        );
        drop(second);
        assert_eq!(
            admission.try_reserve(61, 1_000, limits).unwrap_err(),
            AdmissionError::AggregateBytes
        );
        assert_eq!(
            admission.try_reserve(50, 100, limits).unwrap_err(),
            AdmissionError::DiskHeadroom
        );
        drop(first);
        assert_eq!(admission.snapshot(), (0, 0));

        let cleanup = admission.begin_cleanup(7).unwrap().unwrap();
        assert_eq!(
            admission.try_reserve(1, 1_000, limits).unwrap_err(),
            AdmissionError::CleanupBusy
        );
        cleanup.complete();
        assert!(admission.begin_cleanup(7).unwrap().is_none());
        assert!(admission.begin_cleanup(8).unwrap().is_some());
    }
}
