//! Cancelable, single-flight and bounded media-library indexing.
//!
//! Directory enumeration and classification live exclusively on one worker.
//! Main receives a generation-tagged completed index through a newest-only
//! slot and performs no directory or metadata I/O. Web publication exposes a
//! small current page; other bounded pages/searches use the authenticated
//! `/library-index` endpoint.

use std::cmp::Ordering as CmpOrdering;
use std::collections::BinaryHeap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use serde::{Deserialize, Serialize};

pub const LIBRARY_INDEX_MAX_ENTRIES: usize = 100_000;
pub const LIBRARY_INDEX_DEFAULT_PAGE_SIZE: usize = 64;
pub const LIBRARY_INDEX_MAX_PAGE_SIZE: usize = 128;
pub const LIBRARY_INDEX_MAX_QUERY_BYTES: usize = 128;
pub const LIBRARY_INDEX_MAX_FILENAME_BYTES: usize = 1_024;
pub const LIBRARY_INDEX_MAX_ERROR_BYTES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LibraryEntryKind {
    Visual,
    Audio,
}

impl LibraryEntryKind {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "visual" => Some(Self::Visual),
            "audio" => Some(Self::Audio),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LibraryScanStatus {
    #[default]
    Empty,
    Scanning,
    Ready,
    Truncated,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LibraryPageEntry {
    pub filename: String,
    pub kind: LibraryEntryKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LibraryPageSnapshot {
    pub revision: u64,
    pub generation: u64,
    pub kind: Option<LibraryEntryKind>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub query: String,
    pub offset: u32,
    pub limit: u16,
    pub total_matches: u64,
    pub index_truncated: bool,
    pub has_more: bool,
    pub entries: Vec<LibraryPageEntry>,
}

impl Default for LibraryPageSnapshot {
    fn default() -> Self {
        Self {
            revision: 0,
            generation: 0,
            kind: Some(LibraryEntryKind::Visual),
            query: String::new(),
            offset: 0,
            limit: LIBRARY_INDEX_DEFAULT_PAGE_SIZE as u16,
            total_matches: 0,
            index_truncated: false,
            has_more: false,
            entries: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LibraryIndexSnapshot {
    pub revision: u64,
    pub generation: u64,
    pub status: LibraryScanStatus,
    pub discovered_entries: u64,
    pub indexed_entries: u32,
    pub visual_entries: u64,
    pub audio_entries: u64,
    pub skipped_entries: u64,
    pub truncated: bool,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub error: String,
    pub current_page: LibraryPageSnapshot,
}

impl Default for LibraryIndexSnapshot {
    fn default() -> Self {
        Self {
            revision: 0,
            generation: 0,
            status: LibraryScanStatus::Empty,
            discovered_entries: 0,
            indexed_entries: 0,
            visual_entries: 0,
            audio_entries: 0,
            skipped_entries: 0,
            truncated: false,
            error: String::new(),
            current_page: LibraryPageSnapshot::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LibraryEntry {
    filename: String,
    search_key: String,
    kind: LibraryEntryKind,
}

#[derive(Debug, Clone)]
pub struct LibraryIndex {
    revision: u64,
    generation: u64,
    status: LibraryScanStatus,
    discovered_entries: u64,
    visual_entries: u64,
    audio_entries: u64,
    skipped_entries: u64,
    truncated: bool,
    error: String,
    entries: Vec<LibraryEntry>,
}

impl LibraryIndex {
    pub fn empty() -> Self {
        Self {
            revision: 0,
            generation: 0,
            status: LibraryScanStatus::Empty,
            discovered_entries: 0,
            visual_entries: 0,
            audio_entries: 0,
            skipped_entries: 0,
            truncated: false,
            error: String::new(),
            entries: Vec::new(),
        }
    }

    pub fn scanning(generation: u64) -> Self {
        Self {
            generation,
            status: LibraryScanStatus::Scanning,
            ..Self::empty()
        }
    }

    pub fn error(generation: u64, error: &str) -> Self {
        Self {
            generation,
            status: LibraryScanStatus::Error,
            error: bounded_error(error),
            ..Self::empty()
        }
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn status(&self) -> LibraryScanStatus {
        self.status
    }

    pub fn set_revision(&mut self, revision: u64) {
        self.revision = revision;
    }

    pub fn snapshot(&self) -> LibraryIndexSnapshot {
        let current_page = self
            .page(LibraryPageRequest::current_visual())
            .expect("the built-in current-page request is valid");
        LibraryIndexSnapshot {
            revision: self.revision,
            generation: self.generation,
            status: self.status,
            discovered_entries: self.discovered_entries,
            indexed_entries: u32::try_from(self.entries.len()).unwrap_or(u32::MAX),
            visual_entries: self.visual_entries,
            audio_entries: self.audio_entries,
            skipped_entries: self.skipped_entries,
            truncated: self.truncated,
            error: self.error.clone(),
            current_page,
        }
    }

    pub fn page(
        &self,
        request: LibraryPageRequest,
    ) -> Result<LibraryPageSnapshot, LibraryPageError> {
        request.validate(self.revision)?;
        let query_key = request.query.to_lowercase();
        let mut total_matches = 0_u64;
        let mut page = Vec::with_capacity(request.limit);
        for entry in &self.entries {
            if request.kind.is_some_and(|kind| kind != entry.kind)
                || (!query_key.is_empty() && !entry.search_key.contains(&query_key))
            {
                continue;
            }
            let match_index = total_matches;
            total_matches = total_matches.saturating_add(1);
            if match_index < request.offset as u64 || page.len() == request.limit {
                continue;
            }
            page.push(LibraryPageEntry {
                filename: entry.filename.clone(),
                kind: entry.kind,
            });
        }
        let returned_end = u64::from(request.offset).saturating_add(page.len() as u64);
        Ok(LibraryPageSnapshot {
            revision: self.revision,
            generation: self.generation,
            kind: request.kind,
            query: request.query,
            offset: request.offset,
            limit: u16::try_from(request.limit).unwrap_or(u16::MAX),
            total_matches,
            index_truncated: self.truncated,
            has_more: returned_end < total_matches,
            entries: page,
        })
    }
}

impl Default for LibraryIndex {
    fn default() -> Self {
        Self::empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LibraryPageRequest {
    pub revision: Option<u64>,
    pub offset: u32,
    pub limit: usize,
    pub query: String,
    pub kind: Option<LibraryEntryKind>,
}

impl LibraryPageRequest {
    pub fn new(
        revision: Option<u64>,
        offset: u32,
        limit: usize,
        query: Option<&str>,
        kind: Option<LibraryEntryKind>,
    ) -> Result<Self, LibraryPageError> {
        let query = query.unwrap_or_default().trim().to_owned();
        let request = Self {
            revision,
            offset,
            limit,
            query,
            kind,
        };
        request.validate_without_revision()?;
        Ok(request)
    }

    pub fn current_visual() -> Self {
        Self::current(LibraryEntryKind::Visual)
    }

    pub fn current(kind: LibraryEntryKind) -> Self {
        Self {
            revision: None,
            offset: 0,
            limit: LIBRARY_INDEX_DEFAULT_PAGE_SIZE,
            query: String::new(),
            kind: Some(kind),
        }
    }

    fn validate(&self, current_revision: u64) -> Result<(), LibraryPageError> {
        self.validate_without_revision()?;
        if let Some(revision) = self.revision {
            if revision != current_revision {
                return Err(LibraryPageError::RevisionMismatch {
                    requested: revision,
                    current: current_revision,
                });
            }
        }
        Ok(())
    }

    fn validate_without_revision(&self) -> Result<(), LibraryPageError> {
        if self.limit == 0 || self.limit > LIBRARY_INDEX_MAX_PAGE_SIZE {
            return Err(LibraryPageError::InvalidLimit);
        }
        if self.query.len() > LIBRARY_INDEX_MAX_QUERY_BYTES
            || self.query.chars().any(char::is_control)
        {
            return Err(LibraryPageError::InvalidQuery);
        }
        if self.offset as usize > LIBRARY_INDEX_MAX_ENTRIES {
            return Err(LibraryPageError::InvalidOffset);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LibraryPageError {
    InvalidLimit,
    InvalidQuery,
    InvalidOffset,
    RevisionMismatch { requested: u64, current: u64 },
}

impl std::fmt::Display for LibraryPageError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLimit => write!(
                formatter,
                "page limit must be between 1 and {LIBRARY_INDEX_MAX_PAGE_SIZE}"
            ),
            Self::InvalidQuery => write!(
                formatter,
                "search query must be at most {LIBRARY_INDEX_MAX_QUERY_BYTES} bytes and contain no controls"
            ),
            Self::InvalidOffset => write!(
                formatter,
                "page offset must be no greater than {LIBRARY_INDEX_MAX_ENTRIES}"
            ),
            Self::RevisionMismatch { requested, current } => write!(
                formatter,
                "library revision {requested} is stale; current revision is {current}"
            ),
        }
    }
}

#[derive(Debug)]
pub struct LibraryScanCompletion {
    pub generation: u64,
    pub index: LibraryIndex,
    pub visual_files: Vec<PathBuf>,
    pub audio_files: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Candidate {
    sort_key: String,
    filename: String,
    path: PathBuf,
    kind: LibraryEntryKind,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        (&self.sort_key, &self.filename, self.kind, &self.path).cmp(&(
            &other.sort_key,
            &other.filename,
            other.kind,
            &other.path,
        ))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

trait LibraryEnumerator: Send + Sync {
    fn enumerate(
        &self,
        folder: &Path,
        is_current: &dyn Fn() -> bool,
        visit: &mut dyn FnMut(PathBuf) -> bool,
    ) -> Result<(), &'static str>;
}

struct FilesystemEnumerator;

impl LibraryEnumerator for FilesystemEnumerator {
    fn enumerate(
        &self,
        folder: &Path,
        is_current: &dyn Fn() -> bool,
        visit: &mut dyn FnMut(PathBuf) -> bool,
    ) -> Result<(), &'static str> {
        let entries = std::fs::read_dir(folder)
            .map_err(|_| "library directory could not be opened (permission or I/O error)")?;
        for entry in entries {
            if !is_current() {
                break;
            }
            let entry = entry
                .map_err(|_| "library directory enumeration failed (permission or I/O error)")?;
            let file_type = entry
                .file_type()
                .map_err(|_| "library entry metadata failed (permission or I/O error)")?;
            if file_type.is_file() && !visit(entry.path()) {
                break;
            }
        }
        Ok(())
    }
}

fn scan_library(
    enumerator: &dyn LibraryEnumerator,
    folder: &Path,
    generation: u64,
    cap: usize,
    is_current: &dyn Fn() -> bool,
) -> Option<LibraryScanCompletion> {
    let mut heap = BinaryHeap::with_capacity(cap.saturating_add(1));
    let mut discovered_entries = 0_u64;
    let mut visual_entries = 0_u64;
    let mut audio_entries = 0_u64;
    let mut skipped_entries = 0_u64;
    let result = enumerator.enumerate(folder, is_current, &mut |path| {
        if !is_current() {
            return false;
        }
        if is_upload_reservation(&path) {
            skipped_entries = skipped_entries.saturating_add(1);
            return true;
        }
        let kind = if crate::is_supported_visual_file(&path) {
            Some(LibraryEntryKind::Visual)
        } else if crate::audio::is_supported_audio_file(&path) {
            Some(LibraryEntryKind::Audio)
        } else {
            None
        };
        let Some(kind) = kind else {
            skipped_entries = skipped_entries.saturating_add(1);
            return true;
        };
        let Some(filename) = path.file_name().and_then(|name| name.to_str()) else {
            skipped_entries = skipped_entries.saturating_add(1);
            return true;
        };
        if filename.len() > LIBRARY_INDEX_MAX_FILENAME_BYTES {
            skipped_entries = skipped_entries.saturating_add(1);
            return true;
        }
        discovered_entries = discovered_entries.saturating_add(1);
        match kind {
            LibraryEntryKind::Visual => visual_entries = visual_entries.saturating_add(1),
            LibraryEntryKind::Audio => audio_entries = audio_entries.saturating_add(1),
        }
        let candidate = Candidate {
            sort_key: filename.to_lowercase(),
            filename: filename.to_owned(),
            path,
            kind,
        };
        if heap.len() < cap {
            heap.push(candidate);
        } else if heap.peek().is_some_and(|largest| candidate < *largest) {
            let _ = heap.pop();
            heap.push(candidate);
        }
        true
    });
    if !is_current() {
        return None;
    }
    if let Err(error) = result {
        return Some(LibraryScanCompletion {
            generation,
            index: LibraryIndex {
                revision: 0,
                generation,
                status: LibraryScanStatus::Error,
                discovered_entries,
                visual_entries,
                audio_entries,
                skipped_entries,
                truncated: false,
                error: bounded_error(error),
                entries: Vec::new(),
            },
            visual_files: Vec::new(),
            audio_files: Vec::new(),
        });
    }

    let mut candidates = heap.into_vec();
    candidates.sort();
    let mut entries = Vec::with_capacity(candidates.len());
    let visual_capacity = candidates
        .iter()
        .filter(|entry| entry.kind == LibraryEntryKind::Visual)
        .count();
    let mut visual_files = Vec::with_capacity(visual_capacity);
    let mut audio_files = Vec::with_capacity(candidates.len().saturating_sub(visual_capacity));
    for candidate in candidates {
        match candidate.kind {
            LibraryEntryKind::Visual => visual_files.push(candidate.path),
            LibraryEntryKind::Audio => audio_files.push(candidate.path),
        }
        entries.push(LibraryEntry {
            filename: candidate.filename,
            search_key: candidate.sort_key,
            kind: candidate.kind,
        });
    }
    let truncated = discovered_entries > entries.len() as u64;
    Some(LibraryScanCompletion {
        generation,
        index: LibraryIndex {
            revision: 0,
            generation,
            status: if truncated {
                LibraryScanStatus::Truncated
            } else {
                LibraryScanStatus::Ready
            },
            discovered_entries,
            visual_entries,
            audio_entries,
            skipped_entries,
            truncated,
            error: String::new(),
            entries,
        },
        visual_files,
        audio_files,
    })
}

fn is_upload_reservation(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with(".upload-"))
}

fn bounded_error(error: &str) -> String {
    if error.len() <= LIBRARY_INDEX_MAX_ERROR_BYTES {
        return error.to_owned();
    }
    let mut boundary = LIBRARY_INDEX_MAX_ERROR_BYTES.saturating_sub(3);
    while boundary > 0 && !error.is_char_boundary(boundary) {
        boundary -= 1;
    }
    format!("{}...", &error[..boundary])
}

#[derive(Debug)]
struct ScanJob {
    generation: u64,
    folder: PathBuf,
}

#[derive(Default)]
struct ScanMailboxState {
    pending: Option<ScanJob>,
    stopped: bool,
}

struct ScanMailbox {
    state: Mutex<ScanMailboxState>,
    wake: Condvar,
    current_generation: AtomicU64,
    stopped: AtomicBool,
}

impl ScanMailbox {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(ScanMailboxState::default()),
            wake: Condvar::new(),
            current_generation: AtomicU64::new(0),
            stopped: AtomicBool::new(false),
        })
    }

    fn submit(&self, job: ScanJob) -> Result<(), &'static str> {
        if job.generation == 0 {
            return Err("library generation must be nonzero");
        }
        let mut state = lock_recover(&self.state);
        if state.stopped {
            return Err("library index worker is stopped");
        }
        self.current_generation
            .store(job.generation, Ordering::Release);
        state.pending = Some(job);
        drop(state);
        self.wake.notify_one();
        Ok(())
    }

    fn take(&self) -> Option<ScanJob> {
        let mut state = lock_recover(&self.state);
        loop {
            if state.stopped {
                return None;
            }
            if let Some(job) = state.pending.take() {
                return Some(job);
            }
            state = self
                .wake
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    fn is_current(&self, generation: u64) -> bool {
        generation != 0
            && self.current_generation.load(Ordering::Acquire) == generation
            && !self.stopped.load(Ordering::Acquire)
    }

    fn stop(&self) {
        let mut state = lock_recover(&self.state);
        state.stopped = true;
        self.stopped.store(true, Ordering::Release);
        state.pending = None;
        self.current_generation.store(0, Ordering::Release);
        drop(state);
        self.wake.notify_all();
    }
}

pub struct LibraryIndexRuntime {
    mailbox: Arc<ScanMailbox>,
    completion: Arc<Mutex<Option<LibraryScanCompletion>>>,
    worker: Option<JoinHandle<()>>,
}

impl LibraryIndexRuntime {
    pub fn new() -> Result<Self, String> {
        Self::with_enumerator(Arc::new(FilesystemEnumerator))
    }

    fn with_enumerator(enumerator: Arc<dyn LibraryEnumerator>) -> Result<Self, String> {
        let mailbox = ScanMailbox::new();
        let completion = Arc::new(Mutex::new(None));
        let worker_mailbox = mailbox.clone();
        let worker_completion = completion.clone();
        let worker = std::thread::Builder::new()
            .name("library-index".to_owned())
            .spawn(move || {
                while let Some(job) = worker_mailbox.take() {
                    let generation = job.generation;
                    let result = scan_library(
                        enumerator.as_ref(),
                        &job.folder,
                        generation,
                        LIBRARY_INDEX_MAX_ENTRIES,
                        &|| worker_mailbox.is_current(generation),
                    );
                    if worker_mailbox.is_current(generation) {
                        if let Some(result) = result {
                            *lock_recover(&worker_completion) = Some(result);
                        }
                    }
                }
            })
            .map_err(|error| format!("failed to spawn library index worker: {error}"))?;
        Ok(Self {
            mailbox,
            completion,
            worker: Some(worker),
        })
    }

    pub fn request(&self, generation: u64, folder: PathBuf) -> Result<(), String> {
        // Hold the completion slot through admission: a refused request leaves
        // the current completed scan intact, while an admitted request clears
        // the prior value before the newly-woken worker can publish its own.
        let mut completion = lock_recover(&self.completion);
        self.mailbox
            .submit(ScanJob { generation, folder })
            .map_err(str::to_owned)?;
        *completion = None;
        Ok(())
    }

    pub fn poll(&self) -> Option<LibraryScanCompletion> {
        lock_recover(&self.completion).take()
    }

    #[cfg(test)]
    pub(crate) fn shutdown_for_test(&mut self) {
        self.mailbox.stop();
        if let Some(worker) = self.worker.take() {
            worker.join().expect("library index test worker panicked");
        }
    }

    #[cfg(test)]
    pub(crate) fn seed_empty_completion_for_test(&self, generation: u64) {
        let mut index = LibraryIndex::empty();
        index.generation = generation;
        *lock_recover(&self.completion) = Some(LibraryScanCompletion {
            generation,
            index,
            visual_files: Vec::new(),
            audio_files: Vec::new(),
        });
    }
}

impl Drop for LibraryIndexRuntime {
    fn drop(&mut self) {
        self.mailbox.stop();
        let Some(worker) = self.worker.take() else {
            return;
        };
        // The event/render thread never joins directory I/O. One bounded
        // lifetime worker is transferred to a tiny retirement task.
        if std::thread::Builder::new()
            .name("library-index-retire".to_owned())
            .spawn(move || {
                let _ = worker.join();
            })
            .is_err()
        {
            log::warn!("library index retirement helper could not be spawned");
        }
    }
}

fn lock_recover<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::time::{Duration, Instant};

    struct GeneratedEnumerator {
        entries: usize,
    }

    impl LibraryEnumerator for GeneratedEnumerator {
        fn enumerate(
            &self,
            _folder: &Path,
            _is_current: &dyn Fn() -> bool,
            visit: &mut dyn FnMut(PathBuf) -> bool,
        ) -> Result<(), &'static str> {
            for index in (0..self.entries).rev() {
                let extension = if index % 2 == 0 { "mp4" } else { "wav" };
                if !visit(PathBuf::from(format!("entry-{index:06}.{extension}"))) {
                    break;
                }
            }
            Ok(())
        }
    }

    #[test]
    fn hundred_thousand_mixed_entries_are_bounded_sorted_and_page_deterministic() {
        let enumerator = GeneratedEnumerator { entries: 100_001 };
        let completed = scan_library(
            &enumerator,
            Path::new("abstract-fixture"),
            7,
            LIBRARY_INDEX_MAX_ENTRIES,
            &|| true,
        )
        .unwrap();
        assert_eq!(completed.index.discovered_entries, 100_001);
        assert_eq!(completed.index.entries.len(), LIBRARY_INDEX_MAX_ENTRIES);
        assert_eq!(completed.visual_files.len(), 50_000);
        assert_eq!(completed.audio_files.len(), 50_000);
        assert_eq!(completed.index.status, LibraryScanStatus::Truncated);
        assert!(completed.index.truncated);
        let publication = completed.index.snapshot();
        assert_eq!(publication.indexed_entries, 100_000);
        assert_eq!(
            publication.current_page.entries.len(),
            LIBRARY_INDEX_DEFAULT_PAGE_SIZE
        );
        assert!(
            serde_json::to_vec(&publication).unwrap().len() < 16 * 1024,
            "the frame snapshot must remain independent of total index size"
        );

        let request = LibraryPageRequest::new(
            None,
            64,
            32,
            Some("entry-000"),
            Some(LibraryEntryKind::Visual),
        )
        .unwrap();
        let first = completed.index.page(request.clone()).unwrap();
        let second = completed.index.page(request).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.entries.len(), 32);
        assert!(first
            .entries
            .windows(2)
            .all(|pair| pair[0].filename < pair[1].filename));
        assert!(first
            .entries
            .iter()
            .all(|entry| entry.kind == LibraryEntryKind::Visual));
    }

    struct GenerationEnumerator {
        first_started: Arc<AtomicBool>,
        first_cancelled_after: Arc<AtomicUsize>,
    }

    impl LibraryEnumerator for GenerationEnumerator {
        fn enumerate(
            &self,
            folder: &Path,
            _is_current: &dyn Fn() -> bool,
            visit: &mut dyn FnMut(PathBuf) -> bool,
        ) -> Result<(), &'static str> {
            let first = folder == Path::new("first");
            if first {
                self.first_started.store(true, Ordering::Release);
            }
            let count = if first { 50_000 } else { 3 };
            for index in 0..count {
                if !visit(folder.join(format!("clip-{index:06}.mp4"))) {
                    if first {
                        self.first_cancelled_after.store(index, Ordering::Release);
                    }
                    break;
                }
                if first && index % 32 == 0 {
                    std::thread::yield_now();
                }
            }
            Ok(())
        }
    }

    #[test]
    fn newer_generation_cancels_stale_scan_and_only_current_result_publishes() {
        let started = Arc::new(AtomicBool::new(false));
        let cancelled_after = Arc::new(AtomicUsize::new(usize::MAX));
        let mut runtime = LibraryIndexRuntime::with_enumerator(Arc::new(GenerationEnumerator {
            first_started: started.clone(),
            first_cancelled_after: cancelled_after.clone(),
        }))
        .unwrap();
        runtime.request(1, PathBuf::from("first")).unwrap();
        let start_deadline = Instant::now() + Duration::from_secs(1);
        while !started.load(Ordering::Acquire) {
            assert!(Instant::now() < start_deadline);
            std::thread::yield_now();
        }
        runtime.request(2, PathBuf::from("second")).unwrap();

        let deadline = Instant::now() + Duration::from_secs(2);
        let completed = loop {
            if let Some(completed) = runtime.poll() {
                break completed;
            }
            assert!(Instant::now() < deadline, "current scan did not publish");
            std::thread::yield_now();
        };
        assert_eq!(completed.generation, 2);
        assert_eq!(completed.visual_files.len(), 3);
        assert!(cancelled_after.load(Ordering::Acquire) < 50_000);
        runtime.shutdown_for_test();
    }

    struct ThreadRecordingEnumerator {
        observed: Arc<Mutex<Option<std::thread::ThreadId>>>,
    }

    impl LibraryEnumerator for ThreadRecordingEnumerator {
        fn enumerate(
            &self,
            _folder: &Path,
            _is_current: &dyn Fn() -> bool,
            visit: &mut dyn FnMut(PathBuf) -> bool,
        ) -> Result<(), &'static str> {
            *lock_recover(&self.observed) = Some(std::thread::current().id());
            let _ = visit(PathBuf::from("worker-only.mp4"));
            Ok(())
        }
    }

    #[test]
    fn directory_enumeration_never_runs_on_the_requesting_render_thread() {
        let caller = std::thread::current().id();
        let observed = Arc::new(Mutex::new(None));
        let mut runtime =
            LibraryIndexRuntime::with_enumerator(Arc::new(ThreadRecordingEnumerator {
                observed: observed.clone(),
            }))
            .unwrap();
        runtime
            .request(4, PathBuf::from("abstract-fixture"))
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        while runtime.poll().is_none() {
            assert!(Instant::now() < deadline, "worker did not publish");
            std::thread::yield_now();
        }
        assert_ne!(lock_recover(&observed).as_ref(), Some(&caller));
        runtime.shutdown_for_test();
    }

    #[test]
    fn stopped_mailbox_refusal_preserves_the_current_completed_scan() {
        let mut runtime =
            LibraryIndexRuntime::with_enumerator(Arc::new(GeneratedEnumerator { entries: 0 }))
                .unwrap();
        runtime.shutdown_for_test();
        runtime.seed_empty_completion_for_test(17);

        assert!(runtime
            .request(18, PathBuf::from("refused-after-stop"))
            .is_err());
        assert_eq!(
            runtime.poll().map(|completion| completion.generation),
            Some(17),
            "a refused admission must not erase the still-current completed scan",
        );
    }

    struct ErrorEnumerator;

    impl LibraryEnumerator for ErrorEnumerator {
        fn enumerate(
            &self,
            _folder: &Path,
            _is_current: &dyn Fn() -> bool,
            visit: &mut dyn FnMut(PathBuf) -> bool,
        ) -> Result<(), &'static str> {
            assert!(visit(PathBuf::from("partial.mp4")));
            assert!(visit(PathBuf::from("partial.wav")));
            assert!(visit(PathBuf::from("partial.txt")));
            Err("injected enumeration failure")
        }
    }

    #[test]
    fn errors_and_hostile_page_requests_are_truthful_and_bounded() {
        let completed = scan_library(
            &ErrorEnumerator,
            Path::new("failure"),
            9,
            LIBRARY_INDEX_MAX_ENTRIES,
            &|| true,
        )
        .unwrap();
        assert_eq!(completed.index.status, LibraryScanStatus::Error);
        assert_eq!(completed.index.error, "injected enumeration failure");
        assert_eq!(completed.index.discovered_entries, 2);
        assert_eq!(completed.index.visual_entries, 1);
        assert_eq!(completed.index.audio_entries, 1);
        assert_eq!(completed.index.skipped_entries, 1);
        assert!(completed.index.entries.is_empty());

        let bounded = LibraryIndex::error(9, &"x".repeat(1_000));
        assert!(bounded.error.len() <= LIBRARY_INDEX_MAX_ERROR_BYTES);
        assert!(bounded.error.ends_with("..."));

        let mut index = LibraryIndex::empty();
        index.set_revision(11);
        assert!(matches!(
            LibraryPageRequest::new(None, 0, LIBRARY_INDEX_MAX_PAGE_SIZE + 1, None, None),
            Err(LibraryPageError::InvalidLimit)
        ));
        assert!(matches!(
            LibraryPageRequest::new(None, 0, 1, Some(&"x".repeat(129)), None),
            Err(LibraryPageError::InvalidQuery)
        ));
        let stale = LibraryPageRequest::new(Some(10), 0, 1, None, None).unwrap();
        assert!(matches!(
            index.page(stale),
            Err(LibraryPageError::RevisionMismatch {
                requested: 10,
                current: 11
            })
        ));
    }
}
