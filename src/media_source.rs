//! Shared file-source resolution and bounded content fingerprinting.
//!
//! Ordinary patches retain their historical path-first lookup. Procedurally
//! generated patches may instead carry a path-independent `cos-sha256://`
//! reference. Those references are resolved only after the candidate bytes
//! match the embedded SHA-256 identity.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const CONTENT_SHA256_PREFIX: &str = "cos-sha256://";
pub const FINGERPRINT_BUFFER_BYTES: usize = 1024 * 1024;
pub const DEFAULT_MAX_SEARCH_ENTRIES: usize = 4096;
pub const DEFAULT_MAX_FINGERPRINT_BYTES: u64 = 64 * 1024 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentIdentity {
    pub sha256: String,
    pub byte_len: u64,
}

impl ContentIdentity {
    pub fn new(sha256: impl Into<String>, byte_len: u64) -> Result<Self, SourceResolveError> {
        let sha256 = sha256.into().to_ascii_lowercase();
        if !valid_sha256_hex(&sha256) {
            return Err(SourceResolveError::InvalidContentReference(
                "SHA-256 must contain exactly 64 hexadecimal characters".into(),
            ));
        }
        Ok(Self { sha256, byte_len })
    }

    pub fn source_reference(&self) -> String {
        format!("{CONTENT_SHA256_PREFIX}{}/{}", self.sha256, self.byte_len)
    }
}

fn valid_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

pub fn parse_content_reference(value: &str) -> Result<Option<ContentIdentity>, SourceResolveError> {
    let Some(encoded) = value.strip_prefix(CONTENT_SHA256_PREFIX) else {
        return Ok(None);
    };
    let (sha256, byte_len) = encoded.split_once('/').ok_or_else(|| {
        SourceResolveError::InvalidContentReference(format!(
            "content reference must be {CONTENT_SHA256_PREFIX}<sha256>/<bytes>"
        ))
    })?;
    if byte_len.contains('/') {
        return Err(SourceResolveError::InvalidContentReference(
            "content reference contains unexpected trailing components".into(),
        ));
    }
    let byte_len = byte_len.parse::<u64>().map_err(|_| {
        SourceResolveError::InvalidContentReference(
            "content reference byte length is not an unsigned integer".into(),
        )
    })?;
    ContentIdentity::new(sha256, byte_len).map(Some)
}

#[derive(Clone, Debug, Default)]
pub struct ResolveContext {
    pub patch_dir: Option<PathBuf>,
    pub library_dir: Option<PathBuf>,
}

impl ResolveContext {
    pub fn new(patch_dir: Option<PathBuf>, library_dir: Option<PathBuf>) -> Self {
        Self {
            patch_dir,
            library_dir,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FingerprintLimits {
    pub max_search_entries: usize,
    pub max_total_bytes: u64,
}

impl Default for FingerprintLimits {
    fn default() -> Self {
        Self {
            max_search_entries: DEFAULT_MAX_SEARCH_ENTRIES,
            max_total_bytes: DEFAULT_MAX_FINGERPRINT_BYTES,
        }
    }
}

#[derive(Clone, Debug)]
struct CachedFingerprint {
    identity: ContentIdentity,
    modified: Option<SystemTime>,
}

/// One invocation's bounded, cancellation-aware fingerprint state.
///
/// Repeated layers referencing the same canonical file reuse the first digest.
/// The byte budget counts physical file reads, not references.
pub struct FingerprintSession {
    limits: FingerprintLimits,
    cancel: Option<Arc<AtomicBool>>,
    cache: HashMap<PathBuf, CachedFingerprint>,
    bytes_hashed: u64,
    files_hashed: usize,
    search_entries_examined: usize,
}

impl FingerprintSession {
    pub fn new(limits: FingerprintLimits) -> Result<Self, SourceResolveError> {
        Self::with_cancel(limits, None)
    }

    pub fn with_cancel(
        limits: FingerprintLimits,
        cancel: Option<Arc<AtomicBool>>,
    ) -> Result<Self, SourceResolveError> {
        if limits.max_search_entries == 0 {
            return Err(SourceResolveError::InvalidLimit(
                "maximum search entries must be greater than zero".into(),
            ));
        }
        if limits.max_total_bytes == 0 {
            return Err(SourceResolveError::InvalidLimit(
                "maximum fingerprint bytes must be greater than zero".into(),
            ));
        }
        Ok(Self {
            limits,
            cancel,
            cache: HashMap::new(),
            bytes_hashed: 0,
            files_hashed: 0,
            search_entries_examined: 0,
        })
    }

    #[cfg(test)]
    pub fn bytes_hashed(&self) -> u64 {
        self.bytes_hashed
    }

    #[cfg(test)]
    pub fn files_hashed(&self) -> usize {
        self.files_hashed
    }

    fn check_cancelled(&self) -> Result<(), SourceResolveError> {
        if self
            .cancel
            .as_ref()
            .is_some_and(|cancel| cancel.load(Ordering::Acquire))
        {
            Err(SourceResolveError::Cancelled)
        } else {
            Ok(())
        }
    }

    fn note_search_entry(&mut self) -> Result<(), SourceResolveError> {
        self.search_entries_examined = self
            .search_entries_examined
            .checked_add(1)
            .ok_or_else(|| SourceResolveError::SearchBudget("entry count overflows".into()))?;
        if self.search_entries_examined > self.limits.max_search_entries {
            return Err(SourceResolveError::SearchBudget(format!(
                "source search exceeds the {}-entry limit",
                self.limits.max_search_entries
            )));
        }
        Ok(())
    }

    pub fn fingerprint(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<ContentIdentity, SourceResolveError> {
        self.check_cancelled()?;
        let path = path.as_ref();
        let canonical = std::fs::canonicalize(path).map_err(|error| {
            SourceResolveError::Io(format!("cannot canonicalize {}: {error}", path.display()))
        })?;
        let before = std::fs::metadata(&canonical).map_err(|error| {
            SourceResolveError::Io(format!("cannot inspect {}: {error}", canonical.display()))
        })?;
        if !before.is_file() {
            return Err(SourceResolveError::Io(format!(
                "source is not a regular file: {}",
                canonical.display()
            )));
        }
        let modified = before.modified().ok();
        if let Some(cached) = self.cache.get(&canonical) {
            if cached.identity.byte_len == before.len() && cached.modified == modified {
                return Ok(cached.identity.clone());
            }
        }

        let next_total = self
            .bytes_hashed
            .checked_add(before.len())
            .ok_or_else(|| SourceResolveError::FingerprintBudget("byte count overflows".into()))?;
        if next_total > self.limits.max_total_bytes {
            return Err(SourceResolveError::FingerprintBudget(format!(
                "fingerprinting {} bytes would exceed the {}-byte invocation limit",
                before.len(),
                self.limits.max_total_bytes
            )));
        }
        // Reserve the complete metadata length before I/O. A failed read still
        // consumes this invocation's budget rather than permitting retries to
        // bypass the declared bound.
        self.bytes_hashed = next_total;
        self.files_hashed = self
            .files_hashed
            .checked_add(1)
            .ok_or_else(|| SourceResolveError::FingerprintBudget("file count overflows".into()))?;

        let mut file = File::open(&canonical).map_err(|error| {
            SourceResolveError::Io(format!("cannot open {}: {error}", canonical.display()))
        })?;
        let mut buffer = vec![0u8; FINGERPRINT_BUFFER_BYTES];
        let mut observed = 0u64;
        let mut hasher = Sha256::new();
        loop {
            self.check_cancelled()?;
            let count = file.read(&mut buffer).map_err(|error| {
                SourceResolveError::Io(format!("cannot read {}: {error}", canonical.display()))
            })?;
            if count == 0 {
                break;
            }
            observed = observed.checked_add(count as u64).ok_or_else(|| {
                SourceResolveError::FingerprintBudget("read count overflows".into())
            })?;
            if observed > before.len() {
                return Err(SourceResolveError::ChangedDuringFingerprint(
                    canonical.display().to_string(),
                ));
            }
            hasher.update(&buffer[..count]);
        }
        let after = std::fs::metadata(&canonical).map_err(|error| {
            SourceResolveError::Io(format!(
                "cannot re-inspect {} after fingerprinting: {error}",
                canonical.display()
            ))
        })?;
        if observed != before.len()
            || after.len() != before.len()
            || after.modified().ok() != modified
        {
            return Err(SourceResolveError::ChangedDuringFingerprint(
                canonical.display().to_string(),
            ));
        }

        let identity = ContentIdentity {
            sha256: format!("{:x}", hasher.finalize()),
            byte_len: observed,
        };
        self.cache.insert(
            canonical,
            CachedFingerprint {
                identity: identity.clone(),
                modified,
            },
        );
        Ok(identity)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedFile {
    pub path: PathBuf,
    pub identity: Option<ContentIdentity>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolvedVisualSource {
    File(ResolvedFile),
    Spout { sender: String },
}

#[derive(Debug, PartialEq, Eq)]
pub enum SourceResolveError {
    InvalidContentReference(String),
    InvalidLimit(String),
    Missing(String),
    ContentMismatch(String),
    FingerprintBudget(String),
    SearchBudget(String),
    ChangedDuringFingerprint(String),
    Cancelled,
    Io(String),
}

impl fmt::Display for SourceResolveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidContentReference(detail) => {
                write!(formatter, "invalid content source reference: {detail}")
            }
            Self::InvalidLimit(detail) => write!(formatter, "invalid source limit: {detail}"),
            Self::Missing(name) => write!(formatter, "source not found: {name}"),
            Self::ContentMismatch(name) => {
                write!(
                    formatter,
                    "no candidate matches the recorded content for {name}"
                )
            }
            Self::FingerprintBudget(detail) => {
                write!(formatter, "fingerprint byte budget exceeded: {detail}")
            }
            Self::SearchBudget(detail) => {
                write!(formatter, "source search budget exceeded: {detail}")
            }
            Self::ChangedDuringFingerprint(path) => {
                write!(
                    formatter,
                    "source changed while it was fingerprinted: {path}"
                )
            }
            Self::Cancelled => formatter.write_str("source fingerprinting cancelled"),
            Self::Io(detail) => formatter.write_str(detail),
        }
    }
}

impl std::error::Error for SourceResolveError {}

pub fn resolve_visual_source<F>(
    source_path: &str,
    logical_name: &str,
    context: &ResolveContext,
    expected: Option<&ContentIdentity>,
    accepts: F,
    fingerprints: &mut FingerprintSession,
) -> Result<ResolvedVisualSource, SourceResolveError>
where
    F: Fn(&Path) -> bool,
{
    if let Some(sender) = crate::layers::spout_sender_from_source_path(source_path) {
        return Ok(ResolvedVisualSource::Spout {
            sender: sender.to_string(),
        });
    }
    resolve_file_source(
        source_path,
        logical_name,
        context,
        expected,
        accepts,
        fingerprints,
    )
    .map(ResolvedVisualSource::File)
}

pub fn resolve_file_source<F>(
    source_path: &str,
    logical_name: &str,
    context: &ResolveContext,
    explicit_expected: Option<&ContentIdentity>,
    accepts: F,
    fingerprints: &mut FingerprintSession,
) -> Result<ResolvedFile, SourceResolveError>
where
    F: Fn(&Path) -> bool,
{
    let embedded_expected = parse_content_reference(source_path)?;
    if let (Some(embedded), Some(explicit)) = (&embedded_expected, explicit_expected) {
        if embedded != explicit {
            return Err(SourceResolveError::InvalidContentReference(
                "embedded and externally supplied content identities disagree".into(),
            ));
        }
    }
    let expected = explicit_expected.or(embedded_expected.as_ref());
    let is_content_reference = embedded_expected.is_some();

    let logical_path = PathBuf::from(logical_name);
    let mut raw_candidates = Vec::new();
    if !source_path.is_empty() && !is_content_reference {
        raw_candidates.push(PathBuf::from(source_path));
    }
    if logical_path.is_absolute() {
        raw_candidates.push(logical_path.clone());
    } else if !logical_name.is_empty() {
        if let Some(patch_dir) = &context.patch_dir {
            raw_candidates.push(patch_dir.join(&logical_path));
        }
        if let Some(library_dir) = &context.library_dir {
            raw_candidates.push(library_dir.join(&logical_path));
        }
        raw_candidates.push(logical_path);
    }

    let mut seen = HashSet::new();
    for candidate in raw_candidates {
        let Some(canonical) = canonical_accepted_file(&candidate, &accepts) else {
            continue;
        };
        if !seen.insert(canonical.clone()) {
            continue;
        }
        if let Some(expected) = expected {
            if std::fs::metadata(&canonical)
                .ok()
                .is_none_or(|metadata| metadata.len() != expected.byte_len)
            {
                continue;
            }
            let observed = fingerprints.fingerprint(&canonical)?;
            if observed == *expected {
                return Ok(ResolvedFile {
                    path: canonical,
                    identity: Some(observed),
                });
            }
        } else {
            return Ok(ResolvedFile {
                path: canonical,
                identity: None,
            });
        }
    }

    if let Some(expected) = expected {
        let mut roots = Vec::new();
        if let Some(root) = &context.patch_dir {
            roots.push(root.clone());
        }
        if let Some(root) = &context.library_dir {
            roots.push(root.clone());
        }
        roots.sort();
        roots.dedup();

        let mut fallback_candidates = Vec::new();
        for root in roots {
            let Ok(entries) = std::fs::read_dir(&root) else {
                continue;
            };
            for entry in entries {
                fingerprints.note_search_entry()?;
                let Ok(entry) = entry else {
                    continue;
                };
                let Ok(file_type) = entry.file_type() else {
                    continue;
                };
                // Do not recurse or follow symlinks during content search.
                if !file_type.is_file() {
                    continue;
                }
                let path = entry.path();
                if !accepts(&path)
                    || std::fs::metadata(&path)
                        .ok()
                        .is_none_or(|metadata| metadata.len() != expected.byte_len)
                {
                    continue;
                }
                if let Some(canonical) = canonical_accepted_file(&path, &accepts) {
                    if seen.insert(canonical.clone()) {
                        fallback_candidates.push(canonical);
                    }
                }
            }
        }
        fallback_candidates.sort();
        for candidate in fallback_candidates {
            let observed = fingerprints.fingerprint(&candidate)?;
            if observed == *expected {
                return Ok(ResolvedFile {
                    path: candidate,
                    identity: Some(observed),
                });
            }
        }
        return Err(SourceResolveError::ContentMismatch(display_name(
            logical_name,
            expected,
        )));
    }

    Err(SourceResolveError::Missing(if logical_name.is_empty() {
        source_path.to_string()
    } else {
        logical_name.to_string()
    }))
}

fn canonical_accepted_file<F>(path: &Path, accepts: &F) -> Option<PathBuf>
where
    F: Fn(&Path) -> bool,
{
    if !path.is_file() || !accepts(path) {
        return None;
    }
    std::fs::canonicalize(path).ok()
}

fn display_name(logical_name: &str, expected: &ContentIdentity) -> String {
    if logical_name.is_empty() {
        format!("sha256:{}", &expected.sha256[..12])
    } else {
        Path::new(logical_name)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| logical_name.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{Duration, UNIX_EPOCH};

    fn temp_root(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "collide-o-scope-media-source-{label}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_nanos()
        ))
    }

    fn accepts_bin(path: &Path) -> bool {
        path.extension().and_then(|ext| ext.to_str()) == Some("bin")
    }

    #[test]
    fn sha256_stream_matches_the_standard_golden_vector_and_reuses_cache() {
        let root = temp_root("golden");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("abc.bin");
        fs::write(&path, b"abc").unwrap();
        let mut session = FingerprintSession::new(FingerprintLimits::default()).unwrap();
        let first = session.fingerprint(&path).unwrap();
        let second = session.fingerprint(&path).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            first.sha256,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(first.byte_len, 3);
        assert_eq!(session.files_hashed(), 1);
        assert_eq!(session.bytes_hashed(), 3);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn legacy_resolution_preserves_patch_before_library_precedence() {
        let root = temp_root("precedence");
        let patch = root.join("patch");
        let library = root.join("library");
        fs::create_dir_all(&patch).unwrap();
        fs::create_dir_all(&library).unwrap();
        fs::write(patch.join("clip.bin"), b"patch").unwrap();
        fs::write(library.join("clip.bin"), b"library").unwrap();
        let context = ResolveContext::new(Some(patch.clone()), Some(library));
        let mut fingerprints = FingerprintSession::new(FingerprintLimits::default()).unwrap();
        let resolved = resolve_file_source(
            "",
            "clip.bin",
            &context,
            None,
            accepts_bin,
            &mut fingerprints,
        )
        .unwrap();
        assert_eq!(
            resolved.path,
            fs::canonicalize(patch.join("clip.bin")).unwrap()
        );
        assert!(resolved.identity.is_none());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn expected_digest_rejects_wrong_saved_path_and_finds_renamed_content() {
        let root = temp_root("digest-fallback");
        let library = root.join("library");
        fs::create_dir_all(&library).unwrap();
        let wrong = root.join("saved.bin");
        let renamed = library.join("renamed.bin");
        fs::write(&wrong, b"wrong").unwrap();
        fs::write(&renamed, b"right-content").unwrap();
        let mut source_fingerprints =
            FingerprintSession::new(FingerprintLimits::default()).unwrap();
        let expected = source_fingerprints.fingerprint(&renamed).unwrap();
        let context = ResolveContext::new(None, Some(library));
        let mut resolver_fingerprints =
            FingerprintSession::new(FingerprintLimits::default()).unwrap();
        let resolved = resolve_file_source(
            &wrong.to_string_lossy(),
            "missing.bin",
            &context,
            Some(&expected),
            accepts_bin,
            &mut resolver_fingerprints,
        )
        .unwrap();
        assert_eq!(resolved.path, fs::canonicalize(renamed).unwrap());
        assert_eq!(resolved.identity, Some(expected));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fingerprint_and_search_limits_fail_before_unbounded_work() {
        let root = temp_root("limits");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("large.bin");
        fs::write(&path, b"three").unwrap();
        let mut fingerprints = FingerprintSession::new(FingerprintLimits {
            max_search_entries: 1,
            max_total_bytes: 4,
        })
        .unwrap();
        assert!(matches!(
            fingerprints.fingerprint(&path),
            Err(SourceResolveError::FingerprintBudget(_))
        ));

        let expected = ContentIdentity::new(
            "0000000000000000000000000000000000000000000000000000000000000000",
            5,
        )
        .unwrap();
        fs::write(root.join("other.bin"), b"other").unwrap();
        let context = ResolveContext::new(None, Some(root.clone()));
        let mut search = FingerprintSession::new(FingerprintLimits {
            max_search_entries: 1,
            max_total_bytes: 100,
        })
        .unwrap();
        assert!(matches!(
            resolve_file_source(
                "",
                "missing.bin",
                &context,
                Some(&expected),
                accepts_bin,
                &mut search,
            ),
            Err(SourceResolveError::SearchBudget(_))
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cancellation_stops_fingerprinting_before_file_io_is_accounted() {
        let root = temp_root("cancelled");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("source.bin");
        fs::write(&path, b"source bytes").unwrap();
        let cancel = Arc::new(AtomicBool::new(true));
        let mut fingerprints =
            FingerprintSession::with_cancel(FingerprintLimits::default(), Some(cancel)).unwrap();

        assert_eq!(
            fingerprints.fingerprint(&path),
            Err(SourceResolveError::Cancelled)
        );
        assert_eq!(fingerprints.files_hashed(), 0);
        assert_eq!(fingerprints.bytes_hashed(), 0);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn content_reference_round_trips_and_spout_never_touches_the_filesystem() {
        let identity = ContentIdentity::new(
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            42,
        )
        .unwrap();
        assert_eq!(
            parse_content_reference(&identity.source_reference()).unwrap(),
            Some(identity)
        );
        let context = ResolveContext::default();
        let mut fingerprints = FingerprintSession::new(FingerprintLimits::default()).unwrap();
        assert_eq!(
            resolve_visual_source(
                "spout://Camera",
                "Camera",
                &context,
                None,
                |_| false,
                &mut fingerprints,
            )
            .unwrap(),
            ResolvedVisualSource::Spout {
                sender: "Camera".into()
            }
        );
        assert_eq!(fingerprints.files_hashed(), 0);
    }
}
