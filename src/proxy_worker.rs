//! The bounded FFV1/Matroska proxy-cache worker.
//!
//! This module executes what `proxy.rs` decides. Every consumption question
//! is answered by [`crate::proxy::plan_proxy_input`] — the one contract
//! function — and every publication follows `ATOMIC_PROXY_CACHE_COMMIT_LAW`:
//! create-new staging in the cache directory, staging fsync, atomic replace,
//! parent-directory sync, with the prior artifact readable throughout. A
//! crash leaves at worst a staging file, which recovery removes and never
//! publishes or counts. The directory itself is the index: a scan rebuilds
//! it, so there is no metadata file to corrupt. Last-used ordinals are
//! session-local — a fresh process starts every entry at zero and the pure
//! preflight's `(ordinal, key)` order breaks ties deterministically.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use crate::image_routing::StableLayerId;
use crate::media_safety::{MediaDeviceLimits, MediaSafetyPolicy, MediaSourceKind};
use crate::media_source::{ContentIdentity, FingerprintLimits, FingerprintSession};
use crate::proxy::{
    plan_proxy_input, ProxyAudioInputLaw, ProxyCacheEntry, ProxyCacheKey, ProxyCacheLimits,
    ProxyCachePlan, ProxyError, ProxyFrameTimingLaw, ProxyInputPlan, ProxySettings,
    ProxySourceProbe,
};
use crate::video::{SeedSelectError, ThreadedDecoder};

/// Poll cadence for the encode child. The deadline is absolute and checked
/// first each iteration, so this only bounds reaction latency.
pub const PROXY_ENCODE_POLL_MS: u64 = 100;
/// Captured child stderr is bounded like every other media helper's.
pub const PROXY_ENCODE_STDERR_LIMIT: usize = 256 * 1024;
/// FFV1 encodes write nothing useful to stdout; keep the cap tight.
pub const PROXY_ENCODE_STDOUT_LIMIT: usize = 64 * 1024;
/// Validation decodes the first video frame; a valid artifact yields one
/// within a bounded packet count or it is refused.
pub const PROXY_VALIDATE_MAX_PACKETS: usize = 512;
/// Staging files carry this suffix beside their final name, in the same
/// directory, so the atomic replace is a same-volume rename.
pub const PROXY_STAGING_SUFFIX: &str = ".staging";
/// Each published artifact is sealed by a sidecar carrying the SHA-256 of
/// its exact bytes. Publication renames the artifact first and the sidecar
/// second, so a crash between the two leaves an artifact without a seal —
/// which recovery removes and never serves. Consumption re-hashes the
/// artifact against the seal, so corruption anywhere in the file is refused,
/// not just corruption a first-frame decode would notice.
pub const PROXY_DIGEST_SUFFIX: &str = ".sha256";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProxyWorkerError {
    CacheDirectory(String),
    StagingUnavailable(String),
    SourceFingerprintMismatch { expected: String, observed: String },
    SourceUnreadable(String),
    Probe(String),
    Contract(ProxyError),
    Admission(String),
    EncoderMissing,
    EncodeFailed { status: Option<i32>, stderr: String },
    DeadlineExceeded { seconds: u64 },
    StagingOverCap { bytes: u64, limit: u64 },
    Cancelled,
    InvalidArtifact(String),
    Publish(String),
    Evict(String),
}

impl std::fmt::Display for ProxyWorkerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CacheDirectory(error) => write!(formatter, "proxy cache directory: {error}"),
            Self::StagingUnavailable(error) => {
                write!(formatter, "proxy staging file: {error}")
            }
            Self::SourceFingerprintMismatch { expected, observed } => write!(
                formatter,
                "source bytes changed since verification: expected {expected}, observed {observed}"
            ),
            Self::SourceUnreadable(error) => write!(formatter, "proxy source: {error}"),
            Self::Probe(error) => write!(formatter, "proxy source probe: {error}"),
            Self::Contract(error) => write!(formatter, "proxy input contract: {error}"),
            Self::Admission(error) => write!(formatter, "proxy source admission: {error}"),
            Self::EncoderMissing => {
                formatter.write_str("ffmpeg is not on PATH; the proxy encoder is unavailable")
            }
            Self::EncodeFailed { status, stderr } => write!(
                formatter,
                "proxy encode failed (status {status:?}): {stderr}"
            ),
            Self::DeadlineExceeded { seconds } => write!(
                formatter,
                "proxy encode exceeded its absolute {seconds}s deadline"
            ),
            Self::StagingOverCap { bytes, limit } => write!(
                formatter,
                "proxy staging reached {bytes} bytes; the per-artifact limit is {limit}"
            ),
            Self::Cancelled => formatter.write_str("proxy encode cancelled"),
            Self::InvalidArtifact(error) => {
                write!(formatter, "proxy artifact failed validation: {error}")
            }
            Self::Publish(error) => write!(formatter, "proxy publication: {error}"),
            Self::Evict(error) => write!(formatter, "proxy eviction: {error}"),
        }
    }
}

impl std::error::Error for ProxyWorkerError {}

impl From<ProxyError> for ProxyWorkerError {
    fn from(error: ProxyError) -> Self {
        Self::Contract(error)
    }
}

/// What a cache-directory scan found and repaired. Staging leftovers are a
/// crash's only residue; they are removed and never published or counted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ProxyCacheRecovery {
    pub artifacts: usize,
    pub staging_removed: usize,
    /// Artifacts found without a valid digest seal (or seals without an
    /// artifact): the residue of a crash between the two publication
    /// renames. Removed, never served, never counted.
    pub incomplete_removed: usize,
    pub foreign_entries: usize,
}

#[derive(Debug, Clone)]
struct IndexEntry {
    artifact_bytes: u64,
    last_used_ordinal: u64,
    artifact_digest: String,
}

/// The cache directory plus its rebuilt in-memory index. One instance is
/// shared behind a mutex between the render thread (consumption, touch) and
/// the encode worker (preflight, evict, publish, insert).
pub struct ProxyCacheStore {
    root: PathBuf,
    entries: BTreeMap<ProxyCacheKey, IndexEntry>,
    next_ordinal: u64,
}

impl ProxyCacheStore {
    /// Open (creating if needed) and scan the cache directory. Scanning
    /// removes crash-leftover staging files, admits only artifacts whose
    /// file name is a canonical 64-hex cache key with the `.mkv` extension,
    /// and leaves every foreign file untouched.
    pub fn open(root: PathBuf) -> Result<(Self, ProxyCacheRecovery), ProxyWorkerError> {
        std::fs::create_dir_all(&root)
            .map_err(|error| ProxyWorkerError::CacheDirectory(error.to_string()))?;
        let mut recovery = ProxyCacheRecovery::default();
        let mut artifacts: BTreeMap<ProxyCacheKey, u64> = BTreeMap::new();
        let mut seals: BTreeMap<ProxyCacheKey, String> = BTreeMap::new();
        let listing = std::fs::read_dir(&root)
            .map_err(|error| ProxyWorkerError::CacheDirectory(error.to_string()))?;
        for entry in listing {
            let entry =
                entry.map_err(|error| ProxyWorkerError::CacheDirectory(error.to_string()))?;
            let path = entry.path();
            if !path.is_file() {
                recovery.foreign_entries += 1;
                continue;
            }
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                recovery.foreign_entries += 1;
                continue;
            };
            if name.ends_with(PROXY_STAGING_SUFFIX) {
                std::fs::remove_file(&path)
                    .map_err(|error| ProxyWorkerError::CacheDirectory(error.to_string()))?;
                recovery.staging_removed += 1;
                continue;
            }
            if let Some(stem) = name
                .strip_suffix(PROXY_DIGEST_SUFFIX)
                .and_then(|stem| stem.strip_suffix(".mkv"))
            {
                let (Ok(key), Ok(digest)) = (
                    ProxyCacheKey::from_hex(stem),
                    std::fs::read_to_string(&path),
                ) else {
                    recovery.foreign_entries += 1;
                    continue;
                };
                let digest = digest.trim().to_owned();
                if digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                    seals.insert(key, digest);
                } else {
                    std::fs::remove_file(&path)
                        .map_err(|error| ProxyWorkerError::CacheDirectory(error.to_string()))?;
                    recovery.incomplete_removed += 1;
                }
                continue;
            }
            let Some(stem) = name.strip_suffix(".mkv") else {
                recovery.foreign_entries += 1;
                continue;
            };
            let Ok(key) = ProxyCacheKey::from_hex(stem) else {
                recovery.foreign_entries += 1;
                continue;
            };
            let bytes = entry
                .metadata()
                .map_err(|error| ProxyWorkerError::CacheDirectory(error.to_string()))?
                .len();
            if bytes == 0 {
                recovery.foreign_entries += 1;
                continue;
            }
            artifacts.insert(key, bytes);
        }

        // Pair artifacts with their seals. Either half alone is the residue
        // of an interrupted publication: remove it, count it, serve nothing.
        let mut entries = BTreeMap::new();
        for (key, bytes) in artifacts {
            match seals.remove(&key) {
                Some(digest) => {
                    entries.insert(
                        key,
                        IndexEntry {
                            artifact_bytes: bytes,
                            last_used_ordinal: 0,
                            artifact_digest: digest,
                        },
                    );
                    recovery.artifacts += 1;
                }
                None => {
                    let orphan = root.join(format!("{}.mkv", key.to_hex()));
                    std::fs::remove_file(&orphan)
                        .map_err(|error| ProxyWorkerError::CacheDirectory(error.to_string()))?;
                    recovery.incomplete_removed += 1;
                }
            }
        }
        for (key, _) in seals {
            let orphan = root.join(format!("{}.mkv{}", key.to_hex(), PROXY_DIGEST_SUFFIX));
            std::fs::remove_file(&orphan)
                .map_err(|error| ProxyWorkerError::CacheDirectory(error.to_string()))?;
            recovery.incomplete_removed += 1;
        }
        Ok((
            Self {
                root,
                entries,
                next_ordinal: 1,
            },
            recovery,
        ))
    }

    pub fn artifact_path(&self, key: ProxyCacheKey) -> PathBuf {
        self.root
            .join(key.artifact_file_name(crate::proxy::ProxyFormat::Ffv1Matroska))
    }

    fn staging_path(&self, key: ProxyCacheKey) -> PathBuf {
        let mut name = key
            .artifact_file_name(crate::proxy::ProxyFormat::Ffv1Matroska)
            .into_bytes();
        name.extend_from_slice(PROXY_STAGING_SUFFIX.as_bytes());
        self.root.join(String::from_utf8(name).expect("ascii name"))
    }

    fn seal_path(&self, key: ProxyCacheKey) -> PathBuf {
        let mut name = key
            .artifact_file_name(crate::proxy::ProxyFormat::Ffv1Matroska)
            .into_bytes();
        name.extend_from_slice(PROXY_DIGEST_SUFFIX.as_bytes());
        self.root.join(String::from_utf8(name).expect("ascii name"))
    }

    pub fn contains(&self, key: ProxyCacheKey) -> bool {
        self.entries.contains_key(&key)
    }

    /// Record a use so eviction order reflects it. Unknown keys are inert.
    pub fn touch(&mut self, key: ProxyCacheKey) {
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_used_ordinal = self.next_ordinal;
            self.next_ordinal = self.next_ordinal.saturating_add(1);
        }
    }

    pub fn entries_for_preflight(&self) -> Vec<ProxyCacheEntry> {
        self.entries
            .iter()
            .map(|(key, entry)| ProxyCacheEntry {
                key: *key,
                artifact_bytes: entry.artifact_bytes,
                last_used_ordinal: entry.last_used_ordinal,
            })
            .collect()
    }

    /// Delete the artifacts a pure preflight chose to evict and return the
    /// receipt. Keys and byte counts only — no path enters the receipt.
    pub fn evict_for_plan(
        &mut self,
        plan: &ProxyCachePlan,
    ) -> Result<ProxyEvictionReceipt, ProxyWorkerError> {
        let mut evicted = Vec::with_capacity(plan.evict.len());
        for key in plan.evict.iter() {
            let Some(entry) = self.entries.remove(key) else {
                return Err(ProxyWorkerError::Evict(format!(
                    "eviction plan names unknown key {}",
                    key.to_hex()
                )));
            };
            let path = self.artifact_path(*key);
            std::fs::remove_file(&path)
                .map_err(|error| ProxyWorkerError::Evict(error.to_string()))?;
            let _ = std::fs::remove_file(self.seal_path(*key));
            evicted.push((*key, entry.artifact_bytes));
        }
        Ok(ProxyEvictionReceipt {
            evicted,
            retained_bytes_before_stage: plan.retained_bytes_before_stage,
            committed_bytes: plan.committed_bytes,
            committed_entries: plan.committed_entries,
        })
    }

    /// Execute the publication half of the commit law for a staged artifact:
    /// staging fsync, atomic replace onto the final name, digest-seal
    /// publication through its own staged rename, parent-directory sync,
    /// then index insertion with a fresh ordinal. The artifact renames
    /// first and the seal second, so a crash between the two leaves an
    /// unsealed artifact that recovery removes rather than serves.
    pub fn publish_staged(
        &mut self,
        key: ProxyCacheKey,
        artifact_bytes: u64,
        artifact_digest: String,
    ) -> Result<(), ProxyWorkerError> {
        let staging = self.staging_path(key);
        let target = self.artifact_path(key);
        // Windows refuses FlushFileBuffers on a read-only handle, so the
        // pre-publish sync must open the staging file writable.
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&staging)
            .map_err(|error| ProxyWorkerError::Publish(error.to_string()))?;
        file.sync_all()
            .map_err(|error| ProxyWorkerError::Publish(error.to_string()))?;
        drop(file);
        std::fs::rename(&staging, &target)
            .map_err(|error| ProxyWorkerError::Publish(error.to_string()))?;

        let seal = self.seal_path(key);
        let mut seal_staging = seal.clone().into_os_string();
        seal_staging.push(PROXY_STAGING_SUFFIX);
        let seal_staging = PathBuf::from(seal_staging);
        let _ = std::fs::remove_file(&seal_staging);
        {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&seal_staging)
                .map_err(|error| ProxyWorkerError::Publish(error.to_string()))?;
            file.write_all(artifact_digest.as_bytes())
                .map_err(|error| ProxyWorkerError::Publish(error.to_string()))?;
            file.sync_all()
                .map_err(|error| ProxyWorkerError::Publish(error.to_string()))?;
        }
        std::fs::rename(&seal_staging, &seal)
            .map_err(|error| ProxyWorkerError::Publish(error.to_string()))?;
        sync_parent_directory(&target)
            .map_err(|error| ProxyWorkerError::Publish(error.to_string()))?;
        self.entries.insert(
            key,
            IndexEntry {
                artifact_bytes,
                last_used_ordinal: self.next_ordinal,
                artifact_digest,
            },
        );
        self.next_ordinal = self.next_ordinal.saturating_add(1);
        Ok(())
    }

    fn artifact_digest(&self, key: ProxyCacheKey) -> Option<&str> {
        self.entries
            .get(&key)
            .map(|entry| entry.artifact_digest.as_str())
    }

    /// Refuse and remove a cache entry whose artifact failed validation. A
    /// cache entry is regenerable by construction, so deletion is safe; the
    /// caller reports the refusal.
    pub fn discard_invalid(&mut self, key: ProxyCacheKey) {
        self.entries.remove(&key);
        let _ = std::fs::remove_file(self.artifact_path(key));
        let _ = std::fs::remove_file(self.seal_path(key));
    }
}

/// The eviction receipt: which keys were evicted at what size, and the
/// planned totals they made room for. Deliberately path-free.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProxyEvictionReceipt {
    pub evicted: Vec<(ProxyCacheKey, u64)>,
    pub retained_bytes_before_stage: u64,
    pub committed_bytes: u64,
    pub committed_entries: usize,
}

/// Streaming SHA-256 of a file's exact bytes, 1 MiB per read.
fn hash_artifact_bytes(path: &Path) -> Result<String, ProxyWorkerError> {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path)
        .map_err(|error| ProxyWorkerError::InvalidArtifact(error.to_string()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| ProxyWorkerError::InvalidArtifact(error.to_string()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(hex, "{byte:02x}");
    }
    Ok(hex)
}

fn sync_parent_directory(child: &Path) -> std::io::Result<()> {
    let Some(parent) = child.parent() else {
        return Ok(());
    };
    #[cfg(unix)]
    {
        std::fs::File::open(parent)?.sync_all()
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
        // FlushFileBuffers demands a writable handle even for a directory;
        // FILE_FLAG_BACKUP_SEMANTICS is what makes opening one legal at all.
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
            .open(parent)?
            .sync_all()
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = parent;
        Ok(())
    }
}

/// The default host cache root, beside the TLS material:
/// `%LOCALAPPDATA%\collide-o-scope\proxy-cache`, or `./.proxy-cache` when the
/// environment variable is absent.
pub fn default_proxy_cache_root() -> PathBuf {
    std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .map(|base| base.join("collide-o-scope").join("proxy-cache"))
        .unwrap_or_else(|| PathBuf::from(".proxy-cache"))
}

/// Probe the source container's bounded metadata and report which stream the
/// `best(Type::Video)` law selects, so the encode maps that exact stream.
pub fn probe_proxy_source(path: &Path) -> Result<(ProxySourceProbe, usize), ProxyWorkerError> {
    ffmpeg_next::init().map_err(|error| ProxyWorkerError::Probe(error.to_string()))?;
    let input = ffmpeg_next::format::input(&path)
        .map_err(|error| ProxyWorkerError::Probe(error.to_string()))?;
    let mut video_streams: u32 = 0;
    let mut audio_streams: u32 = 0;
    let mut container_streams: u32 = 0;
    for stream in input.streams() {
        container_streams = container_streams.saturating_add(1);
        match stream.parameters().medium() {
            ffmpeg_next::media::Type::Video => video_streams = video_streams.saturating_add(1),
            ffmpeg_next::media::Type::Audio => audio_streams = audio_streams.saturating_add(1),
            _ => {}
        }
    }
    let best = input
        .streams()
        .best(ffmpeg_next::media::Type::Video)
        .ok_or(ProxyWorkerError::Contract(ProxyError::NoVideoStream))?;
    let best_index = best.index();
    let decoder = ffmpeg_next::codec::context::Context::from_parameters(best.parameters())
        .map_err(|error| ProxyWorkerError::Probe(error.to_string()))?
        .decoder()
        .video()
        .map_err(|error| ProxyWorkerError::Probe(error.to_string()))?;
    let duration_micros = u64::try_from(input.duration()).unwrap_or(0);
    Ok((
        ProxySourceProbe {
            container_streams,
            video_streams,
            audio_streams,
            video_width: decoder.width(),
            video_height: decoder.height(),
            duration_micros,
        },
        best_index,
    ))
}

/// Build the complete encode argv from the contract's plan, separately from
/// any subprocess so the exact invocation is provable without one — the
/// `build_ffmpeg_args` precedent. The staging path was already created
/// create-new by the caller, so `-y` here only truncates our own reservation
/// and can never clobber an unknown existing file.
pub fn build_proxy_encode_args(
    source: &Path,
    staging: &Path,
    probe: ProxySourceProbe,
    plan: &ProxyInputPlan,
    best_video_stream_index: usize,
) -> Vec<String> {
    let mut args = vec![
        "-y".to_owned(),
        "-hide_banner".to_owned(),
        "-loglevel".to_owned(),
        "error".to_owned(),
        "-nostats".to_owned(),
        "-i".to_owned(),
        source.to_string_lossy().into_owned(),
        "-map".to_owned(),
        format!("0:{best_video_stream_index}"),
    ];
    let mut filters = Vec::new();
    if let ProxyFrameTimingLaw::ResampleToConstantRate {
        numerator,
        denominator,
    } = plan.frame_timing
    {
        filters.push(format!("fps={numerator}/{denominator}"));
    }
    if plan.output_width != probe.video_width || plan.output_height != probe.video_height {
        filters.push(format!(
            "scale={}:{}",
            plan.output_width, plan.output_height
        ));
    }
    if filters.is_empty() {
        if plan.frame_timing == ProxyFrameTimingLaw::PreserveSourceTiming {
            args.extend(["-fps_mode".to_owned(), "passthrough".to_owned()]);
        }
    } else {
        args.extend(["-vf".to_owned(), filters.join(",")]);
    }
    args.extend(["-c:v".to_owned(), "ffv1".to_owned()]);
    match plan.audio {
        ProxyAudioInputLaw::FirstOrderedStreamBitExactCopy => {
            args.extend([
                "-map".to_owned(),
                "0:a:0".to_owned(),
                "-c:a".to_owned(),
                "copy".to_owned(),
            ]);
        }
        ProxyAudioInputLaw::NoAudioTrack { .. } => args.push("-an".to_owned()),
    }
    args.extend([
        "-map_metadata".to_owned(),
        "-1".to_owned(),
        "-f".to_owned(),
        "matroska".to_owned(),
        staging.to_string_lossy().into_owned(),
    ]);
    args
}

/// Validate a proxy artifact's decoded identity against the plan that names
/// it. This is the one validation predicate: publication calls it before an
/// artifact may be renamed into the cache, and consumption calls it before a
/// decoder may open one. Exit codes prove a process ran; this proves the
/// artifact decodes to what the plan promised.
pub fn validate_proxy_artifact(
    path: &Path,
    plan: &ProxyInputPlan,
) -> Result<u64, ProxyWorkerError> {
    let bytes = std::fs::metadata(path)
        .map_err(|error| ProxyWorkerError::InvalidArtifact(error.to_string()))?
        .len();
    if bytes == 0 {
        return Err(ProxyWorkerError::InvalidArtifact(
            "artifact is empty".to_owned(),
        ));
    }
    ffmpeg_next::init().map_err(|error| ProxyWorkerError::InvalidArtifact(error.to_string()))?;
    let mut input = ffmpeg_next::format::input(&path)
        .map_err(|error| ProxyWorkerError::InvalidArtifact(error.to_string()))?;
    let format_name = input.format().name().to_owned();
    if !format_name.contains("matroska") {
        return Err(ProxyWorkerError::InvalidArtifact(format!(
            "container is {format_name}, not matroska"
        )));
    }
    let audio_streams = input
        .streams()
        .filter(|stream| stream.parameters().medium() == ffmpeg_next::media::Type::Audio)
        .count();
    let expected_audio = match plan.audio {
        ProxyAudioInputLaw::FirstOrderedStreamBitExactCopy => 1,
        ProxyAudioInputLaw::NoAudioTrack { .. } => 0,
    };
    if audio_streams != expected_audio {
        return Err(ProxyWorkerError::InvalidArtifact(format!(
            "artifact carries {audio_streams} audio streams; the plan requires {expected_audio}"
        )));
    }
    let stream = input
        .streams()
        .best(ffmpeg_next::media::Type::Video)
        .ok_or_else(|| ProxyWorkerError::InvalidArtifact("no video stream".to_owned()))?;
    if stream.parameters().id() != ffmpeg_next::codec::Id::FFV1 {
        return Err(ProxyWorkerError::InvalidArtifact(format!(
            "video codec is {:?}, not FFV1",
            stream.parameters().id()
        )));
    }
    let stream_index = stream.index();
    let mut decoder = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
        .map_err(|error| ProxyWorkerError::InvalidArtifact(error.to_string()))?
        .decoder()
        .video()
        .map_err(|error| ProxyWorkerError::InvalidArtifact(error.to_string()))?;
    if decoder.width() != plan.output_width || decoder.height() != plan.output_height {
        return Err(ProxyWorkerError::InvalidArtifact(format!(
            "artifact is {}x{}; the plan requires {}x{}",
            decoder.width(),
            decoder.height(),
            plan.output_width,
            plan.output_height
        )));
    }
    let mut frame = ffmpeg_next::util::frame::video::Video::empty();
    for (inspected, (stream, packet)) in input.packets().enumerate() {
        if inspected >= PROXY_VALIDATE_MAX_PACKETS {
            break;
        }
        if stream.index() != stream_index {
            continue;
        }
        if decoder.send_packet(&packet).is_err() {
            return Err(ProxyWorkerError::InvalidArtifact(
                "first video packet failed to decode".to_owned(),
            ));
        }
        if decoder.receive_frame(&mut frame).is_ok() {
            if frame.width() != plan.output_width || frame.height() != plan.output_height {
                return Err(ProxyWorkerError::InvalidArtifact(format!(
                    "decoded frame is {}x{}; the plan requires {}x{}",
                    frame.width(),
                    frame.height(),
                    plan.output_width,
                    plan.output_height
                )));
            }
            return Ok(bytes);
        }
    }
    Err(ProxyWorkerError::InvalidArtifact(format!(
        "no video frame decoded within {PROXY_VALIDATE_MAX_PACKETS} packets"
    )))
}

/// A request to encode one proxy. The identity was verified when the source
/// was admitted; the job re-fingerprints before consuming a single byte so a
/// post-load mutation cannot ride a stale digest into a content-keyed
/// artifact.
#[derive(Debug, Clone)]
pub struct ProxyEncodeJob {
    pub source_path: PathBuf,
    pub identity: ContentIdentity,
    pub settings: ProxySettings,
}

/// What a completed encode produced. Keys, byte counts, and the receipt —
/// deliberately no path and no filesystem metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProxyEncodeOutcome {
    pub key: ProxyCacheKey,
    pub artifact_bytes: u64,
    pub already_cached: bool,
    pub eviction: Option<ProxyEvictionReceipt>,
    pub encode_seconds: u64,
}

struct StagingGuard {
    path: PathBuf,
    armed: bool,
}

impl Drop for StagingGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Run one complete encode job against the shared store. Locks the store
/// only for index reads, eviction, and publication — never across the
/// encode itself.
pub fn run_proxy_encode_job(
    store: &Mutex<ProxyCacheStore>,
    job: &ProxyEncodeJob,
    limits: ProxyCacheLimits,
    media_policy: &MediaSafetyPolicy,
    cancel: &AtomicBool,
) -> Result<ProxyEncodeOutcome, ProxyWorkerError> {
    let settings = job.settings.validate()?;
    let key = ProxyCacheKey::derive(&job.identity, settings)?;

    // A post-verification byte change must be refused, not encoded under the
    // old digest. A fresh session defeats the per-session memoization.
    let mut fingerprints = FingerprintSession::new(FingerprintLimits::default())
        .map_err(|error| ProxyWorkerError::SourceUnreadable(error.to_string()))?;
    let observed = fingerprints
        .fingerprint(&job.source_path)
        .map_err(|error| ProxyWorkerError::SourceUnreadable(error.to_string()))?;
    if observed != job.identity {
        return Err(ProxyWorkerError::SourceFingerprintMismatch {
            expected: job.identity.source_reference(),
            observed: observed.source_reference(),
        });
    }

    let (probe, best_video_stream_index) = probe_proxy_source(&job.source_path)?;
    let plan = plan_proxy_input(probe, settings)?;

    // Source admission is MediaSafetyPolicy's, held for the encode lifetime.
    let _reservation = media_policy
        .reserve_source(
            MediaSourceKind::Video,
            probe.video_width,
            probe.video_height,
            MediaDeviceLimits::none(),
        )
        .map_err(|error| ProxyWorkerError::Admission(error.to_string()))?;

    // An existing valid artifact is a cache hit, not a re-encode; an invalid
    // one is refused, discarded, and replaced. The seal check runs here too,
    // so a corrupt cached artifact can never be reported as already cached.
    let existing = {
        let store = lock_store(store);
        store.contains(key).then(|| {
            (
                store.artifact_path(key),
                store.artifact_digest(key).map(str::to_owned),
            )
        })
    };
    if let Some((path, sealed_digest)) = existing {
        let seal_intact = match (&sealed_digest, hash_artifact_bytes(&path)) {
            (Some(sealed), Ok(observed)) => *sealed == observed,
            _ => false,
        };
        if seal_intact {
            match validate_proxy_artifact(&path, &plan) {
                Ok(bytes) => {
                    let mut store = lock_store(store);
                    store.touch(key);
                    return Ok(ProxyEncodeOutcome {
                        key,
                        artifact_bytes: bytes,
                        already_cached: true,
                        eviction: None,
                        encode_seconds: 0,
                    });
                }
                Err(_) => lock_store(store).discard_invalid(key),
            }
        } else {
            lock_store(store).discard_invalid(key);
        }
    }

    let staging = {
        let store = lock_store(store);
        store.staging_path(key)
    };
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&staging)
    {
        Ok(file) => drop(file),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            std::fs::remove_file(&staging)
                .map_err(|error| ProxyWorkerError::StagingUnavailable(error.to_string()))?;
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&staging)
                .map_err(|error| ProxyWorkerError::StagingUnavailable(error.to_string()))?;
        }
        Err(error) => return Err(ProxyWorkerError::StagingUnavailable(error.to_string())),
    }
    let mut guard = StagingGuard {
        path: staging.clone(),
        armed: true,
    };

    let args = build_proxy_encode_args(
        &job.source_path,
        &staging,
        probe,
        &plan,
        best_video_stream_index,
    );
    let started = Instant::now();
    run_bounded_proxy_encode(&args, &staging, plan.deadline_seconds, limits, cancel)?;
    let encode_seconds = started.elapsed().as_secs();

    let artifact_bytes = validate_proxy_artifact(&staging, &plan)?;
    let artifact_digest = hash_artifact_bytes(&staging)?;

    let mut locked = lock_store(store);
    let cache_plan =
        ProxyCachePlan::preflight(key, artifact_bytes, &locked.entries_for_preflight(), limits)?;
    let receipt = locked.evict_for_plan(&cache_plan)?;
    locked.publish_staged(key, artifact_bytes, artifact_digest)?;
    guard.armed = false;
    drop(locked);

    Ok(ProxyEncodeOutcome {
        key,
        artifact_bytes,
        already_cached: false,
        eviction: Some(receipt),
        encode_seconds,
    })
}

fn lock_store(store: &Mutex<ProxyCacheStore>) -> std::sync::MutexGuard<'_, ProxyCacheStore> {
    store
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Spawn and babysit the encode child. Modeled on the thumbnail helpers'
/// bounded runner, with two deliberate differences that keep this a sibling
/// rather than a second caller of that seam: cancellation is a caller-owned
/// flag instead of the library generation (a proxy is content-keyed and
/// survives library changes), and the loop additionally polls the staging
/// file against the per-artifact byte cap so a runaway encode is killed at
/// the bound instead of at the deadline.
fn run_bounded_proxy_encode(
    args: &[String],
    staging: &Path,
    deadline_seconds: u64,
    limits: ProxyCacheLimits,
    cancel: &AtomicBool,
) -> Result<(), ProxyWorkerError> {
    let limits = limits.validate()?;
    let mut child = std::process::Command::new("ffmpeg")
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                ProxyWorkerError::EncoderMissing
            } else {
                ProxyWorkerError::EncodeFailed {
                    status: None,
                    stderr: error.to_string(),
                }
            }
        })?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let stdout_reader = std::thread::spawn(move || read_bounded(stdout, PROXY_ENCODE_STDOUT_LIMIT));
    let stderr_reader = std::thread::spawn(move || read_bounded(stderr, PROXY_ENCODE_STDERR_LIMIT));

    let started = Instant::now();
    let mut kill_cause: Option<ProxyWorkerError> = None;
    let status = loop {
        if kill_cause.is_none() {
            if started.elapsed().as_secs() >= deadline_seconds {
                kill_cause = Some(ProxyWorkerError::DeadlineExceeded {
                    seconds: deadline_seconds,
                });
                let _ = child.kill();
            } else if cancel.load(Ordering::Relaxed) {
                kill_cause = Some(ProxyWorkerError::Cancelled);
                let _ = child.kill();
            } else if let Ok(metadata) = std::fs::metadata(staging) {
                if metadata.len() > limits.max_entry_bytes {
                    kill_cause = Some(ProxyWorkerError::StagingOverCap {
                        bytes: metadata.len(),
                        limit: limits.max_entry_bytes,
                    });
                    let _ = child.kill();
                }
            }
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => std::thread::sleep(Duration::from_millis(PROXY_ENCODE_POLL_MS)),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdout_reader.join();
                let _ = stderr_reader.join();
                return Err(kill_cause.unwrap_or(ProxyWorkerError::EncodeFailed {
                    status: None,
                    stderr: error.to_string(),
                }));
            }
        }
    };
    let _ = stdout_reader.join();
    let stderr_bytes = stderr_reader.join().unwrap_or_default();
    if let Some(cause) = kill_cause {
        return Err(cause);
    }
    if !status.success() {
        return Err(ProxyWorkerError::EncodeFailed {
            status: status.code(),
            stderr: String::from_utf8_lossy(&stderr_bytes).into_owned(),
        });
    }
    Ok(())
}

fn read_bounded(pipe: Option<impl Read>, limit: usize) -> Vec<u8> {
    let Some(pipe) = pipe else {
        return Vec::new();
    };
    let mut buffer = Vec::new();
    let _ = pipe.take(limit as u64 + 1).read_to_end(&mut buffer);
    buffer.truncate(limit);
    buffer
}

/// What the render thread hears back from the worker, drained nonblockingly
/// once per frame.
#[derive(Debug, Clone)]
pub enum ProxyWorkerEvent {
    Started {
        identity: ContentIdentity,
        key: ProxyCacheKey,
    },
    Finished {
        identity: ContentIdentity,
        outcome: ProxyEncodeOutcome,
    },
    Failed {
        identity: ContentIdentity,
        error: String,
    },
}

/// The single encode worker: one thread, a one-slot job queue with a
/// drop-new refusal while busy, and a nonblocking event channel back.
pub struct ProxyEncodeWorker {
    jobs: mpsc::SyncSender<ProxyEncodeJob>,
    events: mpsc::Receiver<ProxyWorkerEvent>,
    cancel: Arc<AtomicBool>,
}

impl ProxyEncodeWorker {
    pub fn spawn(
        store: Arc<Mutex<ProxyCacheStore>>,
        limits: ProxyCacheLimits,
        media_policy: MediaSafetyPolicy,
    ) -> Self {
        let (job_sender, job_receiver) = mpsc::sync_channel::<ProxyEncodeJob>(1);
        let (event_sender, event_receiver) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = Arc::clone(&cancel);
        let builder = std::thread::Builder::new().name("proxy-encode".to_owned());
        let spawned = builder.spawn(move || {
            while let Ok(job) = job_receiver.recv() {
                let key = match ProxyCacheKey::derive(&job.identity, job.settings) {
                    Ok(key) => key,
                    Err(error) => {
                        let _ = event_sender.send(ProxyWorkerEvent::Failed {
                            identity: job.identity.clone(),
                            error: error.to_string(),
                        });
                        continue;
                    }
                };
                let _ = event_sender.send(ProxyWorkerEvent::Started {
                    identity: job.identity.clone(),
                    key,
                });
                match run_proxy_encode_job(&store, &job, limits, &media_policy, &worker_cancel) {
                    Ok(outcome) => {
                        let _ = event_sender.send(ProxyWorkerEvent::Finished {
                            identity: job.identity.clone(),
                            outcome,
                        });
                    }
                    Err(error) => {
                        let _ = event_sender.send(ProxyWorkerEvent::Failed {
                            identity: job.identity.clone(),
                            error: error.to_string(),
                        });
                    }
                }
            }
        });
        if let Err(error) = spawned {
            log::warn!("proxy encode worker thread failed to spawn: {error}");
        }
        Self {
            jobs: job_sender,
            events: event_receiver,
            cancel,
        }
    }

    /// Submit a job; a busy worker refuses rather than queueing a backlog.
    pub fn submit(&self, job: ProxyEncodeJob) -> Result<(), &'static str> {
        self.jobs
            .try_send(job)
            .map_err(|_| "a proxy encode is already running")
    }

    pub fn drain_events(&self) -> Vec<ProxyWorkerEvent> {
        let mut events = Vec::new();
        while let Ok(event) = self.events.try_recv() {
            events.push(event);
        }
        events
    }

    pub fn cancel_all(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// What consulting the cache for a source decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProxyConsultation {
    /// A validated artifact exists; open the decoder on this path while the
    /// layer keeps the original's identity for persistence and export.
    Adopted {
        key: ProxyCacheKey,
        artifact_path: PathBuf,
    },
    /// No artifact for this identity/settings pair; use the original.
    NoArtifact,
    /// An artifact existed but failed decoded-identity validation. It was
    /// refused and discarded — a partial or corrupt artifact is never
    /// served — and the original is used.
    RefusedInvalid { key: ProxyCacheKey, error: String },
}

/// Consult the cache before opening a decoder on a content-referenced
/// source. Validation is the same predicate publication used, so a corrupt
/// artifact is refused at both boundaries by one law.
pub fn consult_proxy_cache(
    store: &Mutex<ProxyCacheStore>,
    identity: &ContentIdentity,
    settings: ProxySettings,
    original_source: &Path,
) -> ProxyConsultation {
    let Ok(key) = ProxyCacheKey::derive(identity, settings) else {
        return ProxyConsultation::NoArtifact;
    };
    let (artifact_path, sealed_digest) = {
        let locked = lock_store(store);
        if !locked.contains(key) {
            return ProxyConsultation::NoArtifact;
        }
        let Some(digest) = locked.artifact_digest(key) else {
            return ProxyConsultation::NoArtifact;
        };
        (locked.artifact_path(key), digest.to_owned())
    };
    // Integrity first: the artifact's bytes must still be exactly the bytes
    // the publication sealed, so corruption anywhere in the file — not only
    // where a first-frame decode would look — is refused.
    match hash_artifact_bytes(&artifact_path) {
        Ok(observed) if observed == sealed_digest => {}
        Ok(_) => {
            lock_store(store).discard_invalid(key);
            return ProxyConsultation::RefusedInvalid {
                key,
                error: "artifact bytes no longer match their published seal".to_owned(),
            };
        }
        Err(error) => {
            lock_store(store).discard_invalid(key);
            return ProxyConsultation::RefusedInvalid {
                key,
                error: error.to_string(),
            };
        }
    }
    let plan = match probe_proxy_source(original_source)
        .map_err(|error| error.to_string())
        .and_then(|(probe, _)| plan_proxy_input(probe, settings).map_err(|error| error.to_string()))
    {
        Ok(plan) => plan,
        Err(error) => {
            return ProxyConsultation::RefusedInvalid { key, error };
        }
    };
    match validate_proxy_artifact(&artifact_path, &plan) {
        Ok(_) => {
            lock_store(store).touch(key);
            ProxyConsultation::Adopted { key, artifact_path }
        }
        Err(error) => {
            lock_store(store).discard_invalid(key);
            ProxyConsultation::RefusedInvalid {
                key,
                error: error.to_string(),
            }
        }
    }
}

// --- Hot adoption ----------------------------------------------------------

/// The playhead seed for an adoption decoder follows the performance
/// preparer's deadline shape: an absolute timeout, a short poll that only
/// bounds reaction latency.
pub const PROXY_ADOPTION_SEED_TIMEOUT: Duration = Duration::from_secs(5);
pub const PROXY_ADOPTION_SEED_POLL: Duration = Duration::from_millis(2);

/// One live layer's claim on a pending hot adoption, captured on the render
/// thread when an encode completion arrives. Every field is re-validated
/// against the live layer before the prepared decoder is installed: the
/// stable layer ID must still resolve, the source-resource epoch must be
/// unchanged (a clip-slot switch makes the claim stale; a patch apply mints
/// new layer IDs entirely), and the layer must still carry this identity
/// un-proxied.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProxyAdoptionCandidate {
    pub layer_id: StableLayerId,
    pub source_resource_epoch: u64,
    /// The live playhead at capture, normalized `0..=1`. The seed frame is
    /// decoded here so the swap presents the position the audience is
    /// watching rather than the artifact's first frame.
    pub start_position: f64,
}

/// A hot-adoption request for one published identity. `original_source` is
/// the layers' current runtime path — the original file — which the cache
/// consultation re-probes and re-plans before the artifact is trusted.
pub struct ProxyAdoptionJob {
    pub identity: ContentIdentity,
    pub settings: ProxySettings,
    pub original_source: PathBuf,
    pub device_limits: MediaDeviceLimits,
    pub candidates: Vec<ProxyAdoptionCandidate>,
}

/// What adoption preparation produced. Unlike [`ProxyWorkerEvent`], the
/// `Prepared` variant is a resource hand-off to the owning render thread,
/// not telemetry: the artifact path must ride along because the layer
/// records it as its runtime source path — exactly as patch-load adoption
/// does — and it travels no further than that field.
pub enum ProxyAdoptionEvent {
    Prepared {
        identity: ContentIdentity,
        key: ProxyCacheKey,
        candidate: ProxyAdoptionCandidate,
        artifact_path: PathBuf,
        decoder: Box<ThreadedDecoder>,
        first_rgba: Vec<u8>,
        width: u32,
        height: u32,
        source_fps: f32,
        preload_bytes: u64,
    },
    Refused {
        identity: ContentIdentity,
        error: String,
    },
}

/// The bounded preparation for one adoption job: one cache consultation for
/// the identity — integrity seal, source re-probe, re-plan, decoded-identity
/// validation, exactly the patch-load adoption law — then one decoder open
/// and one playhead-seeded frame per candidate. Every refusal becomes a
/// typed, operator-facing event; nothing here touches a live layer.
fn prepare_adoption_job(
    store: &Mutex<ProxyCacheStore>,
    media_policy: &MediaSafetyPolicy,
    job: ProxyAdoptionJob,
    cancel: &AtomicBool,
) -> Vec<ProxyAdoptionEvent> {
    let mut events = Vec::new();
    if job.candidates.is_empty() {
        return events;
    }
    let (key, artifact_path) =
        match consult_proxy_cache(store, &job.identity, job.settings, &job.original_source) {
            ProxyConsultation::Adopted { key, artifact_path } => (key, artifact_path),
            ProxyConsultation::NoArtifact => {
                events.push(ProxyAdoptionEvent::Refused {
                    identity: job.identity,
                    error: "no published artifact for this identity".to_owned(),
                });
                return events;
            }
            ProxyConsultation::RefusedInvalid { error, .. } => {
                events.push(ProxyAdoptionEvent::Refused {
                    identity: job.identity,
                    error: format!("proxy artifact refused and discarded: {error}"),
                });
                return events;
            }
        };
    let Some(artifact_text) = artifact_path.to_str().map(str::to_owned) else {
        events.push(ProxyAdoptionEvent::Refused {
            identity: job.identity,
            error: "proxy artifact path is not valid Unicode".to_owned(),
        });
        return events;
    };
    for candidate in job.candidates {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let mut decoder = match ThreadedDecoder::open_with_media_policy(
            &artifact_text,
            media_policy,
            job.device_limits,
        ) {
            Ok(decoder) => decoder,
            Err(error) => {
                events.push(ProxyAdoptionEvent::Refused {
                    identity: job.identity.clone(),
                    error: format!("proxy artifact failed to open: {error}"),
                });
                continue;
            }
        };
        let seed = match decoder.select_seed_frame_at(
            candidate.start_position,
            PROXY_ADOPTION_SEED_TIMEOUT,
            PROXY_ADOPTION_SEED_POLL,
            &|| !cancel.load(Ordering::Relaxed),
        ) {
            Ok(frame) => frame,
            Err(SeedSelectError::Superseded) => break,
            Err(SeedSelectError::NoSeedFrame) => {
                events.push(ProxyAdoptionEvent::Refused {
                    identity: job.identity.clone(),
                    error: "proxy artifact opened without a decoded first frame".to_owned(),
                });
                continue;
            }
            Err(SeedSelectError::Decode(error)) => {
                events.push(ProxyAdoptionEvent::Refused {
                    identity: job.identity.clone(),
                    error,
                });
                continue;
            }
            Err(SeedSelectError::Timeout { target_seconds }) => {
                events.push(ProxyAdoptionEvent::Refused {
                    identity: job.identity.clone(),
                    error: format!(
                        "proxy artifact did not produce source position {:.6}s within {:.1}s",
                        target_seconds,
                        PROXY_ADOPTION_SEED_TIMEOUT.as_secs_f64()
                    ),
                });
                continue;
            }
        };
        let preload_bytes = decoder.media_allocation_plan().working_set_bytes;
        events.push(ProxyAdoptionEvent::Prepared {
            identity: job.identity.clone(),
            key,
            candidate,
            artifact_path: artifact_path.clone(),
            width: decoder.width,
            height: decoder.height,
            source_fps: decoder.fps,
            first_rgba: seed.rgba,
            decoder: Box::new(decoder),
            preload_bytes,
        });
    }
    events
}

/// Supervisor for hot adoption: one thread, a one-slot job queue that
/// refuses new work while busy instead of queueing a backlog, and a
/// nonblocking event channel drained once per frame by the render thread.
pub struct ProxyAdoptionWorker {
    jobs: mpsc::SyncSender<ProxyAdoptionJob>,
    events: mpsc::Receiver<ProxyAdoptionEvent>,
    cancel: Arc<AtomicBool>,
}

impl ProxyAdoptionWorker {
    pub fn spawn(store: Arc<Mutex<ProxyCacheStore>>, media_policy: MediaSafetyPolicy) -> Self {
        let (job_sender, job_receiver) = mpsc::sync_channel::<ProxyAdoptionJob>(1);
        let (event_sender, event_receiver) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = Arc::clone(&cancel);
        let builder = std::thread::Builder::new().name("proxy-adopt".to_owned());
        let spawned = builder.spawn(move || {
            while let Ok(job) = job_receiver.recv() {
                for event in prepare_adoption_job(&store, &media_policy, job, &worker_cancel) {
                    let _ = event_sender.send(event);
                }
            }
        });
        if let Err(error) = spawned {
            log::warn!("proxy adoption worker thread failed to spawn: {error}");
        }
        Self {
            jobs: job_sender,
            events: event_receiver,
            cancel,
        }
    }

    /// Submit a job; a busy worker refuses rather than queueing a backlog.
    pub fn submit(&self, job: ProxyAdoptionJob) -> Result<(), &'static str> {
        self.jobs
            .try_send(job)
            .map_err(|_| "a proxy adoption is already preparing")
    }

    pub fn drain_events(&self) -> Vec<ProxyAdoptionEvent> {
        let mut events = Vec::new();
        while let Ok(event) = self.events.try_recv() {
            events.push(event);
        }
        events
    }

    pub fn cancel_all(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proxy::ProxyAudioAbsenceCause;

    fn temp_root(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "collide-o-scope-proxy-worker-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    fn identity(digit: char, bytes: u64) -> ContentIdentity {
        ContentIdentity::new(digit.to_string().repeat(64), bytes).unwrap()
    }

    fn key(digit: char) -> ProxyCacheKey {
        ProxyCacheKey::derive(&identity(digit, 4_096), ProxySettings::default()).unwrap()
    }

    fn plan_for(probe: ProxySourceProbe, settings: ProxySettings) -> ProxyInputPlan {
        plan_proxy_input(probe, settings).unwrap()
    }

    fn probe_1080p() -> ProxySourceProbe {
        ProxySourceProbe {
            container_streams: 2,
            video_streams: 1,
            audio_streams: 1,
            video_width: 1_920,
            video_height: 1_080,
            duration_micros: 10_000_000,
        }
    }

    fn artifact_name(cache_key: ProxyCacheKey) -> String {
        cache_key.artifact_file_name(crate::proxy::ProxyFormat::Ffv1Matroska)
    }

    #[test]
    fn a_crash_leftover_staging_file_is_removed_and_never_published_or_counted() {
        let root = temp_root("crash");
        let staged_key = key('a');
        let published_key = key('b');
        let unsealed_key = key('c');
        let orphan_seal_key = key('d');
        let staging_name = format!("{}{}", artifact_name(staged_key), PROXY_STAGING_SUFFIX);
        std::fs::write(root.join(&staging_name), b"partial encode from a crash").unwrap();
        std::fs::write(
            root.join(artifact_name(published_key)),
            b"published artifact",
        )
        .unwrap();
        std::fs::write(
            root.join(format!(
                "{}{}",
                artifact_name(published_key),
                PROXY_DIGEST_SUFFIX
            )),
            "e".repeat(64),
        )
        .unwrap();
        // A crash between the artifact rename and the seal rename leaves an
        // unsealed artifact; the inverse leaves an orphan seal. Both are
        // interrupted publications: removed, never served.
        std::fs::write(root.join(artifact_name(unsealed_key)), b"no seal").unwrap();
        std::fs::write(
            root.join(format!(
                "{}{}",
                artifact_name(orphan_seal_key),
                PROXY_DIGEST_SUFFIX
            )),
            "f".repeat(64),
        )
        .unwrap();

        let (store, recovery) = ProxyCacheStore::open(root.clone()).unwrap();
        assert_eq!(recovery.staging_removed, 1);
        assert_eq!(recovery.incomplete_removed, 2);
        assert_eq!(recovery.artifacts, 1);
        assert!(
            !root.join(&staging_name).exists(),
            "a crash leftover must be removed, not retained"
        );
        assert!(!root.join(artifact_name(unsealed_key)).exists());
        assert!(!store.contains(staged_key));
        assert!(!store.contains(unsealed_key));
        assert!(store.contains(published_key));
        let entries = store.entries_for_preflight();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].key, published_key);

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn corruption_anywhere_in_a_published_artifact_is_refused_by_its_seal() {
        let root = temp_root("seal");
        let (store, _) = ProxyCacheStore::open(root.clone()).unwrap();
        let store = Mutex::new(store);
        let source_identity = identity('f', 4_096);
        let cached_key = ProxyCacheKey::derive(&source_identity, ProxySettings::default()).unwrap();
        let staging = lock_store(&store).staging_path(cached_key);
        std::fs::write(&staging, vec![0x11_u8; 8_192]).unwrap();
        let digest = hash_artifact_bytes(&staging).unwrap();
        lock_store(&store)
            .publish_staged(cached_key, 8_192, digest)
            .unwrap();

        // Flip bytes in the middle — where no first-frame decode looks.
        let artifact = lock_store(&store).artifact_path(cached_key);
        let mut bytes = std::fs::read(&artifact).unwrap();
        for byte in bytes.iter_mut().skip(4_000).take(64) {
            *byte ^= 0xFF;
        }
        std::fs::write(&artifact, &bytes).unwrap();

        let refused = consult_proxy_cache(
            &store,
            &source_identity,
            ProxySettings::default(),
            Path::new("unused-original.mp4"),
        );
        assert!(matches!(
            refused,
            ProxyConsultation::RefusedInvalid { key, ref error }
                if key == cached_key && error.contains("seal")
        ));
        assert!(
            !artifact.exists(),
            "a refused artifact is discarded, not retained"
        );
        assert!(!lock_store(&store).contains(cached_key));

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn the_atomic_publish_law_keeps_the_prior_artifact_readable_until_replacement() {
        let root = temp_root("publish");
        let (store, _) = ProxyCacheStore::open(root.clone()).unwrap();
        let store = Mutex::new(store);
        let target_key = key('c');

        let staging = lock_store(&store).staging_path(target_key);
        std::fs::write(&staging, b"first artifact").unwrap();
        let first_digest = hash_artifact_bytes(&staging).unwrap();
        lock_store(&store)
            .publish_staged(target_key, 14, first_digest.clone())
            .unwrap();
        let artifact = lock_store(&store).artifact_path(target_key);
        let seal = lock_store(&store).seal_path(target_key);
        assert_eq!(std::fs::read(&artifact).unwrap(), b"first artifact");
        assert_eq!(std::fs::read_to_string(&seal).unwrap(), first_digest);
        assert!(!staging.exists());

        // Stage a replacement: the prior artifact stays readable while the
        // complete replacement exists beside it, exactly what the preflight's
        // simultaneous double-count models.
        std::fs::write(&staging, b"second artifact!").unwrap();
        assert_eq!(std::fs::read(&artifact).unwrap(), b"first artifact");
        let second_digest = hash_artifact_bytes(&staging).unwrap();
        lock_store(&store)
            .publish_staged(target_key, 16, second_digest.clone())
            .unwrap();
        assert_eq!(std::fs::read(&artifact).unwrap(), b"second artifact!");
        assert_eq!(std::fs::read_to_string(&seal).unwrap(), second_digest);
        let entries = lock_store(&store).entries_for_preflight();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].artifact_bytes, 16);

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn eviction_follows_the_pure_plan_and_returns_a_path_free_receipt() {
        let root = temp_root("evict");
        let (store, _) = ProxyCacheStore::open(root.clone()).unwrap();
        let store = Mutex::new(store);
        for (digit, bytes) in [('a', 60_u64), ('b', 60)] {
            let entry_key = key(digit);
            let staging = lock_store(&store).staging_path(entry_key);
            std::fs::write(&staging, vec![0_u8; bytes as usize]).unwrap();
            let digest = hash_artifact_bytes(&staging).unwrap();
            lock_store(&store)
                .publish_staged(entry_key, bytes, digest)
                .unwrap();
        }
        let limits = ProxyCacheLimits {
            max_entries: 2,
            max_entry_bytes: 100,
            max_total_bytes: 200,
        };

        // Touch the smaller key so LRU order — not key order — decides.
        let first_key = key('a').min(key('b'));
        let second_key = key('a').max(key('b'));
        lock_store(&store).touch(first_key);

        let incoming = key('d');
        let plan = ProxyCachePlan::preflight(
            incoming,
            100,
            &lock_store(&store).entries_for_preflight(),
            limits,
        )
        .unwrap();
        let receipt = lock_store(&store).evict_for_plan(&plan).unwrap();
        assert_eq!(receipt.evicted, vec![(second_key, 60)]);
        assert_eq!(
            receipt.retained_bytes_before_stage,
            plan.retained_bytes_before_stage
        );
        assert!(!lock_store(&store).artifact_path(second_key).exists());
        assert!(!lock_store(&store).seal_path(second_key).exists());
        assert!(lock_store(&store).artifact_path(first_key).exists());
        assert!(lock_store(&store).contains(first_key));
        assert!(!lock_store(&store).contains(second_key));

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn foreign_files_are_counted_but_never_touched() {
        let root = temp_root("foreign");
        std::fs::write(root.join("README.txt"), b"not a cache entry").unwrap();
        std::fs::write(root.join("deadbeef.mkv"), b"short hex is not a key").unwrap();
        std::fs::create_dir_all(root.join("subdir")).unwrap();

        let (store, recovery) = ProxyCacheStore::open(root.clone()).unwrap();
        assert_eq!(recovery.artifacts, 0);
        assert_eq!(recovery.foreign_entries, 3);
        assert!(root.join("README.txt").exists());
        assert!(root.join("deadbeef.mkv").exists());
        assert!(store.entries_for_preflight().is_empty());

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn the_encode_argv_is_derived_from_the_contract_plan() {
        let source = Path::new("source.mp4");
        let staging = Path::new("staging.mkv.staging");
        let probe = probe_1080p();

        let default_plan = plan_for(probe, ProxySettings::default());
        let args = build_proxy_encode_args(source, staging, probe, &default_plan, 1).join(" ");
        assert!(
            args.contains("-map 0:1"),
            "best video stream is mapped absolutely: {args}"
        );
        assert!(
            args.contains("-vf scale=960:540"),
            "half scale applies the even-floored contract dims: {args}"
        );
        assert!(args.contains("-c:v ffv1"));
        assert!(
            args.contains("-map 0:a:0 -c:a copy"),
            "first ordered stream copied: {args}"
        );
        assert!(args.contains("-map_metadata -1"));
        assert!(args.ends_with(&format!("-f matroska {}", staging.display())));
        assert!(
            !args.contains("-fps_mode"),
            "a scale filter already fixes the graph; passthrough is only for the filterless case"
        );

        let silent = ProxySettings {
            include_audio: false,
            scale: crate::proxy::ProxyScale::Original,
            ..ProxySettings::default()
        };
        let silent_plan = plan_for(probe, silent);
        let silent_args =
            build_proxy_encode_args(source, staging, probe, &silent_plan, 0).join(" ");
        assert!(silent_args.contains("-an"));
        assert!(!silent_args.contains("-c:a"));
        assert!(
            silent_args.contains("-fps_mode passthrough"),
            "original scale plus source timing preserves VFR explicitly: {silent_args}"
        );
        assert!(!silent_args.contains("-vf"));

        let resampled = ProxySettings {
            frame_rate: crate::proxy::ProxyFrameRate::Fixed {
                numerator: 30_000,
                denominator: 1_001,
            },
            ..ProxySettings::default()
        };
        let resampled_plan = plan_for(probe, resampled);
        let resampled_args =
            build_proxy_encode_args(source, staging, probe, &resampled_plan, 0).join(" ");
        assert!(
            resampled_args.contains("-vf fps=30000/1001,scale=960:540"),
            "rate resample precedes the scale in one graph: {resampled_args}"
        );

        let silent_source_probe = ProxySourceProbe {
            container_streams: 1,
            audio_streams: 0,
            ..probe
        };
        let silent_source_plan = plan_for(silent_source_probe, ProxySettings::default());
        assert_eq!(
            silent_source_plan.audio,
            ProxyAudioInputLaw::NoAudioTrack {
                cause: ProxyAudioAbsenceCause::SourceCarriesNoAudioStream,
            }
        );
        let silent_source_args =
            build_proxy_encode_args(source, staging, silent_source_probe, &silent_source_plan, 0)
                .join(" ");
        assert!(silent_source_args.contains("-an"));
    }

    #[test]
    fn garbage_bytes_fail_decoded_identity_validation() {
        let root = temp_root("garbage");
        let artifact = root.join("garbage.bin");
        std::fs::write(&artifact, vec![0xA7_u8; 4_096]).unwrap();
        let plan = plan_for(probe_1080p(), ProxySettings::default());
        assert!(matches!(
            validate_proxy_artifact(&artifact, &plan),
            Err(ProxyWorkerError::InvalidArtifact(_))
        ));
        let empty = root.join("empty.mkv");
        std::fs::write(&empty, b"").unwrap();
        assert!(matches!(
            validate_proxy_artifact(&empty, &plan),
            Err(ProxyWorkerError::InvalidArtifact(_))
        ));
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_job_refuses_a_mutated_or_unreadable_source_before_encoding() {
        let root = temp_root("mutated");
        let (store, _) = ProxyCacheStore::open(root.clone()).unwrap();
        let store = Mutex::new(store);
        let policy = MediaSafetyPolicy::safe();
        let cancel = AtomicBool::new(false);

        let missing = ProxyEncodeJob {
            source_path: root.join("does-not-exist.mp4"),
            identity: identity('a', 64),
            settings: ProxySettings::default(),
        };
        assert!(matches!(
            run_proxy_encode_job(
                &store,
                &missing,
                ProxyCacheLimits::default(),
                &policy,
                &cancel
            ),
            Err(ProxyWorkerError::SourceUnreadable(_))
        ));

        let mutated_path = root.join("mutated.mp4");
        std::fs::write(&mutated_path, b"these are not the verified bytes").unwrap();
        let mutated = ProxyEncodeJob {
            source_path: mutated_path,
            identity: identity('a', 64),
            settings: ProxySettings::default(),
        };
        assert!(matches!(
            run_proxy_encode_job(
                &store,
                &mutated,
                ProxyCacheLimits::default(),
                &policy,
                &cancel
            ),
            Err(ProxyWorkerError::SourceFingerprintMismatch { .. })
        ));

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn consultation_without_an_artifact_is_inert() {
        let root = temp_root("consult-none");
        let (store, _) = ProxyCacheStore::open(root.clone()).unwrap();
        let store = Mutex::new(store);
        assert_eq!(
            consult_proxy_cache(
                &store,
                &identity('a', 64),
                ProxySettings::default(),
                Path::new("unused.mp4"),
            ),
            ProxyConsultation::NoArtifact
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    fn generate_test_source(path: &Path, with_audio: bool) {
        let mut command = std::process::Command::new("ffmpeg");
        command.args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=64x36:rate=10",
        ]);
        if with_audio {
            command.args(["-f", "lavfi", "-i", "sine=frequency=440:duration=1"]);
        }
        command.args(["-c:v", "libx264", "-pix_fmt", "yuv420p"]);
        if with_audio {
            command.args(["-c:a", "aac", "-shortest"]);
        }
        command.arg(path);
        let status = command.status().expect("ffmpeg CLI on PATH");
        assert!(status.success(), "test source generation failed");
    }

    fn fingerprint_of(path: &Path) -> ContentIdentity {
        FingerprintSession::new(FingerprintLimits::default())
            .unwrap()
            .fingerprint(path)
            .unwrap()
    }

    /// The delivery fixture: several discriminating observations from one
    /// warm harness. Requires the ffmpeg CLI on PATH (CI's Unix FFmpeg is
    /// built with --disable-programs, so this is opt-in like effects_audit).
    #[test]
    #[ignore]
    fn proxy_worker_end_to_end_encode_publish_rename_and_corruption_survival() {
        let root = temp_root("e2e");
        let source = root.join("clip.mp4");
        generate_test_source(&source, true);
        let source_identity = fingerprint_of(&source);
        let settings = ProxySettings::default();
        let policy = MediaSafetyPolicy::safe();
        let cancel = AtomicBool::new(false);
        let limits = ProxyCacheLimits::default();

        let cache_root = root.join("cache");
        let (store, _) = ProxyCacheStore::open(cache_root.clone()).unwrap();
        let store = Mutex::new(store);

        // 1. The artifact is published, readable, and validated: half scale
        //    of 64x36 is 32x18, FFV1 in Matroska, first audio stream carried.
        let job = ProxyEncodeJob {
            source_path: source.clone(),
            identity: source_identity.clone(),
            settings,
        };
        let outcome = run_proxy_encode_job(&store, &job, limits, &policy, &cancel).unwrap();
        assert!(!outcome.already_cached);
        let artifact = lock_store(&store).artifact_path(outcome.key);
        assert!(artifact.exists());
        let (probe, _) = probe_proxy_source(&source).unwrap();
        assert_eq!(probe.audio_streams, 1);
        let plan = plan_proxy_input(probe, settings).unwrap();
        assert_eq!((plan.output_width, plan.output_height), (32, 18));
        assert_eq!(
            plan.audio,
            ProxyAudioInputLaw::FirstOrderedStreamBitExactCopy
        );
        validate_proxy_artifact(&artifact, &plan).unwrap();

        // 2. A second run is a cache hit, not a re-encode.
        let hit = run_proxy_encode_job(&store, &job, limits, &policy, &cancel).unwrap();
        assert!(hit.already_cached);
        assert_eq!(hit.key, outcome.key);

        // 3. Identical bytes at a different path hit the same key and adopt.
        let renamed = root.join("renamed and relocated clip.mp4");
        std::fs::copy(&source, &renamed).unwrap();
        let renamed_identity = fingerprint_of(&renamed);
        assert_eq!(renamed_identity, source_identity);
        match consult_proxy_cache(&store, &renamed_identity, settings, &renamed) {
            ProxyConsultation::Adopted { key, artifact_path } => {
                assert_eq!(key, outcome.key);
                assert_eq!(artifact_path, artifact);
            }
            other => panic!("expected adoption, got {other:?}"),
        }

        // 4. A corrupted artifact is refused, discarded, and never served;
        //    the next job re-encodes it fresh.
        let valid_bytes = std::fs::read(&artifact).unwrap();
        let mut corrupt = valid_bytes.clone();
        let mid = corrupt.len() / 2;
        for byte in corrupt.iter_mut().skip(mid).take(512) {
            *byte ^= 0xFF;
        }
        std::fs::write(&artifact, &corrupt).unwrap();
        match consult_proxy_cache(&store, &source_identity, settings, &source) {
            ProxyConsultation::RefusedInvalid { key, .. } => assert_eq!(key, outcome.key),
            other => panic!("expected refusal of the corrupted artifact, got {other:?}"),
        }
        assert!(
            !artifact.exists(),
            "a refused artifact is discarded, not retained"
        );
        let reencoded = run_proxy_encode_job(&store, &job, limits, &policy, &cancel).unwrap();
        assert!(!reencoded.already_cached);
        validate_proxy_artifact(&artifact, &plan).unwrap();

        // 4b. The job's own cache-hit path is sealed too: corrupting the
        //     artifact and submitting the job directly (no consultation
        //     in between) must re-encode, never report a corrupt hit.
        let mut resealed = std::fs::read(&artifact).unwrap();
        let tail = resealed.len().saturating_sub(64);
        for byte in resealed.iter_mut().skip(tail) {
            *byte ^= 0xFF;
        }
        std::fs::write(&artifact, &resealed).unwrap();
        let after_corruption =
            run_proxy_encode_job(&store, &job, limits, &policy, &cancel).unwrap();
        assert!(
            !after_corruption.already_cached,
            "a corrupt artifact must never be reported as already cached"
        );
        validate_proxy_artifact(&artifact, &plan).unwrap();

        // 5. A crash leftover: stage a partial file beside the artifact,
        //    reopen the store, and prove it is removed while the published
        //    artifact survives untouched.
        let staging = lock_store(&store).staging_path(outcome.key);
        std::fs::write(&staging, b"interrupted partial encode").unwrap();
        drop(store);
        let (recovered, recovery) = ProxyCacheStore::open(cache_root).unwrap();
        assert_eq!(recovery.staging_removed, 1);
        assert!(!staging.exists());
        assert!(recovered.contains(outcome.key));
        validate_proxy_artifact(&artifact, &plan).unwrap();
        let store = Mutex::new(recovered);

        // 6. include_audio=false is a different key whose artifact carries no
        //    audio track; a silent source under default settings likewise
        //    yields no audio track as the defined result, not an error.
        let silent_settings = ProxySettings {
            include_audio: false,
            ..settings
        };
        let silent_job = ProxyEncodeJob {
            source_path: source.clone(),
            identity: source_identity.clone(),
            settings: silent_settings,
        };
        let silent_outcome =
            run_proxy_encode_job(&store, &silent_job, limits, &policy, &cancel).unwrap();
        assert_ne!(silent_outcome.key, outcome.key);
        let silent_plan = plan_proxy_input(probe, silent_settings).unwrap();
        validate_proxy_artifact(
            &lock_store(&store).artifact_path(silent_outcome.key),
            &silent_plan,
        )
        .unwrap();

        let silent_source = root.join("silent.mp4");
        generate_test_source(&silent_source, false);
        let silent_source_identity = fingerprint_of(&silent_source);
        let silent_source_job = ProxyEncodeJob {
            source_path: silent_source.clone(),
            identity: silent_source_identity,
            settings,
        };
        let silent_source_outcome =
            run_proxy_encode_job(&store, &silent_source_job, limits, &policy, &cancel).unwrap();
        let (silent_probe, _) = probe_proxy_source(&silent_source).unwrap();
        assert_eq!(silent_probe.audio_streams, 0);
        let silent_source_plan = plan_proxy_input(silent_probe, settings).unwrap();
        assert_eq!(
            silent_source_plan.audio,
            ProxyAudioInputLaw::NoAudioTrack {
                cause: ProxyAudioAbsenceCause::SourceCarriesNoAudioStream,
            }
        );
        validate_proxy_artifact(
            &lock_store(&store).artifact_path(silent_source_outcome.key),
            &silent_source_plan,
        )
        .unwrap();

        std::fs::remove_dir_all(&root).unwrap();
    }

    /// The two kill bounds, opt-in for the same CLI reason: an exhausted
    /// absolute deadline and a staging file over the per-artifact cap both
    /// kill the child and surface typed errors, and neither publishes.
    #[test]
    #[ignore]
    fn proxy_encode_kill_bounds_are_typed_and_publish_nothing() {
        let root = temp_root("kill");
        let source = root.join("clip.mp4");
        generate_test_source(&source, false);
        let identity = fingerprint_of(&source);
        let policy = MediaSafetyPolicy::safe();
        let cancel = AtomicBool::new(false);

        let (store, _) = ProxyCacheStore::open(root.join("cache")).unwrap();
        let store = Mutex::new(store);

        let staging = root.join("deadline.mkv.staging");
        let args = vec![
            "-hide_banner".to_owned(),
            "-loglevel".to_owned(),
            "error".to_owned(),
            "-version".to_owned(),
        ];
        assert!(matches!(
            run_bounded_proxy_encode(&args, &staging, 0, ProxyCacheLimits::default(), &cancel),
            Err(ProxyWorkerError::DeadlineExceeded { seconds: 0 })
        ));

        // The tightest legal per-artifact cap: any real FFV1 output exceeds
        // one byte almost immediately, so the size poll kills the child.
        let tiny = ProxyCacheLimits {
            max_entries: 2,
            max_entry_bytes: 1,
            max_total_bytes: 2,
        };
        let job = ProxyEncodeJob {
            source_path: source,
            identity,
            settings: ProxySettings::default(),
        };
        match run_proxy_encode_job(&store, &job, tiny, &policy, &cancel) {
            Err(ProxyWorkerError::StagingOverCap { limit: 1, .. }) => {}
            other => panic!("expected the staging size cap to kill the encode, got {other:?}"),
        }
        assert!(
            lock_store(&store).entries_for_preflight().is_empty(),
            "a killed encode publishes nothing"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_corrupt_cached_artifact_is_refused_and_discarded_at_consultation() {
        let root = temp_root("consult-corrupt");
        let (store, _) = ProxyCacheStore::open(root.clone()).unwrap();
        let store = Mutex::new(store);
        let source_identity = identity('e', 4_096);
        let cached_key = ProxyCacheKey::derive(&source_identity, ProxySettings::default()).unwrap();
        let staging = lock_store(&store).staging_path(cached_key);
        std::fs::write(&staging, vec![0x5C_u8; 2_048]).unwrap();
        let digest = hash_artifact_bytes(&staging).unwrap();
        lock_store(&store)
            .publish_staged(cached_key, 2_048, digest)
            .unwrap();

        // The probe of the original fails on a non-media file, which is also
        // a refusal — never an adoption of unvalidated bytes.
        let fake_source = root.join("source.mp4");
        std::fs::write(&fake_source, b"not media").unwrap();
        let refused = consult_proxy_cache(
            &store,
            &source_identity,
            ProxySettings::default(),
            &fake_source,
        );
        assert!(matches!(
            refused,
            ProxyConsultation::RefusedInvalid { key, .. } if key == cached_key
        ));

        std::fs::remove_dir_all(&root).unwrap();
    }

    fn adoption_candidate(id: u64, epoch: u64, position: f64) -> ProxyAdoptionCandidate {
        ProxyAdoptionCandidate {
            layer_id: StableLayerId::new(id).unwrap(),
            source_resource_epoch: epoch,
            start_position: position,
        }
    }

    #[test]
    fn hot_adoption_prepares_nothing_without_an_artifact_and_names_the_refusal() {
        let root = temp_root("adopt-none");
        let (store, _) = ProxyCacheStore::open(root.clone()).unwrap();
        let store = Mutex::new(store);
        let policy = MediaSafetyPolicy::safe();
        let cancel = AtomicBool::new(false);

        // No candidates: no consultation, no events.
        let idle = prepare_adoption_job(
            &store,
            &policy,
            ProxyAdoptionJob {
                identity: identity('a', 64),
                settings: ProxySettings::default(),
                original_source: root.join("unused.mp4"),
                device_limits: MediaDeviceLimits::none(),
                candidates: Vec::new(),
            },
            &cancel,
        );
        assert!(idle.is_empty());

        // A candidate against an empty cache is one named refusal, and the
        // refusal never fabricates a per-layer preparation.
        let events = prepare_adoption_job(
            &store,
            &policy,
            ProxyAdoptionJob {
                identity: identity('a', 64),
                settings: ProxySettings::default(),
                original_source: root.join("unused.mp4"),
                device_limits: MediaDeviceLimits::none(),
                candidates: vec![adoption_candidate(7, 3, 0.5)],
            },
            &cancel,
        );
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            ProxyAdoptionEvent::Refused { identity: refused, error }
                if *refused == identity('a', 64)
                    && error.contains("no published artifact")
        ));

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn hot_adoption_refuses_through_the_same_consultation_law_as_patch_load() {
        let root = temp_root("adopt-refuse");
        let (store, _) = ProxyCacheStore::open(root.clone()).unwrap();
        let store = Mutex::new(store);
        let policy = MediaSafetyPolicy::safe();
        let cancel = AtomicBool::new(false);
        let source_identity = identity('b', 4_096);
        let cached_key = ProxyCacheKey::derive(&source_identity, ProxySettings::default()).unwrap();

        // A sealed artifact whose original cannot be re-probed is refused by
        // the shared consultation — hot adoption has no laxer private path.
        let staging = lock_store(&store).staging_path(cached_key);
        std::fs::write(&staging, vec![0x2A_u8; 2_048]).unwrap();
        let digest = hash_artifact_bytes(&staging).unwrap();
        lock_store(&store)
            .publish_staged(cached_key, 2_048, digest)
            .unwrap();
        let fake_source = root.join("source.mp4");
        std::fs::write(&fake_source, b"not media").unwrap();

        let events = prepare_adoption_job(
            &store,
            &policy,
            ProxyAdoptionJob {
                identity: source_identity.clone(),
                settings: ProxySettings::default(),
                original_source: fake_source.clone(),
                device_limits: MediaDeviceLimits::none(),
                candidates: vec![adoption_candidate(1, 1, 0.0), adoption_candidate(2, 1, 0.5)],
            },
            &cancel,
        );
        assert_eq!(
            events.len(),
            1,
            "one job-level refusal, not one per candidate"
        );
        assert!(matches!(
            &events[0],
            ProxyAdoptionEvent::Refused { error, .. }
                if error.contains("refused and discarded")
        ));

        // Post-seal corruption is likewise refused and the artifact is
        // discarded, exactly as consultation at patch load would.
        let staging = lock_store(&store).staging_path(cached_key);
        std::fs::write(&staging, vec![0x2A_u8; 2_048]).unwrap();
        let digest = hash_artifact_bytes(&staging).unwrap();
        lock_store(&store)
            .publish_staged(cached_key, 2_048, digest)
            .unwrap();
        let artifact = lock_store(&store).artifact_path(cached_key);
        let mut bytes = std::fs::read(&artifact).unwrap();
        for byte in bytes.iter_mut().skip(1_000).take(32) {
            *byte ^= 0xFF;
        }
        std::fs::write(&artifact, &bytes).unwrap();
        let events = prepare_adoption_job(
            &store,
            &policy,
            ProxyAdoptionJob {
                identity: source_identity.clone(),
                settings: ProxySettings::default(),
                original_source: fake_source,
                device_limits: MediaDeviceLimits::none(),
                candidates: vec![adoption_candidate(1, 1, 0.0)],
            },
            &cancel,
        );
        assert!(matches!(
            &events[0],
            ProxyAdoptionEvent::Refused { error, .. } if error.contains("seal")
        ));
        assert!(
            !artifact.exists(),
            "a refused artifact is discarded, not retained"
        );

        std::fs::remove_dir_all(&root).unwrap();
    }

    /// The adoption delivery fixture. Requires the ffmpeg CLI on PATH
    /// (CI's Unix FFmpeg is built with --disable-programs, so this is
    /// opt-in like effects_audit).
    #[test]
    #[ignore]
    fn proxy_hot_adoption_prepares_playhead_seeded_decoders_end_to_end() {
        let root = temp_root("adopt-e2e");
        let source = root.join("clip.mp4");
        generate_test_source(&source, false);
        let source_identity = fingerprint_of(&source);
        let settings = ProxySettings::default();
        let policy = MediaSafetyPolicy::safe();
        let cancel = AtomicBool::new(false);
        let (store, _) = ProxyCacheStore::open(root.join("cache")).unwrap();
        let store = Mutex::new(store);
        let job = ProxyEncodeJob {
            source_path: source.clone(),
            identity: source_identity.clone(),
            settings,
        };
        let outcome =
            run_proxy_encode_job(&store, &job, ProxyCacheLimits::default(), &policy, &cancel)
                .unwrap();

        // Two candidates at different playheads: each gets its own decoder,
        // seeded where its audience is watching, at the artifact's half
        // scale, with the candidate's claim passed through for the render
        // thread's staleness guards.
        let events = prepare_adoption_job(
            &store,
            &policy,
            ProxyAdoptionJob {
                identity: source_identity.clone(),
                settings,
                original_source: source.clone(),
                device_limits: MediaDeviceLimits::none(),
                candidates: vec![
                    adoption_candidate(11, 4, 0.0),
                    adoption_candidate(12, 9, 0.5),
                ],
            },
            &cancel,
        );
        assert_eq!(events.len(), 2);
        let mut seeds = Vec::new();
        for (index, event) in events.into_iter().enumerate() {
            match event {
                ProxyAdoptionEvent::Prepared {
                    identity,
                    key,
                    candidate,
                    width,
                    height,
                    first_rgba,
                    preload_bytes,
                    ..
                } => {
                    assert_eq!(identity, source_identity);
                    assert_eq!(key, outcome.key);
                    assert_eq!((width, height), (32, 18));
                    assert_eq!(first_rgba.len(), 32 * 18 * 4);
                    assert!(preload_bytes > 0);
                    let expected = if index == 0 {
                        adoption_candidate(11, 4, 0.0)
                    } else {
                        adoption_candidate(12, 9, 0.5)
                    };
                    assert_eq!(candidate, expected);
                    seeds.push(first_rgba);
                }
                ProxyAdoptionEvent::Refused { error, .. } => {
                    panic!("expected preparation, got refusal: {error}")
                }
            }
        }
        assert_ne!(
            seeds[0], seeds[1],
            "a mid-clip playhead must seed a different frame than the start"
        );

        std::fs::remove_dir_all(&root).unwrap();
    }

    /// The physical claim: a prepared adoption installs into a live layer
    /// through the infallible field swap — identity, filename, and playhead
    /// kept; decoder, texture, and dimensions moved. Requires the ffmpeg CLI
    /// and a GPU adapter.
    #[test]
    #[ignore]
    fn gpu_proxy_hot_adoption_swaps_a_live_layer_and_keeps_identity_and_playhead() {
        use crate::layers::{Layer, LayerSourceActivation};
        use crate::transport::{ClipTransportState, NormalizedTime, PlaybackDirection};

        let root = temp_root("adopt-gpu");
        let source = root.join("clip.mp4");
        generate_test_source(&source, false);
        let source_identity = fingerprint_of(&source);
        let settings = ProxySettings::default();
        let policy = MediaSafetyPolicy::safe();
        let cancel = AtomicBool::new(false);
        let (store, _) = ProxyCacheStore::open(root.join("cache")).unwrap();
        let store = Mutex::new(store);
        run_proxy_encode_job(
            &store,
            &ProxyEncodeJob {
                source_path: source.clone(),
                identity: source_identity.clone(),
                settings,
            },
            ProxyCacheLimits::default(),
            &policy,
            &cancel,
        )
        .unwrap();

        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("GPU adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Proxy Hot Adoption Test"),
            ..Default::default()
        }))
        .expect("GPU device");

        let mut layer =
            Layer::new_with_media_policy(source.to_str().unwrap(), &device, &policy).unwrap();
        layer.set_persisted_source_reference(Some(source_identity.source_reference()));
        layer.clip_transport =
            ClipTransportState::at(NormalizedTime::clamped(0.5), PlaybackDirection::Forward);
        let authored_filename = layer.filename.clone();
        let authored_reference = layer.source_reference_for_persistence().to_owned();
        let prior_epoch = layer.source_resource_epoch();
        assert_eq!((layer.width, layer.height), (64, 36));
        assert!(layer.proxy_backing().is_none());

        let candidate = ProxyAdoptionCandidate {
            layer_id: layer.stable_layer_id(),
            source_resource_epoch: prior_epoch,
            start_position: layer.clip_transport.position.get(),
        };
        let events = prepare_adoption_job(
            &store,
            &policy,
            ProxyAdoptionJob {
                identity: source_identity.clone(),
                settings,
                original_source: source.clone(),
                device_limits: MediaDeviceLimits::none(),
                candidates: vec![candidate],
            },
            &cancel,
        );
        assert_eq!(events.len(), 1);
        let ProxyAdoptionEvent::Prepared {
            key,
            artifact_path,
            decoder,
            first_rgba,
            width,
            height,
            source_fps,
            preload_bytes,
            ..
        } = events.into_iter().next().unwrap()
        else {
            panic!("expected a prepared adoption");
        };

        let activation = LayerSourceActivation::stage(
            &device,
            &queue,
            artifact_path.to_string_lossy().into_owned(),
            Some(authored_reference.clone()),
            authored_filename.clone(),
            crate::layers::LayerSource::Video(*decoder),
            width,
            height,
            source_fps,
            preload_bytes,
            &first_rgba,
        )
        .unwrap();
        let displaced = layer.commit_adopted_proxy(activation, key.to_hex());
        drop(displaced);

        assert_eq!(layer.proxy_backing(), Some(key.to_hex().as_str()));
        assert_eq!((layer.width, layer.height), (32, 18));
        assert_eq!(layer.filename, authored_filename);
        assert_eq!(layer.source_reference_for_persistence(), authored_reference);
        assert_eq!(
            layer.source_path,
            artifact_path.to_string_lossy().into_owned(),
            "the runtime open path is the only proxy fact"
        );
        assert!(
            layer.clip_transport.position == NormalizedTime::clamped(0.5),
            "the audience playhead survives the swap"
        );
        assert!(
            layer.source_resource_epoch() != prior_epoch,
            "an adoption is a source-resource change and must advance the epoch"
        );

        // Decoder drop signals its worker without joining, so the artifact
        // handle closes asynchronously; Windows refuses to remove a
        // directory holding an open file. Bound the wait rather than racing.
        drop(layer);
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match std::fs::remove_dir_all(&root) {
                Ok(()) => break,
                Err(_) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(error) => panic!("temp root cleanup failed: {error}"),
            }
        }
    }
}
