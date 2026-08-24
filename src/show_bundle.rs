//! Deterministic, versioned `.cosbundle` build/inspect/import core.
//!
//! The format is deliberately uncompressed. That makes the expansion ratio
//! exactly one, permits bounded streaming verification, and leaves no archive
//! entry type with which a hostile bundle could encode a symlink or device.
//! A complete imported show is published as one no-replace directory
//! generation, so no partial library becomes visible.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::durable_file::{
    publish_directory_noreplace, sync_directory, PublishMode, StagedPublication,
};
use crate::media_source::{parse_content_reference, ContentIdentity};
use crate::patch::PatchState;

pub(crate) const COSBUNDLE_FORMAT_VERSION: u16 = 1;
const COSBUNDLE_MAGIC: &[u8; 8] = b"COSBNDL\0";
const COSBUNDLE_HEADER_BYTES: usize = 60;
const COSBUNDLE_FLAGS: u16 = 0;
const STREAM_BUFFER_BYTES: usize = 1024 * 1024;
const IMPORT_STAGE_PREFIX: &str = ".cosbundle-import-stage-";
const IMPORT_GENERATION_PREFIX: &str = ".cosshow-";
const IMPORT_IDENTITY_FILE: &str = ".cosbundle.identity";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BundleLimits {
    pub max_bundle_bytes: u64,
    pub max_manifest_bytes: usize,
    pub max_entries: usize,
    pub max_entry_bytes: u64,
    pub max_document_bytes: usize,
    pub max_expanded_bytes: u64,
    pub max_path_depth: usize,
    pub max_name_bytes: usize,
}

impl Default for BundleLimits {
    fn default() -> Self {
        Self {
            max_bundle_bytes: 64 * 1024 * 1024 * 1024,
            max_manifest_bytes: 4 * 1024 * 1024,
            max_entries: 4096,
            max_entry_bytes: 64 * 1024 * 1024 * 1024,
            max_document_bytes: crate::patch::editor::MAX_PATCH_FILE_BYTES,
            max_expanded_bytes: 64 * 1024 * 1024 * 1024,
            max_path_depth: 4,
            max_name_bytes: 240,
        }
    }
}

impl BundleLimits {
    fn validate(self) -> Result<Self, BundleError> {
        if self.max_bundle_bytes < COSBUNDLE_HEADER_BYTES as u64
            || self.max_manifest_bytes == 0
            || self.max_entries == 0
            || self.max_entry_bytes == 0
            || self.max_document_bytes == 0
            || self.max_expanded_bytes == 0
            || self.max_path_depth == 0
            || self.max_name_bytes == 0
        {
            return Err(BundleError::Invalid(
                "bundle limits must all admit at least one header, entry, and byte".to_owned(),
            ));
        }
        Ok(self)
    }
}

#[derive(Debug)]
pub(crate) enum BundleError {
    Invalid(String),
    Io(String),
    Collision(String),
    Cancelled,
}

impl fmt::Display for BundleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(detail) => write!(formatter, "invalid .cosbundle: {detail}"),
            Self::Io(detail) => formatter.write_str(detail),
            Self::Collision(detail) => write!(formatter, "bundle collision: {detail}"),
            Self::Cancelled => formatter.write_str("bundle operation cancelled"),
        }
    }
}

impl std::error::Error for BundleError {}

impl From<io::Error> for BundleError {
    fn from(error: io::Error) -> Self {
        Self::Io(error.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BundleMediaRole {
    Original,
    Proxy { original_sha256: String },
}

#[derive(Debug, Clone)]
pub(crate) struct BundleMediaInput {
    pub source: PathBuf,
    pub logical_name: String,
    /// Every exact path/name spelling in the captured patch that this original
    /// replaces with one `cos-sha256://` identity. Proxies must leave this empty.
    pub patch_references: Vec<String>,
    pub expected_identity: Option<ContentIdentity>,
    pub license: Option<String>,
    pub role: BundleMediaRole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BundleDocumentKind {
    ControllerProfile,
    VenueProfile,
    Receipt,
}

#[derive(Debug, Clone)]
pub(crate) struct BundleDocumentInput {
    pub kind: BundleDocumentKind,
    pub logical_name: String,
    pub bytes: Vec<u8>,
    pub license: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BundleOutputCollision {
    Fail,
    Replace,
}

#[derive(Clone)]
pub(crate) struct BundleBuildRequest {
    pub patch: PatchState,
    pub media: Vec<BundleMediaInput>,
    pub documents: Vec<BundleDocumentInput>,
    pub output_collision: BundleOutputCollision,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BundleBuildReceipt {
    pub bundle_sha256: String,
    pub patch_sha256: String,
    pub byte_len: u64,
    pub entry_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum BundleEntryKind {
    Patch,
    OriginalMedia,
    Study,
    Gesture,
    PerformanceTake,
    ControllerProfile,
    VenueProfile,
    Receipt,
    Proxy,
}

impl BundleEntryKind {
    fn is_document(self) -> bool {
        !matches!(self, Self::OriginalMedia | Self::Proxy)
    }

    fn is_authoritative(self) -> bool {
        !matches!(self, Self::Proxy | Self::Receipt)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BundleManifestEntry {
    path: String,
    kind: BundleEntryKind,
    logical_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    license: Option<String>,
    sha256: String,
    byte_len: u64,
    stored_len: u64,
    offset: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    original_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BundleManifest {
    schema_version: u16,
    patch_sha256: String,
    entries: Vec<BundleManifestEntry>,
}

#[derive(Debug)]
enum PreparedContent {
    Memory(Vec<u8>),
    File(PreparedFile),
}

#[derive(Debug)]
struct PreparedFile {
    path: PathBuf,
    byte_len: u64,
    modified: Option<SystemTime>,
    sha256: String,
}

#[derive(Debug)]
struct PreparedEntry {
    manifest: BundleManifestEntry,
    content: PreparedContent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BundleFaultPhase {
    BuildAfterPreflight,
    BuildWrite,
    BuildBeforePublish,
    InspectRead,
    ImportWrite,
    ImportBeforePublish,
}

trait BundleFaultInjector {
    fn check(&self, phase: BundleFaultPhase, progress: u64) -> io::Result<()>;
}

impl<F> BundleFaultInjector for F
where
    F: Fn(BundleFaultPhase, u64) -> io::Result<()>,
{
    fn check(&self, phase: BundleFaultPhase, progress: u64) -> io::Result<()> {
        self(phase, progress)
    }
}

struct NoBundleFaults;

impl BundleFaultInjector for NoBundleFaults {
    fn check(&self, _phase: BundleFaultPhase, _progress: u64) -> io::Result<()> {
        Ok(())
    }
}

pub(crate) fn build_show_bundle(
    destination: &Path,
    request: BundleBuildRequest,
    limits: BundleLimits,
    cancel: &AtomicBool,
) -> Result<BundleBuildReceipt, BundleError> {
    build_show_bundle_with_faults(destination, request, limits, cancel, &NoBundleFaults)
}

fn build_show_bundle_with_faults(
    destination: &Path,
    mut request: BundleBuildRequest,
    limits: BundleLimits,
    cancel: &AtomicBool,
    faults: &dyn BundleFaultInjector,
) -> Result<BundleBuildReceipt, BundleError> {
    let limits = limits.validate()?;
    check_cancelled(cancel)?;
    if destination
        .extension()
        .and_then(|extension| extension.to_str())
        != Some("cosbundle")
    {
        return Err(BundleError::Invalid(
            "bundle destination must use the .cosbundle extension".to_owned(),
        ));
    }
    let planned_entries = 1_usize
        .checked_add(request.media.len())
        .and_then(|count| count.checked_add(request.documents.len()))
        .and_then(|count| count.checked_add(request.patch.studies.len()))
        .and_then(|count| count.checked_add(usize::from(request.patch.gesture_track.is_some())))
        .and_then(|count| count.checked_add(usize::from(request.patch.performance_take.is_some())))
        .ok_or_else(|| BundleError::Invalid("planned entry count overflows".to_owned()))?;
    if planned_entries > limits.max_entries {
        return Err(BundleError::Invalid(format!(
            "planned {planned_entries} entries exceed the {}-entry limit",
            limits.max_entries
        )));
    }
    let mut prepared = prepare_media(&request.media, limits, cancel)?;
    let identities = original_reference_map(&request.media, &prepared)?;
    rewrite_patch_to_content_identities(&mut request.patch, &identities)?;

    let original_identities = prepared
        .iter()
        .filter(|entry| entry.manifest.kind == BundleEntryKind::OriginalMedia)
        .map(|entry| (entry.manifest.sha256.clone(), entry.manifest.byte_len))
        .collect::<BTreeMap<_, _>>();
    validate_patch_original_coverage(&request.patch, &original_identities)?;

    let patch_bytes = canonical_patch_bytes(&request.patch, limits)?;
    reject_secret_markers(&patch_bytes, "canonical patch")?;
    let patch_sha256 = sha256_hex(&patch_bytes);
    prepared.push(memory_entry(
        "patch/show.yaml".to_owned(),
        BundleEntryKind::Patch,
        "show.cos.yaml".to_owned(),
        None,
        patch_bytes,
        None,
    )?);

    prepare_embedded_documents(&request.patch, &mut prepared, limits)?;
    prepare_selected_documents(&request.documents, &mut prepared, limits)?;
    prepared.sort_by(|left, right| left.manifest.path.cmp(&right.manifest.path));
    if prepared.len() > limits.max_entries {
        return Err(BundleError::Invalid(format!(
            "{} entries exceed the {}-entry limit",
            prepared.len(),
            limits.max_entries
        )));
    }

    let mut offset = 0_u64;
    for entry in &mut prepared {
        entry.manifest.offset = offset;
        offset = offset
            .checked_add(entry.manifest.stored_len)
            .ok_or_else(|| BundleError::Invalid("payload length overflows u64".to_owned()))?;
    }
    let manifest = BundleManifest {
        schema_version: COSBUNDLE_FORMAT_VERSION,
        patch_sha256: patch_sha256.clone(),
        entries: prepared
            .iter()
            .map(|entry| entry.manifest.clone())
            .collect(),
    };
    validate_manifest(&manifest, offset, limits)?;
    let manifest_bytes = serde_json::to_vec(&manifest)
        .map_err(|error| BundleError::Invalid(format!("serialize manifest: {error}")))?;
    if manifest_bytes.len() > limits.max_manifest_bytes {
        return Err(BundleError::Invalid(format!(
            "manifest is {} bytes; limit is {}",
            manifest_bytes.len(),
            limits.max_manifest_bytes
        )));
    }
    let total_len = (COSBUNDLE_HEADER_BYTES as u64)
        .checked_add(u64::try_from(manifest_bytes.len()).unwrap_or(u64::MAX))
        .and_then(|bytes| bytes.checked_add(offset))
        .ok_or_else(|| BundleError::Invalid("bundle length overflows u64".to_owned()))?;
    if total_len > limits.max_bundle_bytes {
        return Err(BundleError::Invalid(format!(
            "bundle is {total_len} bytes; limit is {}",
            limits.max_bundle_bytes
        )));
    }
    faults.check(BundleFaultPhase::BuildAfterPreflight, 0)?;
    check_cancelled(cancel)?;

    let manifest_digest: [u8; 32] = Sha256::digest(&manifest_bytes).into();
    let header = encode_header(
        u32::try_from(manifest_bytes.len())
            .map_err(|_| BundleError::Invalid("manifest length does not fit u32".to_owned()))?,
        u32::try_from(prepared.len())
            .map_err(|_| BundleError::Invalid("entry count does not fit u32".to_owned()))?,
        offset,
        manifest_digest,
    );
    let (publication, mut output) =
        StagedPublication::create(destination, "cosbundle-stage").map_err(BundleError::from)?;
    let mut bundle_hasher = Sha256::new();
    let mut written = 0_u64;
    write_build_chunk(
        &mut output,
        &header,
        &mut bundle_hasher,
        &mut written,
        cancel,
        faults,
    )?;
    write_build_chunk(
        &mut output,
        &manifest_bytes,
        &mut bundle_hasher,
        &mut written,
        cancel,
        faults,
    )?;
    for entry in &prepared {
        match &entry.content {
            PreparedContent::Memory(bytes) => write_build_chunk(
                &mut output,
                bytes,
                &mut bundle_hasher,
                &mut written,
                cancel,
                faults,
            )?,
            PreparedContent::File(file) => copy_prepared_file(
                file,
                &mut output,
                &mut bundle_hasher,
                &mut written,
                cancel,
                faults,
            )?,
        }
    }
    if written != total_len {
        return Err(BundleError::Invalid(format!(
            "builder wrote {written} bytes; planned {total_len}"
        )));
    }
    let mode = match request.output_collision {
        BundleOutputCollision::Fail => PublishMode::NoReplace,
        BundleOutputCollision::Replace => PublishMode::Replace,
    };
    publication
        .commit_if(output, mode, || {
            check_cancelled_io(cancel)?;
            faults.check(BundleFaultPhase::BuildBeforePublish, written)
        })
        .map_err(|error| {
            if error.kind() == io::ErrorKind::AlreadyExists {
                BundleError::Collision("destination already exists".to_owned())
            } else {
                BundleError::from(error)
            }
        })?;
    Ok(BundleBuildReceipt {
        bundle_sha256: format!("{:x}", bundle_hasher.finalize()),
        patch_sha256,
        byte_len: written,
        entry_count: prepared.len(),
    })
}

fn prepare_media(
    inputs: &[BundleMediaInput],
    limits: BundleLimits,
    cancel: &AtomicBool,
) -> Result<Vec<PreparedEntry>, BundleError> {
    let mut declared_bytes = 0_u64;
    let mut logical_names = BTreeSet::new();
    for input in inputs {
        validate_logical_name(&input.logical_name, limits)?;
        if !logical_names.insert(portable_fold(&input.logical_name)) {
            return Err(BundleError::Invalid(format!(
                "duplicate or case-fold-colliding media name {}",
                input.logical_name
            )));
        }
        validate_optional_text(input.license.as_deref(), "license", 8 * 1024)?;
        let metadata = fs::symlink_metadata(&input.source).map_err(|error| {
            BundleError::Io(format!("inspect {}: {error}", input.source.display()))
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(BundleError::Invalid(format!(
                "source {} is not a regular no-follow file",
                input.source.display()
            )));
        }
        declared_bytes = declared_bytes
            .checked_add(metadata.len())
            .ok_or_else(|| BundleError::Invalid("media byte total overflows".to_owned()))?;
        if declared_bytes > limits.max_expanded_bytes {
            return Err(BundleError::Invalid(format!(
                "media bytes {declared_bytes} exceed expanded-byte limit {}",
                limits.max_expanded_bytes
            )));
        }
    }
    let mut prepared = Vec::with_capacity(inputs.len());
    let mut original_digests = BTreeSet::new();
    for input in inputs {
        check_cancelled(cancel)?;
        validate_logical_name(&input.logical_name, limits)?;
        validate_optional_text(input.license.as_deref(), "license", 8 * 1024)?;
        match &input.role {
            BundleMediaRole::Original if input.patch_references.is_empty() => {
                return Err(BundleError::Invalid(format!(
                    "original '{}' has no captured patch reference",
                    input.logical_name
                )));
            }
            BundleMediaRole::Proxy { .. } if !input.patch_references.is_empty() => {
                return Err(BundleError::Invalid(format!(
                    "proxy '{}' may not replace an authored patch reference",
                    input.logical_name
                )));
            }
            _ => {}
        }
        let file = fingerprint_regular_file(&input.source, limits.max_entry_bytes, cancel)?;
        if let Some(expected) = &input.expected_identity {
            if expected.sha256 != file.sha256 || expected.byte_len != file.byte_len {
                return Err(BundleError::Invalid(format!(
                    "{} does not match its expected SHA-256 identity",
                    input.logical_name
                )));
            }
        }
        let (kind, path, original_sha256) = match &input.role {
            BundleMediaRole::Original => {
                if !original_digests.insert(file.sha256.clone()) {
                    return Err(BundleError::Invalid(format!(
                        "duplicate original media identity {}",
                        file.sha256
                    )));
                }
                (
                    BundleEntryKind::OriginalMedia,
                    format!("media/{}", file.sha256),
                    None,
                )
            }
            BundleMediaRole::Proxy { original_sha256 } => {
                validate_sha256(original_sha256, "proxy original digest")?;
                (
                    BundleEntryKind::Proxy,
                    format!("proxy/{}", file.sha256),
                    Some(original_sha256.to_ascii_lowercase()),
                )
            }
        };
        prepared.push(PreparedEntry {
            manifest: BundleManifestEntry {
                path,
                kind,
                logical_name: input.logical_name.clone(),
                license: input.license.clone(),
                sha256: file.sha256.clone(),
                byte_len: file.byte_len,
                stored_len: file.byte_len,
                offset: 0,
                original_sha256,
            },
            content: PreparedContent::File(file),
        });
    }
    Ok(prepared)
}

fn original_reference_map(
    inputs: &[BundleMediaInput],
    prepared: &[PreparedEntry],
) -> Result<BTreeMap<String, ContentIdentity>, BundleError> {
    let mut identities = BTreeMap::new();
    let originals = inputs
        .iter()
        .filter(|input| matches!(input.role, BundleMediaRole::Original));
    let prepared_originals = prepared
        .iter()
        .filter(|entry| entry.manifest.kind == BundleEntryKind::OriginalMedia);
    for (input, entry) in originals.zip(prepared_originals) {
        let identity = ContentIdentity::new(entry.manifest.sha256.clone(), entry.manifest.byte_len)
            .map_err(|error| BundleError::Invalid(error.to_string()))?;
        for reference in &input.patch_references {
            if reference.is_empty() {
                return Err(BundleError::Invalid(format!(
                    "{} contains an empty patch reference",
                    input.logical_name
                )));
            }
            if identities
                .insert(reference.clone(), identity.clone())
                .is_some()
            {
                return Err(BundleError::Invalid(format!(
                    "duplicate captured patch reference '{reference}'"
                )));
            }
        }
    }
    Ok(identities)
}

fn rewrite_patch_to_content_identities(
    patch: &mut PatchState,
    identities: &BTreeMap<String, ContentIdentity>,
) -> Result<(), BundleError> {
    for layer in &mut patch.layers {
        rewrite_source_reference(&mut layer.source_path, &layer.filename, identities)?;
        let slot_ids = layer
            .clip_slots
            .iter()
            .map(|slot| slot.id)
            .collect::<Vec<_>>();
        for slot_id in slot_ids {
            let slot = layer
                .clip_slots
                .get_mut(slot_id)
                .expect("captured slot ID remains present while rewriting");
            rewrite_source_reference(&mut slot.source_path, &slot.filename, identities)?;
        }
    }
    if let Some(modulation) = &mut patch.modulation {
        if crate::modulation::normalize_audio_source_kind(&modulation.audio_source_kind)
            == crate::modulation::AUDIO_SOURCE_FILE
            && !modulation.audio_clip_path.is_empty()
        {
            rewrite_source_reference(&mut modulation.audio_clip_path, "audio clip", identities)?;
        }
    }
    Ok(())
}

fn rewrite_source_reference(
    source_path: &mut String,
    logical_name: &str,
    identities: &BTreeMap<String, ContentIdentity>,
) -> Result<(), BundleError> {
    if is_self_contained_source(source_path) {
        return Ok(());
    }
    if parse_content_reference(source_path)
        .map_err(|error| BundleError::Invalid(error.to_string()))?
        .is_some()
    {
        return Ok(());
    }
    let lookup = if source_path.is_empty() {
        logical_name
    } else {
        source_path.as_str()
    };
    let identity = identities.get(lookup).ok_or_else(|| {
        BundleError::Invalid(format!(
            "missing original media for authored source '{logical_name}' ({lookup})"
        ))
    })?;
    *source_path = identity.source_reference();
    Ok(())
}

fn is_self_contained_source(source: &str) -> bool {
    source == crate::layers::PATTERN_SOURCE_PATH
        || source == crate::layers::TEXT_PAGE_SOURCE_PATH
        || crate::layers::spout_sender_from_source_path(source).is_some()
}

fn validate_patch_original_coverage(
    patch: &PatchState,
    originals: &BTreeMap<String, u64>,
) -> Result<(), BundleError> {
    let mut required = BTreeMap::new();
    for layer in &patch.layers {
        note_required_source(&layer.source_path, &mut required)?;
        for slot in layer.clip_slots.iter() {
            note_required_source(&slot.source_path, &mut required)?;
        }
    }
    if let Some(modulation) = &patch.modulation {
        if crate::modulation::normalize_audio_source_kind(&modulation.audio_source_kind)
            == crate::modulation::AUDIO_SOURCE_FILE
        {
            note_required_source(&modulation.audio_clip_path, &mut required)?;
        }
    }
    for (digest, bytes) in required {
        match originals.get(&digest) {
            Some(observed) if *observed == bytes => {}
            Some(observed) => {
                return Err(BundleError::Invalid(format!(
                    "original {digest} has {observed} bytes but patch requires {bytes}"
                )));
            }
            None => {
                return Err(BundleError::Invalid(format!(
                    "patch requires missing original SHA-256 {digest}"
                )));
            }
        }
    }
    Ok(())
}

fn validate_embedded_sidecar_coverage(
    patch: &PatchState,
    manifest: &BundleManifest,
) -> Result<(), BundleError> {
    let observed = |kind| {
        manifest
            .entries
            .iter()
            .filter(|entry| entry.kind == kind)
            .map(|entry| entry.sha256.clone())
            .collect::<BTreeSet<_>>()
    };
    let studies = patch
        .studies
        .iter()
        .map(|study| {
            study
                .to_json_bytes()
                .map(|bytes| sha256_hex(&bytes))
                .map_err(|error| BundleError::Invalid(format!("canonical Study: {error}")))
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    if studies != observed(BundleEntryKind::Study) {
        return Err(BundleError::Invalid(
            "Study sidecars do not exactly match the canonical patch documents".to_owned(),
        ));
    }
    let gesture = patch
        .gesture_track
        .as_ref()
        .map(|document| {
            document
                .to_json_bytes()
                .map(|bytes| BTreeSet::from([sha256_hex(&bytes)]))
                .map_err(|error| BundleError::Invalid(format!("canonical gesture: {error}")))
        })
        .transpose()?
        .unwrap_or_default();
    if gesture != observed(BundleEntryKind::Gesture) {
        return Err(BundleError::Invalid(
            "gesture sidecar does not exactly match the canonical patch document".to_owned(),
        ));
    }
    let take = patch
        .performance_take
        .as_ref()
        .map(|document| {
            document
                .to_json_bytes()
                .map(|bytes| BTreeSet::from([sha256_hex(&bytes)]))
                .map_err(|error| BundleError::Invalid(format!("canonical take: {error}")))
        })
        .transpose()?
        .unwrap_or_default();
    if take != observed(BundleEntryKind::PerformanceTake) {
        return Err(BundleError::Invalid(
            "performance-take sidecar does not exactly match the canonical patch document"
                .to_owned(),
        ));
    }
    Ok(())
}

fn note_required_source(
    source: &str,
    required: &mut BTreeMap<String, u64>,
) -> Result<(), BundleError> {
    if source.is_empty() || is_self_contained_source(source) {
        return Ok(());
    }
    let identity = parse_content_reference(source)
        .map_err(|error| BundleError::Invalid(error.to_string()))?
        .ok_or_else(|| {
            BundleError::Invalid(format!(
                "portable patch retained a host path instead of a content identity: {source}"
            ))
        })?;
    if let Some(previous) = required.insert(identity.sha256.clone(), identity.byte_len) {
        if previous != identity.byte_len {
            return Err(BundleError::Invalid(format!(
                "patch assigns conflicting lengths to SHA-256 {}",
                identity.sha256
            )));
        }
    }
    Ok(())
}

fn canonical_patch_bytes(patch: &PatchState, limits: BundleLimits) -> Result<Vec<u8>, BundleError> {
    let bytes = serde_yaml::to_string(patch)
        .map_err(|error| BundleError::Invalid(format!("serialize patch: {error}")))?
        .into_bytes();
    if bytes.len() > limits.max_document_bytes {
        return Err(BundleError::Invalid(format!(
            "canonical patch is {} bytes; document limit is {}",
            bytes.len(),
            limits.max_document_bytes
        )));
    }
    let round_trip = crate::patch::editor::parse_patch_bytes(&bytes)
        .map_err(|error| BundleError::Invalid(format!("round-trip patch: {error}")))?;
    let round_trip_bytes = serde_yaml::to_string(&round_trip)
        .map_err(|error| BundleError::Invalid(format!("re-serialize patch: {error}")))?
        .into_bytes();
    if round_trip_bytes != bytes {
        return Err(BundleError::Invalid(
            "canonical patch bytes change across the hostile round trip".to_owned(),
        ));
    }
    Ok(bytes)
}

fn prepare_embedded_documents(
    patch: &PatchState,
    prepared: &mut Vec<PreparedEntry>,
    limits: BundleLimits,
) -> Result<(), BundleError> {
    for study in &patch.studies {
        let bytes = study
            .to_json_bytes()
            .map_err(|error| BundleError::Invalid(format!("serialize Study: {error}")))?;
        crate::study::StudyDocument::from_json_bytes(&bytes)
            .map_err(|error| BundleError::Invalid(format!("round-trip Study: {error}")))?;
        push_bounded_memory_document(prepared, BundleEntryKind::Study, "study", bytes, limits)?;
    }
    if let Some(gesture) = &patch.gesture_track {
        let bytes = gesture
            .to_json_bytes()
            .map_err(|error| BundleError::Invalid(format!("serialize gesture: {error}")))?;
        crate::gesture::GestureTrackDocument::from_json_bytes(&bytes)
            .map_err(|error| BundleError::Invalid(format!("round-trip gesture: {error}")))?;
        push_bounded_memory_document(prepared, BundleEntryKind::Gesture, "gesture", bytes, limits)?;
    }
    if let Some(take) = &patch.performance_take {
        let bytes = take
            .to_json_bytes()
            .map_err(|error| BundleError::Invalid(format!("serialize take: {error}")))?;
        crate::performance_track::PerformanceTakeDocument::from_json_bytes(&bytes)
            .map_err(|error| BundleError::Invalid(format!("round-trip take: {error}")))?;
        push_bounded_memory_document(
            prepared,
            BundleEntryKind::PerformanceTake,
            "performance-take",
            bytes,
            limits,
        )?;
    }
    Ok(())
}

fn push_bounded_memory_document(
    prepared: &mut Vec<PreparedEntry>,
    kind: BundleEntryKind,
    label: &str,
    bytes: Vec<u8>,
    limits: BundleLimits,
) -> Result<(), BundleError> {
    if bytes.len() > limits.max_document_bytes {
        return Err(BundleError::Invalid(format!(
            "{label} is {} bytes; document limit is {}",
            bytes.len(),
            limits.max_document_bytes
        )));
    }
    reject_secret_markers(&bytes, label)?;
    let digest = sha256_hex(&bytes);
    prepared.push(memory_entry(
        format!("{label}/{digest}.json"),
        kind,
        format!("{digest}.json"),
        None,
        bytes,
        None,
    )?);
    Ok(())
}

fn prepare_selected_documents(
    documents: &[BundleDocumentInput],
    prepared: &mut Vec<PreparedEntry>,
    limits: BundleLimits,
) -> Result<(), BundleError> {
    for document in documents {
        validate_logical_name(&document.logical_name, limits)?;
        validate_optional_text(document.license.as_deref(), "license", 8 * 1024)?;
        if document.bytes.len() > limits.max_document_bytes {
            return Err(BundleError::Invalid(format!(
                "{} is {} bytes; document limit is {}",
                document.logical_name,
                document.bytes.len(),
                limits.max_document_bytes
            )));
        }
        std::str::from_utf8(&document.bytes).map_err(|_| {
            BundleError::Invalid(format!(
                "selected document '{}' is not UTF-8",
                document.logical_name
            ))
        })?;
        reject_secret_markers(&document.bytes, &document.logical_name)?;
        let (kind, prefix) = match document.kind {
            BundleDocumentKind::ControllerProfile => {
                (BundleEntryKind::ControllerProfile, "profile/controller")
            }
            BundleDocumentKind::VenueProfile => (BundleEntryKind::VenueProfile, "profile/venue"),
            BundleDocumentKind::Receipt => (BundleEntryKind::Receipt, "receipt"),
        };
        let digest = sha256_hex(&document.bytes);
        prepared.push(memory_entry(
            format!("{prefix}/{digest}"),
            kind,
            document.logical_name.clone(),
            document.license.clone(),
            document.bytes.clone(),
            None,
        )?);
    }
    Ok(())
}

fn memory_entry(
    path: String,
    kind: BundleEntryKind,
    logical_name: String,
    license: Option<String>,
    bytes: Vec<u8>,
    original_sha256: Option<String>,
) -> Result<PreparedEntry, BundleError> {
    let byte_len = u64::try_from(bytes.len())
        .map_err(|_| BundleError::Invalid("document length does not fit u64".to_owned()))?;
    let sha256 = sha256_hex(&bytes);
    Ok(PreparedEntry {
        manifest: BundleManifestEntry {
            path,
            kind,
            logical_name,
            license,
            sha256,
            byte_len,
            stored_len: byte_len,
            offset: 0,
            original_sha256,
        },
        content: PreparedContent::Memory(bytes),
    })
}

fn fingerprint_regular_file(
    path: &Path,
    max_bytes: u64,
    cancel: &AtomicBool,
) -> Result<PreparedFile, BundleError> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| BundleError::Io(format!("inspect {}: {error}", path.display())))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(BundleError::Invalid(format!(
            "source {} is not a regular no-follow file",
            path.display()
        )));
    }
    if metadata.len() > max_bytes {
        return Err(BundleError::Invalid(format!(
            "source {} is {} bytes; entry limit is {max_bytes}",
            path.display(),
            metadata.len()
        )));
    }
    let modified = metadata.modified().ok();
    let mut file = File::open(path)
        .map_err(|error| BundleError::Io(format!("open {}: {error}", path.display())))?;
    let mut buffer = vec![0_u8; STREAM_BUFFER_BYTES];
    let mut hasher = Sha256::new();
    let mut observed = 0_u64;
    loop {
        check_cancelled(cancel)?;
        let count = file
            .read(&mut buffer)
            .map_err(|error| BundleError::Io(format!("read {}: {error}", path.display())))?;
        if count == 0 {
            break;
        }
        observed = observed
            .checked_add(u64::try_from(count).unwrap_or(u64::MAX))
            .ok_or_else(|| BundleError::Invalid("source read length overflows".to_owned()))?;
        if observed > metadata.len() {
            return Err(BundleError::Invalid(format!(
                "source {} grew during preflight",
                path.display()
            )));
        }
        hasher.update(&buffer[..count]);
    }
    let after = fs::symlink_metadata(path)
        .map_err(|error| BundleError::Io(format!("re-inspect {}: {error}", path.display())))?;
    if after.file_type().is_symlink()
        || !after.is_file()
        || observed != metadata.len()
        || after.len() != metadata.len()
        || after.modified().ok() != modified
    {
        return Err(BundleError::Invalid(format!(
            "source {} changed or short-read during preflight",
            path.display()
        )));
    }
    Ok(PreparedFile {
        path: path.to_path_buf(),
        byte_len: observed,
        modified,
        sha256: format!("{:x}", hasher.finalize()),
    })
}

fn copy_prepared_file(
    prepared: &PreparedFile,
    output: &mut File,
    bundle_hasher: &mut Sha256,
    written: &mut u64,
    cancel: &AtomicBool,
    faults: &dyn BundleFaultInjector,
) -> Result<(), BundleError> {
    let metadata = fs::symlink_metadata(&prepared.path)?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() != prepared.byte_len
        || metadata.modified().ok() != prepared.modified
    {
        return Err(BundleError::Invalid(format!(
            "source {} changed after bundle preflight",
            prepared.path.display()
        )));
    }
    let mut input = File::open(&prepared.path)?;
    let mut remaining = prepared.byte_len;
    let mut source_hasher = Sha256::new();
    let mut buffer = vec![0_u8; STREAM_BUFFER_BYTES];
    while remaining != 0 {
        check_cancelled(cancel)?;
        let wanted = usize::try_from(remaining.min(buffer.len() as u64)).unwrap_or(buffer.len());
        let count = input.read(&mut buffer[..wanted])?;
        if count == 0 {
            return Err(BundleError::Invalid(format!(
                "source {} short-read after preflight",
                prepared.path.display()
            )));
        }
        source_hasher.update(&buffer[..count]);
        write_build_chunk(
            output,
            &buffer[..count],
            bundle_hasher,
            written,
            cancel,
            faults,
        )?;
        remaining -= u64::try_from(count).unwrap_or(u64::MAX);
    }
    let mut trailing = [0_u8; 1];
    if input.read(&mut trailing)? != 0
        || format!("{:x}", source_hasher.finalize()) != prepared.sha256
    {
        return Err(BundleError::Invalid(format!(
            "source {} changed after bundle preflight",
            prepared.path.display()
        )));
    }
    Ok(())
}

fn write_build_chunk(
    output: &mut File,
    bytes: &[u8],
    hasher: &mut Sha256,
    written: &mut u64,
    cancel: &AtomicBool,
    faults: &dyn BundleFaultInjector,
) -> Result<(), BundleError> {
    check_cancelled(cancel)?;
    faults.check(BundleFaultPhase::BuildWrite, *written)?;
    output.write_all(bytes)?;
    hasher.update(bytes);
    *written = written
        .checked_add(u64::try_from(bytes.len()).unwrap_or(u64::MAX))
        .ok_or_else(|| BundleError::Invalid("bundle write length overflows".to_owned()))?;
    Ok(())
}

fn encode_header(
    manifest_len: u32,
    entry_count: u32,
    payload_len: u64,
    manifest_digest: [u8; 32],
) -> [u8; COSBUNDLE_HEADER_BYTES] {
    let mut header = [0_u8; COSBUNDLE_HEADER_BYTES];
    header[..8].copy_from_slice(COSBUNDLE_MAGIC);
    header[8..10].copy_from_slice(&COSBUNDLE_FORMAT_VERSION.to_le_bytes());
    header[10..12].copy_from_slice(&COSBUNDLE_FLAGS.to_le_bytes());
    header[12..16].copy_from_slice(&manifest_len.to_le_bytes());
    header[16..20].copy_from_slice(&entry_count.to_le_bytes());
    header[20..28].copy_from_slice(&payload_len.to_le_bytes());
    header[28..60].copy_from_slice(&manifest_digest);
    header
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BundlePreviewEntry {
    pub path: String,
    pub kind: String,
    pub logical_name: String,
    pub license: Option<String>,
    pub sha256: String,
    pub byte_len: u64,
    pub authoritative: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BundlePreview {
    pub format_version: u16,
    pub bundle_sha256: String,
    pub patch_sha256: String,
    pub bundle_bytes: u64,
    pub expanded_bytes: u64,
    pub entries: Vec<BundlePreviewEntry>,
}

#[derive(Debug)]
struct BundleInspection {
    preview: BundlePreview,
    manifest: BundleManifest,
    payload_start: u64,
}

pub(crate) fn inspect_show_bundle(
    path: &Path,
    limits: BundleLimits,
    cancel: &AtomicBool,
) -> Result<BundlePreview, BundleError> {
    Ok(inspect_show_bundle_with_faults(path, limits, cancel, &NoBundleFaults)?.preview)
}

fn inspect_show_bundle_with_faults(
    path: &Path,
    limits: BundleLimits,
    cancel: &AtomicBool,
    faults: &dyn BundleFaultInjector,
) -> Result<BundleInspection, BundleError> {
    let limits = limits.validate()?;
    check_cancelled(cancel)?;
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| BundleError::Io(format!("inspect {}: {error}", path.display())))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(BundleError::Invalid(
            "bundle input is not a regular no-follow file".to_owned(),
        ));
    }
    if metadata.len() > limits.max_bundle_bytes {
        return Err(BundleError::Invalid(format!(
            "bundle is {} bytes; limit is {}",
            metadata.len(),
            limits.max_bundle_bytes
        )));
    }
    if metadata.len() < COSBUNDLE_HEADER_BYTES as u64 {
        return Err(BundleError::Invalid(
            "bundle header is truncated".to_owned(),
        ));
    }
    let mut file = File::open(path)?;
    let mut bundle_hasher = Sha256::new();
    let mut header = [0_u8; COSBUNDLE_HEADER_BYTES];
    read_exact_inspect(
        &mut file,
        &mut header,
        &mut bundle_hasher,
        cancel,
        faults,
        0,
    )?;
    if &header[..8] != COSBUNDLE_MAGIC {
        return Err(BundleError::Invalid(
            "unknown magic (ZIP/compressed archives are not accepted)".to_owned(),
        ));
    }
    let version = u16::from_le_bytes(header[8..10].try_into().unwrap());
    let flags = u16::from_le_bytes(header[10..12].try_into().unwrap());
    if version != COSBUNDLE_FORMAT_VERSION || flags != COSBUNDLE_FLAGS {
        return Err(BundleError::Invalid(format!(
            "unsupported header version {version} or flags {flags}"
        )));
    }
    let manifest_len =
        usize::try_from(u32::from_le_bytes(header[12..16].try_into().unwrap())).unwrap();
    let entry_count =
        usize::try_from(u32::from_le_bytes(header[16..20].try_into().unwrap())).unwrap();
    let payload_len = u64::from_le_bytes(header[20..28].try_into().unwrap());
    let manifest_digest: [u8; 32] = header[28..60].try_into().unwrap();
    if manifest_len == 0 || manifest_len > limits.max_manifest_bytes {
        return Err(BundleError::Invalid(format!(
            "manifest length {manifest_len} is outside 1..={} ",
            limits.max_manifest_bytes
        )));
    }
    if entry_count == 0 || entry_count > limits.max_entries {
        return Err(BundleError::Invalid(format!(
            "entry count {entry_count} is outside 1..={} ",
            limits.max_entries
        )));
    }
    let payload_start = (COSBUNDLE_HEADER_BYTES as u64)
        .checked_add(u64::try_from(manifest_len).unwrap_or(u64::MAX))
        .ok_or_else(|| BundleError::Invalid("payload offset overflows".to_owned()))?;
    let planned_total = payload_start
        .checked_add(payload_len)
        .ok_or_else(|| BundleError::Invalid("bundle length overflows".to_owned()))?;
    if planned_total != metadata.len() {
        return Err(BundleError::Invalid(format!(
            "header plans {planned_total} bytes but file has {} (short-read or trailing data)",
            metadata.len()
        )));
    }
    let mut manifest_bytes = vec![0_u8; manifest_len];
    read_exact_inspect(
        &mut file,
        &mut manifest_bytes,
        &mut bundle_hasher,
        cancel,
        faults,
        COSBUNDLE_HEADER_BYTES as u64,
    )?;
    if <[u8; 32]>::from(Sha256::digest(&manifest_bytes)) != manifest_digest {
        return Err(BundleError::Invalid("manifest SHA-256 mismatch".to_owned()));
    }
    let manifest: BundleManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| BundleError::Invalid(format!("parse manifest: {error}")))?;
    let canonical = serde_json::to_vec(&manifest)
        .map_err(|error| BundleError::Invalid(format!("re-serialize manifest: {error}")))?;
    if canonical != manifest_bytes {
        return Err(BundleError::Invalid(
            "manifest is not in canonical serialized form".to_owned(),
        ));
    }
    if manifest.entries.len() != entry_count {
        return Err(BundleError::Invalid(format!(
            "header declares {entry_count} entries but manifest has {}",
            manifest.entries.len()
        )));
    }
    validate_manifest(&manifest, payload_len, limits)?;

    let mut patch_bytes = None;
    let mut progress = payload_start;
    let mut buffer = vec![0_u8; STREAM_BUFFER_BYTES];
    for entry in &manifest.entries {
        check_cancelled(cancel)?;
        let mut remaining = entry.stored_len;
        let mut entry_hasher = Sha256::new();
        let mut document = entry.kind.is_document().then(Vec::new);
        while remaining != 0 {
            let wanted =
                usize::try_from(remaining.min(buffer.len() as u64)).unwrap_or(buffer.len());
            let count = file.read(&mut buffer[..wanted])?;
            if count == 0 {
                return Err(BundleError::Invalid(format!(
                    "short read in entry {}",
                    entry.path
                )));
            }
            faults.check(BundleFaultPhase::InspectRead, progress)?;
            check_cancelled(cancel)?;
            let bytes = &buffer[..count];
            bundle_hasher.update(bytes);
            entry_hasher.update(bytes);
            if let Some(document) = &mut document {
                document.extend_from_slice(bytes);
            }
            let count = u64::try_from(count).unwrap_or(u64::MAX);
            remaining -= count;
            progress = progress
                .checked_add(count)
                .ok_or_else(|| BundleError::Invalid("inspection progress overflows".to_owned()))?;
        }
        let observed = format!("{:x}", entry_hasher.finalize());
        if observed != entry.sha256 {
            return Err(BundleError::Invalid(format!(
                "entry {} SHA-256 mismatch",
                entry.path
            )));
        }
        if let Some(document) = document {
            validate_document_entry(entry, &document)?;
            if entry.kind == BundleEntryKind::Patch {
                patch_bytes = Some(document);
            }
        }
    }
    if progress != metadata.len() {
        return Err(BundleError::Invalid(
            "bundle inspection did not consume the planned file".to_owned(),
        ));
    }
    let patch_bytes = patch_bytes.expect("manifest validation guarantees one patch document");
    let patch = crate::patch::editor::parse_patch_bytes(&patch_bytes)
        .map_err(|error| BundleError::Invalid(format!("validate bundled patch: {error}")))?;
    let canonical_patch = serde_yaml::to_string(&patch)
        .map_err(|error| BundleError::Invalid(format!("re-serialize bundled patch: {error}")))?
        .into_bytes();
    if canonical_patch != patch_bytes {
        return Err(BundleError::Invalid(
            "bundled patch is valid but not canonical".to_owned(),
        ));
    }
    let originals = manifest
        .entries
        .iter()
        .filter(|entry| entry.kind == BundleEntryKind::OriginalMedia)
        .map(|entry| (entry.sha256.clone(), entry.byte_len))
        .collect::<BTreeMap<_, _>>();
    validate_patch_original_coverage(&patch, &originals)?;
    validate_embedded_sidecar_coverage(&patch, &manifest)?;

    let expanded_bytes = manifest
        .entries
        .iter()
        .try_fold(0_u64, |sum, entry| sum.checked_add(entry.byte_len))
        .ok_or_else(|| BundleError::Invalid("expanded byte total overflows".to_owned()))?;
    let preview = BundlePreview {
        format_version: version,
        bundle_sha256: format!("{:x}", bundle_hasher.finalize()),
        patch_sha256: manifest.patch_sha256.clone(),
        bundle_bytes: metadata.len(),
        expanded_bytes,
        entries: manifest
            .entries
            .iter()
            .map(|entry| BundlePreviewEntry {
                path: entry.path.clone(),
                kind: serde_json::to_value(entry.kind)
                    .ok()
                    .and_then(|value| value.as_str().map(str::to_owned))
                    .unwrap_or_else(|| "unknown".to_owned()),
                logical_name: entry.logical_name.clone(),
                license: entry.license.clone(),
                sha256: entry.sha256.clone(),
                byte_len: entry.byte_len,
                authoritative: entry.kind.is_authoritative(),
            })
            .collect(),
    };
    Ok(BundleInspection {
        preview,
        manifest,
        payload_start,
    })
}

fn read_exact_inspect(
    file: &mut File,
    bytes: &mut [u8],
    hasher: &mut Sha256,
    cancel: &AtomicBool,
    faults: &dyn BundleFaultInjector,
    progress: u64,
) -> Result<(), BundleError> {
    check_cancelled(cancel)?;
    faults.check(BundleFaultPhase::InspectRead, progress)?;
    file.read_exact(bytes)
        .map_err(|error| BundleError::Io(format!("short bundle read: {error}")))?;
    hasher.update(bytes);
    Ok(())
}

fn validate_manifest(
    manifest: &BundleManifest,
    payload_len: u64,
    limits: BundleLimits,
) -> Result<(), BundleError> {
    if manifest.schema_version != COSBUNDLE_FORMAT_VERSION {
        return Err(BundleError::Invalid(format!(
            "unsupported manifest version {}",
            manifest.schema_version
        )));
    }
    validate_sha256(&manifest.patch_sha256, "patch digest")?;
    if manifest.entries.is_empty() || manifest.entries.len() > limits.max_entries {
        return Err(BundleError::Invalid(format!(
            "manifest entry count {} is outside 1..={} ",
            manifest.entries.len(),
            limits.max_entries
        )));
    }
    let mut paths = BTreeSet::new();
    let mut logical_names = BTreeSet::new();
    let mut originals = BTreeSet::new();
    let mut expected_offset = 0_u64;
    let mut patch_count = 0_usize;
    let mut previous_path: Option<&str> = None;
    for entry in &manifest.entries {
        validate_bundle_path(&entry.path, limits)?;
        validate_logical_name(&entry.logical_name, limits)?;
        validate_optional_text(entry.license.as_deref(), "license", 8 * 1024)?;
        reject_secret_markers(entry.logical_name.as_bytes(), "logical name")?;
        if let Some(license) = &entry.license {
            reject_secret_markers(license.as_bytes(), "license")?;
        }
        validate_sha256(&entry.sha256, "entry digest")?;
        if entry.byte_len != entry.stored_len {
            return Err(BundleError::Invalid(format!(
                "entry {} requests expansion/compression; format 1 requires a 1:1 byte ratio",
                entry.path
            )));
        }
        if entry.byte_len > limits.max_entry_bytes {
            return Err(BundleError::Invalid(format!(
                "entry {} is {} bytes; limit is {}",
                entry.path, entry.byte_len, limits.max_entry_bytes
            )));
        }
        if entry.kind.is_document()
            && entry.byte_len > u64::try_from(limits.max_document_bytes).unwrap_or(u64::MAX)
        {
            return Err(BundleError::Invalid(format!(
                "document {} is {} bytes; limit is {}",
                entry.path, entry.byte_len, limits.max_document_bytes
            )));
        }
        if entry.offset != expected_offset {
            return Err(BundleError::Invalid(format!(
                "entry {} offset {} is not contiguous expected offset {expected_offset}",
                entry.path, entry.offset
            )));
        }
        expected_offset = expected_offset
            .checked_add(entry.stored_len)
            .ok_or_else(|| BundleError::Invalid("entry offsets overflow".to_owned()))?;
        if let Some(previous) = previous_path {
            if previous >= entry.path.as_str() {
                return Err(BundleError::Invalid(
                    "manifest entries are not in strict deterministic path order".to_owned(),
                ));
            }
        }
        previous_path = Some(&entry.path);
        if !paths.insert(portable_fold(&entry.path)) {
            return Err(BundleError::Invalid(format!(
                "duplicate or case-fold-colliding entry path {}",
                entry.path
            )));
        }
        if !logical_names.insert(portable_fold(&entry.logical_name)) {
            return Err(BundleError::Invalid(format!(
                "duplicate or case-fold-colliding logical name {}",
                entry.logical_name
            )));
        }
        match entry.kind {
            BundleEntryKind::Patch => {
                patch_count += 1;
                if entry.sha256 != manifest.patch_sha256 {
                    return Err(BundleError::Invalid(
                        "patch entry digest disagrees with manifest patch identity".to_owned(),
                    ));
                }
                if entry.original_sha256.is_some() {
                    return Err(BundleError::Invalid(
                        "patch entry may not be derived from media".to_owned(),
                    ));
                }
            }
            BundleEntryKind::OriginalMedia => {
                originals.insert(entry.sha256.clone());
                if entry.original_sha256.is_some() {
                    return Err(BundleError::Invalid(
                        "original media may not declare a derived identity".to_owned(),
                    ));
                }
            }
            BundleEntryKind::Proxy => {
                let original = entry.original_sha256.as_deref().ok_or_else(|| {
                    BundleError::Invalid("proxy lacks its original SHA-256 link".to_owned())
                })?;
                validate_sha256(original, "proxy original digest")?;
            }
            _ if entry.original_sha256.is_some() => {
                return Err(BundleError::Invalid(format!(
                    "non-proxy entry {} declares a derived identity",
                    entry.path
                )));
            }
            _ => {}
        }
    }
    if patch_count != 1 {
        return Err(BundleError::Invalid(format!(
            "manifest must contain exactly one patch, found {patch_count}"
        )));
    }
    for proxy in manifest
        .entries
        .iter()
        .filter(|entry| entry.kind == BundleEntryKind::Proxy)
    {
        if !originals.contains(proxy.original_sha256.as_deref().unwrap()) {
            return Err(BundleError::Invalid(format!(
                "proxy {} links to an absent original",
                proxy.path
            )));
        }
    }
    if expected_offset != payload_len {
        return Err(BundleError::Invalid(format!(
            "entries total {expected_offset} bytes but header payload is {payload_len}"
        )));
    }
    if expected_offset > limits.max_expanded_bytes {
        return Err(BundleError::Invalid(format!(
            "expanded bytes {expected_offset} exceed {}",
            limits.max_expanded_bytes
        )));
    }
    Ok(())
}

fn validate_document_entry(entry: &BundleManifestEntry, bytes: &[u8]) -> Result<(), BundleError> {
    reject_secret_markers(bytes, &entry.logical_name)?;
    match entry.kind {
        BundleEntryKind::Patch => {
            crate::patch::editor::parse_patch_bytes(bytes)
                .map_err(|error| BundleError::Invalid(format!("bundled patch: {error}")))?;
        }
        BundleEntryKind::Study => {
            crate::study::StudyDocument::from_json_bytes(bytes)
                .map_err(|error| BundleError::Invalid(format!("bundled Study: {error}")))?;
        }
        BundleEntryKind::Gesture => {
            crate::gesture::GestureTrackDocument::from_json_bytes(bytes)
                .map_err(|error| BundleError::Invalid(format!("bundled gesture: {error}")))?;
        }
        BundleEntryKind::PerformanceTake => {
            crate::performance_track::PerformanceTakeDocument::from_json_bytes(bytes)
                .map_err(|error| BundleError::Invalid(format!("bundled take: {error}")))?;
        }
        BundleEntryKind::ControllerProfile
        | BundleEntryKind::VenueProfile
        | BundleEntryKind::Receipt => {
            std::str::from_utf8(bytes).map_err(|_| {
                BundleError::Invalid(format!("document {} is not UTF-8", entry.path))
            })?;
        }
        BundleEntryKind::OriginalMedia | BundleEntryKind::Proxy => {
            return Err(BundleError::Invalid(
                "internal document classifier received media".to_owned(),
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BundleImportCollision {
    Fail,
    ReuseVerified,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BundleImportReceipt {
    pub generation_root: PathBuf,
    pub patch_path: PathBuf,
    pub media_paths: Vec<PathBuf>,
    pub bundle_sha256: String,
    pub reused: bool,
}

pub(crate) fn import_show_bundle(
    bundle: &Path,
    library_root: &Path,
    collision: BundleImportCollision,
    limits: BundleLimits,
    cancel: &AtomicBool,
) -> Result<BundleImportReceipt, BundleError> {
    import_show_bundle_with_faults(
        bundle,
        library_root,
        collision,
        limits,
        cancel,
        &NoBundleFaults,
    )
}

fn import_show_bundle_with_faults(
    bundle: &Path,
    library_root: &Path,
    collision: BundleImportCollision,
    limits: BundleLimits,
    cancel: &AtomicBool,
    faults: &dyn BundleFaultInjector,
) -> Result<BundleImportReceipt, BundleError> {
    let inspection = inspect_show_bundle_with_faults(bundle, limits, cancel, faults)?;
    let root_metadata = fs::symlink_metadata(library_root)
        .map_err(|error| BundleError::Io(format!("inspect import root: {error}")))?;
    if root_metadata.file_type().is_symlink() || !root_metadata.is_dir() {
        return Err(BundleError::Invalid(
            "import root is not a regular no-follow directory".to_owned(),
        ));
    }
    let generation_root = import_generation_path(library_root, &inspection.preview.bundle_sha256);
    if generation_root.exists() {
        return match collision {
            BundleImportCollision::Fail => Err(BundleError::Collision(format!(
                "generation {} already exists",
                generation_root.display()
            ))),
            BundleImportCollision::ReuseVerified => {
                verify_existing_generation(&generation_root, &inspection)?;
                Ok(import_receipt(&generation_root, &inspection, true))
            }
        };
    }

    let mut staging = ImportStaging::create(library_root)?;
    let mut input = File::open(bundle)?;
    let mut synced_directories = BTreeSet::new();
    for entry in &inspection.manifest.entries {
        check_cancelled(cancel)?;
        let relative = import_relative_path(entry);
        let destination = staging.path().join(&relative);
        let parent = destination
            .parent()
            .expect("import entry always has a parent");
        fs::create_dir_all(parent)?;
        synced_directories.insert(parent.to_path_buf());
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&destination)?;
        input.seek(SeekFrom::Start(
            inspection
                .payload_start
                .checked_add(entry.offset)
                .ok_or_else(|| BundleError::Invalid("entry seek offset overflows".to_owned()))?,
        ))?;
        let mut remaining = entry.stored_len;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0_u8; STREAM_BUFFER_BYTES];
        let mut progress = 0_u64;
        while remaining != 0 {
            check_cancelled(cancel)?;
            faults.check(BundleFaultPhase::ImportWrite, progress)?;
            let wanted =
                usize::try_from(remaining.min(buffer.len() as u64)).unwrap_or(buffer.len());
            let count = input.read(&mut buffer[..wanted])?;
            if count == 0 {
                return Err(BundleError::Invalid(format!(
                    "short read while importing {}",
                    entry.path
                )));
            }
            output.write_all(&buffer[..count])?;
            hasher.update(&buffer[..count]);
            let count = u64::try_from(count).unwrap_or(u64::MAX);
            remaining -= count;
            progress = progress
                .checked_add(count)
                .ok_or_else(|| BundleError::Invalid("import progress overflows".to_owned()))?;
        }
        if format!("{:x}", hasher.finalize()) != entry.sha256 {
            return Err(BundleError::Invalid(format!(
                "entry {} changed between inspection and import",
                entry.path
            )));
        }
        output.flush()?;
        output.sync_all()?;
    }
    let identity_path = staging.path().join(IMPORT_IDENTITY_FILE);
    let mut identity = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&identity_path)?;
    writeln!(identity, "version={COSBUNDLE_FORMAT_VERSION}")?;
    writeln!(identity, "sha256={}", inspection.preview.bundle_sha256)?;
    identity.flush()?;
    identity.sync_all()?;
    drop(identity);
    synced_directories.insert(staging.path().to_path_buf());
    let mut directories = synced_directories.into_iter().collect::<Vec<_>>();
    directories.sort_by_key(|path| std::cmp::Reverse(path.components().count()));
    for directory in directories {
        sync_directory(&directory)?;
    }
    check_cancelled(cancel)?;
    faults.check(BundleFaultPhase::ImportBeforePublish, 0)?;
    publish_directory_noreplace(staging.path(), &generation_root).map_err(|error| {
        if error.kind() == io::ErrorKind::AlreadyExists {
            BundleError::Collision("external writer won the generation name".to_owned())
        } else {
            BundleError::from(error)
        }
    })?;
    staging.disarm();
    Ok(import_receipt(&generation_root, &inspection, false))
}

fn import_generation_path(root: &Path, digest: &str) -> PathBuf {
    root.join(format!("{IMPORT_GENERATION_PREFIX}{digest}"))
}

fn import_relative_path(entry: &BundleManifestEntry) -> PathBuf {
    match entry.kind {
        BundleEntryKind::Patch => PathBuf::from("show.cos.yaml"),
        BundleEntryKind::OriginalMedia => PathBuf::from(&entry.logical_name),
        BundleEntryKind::Proxy => PathBuf::from(".derived")
            .join(entry.original_sha256.as_deref().unwrap())
            .join(&entry.logical_name),
        _ => PathBuf::from(".cosbundle-meta").join(&entry.path),
    }
}

fn import_receipt(
    generation_root: &Path,
    inspection: &BundleInspection,
    reused: bool,
) -> BundleImportReceipt {
    BundleImportReceipt {
        generation_root: generation_root.to_path_buf(),
        patch_path: generation_root.join("show.cos.yaml"),
        media_paths: inspection
            .manifest
            .entries
            .iter()
            .filter(|entry| entry.kind == BundleEntryKind::OriginalMedia)
            .map(|entry| generation_root.join(&entry.logical_name))
            .collect(),
        bundle_sha256: inspection.preview.bundle_sha256.clone(),
        reused,
    }
}

fn verify_existing_generation(
    root: &Path,
    inspection: &BundleInspection,
) -> Result<(), BundleError> {
    let metadata = fs::symlink_metadata(root)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(BundleError::Collision(
            "existing generation is not a regular directory".to_owned(),
        ));
    }
    let identity = fs::read_to_string(root.join(IMPORT_IDENTITY_FILE))?;
    if !identity
        .lines()
        .any(|line| line.strip_prefix("sha256=") == Some(inspection.preview.bundle_sha256.as_str()))
    {
        return Err(BundleError::Collision(
            "existing generation has a different bundle identity".to_owned(),
        ));
    }
    for entry in &inspection.manifest.entries {
        let path = root.join(import_relative_path(entry));
        let observed = fingerprint_regular_file(&path, entry.byte_len, &AtomicBool::new(false))?;
        if observed.byte_len != entry.byte_len || observed.sha256 != entry.sha256 {
            return Err(BundleError::Collision(format!(
                "existing generation entry {} is not byte-identical",
                entry.logical_name
            )));
        }
    }
    Ok(())
}

struct ImportStaging {
    path: Option<PathBuf>,
}

impl ImportStaging {
    fn create(root: &Path) -> Result<Self, BundleError> {
        for _ in 0..16 {
            let mut nonce = [0_u8; 16];
            getrandom::fill(&mut nonce)
                .map_err(|error| BundleError::Io(format!("staging entropy: {error}")))?;
            let suffix = nonce
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            let path = root.join(format!(
                "{IMPORT_STAGE_PREFIX}{}-{suffix}.part",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path: Some(path) }),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(BundleError::from(error)),
            }
        }
        Err(BundleError::Collision(
            "could not reserve a unique import staging directory".to_owned(),
        ))
    }

    fn path(&self) -> &Path {
        self.path
            .as_deref()
            .expect("active import staging has a path")
    }

    fn disarm(&mut self) {
        self.path = None;
    }
}

impl Drop for ImportStaging {
    fn drop(&mut self) {
        if let Some(path) = self.path.take() {
            let _ = fs::remove_dir_all(path);
        }
    }
}

pub(crate) fn cleanup_show_bundle_orphans(
    library_root: &Path,
    max_entries: usize,
) -> Result<usize, BundleError> {
    let metadata = fs::symlink_metadata(library_root)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(BundleError::Invalid(
            "orphan cleanup root is not a regular no-follow directory".to_owned(),
        ));
    }
    let mut removed = 0_usize;
    for entry in fs::read_dir(library_root)?.take(max_entries) {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if !name.starts_with(IMPORT_STAGE_PREFIX) || !name.ends_with(".part") {
            continue;
        }
        let metadata = fs::symlink_metadata(entry.path())?;
        if metadata.file_type().is_symlink() {
            fs::remove_file(entry.path())?;
            removed += 1;
        } else if metadata.is_dir() {
            fs::remove_dir_all(entry.path())?;
            removed += 1;
        }
    }
    Ok(removed)
}

fn validate_bundle_path(path: &str, limits: BundleLimits) -> Result<(), BundleError> {
    if path.is_empty()
        || path.len() > limits.max_name_bytes.saturating_mul(limits.max_path_depth)
        || path.contains('\\')
        || path.contains(':')
        || !path.is_ascii()
    {
        return Err(BundleError::Invalid(format!(
            "unsafe logical entry path '{path}'"
        )));
    }
    let candidate = Path::new(path);
    if candidate.is_absolute()
        || candidate
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(BundleError::Invalid(format!(
            "entry path is absolute or traversing: {path}"
        )));
    }
    let components = path.split('/').collect::<Vec<_>>();
    if components.len() > limits.max_path_depth {
        return Err(BundleError::Invalid(format!(
            "entry path depth {} exceeds {}",
            components.len(),
            limits.max_path_depth
        )));
    }
    for component in components {
        validate_portable_component(component, limits.max_name_bytes)?;
    }
    Ok(())
}

fn validate_logical_name(name: &str, limits: BundleLimits) -> Result<(), BundleError> {
    if name.contains('/') || name.contains('\\') || Path::new(name).is_absolute() {
        return Err(BundleError::Invalid(format!(
            "logical name must be one portable file component: {name}"
        )));
    }
    validate_portable_component(name, limits.max_name_bytes)
}

fn validate_portable_component(component: &str, max_bytes: usize) -> Result<(), BundleError> {
    if component.is_empty()
        || component == "."
        || component == ".."
        || component.len() > max_bytes
        || !component.is_ascii()
        || component.ends_with(' ')
        || component.ends_with('.')
        || component.bytes().any(|byte| {
            byte < 0x20
                || matches!(
                    byte,
                    b'<' | b'>' | b':' | b'"' | b'/' | b'\\' | b'|' | b'?' | b'*'
                )
        })
        || windows_device_name(component)
    {
        return Err(BundleError::Invalid(format!(
            "unsafe portable name component '{component}'"
        )));
    }
    Ok(())
}

fn windows_device_name(component: &str) -> bool {
    let stem = component
        .split_once('.')
        .map_or(component, |(stem, _)| stem)
        .to_ascii_uppercase();
    matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || stem
            .strip_prefix("COM")
            .or_else(|| stem.strip_prefix("LPT"))
            .is_some_and(|suffix| suffix.len() == 1 && matches!(suffix.as_bytes()[0], b'1'..=b'9'))
}

fn validate_optional_text(
    text: Option<&str>,
    label: &str,
    max_bytes: usize,
) -> Result<(), BundleError> {
    if let Some(text) = text {
        if text.len() > max_bytes
            || text
                .chars()
                .any(|character| character.is_control() && !matches!(character, '\n' | '\r' | '\t'))
        {
            return Err(BundleError::Invalid(format!(
                "{label} is not bounded printable text"
            )));
        }
    }
    Ok(())
}

fn reject_secret_markers(bytes: &[u8], label: &str) -> Result<(), BundleError> {
    let lowercase = bytes.iter().map(u8::to_ascii_lowercase).collect::<Vec<_>>();
    for marker in [
        b"access_token".as_slice(),
        b"private_key".as_slice(),
        b"authorization:".as_slice(),
        b"cookie:".as_slice(),
        b"bearer ".as_slice(),
        b"?key=".as_slice(),
    ] {
        if lowercase
            .windows(marker.len())
            .any(|window| window == marker)
        {
            return Err(BundleError::Invalid(format!(
                "{label} contains a secret-bearing marker"
            )));
        }
    }
    Ok(())
}

fn validate_sha256(digest: &str, label: &str) -> Result<(), BundleError> {
    if digest.len() != 64
        || digest
            .bytes()
            .any(|byte| !byte.is_ascii_hexdigit() || byte.is_ascii_uppercase())
    {
        return Err(BundleError::Invalid(format!(
            "{label} must be 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn portable_fold(value: &str) -> String {
    value.to_ascii_lowercase()
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn check_cancelled(cancel: &AtomicBool) -> Result<(), BundleError> {
    if cancel.load(Ordering::Acquire) {
        Err(BundleError::Cancelled)
    } else {
        Ok(())
    }
}

fn check_cancelled_io(cancel: &AtomicBool) -> io::Result<()> {
    if cancel.load(Ordering::Acquire) {
        Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "bundle operation cancelled",
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::sync::atomic::AtomicU64;

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(label: &str) -> Self {
            let ordinal = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "collide-o-scope-cosbundle-{label}-{}-{ordinal}",
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

    fn fixture_patch(reference: &str) -> PatchState {
        let yaml = format!(
            r#"master: {{}}
layers:
  - filename: clip.mp4
    source_path: '{reference}'
    clip_slots:
      - id: 1
        filename: clip.mp4
        source_path: '{reference}'
"#
        );
        crate::patch::editor::parse_patch_bytes(yaml.as_bytes()).unwrap()
    }

    fn fixture_request(root: &Path, bytes: &[u8]) -> BundleBuildRequest {
        let media = root.join("source.mp4");
        fs::write(&media, bytes).unwrap();
        BundleBuildRequest {
            patch: fixture_patch("fixture-source"),
            media: vec![BundleMediaInput {
                source: media,
                logical_name: "clip.mp4".to_owned(),
                patch_references: vec!["fixture-source".to_owned()],
                expected_identity: None,
                license: Some("CC0-1.0".to_owned()),
                role: BundleMediaRole::Original,
            }],
            documents: Vec::new(),
            output_collision: BundleOutputCollision::Fail,
        }
    }

    fn no_cancel() -> AtomicBool {
        AtomicBool::new(false)
    }

    fn assert_no_staging(root: &Path) {
        assert!(fs::read_dir(root).unwrap().all(|entry| {
            let name = entry.unwrap().file_name().to_string_lossy().into_owned();
            !name.contains("cosbundle-stage") && !name.starts_with(IMPORT_STAGE_PREFIX)
        }));
    }

    #[test]
    fn deterministic_build_preview_and_atomic_import_resolve_original_identity() {
        let source = TempDir::new("deterministic");
        let first = source.0.join("first.cosbundle");
        let second = source.0.join("second.cosbundle");
        let cancel = no_cancel();
        let first_receipt = build_show_bundle(
            &first,
            fixture_request(&source.0, b"canonical original media"),
            BundleLimits::default(),
            &cancel,
        )
        .unwrap();
        let second_receipt = build_show_bundle(
            &second,
            fixture_request(&source.0, b"canonical original media"),
            BundleLimits::default(),
            &cancel,
        )
        .unwrap();
        assert_eq!(fs::read(&first).unwrap(), fs::read(&second).unwrap());
        assert_eq!(first_receipt.bundle_sha256, second_receipt.bundle_sha256);

        let import_root = source.0.join("library");
        fs::create_dir(&import_root).unwrap();
        let before = fs::read_dir(&import_root).unwrap().count();
        let preview = inspect_show_bundle(&first, BundleLimits::default(), &cancel).unwrap();
        assert_eq!(fs::read_dir(&import_root).unwrap().count(), before);
        assert_eq!(preview.entries.len(), 2);
        let original = preview
            .entries
            .iter()
            .find(|entry| entry.kind == "original_media")
            .unwrap();
        assert!(original.authoritative);
        assert_eq!(original.license.as_deref(), Some("CC0-1.0"));

        let imported = import_show_bundle(
            &first,
            &import_root,
            BundleImportCollision::Fail,
            BundleLimits::default(),
            &cancel,
        )
        .unwrap();
        assert_eq!(
            fs::read(&imported.media_paths[0]).unwrap(),
            b"canonical original media"
        );
        let patch = crate::patch::editor::load_patch_path(&imported.patch_path).unwrap();
        let identity = parse_content_reference(&patch.layers[0].source_path)
            .unwrap()
            .unwrap();
        assert_eq!(identity.sha256, original.sha256);
        assert_eq!(identity.byte_len, original.byte_len);

        let reused = import_show_bundle(
            &first,
            &import_root,
            BundleImportCollision::ReuseVerified,
            BundleLimits::default(),
            &cancel,
        )
        .unwrap();
        assert!(reused.reused);
        assert_eq!(reused.generation_root, imported.generation_root);
    }

    #[test]
    fn proxy_documents_and_atomic_replace_preserve_authoritative_originals() {
        let directory = TempDir::new("proxy-documents-replace");
        let bundle = directory.0.join("show.cosbundle");
        let original_bytes = b"authoritative original";
        let original_sha256 = sha256_hex(original_bytes);
        let proxy_path = directory.0.join("source.proxy.mp4");
        fs::write(&proxy_path, b"derived proxy").unwrap();

        let mut request = fixture_request(&directory.0, original_bytes);
        request.media.push(BundleMediaInput {
            source: proxy_path,
            logical_name: "clip.proxy.mp4".to_owned(),
            patch_references: Vec::new(),
            expected_identity: None,
            license: Some("CC0-1.0".to_owned()),
            role: BundleMediaRole::Proxy {
                original_sha256: original_sha256.clone(),
            },
        });
        request.documents.extend([
            BundleDocumentInput {
                kind: BundleDocumentKind::VenueProfile,
                logical_name: "venue.json".to_owned(),
                bytes: br#"{"venue":"fixture"}"#.to_vec(),
                license: None,
            },
            BundleDocumentInput {
                kind: BundleDocumentKind::Receipt,
                logical_name: "receipt.json".to_owned(),
                bytes: br#"{"receipt":"fixture"}"#.to_vec(),
                license: None,
            },
        ]);
        let original_receipt =
            build_show_bundle(&bundle, request, BundleLimits::default(), &no_cancel()).unwrap();
        let preview = inspect_show_bundle(&bundle, BundleLimits::default(), &no_cancel()).unwrap();
        let proxy = preview
            .entries
            .iter()
            .find(|entry| entry.kind == "proxy")
            .unwrap();
        assert!(!proxy.authoritative);
        assert!(preview
            .entries
            .iter()
            .find(|entry| entry.kind == "receipt")
            .is_some_and(|entry| !entry.authoritative));
        assert!(preview
            .entries
            .iter()
            .find(|entry| entry.kind == "venue_profile")
            .is_some_and(|entry| entry.authoritative));

        let library = directory.0.join("library");
        fs::create_dir(&library).unwrap();
        let imported = import_show_bundle(
            &bundle,
            &library,
            BundleImportCollision::Fail,
            BundleLimits::default(),
            &no_cancel(),
        )
        .unwrap();
        assert_eq!(
            fs::read(
                imported
                    .generation_root
                    .join(".derived")
                    .join(&original_sha256)
                    .join("clip.proxy.mp4")
            )
            .unwrap(),
            b"derived proxy"
        );

        let mut replacement = fixture_request(&directory.0, b"replacement original");
        replacement.output_collision = BundleOutputCollision::Replace;
        let replacement_receipt =
            build_show_bundle(&bundle, replacement, BundleLimits::default(), &no_cancel()).unwrap();
        assert_ne!(
            replacement_receipt.bundle_sha256,
            original_receipt.bundle_sha256
        );
        inspect_show_bundle(&bundle, BundleLimits::default(), &no_cancel()).unwrap();
    }

    #[test]
    fn every_hard_size_and_count_boundary_rejects_one_over() {
        let directory = TempDir::new("one-over");
        let output = directory.0.join("limited.cosbundle");

        let count_limits = BundleLimits {
            max_entries: 1,
            ..BundleLimits::default()
        };
        assert!(build_show_bundle(
            &output,
            fixture_request(&directory.0, b"x"),
            count_limits,
            &no_cancel(),
        )
        .is_err());
        assert!(!output.exists());

        let entry_limits = BundleLimits {
            max_entry_bytes: 1,
            ..BundleLimits::default()
        };
        assert!(build_show_bundle(
            &output,
            fixture_request(&directory.0, b"xx"),
            entry_limits,
            &no_cancel(),
        )
        .is_err());
        assert!(!output.exists());

        let mut two_byte_patch = skeletal_entry(
            "patch/show.yaml",
            BundleEntryKind::Patch,
            "show.cos.yaml",
            'a',
        );
        two_byte_patch.byte_len = 2;
        two_byte_patch.stored_len = 2;
        let manifest = skeletal_manifest(vec![two_byte_patch]);
        let document_limits = BundleLimits {
            max_document_bytes: 1,
            ..BundleLimits::default()
        };
        assert!(validate_manifest(&manifest, 2, document_limits).is_err());
        let expanded_limits = BundleLimits {
            max_expanded_bytes: 1,
            ..BundleLimits::default()
        };
        assert!(validate_manifest(&manifest, 2, expanded_limits).is_err());

        let receipt = build_show_bundle(
            &output,
            fixture_request(&directory.0, b"x"),
            BundleLimits::default(),
            &no_cancel(),
        )
        .unwrap();
        let bundle_limits = BundleLimits {
            max_bundle_bytes: receipt.byte_len - 1,
            ..BundleLimits::default()
        };
        assert!(inspect_show_bundle(&output, bundle_limits, &no_cancel()).is_err());
    }

    fn skeletal_manifest(entries: Vec<BundleManifestEntry>) -> BundleManifest {
        let patch_sha256 = entries
            .iter()
            .find(|entry| entry.kind == BundleEntryKind::Patch)
            .map(|entry| entry.sha256.clone())
            .unwrap_or_else(|| "a".repeat(64));
        BundleManifest {
            schema_version: COSBUNDLE_FORMAT_VERSION,
            patch_sha256,
            entries,
        }
    }

    fn skeletal_entry(
        path: &str,
        kind: BundleEntryKind,
        logical_name: &str,
        digest_byte: char,
    ) -> BundleManifestEntry {
        BundleManifestEntry {
            path: path.to_owned(),
            kind,
            logical_name: logical_name.to_owned(),
            license: None,
            sha256: digest_byte.to_string().repeat(64),
            byte_len: 0,
            stored_len: 0,
            offset: 0,
            original_sha256: None,
        }
    }

    #[test]
    fn traversal_absolute_device_casefold_duplicate_and_expansion_are_rejected() {
        let limits = BundleLimits::default();
        for hostile in ["../show.yaml", "/absolute/show.yaml", "C:/show.yaml"] {
            let manifest = skeletal_manifest(vec![skeletal_entry(
                hostile,
                BundleEntryKind::Patch,
                "show.cos.yaml",
                'a',
            )]);
            assert!(
                validate_manifest(&manifest, 0, limits).is_err(),
                "{hostile}"
            );
        }
        let device = skeletal_manifest(vec![skeletal_entry(
            "patch/show.yaml",
            BundleEntryKind::Patch,
            "CON",
            'a',
        )]);
        assert!(validate_manifest(&device, 0, limits).is_err());

        let casefold = skeletal_manifest(vec![
            skeletal_entry("media/a", BundleEntryKind::OriginalMedia, "CLIP.MP4", 'b'),
            skeletal_entry("media/b", BundleEntryKind::OriginalMedia, "clip.mp4", 'c'),
            skeletal_entry(
                "patch/show.yaml",
                BundleEntryKind::Patch,
                "show.cos.yaml",
                'a',
            ),
        ]);
        assert!(validate_manifest(&casefold, 0, limits).is_err());

        let duplicate = skeletal_manifest(vec![
            skeletal_entry("patch/show.yaml", BundleEntryKind::Patch, "one.yaml", 'a'),
            skeletal_entry(
                "patch/show.yaml",
                BundleEntryKind::OriginalMedia,
                "two.mp4",
                'b',
            ),
        ]);
        assert!(validate_manifest(&duplicate, 0, limits).is_err());

        let mut expanded = skeletal_entry(
            "patch/show.yaml",
            BundleEntryKind::Patch,
            "show.cos.yaml",
            'a',
        );
        expanded.byte_len = 2;
        expanded.stored_len = 1;
        let expanded = skeletal_manifest(vec![expanded]);
        assert!(validate_manifest(&expanded, 1, limits).is_err());
    }

    #[test]
    fn unknown_symlink_entry_and_zip_bomb_envelope_are_refused() {
        let unknown = format!(
            r#"{{"schema_version":1,"patch_sha256":"{}","entries":[{{"path":"patch/show.yaml","kind":"symlink","logical_name":"show.cos.yaml","sha256":"{}","byte_len":0,"stored_len":0,"offset":0}}]}}"#,
            "a".repeat(64),
            "a".repeat(64)
        );
        assert!(serde_json::from_str::<BundleManifest>(&unknown).is_err());

        let directory = TempDir::new("zip-bomb");
        let path = directory.0.join("bomb.cosbundle");
        let mut bytes = vec![0_u8; COSBUNDLE_HEADER_BYTES + 128];
        bytes[..4].copy_from_slice(b"PK\x03\x04");
        fs::write(&path, bytes).unwrap();
        let error = inspect_show_bundle(&path, BundleLimits::default(), &no_cancel()).unwrap_err();
        assert!(error.to_string().contains("ZIP/compressed"));
    }

    #[test]
    fn tamper_short_read_and_missing_original_fail_before_import_mutation() {
        let directory = TempDir::new("tamper");
        let bundle = directory.0.join("show.cosbundle");
        build_show_bundle(
            &bundle,
            fixture_request(&directory.0, b"tamper target"),
            BundleLimits::default(),
            &no_cancel(),
        )
        .unwrap();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&bundle)
            .unwrap();
        file.seek(SeekFrom::End(-1)).unwrap();
        file.write_all(&[0x5a]).unwrap();
        drop(file);
        assert!(inspect_show_bundle(&bundle, BundleLimits::default(), &no_cancel()).is_err());

        let short = directory.0.join("short.cosbundle");
        build_show_bundle(
            &short,
            fixture_request(&directory.0, b"short target"),
            BundleLimits::default(),
            &no_cancel(),
        )
        .unwrap();
        let length = fs::metadata(&short).unwrap().len();
        OpenOptions::new()
            .write(true)
            .open(&short)
            .unwrap()
            .set_len(length - 1)
            .unwrap();
        assert!(inspect_show_bundle(&short, BundleLimits::default(), &no_cancel()).is_err());

        let missing = directory.0.join("missing.cosbundle");
        let request = BundleBuildRequest {
            patch: fixture_patch("fixture-source"),
            media: Vec::new(),
            documents: Vec::new(),
            output_collision: BundleOutputCollision::Fail,
        };
        assert!(
            build_show_bundle(&missing, request, BundleLimits::default(), &no_cancel()).is_err()
        );
        assert!(!missing.exists());
    }

    #[test]
    fn secret_documents_and_authoritative_proxy_shapes_are_rejected() {
        let directory = TempDir::new("secret");
        let bundle = directory.0.join("secret.cosbundle");
        let mut request = fixture_request(&directory.0, b"original");
        request.documents.push(BundleDocumentInput {
            kind: BundleDocumentKind::ControllerProfile,
            logical_name: "controller.json".to_owned(),
            bytes: br#"{"access_token":"seeded-secret"}"#.to_vec(),
            license: None,
        });
        let error =
            build_show_bundle(&bundle, request, BundleLimits::default(), &no_cancel()).unwrap_err();
        assert!(error.to_string().contains("secret-bearing"));
        assert!(!bundle.exists());

        let mut proxy = skeletal_entry(
            "proxy/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            BundleEntryKind::Proxy,
            "proxy.mkv",
            'b',
        );
        proxy.original_sha256 = None;
        let manifest = skeletal_manifest(vec![
            skeletal_entry(
                "media/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                BundleEntryKind::OriginalMedia,
                "original.mp4",
                'a',
            ),
            skeletal_entry(
                "patch/show.yaml",
                BundleEntryKind::Patch,
                "show.cos.yaml",
                'c',
            ),
            proxy,
        ]);
        assert!(validate_manifest(&manifest, 0, BundleLimits::default()).is_err());
    }

    #[test]
    fn source_symlink_or_non_regular_input_is_rejected_without_following() {
        let directory = TempDir::new("nofollow");
        let target = directory.0.join("target.mp4");
        fs::write(&target, b"target").unwrap();
        let link = directory.0.join("link.mp4");
        #[cfg(windows)]
        let linked = std::os::windows::fs::symlink_file(&target, &link).is_ok();
        #[cfg(unix)]
        let linked = std::os::unix::fs::symlink(&target, &link).is_ok();
        #[cfg(not(any(windows, unix)))]
        let linked = false;
        if linked {
            let mut request = fixture_request(&directory.0, b"unused");
            request.media[0].source = link;
            assert!(build_show_bundle(
                &directory.0.join("link.cosbundle"),
                request,
                BundleLimits::default(),
                &no_cancel()
            )
            .is_err());
        }
        let mut request = fixture_request(&directory.0, b"unused");
        request.media[0].source = directory.0.clone();
        assert!(build_show_bundle(
            &directory.0.join("directory.cosbundle"),
            request,
            BundleLimits::default(),
            &no_cancel()
        )
        .is_err());
    }

    #[test]
    fn disk_full_cancel_and_prepublication_crash_publish_nothing() {
        let directory = TempDir::new("faults");
        let disk_full = directory.0.join("disk-full.cosbundle");
        let fail_write = |phase, _| {
            if phase == BundleFaultPhase::BuildWrite {
                Err(io::Error::other("injected disk full"))
            } else {
                Ok(())
            }
        };
        assert!(build_show_bundle_with_faults(
            &disk_full,
            fixture_request(&directory.0, b"payload"),
            BundleLimits::default(),
            &no_cancel(),
            &fail_write,
        )
        .is_err());
        assert!(!disk_full.exists());
        assert_no_staging(&directory.0);

        let cancelled = directory.0.join("cancelled.cosbundle");
        let cancel = AtomicBool::new(false);
        let set_cancel = |phase, _| {
            if phase == BundleFaultPhase::BuildWrite {
                cancel.store(true, Ordering::Release);
            }
            Ok(())
        };
        assert!(build_show_bundle_with_faults(
            &cancelled,
            fixture_request(&directory.0, b"payload"),
            BundleLimits::default(),
            &cancel,
            &set_cancel,
        )
        .is_err());
        assert!(!cancelled.exists());
        assert_no_staging(&directory.0);

        let crashed = directory.0.join("crashed.cosbundle");
        let fail_publish = |phase, _| {
            if phase == BundleFaultPhase::BuildBeforePublish {
                Err(io::Error::other("injected crash before publication"))
            } else {
                Ok(())
            }
        };
        assert!(build_show_bundle_with_faults(
            &crashed,
            fixture_request(&directory.0, b"payload"),
            BundleLimits::default(),
            &no_cancel(),
            &fail_publish,
        )
        .is_err());
        assert!(!crashed.exists());
        assert_no_staging(&directory.0);
    }

    #[test]
    fn final_name_races_never_overwrite_external_winners() {
        let directory = TempDir::new("races");
        let bundle = directory.0.join("race.cosbundle");
        let won = AtomicBool::new(false);
        let build_race = |phase, _| {
            if phase == BundleFaultPhase::BuildBeforePublish && !won.swap(true, Ordering::AcqRel) {
                fs::write(&bundle, b"external bundle winner")?;
            }
            Ok(())
        };
        assert!(build_show_bundle_with_faults(
            &bundle,
            fixture_request(&directory.0, b"payload"),
            BundleLimits::default(),
            &no_cancel(),
            &build_race,
        )
        .is_err());
        assert_eq!(fs::read(&bundle).unwrap(), b"external bundle winner");
        assert_no_staging(&directory.0);

        let valid = directory.0.join("valid.cosbundle");
        build_show_bundle(
            &valid,
            fixture_request(&directory.0, b"payload"),
            BundleLimits::default(),
            &no_cancel(),
        )
        .unwrap();
        let preview = inspect_show_bundle(&valid, BundleLimits::default(), &no_cancel()).unwrap();
        let library = directory.0.join("library");
        fs::create_dir(&library).unwrap();
        let winner = import_generation_path(&library, &preview.bundle_sha256);
        let import_won = AtomicBool::new(false);
        let import_race = |phase, _| {
            if phase == BundleFaultPhase::ImportBeforePublish
                && !import_won.swap(true, Ordering::AcqRel)
            {
                fs::create_dir(&winner)?;
                fs::write(winner.join("winner"), b"external import winner")?;
            }
            Ok(())
        };
        assert!(import_show_bundle_with_faults(
            &valid,
            &library,
            BundleImportCollision::Fail,
            BundleLimits::default(),
            &no_cancel(),
            &import_race,
        )
        .is_err());
        assert_eq!(
            fs::read(winner.join("winner")).unwrap(),
            b"external import winner"
        );
        assert_no_staging(&library);
    }

    #[test]
    fn import_disk_full_and_cancel_leave_no_partial_generation_and_cleanup_is_scoped() {
        let directory = TempDir::new("import-fault");
        let bundle = directory.0.join("show.cosbundle");
        build_show_bundle(
            &bundle,
            fixture_request(&directory.0, b"payload"),
            BundleLimits::default(),
            &no_cancel(),
        )
        .unwrap();
        let library = directory.0.join("library");
        fs::create_dir(&library).unwrap();
        let disk_full = |phase, _| {
            if phase == BundleFaultPhase::ImportWrite {
                Err(io::Error::other("injected disk full"))
            } else {
                Ok(())
            }
        };
        assert!(import_show_bundle_with_faults(
            &bundle,
            &library,
            BundleImportCollision::Fail,
            BundleLimits::default(),
            &no_cancel(),
            &disk_full,
        )
        .is_err());
        assert_eq!(fs::read_dir(&library).unwrap().count(), 0);

        let cancel = AtomicBool::new(false);
        let cancel_during_import = |phase, _| {
            if phase == BundleFaultPhase::ImportWrite {
                cancel.store(true, Ordering::Release);
            }
            Ok(())
        };
        assert!(import_show_bundle_with_faults(
            &bundle,
            &library,
            BundleImportCollision::Fail,
            BundleLimits::default(),
            &cancel,
            &cancel_during_import,
        )
        .is_err());
        assert_eq!(fs::read_dir(&library).unwrap().count(), 0);

        let orphan = library.join(format!("{IMPORT_STAGE_PREFIX}dead.part"));
        fs::create_dir(&orphan).unwrap();
        fs::write(orphan.join("partial"), b"partial").unwrap();
        let keep = library.join("operator-library");
        fs::create_dir(&keep).unwrap();
        assert_eq!(cleanup_show_bundle_orphans(&library, 16).unwrap(), 1);
        assert!(!orphan.exists());
        assert!(keep.exists());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(8))]

        #[test]
        fn random_media_bytes_produce_one_deterministic_bundle(bytes in proptest::collection::vec(any::<u8>(), 0..4096)) {
            let directory = TempDir::new("property");
            let first = directory.0.join("a.cosbundle");
            let second = directory.0.join("b.cosbundle");
            let cancel = no_cancel();
            build_show_bundle(
                &first,
                fixture_request(&directory.0, &bytes),
                BundleLimits::default(),
                &cancel,
            ).unwrap();
            build_show_bundle(
                &second,
                fixture_request(&directory.0, &bytes),
                BundleLimits::default(),
                &cancel,
            ).unwrap();
            prop_assert_eq!(fs::read(first).unwrap(), fs::read(second).unwrap());
        }
    }
}
