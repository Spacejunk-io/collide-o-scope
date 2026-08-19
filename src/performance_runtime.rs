//! Bounded, nonblocking preparation and atomic activation of performance media.
//!
//! CPU source opening and first-frame acquisition run on exactly two workers
//! behind a newest-only mailbox. GPU allocation/upload is intentionally polled
//! and driven by the render thread. A [`GpuCommitPayload`] contains no fallible
//! work: after one final topology/config validation, all target layers swap at
//! the same caller-supplied [`Instant`].

use std::collections::VecDeque;
use std::fmt;
use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::image_routing::StableLayerId;
use crate::layers::{
    is_still_image_file, DisplacedLayerSource, Layer, LayerSource, LayerSourceActivation,
    SPOUT_SOURCE_PREFIX,
};
use crate::media_safety::{
    MediaDeviceLimits, MediaSafetyPolicy, MediaSourceKind, PerformanceResourceBudget,
};
use crate::media_source::{
    resolve_visual_source, FingerprintLimits, FingerprintSession, ResolveContext,
    ResolvedVisualSource,
};
use crate::performance::{ClipSlotConfig, ClipSlotId, Scene, SceneId, MAX_SCENE_BINDINGS};
use crate::spout_in::SpoutIn;
use crate::transport::{CueId, NormalizedTime};
use crate::video::{
    decode_still_image_with_media_policy, SeedSelectError, StillImage, ThreadedDecoder,
};

pub const DEFAULT_SPOUT_PREPARE_TIMEOUT: Duration = Duration::from_secs(5);
const SPOUT_PREPARE_POLL_INTERVAL: Duration = Duration::from_millis(10);
const VIDEO_PREPARE_SELECT_TIMEOUT: Duration = Duration::from_secs(5);
const VIDEO_PREPARE_POLL_INTERVAL: Duration = Duration::from_millis(2);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PreparationGeneration(u64);

impl PreparationGeneration {
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PreparedTransactionKey {
    LayerSlot {
        layer_id: StableLayerId,
        slot_id: ClipSlotId,
    },
    Scene(SceneId),
}

#[derive(Debug, Clone)]
pub enum ResolvedPreparedSource {
    /// Resolve/fingerprint the authored identity entirely on a staging worker.
    /// Main supplies immutable context only; it never hashes inactive clips on
    /// the render thread.
    Authored {
        source_path: String,
        filename: String,
        context: ResolveContext,
    },
    /// `path` is the already resolved host path. A content-addressed or other
    /// authored identity can be retained independently for later persistence.
    #[cfg(test)]
    File {
        path: PathBuf,
        persisted_reference: Option<String>,
    },
    Spout {
        sender: String,
    },
}

impl ResolvedPreparedSource {
    pub fn authored(
        source_path: impl Into<String>,
        filename: impl Into<String>,
        context: ResolveContext,
    ) -> Self {
        Self::Authored {
            source_path: source_path.into(),
            filename: filename.into(),
            context,
        }
    }

    #[cfg(test)]
    pub fn file(path: impl Into<PathBuf>) -> Self {
        Self::File {
            path: path.into(),
            persisted_reference: None,
        }
    }

    pub fn spout(sender: impl Into<String>) -> Self {
        Self::Spout {
            sender: sender.into(),
        }
    }

    fn validate(&self) -> Result<(), String> {
        match self {
            Self::Authored {
                source_path,
                filename,
                ..
            } if source_path.is_empty() && filename.is_empty() => {
                Err("authored source path and filename are both empty".to_string())
            }
            #[cfg(test)]
            Self::File { path, .. } if path.as_os_str().is_empty() => {
                Err("resolved media path is empty".to_string())
            }
            Self::Spout { sender } if sender.trim().is_empty() => {
                Err("resolved Spout sender is empty".to_string())
            }
            Self::Authored { .. } | Self::Spout { .. } => Ok(()),
            #[cfg(test)]
            Self::File { .. } => Ok(()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SourcePreparationRequest {
    pub target_layer: StableLayerId,
    /// Exact slot state observed before staging. `None` means a newly authored
    /// slot which must remain absent from the live Set until atomic commit.
    pub expected_prior_slot: Option<ClipSlotConfig>,
    /// Metadata installed at the same instant as the prepared pixels/source.
    pub desired_slot: ClipSlotConfig,
    pub cue_id: Option<CueId>,
    pub resolved_source: ResolvedPreparedSource,
    start_position: NormalizedTime,
}

impl SourcePreparationRequest {
    pub fn new(
        target_layer: StableLayerId,
        expected_prior_slot: Option<ClipSlotConfig>,
        desired_slot: ClipSlotConfig,
        cue_id: Option<CueId>,
        resolved_source: ResolvedPreparedSource,
    ) -> Result<Self, PreparationRequestError> {
        if expected_prior_slot
            .as_ref()
            .is_some_and(|prior| prior.id != desired_slot.id)
        {
            return Err(PreparationRequestError::invalid(
                "expected and desired clip slot IDs differ",
            ));
        }
        resolved_source
            .validate()
            .map_err(PreparationRequestError::invalid)?;
        let start_position = match cue_id {
            Some(cue_id) => desired_slot
                .transport
                .cue(cue_id)
                .map(|cue| cue.at)
                .ok_or_else(|| {
                    PreparationRequestError::invalid(format!(
                        "cue {} is absent from clip slot {}",
                        cue_id.get(),
                        desired_slot.id.get()
                    ))
                })?,
            None => desired_slot.saved_playhead,
        };
        Ok(Self {
            target_layer,
            expected_prior_slot,
            desired_slot,
            cue_id,
            resolved_source,
            start_position,
        })
    }
}

#[derive(Debug, Clone)]
pub struct PreparationRequest {
    pub key: PreparedTransactionKey,
    replacements: Vec<SourcePreparationRequest>,
}

impl PreparationRequest {
    pub fn from_replacements(
        key: PreparedTransactionKey,
        replacements: Vec<SourcePreparationRequest>,
    ) -> Result<Self, PreparationRequestError> {
        if replacements.len() > MAX_SCENE_BINDINGS {
            return Err(PreparationRequestError::invalid(format!(
                "a preparation transaction may address at most {MAX_SCENE_BINDINGS} layers"
            )));
        }
        let mut identities = Vec::with_capacity(replacements.len());
        for replacement in &replacements {
            replacement
                .resolved_source
                .validate()
                .map_err(PreparationRequestError::invalid)?;
            if identities.contains(&replacement.target_layer) {
                return Err(PreparationRequestError::invalid(format!(
                    "layer {} appears more than once in one preparation transaction",
                    replacement.target_layer.get()
                )));
            }
            identities.push(replacement.target_layer);
        }
        Ok(Self { key, replacements })
    }

    pub fn for_layer_slot(
        layer: &Layer,
        slot_id: ClipSlotId,
        cue_id: Option<CueId>,
        resolved_source: ResolvedPreparedSource,
    ) -> Result<Self, PreparationRequestError> {
        let slot = layer.clip_slots.get(slot_id).cloned().ok_or_else(|| {
            PreparationRequestError::invalid(format!(
                "clip slot {} is absent from layer {}",
                slot_id.get(),
                layer.layer_id()
            ))
        })?;
        let target_layer = layer.stable_layer_id();
        let replacement = SourcePreparationRequest::new(
            target_layer,
            Some(slot.clone()),
            slot,
            cue_id,
            resolved_source,
        )?;
        Self::from_replacements(
            PreparedTransactionKey::LayerSlot {
                layer_id: target_layer,
                slot_id,
            },
            vec![replacement],
        )
    }

    /// Stage an existing-slot replacement or a brand-new slot without first
    /// exposing its metadata in the live layer. `desired_slot` is cached only
    /// inside the request until the final source swap.
    pub fn for_layer_slot_update(
        layer: &Layer,
        desired_slot: ClipSlotConfig,
        cue_id: Option<CueId>,
        resolved_source: ResolvedPreparedSource,
    ) -> Result<Self, PreparationRequestError> {
        let target_layer = layer.stable_layer_id();
        let slot_id = desired_slot.id;
        let expected_prior_slot = layer.clip_slots.get(slot_id).cloned();
        if expected_prior_slot.is_none()
            && layer.clip_slots.len() == crate::performance::MAX_CLIP_SLOTS_PER_LAYER
        {
            return Err(PreparationRequestError::invalid(format!(
                "layer {} already contains the maximum number of clip slots",
                layer.layer_id()
            )));
        }
        let replacement = SourcePreparationRequest::new(
            target_layer,
            expected_prior_slot,
            desired_slot,
            cue_id,
            resolved_source,
        )?;
        Self::from_replacements(
            PreparedTransactionKey::LayerSlot {
                layer_id: target_layer,
                slot_id,
            },
            vec![replacement],
        )
    }

    /// Resolve saved positions to the current stable identities, validate every
    /// slot/cue reference, then ask the caller to resolve authored source
    /// identities into explicit host paths or a Spout sender.
    pub fn for_scene(
        scene: &Scene,
        layers: &[Layer],
        mut resolve: impl FnMut(
            StableLayerId,
            &ClipSlotConfig,
        ) -> Result<ResolvedPreparedSource, String>,
    ) -> Result<Self, PreparationRequestError> {
        let mut replacements = Vec::with_capacity(scene.bindings.len());
        for binding in scene.bindings.iter() {
            let position = usize::try_from(binding.layer_position.get()).map_err(|_| {
                PreparationRequestError::invalid(format!(
                    "scene {} layer position {} is not addressable on this host",
                    scene.id.get(),
                    binding.layer_position.get()
                ))
            })?;
            let layer = layers.get(position).ok_or_else(|| {
                PreparationRequestError::invalid(format!(
                    "scene {} references missing layer position {}",
                    scene.id.get(),
                    binding.layer_position.get()
                ))
            })?;
            let slot = layer
                .clip_slots
                .get(binding.slot_id)
                .cloned()
                .ok_or_else(|| {
                    PreparationRequestError::invalid(format!(
                        "scene {} references missing slot {} on layer {}",
                        scene.id.get(),
                        binding.slot_id.get(),
                        layer.layer_id()
                    ))
                })?;
            if let Some(cue_id) = binding.cue_id {
                if slot.transport.cue(cue_id).is_none() {
                    return Err(PreparationRequestError::invalid(format!(
                        "scene {} references missing cue {} in slot {} on layer {}",
                        scene.id.get(),
                        cue_id.get(),
                        slot.id.get(),
                        layer.layer_id()
                    )));
                }
            }
            let layer_id = layer.stable_layer_id();
            let resolved_source = resolve(layer_id, &slot).map_err(|error| {
                PreparationRequestError::invalid(format!(
                    "cannot resolve scene {} slot {} for layer {}: {error}",
                    scene.id.get(),
                    slot.id.get(),
                    layer.layer_id()
                ))
            })?;
            replacements.push(SourcePreparationRequest::new(
                layer_id,
                Some(slot.clone()),
                slot,
                binding.cue_id,
                resolved_source,
            )?);
        }
        Self::from_replacements(PreparedTransactionKey::Scene(scene.id), replacements)
    }

    pub fn replacement_count(&self) -> usize {
        self.replacements.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparationRequestError {
    pub message: String,
}

impl PreparationRequestError {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for PreparationRequestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for PreparationRequestError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreparationFailureKind {
    InvalidRequest,
    SourceOpen,
    SourceTimeout,
    ResourceBudget,
    GpuActivation,
    CommitConflict,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparationFailure {
    pub generation: PreparationGeneration,
    pub key: PreparedTransactionKey,
    pub kind: PreparationFailureKind,
    pub message: String,
}

impl fmt::Display for PreparationFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for PreparationFailure {}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PreparedBudgetSnapshot {
    pub source_count: usize,
    pub preload_bytes: u64,
}

struct ResourceLedger {
    budget: PerformanceResourceBudget,
    state: Mutex<PreparedBudgetSnapshot>,
}

impl ResourceLedger {
    fn new(budget: PerformanceResourceBudget) -> Arc<Self> {
        Arc::new(Self {
            budget,
            state: Mutex::new(PreparedBudgetSnapshot::default()),
        })
    }

    fn acquire(self: &Arc<Self>, bytes: u64) -> Result<PreparedLease, &'static str> {
        let mut state = lock_recover(&self.state);
        self.budget.admit_prepared_source(state.source_count)?;
        let preload_bytes = self
            .budget
            .admit_preload_bytes(state.preload_bytes, bytes)?;
        state.source_count = state
            .source_count
            .checked_add(1)
            .ok_or("prepared source accounting overflow")?;
        state.preload_bytes = preload_bytes;
        drop(state);
        Ok(PreparedLease {
            ledger: self.clone(),
            bytes,
        })
    }

    fn snapshot(&self) -> PreparedBudgetSnapshot {
        *lock_recover(&self.state)
    }
}

struct PreparedLease {
    ledger: Arc<ResourceLedger>,
    bytes: u64,
}

impl PreparedLease {
    /// Reassign an existing prepared-source slot to a different owned source.
    /// Source count is unchanged; only the conservative byte charge moves.
    /// Failure leaves both the lease and aggregate ledger byte-for-byte intact.
    fn try_resize(&mut self, bytes: u64) -> Result<(), &'static str> {
        let mut state = lock_recover(&self.ledger.state);
        let without_current = state
            .preload_bytes
            .checked_sub(self.bytes)
            .ok_or("prepared source byte ledger underflow")?;
        let resized = self
            .ledger
            .budget
            .admit_preload_bytes(without_current, bytes)?;
        state.preload_bytes = resized;
        self.bytes = bytes;
        Ok(())
    }
}

impl Drop for PreparedLease {
    fn drop(&mut self) {
        let mut state = lock_recover(&self.ledger.state);
        state.source_count = state.source_count.saturating_sub(1);
        state.preload_bytes = state.preload_bytes.saturating_sub(self.bytes);
    }
}

enum CpuLayerSource {
    Video(ThreadedDecoder),
    Still(StillImage),
    Spout(SpoutIn),
    #[cfg(test)]
    Synthetic(StillImage),
}

impl CpuLayerSource {
    fn into_layer_source(self) -> Result<LayerSource, String> {
        match self {
            Self::Video(decoder) => Ok(LayerSource::Video(decoder)),
            Self::Still(image) => Ok(LayerSource::Still(image)),
            Self::Spout(receiver) => Ok(LayerSource::Spout(receiver)),
            #[cfg(test)]
            Self::Synthetic(image) => Ok(LayerSource::Still(image)),
        }
    }
}

struct CpuPreparedSource {
    source: CpuLayerSource,
    runtime_source_path: String,
    persisted_reference: Option<String>,
    width: u32,
    height: u32,
    source_fps: f32,
    first_rgba: Vec<u8>,
    preload_bytes: u64,
}

impl CpuPreparedSource {
    fn validate(&self) -> Result<(), String> {
        let expected = u64::from(self.width)
            .checked_mul(u64::from(self.height))
            .and_then(|pixels| pixels.checked_mul(4))
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| {
                format!(
                    "prepared source dimensions overflow for {}x{}",
                    self.width, self.height
                )
            })?;
        if self.width == 0 || self.height == 0 || self.first_rgba.len() != expected {
            return Err(format!(
                "prepared source supplied {} bytes; expected {expected} for {}x{}",
                self.first_rgba.len(),
                self.width,
                self.height
            ));
        }
        Ok(())
    }
}

enum SourcePrepareError {
    Superseded,
    Failed(PreparationFailureKind, String),
}

trait SourcePreparer: Send + Sync {
    fn prepare(
        &self,
        source: &ResolvedPreparedSource,
        start_position: NormalizedTime,
        cancel: Arc<AtomicBool>,
        is_current: &dyn Fn() -> bool,
    ) -> Result<CpuPreparedSource, SourcePrepareError>;
}

struct SystemSourcePreparer {
    media_policy: MediaSafetyPolicy,
    device_limits: MediaDeviceLimits,
    spout_timeout: Duration,
    fingerprint_limits: FingerprintLimits,
}

impl SourcePreparer for SystemSourcePreparer {
    fn prepare(
        &self,
        source: &ResolvedPreparedSource,
        start_position: NormalizedTime,
        cancel: Arc<AtomicBool>,
        is_current: &dyn Fn() -> bool,
    ) -> Result<CpuPreparedSource, SourcePrepareError> {
        if !is_current() {
            return Err(SourcePrepareError::Superseded);
        }
        match source {
            ResolvedPreparedSource::Authored {
                source_path,
                filename,
                context,
            } => self.prepare_authored(
                source_path,
                filename,
                context,
                start_position,
                cancel,
                is_current,
            ),
            #[cfg(test)]
            ResolvedPreparedSource::File {
                path,
                persisted_reference,
            } => self.prepare_file(
                path,
                persisted_reference.clone(),
                start_position,
                is_current,
            ),
            ResolvedPreparedSource::Spout { sender } => self.prepare_spout(sender, is_current),
        }
    }
}

impl SystemSourcePreparer {
    fn prepare_authored(
        &self,
        source_path: &str,
        filename: &str,
        context: &ResolveContext,
        start_position: NormalizedTime,
        cancel: Arc<AtomicBool>,
        is_current: &dyn Fn() -> bool,
    ) -> Result<CpuPreparedSource, SourcePrepareError> {
        // `resolve_visual_source` recognizes spout:// before performing any
        // path or fingerprint operation, preserving a strict no-filesystem
        // live-input path.
        let mut fingerprints = FingerprintSession::with_cancel(
            self.fingerprint_limits,
            Some(cancel),
        )
        .map_err(|error| {
            SourcePrepareError::Failed(PreparationFailureKind::SourceOpen, error.to_string())
        })?;
        let resolved = resolve_visual_source(
            source_path,
            filename,
            context,
            None,
            crate::layers::is_supported_visual_file,
            &mut fingerprints,
        )
        .map_err(|error| match error {
            crate::media_source::SourceResolveError::Cancelled => SourcePrepareError::Superseded,
            error => SourcePrepareError::Failed(
                PreparationFailureKind::SourceOpen,
                format!("cannot resolve prepared source '{filename}': {error}"),
            ),
        })?;
        if !is_current() {
            return Err(SourcePrepareError::Superseded);
        }
        match resolved {
            ResolvedVisualSource::Spout { sender } => self.prepare_spout(&sender, is_current),
            ResolvedVisualSource::File(file) => {
                let persisted_reference = file.identity.map(|identity| identity.source_reference());
                self.prepare_file(&file.path, persisted_reference, start_position, is_current)
            }
        }
    }

    fn prepare_file(
        &self,
        path: &Path,
        persisted_reference: Option<String>,
        start_position: NormalizedTime,
        is_current: &dyn Fn() -> bool,
    ) -> Result<CpuPreparedSource, SourcePrepareError> {
        let path_text = path.to_str().ok_or_else(|| {
            SourcePrepareError::Failed(
                PreparationFailureKind::SourceOpen,
                format!(
                    "resolved media path is not valid Unicode: {}",
                    path.display()
                ),
            )
        })?;
        let runtime_source_path = std::fs::canonicalize(path)
            .unwrap_or_else(|_| path.to_path_buf())
            .to_string_lossy()
            .into_owned();

        if is_still_image_file(path) {
            let decoded =
                decode_still_image_with_media_policy(path, &self.media_policy, self.device_limits)
                    .map_err(|error| {
                        SourcePrepareError::Failed(PreparationFailureKind::SourceOpen, error)
                    })?;
            if !is_current() {
                return Err(SourcePrepareError::Superseded);
            }
            let width = decoded.width;
            let height = decoded.height;
            let preload_bytes = decoded.media_allocation_plan().working_set_bytes;
            let mut image = StillImage::from_decoded(decoded);
            let first_rgba = image.take_frame().ok_or_else(|| {
                SourcePrepareError::Failed(
                    PreparationFailureKind::SourceOpen,
                    format!("still image {} produced no first frame", path.display()),
                )
            })?;
            return Ok(CpuPreparedSource {
                source: CpuLayerSource::Still(image),
                runtime_source_path,
                persisted_reference,
                width,
                height,
                source_fps: 30.0,
                first_rgba,
                preload_bytes,
            });
        }

        let mut decoder = ThreadedDecoder::open_with_media_policy(
            path_text,
            &self.media_policy,
            self.device_limits,
        )
        .map_err(|error| SourcePrepareError::Failed(PreparationFailureKind::SourceOpen, error))?;
        if !is_current() {
            return Err(SourcePrepareError::Superseded);
        }
        let first = decoder
            .select_seed_frame_at(
                start_position.get(),
                VIDEO_PREPARE_SELECT_TIMEOUT,
                VIDEO_PREPARE_POLL_INTERVAL,
                is_current,
            )
            .map_err(|error| match error {
                SeedSelectError::Superseded => SourcePrepareError::Superseded,
                SeedSelectError::NoSeedFrame => SourcePrepareError::Failed(
                    PreparationFailureKind::SourceOpen,
                    format!(
                        "video {} opened without a decoded first frame",
                        path.display()
                    ),
                ),
                SeedSelectError::Decode(error) => {
                    SourcePrepareError::Failed(PreparationFailureKind::SourceOpen, error)
                }
                SeedSelectError::Timeout { target_seconds } => SourcePrepareError::Failed(
                    PreparationFailureKind::SourceTimeout,
                    format!(
                        "video {} did not produce source position {:.6}s within {:.1}s",
                        path.display(),
                        target_seconds,
                        VIDEO_PREPARE_SELECT_TIMEOUT.as_secs_f64()
                    ),
                ),
            })?;
        let preload_bytes = decoder.media_allocation_plan().working_set_bytes;
        Ok(CpuPreparedSource {
            width: decoder.width,
            height: decoder.height,
            source_fps: decoder.fps,
            source: CpuLayerSource::Video(decoder),
            runtime_source_path,
            persisted_reference,
            first_rgba: first.rgba,
            preload_bytes,
        })
    }

    fn prepare_spout(
        &self,
        sender: &str,
        is_current: &dyn Fn() -> bool,
    ) -> Result<CpuPreparedSource, SourcePrepareError> {
        let mut receiver =
            SpoutIn::new_with_media_policy(sender, self.media_policy.clone(), self.device_limits);
        let sanitized_sender = receiver.status().sender_name;
        if sanitized_sender.is_empty() {
            return Err(SourcePrepareError::Failed(
                PreparationFailureKind::SourceOpen,
                "Spout sender name is empty after sanitization".to_string(),
            ));
        }
        receiver.start();
        let started = Instant::now();
        loop {
            if !is_current() {
                return Err(SourcePrepareError::Superseded);
            }
            if let Some(frame) = receiver.try_recv() {
                let plan = self
                    .media_policy
                    .plan(
                        MediaSourceKind::Spout,
                        frame.width,
                        frame.height,
                        self.device_limits,
                    )
                    .map_err(|error| {
                        SourcePrepareError::Failed(
                            PreparationFailureKind::SourceOpen,
                            format!("Spout source rejected before activation: {error}"),
                        )
                    })?;
                let prepared = CpuPreparedSource {
                    source: CpuLayerSource::Spout(receiver),
                    runtime_source_path: format!("{SPOUT_SOURCE_PREFIX}{sanitized_sender}"),
                    persisted_reference: None,
                    width: frame.width,
                    height: frame.height,
                    source_fps: 30.0,
                    first_rgba: frame.pixels,
                    preload_bytes: plan.working_set_bytes,
                };
                prepared.validate().map_err(|error| {
                    SourcePrepareError::Failed(PreparationFailureKind::SourceOpen, error)
                })?;
                // There is deliberately no 1x1-black fallback here. Readiness
                // exists only after `try_recv` supplied a real sequence frame.
                return Ok(prepared);
            }

            let status = receiver.status();
            if !receiver.is_running() && !status.error.is_empty() {
                return Err(SourcePrepareError::Failed(
                    PreparationFailureKind::SourceOpen,
                    format!(
                        "Spout sender '{sanitized_sender}' is unavailable: {}",
                        status.error
                    ),
                ));
            }
            if started.elapsed() >= self.spout_timeout {
                let detail = if status.error.is_empty() {
                    "no valid frame arrived".to_string()
                } else {
                    status.error
                };
                return Err(SourcePrepareError::Failed(
                    PreparationFailureKind::SourceTimeout,
                    format!(
                        "Spout sender '{sanitized_sender}' did not publish a valid frame within {:.1}s: {detail}",
                        self.spout_timeout.as_secs_f64()
                    ),
                ));
            }
            std::thread::sleep(SPOUT_PREPARE_POLL_INTERVAL);
        }
    }
}

struct PreparedReplacement {
    target_layer: StableLayerId,
    expected_prior_slot: Option<ClipSlotConfig>,
    desired_slot: ClipSlotConfig,
    cue_id: Option<CueId>,
    start_position: NormalizedTime,
    prepared: CpuPreparedSource,
    lease: PreparedLease,
}

pub struct PreparedTransaction {
    generation: PreparationGeneration,
    key: PreparedTransactionKey,
    replacements: Vec<PreparedReplacement>,
}

impl PreparedTransaction {
    pub fn key(&self) -> &PreparedTransactionKey {
        &self.key
    }

    pub fn preload_bytes(&self) -> u64 {
        self.replacements
            .iter()
            .map(|replacement| replacement.lease.bytes)
            .sum()
    }

    /// Allocate and upload every replacement texture without mutating a live
    /// layer. A failure drops only staged resources; current output is intact.
    pub fn activate_gpu(
        self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layers: &[Layer],
    ) -> Result<GpuCommitPayload, PreparationFailure> {
        if let Err(message) = validate_live_targets(&self.replacements, layers) {
            return Err(PreparationFailure {
                generation: self.generation,
                key: self.key,
                kind: PreparationFailureKind::CommitConflict,
                message,
            });
        }

        let generation = self.generation;
        let key = self.key;
        let mut replacements = Vec::with_capacity(self.replacements.len());
        for replacement in self.replacements {
            let PreparedReplacement {
                target_layer,
                expected_prior_slot,
                desired_slot,
                cue_id,
                start_position,
                prepared,
                lease,
            } = replacement;
            let CpuPreparedSource {
                source,
                runtime_source_path,
                persisted_reference,
                width,
                height,
                source_fps,
                first_rgba,
                ..
            } = prepared;
            let source = source
                .into_layer_source()
                .map_err(|message| PreparationFailure {
                    generation,
                    key: key.clone(),
                    kind: PreparationFailureKind::GpuActivation,
                    message,
                })?;
            let activation = LayerSourceActivation::stage(
                device,
                queue,
                runtime_source_path,
                persisted_reference,
                desired_slot.filename.clone(),
                source,
                width,
                height,
                source_fps,
                lease.bytes,
                &first_rgba,
            )
            .map_err(|message| PreparationFailure {
                generation,
                key: key.clone(),
                kind: PreparationFailureKind::GpuActivation,
                message,
            })?;
            replacements.push(GpuReplacement {
                target_layer,
                expected_prior_slot,
                desired_slot,
                cue_id,
                start_position,
                activation,
                lease,
            });
        }
        Ok(GpuCommitPayload {
            generation,
            key,
            replacements,
        })
    }

    #[cfg(test)]
    fn first_rgba(&self) -> Option<&[u8]> {
        self.replacements
            .first()
            .map(|replacement| replacement.prepared.first_rgba.as_slice())
    }

    #[cfg(test)]
    fn first_source_identity(&self) -> Option<(&str, Option<&str>)> {
        self.replacements.first().map(|replacement| {
            (
                replacement.prepared.runtime_source_path.as_str(),
                replacement.prepared.persisted_reference.as_deref(),
            )
        })
    }

    #[cfg(test)]
    fn first_slot_transition(&self) -> Option<(Option<&ClipSlotConfig>, &ClipSlotConfig)> {
        self.replacements.first().map(|replacement| {
            (
                replacement.expected_prior_slot.as_ref(),
                &replacement.desired_slot,
            )
        })
    }
}

struct GpuReplacement {
    target_layer: StableLayerId,
    expected_prior_slot: Option<ClipSlotConfig>,
    desired_slot: ClipSlotConfig,
    #[allow(dead_code)]
    cue_id: Option<CueId>,
    start_position: NormalizedTime,
    activation: LayerSourceActivation,
    lease: PreparedLease,
}

pub struct GpuCommitPayload {
    generation: PreparationGeneration,
    key: PreparedTransactionKey,
    replacements: Vec<GpuReplacement>,
}

impl GpuCommitPayload {
    pub fn key(&self) -> &PreparedTransactionKey {
        &self.key
    }

    /// Aggregate conservative working-set charge carried by this inactive
    /// payload. It is derived from the same leases that enforce the hard host
    /// budget, not from a second best-effort estimate.
    pub fn preload_bytes(&self) -> u64 {
        self.replacements
            .iter()
            .map(|replacement| replacement.lease.bytes)
            .sum()
    }

    /// Descriptor for the one-source LayerSlot payloads produced when an
    /// active source is displaced. Scene payloads intentionally return None.
    pub fn single_layer_slot_descriptor(
        &self,
    ) -> Option<(StableLayerId, ClipSlotConfig, Option<CueId>)> {
        if self.replacements.len() != 1 {
            return None;
        }
        let replacement = &self.replacements[0];
        match &self.key {
            PreparedTransactionKey::LayerSlot { layer_id, slot_id }
                if *layer_id == replacement.target_layer
                    && *slot_id == replacement.desired_slot.id =>
            {
                Some((
                    *layer_id,
                    replacement.desired_slot.clone(),
                    replacement.cue_id,
                ))
            }
            PreparedTransactionKey::LayerSlot { .. } | PreparedTransactionKey::Scene(_) => None,
        }
    }

    /// Publish metadata for a successfully prepared inactive source without
    /// activating its pixels yet. Validation is completed for the whole
    /// transaction before any live Set is changed, and active-slot
    /// replacement is deliberately rejected: callers that requested
    /// activation must use [`Self::commit`] instead.
    ///
    /// After publication the payload's expected state advances to the newly
    /// authored metadata, so a later quantized activation still performs the
    /// same exact conflict check and remains all-or-nothing.
    pub fn publish_inactive_slot_metadata(
        &mut self,
        layers: &mut [Layer],
    ) -> Result<Vec<StableLayerId>, String> {
        let positions = validate_gpu_targets(&self.replacements, layers)?;
        let mut staged_sets = Vec::with_capacity(self.replacements.len());
        for (replacement, position) in self.replacements.iter().zip(positions.iter().copied()) {
            if layers[position].active_clip_slot == replacement.desired_slot.id {
                return Err(format!(
                    "cannot publish inactive metadata over active slot {} on layer {}",
                    replacement.desired_slot.id.get(),
                    replacement.target_layer.get()
                ));
            }
            let mut slots = layers[position].clip_slots.clone();
            slots
                .upsert(replacement.desired_slot.clone())
                .map_err(|error| error.to_string())?;
            staged_sets.push((position, slots));
        }

        for (position, slots) in staged_sets {
            layers[position].clip_slots = slots;
        }
        let mut published = Vec::with_capacity(self.replacements.len());
        for replacement in &mut self.replacements {
            replacement.expected_prior_slot = Some(replacement.desired_slot.clone());
            published.push(replacement.target_layer);
        }
        Ok(published)
    }

    /// Commit all source swaps at one logical instant. The topology and slot
    /// configs are validated for the second time before any mutation, so a
    /// reorder is harmless and a deletion/edit rejects the whole payload.
    pub fn commit(
        self,
        layers: &mut [Layer],
        committed_at: Instant,
    ) -> Result<CommitReceipt, CommitRejected> {
        let positions = match validate_gpu_targets(&self.replacements, layers) {
            Ok(positions) => positions,
            Err(message) => return Err(CommitRejected::new(self, message)),
        };

        let generation = self.generation;
        let key = self.key;
        let mut committed_layers = Vec::with_capacity(self.replacements.len());
        let mut displaced_sources = Vec::with_capacity(self.replacements.len());
        let mut uncached_displaced_slots = Vec::new();
        for (replacement, position) in self.replacements.into_iter().zip(positions) {
            let GpuReplacement {
                target_layer,
                desired_slot,
                start_position,
                activation,
                lease,
                ..
            } = replacement;
            let displaced = layers[position].commit_prepared_source(
                activation,
                &desired_slot,
                start_position,
                committed_at,
            );
            committed_layers.push(target_layer);

            // A same-slot source replacement makes the displaced revision
            // obsolete by definition. For an actual slot switch, transfer the
            // incoming source's lease to the exact old decoder/texture. This
            // keeps A→B→A GPU-ready without increasing prepared-source count.
            if displaced.slot.id == desired_slot.id {
                drop(lease);
                drop(displaced);
                continue;
            }
            match displaced_payload(generation, displaced, lease) {
                Ok(payload) => displaced_sources.push(payload),
                Err((layer_id, slot_id)) => {
                    // The new active output has already committed and remains
                    // valid. Only the now-inactive old source is evicted when
                    // its larger working set cannot fit the aggregate budget.
                    uncached_displaced_slots.push((layer_id, slot_id));
                }
            }
        }
        Ok(CommitReceipt {
            key,
            committed_layers,
            displaced_sources,
            uncached_displaced_slots,
        })
    }
}

fn displaced_payload(
    generation: PreparationGeneration,
    displaced: DisplacedLayerSource,
    mut lease: PreparedLease,
) -> Result<GpuCommitPayload, (StableLayerId, ClipSlotId)> {
    let DisplacedLayerSource {
        target_layer,
        slot,
        start_position,
        activation,
    } = displaced;
    let slot_id = slot.id;
    if lease.try_resize(activation.preload_bytes()).is_err() {
        return Err((target_layer, slot_id));
    }
    Ok(GpuCommitPayload {
        generation,
        key: PreparedTransactionKey::LayerSlot {
            layer_id: target_layer,
            slot_id,
        },
        replacements: vec![GpuReplacement {
            target_layer,
            expected_prior_slot: Some(slot.clone()),
            desired_slot: slot,
            cue_id: None,
            start_position,
            activation,
            lease,
        }],
    })
}

pub struct CommitRejected {
    pub failure: PreparationFailure,
    /// Retains the fully prepared resources so diagnostic/test callers can
    /// prove a rejected atomic commit did not corrupt or consume the payload.
    #[allow(dead_code)]
    payload: GpuCommitPayload,
}

impl CommitRejected {
    fn new(payload: GpuCommitPayload, message: String) -> Self {
        let failure = PreparationFailure {
            generation: payload.generation,
            key: payload.key.clone(),
            kind: PreparationFailureKind::CommitConflict,
            message,
        };
        Self { failure, payload }
    }

    #[allow(dead_code)]
    pub fn into_payload(self) -> GpuCommitPayload {
        self.payload
    }
}

pub struct CommitReceipt {
    pub key: PreparedTransactionKey,
    pub committed_layers: Vec<StableLayerId>,
    /// Exact active sources displaced by this commit and retained under the
    /// same hard prepared-resource ledger. Each is a one-source LayerSlot
    /// payload ready for an infallible return switch.
    pub displaced_sources: Vec<GpuCommitPayload>,
    /// Displaced slots dropped because their working set could not replace the
    /// incoming lease within the aggregate byte budget. A future activation
    /// safely reopens these sources; current output is never affected.
    pub uncached_displaced_slots: Vec<(StableLayerId, ClipSlotId)>,
}

fn validate_live_targets(
    replacements: &[PreparedReplacement],
    layers: &[Layer],
) -> Result<(), String> {
    for replacement in replacements {
        let layer = layers
            .iter()
            .find(|layer| layer.stable_layer_id() == replacement.target_layer)
            .ok_or_else(|| {
                format!(
                    "prepared target layer {} no longer exists",
                    replacement.target_layer.get()
                )
            })?;
        if !layer.prepared_slot_state_matches(
            replacement.desired_slot.id,
            replacement.expected_prior_slot.as_ref(),
        ) {
            return Err(format!(
                "clip slot {} changed while preparing layer {}",
                replacement.desired_slot.id.get(),
                replacement.target_layer.get()
            ));
        }
    }
    Ok(())
}

fn validate_gpu_targets(
    replacements: &[GpuReplacement],
    layers: &[Layer],
) -> Result<Vec<usize>, String> {
    let mut positions = Vec::with_capacity(replacements.len());
    for replacement in replacements {
        let position = layers
            .iter()
            .position(|layer| layer.stable_layer_id() == replacement.target_layer)
            .ok_or_else(|| {
                format!(
                    "prepared target layer {} no longer exists",
                    replacement.target_layer.get()
                )
            })?;
        if !layers[position].prepared_slot_state_matches(
            replacement.desired_slot.id,
            replacement.expected_prior_slot.as_ref(),
        ) {
            return Err(format!(
                "clip slot {} changed while preparing layer {}",
                replacement.desired_slot.id.get(),
                replacement.target_layer.get()
            ));
        }
        positions.push(position);
    }
    Ok(positions)
}

struct PreparationJob {
    generation: PreparationGeneration,
    request: PreparationRequest,
    cancel: Arc<AtomicBool>,
}

struct WorkerCompletion {
    generation: PreparationGeneration,
    result: Result<PreparedTransaction, PreparationFailure>,
}

struct JobMailboxState {
    pending: Option<PreparationJob>,
    current_cancel: Option<Arc<AtomicBool>>,
    stopped: bool,
}

struct JobMailbox {
    state: Mutex<JobMailboxState>,
    wake: Condvar,
    latest_generation: AtomicU64,
    stopped: AtomicBool,
}

impl JobMailbox {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(JobMailboxState {
                pending: None,
                current_cancel: None,
                stopped: false,
            }),
            wake: Condvar::new(),
            latest_generation: AtomicU64::new(0),
            stopped: AtomicBool::new(false),
        })
    }

    fn submit(&self, job: PreparationJob) {
        self.latest_generation
            .store(job.generation.get(), Ordering::Release);
        let mut state = lock_recover(&self.state);
        if let Some(cancel) = state.current_cancel.replace(job.cancel.clone()) {
            cancel.store(true, Ordering::Release);
        }
        state.pending = Some(job);
        drop(state);
        self.wake.notify_one();
    }

    fn cancel(&self, generation: PreparationGeneration) {
        self.latest_generation
            .store(generation.get(), Ordering::Release);
        let mut state = lock_recover(&self.state);
        if let Some(cancel) = state.current_cancel.take() {
            cancel.store(true, Ordering::Release);
        }
        state.pending = None;
    }

    fn take(&self) -> Option<PreparationJob> {
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

    fn is_current(&self, generation: PreparationGeneration) -> bool {
        !self.stopped.load(Ordering::Acquire)
            && self.latest_generation.load(Ordering::Acquire) == generation.get()
    }

    fn stop(&self) {
        self.stopped.store(true, Ordering::Release);
        let mut state = lock_recover(&self.state);
        state.stopped = true;
        if let Some(cancel) = state.current_cancel.take() {
            cancel.store(true, Ordering::Release);
        }
        state.pending = None;
        drop(state);
        self.wake.notify_all();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreparationPoll {
    Idle,
    Pending {
        generation: PreparationGeneration,
    },
    Ready {
        generation: PreparationGeneration,
        key: PreparedTransactionKey,
    },
    Failed(PreparationFailure),
}

/// Host-owned, nonblocking front end to the fixed preparation workers.
pub struct PerformancePreparationRuntime {
    budget: PerformanceResourceBudget,
    ledger: Arc<ResourceLedger>,
    mailbox: Arc<JobMailbox>,
    completion: Arc<Mutex<Option<WorkerCompletion>>>,
    workers: Vec<JoinHandle<()>>,
    ready: VecDeque<PreparedTransaction>,
    next_generation: u64,
    pending_generation: Option<PreparationGeneration>,
}

impl PerformancePreparationRuntime {
    pub fn new(
        media_policy: MediaSafetyPolicy,
        device_limits: MediaDeviceLimits,
        budget: PerformanceResourceBudget,
    ) -> Result<Self, String> {
        Self::new_with_spout_timeout(
            media_policy,
            device_limits,
            budget,
            DEFAULT_SPOUT_PREPARE_TIMEOUT,
        )
    }

    pub fn new_with_spout_timeout(
        media_policy: MediaSafetyPolicy,
        device_limits: MediaDeviceLimits,
        budget: PerformanceResourceBudget,
        spout_timeout: Duration,
    ) -> Result<Self, String> {
        Self::with_preparer(
            budget,
            Arc::new(SystemSourcePreparer {
                media_policy,
                device_limits,
                spout_timeout,
                fingerprint_limits: FingerprintLimits::default(),
            }),
        )
    }

    fn with_preparer(
        budget: PerformanceResourceBudget,
        preparer: Arc<dyn SourcePreparer>,
    ) -> Result<Self, String> {
        let mailbox = JobMailbox::new();
        let completion = Arc::new(Mutex::new(None));
        let ledger = ResourceLedger::new(budget);
        let mut workers: Vec<JoinHandle<()>> = Vec::with_capacity(budget.staging_workers());
        for worker_index in 0..budget.staging_workers() {
            let worker_mailbox = mailbox.clone();
            let worker_completion = completion.clone();
            let worker_preparer = preparer.clone();
            let worker_ledger = ledger.clone();
            let worker = std::thread::Builder::new()
                .name(format!("performance-prepare-{}", worker_index + 1))
                .spawn(move || {
                    worker_loop(
                        worker_mailbox,
                        worker_completion,
                        worker_preparer,
                        worker_ledger,
                    );
                })
                .map_err(|error| {
                    mailbox.stop();
                    for worker in workers.drain(..) {
                        let _ = worker.join();
                    }
                    format!("could not start performance preparation worker: {error}")
                })?;
            workers.push(worker);
        }
        Ok(Self {
            budget,
            ledger,
            mailbox,
            completion,
            workers,
            ready: VecDeque::new(),
            next_generation: 0,
            pending_generation: None,
        })
    }

    pub fn submit(
        &mut self,
        request: PreparationRequest,
    ) -> Result<PreparationGeneration, PreparationFailure> {
        let replacement_count = request.replacement_count();
        if replacement_count > self.budget.max_prepared_sources() {
            return Err(PreparationFailure {
                generation: PreparationGeneration(self.next_generation),
                key: request.key,
                kind: PreparationFailureKind::ResourceBudget,
                message: format!(
                    "transaction needs {} prepared sources; host limit is {}",
                    replacement_count,
                    self.budget.max_prepared_sources()
                ),
            });
        }

        // Replacing the same prepared key never requires keeping the older
        // inactive resources. Further LRU entries are evicted only as needed
        // for the request's known source-count bound; live Layers are external.
        self.remove_ready(&request.key);
        while self
            .ledger
            .snapshot()
            .source_count
            .saturating_add(replacement_count)
            > self.budget.max_prepared_sources()
        {
            if self.ready.pop_front().is_none() {
                break;
            }
        }

        self.next_generation =
            self.next_generation
                .checked_add(1)
                .ok_or_else(|| PreparationFailure {
                    generation: PreparationGeneration(u64::MAX),
                    key: request.key.clone(),
                    kind: PreparationFailureKind::InvalidRequest,
                    message: "performance preparation generation exhausted".to_string(),
                })?;
        let generation = PreparationGeneration(self.next_generation);
        self.pending_generation = Some(generation);
        self.mailbox.submit(PreparationJob {
            generation,
            request,
            cancel: Arc::new(AtomicBool::new(false)),
        });
        Ok(generation)
    }

    /// Supersede queued/running work. Workers check this generation between
    /// sources and while waiting for Spout, then drop any partial transaction.
    pub fn cancel_pending(&mut self) -> Result<PreparationGeneration, String> {
        self.next_generation = self
            .next_generation
            .checked_add(1)
            .ok_or_else(|| "performance preparation generation exhausted".to_string())?;
        let generation = PreparationGeneration(self.next_generation);
        self.pending_generation = None;
        self.mailbox.cancel(generation);
        *lock_recover(&self.completion) = None;
        Ok(generation)
    }

    /// Never blocks. A successful completion is inserted into the inactive LRU
    /// cache and remains budget-accounted until taken, replaced, or evicted.
    pub fn poll(&mut self) -> PreparationPoll {
        let completion = lock_recover(&self.completion).take();
        let Some(completion) = completion else {
            return self
                .pending_generation
                .map_or(PreparationPoll::Idle, |generation| {
                    PreparationPoll::Pending { generation }
                });
        };
        if self.pending_generation != Some(completion.generation) {
            return self
                .pending_generation
                .map_or(PreparationPoll::Idle, |generation| {
                    PreparationPoll::Pending { generation }
                });
        }
        self.pending_generation = None;
        match completion.result {
            Ok(transaction) => {
                let key = transaction.key.clone();
                let generation = transaction.generation;
                self.remove_ready(&key);
                self.ready.push_back(transaction);
                PreparationPoll::Ready { generation, key }
            }
            Err(failure) => PreparationPoll::Failed(failure),
        }
    }

    #[cfg(test)]
    pub fn has_ready(&self, key: &PreparedTransactionKey) -> bool {
        self.ready
            .iter()
            .any(|transaction| transaction.key() == key)
    }

    /// Remove a prepared transaction from the evictable cache for GPU
    /// activation. Its budget lease follows the returned value.
    pub fn take_ready(&mut self, key: &PreparedTransactionKey) -> Option<PreparedTransaction> {
        let position = self
            .ready
            .iter()
            .position(|transaction| transaction.key() == key)?;
        self.ready.remove(position)
    }

    pub fn clear_ready(&mut self) {
        self.ready.clear();
    }

    pub fn budget_snapshot(&self) -> PreparedBudgetSnapshot {
        self.ledger.snapshot()
    }

    #[cfg(test)]
    pub fn latest_generation(&self) -> PreparationGeneration {
        PreparationGeneration(self.next_generation)
    }

    fn remove_ready(&mut self, key: &PreparedTransactionKey) {
        if let Some(position) = self
            .ready
            .iter()
            .position(|transaction| transaction.key() == key)
        {
            self.ready.remove(position);
        }
    }
}

impl Drop for PerformancePreparationRuntime {
    fn drop(&mut self) {
        self.mailbox.stop();
        for worker in self.workers.drain(..) {
            if worker.join().is_err() {
                log::warn!("performance preparation worker panicked during shutdown");
            }
        }
    }
}

fn worker_loop(
    mailbox: Arc<JobMailbox>,
    completion: Arc<Mutex<Option<WorkerCompletion>>>,
    preparer: Arc<dyn SourcePreparer>,
    ledger: Arc<ResourceLedger>,
) {
    while let Some(job) = mailbox.take() {
        let generation = job.generation;
        let failure_key = job.request.key.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            prepare_job(&job, &mailbox, preparer.as_ref(), &ledger)
        }))
        .unwrap_or_else(|_| {
            Some(Err(PreparationFailure {
                generation,
                key: failure_key,
                kind: PreparationFailureKind::SourceOpen,
                message: "performance source preparation worker panicked".to_string(),
            }))
        });
        if !mailbox.is_current(generation) {
            continue;
        }
        if let Some(result) = result {
            *lock_recover(&completion) = Some(WorkerCompletion { generation, result });
        }
    }
}

fn prepare_job(
    job: &PreparationJob,
    mailbox: &JobMailbox,
    preparer: &dyn SourcePreparer,
    ledger: &Arc<ResourceLedger>,
) -> Option<Result<PreparedTransaction, PreparationFailure>> {
    let mut replacements = Vec::with_capacity(job.request.replacements.len());
    for request in &job.request.replacements {
        let is_current = || mailbox.is_current(job.generation);
        let prepared = match preparer.prepare(
            &request.resolved_source,
            request.start_position,
            job.cancel.clone(),
            &is_current,
        ) {
            Ok(prepared) => prepared,
            Err(SourcePrepareError::Superseded) => return None,
            Err(SourcePrepareError::Failed(kind, message)) => {
                return Some(Err(PreparationFailure {
                    generation: job.generation,
                    key: job.request.key.clone(),
                    kind,
                    message,
                }));
            }
        };
        if !mailbox.is_current(job.generation) {
            return None;
        }
        if let Err(message) = prepared.validate() {
            return Some(Err(PreparationFailure {
                generation: job.generation,
                key: job.request.key.clone(),
                kind: PreparationFailureKind::SourceOpen,
                message,
            }));
        }
        let lease = match ledger.acquire(prepared.preload_bytes) {
            Ok(lease) => lease,
            Err(message) => {
                return Some(Err(PreparationFailure {
                    generation: job.generation,
                    key: job.request.key.clone(),
                    kind: PreparationFailureKind::ResourceBudget,
                    message: format!(
                        "cannot prepare slot {} for layer {}: {message}",
                        request.desired_slot.id.get(),
                        request.target_layer.get()
                    ),
                }));
            }
        };
        replacements.push(PreparedReplacement {
            target_layer: request.target_layer,
            expected_prior_slot: request.expected_prior_slot.clone(),
            desired_slot: request.desired_slot.clone(),
            cue_id: request.cue_id,
            start_position: request.start_position,
            prepared,
            lease,
        });
    }
    mailbox.is_current(job.generation).then(|| {
        Ok(PreparedTransaction {
            generation: job.generation,
            key: job.request.key.clone(),
            replacements,
        })
    })
}

fn lock_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::ClipTransportConfig;
    use std::sync::atomic::AtomicUsize;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct FakePreparer;

    impl SourcePreparer for FakePreparer {
        fn prepare(
            &self,
            source: &ResolvedPreparedSource,
            _start_position: NormalizedTime,
            _cancel: Arc<AtomicBool>,
            is_current: &dyn Fn() -> bool,
        ) -> Result<CpuPreparedSource, SourcePrepareError> {
            let ResolvedPreparedSource::File { path, .. } = source else {
                unreachable!()
            };
            let name = path.to_string_lossy();
            let delay_ms = if name.contains("slow") { 120 } else { 10 };
            let started = Instant::now();
            while started.elapsed() < Duration::from_millis(delay_ms) {
                if !is_current() {
                    return Err(SourcePrepareError::Superseded);
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            if name.contains("fail") {
                return Err(SourcePrepareError::Failed(
                    PreparationFailureKind::SourceOpen,
                    "synthetic open failure".to_string(),
                ));
            }
            let preload_bytes = name
                .split('-')
                .find_map(|part| part.parse::<u64>().ok())
                .unwrap_or(4);
            let pixel = if name.contains("real") { 0x7f } else { 0x22 };
            let first_rgba = vec![pixel, 2, 3, 255];
            let synthetic = crate::video::DecodedStillImage::from_rgba(1, 1, first_rgba.clone())
                .map(StillImage::from_decoded)
                .map_err(|error| {
                    SourcePrepareError::Failed(PreparationFailureKind::SourceOpen, error)
                })?;
            Ok(CpuPreparedSource {
                source: CpuLayerSource::Synthetic(synthetic),
                runtime_source_path: name.into_owned(),
                persisted_reference: None,
                width: 1,
                height: 1,
                source_fps: 30.0,
                first_rgba,
                preload_bytes,
            })
        }
    }

    struct CountingPreparer {
        opens: Arc<AtomicUsize>,
        preload_bytes: u64,
    }

    impl SourcePreparer for CountingPreparer {
        fn prepare(
            &self,
            source: &ResolvedPreparedSource,
            _start_position: NormalizedTime,
            _cancel: Arc<AtomicBool>,
            is_current: &dyn Fn() -> bool,
        ) -> Result<CpuPreparedSource, SourcePrepareError> {
            if !is_current() {
                return Err(SourcePrepareError::Superseded);
            }
            self.opens.fetch_add(1, Ordering::AcqRel);
            let ResolvedPreparedSource::File { path, .. } = source else {
                unreachable!()
            };
            let runtime_source_path = path.to_string_lossy().into_owned();
            let pixel = if runtime_source_path.contains('a') {
                0xa1
            } else {
                0xb2
            };
            let first_rgba = vec![pixel, 4, 9, 255];
            let image = crate::video::DecodedStillImage::from_rgba(1, 1, first_rgba.clone())
                .map(StillImage::from_decoded)
                .map_err(|error| {
                    SourcePrepareError::Failed(PreparationFailureKind::SourceOpen, error)
                })?;
            Ok(CpuPreparedSource {
                source: CpuLayerSource::Synthetic(image),
                runtime_source_path,
                persisted_reference: None,
                width: 1,
                height: 1,
                source_fps: 30.0,
                first_rgba,
                preload_bytes: self.preload_bytes,
            })
        }
    }

    fn slot(id: u16) -> ClipSlotConfig {
        ClipSlotConfig {
            id: ClipSlotId::new(id).unwrap(),
            name: format!("Slot {id}"),
            filename: format!("slot-{id}.mp4"),
            source_path: format!("slot-{id}.mp4"),
            transport: ClipTransportConfig::default(),
            saved_playhead: NormalizedTime::ZERO,
        }
    }

    fn replacement(layer: u64, id: u16, path: &str) -> SourcePreparationRequest {
        let desired = slot(id);
        SourcePreparationRequest::new(
            StableLayerId::new(layer).unwrap(),
            Some(desired.clone()),
            desired,
            None,
            ResolvedPreparedSource::file(path),
        )
        .unwrap()
    }

    fn request(scene: u16, replacements: Vec<SourcePreparationRequest>) -> PreparationRequest {
        PreparationRequest::from_replacements(
            PreparedTransactionKey::Scene(SceneId::new(scene).unwrap()),
            replacements,
        )
        .unwrap()
    }

    fn runtime(budget: PerformanceResourceBudget) -> PerformancePreparationRuntime {
        PerformancePreparationRuntime::with_preparer(budget, Arc::new(FakePreparer)).unwrap()
    }

    fn wait_terminal(runtime: &mut PerformancePreparationRuntime) -> PreparationPoll {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let poll = runtime.poll();
            if !matches!(poll, PreparationPoll::Pending { .. }) {
                return poll;
            }
            assert!(Instant::now() < deadline, "preparation timed out in test");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    fn test_gpu() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("GPU adapter");
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Prepared Slot Reuse Test"),
            ..Default::default()
        }))
        .expect("GPU device")
    }

    fn live_still_layer(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
    ) -> (Layer, PathBuf) {
        let path = std::env::temp_dir().join(format!(
            "collide-o-scope-slot-a-{}-{}.png",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_nanos()
        ));
        let mut pixels = Vec::with_capacity(width as usize * 4);
        for index in 0..width {
            pixels.extend_from_slice(&[0x31 + index as u8, 7, 11, 255]);
        }
        image::RgbaImage::from_raw(width, 1, pixels)
            .unwrap()
            .save(&path)
            .unwrap();
        let mut layer = Layer::new_with_media_policy(
            &path.to_string_lossy(),
            device,
            &MediaSafetyPolicy::safe(),
        )
        .unwrap();
        let frame = layer.take_ready_media_frame().unwrap().unwrap();
        layer.upload_frame(device, queue, &frame.rgba).unwrap();
        assert!(layer.source_frame_initialized());
        (layer, path)
    }

    fn prepare_slot_b(
        runtime: &mut PerformancePreparationRuntime,
        layer: &mut Layer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> GpuCommitPayload {
        let mut desired = slot(2);
        desired.filename = "prepared-b.png".into();
        desired.source_path = "prepared-b.png".into();
        layer.clip_slots.upsert(desired.clone()).unwrap();
        let key = PreparedTransactionKey::LayerSlot {
            layer_id: layer.stable_layer_id(),
            slot_id: desired.id,
        };
        let replacement = SourcePreparationRequest::new(
            layer.stable_layer_id(),
            Some(desired.clone()),
            desired,
            None,
            ResolvedPreparedSource::file("prepared-b"),
        )
        .unwrap();
        runtime
            .submit(PreparationRequest::from_replacements(key.clone(), vec![replacement]).unwrap())
            .unwrap();
        assert!(matches!(
            wait_terminal(runtime),
            PreparationPoll::Ready { .. }
        ));
        runtime
            .take_ready(&key)
            .unwrap()
            .activate_gpu(device, queue, std::slice::from_ref(layer))
            .unwrap_or_else(|error| panic!("GPU activation failed: {error}"))
    }

    #[test]
    fn transferred_lease_resize_is_atomic_and_never_weakens_the_budget() {
        // A planning budget of 100 assigns 80 bytes to prepared preloads.
        let ledger = ResourceLedger::new(PerformanceResourceBudget::for_planning_budget(100));
        let mut first = ledger.acquire(30).unwrap();
        let mut second = ledger.acquire(30).unwrap();
        assert_eq!(
            ledger.snapshot(),
            PreparedBudgetSnapshot {
                source_count: 2,
                preload_bytes: 60,
            }
        );

        first.try_resize(50).unwrap();
        assert_eq!(
            ledger.snapshot(),
            PreparedBudgetSnapshot {
                source_count: 2,
                preload_bytes: 80,
            }
        );

        // Growing the second displaced source would exceed the aggregate
        // ceiling. Both the lease and ledger retain their exact prior charge.
        assert_eq!(
            second.try_resize(40),
            Err("performance preload byte budget exceeded")
        );
        assert_eq!(second.bytes, 30);
        assert_eq!(
            ledger.snapshot(),
            PreparedBudgetSnapshot {
                source_count: 2,
                preload_bytes: 80,
            }
        );

        drop(first);
        drop(second);
        assert_eq!(ledger.snapshot(), PreparedBudgetSnapshot::default());
    }

    #[test]
    fn transaction_failure_publishes_no_partial_ready_state() {
        let mut runtime = runtime(PerformanceResourceBudget::default());
        let key = PreparedTransactionKey::Scene(SceneId::new(1).unwrap());
        runtime
            .submit(request(
                1,
                vec![replacement(1, 1, "ok"), replacement(2, 1, "fail")],
            ))
            .unwrap();
        let PreparationPoll::Failed(failure) = wait_terminal(&mut runtime) else {
            panic!("expected atomic failure")
        };
        assert_eq!(failure.kind, PreparationFailureKind::SourceOpen);
        assert!(!runtime.has_ready(&key));
        assert_eq!(runtime.budget_snapshot(), PreparedBudgetSnapshot::default());
    }

    #[test]
    fn preparing_active_replacement_exposes_no_intermediate_metadata() {
        let prior = slot(1);
        let mut desired = prior.clone();
        desired.filename = "new-source.mp4".into();
        desired.source_path = "new-source.mp4".into();
        let live_slots = crate::performance::ClipSlots::singleton(prior.clone());
        let replacement = SourcePreparationRequest::new(
            StableLayerId::new(1).unwrap(),
            Some(prior.clone()),
            desired.clone(),
            None,
            ResolvedPreparedSource::file("ok"),
        )
        .unwrap();
        let key = PreparedTransactionKey::LayerSlot {
            layer_id: StableLayerId::new(1).unwrap(),
            slot_id: ClipSlotId::LEGACY,
        };
        let mut runtime = runtime(PerformanceResourceBudget::default());
        runtime
            .submit(PreparationRequest::from_replacements(key.clone(), vec![replacement]).unwrap())
            .unwrap();

        // Staging owns desired metadata; the independently captured live Set
        // remains byte-for-byte on the prior source until commit is requested.
        assert_eq!(live_slots.get(ClipSlotId::LEGACY), Some(&prior));
        assert!(matches!(
            wait_terminal(&mut runtime),
            PreparationPoll::Ready { .. }
        ));
        assert_eq!(live_slots.get(ClipSlotId::LEGACY), Some(&prior));
        let prepared = runtime.take_ready(&key).unwrap();
        let (expected, cached_desired) = prepared.first_slot_transition().unwrap();
        assert_eq!(expected, Some(&prior));
        assert_eq!(cached_desired, &desired);
    }

    #[test]
    fn no_placeholder_black_source_is_reported_ready() {
        let mut runtime = runtime(PerformanceResourceBudget::default());
        let key = PreparedTransactionKey::Scene(SceneId::new(2).unwrap());
        runtime
            .submit(request(2, vec![replacement(1, 1, "spout-real")]))
            .unwrap();
        assert!(matches!(runtime.poll(), PreparationPoll::Pending { .. }));
        assert!(matches!(
            wait_terminal(&mut runtime),
            PreparationPoll::Ready { .. }
        ));
        let prepared = runtime.take_ready(&key).unwrap();
        assert_eq!(prepared.first_rgba().unwrap(), [0x7f, 2, 3, 255]);
    }

    #[test]
    fn aggregate_preload_budget_rejects_whole_transaction_and_releases_leases() {
        let mut runtime = runtime(PerformanceResourceBudget::for_planning_budget(100));
        runtime
            .submit(request(
                3,
                vec![replacement(1, 1, "bytes-50"), replacement(2, 1, "bytes-50")],
            ))
            .unwrap();
        let PreparationPoll::Failed(failure) = wait_terminal(&mut runtime) else {
            panic!("expected budget failure")
        };
        assert_eq!(failure.kind, PreparationFailureKind::ResourceBudget);
        assert_eq!(runtime.budget_snapshot(), PreparedBudgetSnapshot::default());
    }

    #[test]
    fn rapid_newer_submission_supersedes_running_and_queued_work() {
        let mut runtime = runtime(PerformanceResourceBudget::default());
        let old = runtime
            .submit(request(4, vec![replacement(1, 1, "slow-old")]))
            .unwrap();
        std::thread::sleep(Duration::from_millis(5));
        let new_key = PreparedTransactionKey::Scene(SceneId::new(5).unwrap());
        let new = runtime
            .submit(request(5, vec![replacement(1, 1, "new")]))
            .unwrap();
        assert!(new > old);
        let PreparationPoll::Ready { generation, key } = wait_terminal(&mut runtime) else {
            panic!("newest request did not become ready")
        };
        assert_eq!(generation, new);
        assert_eq!(key, new_key);
        std::thread::sleep(Duration::from_millis(140));
        assert_eq!(runtime.poll(), PreparationPoll::Idle);
        assert!(runtime.has_ready(&new_key));
        assert!(!runtime.has_ready(&PreparedTransactionKey::Scene(SceneId::new(4).unwrap())));
    }

    #[test]
    fn source_count_limit_is_rejected_before_worker_submission() {
        let mut runtime = runtime(PerformanceResourceBudget::default());
        let replacements = (0..17)
            .map(|index| replacement(index + 1, 1, "ok"))
            .collect();
        let error = runtime.submit(request(6, replacements)).unwrap_err();
        assert_eq!(error.kind, PreparationFailureKind::ResourceBudget);
        assert_eq!(runtime.latest_generation().get(), 0);
    }

    #[test]
    fn authored_content_reference_resolves_and_fingerprints_on_worker() {
        let root = std::env::temp_dir().join(format!(
            "collide-o-scope-prepared-resolver-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let media_path = root.join("renamed.png");
        image::RgbaImage::from_raw(2, 1, vec![1, 2, 3, 255, 5, 6, 7, 255])
            .unwrap()
            .save(&media_path)
            .unwrap();
        let mut fingerprints = FingerprintSession::new(FingerprintLimits::default()).unwrap();
        let reference = fingerprints
            .fingerprint(&media_path)
            .unwrap()
            .source_reference();

        let layer_id = StableLayerId::new(1).unwrap();
        let mut authored_slot = slot(1);
        authored_slot.filename = "missing-name.png".into();
        authored_slot.source_path.clone_from(&reference);
        let replacement = SourcePreparationRequest::new(
            layer_id,
            None,
            authored_slot,
            None,
            ResolvedPreparedSource::authored(
                reference.clone(),
                "missing-name.png",
                ResolveContext::new(None, Some(root.clone())),
            ),
        )
        .unwrap();
        let key = PreparedTransactionKey::LayerSlot {
            layer_id,
            slot_id: ClipSlotId::LEGACY,
        };
        let request =
            PreparationRequest::from_replacements(key.clone(), vec![replacement]).unwrap();
        let mut runtime = PerformancePreparationRuntime::new(
            MediaSafetyPolicy::safe(),
            MediaDeviceLimits::none(),
            PerformanceResourceBudget::default(),
        )
        .unwrap();
        runtime.submit(request).unwrap();
        assert!(matches!(
            wait_terminal(&mut runtime),
            PreparationPoll::Ready { .. }
        ));
        let prepared = runtime.take_ready(&key).unwrap();
        let (runtime_path, persisted) = prepared.first_source_identity().unwrap();
        assert_eq!(
            Path::new(runtime_path),
            std::fs::canonicalize(&media_path).unwrap()
        );
        assert_eq!(persisted, Some(reference.as_str()));
        drop(prepared);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn video_preparation_publishes_the_requested_cue_frame_not_the_zero_seed() {
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("loop-72f.mp4");
        let cue_id = CueId::new(7).unwrap();
        let cue_at = NormalizedTime::clamped(0.7);
        let mut desired = slot(1);
        assert!(desired.transport.cues.insert(crate::transport::CuePoint {
            id: cue_id,
            at: cue_at
        }));
        desired.saved_playhead = NormalizedTime::clamped(0.1);
        desired.filename = "loop-72f.mp4".into();
        desired.source_path = fixture.to_string_lossy().into_owned();
        let layer_id = StableLayerId::new(77).unwrap();
        let replacement = SourcePreparationRequest::new(
            layer_id,
            Some(desired.clone()),
            desired,
            Some(cue_id),
            ResolvedPreparedSource::file(&fixture),
        )
        .unwrap();
        let key = PreparedTransactionKey::LayerSlot {
            layer_id,
            slot_id: ClipSlotId::LEGACY,
        };
        let mut runtime = PerformancePreparationRuntime::new(
            MediaSafetyPolicy::safe(),
            MediaDeviceLimits::none(),
            PerformanceResourceBudget::default(),
        )
        .unwrap();
        runtime
            .submit(PreparationRequest::from_replacements(key.clone(), vec![replacement]).unwrap())
            .unwrap();
        assert!(matches!(
            wait_terminal(&mut runtime),
            PreparationPoll::Ready { .. }
        ));
        let prepared = runtime.take_ready(&key).unwrap();

        let mut expected = crate::video::VideoDecoder::open(&fixture.to_string_lossy()).unwrap();
        let target_seconds = cue_at.get() * expected.duration_seconds();
        let expected = expected.seek_decode(target_seconds).unwrap();
        assert_eq!(prepared.first_rgba().unwrap(), expected.rgba);
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn displaced_a_to_b_to_a_reuses_exact_source_without_another_open_or_black() {
        let (device, queue) = test_gpu();
        let (mut layer, path) = live_still_layer(&device, &queue, 1);
        let opens = Arc::new(AtomicUsize::new(0));
        let mut runtime = PerformancePreparationRuntime::with_preparer(
            PerformanceResourceBudget::default(),
            Arc::new(CountingPreparer {
                opens: Arc::clone(&opens),
                preload_bytes: 24,
            }),
        )
        .unwrap();

        let prepared_b = prepare_slot_b(&mut runtime, &mut layer, &device, &queue);
        assert_eq!(opens.load(Ordering::Acquire), 1);
        let mut first_receipt =
            match prepared_b.commit(std::slice::from_mut(&mut layer), Instant::now()) {
                Ok(receipt) => receipt,
                Err(rejected) => panic!("B commit failed: {}", rejected.failure),
            };
        assert_eq!(layer.active_clip_slot, ClipSlotId::new(2).unwrap());
        assert!(
            layer.source_frame_initialized(),
            "prepared B must be defined"
        );
        assert_eq!(first_receipt.displaced_sources.len(), 1);
        assert!(first_receipt.uncached_displaced_slots.is_empty());
        assert_eq!(runtime.budget_snapshot().source_count, 1);

        let prepared_a = first_receipt.displaced_sources.pop().unwrap();
        let (layer_id, cached_slot, cue_id) = prepared_a.single_layer_slot_descriptor().unwrap();
        assert_eq!(layer_id, layer.stable_layer_id());
        assert_eq!(cached_slot.id, ClipSlotId::LEGACY);
        assert_eq!(cue_id, None);
        let second_receipt =
            match prepared_a.commit(std::slice::from_mut(&mut layer), Instant::now()) {
                Ok(receipt) => receipt,
                Err(rejected) => panic!("A return commit failed: {}", rejected.failure),
            };

        assert_eq!(layer.active_clip_slot, ClipSlotId::LEGACY);
        assert!(
            layer.source_frame_initialized(),
            "the exact prior A texture remains initialized across reuse"
        );
        assert_eq!(
            opens.load(Ordering::Acquire),
            1,
            "returning to displaced A must not reopen or first-frame decode"
        );
        assert_eq!(second_receipt.displaced_sources.len(), 1);
        drop(second_receipt);
        assert_eq!(runtime.budget_snapshot(), PreparedBudgetSnapshot::default());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn displaced_payload_rejects_stale_slot_edit_and_removed_layer_without_mutation() {
        let (device, queue) = test_gpu();
        let (mut layer, path) = live_still_layer(&device, &queue, 1);
        let opens = Arc::new(AtomicUsize::new(0));
        let mut runtime = PerformancePreparationRuntime::with_preparer(
            PerformanceResourceBudget::default(),
            Arc::new(CountingPreparer {
                opens,
                preload_bytes: 24,
            }),
        )
        .unwrap();
        let prepared_b = prepare_slot_b(&mut runtime, &mut layer, &device, &queue);
        let mut receipt = prepared_b
            .commit(std::slice::from_mut(&mut layer), Instant::now())
            .unwrap_or_else(|rejected| panic!("B commit failed: {}", rejected.failure));
        let prepared_a = receipt.displaced_sources.pop().unwrap();

        layer
            .clip_slots
            .get_mut(ClipSlotId::LEGACY)
            .unwrap()
            .name
            .push_str(" edited");
        let rejected = match prepared_a.commit(std::slice::from_mut(&mut layer), Instant::now()) {
            Ok(_) => panic!("stale edited slot unexpectedly committed"),
            Err(rejected) => rejected,
        };
        assert_eq!(
            rejected.failure.kind,
            PreparationFailureKind::CommitConflict
        );
        assert_eq!(layer.active_clip_slot, ClipSlotId::new(2).unwrap());
        assert!(layer.source_frame_initialized());

        let prepared_a = rejected.into_payload();
        let mut removed_layers = Vec::new();
        let removed = match prepared_a.commit(&mut removed_layers, Instant::now()) {
            Ok(_) => panic!("removed target layer unexpectedly committed"),
            Err(rejected) => rejected,
        };
        assert_eq!(removed.failure.kind, PreparationFailureKind::CommitConflict);
        assert!(removed.failure.message.contains("no longer exists"));
        drop(layer);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn over_budget_displaced_source_is_evicted_and_reopened_without_blanking_live_output() {
        let (device, queue) = test_gpu();
        // A 2x1 still carries a 48-byte conservative working set. The test
        // budget exposes only 40 prepared-preload bytes, while B needs 24.
        let (mut layer, path) = live_still_layer(&device, &queue, 2);
        let opens = Arc::new(AtomicUsize::new(0));
        let mut runtime = PerformancePreparationRuntime::with_preparer(
            PerformanceResourceBudget::for_planning_budget(50),
            Arc::new(CountingPreparer {
                opens: Arc::clone(&opens),
                preload_bytes: 24,
            }),
        )
        .unwrap();
        let prepared_b = prepare_slot_b(&mut runtime, &mut layer, &device, &queue);
        let receipt = prepared_b
            .commit(std::slice::from_mut(&mut layer), Instant::now())
            .unwrap_or_else(|rejected| panic!("B commit failed: {}", rejected.failure));
        assert!(receipt.displaced_sources.is_empty());
        assert_eq!(
            receipt.uncached_displaced_slots,
            vec![(layer.stable_layer_id(), ClipSlotId::LEGACY)]
        );
        assert_eq!(runtime.budget_snapshot(), PreparedBudgetSnapshot::default());
        assert_eq!(layer.active_clip_slot, ClipSlotId::new(2).unwrap());
        assert!(
            layer.source_frame_initialized(),
            "B remains live after eviction"
        );

        let slot_a = layer.clip_slots.get(ClipSlotId::LEGACY).unwrap().clone();
        let key = PreparedTransactionKey::LayerSlot {
            layer_id: layer.stable_layer_id(),
            slot_id: ClipSlotId::LEGACY,
        };
        let reopen = SourcePreparationRequest::new(
            layer.stable_layer_id(),
            Some(slot_a.clone()),
            slot_a,
            None,
            ResolvedPreparedSource::file("reopen-a"),
        )
        .unwrap();
        runtime
            .submit(PreparationRequest::from_replacements(key.clone(), vec![reopen]).unwrap())
            .unwrap();
        assert!(matches!(
            wait_terminal(&mut runtime),
            PreparationPoll::Ready { .. }
        ));
        let prepared_a = runtime
            .take_ready(&key)
            .unwrap()
            .activate_gpu(&device, &queue, std::slice::from_ref(&layer))
            .unwrap_or_else(|error| panic!("A reopen GPU activation failed: {error}"));
        let return_receipt = prepared_a
            .commit(std::slice::from_mut(&mut layer), Instant::now())
            .unwrap_or_else(|rejected| panic!("reopened A commit failed: {}", rejected.failure));
        assert_eq!(opens.load(Ordering::Acquire), 2);
        assert_eq!(layer.active_clip_slot, ClipSlotId::LEGACY);
        assert!(layer.source_frame_initialized());
        drop(return_receipt);
        std::fs::remove_file(path).unwrap();
    }
}
