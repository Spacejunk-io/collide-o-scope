use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::effects::EffectUniforms;
use crate::image_routing::{LayerMatte, StableLayerId};
use crate::media_safety::{
    validate_safe_dimensions, MediaAllocationPlan, MediaDeviceLimits, MediaSafetyPolicy,
    MediaSourceKind,
};
use crate::motion::MotionParams;
use crate::performance::{ClipSlotConfig, ClipSlotId, ClipSlots};
use crate::spatial::SpatialTransform;
use crate::spout_in::{SpoutFrame, SpoutIn, SpoutStatus};
use crate::transport::{
    ClipTransportState, CueId, FrameSelection, NormalizedTime, PlaybackDirection,
    ProgramTransportTick, TransportTimeline,
};
use crate::video::threaded::{DecoderHealth, DecoderTelemetry, ReadyFrame};
use crate::video::{
    decode_still_image_with_media_policy, CodecMotionFrame, StillImage, ThreadedDecoder,
};
use crate::visual_rack::{LegacyRackScope, RuntimeVisualRack};

pub const SPOUT_SOURCE_PREFIX: &str = "spout://";
pub const VIDEO_EXTENSIONS: &[&str] = &["mp4", "webm", "mov", "avi", "mkv"];
pub const STILL_IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "bmp", "webp"];

static NEXT_LAYER_ID: AtomicU64 = AtomicU64::new(1);

const fn default_bypass_master_fx() -> bool {
    false
}

fn allocate_layer_id() -> u64 {
    NEXT_LAYER_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .expect("live layer identity space exhausted")
}

/// Advance the process-lifetime allocator beyond an identity restored by the
/// private manual-history transaction store. History is the only path that
/// restores runtime identities: persisted patches continue to allocate fresh
/// IDs, so a file can never manufacture or reuse a live identity.
fn observe_restored_layer_id(layer_id: u64) -> Result<(), &'static str> {
    let next = layer_id
        .checked_add(1)
        .filter(|next| *next != 0)
        .ok_or("restored layer identity exhausts the process identity space")?;
    NEXT_LAYER_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            Some(current.max(next))
        })
        .map(|_| ())
        .map_err(|_| "could not advance the process layer identity cursor")
}

fn default_layer_rack() -> RuntimeVisualRack {
    RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer)
}

fn default_layer_motion() -> MotionParams {
    MotionParams::default()
}

fn reset_layer_motion(motion: &mut MotionParams) {
    *motion = MotionParams::default();
}

fn take_matching_codec_motion(
    frame: &mut ReadyFrame,
    source_dimensions: [u32; 2],
) -> Option<CodecMotionFrame> {
    frame.codec_motion.take().filter(|motion| {
        motion.source_generation == frame.source_generation
            && motion.source_dimensions == source_dimensions
            && motion.algorithm_version == crate::motion::MOTION_ALGORITHM_VERSION
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    Normal,
    Screen,
    Multiply,
    Difference,
    Add,
    Subtract,
    Darken,
    Lighten,
    Overlay,
    SoftLight,
    HardLight,
    Exclusion,
    Dodge,
    Burn,
    AlphaCut,
}

impl BlendMode {
    /// Every protocol value in its permanent append-only numeric order.
    pub const ALL: [Self; 15] = [
        Self::Normal,
        Self::Screen,
        Self::Multiply,
        Self::Difference,
        Self::Add,
        Self::Subtract,
        Self::Darken,
        Self::Lighten,
        Self::Overlay,
        Self::SoftLight,
        Self::HardLight,
        Self::Exclusion,
        Self::Dodge,
        Self::Burn,
        Self::AlphaCut,
    ];

    /// Stable append-only shader/protocol code. Never reorder or recycle one.
    pub fn as_u32(self) -> u32 {
        match self {
            Self::Normal => 0,
            Self::Screen => 1,
            Self::Multiply => 2,
            Self::Difference => 3,
            Self::Add => 4,
            Self::Subtract => 5,
            Self::Darken => 6,
            Self::Lighten => 7,
            Self::Overlay => 8,
            Self::SoftLight => 9,
            Self::HardLight => 10,
            Self::Exclusion => 11,
            Self::Dodge => 12,
            Self::Burn => 13,
            Self::AlphaCut => 14,
        }
    }

    /// Decode a stable shader/protocol code. Unknown future values must be
    /// rejected by callers rather than silently reinterpreted.
    pub fn from_u32(code: u32) -> Option<Self> {
        Self::ALL.get(code as usize).copied()
    }

    /// Stable lowercase value used by patches and the web protocol.
    /// Keep this separate from the human-facing title-case label so a
    /// snapshot always matches the HTML `<option value>` exactly.
    pub fn key(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Screen => "screen",
            Self::Multiply => "multiply",
            Self::Difference => "difference",
            Self::Add => "add",
            Self::Subtract => "subtract",
            Self::Darken => "darken",
            Self::Lighten => "lighten",
            Self::Overlay => "overlay",
            Self::SoftLight => "soft_light",
            Self::HardLight => "hard_light",
            Self::Exclusion => "exclusion",
            Self::Dodge => "dodge",
            Self::Burn => "burn",
            Self::AlphaCut => "alpha_cut",
        }
    }

    /// Parse the exact stable patch/web key without inventing aliases.
    pub fn from_key(key: &str) -> Option<Self> {
        Some(match key {
            "normal" => Self::Normal,
            "screen" => Self::Screen,
            "multiply" => Self::Multiply,
            "difference" => Self::Difference,
            "add" => Self::Add,
            "subtract" => Self::Subtract,
            "darken" => Self::Darken,
            "lighten" => Self::Lighten,
            "overlay" => Self::Overlay,
            "soft_light" => Self::SoftLight,
            "hard_light" => Self::HardLight,
            "exclusion" => Self::Exclusion,
            "dodge" => Self::Dodge,
            "burn" => Self::Burn,
            "alpha_cut" => Self::AlphaCut,
            _ => return None,
        })
    }
}

pub enum LayerSource {
    Video(ThreadedDecoder),
    Still(StillImage),
    Spout(SpoutIn),
}

/// A fully allocated and initialized replacement for a layer's source-only
/// resources. Building this value is fallible; installing it is an infallible
/// field swap performed only after an entire Scene transaction is ready.
pub(crate) struct LayerSourceActivation {
    source_path: String,
    persisted_source_reference: Option<String>,
    filename: String,
    source: LayerSource,
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    width: u32,
    height: u32,
    source_fps: f32,
    preload_bytes: u64,
    source_error: String,
    source_frame_initialized: bool,
    codec_motion: Option<CodecMotionFrame>,
}

/// Ownership returned when an already-live source is displaced by a prepared
/// activation. The source texture and decoder remain intact so the caller can
/// keep this exact slot GPU-ready instead of reopening it on the next switch.
pub(crate) struct DisplacedLayerSource {
    pub(crate) target_layer: StableLayerId,
    pub(crate) slot: ClipSlotConfig,
    pub(crate) start_position: NormalizedTime,
    pub(crate) activation: LayerSourceActivation,
}

impl LayerSourceActivation {
    pub(crate) const fn preload_bytes(&self) -> u64 {
        self.preload_bytes
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn stage(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source_path: String,
        persisted_source_reference: Option<String>,
        filename: String,
        source: LayerSource,
        width: u32,
        height: u32,
        source_fps: f32,
        preload_bytes: u64,
        first_rgba: &[u8],
    ) -> Result<Self, String> {
        let (texture, texture_view) =
            create_layer_texture(device, width, height, "Prepared Layer Texture")?;
        write_layer_texture_checked(
            device,
            queue,
            &texture,
            first_rgba,
            width,
            height,
            "Prepared Layer Texture",
        )?;
        Ok(Self {
            source_path,
            persisted_source_reference,
            filename,
            source,
            texture,
            texture_view,
            width,
            height,
            source_fps: if source_fps.is_finite() && source_fps > 0.0 {
                source_fps
            } else {
                30.0
            },
            preload_bytes,
            source_error: String::new(),
            source_frame_initialized: true,
            codec_motion: None,
        })
    }
}

impl LayerSource {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Video(_) => "video",
            Self::Still(_) => "image",
            Self::Spout(_) => "spout",
        }
    }
}

fn source_reference_for_persistence<'a>(
    runtime_path: &'a str,
    persisted_reference: Option<&'a str>,
) -> &'a str {
    persisted_reference.unwrap_or(runtime_path)
}

fn content_identity_for_proxy_reference(
    reference: &str,
) -> Option<crate::media_source::ContentIdentity> {
    crate::media_source::parse_content_reference(reference)
        .ok()
        .flatten()
}

pub struct Layer {
    /// Process-lifetime identity for this live instance. Moving/reordering a
    /// layer retains the ID; constructing a replacement (including patch
    /// load) allocates a new one. It is intentionally absent from patches.
    layer_id: u64,
    /// Stable source identity used when persisting/reopening this layer. This
    /// is the canonical path when canonicalization succeeds, otherwise it is
    /// the caller-provided path verbatim.
    pub source_path: String,
    /// Content-addressed identity retained across runtime path resolution.
    ///
    /// Decoders continue to use `source_path`; patch capture and export use
    /// this reference when present so moving or changing the resolved host
    /// file cannot silently weaken a `cos-sha256://` source contract.
    persisted_source_reference: Option<String>,
    pub filename: String,
    pub source: LayerSource,
    pub texture: wgpu::Texture,
    pub texture_view: wgpu::TextureView,
    /// Generation of the GPU texture/view identity (not its pixel contents).
    /// Advanced composition bind groups retain views, so same-size source
    /// replacement must invalidate them without rebuilding on ordinary frame
    /// uploads.
    source_resource_epoch: u64,
    pub opacity: f32,
    pub blend_mode: BlendMode,
    pub paused: bool,
    pub visible: bool,
    /// When true, this layer skips the shared master shader stage while
    /// retaining its own effects and all later program-wide stages.
    pub bypass_master_fx: bool,
    /// Advance this layer's deterministic pattern seed at each authoritative
    /// video-loop boundary. Stills and live inputs ignore the flag.
    pub reroll_on_loop: bool,
    pub effects: EffectUniforms,
    /// Ordered creative processing owned by this stable live layer. Legacy
    /// sources begin with the frozen whole-scope marker rack; source swaps do
    /// not replace it because the rack belongs to the layer, not its decoder.
    pub rack: RuntimeVisualRack,
    /// Resolution-independent authored geometry. New interactive layers use
    /// Fit + Transparent; patch application may replace this with the exact
    /// persisted value (including the inactive historical identity).
    pub transform: SpatialTransform,
    /// Bounded motion-field, transplant, and curved-shutter authoring owned by
    /// this stable visual identity. Source swaps preserve it exactly.
    pub motion: MotionParams,
    /// Prepared source set owned by this persistent visual identity. Slot IDs
    /// are searched, never interpreted as vector indices.
    pub clip_slots: ClipSlots,
    pub active_clip_slot: ClipSlotId,
    /// Runtime-only playhead/generation state for the active slot.
    pub clip_transport: ClipTransportState,
    /// Explicit image relationship evaluated after local effects and before
    /// opacity/blending. Disabled is the exact legacy compositor path.
    pub matte: LayerMatte,
    /// One-shot UI requests consumed by the next authoritative program tick.
    pending_transport_seek: Option<NormalizedTime>,
    pending_transport_cue: Option<CueId>,
    /// Last pure timeline result. Transport transparency is kept separate from
    /// authored visibility so a completed OneShot never edits the patch.
    last_transport_selection: Option<FrameSelection>,
    pub width: u32,
    pub height: u32,
    /// Render-side source error (for example a live frame exceeding GPU
    /// limits). Spout worker errors are merged into `spout_status()`.
    source_error: String,
    /// True after the first source image has reached the GPU texture. A file
    /// layer may upload its decoder seed while globally frozen, but no later
    /// in-flight completion may leak through that freeze.
    source_frame_initialized: bool,
    /// Decoder metadata committed with the exact RGBA upload currently held
    /// by `texture`. Runtime-only: patches and prepared authoring never see it.
    codec_motion: Option<CodecMotionFrame>,
    /// Conservative CPU/GPU working-set charge used while this source is an
    /// inactive prepared slot. Active sources are deliberately outside that
    /// evictable ledger, but the exact charge follows ownership on a swap.
    source_preload_bytes: u64,
    // Transport
    pub speed: f32, // 0.25..4.0 playback multiplier (1.0 = normal)
    pub fps: f32,   // target decode FPS (e.g. 30.0)
}

fn initial_performance_state(
    filename: &str,
    source_path: &str,
    fps: f32,
) -> (ClipSlots, ClipSlotId, ClipTransportState) {
    let slot = ClipSlotConfig::from_legacy(filename.to_owned(), source_path.to_owned(), 1.0, fps);
    (
        ClipSlots::singleton(slot),
        ClipSlotId::LEGACY,
        ClipTransportState::at(NormalizedTime::ZERO, PlaybackDirection::Forward),
    )
}

fn evaluate_transport_selection(
    config: &crate::transport::ClipTransportConfig,
    mut state: ClipTransportState,
    tick: ProgramTransportTick,
    decoder_generation: Option<u64>,
) -> (ClipTransportState, FrameSelection) {
    if let Some(decoder_generation) = decoder_generation {
        state.generation = state.generation.max(decoder_generation);
    }
    TransportTimeline::select(config, state, tick)
}

impl Layer {
    /// Open a persisted visual source. Still-image extensions use a bounded
    /// one-shot decoder; all other paths retain the ordinary video path so
    /// codec/container probing and contextual FFmpeg errors stay unchanged.
    #[allow(dead_code)]
    pub fn new(path: &str, device: &wgpu::Device) -> Result<Self, String> {
        let media_policy = MediaSafetyPolicy::safe();
        Self::new_with_media_policy(path, device, &media_policy)
    }

    /// Open a source under an explicit host-local media policy. This policy is
    /// supplied by the running host and is deliberately absent from patches.
    pub fn new_with_media_policy(
        path: &str,
        device: &wgpu::Device,
        media_policy: &MediaSafetyPolicy,
    ) -> Result<Self, String> {
        if is_still_image_file(std::path::Path::new(path)) {
            return Self::new_still(path, device, media_policy);
        }
        Self::new_video(path, device, media_policy)
    }

    fn new_video(
        path: &str,
        device: &wgpu::Device,
        media_policy: &MediaSafetyPolicy,
    ) -> Result<Self, String> {
        let device_limits = media_device_limits(device);
        let decoder = ThreadedDecoder::open_with_media_policy(path, media_policy, device_limits)?;
        let width = decoder.width;
        let height = decoder.height;
        let fps = decoder.fps;
        let media_plan = decoder.media_allocation_plan();
        let source_preload_bytes = media_plan.working_set_bytes;
        debug_assert_eq!((media_plan.width, media_plan.height), (width, height));
        validate_source_texture_dimensions_with_media_policy(
            width,
            height,
            device_limits,
            MediaSourceKind::Video,
            "video",
            media_policy,
        )?;
        let (texture, texture_view) = create_layer_texture(device, width, height, "Layer Texture")?;

        // Preserve a stable path independently from the short display label.
        // Canonicalization makes relative drag/drop and file-dialog paths
        // deterministic for later patch capture and export.
        let source_path = std::fs::canonicalize(path)
            .unwrap_or_else(|_| std::path::PathBuf::from(path))
            .to_string_lossy()
            .into_owned();

        // Extract just the filename from the path.
        let filename = std::path::Path::new(path)
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| path.to_string());

        let effects = EffectUniforms {
            resolution: [width as f32, height as f32],
            ..Default::default()
        };
        let (clip_slots, active_clip_slot, clip_transport) =
            initial_performance_state(&filename, &source_path, fps);

        Ok(Self {
            layer_id: allocate_layer_id(),
            source_path,
            persisted_source_reference: None,
            filename,
            source: LayerSource::Video(decoder),
            texture,
            texture_view,
            source_resource_epoch: 1,
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
            paused: false,
            visible: true,
            bypass_master_fx: default_bypass_master_fx(),
            reroll_on_loop: false,
            effects,
            rack: default_layer_rack(),
            transform: SpatialTransform::new_layer_default(),
            motion: default_layer_motion(),
            clip_slots,
            active_clip_slot,
            clip_transport,
            matte: LayerMatte::default(),
            pending_transport_seek: None,
            pending_transport_cue: None,
            last_transport_selection: None,
            width,
            height,
            source_error: String::new(),
            source_frame_initialized: false,
            codec_motion: None,
            source_preload_bytes,
            speed: 1.0,
            fps,
        })
    }

    fn new_still(
        path: &str,
        device: &wgpu::Device,
        media_policy: &MediaSafetyPolicy,
    ) -> Result<Self, String> {
        let device_limits = media_device_limits(device);
        let decoded = decode_still_image_with_media_policy(
            std::path::Path::new(path),
            media_policy,
            device_limits,
        )?;
        let width = decoded.width;
        let height = decoded.height;
        let source_preload_bytes = decoded.media_allocation_plan().working_set_bytes;
        debug_assert_eq!(
            (
                decoded.media_allocation_plan().width,
                decoded.media_allocation_plan().height
            ),
            (width, height)
        );
        validate_source_texture_dimensions_with_media_policy(
            width,
            height,
            device_limits,
            MediaSourceKind::Still,
            "still image",
            media_policy,
        )?;
        let (texture, texture_view) =
            create_layer_texture(device, width, height, "Still Layer Texture")?;

        let source_path = std::fs::canonicalize(path)
            .unwrap_or_else(|_| std::path::PathBuf::from(path))
            .to_string_lossy()
            .into_owned();
        let filename = std::path::Path::new(path)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string());
        let effects = EffectUniforms {
            resolution: [width as f32, height as f32],
            ..Default::default()
        };
        let (clip_slots, active_clip_slot, clip_transport) =
            initial_performance_state(&filename, &source_path, 30.0);
        Ok(Self {
            layer_id: allocate_layer_id(),
            source_path,
            persisted_source_reference: None,
            filename,
            source: LayerSource::Still(StillImage::from_decoded(decoded)),
            texture,
            texture_view,
            source_resource_epoch: 1,
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
            paused: false,
            visible: true,
            bypass_master_fx: default_bypass_master_fx(),
            reroll_on_loop: false,
            effects,
            rack: default_layer_rack(),
            transform: SpatialTransform::new_layer_default(),
            motion: default_layer_motion(),
            clip_slots,
            active_clip_slot,
            clip_transport,
            matte: LayerMatte::default(),
            pending_transport_seek: None,
            pending_transport_cue: None,
            last_transport_selection: None,
            width,
            height,
            source_error: String::new(),
            source_frame_initialized: false,
            codec_motion: None,
            source_preload_bytes,
            speed: 1.0,
            // A still has no source cadence. Retaining the conventional value
            // keeps old patch/config consumers finite while transport methods
            // below deliberately schedule no decode work.
            fps: 30.0,
        })
    }

    /// Create a live Spout receiver layer. The texture is initialized to
    /// transparent black so a missing/warming sender never exposes
    /// uninitialized GPU memory or masks lower layers. `SpoutIn` sanitizes the
    /// requested sender name before this stable `spout://` identity is persisted.
    #[allow(dead_code)]
    pub fn new_spout(
        sender_name: &str,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Self, String> {
        let media_policy = MediaSafetyPolicy::safe();
        Self::new_spout_with_media_policy(sender_name, device, queue, &media_policy)
    }

    pub fn new_spout_with_media_policy(
        sender_name: &str,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        media_policy: &MediaSafetyPolicy,
    ) -> Result<Self, String> {
        let device_limits = media_device_limits(device);
        let mut receiver =
            SpoutIn::new_with_media_policy(sender_name, media_policy.clone(), device_limits);
        let sanitized_name = receiver.status().sender_name;
        if sanitized_name.is_empty() {
            return Err("Spout sender name is empty after sanitization".to_string());
        }

        let width = 1;
        let height = 1;
        let source_preload_bytes = media_policy
            .plan(MediaSourceKind::Spout, width, height, device_limits)
            .map_err(|error| format!("Spout source rejected: {error}"))?
            .working_set_bytes;
        let (texture, texture_view) =
            create_layer_texture(device, width, height, "Spout Layer Texture")?;
        write_layer_texture_checked(
            device,
            queue,
            &texture,
            &[0, 0, 0, 0],
            width,
            height,
            "Spout Layer Texture",
        )?;
        receiver.start();

        let effects = EffectUniforms {
            resolution: [width as f32, height as f32],
            ..Default::default()
        };
        let source_path = format!("{SPOUT_SOURCE_PREFIX}{sanitized_name}");
        let filename = format!("Spout: {sanitized_name}");
        let (clip_slots, active_clip_slot, clip_transport) =
            initial_performance_state(&filename, &source_path, 30.0);
        Ok(Self {
            layer_id: allocate_layer_id(),
            source_path,
            persisted_source_reference: None,
            filename,
            source: LayerSource::Spout(receiver),
            texture,
            texture_view,
            source_resource_epoch: 1,
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
            paused: false,
            visible: true,
            bypass_master_fx: default_bypass_master_fx(),
            reroll_on_loop: false,
            effects,
            rack: default_layer_rack(),
            transform: SpatialTransform::new_layer_default(),
            motion: default_layer_motion(),
            clip_slots,
            active_clip_slot,
            clip_transport,
            matte: LayerMatte::default(),
            pending_transport_seek: None,
            pending_transport_cue: None,
            last_transport_selection: None,
            width,
            height,
            source_error: String::new(),
            source_frame_initialized: false,
            codec_motion: None,
            source_preload_bytes,
            speed: 1.0,
            fps: 30.0,
        })
    }

    pub fn source_kind(&self) -> &'static str {
        self.source.kind()
    }

    /// Reset only authored M4 motion controls. Source/transport resources,
    /// stable identity, geometry, effects, and routing remain untouched.
    pub fn reset_motion(&mut self) {
        reset_layer_motion(&mut self.motion);
    }

    /// Immutable identity of this live layer instance.
    pub fn layer_id(&self) -> u64 {
        self.layer_id
    }

    pub fn stable_layer_id(&self) -> StableLayerId {
        StableLayerId::new(self.layer_id)
            .expect("allocated live layer identities are always non-zero")
    }

    /// Restore an exact process-local identity while a complete replacement
    /// stack is still detached. Main validates uniqueness and stack length
    /// before calling this; creative routes are resolved only afterwards.
    pub(crate) fn restore_stable_layer_id_for_history(
        &mut self,
        layer_id: StableLayerId,
    ) -> Result<(), String> {
        observe_restored_layer_id(layer_id.get()).map_err(str::to_string)?;
        self.layer_id = layer_id.get();
        Ok(())
    }

    pub fn source_resource_epoch(&self) -> u64 {
        self.source_resource_epoch
    }

    pub fn active_clip_config(&self) -> &ClipSlotConfig {
        self.clip_slots
            .get(self.active_clip_slot)
            .expect("Layer maintains a valid active ClipSlot ID")
    }

    /// Install already-sanitized Set metadata after exact source
    /// reconstruction. The current decoder/texture must already represent the
    /// selected slot; this method changes no GPU or source resource.
    pub fn install_performance_state(
        &mut self,
        clip_slots: ClipSlots,
        active_clip_slot: ClipSlotId,
        matte: LayerMatte,
    ) -> Result<(), String> {
        let slot = clip_slots.get(active_clip_slot).ok_or_else(|| {
            format!(
                "active clip slot {} is absent from layer '{}'",
                active_clip_slot.get(),
                self.filename
            )
        })?;
        let saved_playhead = slot.saved_playhead;
        self.filename.clone_from(&slot.filename);
        let transport = slot.transport.sanitized();
        // `ClipTransportConfig` is authoritative. These public fields remain
        // compatibility mirrors only; they must not narrow the 0..=16 law.
        self.speed = transport.rate as f32;
        self.fps = slot
            .transport
            .sample_fps
            .unwrap_or(f64::from(self.fps))
            .clamp(0.25, 480.0) as f32;
        self.clip_transport = ClipTransportState::at(slot.saved_playhead, transport.direction);
        self.clip_slots = clip_slots;
        self.active_clip_slot = active_clip_slot;
        self.matte = matte.sanitized();
        // Layer construction seeds video at t=0. Exact patch reconstruction
        // must publish an absolute generation-tagged selection for the saved
        // playhead on its first program tick, including while Media Freeze is
        // holding presentation pixels, so resume can never expose the seed.
        self.pending_transport_seek = Some(saved_playhead);
        self.pending_transport_cue = None;
        self.last_transport_selection = None;
        self.codec_motion = None;
        Ok(())
    }

    /// Capture the canonical Set while keeping the legacy public cadence
    /// mirrors synchronized with the active slot.
    pub fn clip_slots_for_persistence(&self) -> ClipSlots {
        let mut slots = self.clip_slots.clone();
        if let Some(active) = slots.get_mut(self.active_clip_slot) {
            active.filename.clone_from(&self.filename);
            active.source_path = self.source_reference_for_persistence().to_owned();
            active.saved_playhead = self.clip_transport.position;
        }
        slots
    }

    /// Source duration supplied to the pure transport evaluator. Still and
    /// live inputs have no finite seekable duration and therefore report zero.
    pub fn source_duration_seconds(&self) -> f64 {
        match &self.source {
            LayerSource::Video(decoder) => decoder.duration_seconds,
            LayerSource::Still(_) | LayerSource::Spout(_) => 0.0,
        }
    }

    /// Bounded source-frame estimate used only to derive an exact selected
    /// frame index for deterministic diagnostics/export parity.
    pub fn source_frame_count(&self) -> u64 {
        match &self.source {
            LayerSource::Video(decoder) => {
                let estimate = decoder.duration_seconds * f64::from(decoder.fps);
                if estimate.is_finite() && estimate > 0.0 {
                    estimate.round().clamp(1.0, f64::from(u32::MAX)) as u64
                } else {
                    0
                }
            }
            LayerSource::Still(_) => 1,
            LayerSource::Spout(_) => 0,
        }
    }

    pub fn request_transport_seek(&mut self, position: NormalizedTime) {
        self.pending_transport_seek = Some(position);
        self.pending_transport_cue = None;
        self.codec_motion = None;
    }

    pub fn request_transport_cue(&mut self, cue_id: CueId) {
        self.pending_transport_cue = Some(cue_id);
        self.pending_transport_seek = None;
        self.codec_motion = None;
    }

    /// Apply one shared program-clock observation to the active slot and, for
    /// seekable video, publish a generation-tagged absolute selection to the
    /// decoder's newest-only mailbox. Still/Spout sources retain the same pure
    /// timeline snapshot but deliberately schedule no decoder work.
    /// Apply one program-clock observation using frame-local cadence
    /// overrides without mutating the authored ClipSlot. This preserves the
    /// legacy speed/FPS modulation surface while the Set retains the wider
    /// canonical transport ranges used by live and offline playback.
    pub fn apply_transport_tick_with_overrides(
        &mut self,
        mut tick: ProgramTransportTick,
        effective_rate: f64,
        effective_sample_fps: Option<f64>,
    ) -> Result<FrameSelection, String> {
        tick.source_duration_seconds = self.source_duration_seconds();
        tick.source_frame_count = self.source_frame_count();

        let queued_seek = self.pending_transport_seek.take();
        let queued_cue = self.pending_transport_cue.take();
        if tick.seek_to.is_none() {
            tick.seek_to = queued_seek;
        }
        if tick.cue_id.is_none() && tick.seek_to.is_none() {
            tick.cue_id = queued_cue;
        }

        // A decoder may already have received compatibility seek requests.
        // Rebase the pure state's generation before selecting so the next
        // absolute request can never be rejected as stale.
        let decoder_generation = match &self.source {
            LayerSource::Video(decoder) => Some(decoder.source_generation()),
            LayerSource::Still(_) | LayerSource::Spout(_) => None,
        };
        let mut config = self.active_clip_config().transport;
        config.rate = effective_rate;
        config.sample_fps = effective_sample_fps;
        let config = config.sanitized();
        let (next_state, selection) =
            evaluate_transport_selection(&config, self.clip_transport, tick, decoder_generation);
        self.clip_transport = next_state;
        if selection.discontinuity {
            self.codec_motion = None;
        }

        if selection.sample_due && !selection.transparent {
            if let LayerSource::Video(decoder) = &mut self.source {
                let accepted =
                    decoder.request_source_time(selection.generation, selection.source_seconds)?;
                if !accepted {
                    return Err(format!(
                        "decoder rejected stale transport generation {} for layer '{}'",
                        selection.generation, self.filename
                    ));
                }
            }
        }
        self.last_transport_selection = Some(selection);
        Ok(selection)
    }

    /// Authored `visible` remains independent; the evaluator combines it with
    /// this gate when deciding whether a OneShot source contributes.
    pub fn transport_visible(&self) -> bool {
        self.last_transport_selection
            .is_none_or(|selection| !selection.transparent)
    }

    pub(crate) fn prepared_slot_state_matches(
        &self,
        slot_id: ClipSlotId,
        expected_prior: Option<&ClipSlotConfig>,
    ) -> bool {
        self.clip_slots.get(slot_id) == expected_prior
            && (expected_prior.is_some()
                || self.clip_slots.len() < crate::performance::MAX_CLIP_SLOTS_PER_LAYER)
    }

    /// Infallible source-only half of an already validated atomic commit.
    /// Layer identity, effects, transform, blend, modulation-facing fields,
    /// matte, authored visibility, and pause state are deliberately untouched.
    pub(crate) fn commit_prepared_source(
        &mut self,
        activation: LayerSourceActivation,
        slot: &ClipSlotConfig,
        start_position: NormalizedTime,
        now: Instant,
    ) -> DisplacedLayerSource {
        // Persist the actual runtime playhead and source identity before the
        // active ownership is moved out. This exact value is also the
        // optimistic-concurrency revision of the returned prepared payload.
        let target_layer = self.stable_layer_id();
        let prior_start_position = self.clip_transport.position;
        let prior_source_fps = match &self.source {
            LayerSource::Video(decoder) => decoder.fps,
            LayerSource::Still(_) | LayerSource::Spout(_) => 30.0,
        };
        let mut prior_slot = self.active_clip_config().clone();
        prior_slot.filename.clone_from(&self.filename);
        prior_slot.source_path = self.source_reference_for_persistence().to_owned();
        prior_slot.saved_playhead = prior_start_position;
        self.clip_slots
            .upsert(prior_slot.clone())
            .expect("the active slot already occupies bounded Set capacity");

        let LayerSourceActivation {
            source_path,
            persisted_source_reference,
            filename,
            source,
            texture,
            texture_view,
            width,
            height,
            source_fps,
            preload_bytes,
            source_error,
            source_frame_initialized,
            codec_motion,
        } = activation;
        let displaced = LayerSourceActivation {
            source_path: std::mem::replace(&mut self.source_path, source_path),
            persisted_source_reference: std::mem::replace(
                &mut self.persisted_source_reference,
                persisted_source_reference,
            ),
            filename: std::mem::replace(&mut self.filename, filename),
            source: std::mem::replace(&mut self.source, source),
            texture: std::mem::replace(&mut self.texture, texture),
            texture_view: std::mem::replace(&mut self.texture_view, texture_view),
            width: self.width,
            height: self.height,
            source_fps: prior_source_fps,
            preload_bytes: self.source_preload_bytes,
            source_error: std::mem::replace(&mut self.source_error, source_error),
            source_frame_initialized: self.source_frame_initialized,
            codec_motion: std::mem::replace(&mut self.codec_motion, codec_motion),
        };
        self.width = width;
        self.height = height;
        self.source_resource_epoch = self.source_resource_epoch.wrapping_add(1).max(1);
        self.effects.resolution = [width as f32, height as f32];
        self.source_frame_initialized = source_frame_initialized;
        self.source_preload_bytes = preload_bytes;

        let transport = slot.transport.sanitized();
        self.clip_slots
            .upsert(slot.clone())
            .expect("prepared slot capacity and prior state were validated before commit");
        self.active_clip_slot = slot.id;
        self.clip_transport = ClipTransportState::at(start_position, transport.direction);
        self.speed = transport.rate as f32;
        self.fps = transport
            .sample_fps
            .unwrap_or(f64::from(source_fps))
            .clamp(0.25, 480.0) as f32;
        self.pending_transport_seek = None;
        self.pending_transport_cue = None;
        self.last_transport_selection = None;
        self.reset_transport_timing_at(now);

        DisplacedLayerSource {
            target_layer,
            slot: prior_slot,
            start_position: prior_start_position,
            activation: displaced,
        }
    }

    /// Keep a verified content reference while the decoder uses its resolved
    /// host path. Passing `None` restores ordinary path-based persistence.
    pub(crate) fn set_persisted_source_reference(&mut self, reference: Option<String>) {
        self.persisted_source_reference = reference;
    }

    /// Source identity written to patches and handed to offline export.
    pub fn source_reference_for_persistence(&self) -> &str {
        source_reference_for_persistence(
            &self.source_path,
            self.persisted_source_reference.as_deref(),
        )
    }

    /// Return a previously verified portable identity for proxy assessment.
    /// Ordinary/invalid host paths intentionally produce no identity and this
    /// accessor performs no filesystem read or warm-path fingerprinting.
    pub fn content_identity_for_proxy(&self) -> Option<crate::media_source::ContentIdentity> {
        content_identity_for_proxy_reference(self.source_reference_for_persistence())
    }

    pub fn is_video(&self) -> bool {
        matches!(self.source, LayerSource::Video(_))
    }

    /// Persisted sources that can be reconstructed for deterministic offline
    /// export. Live Spout input is intentionally excluded.
    pub fn is_file_media(&self) -> bool {
        matches!(self.source, LayerSource::Video(_) | LayerSource::Still(_))
    }

    pub fn progress(&self) -> f32 {
        match &self.source {
            LayerSource::Video(decoder) => decoder.progress(),
            LayerSource::Still(_) => 0.0,
            LayerSource::Spout(_) => 0.0,
        }
    }

    pub fn source_frame_initialized(&self) -> bool {
        self.source_frame_initialized
    }

    /// Codec metadata committed with the exact pixels currently in the source
    /// texture. Intra/unavailable/rejected products remain visible here for
    /// truthful source diagnostics; only `Available` products yield vectors.
    pub fn codec_motion(&self) -> Option<&CodecMotionFrame> {
        self.codec_motion.as_ref()
    }

    /// Take a file source's initial/newest frame. Video publishes the initial
    /// seed plus requested advances; a still publishes exactly one immutable
    /// RGBA frame. Call this before pause gating so either kind starts defined.
    pub fn take_ready_media_frame(&mut self) -> Result<Option<ReadyFrame>, String> {
        match &mut self.source {
            LayerSource::Video(decoder) => decoder.try_next_ready_frame_result(),
            LayerSource::Still(image) => Ok(image.take_frame().map(ReadyFrame::still)),
            LayerSource::Spout(_) => Ok(None),
        }
    }

    pub fn restore_ready_media_frame_after_failed_upload(&mut self, frame: ReadyFrame) {
        if let LayerSource::Still(image) = &mut self.source {
            image.restore_frame_after_failed_upload(frame.rgba);
        }
    }

    /// Upload one decoded/still product and commit its codec metadata only
    /// after the matching RGBA write succeeds. On error the caller retains the
    /// complete frame for existing still-image retry behavior.
    pub fn upload_ready_media_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &mut ReadyFrame,
    ) -> Result<(), String> {
        let upload_started = Instant::now();
        let upload_result = self.upload_frame(device, queue, &frame.rgba);
        let upload_duration = upload_started.elapsed();
        if upload_result.is_ok() {
            if let LayerSource::Video(decoder) = &mut self.source {
                decoder.record_upload_duration(upload_duration);
            }
        }
        upload_result?;
        self.codec_motion = take_matching_codec_motion(frame, [self.width, self.height]);
        Ok(())
    }

    /// Compatibility name retained for existing render-loop callers. It now
    /// harvests either supported file source; new call sites should use the
    /// accurately named `take_ready_media_frame` wrapper above.
    #[allow(dead_code)]
    pub fn take_ready_video_frame(&mut self) -> Result<Option<Vec<u8>>, String> {
        self.take_ready_media_frame()
            .map(|ready| ready.map(|frame| frame.rgba))
    }

    /// Stable video decoder health for state snapshots.
    pub fn video_health(&self) -> Option<DecoderHealth> {
        match &self.source {
            LayerSource::Video(decoder) => Some(decoder.health()),
            LayerSource::Still(_) => None,
            LayerSource::Spout(_) => None,
        }
    }

    /// Allocation-free newest-only decoder timing/loss truth. Still and live
    /// inputs have no threaded decoder and therefore report absence rather
    /// than manufactured zeroes.
    pub fn video_telemetry(&self) -> Option<DecoderTelemetry> {
        match &self.source {
            LayerSource::Video(decoder) => Some(decoder.telemetry()),
            LayerSource::Still(_) | LayerSource::Spout(_) => None,
        }
    }

    pub fn spout_status(&self) -> Option<SpoutStatus> {
        match &self.source {
            LayerSource::Spout(receiver) => {
                let mut status = receiver.status();
                if !self.source_error.is_empty() {
                    status.error.clone_from(&self.source_error);
                }
                Some(status)
            }
            LayerSource::Video(_) | LayerSource::Still(_) => None,
        }
    }

    pub fn source_error(&self) -> &str {
        &self.source_error
    }

    /// Take at most the newest live frame. The receiver's one-frame slot
    /// discards intermediate images, so a slow render tick never builds lag.
    pub fn try_spout_frame(&self) -> Option<SpoutFrame> {
        match &self.source {
            LayerSource::Spout(receiver) => receiver.try_recv(),
            LayerSource::Video(_) | LayerSource::Still(_) => None,
        }
    }

    /// Resize the GPU source texture atomically when a Spout sender changes
    /// dimensions, then upload the complete RGBA frame.
    pub fn upload_spout_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: SpoutFrame,
    ) -> Result<(), String> {
        if !matches!(self.source, LayerSource::Spout(_)) {
            return Err("cannot upload a Spout frame to a file-backed layer".to_string());
        }
        let expected_len = match (frame.width as usize)
            .checked_mul(frame.height as usize)
            .and_then(|pixels| pixels.checked_mul(4))
        {
            Some(len) => len,
            None => {
                let error = format!(
                    "Spout frame dimensions overflow: {}x{}",
                    frame.width, frame.height
                );
                self.source_error.clone_from(&error);
                return Err(error);
            }
        };
        if frame.width == 0 || frame.height == 0 || frame.pixels.len() != expected_len {
            let error = format!(
                "invalid Spout frame {}x{}: expected {expected_len} RGBA bytes, got {}",
                frame.width,
                frame.height,
                frame.pixels.len()
            );
            self.source_error.clone_from(&error);
            return Err(error);
        }
        let max_dimension = device.limits().max_texture_dimension_2d;
        if frame.width > max_dimension || frame.height > max_dimension {
            let error = format!(
                "Spout frame {}x{} exceeds this GPU's {max_dimension}px 2D texture limit",
                frame.width, frame.height
            );
            self.source_error.clone_from(&error);
            return Err(error);
        }

        if frame.width != self.width || frame.height != self.height {
            let (texture, texture_view) =
                create_layer_texture(device, frame.width, frame.height, "Spout Layer Texture")
                    .inspect_err(|error| self.source_error.clone_from(error))?;
            if let Err(error) = write_layer_texture_checked(
                device,
                queue,
                &texture,
                &frame.pixels,
                frame.width,
                frame.height,
                "Spout Layer Texture",
            ) {
                self.source_error.clone_from(&error);
                return Err(error);
            }
            self.texture = texture;
            self.texture_view = texture_view;
            self.source_resource_epoch = self.source_resource_epoch.wrapping_add(1).max(1);
            self.width = frame.width;
            self.height = frame.height;
            self.effects.resolution = [frame.width as f32, frame.height as f32];
            self.source_frame_initialized = true;
            self.codec_motion = None;
            // Spout uses the same four-RGBA-buffer conservative planning
            // multiplier as its admission plan. The frame length is already
            // validated above, so this conversion is exact and bounded.
            self.source_preload_bytes = u64::try_from(frame.pixels.len())
                .unwrap_or(u64::MAX)
                .saturating_mul(4);
            self.source_error.clear();
            return Ok(());
        }
        self.upload_frame(device, queue, &frame.pixels)
    }

    /// Compatibility seam retained for callers that reset legacy pacing on a
    /// source discontinuity. Absolute `TransportTimeline` selection carries
    /// no local pacing debt, so the operation is intentionally a no-op.
    pub fn reset_transport_timing(&mut self) {
        self.reset_transport_timing_at(Instant::now());
    }

    /// Timestamped compatibility form; see [`Self::reset_transport_timing`].
    pub(crate) fn reset_transport_timing_at(&mut self, _now: Instant) {}

    pub fn upload_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        rgba_data: &[u8],
    ) -> Result<(), String> {
        match write_layer_texture_checked(
            device,
            queue,
            &self.texture,
            rgba_data,
            self.width,
            self.height,
            "Layer Source Texture",
        ) {
            Ok(()) => {
                self.source_frame_initialized = true;
                // Generic RGBA uploads have no paired decoder product. The
                // motion-aware sibling writes a matching product afterward.
                self.codec_motion = None;
                self.source_error.clear();
                Ok(())
            }
            Err(error) => {
                self.source_error.clone_from(&error);
                Err(error)
            }
        }
    }
}

fn write_layer_texture_checked(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    rgba_data: &[u8],
    width: u32,
    height: u32,
    label: &str,
) -> Result<(), String> {
    let expected = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(4))
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or_else(|| format!("{label} upload dimensions overflow for {width}x{height}"))?;
    if rgba_data.len() != expected {
        return Err(format!(
            "{label} upload has {} bytes; expected {expected} for {width}x{height}",
            rgba_data.len()
        ));
    }

    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba_data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * width),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    let _ = device.poll(wgpu::PollType::Poll);
    let errors = [
        pollster::block_on(out_of_memory.pop()),
        pollster::block_on(internal.pop()),
        pollster::block_on(validation.pop()),
    ]
    .into_iter()
    .flatten()
    .map(|error| error.to_string())
    .collect::<Vec<_>>();
    if let Some(error) = layer_texture_upload_error(label, width, height, &errors) {
        Err(error)
    } else {
        Ok(())
    }
}

fn layer_texture_upload_error(
    label: &str,
    width: u32,
    height: u32,
    errors: &[String],
) -> Option<String> {
    (!errors.is_empty()).then(|| {
        format!(
            "could not upload {label} at {width}x{height}: {}",
            errors.join("; ")
        )
    })
}

fn create_layer_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &str,
) -> Result<(wgpu::Texture, wgpu::TextureView), String> {
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let errors = [
        pollster::block_on(out_of_memory.pop()),
        pollster::block_on(internal.pop()),
        pollster::block_on(validation.pop()),
    ];
    if let Some(error) = errors.into_iter().flatten().next() {
        return Err(format!(
            "could not allocate {label} at {width}x{height}: {error}"
        ));
    }
    Ok((texture, texture_view))
}

#[allow(dead_code)] // Compatibility Safe wrapper for legacy validation call sites.
pub(crate) fn validate_source_texture_dimensions(
    width: u32,
    height: u32,
    max_dimension: u32,
    source_kind: &str,
) -> Result<(), String> {
    validate_safe_dimensions(
        MediaSourceKind::Video,
        width,
        height,
        MediaDeviceLimits::texture_only(max_dimension),
    )
    .map(|_| ())
    .map_err(|error| format!("{source_kind} source rejected: {error}"))
}

/// Policy-aware source-texture admission used before decoder and GPU
/// allocation. The returned plan can be surfaced in operator diagnostics.
pub(crate) fn validate_source_texture_dimensions_with_media_policy(
    width: u32,
    height: u32,
    device_limits: MediaDeviceLimits,
    source_kind: MediaSourceKind,
    source_label: &str,
    media_policy: &MediaSafetyPolicy,
) -> Result<MediaAllocationPlan, String> {
    media_policy
        .plan(source_kind, width, height, device_limits)
        .map_err(|error| format!("{source_label} source rejected: {error}"))
}

fn media_device_limits(device: &wgpu::Device) -> MediaDeviceLimits {
    let limits = device.limits();
    MediaDeviceLimits::new(limits.max_texture_dimension_2d, limits.max_buffer_size)
}

/// Parse the stable source identity used for persisted Spout receiver layers.
/// An empty suffix is returned as `Some("")` so callers can distinguish an
/// invalid Spout source from an ordinary video path and report it precisely.
pub fn spout_sender_from_source_path(source_path: &str) -> Option<&str> {
    source_path.strip_prefix(SPOUT_SOURCE_PREFIX)
}

/// Valid video file extensions for drag-and-drop.
pub fn is_video_file(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension_in(extension, VIDEO_EXTENSIONS))
}

/// Supported immutable visual sources. Animated image formats are omitted on
/// purpose: each accepted image has unambiguous one-frame transport semantics.
pub fn is_still_image_file(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension_in(extension, STILL_IMAGE_EXTENSIONS))
}

pub fn is_supported_visual_extension(extension: &str) -> bool {
    extension_in(extension, VIDEO_EXTENSIONS) || extension_in(extension, STILL_IMAGE_EXTENSIONS)
}

/// File types that can become ordinary effect/composite layers and can be
/// reconstructed frame-for-frame during offline export.
pub fn is_supported_visual_file(path: &std::path::Path) -> bool {
    is_video_file(path) || is_still_image_file(path)
}

fn extension_in(extension: &str, allowed: &[&str]) -> bool {
    allowed
        .iter()
        .any(|candidate| extension.eq_ignore_ascii_case(candidate))
}

#[cfg(test)]
mod tests {
    use super::{
        allocate_layer_id, content_identity_for_proxy_reference, create_layer_texture,
        default_bypass_master_fx, default_layer_motion, default_layer_rack,
        evaluate_transport_selection, is_still_image_file, is_supported_visual_extension,
        is_supported_visual_file, is_video_file, layer_texture_upload_error, reset_layer_motion,
        source_reference_for_persistence, spout_sender_from_source_path,
        take_matching_codec_motion, validate_source_texture_dimensions,
        write_layer_texture_checked, BlendMode,
    };
    use crate::transport::{
        ClipTransportConfig, ClipTransportState, EndBehavior, NormalizedTime, PlaybackDirection,
        ProgramTransportTick,
    };
    #[test]
    fn live_transport_seam_covers_freeze_reverse_seek_generation_and_terminal_transparency() {
        let reverse = ClipTransportConfig {
            direction: PlaybackDirection::Reverse,
            rate: 1.0,
            ..ClipTransportConfig::default()
        };
        let state =
            ClipTransportState::at(NormalizedTime::clamped(0.5), PlaybackDirection::Reverse);
        let frozen_tick = ProgramTransportTick {
            delta_seconds: 1.0,
            program_running: false,
            media_running: true,
            source_duration_seconds: 10.0,
            source_frame_count: 100,
            ..ProgramTransportTick::default()
        };
        let (state, frozen) = evaluate_transport_selection(&reverse, state, frozen_tick, Some(7));
        assert_eq!(frozen.logical_time, NormalizedTime::clamped(0.5));
        assert!(frozen.held);
        assert!(frozen.generation >= 7);

        let seek_tick = ProgramTransportTick {
            source_duration_seconds: 10.0,
            source_frame_count: 100,
            seek_to: Some(NormalizedTime::clamped(0.8)),
            ..ProgramTransportTick::default()
        };
        let (state, sought) = evaluate_transport_selection(&reverse, state, seek_tick, Some(9));
        assert_eq!(sought.logical_time, NormalizedTime::clamped(0.8));
        assert!(sought.discontinuity);
        assert!(sought.generation > 9);

        let reverse_tick = ProgramTransportTick {
            delta_seconds: 1.0,
            source_duration_seconds: 10.0,
            source_frame_count: 100,
            ..ProgramTransportTick::default()
        };
        let (_, reversed) = evaluate_transport_selection(&reverse, state, reverse_tick, None);
        assert!((reversed.logical_time.get() - 0.7).abs() < 1e-9);

        let one_shot = ClipTransportConfig {
            end_behavior: EndBehavior::OneShot,
            rate: 1.0,
            ..ClipTransportConfig::default()
        };
        let state =
            ClipTransportState::at(NormalizedTime::clamped(0.9), PlaybackDirection::Forward);
        let crossing_tick = ProgramTransportTick {
            delta_seconds: 2.0,
            source_duration_seconds: 10.0,
            source_frame_count: 100,
            ..ProgramTransportTick::default()
        };
        let (state, terminal) = evaluate_transport_selection(&one_shot, state, crossing_tick, None);
        assert!(terminal.completed);
        assert!(
            !terminal.transparent,
            "terminal frame presents exactly once"
        );
        let (_, after_terminal) = evaluate_transport_selection(
            &one_shot,
            state,
            ProgramTransportTick {
                source_duration_seconds: 10.0,
                source_frame_count: 100,
                ..ProgramTransportTick::default()
            },
            None,
        );
        assert!(after_terminal.transparent);
    }

    #[test]
    fn spout_source_identity_is_unambiguous_and_round_trippable() {
        assert_eq!(
            spout_sender_from_source_path("spout://Resolume Composition"),
            Some("Resolume Composition")
        );
        assert_eq!(spout_sender_from_source_path("spout://"), Some(""));
        assert_eq!(spout_sender_from_source_path("C:\\clips\\spout.mp4"), None);
    }

    #[test]
    fn resolved_runtime_path_does_not_replace_retained_content_identity() {
        let runtime = r"D:\host\resolved\clip.mp4";
        let reference = format!("cos-sha256://{}/1234", "a".repeat(64));

        assert_eq!(source_reference_for_persistence(runtime, None), runtime);
        assert_eq!(
            source_reference_for_persistence(runtime, Some(&reference)),
            reference
        );
    }

    #[test]
    fn proxy_identity_uses_only_a_valid_retained_content_reference() {
        let digest = "a".repeat(64);
        let reference = format!("cos-sha256://{digest}/1234");
        let identity = content_identity_for_proxy_reference(&reference).unwrap();
        assert_eq!(identity.sha256, digest);
        assert_eq!(identity.byte_len, 1234);
        assert_eq!(
            content_identity_for_proxy_reference(r"D:\host\clip.mp4"),
            None
        );
        assert_eq!(
            content_identity_for_proxy_reference("cos-sha256://not-a-digest/1234"),
            None
        );
    }

    #[test]
    fn blend_mode_protocol_keys_match_web_option_values() {
        let keys = [
            "normal",
            "screen",
            "multiply",
            "difference",
            "add",
            "subtract",
            "darken",
            "lighten",
            "overlay",
            "soft_light",
            "hard_light",
            "exclusion",
            "dodge",
            "burn",
            "alpha_cut",
        ];
        for (code, (mode, key)) in BlendMode::ALL.into_iter().zip(keys).enumerate() {
            assert_eq!(mode.as_u32(), code as u32);
            assert_eq!(BlendMode::from_u32(code as u32), Some(mode));
            assert_eq!(mode.key(), key);
            assert_eq!(BlendMode::from_key(key), Some(mode));
        }
        assert_eq!(BlendMode::from_key("Soft Light"), None);
        assert_eq!(BlendMode::from_key("unknown"), None);
        assert_eq!(BlendMode::from_u32(15), None);
        assert_eq!(BlendMode::from_u32(u32::MAX), None);
    }

    #[test]
    fn live_layer_ids_are_nonzero_and_unique() {
        let first = allocate_layer_id();
        let second = allocate_layer_id();
        assert_ne!(first, 0);
        assert_ne!(second, 0);
        assert_ne!(first, second);
    }

    #[test]
    fn every_layer_constructor_uses_opt_in_master_fx_bypass_default() {
        // Video, still, and Spout constructors all call this one default,
        // preventing one source kind from unexpectedly entering bypass mode.
        assert!(!default_bypass_master_fx());
    }

    #[test]
    fn every_layer_constructor_starts_with_the_exact_frozen_legacy_rack() {
        let rack = default_layer_rack();
        assert!(rack.is_exact_legacy(crate::visual_rack::LegacyRackScope::Layer));
        assert_eq!(
            rack,
            crate::visual_rack::RuntimeVisualRack::synthetic_legacy(
                crate::visual_rack::LegacyRackScope::Layer
            )
        );
    }

    #[test]
    fn every_layer_constructor_and_motion_reset_use_the_exact_no_op_contract() {
        let defaults = default_layer_motion();
        assert_eq!(defaults, crate::motion::MotionParams::default());
        assert!(defaults.is_exact_zero());

        let mut authored = crate::motion::MotionParams {
            transplant: crate::motion::FaradayParams {
                amount: 0.8,
                ..crate::motion::FaradayParams::default()
            },
            shutter: crate::motion::CurvedShutterParams {
                angle_degrees: 270.0,
                ..crate::motion::CurvedShutterParams::default()
            },
            ..crate::motion::MotionParams::default()
        };
        reset_layer_motion(&mut authored);
        assert_eq!(authored, defaults);
    }

    #[test]
    fn codec_metadata_commits_only_for_matching_pixels_generation_and_dimensions() {
        use crate::motion::MOTION_ALGORITHM_VERSION;
        use crate::video::threaded::ReadyFrame;
        use crate::video::{
            CodecMotionFrame, CodecMotionFrameType, CodecMotionProvenance, CodecMotionStatus,
        };

        fn ready(source_generation: u64, motion_generation: u64) -> ReadyFrame {
            ReadyFrame {
                rgba: vec![0; 16 * 16 * 4],
                codec_motion: Some(CodecMotionFrame {
                    source_dimensions: [16, 16],
                    frame_delta_seconds: 1.0 / 30.0,
                    source_generation: motion_generation,
                    frame_ordinal: 3,
                    algorithm_version: MOTION_ALGORITHM_VERSION,
                    provenance: CodecMotionProvenance::FfmpegExportMvs,
                    frame_type: CodecMotionFrameType::Intra,
                    status: CodecMotionStatus::Intra,
                    vectors: Vec::new(),
                }),
                loops_advanced: 0,
                source_generation,
                pts: Some(3),
                source_seconds: 0.1,
                duration_seconds: 1.0,
            }
        }

        let mut matching = ready(7, 7);
        assert_eq!(
            take_matching_codec_motion(&mut matching, [16, 16])
                .expect("matching pixels and metadata")
                .source_generation,
            7
        );
        assert!(matching.codec_motion.is_none());

        let mut stale = ready(8, 7);
        assert!(take_matching_codec_motion(&mut stale, [16, 16]).is_none());
        assert!(stale.codec_motion.is_none(), "stale metadata is discarded");

        let mut wrong_dimensions = ready(7, 7);
        assert!(take_matching_codec_motion(&mut wrong_dimensions, [8, 8]).is_none());
        assert!(wrong_dimensions.codec_motion.is_none());
    }

    #[test]
    fn source_texture_dimensions_reject_zero_and_gpu_oversize() {
        assert!(validate_source_texture_dimensions(0, 1080, 8192, "video").is_err());
        assert!(validate_source_texture_dimensions(1920, 0, 8192, "video").is_err());
        assert!(validate_source_texture_dimensions(8193, 1080, 8192, "video").is_err());
        assert!(validate_source_texture_dimensions(1920, 1080, 8192, "video").is_ok());
        assert!(validate_source_texture_dimensions(3840, 2160, 16_384, "video").is_ok());
        assert!(validate_source_texture_dimensions(3840, 2161, 16_384, "video").is_err());
        assert!(validate_source_texture_dimensions(3000, 3000, 16_384, "video").is_err());
    }

    #[test]
    fn source_texture_upload_errors_are_contextual_and_recoverable() {
        assert!(layer_texture_upload_error("Layer", 7680, 4320, &[]).is_none());
        let error = layer_texture_upload_error(
            "Layer",
            7680,
            4320,
            &["Out of Memory".to_string(), "validation failed".to_string()],
        )
        .expect("scoped GPU error");
        assert!(error.contains("Layer"));
        assert!(error.contains("7680x4320"));
        assert!(error.contains("Out of Memory"));
        assert!(error.contains("validation failed"));
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_source_upload_scope_reports_invalid_extent_and_accepts_valid_rgba() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("GPU adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Layer Upload Scope Test"),
            ..Default::default()
        }))
        .expect("GPU device");
        let (texture, _view) =
            create_layer_texture(&device, 1, 1, "Layer Upload Scope Test").unwrap();

        let error = write_layer_texture_checked(
            &device,
            &queue,
            &texture,
            &[0; 8],
            2,
            1,
            "Layer Upload Scope Test",
        )
        .expect_err("extent wider than the texture must be recoverable");
        assert!(error.contains("Layer Upload Scope Test"));
        assert!(write_layer_texture_checked(
            &device,
            &queue,
            &texture,
            &[10, 20, 30, 255],
            1,
            1,
            "Layer Upload Scope Test",
        )
        .is_ok());
    }

    #[test]
    fn visual_file_classifier_accepts_supported_stills_case_insensitively() {
        for filename in [
            "frame.png",
            "frame.PNG",
            "photo.jpg",
            "photo.JPEG",
            "plate.bmp",
            "overlay.WeBp",
        ] {
            let path = std::path::Path::new(filename);
            assert!(is_still_image_file(path), "{filename}");
            assert!(is_supported_visual_file(path), "{filename}");
            assert!(!is_video_file(path), "{filename}");
            assert!(is_supported_visual_extension(
                path.extension().unwrap().to_str().unwrap()
            ));
        }
        for filename in ["clip.webm", "clip.mp4"] {
            let path = std::path::Path::new(filename);
            assert!(!is_still_image_file(path), "{filename}");
            assert!(is_supported_visual_file(path), "{filename}");
        }
        for filename in ["animation.gif", "sound.wav", "frame", "frame.png.exe"] {
            assert!(!is_supported_visual_file(std::path::Path::new(filename)));
        }
    }
}
