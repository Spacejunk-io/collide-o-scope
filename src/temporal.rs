//! Pure temporal-domain planning shared by Exact, Advanced, live, and export.
//!
//! This module owns CPU state and laws only. It deliberately has no `wgpu`,
//! clock, filesystem, or UI dependency: callers provide elapsed program time
//! and an explicit event batch, stage one frame, and then either commit or
//! discard it at the outer accepted-frame boundary.

#![allow(
    dead_code,
    reason = "shared temporal event/reset adapters land across the staged M3 host integrations"
)]

use crate::effects::params::{normalized_slit_direction, TemporalParams, TEMPORAL_REFERENCE_FPS};
use crate::image_routing::StableLayerId;
use crate::performance::SavedLayerPosition;

/// Frames of clean output history retained at the 30 Hz authoring rate.
pub(crate) const TEMPORAL_HISTORY_LEN: u32 = 24;

/// The topology used to turn image position into a clean-history age.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TemporalTopology {
    #[default]
    Linear,
    Radial,
    Spiral,
    Contour,
    Folded,
    Kaleidoscopic,
}

/// Sampling law for a non-integral clean-history age.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TemporalInterpolation {
    /// The exact legacy slit-scan law.
    #[default]
    Floor,
    Linear,
}

impl TemporalTopology {
    pub(crate) const fn gpu_code(self) -> u32 {
        match self {
            Self::Linear => 0,
            Self::Radial => 1,
            Self::Spiral => 2,
            Self::Contour => 3,
            Self::Folded => 4,
            Self::Kaleidoscopic => 5,
        }
    }
}

impl TemporalInterpolation {
    pub(crate) const fn gpu_code(self) -> u32 {
        match self {
            Self::Floor => 0,
            Self::Linear => 1,
        }
    }
}

/// Zero/default authoring surface for Temporal Topology Loom. `amount == 0`
/// selects the frozen legacy shader; every non-zero amount selects the bounded
/// additive originals pipeline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemporalLoomParams {
    pub amount: f32,
    pub topology: TemporalTopology,
    pub interpolation: TemporalInterpolation,
    pub depth: f32,
    pub phase: f32,
    pub scale: f32,
    pub angle: f32,
    pub folds: u8,
    /// Zero means unquantized; non-zero values are bounded by the ring size.
    pub quantization: u8,
}

impl Default for TemporalLoomParams {
    fn default() -> Self {
        Self {
            amount: 0.0,
            topology: TemporalTopology::Linear,
            interpolation: TemporalInterpolation::Floor,
            depth: 1.0,
            phase: 0.0,
            scale: 1.0,
            angle: 0.0,
            folds: 1,
            quantization: 0,
        }
    }
}

impl TemporalLoomParams {
    fn sanitized(self) -> Self {
        Self {
            amount: finite_or(self.amount, 0.0).clamp(0.0, 1.0),
            topology: self.topology,
            interpolation: self.interpolation,
            depth: finite_or(self.depth, 1.0).clamp(0.0, 1.0),
            phase: finite_or(self.phase, 0.0).clamp(-1_000.0, 1_000.0),
            scale: finite_or(self.scale, 1.0).clamp(0.01, 100.0),
            angle: finite_or(self.angle, 0.0).clamp(-180.0, 180.0),
            folds: self.folds.clamp(1, 16),
            quantization: self.quantization.min(TEMPORAL_HISTORY_LEN as u8),
        }
    }
}

/// Zero/default authoring surface for the deterministic Collision Atlas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CollisionAtlasParams {
    pub amount: f32,
    pub seed: u32,
    pub territories: u8,
    pub collision: f32,
}

impl Default for CollisionAtlasParams {
    fn default() -> Self {
        Self {
            amount: 0.0,
            seed: 0,
            territories: 8,
            collision: 0.0,
        }
    }
}

impl CollisionAtlasParams {
    fn sanitized(self) -> Self {
        Self {
            amount: finite_or(self.amount, 0.0).clamp(0.0, 1.0),
            seed: self.seed,
            territories: self.territories.clamp(1, 64),
            collision: finite_or(self.collision, 0.0).clamp(0.0, 1.0),
        }
    }
}

/// Admission signal selected by Refresh Garden.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RefreshGardenGate {
    #[default]
    TemporalDelta,
    Luma,
    Chroma,
    CellularRidge,
    AudioEnergy,
    AudioOnset,
    Matte,
}

impl RefreshGardenGate {
    pub(crate) const fn gpu_code(self) -> u32 {
        match self {
            Self::TemporalDelta => 0,
            Self::Luma => 1,
            Self::Chroma => 2,
            Self::CellularRidge => 3,
            Self::AudioEnergy => 4,
            Self::AudioOnset => 5,
            Self::Matte => 6,
        }
    }
}

/// Zero/default authoring surface for Refresh Garden.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RefreshGardenParams {
    pub amount: f32,
    pub gate: RefreshGardenGate,
    pub threshold: f32,
    pub softness: f32,
    pub decay: f32,
    /// Zero disables forced periodic admission.
    pub max_hold_ticks: u32,
}

impl Default for RefreshGardenParams {
    fn default() -> Self {
        Self {
            amount: 0.0,
            gate: RefreshGardenGate::TemporalDelta,
            threshold: 0.1,
            softness: 0.03,
            decay: 1.0,
            max_hold_ticks: 0,
        }
    }
}

impl RefreshGardenParams {
    fn sanitized(self) -> Self {
        Self {
            amount: finite_or(self.amount, 0.0).clamp(0.0, 1.0),
            gate: self.gate,
            threshold: finite_or(self.threshold, 0.1).clamp(0.0, 1.0),
            softness: finite_or(self.softness, 0.03).clamp(0.0, 0.5),
            decay: finite_or(self.decay, 1.0).clamp(0.0, 1.0),
            max_hold_ticks: self.max_hold_ticks,
        }
    }
}

/// Explicit event stream capable of advancing Collision Score.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CollisionScoreTrigger {
    #[default]
    Boundary,
    Downbeat,
    AudioOnset,
    Manual,
}

/// Runtime identity of the loop source allowed to conduct a boundary-driven
/// Collision Score. Patch persistence stores only the saved position; a live
/// stable identity is resolved again after loading.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CollisionScoreLoopDriver {
    #[default]
    None,
    SelectedLayer {
        layer_id: StableLayerId,
        saved_position: SavedLayerPosition,
    },
    MissingSelectedLayer {
        saved_position: SavedLayerPosition,
    },
}

/// Zero/default authoring surface for the deterministic Collision Score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollisionScoreParams {
    pub enabled: bool,
    pub seed: u32,
    pub state_count: u8,
    pub trigger: CollisionScoreTrigger,
    pub loop_driver: CollisionScoreLoopDriver,
}

impl Default for CollisionScoreParams {
    fn default() -> Self {
        Self {
            enabled: false,
            seed: 0,
            state_count: 4,
            trigger: CollisionScoreTrigger::Boundary,
            loop_driver: CollisionScoreLoopDriver::None,
        }
    }
}

impl CollisionScoreParams {
    fn sanitized(self) -> Self {
        Self {
            enabled: self.enabled,
            seed: self.seed,
            state_count: self.state_count.clamp(2, 16),
            trigger: self.trigger,
            loop_driver: self.loop_driver,
        }
    }
}

/// Event-local memory reset selected by the temporal originals authoring
/// surface. Reset execution is deliberately outside the GPU shader.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TemporalEventResetMode {
    #[default]
    None,
    Score,
    Memory,
    All,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TemporalResetPolicy {
    pub loop_boundary: TemporalEventResetMode,
    pub downbeat: TemporalEventResetMode,
}

/// M3 authoring contract. Every member defaults to a strict zero/no-op.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TemporalOriginalsParams {
    pub loom: TemporalLoomParams,
    pub atlas: CollisionAtlasParams,
    pub garden: RefreshGardenParams,
    pub score: CollisionScoreParams,
    pub reset: TemporalResetPolicy,
}

impl TemporalOriginalsParams {
    pub(crate) fn sanitized(self) -> Self {
        Self {
            loom: self.loom.sanitized(),
            atlas: self.atlas.sanitized(),
            garden: self.garden.sanitized(),
            score: self.score.sanitized(),
            reset: self.reset,
        }
    }

    pub(crate) fn is_zero(self) -> bool {
        self.loom.amount == 0.0
            && self.atlas.amount == 0.0
            && self.garden.amount == 0.0
            && !self.score.enabled
    }
}

/// Per-frame event counts. Counts, rather than booleans, preserve multiple
/// loop/downbeat/onset/manual events that occur between two rendered frames.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct TemporalFrameEvents {
    pub boundary_events: u32,
    pub downbeat_events: u32,
    pub audio_onset_events: u32,
    pub manual_events: u32,
}

impl TemporalFrameEvents {
    fn count_for(self, trigger: CollisionScoreTrigger) -> u32 {
        match trigger {
            CollisionScoreTrigger::Boundary => self.boundary_events,
            CollisionScoreTrigger::Downbeat => self.downbeat_events,
            CollisionScoreTrigger::AudioOnset => self.audio_onset_events,
            CollisionScoreTrigger::Manual => self.manual_events,
        }
    }
}

/// Shared audio-onset hysteresis for live and deterministic export. The latch
/// follows the envelope while Program is frozen, but only a low-to-high edge
/// observed on a Program-advancing frame becomes an event.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct TemporalAudioOnsetTracker {
    high: bool,
}

impl TemporalAudioOnsetTracker {
    const ATTACK: f32 = 0.5;
    const RELEASE: f32 = 0.2;

    pub(crate) fn observe(&mut self, value: f32, program_advances: bool) -> u32 {
        let value = sanitized_unit(value);
        let was_high = self.high;
        self.high = if was_high {
            value >= Self::RELEASE
        } else {
            value >= Self::ATTACK
        };
        u32::from(program_advances && !was_high && self.high)
    }

    /// Anchor after a source/cue/reset edge without manufacturing an onset.
    pub(crate) fn reanchor(&mut self, value: f32) {
        self.high = sanitized_unit(value) >= Self::ATTACK;
    }
}

fn sanitized_unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// The four independent Program/Media transport states.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum TemporalFreezeState {
    #[default]
    Running,
    ProgramFrozen,
    MediaFrozen,
    ProgramAndMediaFrozen,
}

impl TemporalFreezeState {
    pub(crate) const fn program_advances(self) -> bool {
        matches!(self, Self::Running | Self::MediaFrozen)
    }

    pub(crate) const fn media_advances(self) -> bool {
        matches!(self, Self::Running | Self::ProgramFrozen)
    }
}

/// All temporal inputs are explicit. Blackout is recorded for downstream
/// policy/provenance, but intentionally does not stop hidden temporal evolution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TemporalFrameInput {
    pub delta_seconds: f32,
    pub freeze: TemporalFreezeState,
    pub blackout: bool,
    pub events: TemporalFrameEvents,
    /// Normalized deterministic analysis energy for Refresh Garden. Existing
    /// callers and patches retain exact behavior because `new` anchors this
    /// additive signal at zero.
    pub audio_energy: f32,
}

impl TemporalFrameInput {
    pub(crate) fn new(
        delta_seconds: f32,
        freeze: TemporalFreezeState,
        blackout: bool,
        events: TemporalFrameEvents,
    ) -> Self {
        Self {
            delta_seconds: sanitize_delta(delta_seconds),
            freeze,
            blackout,
            events,
            audio_energy: 0.0,
        }
    }

    pub(crate) fn with_audio_energy(mut self, audio_energy: f32) -> Self {
        self.audio_energy = sanitized_unit(audio_energy);
        self
    }

    /// Compatibility adapter for current Exact/Advanced call sites.
    pub(crate) fn legacy(delta_seconds: f32, advance_program: bool) -> Self {
        let freeze = if advance_program {
            TemporalFreezeState::Running
        } else {
            TemporalFreezeState::ProgramFrozen
        };
        Self::new(delta_seconds, freeze, false, TemporalFrameEvents::default())
    }
}

/// Typed causes let later integration distinguish hard generation cuts from
/// loop/downbeat/blackout events instead of collapsing them into one boolean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TemporalResetCause {
    PatchGeneration,
    ApplyLook,
    SourceCut,
    Seek,
    Resize,
    BroadRevert,
    ManualClear,
    LoopBoundary,
    Downbeat,
    BlackoutTransition,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct TemporalResetDomains {
    pub clean_history: bool,
    pub carrier: bool,
    pub score: bool,
    pub freeze_hold: bool,
}

impl TemporalResetDomains {
    pub(crate) const NONE: Self = Self {
        clean_history: false,
        carrier: false,
        score: false,
        freeze_hold: false,
    };

    pub(crate) const HARD: Self = Self {
        clean_history: true,
        carrier: true,
        score: true,
        freeze_hold: true,
    };

    pub(crate) const MANUAL_MEMORY: Self = Self {
        clean_history: true,
        carrier: true,
        score: true,
        // A manual memory clear while paused must not destroy the audience
        // image already being held. Resume ignores the invalid carrier.
        freeze_hold: false,
    };

    pub(crate) const fn for_cause(cause: TemporalResetCause) -> Self {
        match cause {
            TemporalResetCause::PatchGeneration
            | TemporalResetCause::ApplyLook
            | TemporalResetCause::SourceCut
            | TemporalResetCause::Seek
            | TemporalResetCause::Resize
            | TemporalResetCause::BroadRevert => Self::HARD,
            TemporalResetCause::ManualClear => Self::MANUAL_MEMORY,
            TemporalResetCause::LoopBoundary
            | TemporalResetCause::Downbeat
            | TemporalResetCause::BlackoutTransition => Self::NONE,
        }
    }

    const fn union(self, other: Self) -> Self {
        Self {
            clean_history: self.clean_history || other.clean_history,
            carrier: self.carrier || other.carrier,
            score: self.score || other.score,
            freeze_hold: self.freeze_hold || other.freeze_hold,
        }
    }
}

const fn event_reset_mode_domains(mode: TemporalEventResetMode) -> TemporalResetDomains {
    match mode {
        TemporalEventResetMode::None => TemporalResetDomains::NONE,
        TemporalEventResetMode::Score => TemporalResetDomains {
            clean_history: false,
            carrier: false,
            score: true,
            freeze_hold: false,
        },
        TemporalEventResetMode::Memory => TemporalResetDomains {
            clean_history: true,
            carrier: true,
            score: false,
            freeze_hold: false,
        },
        TemporalEventResetMode::All => TemporalResetDomains {
            clean_history: true,
            carrier: true,
            score: true,
            freeze_hold: false,
        },
    }
}

fn event_reset_domains(
    policy: TemporalResetPolicy,
    events: TemporalFrameEvents,
) -> TemporalResetDomains {
    let loop_reset = if events.boundary_events == 0 {
        TemporalResetDomains::NONE
    } else {
        event_reset_mode_domains(policy.loop_boundary)
    };
    let downbeat_reset = if events.downbeat_events == 0 {
        TemporalResetDomains::NONE
    } else {
        event_reset_mode_domains(policy.downbeat)
    };
    loop_reset.union(downbeat_reset)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CollisionScoreState {
    pub state_index: u8,
    pub event_ordinal: u64,
}

/// The exact four-vec4 legacy shader contract. Originals use a second binding,
/// so these 64 bytes retain the old field order and IEEE-754 byte values.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct TemporalGpuUniforms {
    pub feedback: f32,
    pub fb_zoom: f32,
    pub fb_rotate: f32,
    pub slitscan: f32,
    pub history_len: f32,
    pub write_index: f32,
    pub valid_history: f32,
    pub feedback_valid: f32,
    pub slit_direction: [f32; 2],
    pub key_reference_layer: f32,
    pub key_valid: f32,
    pub key_mode: f32,
    pub key_threshold: f32,
    pub key_softness: f32,
    pub _pad: f32,
}

const _: () = assert!(std::mem::size_of::<TemporalGpuUniforms>() == 64);

/// Additive M3 shader contract. The legacy uniform above is intentionally
/// frozen; originals live in a second fixed binding so a zero/default patch
/// can keep executing the old shader and pipeline literally unchanged.
///
/// Every member is a complete 16-byte lane, mirroring eight WGSL vec4 values
/// without relying on host-specific struct padding.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct TemporalOriginalsGpuUniforms {
    /// amount, depth, phase, scale
    pub loom_values: [f32; 4],
    /// angle in radians, output aspect ratio, reserved, reserved
    pub loom_geometry: [f32; 4],
    /// topology, interpolation, folds, quantization
    pub loom_modes: [u32; 4],
    /// amount, collision, reserved, reserved
    pub atlas_values: [f32; 4],
    /// seed, territories, score state count, flags (bit 0 = score enabled)
    pub atlas_modes: [u32; 4],
    /// score seed, state index, event ordinal low, event ordinal high
    pub score_runtime: [u32; 4],
    /// amount, threshold, softness, decay
    pub garden_values: [f32; 4],
    /// gate, max hold ticks, observation ticks, packed runtime signals
    pub garden_modes: [u32; 4],
}

const _: () = assert!(std::mem::size_of::<TemporalOriginalsGpuUniforms>() == 128);

impl TemporalOriginalsGpuUniforms {
    fn for_frame(
        params: TemporalOriginalsParams,
        score: CollisionScoreState,
        dimensions: [u32; 2],
        observation_ticks: u32,
        audio_energy: f32,
        audio_onset: bool,
        force_garden_refresh: bool,
    ) -> Self {
        let params = params.sanitized();
        let aspect = dimensions[0].max(1) as f32 / dimensions[1].max(1) as f32;
        let ordinal = score.event_ordinal;
        Self {
            loom_values: [
                params.loom.amount,
                params.loom.depth,
                params.loom.phase,
                params.loom.scale,
            ],
            loom_geometry: [params.loom.angle.to_radians(), aspect, 0.0, 0.0],
            loom_modes: [
                params.loom.topology.gpu_code(),
                params.loom.interpolation.gpu_code(),
                u32::from(params.loom.folds),
                u32::from(params.loom.quantization),
            ],
            atlas_values: [params.atlas.amount, params.atlas.collision, 0.0, 0.0],
            atlas_modes: [
                params.atlas.seed,
                u32::from(params.atlas.territories),
                u32::from(params.score.state_count),
                u32::from(params.score.enabled),
            ],
            score_runtime: [
                params.score.seed,
                u32::from(score.state_index),
                ordinal as u32,
                (ordinal >> 32) as u32,
            ],
            garden_values: [
                params.garden.amount,
                params.garden.threshold,
                params.garden.softness,
                params.garden.decay,
            ],
            garden_modes: [
                params.garden.gate.gpu_code(),
                params.garden.max_hold_ticks,
                observation_ticks,
                pack_garden_runtime(audio_energy, audio_onset, force_garden_refresh),
            ],
        }
    }
}

const GARDEN_AUDIO_QUANTIZATION: f32 = 65_535.0;
const GARDEN_AUDIO_ONSET_BIT: u32 = 1 << 16;
const GARDEN_FORCE_REFRESH_BIT: u32 = 1 << 17;

fn pack_garden_runtime(audio_energy: f32, audio_onset: bool, force_refresh: bool) -> u32 {
    let energy = (sanitized_unit(audio_energy) * GARDEN_AUDIO_QUANTIZATION).round() as u32;
    energy
        | (if audio_onset {
            GARDEN_AUDIO_ONSET_BIT
        } else {
            0
        })
        | (if force_refresh {
            GARDEN_FORCE_REFRESH_BIT
        } else {
            0
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TemporalFrameAction {
    PrimeFrozenOutput,
    HoldFrozenOutput,
    Advance { record_history: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TemporalReadSnapshot {
    pub virtual_write: usize,
    pub virtual_valid: u32,
    pub key_reference: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TemporalFramePlan {
    pub action: TemporalFrameAction,
    pub read: TemporalReadSnapshot,
    pub uniforms: TemporalGpuUniforms,
    pub originals_uniforms: TemporalOriginalsGpuUniforms,
    pub legacy_shader_active: bool,
    pub originals_shader_active: bool,
    pub history_write_target: Option<usize>,
    pub observation_ticks: u32,
    pub score_events_consumed: u32,
    pub garden_force_refresh: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct TemporalStateSnapshot {
    history_write: usize,
    history_valid: u32,
    history_accumulator: f64,
    feedback_valid: bool,
    freeze_hold_valid: bool,
    initialized: bool,
    total_history_frames: usize,
    total_reference_ticks: u64,
    score: CollisionScoreState,
    garden_hold_ticks: u32,
    last_reset: Option<TemporalResetCause>,
}

/// Read-only engine-domain telemetry. This deliberately carries no web/UI
/// vocabulary, so Exact, Advanced, tests, and future hosts share one truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TemporalStateMetrics {
    pub history_valid: u32,
    pub history_capacity: u32,
    pub carrier_valid: bool,
    pub freeze_hold_valid: bool,
    pub total_reference_ticks: u64,
    pub score_state: u8,
    pub score_event_ordinal: u64,
    pub frame_staged: bool,
    pub last_reset: Option<TemporalResetCause>,
}

/// Shared transactional temporal CPU state.
#[derive(Debug, Clone)]
pub(crate) struct TemporalState {
    pub(crate) history_write: usize,
    pub(crate) history_valid: u32,
    pub(crate) history_accumulator: f64,
    /// Validity of the post-temporal feedback/Garden carrier.
    pub(crate) feedback_valid: bool,
    /// Validity of the exact audience image used by Program Freeze.
    pub(crate) freeze_hold_valid: bool,
    pub(crate) initialized: bool,
    pub(crate) total_history_frames: usize,
    pub(crate) total_reference_ticks: u64,
    pub(crate) score: CollisionScoreState,
    /// Reference ticks since the last periodic all-pixel Garden admission.
    /// Actual per-pixel admissions need no readback; this deterministic phase
    /// guarantees that no pixel can remain gated out beyond max-hold.
    pub(crate) garden_hold_ticks: u32,
    pub(crate) last_reset: Option<TemporalResetCause>,
    staged_snapshot: Option<TemporalStateSnapshot>,
}

impl Default for TemporalState {
    fn default() -> Self {
        Self {
            history_write: 0,
            history_valid: 0,
            history_accumulator: 0.0,
            feedback_valid: false,
            freeze_hold_valid: false,
            initialized: false,
            total_history_frames: 0,
            total_reference_ticks: 0,
            score: CollisionScoreState::default(),
            garden_hold_ticks: 0,
            last_reset: None,
            staged_snapshot: None,
        }
    }
}

impl TemporalState {
    fn snapshot(&self) -> TemporalStateSnapshot {
        TemporalStateSnapshot {
            history_write: self.history_write,
            history_valid: self.history_valid,
            history_accumulator: self.history_accumulator,
            feedback_valid: self.feedback_valid,
            freeze_hold_valid: self.freeze_hold_valid,
            initialized: self.initialized,
            total_history_frames: self.total_history_frames,
            total_reference_ticks: self.total_reference_ticks,
            score: self.score,
            garden_hold_ticks: self.garden_hold_ticks,
            last_reset: self.last_reset,
        }
    }

    fn restore(&mut self, snapshot: TemporalStateSnapshot) {
        self.history_write = snapshot.history_write;
        self.history_valid = snapshot.history_valid;
        self.history_accumulator = snapshot.history_accumulator;
        self.feedback_valid = snapshot.feedback_valid;
        self.freeze_hold_valid = snapshot.freeze_hold_valid;
        self.initialized = snapshot.initialized;
        self.total_history_frames = snapshot.total_history_frames;
        self.total_reference_ticks = snapshot.total_reference_ticks;
        self.score = snapshot.score;
        self.garden_hold_ticks = snapshot.garden_hold_ticks;
        self.last_reset = snapshot.last_reset;
    }

    /// Stage exactly one render. Callers must commit or discard before staging
    /// another; this prevents abandoned command encoders from advancing CPU
    /// cadence, ring validity, or Score state.
    pub(crate) fn stage_frame(
        &mut self,
        params: &TemporalParams,
        input: TemporalFrameInput,
        dimensions: [u32; 2],
    ) -> TemporalFramePlan {
        assert!(
            self.staged_snapshot.is_none(),
            "temporal frame already staged; commit or discard it first"
        );
        let original = self.snapshot();
        self.staged_snapshot = Some(original.clone());
        let frame_params = params.for_frame_delta(input.delta_seconds);
        if input.freeze.program_advances() {
            let reset_domains = event_reset_domains(frame_params.originals.reset, input.events);
            self.apply_reset_values(reset_domains);
            if reset_domains != TemporalResetDomains::NONE {
                self.last_reset = Some(
                    if input.events.downbeat_events > 0
                        && frame_params.originals.reset.downbeat != TemporalEventResetMode::None
                    {
                        TemporalResetCause::Downbeat
                    } else {
                        TemporalResetCause::LoopBoundary
                    },
                );
            }
        }
        // Reads and uniforms observe event resets from this same accepted-frame
        // transaction. Discard still restores `original`, before those resets.
        let before = self.snapshot();
        let read = temporal_read_snapshot(
            before.history_write,
            before.history_valid,
            frame_params.key_history,
        );
        let uniforms = TemporalGpuUniforms {
            feedback: frame_params.feedback,
            fb_zoom: frame_params.fb_zoom,
            fb_rotate: frame_params.fb_rotate,
            slitscan: frame_params.slitscan,
            history_len: TEMPORAL_HISTORY_LEN as f32,
            write_index: read.virtual_write as f32,
            valid_history: read.virtual_valid as f32,
            feedback_valid: f32::from(before.feedback_valid),
            slit_direction: normalized_slit_direction(
                frame_params.slit_angle,
                dimensions[0],
                dimensions[1],
            ),
            key_reference_layer: read.key_reference.unwrap_or(0) as f32,
            key_valid: f32::from(read.key_reference.is_some()),
            key_mode: frame_params.key_mode,
            key_threshold: frame_params.key_threshold,
            key_softness: frame_params.key_softness,
            _pad: 0.0,
        };

        if !input.freeze.program_advances() {
            let action = if before.freeze_hold_valid {
                TemporalFrameAction::HoldFrozenOutput
            } else {
                self.feedback_valid = true;
                self.freeze_hold_valid = true;
                TemporalFrameAction::PrimeFrozenOutput
            };
            return TemporalFramePlan {
                action,
                read,
                uniforms,
                originals_uniforms: TemporalOriginalsGpuUniforms::for_frame(
                    frame_params.originals,
                    before.score,
                    dimensions,
                    0,
                    input.audio_energy,
                    false,
                    false,
                ),
                legacy_shader_active: frame_params.is_active(),
                originals_shader_active: !frame_params.originals.is_zero(),
                history_write_target: None,
                observation_ticks: 0,
                score_events_consumed: 0,
                garden_force_refresh: false,
            };
        }

        let observation_ticks = self.history_ticks_for_delta(input.delta_seconds);
        let record_history = observation_ticks > 0;
        let history_write_target = record_history.then(|| {
            self.record_history_frame();
            self.history_write
        });

        let garden_force_refresh =
            self.advance_garden_hold(frame_params.originals.garden, observation_ticks);

        let score_events_consumed = if frame_params.originals.score.enabled {
            let score = frame_params.originals.score;
            let count = input.events.count_for(score.trigger);
            self.score.event_ordinal = self.score.event_ordinal.saturating_add(u64::from(count));
            self.score.state_index = ((u64::from(self.score.state_index) + u64::from(count))
                % u64::from(score.state_count)) as u8;
            count
        } else {
            0
        };

        // Current legacy behavior always refreshes the shared feedback/hold
        // texture after an advancing frame, even when the shader is disabled.
        self.feedback_valid = true;
        self.freeze_hold_valid = true;

        TemporalFramePlan {
            action: TemporalFrameAction::Advance { record_history },
            read,
            uniforms,
            originals_uniforms: TemporalOriginalsGpuUniforms::for_frame(
                frame_params.originals,
                self.score,
                dimensions,
                observation_ticks,
                input.audio_energy,
                input.events.audio_onset_events > 0,
                garden_force_refresh,
            ),
            legacy_shader_active: frame_params.is_active(),
            originals_shader_active: !frame_params.originals.is_zero(),
            history_write_target,
            observation_ticks,
            score_events_consumed,
            garden_force_refresh,
        }
    }

    pub(crate) fn commit_staged(&mut self) {
        self.staged_snapshot = None;
    }

    pub(crate) fn discard_staged(&mut self) {
        if let Some(snapshot) = self.staged_snapshot.take() {
            self.restore(snapshot);
        }
    }

    pub(crate) fn has_staged_frame(&self) -> bool {
        self.staged_snapshot.is_some()
    }

    /// Compatibility hard reset used by current generation-cut call sites.
    pub(crate) fn reset(&mut self) {
        self.reset_for(TemporalResetCause::PatchGeneration);
    }

    pub(crate) fn reset_for(&mut self, cause: TemporalResetCause) {
        self.apply_reset_domains(TemporalResetDomains::for_cause(cause));
        self.last_reset = Some(cause);
    }

    pub(crate) fn apply_reset_domains(&mut self, domains: TemporalResetDomains) {
        if domains == TemporalResetDomains::NONE {
            return;
        }
        self.staged_snapshot = None;
        self.apply_reset_values(domains);
    }

    fn apply_reset_values(&mut self, domains: TemporalResetDomains) {
        if domains.clean_history {
            self.history_write = 0;
            self.history_valid = 0;
            self.history_accumulator = 0.0;
            self.initialized = false;
            self.total_history_frames = 0;
            self.total_reference_ticks = 0;
        }
        if domains.carrier {
            self.feedback_valid = false;
            self.garden_hold_ticks = 0;
        }
        if domains.freeze_hold {
            self.freeze_hold_valid = false;
        }
        if domains.score {
            self.score = CollisionScoreState::default();
        }
    }

    fn advance_garden_hold(&mut self, garden: RefreshGardenParams, ticks: u32) -> bool {
        if garden.amount <= 0.0 || garden.max_hold_ticks == 0 {
            self.garden_hold_ticks = 0;
            return false;
        }
        let accumulated = self.garden_hold_ticks.saturating_add(ticks);
        if accumulated < garden.max_hold_ticks {
            self.garden_hold_ticks = accumulated;
            false
        } else {
            self.garden_hold_ticks = accumulated % garden.max_hold_ticks;
            true
        }
    }

    pub(crate) fn metrics(&self) -> TemporalStateMetrics {
        TemporalStateMetrics {
            history_valid: self.history_valid,
            history_capacity: TEMPORAL_HISTORY_LEN,
            carrier_valid: self.feedback_valid,
            freeze_hold_valid: self.freeze_hold_valid,
            total_reference_ticks: self.total_reference_ticks,
            score_state: self.score.state_index,
            score_event_ordinal: self.score.event_ordinal,
            frame_staged: self.staged_snapshot.is_some(),
            last_reset: self.last_reset,
        }
    }

    fn history_ticks_for_delta(&mut self, delta_seconds: f32) -> u32 {
        if !self.initialized {
            self.initialized = true;
            self.total_reference_ticks = self.total_reference_ticks.saturating_add(1);
            return 1;
        }

        let reference_delta = 1.0 / f64::from(TEMPORAL_REFERENCE_FPS);
        self.history_accumulator += f64::from(sanitize_delta(delta_seconds));
        let elapsed = (self.history_accumulator / reference_delta).floor() as u64;
        if elapsed == 0 {
            return 0;
        }
        self.history_accumulator -= elapsed as f64 * reference_delta;
        let bounded = elapsed.min(u64::from(TEMPORAL_HISTORY_LEN)) as u32;
        self.total_reference_ticks = self
            .total_reference_ticks
            .saturating_add(u64::from(bounded));
        bounded
    }

    fn record_history_frame(&mut self) {
        if self.history_valid > 0 {
            self.history_write = (self.history_write + 1) % TEMPORAL_HISTORY_LEN as usize;
        }
        self.history_valid = (self.history_valid + 1).min(TEMPORAL_HISTORY_LEN);
        self.total_history_frames = self.total_history_frames.saturating_add(1);
    }
}

pub(crate) fn temporal_key_reference_layer(
    history_write: usize,
    history_valid: u32,
    requested_depth: f32,
) -> Option<usize> {
    let depth = if requested_depth.is_finite() {
        requested_depth
            .round()
            .clamp(1.0, (TEMPORAL_HISTORY_LEN - 1) as f32) as usize
    } else {
        1
    };
    let offset = depth.saturating_sub(1);
    if history_valid == 0 || offset >= history_valid as usize {
        return None;
    }
    Some((history_write + TEMPORAL_HISTORY_LEN as usize - offset) % TEMPORAL_HISTORY_LEN as usize)
}

pub(crate) fn temporal_read_snapshot(
    previous_write: usize,
    previous_valid: u32,
    key_history: f32,
) -> TemporalReadSnapshot {
    TemporalReadSnapshot {
        virtual_write: if previous_valid == 0 {
            0
        } else {
            (previous_write + 1) % TEMPORAL_HISTORY_LEN as usize
        },
        virtual_valid: if previous_valid == 0 {
            0
        } else {
            (previous_valid + 1).min(TEMPORAL_HISTORY_LEN)
        },
        key_reference: temporal_key_reference_layer(previous_write, previous_valid, key_history),
    }
}

fn sanitize_delta(delta_seconds: f32) -> f32 {
    if delta_seconds.is_finite() {
        delta_seconds.max(0.0)
    } else {
        1.0 / TEMPORAL_REFERENCE_FPS
    }
}

fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

/// CPU reference for one Loom coordinate. This is intentionally independent
/// of the WGSL implementation so unit tests can pin the artistic topology and
/// startup bounds without requiring an adapter.
pub(crate) fn temporal_loom_age(
    uv: [f32; 2],
    params: TemporalLoomParams,
    aspect_ratio: f32,
) -> f32 {
    let params = params.sanitized();
    let aspect = finite_or(aspect_ratio, 1.0).clamp(1.0 / 100.0, 100.0);
    let mut p = [
        (finite_or(uv[0], 0.5) - 0.5) * aspect * params.scale,
        (finite_or(uv[1], 0.5) - 0.5) * params.scale,
    ];
    let angle = params.angle.to_radians();
    let (sine, cosine) = angle.sin_cos();
    p = [p[0] * cosine + p[1] * sine, -p[0] * sine + p[1] * cosine];
    let radius = p[0].hypot(p[1]);
    let turn = p[1].atan2(p[0]) / std::f32::consts::TAU;
    let folds = f32::from(params.folds);
    let raw = match params.topology {
        TemporalTopology::Linear => p[1] + 0.5 + params.phase,
        TemporalTopology::Radial => radius * 2.0 + params.phase,
        TemporalTopology::Spiral => radius * 2.0 + turn + params.phase,
        TemporalTopology::Contour => (p[0].abs() + p[1].abs()) * 2.0 + params.phase,
        TemporalTopology::Folded => triangle_wave((p[1] + 0.5) * folds + params.phase),
        TemporalTopology::Kaleidoscopic => {
            let sector = triangle_wave((turn + 0.5) * folds);
            radius * 2.0 + sector + params.phase
        }
    };
    quantize_age(unit_fraction(raw), params.quantization)
}

fn triangle_wave(value: f32) -> f32 {
    1.0 - (unit_fraction(value) * 2.0 - 1.0).abs()
}

fn unit_fraction(value: f32) -> f32 {
    let value = finite_or(value, 0.0);
    value - value.floor()
}

fn quantize_age(age: f32, quantization: u8) -> f32 {
    match quantization {
        0 => age.clamp(0.0, 1.0),
        1 => 0.0,
        levels => {
            let intervals = f32::from(levels - 1);
            (age.clamp(0.0, 1.0) * intervals).round() / intervals
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TemporalHistorySamplePlan {
    /// Zero denotes the virtual current image; positive ages address the ring.
    pub lower_age: u32,
    pub upper_age: u32,
    pub mix: f32,
}

/// Resolve an age without ever exposing an unwritten array layer. The valid
/// count includes the virtual current image, matching `TemporalReadSnapshot`.
pub(crate) fn temporal_history_sample_plan(
    requested_age: f32,
    virtual_valid: u32,
    interpolation: TemporalInterpolation,
) -> TemporalHistorySamplePlan {
    let max_age = virtual_valid
        .saturating_sub(1)
        .min(TEMPORAL_HISTORY_LEN - 1);
    let age = finite_or(requested_age, 0.0).clamp(0.0, max_age as f32);
    let lower_age = age.floor() as u32;
    match interpolation {
        TemporalInterpolation::Floor => TemporalHistorySamplePlan {
            lower_age,
            upper_age: lower_age,
            mix: 0.0,
        },
        TemporalInterpolation::Linear => TemporalHistorySamplePlan {
            lower_age,
            upper_age: age.ceil() as u32,
            mix: age.fract(),
        },
    }
}

/// Deterministic, analytical 3x3 Worley/Voronoi reference. Seed zero is an
/// ordinary seed and never selects a fallback/random source.
pub(crate) fn collision_atlas_age(
    uv: [f32; 2],
    params: CollisionAtlasParams,
    aspect_ratio: f32,
    score_params: CollisionScoreParams,
    score: CollisionScoreState,
) -> f32 {
    let params = params.sanitized();
    let score_params = score_params.sanitized();
    let aspect = finite_or(aspect_ratio, 1.0).clamp(1.0 / 100.0, 100.0);
    let grid_scale = f32::from(params.territories).sqrt().max(1.0);
    let p = [
        finite_or(uv[0], 0.5) * aspect * grid_scale,
        finite_or(uv[1], 0.5) * grid_scale,
    ];
    let base = [p[0].floor() as i32, p[1].floor() as i32];
    let seed = effective_atlas_seed(params.seed, score_params, score);
    let mut nearest = (f32::INFINITY, 0.0_f32);
    let mut second = (f32::INFINITY, 0.0_f32);
    for y_offset in -1_i32..=1 {
        for x_offset in -1_i32..=1 {
            let cell = [base[0] + x_offset, base[1] + y_offset];
            let hash = atlas_cell_hash(cell[0], cell[1], seed);
            let feature = [
                cell[0] as f32 + hash_unit(hash),
                cell[1] as f32 + hash_unit(mix_u32(hash ^ 0x68bc_21ebu32)),
            ];
            let delta = [feature[0] - p[0], feature[1] - p[1]];
            let distance = delta[0].mul_add(delta[0], delta[1] * delta[1]);
            let age = hash_unit(mix_u32(hash ^ 0x02e5_be93u32));
            if distance < nearest.0 {
                second = nearest;
                nearest = (distance, age);
            } else if distance < second.0 {
                second = (distance, age);
            }
        }
    }
    let ridge_gap = second.0.sqrt() - nearest.0.sqrt();
    let ridge = 1.0 - smoothstep(0.0, 0.35, ridge_gap);
    nearest.1 + (second.1 - nearest.1) * ridge * params.collision
}

/// Shader-independent Garden gate inputs used by CPU laws and deterministic
/// host fixtures. Pixel-derived channels are already normalized by callers.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct RefreshGardenSignals {
    pub temporal_delta: f32,
    pub luma: f32,
    pub chroma: f32,
    pub cellular_ridge: f32,
    pub audio_energy: f32,
    pub audio_onset: bool,
    pub matte: f32,
    pub force_refresh: bool,
}

impl RefreshGardenSignals {
    fn selected(self, gate: RefreshGardenGate) -> f32 {
        match gate {
            RefreshGardenGate::TemporalDelta => self.temporal_delta,
            RefreshGardenGate::Luma => self.luma,
            RefreshGardenGate::Chroma => self.chroma,
            RefreshGardenGate::CellularRidge => self.cellular_ridge,
            RefreshGardenGate::AudioEnergy => self.audio_energy,
            RefreshGardenGate::AudioOnset => f32::from(self.audio_onset),
            RefreshGardenGate::Matte => self.matte,
        }
    }
}

/// Per-reference-tick admission mask. A forced max-hold refresh opens every
/// gate but retains the authored Garden amount as its bounded blend strength.
pub(crate) fn refresh_garden_admission(
    params: RefreshGardenParams,
    signals: RefreshGardenSignals,
) -> f32 {
    let params = params.sanitized();
    let gate = if signals.force_refresh {
        1.0
    } else {
        let signal = sanitized_unit(signals.selected(params.gate));
        if params.softness <= f32::EPSILON {
            f32::from(signal >= params.threshold)
        } else {
            smoothstep(
                (params.threshold - params.softness).max(0.0),
                (params.threshold + params.softness).min(1.0),
                signal,
            )
        }
    };
    params.amount * gate
}

/// Closed-form deterministic recurrence for a batch of reference ticks:
/// `memory = memory * decay * (1 - admission) + current * admission`.
/// Grouping identical observations into 24/30/60 Hz frames therefore produces
/// the same result without per-tick full-frame passes.
pub(crate) fn refresh_garden_pixel(
    current: [f32; 4],
    previous: [f32; 4],
    params: RefreshGardenParams,
    signals: RefreshGardenSignals,
    observation_ticks: u32,
    carrier_valid: bool,
) -> [f32; 4] {
    if !carrier_valid {
        return current;
    }
    if observation_ticks == 0 {
        return previous;
    }
    let params = params.sanitized();
    let admission = refresh_garden_admission(params, signals);
    let coefficient = (params.decay * (1.0 - admission)).clamp(0.0, 1.0);
    let retained = coefficient.powf(observation_ticks as f32);
    let injected = if admission <= 0.0 {
        0.0
    } else if (1.0 - coefficient).abs() <= f32::EPSILON {
        admission * observation_ticks as f32
    } else {
        admission * (1.0 - retained) / (1.0 - coefficient)
    };
    std::array::from_fn(|channel| previous[channel].mul_add(retained, current[channel] * injected))
}

fn effective_atlas_seed(
    atlas_seed: u32,
    score_params: CollisionScoreParams,
    score: CollisionScoreState,
) -> u32 {
    if !score_params.enabled {
        return atlas_seed;
    }
    let ordinal = score.event_ordinal;
    let conductor = score_params.seed
        ^ u32::from(score.state_index).wrapping_mul(0x9e37_79b9)
        ^ ordinal as u32
        ^ (ordinal >> 32) as u32;
    atlas_seed ^ mix_u32(conductor)
}

fn atlas_cell_hash(x: i32, y: i32, seed: u32) -> u32 {
    let combined =
        (x as u32).wrapping_mul(0x8da6_b343) ^ (y as u32).wrapping_mul(0xd816_3841) ^ seed;
    mix_u32(combined)
}

fn mix_u32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

fn hash_unit(value: u32) -> f32 {
    (value >> 8) as f32 * (1.0 / 16_777_216.0)
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn running(delta_seconds: f32) -> TemporalFrameInput {
        TemporalFrameInput::new(
            delta_seconds,
            TemporalFreezeState::Running,
            false,
            TemporalFrameEvents::default(),
        )
    }

    fn stage_and_commit(
        state: &mut TemporalState,
        params: &TemporalParams,
        input: TemporalFrameInput,
    ) -> TemporalFramePlan {
        let plan = state.stage_frame(params, input, [1_920, 1_080]);
        state.commit_staged();
        plan
    }

    #[test]
    fn legacy_uniform_bytes_are_the_frozen_four_vec4_contract() {
        let params = TemporalParams {
            feedback: 0.5,
            fb_zoom: 1.0,
            fb_rotate: 0.0,
            slitscan: 0.25,
            slit_angle: 0.0,
            slit_axis: 0.0,
            key_mode: 1.0,
            key_threshold: 0.5,
            key_softness: 0.25,
            key_history: 1.0,
            originals: TemporalOriginalsParams::default(),
        };
        let plan =
            TemporalState::default().stage_frame(&params, running(1.0 / 30.0), [1_920, 1_080]);
        let expected: [u8; 64] = [
            0, 0, 0, 63, 0, 0, 128, 63, 0, 0, 0, 0, 0, 0, 128, 62, 0, 0, 192, 65, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0,
            0, 0, 63, 0, 0, 128, 62, 0, 0, 0, 0,
        ];
        assert_eq!(bytemuck::bytes_of(&plan.uniforms), &expected);
        assert_eq!(std::mem::size_of::<TemporalGpuUniforms>(), 64);
    }

    #[test]
    fn exact_and_advanced_consumers_receive_identical_shared_plans() {
        let params = TemporalParams {
            feedback: 0.81,
            slitscan: 0.7,
            key_mode: 1.0,
            key_history: 2.0,
            ..TemporalParams::default()
        };
        let mut exact = TemporalState::default();
        let mut advanced = TemporalState::default();
        for (delta, freeze) in [
            (1.0 / 60.0, TemporalFreezeState::Running),
            (1.0 / 60.0, TemporalFreezeState::Running),
            (0.0, TemporalFreezeState::ProgramFrozen),
            (1.0 / 30.0, TemporalFreezeState::MediaFrozen),
        ] {
            let input =
                TemporalFrameInput::new(delta, freeze, false, TemporalFrameEvents::default());
            let exact_plan = exact.stage_frame(&params, input, [1_280, 720]);
            let advanced_plan = advanced.stage_frame(&params, input, [1_280, 720]);
            assert_eq!(exact_plan, advanced_plan);
            exact.commit_staged();
            advanced.commit_staged();
        }
        assert_eq!(exact.snapshot(), advanced.snapshot());
    }

    #[test]
    fn cadence_is_identical_at_24_30_and_60_display_fps() {
        for (fps, expected_records) in [(24, 24), (30, 30), (60, 30)] {
            let mut state = TemporalState::default();
            let mut records = 0;
            for _ in 0..fps {
                let plan = stage_and_commit(
                    &mut state,
                    &TemporalParams::default(),
                    running(1.0 / fps as f32),
                );
                records += usize::from(matches!(
                    plan.action,
                    TemporalFrameAction::Advance {
                        record_history: true
                    }
                ));
            }
            assert_eq!(records, expected_records, "{fps} fps");
        }
    }

    #[test]
    fn staged_state_commits_or_rolls_back_as_one_transaction() {
        let mut state = TemporalState::default();
        let first = state.stage_frame(&TemporalParams::default(), running(1.0 / 30.0), [640, 360]);
        assert_eq!(first.history_write_target, Some(0));
        assert_eq!(state.history_valid, 1);
        assert!(state.has_staged_frame());
        state.discard_staged();
        assert_eq!(state.history_valid, 0);
        assert!(!state.feedback_valid);
        assert!(!state.has_staged_frame());

        let committed =
            state.stage_frame(&TemporalParams::default(), running(1.0 / 30.0), [640, 360]);
        assert_eq!(committed.history_write_target, Some(0));
        state.commit_staged();
        assert_eq!(state.history_valid, 1);
        assert!(state.feedback_valid);
    }

    #[test]
    fn freeze_and_blackout_truth_table_is_explicit() {
        let params = TemporalParams::default();
        for (freeze, program_advances, media_advances) in [
            (TemporalFreezeState::Running, true, true),
            (TemporalFreezeState::ProgramFrozen, false, true),
            (TemporalFreezeState::MediaFrozen, true, false),
            (TemporalFreezeState::ProgramAndMediaFrozen, false, false),
        ] {
            assert_eq!(freeze.program_advances(), program_advances);
            assert_eq!(freeze.media_advances(), media_advances);
        }

        let mut clear = TemporalState::default();
        let mut blacked_out = TemporalState::default();
        let clear_plan = clear.stage_frame(
            &params,
            TemporalFrameInput::new(
                1.0 / 30.0,
                TemporalFreezeState::Running,
                false,
                TemporalFrameEvents::default(),
            ),
            [640, 360],
        );
        let blackout_plan = blacked_out.stage_frame(
            &params,
            TemporalFrameInput::new(
                1.0 / 30.0,
                TemporalFreezeState::Running,
                true,
                TemporalFrameEvents::default(),
            ),
            [640, 360],
        );
        assert_eq!(clear_plan, blackout_plan, "blackout is output-only");
        assert_eq!(clear.snapshot(), blacked_out.snapshot());

        clear.commit_staged();
        let held = clear.stage_frame(
            &params,
            TemporalFrameInput::new(
                10.0,
                TemporalFreezeState::ProgramFrozen,
                false,
                TemporalFrameEvents {
                    boundary_events: 9,
                    ..TemporalFrameEvents::default()
                },
            ),
            [640, 360],
        );
        assert_eq!(held.action, TemporalFrameAction::HoldFrozenOutput);
        assert_eq!(held.observation_ticks, 0);
        assert_eq!(held.score_events_consumed, 0);
    }

    #[test]
    fn reset_event_truth_table_preserves_blackout_and_manual_freeze_hold() {
        for cause in [
            TemporalResetCause::PatchGeneration,
            TemporalResetCause::ApplyLook,
            TemporalResetCause::SourceCut,
            TemporalResetCause::Seek,
            TemporalResetCause::Resize,
            TemporalResetCause::BroadRevert,
        ] {
            assert_eq!(
                TemporalResetDomains::for_cause(cause),
                TemporalResetDomains::HARD
            );
        }
        assert_eq!(
            TemporalResetDomains::for_cause(TemporalResetCause::ManualClear),
            TemporalResetDomains::MANUAL_MEMORY
        );
        for cause in [
            TemporalResetCause::LoopBoundary,
            TemporalResetCause::Downbeat,
            TemporalResetCause::BlackoutTransition,
        ] {
            assert_eq!(
                TemporalResetDomains::for_cause(cause),
                TemporalResetDomains::NONE
            );
        }

        let mut state = TemporalState::default();
        stage_and_commit(&mut state, &TemporalParams::default(), running(1.0 / 30.0));
        state.reset_for(TemporalResetCause::ManualClear);
        assert_eq!(state.history_valid, 0);
        assert!(!state.feedback_valid);
        assert!(state.freeze_hold_valid);
        let held = state.stage_frame(
            &TemporalParams::default(),
            TemporalFrameInput::new(
                0.0,
                TemporalFreezeState::ProgramFrozen,
                false,
                TemporalFrameEvents::default(),
            ),
            [640, 360],
        );
        assert_eq!(held.action, TemporalFrameAction::HoldFrozenOutput);

        state.discard_staged();
        let committed_history = state.history_valid;
        state.stage_frame(&TemporalParams::default(), running(1.0 / 30.0), [640, 360]);
        state.reset_for(TemporalResetCause::BlackoutTransition);
        assert!(
            state.has_staged_frame(),
            "blackout cannot commit a transaction"
        );
        state.discard_staged();
        assert_eq!(state.history_valid, committed_history);
    }

    #[test]
    fn score_consumes_every_boundary_event_without_a_wall_clock() {
        let params = TemporalParams {
            originals: TemporalOriginalsParams {
                score: CollisionScoreParams {
                    enabled: true,
                    seed: 0,
                    state_count: 4,
                    trigger: CollisionScoreTrigger::Boundary,
                    loop_driver: CollisionScoreLoopDriver::None,
                },
                ..TemporalOriginalsParams::default()
            },
            ..TemporalParams::default()
        };
        let input = TemporalFrameInput::new(
            0.0,
            TemporalFreezeState::Running,
            false,
            TemporalFrameEvents {
                boundary_events: 7,
                ..TemporalFrameEvents::default()
            },
        );
        let mut live = TemporalState::default();
        let mut export = TemporalState::default();
        let live_plan = live.stage_frame(&params, input, [640, 360]);
        let export_plan = export.stage_frame(&params, input, [640, 360]);
        assert_eq!(live_plan.score_events_consumed, 7);
        assert_eq!(live.score.state_index, 3);
        assert_eq!(live.score.event_ordinal, 7);
        assert_eq!(live_plan, export_plan);
        assert_eq!(live.snapshot(), export.snapshot());
    }

    #[test]
    fn zero_originals_are_sanitized_no_ops() {
        let clean = TemporalOriginalsParams::default().sanitized();
        assert!(clean.is_zero());
        assert_eq!(clean.loom.topology, TemporalTopology::Linear);
        assert_eq!(clean.loom.interpolation, TemporalInterpolation::Floor);
        assert_eq!(clean.atlas.seed, 0);
        assert!(!clean.score.enabled);
    }

    #[test]
    fn originals_uniform_is_eight_explicit_vec4_lanes_and_sanitizes_controls() {
        let params = TemporalOriginalsParams {
            loom: TemporalLoomParams {
                amount: f32::NAN,
                depth: f32::INFINITY,
                phase: f32::NEG_INFINITY,
                scale: f32::NAN,
                angle: f32::INFINITY,
                folds: 0,
                quantization: u8::MAX,
                ..TemporalLoomParams::default()
            },
            atlas: CollisionAtlasParams {
                amount: f32::INFINITY,
                seed: 0,
                territories: 0,
                collision: f32::NAN,
            },
            ..TemporalOriginalsParams::default()
        };
        let uniforms = TemporalOriginalsGpuUniforms::for_frame(
            params,
            CollisionScoreState {
                state_index: 3,
                event_ordinal: 0x1122_3344_5566_7788,
            },
            [1_920, 1_080],
            7,
            0.0,
            false,
            false,
        );
        assert_eq!(std::mem::size_of::<TemporalOriginalsGpuUniforms>(), 128);
        assert_eq!(
            std::mem::offset_of!(TemporalOriginalsGpuUniforms, loom_values),
            0
        );
        assert_eq!(
            std::mem::offset_of!(TemporalOriginalsGpuUniforms, loom_geometry),
            16
        );
        assert_eq!(
            std::mem::offset_of!(TemporalOriginalsGpuUniforms, loom_modes),
            32
        );
        assert_eq!(
            std::mem::offset_of!(TemporalOriginalsGpuUniforms, atlas_values),
            48
        );
        assert_eq!(
            std::mem::offset_of!(TemporalOriginalsGpuUniforms, atlas_modes),
            64
        );
        assert_eq!(
            std::mem::offset_of!(TemporalOriginalsGpuUniforms, score_runtime),
            80
        );
        assert_eq!(
            std::mem::offset_of!(TemporalOriginalsGpuUniforms, garden_values),
            96
        );
        assert_eq!(
            std::mem::offset_of!(TemporalOriginalsGpuUniforms, garden_modes),
            112
        );
        assert_eq!(uniforms.loom_values, [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(uniforms.loom_geometry, [0.0, 16.0 / 9.0, 0.0, 0.0]);
        assert_eq!(uniforms.loom_modes, [0, 0, 1, 24]);
        assert_eq!(uniforms.atlas_values, [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(uniforms.atlas_modes[0..2], [0, 1]);
        assert_eq!(uniforms.score_runtime[1..], [3, 0x5566_7788, 0x1122_3344]);
        assert_eq!(uniforms.garden_modes[2], 7);
    }

    #[test]
    fn independent_cpu_reference_pins_every_loom_topology() {
        let base = TemporalLoomParams {
            amount: 1.0,
            depth: 1.0,
            phase: 0.17,
            scale: 1.4,
            angle: 23.0,
            folds: 7,
            quantization: 0,
            ..TemporalLoomParams::default()
        };
        let expected = [
            0.072_603_17,
            0.573_607_7,
            0.411_536_16,
            0.101_395_77,
            0.976_444_3,
            0.304_606_02,
        ];
        for (topology, expected) in [
            TemporalTopology::Linear,
            TemporalTopology::Radial,
            TemporalTopology::Spiral,
            TemporalTopology::Contour,
            TemporalTopology::Folded,
            TemporalTopology::Kaleidoscopic,
        ]
        .into_iter()
        .zip(expected)
        {
            let actual = temporal_loom_age(
                [0.73, 0.21],
                TemporalLoomParams { topology, ..base },
                16.0 / 9.0,
            );
            assert!((actual - expected).abs() < 2.0e-6, "{topology:?}: {actual}");
        }

        let quantized = temporal_loom_age(
            [0.73, 0.21],
            TemporalLoomParams {
                quantization: 5,
                ..base
            },
            16.0 / 9.0,
        );
        assert_eq!(quantized, 0.0, "five levels include both endpoints");
    }

    #[test]
    fn history_interpolation_honors_virtual_current_and_valid_layer_bounds() {
        assert_eq!(
            temporal_history_sample_plan(99.0, 0, TemporalInterpolation::Linear),
            TemporalHistorySamplePlan {
                lower_age: 0,
                upper_age: 0,
                mix: 0.0,
            }
        );
        assert_eq!(
            temporal_history_sample_plan(99.0, 2, TemporalInterpolation::Linear),
            TemporalHistorySamplePlan {
                lower_age: 1,
                upper_age: 1,
                mix: 0.0,
            }
        );
        assert_eq!(
            temporal_history_sample_plan(3.75, 24, TemporalInterpolation::Floor),
            TemporalHistorySamplePlan {
                lower_age: 3,
                upper_age: 3,
                mix: 0.0,
            }
        );
        assert_eq!(
            temporal_history_sample_plan(3.75, 24, TemporalInterpolation::Linear),
            TemporalHistorySamplePlan {
                lower_age: 3,
                upper_age: 4,
                mix: 0.75,
            }
        );
        assert_eq!(
            temporal_history_sample_plan(f32::NAN, 24, TemporalInterpolation::Linear).lower_age,
            0
        );
    }

    #[test]
    fn collision_atlas_seed_zero_is_valid_repeatable_and_score_driven() {
        let atlas = CollisionAtlasParams {
            amount: 1.0,
            seed: 0,
            territories: 19,
            collision: 0.75,
        };
        let uv = [0.617, 0.283];
        let plain_score = CollisionScoreParams::default();
        let state = CollisionScoreState::default();
        let first = collision_atlas_age(uv, atlas, 16.0 / 9.0, plain_score, state);
        let second = collision_atlas_age(uv, atlas, 16.0 / 9.0, plain_score, state);
        assert_eq!(first.to_bits(), second.to_bits());
        assert!(first.is_finite() && (0.0..=1.0).contains(&first));
        let another_seed = collision_atlas_age(
            uv,
            CollisionAtlasParams { seed: 1, ..atlas },
            16.0 / 9.0,
            plain_score,
            state,
        );
        assert_ne!(first.to_bits(), another_seed.to_bits());

        let conducted = collision_atlas_age(
            uv,
            atlas,
            16.0 / 9.0,
            CollisionScoreParams {
                enabled: true,
                seed: 0,
                ..CollisionScoreParams::default()
            },
            CollisionScoreState {
                state_index: 1,
                event_ordinal: 1,
            },
        );
        assert_ne!(first.to_bits(), conducted.to_bits());
    }

    #[test]
    fn event_resets_are_one_transaction_before_reads_and_full_score_counts() {
        let mut state = TemporalState::default();
        stage_and_commit(&mut state, &TemporalParams::default(), running(1.0 / 30.0));
        stage_and_commit(&mut state, &TemporalParams::default(), running(1.0 / 30.0));
        state.score = CollisionScoreState {
            state_index: 3,
            event_ordinal: 11,
        };
        let before = state.snapshot();
        let params = TemporalParams {
            originals: TemporalOriginalsParams {
                score: CollisionScoreParams {
                    enabled: true,
                    state_count: 4,
                    trigger: CollisionScoreTrigger::Boundary,
                    ..CollisionScoreParams::default()
                },
                reset: TemporalResetPolicy {
                    loop_boundary: TemporalEventResetMode::Memory,
                    downbeat: TemporalEventResetMode::Score,
                },
                ..TemporalOriginalsParams::default()
            },
            ..TemporalParams::default()
        };
        let plan = state.stage_frame(
            &params,
            TemporalFrameInput::new(
                1.0 / 30.0,
                TemporalFreezeState::Running,
                false,
                TemporalFrameEvents {
                    boundary_events: 6,
                    downbeat_events: 3,
                    ..TemporalFrameEvents::default()
                },
            ),
            [640, 360],
        );
        assert_eq!(plan.read.virtual_valid, 0, "reset precedes ring reads");
        assert_eq!(plan.score_events_consumed, 6);
        assert_eq!(state.history_valid, 1, "one new observation follows reset");
        assert_eq!(state.score.state_index, 2);
        assert_eq!(state.score.event_ordinal, 6);
        state.discard_staged();
        assert_eq!(state.snapshot(), before, "discard restores pre-event state");

        let frozen = state.stage_frame(
            &params,
            TemporalFrameInput::new(
                99.0,
                TemporalFreezeState::ProgramFrozen,
                false,
                TemporalFrameEvents {
                    boundary_events: 100,
                    downbeat_events: 100,
                    ..TemporalFrameEvents::default()
                },
            ),
            [640, 360],
        );
        assert_eq!(frozen.action, TemporalFrameAction::HoldFrozenOutput);
        assert_eq!(frozen.score_events_consumed, 0);
        assert_eq!(state.history_valid, before.history_valid);
        assert_eq!(state.score, before.score);
    }

    #[test]
    fn audio_onset_tracker_has_shared_hysteresis_and_pause_reanchoring() {
        let mut tracker = TemporalAudioOnsetTracker::default();
        assert_eq!(tracker.observe(0.49, true), 0);
        assert_eq!(tracker.observe(0.5, true), 1);
        assert_eq!(tracker.observe(0.3, true), 0);
        assert_eq!(tracker.observe(0.2, true), 0);
        assert_eq!(tracker.observe(0.19, true), 0);
        assert_eq!(tracker.observe(0.5, false), 0, "paused edge is consumed");
        assert_eq!(tracker.observe(0.9, true), 0, "resume cannot replay it");
        tracker.reanchor(0.0);
        assert_eq!(tracker.observe(1.0, true), 1);
        tracker.reanchor(f32::NAN);
        assert_eq!(tracker.observe(0.5, true), 1);
    }

    #[test]
    fn garden_gate_laws_cover_pixel_analytical_and_audio_signals() {
        let params = RefreshGardenParams {
            amount: 0.8,
            threshold: 0.5,
            softness: 0.0,
            ..RefreshGardenParams::default()
        };
        let signals = RefreshGardenSignals {
            temporal_delta: 0.6,
            luma: 0.4,
            chroma: 0.7,
            cellular_ridge: 0.3,
            audio_energy: 0.9,
            audio_onset: true,
            matte: 0.2,
            force_refresh: false,
        };
        for (gate, expected) in [
            (RefreshGardenGate::TemporalDelta, 0.8),
            (RefreshGardenGate::Luma, 0.0),
            (RefreshGardenGate::Chroma, 0.8),
            (RefreshGardenGate::CellularRidge, 0.0),
            (RefreshGardenGate::AudioEnergy, 0.8),
            (RefreshGardenGate::AudioOnset, 0.8),
            (RefreshGardenGate::Matte, 0.0),
        ] {
            assert_eq!(
                refresh_garden_admission(RefreshGardenParams { gate, ..params }, signals),
                expected,
                "{gate:?}"
            );
        }
        assert_eq!(
            refresh_garden_admission(
                RefreshGardenParams {
                    gate: RefreshGardenGate::Matte,
                    ..params
                },
                RefreshGardenSignals {
                    force_refresh: true,
                    ..signals
                }
            ),
            0.8
        );
    }

    #[test]
    fn garden_closed_form_is_invariant_to_24_30_60_frame_tick_grouping() {
        fn one_second(fps: u32) -> [f32; 4] {
            let garden = RefreshGardenParams {
                amount: 0.31,
                gate: RefreshGardenGate::Luma,
                threshold: 0.2,
                softness: 0.0,
                decay: 0.97,
                max_hold_ticks: 0,
            };
            let params = TemporalParams {
                originals: TemporalOriginalsParams {
                    garden,
                    ..TemporalOriginalsParams::default()
                },
                ..TemporalParams::default()
            };
            let current = [0.8, 0.4, 0.2, 1.0];
            let signals = RefreshGardenSignals {
                luma: 0.5,
                ..RefreshGardenSignals::default()
            };
            let mut memory = [0.1, 0.2, 0.3, 0.4];
            let mut state = TemporalState {
                feedback_valid: true,
                initialized: true,
                ..TemporalState::default()
            };
            // Prime an identical valid carrier without consuming a reference
            // tick, then compare only one second of authored observations.
            for _ in 0..fps {
                let plan = state.stage_frame(&params, running(1.0 / fps as f32), [640, 360]);
                memory = refresh_garden_pixel(
                    current,
                    memory,
                    garden,
                    signals,
                    plan.observation_ticks,
                    true,
                );
                state.commit_staged();
            }
            assert_eq!(state.total_reference_ticks, 30, "{fps} fps");
            memory
        }

        let reference = one_second(30);
        for fps in [24, 60] {
            let actual = one_second(fps);
            for channel in 0..4 {
                assert!(
                    (actual[channel] - reference[channel]).abs() < 2.0e-6,
                    "{fps} fps channel {channel}: {actual:?} vs {reference:?}"
                );
            }
        }
    }

    #[test]
    fn garden_max_hold_is_transactional_and_freeze_safe() {
        let params = TemporalParams {
            originals: TemporalOriginalsParams {
                garden: RefreshGardenParams {
                    amount: 1.0,
                    max_hold_ticks: 3,
                    ..RefreshGardenParams::default()
                },
                ..TemporalOriginalsParams::default()
            },
            ..TemporalParams::default()
        };
        let mut state = TemporalState::default();
        for expected in [false, false] {
            let plan = stage_and_commit(&mut state, &params, running(1.0 / 30.0));
            assert_eq!(plan.garden_force_refresh, expected);
        }
        let before = state.metrics();
        let staged = state.stage_frame(&params, running(1.0 / 30.0), [640, 360]);
        assert!(staged.garden_force_refresh);
        assert_eq!(
            staged.originals_uniforms.garden_modes[3] & GARDEN_FORCE_REFRESH_BIT,
            GARDEN_FORCE_REFRESH_BIT
        );
        state.discard_staged();
        assert_eq!(state.metrics(), before);

        let frozen = state.stage_frame(
            &params,
            TemporalFrameInput::new(
                10.0,
                TemporalFreezeState::ProgramFrozen,
                true,
                TemporalFrameEvents::default(),
            ),
            [640, 360],
        );
        assert_eq!(frozen.action, TemporalFrameAction::HoldFrozenOutput);
        assert!(!frozen.garden_force_refresh);
        state.commit_staged();
        assert_eq!(state.garden_hold_ticks, 2);

        let accepted = stage_and_commit(&mut state, &params, running(1.0 / 30.0));
        assert!(accepted.garden_force_refresh);
        assert_eq!(state.garden_hold_ticks, 0);
        state.reset_for(TemporalResetCause::ManualClear);
        assert_eq!(state.garden_hold_ticks, 0);
        assert!(state.freeze_hold_valid);
    }

    #[test]
    fn garden_runtime_packing_and_metrics_are_stable_engine_contracts() {
        let params = TemporalParams {
            originals: TemporalOriginalsParams {
                garden: RefreshGardenParams {
                    amount: 1.0,
                    max_hold_ticks: 1,
                    ..RefreshGardenParams::default()
                },
                ..TemporalOriginalsParams::default()
            },
            ..TemporalParams::default()
        };
        let input = TemporalFrameInput::new(
            1.0 / 30.0,
            TemporalFreezeState::Running,
            false,
            TemporalFrameEvents {
                audio_onset_events: 2,
                ..TemporalFrameEvents::default()
            },
        )
        .with_audio_energy(0.5);
        let mut state = TemporalState::default();
        let plan = state.stage_frame(&params, input, [640, 360]);
        assert_eq!(
            plan.originals_uniforms.garden_modes[3],
            32_768 | GARDEN_AUDIO_ONSET_BIT | GARDEN_FORCE_REFRESH_BIT
        );
        let staged = state.metrics();
        assert_eq!(staged.history_capacity, TEMPORAL_HISTORY_LEN);
        assert_eq!(staged.history_valid, 1);
        assert!(staged.carrier_valid);
        assert!(staged.freeze_hold_valid);
        assert!(staged.frame_staged);
        state.commit_staged();
        assert!(!state.metrics().frame_staged);
        state.reset_for(TemporalResetCause::ManualClear);
        let reset = state.metrics();
        assert_eq!(reset.last_reset, Some(TemporalResetCause::ManualClear));
        assert!(!reset.carrier_valid);
        assert!(reset.freeze_hold_valid);
    }

    #[test]
    fn originals_shader_contract_is_additive_bounded_and_reuses_three_texture_resources() {
        use sha2::{Digest, Sha256};

        let legacy = include_str!("shaders/temporal.wgsl");
        let originals = include_str!("shaders/temporal_originals.wgsl");
        assert_eq!(
            format!("{:x}", Sha256::digest(legacy.as_bytes())),
            "388fa95fc027c00078dc5d7d380370d335fc3fa332ec0e2dead9f7b202f269d6",
            "the shared legacy/Advanced shader source is a frozen contract"
        );
        assert!(!legacy.contains("TemporalOriginalsUniforms"));
        assert!(legacy.contains("if u._pad0 < 0.5 { return legacy_temporal(uv); }"));
        assert!(legacy.contains("advanced_feedback_premultiplied_linear"));
        assert_eq!(legacy.matches("textureLoad(feedback_tex").count(), 4);
        assert!(originals.contains("@group(1) @binding(0) var<uniform> u"));
        assert!(originals.contains("@group(1) @binding(1) var<uniform> originals"));
        assert!(originals.contains("for (var y_offset = -1; y_offset <= 1"));
        assert!(originals.contains("for (var x_offset = -1; x_offset <= 1"));
        assert!(!originals.contains("atlas_tex"));
        assert!(!originals.contains("garden_tex"));
        let legacy_originals = originals
            .split("fn premultiply_originals")
            .next()
            .expect("legacy Originals source prefix");
        assert_eq!(originals.matches("var feedback_tex: texture_2d").count(), 1);
        assert_eq!(
            legacy_originals
                .matches("textureSample(feedback_tex")
                .count(),
            1,
            "legacy feedback and Garden must share one carrier sample"
        );
        assert_eq!(legacy_originals.matches("textureSample(").count(), 5);
        assert_eq!(originals.matches("textureLoad(feedback_tex").count(), 4);
        assert!(originals.contains("advanced_woven_history"));
        assert!(originals.contains("return mix(lower, upper, fract(depth));"));
        let guard = legacy_originals
            .find("if discrete_age == 0u")
            .expect("virtual-current guard");
        let history_sample = legacy_originals
            .find("return textureSample(history_tex")
            .expect("history sample");
        assert!(guard < history_sample);
    }

    #[test]
    fn temporal_zero_and_originals_sequences_have_fixed_24_30_60_hashes() {
        use sha2::{Digest, Sha256};

        fn sequence_hash(fps: u32, originals: bool) -> String {
            let originals = if originals {
                TemporalOriginalsParams {
                    loom: TemporalLoomParams {
                        amount: 0.83,
                        topology: TemporalTopology::Kaleidoscopic,
                        interpolation: TemporalInterpolation::Linear,
                        depth: 0.91,
                        phase: 0.137,
                        scale: 1.23,
                        angle: 17.0,
                        folds: 7,
                        quantization: 8,
                    },
                    atlas: CollisionAtlasParams {
                        amount: 0.62,
                        seed: 0,
                        territories: 19,
                        collision: 0.8,
                    },
                    score: CollisionScoreParams {
                        enabled: true,
                        seed: 0,
                        state_count: 5,
                        trigger: CollisionScoreTrigger::Boundary,
                        ..CollisionScoreParams::default()
                    },
                    ..TemporalOriginalsParams::default()
                }
            } else {
                TemporalOriginalsParams::default()
            };
            let params = TemporalParams {
                feedback: 0.81,
                fb_zoom: 1.03,
                fb_rotate: 2.0,
                slitscan: 0.77,
                key_mode: 1.0,
                key_history: 2.0,
                originals,
                ..TemporalParams::default()
            };
            let mut state = TemporalState::default();
            let mut hash = Sha256::new();
            for frame in 0..fps {
                let boundary_events = u32::from(frame.saturating_mul(4) % fps == 0);
                let plan = state.stage_frame(
                    &params,
                    TemporalFrameInput::new(
                        1.0 / fps as f32,
                        TemporalFreezeState::Running,
                        false,
                        TemporalFrameEvents {
                            boundary_events,
                            ..TemporalFrameEvents::default()
                        },
                    ),
                    [1_920, 1_080],
                );
                hash.update(bytemuck::bytes_of(&plan.uniforms));
                hash.update(bytemuck::bytes_of(&plan.originals_uniforms));
                hash.update([
                    match plan.action {
                        TemporalFrameAction::PrimeFrozenOutput => 0,
                        TemporalFrameAction::HoldFrozenOutput => 1,
                        TemporalFrameAction::Advance {
                            record_history: false,
                        } => 2,
                        TemporalFrameAction::Advance {
                            record_history: true,
                        } => 3,
                    },
                    u8::from(plan.legacy_shader_active),
                    u8::from(plan.originals_shader_active),
                    0,
                ]);
                hash.update(plan.observation_ticks.to_le_bytes());
                hash.update(plan.score_events_consumed.to_le_bytes());
                hash.update(
                    plan.history_write_target
                        .and_then(|value| u32::try_from(value).ok())
                        .unwrap_or(u32::MAX)
                        .to_le_bytes(),
                );
                state.commit_staged();
            }
            format!("{:x}", hash.finalize())
        }

        let zero = [24, 30, 60].map(|fps| sequence_hash(fps, false));
        let originals = [24, 30, 60].map(|fps| sequence_hash(fps, true));
        assert_eq!(
            zero,
            [
                "c10caaf3ebbacbdc4253f491fc7e5019a2c442e3beed708299dfd063712f16c9",
                "cfca57df5fa8474a2fdfcb2d86d6c98a2038798d169d5c9322c863c46fb988ec",
                "bcdefa11815cef6a192ce01c95c8a3d0635178b25d67b4a5e765aa6671a5dc3c",
            ]
            .map(str::to_string)
        );
        assert_eq!(
            originals,
            [
                "d3e7fef535d9302210ddd2bf86c5b39b79c3ff083e1edf1e7095271cde4c1c67",
                "8a08e33e55014f398c84b4638fa7e7cb48a1f62ca0736b6ebd48e47fd3e5a184",
                "a7a8e3d965adcaa2ece191274d7280d8c1dafde21c3323bbcdd7fa17d81d529e",
            ]
            .map(str::to_string)
        );
    }
}
