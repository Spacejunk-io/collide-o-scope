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

use crate::effects::params::{
    normalized_slit_direction, FeedbackRigParams, TemporalParams, TEMPORAL_REFERENCE_FPS,
};
use crate::image_routing::{LayerImageStage, StableLayerId};
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

/// The B12 time-displace map: how slit-scan turns image position into a
/// clean-history age. `Ramp` is the exact existing angle path and the default.
/// The map vocabulary is derived from BENDR (MIT, © 2026 Steve Blythe), a
/// browser circuit-bent video processor; every law here is a rewrite.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TimeDisplaceMap {
    /// The existing angled ramp along `slit_direction`. Exact legacy path.
    #[default]
    Ramp,
    /// The picture times itself: bright things lag dark ones.
    Brightness,
    /// Time pushed out from the centre, aspect-correct.
    Radial,
    /// A per-scanline failure ramp — exactly what a time-base corrector does
    /// when it fails: a sawtooth over each 8-scanline group.
    TbcRamp,
    /// A wrapped horizontal ramp travelling across the frame on the 30 Hz
    /// reference clock; the fixed period is
    /// [`TIME_DISPLACE_SWEEP_PERIOD_TICKS`].
    Sweep,
}

impl TimeDisplaceMap {
    /// Permanent append-only shader code. Never renumber an existing entry.
    pub const fn gpu_code(self) -> u32 {
        match self {
            Self::Ramp => 0,
            Self::Brightness => 1,
            Self::Radial => 2,
            Self::TbcRamp => 3,
            Self::Sweep => 4,
        }
    }

    pub const ALL: [Self; 5] = [
        Self::Ramp,
        Self::Brightness,
        Self::Radial,
        Self::TbcRamp,
        Self::Sweep,
    ];
}

/// Scanlines per TBC-failure sawtooth period. Fixed law, not an authored
/// control: the 8-line group is the physical vocabulary of a failing
/// time-base corrector, matching the derived BENDR mechanism.
pub(crate) const TIME_DISPLACE_TBC_LINES: f32 = 8.0;

/// Reference ticks per full Sweep crossing. Fixed law: 600 ticks is 20
/// seconds at the 30 Hz reference, BENDR's map-drift rate (0.05 cycles per
/// second) expressed on our reference clock.
pub(crate) const TIME_DISPLACE_SWEEP_PERIOD_TICKS: u64 = 600;

/// Deterministic Sweep phase for one accepted frame, derived from the same
/// accumulated reference ticks the rig's noise epoch uses. Program Freeze
/// holds the tick count, so the sweep holds; export accumulates the same
/// ticks from frame-indexed time, so live and offline agree structurally.
pub(crate) fn time_displace_sweep_phase(total_reference_ticks: u64) -> f32 {
    (total_reference_ticks % TIME_DISPLACE_SWEEP_PERIOD_TICKS) as f32
        / TIME_DISPLACE_SWEEP_PERIOD_TICKS as f32
}

/// CPU reference for the B12 time-displace coordinate law, mirrored by the
/// slit-scan blocks of `temporal_originals.wgsl` expression for expression.
/// `uv` is the output coordinate, `slit_direction` the aspect-corrected
/// normalized scan direction, `aspect` the output aspect ratio,
/// `covered_luma` the current sample's alpha-covered luma, `frame_height`
/// the output height in scanlines, and `sweep_phase` the deterministic
/// phase from [`time_displace_sweep_phase`]. The return is the normalized
/// history coordinate in `[0, 1]`; depth clamping against the valid-history
/// counter stays in the caller, exactly as the legacy slit-scan law.
pub(crate) fn time_displace_coord(
    map: TimeDisplaceMap,
    uv: [f32; 2],
    slit_direction: [f32; 2],
    aspect: f32,
    covered_luma: f32,
    frame_height: f32,
    sweep_phase: f32,
) -> f32 {
    match map {
        TimeDisplaceMap::Ramp => {
            ((uv[0] - 0.5) * slit_direction[0] + (uv[1] - 0.5) * slit_direction[1] + 0.5)
                .clamp(0.0, 1.0)
        }
        TimeDisplaceMap::Brightness => covered_luma.clamp(0.0, 1.0),
        TimeDisplaceMap::Radial => {
            let centered = [(uv[0] - 0.5) * aspect, uv[1] - 0.5];
            ((centered[0] * centered[0] + centered[1] * centered[1]).sqrt() * 1.6).clamp(0.0, 1.0)
        }
        TimeDisplaceMap::TbcRamp => {
            let line_phase = uv[1].clamp(0.0, 1.0) * frame_height / TIME_DISPLACE_TBC_LINES;
            (line_phase - line_phase.floor()).clamp(0.0, 1.0)
        }
        TimeDisplaceMap::Sweep => {
            let travelled = uv[0] - sweep_phase;
            (travelled - travelled.floor()).clamp(0.0, 1.0)
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
    Motion,
}

/// Stable current-frame image route used only by the Garden Matte gate.
///
/// The saved position is provenance for patch/Morph capture. Live routing is
/// always by `layer_id`; a removed donor becomes the explicit Missing variant
/// and can never bind a replacement that later occupies the old position.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RefreshGardenMatteRoute {
    #[default]
    None,
    SelectedLayer {
        layer_id: StableLayerId,
        saved_position: SavedLayerPosition,
        stage: LayerImageStage,
    },
    MissingSelectedLayer {
        saved_position: SavedLayerPosition,
        stage: LayerImageStage,
    },
}

impl RefreshGardenMatteRoute {
    pub(crate) fn mark_layer_missing(
        &mut self,
        removed: StableLayerId,
        removed_position: SavedLayerPosition,
    ) {
        let Self::SelectedLayer {
            layer_id, stage, ..
        } = *self
        else {
            return;
        };
        if layer_id == removed {
            *self = Self::MissingSelectedLayer {
                saved_position: removed_position,
                stage,
            };
        }
    }

    pub(crate) fn refresh_saved_position(
        &mut self,
        mut position_of: impl FnMut(StableLayerId) -> Option<SavedLayerPosition>,
    ) {
        let Self::SelectedLayer {
            layer_id,
            saved_position,
            ..
        } = self
        else {
            return;
        };
        if let Some(position) = position_of(*layer_id) {
            *saved_position = position;
        }
    }
}

/// Stable selected-layer route used by the Garden Motion gate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RefreshGardenMotionRoute {
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

impl RefreshGardenMotionRoute {
    pub(crate) fn mark_layer_missing(
        &mut self,
        removed: StableLayerId,
        removed_position: SavedLayerPosition,
    ) {
        if matches!(self, Self::SelectedLayer { layer_id, .. } if *layer_id == removed) {
            *self = Self::MissingSelectedLayer {
                saved_position: removed_position,
            };
        }
    }

    pub(crate) fn refresh_saved_position(
        &mut self,
        mut position_of: impl FnMut(StableLayerId) -> Option<SavedLayerPosition>,
    ) {
        let Self::SelectedLayer {
            layer_id,
            saved_position,
        } = self
        else {
            return;
        };
        if let Some(position) = position_of(*layer_id) {
            *saved_position = position;
        }
    }
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
            Self::Motion => 7,
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
    pub matte_route: RefreshGardenMatteRoute,
    pub motion_route: RefreshGardenMotionRoute,
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
            matte_route: RefreshGardenMatteRoute::None,
            motion_route: RefreshGardenMotionRoute::None,
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
            matte_route: self.matte_route,
            motion_route: self.motion_route,
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

/// The bounded laws selected by one Collision Score state. This is a true
/// finite-state table: revisiting a state produces the same law for the same
/// authored seed, independent of render cadence and wall time. The event
/// ordinal remains available to Collision Atlas as its established seeded
/// territory stream, but it is not allowed to make the state table unbounded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CollisionScoreMemoryLaw {
    pub loom_depth_scale: f32,
    pub loom_phase_offset: f32,
    pub atlas_collision_offset: f32,
    pub garden_threshold_offset: f32,
    pub garden_decay_scale: f32,
}

impl Default for CollisionScoreMemoryLaw {
    fn default() -> Self {
        Self {
            loom_depth_scale: 1.0,
            loom_phase_offset: 0.0,
            atlas_collision_offset: 0.0,
            garden_threshold_offset: 0.0,
            garden_decay_scale: 1.0,
        }
    }
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

/// Resolve the deterministic bounded law for one Score state. Five isolated
/// hash lanes prevent adding a law from perturbing the established lanes.
pub(crate) fn collision_score_memory_law(
    params: CollisionScoreParams,
    state: CollisionScoreState,
) -> CollisionScoreMemoryLaw {
    let params = params.sanitized();
    if !params.enabled {
        return CollisionScoreMemoryLaw::default();
    }
    let state = u32::from(state.state_index % params.state_count);
    let base = params.seed ^ state.wrapping_mul(0x9e37_79b9);
    let lane = |domain: u32| hash_unit(mix_u32(base ^ domain));
    CollisionScoreMemoryLaw {
        loom_depth_scale: 0.5 + lane(0x10d3_71a5) * 0.5,
        loom_phase_offset: lane(0x8f4c_2bd7) - 0.5,
        atlas_collision_offset: (lane(0x6a09_e667) - 0.5) * 0.5,
        garden_threshold_offset: (lane(0xbb67_ae85) - 0.5) * 0.5,
        garden_decay_scale: 0.75 + lane(0x3c6e_f372) * 0.25,
    }
}

fn score_effective_originals(
    params: TemporalOriginalsParams,
    state: CollisionScoreState,
) -> TemporalOriginalsParams {
    let mut params = params.sanitized();
    let law = collision_score_memory_law(params.score, state);
    params.loom.depth = (params.loom.depth * law.loom_depth_scale).clamp(0.0, 1.0);
    params.loom.phase = (params.loom.phase + law.loom_phase_offset).clamp(-1_000.0, 1_000.0);
    params.atlas.collision = (params.atlas.collision + law.atlas_collision_offset).clamp(0.0, 1.0);
    params.garden.threshold =
        (params.garden.threshold + law.garden_threshold_offset).clamp(0.0, 1.0);
    params.garden.decay = (params.garden.decay * law.garden_decay_scale).clamp(0.0, 1.0);
    params
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

/// Photographic long-exposure ghosting over the clean fixed-rate history.
///
/// `shutter_frames` counts the virtual current frame plus prior 30 Hz ring
/// samples. The shader clamps it against initialized history, so startup is
/// deterministic. The shader evaluates short shutters exactly and uniformly
/// stratifies longer ones into at most eight total samples, so the maximum
/// per-pixel work stays independent of the 24-frame authored span.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LongExposureParams {
    pub amount: f32,
    pub shutter_frames: u8,
}

impl Default for LongExposureParams {
    fn default() -> Self {
        Self {
            amount: 0.0,
            shutter_frames: 12,
        }
    }
}

impl LongExposureParams {
    pub(crate) fn sanitized(self) -> Self {
        Self {
            amount: finite_or(self.amount, 0.0).clamp(0.0, 1.0),
            shutter_frames: self.shutter_frames.clamp(2, TEMPORAL_HISTORY_LEN as u8),
        }
    }
}

/// M3 authoring contract. Every member defaults to a strict zero/no-op.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TemporalOriginalsParams {
    pub loom: TemporalLoomParams,
    pub atlas: CollisionAtlasParams,
    pub garden: RefreshGardenParams,
    /// Bounded photographic integration over the clean 30 Hz history ring.
    /// Amount zero is an exact no-op; the runtime never allocates a second
    /// history and never samples an unwritten layer.
    pub long_exposure: LongExposureParams,
    pub score: CollisionScoreParams,
    pub reset: TemporalResetPolicy,
}

impl TemporalOriginalsParams {
    pub(crate) fn sanitized(self) -> Self {
        Self {
            loom: self.loom.sanitized(),
            atlas: self.atlas.sanitized(),
            garden: self.garden.sanitized(),
            long_exposure: self.long_exposure.sanitized(),
            score: self.score.sanitized(),
            reset: self.reset,
        }
    }

    pub(crate) fn is_zero(self) -> bool {
        self.loom.amount == 0.0
            && self.atlas.amount == 0.0
            && self.garden.amount == 0.0
            && self.long_exposure.amount == 0.0
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
    /// Explicit all-pixel Refresh Garden admissions. Multiple events in one
    /// rendered frame are counted for replay/telemetry, while the pixel law
    /// needs only one open gate for that batch of reference ticks.
    pub garden_refresh_events: u32,
}

/// Hard cap for explicit performance events retained for deterministic
/// offline replay. Boundary/downbeat/audio events are regenerated from their
/// authoritative sources and therefore do not consume this track.
pub(crate) const MAX_RECORDED_TEMPORAL_EVENT_POINTS: usize = 4_096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RecordedTemporalEventPoint {
    /// Reference tick relative to the first recorded explicit event.
    pub reference_tick: u64,
    pub manual_score_events: u32,
    pub garden_refresh_events: u32,
}

/// Portable, bounded explicit-event track passed by value into an export job.
/// It contains no wall time, GPU state, source path, or runtime identity.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct TemporalEventTrack {
    origin_tick: Option<u64>,
    events: Vec<RecordedTemporalEventPoint>,
    truncated: bool,
}

impl TemporalEventTrack {
    pub(crate) fn clear(&mut self) {
        self.origin_tick = None;
        self.events.clear();
        self.truncated = false;
    }

    pub(crate) fn events(&self) -> &[RecordedTemporalEventPoint] {
        &self.events
    }

    pub(crate) fn truncated(&self) -> bool {
        self.truncated
    }

    pub(crate) fn record_accepted(
        &mut self,
        absolute_reference_tick: u64,
        events: TemporalFrameEvents,
    ) -> bool {
        if events.manual_events == 0 && events.garden_refresh_events == 0 {
            return true;
        }
        let origin = *self.origin_tick.get_or_insert(absolute_reference_tick);
        let reference_tick = absolute_reference_tick.saturating_sub(origin);
        if let Some(last) = self
            .events
            .last_mut()
            .filter(|last| last.reference_tick == reference_tick)
        {
            last.manual_score_events = last
                .manual_score_events
                .saturating_add(events.manual_events);
            last.garden_refresh_events = last
                .garden_refresh_events
                .saturating_add(events.garden_refresh_events);
            return true;
        }
        if self.events.len() >= MAX_RECORDED_TEMPORAL_EVENT_POINTS {
            self.truncated = true;
            return false;
        }
        self.events.push(RecordedTemporalEventPoint {
            reference_tick,
            manual_score_events: events.manual_events,
            garden_refresh_events: events.garden_refresh_events,
        });
        true
    }

    pub(crate) fn replay(&self) -> TemporalEventReplay<'_> {
        TemporalEventReplay {
            events: &self.events,
            cursor: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TemporalEventReplay<'a> {
    events: &'a [RecordedTemporalEventPoint],
    cursor: usize,
}

impl TemporalEventReplay<'_> {
    /// Consume every explicit event whose reference tick is due. A low display
    /// rate may cross several points in one frame; counts are saturated rather
    /// than silently dropping intermediate gestures.
    pub(crate) fn events_due(&mut self, reference_tick: u64) -> TemporalFrameEvents {
        let mut due = TemporalFrameEvents::default();
        while let Some(point) = self
            .events
            .get(self.cursor)
            .filter(|point| point.reference_tick <= reference_tick)
        {
            due.manual_events = due.manual_events.saturating_add(point.manual_score_events);
            due.garden_refresh_events = due
                .garden_refresh_events
                .saturating_add(point.garden_refresh_events);
            self.cursor += 1;
        }
        due
    }
}

/// Accepted-frame clock for live event recording. Rejected frames do not
/// advance it; Program Freeze does not call it. Its only output is a 30 Hz
/// integer address, matching the temporal authoring reference.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct TemporalEventRecorder {
    track: TemporalEventTrack,
    accepted_seconds: f64,
}

impl TemporalEventRecorder {
    pub(crate) fn record_accepted(
        &mut self,
        delta_seconds: f32,
        events: TemporalFrameEvents,
    ) -> bool {
        let tick = (self.accepted_seconds * f64::from(TEMPORAL_REFERENCE_FPS))
            .round()
            .clamp(0.0, u64::MAX as f64) as u64;
        let recorded = self.track.record_accepted(tick, events);
        self.accepted_seconds += f64::from(sanitize_delta(delta_seconds));
        recorded
    }

    pub(crate) fn track(&self) -> &TemporalEventTrack {
        &self.track
    }

    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }
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

    /// The program-advancing delta: zero while the program is frozen, so a
    /// consumer clocked by this (the B4 display stage's phosphor decay and
    /// field clock) holds exactly as the program holds.
    pub(crate) fn program_advancing_delta(&self) -> f32 {
        if self.freeze.program_advances() {
            self.delta_seconds
        } else {
            0.0
        }
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

/// The B3 feedback-rig shader contract. The legacy 64-byte uniform above is
/// frozen, so the rig lives in its own third fixed binding exactly as the
/// originals took a second: a default patch keeps executing the old
/// expression literally unchanged, gated by `modes[3] == 0`.
///
/// Six complete 16-byte lanes, mirroring six WGSL vec4 values.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct TemporalRigGpuUniforms {
    /// offset_x, offset_y, hue rotate in radians, saturation
    pub values_a: [f32; 4],
    /// chroma displace, blur, sharpen, drive
    pub values_b: [f32; 4],
    /// pivot, threshold, noise, tick mix (clamped tick fraction)
    pub values_c: [f32; 4],
    /// gain r, gain g, gain b, servo strength (0 off/defeated, 1 engaged)
    pub values_d: [f32; 4],
    /// reflect x, reflect y, shape code, edge code
    pub modes_a: [u32; 4],
    /// noise epoch (30 Hz reference ticks), rig active flag, reserved x2
    pub modes_b: [u32; 4],
}

const _: () = assert!(std::mem::size_of::<TemporalRigGpuUniforms>() == 96);

/// CPU reference for the B3 rig coordinate law, mirroring
/// `rig_reflect_offset` + `rig_resolve_edge` in the temporal shaders,
/// expression for expression. `p_centered` is the already rotated/zoomed
/// centred lookup; the return is the resolved texture coordinate and its
/// coverage. `Transparent` is the exact historical inside test; the other
/// three laws always cover.
pub(crate) fn feedback_rig_resolve(
    p_centered: [f32; 2],
    rig: &FeedbackRigParams,
) -> ([f32; 2], f32) {
    fn mirror_unit(value: f32) -> f32 {
        let half = value / 2.0;
        let period = (half - half.floor()) * 2.0;
        let folded = if period > 1.0 { 2.0 - period } else { period };
        folded.clamp(0.0, 1.0)
    }
    let mut p = p_centered;
    if rig.reflect_x {
        p[0] = -p[0];
    }
    if rig.reflect_y {
        p[1] = -p[1];
    }
    let p = [p[0] + rig.offset_x + 0.5, p[1] + rig.offset_y + 0.5];
    match rig.edge {
        crate::motion::MotionBoundaryMode::Mirror => ([mirror_unit(p[0]), mirror_unit(p[1])], 1.0),
        crate::motion::MotionBoundaryMode::Wrap => (
            [
                (p[0] - p[0].floor()).clamp(0.0, 1.0),
                (p[1] - p[1].floor()).clamp(0.0, 1.0),
            ],
            1.0,
        ),
        crate::motion::MotionBoundaryMode::Hold => {
            ([p[0].clamp(0.0, 1.0), p[1].clamp(0.0, 1.0)], 1.0)
        }
        crate::motion::MotionBoundaryMode::Transparent => {
            let inside = if p[0] >= 0.0 && p[0] <= 1.0 && p[1] >= 0.0 && p[1] <= 1.0 {
                1.0
            } else {
                0.0
            };
            ([p[0].clamp(0.0, 1.0), p[1].clamp(0.0, 1.0)], inside)
        }
    }
}

fn rig_reference_avalanche(value: u32) -> u32 {
    let mut x = value;
    x = (x ^ (x >> 16)).wrapping_mul(0x7feb_352d);
    x = (x ^ (x >> 15)).wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

fn rig_reference_shape(x: f32, shape: crate::effects::params::FeedbackShape) -> f32 {
    use crate::effects::params::FeedbackShape;
    match shape {
        FeedbackShape::Soft => (2.0 * x).tanh() * 0.5,
        FeedbackShape::Wrap => {
            let v = x + 0.5;
            (v - v.floor()) - 0.5
        }
        FeedbackShape::Fold => {
            let v = (x + 0.5) / 2.0;
            let t = (v - v.floor()) * 2.0;
            if t > 1.0 {
                1.5 - t
            } else {
                t - 0.5
            }
        }
        FeedbackShape::Clamp => x.clamp(-0.5, 0.5),
    }
}

/// CPU reference for the B3 rig colour pipeline, mirroring `rig_grade` in the
/// temporal shaders, expression for expression: gains, YIQ hue rotation,
/// saturation, the tick-mixed waveshaper, threshold decay, deterministic loop
/// noise, and the compressive servo. `frag_px` is the fragment's quantized
/// pixel in feedback-texture space and `noise_epoch` the 30 Hz reference tick.
pub(crate) fn feedback_rig_grade(
    rgb: [f32; 3],
    rig: &FeedbackRigParams,
    tick_mix: f32,
    frag_px: [u32; 2],
    noise_epoch: u32,
) -> [f32; 3] {
    use crate::effects::params::FeedbackShape;
    let servo_strength = if rig.servo && !rig.servo_defeated {
        1.0
    } else {
        0.0
    };
    let mut color = [
        rgb[0] * rig.gain_r,
        rgb[1] * rig.gain_g,
        rgb[2] * rig.gain_b,
    ];
    let hue = rig.hue_rotate.to_radians();
    if hue.abs() > 1.0e-6 {
        let y = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2];
        let i = 0.596 * color[0] - 0.274 * color[1] - 0.322 * color[2];
        let q = 0.211 * color[0] - 0.523 * color[1] + 0.312 * color[2];
        let (sin, cos) = hue.sin_cos();
        let i2 = i * cos - q * sin;
        let q2 = i * sin + q * cos;
        color = [
            y + 0.956 * i2 + 0.621 * q2,
            y - 0.272 * i2 - 0.647 * q2,
            y - 1.106 * i2 + 1.703 * q2,
        ];
    }
    if (rig.saturation - 1.0).abs() > 1.0e-6 {
        let luma = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2];
        for channel in &mut color {
            *channel = luma + (*channel - luma) * rig.saturation;
        }
    }
    let tick_mix = tick_mix.clamp(0.0, 1.0);
    if rig.shape != FeedbackShape::Clamp || (rig.drive - 1.0).abs() > 1.0e-6 {
        for channel in &mut color {
            let x = (*channel - rig.pivot) * rig.drive;
            let shaped = rig_reference_shape(x, rig.shape) + rig.pivot;
            *channel += (shaped - *channel) * tick_mix;
        }
    }
    if rig.threshold > 0.0 {
        let luma = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2];
        let edge0 = rig.threshold - 0.05;
        let edge1 = rig.threshold + 0.05;
        let t = ((luma - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
        let gate = t * t * (3.0 - 2.0 * t);
        let factor = 1.0 + (gate - 1.0) * tick_mix;
        for channel in &mut color {
            *channel *= factor;
        }
    }
    if rig.noise > 0.0 {
        let seed = rig_reference_avalanche(
            frag_px[0]
                ^ frag_px[1].wrapping_mul(0x9e37_79b9)
                ^ noise_epoch.wrapping_mul(0x85eb_ca6b)
                ^ 0x4233_5247,
        );
        let sample = (seed & 0x00ff_ffff) as f32 / 16_777_216.0;
        let offset = (sample * 2.0 - 1.0) * rig.noise * 0.25;
        for channel in &mut color {
            *channel += offset;
        }
    }
    if servo_strength > 0.0 {
        let luma = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2];
        let compression = 1.0 + (luma - 1.0).max(0.0);
        let mix = tick_mix * servo_strength;
        for channel in &mut color {
            let compressed = *channel / compression;
            *channel += (compressed - *channel) * mix;
        }
    }
    [color[0].max(0.0), color[1].max(0.0), color[2].max(0.0)]
}

impl TemporalRigGpuUniforms {
    fn for_frame(
        rig: FeedbackRigParams,
        authored_identity: bool,
        tick_mix: f32,
        noise_epoch: u64,
    ) -> Self {
        let servo_strength = if rig.servo && !rig.servo_defeated {
            1.0
        } else {
            0.0
        };
        Self {
            values_a: [
                rig.offset_x,
                rig.offset_y,
                rig.hue_rotate.to_radians(),
                rig.saturation,
            ],
            values_b: [rig.chroma_displace, rig.blur, rig.sharpen, rig.drive],
            values_c: [rig.pivot, rig.threshold, rig.noise, tick_mix],
            values_d: [rig.gain_r, rig.gain_g, rig.gain_b, servo_strength],
            modes_a: [
                u32::from(rig.reflect_x),
                u32::from(rig.reflect_y),
                rig.shape.code(),
                rig.edge.code(),
            ],
            modes_b: [
                (noise_epoch & 0xffff_ffff) as u32,
                // The activity flag answers from the AUTHORED values, not the
                // frame-scaled copies: a frame-scaled gain is transiently 1 at
                // dt 0 while the authored loop is emphatically not identity.
                u32::from(!authored_identity),
                0,
                0,
            ],
        }
    }
}

/// Additive M3 shader contract. The legacy uniform above is intentionally
/// frozen; originals live in a second fixed binding so a zero/default patch
/// can keep executing the old shader and pipeline literally unchanged.
///
/// Every member is a complete 16-byte lane, mirroring nine WGSL vec4 values
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
    /// amount, shutter frames (current inclusive), reserved, reserved
    pub long_exposure_values: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<TemporalOriginalsGpuUniforms>() == 144);

impl TemporalOriginalsGpuUniforms {
    #[allow(
        clippy::too_many_arguments,
        reason = "one private builder with every frame fact named beats a second struct"
    )]
    fn for_frame(
        params: TemporalOriginalsParams,
        score: CollisionScoreState,
        dimensions: [u32; 2],
        observation_ticks: u32,
        audio_energy: f32,
        audio_onset: bool,
        force_garden_refresh: bool,
        time_displace_map: TimeDisplaceMap,
        time_displace_interp: bool,
        total_reference_ticks: u64,
    ) -> Self {
        let params = score_effective_originals(params, score);
        let aspect = dimensions[0].max(1) as f32 / dimensions[1].max(1) as f32;
        let ordinal = score.event_ordinal;
        // B12 lanes ride the two reserved loom_geometry slots plus one
        // reserved atlas_values slot. A default map keeps all three at zero,
        // so pre-B12 uniform bytes are unchanged; the sweep phase is only
        // populated when Sweep is authored, so a default patch's uniforms do
        // not vary with the tick counter.
        let sweep_phase = if time_displace_map == TimeDisplaceMap::Sweep {
            time_displace_sweep_phase(total_reference_ticks)
        } else {
            0.0
        };
        Self {
            loom_values: [
                params.loom.amount,
                params.loom.depth,
                params.loom.phase,
                params.loom.scale,
            ],
            loom_geometry: [
                params.loom.angle.to_radians(),
                aspect,
                time_displace_map.gpu_code() as f32,
                f32::from(time_displace_interp),
            ],
            loom_modes: [
                params.loom.topology.gpu_code(),
                params.loom.interpolation.gpu_code(),
                u32::from(params.loom.folds),
                u32::from(params.loom.quantization),
            ],
            atlas_values: [
                params.atlas.amount,
                params.atlas.collision,
                sweep_phase,
                0.0,
            ],
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
            long_exposure_values: [
                params.long_exposure.amount,
                f32::from(params.long_exposure.shutter_frames),
                0.0,
                0.0,
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
    /// B3 feedback-rig lanes for the third fixed binding.
    pub rig_uniforms: TemporalRigGpuUniforms,
    pub legacy_shader_active: bool,
    pub originals_shader_active: bool,
    pub history_write_target: Option<usize>,
    pub observation_ticks: u32,
    pub score_events_consumed: u32,
    pub garden_refresh_events_consumed: u32,
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
                    frame_params.slit_map,
                    frame_params.slit_interp,
                    before.total_reference_ticks,
                ),
                rig_uniforms: TemporalRigGpuUniforms::for_frame(
                    frame_params.rig,
                    params.rig.is_identity(),
                    TemporalParams::rig_tick_mix(input.delta_seconds),
                    before.total_reference_ticks,
                ),
                legacy_shader_active: frame_params.is_active(),
                originals_shader_active: !frame_params.originals.is_zero()
                    || frame_params.time_displace_active(),
                history_write_target: None,
                observation_ticks: 0,
                score_events_consumed: 0,
                garden_refresh_events_consumed: 0,
                garden_force_refresh: false,
            };
        }

        let observation_ticks = self.history_ticks_for_delta(input.delta_seconds);
        let record_history = observation_ticks > 0;
        let history_write_target = record_history.then(|| {
            self.record_history_frame();
            self.history_write
        });

        let periodic_garden_refresh =
            self.advance_garden_hold(frame_params.originals.garden, observation_ticks);
        let garden_refresh_events_consumed = input.events.garden_refresh_events;
        let garden_force_refresh = periodic_garden_refresh || garden_refresh_events_consumed > 0;

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
                frame_params.slit_map,
                frame_params.slit_interp,
                before.total_reference_ticks,
            ),
            rig_uniforms: TemporalRigGpuUniforms::for_frame(
                frame_params.rig,
                params.rig.is_identity(),
                TemporalParams::rig_tick_mix(input.delta_seconds),
                before.total_reference_ticks,
            ),
            legacy_shader_active: frame_params.is_active(),
            originals_shader_active: !frame_params.originals.is_zero()
                || frame_params.time_displace_active(),
            history_write_target,
            observation_ticks,
            score_events_consumed,
            garden_refresh_events_consumed,
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

/// CPU reference for the Amdahl-bounded long-exposure sampling law. The
/// returned ages exclude the virtual current frame (age zero). Short shutters
/// visit every initialized age; shutters above eight total samples retain the
/// endpoints and distribute seven history reads uniformly over the interval.
#[cfg(test)]
pub(crate) fn long_exposure_sample_ages(shutter_frames: u8, virtual_valid: u32) -> Vec<u32> {
    if virtual_valid <= 1 {
        return Vec::new();
    }
    let requested = u32::from(shutter_frames.clamp(2, TEMPORAL_HISTORY_LEN as u8));
    let available = requested.min(virtual_valid.min(TEMPORAL_HISTORY_LEN));
    let sample_count = available.min(8);
    (1..sample_count)
        .map(|sample_index| {
            ((sample_index as f32 * (available - 1) as f32 / (sample_count - 1) as f32).round())
                as u32
        })
        .collect()
}

/// The one hostile-dt law. It is shared with the gesture recorder so both
/// reference-tick clocks treat a non-finite or negative frame delta the same
/// way; there is deliberately no second copy of this rule.
pub(crate) fn sanitize_delta(delta_seconds: f32) -> f32 {
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
    let mut params = params.sanitized();
    let score_params = score_params.sanitized();
    let score_law = collision_score_memory_law(score_params, score);
    params.collision = (params.collision + score_law.atlas_collision_offset).clamp(0.0, 1.0);
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
    pub motion: f32,
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
            RefreshGardenGate::Motion => self.motion,
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
            rig: FeedbackRigParams::default(),
            ..TemporalParams::default()
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
    fn originals_uniform_is_nine_explicit_vec4_lanes_and_sanitizes_controls() {
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
            long_exposure: LongExposureParams {
                amount: f32::NAN,
                shutter_frames: 0,
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
            TimeDisplaceMap::Ramp,
            false,
            0,
        );
        assert_eq!(std::mem::size_of::<TemporalOriginalsGpuUniforms>(), 144);
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
        assert_eq!(
            std::mem::offset_of!(TemporalOriginalsGpuUniforms, long_exposure_values),
            128
        );
        assert_eq!(uniforms.loom_values, [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(uniforms.loom_geometry, [0.0, 16.0 / 9.0, 0.0, 0.0]);
        assert_eq!(uniforms.loom_modes, [0, 0, 1, 24]);
        assert_eq!(uniforms.atlas_values, [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(uniforms.atlas_modes[0..2], [0, 1]);
        assert_eq!(uniforms.long_exposure_values, [0.0, 2.0, 0.0, 0.0]);
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
    fn garden_routes_refresh_provenance_and_tombstone_removed_stable_ids() {
        let selected = StableLayerId::new(91).unwrap();
        let unrelated = StableLayerId::new(44).unwrap();
        let original_position = SavedLayerPosition::new(3).unwrap();
        let moved_position = SavedLayerPosition::new(1).unwrap();
        let removed_position = SavedLayerPosition::new(7).unwrap();
        let mut matte = RefreshGardenMatteRoute::SelectedLayer {
            layer_id: selected,
            saved_position: original_position,
            stage: LayerImageStage::PostLocalEffects,
        };
        let mut motion = RefreshGardenMotionRoute::SelectedLayer {
            layer_id: selected,
            saved_position: original_position,
        };
        matte.refresh_saved_position(|id| (id == selected).then_some(moved_position));
        motion.refresh_saved_position(|id| (id == selected).then_some(moved_position));
        assert!(matches!(
            matte,
            RefreshGardenMatteRoute::SelectedLayer { saved_position, .. }
                if saved_position == moved_position
        ));
        assert!(matches!(
            motion,
            RefreshGardenMotionRoute::SelectedLayer { saved_position, .. }
                if saved_position == moved_position
        ));
        matte.mark_layer_missing(unrelated, removed_position);
        motion.mark_layer_missing(unrelated, removed_position);
        assert!(matches!(
            matte,
            RefreshGardenMatteRoute::SelectedLayer { .. }
        ));
        assert!(matches!(
            motion,
            RefreshGardenMotionRoute::SelectedLayer { .. }
        ));
        matte.mark_layer_missing(selected, removed_position);
        motion.mark_layer_missing(selected, removed_position);
        assert_eq!(
            matte,
            RefreshGardenMatteRoute::MissingSelectedLayer {
                saved_position: removed_position,
                stage: LayerImageStage::PostLocalEffects,
            }
        );
        assert_eq!(
            motion,
            RefreshGardenMotionRoute::MissingSelectedLayer {
                saved_position: removed_position,
            }
        );

        let sanitized = RefreshGardenParams {
            amount: f32::NAN,
            matte_route: matte,
            motion_route: motion,
            ..RefreshGardenParams::default()
        }
        .sanitized();
        assert_eq!(sanitized.amount, 0.0);
        assert_eq!(sanitized.matte_route, matte);
        assert_eq!(sanitized.motion_route, motion);
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
            motion: 0.75,
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
            (RefreshGardenGate::Motion, 0.8),
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
                ..RefreshGardenParams::default()
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

        let legacy = include_str!("shaders/temporal.wgsl").replace("\r\n", "\n");
        let originals = include_str!("shaders/temporal_originals.wgsl");
        // B3 re-froze this hash: the feedback rig joined both variants behind
        // its identity gate, with the historical expression untouched in the
        // rig-inactive branch.
        assert_eq!(
            format!("{:x}", Sha256::digest(legacy.as_bytes())),
            "07e5cde0a6f24ed03dfa0e9f8ddf086fe3e3d4de4227a17b6a8ccf8773e59486",
            "the LF-canonical shared legacy/Advanced shader source is a frozen contract"
        );
        assert!(!legacy.contains("TemporalOriginalsUniforms"));
        assert!(legacy.contains("if u._pad0 < 0.5 { return legacy_temporal(uv); }"));
        assert!(legacy.contains("advanced_feedback_premultiplied_linear"));
        assert_eq!(legacy.matches("textureLoad(feedback_tex").count(), 4);
        assert!(originals.contains("@group(1) @binding(0) var<uniform> u"));
        assert!(originals.contains("@group(1) @binding(1) var<uniform> originals"));
        assert!(originals.contains("long_exposure_values: vec4f"));
        assert_eq!(
            originals
                .matches("for (var sample_index = 1u; sample_index < 8u")
                .count(),
            2
        );
        assert_eq!(
            originals
                .matches("let sample_count = min(available, 8u)")
                .count(),
            2
        );
        assert!(originals.contains("let available = min(requested, u32(u.valid_history))"));
        assert_eq!(originals.matches("textureLoad(history_tex").count(), 1);
        assert!(originals.contains("let coords = min(vec2u(normalized * vec2f(dimensions))"));
        assert!(originals.contains("for (var y_offset = -1; y_offset <= 1"));
        assert!(originals.contains("for (var x_offset = -1; x_offset <= 1"));
        assert!(!originals.contains("atlas_tex"));
        assert!(!originals.contains("garden_tex"));
        let legacy_originals = originals
            .split("fn premultiply_originals")
            .next()
            .expect("legacy Originals source prefix");
        assert_eq!(originals.matches("var feedback_tex: texture_2d").count(), 1);
        // The frozen shared read stays singular; the other seven live inside
        // `rig_sample_legacy`, which only an authored rig reaches (one base,
        // two chromatic, four blur/sharpen cross taps).
        assert_eq!(
            legacy_originals
                .matches("textureSample(feedback_tex")
                .count(),
            8,
            "one shared legacy carrier sample plus the rig's seven gated taps"
        );
        assert_eq!(
            legacy_originals
                .split("fn rig_sample_legacy")
                .next()
                .expect("pre-rig prefix")
                .matches("textureSample(feedback_tex")
                .count(),
            0,
            "no feedback sample precedes the rig helper block"
        );
        // B12 re-counted this: the legacy slit block's inline history sample
        // now routes through `history_age_sample`, so one inline occurrence
        // left and no new sampling text arrived. The interpolation path's
        // extra load reuses the same helper.
        assert_eq!(legacy_originals.matches("textureSample(").count(), 11);
        // B12 time-displace maps: one shared coordinate helper serves both
        // variants, the map/interp lanes ride the reserved loom_geometry
        // slots, and both slit blocks keep the valid-history depth clamp.
        assert!(originals.contains("fn time_displace_coord"));
        assert!(originals.contains("const TIME_DISPLACE_TBC_LINES: f32 = 8.0;"));
        assert_eq!(
            originals
                .matches("if originals.loom_geometry.w > 0.5 {")
                .count(),
            2,
            "both variants carry the interpolation toggle"
        );
        assert_eq!(
            originals
                .matches("let max_depth = max(u.valid_history - 1.0, 0.0);")
                .count(),
            4,
            "slit-scan and Loom keep the valid-history clamp in both variants"
        );
        // Both shaders bind the rig at the same third fixed slot.
        assert!(legacy.contains("@group(1) @binding(2) var<uniform> rig"));
        assert!(originals.contains("@group(1) @binding(2) var<uniform> rig"));
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
    fn time_displace_map_codes_are_permanent_and_closed() {
        let expected: [(TimeDisplaceMap, u32); 5] = [
            (TimeDisplaceMap::Ramp, 0),
            (TimeDisplaceMap::Brightness, 1),
            (TimeDisplaceMap::Radial, 2),
            (TimeDisplaceMap::TbcRamp, 3),
            (TimeDisplaceMap::Sweep, 4),
        ];
        assert_eq!(TimeDisplaceMap::ALL.len(), expected.len());
        for (index, (map, code)) in expected.into_iter().enumerate() {
            assert_eq!(TimeDisplaceMap::ALL[index], map);
            assert_eq!(map.gpu_code(), code, "{map:?}");
        }
        assert_eq!(TimeDisplaceMap::default(), TimeDisplaceMap::Ramp);
    }

    #[test]
    fn time_displace_coord_matches_the_analytic_map_laws() {
        let close = |a: f32, b: f32| assert!((a - b).abs() < 1.0e-4, "{a} != {b}");
        let aspect = 16.0 / 9.0;
        let coord = |map, uv: [f32; 2], luma: f32, phase: f32| {
            time_displace_coord(map, uv, [0.6, 0.4], aspect, luma, 1_080.0, phase)
        };

        // Ramp is the exact legacy expression on the slit direction.
        close(coord(TimeDisplaceMap::Ramp, [0.3, 0.8], 0.0, 0.0), 0.5);
        close(
            coord(TimeDisplaceMap::Ramp, [1.0, 1.0], 0.0, 0.0),
            (0.5f32 * 0.6 + 0.5 * 0.4 + 0.5).clamp(0.0, 1.0),
        );
        // Brightness passes the alpha-covered luma through, clamped: bright
        // things lag dark ones.
        close(
            coord(TimeDisplaceMap::Brightness, [0.1, 0.9], 0.37, 0.0),
            0.37,
        );
        close(
            coord(TimeDisplaceMap::Brightness, [0.1, 0.9], 4.0, 0.0),
            1.0,
        );
        // Radial is aspect-correct distance from the centre, reach 1.6.
        close(coord(TimeDisplaceMap::Radial, [0.5, 0.5], 1.0, 0.0), 0.0);
        close(coord(TimeDisplaceMap::Radial, [0.5, 0.75], 0.0, 0.0), 0.4);
        close(
            coord(TimeDisplaceMap::Radial, [0.75, 0.5], 0.0, 0.0),
            0.25 * aspect * 1.6,
        );
        close(coord(TimeDisplaceMap::Radial, [1.0, 1.0], 0.0, 0.0), 1.0);
        // TbcRamp is a sawtooth over each 8-scanline group, constant in x.
        close(
            coord(TimeDisplaceMap::TbcRamp, [0.1, 4.0 / 1_080.0], 0.0, 0.0),
            0.5,
        );
        close(
            coord(TimeDisplaceMap::TbcRamp, [0.9, 4.0 / 1_080.0], 0.0, 0.0),
            0.5,
        );
        close(
            coord(TimeDisplaceMap::TbcRamp, [0.5, 10.0 / 1_080.0], 0.0, 0.0),
            0.25,
        );
        // Sweep is the wrapped horizontal ramp travelling by phase.
        close(coord(TimeDisplaceMap::Sweep, [0.25, 0.5], 0.0, 0.0), 0.25);
        close(coord(TimeDisplaceMap::Sweep, [0.25, 0.5], 0.0, 0.5), 0.75);
        close(coord(TimeDisplaceMap::Sweep, [0.9, 0.5], 0.0, 0.15), 0.75);

        // Hostile inputs stay inside the normalized coordinate for every map.
        for map in TimeDisplaceMap::ALL {
            for uv in [[-4.0, 9.0], [0.0, 1.0], [7.5, -0.25]] {
                let value = coord(map, uv, 123.0, 0.99);
                assert!((0.0..=1.0).contains(&value), "{map:?} {uv:?} -> {value}");
            }
        }
    }

    #[test]
    fn time_displace_sweep_phase_is_deterministic_on_the_reference_clock() {
        assert_eq!(time_displace_sweep_phase(0), 0.0);
        assert_eq!(time_displace_sweep_phase(150), 0.25);
        assert_eq!(time_displace_sweep_phase(600), 0.0);
        assert_eq!(time_displace_sweep_phase(624), 0.04);
        // The same tick count always yields the same phase: freeze holds it.
        assert_eq!(
            time_displace_sweep_phase(1_234_567),
            time_displace_sweep_phase(1_234_567)
        );
    }

    /// The unwritten-history guard, on the exact depth expressions the two
    /// slit blocks execute: every produced age stays strictly inside the
    /// valid-history counter, for both the banded floor law and the
    /// interpolated pair, at every map coordinate.
    #[test]
    fn time_displace_depth_clamps_against_the_valid_history_counter() {
        fn shader_depth_ages(
            coord: f32,
            slitscan: f32,
            valid_history: u32,
            interp: bool,
        ) -> Vec<u32> {
            if valid_history == 0 {
                // The whole block is gated on `u.valid_history > 0.5`.
                return Vec::new();
            }
            let max_depth = (valid_history as f32 - 1.0).max(0.0);
            let requested = coord * slitscan * (TEMPORAL_HISTORY_LEN as f32 - 1.0);
            let depth = requested.min(max_depth);
            if interp {
                vec![depth.floor() as u32, depth.ceil() as u32]
            } else if requested >= 1.0 && max_depth >= 1.0 {
                vec![depth.floor() as u32]
            } else {
                Vec::new()
            }
        }

        for valid_history in 0..=TEMPORAL_HISTORY_LEN {
            for step in 0..=20 {
                let coord = step as f32 / 20.0;
                for slitscan in [0.05, 0.5, 1.0] {
                    for interp in [false, true] {
                        for age in shader_depth_ages(coord, slitscan, valid_history, interp) {
                            assert!(
                                age < valid_history.max(1),
                                "age {age} escaped valid {valid_history} \
                                 (coord {coord}, slitscan {slitscan}, interp {interp})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn time_displace_selects_the_originals_shader_only_off_the_exact_ramp_path() {
        let plan_for = |map, interp, slitscan| {
            let params = TemporalParams {
                slitscan,
                slit_map: map,
                slit_interp: interp,
                ..TemporalParams::default()
            };
            TemporalState::default().stage_frame(&params, running(1.0 / 30.0), [1_920, 1_080])
        };

        // The authored default keeps the frozen legacy shader byte for byte.
        let default_plan = plan_for(TimeDisplaceMap::Ramp, false, 0.5);
        assert!(default_plan.legacy_shader_active);
        assert!(!default_plan.originals_shader_active);
        assert_eq!(default_plan.originals_uniforms.loom_geometry[2], 0.0);
        assert_eq!(default_plan.originals_uniforms.loom_geometry[3], 0.0);
        assert_eq!(default_plan.originals_uniforms.atlas_values[2], 0.0);

        // A non-default map or the interpolation toggle selects the bounded
        // additive originals pipeline and rides the reserved lanes.
        let radial = plan_for(TimeDisplaceMap::Radial, false, 0.5);
        assert!(radial.originals_shader_active);
        assert_eq!(radial.originals_uniforms.loom_geometry[2], 2.0);
        assert_eq!(radial.originals_uniforms.loom_geometry[3], 0.0);
        let interpolated = plan_for(TimeDisplaceMap::Ramp, true, 0.5);
        assert!(interpolated.originals_shader_active);
        assert_eq!(interpolated.originals_uniforms.loom_geometry[2], 0.0);
        assert_eq!(interpolated.originals_uniforms.loom_geometry[3], 1.0);

        // With slit-scan off there is no time displacement to run.
        let dormant = plan_for(TimeDisplaceMap::Sweep, true, 0.0);
        assert!(!dormant.originals_shader_active);

        // Sweep populates its phase lane from the accumulated reference
        // ticks; every other map leaves the lane at zero even as ticks
        // advance, so a default patch's uniform bytes never vary with time.
        let mut state = TemporalState::default();
        let sweep_params = TemporalParams {
            slitscan: 0.5,
            slit_map: TimeDisplaceMap::Sweep,
            ..TemporalParams::default()
        };
        let ramp_params = TemporalParams {
            slitscan: 0.5,
            ..TemporalParams::default()
        };
        for _ in 0..150 {
            state.stage_frame(&ramp_params, running(1.0 / 30.0), [1_920, 1_080]);
            state.commit_staged();
        }
        let ticks = state.metrics().total_reference_ticks;
        assert!(
            ticks > 100,
            "reference ticks must have accumulated: {ticks}"
        );
        let sweep_plan = state.stage_frame(&sweep_params, running(1.0 / 30.0), [1_920, 1_080]);
        assert_eq!(
            sweep_plan.originals_uniforms.atlas_values[2],
            time_displace_sweep_phase(ticks)
        );
        state.discard_staged();
        let ramp_plan = state.stage_frame(&ramp_params, running(1.0 / 30.0), [1_920, 1_080]);
        assert_eq!(ramp_plan.originals_uniforms.atlas_values[2], 0.0);
    }

    #[test]
    fn long_exposure_is_additive_bounded_and_neutral_by_default() {
        let default_plan = TemporalState::default().stage_frame(
            &TemporalParams::default(),
            running(1.0 / 30.0),
            [1_920, 1_080],
        );
        assert!(!default_plan.originals_shader_active);
        assert_eq!(default_plan.originals_uniforms.long_exposure_values[0], 0.0);

        let params = TemporalParams {
            originals: TemporalOriginalsParams {
                long_exposure: LongExposureParams {
                    amount: 0.75,
                    shutter_frames: u8::MAX,
                },
                ..TemporalOriginalsParams::default()
            },
            ..TemporalParams::default()
        };
        let plan =
            TemporalState::default().stage_frame(&params, running(1.0 / 30.0), [1_920, 1_080]);
        assert!(!plan.legacy_shader_active);
        assert!(plan.originals_shader_active);
        assert_eq!(plan.originals_uniforms.long_exposure_values[0], 0.75);
        assert_eq!(
            plan.originals_uniforms.long_exposure_values[1],
            TEMPORAL_HISTORY_LEN as f32
        );

        let dry = TemporalParams::default();
        assert_eq!(dry.originals.long_exposure, LongExposureParams::default());
        assert!(dry.originals.is_zero());

        assert!(long_exposure_sample_ages(24, 1).is_empty());
        assert_eq!(long_exposure_sample_ages(5, 24), vec![1, 2, 3, 4]);
        assert_eq!(
            long_exposure_sample_ages(24, 24),
            vec![3, 7, 10, 13, 16, 20, 23]
        );
        for valid in 0..=TEMPORAL_HISTORY_LEN {
            let ages = long_exposure_sample_ages(24, valid);
            assert!(ages.len() <= 7, "at most seven ring reads plus current");
            assert!(ages.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(ages.iter().all(|age| *age < valid.max(1)));
        }
    }

    #[test]
    fn explicit_event_track_is_bounded_counted_and_display_rate_invariant() {
        fn record_at(fps: u32) -> TemporalEventTrack {
            let mut recorder = TemporalEventRecorder::default();
            for frame in 0..fps * 3 {
                let events = TemporalFrameEvents {
                    manual_events: u32::from(frame == 0) + u32::from(frame == fps * 2),
                    garden_refresh_events: u32::from(frame == fps),
                    ..TemporalFrameEvents::default()
                };
                assert!(recorder.record_accepted(1.0 / fps as f32, events));
            }
            recorder.track().clone()
        }

        let expected = [
            RecordedTemporalEventPoint {
                reference_tick: 0,
                manual_score_events: 1,
                garden_refresh_events: 0,
            },
            RecordedTemporalEventPoint {
                reference_tick: 30,
                manual_score_events: 0,
                garden_refresh_events: 1,
            },
            RecordedTemporalEventPoint {
                reference_tick: 60,
                manual_score_events: 1,
                garden_refresh_events: 0,
            },
        ];
        for fps in [24, 30, 60] {
            assert_eq!(record_at(fps).events(), expected, "{fps} fps");
        }

        let track = record_at(60);
        let mut replay = track.replay();
        assert_eq!(replay.events_due(0).manual_events, 1);
        assert_eq!(replay.events_due(29), TemporalFrameEvents::default());
        let tick_60 = replay.events_due(60);
        assert_eq!(tick_60.garden_refresh_events, 1);
        assert_eq!(tick_60.manual_events, 1);
        assert_eq!(replay.events_due(u64::MAX), TemporalFrameEvents::default());

        let mut capped = TemporalEventTrack::default();
        for tick in 0..MAX_RECORDED_TEMPORAL_EVENT_POINTS as u64 {
            assert!(capped.record_accepted(
                tick,
                TemporalFrameEvents {
                    manual_events: 1,
                    ..TemporalFrameEvents::default()
                }
            ));
        }
        assert!(!capped.record_accepted(
            MAX_RECORDED_TEMPORAL_EVENT_POINTS as u64,
            TemporalFrameEvents {
                manual_events: 1,
                ..TemporalFrameEvents::default()
            }
        ));
        assert!(capped.truncated());
        assert_eq!(capped.events().len(), MAX_RECORDED_TEMPORAL_EVENT_POINTS);
        capped.clear();
        assert!(capped.events().is_empty());
        assert!(!capped.truncated());
    }

    #[test]
    fn collision_score_selects_repeatable_bounded_multi_law_states() {
        let disabled = CollisionScoreParams::default();
        assert_eq!(
            collision_score_memory_law(
                disabled,
                CollisionScoreState {
                    state_index: 9,
                    event_ordinal: u64::MAX,
                },
            ),
            CollisionScoreMemoryLaw::default()
        );

        let params = CollisionScoreParams {
            enabled: true,
            seed: 0x5eed_c011,
            state_count: 5,
            ..CollisionScoreParams::default()
        };
        let first = collision_score_memory_law(
            params,
            CollisionScoreState {
                state_index: 2,
                event_ordinal: 1,
            },
        );
        let repeated = collision_score_memory_law(
            params,
            CollisionScoreState {
                state_index: 7,
                event_ordinal: u64::MAX,
            },
        );
        let next = collision_score_memory_law(
            params,
            CollisionScoreState {
                state_index: 3,
                event_ordinal: 2,
            },
        );
        assert_eq!(first, repeated, "the finite state table wraps exactly");
        assert_ne!(first, next, "adjacent states select distinct bounded laws");
        for law in [first, next] {
            assert!((0.5..=1.0).contains(&law.loom_depth_scale));
            assert!((-0.5..=0.5).contains(&law.loom_phase_offset));
            assert!((-0.25..=0.25).contains(&law.atlas_collision_offset));
            assert!((-0.25..=0.25).contains(&law.garden_threshold_offset));
            assert!((0.75..=1.0).contains(&law.garden_decay_scale));
        }

        let authored = TemporalOriginalsParams {
            loom: TemporalLoomParams {
                amount: 1.0,
                depth: 1.0,
                ..TemporalLoomParams::default()
            },
            atlas: CollisionAtlasParams {
                amount: 1.0,
                collision: 0.5,
                ..CollisionAtlasParams::default()
            },
            garden: RefreshGardenParams {
                amount: 1.0,
                threshold: 0.5,
                decay: 1.0,
                ..RefreshGardenParams::default()
            },
            score: params,
            ..TemporalOriginalsParams::default()
        };
        let effective = score_effective_originals(
            authored,
            CollisionScoreState {
                state_index: 2,
                event_ordinal: 999,
            },
        );
        assert_eq!(effective.loom.depth, first.loom_depth_scale);
        assert_eq!(effective.loom.phase, first.loom_phase_offset);
        assert_eq!(
            effective.atlas.collision,
            (0.5 + first.atlas_collision_offset).clamp(0.0, 1.0)
        );
        assert_eq!(
            effective.garden.threshold,
            (0.5 + first.garden_threshold_offset).clamp(0.0, 1.0)
        );
        assert_eq!(effective.garden.decay, first.garden_decay_scale);
    }

    #[test]
    fn explicit_garden_refresh_is_counted_freeze_safe_and_transactional() {
        let params = TemporalParams {
            originals: TemporalOriginalsParams {
                garden: RefreshGardenParams {
                    amount: 0.75,
                    max_hold_ticks: 0,
                    ..RefreshGardenParams::default()
                },
                ..TemporalOriginalsParams::default()
            },
            ..TemporalParams::default()
        };
        let events = TemporalFrameEvents {
            garden_refresh_events: 3,
            ..TemporalFrameEvents::default()
        };
        let mut state = TemporalState::default();
        let frozen = state.stage_frame(
            &params,
            TemporalFrameInput::new(
                1.0 / 60.0,
                TemporalFreezeState::ProgramFrozen,
                false,
                events,
            ),
            [64, 36],
        );
        assert_eq!(frozen.garden_refresh_events_consumed, 0);
        assert!(!frozen.garden_force_refresh);
        state.commit_staged();

        let advancing = state.stage_frame(
            &params,
            TemporalFrameInput::new(1.0 / 60.0, TemporalFreezeState::Running, false, events),
            [64, 36],
        );
        assert_eq!(advancing.garden_refresh_events_consumed, 3);
        assert!(advancing.garden_force_refresh);
        assert_ne!(
            advancing.originals_uniforms.garden_modes[3] & GARDEN_FORCE_REFRESH_BIT,
            0
        );
        state.discard_staged();
        assert_eq!(state.metrics().total_reference_ticks, 0);

        let replay = state.stage_frame(
            &params,
            TemporalFrameInput::new(1.0 / 60.0, TemporalFreezeState::Running, false, events),
            [64, 36],
        );
        assert_eq!(replay.garden_refresh_events_consumed, 3);
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
                // Keep this pre-Long-Exposure sequence frozen over the
                // established eight Originals lanes. The new ninth lane has
                // its own active/default and live/export parity proofs above,
                // so an additive neutral tail does not rewrite this golden.
                hash.update(&bytemuck::bytes_of(&plan.originals_uniforms)[..128]);
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
                "a1891de9ed1ce055d199298e2f8d7ea898e2a4fc750a7c139b1f1f0490b5a3c3",
                "deab90dbae86952f7a32056d2fe50114f5d83562096bd9aa586e96cffa41dd64",
                "332c187ce0c7be98314b5ad483af48b0d756c1c77deb990ed265e4333359fc5e",
            ]
            .map(str::to_string)
        );
    }

    // -----------------------------------------------------------------
    // B3 feedback rig
    // -----------------------------------------------------------------

    /// A small CPU feedback loop over the legacy variant semantics: rotate/
    /// zoom, the rig coordinate and colour laws, bilinear clamp sampling, and
    /// the frozen max-combine. This is the low-resolution instrument the
    /// regime fixtures measure.
    fn simulate_rig_loop(
        params: &TemporalParams,
        dt: f32,
        frames: usize,
        size: usize,
        current: &dyn Fn(usize, usize) -> [f32; 3],
    ) -> Vec<Vec<[f32; 3]>> {
        let frame_params = params.for_frame_delta(dt);
        let tick_mix = TemporalParams::rig_tick_mix(dt);
        let rig_identity = params.rig.is_identity();
        let sample = |grid: &Vec<Vec<[f32; 3]>>, uv: [f32; 2]| -> [f32; 3] {
            let n = size as f32;
            let x = (uv[0] * n - 0.5).clamp(0.0, n - 1.0);
            let y = (uv[1] * n - 0.5).clamp(0.0, n - 1.0);
            let x0 = x.floor() as usize;
            let y0 = y.floor() as usize;
            let x1 = (x0 + 1).min(size - 1);
            let y1 = (y0 + 1).min(size - 1);
            let fx = x - x0 as f32;
            let fy = y - y0 as f32;
            let mut out = [0.0_f32; 3];
            for c in 0..3 {
                let top = grid[y0][x0][c] * (1.0 - fx) + grid[y0][x1][c] * fx;
                let bottom = grid[y1][x0][c] * (1.0 - fx) + grid[y1][x1][c] * fx;
                out[c] = top * (1.0 - fy) + bottom * fy;
            }
            out
        };
        let mut previous = vec![vec![[0.0_f32; 3]; size]; size];
        let mut feedback_valid = false;
        for frame in 0..frames {
            let mut next = vec![vec![[0.0_f32; 3]; size]; size];
            for (y, row) in next.iter_mut().enumerate() {
                for (x, cell) in row.iter_mut().enumerate() {
                    let mut color = current(x, y);
                    if frame_params.feedback > 0.001 && feedback_valid {
                        let uv = [
                            (x as f32 + 0.5) / size as f32,
                            (y as f32 + 0.5) / size as f32,
                        ];
                        let angle = frame_params.fb_rotate * 0.017_453_3;
                        let (sin, cos) = (angle.sin(), angle.cos());
                        let centered = [uv[0] - 0.5, uv[1] - 0.5];
                        let zoom = frame_params.fb_zoom.max(0.01);
                        let p = [
                            (centered[0] * cos - centered[1] * sin) / zoom,
                            (centered[0] * sin + centered[1] * cos) / zoom,
                        ];
                        let (resolved, inside, graded) = if rig_identity {
                            let q = [p[0] + 0.5, p[1] + 0.5];
                            let inside = if q[0] >= 0.0 && q[0] <= 1.0 && q[1] >= 0.0 && q[1] <= 1.0
                            {
                                1.0
                            } else {
                                0.0
                            };
                            let clamped = [q[0].clamp(0.0, 1.0), q[1].clamp(0.0, 1.0)];
                            let prev = sample(&previous, clamped);
                            (clamped, inside, prev)
                        } else {
                            let (resolved, inside) = feedback_rig_resolve(p, &frame_params.rig);
                            let prev = sample(&previous, resolved);
                            let frag_px =
                                [(uv[0] * size as f32) as u32, (uv[1] * size as f32) as u32];
                            let graded = feedback_rig_grade(
                                prev,
                                &frame_params.rig,
                                tick_mix,
                                frag_px,
                                frame as u32,
                            );
                            (resolved, inside, graded)
                        };
                        let _ = resolved;
                        for c in 0..3 {
                            color[c] = color[c].max(graded[c] * frame_params.feedback * inside);
                        }
                    }
                    *cell = color;
                }
            }
            previous = next;
            feedback_valid = true;
        }
        previous
    }

    fn impulse_current(_size: usize, at: (usize, usize)) -> impl Fn(usize, usize) -> [f32; 3] {
        move |x, y| {
            if (x, y) == at {
                [1.0, 1.0, 1.0]
            } else {
                [0.0, 0.0, 0.0]
            }
        }
    }

    fn luma_at(grid: &[Vec<[f32; 3]>], x: usize, y: usize) -> f32 {
        let c = grid[y][x];
        0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    }

    #[test]
    fn whole_fraction_rotation_locks_into_four_arms_and_detune_shears_them() {
        // Authored rotation is 5 degrees per reference tick; eighteen ticks
        // per simulated frame make each application a quarter turn, so an
        // impulse fed back four times returns to itself: four locked arms.
        let size = 41;
        let impulse = (30_usize, 20_usize);
        let params = TemporalParams {
            feedback: 0.95,
            fb_rotate: 5.0,
            ..TemporalParams::default()
        };
        let locked = simulate_rig_loop(
            &params,
            18.0 / TEMPORAL_REFERENCE_FPS,
            8,
            size,
            &impulse_current(size, impulse),
        );
        // The impulse sits at +x from centre (20, 20); each fed-back hop
        // rotates it a quarter turn and decays it by feedback^18, so the
        // three ghost arms carry the analytic retention powers (bilinear
        // resampling loses a little more, hence the 0.3 factor).
        let retention = 0.95_f32.powf(18.0);
        let arms = [(20, 10), (10, 20), (20, 30)];
        for (hop, &(x, y)) in arms.iter().enumerate() {
            let expected = retention.powi(hop as i32 + 1);
            assert!(
                luma_at(&locked, x, y) > expected * 0.3,
                "locked hop {hop} at ({x}, {y}) = {} (expected near {expected})",
                luma_at(&locked, x, y)
            );
        }
        // Detuned to eighty degrees per application, the quarter-turn
        // positions no longer coincide with the trail: the arms shear away.
        let detuned = simulate_rig_loop(
            &params,
            16.0 / TEMPORAL_REFERENCE_FPS,
            8,
            size,
            &impulse_current(size, impulse),
        );
        let locked_energy: f32 = arms.iter().map(|&(x, y)| luma_at(&locked, x, y)).sum();
        let detuned_energy: f32 = arms.iter().map(|&(x, y)| luma_at(&detuned, x, y)).sum();
        assert!(
            detuned_energy < locked_energy * 0.25,
            "detuned arms must shear off the lock: locked {locked_energy}, detuned {detuned_energy}"
        );
    }

    #[test]
    fn reflection_is_the_alternating_regime_no_rotation_reaches() {
        let size = 41;
        let impulse = (30_usize, 20_usize);
        let mirrored = (size - 1 - 30, 20_usize);
        let mut params = TemporalParams {
            feedback: 0.9,
            ..TemporalParams::default()
        };
        params.rig.reflect_x = true;
        let reflected = simulate_rig_loop(
            &params,
            1.0 / TEMPORAL_REFERENCE_FPS,
            3,
            size,
            &impulse_current(size, impulse),
        );
        assert!(
            luma_at(&reflected, mirrored.0, mirrored.1) > 0.5,
            "the mirrored ghost must appear"
        );
        // Without the reflection nothing ever reaches the mirrored position.
        let plain = simulate_rig_loop(
            &TemporalParams {
                feedback: 0.9,
                ..TemporalParams::default()
            },
            1.0 / TEMPORAL_REFERENCE_FPS,
            3,
            size,
            &impulse_current(size, impulse),
        );
        assert!(luma_at(&plain, mirrored.0, mirrored.1) < 1.0e-3);
    }

    #[test]
    fn the_servo_bounds_a_hot_loop_and_defeat_lets_it_run_to_white_and_stay() {
        let size = 9;
        let gray = |_: usize, _: usize| [0.5_f32, 0.5, 0.5];
        let mut hot = TemporalParams {
            feedback: 0.95,
            ..TemporalParams::default()
        };
        hot.rig.gain_r = 2.0;
        hot.rig.gain_g = 2.0;
        hot.rig.gain_b = 2.0;
        hot.rig.servo = true;
        let dt = 1.0 / TEMPORAL_REFERENCE_FPS;
        let served = simulate_rig_loop(&hot, dt, 40, size, &gray);
        let served_luma = luma_at(&served, 4, 4);
        assert!(
            served_luma < 2.5,
            "the engaged servo must bound the loop, got {served_luma}"
        );
        // Defeated, the same loop runs away and never comes back down.
        hot.rig.servo_defeated = true;
        let defeated_short = simulate_rig_loop(&hot, dt, 20, size, &gray);
        let defeated_long = simulate_rig_loop(&hot, dt, 40, size, &gray);
        let short_luma = luma_at(&defeated_short, 4, 4);
        let long_luma = luma_at(&defeated_long, 4, 4);
        assert!(
            long_luma > served_luma * 10.0,
            "defeat must let the loop exceed the served level: {long_luma}"
        );
        assert!(
            long_luma >= short_luma,
            "a defeated runaway stays where it ran: {short_luma} -> {long_luma}"
        );
    }

    #[test]
    fn rig_uniform_lanes_carry_the_authored_activity_and_servo_laws() {
        use crate::effects::params::FeedbackShape;
        assert_eq!(std::mem::size_of::<TemporalRigGpuUniforms>(), 96);
        // Authored identity keeps the activity flag closed even though the
        // frame-scaled gains transiently read 1 at dt 0.
        let identity = TemporalRigGpuUniforms::for_frame(
            FeedbackRigParams::default().for_frame_scale(0.0),
            true,
            0.0,
            77,
        );
        assert_eq!(identity.modes_b[1], 0);
        assert_eq!(identity.modes_b[0], 77);
        let mut authored = FeedbackRigParams {
            reflect_y: true,
            hue_rotate: 90.0,
            shape: FeedbackShape::Fold,
            edge: crate::motion::MotionBoundaryMode::Mirror,
            servo: true,
            ..FeedbackRigParams::default()
        };
        let active = TemporalRigGpuUniforms::for_frame(
            authored.for_frame_scale(1.0),
            authored.is_identity(),
            1.0,
            (1_u64 << 40) | 5,
        );
        assert_eq!(active.modes_b[1], 1);
        // The epoch lane is the low 32 bits of the reference tick.
        assert_eq!(active.modes_b[0], 5);
        assert_eq!(active.modes_a, [0, 1, 3, 1]);
        assert!((active.values_a[2] - std::f32::consts::FRAC_PI_2).abs() < 1.0e-6);
        assert_eq!(active.values_d[3], 1.0);
        // Defeat wins over engage.
        authored.servo_defeated = true;
        let defeated = TemporalRigGpuUniforms::for_frame(
            authored.for_frame_scale(1.0),
            authored.is_identity(),
            1.0,
            0,
        );
        assert_eq!(defeated.values_d[3], 0.0);
    }

    #[test]
    fn rig_reference_edge_laws_share_the_frozen_boundary_numbering() {
        use crate::motion::MotionBoundaryMode;
        let rig_with_edge = |edge: MotionBoundaryMode| FeedbackRigParams {
            edge,
            offset_x: 0.4,
            ..FeedbackRigParams::default()
        };
        // Outside the unit square: Transparent removes coverage, the other
        // three always cover, exactly the program-wide boundary table.
        let outside = [0.4_f32, 0.0];
        let (_, transparent) =
            feedback_rig_resolve(outside, &rig_with_edge(MotionBoundaryMode::Transparent));
        assert_eq!(transparent, 0.0);
        for edge in [
            MotionBoundaryMode::Mirror,
            MotionBoundaryMode::Wrap,
            MotionBoundaryMode::Hold,
        ] {
            let (resolved, coverage) = feedback_rig_resolve(outside, &rig_with_edge(edge));
            assert_eq!(coverage, 1.0, "{edge:?}");
            assert!((0.0..=1.0).contains(&resolved[0]));
        }
        // Hold clamps, Wrap wraps, Mirror reflects.
        let (hold, _) = feedback_rig_resolve(outside, &rig_with_edge(MotionBoundaryMode::Hold));
        assert_eq!(hold[0], 1.0);
        let (wrap, _) = feedback_rig_resolve(outside, &rig_with_edge(MotionBoundaryMode::Wrap));
        assert!((wrap[0] - 0.3).abs() < 1.0e-6);
        let (mirror, _) = feedback_rig_resolve(outside, &rig_with_edge(MotionBoundaryMode::Mirror));
        assert!((mirror[0] - 0.7).abs() < 1.0e-6);
    }
}
