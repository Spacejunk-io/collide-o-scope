//! Modulation matrix: internal LFO sources routed to effect parameters.
//!
//! This is the engine's autonomous heartbeat. Each frame the render loop
//! calls `update` (advancing the beat clock and sampling every LFO), then
//! `modulate` to produce *modulated copies* of the master effects and NTSC
//! params. Base values — what the UI sliders edit — are never mutated, so
//! manual control and modulation compose: the slider sets the center, the
//! LFO breathes around it.
//!
//! LFO rates are expressed in beats (quarter notes), synced to a BPM clock
//! driven by tap tempo. Every source added later (audio transients, MIDI
//! CCs, gyroscope axes) should enter through this same matrix: a source
//! produces a value in [-1, 1], a routing scales it into a parameter range.

use std::borrow::Cow;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::audio::{AudioBandConfig, AudioLevels, MAX_AUDIO_BANDS};
use crate::effects::params::TemporalParams;
use crate::effects::EffectUniforms;
use crate::ntsc::NtscParams;

pub const NUM_LFOS: usize = 4;
pub const MAX_ROUTINGS: usize = 64;
/// Maximum live stack size advertised by the modulation matrix and panel.
/// Keeping this explicit prevents a layer from existing without routable
/// opacity, transport, key, and effect controls.
pub const MAX_MOD_LAYERS: usize = 16;
pub const AUDIO_SOURCE_LIVE: &str = "live";
pub const AUDIO_SOURCE_FILE: &str = "file";

pub fn normalize_audio_source_kind(value: &str) -> &'static str {
    if value.eq_ignore_ascii_case(AUDIO_SOURCE_FILE) {
        AUDIO_SOURCE_FILE
    } else {
        AUDIO_SOURCE_LIVE
    }
}

/// Modulation targets: (key, min, max). Depth ±1.0 spans half the range
/// in each direction from the base value, clamped to [min, max].
pub const TARGETS: &[(&str, f32, f32)] = &[
    ("pixelate", 1.0, 32.0),
    ("rgb_split", 0.0, 30.0),
    ("hue_shift", -180.0, 180.0),
    ("saturation", -1.0, 1.0),
    ("brightness", -1.0, 1.0),
    ("contrast", -1.0, 1.0),
    ("posterize", 0.0, 16.0),
    ("grain_intensity", 0.0, 0.3),
    ("grain_size", 1.0, 4.0),
    ("vignette", 0.0, 1.5),
    ("color_drift", 0.0, 0.02),
    ("downsample", 0.05, 1.0),
    ("breathe_scale", 0.0, 0.05),
    ("breathe_rotation", 0.0, 2.0),
    ("breathe_position", 0.0, 0.02),
    ("key_threshold", 0.0, 1.0),
    ("key_softness", 0.0, 0.5),
    ("key_color_r", 0.0, 1.0),
    ("key_color_g", 0.0, 1.0),
    ("key_color_b", 0.0, 1.0),
    ("key_tolerance", 0.0, 1.0),
    ("cellular_amount", 0.0, 1.0),
    ("cellular_scale", 2.0, 32.0),
    ("cellular_warp", 0.0, 1.0),
    ("cellular_speed", 0.0, 2.0),
    ("cellular_gap_amount", 0.0, 1.0),
    ("cellular_gap_threshold", 0.0, 1.0),
    ("cellular_gap_softness", 0.0, 0.5),
    ("ntsc_snow", 0.0, 1.0),
    ("ntsc_tracking_snow", 0.0, 1.0),
    ("ntsc_edge_wave", 0.0, 20.0),
    ("ntsc_edge_wave_speed", 0.0, 10.0),
    ("ntsc_head_shift", -100.0, 100.0),
    ("ntsc_tracking_wave", 0.0, 50.0),
    ("ntsc_chroma_loss", 0.0, 0.01),
    ("ntsc_composite_noise", 0.0, 0.5),
    ("ntsc_luma_noise", 0.0, 0.2),
    ("ntsc_chroma_noise", 0.0, 0.5),
    ("ntsc_luma_smear", 0.0, 1.0),
    ("ntsc_sharpening", -1.0, 2.0),
    ("temporal_feedback", 0.0, 0.95),
    ("temporal_slitscan", 0.0, 1.0),
    ("temporal_fb_zoom", 0.9, 1.1),
    ("temporal_fb_rotate", -5.0, 5.0),
    ("temporal_slit_angle", -180.0, 180.0),
    ("temporal_key_threshold", 0.0, 1.0),
    ("temporal_key_softness", 0.0, 0.5),
    ("temporal_key_history", 1.0, 23.0),
    // The patch-morph crossfader; applied by the app, not apply_offset.
    ("morph", 0.0, 1.0),
];
const MORPH_TARGET_INDEX: usize = TARGETS.len() - 1;

/// Canonicalize the only retired target spelling. Runtime/UI output always
/// uses `layerN_key_threshold`; old patch files remain loadable.
pub fn canonical_target(target: &str) -> Cow<'_, str> {
    let Some(rest) = target.strip_prefix("layer") else {
        return Cow::Borrowed(target);
    };
    let Some((number, suffix)) = rest.split_once('_') else {
        return Cow::Borrowed(target);
    };
    if suffix != "key" {
        return Cow::Borrowed(target);
    }
    let Ok(layer) = number.parse::<usize>() else {
        return Cow::Borrowed(target);
    };
    if !(1..=MAX_MOD_LAYERS).contains(&layer) {
        return Cow::Borrowed(target);
    }
    Cow::Owned(format!("layer{layer}_key_threshold"))
}

/// Resolve a target's legal value range, including dynamically named layer
/// targets up to [`MAX_MOD_LAYERS`].
pub fn target_range(target: &str) -> Option<(f32, f32)> {
    let target = canonical_target(target);
    let target = target.as_ref();
    if let Some((_, min, max)) = TARGETS.iter().find(|(key, _, _)| *key == target) {
        return Some((*min, *max));
    }

    let rest = target.strip_prefix("layer")?;
    let (number, suffix) = rest.split_once('_')?;
    let layer = number.parse::<usize>().ok()?;
    if !(1..=MAX_MOD_LAYERS).contains(&layer) {
        return None;
    }
    match suffix {
        "opacity" | "key_threshold" => Some((0.0, 1.0)),
        "speed" => Some((0.25, 4.0)),
        "fps" => Some((1.0, 240.0)),
        "pixelate" => Some((1.0, 32.0)),
        "rgb_split" => Some((0.0, 30.0)),
        "hue_shift" => Some((-180.0, 180.0)),
        "saturation" | "brightness" | "contrast" => Some((-1.0, 1.0)),
        "posterize" => Some((0.0, 16.0)),
        "grain_intensity" => Some((0.0, 0.3)),
        "grain_size" => Some((1.0, 4.0)),
        "vignette" => Some((0.0, 1.5)),
        "color_drift" => Some((0.0, 0.02)),
        "breathe_scale" => Some((0.0, 0.05)),
        "breathe_rotation" => Some((0.0, 2.0)),
        "breathe_position" => Some((0.0, 0.02)),
        "key_softness" => Some((0.0, 0.5)),
        "key_color_r" | "key_color_g" | "key_color_b" | "key_tolerance" => Some((0.0, 1.0)),
        "downsample" => Some((0.05, 1.0)),
        "cellular_amount" => Some((0.0, 1.0)),
        "cellular_scale" => Some((2.0, 32.0)),
        "cellular_warp" => Some((0.0, 1.0)),
        "cellular_speed" => Some((0.0, 2.0)),
        "cellular_gap_amount" => Some((0.0, 1.0)),
        "cellular_gap_threshold" => Some((0.0, 1.0)),
        "cellular_gap_softness" => Some((0.0, 0.5)),
        _ => None,
    }
}

pub fn is_valid_target(target: &str) -> bool {
    target_range(target).is_some()
}

/// Per-layer values after modulation, aligned with the layers vec.
#[derive(Debug, Clone, Copy)]
pub struct LayerModulation {
    pub opacity: f32,
    pub speed: f32,
    pub fps: f32,
    pub effects: EffectUniforms,
}

/// One-frame modulation cache. Every route is accumulated exactly once; all
/// master, morph, and layer consumers then read fixed-size indexed storage.
/// Keeping this frame-local makes route edits immediately authoritative while
/// avoiding repeated scans and target parsing in both live and export paths.
pub struct ModulationFrame {
    offsets: RoutingOffsets,
}

impl ModulationFrame {
    /// Offset for the program morph crossfader. Its compiled slot avoids a
    /// target-name scan on every live and offline frame.
    pub fn morph_offset(&self) -> f32 {
        let (_, min, max) = TARGETS[MORPH_TARGET_INDEX];
        self.offsets.master[MORPH_TARGET_INDEX] * (max - min) * 0.5
    }

    pub fn modulate(
        &self,
        effects: &EffectUniforms,
        ntsc: &NtscParams,
        temporal: &TemporalParams,
    ) -> (EffectUniforms, NtscParams, TemporalParams) {
        ModMatrix::modulate_from_offsets(effects, ntsc, temporal, &self.offsets)
    }

    pub fn modulate_layers<'a>(
        &self,
        layers: impl IntoIterator<Item = (&'a EffectUniforms, f32, f32, f32)>,
    ) -> Vec<LayerModulation> {
        layers
            .into_iter()
            .enumerate()
            .map(|(index, (effects, opacity, speed, fps))| {
                ModMatrix::modulate_layer_from_offsets(
                    index,
                    effects,
                    opacity,
                    speed,
                    fps,
                    &self.offsets,
                )
            })
            .collect()
    }
}

/// Parsed destination kept beside a route so the render path never reparses
/// `layerN_*` strings or allocates formatted target names at frame rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompiledTarget {
    Master(usize),
    Layer { index: usize, suffix: usize },
    Invalid,
}

fn master_target_index(target: &str) -> Option<usize> {
    TARGETS.iter().position(|(key, _, _)| *key == target)
}

fn compile_target(target: &str) -> CompiledTarget {
    let target = canonical_target(target);
    if let Some(index) = master_target_index(target.as_ref()) {
        return CompiledTarget::Master(index);
    }
    let Some((layer, suffix)) = parse_layer_target(target.as_ref()) else {
        return CompiledTarget::Invalid;
    };
    layer_suffix_index(suffix).map_or(CompiledTarget::Invalid, |suffix| CompiledTarget::Layer {
        index: layer,
        suffix,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LfoShape {
    Sine,
    Triangle,
    Saw,
    Square,
    SampleHold,
}

impl LfoShape {
    pub fn from_str(s: &str) -> Self {
        match s {
            "triangle" => Self::Triangle,
            "saw" => Self::Saw,
            "square" => Self::Square,
            "sample_hold" => Self::SampleHold,
            _ => Self::Sine,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Sine => "sine",
            Self::Triangle => "triangle",
            Self::Saw => "saw",
            Self::Square => "square",
            Self::SampleHold => "sample_hold",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Lfo {
    pub shape: LfoShape,
    /// Cycle length in beats (quarter notes): 4.0 = one cycle per 4/4 bar.
    pub beats: f32,
    /// Phase offset, 0..1 of a cycle.
    pub phase: f32,
}

impl Default for Lfo {
    fn default() -> Self {
        Self {
            shape: LfoShape::Sine,
            beats: 4.0,
            phase: 0.0,
        }
    }
}

impl Lfo {
    /// Set a finite phase offset and wrap it into one cycle.
    /// Non-finite external values safely reset to the cycle origin.
    pub fn set_phase(&mut self, phase: f32) {
        self.phase = finite_or(phase, 0.0).rem_euclid(1.0);
    }

    /// Finite, wrapped phase for snapshots and other read-only consumers.
    pub fn normalized_phase(&self) -> f32 {
        finite_or(self.phase, 0.0).rem_euclid(1.0)
    }

    /// Bipolar output in [-1, 1] at the given global beat position.
    pub fn value(&self, beat: f64, lfo_index: usize) -> f32 {
        let beats = self.beats.max(0.0625) as f64;
        // Keep the sampler finite even if a caller bypasses `set_phase` and
        // writes the public compatibility field directly.
        let phase = self.normalized_phase() as f64;
        let cycles = beat / beats + phase;
        let p = cycles.rem_euclid(1.0) as f32;
        match self.shape {
            LfoShape::Sine => (p * std::f32::consts::TAU).sin(),
            LfoShape::Triangle => 1.0 - 4.0 * (p - 0.5).abs(),
            LfoShape::Saw => 2.0 * p - 1.0,
            LfoShape::Square => {
                if p < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
            LfoShape::SampleHold => {
                // Deterministic pseudo-random value held for each full cycle.
                let cycle = cycles.floor() as i64 as u64;
                let mut h = cycle
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(lfo_index as u64 + 1);
                h ^= h >> 33;
                h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
                h ^= h >> 33;
                (h as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32
            }
        }
    }
}

/// Number of assignable MIDI CC slots (A–D in the UI).
pub const NUM_MIDI_SLOTS: usize = 4;

/// A modulation source: an internal LFO, an audio analysis band, or a
/// MIDI CC slot. LFOs are bipolar [-1, 1]; audio and MIDI are unipolar [0, 1].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModSource {
    Lfo(usize),
    AudioLevel,
    AudioBass,
    AudioMid,
    AudioHigh,
    /// One of the configurable analysis bands, zero-indexed internally and
    /// serialized as `audio_band1` through `audio_band8`.
    AudioBand(usize),
    AudioOnset,
    AudioBright,
    AudioNoise,
    Midi(usize),
    GyroYaw,
    GyroPitch,
    GyroRoll,
    PadX,
    PadY,
}

impl ModSource {
    pub fn try_from_str(s: &str) -> Option<Self> {
        Some(match s {
            "lfo0" => Self::Lfo(0),
            "lfo1" => Self::Lfo(1),
            "lfo2" => Self::Lfo(2),
            "lfo3" => Self::Lfo(3),
            "audio_level" => Self::AudioLevel,
            "audio_bass" => Self::AudioBass,
            "audio_mid" => Self::AudioMid,
            "audio_high" => Self::AudioHigh,
            "audio_band1" => Self::AudioBand(0),
            "audio_band2" => Self::AudioBand(1),
            "audio_band3" => Self::AudioBand(2),
            "audio_band4" => Self::AudioBand(3),
            "audio_band5" => Self::AudioBand(4),
            "audio_band6" => Self::AudioBand(5),
            "audio_band7" => Self::AudioBand(6),
            "audio_band8" => Self::AudioBand(7),
            "audio_onset" => Self::AudioOnset,
            "audio_bright" => Self::AudioBright,
            "audio_noise" => Self::AudioNoise,
            "midi_a" => Self::Midi(0),
            "midi_b" => Self::Midi(1),
            "midi_c" => Self::Midi(2),
            "midi_d" => Self::Midi(3),
            "gyro_yaw" => Self::GyroYaw,
            "gyro_pitch" => Self::GyroPitch,
            "gyro_roll" => Self::GyroRoll,
            "pad_x" => Self::PadX,
            "pad_y" => Self::PadY,
            _ => return None,
        })
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Lfo(0) => "lfo0",
            Self::Lfo(1) => "lfo1",
            Self::Lfo(2) => "lfo2",
            Self::Lfo(_) => "lfo3",
            Self::AudioLevel => "audio_level",
            Self::AudioBass => "audio_bass",
            Self::AudioMid => "audio_mid",
            Self::AudioHigh => "audio_high",
            Self::AudioBand(0) => "audio_band1",
            Self::AudioBand(1) => "audio_band2",
            Self::AudioBand(2) => "audio_band3",
            Self::AudioBand(3) => "audio_band4",
            Self::AudioBand(4) => "audio_band5",
            Self::AudioBand(5) => "audio_band6",
            Self::AudioBand(6) => "audio_band7",
            Self::AudioBand(_) => "audio_band8",
            Self::AudioOnset => "audio_onset",
            Self::AudioBright => "audio_bright",
            Self::AudioNoise => "audio_noise",
            Self::Midi(0) => "midi_a",
            Self::Midi(1) => "midi_b",
            Self::Midi(2) => "midi_c",
            Self::Midi(_) => "midi_d",
            Self::GyroYaw => "gyro_yaw",
            Self::GyroPitch => "gyro_pitch",
            Self::GyroRoll => "gyro_roll",
            Self::PadX => "pad_x",
            Self::PadY => "pad_y",
        }
    }
}

/// Response shape applied to a source before routing depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
// `SCurve` is part of the persisted patch and web-control vocabulary.
#[allow(clippy::enum_variant_names)]
pub enum Curve {
    Linear,
    Exp,
    Log,
    SCurve,
    Steps,
}

impl Curve {
    pub fn from_str(value: &str) -> Self {
        match value {
            "exp" => Self::Exp,
            "log" => Self::Log,
            "s_curve" => Self::SCurve,
            "steps" => Self::Steps,
            _ => Self::Linear,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::SCurve => "s_curve",
            Self::Steps => "steps",
        }
    }
}

/// Runtime and persisted configuration for one gyroscope axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GyroAxisConfig {
    /// Orientation which maps to the centered value 0.5.
    pub center_degrees: f32,
    /// Absolute degrees from center which reach either end of the range.
    pub range_degrees: f32,
    /// Centered exponent control: k = 2^expo.
    pub expo: f32,
    pub invert: bool,
}

impl GyroAxisConfig {
    fn with_range(range_degrees: f32) -> Self {
        Self {
            center_degrees: 0.0,
            range_degrees,
            expo: 0.0,
            invert: false,
        }
    }
}

/// Runtime and persisted shaping for one XY-pad axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PadAxisConfig {
    pub curve: Curve,
    pub curve_amount: f32,
    /// Number of evenly spaced positions, including both 0 and 1. Values are
    /// snapped to the nearest position; zero or one disables quantization.
    pub quantize: u32,
}

impl Default for PadAxisConfig {
    fn default() -> Self {
        Self {
            curve: Curve::Linear,
            curve_amount: 0.0,
            quantize: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PadConfig {
    pub axes: [PadAxisConfig; 2],
    pub spring_enabled: bool,
    /// Exponential return rate in inverse seconds.
    pub spring_rate: f32,
}

impl Default for PadConfig {
    fn default() -> Self {
        Self {
            axes: [PadAxisConfig::default(); 2],
            spring_enabled: false,
            spring_rate: 4.0,
        }
    }
}

/// A single modulation routing: source → parameter, shaped and scaled by depth.
#[derive(Debug, Clone)]
pub struct Routing {
    id: u64,
    pub source: ModSource,
    target: String,
    pub depth: f32,
    pub curve: Curve,
    pub curve_amount: f32,
    /// Seconds to follow a rising source. Zero is instantaneous.
    pub attack: f32,
    /// Seconds to follow a falling source. Zero is instantaneous.
    pub release: f32,
    compiled_target: CompiledTarget,
    state: f32,
    cached: f32,
}

static NEXT_ROUTING_ID: AtomicU64 = AtomicU64::new(1);

fn allocate_routing_id() -> u64 {
    NEXT_ROUTING_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .expect("modulation routing id space exhausted")
}

impl Routing {
    pub fn new(source: ModSource, target: impl Into<String>, depth: f32) -> Self {
        let target = target.into();
        let target = canonical_target(&target).into_owned();
        Self {
            id: allocate_routing_id(),
            source,
            compiled_target: compile_target(&target),
            target,
            depth,
            curve: Curve::Linear,
            curve_amount: 0.0,
            attack: 0.0,
            release: 0.0,
            state: 0.0,
            cached: 0.0,
        }
    }

    pub fn route_id(&self) -> u64 {
        self.id
    }

    pub fn cached_value(&self) -> f32 {
        finite_or(self.cached, 0.0).clamp(-1.0, 1.0)
    }

    pub fn target(&self) -> &str {
        &self.target
    }

    /// Clear transient state after replacing a route's identity/configuration.
    pub fn reset_runtime(&mut self) {
        self.state = 0.0;
        self.cached = 0.0;
    }

    /// Change a semantic destination atomically and clear transient response
    /// state. Source/target changes define a new signal; response-time edits do
    /// not and therefore deliberately preserve continuity.
    pub fn set_target(&mut self, target: impl Into<String>) -> bool {
        let target = target.into();
        let target = canonical_target(&target).into_owned();
        if self.target == target {
            return false;
        }
        self.compiled_target = compile_target(&target);
        self.target = target;
        self.reset_runtime();
        true
    }

    fn advance(&mut self, desired: f32, dt: f32) {
        let tau = if desired >= self.state {
            self.attack
        } else {
            self.release
        };
        self.state = exponential_follow(self.state, desired, dt, tau);
        self.cached = self.state;
    }
}

/// Beat clock with tap tempo. Tapping re-anchors the downbeat, so the
/// performer's taps both set the tempo and align the LFO phase to it.
pub struct Clock {
    pub bpm: f32,
    anchor: Instant,
    taps: Vec<Instant>,
    /// When Some, an external MIDI clock owns the beat position; the
    /// internal anchor-based clock is bypassed until it goes away.
    external_beat: Option<f64>,
    /// Keeps the public beat continuous when a paused program resumes while
    /// an external MIDI clock has continued advancing in the background.
    external_offset: f64,
    /// A frozen logical beat while the master program transport is paused.
    /// Incoming MIDI clock telemetry may continue updating `external_beat`,
    /// but it cannot move the rendered modulation phase until resume.
    paused_beat: Option<f64>,
}

const TAP_TIMEOUT: Duration = Duration::from_secs(2);
const MAX_TAPS: usize = 8;

impl Clock {
    pub fn new() -> Self {
        Self {
            bpm: 120.0,
            anchor: Instant::now(),
            taps: Vec::new(),
            external_beat: None,
            external_offset: 0.0,
            paused_beat: None,
        }
    }

    /// Global beat position (quarter notes since the last downbeat anchor),
    /// or the external MIDI clock's position when one is driving.
    pub fn beat(&self, now: Instant) -> f64 {
        if let Some(beat) = self.paused_beat {
            return beat;
        }
        match self.external_beat {
            Some(beat) => beat + self.external_offset,
            None => self.internal_beat(now),
        }
    }

    fn internal_beat(&self, now: Instant) -> f64 {
        now.saturating_duration_since(self.anchor).as_secs_f64() * (self.bpm as f64 / 60.0)
    }

    fn anchor_internal_at(&mut self, beat: f64, now: Instant) {
        let elapsed = finite_f64_or(beat, 0.0).max(0.0) / (self.bpm as f64 / 60.0);
        self.anchor = now
            .checked_sub(Duration::from_secs_f64(elapsed))
            .unwrap_or(now);
    }

    /// Freeze or resume the logical beat without accumulating wall-clock or
    /// external MIDI catch-up. This is idempotent so repeated absolute web
    /// transport commands cannot perturb phase.
    pub fn set_paused(&mut self, paused: bool, now: Instant) {
        match (paused, self.paused_beat) {
            (true, None) => self.paused_beat = Some(self.beat(now)),
            (false, Some(frozen)) => {
                self.paused_beat = None;
                if let Some(raw_external) = self.external_beat {
                    self.external_offset = frozen - raw_external;
                } else {
                    self.anchor_internal_at(frozen, now);
                }
            }
            _ => {}
        }
    }

    #[cfg(test)]
    pub fn is_paused(&self) -> bool {
        self.paused_beat.is_some()
    }

    /// Hand the beat position to (or take it back from) an external clock.
    /// When the external clock disappears, re-anchor so the internal clock
    /// continues from the same position instead of jumping.
    pub fn set_external_beat(&mut self, beat: Option<f64>, now: Instant) {
        if let Some(raw) = beat {
            let raw = finite_f64_or(raw, 0.0).max(0.0);
            if self.external_beat.is_none() {
                // Preserve the legacy handoff (external beat is authoritative)
                // while running. During pause, align the new source underneath
                // the frozen phase so its eventual resume remains continuous.
                self.external_offset = self.paused_beat.map_or(0.0, |frozen| frozen - raw);
            }
            self.external_beat = Some(raw);
            return;
        }

        if let Some(last_raw) = self.external_beat.take() {
            let logical = self.paused_beat.unwrap_or(last_raw + self.external_offset);
            self.anchor_internal_at(logical, now);
        }
        self.external_offset = 0.0;
    }

    pub fn is_external(&self) -> bool {
        self.external_beat.is_some()
    }

    /// Set the internal tempo without moving the current beat position.
    ///
    /// Callers that already sampled a timestamp should use [`Self::set_bpm_at`]
    /// so the tempo change and the surrounding clock work share one instant.
    pub fn set_bpm(&mut self, bpm: f32) {
        self.set_bpm_at(bpm, Instant::now());
    }

    /// Timestamped form of [`Self::set_bpm`]. Changing the slope of an
    /// anchor-based clock must also move its anchor; otherwise elapsed time is
    /// retroactively multiplied by the new BPM and the beat jumps immediately.
    pub fn set_bpm_at(&mut self, bpm: f32, now: Instant) {
        let bpm = finite_or(bpm, self.bpm).clamp(30.0, 300.0);
        if bpm == self.bpm {
            return;
        }

        if self.external_beat.is_none() {
            let beat = self.beat(now);
            self.bpm = bpm;
            self.anchor_internal_at(beat, now);
            return;
        }
        self.bpm = bpm;
    }

    pub fn tap(&mut self, now: Instant) {
        if let Some(&last) = self.taps.last() {
            if now.duration_since(last) > TAP_TIMEOUT {
                self.taps.clear();
            }
        }
        self.taps.push(now);
        if self.taps.len() > MAX_TAPS {
            self.taps.remove(0);
        }
        if self.taps.len() >= 2 {
            let first = self.taps[0];
            let span = now.duration_since(first).as_secs_f64();
            let intervals = (self.taps.len() - 1) as f64;
            let avg = span / intervals;
            if avg > 0.0 {
                self.bpm = ((60.0 / avg) as f32).clamp(30.0, 300.0);
            }
        }
        // Each tap is a downbeat: re-anchor so beat 0 lands on it.
        self.anchor = now;
        if self.paused_beat.is_some() {
            self.paused_beat = Some(0.0);
        }
    }
}

pub struct ModMatrix {
    pub clock: Clock,
    pub lfos: [Lfo; NUM_LFOS],
    pub routings: Vec<Routing>,
    /// Latest sampled LFO values (refreshed by `update`), for UI meters.
    pub lfo_values: [f32; NUM_LFOS],
    /// Beat position at the last update, for the panel's beat light.
    pub current_beat: f64,
    last_update: Option<Instant>,
    /// Latest audio levels (pushed by the app each frame from the analyzer).
    pub audio: AudioLevels,
    /// Whether audio capture should be running (the app syncs the analyzer).
    pub audio_enabled: bool,
    /// Gain applied to normalized audio levels before routing.
    pub audio_gain: f32,
    /// Preferred input device name; empty = system default.
    pub audio_device: String,
    /// `live` captures a CPAL input/system-playback source; `file` analyzes
    /// `audio_clip_path` against the piece-local program clock.
    pub audio_source_kind: String,
    /// Persisted source identity for deterministic, circular file analysis.
    pub audio_clip_path: String,
    /// Validated 3–8-band layout mirrored into `AudioAnalyzer`.
    pub audio_band_config: AudioBandConfig,
    /// Latest MIDI slot values 0..1 (pushed by the app from the MIDI engine).
    pub midi: [f32; NUM_MIDI_SLOTS],
    /// Whether MIDI input should be connected (the app syncs the engine).
    pub midi_enabled: bool,
    /// CC number bound to each slot.
    pub midi_ccs: [u8; NUM_MIDI_SLOTS],
    /// When Some(slot), the next CC message seen binds that slot (MIDI learn).
    pub midi_learn: Option<usize>,
    /// Follow external MIDI timing clock (0xF8) for BPM and beat position.
    pub midi_clock_sync: bool,
    /// Phone orientation [yaw, pitch, roll], each 0..1 (0.5 = level).
    /// Streamed from the web remote; holds the last value received.
    pub gyro: [f32; 3],
    /// Most recent DeviceOrientation degrees [alpha, beta, gamma].
    pub gyro_raw: [f32; 3],
    pub gyro_config: [GyroAxisConfig; 3],
    /// XY performance pad [x, y], each 0..1. Touched from the web remote;
    /// optionally springs toward center after release.
    pub pad: [f32; 2],
    pub pad_active: bool,
    pub pad_config: PadConfig,
}

impl ModMatrix {
    pub fn new() -> Self {
        Self {
            clock: Clock::new(),
            lfos: std::array::from_fn(|_| Lfo::default()),
            routings: Vec::new(),
            lfo_values: [0.0; NUM_LFOS],
            current_beat: 0.0,
            last_update: None,
            audio: AudioLevels::default(),
            audio_enabled: false,
            audio_gain: 1.0,
            audio_device: String::new(),
            audio_source_kind: AUDIO_SOURCE_LIVE.to_string(),
            audio_clip_path: String::new(),
            audio_band_config: AudioBandConfig::default(),
            midi: [0.0; NUM_MIDI_SLOTS],
            midi_enabled: false,
            // CC1 (mod wheel) and the common first knobs on most controllers.
            midi_ccs: [1, 2, 3, 4],
            midi_learn: None,
            midi_clock_sync: false,
            gyro: [0.5; 3],
            gyro_raw: [0.0; 3],
            gyro_config: [
                GyroAxisConfig::with_range(180.0),
                GyroAxisConfig::with_range(90.0),
                GyroAxisConfig::with_range(90.0),
            ],
            pad: [0.5; 2],
            pad_active: false,
            pad_config: PadConfig::default(),
        }
    }

    /// Advance all time-dependent modulation state exactly once per live frame.
    pub fn update(&mut self, now: Instant) {
        let dt = self
            .last_update
            .map(|last| now.saturating_duration_since(last).as_secs_f32())
            .unwrap_or(0.0);
        self.last_update = Some(now);
        self.update_at_beat(self.clock.beat(now), dt);
    }

    /// Forget only the live-frame timestamp used to derive modulation `dt`.
    ///
    /// The next [`Self::update`] then advances with `dt = 0`, preventing time
    /// spent rebuilding a patch from being applied as spring or slew motion.
    /// Beat phase and the most recently published beat remain untouched.
    pub fn reset_update_timing(&mut self) {
        self.last_update = None;
    }

    /// Advance at an explicit beat and time step. The offline exporter passes
    /// `frame_index / fps` and `1 / fps`, making slew and spring motion fully
    /// deterministic and independent of render performance.
    pub fn update_at_beat(&mut self, beat: f64, dt: f32) {
        let dt = finite_or(dt, 0.0).max(0.0);
        self.current_beat = beat;
        for (i, lfo) in self.lfos.iter().enumerate() {
            self.lfo_values[i] = lfo.value(beat, i);
        }

        if self.pad_config.spring_enabled && !self.pad_active {
            let rate = finite_or(self.pad_config.spring_rate, 0.0).max(0.0);
            if rate > 0.0 && dt > 0.0 {
                let alpha = 1.0 - (-rate * dt).exp();
                for value in &mut self.pad {
                    *value += (0.5 - *value) * alpha;
                    *value = (*value).clamp(0.0, 1.0);
                }
            }
        }

        // Source state is independent of routing response caches, so each
        // route can sample and advance in place without a frame-local heap
        // allocation. Consumers below only read the cache; every route still
        // advances exactly once per frame.
        for index in 0..self.routings.len() {
            let (source, curve, curve_amount) = {
                let routing = &self.routings[index];
                (routing.source, routing.curve, routing.curve_amount)
            };
            let desired = shape(self.source_value(source), curve, curve_amount);
            self.routings[index].advance(desired, dt);
        }
    }

    /// Store a DeviceOrientation sample and apply calibration/range/expo.
    pub fn set_gyro_degrees(&mut self, alpha: f32, beta: f32, gamma: f32) {
        self.gyro_raw = [alpha, beta, gamma].map(|v| finite_or(v, 0.0));
        self.recompute_gyro();
    }

    /// Make the current orientation the centered (0.5) position on all axes.
    pub fn calibrate_gyro(&mut self) {
        for (axis, raw) in self.gyro_config.iter_mut().zip(self.gyro_raw) {
            axis.center_degrees = raw;
        }
        self.recompute_gyro();
    }

    /// Release a vanished phone stream without leaving its last pose applied.
    ///
    /// Raw values move to the persisted calibration centers as well as the
    /// normalized outputs moving to 0.5. A later physical sample therefore
    /// resumes from the same calibration instead of changing that contract.
    pub fn recenter_gyro(&mut self) {
        for (raw, config) in self.gyro_raw.iter_mut().zip(self.gyro_config) {
            *raw = finite_or(config.center_degrees, 0.0);
        }
        self.recompute_gyro();
    }

    /// Re-apply gyroscope configuration after a config field changes.
    pub fn recompute_gyro(&mut self) {
        for i in 0..3 {
            let cfg = self.gyro_config[i];
            let mut delta = self.gyro_raw[i] - finite_or(cfg.center_degrees, 0.0);
            if i == 0 {
                // Yaw wraps at 360 degrees; choose the shortest calibrated arc.
                delta = (delta + 180.0).rem_euclid(360.0) - 180.0;
            }
            let range = finite_or(cfg.range_degrees, 90.0).abs().max(0.001);
            let mut centered = (delta / range).clamp(-1.0, 1.0);
            if cfg.invert {
                centered = -centered;
            }
            let exponent = 2.0_f32.powf(finite_or(cfg.expo, 0.0).clamp(-2.0, 2.0));
            centered = centered.signum() * centered.abs().powf(exponent);
            self.gyro[i] = (0.5 + centered * 0.5).clamp(0.0, 1.0);
        }
    }

    pub fn set_pad(&mut self, x: f32, y: f32, active: bool) {
        self.pad = [
            finite_or(x, 0.5).clamp(0.0, 1.0),
            finite_or(y, 0.5).clamp(0.0, 1.0),
        ];
        self.pad_active = active;
    }

    /// Current value of a modulation source.
    pub fn source_value(&self, source: ModSource) -> f32 {
        match source {
            ModSource::Lfo(i) => self.lfo_values[i.min(NUM_LFOS - 1)],
            ModSource::AudioLevel => self.audio.level,
            ModSource::AudioBass => self.audio.bass,
            ModSource::AudioMid => self.audio.mid,
            ModSource::AudioHigh => self.audio.high,
            ModSource::AudioBand(i) => self.audio.bands[i.min(MAX_AUDIO_BANDS - 1)],
            ModSource::AudioOnset => self.audio.onset,
            ModSource::AudioBright => self.audio.bright,
            ModSource::AudioNoise => self.audio.noise,
            ModSource::Midi(i) => self.midi[i.min(NUM_MIDI_SLOTS - 1)],
            ModSource::GyroYaw => finite_or(self.gyro[0], 0.5).clamp(0.0, 1.0) * 2.0 - 1.0,
            ModSource::GyroPitch => finite_or(self.gyro[1], 0.5).clamp(0.0, 1.0) * 2.0 - 1.0,
            ModSource::GyroRoll => finite_or(self.gyro[2], 0.5).clamp(0.0, 1.0) * 2.0 - 1.0,
            ModSource::PadX => self.pad_source_value(0),
            ModSource::PadY => self.pad_source_value(1),
        }
    }

    fn pad_source_value(&self, axis: usize) -> f32 {
        let cfg = self.pad_config.axes[axis.min(1)];
        let centered = finite_or(self.pad[axis.min(1)], 0.5).clamp(0.0, 1.0) * 2.0 - 1.0;
        let value = shape(centered, cfg.curve, cfg.curve_amount).clamp(-1.0, 1.0);
        if cfg.quantize > 1 {
            let intervals = (cfg.quantize - 1) as f32;
            let unit = (value + 1.0) * 0.5;
            ((unit * intervals).round() / intervals) * 2.0 - 1.0
        } else {
            value
        }
    }

    /// Produce modulated copies of the effect, NTSC, and temporal params.
    /// Base values are untouched; each routing adds
    /// `source × depth × half-range`, clamped.
    #[cfg(test)]
    pub fn modulate(
        &self,
        effects: &EffectUniforms,
        ntsc: &NtscParams,
        temporal: &TemporalParams,
    ) -> (EffectUniforms, NtscParams, TemporalParams) {
        Self::modulate_from_offsets(effects, ntsc, temporal, &self.accumulate_offsets())
    }

    fn modulate_from_offsets(
        effects: &EffectUniforms,
        ntsc: &NtscParams,
        temporal: &TemporalParams,
        offsets: &RoutingOffsets,
    ) -> (EffectUniforms, NtscParams, TemporalParams) {
        let mut fx = *effects;
        let mut np = ntsc.clone();
        let mut tp = *temporal;

        for (index, &(target, min, max)) in TARGETS.iter().enumerate() {
            let offset = offsets.master[index] * (max - min) * 0.5;
            if offset != 0.0 {
                apply_offset(&mut fx, &mut np, &mut tp, target, offset, min, max);
            }
        }

        (fx, np, tp)
    }

    fn modulate_layer_from_offsets(
        index: usize,
        base_effects: &EffectUniforms,
        base_opacity: f32,
        base_speed: f32,
        base_fps: f32,
        offsets: &RoutingOffsets,
    ) -> LayerModulation {
        let mut effects = *base_effects;
        let offset = |suffix: &'static str, min: f32, max: f32| {
            offsets.layer_value(index, suffix) * (max - min) * 0.5
        };
        let opacity = (base_opacity + offset("opacity", 0.0, 1.0)).clamp(0.0, 1.0);
        let speed = (base_speed + offset("speed", 0.25, 4.0)).clamp(0.25, 4.0);
        let fps = (base_fps + offset("fps", 1.0, 240.0)).clamp(1.0, 240.0);

        macro_rules! apply {
            ($field:ident, $suffix:literal, $min:expr, $max:expr) => {
                effects.$field = (effects.$field + offset($suffix, $min, $max)).clamp($min, $max);
            };
        }
        apply!(key_threshold, "key_threshold", 0.0, 1.0);
        for (channel, suffix) in
            effects
                .key_color
                .iter_mut()
                .zip(["key_color_r", "key_color_g", "key_color_b"])
        {
            *channel = (*channel + offset(suffix, 0.0, 1.0)).clamp(0.0, 1.0);
        }
        apply!(key_tolerance, "key_tolerance", 0.0, 1.0);
        apply!(pixelate_size, "pixelate", 1.0, 32.0);
        apply!(rgb_split, "rgb_split", 0.0, 30.0);
        apply!(hue_shift, "hue_shift", -180.0, 180.0);
        apply!(saturation, "saturation", -1.0, 1.0);
        apply!(brightness, "brightness", -1.0, 1.0);
        apply!(contrast, "contrast", -1.0, 1.0);
        apply!(posterize, "posterize", 0.0, 16.0);
        apply!(grain_intensity, "grain_intensity", 0.0, 0.3);
        apply!(grain_size, "grain_size", 1.0, 4.0);
        apply!(vignette, "vignette", 0.0, 1.5);
        apply!(color_drift, "color_drift", 0.0, 0.02);
        apply!(breathe_scale, "breathe_scale", 0.0, 0.05);
        apply!(breathe_rotation, "breathe_rotation", 0.0, 2.0);
        apply!(breathe_position, "breathe_position", 0.0, 0.02);
        apply!(key_softness, "key_softness", 0.0, 0.5);
        apply!(downsample, "downsample", 0.05, 1.0);
        apply!(cellular_amount, "cellular_amount", 0.0, 1.0);
        apply!(cellular_scale, "cellular_scale", 2.0, 32.0);
        apply!(cellular_warp, "cellular_warp", 0.0, 1.0);
        apply!(cellular_speed, "cellular_speed", 0.0, 2.0);
        apply!(cellular_gap_amount, "cellular_gap_amount", 0.0, 1.0);
        apply!(cellular_gap_threshold, "cellular_gap_threshold", 0.0, 1.0);
        apply!(cellular_gap_softness, "cellular_gap_softness", 0.0, 0.5);

        LayerModulation {
            opacity,
            speed,
            fps,
            effects,
        }
    }

    /// Modulate a complete stack from one O(routes) accumulator pass. This is
    /// the live/export hot-path API; the single-layer method remains as a
    /// compatibility convenience for patch tooling and focused tests.
    pub fn frame(&self) -> ModulationFrame {
        ModulationFrame {
            offsets: self.accumulate_offsets(),
        }
    }

    /// Modulate one layer. Batch callers should prefer [`Self::modulate_layers`]
    /// so all layer destinations share one routing accumulator pass.
    #[cfg(test)]
    pub fn modulate_layer_full(
        &self,
        index: usize,
        base_effects: &EffectUniforms,
        base_opacity: f32,
        base_speed: f32,
        base_fps: f32,
    ) -> LayerModulation {
        let offsets = self.accumulate_offsets();
        Self::modulate_layer_from_offsets(
            index,
            base_effects,
            base_opacity,
            base_speed,
            base_fps,
            &offsets,
        )
    }

    /// Summed modulation offset for one named target — for values the app
    /// applies itself (e.g. the morph crossfader) rather than via
    /// `modulate`. Uses the same depth × half-range scaling.
    #[cfg(test)]
    pub fn target_offset(&self, target: &str) -> f32 {
        let Some((min, max)) = target_range(target) else {
            return 0.0;
        };
        let compiled = compile_target(target);
        self.accumulate_offsets().value(compiled) * (max - min) * 0.5
    }

    fn accumulate_offsets(&self) -> RoutingOffsets {
        let mut offsets = RoutingOffsets::default();
        for routing in &self.routings {
            let amount = routing.cached_value() * finite_or(routing.depth, 0.0);
            if amount != 0.0 {
                offsets.add(routing.compiled_target, amount);
            }
        }
        offsets
    }

    pub fn add_routing(&mut self) {
        if self.routings.len() < MAX_ROUTINGS {
            self.routings
                .push(Routing::new(ModSource::Lfo(0), "rgb_split", 0.0));
        }
    }

    pub fn remove_routing(&mut self, index: usize) {
        if index < self.routings.len() {
            self.routings.remove(index);
        }
    }

    /// Keep positional layer targets attached to the same logical layer when
    /// one layer is removed. Routes for the removed layer are discarded.
    pub fn remap_layer_targets_after_remove(&mut self, removed: usize) {
        self.routings.retain_mut(|routing| {
            let Some((layer, suffix)) = parse_layer_target(&routing.target) else {
                return true;
            };
            if layer == removed {
                return false;
            }
            let remapped = if layer > removed { layer - 1 } else { layer };
            let _ = routing.set_target(format!("layer{}_{suffix}", remapped + 1));
            true
        });
    }

    /// Apply the same stable permutation as moving an element in a Vec.
    pub fn remap_layer_targets_after_move(&mut self, from: usize, to: usize) {
        if from == to {
            return;
        }
        for routing in &mut self.routings {
            let Some((layer, suffix)) = parse_layer_target(&routing.target) else {
                continue;
            };
            let remapped = if layer == from {
                to
            } else if from < to && layer > from && layer <= to {
                layer - 1
            } else if to < from && layer >= to && layer < from {
                layer + 1
            } else {
                layer
            };
            let _ = routing.set_target(format!("layer{}_{suffix}", remapped + 1));
        }
    }

    /// Reset LFOs and routings; tempo is left alone (losing a dialed-in
    /// BPM mid-set would be crueler than any stale routing).
    pub fn reset(&mut self) {
        self.lfos = std::array::from_fn(|_| Lfo::default());
        self.routings.clear();
    }
}

fn parse_layer_target(target: &str) -> Option<(usize, &str)> {
    let rest = target.strip_prefix("layer")?;
    let (number, suffix) = rest.split_once('_')?;
    let one_based = number.parse::<usize>().ok()?;
    (1..=MAX_MOD_LAYERS)
        .contains(&one_based)
        .then_some((one_based - 1, suffix))
}

/// Source-depth sums, before multiplication by each destination's half-range.
/// The arrays are small, fixed, stack-resident, and rebuilt once per consumer
/// batch so routing edits never need an invalidation protocol.
#[derive(Clone)]
struct RoutingOffsets {
    master: [f32; TARGETS.len()],
    layer: [[f32; LAYER_TARGET_SUFFIXES.len()]; MAX_MOD_LAYERS],
}

const LAYER_TARGET_SUFFIXES: &[&str] = &[
    "opacity",
    "speed",
    "fps",
    "key_threshold",
    "key_color_r",
    "key_color_g",
    "key_color_b",
    "key_tolerance",
    "pixelate",
    "rgb_split",
    "hue_shift",
    "saturation",
    "brightness",
    "contrast",
    "posterize",
    "grain_intensity",
    "grain_size",
    "vignette",
    "color_drift",
    "breathe_scale",
    "breathe_rotation",
    "breathe_position",
    "cellular_amount",
    "cellular_scale",
    "cellular_warp",
    "cellular_speed",
    "cellular_gap_amount",
    "cellular_gap_threshold",
    "cellular_gap_softness",
    "key_softness",
    "downsample",
];

impl Default for RoutingOffsets {
    fn default() -> Self {
        Self {
            master: [0.0; TARGETS.len()],
            layer: [[0.0; LAYER_TARGET_SUFFIXES.len()]; MAX_MOD_LAYERS],
        }
    }
}

fn layer_suffix_index(suffix: &str) -> Option<usize> {
    Some(match suffix {
        "opacity" => 0,
        "speed" => 1,
        "fps" => 2,
        "key_threshold" => 3,
        "key_color_r" => 4,
        "key_color_g" => 5,
        "key_color_b" => 6,
        "key_tolerance" => 7,
        "pixelate" => 8,
        "rgb_split" => 9,
        "hue_shift" => 10,
        "saturation" => 11,
        "brightness" => 12,
        "contrast" => 13,
        "posterize" => 14,
        "grain_intensity" => 15,
        "grain_size" => 16,
        "vignette" => 17,
        "color_drift" => 18,
        "breathe_scale" => 19,
        "breathe_rotation" => 20,
        "breathe_position" => 21,
        "cellular_amount" => 22,
        "cellular_scale" => 23,
        "cellular_warp" => 24,
        "cellular_speed" => 25,
        "cellular_gap_amount" => 26,
        "cellular_gap_threshold" => 27,
        "cellular_gap_softness" => 28,
        "key_softness" => 29,
        "downsample" => 30,
        _ => return None,
    })
}

impl RoutingOffsets {
    fn add(&mut self, target: CompiledTarget, amount: f32) {
        match target {
            CompiledTarget::Master(index) => self.master[index] += amount,
            CompiledTarget::Layer { index, suffix } => self.layer[index][suffix] += amount,
            CompiledTarget::Invalid => {}
        }
    }

    #[cfg(test)]
    fn value(&self, target: CompiledTarget) -> f32 {
        match target {
            CompiledTarget::Master(index) => self.master[index],
            CompiledTarget::Layer { index, suffix } => self.layer[index][suffix],
            CompiledTarget::Invalid => 0.0,
        }
    }

    fn layer_value(&self, layer: usize, suffix: &str) -> f32 {
        let Some(suffix) = layer_suffix_index(suffix) else {
            return 0.0;
        };
        self.layer
            .get(layer)
            .and_then(|values| values.get(suffix))
            .copied()
            .unwrap_or(0.0)
    }
}

fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

fn finite_f64_or(value: f64, fallback: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

/// Shape a signed value without ever discarding its polarity.
pub fn shape(value: f32, curve: Curve, amount: f32) -> f32 {
    let value = finite_or(value, 0.0).clamp(-1.0, 1.0);
    let sign = value.signum();
    let magnitude = value.abs();
    let amount = finite_or(amount, 0.0).clamp(-2.0, 2.0);
    let shaped = match curve {
        Curve::Linear => magnitude,
        Curve::Exp => magnitude.powf(2.0_f32.powf(amount)),
        Curve::Log => magnitude.powf(2.0_f32.powf(-amount)),
        Curve::SCurve => magnitude * magnitude * (3.0 - 2.0 * magnitude),
        Curve::Steps => {
            // -2..+2 maps to 2, 4, 8, 16, 32 equal increments.
            let steps = 2.0_f32.powf(amount + 3.0).round().clamp(2.0, 32.0);
            (magnitude * steps).floor() / steps
        }
    };
    sign * shaped.clamp(0.0, 1.0)
}

fn exponential_follow(current: f32, desired: f32, dt: f32, tau: f32) -> f32 {
    let dt = finite_or(dt, 0.0).max(0.0);
    let tau = finite_or(tau, 0.0).max(0.0);
    if tau <= f32::EPSILON {
        return desired;
    }
    let alpha = 1.0 - (-dt / tau).exp();
    current + (desired - current) * alpha
}

fn apply_offset(
    fx: &mut EffectUniforms,
    np: &mut NtscParams,
    tp: &mut TemporalParams,
    target: &str,
    offset: f32,
    min: f32,
    max: f32,
) {
    let slot: &mut f32 = match target {
        "pixelate" => &mut fx.pixelate_size,
        "rgb_split" => &mut fx.rgb_split,
        "hue_shift" => &mut fx.hue_shift,
        "saturation" => &mut fx.saturation,
        "brightness" => &mut fx.brightness,
        "contrast" => &mut fx.contrast,
        "posterize" => &mut fx.posterize,
        "grain_intensity" => &mut fx.grain_intensity,
        "grain_size" => &mut fx.grain_size,
        "vignette" => &mut fx.vignette,
        "color_drift" => &mut fx.color_drift,
        "downsample" => &mut fx.downsample,
        "breathe_scale" => &mut fx.breathe_scale,
        "breathe_rotation" => &mut fx.breathe_rotation,
        "breathe_position" => &mut fx.breathe_position,
        "key_threshold" => &mut fx.key_threshold,
        "key_softness" => &mut fx.key_softness,
        "key_color_r" => &mut fx.key_color[0],
        "key_color_g" => &mut fx.key_color[1],
        "key_color_b" => &mut fx.key_color[2],
        "key_tolerance" => &mut fx.key_tolerance,
        "cellular_amount" => &mut fx.cellular_amount,
        "cellular_scale" => &mut fx.cellular_scale,
        "cellular_warp" => &mut fx.cellular_warp,
        "cellular_speed" => &mut fx.cellular_speed,
        "cellular_gap_amount" => &mut fx.cellular_gap_amount,
        "cellular_gap_threshold" => &mut fx.cellular_gap_threshold,
        "cellular_gap_softness" => &mut fx.cellular_gap_softness,
        "ntsc_snow" => &mut np.snow_intensity,
        "ntsc_tracking_snow" => &mut np.tracking_noise_snow,
        "ntsc_edge_wave" => &mut np.edge_wave_intensity,
        "ntsc_edge_wave_speed" => &mut np.edge_wave_speed,
        "ntsc_head_shift" => &mut np.head_switching_shift,
        "ntsc_tracking_wave" => &mut np.tracking_noise_wave,
        "ntsc_chroma_loss" => &mut np.chroma_loss,
        "ntsc_composite_noise" => &mut np.composite_noise_intensity,
        "ntsc_luma_noise" => &mut np.luma_noise_intensity,
        "ntsc_chroma_noise" => &mut np.chroma_noise_intensity,
        "ntsc_luma_smear" => &mut np.luma_smear,
        "ntsc_sharpening" => &mut np.composite_sharpening,
        "temporal_feedback" => &mut tp.feedback,
        "temporal_slitscan" => &mut tp.slitscan,
        "temporal_fb_zoom" => &mut tp.fb_zoom,
        "temporal_fb_rotate" => &mut tp.fb_rotate,
        "temporal_slit_angle" => &mut tp.slit_angle,
        "temporal_key_threshold" => &mut tp.key_threshold,
        "temporal_key_softness" => &mut tp.key_softness,
        "temporal_key_history" => &mut tp.key_history,
        _ => return,
    };
    *slot = (*slot + offset).clamp(min, max);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn manual_bpm_changes_preserve_beat_phase_and_use_the_new_rate() {
        for new_bpm in [60.0, 300.0] {
            let mut clock = Clock::new();
            let downbeat = Instant::now();
            clock.tap(downbeat);
            let change = downbeat + Duration::from_millis(1_950);
            let beat_before = clock.beat(change);
            assert!((beat_before - 3.9).abs() < 1e-9);

            clock.set_bpm_at(new_bpm, change);

            let beat_after = clock.beat(change);
            assert!(
                (beat_after - beat_before).abs() < 1e-8,
                "{new_bpm} BPM moved beat {beat_before} to {beat_after}"
            );
            let one_second_later = clock.beat(change + Duration::from_secs(1));
            let expected = beat_before + new_bpm as f64 / 60.0;
            assert!((one_second_later - expected).abs() < 1e-8);
        }
    }

    #[test]
    fn internal_clock_pause_is_idempotent_and_resumes_without_catch_up() {
        let mut clock = Clock::new();
        let downbeat = Instant::now();
        clock.tap(downbeat);
        let pause = downbeat + Duration::from_secs(2);
        clock.set_paused(true, pause);
        let frozen = clock.beat(pause);
        assert!((frozen - 4.0).abs() < 1e-9);
        assert!(clock.is_paused());

        clock.set_paused(true, pause + Duration::from_secs(20));
        assert_eq!(clock.beat(pause + Duration::from_secs(20)), frozen);

        let resume = pause + Duration::from_secs(20);
        clock.set_paused(false, resume);
        assert!(!clock.is_paused());
        assert!((clock.beat(resume) - frozen).abs() < 1e-9);
        assert!((clock.beat(resume + Duration::from_millis(500)) - (frozen + 1.0)).abs() < 1e-9);
    }

    #[test]
    fn external_clock_telemetry_advances_under_pause_without_moving_program_phase() {
        let mut clock = Clock::new();
        let start = Instant::now();
        clock.set_external_beat(Some(10.0), start);
        clock.set_paused(true, start);
        assert_eq!(clock.beat(start), 10.0);

        // Hardware continues to publish its absolute transport while the
        // visual program is frozen.
        clock.set_external_beat(Some(42.0), start + Duration::from_secs(8));
        assert_eq!(clock.beat(start + Duration::from_secs(8)), 10.0);

        let resume = start + Duration::from_secs(8);
        clock.set_paused(false, resume);
        assert_eq!(clock.beat(resume), 10.0);
        clock.set_external_beat(Some(42.5), resume + Duration::from_millis(250));
        assert_eq!(clock.beat(resume + Duration::from_millis(250)), 10.5);

        // Falling back to the internal clock also preserves that logical
        // position rather than exposing the raw external count.
        let handoff = resume + Duration::from_millis(250);
        clock.set_external_beat(None, handoff);
        assert!((clock.beat(handoff) - 10.5).abs() < 1e-9);
    }

    #[test]
    fn curves_preserve_sign_and_endpoints() {
        for curve in [
            Curve::Linear,
            Curve::Exp,
            Curve::Log,
            Curve::SCurve,
            Curve::Steps,
        ] {
            approx(shape(0.0, curve, 1.0), 0.0);
            approx(shape(1.0, curve, 1.0), 1.0);
            approx(shape(-1.0, curve, 1.0), -1.0);
            assert!(shape(0.4, curve, 1.0) >= 0.0);
            assert!(shape(-0.4, curve, 1.0) <= 0.0);
        }
        assert!(shape(0.5, Curve::Exp, 1.0) < 0.5);
        assert!(shape(0.5, Curve::Log, 1.0) > 0.5);
        approx(shape(0.5, Curve::SCurve, 0.0), 0.5);
    }

    #[test]
    fn configurable_audio_band_sources_roundtrip_and_legacy_aliases_hold() {
        let mut matrix = ModMatrix::new();
        matrix.audio.bands = [0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88];
        matrix.audio.bass = matrix.audio.bands[0];
        matrix.audio.mid = matrix.audio.bands[1];
        matrix.audio.high = matrix.audio.bands[2];

        for index in 0..MAX_AUDIO_BANDS {
            let source = ModSource::AudioBand(index);
            assert_eq!(ModSource::try_from_str(source.as_str()), Some(source));
            approx(matrix.source_value(source), matrix.audio.bands[index]);
        }
        approx(
            matrix.source_value(ModSource::AudioBass),
            matrix.source_value(ModSource::AudioBand(0)),
        );
        approx(
            matrix.source_value(ModSource::AudioMid),
            matrix.source_value(ModSource::AudioBand(1)),
        );
        approx(
            matrix.source_value(ModSource::AudioHigh),
            matrix.source_value(ModSource::AudioBand(2)),
        );
        assert_eq!(ModSource::try_from_str("audio_band9"), None);
    }

    #[test]
    fn exponential_slew_uses_distinct_attack_and_release() {
        let mut matrix = ModMatrix::new();
        let mut route = Routing::new(ModSource::Midi(0), "brightness", 1.0);
        route.attack = 1.0;
        route.release = 2.0;
        matrix.routings.push(route);

        matrix.midi[0] = 1.0;
        matrix.update_at_beat(0.0, 1.0);
        let attacked = 1.0 - (-1.0_f32).exp();
        approx(matrix.routings[0].cached, attacked);

        matrix.midi[0] = 0.0;
        matrix.update_at_beat(0.0, 2.0);
        approx(matrix.routings[0].cached, attacked * (-1.0_f32).exp());

        let one_step = exponential_follow(0.0, 1.0, 1.0, 0.7);
        let mut ten_steps = 0.0;
        for _ in 0..10 {
            ten_steps = exponential_follow(ten_steps, 1.0, 0.1, 0.7);
        }
        approx(one_step, ten_steps);
    }

    #[test]
    fn update_timing_reset_zeroes_one_delta_without_reanchoring_beat() {
        let mut matrix = ModMatrix::new();
        let downbeat = Instant::now();
        matrix.clock.tap(downbeat);
        matrix.pad_config.spring_enabled = true;
        matrix.pad_config.spring_rate = 4.0;
        matrix.set_pad(1.0, 0.0, false);
        matrix.update(downbeat + Duration::from_millis(250));

        let published_beat = matrix.current_beat;
        let future = downbeat + Duration::from_secs(10);
        let future_beat = matrix.clock.beat(future);
        matrix.set_pad(1.0, 0.0, false);

        matrix.reset_update_timing();

        assert_eq!(matrix.current_beat, published_beat);
        assert_eq!(matrix.clock.beat(future), future_beat);
        matrix.update(future);
        assert_eq!(matrix.pad, [1.0, 0.0]);

        matrix.update(future + Duration::from_millis(250));
        assert!(matrix.pad[0] < 1.0);
        assert!(matrix.pad[1] > 0.0);
    }

    #[test]
    fn offsets_sum_before_clamp_and_bases_are_immutable() {
        fn render(reversed: bool) -> (f32, f32) {
            let mut matrix = ModMatrix::new();
            matrix.midi[0] = 1.0;
            let positive = Routing::new(ModSource::Midi(0), "brightness", 1.0);
            let negative = Routing::new(ModSource::Midi(0), "brightness", -1.0);
            matrix.routings = if reversed {
                vec![negative, positive]
            } else {
                vec![positive, negative]
            };
            matrix.update_at_beat(0.0, 1.0 / 30.0);
            let base = EffectUniforms {
                brightness: 0.9,
                ..Default::default()
            };
            let (modulated, _, _) =
                matrix.modulate(&base, &NtscParams::default(), &TemporalParams::default());
            (base.brightness, modulated.brightness)
        }

        let forward = render(false);
        let reverse = render(true);
        approx(forward.0, 0.9);
        approx(reverse.0, 0.9);
        approx(forward.1, 0.9);
        approx(reverse.1, forward.1);
    }

    #[test]
    fn consumers_are_pure_and_routing_lifecycle_keeps_state_aligned() {
        let mut matrix = ModMatrix::new();
        matrix.midi = [0.25, 0.75, 0.0, 0.0];
        let mut first = Routing::new(ModSource::Midi(0), "brightness", 1.0);
        first.attack = 1.0;
        let mut second = Routing::new(ModSource::Midi(1), "contrast", 1.0);
        second.attack = 1.0;
        matrix.routings = vec![first, second];
        matrix.update_at_beat(0.0, 0.5);
        let cached = matrix.routings[1].cached;

        let _ = matrix.target_offset("contrast");
        let _ = matrix.modulate(
            &EffectUniforms::default(),
            &NtscParams::default(),
            &TemporalParams::default(),
        );
        let _ = matrix.modulate_layer_full(0, &EffectUniforms::default(), 1.0, 1.0, 30.0);
        approx(matrix.routings[1].cached, cached);

        matrix.remove_routing(0);
        approx(matrix.routings[0].cached, cached);
        matrix.add_routing();
        approx(matrix.routings[1].cached, 0.0);
        matrix.routings[0].reset_runtime();
        approx(matrix.routings[0].cached, 0.0);
    }

    #[test]
    fn compiled_targets_and_batched_layers_match_single_layer_semantics() {
        let mut matrix = ModMatrix::new();
        matrix.midi[0] = 0.8;
        matrix.routings = vec![
            Routing::new(ModSource::Midi(0), "brightness", 0.5),
            Routing::new(ModSource::Midi(0), "layer1_opacity", -0.25),
            Routing::new(ModSource::Midi(0), "layer2_cellular_gap_softness", 0.75),
            Routing::new(ModSource::Midi(0), "layer2_fps", 0.2),
        ];
        matrix.update_at_beat(0.0, 0.0);
        let first = EffectUniforms::default();
        let second = EffectUniforms::default();
        let expected_first = matrix.modulate_layer_full(0, &first, 0.9, 1.0, 30.0);
        let expected_second = matrix.modulate_layer_full(1, &second, 0.7, 1.5, 24.0);

        let batched = matrix
            .frame()
            .modulate_layers([(&first, 0.9, 1.0, 30.0), (&second, 0.7, 1.5, 24.0)]);

        approx(batched[0].opacity, expected_first.opacity);
        approx(
            batched[0].effects.brightness,
            expected_first.effects.brightness,
        );
        approx(batched[1].fps, expected_second.fps);
        approx(
            batched[1].effects.cellular_gap_softness,
            expected_second.effects.cellular_gap_softness,
        );
        let frame = matrix.frame();
        let cached = frame.modulate_layers([(&first, 0.9, 1.0, 30.0), (&second, 0.7, 1.5, 24.0)]);
        approx(cached[0].opacity, batched[0].opacity);
        approx(cached[1].fps, batched[1].fps);
        let (master, _, _) = frame.modulate(
            &EffectUniforms::default(),
            &NtscParams::default(),
            &TemporalParams::default(),
        );
        approx(master.brightness, 0.4);
        assert_eq!(
            matrix.routings[2].compiled_target,
            compile_target("layer2_cellular_gap_softness")
        );
    }

    #[test]
    fn target_change_resets_signal_but_response_time_change_preserves_it() {
        let mut route = Routing::new(ModSource::Midi(0), "brightness", 1.0);
        route.attack = 1.0;
        route.advance(1.0, 1.0);
        let live = route.cached_value();
        assert!(live > 0.0);

        route.attack = 2.0;
        route.release = 3.0;
        approx(route.cached_value(), live);
        assert!(!route.set_target("brightness"));
        approx(route.cached_value(), live);
        assert!(route.set_target("layer1_brightness"));
        approx(route.cached_value(), 0.0);
        assert_eq!(route.compiled_target, compile_target("layer1_brightness"));

        // Target identity and its compiled destination change atomically.
        route.state = 1.0;
        route.cached = 1.0;
        assert!(route.set_target("contrast"));
        route.state = 1.0;
        route.cached = 1.0;
        let mut matrix = ModMatrix::new();
        matrix.routings.push(route);
        let frame = matrix.frame();
        let (effects, _, _) = frame.modulate(
            &EffectUniforms::default(),
            &NtscParams::default(),
            &TemporalParams::default(),
        );
        approx(effects.brightness, 0.0);
        approx(effects.contrast, 1.0);
    }

    #[test]
    fn gyro_calibration_wrap_and_invert_are_stable() {
        let mut matrix = ModMatrix::new();
        matrix.set_gyro_degrees(359.0, 10.0, -20.0);
        matrix.calibrate_gyro();
        for value in matrix.gyro {
            approx(value, 0.5);
        }

        matrix.set_gyro_degrees(1.0, 20.0, -20.0);
        assert!(matrix.gyro[0] > 0.5, "yaw must cross 360 on shortest arc");
        assert!(matrix.gyro[1] > 0.5);
        matrix.gyro_config[1].invert = true;
        matrix.recompute_gyro();
        assert!(matrix.gyro[1] < 0.5);
    }

    #[test]
    fn gyro_recenter_releases_last_pose_without_losing_calibration() {
        let mut matrix = ModMatrix::new();
        matrix.set_gyro_degrees(270.0, 25.0, -30.0);
        matrix.calibrate_gyro();
        let centers = matrix.gyro_raw;
        matrix.set_gyro_degrees(320.0, 60.0, 20.0);
        assert_ne!(matrix.gyro, [0.5; 3]);

        matrix.recenter_gyro();

        assert_eq!(matrix.gyro_raw, centers);
        assert_eq!(matrix.gyro, [0.5; 3]);
    }

    #[test]
    fn pad_quantize_and_spring_are_deterministic() {
        let mut matrix = ModMatrix::new();
        matrix.pad_config.axes[0].quantize = 4;
        matrix.set_pad(0.74, 0.9, true);
        approx(matrix.source_value(ModSource::PadX), 1.0 / 3.0);

        // N means exactly N evenly spaced positions, inclusive of endpoints,
        // with nearest-position snapping and symmetric midpoint behavior.
        let samples = [0.0, 0.16, 0.34, 0.5, 0.66, 0.84, 1.0];
        let quantized: Vec<f32> = samples
            .into_iter()
            .map(|value| {
                matrix.set_pad(value, 0.5, true);
                matrix.source_value(ModSource::PadX)
            })
            .collect();
        for (actual, expected) in
            quantized
                .into_iter()
                .zip([-1.0, -1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0, 1.0])
        {
            approx(actual, expected);
        }

        matrix.pad_config.spring_enabled = true;
        matrix.pad_config.spring_rate = 4.0;
        matrix.set_pad(0.74, 0.9, false);
        matrix.update_at_beat(0.0, 0.25);
        approx(matrix.pad[0], 0.5 + 0.24 * (-1.0_f32).exp());
        approx(matrix.pad[1], 0.5 + 0.4 * (-1.0_f32).exp());

        let mut one = ModMatrix::new();
        one.pad_config.spring_enabled = true;
        one.pad_config.spring_rate = 4.0;
        one.set_pad(1.0, 0.0, false);
        one.update_at_beat(0.0, 1.0);
        let mut ten = ModMatrix::new();
        ten.pad_config.spring_enabled = true;
        ten.pad_config.spring_rate = 4.0;
        ten.set_pad(1.0, 0.0, false);
        for frame in 0..10 {
            ten.update_at_beat(frame as f64, 0.1);
        }
        approx(one.pad[0], ten.pad[0]);
        approx(one.pad[1], ten.pad[1]);
    }

    #[test]
    fn lfo_phase_setter_and_sampler_reject_nonfinite_values() {
        let mut lfo = Lfo::default();
        lfo.set_phase(1.25);
        approx(lfo.phase, 0.25);
        lfo.set_phase(-0.25);
        approx(lfo.phase, 0.75);

        for phase in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            lfo.set_phase(phase);
            approx(lfo.phase, 0.0);
            assert!(lfo.value(3.0, 0).is_finite());

            // Direct field writes remain safe for compatibility with existing
            // internal callers while they migrate to `set_phase`.
            lfo.phase = phase;
            assert!(lfo.value(3.0, 0).is_finite());
        }
    }

    #[test]
    fn every_supported_layer_has_full_target_validation_and_modulation() {
        assert!(include_str!("../../static/app.js")
            .contains(&format!("const MAX_MOD_LAYERS = {MAX_MOD_LAYERS};")));
        assert_eq!(target_range("layer16_brightness"), Some((-1.0, 1.0)));
        assert_eq!(target_range("layer16_downsample"), Some((0.05, 1.0)));
        assert_eq!(target_range("layer17_opacity"), None);
        assert_eq!(target_range("layer0_speed"), None);
        assert_eq!(target_range("layer16_unknown"), None);

        let mut matrix = ModMatrix::new();
        matrix.midi[0] = 1.0;
        matrix
            .routings
            .push(Routing::new(ModSource::Midi(0), "layer16_brightness", 1.0));
        matrix.update_at_beat(0.0, 0.0);
        let base = EffectUniforms::default();
        let modulated = matrix.modulate_layer_full(15, &base, 1.0, 1.0, 30.0);
        approx(modulated.effects.brightness, 1.0);
        approx(base.brightness, 0.0);
    }

    #[test]
    fn cellular_targets_modulate_master_and_layers_without_mutating_bases() {
        for (target, range) in [
            ("cellular_amount", (0.0, 1.0)),
            ("cellular_scale", (2.0, 32.0)),
            ("cellular_warp", (0.0, 1.0)),
            ("cellular_speed", (0.0, 2.0)),
            ("cellular_gap_amount", (0.0, 1.0)),
            ("cellular_gap_threshold", (0.0, 1.0)),
            ("cellular_gap_softness", (0.0, 0.5)),
            ("layer16_cellular_amount", (0.0, 1.0)),
            ("layer16_cellular_scale", (2.0, 32.0)),
            ("layer16_cellular_warp", (0.0, 1.0)),
            ("layer16_cellular_speed", (0.0, 2.0)),
            ("layer16_cellular_gap_amount", (0.0, 1.0)),
            ("layer16_cellular_gap_threshold", (0.0, 1.0)),
            ("layer16_cellular_gap_softness", (0.0, 0.5)),
        ] {
            assert_eq!(target_range(target), Some(range));
        }

        let mut matrix = ModMatrix::new();
        matrix.midi[0] = 1.0;
        for target in [
            "cellular_amount",
            "cellular_scale",
            "cellular_warp",
            "cellular_speed",
            "cellular_gap_amount",
            "cellular_gap_threshold",
            "cellular_gap_softness",
            "layer1_cellular_amount",
            "layer1_cellular_scale",
            "layer1_cellular_warp",
            "layer1_cellular_speed",
            "layer1_cellular_gap_amount",
            "layer1_cellular_gap_threshold",
            "layer1_cellular_gap_softness",
        ] {
            matrix
                .routings
                .push(Routing::new(ModSource::Midi(0), target, 1.0));
        }
        matrix.update_at_beat(0.0, 0.0);

        let base = EffectUniforms::default();
        let (master, _, _) =
            matrix.modulate(&base, &NtscParams::default(), &TemporalParams::default());
        approx(master.cellular_amount, 0.5);
        approx(master.cellular_scale, 25.0);
        approx(master.cellular_warp, 0.85);
        approx(master.cellular_speed, 1.25);
        approx(master.cellular_gap_amount, 0.5);
        approx(master.cellular_gap_threshold, 1.0);
        approx(master.cellular_gap_softness, 0.33);

        let layer = matrix.modulate_layer_full(0, &base, 1.0, 1.0, 30.0);
        approx(layer.effects.cellular_amount, 0.5);
        approx(layer.effects.cellular_scale, 25.0);
        approx(layer.effects.cellular_warp, 0.85);
        approx(layer.effects.cellular_speed, 1.25);
        approx(layer.effects.cellular_gap_amount, 0.5);
        approx(layer.effects.cellular_gap_threshold, 1.0);
        approx(layer.effects.cellular_gap_softness, 0.33);

        approx(base.cellular_amount, 0.0);
        approx(base.cellular_scale, 10.0);
        approx(base.cellular_warp, 0.35);
        approx(base.cellular_speed, 0.25);
        approx(base.cellular_gap_amount, 0.0);
        approx(base.cellular_gap_threshold, 0.65);
        approx(base.cellular_gap_softness, 0.08);
    }

    #[test]
    fn centered_performance_sources_are_bipolar_without_changing_telemetry() {
        let mut matrix = ModMatrix::new();
        matrix.gyro = [0.5, 0.25, 0.75];
        matrix.set_pad(0.5, 0.25, true);
        assert_eq!(matrix.gyro, [0.5, 0.25, 0.75]);
        assert_eq!(matrix.pad, [0.5, 0.25]);
        approx(matrix.source_value(ModSource::GyroYaw), 0.0);
        approx(matrix.source_value(ModSource::GyroPitch), -0.5);
        approx(matrix.source_value(ModSource::GyroRoll), 0.5);
        approx(matrix.source_value(ModSource::PadX), 0.0);
        approx(matrix.source_value(ModSource::PadY), -0.5);
    }

    #[test]
    fn layer_route_targets_follow_move_and_remove_permutations_by_identity() {
        let mut matrix = ModMatrix::new();
        matrix.routings = vec![
            Routing::new(ModSource::Lfo(0), "layer1_brightness", 0.1),
            Routing::new(ModSource::Lfo(1), "layer2_key", 0.2),
            Routing::new(ModSource::Lfo(2), "layer3_opacity", 0.3),
            Routing::new(ModSource::Lfo(3), "morph", 0.4),
        ];
        let ids: Vec<u64> = matrix.routings.iter().map(Routing::route_id).collect();
        assert_eq!(matrix.routings[1].target, "layer2_key_threshold");

        matrix.remap_layer_targets_after_move(0, 2);
        assert_eq!(matrix.routings[0].target, "layer3_brightness");
        assert_eq!(matrix.routings[1].target, "layer1_key_threshold");
        assert_eq!(matrix.routings[2].target, "layer2_opacity");
        assert_eq!(matrix.routings[3].target, "morph");
        assert_eq!(
            matrix
                .routings
                .iter()
                .map(Routing::route_id)
                .collect::<Vec<_>>(),
            ids
        );

        matrix.remap_layer_targets_after_remove(1);
        assert_eq!(matrix.routings.len(), 3);
        assert_eq!(matrix.routings[0].target, "layer2_brightness");
        assert_eq!(matrix.routings[1].target, "layer1_key_threshold");
        assert_eq!(matrix.routings[2].target, "morph");
        assert_eq!(matrix.routings[0].route_id(), ids[0]);
        assert_eq!(matrix.routings[1].route_id(), ids[1]);
        assert_eq!(matrix.routings[2].route_id(), ids[3]);
    }

    #[test]
    fn expanded_continuous_targets_include_key_temporal_ntsc_and_layer_fps() {
        for (target, range) in [
            ("key_color_r", (0.0, 1.0)),
            ("key_tolerance", (0.0, 1.0)),
            ("ntsc_edge_wave_speed", (0.0, 10.0)),
            ("ntsc_tracking_wave", (0.0, 50.0)),
            ("ntsc_composite_noise", (0.0, 0.5)),
            ("ntsc_chroma_noise", (0.0, 0.5)),
            ("ntsc_luma_smear", (0.0, 1.0)),
            ("ntsc_sharpening", (-1.0, 2.0)),
            ("temporal_key_threshold", (0.0, 1.0)),
            ("temporal_key_softness", (0.0, 0.5)),
            ("temporal_key_history", (1.0, 23.0)),
            ("layer1_fps", (1.0, 240.0)),
            ("layer1_key", (0.0, 1.0)),
            ("layer1_key_threshold", (0.0, 1.0)),
        ] {
            assert_eq!(target_range(target), Some(range), "{target}");
        }

        let mut matrix = ModMatrix::new();
        matrix.midi[0] = 1.0;
        matrix
            .routings
            .push(Routing::new(ModSource::Midi(0), "layer1_fps", 1.0));
        matrix.update_at_beat(0.0, 1.0 / 30.0);
        let base = EffectUniforms::default();
        let layer = matrix.modulate_layer_full(0, &base, 1.0, 1.0, 30.0);
        approx(layer.fps, 149.5);
        approx(base.key_threshold, EffectUniforms::default().key_threshold);
    }
}
