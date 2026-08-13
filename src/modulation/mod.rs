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
    ("ntsc_snow", 0.0, 1.0),
    ("ntsc_tracking_snow", 0.0, 1.0),
    ("ntsc_edge_wave", 0.0, 20.0),
    ("ntsc_head_shift", -100.0, 100.0),
    ("ntsc_chroma_loss", 0.0, 0.01),
    ("ntsc_luma_noise", 0.0, 0.2),
    ("temporal_feedback", 0.0, 0.95),
    ("temporal_slitscan", 0.0, 1.0),
    ("temporal_fb_zoom", 0.9, 1.1),
    ("temporal_fb_rotate", -5.0, 5.0),
    ("temporal_slit_angle", -180.0, 180.0),
    // The patch-morph crossfader; applied by the app, not apply_offset.
    ("morph", 0.0, 1.0),
];

/// Resolve a target's legal value range, including dynamically named layer
/// targets up to [`MAX_MOD_LAYERS`].
pub fn target_range(target: &str) -> Option<(f32, f32)> {
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
        "opacity" | "key" => Some((0.0, 1.0)),
        "speed" => Some((0.25, 4.0)),
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
        "downsample" => Some((0.05, 1.0)),
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
    pub effects: EffectUniforms,
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
    pub source: ModSource,
    pub target: String,
    pub depth: f32,
    pub curve: Curve,
    pub curve_amount: f32,
    /// Seconds to follow a rising source. Zero is instantaneous.
    pub attack: f32,
    /// Seconds to follow a falling source. Zero is instantaneous.
    pub release: f32,
    state: f32,
    cached: f32,
}

impl Routing {
    pub fn new(source: ModSource, target: impl Into<String>, depth: f32) -> Self {
        Self {
            source,
            target: target.into(),
            depth,
            curve: Curve::Linear,
            curve_amount: 0.0,
            attack: 0.0,
            release: 0.0,
            state: 0.0,
            cached: 0.0,
        }
    }

    /// Clear transient state after replacing a route's identity/configuration.
    pub fn reset_runtime(&mut self) {
        self.state = 0.0;
        self.cached = 0.0;
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
        }
    }

    /// Global beat position (quarter notes since the last downbeat anchor),
    /// or the external MIDI clock's position when one is driving.
    pub fn beat(&self, now: Instant) -> f64 {
        match self.external_beat {
            Some(beat) => beat,
            None => now.duration_since(self.anchor).as_secs_f64() * (self.bpm as f64 / 60.0),
        }
    }

    /// Hand the beat position to (or take it back from) an external clock.
    /// When the external clock disappears, re-anchor so the internal clock
    /// continues from the same position instead of jumping.
    pub fn set_external_beat(&mut self, beat: Option<f64>, now: Instant) {
        if beat.is_none() {
            if let Some(last) = self.external_beat {
                let elapsed = last / (self.bpm as f64 / 60.0);
                self.anchor = now - Duration::from_secs_f64(elapsed.max(0.0));
            }
        }
        self.external_beat = beat;
    }

    pub fn is_external(&self) -> bool {
        self.external_beat.is_some()
    }

    pub fn set_bpm(&mut self, bpm: f32) {
        self.bpm = bpm.clamp(30.0, 300.0);
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

        // Sample every route before mutating any route. Consumers below only
        // read this cache, so a route advances once even if master, morph and
        // multiple layer paths all query it during the frame.
        let desired: Vec<f32> = self
            .routings
            .iter()
            .map(|routing| {
                let raw = self.source_value(routing.source);
                shape(raw, routing.curve, routing.curve_amount)
            })
            .collect();
        for (routing, desired) in self.routings.iter_mut().zip(desired) {
            routing.advance(desired, dt);
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
            ModSource::GyroYaw => self.gyro[0],
            ModSource::GyroPitch => self.gyro[1],
            ModSource::GyroRoll => self.gyro[2],
            ModSource::PadX => self.pad_source_value(0),
            ModSource::PadY => self.pad_source_value(1),
        }
    }

    fn pad_source_value(&self, axis: usize) -> f32 {
        let cfg = self.pad_config.axes[axis.min(1)];
        let value = shape(self.pad[axis.min(1)], cfg.curve, cfg.curve_amount).clamp(0.0, 1.0);
        if cfg.quantize > 1 {
            let intervals = (cfg.quantize - 1) as f32;
            (value * intervals).round() / intervals
        } else {
            value
        }
    }

    /// Produce modulated copies of the effect, NTSC, and temporal params.
    /// Base values are untouched; each routing adds
    /// `source × depth × half-range`, clamped.
    pub fn modulate(
        &self,
        effects: &EffectUniforms,
        ntsc: &NtscParams,
        temporal: &TemporalParams,
    ) -> (EffectUniforms, NtscParams, TemporalParams) {
        let mut fx = *effects;
        let mut np = ntsc.clone();
        let mut tp = *temporal;

        for &(target, min, max) in TARGETS {
            let offset = self.routing_offset(target, min, max);
            if offset != 0.0 {
                apply_offset(&mut fx, &mut np, &mut tp, target, offset, min, max);
            }
        }

        (fx, np, tp)
    }

    /// Modulate every continuous per-layer effect through the same cached
    /// routing path as master effects. Discrete selectors (blend, key mode,
    /// grain algorithm) intentionally remain base controls.
    pub fn modulate_layer_full(
        &self,
        index: usize,
        base_effects: &EffectUniforms,
        base_opacity: f32,
        base_speed: f32,
    ) -> LayerModulation {
        let mut effects = *base_effects;
        let opacity =
            (base_opacity + self.layer_routing_offset(index, "opacity", 0.0, 1.0)).clamp(0.0, 1.0);
        let speed =
            (base_speed + self.layer_routing_offset(index, "speed", 0.25, 4.0)).clamp(0.25, 4.0);

        macro_rules! apply {
            ($field:ident, $suffix:literal, $min:expr, $max:expr) => {
                effects.$field = (effects.$field
                    + self.layer_routing_offset(index, $suffix, $min, $max))
                .clamp($min, $max);
            };
        }
        apply!(key_threshold, "key", 0.0, 1.0);
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

        LayerModulation {
            opacity,
            speed,
            effects,
        }
    }

    /// Summed modulation offset for one named target — for values the app
    /// applies itself (e.g. the morph crossfader) rather than via
    /// `modulate`. Uses the same depth × half-range scaling.
    pub fn target_offset(&self, target: &str) -> f32 {
        let Some((min, max)) = target_range(target) else {
            return 0.0;
        };
        self.routing_offset(target, min, max)
    }

    fn routing_offset(&self, target: &str, min: f32, max: f32) -> f32 {
        let half_range = (max - min) * 0.5;
        self.routings
            .iter()
            .filter(|routing| routing.target == target && routing.depth != 0.0)
            .map(|routing| routing.cached * routing.depth * half_range)
            .sum()
    }

    fn layer_routing_offset(&self, index: usize, suffix: &str, min: f32, max: f32) -> f32 {
        self.routing_offset(&format!("layer{}_{suffix}", index + 1), min, max)
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

    /// Reset LFOs and routings; tempo is left alone (losing a dialed-in
    /// BPM mid-set would be crueler than any stale routing).
    pub fn reset(&mut self) {
        self.lfos = std::array::from_fn(|_| Lfo::default());
        self.routings.clear();
    }
}

fn finite_or(value: f32, fallback: f32) -> f32 {
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
        "ntsc_snow" => &mut np.snow_intensity,
        "ntsc_tracking_snow" => &mut np.tracking_noise_snow,
        "ntsc_edge_wave" => &mut np.edge_wave_intensity,
        "ntsc_head_shift" => &mut np.head_switching_shift,
        "ntsc_chroma_loss" => &mut np.chroma_loss,
        "ntsc_luma_noise" => &mut np.luma_noise_intensity,
        "temporal_feedback" => &mut tp.feedback,
        "temporal_slitscan" => &mut tp.slitscan,
        "temporal_fb_zoom" => &mut tp.fb_zoom,
        "temporal_fb_rotate" => &mut tp.fb_rotate,
        "temporal_slit_angle" => &mut tp.slit_angle,
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
        let _ = matrix.modulate_layer_full(0, &EffectUniforms::default(), 1.0, 1.0);
        approx(matrix.routings[1].cached, cached);

        matrix.remove_routing(0);
        approx(matrix.routings[0].cached, cached);
        matrix.add_routing();
        approx(matrix.routings[1].cached, 0.0);
        matrix.routings[0].reset_runtime();
        approx(matrix.routings[0].cached, 0.0);
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
    fn pad_quantize_and_spring_are_deterministic() {
        let mut matrix = ModMatrix::new();
        matrix.pad_config.axes[0].quantize = 4;
        matrix.set_pad(0.74, 0.9, true);
        approx(matrix.source_value(ModSource::PadX), 2.0 / 3.0);

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
        assert_eq!(
            quantized,
            vec![0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 1.0, 1.0]
        );

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
        let modulated = matrix.modulate_layer_full(15, &base, 1.0, 1.0);
        approx(modulated.effects.brightness, 1.0);
        approx(base.brightness, 0.0);
    }
}
