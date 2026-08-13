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

use crate::audio::AudioLevels;
use crate::effects::params::TemporalParams;
use crate::effects::EffectUniforms;
use crate::ntsc::NtscParams;

pub const NUM_LFOS: usize = 4;
pub const MAX_ROUTINGS: usize = 16;

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
    ("vignette", 0.0, 1.5),
    ("color_drift", 0.0, 0.02),
    ("breathe_scale", 0.0, 0.05),
    ("breathe_rotation", 0.0, 2.0),
    ("breathe_position", 0.0, 0.02),
    ("ntsc_snow", 0.0, 1.0),
    ("ntsc_tracking_snow", 0.0, 1.0),
    ("ntsc_edge_wave", 0.0, 20.0),
    ("ntsc_head_shift", -100.0, 100.0),
    ("ntsc_chroma_loss", 0.0, 0.01),
    ("ntsc_luma_noise", 0.0, 0.2),
    ("layer1_opacity", 0.0, 1.0),
    ("layer2_opacity", 0.0, 1.0),
    ("layer3_opacity", 0.0, 1.0),
    ("layer4_opacity", 0.0, 1.0),
    ("layer1_speed", 0.25, 4.0),
    ("layer2_speed", 0.25, 4.0),
    ("layer3_speed", 0.25, 4.0),
    ("layer4_speed", 0.25, 4.0),
    ("layer1_key", 0.0, 1.0),
    ("layer2_key", 0.0, 1.0),
    ("layer3_key", 0.0, 1.0),
    ("layer4_key", 0.0, 1.0),
    ("temporal_feedback", 0.0, 0.95),
    ("temporal_slitscan", 0.0, 1.0),
    ("temporal_fb_zoom", 0.9, 1.1),
    ("temporal_fb_rotate", -5.0, 5.0),
];

/// Per-layer values after modulation, aligned with the layers vec.
#[derive(Debug, Clone, Copy)]
pub struct LayerModulation {
    pub opacity: f32,
    pub speed: f32,
    pub key_threshold: f32,
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
    /// Bipolar output in [-1, 1] at the given global beat position.
    pub fn value(&self, beat: f64, lfo_index: usize) -> f32 {
        let beats = self.beats.max(0.0625) as f64;
        let cycles = beat / beats + self.phase as f64;
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
    pub fn from_str(s: &str) -> Self {
        match s {
            "lfo1" => Self::Lfo(1),
            "lfo2" => Self::Lfo(2),
            "lfo3" => Self::Lfo(3),
            "audio_level" => Self::AudioLevel,
            "audio_bass" => Self::AudioBass,
            "audio_mid" => Self::AudioMid,
            "audio_high" => Self::AudioHigh,
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
            _ => Self::Lfo(0),
        }
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

/// A single modulation routing: source → parameter, scaled by depth.
#[derive(Debug, Clone)]
pub struct Routing {
    pub source: ModSource,
    pub target: String,
    pub depth: f32,
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
    /// Latest audio levels (pushed by the app each frame from the analyzer).
    pub audio: AudioLevels,
    /// Whether audio capture should be running (the app syncs the analyzer).
    pub audio_enabled: bool,
    /// Gain applied to normalized audio levels before routing.
    pub audio_gain: f32,
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
    /// XY performance pad [x, y], each 0..1. Touched from the web remote;
    /// holds its position when released, like a hardware pad.
    pub pad: [f32; 2],
}

impl ModMatrix {
    pub fn new() -> Self {
        Self {
            clock: Clock::new(),
            lfos: std::array::from_fn(|_| Lfo::default()),
            routings: Vec::new(),
            lfo_values: [0.0; NUM_LFOS],
            audio: AudioLevels::default(),
            audio_enabled: false,
            audio_gain: 1.0,
            midi: [0.0; NUM_MIDI_SLOTS],
            midi_enabled: false,
            // CC1 (mod wheel) and the common first knobs on most controllers.
            midi_ccs: [1, 2, 3, 4],
            midi_learn: None,
            midi_clock_sync: false,
            gyro: [0.5; 3],
            pad: [0.5; 2],
        }
    }

    /// Advance the clock and sample every LFO. Call once per frame.
    pub fn update(&mut self, now: Instant) {
        self.update_at_beat(self.clock.beat(now));
    }

    /// Sample every LFO at an explicit beat position. Used directly by the
    /// offline exporter, where the beat is derived from the frame index so
    /// the same patch renders the same file every time.
    pub fn update_at_beat(&mut self, beat: f64) {
        for (i, lfo) in self.lfos.iter().enumerate() {
            self.lfo_values[i] = lfo.value(beat, i);
        }
    }

    /// Current value of a modulation source.
    pub fn source_value(&self, source: ModSource) -> f32 {
        match source {
            ModSource::Lfo(i) => self.lfo_values[i.min(NUM_LFOS - 1)],
            ModSource::AudioLevel => self.audio.level,
            ModSource::AudioBass => self.audio.bass,
            ModSource::AudioMid => self.audio.mid,
            ModSource::AudioHigh => self.audio.high,
            ModSource::AudioOnset => self.audio.onset,
            ModSource::AudioBright => self.audio.bright,
            ModSource::AudioNoise => self.audio.noise,
            ModSource::Midi(i) => self.midi[i.min(NUM_MIDI_SLOTS - 1)],
            ModSource::GyroYaw => self.gyro[0],
            ModSource::GyroPitch => self.gyro[1],
            ModSource::GyroRoll => self.gyro[2],
            ModSource::PadX => self.pad[0],
            ModSource::PadY => self.pad[1],
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

        for routing in &self.routings {
            if routing.depth == 0.0 {
                continue;
            }
            let value = self.source_value(routing.source);
            let Some(&(_, min, max)) = TARGETS.iter().find(|(k, _, _)| *k == routing.target)
            else {
                continue;
            };
            let offset = value * routing.depth * (max - min) * 0.5;
            apply_offset(&mut fx, &mut np, &mut tp, &routing.target, offset, min, max);
        }

        (fx, np, tp)
    }

    /// Modulate one layer's routable values (index is 0-based; targets are
    /// named layer1..layer4). Bases come in, modulated copies come out.
    pub fn modulate_layer(
        &self,
        index: usize,
        base_opacity: f32,
        base_speed: f32,
        base_key: f32,
    ) -> LayerModulation {
        let mut result = LayerModulation {
            opacity: base_opacity,
            speed: base_speed,
            key_threshold: base_key,
        };
        let prefix = format!("layer{}_", index + 1);

        for routing in &self.routings {
            if routing.depth == 0.0 || !routing.target.starts_with(&prefix) {
                continue;
            }
            let value = self.source_value(routing.source);
            let Some(&(_, min, max)) = TARGETS.iter().find(|(k, _, _)| *k == routing.target)
            else {
                continue;
            };
            let offset = value * routing.depth * (max - min) * 0.5;
            let slot = match &routing.target[prefix.len()..] {
                "opacity" => &mut result.opacity,
                "speed" => &mut result.speed,
                "key" => &mut result.key_threshold,
                _ => continue,
            };
            *slot = (*slot + offset).clamp(min, max);
        }
        result
    }

    pub fn add_routing(&mut self) {
        if self.routings.len() < MAX_ROUTINGS {
            self.routings.push(Routing {
                source: ModSource::Lfo(0),
                target: "rgb_split".to_string(),
                depth: 0.0,
            });
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
        "vignette" => &mut fx.vignette,
        "color_drift" => &mut fx.color_drift,
        "breathe_scale" => &mut fx.breathe_scale,
        "breathe_rotation" => &mut fx.breathe_rotation,
        "breathe_position" => &mut fx.breathe_position,
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
        _ => return,
    };
    *slot = (*slot + offset).clamp(min, max);
}
