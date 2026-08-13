pub mod editor;

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::effects::params::TemporalParams;
use crate::effects::EffectUniforms;
use crate::layers::{BlendMode, Layer};
use crate::modulation::{
    Curve, GyroAxisConfig, Lfo, LfoShape, ModMatrix, ModSource, PadAxisConfig, PadConfig, Routing,
    MAX_ROUTINGS, NUM_LFOS,
};
use crate::ntsc::NtscParams;

// --- Helpers for serde defaults ---

fn one() -> f32 {
    1.0
}
fn default_fps() -> f32 {
    30.0
}
fn default_cellular_scale() -> f32 {
    10.0
}
fn default_cellular_warp() -> f32 {
    0.35
}
fn default_cellular_speed() -> f32 {
    0.25
}
fn default_cellular_gap_threshold() -> f32 {
    0.65
}
fn default_cellular_gap_softness() -> f32 {
    0.08
}
fn default_key_color() -> [f32; 3] {
    [0.0, 1.0, 0.0]
}
fn default_key_tolerance() -> f32 {
    0.15
}
fn default_temporal_key_threshold() -> f32 {
    0.1
}
fn default_temporal_key_softness() -> f32 {
    0.03
}

// --- Parameter metadata for stepping & comments ---

pub struct ParamMeta {
    pub step: f32,
    pub min: f32,
    pub max: f32,
    pub desc: &'static str,
}

pub fn param_meta(name: &str) -> Option<ParamMeta> {
    match name {
        "pixelate" => Some(ParamMeta {
            step: 1.0,
            min: 1.0,
            max: 32.0,
            desc: "pixel block size",
        }),
        "rgb_split" => Some(ParamMeta {
            step: 0.5,
            min: 0.0,
            max: 30.0,
            desc: "chromatic split px",
        }),
        "hue_shift" => Some(ParamMeta {
            step: 5.0,
            min: -180.0,
            max: 180.0,
            desc: "degrees",
        }),
        "saturation" => Some(ParamMeta {
            step: 0.05,
            min: -1.0,
            max: 1.0,
            desc: "color intensity",
        }),
        "brightness" => Some(ParamMeta {
            step: 0.05,
            min: -1.0,
            max: 1.0,
            desc: "exposure",
        }),
        "contrast" => Some(ParamMeta {
            step: 0.05,
            min: -1.0,
            max: 1.0,
            desc: "dynamic range",
        }),
        "posterize" => Some(ParamMeta {
            step: 1.0,
            min: 0.0,
            max: 16.0,
            desc: "color levels (0=off)",
        }),
        "downsample" => Some(ParamMeta {
            step: 0.05,
            min: 0.05,
            max: 1.0,
            desc: "render resolution fraction",
        }),
        "cellular_amount" => Some(ParamMeta {
            step: 0.05,
            min: 0.0,
            max: 1.0,
            desc: "cellular effect mix",
        }),
        "cellular_scale" => Some(ParamMeta {
            step: 1.0,
            min: 2.0,
            max: 32.0,
            desc: "cells across frame height",
        }),
        "cellular_warp" => Some(ParamMeta {
            step: 0.05,
            min: 0.0,
            max: 1.0,
            desc: "bounded domain displacement",
        }),
        "cellular_speed" => Some(ParamMeta {
            step: 0.05,
            min: 0.0,
            max: 2.0,
            desc: "feature target epochs per second",
        }),
        "cellular_gap_amount" => Some(ParamMeta {
            step: 0.05,
            min: 0.0,
            max: 1.0,
            desc: "cell ridge transparency",
        }),
        "cellular_gap_threshold" => Some(ParamMeta {
            step: 0.05,
            min: 0.0,
            max: 1.0,
            desc: "ridge strength keyed out",
        }),
        "cellular_gap_softness" => Some(ParamMeta {
            step: 0.01,
            min: 0.0,
            max: 0.5,
            desc: "transparent gap edge feather",
        }),
        "grain_intensity" => Some(ParamMeta {
            step: 0.01,
            min: 0.0,
            max: 0.3,
            desc: "film grain amount",
        }),
        "grain_size" => Some(ParamMeta {
            step: 0.25,
            min: 1.0,
            max: 4.0,
            desc: "grain particle scale",
        }),
        "grain_algo" => Some(ParamMeta {
            step: 1.0,
            min: 0.0,
            max: 3.0,
            desc: "0=value 1=perlin 2=gaussian 3=salt&pepper",
        }),
        "breathe_scale" => Some(ParamMeta {
            step: 0.005,
            min: 0.0,
            max: 0.05,
            desc: "zoom oscillation",
        }),
        "breathe_rotation" => Some(ParamMeta {
            step: 0.1,
            min: 0.0,
            max: 2.0,
            desc: "rotation oscillation deg",
        }),
        "breathe_position" => Some(ParamMeta {
            step: 0.002,
            min: 0.0,
            max: 0.02,
            desc: "position drift",
        }),
        "vignette" => Some(ParamMeta {
            step: 0.05,
            min: 0.0,
            max: 1.5,
            desc: "edge darkening",
        }),
        "color_drift" => Some(ParamMeta {
            step: 0.002,
            min: 0.0,
            max: 0.02,
            desc: "chromatic aberration",
        }),
        "key_mode" => Some(ParamMeta {
            step: 1.0,
            min: 0.0,
            max: 4.0,
            desc: "0=off 1=bright 2=dark 3=remove chroma 4=keep chroma",
        }),
        "key_threshold" | "key_tolerance" | "key_color_r" | "key_color_g" | "key_color_b" => {
            Some(ParamMeta {
                step: 0.01,
                min: 0.0,
                max: 1.0,
                desc: "normalized key control",
            })
        }
        "key_softness" => Some(ParamMeta {
            step: 0.01,
            min: 0.0,
            max: 0.5,
            desc: "key edge feather",
        }),
        "opacity" => Some(ParamMeta {
            step: 0.05,
            min: 0.0,
            max: 1.0,
            desc: "layer transparency",
        }),
        "speed" => Some(ParamMeta {
            step: 0.25,
            min: 0.25,
            max: 4.0,
            desc: "playback multiplier",
        }),
        "fps" => Some(ParamMeta {
            step: 1.0,
            min: 1.0,
            max: 240.0,
            desc: "decode frame rate",
        }),
        _ => None,
    }
}

// --- Serializable patch state ---

#[derive(Serialize, Deserialize, Clone)]
pub struct PatchState {
    pub master: EffectsConfig,
    pub layers: Vec<LayerConfig>,
    #[serde(default)]
    pub master_paused: bool,
    #[serde(default)]
    pub ntsc: Option<NtscConfig>,
    #[serde(default)]
    pub modulation: Option<ModConfig>,
    #[serde(default)]
    pub temporal: Option<TemporalConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub morph: Option<crate::morph::MorphStateSnapshot>,
}

/// Serializable temporal (feedback/slit-scan) parameters for patch files.
#[derive(Serialize, Deserialize, Clone)]
pub struct TemporalConfig {
    #[serde(default)]
    pub feedback: f32,
    #[serde(default = "one")]
    pub fb_zoom: f32,
    #[serde(default)]
    pub fb_rotate: f32,
    #[serde(default)]
    pub slitscan: f32,
    #[serde(default)]
    pub slit_axis: f32,
    /// Arbitrary angle added after the original row/column-only format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slit_angle: Option<f32>,
    #[serde(default)]
    pub key_mode: u32,
    #[serde(default = "default_temporal_key_threshold")]
    pub key_threshold: f32,
    #[serde(default = "default_temporal_key_softness")]
    pub key_softness: f32,
    #[serde(default = "one")]
    pub key_history: f32,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self::from_params(&TemporalParams::default())
    }
}

impl TemporalConfig {
    pub fn from_params(p: &TemporalParams) -> Self {
        Self {
            feedback: p.feedback,
            fb_zoom: p.fb_zoom,
            fb_rotate: p.fb_rotate,
            slitscan: p.slitscan,
            slit_axis: p.slit_axis,
            slit_angle: Some(p.slit_angle),
            key_mode: p.key_mode as u32,
            key_threshold: p.key_threshold,
            key_softness: p.key_softness,
            key_history: p.key_history,
        }
    }

    pub fn to_params(&self) -> TemporalParams {
        TemporalParams {
            feedback: finite_or(self.feedback, 0.0).clamp(0.0, 0.95),
            fb_zoom: finite_or(self.fb_zoom, 1.0).clamp(0.9, 1.1),
            fb_rotate: finite_or(self.fb_rotate, 0.0).clamp(-5.0, 5.0),
            slitscan: finite_or(self.slitscan, 0.0).clamp(0.0, 1.0),
            slit_angle: self
                .slit_angle
                .map(|angle| finite_or(angle, 0.0).clamp(-180.0, 180.0))
                .unwrap_or_else(|| finite_or(self.slit_axis, 0.0).clamp(0.0, 1.0) * 90.0),
            slit_axis: finite_or(self.slit_axis, 0.0).clamp(0.0, 1.0),
            key_mode: self.key_mode.min(4) as f32,
            key_threshold: finite_or(self.key_threshold, 0.1).clamp(0.0, 1.0),
            key_softness: finite_or(self.key_softness, 0.03).clamp(0.0, 0.5),
            key_history: finite_or(self.key_history, 1.0).round().clamp(1.0, 23.0),
        }
    }
}

/// Serializable modulation matrix state for patch files.
#[derive(Serialize, Deserialize, Clone)]
pub struct ModConfig {
    #[serde(default = "default_bpm")]
    pub bpm: f32,
    #[serde(default)]
    pub lfos: Vec<LfoConfig>,
    #[serde(default)]
    pub routings: Vec<RoutingConfig>,
    #[serde(default)]
    pub audio_enabled: bool,
    #[serde(default = "one")]
    pub audio_gain: f32,
    #[serde(default)]
    pub audio_device: String,
    #[serde(default = "default_audio_source_kind")]
    pub audio_source_kind: String,
    #[serde(default)]
    pub audio_clip_path: String,
    /// Number of routable FFT bands. Older patches omit this and remain at
    /// the historical three-band layout.
    #[serde(default = "default_audio_band_count")]
    pub audio_band_count: usize,
    /// Ordered crossovers. Current patches store exactly count - 1 entries.
    /// Legacy patches stored `[bass, mid, analysis_ceiling]`; application
    /// migrates that third value into `audio_band_ceiling_hz`.
    #[serde(default = "default_audio_band_edges")]
    pub audio_band_edges: Vec<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_band_ceiling_hz: Option<f32>,
    #[serde(default)]
    pub midi_enabled: bool,
    #[serde(default = "default_midi_ccs")]
    pub midi_ccs: Vec<u8>,
    #[serde(default)]
    pub midi_clock_sync: bool,
    #[serde(default = "default_gyro_axes")]
    pub gyro: Vec<GyroAxisPatchConfig>,
    /// Latest DeviceOrientation sample in degrees. Older patches omit this;
    /// those load centered on their saved calibration instead of inventing
    /// an offset from a zero-valued sample.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gyro_raw: Option<Vec<f32>>,
    #[serde(default)]
    pub pad: PadPatchConfig,
    /// Saved XY gesture position. A loaded/exported patch owns no live
    /// pointer, so spring return (when enabled) resumes from this position.
    #[serde(default = "default_pad_position")]
    pub pad_position: Vec<f32>,
}

fn default_midi_ccs() -> Vec<u8> {
    vec![1, 2, 3, 4]
}

fn default_bpm() -> f32 {
    120.0
}

fn default_audio_band_edges() -> Vec<f32> {
    vec![250.0, 2000.0, 8000.0]
}

fn default_audio_band_count() -> usize {
    3
}

fn default_audio_source_kind() -> String {
    crate::modulation::AUDIO_SOURCE_LIVE.to_string()
}

fn four() -> f32 {
    4.0
}

fn default_gyro_range() -> f32 {
    90.0
}

fn default_gyro_axes() -> Vec<GyroAxisPatchConfig> {
    vec![
        GyroAxisPatchConfig {
            range_degrees: 180.0,
            ..Default::default()
        },
        GyroAxisPatchConfig::default(),
        GyroAxisPatchConfig::default(),
    ]
}

fn default_pad_position() -> Vec<f32> {
    vec![0.5, 0.5]
}

#[derive(Serialize, Deserialize, Clone)]
pub struct LfoConfig {
    #[serde(default = "default_shape")]
    pub shape: String,
    #[serde(default = "default_beats")]
    pub beats: f32,
    #[serde(default)]
    pub phase: f32,
}

fn default_shape() -> String {
    "sine".to_string()
}
fn default_beats() -> f32 {
    4.0
}

#[derive(Serialize, Deserialize, Clone)]
pub struct RoutingConfig {
    pub source: String,
    pub target: String,
    #[serde(default)]
    pub depth: f32,
    #[serde(default = "default_curve")]
    pub curve: String,
    #[serde(default)]
    pub curve_amount: f32,
    #[serde(default)]
    pub attack: f32,
    #[serde(default)]
    pub release: f32,
}

fn default_curve() -> String {
    "linear".to_string()
}

#[derive(Serialize, Deserialize, Clone)]
pub struct GyroAxisPatchConfig {
    #[serde(default)]
    pub center_degrees: f32,
    #[serde(default = "default_gyro_range")]
    pub range_degrees: f32,
    #[serde(default)]
    pub expo: f32,
    #[serde(default)]
    pub invert: bool,
}

impl Default for GyroAxisPatchConfig {
    fn default() -> Self {
        Self {
            center_degrees: 0.0,
            range_degrees: 90.0,
            expo: 0.0,
            invert: false,
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct PadAxisPatchConfig {
    #[serde(default = "default_curve")]
    pub curve: String,
    #[serde(default)]
    pub curve_amount: f32,
    #[serde(default)]
    pub quantize: u32,
}

impl Default for PadAxisPatchConfig {
    fn default() -> Self {
        Self {
            curve: default_curve(),
            curve_amount: 0.0,
            quantize: 0,
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct PadPatchConfig {
    #[serde(default)]
    pub x: PadAxisPatchConfig,
    #[serde(default)]
    pub y: PadAxisPatchConfig,
    #[serde(default)]
    pub spring_enabled: bool,
    #[serde(default = "four")]
    pub spring_rate: f32,
}

impl Default for PadPatchConfig {
    fn default() -> Self {
        Self {
            x: PadAxisPatchConfig::default(),
            y: PadAxisPatchConfig::default(),
            spring_enabled: false,
            spring_rate: 4.0,
        }
    }
}

impl ModConfig {
    pub fn from_matrix(m: &ModMatrix) -> Self {
        Self {
            bpm: m.clock.bpm,
            lfos: m
                .lfos
                .iter()
                .map(|l| LfoConfig {
                    shape: l.shape.as_str().to_string(),
                    beats: l.beats,
                    phase: l.normalized_phase(),
                })
                .collect(),
            routings: m
                .routings
                .iter()
                .map(|r| RoutingConfig {
                    source: r.source.as_str().to_string(),
                    target: r.target().to_owned(),
                    depth: r.depth,
                    curve: r.curve.as_str().to_string(),
                    curve_amount: r.curve_amount,
                    attack: r.attack,
                    release: r.release,
                })
                .collect(),
            audio_enabled: m.audio_enabled,
            audio_gain: m.audio_gain,
            audio_device: m.audio_device.clone(),
            audio_source_kind: crate::modulation::normalize_audio_source_kind(&m.audio_source_kind)
                .to_string(),
            audio_clip_path: m.audio_clip_path.clone(),
            audio_band_count: m.audio_band_config.count(),
            audio_band_edges: m.audio_band_config.crossovers().to_vec(),
            audio_band_ceiling_hz: Some(m.audio_band_config.ceiling_hz()),
            midi_enabled: m.midi_enabled,
            midi_ccs: m.midi_ccs.to_vec(),
            midi_clock_sync: m.midi_clock_sync,
            gyro: m
                .gyro_config
                .iter()
                .map(|cfg| GyroAxisPatchConfig {
                    center_degrees: cfg.center_degrees,
                    range_degrees: cfg.range_degrees,
                    expo: cfg.expo,
                    invert: cfg.invert,
                })
                .collect(),
            gyro_raw: Some(m.gyro_raw.to_vec()),
            pad: PadPatchConfig {
                x: PadAxisPatchConfig::from_axis(m.pad_config.axes[0]),
                y: PadAxisPatchConfig::from_axis(m.pad_config.axes[1]),
                spring_enabled: m.pad_config.spring_enabled,
                spring_rate: m.pad_config.spring_rate,
            },
            pad_position: m.pad.to_vec(),
        }
    }

    pub fn apply_to_matrix(&self, m: &mut ModMatrix) {
        m.clock.set_bpm(finite_or(self.bpm, 120.0));
        m.lfos = std::array::from_fn(|_| Lfo::default());
        for (i, cfg) in self.lfos.iter().take(NUM_LFOS).enumerate() {
            let mut lfo = Lfo {
                shape: LfoShape::from_str(&cfg.shape),
                beats: finite_or(cfg.beats, 4.0).clamp(0.0625, 64.0),
                phase: 0.0,
            };
            lfo.set_phase(cfg.phase);
            m.lfos[i] = lfo;
        }
        // If a transitional patch happens to contain both spellings, the
        // canonical route wins and the legacy alias is ignored rather than
        // applying the same semantic destination twice.
        let canonical_key_targets: HashSet<String> = self
            .routings
            .iter()
            .take(MAX_ROUTINGS)
            .filter(|routing| {
                routing.target.starts_with("layer") && routing.target.ends_with("_key_threshold")
            })
            .map(|routing| routing.target.clone())
            .collect();
        m.routings = self
            .routings
            .iter()
            .take(MAX_ROUTINGS)
            .filter_map(|r| {
                let source = ModSource::try_from_str(&r.source)?;
                let target = crate::modulation::canonical_target(&r.target);
                if target.as_ref() != r.target && canonical_key_targets.contains(target.as_ref()) {
                    return None;
                }
                if !crate::modulation::is_valid_target(target.as_ref()) {
                    return None;
                }
                let mut routing = Routing::new(
                    source,
                    target.as_ref(),
                    finite_or(r.depth, 0.0).clamp(-1.0, 1.0),
                );
                routing.curve = Curve::from_str(&r.curve);
                routing.curve_amount = finite_or(r.curve_amount, 0.0).clamp(-2.0, 2.0);
                routing.attack = finite_or(r.attack, 0.0).clamp(0.0, 10.0);
                routing.release = finite_or(r.release, 0.0).clamp(0.0, 10.0);
                Some(routing)
            })
            .collect();
        m.audio_enabled = self.audio_enabled;
        m.audio_gain = finite_or(self.audio_gain, 1.0).clamp(0.0, 8.0);
        m.audio_device = self.audio_device.clone();
        m.audio_source_kind =
            crate::modulation::normalize_audio_source_kind(&self.audio_source_kind).to_string();
        m.audio_clip_path = self.audio_clip_path.clone();
        let count = self
            .audio_band_count
            .clamp(crate::audio::MIN_AUDIO_BANDS, crate::audio::MAX_AUDIO_BANDS);
        let (crossovers, ceiling_hz) = match self.audio_band_ceiling_hz {
            Some(ceiling) => (self.audio_band_edges.as_slice(), ceiling),
            None if self.audio_band_edges.len() >= count => (
                &self.audio_band_edges[..count - 1],
                self.audio_band_edges[count - 1],
            ),
            None => (self.audio_band_edges.as_slice(), 8000.0),
        };
        m.audio_band_config = crate::audio::AudioBandConfig::new(count, crossovers, ceiling_hz);
        m.midi_enabled = self.midi_enabled;
        m.midi_ccs = [1, 2, 3, 4];
        for (i, &cc) in self.midi_ccs.iter().take(m.midi_ccs.len()).enumerate() {
            m.midi_ccs[i] = cc & 0x7F;
        }
        m.midi_clock_sync = self.midi_clock_sync;

        let defaults = default_gyro_axes();
        for i in 0..3 {
            let cfg = self.gyro.get(i).or_else(|| defaults.get(i)).unwrap();
            m.gyro_config[i] = GyroAxisConfig {
                center_degrees: finite_or(cfg.center_degrees, 0.0),
                range_degrees: finite_or(cfg.range_degrees, 90.0).abs().clamp(1.0, 360.0),
                expo: finite_or(cfg.expo, 0.0).clamp(-2.0, 2.0),
                invert: cfg.invert,
            };
        }
        for i in 0..3 {
            m.gyro_raw[i] = self
                .gyro_raw
                .as_ref()
                .and_then(|values| values.get(i))
                .copied()
                .filter(|value| value.is_finite())
                .unwrap_or(m.gyro_config[i].center_degrees);
        }
        m.pad_config = PadConfig {
            axes: [self.pad.x.to_axis(), self.pad.y.to_axis()],
            spring_enabled: self.pad.spring_enabled,
            spring_rate: finite_or(self.pad.spring_rate, 4.0).clamp(0.1, 20.0),
        };
        for i in 0..2 {
            m.pad[i] = self
                .pad_position
                .get(i)
                .copied()
                .filter(|value| value.is_finite())
                .unwrap_or(0.5)
                .clamp(0.0, 1.0);
        }
        // A saved patch has no owning browser pointer. Marking it released is
        // what lets deterministic spring return advance in live and export.
        m.pad_active = false;
        m.recompute_gyro();
    }
}

fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

impl PadAxisPatchConfig {
    fn from_axis(axis: PadAxisConfig) -> Self {
        Self {
            curve: axis.curve.as_str().to_string(),
            curve_amount: axis.curve_amount,
            quantize: axis.quantize,
        }
    }

    fn to_axis(&self) -> PadAxisConfig {
        PadAxisConfig {
            curve: Curve::from_str(&self.curve),
            curve_amount: finite_or(self.curve_amount, 0.0).clamp(-2.0, 2.0),
            quantize: self.quantize.min(64),
        }
    }
}

/// Serializable NTSC/VHS effect parameters for patch files.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct NtscConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub tape_speed: u32,
    #[serde(default)]
    pub chroma_loss: f32,
    #[serde(default)]
    pub edge_wave_enabled: bool,
    #[serde(default)]
    pub edge_wave_intensity: f32,
    #[serde(default = "default_edge_wave_speed")]
    pub edge_wave_speed: f32,
    #[serde(default)]
    pub head_switching_enabled: bool,
    #[serde(default = "default_head_height")]
    pub head_switching_height: i32,
    #[serde(default)]
    pub head_switching_shift: f32,
    #[serde(default)]
    pub tracking_noise_enabled: bool,
    #[serde(default = "default_tracking_height")]
    pub tracking_noise_height: i32,
    #[serde(default)]
    pub tracking_noise_wave: f32,
    #[serde(default)]
    pub tracking_noise_snow: f32,
    #[serde(default)]
    pub snow_intensity: f32,
    #[serde(default)]
    pub composite_noise_intensity: f32,
    #[serde(default)]
    pub luma_noise_intensity: f32,
    #[serde(default)]
    pub chroma_noise_intensity: f32,
    #[serde(default)]
    pub luma_smear: f32,
    #[serde(default)]
    pub composite_sharpening: f32,
}

fn default_edge_wave_speed() -> f32 {
    0.5
}
fn default_head_height() -> i32 {
    8
}
fn default_tracking_height() -> i32 {
    24
}

impl NtscConfig {
    pub fn from_params(p: &NtscParams) -> Self {
        Self {
            enabled: p.enabled,
            tape_speed: p.tape_speed,
            chroma_loss: p.chroma_loss,
            edge_wave_enabled: p.edge_wave_enabled,
            edge_wave_intensity: p.edge_wave_intensity,
            edge_wave_speed: p.edge_wave_speed,
            head_switching_enabled: p.head_switching_enabled,
            head_switching_height: p.head_switching_height,
            head_switching_shift: p.head_switching_shift,
            tracking_noise_enabled: p.tracking_noise_enabled,
            tracking_noise_height: p.tracking_noise_height,
            tracking_noise_wave: p.tracking_noise_wave,
            tracking_noise_snow: p.tracking_noise_snow,
            snow_intensity: p.snow_intensity,
            composite_noise_intensity: p.composite_noise_intensity,
            luma_noise_intensity: p.luma_noise_intensity,
            chroma_noise_intensity: p.chroma_noise_intensity,
            luma_smear: p.luma_smear,
            composite_sharpening: p.composite_sharpening,
        }
    }

    pub fn to_params(&self) -> NtscParams {
        let finite = |value: f32, fallback: f32| {
            if value.is_finite() {
                value
            } else {
                fallback
            }
        };
        NtscParams {
            enabled: self.enabled,
            tape_speed: self.tape_speed.min(2),
            chroma_loss: finite(self.chroma_loss, 0.0).clamp(0.0, 0.01),
            edge_wave_enabled: self.edge_wave_enabled,
            edge_wave_intensity: finite(self.edge_wave_intensity, 0.0).clamp(0.0, 20.0),
            edge_wave_speed: finite(self.edge_wave_speed, 0.5).clamp(0.0, 10.0),
            head_switching_enabled: self.head_switching_enabled,
            head_switching_height: self.head_switching_height.clamp(0, 24),
            head_switching_shift: finite(self.head_switching_shift, 0.0).clamp(-100.0, 100.0),
            tracking_noise_enabled: self.tracking_noise_enabled,
            tracking_noise_height: self.tracking_noise_height.clamp(0, 120),
            tracking_noise_wave: finite(self.tracking_noise_wave, 0.0).clamp(0.0, 50.0),
            tracking_noise_snow: finite(self.tracking_noise_snow, 0.0).clamp(0.0, 1.0),
            snow_intensity: finite(self.snow_intensity, 0.0).clamp(0.0, 1.0),
            composite_noise_intensity: finite(self.composite_noise_intensity, 0.0).clamp(0.0, 0.5),
            luma_noise_intensity: finite(self.luma_noise_intensity, 0.0).clamp(0.0, 0.2),
            chroma_noise_intensity: finite(self.chroma_noise_intensity, 0.0).clamp(0.0, 0.5),
            luma_smear: finite(self.luma_smear, 0.0).clamp(0.0, 1.0),
            composite_sharpening: finite(self.composite_sharpening, 0.0).clamp(-1.0, 2.0),
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct LayerConfig {
    pub filename: String,
    /// Stable identity for sources loaded outside the current library. File
    /// layers store their canonical path; live receivers use
    /// `spout://<sender-name>`. Old patches omit this and continue resolving
    /// video sources by filename.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source_path: String,
    #[serde(default = "one")]
    pub opacity: f32,
    #[serde(default = "default_blend")]
    pub blend_mode: String,
    #[serde(default = "one")]
    pub speed: f32,
    #[serde(default = "default_fps")]
    pub fps: f32,
    #[serde(default)]
    pub paused: bool,
    #[serde(default = "default_true")]
    pub visible: bool,
    /// Skip the shared master shader for this layer. Missing in legacy
    /// patches means the historical behavior: master effects remain active.
    #[serde(default)]
    pub bypass_master_fx: bool,
    #[serde(default)]
    pub effects: EffectsConfig,
}

fn default_blend() -> String {
    "normal".to_string()
}
fn default_true() -> bool {
    true
}
#[derive(Serialize, Deserialize, Clone)]
pub struct EffectsConfig {
    #[serde(default = "one")]
    pub pixelate: f32,
    #[serde(default)]
    pub rgb_split: f32,
    #[serde(default)]
    pub hue_shift: f32,
    #[serde(default)]
    pub saturation: f32,
    #[serde(default)]
    pub brightness: f32,
    #[serde(default)]
    pub contrast: f32,
    #[serde(default)]
    pub posterize: f32,
    #[serde(default)]
    pub invert: bool,
    /// Fraction of full render resolution (1.0 = full resolution).
    #[serde(default = "one")]
    pub downsample: f32,
    #[serde(default)]
    pub grain_intensity: f32,
    #[serde(default = "one")]
    pub grain_size: f32,
    #[serde(default)]
    pub grain_algo: u32,
    #[serde(default)]
    pub color_grain: bool,
    #[serde(default)]
    pub breathe_scale: f32,
    #[serde(default)]
    pub breathe_rotation: f32,
    #[serde(default)]
    pub breathe_position: f32,
    #[serde(default)]
    pub vignette: f32,
    #[serde(default)]
    pub color_drift: f32,
    #[serde(default)]
    pub key_mode: u32,
    #[serde(default = "default_key_threshold")]
    pub key_threshold: f32,
    #[serde(default = "default_key_softness")]
    pub key_softness: f32,
    #[serde(default = "default_key_color")]
    pub key_color: [f32; 3],
    #[serde(default = "default_key_tolerance")]
    pub key_tolerance: f32,
    #[serde(default)]
    pub cellular_amount: f32,
    #[serde(default = "default_cellular_scale")]
    pub cellular_scale: f32,
    #[serde(default = "default_cellular_warp")]
    pub cellular_warp: f32,
    #[serde(default = "default_cellular_speed")]
    pub cellular_speed: f32,
    #[serde(default)]
    pub cellular_gap_amount: f32,
    #[serde(default = "default_cellular_gap_threshold")]
    pub cellular_gap_threshold: f32,
    #[serde(default = "default_cellular_gap_softness")]
    pub cellular_gap_softness: f32,
}

fn default_key_threshold() -> f32 {
    0.5
}
fn default_key_softness() -> f32 {
    0.1
}

impl Default for EffectsConfig {
    fn default() -> Self {
        Self {
            pixelate: 1.0,
            rgb_split: 0.0,
            hue_shift: 0.0,
            saturation: 0.0,
            brightness: 0.0,
            contrast: 0.0,
            posterize: 0.0,
            invert: false,
            downsample: 1.0,
            grain_intensity: 0.0,
            grain_size: 1.0,
            grain_algo: 0,
            color_grain: false,
            breathe_scale: 0.0,
            breathe_rotation: 0.0,
            breathe_position: 0.0,
            vignette: 0.0,
            color_drift: 0.0,
            key_mode: 0,
            key_threshold: 0.5,
            key_softness: 0.1,
            key_color: default_key_color(),
            key_tolerance: default_key_tolerance(),
            cellular_amount: 0.0,
            cellular_scale: 10.0,
            cellular_warp: 0.35,
            cellular_speed: 0.25,
            cellular_gap_amount: 0.0,
            cellular_gap_threshold: 0.65,
            cellular_gap_softness: 0.08,
        }
    }
}

// --- Conversion: EffectUniforms <-> EffectsConfig ---

impl EffectsConfig {
    pub fn from_uniforms(u: &EffectUniforms) -> Self {
        Self {
            pixelate: u.pixelate_size,
            rgb_split: u.rgb_split,
            hue_shift: u.hue_shift,
            saturation: u.saturation,
            brightness: u.brightness,
            contrast: u.contrast,
            posterize: u.posterize,
            invert: u.invert > 0.5,
            downsample: u.downsample,
            grain_intensity: u.grain_intensity,
            grain_size: u.grain_size,
            grain_algo: u.grain_algo as u32,
            color_grain: u.color_grain > 0.5,
            breathe_scale: u.breathe_scale,
            breathe_rotation: u.breathe_rotation,
            breathe_position: u.breathe_position,
            vignette: u.vignette,
            color_drift: u.color_drift,
            key_mode: u.key_mode as u32,
            key_threshold: u.key_threshold,
            key_softness: u.key_softness,
            key_color: u.key_color,
            key_tolerance: u.key_tolerance,
            cellular_amount: u.cellular_amount,
            cellular_scale: u.cellular_scale,
            cellular_warp: u.cellular_warp,
            cellular_speed: u.cellular_speed,
            cellular_gap_amount: u.cellular_gap_amount,
            cellular_gap_threshold: u.cellular_gap_threshold,
            cellular_gap_softness: u.cellular_gap_softness,
        }
    }

    pub fn apply_to_uniforms(&self, u: &mut EffectUniforms) {
        u.pixelate_size = finite_or(self.pixelate, 1.0).clamp(1.0, 32.0);
        u.rgb_split = finite_or(self.rgb_split, 0.0).clamp(0.0, 30.0);
        u.hue_shift = finite_or(self.hue_shift, 0.0).clamp(-180.0, 180.0);
        u.saturation = finite_or(self.saturation, 0.0).clamp(-1.0, 1.0);
        u.brightness = finite_or(self.brightness, 0.0).clamp(-1.0, 1.0);
        u.contrast = finite_or(self.contrast, 0.0).clamp(-1.0, 1.0);
        u.posterize = finite_or(self.posterize, 0.0).clamp(0.0, 16.0);
        u.invert = if self.invert { 1.0 } else { 0.0 };
        u.downsample = finite_or(self.downsample, 1.0).clamp(0.05, 1.0);
        u.grain_intensity = finite_or(self.grain_intensity, 0.0).clamp(0.0, 0.3);
        u.grain_size = finite_or(self.grain_size, 1.0).clamp(1.0, 4.0);
        u.grain_algo = (self.grain_algo.min(3)) as f32;
        u.color_grain = if self.color_grain { 1.0 } else { 0.0 };
        u.breathe_scale = finite_or(self.breathe_scale, 0.0).clamp(0.0, 0.05);
        u.breathe_rotation = finite_or(self.breathe_rotation, 0.0).clamp(0.0, 2.0);
        u.breathe_position = finite_or(self.breathe_position, 0.0).clamp(0.0, 0.02);
        u.vignette = finite_or(self.vignette, 0.0).clamp(0.0, 1.5);
        u.color_drift = finite_or(self.color_drift, 0.0).clamp(0.0, 0.02);
        u.key_mode = self.key_mode.min(4) as f32;
        u.key_threshold = finite_or(self.key_threshold, 0.5).clamp(0.0, 1.0);
        u.key_softness = finite_or(self.key_softness, 0.1).clamp(0.0, 0.5);
        u.key_color = [
            finite_or(self.key_color[0], 0.0).clamp(0.0, 1.0),
            finite_or(self.key_color[1], 1.0).clamp(0.0, 1.0),
            finite_or(self.key_color[2], 0.0).clamp(0.0, 1.0),
        ];
        u.key_tolerance = finite_or(self.key_tolerance, 0.15).clamp(0.0, 1.0);
        u.cellular_amount = finite_or(self.cellular_amount, 0.0).clamp(0.0, 1.0);
        u.cellular_scale = finite_or(self.cellular_scale, 10.0).clamp(2.0, 32.0);
        u.cellular_warp = finite_or(self.cellular_warp, 0.35).clamp(0.0, 1.0);
        u.cellular_speed = finite_or(self.cellular_speed, 0.25).clamp(0.0, 2.0);
        u.cellular_gap_amount = finite_or(self.cellular_gap_amount, 0.0).clamp(0.0, 1.0);
        u.cellular_gap_threshold = finite_or(self.cellular_gap_threshold, 0.65).clamp(0.0, 1.0);
        u.cellular_gap_softness = finite_or(self.cellular_gap_softness, 0.08).clamp(0.0, 0.5);
    }

    /// Get fields organized into groups for display.
    pub fn grouped_fields(&self) -> Vec<(&'static str, Vec<(&'static str, String)>)> {
        vec![
            (
                "digital",
                vec![
                    ("pixelate", format!("{:.1}", self.pixelate)),
                    ("rgb_split", format!("{:.1}", self.rgb_split)),
                    ("hue_shift", format!("{:.1}", self.hue_shift)),
                    ("saturation", format!("{:.2}", self.saturation)),
                    ("brightness", format!("{:.2}", self.brightness)),
                    ("contrast", format!("{:.2}", self.contrast)),
                    ("posterize", format!("{:.1}", self.posterize)),
                    ("invert", format!("{}", self.invert)),
                    ("downsample", format!("{:.2}", self.downsample)),
                ],
            ),
            (
                "cellular",
                vec![
                    ("cellular_amount", format!("{:.2}", self.cellular_amount)),
                    ("cellular_scale", format!("{:.1}", self.cellular_scale)),
                    ("cellular_warp", format!("{:.2}", self.cellular_warp)),
                    ("cellular_speed", format!("{:.2}", self.cellular_speed)),
                    (
                        "cellular_gap_amount",
                        format!("{:.2}", self.cellular_gap_amount),
                    ),
                    (
                        "cellular_gap_threshold",
                        format!("{:.2}", self.cellular_gap_threshold),
                    ),
                    (
                        "cellular_gap_softness",
                        format!("{:.2}", self.cellular_gap_softness),
                    ),
                ],
            ),
            (
                "analog",
                vec![
                    ("grain_intensity", format!("{:.2}", self.grain_intensity)),
                    ("grain_size", format!("{:.2}", self.grain_size)),
                    ("grain_algo", format!("{}", self.grain_algo)),
                    ("color_grain", format!("{}", self.color_grain)),
                    ("vignette", format!("{:.2}", self.vignette)),
                    ("color_drift", format!("{:.3}", self.color_drift)),
                ],
            ),
            (
                "motion",
                vec![
                    ("breathe_scale", format!("{:.3}", self.breathe_scale)),
                    ("breathe_rotation", format!("{:.2}", self.breathe_rotation)),
                    ("breathe_position", format!("{:.3}", self.breathe_position)),
                ],
            ),
            (
                "key",
                vec![
                    ("key_mode", format!("{}", self.key_mode)),
                    ("key_threshold", format!("{:.2}", self.key_threshold)),
                    ("key_softness", format!("{:.2}", self.key_softness)),
                    ("key_color_r", format!("{:.3}", self.key_color[0])),
                    ("key_color_g", format!("{:.3}", self.key_color[1])),
                    ("key_color_b", format!("{:.3}", self.key_color[2])),
                    ("key_tolerance", format!("{:.2}", self.key_tolerance)),
                ],
            ),
        ]
    }

    /// Set a single field by key name. Returns true if the key was recognized.
    pub fn set_field(&mut self, key: &str, value: &str) -> bool {
        match key {
            "pixelate" => {
                if let Ok(v) = value.parse() {
                    self.pixelate = v;
                    return true;
                }
            }
            "rgb_split" => {
                if let Ok(v) = value.parse() {
                    self.rgb_split = v;
                    return true;
                }
            }
            "hue_shift" => {
                if let Ok(v) = value.parse() {
                    self.hue_shift = v;
                    return true;
                }
            }
            "saturation" => {
                if let Ok(v) = value.parse() {
                    self.saturation = v;
                    return true;
                }
            }
            "brightness" => {
                if let Ok(v) = value.parse() {
                    self.brightness = v;
                    return true;
                }
            }
            "contrast" => {
                if let Ok(v) = value.parse() {
                    self.contrast = v;
                    return true;
                }
            }
            "posterize" => {
                if let Ok(v) = value.parse() {
                    self.posterize = v;
                    return true;
                }
            }
            "invert" => {
                if let Ok(v) = value.parse() {
                    self.invert = v;
                    return true;
                }
            }
            "downsample" => {
                if let Ok(v) = value.parse() {
                    self.downsample = v;
                    return true;
                }
            }
            "grain_intensity" => {
                if let Ok(v) = value.parse() {
                    self.grain_intensity = v;
                    return true;
                }
            }
            "grain_size" => {
                if let Ok(v) = value.parse() {
                    self.grain_size = v;
                    return true;
                }
            }
            "grain_algo" => {
                if let Ok(v) = value.parse() {
                    self.grain_algo = v;
                    return true;
                }
            }
            "color_grain" => {
                if let Ok(v) = value.parse() {
                    self.color_grain = v;
                    return true;
                }
            }
            "breathe_scale" => {
                if let Ok(v) = value.parse() {
                    self.breathe_scale = v;
                    return true;
                }
            }
            "breathe_rotation" => {
                if let Ok(v) = value.parse() {
                    self.breathe_rotation = v;
                    return true;
                }
            }
            "breathe_position" => {
                if let Ok(v) = value.parse() {
                    self.breathe_position = v;
                    return true;
                }
            }
            "vignette" => {
                if let Ok(v) = value.parse() {
                    self.vignette = v;
                    return true;
                }
            }
            "color_drift" => {
                if let Ok(v) = value.parse() {
                    self.color_drift = v;
                    return true;
                }
            }
            "key_mode" => {
                if let Ok(v) = value.parse() {
                    self.key_mode = v;
                    return true;
                }
            }
            "key_threshold" => {
                if let Ok(v) = value.parse() {
                    self.key_threshold = v;
                    return true;
                }
            }
            "key_softness" => {
                if let Ok(v) = value.parse() {
                    self.key_softness = v;
                    return true;
                }
            }
            "key_color_r" | "key_color_g" | "key_color_b" => {
                if let Ok(v) = value.parse() {
                    let index = match key {
                        "key_color_r" => 0,
                        "key_color_g" => 1,
                        _ => 2,
                    };
                    self.key_color[index] = v;
                    return true;
                }
            }
            "key_tolerance" => {
                if let Ok(v) = value.parse() {
                    self.key_tolerance = v;
                    return true;
                }
            }
            "cellular_amount" => {
                if let Ok(v) = value.parse() {
                    self.cellular_amount = v;
                    return true;
                }
            }
            "cellular_scale" => {
                if let Ok(v) = value.parse() {
                    self.cellular_scale = v;
                    return true;
                }
            }
            "cellular_warp" => {
                if let Ok(v) = value.parse() {
                    self.cellular_warp = v;
                    return true;
                }
            }
            "cellular_speed" => {
                if let Ok(v) = value.parse() {
                    self.cellular_speed = v;
                    return true;
                }
            }
            "cellular_gap_amount" => {
                if let Ok(v) = value.parse() {
                    self.cellular_gap_amount = v;
                    return true;
                }
            }
            "cellular_gap_threshold" => {
                if let Ok(v) = value.parse() {
                    self.cellular_gap_threshold = v;
                    return true;
                }
            }
            "cellular_gap_softness" => {
                if let Ok(v) = value.parse() {
                    self.cellular_gap_softness = v;
                    return true;
                }
            }
            _ => {}
        }
        false
    }
}

// --- Conversion: Layer <-> LayerConfig ---

impl LayerConfig {
    pub fn from_layer(layer: &Layer) -> Self {
        Self {
            filename: layer.filename.clone(),
            source_path: layer.source_path.clone(),
            opacity: layer.opacity,
            blend_mode: layer.blend_mode.key().to_string(),
            speed: layer.speed,
            fps: layer.fps,
            paused: layer.paused,
            visible: layer.visible,
            bypass_master_fx: layer.bypass_master_fx,
            effects: EffectsConfig::from_uniforms(&layer.effects),
        }
    }

    pub fn apply_to_layer(&self, layer: &mut Layer) {
        layer.opacity = finite_or(self.opacity, 1.0).clamp(0.0, 1.0);
        layer.blend_mode = match self.blend_mode.as_str() {
            "screen" => BlendMode::Screen,
            "multiply" => BlendMode::Multiply,
            "difference" => BlendMode::Difference,
            _ => BlendMode::Normal,
        };
        layer.speed = finite_or(self.speed, 1.0).clamp(0.25, 4.0);
        layer.fps = finite_or(self.fps, 30.0).clamp(1.0, 240.0);
        layer.paused = self.paused;
        layer.visible = self.visible;
        layer.bypass_master_fx = self.bypass_master_fx;
        self.effects.apply_to_uniforms(&mut layer.effects);
    }

    /// Get top-level layer fields as (key, value_string) pairs.
    pub fn top_fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("filename", self.filename.clone()),
            ("opacity", format!("{:.2}", self.opacity)),
            ("blend_mode", self.blend_mode.clone()),
            ("speed", format!("{:.2}", self.speed)),
            ("fps", format!("{:.1}", self.fps)),
            ("paused", format!("{}", self.paused)),
            ("visible", format!("{}", self.visible)),
            ("bypass_master_fx", format!("{}", self.bypass_master_fx)),
        ]
    }

    /// Set a top-level field by key name. Returns true if recognized.
    pub fn set_field(&mut self, key: &str, value: &str) -> bool {
        match key {
            "opacity" => {
                if let Ok(v) = value.parse() {
                    self.opacity = v;
                    return true;
                }
            }
            "blend_mode" => {
                self.blend_mode = value.to_string();
                return true;
            }
            "speed" => {
                if let Ok(v) = value.parse() {
                    self.speed = v;
                    return true;
                }
            }
            "fps" => {
                if let Ok(v) = value.parse() {
                    self.fps = v;
                    return true;
                }
            }
            "paused" => {
                if let Ok(v) = value.parse() {
                    self.paused = v;
                    return true;
                }
            }
            "visible" => {
                if let Ok(v) = value.parse() {
                    self.visible = v;
                    return true;
                }
            }
            "bypass_master_fx" => {
                if let Ok(v) = value.parse() {
                    self.bypass_master_fx = v;
                    return true;
                }
            }
            _ => {}
        }
        false
    }
}

// --- Full patch snapshot ---

impl PatchState {
    pub fn capture(
        master: &EffectUniforms,
        layers: &[Layer],
        ntsc_params: &NtscParams,
        mod_matrix: &ModMatrix,
        temporal: &TemporalParams,
        master_paused: bool,
        morph: &crate::morph::Morph,
    ) -> Self {
        Self {
            master: EffectsConfig::from_uniforms(master),
            layers: layers.iter().map(LayerConfig::from_layer).collect(),
            master_paused,
            ntsc: Some(NtscConfig::from_params(ntsc_params)),
            modulation: Some(ModConfig::from_matrix(mod_matrix)),
            temporal: Some(TemporalConfig::from_params(temporal)),
            morph: Some(morph.snapshot_at_beat(mod_matrix.current_beat)),
        }
    }

    pub fn apply(
        &self,
        master: &mut EffectUniforms,
        layers: &mut [Layer],
        ntsc_params: &mut NtscParams,
        mod_matrix: &mut ModMatrix,
        temporal: &mut TemporalParams,
    ) {
        self.master.apply_to_uniforms(master);
        for (config, layer) in self.layers.iter().zip(layers.iter_mut()) {
            config.apply_to_layer(layer);
        }
        if let Some(ref ntsc) = self.ntsc {
            *ntsc_params = ntsc.to_params();
        }
        if let Some(ref modulation) = self.modulation {
            modulation.apply_to_matrix(mod_matrix);
        }
        if let Some(ref temporal_cfg) = self.temporal {
            *temporal = temporal_cfg.to_params();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_capture_rebases_in_flight_morph_to_remaining_beats() {
        let mut matrix = ModMatrix::new();
        matrix.update_at_beat(103.0, 0.0);
        let mut morph = crate::morph::Morph::default();
        morph.start_glide(1.0, 8.0, 100.0);

        let patch = PatchState::capture(
            &EffectUniforms::default(),
            &[],
            &NtscParams::default(),
            &matrix,
            &TemporalParams::default(),
            false,
            &morph,
        );
        let snapshot = patch.morph.unwrap();
        assert!((snapshot.t - 0.375).abs() < 1e-6);
        let glide = snapshot.glide.unwrap();
        assert_eq!(glide.start_beat, 0.0);
        assert_eq!(glide.duration_beats, 5.0);
        let restored = crate::morph::Morph::from_snapshot(snapshot);
        assert!((restored.position_at_beat(0.0) - 0.375).abs() < 1e-6);
        assert!((restored.position_at_beat(5.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn downsample_persists_for_master_and_layers_with_legacy_default() {
        let master = EffectUniforms {
            downsample: 0.35,
            ..Default::default()
        };
        let layer_effects = EffectsConfig {
            downsample: 0.6,
            ..Default::default()
        };
        let patch = PatchState {
            master: EffectsConfig::from_uniforms(&master),
            layers: vec![LayerConfig {
                filename: "clip.mp4".to_string(),
                source_path: String::new(),
                opacity: 1.0,
                blend_mode: "normal".to_string(),
                speed: 1.0,
                fps: 30.0,
                paused: false,
                visible: true,
                bypass_master_fx: true,
                effects: layer_effects,
            }],
            master_paused: false,
            ntsc: None,
            modulation: None,
            temporal: None,
            morph: None,
        };

        let yaml = serde_yaml::to_string(&patch).unwrap();
        assert!(yaml.contains("downsample: 0.35"));
        let parsed: PatchState = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed.master.downsample, 0.35);
        assert_eq!(parsed.layers[0].effects.downsample, 0.6);
        assert!(parsed.layers[0].bypass_master_fx);

        let mut restored = EffectUniforms::default();
        parsed.master.apply_to_uniforms(&mut restored);
        assert_eq!(restored.downsample, 0.35);

        let legacy: PatchState = serde_yaml::from_str(
            "master: {}\nlayers:\n  - filename: legacy.mp4\n    effects: {}\n",
        )
        .unwrap();
        assert_eq!(legacy.master.downsample, 1.0);
        assert_eq!(legacy.layers[0].effects.downsample, 1.0);
        assert!(!legacy.layers[0].bypass_master_fx);

        let mut invalid = EffectsConfig {
            downsample: f32::NAN,
            ..Default::default()
        };
        invalid.apply_to_uniforms(&mut restored);
        assert_eq!(restored.downsample, 1.0);
        invalid.downsample = -10.0;
        invalid.apply_to_uniforms(&mut restored);
        assert_eq!(restored.downsample, 0.05);

        assert!(invalid.set_field("downsample", "0.4"));
        assert_eq!(invalid.downsample, 0.4);
        assert!(invalid
            .grouped_fields()
            .iter()
            .flat_map(|(_, fields)| fields)
            .any(|(key, _)| *key == "downsample"));
        let metadata = param_meta("downsample").unwrap();
        assert_eq!((metadata.min, metadata.max), (0.05, 1.0));
    }

    #[test]
    fn layer_master_fx_bypass_round_trips_and_native_editor_exposes_it() {
        let legacy: LayerConfig = serde_yaml::from_str(
            "filename: legacy.mp4\nopacity: 1\nblend_mode: normal\neffects: {}\n",
        )
        .unwrap();
        assert!(!legacy.bypass_master_fx);

        let mut edited = legacy;
        assert!(edited.set_field("bypass_master_fx", "true"));
        assert!(edited.bypass_master_fx);
        assert!(edited
            .top_fields()
            .iter()
            .any(|(key, value)| *key == "bypass_master_fx" && value == "true"));

        let yaml = serde_yaml::to_string(&edited).unwrap();
        let restored: LayerConfig = serde_yaml::from_str(&yaml).unwrap();
        assert!(restored.bypass_master_fx);
    }

    #[test]
    fn cellular_controls_round_trip_sanitize_and_keep_legacy_defaults() {
        let configured = EffectsConfig {
            cellular_amount: 0.8,
            cellular_scale: 24.0,
            cellular_warp: 0.65,
            cellular_speed: 1.5,
            cellular_gap_amount: 0.9,
            cellular_gap_threshold: 0.4,
            cellular_gap_softness: 0.12,
            ..Default::default()
        };
        let yaml = serde_yaml::to_string(&configured).unwrap();
        let decoded: EffectsConfig = serde_yaml::from_str(&yaml).unwrap();
        let mut uniforms = EffectUniforms::default();
        decoded.apply_to_uniforms(&mut uniforms);
        assert_eq!(uniforms.cellular_amount, 0.8);
        assert_eq!(uniforms.cellular_scale, 24.0);
        assert_eq!(uniforms.cellular_warp, 0.65);
        assert_eq!(uniforms.cellular_speed, 1.5);
        assert_eq!(uniforms.cellular_gap_amount, 0.9);
        assert_eq!(uniforms.cellular_gap_threshold, 0.4);
        assert_eq!(uniforms.cellular_gap_softness, 0.12);

        let legacy: EffectsConfig = serde_yaml::from_str("pixelate: 4.0\n").unwrap();
        assert_eq!(legacy.cellular_amount, 0.0);
        assert_eq!(legacy.cellular_scale, 10.0);
        assert_eq!(legacy.cellular_warp, 0.35);
        assert_eq!(legacy.cellular_speed, 0.25);
        assert_eq!(legacy.cellular_gap_amount, 0.0);
        assert_eq!(legacy.cellular_gap_threshold, 0.65);
        assert_eq!(legacy.cellular_gap_softness, 0.08);

        let invalid: EffectsConfig = serde_yaml::from_str(
            "cellular_amount: .nan\ncellular_scale: -9\ncellular_warp: 7\ncellular_speed: .inf\ncellular_gap_amount: 9\ncellular_gap_threshold: -4\ncellular_gap_softness: .nan\n",
        )
        .unwrap();
        invalid.apply_to_uniforms(&mut uniforms);
        assert_eq!(uniforms.cellular_amount, 0.0);
        assert_eq!(uniforms.cellular_scale, 2.0);
        assert_eq!(uniforms.cellular_warp, 1.0);
        assert_eq!(uniforms.cellular_speed, 0.25);
        assert_eq!(uniforms.cellular_gap_amount, 1.0);
        assert_eq!(uniforms.cellular_gap_threshold, 0.0);
        assert_eq!(uniforms.cellular_gap_softness, 0.08);

        let mut editable = EffectsConfig::default();
        for (key, value) in [
            ("cellular_amount", "0.7"),
            ("cellular_scale", "18"),
            ("cellular_warp", "0.4"),
            ("cellular_speed", "0.9"),
            ("cellular_gap_amount", "0.8"),
            ("cellular_gap_threshold", "0.55"),
            ("cellular_gap_softness", "0.1"),
        ] {
            assert!(editable.set_field(key, value));
            let metadata = param_meta(key).unwrap();
            assert!(metadata.min < metadata.max);
            assert!(metadata.step > 0.0);
        }
        let cellular_group = editable
            .grouped_fields()
            .into_iter()
            .find(|(name, _)| *name == "cellular")
            .unwrap();
        assert_eq!(cellular_group.1.len(), 7);
    }

    #[test]
    fn chroma_and_temporal_keys_round_trip_with_safe_legacy_defaults() {
        let configured = EffectsConfig {
            key_mode: 3,
            key_color: [0.1, 0.8, 0.2],
            key_tolerance: 0.24,
            key_softness: 0.06,
            ..Default::default()
        };
        let restored: EffectsConfig =
            serde_yaml::from_str(&serde_yaml::to_string(&configured).unwrap()).unwrap();
        let mut uniforms = EffectUniforms::default();
        restored.apply_to_uniforms(&mut uniforms);
        assert_eq!(uniforms.key_mode, 3.0);
        assert_eq!(uniforms.key_color, [0.1, 0.8, 0.2]);
        assert_eq!(uniforms.key_tolerance, 0.24);

        let legacy: EffectsConfig = serde_yaml::from_str("pixelate: 2\n").unwrap();
        assert_eq!(legacy.key_mode, 0);
        assert_eq!(legacy.key_color, [0.0, 1.0, 0.0]);
        assert_eq!(legacy.key_tolerance, 0.15);

        let invalid: EffectsConfig =
            serde_yaml::from_str("key_mode: 99\nkey_color: [.nan, .inf, -2]\nkey_tolerance: 9\n")
                .unwrap();
        invalid.apply_to_uniforms(&mut uniforms);
        assert_eq!(uniforms.key_mode, 4.0);
        assert_eq!(uniforms.key_color, [0.0, 1.0, 0.0]);
        assert_eq!(uniforms.key_tolerance, 1.0);

        let legacy_temporal: TemporalConfig = serde_yaml::from_str("feedback: 0.2\n").unwrap();
        let temporal = legacy_temporal.to_params();
        assert_eq!(temporal.key_mode, 0.0);
        assert_eq!(temporal.key_threshold, 0.1);
        assert_eq!(temporal.key_softness, 0.03);
        assert_eq!(temporal.key_history, 1.0);

        let invalid_temporal: TemporalConfig = serde_yaml::from_str(
            "key_mode: 99\nkey_threshold: .nan\nkey_softness: 8\nkey_history: 99\n",
        )
        .unwrap();
        let temporal = invalid_temporal.to_params();
        assert_eq!(temporal.key_mode, 4.0);
        assert_eq!(temporal.key_threshold, 0.1);
        assert_eq!(temporal.key_softness, 0.5);
        assert_eq!(temporal.key_history, 23.0);
    }

    /// A patch with modulation state survives a YAML round-trip, and old
    /// patches without a `modulation:` section still parse.
    #[test]
    fn mod_config_yaml_round_trip() {
        let mut matrix = ModMatrix::new();
        matrix.clock.set_bpm(140.0);
        matrix.lfos[1].shape = LfoShape::Saw;
        matrix.lfos[1].beats = 2.0;
        matrix.lfos[2].phase = 0.25;
        matrix.audio_enabled = true;
        matrix.audio_gain = 1.5;
        matrix.audio_source_kind = crate::modulation::AUDIO_SOURCE_FILE.to_string();
        matrix.audio_clip_path = "pulse-loop.wav".to_string();
        matrix.audio_band_config = crate::audio::AudioBandConfig::new(
            6,
            &[120.0, 480.0, 1500.0, 4000.0, 9000.0],
            16_000.0,
        );
        matrix.midi_enabled = true;
        matrix.midi_ccs = [21, 22, 23, 24];
        matrix.midi_clock_sync = true;
        matrix.set_gyro_degrees(355.0, 12.0, -18.0);
        matrix.gyro_config[0] = GyroAxisConfig {
            center_degrees: 350.0,
            range_degrees: 30.0,
            expo: 0.75,
            invert: true,
        };
        matrix.gyro_config[1].center_degrees = 5.0;
        matrix.recompute_gyro();
        matrix.set_pad(0.8, 0.2, false);
        matrix.pad_config.axes[0] = PadAxisConfig {
            curve: Curve::Exp,
            curve_amount: 0.75,
            quantize: 8,
        };
        matrix.pad_config.axes[1] = PadAxisConfig {
            curve: Curve::Steps,
            curve_amount: -0.5,
            quantize: 16,
        };
        matrix.pad_config.spring_enabled = true;
        matrix.pad_config.spring_rate = 7.0;
        let mut expressive = Routing::new(ModSource::AudioBright, "layer2_opacity", 0.6);
        expressive.curve = Curve::SCurve;
        expressive.curve_amount = 0.5;
        expressive.attack = 0.08;
        expressive.release = 0.4;
        matrix.routings.push(expressive);
        matrix
            .routings
            .push(Routing::new(ModSource::AudioBass, "ntsc_snow", -0.8));
        matrix
            .routings
            .push(Routing::new(ModSource::Midi(2), "vignette", 1.0));
        matrix
            .routings
            .push(Routing::new(ModSource::Lfo(3), "rgb_split", 0.5));
        matrix
            .routings
            .push(Routing::new(ModSource::AudioBand(5), "contrast", 0.25));

        let temporal = TemporalParams {
            feedback: 0.7,
            fb_zoom: 1.02,
            fb_rotate: -1.5,
            slitscan: 0.4,
            slit_angle: 37.0,
            slit_axis: 1.0,
            key_mode: 3.0,
            key_threshold: 0.22,
            key_softness: 0.05,
            key_history: 4.0,
        };

        let patch = PatchState::capture(
            &EffectUniforms::default(),
            &[],
            &NtscParams::default(),
            &matrix,
            &temporal,
            true,
            &crate::morph::Morph::default(),
        );
        let yaml = serde_yaml::to_string(&patch).unwrap();
        let parsed: PatchState = serde_yaml::from_str(&yaml).unwrap();
        assert!(parsed.master_paused);

        let mut restored = ModMatrix::new();
        let mut restored_temporal = TemporalParams::default();
        parsed.apply(
            &mut EffectUniforms::default(),
            &mut [],
            &mut NtscParams::default(),
            &mut restored,
            &mut restored_temporal,
        );

        assert_eq!(restored_temporal.feedback, 0.7);
        assert_eq!(restored_temporal.fb_zoom, 1.02);
        assert_eq!(restored_temporal.fb_rotate, -1.5);
        assert_eq!(restored_temporal.slitscan, 0.4);
        assert_eq!(restored_temporal.slit_angle, 37.0);
        assert_eq!(restored_temporal.slit_axis, 1.0);
        assert_eq!(restored_temporal.key_mode, 3.0);
        assert_eq!(restored_temporal.key_threshold, 0.22);
        assert_eq!(restored_temporal.key_softness, 0.05);
        assert_eq!(restored_temporal.key_history, 4.0);

        assert_eq!(restored.clock.bpm, 140.0);
        assert_eq!(restored.lfos[1].shape, LfoShape::Saw);
        assert_eq!(restored.lfos[1].beats, 2.0);
        assert_eq!(restored.lfos[2].phase, 0.25);
        assert!(restored.audio_enabled);
        assert_eq!(restored.audio_gain, 1.5);
        assert_eq!(
            restored.audio_source_kind,
            crate::modulation::AUDIO_SOURCE_FILE
        );
        assert_eq!(restored.audio_clip_path, "pulse-loop.wav");
        assert_eq!(restored.audio_band_config.count(), 6);
        assert_eq!(
            restored.audio_band_config.crossovers(),
            &[120.0, 480.0, 1500.0, 4000.0, 9000.0]
        );
        assert_eq!(restored.audio_band_config.ceiling_hz(), 16_000.0);
        assert!(restored.midi_enabled);
        assert_eq!(restored.midi_ccs, [21, 22, 23, 24]);
        assert!(restored.midi_clock_sync);
        assert_eq!(restored.gyro_raw, [355.0, 12.0, -18.0]);
        assert_eq!(restored.gyro_config[0].center_degrees, 350.0);
        assert_eq!(restored.gyro_config[0].range_degrees, 30.0);
        assert_eq!(restored.gyro_config[0].expo, 0.75);
        assert!(restored.gyro_config[0].invert);
        assert_eq!(restored.gyro_config[1].center_degrees, 5.0);
        assert_eq!(restored.pad, [0.8, 0.2]);
        assert!(!restored.pad_active);
        assert_eq!(restored.pad_config.axes[0].curve, Curve::Exp);
        assert_eq!(restored.pad_config.axes[0].curve_amount, 0.75);
        assert_eq!(restored.pad_config.axes[0].quantize, 8);
        assert_eq!(restored.pad_config.axes[1].curve, Curve::Steps);
        assert_eq!(restored.pad_config.axes[1].quantize, 16);
        assert!(restored.pad_config.spring_enabled);
        assert_eq!(restored.pad_config.spring_rate, 7.0);
        assert_eq!(restored.routings.len(), 5);
        assert_eq!(restored.routings[0].source, ModSource::AudioBright);
        assert_eq!(restored.routings[0].target(), "layer2_opacity");
        assert_eq!(restored.routings[0].curve, Curve::SCurve);
        assert_eq!(restored.routings[0].curve_amount, 0.5);
        assert_eq!(restored.routings[0].attack, 0.08);
        assert_eq!(restored.routings[0].release, 0.4);
        assert_eq!(restored.routings[1].source, ModSource::AudioBass);
        assert_eq!(restored.routings[1].target(), "ntsc_snow");
        assert_eq!(restored.routings[1].depth, -0.8);
        assert_eq!(restored.routings[2].source, ModSource::Midi(2));
        assert_eq!(restored.routings[3].source, ModSource::Lfo(3));
        assert_eq!(restored.routings[4].source, ModSource::AudioBand(5));

        // Layer modulation math: bright 0.5 at depth 0.6 on layer 2 opacity.
        restored.audio.bright = 0.5;
        restored.update_at_beat(0.0, 1.0);
        let lm = restored.modulate_layer_full(
            1,
            &crate::effects::EffectUniforms::default(),
            0.4,
            1.0,
            30.0,
        );
        let expected = 0.4 + crate::modulation::shape(0.5, Curve::SCurve, 0.5) * 0.6 * 0.5;
        assert!((lm.opacity - expected).abs() < 1e-4);
        assert_eq!(lm.speed, 1.0, "untargeted values pass through");

        // Legacy patch without modulation section parses and applies cleanly.
        let legacy = "master:\n  pixelate: 4.0\nlayers: []\n";
        let parsed: PatchState = serde_yaml::from_str(legacy).unwrap();
        assert!(!parsed.master_paused);
        assert!(parsed.modulation.is_none());
        assert!(parsed.temporal.is_none());
        let mut untouched = ModMatrix::new();
        untouched.clock.set_bpm(99.0);
        let mut untouched_temporal = TemporalParams {
            feedback: 0.3,
            ..Default::default()
        };
        parsed.apply(
            &mut EffectUniforms::default(),
            &mut [],
            &mut NtscParams::default(),
            &mut untouched,
            &mut untouched_temporal,
        );
        assert_eq!(
            untouched.clock.bpm, 99.0,
            "absent section must not reset matrix"
        );
        assert_eq!(
            untouched_temporal.feedback, 0.3,
            "absent section must not reset temporal"
        );

        // Legacy modulation sections had no current motion samples. They
        // restore gyro axes at their saved centers and the pad at center.
        let legacy_mod = r#"
master: {}
layers: []
modulation:
  gyro:
    - { center_degrees: 90.0, range_degrees: 20.0 }
    - { center_degrees: 10.0, range_degrees: 30.0 }
    - { center_degrees: -5.0, range_degrees: 40.0 }
"#;
        let parsed: PatchState = serde_yaml::from_str(legacy_mod).unwrap();
        let mut legacy_matrix = ModMatrix::new();
        parsed
            .modulation
            .unwrap()
            .apply_to_matrix(&mut legacy_matrix);
        assert_eq!(legacy_matrix.gyro, [0.5; 3]);
        assert_eq!(legacy_matrix.pad, [0.5; 2]);
        assert!(!legacy_matrix.pad_active);

        // The historical three-entry list encoded two crossovers plus the
        // analysis ceiling. It must load exactly as it did before the band
        // count became configurable.
        let legacy_audio = r#"
master: {}
layers: []
modulation:
  audio_band_edges: [250.0, 2000.0, 8000.0]
  routings:
    - { source: audio_bass, target: brightness, depth: 0.5 }
"#;
        let parsed: PatchState = serde_yaml::from_str(legacy_audio).unwrap();
        let mut legacy_audio_matrix = ModMatrix::new();
        parsed
            .modulation
            .unwrap()
            .apply_to_matrix(&mut legacy_audio_matrix);
        assert_eq!(legacy_audio_matrix.audio_band_config.count(), 3);
        assert_eq!(
            legacy_audio_matrix.audio_band_config.crossovers(),
            &[250.0, 2000.0]
        );
        assert_eq!(legacy_audio_matrix.audio_band_config.ceiling_hz(), 8000.0);
        assert_eq!(legacy_audio_matrix.routings[0].source, ModSource::AudioBass);

        let alias_and_canonical: ModConfig = serde_yaml::from_str(
            r#"
bpm: 120
routings:
  - { source: lfo0, target: layer1_key, depth: 0.9 }
  - { source: lfo1, target: layer1_key_threshold, depth: 0.4 }
"#,
        )
        .unwrap();
        let mut normalized = ModMatrix::new();
        alias_and_canonical.apply_to_matrix(&mut normalized);
        assert_eq!(normalized.routings.len(), 1);
        assert_eq!(normalized.routings[0].target(), "layer1_key_threshold");
        assert_eq!(normalized.routings[0].source, ModSource::Lfo(1));

        // Canonical spellings beyond the accepted route window must not
        // suppress a legacy alias that is itself inside that window.
        let mut capped: ModConfig = serde_yaml::from_str("bpm: 120\n").unwrap();
        capped.routings.push(RoutingConfig {
            source: "lfo0".to_string(),
            target: "layer1_key".to_string(),
            depth: 0.5,
            curve: "linear".to_string(),
            curve_amount: 0.0,
            attack: 30.0,
            release: 30.0,
        });
        for _ in 1..MAX_ROUTINGS {
            capped.routings.push(RoutingConfig {
                source: "lfo1".to_string(),
                target: "brightness".to_string(),
                depth: 0.25,
                curve: "linear".to_string(),
                curve_amount: 0.0,
                attack: 0.0,
                release: 0.0,
            });
        }
        capped.routings.push(RoutingConfig {
            source: "lfo2".to_string(),
            target: "layer1_key_threshold".to_string(),
            depth: 0.75,
            curve: "linear".to_string(),
            curve_amount: 0.0,
            attack: 0.0,
            release: 0.0,
        });
        let mut capped_matrix = ModMatrix::new();
        capped.apply_to_matrix(&mut capped_matrix);
        assert_eq!(capped_matrix.routings.len(), MAX_ROUTINGS);
        assert_eq!(capped_matrix.routings[0].target(), "layer1_key_threshold");
        assert_eq!(capped_matrix.routings[0].source, ModSource::Lfo(0));
        assert_eq!(capped_matrix.routings[0].attack, 10.0);
        assert_eq!(capped_matrix.routings[0].release, 10.0);
        assert_eq!(
            legacy_audio_matrix.audio_source_kind,
            crate::modulation::AUDIO_SOURCE_LIVE
        );
        assert!(legacy_audio_matrix.audio_clip_path.is_empty());
    }
}
