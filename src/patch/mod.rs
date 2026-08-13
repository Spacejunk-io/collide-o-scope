#[allow(dead_code)]
pub mod editor;

use serde::{Deserialize, Serialize};

use crate::effects::params::TemporalParams;
use crate::effects::EffectUniforms;
use crate::layers::{BlendMode, Layer};
use crate::modulation::{Lfo, LfoShape, ModMatrix, ModSource, Routing, MAX_ROUTINGS, NUM_LFOS};
use crate::ntsc::NtscParams;

// --- Helpers for serde defaults ---

fn one() -> f32 {
    1.0
}
fn default_fps() -> f32 {
    30.0
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
        "pixelate" => Some(ParamMeta { step: 1.0, min: 1.0, max: 32.0, desc: "pixel block size" }),
        "rgb_split" => Some(ParamMeta { step: 0.5, min: 0.0, max: 30.0, desc: "chromatic split px" }),
        "hue_shift" => Some(ParamMeta { step: 5.0, min: -180.0, max: 180.0, desc: "degrees" }),
        "saturation" => Some(ParamMeta { step: 0.05, min: -1.0, max: 1.0, desc: "color intensity" }),
        "brightness" => Some(ParamMeta { step: 0.05, min: -1.0, max: 1.0, desc: "exposure" }),
        "contrast" => Some(ParamMeta { step: 0.05, min: -1.0, max: 1.0, desc: "dynamic range" }),
        "posterize" => Some(ParamMeta { step: 1.0, min: 0.0, max: 16.0, desc: "color levels (0=off)" }),
        "grain_intensity" => Some(ParamMeta { step: 0.01, min: 0.0, max: 0.3, desc: "film grain amount" }),
        "grain_size" => Some(ParamMeta { step: 0.25, min: 1.0, max: 4.0, desc: "grain particle scale" }),
        "grain_algo" => Some(ParamMeta { step: 1.0, min: 0.0, max: 3.0, desc: "0=value 1=perlin 2=gaussian 3=salt&pepper" }),
        "breathe_scale" => Some(ParamMeta { step: 0.005, min: 0.0, max: 0.05, desc: "zoom oscillation" }),
        "breathe_rotation" => Some(ParamMeta { step: 0.1, min: 0.0, max: 2.0, desc: "rotation oscillation deg" }),
        "breathe_position" => Some(ParamMeta { step: 0.002, min: 0.0, max: 0.02, desc: "position drift" }),
        "vignette" => Some(ParamMeta { step: 0.05, min: 0.0, max: 1.5, desc: "edge darkening" }),
        "color_drift" => Some(ParamMeta { step: 0.002, min: 0.0, max: 0.02, desc: "chromatic aberration" }),
        "opacity" => Some(ParamMeta { step: 0.05, min: 0.0, max: 1.0, desc: "layer transparency" }),
        "speed" => Some(ParamMeta { step: 0.25, min: 0.25, max: 4.0, desc: "playback multiplier" }),
        "fps" => Some(ParamMeta { step: 1.0, min: 1.0, max: 60.0, desc: "decode frame rate" }),
        _ => None,
    }
}


// --- Serializable patch state ---

#[derive(Serialize, Deserialize, Clone)]
pub struct PatchState {
    pub master: EffectsConfig,
    pub layers: Vec<LayerConfig>,
    #[serde(default)]
    pub ntsc: Option<NtscConfig>,
    #[serde(default)]
    pub modulation: Option<ModConfig>,
    #[serde(default)]
    pub temporal: Option<TemporalConfig>,
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
}

impl TemporalConfig {
    pub fn from_params(p: &TemporalParams) -> Self {
        Self {
            feedback: p.feedback,
            fb_zoom: p.fb_zoom,
            fb_rotate: p.fb_rotate,
            slitscan: p.slitscan,
            slit_axis: p.slit_axis,
        }
    }

    pub fn to_params(&self) -> TemporalParams {
        TemporalParams {
            feedback: self.feedback.clamp(0.0, 0.95),
            fb_zoom: self.fb_zoom.clamp(0.9, 1.1),
            fb_rotate: self.fb_rotate.clamp(-5.0, 5.0),
            slitscan: self.slitscan.clamp(0.0, 1.0),
            slit_axis: self.slit_axis.clamp(0.0, 1.0),
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
    pub midi_enabled: bool,
    #[serde(default = "default_midi_ccs")]
    pub midi_ccs: Vec<u8>,
    #[serde(default)]
    pub midi_clock_sync: bool,
}

fn default_midi_ccs() -> Vec<u8> {
    vec![1, 2, 3, 4]
}

fn default_bpm() -> f32 {
    120.0
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
                    phase: l.phase,
                })
                .collect(),
            routings: m
                .routings
                .iter()
                .map(|r| RoutingConfig {
                    source: r.source.as_str().to_string(),
                    target: r.target.clone(),
                    depth: r.depth,
                })
                .collect(),
            audio_enabled: m.audio_enabled,
            audio_gain: m.audio_gain,
            midi_enabled: m.midi_enabled,
            midi_ccs: m.midi_ccs.to_vec(),
            midi_clock_sync: m.midi_clock_sync,
        }
    }

    pub fn apply_to_matrix(&self, m: &mut ModMatrix) {
        m.clock.set_bpm(self.bpm);
        for (i, cfg) in self.lfos.iter().take(NUM_LFOS).enumerate() {
            m.lfos[i] = Lfo {
                shape: LfoShape::from_str(&cfg.shape),
                beats: cfg.beats.clamp(0.0625, 64.0),
                phase: cfg.phase.rem_euclid(1.0),
            };
        }
        m.routings = self
            .routings
            .iter()
            .take(MAX_ROUTINGS)
            .map(|r| Routing {
                source: ModSource::from_str(&r.source),
                target: r.target.clone(),
                depth: r.depth.clamp(-1.0, 1.0),
            })
            .collect();
        m.audio_enabled = self.audio_enabled;
        m.audio_gain = self.audio_gain.clamp(0.0, 8.0);
        m.midi_enabled = self.midi_enabled;
        for (i, &cc) in self.midi_ccs.iter().take(m.midi_ccs.len()).enumerate() {
            m.midi_ccs[i] = cc & 0x7F;
        }
        m.midi_clock_sync = self.midi_clock_sync;
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

fn default_edge_wave_speed() -> f32 { 0.5 }
fn default_head_height() -> i32 { 8 }
fn default_tracking_height() -> i32 { 24 }

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
        NtscParams {
            enabled: self.enabled,
            tape_speed: self.tape_speed,
            chroma_loss: self.chroma_loss,
            edge_wave_enabled: self.edge_wave_enabled,
            edge_wave_intensity: self.edge_wave_intensity,
            edge_wave_speed: self.edge_wave_speed,
            head_switching_enabled: self.head_switching_enabled,
            head_switching_height: self.head_switching_height,
            head_switching_shift: self.head_switching_shift,
            tracking_noise_enabled: self.tracking_noise_enabled,
            tracking_noise_height: self.tracking_noise_height,
            tracking_noise_wave: self.tracking_noise_wave,
            tracking_noise_snow: self.tracking_noise_snow,
            snow_intensity: self.snow_intensity,
            composite_noise_intensity: self.composite_noise_intensity,
            luma_noise_intensity: self.luma_noise_intensity,
            chroma_noise_intensity: self.chroma_noise_intensity,
            luma_smear: self.luma_smear,
            composite_sharpening: self.composite_sharpening,
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct LayerConfig {
    pub filename: String,
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
}

fn default_key_threshold() -> f32 { 0.5 }
fn default_key_softness() -> f32 { 0.1 }

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
        }
    }

    pub fn apply_to_uniforms(&self, u: &mut EffectUniforms) {
        u.pixelate_size = self.pixelate.clamp(1.0, 32.0);
        u.rgb_split = self.rgb_split.clamp(0.0, 30.0);
        u.hue_shift = self.hue_shift.clamp(-180.0, 180.0);
        u.saturation = self.saturation.clamp(-1.0, 1.0);
        u.brightness = self.brightness.clamp(-1.0, 1.0);
        u.contrast = self.contrast.clamp(-1.0, 1.0);
        u.posterize = self.posterize.clamp(0.0, 16.0);
        u.invert = if self.invert { 1.0 } else { 0.0 };
        u.grain_intensity = self.grain_intensity.clamp(0.0, 0.3);
        u.grain_size = self.grain_size.clamp(1.0, 4.0);
        u.grain_algo = (self.grain_algo.min(3)) as f32;
        u.color_grain = if self.color_grain { 1.0 } else { 0.0 };
        u.breathe_scale = self.breathe_scale.clamp(0.0, 0.05);
        u.breathe_rotation = self.breathe_rotation.clamp(0.0, 2.0);
        u.breathe_position = self.breathe_position.clamp(0.0, 0.02);
        u.vignette = self.vignette.clamp(0.0, 1.5);
        u.color_drift = self.color_drift.clamp(0.0, 0.02);
        u.key_mode = self.key_mode.min(2) as f32;
        u.key_threshold = self.key_threshold.clamp(0.0, 1.0);
        u.key_softness = self.key_softness.clamp(0.0, 0.5);
    }

    /// Get fields organized into groups for display.
    pub fn grouped_fields(&self) -> Vec<(&'static str, Vec<(&'static str, String)>)> {
        vec![
            ("digital", vec![
                ("pixelate", format!("{:.1}", self.pixelate)),
                ("rgb_split", format!("{:.1}", self.rgb_split)),
                ("hue_shift", format!("{:.1}", self.hue_shift)),
                ("saturation", format!("{:.2}", self.saturation)),
                ("brightness", format!("{:.2}", self.brightness)),
                ("contrast", format!("{:.2}", self.contrast)),
                ("posterize", format!("{:.1}", self.posterize)),
                ("invert", format!("{}", self.invert)),
            ]),
            ("analog", vec![
                ("grain_intensity", format!("{:.2}", self.grain_intensity)),
                ("grain_size", format!("{:.2}", self.grain_size)),
                ("grain_algo", format!("{}", self.grain_algo)),
                ("color_grain", format!("{}", self.color_grain)),
                ("vignette", format!("{:.2}", self.vignette)),
                ("color_drift", format!("{:.3}", self.color_drift)),
            ]),
            ("motion", vec![
                ("breathe_scale", format!("{:.3}", self.breathe_scale)),
                ("breathe_rotation", format!("{:.2}", self.breathe_rotation)),
                ("breathe_position", format!("{:.3}", self.breathe_position)),
            ]),
        ]
    }

    /// Set a single field by key name. Returns true if the key was recognized.
    pub fn set_field(&mut self, key: &str, value: &str) -> bool {
        match key {
            "pixelate" => { if let Ok(v) = value.parse() { self.pixelate = v; return true; } }
            "rgb_split" => { if let Ok(v) = value.parse() { self.rgb_split = v; return true; } }
            "hue_shift" => { if let Ok(v) = value.parse() { self.hue_shift = v; return true; } }
            "saturation" => { if let Ok(v) = value.parse() { self.saturation = v; return true; } }
            "brightness" => { if let Ok(v) = value.parse() { self.brightness = v; return true; } }
            "contrast" => { if let Ok(v) = value.parse() { self.contrast = v; return true; } }
            "posterize" => { if let Ok(v) = value.parse() { self.posterize = v; return true; } }
            "invert" => { if let Ok(v) = value.parse() { self.invert = v; return true; } }
            "grain_intensity" => { if let Ok(v) = value.parse() { self.grain_intensity = v; return true; } }
            "grain_size" => { if let Ok(v) = value.parse() { self.grain_size = v; return true; } }
            "grain_algo" => { if let Ok(v) = value.parse() { self.grain_algo = v; return true; } }
            "color_grain" => { if let Ok(v) = value.parse() { self.color_grain = v; return true; } }
            "breathe_scale" => { if let Ok(v) = value.parse() { self.breathe_scale = v; return true; } }
            "breathe_rotation" => { if let Ok(v) = value.parse() { self.breathe_rotation = v; return true; } }
            "breathe_position" => { if let Ok(v) = value.parse() { self.breathe_position = v; return true; } }
            "vignette" => { if let Ok(v) = value.parse() { self.vignette = v; return true; } }
            "color_drift" => { if let Ok(v) = value.parse() { self.color_drift = v; return true; } }
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
            opacity: layer.opacity,
            blend_mode: match layer.blend_mode {
                BlendMode::Normal => "normal",
                BlendMode::Screen => "screen",
                BlendMode::Multiply => "multiply",
                BlendMode::Difference => "difference",
            }
            .to_string(),
            speed: layer.speed,
            fps: layer.fps,
            paused: layer.paused,
            visible: layer.visible,
            effects: EffectsConfig::from_uniforms(&layer.effects),
        }
    }

    pub fn apply_to_layer(&self, layer: &mut Layer) {
        layer.opacity = self.opacity.clamp(0.0, 1.0);
        layer.blend_mode = match self.blend_mode.as_str() {
            "screen" => BlendMode::Screen,
            "multiply" => BlendMode::Multiply,
            "difference" => BlendMode::Difference,
            _ => BlendMode::Normal,
        };
        layer.speed = self.speed.clamp(0.25, 4.0);
        layer.fps = self.fps.clamp(1.0, 60.0);
        layer.paused = self.paused;
        layer.visible = self.visible;
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
        ]
    }

    /// Set a top-level field by key name. Returns true if recognized.
    pub fn set_field(&mut self, key: &str, value: &str) -> bool {
        match key {
            "opacity" => { if let Ok(v) = value.parse() { self.opacity = v; return true; } }
            "blend_mode" => { self.blend_mode = value.to_string(); return true; }
            "speed" => { if let Ok(v) = value.parse() { self.speed = v; return true; } }
            "fps" => { if let Ok(v) = value.parse() { self.fps = v; return true; } }
            "paused" => { if let Ok(v) = value.parse() { self.paused = v; return true; } }
            "visible" => { if let Ok(v) = value.parse() { self.visible = v; return true; } }
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
    ) -> Self {
        Self {
            master: EffectsConfig::from_uniforms(master),
            layers: layers.iter().map(LayerConfig::from_layer).collect(),
            ntsc: Some(NtscConfig::from_params(ntsc_params)),
            modulation: Some(ModConfig::from_matrix(mod_matrix)),
            temporal: Some(TemporalConfig::from_params(temporal)),
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
        matrix.midi_enabled = true;
        matrix.midi_ccs = [21, 22, 23, 24];
        matrix.midi_clock_sync = true;
        matrix.routings.push(Routing {
            source: ModSource::AudioBright,
            target: "layer2_opacity".to_string(),
            depth: 0.6,
        });
        matrix.routings.push(Routing {
            source: ModSource::AudioBass,
            target: "ntsc_snow".to_string(),
            depth: -0.8,
        });
        matrix.routings.push(Routing {
            source: ModSource::Midi(2),
            target: "vignette".to_string(),
            depth: 1.0,
        });
        matrix.routings.push(Routing {
            source: ModSource::Lfo(3),
            target: "rgb_split".to_string(),
            depth: 0.5,
        });

        let temporal = TemporalParams {
            feedback: 0.7,
            fb_zoom: 1.02,
            fb_rotate: -1.5,
            slitscan: 0.4,
            slit_axis: 1.0,
        };

        let patch = PatchState::capture(
            &EffectUniforms::default(),
            &[],
            &NtscParams::default(),
            &matrix,
            &temporal,
        );
        let yaml = serde_yaml::to_string(&patch).unwrap();
        let parsed: PatchState = serde_yaml::from_str(&yaml).unwrap();

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
        assert_eq!(restored_temporal.slit_axis, 1.0);

        assert_eq!(restored.clock.bpm, 140.0);
        assert_eq!(restored.lfos[1].shape, LfoShape::Saw);
        assert_eq!(restored.lfos[1].beats, 2.0);
        assert_eq!(restored.lfos[2].phase, 0.25);
        assert!(restored.audio_enabled);
        assert_eq!(restored.audio_gain, 1.5);
        assert!(restored.midi_enabled);
        assert_eq!(restored.midi_ccs, [21, 22, 23, 24]);
        assert!(restored.midi_clock_sync);
        assert_eq!(restored.routings.len(), 4);
        assert_eq!(restored.routings[0].source, ModSource::AudioBright);
        assert_eq!(restored.routings[0].target, "layer2_opacity");
        assert_eq!(restored.routings[1].source, ModSource::AudioBass);
        assert_eq!(restored.routings[1].target, "ntsc_snow");
        assert_eq!(restored.routings[1].depth, -0.8);
        assert_eq!(restored.routings[2].source, ModSource::Midi(2));
        assert_eq!(restored.routings[3].source, ModSource::Lfo(3));

        // Layer modulation math: bright 0.5 at depth 0.6 on layer 2 opacity.
        restored.audio.bright = 0.5;
        let lm = restored.modulate_layer(1, 0.4, 1.0, 0.5);
        assert!((lm.opacity - (0.4 + 0.5 * 0.6 * 0.5)).abs() < 1e-6);
        assert_eq!(lm.speed, 1.0, "untargeted values pass through");

        // Legacy patch without modulation section parses and applies cleanly.
        let legacy = "master:\n  pixelate: 4.0\nlayers: []\n";
        let parsed: PatchState = serde_yaml::from_str(legacy).unwrap();
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
        assert_eq!(untouched.clock.bpm, 99.0, "absent section must not reset matrix");
        assert_eq!(untouched_temporal.feedback, 0.3, "absent section must not reset temporal");
    }
}
