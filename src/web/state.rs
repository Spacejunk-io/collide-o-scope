//! Shared state between the web control panel and the render engine.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, Mutex, RwLock};

use crate::effects::EffectUniforms;

/// Shared state accessible by both the web server and the render loop.
pub struct WebState {
    /// Full app snapshot (pushed from render loop each frame)
    pub app: RwLock<AppSnapshot>,
    /// Broadcast channel for pushing state to all WebSocket clients
    pub tx: broadcast::Sender<String>,
    /// Actions queue: browser pushes commands, render loop drains them
    pub actions: Mutex<Vec<WebAction>>,
    /// Thumbnail cache: filename → JPEG bytes (generated on library scan)
    pub thumbnails: std::sync::RwLock<HashMap<String, Vec<u8>>>,
    /// Preview frames: filename → vec of JPEG frames (for hover animation)
    pub preview_frames: std::sync::RwLock<HashMap<String, Vec<Vec<u8>>>>,
    /// Library folder for clip uploads (set by the app; None until known).
    pub library_folder: std::sync::RwLock<Option<std::path::PathBuf>>,
    /// Per-session access token. Loopback clients are exempt; LAN clients
    /// must present it (the QR code carries it) and get a cookie back.
    pub access_token: String,
    /// Full remote URL (LAN address + token), set by the server at startup.
    pub lan_url: std::sync::RwLock<String>,
}

/// Full app state snapshot sent to the browser each frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSnapshot {
    #[serde(rename = "type")]
    pub msg_type: String,
    pub effects: EffectsSnapshot,
    pub ntsc: NtscSnapshot,
    pub layers: Vec<LayerSnapshot>,
    pub library: Vec<String>,
    pub paused: bool,
    /// Modulation matrix state (BPM, LFOs, routings)
    #[serde(default)]
    pub modulation: ModSnapshot,
    /// Audio input analysis state
    #[serde(default)]
    pub audio: AudioSnapshot,
    /// MIDI input state
    #[serde(default)]
    pub midi: MidiSnapshot,
    /// Temporal (feedback/slit-scan) effect state
    #[serde(default)]
    pub temporal: TemporalSnapshot,
    /// Spout output state
    #[serde(default)]
    pub spout: SpoutSnapshot,
    /// Remote control URL (LAN address with access token) for the QR code
    #[serde(default)]
    pub remote_url: String,
    /// Whether the fullscreen output window is open
    #[serde(default)]
    pub output_window: bool,
    /// Patch morph crossfader state
    #[serde(default)]
    pub morph: MorphSnapshot,
    /// Output is currently cut to black
    #[serde(default)]
    pub blackout: bool,
    /// Export progress: 0.0 = idle, 0.0..1.0 = rendering, 1.0 = done
    #[serde(default)]
    pub export_progress: f32,
    /// Non-empty when export encountered an error
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub export_error: String,
}

impl Default for AppSnapshot {
    fn default() -> Self {
        Self {
            msg_type: "state".to_string(),
            effects: EffectsSnapshot::default(),
            ntsc: NtscSnapshot::default(),
            layers: Vec::new(),
            library: Vec::new(),
            paused: false,
            modulation: ModSnapshot::default(),
            audio: AudioSnapshot::default(),
            midi: MidiSnapshot::default(),
            temporal: TemporalSnapshot::default(),
            spout: SpoutSnapshot::default(),
            remote_url: String::new(),
            output_window: false,
            morph: MorphSnapshot::default(),
            blackout: false,
            export_progress: 0.0,
            export_error: String::new(),
        }
    }
}

/// A JSON-friendly snapshot of the current effect parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectsSnapshot {
    pub pixelate: f32,
    pub rgb_split: f32,
    pub hue_shift: f32,
    pub saturation: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub posterize: f32,
    pub invert: bool,
    pub grain_intensity: f32,
    pub grain_size: f32,
    pub grain_algo: u32,
    pub color_grain: bool,
    pub vignette: f32,
    pub color_drift: f32,
    pub breathe_scale: f32,
    pub breathe_rotation: f32,
    pub breathe_position: f32,
}

impl Default for EffectsSnapshot {
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
            vignette: 0.0,
            color_drift: 0.0,
            breathe_scale: 0.0,
            breathe_rotation: 0.0,
            breathe_position: 0.0,
        }
    }
}

/// NTSC/VHS effect parameters sent to the browser.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NtscSnapshot {
    pub enabled: bool,
    pub tape_speed: u32,
    pub chroma_loss: f32,
    pub edge_wave_enabled: bool,
    pub edge_wave_intensity: f32,
    pub edge_wave_speed: f32,
    pub head_switching_enabled: bool,
    pub head_switching_height: i32,
    pub head_switching_shift: f32,
    pub tracking_noise_enabled: bool,
    pub tracking_noise_height: i32,
    pub tracking_noise_wave: f32,
    pub tracking_noise_snow: f32,
    pub snow_intensity: f32,
    pub composite_noise_intensity: f32,
    pub luma_noise_intensity: f32,
    pub chroma_noise_intensity: f32,
    pub luma_smear: f32,
    pub composite_sharpening: f32,
}

impl Default for NtscSnapshot {
    fn default() -> Self {
        Self {
            enabled: false,
            tape_speed: 0,
            chroma_loss: 0.0,
            edge_wave_enabled: false,
            edge_wave_intensity: 0.0,
            edge_wave_speed: 0.5,
            head_switching_enabled: false,
            head_switching_height: 8,
            head_switching_shift: 0.0,
            tracking_noise_enabled: false,
            tracking_noise_height: 24,
            tracking_noise_wave: 0.0,
            tracking_noise_snow: 0.0,
            snow_intensity: 0.0,
            composite_noise_intensity: 0.0,
            luma_noise_intensity: 0.0,
            chroma_noise_intensity: 0.0,
            luma_smear: 0.0,
            composite_sharpening: 0.0,
        }
    }
}

impl NtscSnapshot {
    pub fn from_params(p: &crate::ntsc::NtscParams) -> Self {
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
}

/// Modulation matrix state sent to the browser.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModSnapshot {
    pub bpm: f32,
    /// Beat position (quarter notes); the panel's beat light pulses on it.
    #[serde(default)]
    pub beat: f64,
    pub lfos: Vec<LfoSnapshot>,
    pub routings: Vec<RoutingSnapshot>,
    /// Phone orientation [yaw, pitch, roll], 0..1 (0.5 = level).
    #[serde(default)]
    pub gyro: [f32; 3],
    /// XY performance pad position, each 0..1.
    #[serde(default)]
    pub pad: [f32; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LfoSnapshot {
    pub shape: String,
    pub beats: f32,
    pub phase: f32,
    /// Live output value in [-1, 1], for UI meters.
    pub value: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingSnapshot {
    pub source: String,
    pub target: String,
    pub depth: f32,
}

/// Audio analysis state sent to the browser.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AudioSnapshot {
    pub enabled: bool,
    pub gain: f32,
    pub level: f32,
    pub bass: f32,
    pub mid: f32,
    pub high: f32,
    pub onset: f32,
    #[serde(default)]
    pub bright: f32,
    #[serde(default)]
    pub noise: f32,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub device: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub error: String,
    /// Available input device names, for the device select.
    #[serde(default)]
    pub devices: Vec<String>,
    /// Preferred device name ("" = system default).
    #[serde(default)]
    pub selected: String,
}

impl ModSnapshot {
    pub fn from_matrix(m: &crate::modulation::ModMatrix) -> Self {
        Self {
            bpm: m.clock.bpm,
            beat: m.current_beat,
            lfos: m
                .lfos
                .iter()
                .zip(m.lfo_values.iter())
                .map(|(l, &v)| LfoSnapshot {
                    shape: l.shape.as_str().to_string(),
                    beats: l.beats,
                    phase: l.phase,
                    value: v,
                })
                .collect(),
            routings: m
                .routings
                .iter()
                .map(|r| RoutingSnapshot {
                    source: r.source.as_str().to_string(),
                    target: r.target.clone(),
                    depth: r.depth,
                })
                .collect(),
            gyro: m.gyro,
            pad: m.pad,
        }
    }
}

/// Temporal (frame-history) effect parameters sent to the browser.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TemporalSnapshot {
    pub feedback: f32,
    pub fb_zoom: f32,
    pub fb_rotate: f32,
    pub slitscan: f32,
    pub slit_axis: f32,
}

impl TemporalSnapshot {
    pub fn from_params(p: &crate::effects::params::TemporalParams) -> Self {
        Self {
            feedback: p.feedback,
            fb_zoom: p.fb_zoom,
            fb_rotate: p.fb_rotate,
            slitscan: p.slitscan,
            slit_axis: p.slit_axis,
        }
    }
}

/// Spout output state sent to the browser.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpoutSnapshot {
    pub enabled: bool,
    pub active: bool,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub error: String,
}

/// Patch morph crossfader state sent to the browser.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MorphSnapshot {
    pub has_a: bool,
    pub has_b: bool,
    /// True while both slots are set (crossfader engaged).
    pub active: bool,
    pub t: f32,
}

/// MIDI input state sent to the browser.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MidiSnapshot {
    pub enabled: bool,
    pub slots: Vec<MidiSlotSnapshot>,
    /// Slot index currently armed for MIDI learn, if any.
    pub learning: Option<usize>,
    /// Follow external MIDI timing clock.
    #[serde(default)]
    pub clock_sync: bool,
    /// True while external clock pulses are actively driving the beat.
    #[serde(default)]
    pub clock_active: bool,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub port: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MidiSlotSnapshot {
    pub cc: u8,
    /// Live value 0..1, for UI meters.
    pub value: f32,
}

/// Per-layer info sent to the browser.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSnapshot {
    pub filename: String,
    pub visible: bool,
    pub paused: bool,
    pub opacity: f32,
    pub speed: f32,
    pub blend_mode: String,
    pub progress: f32,
    #[serde(default)]
    pub key_mode: u32,
    #[serde(default)]
    pub key_threshold: f32,
    #[serde(default)]
    pub key_softness: f32,
}

/// Actions the browser can request (processed by the render loop).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action")]
pub enum WebAction {
    /// Set a master effect parameter
    #[serde(rename = "set_param")]
    SetParam { param: String, value: serde_json::Value },
    /// Add a layer from the library by filename
    #[serde(rename = "add_layer")]
    AddLayer { filename: String },
    /// Remove a layer by index
    #[serde(rename = "remove_layer")]
    RemoveLayer { index: usize },
    /// Toggle layer visibility
    #[serde(rename = "toggle_visibility")]
    ToggleVisibility { index: usize },
    /// Toggle layer play/pause
    #[serde(rename = "toggle_layer_pause")]
    ToggleLayerPause { index: usize },
    /// Toggle master play/pause
    #[serde(rename = "toggle_master_pause")]
    ToggleMasterPause,
    /// Reset all master effects
    #[serde(rename = "reset_fx")]
    ResetFx,
    /// Reset a specific effect group (digital, analog, motion)
    #[serde(rename = "reset_group")]
    ResetGroup { group: String },
    /// Set a per-layer parameter (opacity, speed, blend_mode)
    #[serde(rename = "set_layer_param")]
    SetLayerParam { index: usize, param: String, value: serde_json::Value },
    /// Set an NTSC/VHS effect parameter
    #[serde(rename = "set_ntsc_param")]
    SetNtscParam { param: String, value: serde_json::Value },
    /// Tap tempo: each tap refines BPM and re-anchors the downbeat
    #[serde(rename = "tap_tempo")]
    TapTempo,
    /// Set BPM directly
    #[serde(rename = "set_bpm")]
    SetBpm { value: f32 },
    /// Set an LFO parameter ("shape" | "beats" | "phase")
    #[serde(rename = "set_lfo")]
    SetLfo { index: usize, param: String, value: serde_json::Value },
    /// Append a new modulation routing (defaults)
    #[serde(rename = "add_routing")]
    AddRouting,
    /// Remove a modulation routing by index
    #[serde(rename = "remove_routing")]
    RemoveRouting { index: usize },
    /// Set a routing parameter ("source" | "target" | "depth")
    #[serde(rename = "set_routing")]
    SetRouting { index: usize, param: String, value: serde_json::Value },
    /// Set an audio input parameter ("enabled" | "gain")
    #[serde(rename = "set_audio")]
    SetAudio { param: String, value: serde_json::Value },
    /// Set a MIDI parameter ("enabled" | "learn" | "cc0".."cc3")
    #[serde(rename = "set_midi")]
    SetMidi { param: String, value: serde_json::Value },
    /// Phone orientation sample (degrees, DeviceOrientation convention)
    #[serde(rename = "gyro")]
    Gyro { alpha: f32, beta: f32, gamma: f32 },
    /// XY performance pad position (0..1 each)
    #[serde(rename = "pad")]
    Pad { x: f32, y: f32 },
    /// Open/close the fullscreen output window (second display)
    #[serde(rename = "toggle_output_window")]
    ToggleOutputWindow,
    /// Capture current parameters into morph slot "a" or "b"
    #[serde(rename = "morph_capture")]
    MorphCapture { slot: String },
    /// Clear both morph slots (crossfader disengages)
    #[serde(rename = "morph_clear")]
    MorphClear,
    /// Set the morph crossfader position (0 = A, 1 = B)
    #[serde(rename = "set_morph")]
    SetMorph { value: f32 },
    /// Rescan the library folder (pushed internally after an upload)
    #[serde(rename = "rescan_library")]
    RescanLibrary,
    /// Cut the output to black (toggle)
    #[serde(rename = "toggle_blackout")]
    ToggleBlackout,
    /// Set a temporal effect parameter
    #[serde(rename = "set_temporal")]
    SetTemporal { param: String, value: serde_json::Value },
    /// Enable/disable the Spout output sender
    #[serde(rename = "set_spout")]
    SetSpout { enabled: bool },
    /// Start an offline render export
    #[serde(rename = "start_export")]
    StartExport { width: u32, height: u32, fps: u32, duration_secs: f32 },
    /// Cancel a running export
    #[serde(rename = "cancel_export")]
    CancelExport,
}

impl EffectsSnapshot {
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
            vignette: u.vignette,
            color_drift: u.color_drift,
            breathe_scale: u.breathe_scale,
            breathe_rotation: u.breathe_rotation,
            breathe_position: u.breathe_position,
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
        u.vignette = self.vignette.clamp(0.0, 1.5);
        u.color_drift = self.color_drift.clamp(0.0, 0.02);
        u.breathe_scale = self.breathe_scale.clamp(0.0, 0.05);
        u.breathe_rotation = self.breathe_rotation.clamp(0.0, 2.0);
        u.breathe_position = self.breathe_position.clamp(0.0, 0.02);
    }

    pub fn apply_param(&mut self, param: &str, value: &serde_json::Value) {
        let v = value;
        match param {
            "pixelate" => if let Some(n) = v.as_f64() { self.pixelate = n as f32; },
            "rgb_split" => if let Some(n) = v.as_f64() { self.rgb_split = n as f32; },
            "hue_shift" => if let Some(n) = v.as_f64() { self.hue_shift = n as f32; },
            "saturation" => if let Some(n) = v.as_f64() { self.saturation = n as f32; },
            "brightness" => if let Some(n) = v.as_f64() { self.brightness = n as f32; },
            "contrast" => if let Some(n) = v.as_f64() { self.contrast = n as f32; },
            "posterize" => if let Some(n) = v.as_f64() { self.posterize = n as f32; },
            "invert" => if let Some(b) = v.as_bool() { self.invert = b; },
            "grain_intensity" => if let Some(n) = v.as_f64() { self.grain_intensity = n as f32; },
            "grain_size" => if let Some(n) = v.as_f64() { self.grain_size = n as f32; },
            "grain_algo" => if let Some(n) = v.as_u64() { self.grain_algo = n as u32; },
            "color_grain" => if let Some(b) = v.as_bool() { self.color_grain = b; },
            "vignette" => if let Some(n) = v.as_f64() { self.vignette = n as f32; },
            "color_drift" => if let Some(n) = v.as_f64() { self.color_drift = n as f32; },
            "breathe_scale" => if let Some(n) = v.as_f64() { self.breathe_scale = n as f32; },
            "breathe_rotation" => if let Some(n) = v.as_f64() { self.breathe_rotation = n as f32; },
            "breathe_position" => if let Some(n) = v.as_f64() { self.breathe_position = n as f32; },
            _ => {}
        }
    }
}

impl WebState {
    pub fn new() -> Arc<Self> {
        let (tx, _) = broadcast::channel(64);
        Arc::new(Self {
            app: RwLock::new(AppSnapshot::default()),
            tx,
            actions: Mutex::new(Vec::new()),
            thumbnails: std::sync::RwLock::new(HashMap::new()),
            preview_frames: std::sync::RwLock::new(HashMap::new()),
            library_folder: std::sync::RwLock::new(None),
            access_token: generate_token(),
            lan_url: std::sync::RwLock::new(String::new()),
        })
    }
}

/// Random per-session access token (8 hex chars, OS entropy).
fn generate_token() -> String {
    let mut bytes = [0u8; 4];
    if getrandom::fill(&mut bytes).is_err() {
        // Fallback: time-derived. Weaker, but the gate still functions.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        bytes = nanos.to_le_bytes();
    }
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
