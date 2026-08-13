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
    /// Per-session access token. Every client must present it once (the local
    /// startup URL and QR code carry it) and then receives a strict cookie.
    pub access_token: String,
    /// Full remote URL (LAN address + token), set by the server at startup.
    pub lan_url: std::sync::RwLock<String>,
}

/// Keep browser input bounded even if a client produces controls faster than
/// the render loop can consume them. A small reserve guarantees that safety
/// commands are still admitted while ordinary fader traffic is saturated.
pub const MAX_PENDING_ACTIONS: usize = 512;
const PRIORITY_ACTION_RESERVE: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnqueueOutcome {
    Added,
    Coalesced,
    Dropped,
}

/// Full app state snapshot sent to the browser each frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSnapshot {
    #[serde(rename = "type")]
    pub msg_type: String,
    pub effects: EffectsSnapshot,
    pub ntsc: NtscSnapshot,
    pub layers: Vec<LayerSnapshot>,
    /// Monotonic topology generation used to reject stale multi-controller
    /// reorder requests. Zero means an older server/client with no revision.
    #[serde(default)]
    pub layer_stack_revision: u64,
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
    /// Non-empty when creating or maintaining an output surface failed.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub output_error: String,
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
    /// Stable lifecycle state:
    /// idle | running | cancelling | succeeded | failed | cancelled.
    #[serde(default)]
    pub export_status: String,
    /// Number of coalesced actions waiting for the next downbeat.
    #[serde(default)]
    pub quantized_pending: usize,
}

impl Default for AppSnapshot {
    fn default() -> Self {
        Self {
            msg_type: "state".to_string(),
            effects: EffectsSnapshot::default(),
            ntsc: NtscSnapshot::default(),
            layers: Vec::new(),
            layer_stack_revision: 0,
            library: Vec::new(),
            paused: false,
            modulation: ModSnapshot::default(),
            audio: AudioSnapshot::default(),
            midi: MidiSnapshot::default(),
            temporal: TemporalSnapshot::default(),
            spout: SpoutSnapshot::default(),
            remote_url: String::new(),
            output_window: false,
            output_error: String::new(),
            morph: MorphSnapshot::default(),
            blackout: false,
            export_progress: 0.0,
            export_error: String::new(),
            export_status: "idle".to_string(),
            quantized_pending: 0,
        }
    }
}

/// A JSON-friendly snapshot of the current effect parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectsSnapshot {
    pub pixelate: f32,
    pub downsample: f32,
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
            downsample: 1.0,
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
    /// Last asynchronous worker error, if any.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub error: String,
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
            error: String::new(),
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
            error: String::new(),
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
    /// Engine-owned response settings for each orientation axis.
    #[serde(default)]
    pub gyro_config: GyroConfigSnapshot,
    /// Engine-owned response, quantize, and spring settings for the XY pad.
    #[serde(default)]
    pub pad_config: PadConfigSnapshot,
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
    #[serde(default = "default_curve")]
    pub curve: String,
    #[serde(default)]
    pub curve_amount: f32,
    /// Rise time in seconds (zero is immediate).
    #[serde(default)]
    pub attack: f32,
    /// Fall time in seconds (zero is immediate).
    #[serde(default)]
    pub release: f32,
}

fn default_curve() -> String {
    "linear".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisConfigSnapshot {
    /// Degrees from calibrated center to full-scale output.
    pub range: f32,
    /// Signed response exponent amount in -2..2.
    pub expo: f32,
    pub invert: bool,
}

impl Default for AxisConfigSnapshot {
    fn default() -> Self {
        Self {
            range: 90.0,
            expo: 0.0,
            invert: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GyroConfigSnapshot {
    pub yaw: AxisConfigSnapshot,
    pub pitch: AxisConfigSnapshot,
    pub roll: AxisConfigSnapshot,
}

impl Default for GyroConfigSnapshot {
    fn default() -> Self {
        Self {
            yaw: AxisConfigSnapshot {
                range: 180.0,
                ..Default::default()
            },
            pitch: AxisConfigSnapshot::default(),
            roll: AxisConfigSnapshot::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PadAxisConfigSnapshot {
    #[serde(default = "default_curve")]
    pub curve: String,
    #[serde(default)]
    pub curve_amount: f32,
    /// Number of discrete positions; 0/1 disables quantization.
    #[serde(default)]
    pub quantize: u32,
}

impl Default for PadAxisConfigSnapshot {
    fn default() -> Self {
        Self {
            curve: default_curve(),
            curve_amount: 0.0,
            quantize: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PadConfigSnapshot {
    #[serde(default)]
    pub x: PadAxisConfigSnapshot,
    #[serde(default)]
    pub y: PadAxisConfigSnapshot,
    #[serde(default)]
    pub spring_enabled: bool,
    #[serde(default = "default_spring_rate")]
    pub spring_rate: f32,
}

fn default_spring_rate() -> f32 {
    4.0
}

impl Default for PadConfigSnapshot {
    fn default() -> Self {
        Self {
            x: PadAxisConfigSnapshot::default(),
            y: PadAxisConfigSnapshot::default(),
            spring_enabled: false,
            spring_rate: default_spring_rate(),
        }
    }
}

/// Audio analysis state sent to the browser.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Device that is actually producing samples after fallback resolution.
    #[serde(default)]
    pub active_device: String,
    /// True when `active_device` differs from the requested `selected` device.
    #[serde(default)]
    pub using_fallback: bool,
    #[serde(default = "default_audio_band_count")]
    pub band_count: usize,
    /// The active count - 1 ordered crossovers.
    #[serde(default)]
    pub band_edges: Vec<f32>,
    #[serde(default = "default_audio_band_ceiling")]
    pub band_ceiling_hz: f32,
    /// Normalized configurable band meters, one per active band.
    #[serde(default)]
    pub bands: Vec<f32>,
    #[serde(default)]
    pub spectrum: Vec<f32>,
}

fn default_audio_band_count() -> usize {
    3
}

fn default_audio_band_ceiling() -> f32 {
    8000.0
}

impl Default for AudioSnapshot {
    fn default() -> Self {
        Self {
            enabled: false,
            gain: 0.0,
            level: 0.0,
            bass: 0.0,
            mid: 0.0,
            high: 0.0,
            onset: 0.0,
            bright: 0.0,
            noise: 0.0,
            device: String::new(),
            error: String::new(),
            devices: Vec::new(),
            selected: String::new(),
            active_device: String::new(),
            using_fallback: false,
            band_count: default_audio_band_count(),
            band_edges: vec![250.0, 2000.0],
            band_ceiling_hz: default_audio_band_ceiling(),
            bands: vec![0.0; default_audio_band_count()],
            spectrum: Vec::new(),
        }
    }
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
                    curve: r.curve.as_str().to_string(),
                    curve_amount: r.curve_amount,
                    attack: r.attack,
                    release: r.release,
                })
                .collect(),
            gyro: m.gyro,
            pad: m.pad,
            gyro_config: GyroConfigSnapshot {
                yaw: AxisConfigSnapshot {
                    range: m.gyro_config[0].range_degrees,
                    expo: m.gyro_config[0].expo,
                    invert: m.gyro_config[0].invert,
                },
                pitch: AxisConfigSnapshot {
                    range: m.gyro_config[1].range_degrees,
                    expo: m.gyro_config[1].expo,
                    invert: m.gyro_config[1].invert,
                },
                roll: AxisConfigSnapshot {
                    range: m.gyro_config[2].range_degrees,
                    expo: m.gyro_config[2].expo,
                    invert: m.gyro_config[2].invert,
                },
            },
            pad_config: PadConfigSnapshot {
                x: PadAxisConfigSnapshot {
                    curve: m.pad_config.axes[0].curve.as_str().to_string(),
                    curve_amount: m.pad_config.axes[0].curve_amount,
                    quantize: m.pad_config.axes[0].quantize,
                },
                y: PadAxisConfigSnapshot {
                    curve: m.pad_config.axes[1].curve.as_str().to_string(),
                    curve_amount: m.pad_config.axes[1].curve_amount,
                    quantize: m.pad_config.axes[1].quantize,
                },
                spring_enabled: m.pad_config.spring_enabled,
                spring_rate: m.pad_config.spring_rate,
            },
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
    #[serde(default)]
    pub slit_angle: f32,
    pub slit_axis: f32,
}

impl TemporalSnapshot {
    pub fn from_params(p: &crate::effects::params::TemporalParams) -> Self {
        Self {
            feedback: p.feedback,
            fb_zoom: p.fb_zoom,
            fb_rotate: p.fb_rotate,
            slitscan: p.slitscan,
            slit_angle: p.slit_angle,
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
    #[serde(default)]
    pub blend_law: String,
    #[serde(default)]
    pub gliding: bool,
    #[serde(default)]
    pub glide_target: f32,
    #[serde(default)]
    pub glide_duration_beats: f64,
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
    /// Immutable engine identity. Unlike the display index, this survives a
    /// reorder and lets concurrent controllers address the intended layer.
    #[serde(default)]
    pub layer_id: String,
    pub filename: String,
    pub visible: bool,
    pub paused: bool,
    pub opacity: f32,
    pub speed: f32,
    #[serde(default = "default_layer_fps")]
    pub fps: f32,
    pub blend_mode: String,
    pub progress: f32,
    #[serde(default)]
    pub key_mode: u32,
    #[serde(default)]
    pub key_threshold: f32,
    #[serde(default)]
    pub key_softness: f32,
    #[serde(default)]
    pub effects: EffectsSnapshot,
    /// `video` for ordinary decoded clips or `spout` for a live receiver.
    #[serde(default)]
    pub source_kind: String,
    /// Requested/connected Spout sender name. Empty for video layers.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source_name: String,
    #[serde(default)]
    pub source_active: bool,
    #[serde(default)]
    pub source_width: u32,
    #[serde(default)]
    pub source_height: u32,
    #[serde(default)]
    pub source_sequence: u64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source_error: String,
    /// Explicitly tells the performer how non-file sources behave in export.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub offline_export_policy: String,
}

fn default_layer_fps() -> f32 {
    30.0
}

/// Actions the browser can request (processed by the render loop).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action")]
pub enum WebAction {
    /// Defer an otherwise ordinary action until the next four-beat downbeat.
    #[serde(rename = "quantized")]
    Quantized { inner: Box<WebAction> },
    /// Set a master effect parameter
    #[serde(rename = "set_param")]
    SetParam {
        param: String,
        value: serde_json::Value,
    },
    /// Add a layer from the library by filename
    #[serde(rename = "add_layer")]
    AddLayer { filename: String },
    /// Add a live Spout receiver layer by sender name.
    #[serde(rename = "add_spout_layer")]
    AddSpoutLayer { sender: String },
    /// Remove a layer by index
    #[serde(rename = "remove_layer")]
    RemoveLayer {
        index: usize,
        #[serde(default)]
        layer_id: Option<String>,
    },
    /// Move one layer to another stack position.
    #[serde(rename = "move_layer")]
    MoveLayer {
        from: usize,
        to: usize,
        #[serde(default)]
        layer_id: Option<String>,
        #[serde(default)]
        stack_revision: Option<u64>,
    },
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
    SetLayerParam {
        index: usize,
        #[serde(default)]
        layer_id: Option<String>,
        param: String,
        value: serde_json::Value,
    },
    /// Set one direct per-layer effect parameter.
    #[serde(rename = "set_layer_effect")]
    SetLayerEffect {
        index: usize,
        #[serde(default)]
        layer_id: Option<String>,
        param: String,
        value: serde_json::Value,
    },
    /// Reset all direct effects on one layer.
    #[serde(rename = "reset_layer_fx")]
    ResetLayerFx {
        index: usize,
        #[serde(default)]
        layer_id: Option<String>,
    },
    /// Idempotent layer safety/transport setters. Legacy toggles remain below
    /// for old clients, but the bundled panel never depends on toggle parity.
    #[serde(rename = "set_layer_visibility")]
    SetLayerVisibility {
        index: usize,
        #[serde(default)]
        layer_id: Option<String>,
        visible: bool,
    },
    #[serde(rename = "set_layer_paused")]
    SetLayerPaused {
        index: usize,
        #[serde(default)]
        layer_id: Option<String>,
        paused: bool,
    },
    #[serde(rename = "set_master_paused")]
    SetMasterPaused { paused: bool },
    #[serde(rename = "set_blackout")]
    SetBlackout { enabled: bool },
    /// Set an NTSC/VHS effect parameter
    #[serde(rename = "set_ntsc_param")]
    SetNtscParam {
        param: String,
        value: serde_json::Value,
    },
    /// Tap tempo: each tap refines BPM and re-anchors the downbeat
    #[serde(rename = "tap_tempo")]
    TapTempo,
    /// Set BPM directly
    #[serde(rename = "set_bpm")]
    SetBpm { value: f32 },
    /// Set an LFO parameter ("shape" | "beats" | "phase")
    #[serde(rename = "set_lfo")]
    SetLfo {
        index: usize,
        param: String,
        value: serde_json::Value,
    },
    /// Append a new modulation routing (defaults)
    #[serde(rename = "add_routing")]
    AddRouting,
    /// Remove a modulation routing by index
    #[serde(rename = "remove_routing")]
    RemoveRouting { index: usize },
    /// Set a routing parameter ("source" | "target" | "depth")
    #[serde(rename = "set_routing")]
    SetRouting {
        index: usize,
        param: String,
        value: serde_json::Value,
    },
    /// Set an audio input parameter (`enabled`, `gain`, `device`,
    /// `band_count`, or the atomic `band_edges` layout object).
    #[serde(rename = "set_audio")]
    SetAudio {
        param: String,
        value: serde_json::Value,
    },
    /// Set a MIDI parameter ("enabled" | "learn" | "cc0".."cc3")
    #[serde(rename = "set_midi")]
    SetMidi {
        param: String,
        value: serde_json::Value,
    },
    /// Phone orientation sample (degrees, DeviceOrientation convention)
    #[serde(rename = "gyro")]
    Gyro { alpha: f32, beta: f32, gamma: f32 },
    /// Store the latest raw orientation as the centered (0.5) pose.
    #[serde(rename = "gyro_calibrate")]
    GyroCalibrate,
    /// Set one gyro axis response parameter (range, expo, invert).
    #[serde(rename = "set_gyro_config")]
    SetGyroConfig {
        axis: String,
        param: String,
        value: serde_json::Value,
    },
    /// XY performance pad position (0..1 each)
    #[serde(rename = "pad")]
    Pad {
        x: f32,
        y: f32,
        /// True while a pointer owns the pad; false starts spring return.
        #[serde(default = "bool_true")]
        active: bool,
    },
    /// Set one pad response or spring parameter.
    #[serde(rename = "set_pad_config")]
    SetPadConfig {
        axis: String,
        param: String,
        value: serde_json::Value,
    },
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
    /// Select linear or equal-power interpolation.
    #[serde(rename = "set_morph_law")]
    SetMorphLaw { law: String },
    /// Begin an explicit beat-duration glide to A or B.
    #[serde(rename = "morph_glide")]
    MorphGlide { target: f32, duration_beats: f64 },
    /// Rescan the library folder (pushed internally after an upload)
    #[serde(rename = "rescan_library")]
    RescanLibrary,
    /// Cut the output to black (toggle)
    #[serde(rename = "toggle_blackout")]
    ToggleBlackout,
    /// Set a temporal effect parameter
    #[serde(rename = "set_temporal")]
    SetTemporal {
        param: String,
        value: serde_json::Value,
    },
    /// Enable/disable the Spout output sender
    #[serde(rename = "set_spout")]
    SetSpout { enabled: bool },
    /// Start an offline render export
    #[serde(rename = "start_export")]
    StartExport {
        width: u32,
        height: u32,
        fps: u32,
        duration_secs: f32,
        #[serde(default)]
        audio_layer: Option<usize>,
        #[serde(default)]
        audio_layer_id: Option<String>,
    },
    /// Cancel a running export
    #[serde(rename = "cancel_export")]
    CancelExport,
}

fn bool_true() -> bool {
    true
}

impl WebAction {
    fn layer_key(index: usize, layer_id: &Option<String>) -> String {
        layer_id
            .as_deref()
            .filter(|id| !id.is_empty())
            .map_or_else(|| format!("index:{index}"), |id| format!("id:{id}"))
    }

    /// Absolute controls may replace an older pending value with the same
    /// semantic destination. The replacement is moved to the queue tail so a
    /// later value never jumps ahead of an intervening reset or topology edit.
    fn coalesce_key(&self) -> Option<String> {
        match self {
            Self::Quantized { inner } => inner.coalesce_key().map(|key| format!("quantized:{key}")),
            Self::SetParam { param, .. } => Some(format!("master:{param}")),
            Self::SetLayerParam {
                index,
                layer_id,
                param,
                ..
            } => Some(format!(
                "layer:{}:param:{param}",
                Self::layer_key(*index, layer_id)
            )),
            Self::SetLayerEffect {
                index,
                layer_id,
                param,
                ..
            } => Some(format!(
                "layer:{}:effect:{param}",
                Self::layer_key(*index, layer_id)
            )),
            Self::SetLayerVisibility {
                index, layer_id, ..
            } => Some(format!(
                "layer:{}:visible",
                Self::layer_key(*index, layer_id)
            )),
            Self::SetLayerPaused {
                index, layer_id, ..
            } => Some(format!(
                "layer:{}:paused",
                Self::layer_key(*index, layer_id)
            )),
            Self::SetMasterPaused { .. } => Some("master:paused".into()),
            Self::SetBlackout { .. } => Some("master:blackout".into()),
            Self::SetNtscParam { param, .. } => Some(format!("ntsc:{param}")),
            Self::SetBpm { .. } => Some("mod:bpm".into()),
            Self::SetLfo { index, param, .. } => Some(format!("lfo:{index}:{param}")),
            Self::SetRouting { index, param, .. } => Some(format!("routing:{index}:{param}")),
            Self::SetAudio { param, .. } => Some(format!("audio:{param}")),
            Self::SetMidi { param, .. } => Some(format!("midi:{param}")),
            Self::Gyro { .. } => Some("gyro:sample".into()),
            Self::SetGyroConfig { axis, param, .. } => Some(format!("gyro:{axis}:{param}")),
            Self::Pad { .. } => Some("pad:position".into()),
            Self::SetPadConfig { axis, param, .. } => Some(format!("pad:{axis}:{param}")),
            Self::SetMorph { .. } => Some("morph:position".into()),
            Self::SetMorphLaw { .. } => Some("morph:law".into()),
            Self::SetTemporal { param, .. } => Some(format!("temporal:{param}")),
            Self::SetSpout { .. } => Some("spout:enabled".into()),
            Self::CancelExport => Some("export:cancel".into()),
            Self::RescanLibrary => Some("library:rescan".into()),
            _ => None,
        }
    }

    fn is_priority(&self) -> bool {
        match self {
            Self::CancelExport
            | Self::ToggleBlackout
            | Self::SetBlackout { .. }
            | Self::SetMasterPaused { .. }
            | Self::SetLayerPaused { .. }
            | Self::SetLayerVisibility { .. }
            | Self::RescanLibrary => true,
            Self::Pad { active, .. } => !active,
            Self::Quantized { inner } => inner.is_priority(),
            _ => false,
        }
    }
}

fn enqueue_bounded(queue: &mut Vec<WebAction>, action: WebAction) -> EnqueueOutcome {
    if let Some(key) = action.coalesce_key() {
        if let Some(position) = queue
            .iter()
            .rposition(|candidate| candidate.coalesce_key().as_deref() == Some(key.as_str()))
        {
            queue.remove(position);
            queue.push(action);
            return EnqueueOutcome::Coalesced;
        }
    }

    if action.is_priority() {
        if queue.len() >= MAX_PENDING_ACTIONS {
            let position = queue
                .iter()
                .position(|candidate| !candidate.is_priority())
                .unwrap_or(0);
            queue.remove(position);
        }
    } else if queue.len() >= MAX_PENDING_ACTIONS - PRIORITY_ACTION_RESERVE {
        return EnqueueOutcome::Dropped;
    }

    queue.push(action);
    EnqueueOutcome::Added
}

impl EffectsSnapshot {
    pub fn from_uniforms(u: &EffectUniforms) -> Self {
        Self {
            pixelate: u.pixelate_size,
            downsample: u.downsample,
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
        u.downsample = self.downsample.clamp(0.05, 1.0);
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
            "pixelate" => {
                if let Some(n) = v.as_f64() {
                    self.pixelate = n as f32;
                }
            }
            "downsample" => {
                if let Some(n) = v.as_f64() {
                    self.downsample = n as f32;
                }
            }
            "rgb_split" => {
                if let Some(n) = v.as_f64() {
                    self.rgb_split = n as f32;
                }
            }
            "hue_shift" => {
                if let Some(n) = v.as_f64() {
                    self.hue_shift = n as f32;
                }
            }
            "saturation" => {
                if let Some(n) = v.as_f64() {
                    self.saturation = n as f32;
                }
            }
            "brightness" => {
                if let Some(n) = v.as_f64() {
                    self.brightness = n as f32;
                }
            }
            "contrast" => {
                if let Some(n) = v.as_f64() {
                    self.contrast = n as f32;
                }
            }
            "posterize" => {
                if let Some(n) = v.as_f64() {
                    self.posterize = n as f32;
                }
            }
            "invert" => {
                if let Some(b) = v.as_bool() {
                    self.invert = b;
                }
            }
            "grain_intensity" => {
                if let Some(n) = v.as_f64() {
                    self.grain_intensity = n as f32;
                }
            }
            "grain_size" => {
                if let Some(n) = v.as_f64() {
                    self.grain_size = n as f32;
                }
            }
            "grain_algo" => {
                if let Some(n) = v.as_u64() {
                    self.grain_algo = n as u32;
                }
            }
            "color_grain" => {
                if let Some(b) = v.as_bool() {
                    self.color_grain = b;
                }
            }
            "vignette" => {
                if let Some(n) = v.as_f64() {
                    self.vignette = n as f32;
                }
            }
            "color_drift" => {
                if let Some(n) = v.as_f64() {
                    self.color_drift = n as f32;
                }
            }
            "breathe_scale" => {
                if let Some(n) = v.as_f64() {
                    self.breathe_scale = n as f32;
                }
            }
            "breathe_rotation" => {
                if let Some(n) = v.as_f64() {
                    self.breathe_rotation = n as f32;
                }
            }
            "breathe_position" => {
                if let Some(n) = v.as_f64() {
                    self.breathe_position = n as f32;
                }
            }
            _ => {}
        }
    }
}

impl WebState {
    pub fn new() -> Result<Arc<Self>, String> {
        let (tx, _) = broadcast::channel(64);
        let mut token_bytes = [0_u8; 16];
        // A weak control token is worse than refusing to expose the control
        // server. Entropy failure therefore aborts startup instead of falling
        // back to a predictable clock value.
        getrandom::fill(&mut token_bytes)
            .map_err(|error| format!("OS entropy unavailable for web control token: {error}"))?;
        Ok(Arc::new(Self {
            app: RwLock::new(AppSnapshot::default()),
            tx,
            actions: Mutex::new(Vec::new()),
            thumbnails: std::sync::RwLock::new(HashMap::new()),
            preview_frames: std::sync::RwLock::new(HashMap::new()),
            library_folder: std::sync::RwLock::new(None),
            access_token: token_bytes
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect(),
            lan_url: std::sync::RwLock::new(String::new()),
        }))
    }

    pub async fn enqueue_action(&self, action: WebAction) -> EnqueueOutcome {
        let mut queue = self.actions.lock().await;
        enqueue_bounded(&mut queue, action)
    }
}

#[cfg(test)]
mod protocol_tests {
    use super::*;

    fn assert_json_number_near(value: &serde_json::Value, expected: f64) {
        let actual = value.as_f64().expect("JSON number");
        assert!(
            (actual - expected).abs() < 1.0e-6,
            "expected {expected}, got {actual}"
        );
    }

    fn set_bpm(value: f32) -> WebAction {
        WebAction::SetBpm { value }
    }

    #[test]
    fn web_state_token_is_128_bits_and_queue_is_bounded() {
        let state = WebState::new().expect("OS entropy");
        assert_eq!(state.access_token.len(), 32);
        assert!(state
            .access_token
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit()));

        let mut queue = Vec::new();
        for value in 0..1000 {
            assert_ne!(
                enqueue_bounded(&mut queue, set_bpm(value as f32)),
                EnqueueOutcome::Dropped
            );
        }
        assert_eq!(queue.len(), 1, "absolute fader traffic coalesces");

        for _ in 0..MAX_PENDING_ACTIONS * 2 {
            let _ = enqueue_bounded(&mut queue, WebAction::AddRouting);
        }
        assert!(queue.len() <= MAX_PENDING_ACTIONS - PRIORITY_ACTION_RESERVE);
        assert_ne!(
            enqueue_bounded(&mut queue, WebAction::CancelExport),
            EnqueueOutcome::Dropped
        );
        assert!(queue.len() <= MAX_PENDING_ACTIONS);
        assert!(matches!(queue.last(), Some(WebAction::CancelExport)));
    }

    #[test]
    fn layer_identity_and_direct_effect_protocol_round_trip() {
        let effect: WebAction = serde_json::from_str(
            r#"{"action":"set_layer_effect","index":2,"layer_id":"layer-abc","param":"downsample","value":0.5}"#,
        )
        .unwrap();
        assert!(
            matches!(effect, WebAction::SetLayerEffect { index: 2, layer_id: Some(id), param, .. } if id == "layer-abc" && param == "downsample")
        );

        let reorder: WebAction = serde_json::from_str(
            r#"{"action":"move_layer","from":2,"to":0,"layer_id":"layer-abc","stack_revision":7}"#,
        )
        .unwrap();
        assert!(matches!(
            reorder,
            WebAction::MoveLayer {
                stack_revision: Some(7),
                ..
            }
        ));

        let export: WebAction = serde_json::from_str(
            r#"{"action":"start_export","width":1280,"height":720,"fps":30,"duration_secs":1,"audio_layer":2,"audio_layer_id":"layer-abc"}"#,
        )
        .unwrap();
        assert!(
            matches!(export, WebAction::StartExport { audio_layer_id: Some(id), .. } if id == "layer-abc")
        );
    }

    #[test]
    fn downsample_snapshot_and_browser_control_are_complete() {
        let mut uniforms = EffectUniforms {
            downsample: 0.42,
            ..EffectUniforms::default()
        };
        let snapshot = EffectsSnapshot::from_uniforms(&uniforms);
        assert!((snapshot.downsample - 0.42).abs() < f32::EPSILON);
        let mut changed = snapshot.clone();
        changed.apply_param("downsample", &serde_json::json!(0.25));
        changed.apply_to_uniforms(&mut uniforms);
        assert!((uniforms.downsample - 0.25).abs() < f32::EPSILON);
        assert!(include_str!("../../static/index.html").contains("data-param=\"downsample\""));
        assert!(include_str!("../../static/app.js").contains("'set_layer_effect'"));
    }

    #[test]
    fn browser_scrubs_bootstrap_key_and_reports_every_layer_source_error() {
        let js = include_str!("../../static/app.js");
        let html = include_str!("../../static/index.html");
        assert!(js.contains("url.searchParams.delete('key')"));
        assert!(js.contains("window.history.replaceState"));
        assert!(js.contains("<div class=\"layer-source-status\" role=\"status\""));
        assert!(js.contains("video decoder: ${layer.source_error}"));
        assert!(js.contains("header.setAttribute('aria-expanded'"));
        assert!(js.contains("aria-keyshortcuts=\"ArrowUp ArrowDown Home End\""));
        assert!(js.contains("item.setAttribute('role', 'button')"));
        assert!(js.contains("window.confirm"));
        assert!(html.contains("id=\"output-status\" role=\"status\" aria-live=\"polite\""));
        assert!(js.contains("syncOutputWindow(msg.output_window, msg.output_error)"));
    }

    #[test]
    fn persistent_layer_cards_toggle_from_the_latest_snapshot() {
        let js = include_str!("../../static/app.js");
        assert!(js.contains("card._layerState = layer"));
        assert!(js.contains("const current = card._layerState || layer;"));
        assert!(js.contains("paused: !current.paused"));
        assert!(js.contains("visible: !current.visible"));
    }

    #[test]
    fn new_control_actions_deserialize_with_explicit_protocol_names() {
        let action: WebAction = serde_json::from_str(r#"{"action":"gyro_calibrate"}"#).unwrap();
        assert!(matches!(action, WebAction::GyroCalibrate));

        let action: WebAction = serde_json::from_str(
            r#"{"action":"set_gyro_config","axis":"roll","param":"invert","value":true}"#,
        )
        .unwrap();
        assert!(
            matches!(action, WebAction::SetGyroConfig { axis, param, value }
            if axis == "roll" && param == "invert" && value == serde_json::json!(true))
        );

        let action: WebAction = serde_json::from_str(
            r#"{"action":"set_pad_config","axis":"x","param":"quantize","value":8}"#,
        )
        .unwrap();
        assert!(
            matches!(action, WebAction::SetPadConfig { axis, param, value }
            if axis == "x" && param == "quantize" && value == serde_json::json!(8))
        );

        let action: WebAction =
            serde_json::from_str(r#"{"action":"pad","x":0.2,"y":0.8,"active":false}"#).unwrap();
        assert!(matches!(action, WebAction::Pad { x, y, active }
            if (x - 0.2).abs() < f32::EPSILON && (y - 0.8).abs() < f32::EPSILON && !active));

        let action: WebAction =
            serde_json::from_str(r#"{"action":"move_layer","from":3,"to":1}"#).unwrap();
        assert!(matches!(
            action,
            WebAction::MoveLayer {
                from: 3,
                to: 1,
                layer_id: None,
                stack_revision: None
            }
        ));

        let action: WebAction =
            serde_json::from_str(r#"{"action":"add_spout_layer","sender":"Resolume Composition"}"#)
                .unwrap();
        assert!(matches!(action, WebAction::AddSpoutLayer { sender }
            if sender == "Resolume Composition"));

        let value = serde_json::to_value(WebAction::GyroCalibrate).unwrap();
        assert_eq!(value["action"], "gyro_calibrate");
        let value = serde_json::to_value(WebAction::SetPadConfig {
            axis: "both".into(),
            param: "spring_rate".into(),
            value: serde_json::json!(4.0),
        })
        .unwrap();
        assert_eq!(value["action"], "set_pad_config");
        assert_eq!(value["axis"], "both");
        assert_eq!(value["param"], "spring_rate");
    }

    #[test]
    fn legacy_pad_action_defaults_to_active() {
        let action: WebAction =
            serde_json::from_str(r#"{"action":"pad","x":0.5,"y":0.5}"#).unwrap();
        assert!(matches!(action, WebAction::Pad { active: true, .. }));
    }

    #[test]
    fn quantized_action_wraps_existing_protocol_without_changing_it() {
        let action: WebAction = serde_json::from_str(
            r#"{"action":"quantized","inner":{"action":"set_morph","value":0.75}}"#,
        )
        .unwrap();
        assert!(matches!(
            action,
            WebAction::Quantized { inner }
                if matches!(*inner, WebAction::SetMorph { value } if (value - 0.75).abs() < f32::EPSILON)
        ));
    }

    #[test]
    fn routing_snapshot_accepts_legacy_and_round_trips_response_fields() {
        let legacy: RoutingSnapshot =
            serde_json::from_str(r#"{"source":"lfo0","target":"rgb_split","depth":0.5}"#).unwrap();
        assert_eq!(legacy.curve, "linear");
        assert_eq!(legacy.curve_amount, 0.0);
        assert_eq!(legacy.attack, 0.0);
        assert_eq!(legacy.release, 0.0);

        let routing = RoutingSnapshot {
            source: "pad_x".into(),
            target: "morph".into(),
            depth: -0.75,
            curve: "s_curve".into(),
            curve_amount: 0.4,
            attack: 0.12,
            release: 1.5,
        };
        let value = serde_json::to_value(&routing).unwrap();
        assert_eq!(value["curve"], "s_curve");
        assert_json_number_near(&value["curve_amount"], 0.4);
        assert_json_number_near(&value["attack"], 0.12);
        assert_json_number_near(&value["release"], 1.5);
    }

    #[test]
    fn legacy_mod_snapshot_defaults_new_configs() {
        let legacy = r#"{
            "bpm":120.0,
            "beat":0.0,
            "lfos":[],
            "routings":[],
            "gyro":[0.5,0.5,0.5],
            "pad":[0.5,0.5]
        }"#;
        let snapshot: ModSnapshot = serde_json::from_str(legacy).unwrap();
        assert_eq!(snapshot.gyro_config.yaw.range, 180.0);
        assert_eq!(snapshot.gyro_config.pitch.range, 90.0);
        assert_eq!(snapshot.pad_config.x.curve, "linear");
        assert!(!snapshot.pad_config.spring_enabled);
        assert_eq!(snapshot.pad_config.spring_rate, 4.0);
    }

    #[test]
    fn audio_band_protocol_round_trips_and_defaults_legacy_snapshots() {
        let legacy: AudioSnapshot = serde_json::from_str(
            r#"{"enabled":false,"gain":1.0,"level":0.0,"bass":0.1,"mid":0.2,"high":0.3,"onset":0.0}"#,
        )
        .unwrap();
        assert_eq!(legacy.band_count, 3);
        assert_eq!(legacy.band_ceiling_hz, 8000.0);
        assert!(legacy.bands.is_empty());

        let current = AudioSnapshot {
            band_count: 8,
            band_edges: vec![100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0],
            band_ceiling_hz: 12_800.0,
            bands: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            ..AudioSnapshot::default()
        };
        let value = serde_json::to_value(current).unwrap();
        assert_eq!(value["band_count"], 8);
        assert_eq!(value["band_edges"].as_array().unwrap().len(), 7);
        assert_eq!(value["bands"].as_array().unwrap().len(), 8);
        assert_json_number_near(&value["band_ceiling_hz"], 12_800.0);

        for raw in [
            r#"{"action":"set_audio","param":"band_count","value":6}"#,
            r#"{"action":"set_audio","param":"band_edges","value":{"count":6,"edges":[120,480,1500,4000,9000],"ceiling":16000}}"#,
        ] {
            let action: WebAction = serde_json::from_str(raw).unwrap();
            assert!(matches!(action, WebAction::SetAudio { .. }));
        }
    }

    #[test]
    fn audio_band_browser_contract_exposes_count_edges_meters_and_sources() {
        let html = include_str!("../../static/index.html");
        let js = include_str!("../../static/app.js");
        assert!(html.contains("id=\"audio-band-count\""));
        assert!(html.contains("id=\"audio-band-edges\""));
        assert!(html.contains("id=\"audio-high-edge\""));
        assert!(html.contains("id=\"audio-extra-band-meters\""));
        assert!(js.contains("param: 'band_count'"));
        assert!(js.contains("value: { count, edges, ceiling }"));
        for index in 1..=crate::audio::MAX_AUDIO_BANDS {
            assert!(
                js.contains(&format!("['audio_band{index}', 'Band {index}']")),
                "browser route menu must expose audio band {index}"
            );
        }
    }

    #[test]
    fn config_snapshots_serialize_every_browser_control_field() {
        let gyro = GyroConfigSnapshot {
            yaw: AxisConfigSnapshot {
                range: 30.0,
                expo: 0.5,
                invert: true,
            },
            pitch: AxisConfigSnapshot {
                range: 45.0,
                expo: -0.25,
                invert: false,
            },
            roll: AxisConfigSnapshot {
                range: 60.0,
                expo: 1.0,
                invert: true,
            },
        };
        let gyro_value = serde_json::to_value(gyro).unwrap();
        assert_json_number_near(&gyro_value["yaw"]["range"], 30.0);
        assert_json_number_near(&gyro_value["yaw"]["expo"], 0.5);
        assert_eq!(gyro_value["yaw"]["invert"], true);
        assert_json_number_near(&gyro_value["pitch"]["range"], 45.0);
        assert_json_number_near(&gyro_value["roll"]["range"], 60.0);

        let pad = PadConfigSnapshot {
            x: PadAxisConfigSnapshot {
                curve: "exp".into(),
                curve_amount: 0.8,
                quantize: 8,
            },
            y: PadAxisConfigSnapshot {
                curve: "steps".into(),
                curve_amount: -0.4,
                quantize: 16,
            },
            spring_enabled: true,
            spring_rate: 6.5,
        };
        let pad_value = serde_json::to_value(pad).unwrap();
        assert_eq!(pad_value["x"]["curve"], "exp");
        assert_json_number_near(&pad_value["x"]["curve_amount"], 0.8);
        assert_eq!(pad_value["x"]["quantize"], 8);
        assert_eq!(pad_value["y"]["curve"], "steps");
        assert_json_number_near(&pad_value["y"]["curve_amount"], -0.4);
        assert_eq!(pad_value["y"]["quantize"], 16);
        assert_eq!(pad_value["spring_enabled"], true);
        assert_json_number_near(&pad_value["spring_rate"], 6.5);
    }

    #[test]
    fn legacy_app_snapshot_defaults_export_status() {
        let mut value = serde_json::to_value(AppSnapshot::default()).unwrap();
        value.as_object_mut().unwrap().remove("export_status");
        let restored: AppSnapshot = serde_json::from_value(value).unwrap();
        assert_eq!(restored.export_status, "");
    }
}
