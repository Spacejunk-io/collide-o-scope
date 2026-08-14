//! Shared state between the web control panel and the render engine.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

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
    /// Server-owned phone-stream membership and monotonic sample freshness.
    gyro_streams: std::sync::Mutex<GyroStreamRegistry>,
    next_client_id: AtomicU64,
}

/// A phone normally publishes at roughly 30 Hz. This allows substantial
/// mobile scheduling jitter without permitting a vanished pose to remain
/// applied indefinitely.
pub const GYRO_SAMPLE_TIMEOUT: Duration = Duration::from_millis(1_500);

#[derive(Default)]
struct GyroStreamRegistry {
    /// Clients which have used the explicit start/stop protocol.
    declared_clients: HashSet<u64>,
    /// Clients currently claiming ownership of a live sensor stream.
    streamers: HashSet<u64>,
    last_sample: Option<Instant>,
    ever_enabled: bool,
}

impl GyroStreamRegistry {
    fn set_stream(&mut self, client_id: u64, enabled: bool) {
        self.declared_clients.insert(client_id);
        if enabled {
            self.ever_enabled = true;
            self.streamers.insert(client_id);
        } else {
            self.streamers.remove(&client_id);
        }
    }

    fn note_sample_at(&mut self, client_id: u64, now: Instant) {
        // Older panels have no gyro_stream action. Their first valid sample
        // implicitly starts a stream, while an explicitly stopped new panel
        // cannot be reactivated by a late in-flight sample.
        if !self.declared_clients.contains(&client_id) {
            self.ever_enabled = true;
            self.streamers.insert(client_id);
        }
        if self.streamers.contains(&client_id) {
            self.last_sample = Some(now);
        }
    }

    fn disconnect(&mut self, client_id: u64) {
        self.streamers.remove(&client_id);
        self.declared_clients.remove(&client_id);
    }

    fn status_at(&self, now: Instant) -> GyroStatusSnapshot {
        let sample_age_ms = self.last_sample.map(|sample| {
            now.saturating_duration_since(sample)
                .as_millis()
                .min(u128::from(u64::MAX)) as u64
        });
        let active = !self.streamers.is_empty()
            && self
                .last_sample
                .is_some_and(|sample| now.saturating_duration_since(sample) <= GYRO_SAMPLE_TIMEOUT);
        GyroStatusSnapshot {
            active,
            stale: self.ever_enabled && !active,
            streamers: self.streamers.len(),
            sample_age_ms,
        }
    }
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
    /// Result of the most recent frictionless patch capture. The engine owns
    /// this text so clients never manufacture a successful-save indication.
    #[serde(default)]
    pub patch_save_status: String,
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
            patch_save_status: String::new(),
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
    #[serde(default)]
    pub key_mode: u32,
    #[serde(default = "default_key_color")]
    pub key_color: [f32; 3],
    #[serde(default = "default_key_threshold")]
    pub key_threshold: f32,
    #[serde(default = "default_key_softness")]
    pub key_softness: f32,
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

fn default_cellular_scale() -> f32 {
    10.0
}

fn default_key_color() -> [f32; 3] {
    [0.0, 1.0, 0.0]
}

fn default_key_threshold() -> f32 {
    0.5
}

fn default_key_softness() -> f32 {
    0.1
}

fn default_key_tolerance() -> f32 {
    0.15
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
            key_mode: 0,
            key_color: default_key_color(),
            key_threshold: default_key_threshold(),
            key_softness: default_key_softness(),
            key_tolerance: default_key_tolerance(),
            cellular_amount: 0.0,
            cellular_scale: default_cellular_scale(),
            cellular_warp: default_cellular_warp(),
            cellular_speed: default_cellular_speed(),
            cellular_gap_amount: 0.0,
            cellular_gap_threshold: default_cellular_gap_threshold(),
            cellular_gap_softness: default_cellular_gap_softness(),
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
    /// Authoritative server view of phone stream ownership and freshness.
    #[serde(default)]
    pub gyro_status: GyroStatusSnapshot,
    /// Engine-owned response, quantize, and spring settings for the XY pad.
    #[serde(default)]
    pub pad_config: PadConfigSnapshot,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct GyroStatusSnapshot {
    /// At least one declared/legacy streamer has supplied a recent sample.
    #[serde(default)]
    pub active: bool,
    /// A stream existed or was requested, but no sample is currently fresh.
    #[serde(default)]
    pub stale: bool,
    /// Number of connected clients currently claiming an enabled stream.
    #[serde(default)]
    pub streamers: usize,
    /// Monotonic age of the most recently accepted sample.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_age_ms: Option<u64>,
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
    /// Stable runtime identity. It is deliberately not persisted in patches.
    #[serde(default)]
    pub route_id: String,
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
    /// Cached shaped/slewed source value before route depth, in -1..1.
    #[serde(default)]
    pub value: f32,
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
    /// `live` for device/Windows playback capture, `file` for deterministic
    /// circular analysis of an imported clip.
    #[serde(default = "default_audio_source_kind")]
    pub source_kind: String,
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
    /// Windows output endpoints available through WASAPI loopback capture.
    #[serde(default)]
    pub system_playback_devices: Vec<String>,
    /// Preferred device name ("" = system default).
    #[serde(default)]
    pub selected: String,
    /// Device that is actually producing samples after fallback resolution.
    #[serde(default)]
    pub active_device: String,
    /// True when `active_device` differs from the requested `selected` device.
    #[serde(default)]
    pub using_fallback: bool,
    /// Audio-only files in the active library.
    #[serde(default)]
    pub clip_files: Vec<String>,
    /// Persisted/selected clip source identity.
    #[serde(default)]
    pub clip_path: String,
    #[serde(default)]
    pub clip_loading: bool,
    #[serde(default)]
    pub clip_duration_secs: f64,
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

fn default_audio_source_kind() -> String {
    crate::modulation::AUDIO_SOURCE_LIVE.to_string()
}

fn default_audio_band_ceiling() -> f32 {
    8000.0
}

impl Default for AudioSnapshot {
    fn default() -> Self {
        Self {
            enabled: false,
            source_kind: default_audio_source_kind(),
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
            system_playback_devices: Vec::new(),
            selected: String::new(),
            active_device: String::new(),
            using_fallback: false,
            clip_files: Vec::new(),
            clip_path: String::new(),
            clip_loading: false,
            clip_duration_secs: 0.0,
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
                    route_id: r.route_id().to_string(),
                    source: r.source.as_str().to_string(),
                    target: r.target().to_owned(),
                    depth: r.depth,
                    curve: r.curve.as_str().to_string(),
                    curve_amount: r.curve_amount,
                    attack: r.attack,
                    release: r.release,
                    value: r.cached_value(),
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
            gyro_status: GyroStatusSnapshot::default(),
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalSnapshot {
    pub feedback: f32,
    pub fb_zoom: f32,
    pub fb_rotate: f32,
    pub slitscan: f32,
    #[serde(default)]
    pub slit_angle: f32,
    pub slit_axis: f32,
    #[serde(default)]
    pub key_mode: u32,
    #[serde(default = "default_temporal_key_threshold")]
    pub key_threshold: f32,
    #[serde(default = "default_temporal_key_softness")]
    pub key_softness: f32,
    #[serde(default = "default_temporal_key_history")]
    pub key_history: f32,
}

fn default_temporal_key_threshold() -> f32 {
    0.1
}

fn default_temporal_key_softness() -> f32 {
    0.03
}

fn default_temporal_key_history() -> f32 {
    1.0
}

impl Default for TemporalSnapshot {
    fn default() -> Self {
        Self {
            feedback: 0.0,
            fb_zoom: 1.0,
            fb_rotate: 0.0,
            slitscan: 0.0,
            slit_angle: 0.0,
            slit_axis: 0.0,
            key_mode: 0,
            key_threshold: default_temporal_key_threshold(),
            key_softness: default_temporal_key_softness(),
            key_history: default_temporal_key_history(),
        }
    }
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
            key_mode: p.key_mode.round().clamp(0.0, 4.0) as u32,
            key_threshold: p.key_threshold,
            key_softness: p.key_softness,
            key_history: p.key_history,
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
    /// Remaining beat duration at the published authoritative clock.
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
    /// Current tempo estimate while the external clock is active.
    #[serde(default)]
    pub clock_bpm: f32,
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
    /// True when the layer skips Digital/Analog/Cellular/Motion/VHS master
    /// processing. The later program-wide Temporal stage still applies.
    #[serde(default)]
    pub bypass_master_fx: bool,
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
    #[serde(default = "default_key_color")]
    pub key_color: [f32; 3],
    #[serde(default = "default_key_tolerance")]
    pub key_tolerance: f32,
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
    /// Revert the complete master visual state. Layers, transport, BPM, and
    /// input/device selections are preserved.
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
    RemoveRouting {
        index: usize,
        #[serde(default)]
        route_id: Option<String>,
    },
    /// Set a routing parameter ("source" | "target" | "depth")
    #[serde(rename = "set_routing")]
    SetRouting {
        index: usize,
        #[serde(default)]
        route_id: Option<String>,
        /// Stable identity of a positional `layerN_*` target. Bundled clients
        /// include this when changing a route target so an intervening stack
        /// edit cannot silently retarget the route to a different layer.
        #[serde(default)]
        target_layer_id: Option<String>,
        /// Stack generation observed when `target_layer_id` was captured.
        /// The stable ID remains authoritative across harmless index drift;
        /// this revision is diagnostic/precondition metadata for receivers.
        #[serde(default)]
        layer_stack_revision: Option<u64>,
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
    /// Declare this WebSocket as an enabled/disabled phone sensor streamer.
    /// Older clients remain compatible: their first gyro sample implicitly
    /// enables their connection until it disconnects.
    #[serde(rename = "gyro_stream")]
    GyroStream { enabled: bool },
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
    /// Set fullscreen audience output explicitly. Current clients use this
    /// idempotent command so a delayed/retried packet cannot invert the
    /// performer's requested state.
    #[serde(rename = "set_output_window")]
    SetOutputWindow { enabled: bool },
    /// Legacy open/close command retained for older control panels.
    #[serde(rename = "toggle_output_window")]
    ToggleOutputWindow,
    /// Capture current parameters into morph slot "a" or "b"
    #[serde(rename = "morph_capture")]
    MorphCapture {
        slot: String,
        /// Optional stack generation supplied by current clients. A capture
        /// queued against an older topology is rejected at execution time.
        #[serde(default)]
        stack_revision: Option<u64>,
    },
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
    /// Capture the complete current performance state into the local patch
    /// corpus without opening a native file dialog.
    #[serde(rename = "quick_save_patch")]
    QuickSavePatch,
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

    fn routing_key(index: usize, route_id: &Option<String>) -> String {
        route_id
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
            Self::SetOutputWindow { .. } => Some("output:enabled".into()),
            Self::SetNtscParam { param, .. } => Some(format!("ntsc:{param}")),
            Self::SetBpm { .. } => Some("mod:bpm".into()),
            Self::SetLfo { index, param, .. } => Some(format!("lfo:{index}:{param}")),
            Self::SetRouting {
                index,
                route_id,
                param,
                ..
            } => Some(format!(
                "routing:{}:{param}",
                Self::routing_key(*index, route_id)
            )),
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
            Self::QuickSavePatch => Some("patch:quick-save".into()),
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
            | Self::SetOutputWindow { .. }
            | Self::ToggleOutputWindow
            | Self::SetMasterPaused { .. }
            | Self::ResetFx
            | Self::SetLayerPaused { .. }
            | Self::SetLayerVisibility { .. }
            | Self::QuickSavePatch
            | Self::RescanLibrary => true,
            Self::Pad { active, .. } => !active,
            Self::Quantized { inner } => inner.is_priority(),
            _ => false,
        }
    }
}

fn enqueue_bounded(queue: &mut Vec<WebAction>, action: WebAction) -> EnqueueOutcome {
    // A rescan has no payload. Keep the earliest pending barrier in place so
    // later uploads cannot move it behind a clip-selection action.
    if matches!(action, WebAction::RescanLibrary)
        && queue
            .iter()
            .any(|candidate| matches!(candidate, WebAction::RescanLibrary))
    {
        return EnqueueOutcome::Coalesced;
    }
    if let Some(key) = action.coalesce_key() {
        // Captures, resets, saves, topology changes, and other uncoalesced
        // commands observe ordered state. Never move a later absolute value
        // across one of those semantic barriers.
        let barrier = queue
            .iter()
            .rposition(|candidate| candidate.coalesce_key().is_none())
            .map_or(0, |position| position + 1);
        if let Some(position) = queue[barrier..]
            .iter()
            .rposition(|candidate| candidate.coalesce_key().as_deref() == Some(key.as_str()))
            .map(|position| barrier + position)
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
            key_mode: u.key_mode.round().clamp(0.0, 4.0) as u32,
            key_color: u.key_color,
            key_threshold: u.key_threshold,
            key_softness: u.key_softness,
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
        u.key_mode = self.key_mode.min(4) as f32;
        u.key_color = self.key_color.map(|channel| channel.clamp(0.0, 1.0));
        u.key_threshold = self.key_threshold.clamp(0.0, 1.0);
        u.key_softness = self.key_softness.clamp(0.0, 0.5);
        u.key_tolerance = self.key_tolerance.clamp(0.0, 1.0);
        u.cellular_amount = self.cellular_amount.clamp(0.0, 1.0);
        u.cellular_scale = self.cellular_scale.clamp(2.0, 32.0);
        u.cellular_warp = self.cellular_warp.clamp(0.0, 1.0);
        u.cellular_speed = self.cellular_speed.clamp(0.0, 2.0);
        u.cellular_gap_amount = self.cellular_gap_amount.clamp(0.0, 1.0);
        u.cellular_gap_threshold = self.cellular_gap_threshold.clamp(0.0, 1.0);
        u.cellular_gap_softness = self.cellular_gap_softness.clamp(0.0, 0.5);
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
            "key_mode" => {
                if let Some(n) = v.as_u64() {
                    self.key_mode = (n as u32).min(4);
                }
            }
            "key_color_r" | "key_color_g" | "key_color_b" => {
                if let Some(n) = v.as_f64() {
                    let index = match param {
                        "key_color_r" => 0,
                        "key_color_g" => 1,
                        _ => 2,
                    };
                    self.key_color[index] = n as f32;
                }
            }
            "key_threshold" => {
                if let Some(n) = v.as_f64() {
                    self.key_threshold = n as f32;
                }
            }
            "key_softness" => {
                if let Some(n) = v.as_f64() {
                    self.key_softness = n as f32;
                }
            }
            "key_tolerance" => {
                if let Some(n) = v.as_f64() {
                    self.key_tolerance = n as f32;
                }
            }
            "cellular_amount" => {
                if let Some(n) = v.as_f64() {
                    self.cellular_amount = n as f32;
                }
            }
            "cellular_scale" => {
                if let Some(n) = v.as_f64() {
                    self.cellular_scale = n as f32;
                }
            }
            "cellular_warp" => {
                if let Some(n) = v.as_f64() {
                    self.cellular_warp = n as f32;
                }
            }
            "cellular_speed" => {
                if let Some(n) = v.as_f64() {
                    self.cellular_speed = n as f32;
                }
            }
            "cellular_gap_amount" => {
                if let Some(n) = v.as_f64() {
                    self.cellular_gap_amount = n as f32;
                }
            }
            "cellular_gap_threshold" => {
                if let Some(n) = v.as_f64() {
                    self.cellular_gap_threshold = n as f32;
                }
            }
            "cellular_gap_softness" => {
                if let Some(n) = v.as_f64() {
                    self.cellular_gap_softness = n as f32;
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
            gyro_streams: std::sync::Mutex::new(GyroStreamRegistry::default()),
            next_client_id: AtomicU64::new(1),
        }))
    }

    pub async fn enqueue_action(&self, action: WebAction) -> EnqueueOutcome {
        let mut queue = self.actions.lock().await;
        enqueue_bounded(&mut queue, action)
    }

    pub fn allocate_client_id(&self) -> u64 {
        self.next_client_id.fetch_add(1, Ordering::Relaxed)
    }

    pub fn set_gyro_stream(&self, client_id: u64, enabled: bool) {
        self.gyro_streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .set_stream(client_id, enabled);
    }

    pub fn note_gyro_sample(&self, client_id: u64) {
        self.gyro_streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .note_sample_at(client_id, Instant::now());
    }

    pub fn disconnect_gyro_client(&self, client_id: u64) {
        self.gyro_streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .disconnect(client_id);
    }

    pub fn gyro_status(&self) -> GyroStatusSnapshot {
        self.gyro_streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .status_at(Instant::now())
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
    fn output_window_absolute_state_is_priority_and_coalesces() {
        let enabled: WebAction =
            serde_json::from_str(r#"{"action":"set_output_window","enabled":true}"#).unwrap();
        assert!(matches!(
            enabled,
            WebAction::SetOutputWindow { enabled: true }
        ));
        assert_eq!(enabled.coalesce_key().as_deref(), Some("output:enabled"));
        assert!(enabled.is_priority());

        let mut queue = vec![WebAction::AddRouting; MAX_PENDING_ACTIONS];
        assert_eq!(enqueue_bounded(&mut queue, enabled), EnqueueOutcome::Added);
        assert!(matches!(
            queue.last(),
            Some(WebAction::SetOutputWindow { enabled: true })
        ));
        assert_eq!(
            enqueue_bounded(&mut queue, WebAction::SetOutputWindow { enabled: false },),
            EnqueueOutcome::Coalesced
        );
        assert!(matches!(
            queue.last(),
            Some(WebAction::SetOutputWindow { enabled: false })
        ));

        let legacy: WebAction =
            serde_json::from_str(r#"{"action":"toggle_output_window"}"#).unwrap();
        assert!(matches!(legacy, WebAction::ToggleOutputWindow));
        assert!(legacy.is_priority());
        assert!(legacy.coalesce_key().is_none());
    }

    #[test]
    fn morph_capture_is_an_ordering_barrier_for_fader_coalescing() {
        let mut queue = Vec::new();
        assert_eq!(
            enqueue_bounded(&mut queue, WebAction::SetMorph { value: 0.2 }),
            EnqueueOutcome::Added
        );
        assert_eq!(
            enqueue_bounded(
                &mut queue,
                WebAction::MorphCapture {
                    slot: "a".into(),
                    stack_revision: Some(7),
                },
            ),
            EnqueueOutcome::Added
        );
        assert_eq!(
            enqueue_bounded(&mut queue, WebAction::SetMorph { value: 0.8 }),
            EnqueueOutcome::Added
        );
        assert_eq!(queue.len(), 3);
        assert!(matches!(queue[0], WebAction::SetMorph { value } if value == 0.2));
        assert!(matches!(
            queue[1],
            WebAction::MorphCapture {
                stack_revision: Some(7),
                ..
            }
        ));
        assert!(matches!(queue[2], WebAction::SetMorph { value } if value == 0.8));
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

        let bypass: WebAction = serde_json::from_str(
            r#"{"action":"set_layer_param","index":2,"layer_id":"layer-abc","param":"bypass_master_fx","value":true}"#,
        )
        .unwrap();
        assert!(
            matches!(bypass, WebAction::SetLayerParam { index: 2, layer_id: Some(id), param, value }
                if id == "layer-abc" && param == "bypass_master_fx" && value == serde_json::json!(true))
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
    fn legacy_layer_snapshot_defaults_master_fx_bypass_to_off() {
        let current = LayerSnapshot {
            layer_id: "17".into(),
            filename: "plate.png".into(),
            visible: true,
            bypass_master_fx: true,
            paused: false,
            opacity: 1.0,
            speed: 1.0,
            fps: 30.0,
            blend_mode: "normal".into(),
            progress: 0.0,
            key_mode: 0,
            key_threshold: 0.5,
            key_softness: 0.1,
            key_color: default_key_color(),
            key_tolerance: default_key_tolerance(),
            effects: EffectsSnapshot::default(),
            source_kind: "image".into(),
            source_name: String::new(),
            source_active: true,
            source_width: 1920,
            source_height: 1080,
            source_sequence: 0,
            source_error: String::new(),
            offline_export_policy: String::new(),
        };
        let mut value = serde_json::to_value(current).unwrap();
        value.as_object_mut().unwrap().remove("bypass_master_fx");
        let legacy: LayerSnapshot = serde_json::from_value(value).unwrap();
        assert!(!legacy.bypass_master_fx);
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
    fn cellular_snapshot_controls_and_modulation_targets_are_complete() {
        let mut uniforms = EffectUniforms {
            cellular_amount: 0.7,
            cellular_scale: 18.0,
            cellular_warp: 0.6,
            cellular_speed: 1.25,
            cellular_gap_amount: 0.85,
            cellular_gap_threshold: 0.45,
            cellular_gap_softness: 0.14,
            ..EffectUniforms::default()
        };
        let snapshot = EffectsSnapshot::from_uniforms(&uniforms);
        assert!((snapshot.cellular_amount - 0.7).abs() < f32::EPSILON);
        assert!((snapshot.cellular_scale - 18.0).abs() < f32::EPSILON);
        assert!((snapshot.cellular_warp - 0.6).abs() < f32::EPSILON);
        assert!((snapshot.cellular_speed - 1.25).abs() < f32::EPSILON);
        assert!((snapshot.cellular_gap_amount - 0.85).abs() < f32::EPSILON);
        assert!((snapshot.cellular_gap_threshold - 0.45).abs() < f32::EPSILON);
        assert!((snapshot.cellular_gap_softness - 0.14).abs() < f32::EPSILON);

        let mut changed = EffectsSnapshot::default();
        for (param, value) in [
            ("cellular_amount", 0.5),
            ("cellular_scale", 24.0),
            ("cellular_warp", 0.8),
            ("cellular_speed", 1.5),
            ("cellular_gap_amount", 0.75),
            ("cellular_gap_threshold", 0.6),
            ("cellular_gap_softness", 0.11),
        ] {
            changed.apply_param(param, &serde_json::json!(value));
        }
        changed.apply_to_uniforms(&mut uniforms);
        assert!((uniforms.cellular_amount - 0.5).abs() < f32::EPSILON);
        assert!((uniforms.cellular_scale - 24.0).abs() < f32::EPSILON);
        assert!((uniforms.cellular_warp - 0.8).abs() < f32::EPSILON);
        assert!((uniforms.cellular_speed - 1.5).abs() < f32::EPSILON);
        assert!((uniforms.cellular_gap_amount - 0.75).abs() < f32::EPSILON);
        assert!((uniforms.cellular_gap_threshold - 0.6).abs() < f32::EPSILON);
        assert!((uniforms.cellular_gap_softness - 0.11).abs() < f32::EPSILON);

        let mut legacy_json = serde_json::to_value(EffectsSnapshot::default()).unwrap();
        let legacy_object = legacy_json.as_object_mut().unwrap();
        legacy_object.remove("cellular_gap_amount");
        legacy_object.remove("cellular_gap_threshold");
        legacy_object.remove("cellular_gap_softness");
        let legacy: EffectsSnapshot = serde_json::from_value(legacy_json).unwrap();
        assert_eq!(legacy.cellular_gap_amount, 0.0);
        assert_eq!(legacy.cellular_gap_threshold, 0.65);
        assert_eq!(legacy.cellular_gap_softness, 0.08);

        let html = include_str!("../../static/index.html");
        let js = include_str!("../../static/app.js");
        for field in [
            "cellular_amount",
            "cellular_scale",
            "cellular_warp",
            "cellular_speed",
        ] {
            assert!(html.contains(&format!("data-param=\"{field}\"")));
            assert!(js.contains(&format!("'{field}'")));
        }
        for field in [
            "cellular_gap_amount",
            "cellular_gap_threshold",
            "cellular_gap_softness",
        ] {
            assert!(html.contains(&format!("data-param=\"{field}\"")));
            assert!(js.contains(&format!("['{field}',")));
        }
        assert!(html.contains("keyed ridges can reveal lower content"));
        assert!(js.contains("'Master Cell Gap Key'"));
        assert!(html.contains("data-group=\"cellular\""));
        assert!(js.contains("class=\"layer-cellular-toggle\""));
        assert!(js.contains("aria-controls=\"layer-cellular-body-${index}\""));
        assert!(js.contains("class=\"layer-cellular-body\""));
        for contract in [
            "const MASTER_MOD_TARGETS = MOD_TARGETS.slice()",
            "Math.min(MAX_MOD_LAYERS, latestLayerIdentities.length)",
            "targets.push([`layer${layer}_${suffix}`, `L${layer} ${label}`])",
            "groups.push([`Layer ${layer}`, targets])",
            "routes: routings.map((routing, index)",
        ] {
            assert!(
                js.contains(contract),
                "missing bounded target menu contract: {contract}"
            );
        }
    }

    #[test]
    fn master_control_groups_stay_in_their_intended_columns() {
        let html = include_str!("../../static/index.html");
        let video = html.find("data-fx-column=\"video\"").unwrap();
        let mod_morph = html.find("data-fx-column=\"mod-morph\"").unwrap();
        let sources = html.find("data-fx-column=\"sources\"").unwrap();
        let io = html.find("data-fx-column=\"io\"").unwrap();
        assert!(video < mod_morph && mod_morph < sources && sources < io);

        let video_column = &html[video..mod_morph];
        assert!(video_column.contains("data-group=\"digital\""));
        assert!(video_column.contains("data-group=\"analog\""));
        assert!(video_column.contains("id=\"vhs-group\""));
        assert!(!video_column.contains("data-group=\"cellular\""));

        let second_column = &html[mod_morph..sources];
        let ordered = [
            "id=\"morph-group\"",
            "class=\"fx-group\" data-group=\"cellular\"",
            "class=\"fx-group\" data-group=\"motion\"",
            "id=\"temporal-group\"",
            "id=\"audio-group\"",
        ];
        let mut previous = 0;
        for marker in ordered {
            let position = second_column.find(marker).unwrap();
            assert!(position >= previous, "{marker} is out of column order");
            previous = position;
        }
        assert_eq!(second_column.matches("<div class=\"fx-group\"").count(), 5);
        assert!(!second_column.contains("id=\"mod-group\""));
        assert!(!second_column.contains("id=\"pad-group\""));
        assert!(!second_column.contains("id=\"midi-group\""));

        let third_column = &html[sources..io];
        let ordered = ["id=\"mod-group\"", "id=\"pad-group\"", "id=\"midi-group\""];
        let mut previous = 0;
        for marker in ordered {
            let position = third_column.find(marker).unwrap();
            assert!(position >= previous, "{marker} is out of column order");
            previous = position;
        }
        assert_eq!(third_column.matches("<div class=\"fx-group\"").count(), 3);
        assert!(!third_column.contains("id=\"audio-group\""));
        assert!(!third_column.contains("id=\"gyro-group\""));

        assert!(html.contains(
            "data-fx-column=\"mod-morph\"><!-- morph, time-domain effects, and audio -->"
        ));
        assert!(html
            .contains("data-fx-column=\"sources\"><!-- live modulation controls and sources -->"));
        for group in ["mod", "pad", "midi", "audio"] {
            assert_eq!(html.matches(&format!("id=\"{group}-group\"")).count(), 1);
        }

        let io_column = &html[io..];
        let remote = io_column.find("id=\"remote-group\"").unwrap();
        let gyro = io_column.find("id=\"gyro-group\"").unwrap();
        assert!(remote < gyro);
        assert_eq!(html.matches("id=\"gyro-group\"").count(), 1);
        assert_eq!(html.matches("id=\"temporal-group\"").count(), 1);
    }

    #[test]
    fn quick_patch_capture_uses_an_authoritative_snapshot_status() {
        let action: WebAction = serde_json::from_str(r#"{"action":"quick_save_patch"}"#).unwrap();
        assert!(matches!(action, WebAction::QuickSavePatch));
        let value = serde_json::to_value(WebAction::QuickSavePatch).unwrap();
        assert_eq!(value["action"], "quick_save_patch");

        let html = include_str!("../../static/index.html");
        let js = include_str!("../../static/app.js");
        assert!(html.contains("id=\"patch-capture\""));
        assert!(html.contains("id=\"patch-save-status\" role=\"status\" aria-live=\"polite\""));
        assert!(js.contains("sendAction({ action: 'quick_save_patch' })"));
        assert!(js.contains("syncPatchSave(msg.patch_save_status || '')"));
        assert!(js.contains("text.startsWith('Saving')"));
        assert!(!js.contains("setTimeout(() => patchCaptureButton"));
    }

    #[test]
    fn browser_scrubs_bootstrap_key_and_reports_every_layer_source_error() {
        let js = include_str!("../../static/app.js");
        let html = include_str!("../../static/index.html");
        assert!(js.contains("url.searchParams.delete('key')"));
        assert!(js.contains("window.history.replaceState"));
        assert!(js.contains("<div class=\"layer-source-status\" role=\"status\""));
        assert!(js.contains("video decoder: ${layer.source_error}"));
        assert!(js.contains("chevron?.setAttribute('aria-expanded'"));
        assert!(js.contains("aria-keyshortcuts=\"ArrowUp ArrowDown Home End\""));
        assert!(js.contains("item.setAttribute('role', 'button')"));
        assert!(js.contains("window.confirm"));
        assert!(html.contains("id=\"output-status\" role=\"status\" aria-live=\"polite\""));
        assert!(js.contains("syncOutputWindow(msg.output_window, msg.output_error)"));
        assert!(js.contains("sendAction({ action: 'set_output_window', enabled })"));
        assert!(js.contains("outputPendingOpen"));
        assert!(!js.contains("sendAction({ action: 'toggle_output_window' })"));
    }

    #[test]
    fn every_range_control_has_a_complete_editable_numeric_contract() {
        fn assert_range_tags_are_bounded(source: &str, allow_row_metadata: bool) -> usize {
            let marker = "<input type=\"range\"";
            let mut cursor = 0;
            let mut count = 0;
            while let Some(relative) = source[cursor..].find(marker) {
                let start = cursor + relative;
                let end = start
                    + source[start..]
                        .find('>')
                        .expect("range input must have a closing bracket");
                let tag = &source[start..=end];
                let context_start = if allow_row_metadata {
                    source[..start].rfind("<div").unwrap_or(start)
                } else {
                    start
                };
                let context = &source[context_start..=end];
                for attribute in ["min", "max", "step"] {
                    assert!(
                        tag.contains(&format!("{attribute}=\""))
                            || context.contains(&format!("data-{attribute}=\"")),
                        "range #{count} is missing {attribute}: {tag}"
                    );
                }
                count += 1;
                cursor = end + 1;
            }
            count
        }

        let html = include_str!("../../static/index.html");
        let js = include_str!("../../static/app.js");
        let css = include_str!("../../static/style.css");

        // Exact declaration counts keep every currently shipped static and
        // generated range under this universal contract.
        assert_eq!(assert_range_tags_are_bounded(html, true), 58);
        assert_eq!(assert_range_tags_are_bounded(js, false), 12);

        for contract in [
            "function normalizeRangeValue(slider, rawValue)",
            "Math.min(max, Math.max(min, value))",
            "Math.round((value - base) / step) * step",
            "binding.slider.dispatchEvent(new Event('input', { bubbles: true }))",
            "event.key === 'Enter'",
            "event.key === 'Escape'",
            "editor.addEventListener('blur'",
            "input,select,[data-range-editor]",
            "if (document.activeElement === el) return true;",
            "if (binding.editor.textContent !== textValue)",
            "binding.editor.getAttribute('aria-valuenow') !== ariaValue",
            "if (binding.disabled === disabled) return;",
            "const activeRangeBindings = new Set();",
            "if (!slider.isConnected)",
            "const peer = rangeControlPeers.get(el)",
            "syncRangeEditors(document)",
            "bindRangeEditors(card)",
            "bindRangeEditors(row)",
            "new MutationObserver",
            "editor.setAttribute('contenteditable'",
            "editor.setAttribute('role', 'spinbutton')",
            "editor.setAttribute('inputmode', min < 0 ? 'text' : 'decimal')",
            "event.key === 'ArrowUp'",
            "event.key === 'PageUp'",
            "editor.addEventListener('paste'",
            "event.clipboardData?.getData('text/plain')",
            "editor.setAttribute('aria-valuemin'",
            "editor.setAttribute('aria-valuemax'",
            "editor.setAttribute('aria-valuenow'",
            "editor.setAttribute('aria-invalid'",
        ] {
            assert!(js.contains(contract), "missing range contract: {contract}");
        }

        for coverage in [
            "data-param=\"pixelate\"",
            "data-param=\"cellular_speed\"",
            "data-ntsc=\"snow_intensity\"",
            "data-temporal=\"feedback\"",
            "id=\"audio-gain\"",
            "id=\"morph-t\"",
            "id=\"pad-x-curve-amount\"",
            "id=\"gyro-yaw-expo\"",
        ] {
            assert!(html.contains(coverage), "missing static range: {coverage}");
        }
        for coverage in [
            "const LAYER_EFFECT_CONTROLS",
            "data-layer-effect=\"${param}\"",
            "data-param=\"opacity\"",
            "class=\"routing-depth\"",
            "class=\"routing-curve-amount\"",
        ] {
            assert!(js.contains(coverage), "missing generated range: {coverage}");
        }
        for contract in [
            ".range-value[contenteditable=\"true\"]",
            ".range-value[contenteditable=\"true\"]:focus",
            ".range-value.range-value-invalid",
            ".range-editor-wrap",
            "touch-action: manipulation",
            "min-height: 24px",
        ] {
            assert!(css.contains(contract), "missing range styling: {contract}");
        }

        assert!(html.contains("id=\"btn-revert-master\""));
        assert!(html.contains("Revert master visual state"));
        assert!(js.contains(
            "title=\"Reset layer effects (opacity and transport unchanged)\" aria-label=\"Reset layer effects (opacity and transport unchanged)\""
        ));
    }

    #[test]
    fn windows_run_helper_restarts_only_the_exact_executable_with_bounded_waits() {
        let script = include_str!("../../scripts/build-windows.ps1");
        for contract in [
            "function Test-ExactExecutableProcess",
            "[System.StringComparer]::OrdinalIgnoreCase.Equals",
            "$process.CloseMainWindow()",
            "$process.Kill()",
            "[DateTime]::UtcNow.AddSeconds(5)",
            "[DateTime]::UtcNow.AddSeconds(10)",
            "$remainingExactCopies",
            "Close it manually and retry",
        ] {
            assert!(
                script.contains(contract),
                "missing restart contract: {contract}"
            );
        }
        assert!(!script.contains("Wait-Process"));
        assert!(!script.contains("Stop-Process -Id"));
        assert!(!script.contains("Get-Process -Id"));
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
    fn layer_master_fx_bypass_ui_is_stable_id_accessible_and_explicitly_scoped() {
        let js = include_str!("../../static/app.js");
        for contract in [
            "<label>Bypass Master FX</label>",
            "aria-label=\"Bypass Master FX for layer ${index + 1}\"",
            "aria-describedby=\"layer-master-bypass-help-${index}\"",
            "param: 'bypass_master_fx'",
            "...layerSelector(layer, index)",
            "bypassMasterFx.checked = !!layer.bypass_master_fx",
            "Skips Digital/Analog/Cellular/Motion/VHS master processing; own Layer FX/opacity/key/blend remain; Temporal still affects the final program.",
        ] {
            assert!(js.contains(contract), "missing bypass UI contract: {contract}");
        }
    }

    #[test]
    fn master_transport_uses_absolute_pending_state_and_priority_revert() {
        let js = include_str!("../../static/app.js");
        for contract in [
            "let transportAuthoritativePaused = false",
            "let transportPendingPaused = null",
            "let transportRequestSequence = 0",
            "if (transportPendingPaused !== null) return",
            "sendAction({ action: 'set_master_paused', paused: target })",
            "transportPendingPaused === transportAuthoritativePaused",
            "renderMasterTransport(target, true)",
            "btn.toggleAttribute('aria-busy', pending)",
            "window.setTimeout(() =>",
            "transportRequestSequence === requestSequence",
            "sendAction({ action: 'reset_fx' })",
        ] {
            assert!(
                js.contains(contract),
                "missing transport contract: {contract}"
            );
        }

        let mut queue = vec![WebAction::AddRouting; MAX_PENDING_ACTIONS];
        assert_ne!(
            enqueue_bounded(&mut queue, WebAction::ResetFx),
            EnqueueOutcome::Dropped
        );
        assert!(matches!(queue.last(), Some(WebAction::ResetFx)));
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
        assert!(legacy.route_id.is_empty());
        assert_eq!(legacy.value, 0.0);

        let routing = RoutingSnapshot {
            route_id: "42".into(),
            source: "pad_x".into(),
            target: "morph".into(),
            depth: -0.75,
            curve: "s_curve".into(),
            curve_amount: 0.4,
            attack: 0.12,
            release: 1.5,
            value: -0.25,
        };
        let value = serde_json::to_value(&routing).unwrap();
        assert_eq!(value["curve"], "s_curve");
        assert_json_number_near(&value["curve_amount"], 0.4);
        assert_json_number_near(&value["attack"], 0.12);
        assert_json_number_near(&value["release"], 1.5);
        assert_eq!(value["route_id"], "42");
        assert_json_number_near(&value["value"], -0.25);
    }

    #[test]
    fn routing_actions_accept_stable_ids_and_rescan_keeps_its_barrier_position() {
        let action: WebAction = serde_json::from_str(
            r#"{"action":"set_routing","index":7,"route_id":"42","param":"depth","value":0.5}"#,
        )
        .unwrap();
        assert!(matches!(
            action,
            WebAction::SetRouting {
                index: 7,
                route_id: Some(id),
                target_layer_id: None,
                layer_stack_revision: None,
                ..
            } if id == "42"
        ));

        let mut queue = Vec::new();
        assert_eq!(
            enqueue_bounded(&mut queue, WebAction::RescanLibrary),
            EnqueueOutcome::Added
        );
        let clip = WebAction::SetAudio {
            param: "clip".into(),
            value: serde_json::json!("second.wav"),
        };
        assert_eq!(enqueue_bounded(&mut queue, clip), EnqueueOutcome::Added);
        assert_eq!(
            enqueue_bounded(&mut queue, WebAction::RescanLibrary),
            EnqueueOutcome::Coalesced
        );
        assert!(matches!(queue.first(), Some(WebAction::RescanLibrary)));
        assert!(matches!(
            queue.get(1),
            Some(WebAction::SetAudio { param, .. }) if param == "clip"
        ));
    }

    #[test]
    fn routing_target_identity_is_backward_compatible_and_coalesces_latest_metadata() {
        let js = include_str!("../../static/app.js");
        for contract in [
            "latestLayerIdentities = layers.map((layer) => String(layer.layer_id || ''))",
            "const targetLayerId = latestLayerIdentities[Number(layerMatch[1]) - 1]",
            "targetIdentity = { target_layer_id: targetLayerId }",
            "targetIdentity.layer_stack_revision = layerStackRevision",
            "...selector(), ...targetIdentity, param: 'target', value: target",
        ] {
            assert!(
                js.contains(contract),
                "missing target identity contract: {contract}"
            );
        }

        let legacy: WebAction = serde_json::from_str(
            r#"{"action":"set_routing","index":2,"param":"target","value":"layer2_opacity"}"#,
        )
        .unwrap();
        assert!(matches!(
            legacy,
            WebAction::SetRouting {
                target_layer_id: None,
                layer_stack_revision: None,
                ..
            }
        ));

        let first: WebAction = serde_json::from_str(
            r#"{"action":"set_routing","index":2,"route_id":"42","target_layer_id":"20","layer_stack_revision":7,"param":"target","value":"layer2_opacity"}"#,
        )
        .unwrap();
        let latest: WebAction = serde_json::from_str(
            r#"{"action":"set_routing","index":0,"route_id":"42","target_layer_id":"20","layer_stack_revision":8,"param":"target","value":"layer1_opacity"}"#,
        )
        .unwrap();
        let mut queue = Vec::new();
        assert_eq!(enqueue_bounded(&mut queue, first), EnqueueOutcome::Added);
        assert_eq!(
            enqueue_bounded(&mut queue, latest),
            EnqueueOutcome::Coalesced
        );
        assert!(matches!(
            queue.as_slice(),
            [WebAction::SetRouting {
                route_id: Some(route_id),
                target_layer_id: Some(layer_id),
                layer_stack_revision: Some(8),
                value,
                ..
            }] if route_id == "42" && layer_id == "20" && value == "layer1_opacity"
        ));
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
    fn audio_protocol_round_trips_and_defaults_legacy_snapshots() {
        let legacy: AudioSnapshot = serde_json::from_str(
            r#"{"enabled":false,"gain":1.0,"level":0.0,"bass":0.1,"mid":0.2,"high":0.3,"onset":0.0}"#,
        )
        .unwrap();
        assert_eq!(legacy.band_count, 3);
        assert_eq!(legacy.band_ceiling_hz, 8000.0);
        assert!(legacy.bands.is_empty());
        assert_eq!(legacy.source_kind, "live");
        assert!(legacy.system_playback_devices.is_empty());
        assert!(legacy.clip_files.is_empty());
        assert!(legacy.clip_path.is_empty());
        assert!(!legacy.clip_loading);
        assert_eq!(legacy.clip_duration_secs, 0.0);

        let current = AudioSnapshot {
            band_count: 8,
            band_edges: vec![100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0],
            band_ceiling_hz: 12_800.0,
            bands: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            source_kind: "file".into(),
            system_playback_devices: vec!["Speakers".into()],
            clip_files: vec!["pulse-loop.wav".into()],
            clip_path: "pulse-loop.wav".into(),
            clip_loading: true,
            clip_duration_secs: 2.5,
            ..AudioSnapshot::default()
        };
        let value = serde_json::to_value(current).unwrap();
        assert_eq!(value["band_count"], 8);
        assert_eq!(value["band_edges"].as_array().unwrap().len(), 7);
        assert_eq!(value["bands"].as_array().unwrap().len(), 8);
        assert_json_number_near(&value["band_ceiling_hz"], 12_800.0);
        assert_eq!(value["source_kind"], "file");
        assert_eq!(value["system_playback_devices"][0], "Speakers");
        assert_eq!(value["clip_files"][0], "pulse-loop.wav");
        assert_eq!(value["clip_path"], "pulse-loop.wav");
        assert_eq!(value["clip_loading"], true);
        assert_json_number_near(&value["clip_duration_secs"], 2.5);

        for raw in [
            r#"{"action":"set_audio","param":"band_count","value":6}"#,
            r#"{"action":"set_audio","param":"band_edges","value":{"count":6,"edges":[120,480,1500,4000,9000],"ceiling":16000}}"#,
            r#"{"action":"set_audio","param":"source_kind","value":"file"}"#,
            r#"{"action":"set_audio","param":"clip","value":"pulse-loop.wav"}"#,
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
        assert!(html.contains("id=\"audio-source-kind\""));
        assert!(html.contains("id=\"audio-clip\""));
        assert!(html.contains("id=\"audio-import\""));
        assert!(html.contains(".wav,.mp3,.flac,.ogg,.opus,.m4a,.aac"));
        assert!(html.contains("Windows system playback"));
        assert!(js.contains("param: 'band_count'"));
        assert!(js.contains("param: 'source_kind'"));
        assert!(js.contains("param: 'clip'"));
        assert!(js.contains("MAX_AUDIO_IMPORT_BYTES = 512 * 1024 * 1024"));
        assert!(
            js.contains("slider.value = Number.isFinite(declaredDefault) ? declaredDefault : min")
        );
        assert!(js.contains("const fileMode = audioSourceKind.value === 'file'"));
        assert!(js.contains("audioClipRow.hidden = !fileMode"));
        assert!(js.contains("audioDevice.closest('.param-row').hidden = fileMode"));
        assert!(include_str!("../../static/style.css").contains(".param-row[hidden]"));
        assert!(js.contains("class=\"layer-fx-body\""));
        assert!(js.contains("class=\"layer-cellular-body\""));
        assert!(js.contains("role=\"region\" aria-label=\"Layer ${index + 1} effects\" hidden"));
        assert!(js.contains("system-playback:default"));
        assert!(js.contains("deterministic program-time analysis"));
        assert!(js.contains(".filter(({ layer }) => layer.source_kind === 'video')"));
        assert!(js.contains("const hasAnimatedPreview = /\\.(mp4|webm|mov|avi|mkv)$/i"));
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
        value.as_object_mut().unwrap().remove("patch_save_status");
        let restored: AppSnapshot = serde_json::from_value(value).unwrap();
        assert_eq!(restored.export_status, "");
        assert_eq!(restored.patch_save_status, "");
    }

    #[test]
    fn gyro_registry_tracks_explicit_legacy_timeout_and_disconnect_states() {
        let start = Instant::now();
        let mut registry = GyroStreamRegistry::default();
        assert_eq!(registry.status_at(start), GyroStatusSnapshot::default());

        registry.set_stream(7, true);
        let waiting = registry.status_at(start);
        assert!(!waiting.active);
        assert!(waiting.stale);
        assert_eq!(waiting.streamers, 1);
        assert_eq!(waiting.sample_age_ms, None);

        registry.note_sample_at(7, start);
        let live = registry.status_at(start + Duration::from_millis(250));
        assert!(live.active);
        assert!(!live.stale);
        assert_eq!(live.sample_age_ms, Some(250));

        let timed_out = registry.status_at(start + GYRO_SAMPLE_TIMEOUT + Duration::from_millis(1));
        assert!(!timed_out.active);
        assert!(timed_out.stale);

        registry.disconnect(7);
        let stopped = registry.status_at(start + Duration::from_secs(2));
        assert!(!stopped.active);
        assert!(stopped.stale);
        assert_eq!(stopped.streamers, 0);

        let mut legacy = GyroStreamRegistry::default();
        legacy.note_sample_at(42, start);
        assert!(legacy.status_at(start).active);
        legacy.disconnect(42);
        assert!(legacy.status_at(start).stale);

        let mut explicitly_stopped = GyroStreamRegistry::default();
        explicitly_stopped.set_stream(9, true);
        explicitly_stopped.set_stream(9, false);
        explicitly_stopped.note_sample_at(9, start);
        assert_eq!(explicitly_stopped.status_at(start).streamers, 0);
    }

    #[test]
    fn gyro_stream_protocol_and_telemetry_are_backward_compatible() {
        let action: WebAction =
            serde_json::from_str(r#"{"action":"gyro_stream","enabled":true}"#).unwrap();
        assert!(matches!(action, WebAction::GyroStream { enabled: true }));

        let mut legacy = serde_json::to_value(ModSnapshot::default()).unwrap();
        legacy.as_object_mut().unwrap().remove("gyro_status");
        let restored: ModSnapshot = serde_json::from_value(legacy).unwrap();
        assert_eq!(restored.gyro_status, GyroStatusSnapshot::default());

        let js = include_str!("../../static/app.js");
        assert!(js.contains("sendAction({ action: 'gyro_stream', enabled: true })"));
        assert!(js.contains("sendAction({ action: 'gyro_stream', enabled: false })"));
        assert!(js.contains("syncGyroStatus(m.gyro_status)"));
        assert!(js.contains("sensor data stale"));
        assert!(js.contains("output centered"));
    }
}
