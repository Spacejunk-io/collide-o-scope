#![allow(deprecated)] // egui 0.34 deprecation warnings for panel API renames
#![allow(dead_code)] // Old egui UI code kept as reference during web UI migration

mod audio;
mod effects;
mod input;
mod layers;
mod midi;
mod modulation;
mod ntsc;
mod patch;
mod render_export;
mod renderer;
mod spout_out;
mod video;
mod web;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use egui_wgpu::ScreenDescriptor;
use winit::application::ApplicationHandler;
use winit::event::{KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::ModifiersState;
use winit::window::{Fullscreen, Window, WindowAttributes, WindowId};

use input::{apply_action, map_key, ControlFlow};
use layers::{is_video_file, BlendMode, Layer};
use renderer::Renderer;
use web::state::WebState;

const TARGET_FPS: u64 = 30;
const FRAME_DURATION: Duration = Duration::from_millis(1000 / TARGET_FPS);

struct App {
    initial_video: Option<String>,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    layers: Vec<Layer>,
    selected_layer: Option<usize>,
    master_effects: effects::EffectUniforms,
    master_paused: bool,
    last_frame_time: Instant,
    start_time: Instant,
    modifiers: ModifiersState,
    // Library
    library_folder: Option<PathBuf>,
    library_files: Vec<PathBuf>,
    // YAML editor
    yaml_editor: patch::editor::EditorState,
    // egui state
    egui_ctx: egui::Context,
    egui_winit: Option<egui_winit::State>,
    egui_renderer: Option<egui_wgpu::Renderer>,
    video_egui_texture_id: Option<egui::TextureId>,
    // NTSC/VHS effects (params live here; processing runs on a worker thread)
    ntsc_params: ntsc::NtscParams,
    ntsc_worker: ntsc::NtscWorker,
    // Modulation matrix (BPM clock, LFOs, routings)
    mod_matrix: modulation::ModMatrix,
    // Audio input analysis (modulation source)
    audio: audio::AudioAnalyzer,
    // MIDI input (modulation source)
    midi: midi::MidiEngine,
    // Temporal effects (feedback trails, slit-scan)
    temporal_params: effects::params::TemporalParams,
    // Spout texture-sharing output
    spout: spout_out::SpoutOut,
    spout_enabled: bool,
    // Web control panel
    web_state: Arc<WebState>,
    // Offline render export
    export_job: Option<render_export::ExportJob>,
}

impl App {
    fn new(initial_video: Option<String>, library_folder: Option<PathBuf>, web_state: Arc<WebState>) -> Self {
        let library_files = library_folder
            .as_ref()
            .map(|f| scan_folder(f))
            .unwrap_or_default();

        // Generate thumbnails on background thread
        generate_thumbnails(&library_files, web_state.clone());

        Self {
            initial_video,
            window: None,
            renderer: None,
            layers: Vec::new(),
            selected_layer: None,
            master_effects: effects::EffectUniforms::default(),
            master_paused: false,
            last_frame_time: Instant::now(),
            start_time: Instant::now(),
            modifiers: ModifiersState::empty(),
            library_folder,
            library_files,
            yaml_editor: patch::editor::EditorState::default(),
            egui_ctx: egui::Context::default(),
            egui_winit: None,
            egui_renderer: None,
            video_egui_texture_id: None,
            ntsc_params: ntsc::NtscParams::default(),
            ntsc_worker: ntsc::NtscWorker::new(),
            mod_matrix: modulation::ModMatrix::new(),
            audio: audio::AudioAnalyzer::new(),
            midi: midi::MidiEngine::new(),
            temporal_params: effects::params::TemporalParams::default(),
            spout: spout_out::SpoutOut::new(),
            spout_enabled: false,
            web_state,
            export_job: None,
        }
    }

    fn add_layer(&mut self, path: &str) {
        let renderer = self.renderer.as_ref().unwrap();
        match Layer::new(path, &renderer.device) {
            Ok(layer) => {
                self.layers.push(layer);
                self.selected_layer = Some(self.layers.len() - 1);
            }
            Err(e) => {
                eprintln!("Failed to open video: {e}");
            }
        }
    }

    fn set_library_folder(&mut self, folder: PathBuf) {
        self.library_files = scan_folder(&folder);
        self.library_folder = Some(folder);
    }

    /// Handle an action from the web UI.
    fn handle_web_action(&mut self, action: web::state::WebAction) {
        use web::state::WebAction;
        match action {
            WebAction::SetParam { param, value } => {
                let mut snap = web::state::EffectsSnapshot::from_uniforms(&self.master_effects);
                snap.apply_param(&param, &value);
                snap.apply_to_uniforms(&mut self.master_effects);
            }
            WebAction::AddLayer { filename } => {
                // Find the full path from the library
                if let Some(path) = self.library_files.iter().find(|p| {
                    p.file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .as_deref() == Some(&filename)
                }) {
                    let path_str = path.to_string_lossy().to_string();
                    self.add_layer(&path_str);
                }
            }
            WebAction::RemoveLayer { index } => {
                if index < self.layers.len() {
                    self.layers.remove(index);
                    if self.layers.is_empty() {
                        self.selected_layer = None;
                    } else if let Some(sel) = self.selected_layer {
                        if sel >= self.layers.len() {
                            self.selected_layer = Some(self.layers.len() - 1);
                        }
                    }
                }
            }
            WebAction::ToggleVisibility { index } => {
                if index < self.layers.len() {
                    self.layers[index].visible = !self.layers[index].visible;
                    log::info!("Layer {index} visibility → {}", self.layers[index].visible);
                }
            }
            WebAction::ToggleLayerPause { index } => {
                if index < self.layers.len() {
                    self.layers[index].paused = !self.layers[index].paused;
                    log::info!("Layer {index} paused → {}", self.layers[index].paused);
                }
            }
            WebAction::ToggleMasterPause => {
                self.master_paused = !self.master_paused;
            }
            WebAction::ResetFx => {
                self.master_effects.reset();
            }
            WebAction::ResetGroup { group } => {
                let defaults = crate::effects::EffectUniforms::default();
                match group.as_str() {
                    "digital" => {
                        self.master_effects.pixelate_size = defaults.pixelate_size;
                        self.master_effects.rgb_split = defaults.rgb_split;
                        self.master_effects.hue_shift = defaults.hue_shift;
                        self.master_effects.saturation = defaults.saturation;
                        self.master_effects.brightness = defaults.brightness;
                        self.master_effects.contrast = defaults.contrast;
                        self.master_effects.posterize = defaults.posterize;
                        self.master_effects.invert = defaults.invert;
                    }
                    "analog" => {
                        self.master_effects.grain_intensity = defaults.grain_intensity;
                        self.master_effects.grain_size = defaults.grain_size;
                        self.master_effects.grain_algo = defaults.grain_algo;
                        self.master_effects.color_grain = defaults.color_grain;
                        self.master_effects.vignette = defaults.vignette;
                        self.master_effects.color_drift = defaults.color_drift;
                    }
                    "motion" => {
                        self.master_effects.breathe_scale = defaults.breathe_scale;
                        self.master_effects.breathe_rotation = defaults.breathe_rotation;
                        self.master_effects.breathe_position = defaults.breathe_position;
                    }
                    "vhs" => {
                        self.ntsc_params = ntsc::NtscParams::default();
                    }
                    "mod" => {
                        self.mod_matrix.reset();
                    }
                    "temporal" => {
                        self.temporal_params = effects::params::TemporalParams::default();
                    }
                    _ => {}
                }
            }
            WebAction::TapTempo => {
                self.mod_matrix.clock.tap(Instant::now());
            }
            WebAction::SetBpm { value } => {
                self.mod_matrix.clock.set_bpm(value);
            }
            WebAction::SetLfo { index, param, value } => {
                if let Some(lfo) = self.mod_matrix.lfos.get_mut(index) {
                    match param.as_str() {
                        "shape" => {
                            if let Some(s) = value.as_str() {
                                lfo.shape = modulation::LfoShape::from_str(s);
                            }
                        }
                        "beats" => {
                            if let Some(n) = value.as_f64() {
                                lfo.beats = (n as f32).clamp(0.0625, 64.0);
                            }
                        }
                        "phase" => {
                            if let Some(n) = value.as_f64() {
                                lfo.phase = (n as f32).rem_euclid(1.0);
                            }
                        }
                        _ => {}
                    }
                }
            }
            WebAction::AddRouting => {
                self.mod_matrix.add_routing();
            }
            WebAction::RemoveRouting { index } => {
                self.mod_matrix.remove_routing(index);
            }
            WebAction::SetAudio { param, value } => {
                match param.as_str() {
                    "enabled" => {
                        if let Some(b) = value.as_bool() {
                            self.mod_matrix.audio_enabled = b;
                        }
                    }
                    "gain" => {
                        if let Some(n) = value.as_f64() {
                            self.mod_matrix.audio_gain = (n as f32).clamp(0.0, 8.0);
                        }
                    }
                    _ => {}
                }
            }
            WebAction::SetMidi { param, value } => {
                match param.as_str() {
                    "enabled" => {
                        if let Some(b) = value.as_bool() {
                            self.mod_matrix.midi_enabled = b;
                            if !b {
                                self.mod_matrix.midi_learn = None;
                            }
                        }
                    }
                    "learn" => {
                        // A slot index arms learn; null (or repeat) disarms.
                        match value.as_u64() {
                            Some(slot) if (slot as usize) < modulation::NUM_MIDI_SLOTS
                                && self.mod_matrix.midi_learn != Some(slot as usize) =>
                            {
                                self.mod_matrix.midi_learn = Some(slot as usize);
                                // Discard stale "last CC" so learn binds a fresh twist.
                                let _ = self.midi.take_last_cc();
                            }
                            _ => self.mod_matrix.midi_learn = None,
                        }
                    }
                    "cc0" | "cc1" | "cc2" | "cc3" => {
                        let slot = param.as_bytes()[2] as usize - b'0' as usize;
                        if let Some(n) = value.as_u64() {
                            self.mod_matrix.midi_ccs[slot] = (n as u8) & 0x7F;
                        }
                    }
                    "clock_sync" => {
                        if let Some(b) = value.as_bool() {
                            self.mod_matrix.midi_clock_sync = b;
                        }
                    }
                    _ => {}
                }
            }
            WebAction::Gyro { alpha, beta, gamma } => {
                // DeviceOrientation degrees → unipolar 0..1 (0.5 = level).
                self.mod_matrix.gyro = [
                    (alpha / 360.0).rem_euclid(1.0),
                    ((beta / 90.0).clamp(-1.0, 1.0) + 1.0) * 0.5,
                    ((gamma / 90.0).clamp(-1.0, 1.0) + 1.0) * 0.5,
                ];
            }
            WebAction::Pad { x, y } => {
                self.mod_matrix.pad = [x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)];
            }
            WebAction::SetTemporal { param, value } => {
                let p = &mut self.temporal_params;
                match param.as_str() {
                    "feedback" => {
                        if let Some(n) = value.as_f64() {
                            p.feedback = (n as f32).clamp(0.0, 0.95);
                        }
                    }
                    "fb_zoom" => {
                        if let Some(n) = value.as_f64() {
                            p.fb_zoom = (n as f32).clamp(0.9, 1.1);
                        }
                    }
                    "fb_rotate" => {
                        if let Some(n) = value.as_f64() {
                            p.fb_rotate = (n as f32).clamp(-5.0, 5.0);
                        }
                    }
                    "slitscan" => {
                        if let Some(n) = value.as_f64() {
                            p.slitscan = (n as f32).clamp(0.0, 1.0);
                        }
                    }
                    "slit_axis" => {
                        if let Some(n) = value.as_u64() {
                            p.slit_axis = (n.min(1)) as f32;
                        }
                    }
                    _ => {}
                }
            }
            WebAction::SetSpout { enabled } => {
                self.spout_enabled = enabled;
            }
            WebAction::SetRouting { index, param, value } => {
                if let Some(routing) = self.mod_matrix.routings.get_mut(index) {
                    match param.as_str() {
                        "source" => {
                            if let Some(s) = value.as_str() {
                                routing.source = modulation::ModSource::from_str(s);
                            }
                        }
                        "target" => {
                            if let Some(s) = value.as_str() {
                                if modulation::TARGETS.iter().any(|(k, _, _)| *k == s) {
                                    routing.target = s.to_string();
                                }
                            }
                        }
                        "depth" => {
                            if let Some(n) = value.as_f64() {
                                routing.depth = (n as f32).clamp(-1.0, 1.0);
                            }
                        }
                        _ => {}
                    }
                }
            }
            WebAction::SetLayerParam { index, param, value } => {
                if index < self.layers.len() {
                    let layer = &mut self.layers[index];
                    match param.as_str() {
                        "opacity" => {
                            if let Some(v) = value.as_f64() {
                                layer.opacity = (v as f32).clamp(0.0, 1.0);
                            }
                        }
                        "speed" => {
                            if let Some(v) = value.as_f64() {
                                layer.speed = (v as f32).clamp(0.25, 4.0);
                            }
                        }
                        "blend_mode" => {
                            if let Some(s) = value.as_str() {
                                layer.blend_mode = match s {
                                    "screen" => crate::layers::BlendMode::Screen,
                                    "multiply" => crate::layers::BlendMode::Multiply,
                                    "difference" => crate::layers::BlendMode::Difference,
                                    _ => crate::layers::BlendMode::Normal,
                                };
                            }
                        }
                        "key_mode" => {
                            if let Some(v) = value.as_u64() {
                                layer.effects.key_mode = (v.min(2)) as f32;
                            }
                        }
                        "key_threshold" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.key_threshold = (v as f32).clamp(0.0, 1.0);
                            }
                        }
                        "key_softness" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.key_softness = (v as f32).clamp(0.0, 0.5);
                            }
                        }
                        _ => {}
                    }
                }
            }
            WebAction::SetNtscParam { param, value } => {
                self.ntsc_params.set_param(&param, &value);
            }
            WebAction::StartExport { width, height, fps, duration_secs } => {
                if self.export_job.is_none() || self.export_job.as_ref().unwrap().is_done() {
                    let patch = patch::PatchState::capture(
                        &self.master_effects,
                        &self.layers,
                        &self.ntsc_params,
                        &self.mod_matrix,
                        &self.temporal_params,
                    );
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    let timestamp = now;
                    let output_dir = self.library_folder.as_ref()
                        .map(|f| f.parent().unwrap_or(f).join("renders"))
                        .unwrap_or_else(|| std::path::PathBuf::from("renders"));
                    let output_path = format!(
                        "{}/patch_{}_{width}x{height}.mp4",
                        output_dir.display(),
                        timestamp
                    );
                    let lib_folder = self.library_folder.as_ref()
                        .map(|f| f.to_string_lossy().to_string())
                        .unwrap_or_else(|| ".".to_string());
                    let config = render_export::ExportConfig {
                        width,
                        height,
                        fps,
                        duration_secs,
                        output_path,
                    };
                    self.export_job = Some(render_export::ExportJob::start(patch, config, &lib_folder));
                    log::info!("Export started");
                }
            }
            WebAction::CancelExport => {
                if let Some(ref job) = self.export_job {
                    job.cancel();
                }
            }
        }
    }

    /// Push full app state to the web UI via broadcast.
    fn push_web_state(&self) {
        use web::state::{
            AppSnapshot, AudioSnapshot, EffectsSnapshot, LayerSnapshot, MidiSlotSnapshot,
            MidiSnapshot, ModSnapshot, NtscSnapshot, SpoutSnapshot, TemporalSnapshot,
        };

        let snapshot = AppSnapshot {
            msg_type: "state".to_string(),
            effects: EffectsSnapshot::from_uniforms(&self.master_effects),
            ntsc: NtscSnapshot::from_params(&self.ntsc_params),
            layers: self.layers.iter().map(|l| LayerSnapshot {
                filename: l.filename.clone(),
                visible: l.visible,
                paused: l.paused,
                opacity: l.opacity,
                speed: l.speed,
                blend_mode: l.blend_mode.label().to_string(),
                progress: l.decoder.progress(),
                key_mode: l.effects.key_mode as u32,
                key_threshold: l.effects.key_threshold,
                key_softness: l.effects.key_softness,
            }).collect(),
            library: self.library_files.iter().filter_map(|p| {
                p.file_name().map(|n| n.to_string_lossy().to_string())
            }).collect(),
            paused: self.master_paused,
            modulation: ModSnapshot::from_matrix(&self.mod_matrix),
            audio: AudioSnapshot {
                enabled: self.mod_matrix.audio_enabled,
                gain: self.mod_matrix.audio_gain,
                level: self.mod_matrix.audio.level,
                bass: self.mod_matrix.audio.bass,
                mid: self.mod_matrix.audio.mid,
                high: self.mod_matrix.audio.high,
                onset: self.mod_matrix.audio.onset,
                bright: self.mod_matrix.audio.bright,
                noise: self.mod_matrix.audio.noise,
                device: self.audio.device_name.clone(),
                error: self.audio.error.clone(),
            },
            midi: MidiSnapshot {
                enabled: self.mod_matrix.midi_enabled,
                slots: (0..modulation::NUM_MIDI_SLOTS)
                    .map(|i| MidiSlotSnapshot {
                        cc: self.mod_matrix.midi_ccs[i],
                        value: self.mod_matrix.midi[i],
                    })
                    .collect(),
                learning: self.mod_matrix.midi_learn,
                clock_sync: self.mod_matrix.midi_clock_sync,
                clock_active: self.mod_matrix.clock.is_external(),
                port: self.midi.port_name.clone(),
                error: self.midi.error.clone(),
            },
            temporal: TemporalSnapshot::from_params(&self.temporal_params),
            spout: {
                let status = self.spout.status();
                SpoutSnapshot {
                    enabled: self.spout_enabled,
                    active: status.active,
                    error: status.error,
                }
            },
            remote_url: self.web_state.lan_url.read().map(|s| s.clone()).unwrap_or_default(),
            export_progress: self.export_job.as_ref()
                .map(|j| if j.is_done() { 1.0 } else { j.progress.progress_f32() })
                .unwrap_or(0.0),
            export_error: self.export_job.as_ref()
                .and_then(|j| {
                    if j.is_done() {
                        let err = j.progress.error.lock().unwrap();
                        if err.is_empty() { None } else { Some(err.clone()) }
                    } else {
                        None
                    }
                })
                .unwrap_or_default(),
        };

        // Non-blocking: try to write + broadcast
        if let Ok(mut app) = self.web_state.app.try_write() {
            *app = snapshot.clone();
        }
        let _ = self.web_state.tx.send(serde_json::to_string(&snapshot).unwrap_or_default());
    }
}

/// Scan a directory for video files, returning sorted list of paths.


fn scan_folder(folder: &PathBuf) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(folder) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && is_video_file(p))
        .collect();
    files.sort();
    files
}

/// Generate thumbnails and preview frames for all library files using ffmpeg CLI.
/// Thumbnails are generated first (fast), then preview frames in a second pass.
fn generate_thumbnails(files: &[PathBuf], web_state: Arc<web::state::WebState>) {
    let paths: Vec<PathBuf> = files.to_vec();
    std::thread::Builder::new()
        .name("thumb-gen".into())
        .spawn(move || {
            use std::process::Command;
            use std::sync::atomic::{AtomicUsize, Ordering};

            let count = Arc::new(AtomicUsize::new(0));
            let total = paths.len();

            // Pass 1: Generate static thumbnails (fast, parallel batches of 8)
            for chunk in paths.chunks(8) {
                let handles: Vec<_> = chunk.iter().map(|path| {
                    let path = path.clone();
                    let web_state = web_state.clone();
                    let count = count.clone();
                    std::thread::spawn(move || {
                        let filename = match path.file_name() {
                            Some(n) => n.to_string_lossy().to_string(),
                            None => return,
                        };

                        let output = Command::new("ffmpeg")
                            .args([
                                "-i", &path.to_string_lossy(),
                                "-vframes", "1",
                                "-vf", "scale=180:-1",
                                "-f", "image2pipe",
                                "-vcodec", "mjpeg",
                                "-q:v", "8",
                                "-loglevel", "error",
                                "pipe:1",
                            ])
                            .output();

                        match output {
                            Ok(result) if result.status.success() && !result.stdout.is_empty() => {
                                if let Ok(mut cache) = web_state.thumbnails.write() {
                                    cache.insert(filename, result.stdout);
                                }
                                count.fetch_add(1, Ordering::Relaxed);
                            }
                            Ok(result) => {
                                let err = String::from_utf8_lossy(&result.stderr);
                                log::warn!("Thumb: ffmpeg failed for {filename}: {err}");
                            }
                            Err(e) => {
                                log::warn!("Thumb: can't run ffmpeg for {filename}: {e}");
                            }
                        }
                    })
                }).collect();

                for h in handles {
                    let _ = h.join();
                }
            }

            log::info!("Generated {}/{total} thumbnails", count.load(Ordering::Relaxed));

            // Pass 2: Generate preview frames (~8 per video, parallel batches of 4)
            let preview_count = Arc::new(AtomicUsize::new(0));
            for chunk in paths.chunks(4) {
                let handles: Vec<_> = chunk.iter().map(|path| {
                    let path = path.clone();
                    let web_state = web_state.clone();
                    let preview_count = preview_count.clone();
                    std::thread::spawn(move || {
                        let filename = match path.file_name() {
                            Some(n) => n.to_string_lossy().to_string(),
                            None => return,
                        };

                        // Get video duration with ffprobe
                        let duration = Command::new("ffprobe")
                            .args([
                                "-v", "error",
                                "-show_entries", "format=duration",
                                "-of", "csv=p=0",
                                &path.to_string_lossy(),
                            ])
                            .output()
                            .ok()
                            .and_then(|o| String::from_utf8(o.stdout).ok())
                            .and_then(|s| s.trim().parse::<f64>().ok())
                            .unwrap_or(0.0);

                        if duration < 0.5 {
                            return;
                        }

                        const NUM_FRAMES: usize = 8;
                        let mut frames = Vec::with_capacity(NUM_FRAMES);

                        for i in 0..NUM_FRAMES {
                            let seek = duration * (i as f64) / (NUM_FRAMES as f64);
                            let seek_str = format!("{:.2}", seek);

                            let output = Command::new("ffmpeg")
                                .args([
                                    "-ss", &seek_str,
                                    "-i", &path.to_string_lossy(),
                                    "-vframes", "1",
                                    "-vf", "scale=180:-1",
                                    "-f", "image2pipe",
                                    "-vcodec", "mjpeg",
                                    "-q:v", "10",
                                    "-loglevel", "error",
                                    "pipe:1",
                                ])
                                .output();

                            if let Ok(result) = output {
                                if result.status.success() && !result.stdout.is_empty() {
                                    frames.push(result.stdout);
                                }
                            }
                        }

                        if !frames.is_empty() {
                            if let Ok(mut cache) = web_state.preview_frames.write() {
                                cache.insert(filename, frames);
                            }
                            preview_count.fetch_add(1, Ordering::Relaxed);
                        }
                    })
                }).collect();

                for h in handles {
                    let _ = h.join();
                }
            }

            log::info!("Generated {}/{total} preview strips", preview_count.load(Ordering::Relaxed));
        })
        .ok();
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let mut output_width = 1280u32;
        let mut output_height = 720u32;

        if let Some(ref path) = self.initial_video {
            if let Ok(decoder) = video::VideoDecoder::open(path) {
                output_width = decoder.width;
                output_height = decoder.height;
            }
        }

        log::info!("Output: {}x{}", output_width, output_height);

        let window_w = output_width;
        let window_h = output_height;

        let window_attrs = WindowAttributes::default()
            .with_title("collide-o-scope")
            .with_inner_size(winit::dpi::LogicalSize::new(window_w, window_h));

        let window = Arc::new(event_loop.create_window(window_attrs).unwrap());

        let renderer = Renderer::new(window.clone(), output_width, output_height);

        configure_fonts(&self.egui_ctx);

        let egui_winit = egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        let mut egui_renderer = egui_wgpu::Renderer::new(
            &renderer.device,
            renderer.config.format,
            egui_wgpu::RendererOptions::default(),
        );

        let video_egui_texture_id = egui_renderer.register_native_texture(
            &renderer.device,
            &renderer.output_view,
            wgpu::FilterMode::Linear,
        );

        self.window = Some(window);
        self.renderer = Some(renderer);
        self.egui_winit = Some(egui_winit);
        self.egui_renderer = Some(egui_renderer);
        self.video_egui_texture_id = Some(video_egui_texture_id);

        if let Some(path) = self.initial_video.take() {
            self.add_layer(&path);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(egui_winit) = &mut self.egui_winit {
            let response =
                egui_winit.on_window_event(self.window.as_ref().unwrap(), &event);
            if response.consumed {
                return;
            }
        }

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(new_size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(new_size.width, new_size.height);
                }
            }

            WindowEvent::ScaleFactorChanged { .. } => {
                let size = self.window.as_ref().unwrap().inner_size();
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(size.width, size.height);
                }
            }

            WindowEvent::ModifiersChanged(mods) => {
                self.modifiers = mods.state();
            }

            WindowEvent::DroppedFile(path) => {
                if path.is_dir() {
                    self.set_library_folder(path);
                } else if is_video_file(&path) {
                    if let Some(path_str) = path.to_str() {
                        let path_owned = path_str.to_string();
                        self.add_layer(&path_owned);
                    }
                }
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key,
                        state,
                        ..
                    },
                ..
            } => {
                use winit::keyboard::{KeyCode, PhysicalKey};

                // Ctrl+key shortcuts (editor toggle, save, load)
                if state == winit::event::ElementState::Pressed
                    && self.modifiers.control_key()
                {
                    match physical_key {
                        PhysicalKey::Code(KeyCode::KeyE) => {
                            self.yaml_editor.active = !self.yaml_editor.active;
                            return;
                        }
                        PhysicalKey::Code(KeyCode::KeyS) => {
                            patch::editor::save_patch(
                                &self.master_effects,
                                &self.layers,
                                &self.ntsc_params,
                                &self.mod_matrix,
                                &self.temporal_params,
                            );
                            return;
                        }
                        PhysicalKey::Code(KeyCode::KeyO) => {
                            patch::editor::load_patch(
                                &mut self.master_effects,
                                &mut self.layers,
                                &mut self.ntsc_params,
                                &mut self.mod_matrix,
                                &mut self.temporal_params,
                            );
                            return;
                        }
                        _ => {}
                    }
                }

                let shift = self.modifiers.shift_key();
                let action = map_key(physical_key, state, shift);

                if let Some(idx) = self.selected_layer {
                    if let Some(layer) = self.layers.get_mut(idx) {
                        match apply_action(action, &mut layer.effects) {
                            ControlFlow::Quit => event_loop.exit(),
                            ControlFlow::TogglePause => layer.paused = !layer.paused,
                            ControlFlow::ToggleFullscreen => {
                                if let Some(window) = &self.window {
                                    let current = window.fullscreen();
                                    if current.is_some() {
                                        window.set_fullscreen(None);
                                    } else {
                                        window.set_fullscreen(Some(Fullscreen::Borderless(None)));
                                    }
                                }
                            }
                            ControlFlow::Continue => {}
                        }
                    }
                } else {
                    let mut dummy = effects::EffectUniforms::default();
                    match apply_action(action, &mut dummy) {
                        ControlFlow::Quit => event_loop.exit(),
                        ControlFlow::ToggleFullscreen => {
                            if let Some(window) = &self.window {
                                let current = window.fullscreen();
                                if current.is_some() {
                                    window.set_fullscreen(None);
                                } else {
                                    window.set_fullscreen(Some(Fullscreen::Borderless(None)));
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                let win_size = self.window.as_ref().unwrap().inner_size();
                if win_size.width == 0 || win_size.height == 0 {
                    if let Some(w) = &self.window {
                        w.request_redraw();
                    }
                    return;
                }

                let now = Instant::now();
                if now - self.last_frame_time >= FRAME_DURATION {
                    self.last_frame_time = now;

                    // Consume decoded frames for each layer (per-layer timing, speed, pause).
                    // Decoding happens on background threads; this is a non-blocking recv,
                    // so a slow or looping decoder can never stall the render loop.
                    // Pacing uses the modulated speed (last frame's matrix sample —
                    // one frame stale, imperceptible).
                    if !self.master_paused {
                        let matrix = &self.mod_matrix;
                        for (i, layer) in self.layers.iter_mut().enumerate() {
                            let speed = matrix
                                .modulate_layer(i, layer.opacity, layer.speed, layer.effects.key_threshold)
                                .speed;
                            if !layer.paused && layer.ready_for_frame(speed) {
                                if let Some(frame_data) = layer.decoder.try_next_frame() {
                                    let renderer = self.renderer.as_ref().unwrap();
                                    layer.upload_frame(&renderer.queue, &frame_data);
                                    layer.last_decode = Instant::now();
                                }
                            }
                        }
                    }

                    // Process actions from web UI
                    let pending_actions: Vec<_> = self.web_state.actions
                        .try_lock()
                        .map(|mut a| a.drain(..).collect())
                        .unwrap_or_default();
                    for action in pending_actions {
                        self.handle_web_action(action);
                    }

                    // Build minimal egui frame (video display only, no UI panels)
                    let window = self.window.as_ref().unwrap();
                    let egui_winit = self.egui_winit.as_mut().unwrap();
                    let raw_input = egui_winit.take_egui_input(window);

                    let video_egui_texture_id = self.video_egui_texture_id;
                    let output_width = self.renderer.as_ref().unwrap().output_width;
                    let output_height = self.renderer.as_ref().unwrap().output_height;

                    let full_output = self.egui_ctx.run_ui(raw_input, |ctx| {
                        // Full-window video output (no UI panels)
                        egui::CentralPanel::default()
                            .frame(egui::Frame::NONE.fill(egui::Color32::BLACK))
                            .show(ctx, |ui| {
                                if let Some(tex_id) = video_egui_texture_id {
                                    let available = ui.available_size();
                                    let aspect = output_width as f32 / output_height as f32;
                                    let (w, h) = fit_to_area(available.x, available.y, aspect);
                                    ui.centered_and_justified(|ui| {
                                        ui.image(egui::load::SizedTexture::new(
                                            tex_id,
                                            egui::vec2(w, h),
                                        ));
                                    });
                                }
                            });
                    });

                    // Push full state to web UI
                    self.push_web_state();

                    let window = self.window.as_ref().unwrap();
                    let egui_winit = self.egui_winit.as_mut().unwrap();
                    egui_winit.handle_platform_output(window, full_output.platform_output);

                    let tris = self
                        .egui_ctx
                        .tessellate(full_output.shapes, full_output.pixels_per_point);

                    // Set time uniform on all effects (drives animated noise/breathing)
                    let elapsed = self.start_time.elapsed().as_secs_f32();
                    for layer in &mut self.layers {
                        layer.effects.time = elapsed;
                    }
                    self.master_effects.time = elapsed;

                    // Sync audio capture with the requested state, then feed
                    // the latest analysis into the matrix as a source.
                    if self.mod_matrix.audio_enabled && !self.audio.is_running() {
                        self.audio.start();
                        if !self.audio.error.is_empty() {
                            // Device failed — flip the switch back off so the
                            // UI reflects reality instead of retrying forever.
                            self.mod_matrix.audio_enabled = false;
                        }
                    } else if !self.mod_matrix.audio_enabled && self.audio.is_running() {
                        self.audio.stop();
                    }
                    self.mod_matrix.audio = self.audio.analyze(self.mod_matrix.audio_gain);

                    // Same for MIDI: sync connection state, run learn, read slots.
                    if self.mod_matrix.midi_enabled && !self.midi.is_running() {
                        self.midi.start();
                        if !self.midi.error.is_empty() {
                            self.mod_matrix.midi_enabled = false;
                        }
                    } else if !self.mod_matrix.midi_enabled && self.midi.is_running() {
                        self.midi.stop();
                    }
                    if let Some(slot) = self.mod_matrix.midi_learn {
                        if let Some(cc) = self.midi.take_last_cc() {
                            self.mod_matrix.midi_ccs[slot] = cc;
                            self.mod_matrix.midi_learn = None;
                            log::info!("MIDI learn: slot {slot} → CC{cc}");
                        }
                    }
                    for i in 0..modulation::NUM_MIDI_SLOTS {
                        self.mod_matrix.midi[i] =
                            self.midi.cc_value(self.mod_matrix.midi_ccs[i]);
                    }

                    // MIDI clock sync: while pulses arrive, the external clock
                    // owns BPM and beat position; when they stop, the internal
                    // clock resumes from the same position.
                    if self.mod_matrix.midi_clock_sync && self.midi.is_running() {
                        match self.midi.clock_state() {
                            Some((bpm, beat)) => {
                                self.mod_matrix.clock.set_bpm(bpm);
                                self.mod_matrix.clock.set_external_beat(Some(beat), Instant::now());
                            }
                            None => {
                                self.mod_matrix.clock.set_external_beat(None, Instant::now());
                            }
                        }
                    } else if self.mod_matrix.clock.is_external() {
                        self.mod_matrix.clock.set_external_beat(None, Instant::now());
                    }

                    // Sample the modulation matrix and derive modulated copies.
                    // Base values (what the UI edits) stay untouched; LFOs
                    // breathe around them.
                    self.mod_matrix.update(Instant::now());
                    let (mod_master, mod_ntsc, mod_temporal) = self.mod_matrix.modulate(
                        &self.master_effects,
                        &self.ntsc_params,
                        &self.temporal_params,
                    );

                    let renderer = self.renderer.as_mut().unwrap();
                    let mut encoder = renderer.device.create_command_encoder(
                        &wgpu::CommandEncoderDescriptor {
                            label: Some("Frame Encoder"),
                        },
                    );
                    // Per-layer modulated copies: opacity crossfades, key
                    // thresholds breathing shapes in and out — bases untouched.
                    let layer_mods: Vec<(effects::EffectUniforms, f32)> = self
                        .layers
                        .iter()
                        .enumerate()
                        .map(|(i, layer)| {
                            let lm = self.mod_matrix.modulate_layer(
                                i,
                                layer.opacity,
                                layer.speed,
                                layer.effects.key_threshold,
                            );
                            let mut fx = layer.effects;
                            fx.key_threshold = lm.key_threshold;
                            (fx, lm.opacity)
                        })
                        .collect();

                    renderer.render_layers(&mut encoder, &self.layers, &layer_mods);
                    renderer.render_master_effects(&mut encoder, &mod_master);
                    // Temporal effects (feedback/slit-scan) + history recording.
                    renderer.render_temporal(&mut encoder, &mod_temporal);

                    // Sync the Spout output worker with its toggle.
                    if self.spout_enabled && !self.spout.is_running() {
                        self.spout.start();
                    } else if !self.spout_enabled && self.spout.is_running() {
                        self.spout.stop();
                    }

                    // NTSC/VHS post-process — fully asynchronous. The render
                    // thread never waits on the GPU readback or the CPU effect;
                    // the displayed NTSC image trails the live composite by
                    // ~2 frames, which is invisible for a VHS look. Spout
                    // shares the same readback pipeline: it receives either
                    // the raw composite or, when NTSC is on, the processed
                    // frames — always what the audience sees.
                    let ntsc_enabled = mod_ntsc.enabled;
                    let spout_active = self.spout_enabled && self.spout.is_running();
                    let need_readback = ntsc_enabled || spout_active;
                    if need_readback {
                        // Kick off a copy of the fresh composite into a staging
                        // buffer, submit the render work, then request the map.
                        let slot = renderer.begin_readback(&mut encoder);
                        renderer.queue.submit(std::iter::once(encoder.finish()));
                        if let Some(idx) = slot {
                            renderer.map_readback(idx);
                        }

                        // Fresh encoder for the egui pass.
                        encoder = renderer.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("Post-NTSC Encoder"),
                            },
                        );
                    }

                    // Harvest any completed readback and route it.
                    if let Some(pixels) = renderer.poll_readback() {
                        if ntsc_enabled {
                            self.ntsc_worker.try_submit(
                                pixels,
                                renderer.output_width,
                                renderer.output_height,
                                mod_ntsc.clone(),
                            );
                        } else if spout_active {
                            self.spout.try_submit(
                                pixels,
                                renderer.output_width,
                                renderer.output_height,
                            );
                        }
                        // Otherwise a stale readback from a just-disabled
                        // feature; dropped.
                    }

                    // Any processed frame ready? Overwrite the composite with it.
                    // The queued texture write executes before the egui submit
                    // below, so the egui pass displays the processed frame.
                    if let Some(processed) = self.ntsc_worker.try_recv() {
                        if ntsc_enabled {
                            renderer.write_composite(&processed);
                            if spout_active {
                                self.spout.try_submit(
                                    processed,
                                    renderer.output_width,
                                    renderer.output_height,
                                );
                            }
                        }
                    }

                    let egui_renderer = self.egui_renderer.as_mut().unwrap();
                    for (id, image_delta) in &full_output.textures_delta.set {
                        egui_renderer.update_texture(
                            &renderer.device,
                            &renderer.queue,
                            *id,
                            image_delta,
                        );
                    }

                    let screen_desc = ScreenDescriptor {
                        size_in_pixels: [renderer.config.width, renderer.config.height],
                        pixels_per_point: full_output.pixels_per_point,
                    };

                    egui_renderer.update_buffers(
                        &renderer.device,
                        &renderer.queue,
                        &mut encoder,
                        &tris,
                        &screen_desc,
                    );

                    let surface_texture = match renderer.surface.get_current_texture() {
                        wgpu::CurrentSurfaceTexture::Success(t)
                        | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
                        wgpu::CurrentSurfaceTexture::Outdated
                        | wgpu::CurrentSurfaceTexture::Lost => {
                            let size = window.inner_size();
                            let r = self.renderer.as_mut().unwrap();
                            if size.width > 0 && size.height > 0 {
                                r.resize(size.width, size.height);
                            } else {
                                r.reconfigure_surface();
                            }
                            if let Some(w) = &self.window {
                                w.request_redraw();
                            }
                            return;
                        }
                        _ => {
                            if let Some(w) = &self.window {
                                w.request_redraw();
                            }
                            return;
                        }
                    };
                    let surface_view = surface_texture
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    {
                        let mut render_pass = encoder
                            .begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("egui Pass"),
                                color_attachments: &[Some(
                                    wgpu::RenderPassColorAttachment {
                                        view: &surface_view,
                                        resolve_target: None,
                                        ops: wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                                r: 0.1,
                                                g: 0.1,
                                                b: 0.1,
                                                a: 1.0,
                                            }),
                                            store: wgpu::StoreOp::Store,
                                        },
                                        depth_slice: None,
                                    },
                                )],
                                depth_stencil_attachment: None,
                                ..Default::default()
                            })
                            .forget_lifetime();

                        egui_renderer.render(&mut render_pass, &tris, &screen_desc);
                    }

                    for id in &full_output.textures_delta.free {
                        egui_renderer.free_texture(id);
                    }

                    renderer.queue.submit(std::iter::once(encoder.finish()));
                    surface_texture.present();
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            _ => {}
        }
    }
}

/// Returns an optional path to add as a new layer (needs device access from caller).
fn build_ui(
    ctx: &egui::Context,
    layers: &mut Vec<Layer>,
    selected_layer: &mut Option<usize>,
    master_effects: &mut effects::EffectUniforms,
    master_paused: &mut bool,
    yaml_editor: &mut patch::editor::EditorState,
    library_folder: &mut Option<PathBuf>,
    library_files: &mut Vec<PathBuf>,
    video_egui_texture_id: Option<egui::TextureId>,
    output_width: u32,
    output_height: u32,
) -> Option<String> {
    let mut add_layer_path: Option<String> = None;
    let mut remove_layer: Option<usize> = None;
    let mut move_layer: Option<(usize, usize)> = None;
    let mut change_folder = false;

    // LEFT panel: Layers with collapsible per-layer controls
    egui::Panel::left("left_panel")
        .min_size(240.0)
        .default_size(280.0)
        .show(ctx, |ui| {
            // View switcher tabs
            ui.horizontal(|ui| {
                if ui.selectable_label(!yaml_editor.active, "UI").clicked() {
                    yaml_editor.active = false;
                }
                if ui.selectable_label(yaml_editor.active, "Code").clicked() {
                    yaml_editor.active = true;
                }
            });
            ui.separator();

            if yaml_editor.active {
                // Code view
                patch::editor::build_yaml_editor_content(
                    ui,
                    layers,
                    master_effects,
                    yaml_editor,
                );
            } else {
                // UI view: collapsible layers
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        let layer_count = layers.len();

                        for i in 0..layer_count {
                            let is_selected = *selected_layer == Some(i);

                            // Layer header row with controls
                            ui.horizontal(|ui| {
                                // Grip handle for reorder
                                let grip = ui.label(egui::RichText::new("⠿").weak());
                                if grip.dragged() {
                                    let delta = grip.drag_delta().y;
                                    if delta < -16.0 && i > 0 {
                                        move_layer = Some((i, i - 1));
                                    } else if delta > 16.0 && i < layer_count - 1 {
                                        move_layer = Some((i, i + 1));
                                    }
                                }

                                // Visibility toggle
                                let eye = if layers[i].visible { "👁" } else { "ꞏ" };
                                if ui.small_button(eye).clicked() {
                                    layers[i].visible = !layers[i].visible;
                                }

                                // Remove button
                                if ui.small_button("×").clicked() {
                                    remove_layer = Some(i);
                                }
                            });

                            // Collapsible header with layer name
                            let header_id = ui.make_persistent_id(format!("layer_col_{i}"));
                            let header = egui::CollapsingHeader::new(
                                egui::RichText::new(&layers[i].filename)
                                    .strong(),
                            )
                            .id_salt(header_id)
                            .default_open(is_selected);

                            header.show(ui, |ui| {
                                let layer = &mut layers[i];

                                // Transport
                                ui.horizontal(|ui| {
                                    if ui.button(if layer.paused { "▶" } else { "⏸" }).clicked() {
                                        layer.paused = !layer.paused;
                                    }
                                    if ui.button("Reset FX").clicked() {
                                        layer.effects.reset();
                                    }
                                });

                                labeled_slider(ui, "Speed",
                                    egui::Slider::new(&mut layer.speed, 0.25..=4.0)
                                        .logarithmic(true)
                                        .custom_formatter(|v, _| format!("{:.2}×", v)),
                                );
                                labeled_slider(ui, "FPS",
                                    egui::Slider::new(&mut layer.fps, 1.0..=60.0)
                                        .step_by(1.0)
                                        .custom_formatter(|v, _| format!("{:.0}", v)),
                                );
                                labeled_slider(ui, "Opacity",
                                    egui::Slider::new(&mut layer.opacity, 0.0..=1.0),
                                );
                                ui.horizontal(|ui| {
                                    ui.allocate_ui_with_layout(
                                        egui::vec2(78.0, ui.spacing().interact_size.y),
                                        egui::Layout::left_to_right(egui::Align::Center),
                                        |ui| { ui.label("Blend"); },
                                    );
                                    egui::ComboBox::from_id_salt(format!("blend_mode_{i}"))
                                        .selected_text(layer.blend_mode.label())
                                        .show_ui(ui, |ui| {
                                            for mode in BlendMode::ALL {
                                                ui.selectable_value(
                                                    &mut layer.blend_mode,
                                                    *mode,
                                                    mode.label(),
                                                );
                                            }
                                        });
                                });

                                ui.add_space(4.0);
                                effects_sliders(ui, &mut layer.effects, &format!("layer_{i}"));
                            });

                            ui.separator();
                        }

                        if layers.is_empty() {
                            ui.weak("No active layers");
                            ui.weak("Drop files or add from library →");
                        }
                    });
            }
        });

    // RIGHT panel: Master controls + Library
    egui::Panel::right("right_panel")
        .min_size(260.0)
        .default_size(300.0)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    // === MASTER ===
                    ui.heading("Master");
                    ui.separator();

                    // Transport
                    ui.horizontal(|ui| {
                        if ui
                            .button(if *master_paused { "▶ Play All" } else { "⏸ Pause All" })
                            .clicked()
                        {
                            *master_paused = !*master_paused;
                        }
                        if ui.button("Reset FX").clicked() {
                            master_effects.reset();
                        }
                    });

                    ui.add_space(4.0);
                    effects_sliders(ui, master_effects, "master");

                    ui.add_space(12.0);
                    ui.separator();

                    // === LIBRARY ===
                    ui.heading("Library");
                    ui.separator();

                    // Folder path + change button
                    ui.horizontal(|ui| {
                        let folder_label = library_folder
                            .as_ref()
                            .and_then(|f| f.file_name())
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_else(|| "No folder".into());
                        ui.label(folder_label);
                        if ui.small_button("…").clicked() {
                            change_folder = true;
                        }
                    });

                    ui.add_space(4.0);
                    ui.separator();

                    // File list
                    for file in library_files.iter() {
                        let name = file
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_default();

                        let response = ui.selectable_label(false, &name);
                        if response.double_clicked() {
                            if let Some(path_str) = file.to_str() {
                                add_layer_path = Some(path_str.to_string());
                            }
                        }
                        response.on_hover_text("Double-click to add as layer");
                    }

                    if library_files.is_empty() {
                        ui.weak("No video files");
                        ui.weak("Drop a folder or click …");
                    }
                });
        });

    // Central panel: video output
    egui::CentralPanel::default().show(ctx, |ui| {
        if let Some(tex_id) = video_egui_texture_id {
            let available = ui.available_size();
            let aspect = output_width as f32 / output_height as f32;
            let (w, h) = fit_to_area(available.x, available.y, aspect);
            ui.centered_and_justified(|ui| {
                ui.image(egui::load::SizedTexture::new(tex_id, egui::vec2(w, h)));
            });
        }
    });

    // --- Apply deferred actions ---

    // Change folder (opens native dialog)
    if change_folder {
        if let Some(folder) = rfd::FileDialog::new().pick_folder() {
            *library_files = scan_folder(&folder);
            *library_folder = Some(folder);
        }
    }

    // Move layer
    if let Some((from, to)) = move_layer {
        layers.swap(from, to);
        if *selected_layer == Some(from) {
            *selected_layer = Some(to);
        } else if *selected_layer == Some(to) {
            *selected_layer = Some(from);
        }
    }

    // Remove layer
    if let Some(idx) = remove_layer {
        layers.remove(idx);
        if layers.is_empty() {
            *selected_layer = None;
        } else if let Some(sel) = *selected_layer {
            if sel >= layers.len() {
                *selected_layer = Some(layers.len() - 1);
            }
        }
    }

    add_layer_path
}

/// Inline labeled slider: label on left (fixed width), slider fills remaining space.
fn labeled_slider(ui: &mut egui::Ui, label: &str, slider: egui::Slider<'_>) {
    ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(
            egui::vec2(78.0, ui.spacing().interact_size.y),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| { ui.label(label); },
        );
        ui.add(slider);
    });
}

/// Inline labeled checkbox.
fn labeled_checkbox(ui: &mut egui::Ui, label: &str, value: &mut bool) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(
            egui::vec2(78.0, ui.spacing().interact_size.y),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| { ui.label(label); },
        );
        changed = ui.checkbox(value, "").changed();
    });
    changed
}

/// Shared effects slider UI — used for both master and per-layer effects.
fn effects_sliders(ui: &mut egui::Ui, effects: &mut effects::EffectUniforms, id_prefix: &str) {
    // --- Digital effects ---
    ui.label(egui::RichText::new("Digital").weak().size(11.0));

    labeled_slider(ui, "Pixelate",
        egui::Slider::new(&mut effects.pixelate_size, 1.0..=32.0)
            .step_by(1.0)
            .custom_formatter(|v, _| format!("{:.0}", v)),
    );
    labeled_slider(ui, "RGB Split",
        egui::Slider::new(&mut effects.rgb_split, 0.0..=30.0)
            .step_by(1.0)
            .custom_formatter(|v, _| format!("{:.0}", v)),
    );
    labeled_slider(ui, "Hue",
        egui::Slider::new(&mut effects.hue_shift, -180.0..=180.0)
            .step_by(1.0)
            .suffix("°"),
    );
    labeled_slider(ui, "Saturation",
        egui::Slider::new(&mut effects.saturation, -1.0..=1.0),
    );
    labeled_slider(ui, "Brightness",
        egui::Slider::new(&mut effects.brightness, -1.0..=1.0),
    );
    labeled_slider(ui, "Contrast",
        egui::Slider::new(&mut effects.contrast, -1.0..=1.0),
    );
    labeled_slider(ui, "Posterize",
        egui::Slider::new(&mut effects.posterize, 0.0..=16.0)
            .step_by(1.0)
            .custom_formatter(|v, _| {
                if v < 2.0 { "Off".to_string() } else { format!("{:.0}", v) }
            }),
    );

    let mut invert_on = effects.invert > 0.5;
    if labeled_checkbox(ui, "Invert", &mut invert_on) {
        effects.invert = if invert_on { 1.0 } else { 0.0 };
    }

    ui.add_space(4.0);
    ui.separator();

    // --- Analog effects ---
    ui.label(egui::RichText::new("Analog").weak().size(11.0));

    labeled_slider(ui, "Grain",
        egui::Slider::new(&mut effects.grain_intensity, 0.0..=0.3),
    );

    if effects.grain_intensity > 0.0 {
        labeled_slider(ui, "  Size",
            egui::Slider::new(&mut effects.grain_size, 1.0..=4.0)
                .step_by(1.0)
                .custom_formatter(|v, _| format!("{:.0}", v)),
        );
        ui.horizontal(|ui| {
            ui.allocate_ui_with_layout(
                egui::vec2(78.0, ui.spacing().interact_size.y),
                egui::Layout::left_to_right(egui::Align::Center),
                |ui| { ui.label("  Algo"); },
            );
            egui::ComboBox::from_id_salt(format!("grain_algo_{id_prefix}"))
                .width(90.0)
                .selected_text(match effects.grain_algo as i32 {
                    1 => "Perlin",
                    2 => "Salt&Pepper",
                    3 => "Blue",
                    _ => "Gaussian",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut effects.grain_algo, 0.0, "Gaussian");
                    ui.selectable_value(&mut effects.grain_algo, 1.0, "Perlin");
                    ui.selectable_value(&mut effects.grain_algo, 2.0, "Salt&Pepper");
                    ui.selectable_value(&mut effects.grain_algo, 3.0, "Blue");
                });
        });
        let mut color = effects.color_grain > 0.5;
        if labeled_checkbox(ui, "  Color", &mut color) {
            effects.color_grain = if color { 1.0 } else { 0.0 };
        }
    }

    labeled_slider(ui, "Vignette",
        egui::Slider::new(&mut effects.vignette, 0.0..=1.5),
    );
    labeled_slider(ui, "Drift",
        egui::Slider::new(&mut effects.color_drift, 0.0..=0.02),
    );

    ui.add_space(4.0);
    ui.separator();

    // --- Motion effects ---
    ui.label(egui::RichText::new("Motion").weak().size(11.0));

    labeled_slider(ui, "Bth Scale",
        egui::Slider::new(&mut effects.breathe_scale, 0.0..=0.05),
    );
    labeled_slider(ui, "Bth Rotate",
        egui::Slider::new(&mut effects.breathe_rotation, 0.0..=2.0)
            .suffix("°"),
    );
    labeled_slider(ui, "Bth Drift",
        egui::Slider::new(&mut effects.breathe_position, 0.0..=0.02),
    );
}

/// Fit a rectangle with given aspect ratio into available width/height.
fn fit_to_area(max_w: f32, max_h: f32, aspect: f32) -> (f32, f32) {
    let w = max_w;
    let h = w / aspect;
    if h <= max_h {
        (w, h)
    } else {
        let h = max_h;
        let w = h * aspect;
        (w, h)
    }
}

/// Load custom fonts from `assets/fonts/` at runtime if present.
/// Falls back to egui's built-in fonts on any platform where they're absent,
/// so the build never depends on machine-specific font paths.
fn configure_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();

    let candidates: [(&str, egui::FontFamily, &[&str]); 2] = [
        (
            "CustomSans",
            egui::FontFamily::Proportional,
            &["IBMPlexSans-Regular.otf", "IBMPlexSans-Regular.ttf"],
        ),
        (
            "CustomMono",
            egui::FontFamily::Monospace,
            &["IBMPlexMono-Regular.otf", "IBMPlexMono-Regular.ttf"],
        ),
    ];

    let mut any_loaded = false;
    for (name, family, filenames) in candidates {
        for filename in filenames {
            let path = PathBuf::from("assets/fonts").join(filename);
            if let Ok(bytes) = std::fs::read(&path) {
                fonts.font_data.insert(
                    name.to_owned(),
                    Arc::new(egui::FontData::from_owned(bytes)),
                );
                fonts
                    .families
                    .entry(family.clone())
                    .or_default()
                    .insert(0, name.to_owned());
                any_loaded = true;
                break;
            }
        }
    }

    if any_loaded {
        ctx.set_fonts(fonts);
    }
}

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let arg = args.get(1).cloned();

    // Detect if arg is a folder (library) or a file (single layer)
    let (initial_video, library_folder) = match arg {
        Some(ref path) => {
            let p = PathBuf::from(path);
            if p.is_dir() {
                (None, Some(p))
            } else {
                // It's a file — also use its parent directory as the library
                let parent = p.parent().map(|p| p.to_path_buf());
                (Some(path.clone()), parent)
            }
        }
        None => {
            // Default: use ./videos/ if it exists
            let default_lib = PathBuf::from("videos");
            if default_lib.is_dir() {
                (None, Some(default_lib))
            } else {
                (None, None)
            }
        }
    };

    // Start web control panel server
    let web_state = WebState::new();
    let url = web::server::spawn(web_state.clone(), 3030);
    log::info!("Opening control panel: {}", url);
    let _ = open::that(&url);

    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(initial_video, library_folder, web_state);
    event_loop.run_app(&mut app).unwrap();
}
