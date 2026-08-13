#![allow(deprecated)] // egui 0.34 deprecation warnings for panel API renames

mod audio;
mod effects;
mod input;
mod layers;
mod midi;
mod modulation;
mod morph;
mod ntsc;
mod patch;
mod render_export;
mod renderer;
mod spout_in;
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
use layers::{is_video_file, spout_sender_from_source_path, Layer};
use renderer::Renderer;
use web::state::WebState;

const TARGET_FPS: u64 = 30;
const FRAME_DURATION: Duration = Duration::from_millis(1000 / TARGET_FPS);

fn selected_layer_after_remove(
    selected: Option<usize>,
    removed: usize,
    remaining: usize,
) -> Option<usize> {
    if remaining == 0 {
        return None;
    }
    selected.map(|index| {
        if index > removed {
            index - 1
        } else {
            index.min(remaining - 1)
        }
    })
}

struct App {
    initial_video: Option<String>,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    layers: Vec<Layer>,
    selected_layer: Option<usize>,
    /// Monotonic generation for the live layer stack. Browser actions carry
    /// this (plus a stable layer ID) so stale multi-controller edits cannot
    /// land on a different clip after a topology change.
    layer_stack_revision: u64,
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
    // Last processed NTSC frame presented while the next CPU job runs.
    ntsc_presented: Option<(u64, Vec<u8>)>,
    ntsc_pipeline_enabled: bool,
    // Modulation matrix (BPM clock, LFOs, routings)
    mod_matrix: modulation::ModMatrix,
    // Patch morphing crossfader (A/B slots + t)
    morph: morph::Morph,
    // Blackout: output cut to black (B key / panel button)
    blackout: bool,
    // Advanced at every blackout edge so delayed GPU/CPU frames are rejected.
    visual_epoch: u64,
    // Audio input analysis (modulation source)
    audio: audio::AudioAnalyzer,
    // MIDI input (modulation source)
    midi: midi::MidiEngine,
    // Temporal effects (feedback trails, slit-scan)
    temporal_params: effects::params::TemporalParams,
    // Spout texture-sharing output
    spout: spout_out::SpoutOut,
    spout_enabled: bool,
    output_error: String,
    // Web control panel
    web_state: Arc<WebState>,
    // Offline render export
    export_job: Option<render_export::ExportJob>,
    // Actions latched for the next four-beat downbeat.
    quantized_actions: Vec<web::state::WebAction>,
    quantized_bar: Option<i64>,
}

impl App {
    fn new(
        initial_video: Option<String>,
        library_folder: Option<PathBuf>,
        web_state: Arc<WebState>,
    ) -> Self {
        let library_files = library_folder.as_ref().map(scan_folder).unwrap_or_default();

        // The web server needs the folder for clip uploads.
        if let Ok(mut lf) = web_state.library_folder.write() {
            *lf = library_folder.clone();
        }

        // Generate thumbnails on background thread
        generate_thumbnails(&library_files, web_state.clone());

        Self {
            initial_video,
            window: None,
            renderer: None,
            layers: Vec::new(),
            selected_layer: None,
            layer_stack_revision: 1,
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
            ntsc_presented: None,
            ntsc_pipeline_enabled: false,
            mod_matrix: modulation::ModMatrix::new(),
            morph: morph::Morph::default(),
            blackout: false,
            visual_epoch: 0,
            audio: audio::AudioAnalyzer::new(),
            midi: midi::MidiEngine::new(),
            temporal_params: effects::params::TemporalParams::default(),
            spout: spout_out::SpoutOut::new(),
            spout_enabled: false,
            output_error: String::new(),
            web_state,
            export_job: None,
            quantized_actions: Vec::new(),
            quantized_bar: None,
        }
    }

    fn add_layer(&mut self, path: &str) {
        if self.layers.len() >= modulation::MAX_MOD_LAYERS {
            log::error!(
                "Layer limit reached ({}); every live layer must remain fully routable",
                modulation::MAX_MOD_LAYERS
            );
            return;
        }
        let renderer = self.renderer.as_ref().unwrap();
        match Layer::new(path, &renderer.device) {
            Ok(layer) => {
                self.layers.push(layer);
                self.selected_layer = Some(self.layers.len() - 1);
                self.bump_layer_stack_revision();
                // Morph slots are index-aligned with the layer stack. A
                // topology change invalidates that alignment; clearing is
                // safer than silently morphing the wrong clip.
                self.morph.clear();
            }
            Err(e) => {
                eprintln!("Failed to open video: {e}");
            }
        }
    }

    fn add_spout_layer(&mut self, sender_name: &str) {
        if self.layers.len() >= modulation::MAX_MOD_LAYERS {
            log::error!(
                "Layer limit reached ({}); every live layer must remain fully routable",
                modulation::MAX_MOD_LAYERS
            );
            return;
        }
        let Some(renderer) = self.renderer.as_ref() else {
            log::error!("Cannot add Spout input before the renderer is ready");
            return;
        };
        match Layer::new_spout(sender_name, &renderer.device, &renderer.queue) {
            Ok(layer) => {
                self.layers.push(layer);
                self.selected_layer = Some(self.layers.len() - 1);
                self.bump_layer_stack_revision();
                self.morph.clear();
            }
            Err(error) => log::error!("Failed to create Spout input layer: {error}"),
        }
    }

    fn bump_layer_stack_revision(&mut self) {
        self.layer_stack_revision = self.layer_stack_revision.wrapping_add(1).max(1);
    }

    /// Resolve a browser layer selector. When an ID is supplied it is
    /// authoritative: a stale/unknown ID is rejected instead of falling back
    /// to a positional index that may now name a different layer.
    fn resolve_layer_index(&self, index: usize, layer_id: &Option<String>) -> Option<usize> {
        match layer_id.as_deref().filter(|id| !id.is_empty()) {
            Some(id) => {
                let id = id.parse::<u64>().ok()?;
                self.layers.iter().position(|layer| layer.layer_id() == id)
            }
            None => (index < self.layers.len()).then_some(index),
        }
    }

    fn reset_patch_generation(&mut self) {
        self.quantized_actions.clear();
        // A learn request and its most recent hardware CC belong to the old
        // control generation. Never let either overwrite bindings restored by
        // the newly committed patch.
        self.mod_matrix.midi_learn = None;
        let _ = self.midi.take_last_cc();
        // The file dialog can leave the render loop paused while browser
        // commands accumulate against the old snapshot. Stable IDs reject
        // bundled-client stragglers, and clearing this bounded queue also
        // protects legacy positional clients at the exact commit boundary.
        self.web_state.actions.blocking_lock().clear();
        self.quantized_bar = Some((self.mod_matrix.current_beat / 4.0).floor() as i64);
        self.bump_layer_stack_revision();
        self.visual_epoch = self.visual_epoch.wrapping_add(1);
        self.ntsc_presented = None;
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.reset_visual_generation();
        }
    }

    /// Open or close the dedicated fullscreen output window. When a second
    /// monitor exists, the output goes fullscreen there and the main window
    /// stays with the performer; on a single monitor it takes that one
    /// (Escape or O closes it; the web panel keeps working regardless).
    fn toggle_output_window(&mut self, event_loop: &ActiveEventLoop) {
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        if renderer.has_output() {
            renderer.close_output();
            self.output_error.clear();
            log::info!("Output window closed");
            return;
        }

        let main_monitor = self.window.as_ref().and_then(|w| w.current_monitor());
        let target_monitor = event_loop
            .available_monitors()
            .find(|m| Some(m) != main_monitor.as_ref())
            .or(main_monitor);

        let attrs = WindowAttributes::default()
            .with_title("collide-o-scope — output")
            .with_fullscreen(Some(Fullscreen::Borderless(target_monitor)));

        match event_loop.create_window(attrs) {
            Ok(window) => {
                let window = Arc::new(window);
                match renderer.create_output(window.clone()) {
                    Ok(()) => {
                        self.output_error.clear();
                        log::info!("Output window opened");
                    }
                    Err(error) => {
                        // Dropping the last Arc closes the just-created winit
                        // window; no half-configured output target is retained.
                        log::error!("Output window unavailable: {error}");
                        self.output_error = error;
                        drop(window);
                    }
                }
            }
            Err(e) => {
                self.output_error = format!("could not create output window: {e}");
                log::error!("Output window: {e}");
            }
        }
    }

    fn set_library_folder(&mut self, folder: PathBuf) {
        self.library_files = scan_folder(&folder);
        if let Ok(mut lf) = self.web_state.library_folder.write() {
            *lf = Some(folder.clone());
        }
        self.library_folder = Some(folder);
    }

    /// Apply a patch atomically, rebuilding its saved layer stack before any
    /// live state is replaced. This makes a saved patch a real performance
    /// snapshot instead of merely repainting whichever layers happen to be
    /// open when it is loaded.
    fn apply_loaded_patch(
        &mut self,
        patch: patch::PatchState,
        patch_path: &std::path::Path,
    ) -> Result<(), String> {
        let renderer = self.renderer.as_ref().ok_or("renderer is not ready")?;
        if patch.layers.len() > modulation::MAX_MOD_LAYERS {
            return Err(format!(
                "patch contains {} layers; the fully routable limit is {}",
                patch.layers.len(),
                modulation::MAX_MOD_LAYERS
            ));
        }
        let patch_dir = patch_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));
        let mut rebuilt = Vec::with_capacity(patch.layers.len());

        for config in &patch.layers {
            if let Some(sender_name) = spout_sender_from_source_path(&config.source_path) {
                let mut layer = Layer::new_spout(sender_name, &renderer.device, &renderer.queue)
                    .map_err(|error| format!("{}: {error}", config.filename))?;
                config.apply_to_layer(&mut layer);
                rebuilt.push(layer);
                continue;
            }

            let saved = PathBuf::from(&config.filename);
            let mut candidates = Vec::new();
            if !config.source_path.is_empty() {
                candidates.push(PathBuf::from(&config.source_path));
            }
            if saved.is_absolute() {
                candidates.push(saved.clone());
            } else {
                // A portable patch bundled beside its media is more specific
                // than a same-basename clip in the currently selected library.
                candidates.push(patch_dir.join(&saved));
                if let Some(folder) = &self.library_folder {
                    candidates.push(folder.join(&saved));
                }
                candidates.push(saved.clone());
            }
            let path = candidates
                .into_iter()
                .find(|candidate| candidate.is_file())
                .ok_or_else(|| format!("patch clip not found: {}", config.filename))?;
            let path_text = path.to_string_lossy();
            let mut layer = Layer::new(&path_text, &renderer.device)
                .map_err(|e| format!("{}: {e}", config.filename))?;
            config.apply_to_layer(&mut layer);
            rebuilt.push(layer);
        }

        patch.apply(
            &mut self.master_effects,
            &mut rebuilt,
            &mut self.ntsc_params,
            &mut self.mod_matrix,
            &mut self.temporal_params,
        );
        self.master_paused = patch.master_paused;
        self.layers = rebuilt;
        self.selected_layer = (!self.layers.is_empty()).then_some(0);
        self.morph = patch
            .morph
            .clone()
            .map(|snapshot| {
                morph::Morph::from_snapshot_at_beat(snapshot, self.mod_matrix.current_beat)
            })
            .unwrap_or_default();

        // A replacement patch is a new state and visual generation. Nothing
        // queued against the prior layer/morph topology may fire later, and
        // no temporal/readback/NTSC pixel from the prior patch may leak into
        // this one.
        self.reset_patch_generation();

        // Patch recall stores the requested CPAL device. If a stream is still
        // open for another preference, stop it now; the frame pump reopens
        // the desired device (or its explicit default-device fallback).
        if self.audio.is_running() && !self.audio.is_running_for(&self.mod_matrix.audio_device) {
            self.audio.stop();
        }
        Ok(())
    }

    fn quantized_action_key(action: &web::state::WebAction) -> Option<String> {
        use web::state::WebAction;
        match action {
            WebAction::SetParam { param, .. } => Some(format!("master:{param}")),
            WebAction::SetLayerParam {
                index,
                layer_id,
                param,
                ..
            } => Some(match layer_id.as_deref().filter(|id| !id.is_empty()) {
                Some(id) => format!("layer:id:{id}:{param}"),
                None => format!("layer:index:{index}:{param}"),
            }),
            WebAction::SetLayerEffect {
                index,
                layer_id,
                param,
                ..
            } => Some(match layer_id.as_deref().filter(|id| !id.is_empty()) {
                Some(id) => format!("layer:id:{id}:effect:{param}"),
                None => format!("layer:index:{index}:effect:{param}"),
            }),
            WebAction::SetNtscParam { param, .. } => Some(format!("ntsc:{param}")),
            WebAction::SetTemporal { param, .. } => Some(format!("temporal:{param}")),
            WebAction::SetMorph { .. } => Some("morph:t".to_string()),
            WebAction::MorphGlide { .. } => Some("morph:glide".to_string()),
            WebAction::MorphCapture { slot } => Some(format!("morph:capture:{slot}")),
            WebAction::MorphClear => Some("morph:clear".to_string()),
            _ => None,
        }
    }

    /// Output-window creation needs the event loop and therefore cannot be
    /// delegated to the plain state-action handler. Treat it as immediate even
    /// if an old or hand-authored client wrapped it in one or more latches.
    fn is_output_window_action(action: &web::state::WebAction) -> bool {
        match action {
            web::state::WebAction::ToggleOutputWindow => true,
            web::state::WebAction::Quantized { inner } => Self::is_output_window_action(inner),
            _ => false,
        }
    }

    fn remap_quantized_layers_after_remove(&mut self, removed: usize) {
        self.quantized_actions.retain_mut(|action| match action {
            web::state::WebAction::SetLayerParam {
                index,
                layer_id: None,
                ..
            }
            | web::state::WebAction::SetLayerEffect {
                index,
                layer_id: None,
                ..
            } if *index == removed => false,
            web::state::WebAction::SetLayerParam {
                index,
                layer_id: None,
                ..
            }
            | web::state::WebAction::SetLayerEffect {
                index,
                layer_id: None,
                ..
            } => {
                if *index > removed {
                    *index -= 1;
                }
                true
            }
            _ => true,
        });
    }

    fn remap_quantized_layers_after_move(&mut self, from: usize, to: usize) {
        for action in &mut self.quantized_actions {
            let index = match action {
                web::state::WebAction::SetLayerParam {
                    index,
                    layer_id: None,
                    ..
                }
                | web::state::WebAction::SetLayerEffect {
                    index,
                    layer_id: None,
                    ..
                } => index,
                _ => continue,
            };
            if from == to {
                continue;
            }
            *index = if *index == from {
                to
            } else if from < to && *index > from && *index <= to {
                *index - 1
            } else if to < from && *index >= to && *index < from {
                *index + 1
            } else {
                *index
            };
        }
    }

    fn toggle_blackout(&mut self) {
        self.set_blackout(!self.blackout);
    }

    fn set_blackout(&mut self, enabled: bool) {
        if self.blackout == enabled {
            return;
        }
        self.blackout = enabled;
        self.visual_epoch = self.visual_epoch.saturating_add(1);
        self.ntsc_presented = None;

        self.ensure_spout_black();
    }

    /// Replace Spout's retained texture exactly once for each blackout
    /// generation. This also covers enabling Spout after blackout is active.
    fn ensure_spout_black(&mut self) {
        if !self.blackout
            || !self.spout_enabled
            || !self.spout.is_running()
            || self.spout.black_delivered(self.visual_epoch)
        {
            return;
        }
        let Some(renderer) = self.renderer.as_ref() else {
            return;
        };
        let width = renderer.output_width;
        let height = renderer.output_height;
        self.spout.cut_to_black(width, height, self.visual_epoch);
    }

    fn queue_quantized_action(&mut self, action: web::state::WebAction) {
        let Some(key) = Self::quantized_action_key(&action) else {
            // Quantization is deliberately limited to non-emergency,
            // continuously expressive controls. Unknown wrappers retain
            // normal immediate behavior.
            self.handle_web_action(action);
            return;
        };
        if let Some(existing) = self.quantized_actions.iter().position(|candidate| {
            Self::quantized_action_key(candidate).as_deref() == Some(key.as_str())
        }) {
            self.quantized_actions[existing] = action;
        } else if self.quantized_actions.len() < 256 {
            self.quantized_actions.push(action);
        }
    }

    fn release_quantized_actions_on_downbeat(&mut self) {
        let bar = (self.mod_matrix.current_beat / 4.0).floor() as i64;
        let crossed = self
            .quantized_bar
            .map(|previous| previous != bar)
            .unwrap_or(false);
        self.quantized_bar = Some(bar);
        if !crossed || self.quantized_actions.is_empty() {
            return;
        }
        let actions = std::mem::take(&mut self.quantized_actions);
        for action in actions {
            self.handle_web_action(action);
        }
    }

    /// Handle an action from the web UI.
    fn handle_web_action(&mut self, action: web::state::WebAction) {
        use web::state::WebAction;
        match action {
            WebAction::Quantized { inner } => self.queue_quantized_action(*inner),
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
                        .as_deref()
                        == Some(&filename)
                }) {
                    let path_str = path.to_string_lossy().to_string();
                    self.add_layer(&path_str);
                }
            }
            WebAction::AddSpoutLayer { sender } => self.add_spout_layer(&sender),
            WebAction::RemoveLayer { index, layer_id } => {
                if let Some(index) = self.resolve_layer_index(index, &layer_id) {
                    self.layers.remove(index);
                    self.remap_quantized_layers_after_remove(index);
                    self.bump_layer_stack_revision();
                    self.morph.clear();
                    self.selected_layer =
                        selected_layer_after_remove(self.selected_layer, index, self.layers.len());
                }
            }
            WebAction::MoveLayer {
                from,
                to,
                layer_id,
                stack_revision,
            } => {
                let revision_current = stack_revision
                    .map(|revision| revision == self.layer_stack_revision)
                    .unwrap_or(true);
                let resolved_from = self.resolve_layer_index(from, &layer_id);
                if !revision_current {
                    log::warn!(
                        "Rejected stale layer move at revision {:?}; current revision is {}",
                        stack_revision,
                        self.layer_stack_revision
                    );
                } else if let Some(from) = resolved_from {
                    if to < self.layers.len() && from != to {
                        let layer = self.layers.remove(from);
                        self.layers.insert(to, layer);
                        self.remap_quantized_layers_after_move(from, to);
                        self.bump_layer_stack_revision();
                        self.selected_layer = self.selected_layer.map(|selected| {
                            if selected == from {
                                to
                            } else if from < to && selected > from && selected <= to {
                                selected - 1
                            } else if to < from && selected >= to && selected < from {
                                selected + 1
                            } else {
                                selected
                            }
                        });
                        // Morph slots are index-addressed; never let a reorder
                        // silently bind a captured value to the wrong clip.
                        self.morph.clear();
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
            WebAction::SetLayerVisibility {
                index,
                layer_id,
                visible,
            } => {
                if let Some(index) = self.resolve_layer_index(index, &layer_id) {
                    self.layers[index].visible = visible;
                }
            }
            WebAction::SetLayerPaused {
                index,
                layer_id,
                paused,
            } => {
                if let Some(index) = self.resolve_layer_index(index, &layer_id) {
                    self.layers[index].paused = paused;
                    self.layers[index].reset_transport_timing();
                }
            }
            WebAction::SetMasterPaused { paused } => {
                self.master_paused = paused;
                for layer in &mut self.layers {
                    layer.reset_transport_timing();
                }
            }
            WebAction::SetBlackout { enabled } => self.set_blackout(enabled),
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
                        self.master_effects.downsample = defaults.downsample;
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
            WebAction::SetLfo {
                index,
                param,
                value,
            } => {
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
                                lfo.set_phase(n as f32);
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
                    "device" => {
                        if let Some(s) = value.as_str() {
                            self.mod_matrix.audio_device = s.to_string();
                            // Restart on the new device if currently running;
                            // the frame loop re-opens it while enabled.
                            if self.audio.is_running() {
                                self.audio.stop();
                            }
                        }
                    }
                    "band_count" => {
                        if let Some(count) = value.as_u64() {
                            let config =
                                self.mod_matrix.audio_band_config.with_count(count as usize);
                            self.mod_matrix.audio_band_config = self.audio.set_band_config(config);
                        }
                    }
                    "band_edges" => {
                        let current = self.mod_matrix.audio_band_config;
                        if let Some(raw_edges) =
                            value.get("edges").and_then(serde_json::Value::as_array)
                        {
                            let edges: Vec<f32> = raw_edges
                                .iter()
                                .filter_map(serde_json::Value::as_f64)
                                .map(|number| number as f32)
                                .collect();
                            let count = value
                                .get("count")
                                .and_then(serde_json::Value::as_u64)
                                .map(|count| count as usize)
                                .unwrap_or(current.count());
                            let ceiling = value
                                .get("ceiling")
                                .and_then(serde_json::Value::as_f64)
                                .map(|number| number as f32)
                                .unwrap_or(current.ceiling_hz());
                            let config = audio::AudioBandConfig::new(count, &edges, ceiling);
                            self.mod_matrix.audio_band_config = self.audio.set_band_config(config);
                        } else {
                            // Pre-configurable-count browser clients sent
                            // named bass/mid/high values, with "high" acting
                            // as the analysis ceiling.
                            let legacy = self.audio.band_edges();
                            let read = |key: &str, fallback: f32| {
                                value
                                    .get(key)
                                    .and_then(serde_json::Value::as_f64)
                                    .map(|number| number as f32)
                                    .unwrap_or(fallback)
                            };
                            self.audio.set_band_edges(
                                read("bass", legacy.bass_hz),
                                read("mid", legacy.mid_hz),
                                read("high", legacy.high_hz),
                            );
                            self.mod_matrix.audio_band_config = self.audio.band_config();
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
                            Some(slot)
                                if (slot as usize) < modulation::NUM_MIDI_SLOTS
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
                            self.mod_matrix.midi_ccs[slot] = n.min(127) as u8;
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
                self.mod_matrix.set_gyro_degrees(alpha, beta, gamma);
            }
            WebAction::GyroCalibrate => self.mod_matrix.calibrate_gyro(),
            WebAction::SetGyroConfig { axis, param, value } => {
                let axis_index = match axis.as_str() {
                    "yaw" => Some(0),
                    "pitch" => Some(1),
                    "roll" => Some(2),
                    _ => None,
                };
                if let Some(config) =
                    axis_index.and_then(|i| self.mod_matrix.gyro_config.get_mut(i))
                {
                    match param.as_str() {
                        "range" => {
                            if let Some(n) = value.as_f64() {
                                config.range_degrees = (n as f32).clamp(1.0, 180.0);
                            }
                        }
                        "expo" => {
                            if let Some(n) = value.as_f64() {
                                config.expo = (n as f32).clamp(-2.0, 2.0);
                            }
                        }
                        "invert" => {
                            if let Some(enabled) = value.as_bool() {
                                config.invert = enabled;
                            }
                        }
                        _ => {}
                    }
                    self.mod_matrix.recompute_gyro();
                }
            }
            WebAction::Pad { x, y, active } => self.mod_matrix.set_pad(x, y, active),
            WebAction::SetPadConfig { axis, param, value } => match param.as_str() {
                "spring_enabled" => {
                    if let Some(enabled) = value.as_bool() {
                        self.mod_matrix.pad_config.spring_enabled = enabled;
                    }
                }
                "spring_rate" => {
                    if let Some(n) = value.as_f64() {
                        self.mod_matrix.pad_config.spring_rate = (n as f32).clamp(0.1, 20.0);
                    }
                }
                _ => {
                    let axis_index = match axis.as_str() {
                        "x" => Some(0),
                        "y" => Some(1),
                        _ => None,
                    };
                    if let Some(config) =
                        axis_index.and_then(|i| self.mod_matrix.pad_config.axes.get_mut(i))
                    {
                        match param.as_str() {
                            "curve" => {
                                if let Some(name) = value.as_str() {
                                    config.curve = modulation::Curve::from_str(name);
                                }
                            }
                            "curve_amount" => {
                                if let Some(n) = value.as_f64() {
                                    config.curve_amount = (n as f32).clamp(-2.0, 2.0);
                                }
                            }
                            "quantize" => {
                                if let Some(n) = value.as_u64() {
                                    config.quantize = n.min(64) as u32;
                                }
                            }
                            _ => {}
                        }
                    }
                }
            },
            WebAction::ToggleOutputWindow => {
                // Handled in the action drain loop, which has the event
                // loop needed for window creation. Never reaches here.
            }
            WebAction::MorphCapture { slot } => {
                let snap = morph::MorphSlot::capture(
                    &self.master_effects,
                    &self.ntsc_params,
                    &self.temporal_params,
                    &self.layers,
                );
                match slot.as_str() {
                    "b" => self.morph.b = Some(snap),
                    _ => self.morph.a = Some(snap),
                }
            }
            WebAction::MorphClear => {
                self.morph.clear();
            }
            WebAction::SetMorph { value } => {
                self.morph.set_position(value);
            }
            WebAction::SetMorphLaw { law } => {
                self.morph.blend_law = match law.as_str() {
                    "equal_power" => morph::MorphBlendLaw::EqualPower,
                    _ => morph::MorphBlendLaw::Linear,
                };
            }
            WebAction::MorphGlide {
                target,
                duration_beats,
            } => {
                self.morph
                    .start_glide(target, duration_beats, self.mod_matrix.current_beat);
            }
            WebAction::ToggleBlackout => {
                self.toggle_blackout();
            }
            WebAction::RescanLibrary => {
                if let Some(folder) = self.library_folder.clone() {
                    self.library_files = scan_folder(&folder);
                    // Thumbnails only for clips the cache hasn't seen.
                    let new_files: Vec<PathBuf> = self
                        .library_files
                        .iter()
                        .filter(|p| {
                            let name = p
                                .file_name()
                                .map(|n| n.to_string_lossy().to_string())
                                .unwrap_or_default();
                            !self
                                .web_state
                                .thumbnails
                                .read()
                                .map(|c| c.contains_key(&name))
                                .unwrap_or(false)
                        })
                        .cloned()
                        .collect();
                    if !new_files.is_empty() {
                        log::info!("Library rescan: {} new clip(s)", new_files.len());
                        generate_thumbnails(&new_files, self.web_state.clone());
                    }
                }
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
                            p.slit_angle = p.slit_axis * 90.0;
                        }
                    }
                    "slit_angle" => {
                        if let Some(n) = value.as_f64() {
                            let angle = n as f32;
                            if angle.is_finite() {
                                p.slit_angle = angle.clamp(-180.0, 180.0);
                            }
                        }
                    }
                    _ => {}
                }
            }
            WebAction::SetSpout { enabled } => {
                self.spout_enabled = enabled;
            }
            WebAction::SetRouting {
                index,
                param,
                value,
            } => {
                if let Some(routing) = self.mod_matrix.routings.get_mut(index) {
                    match param.as_str() {
                        "source" => {
                            if let Some(s) = value.as_str() {
                                if let Some(source) = modulation::ModSource::try_from_str(s) {
                                    routing.source = source;
                                    routing.reset_runtime();
                                }
                            }
                        }
                        "target" => {
                            if let Some(s) = value.as_str() {
                                if modulation::is_valid_target(s) {
                                    routing.target = s.to_string();
                                }
                            }
                        }
                        "depth" => {
                            if let Some(n) = value.as_f64() {
                                routing.depth = (n as f32).clamp(-1.0, 1.0);
                            }
                        }
                        "curve" => {
                            if let Some(s) = value.as_str() {
                                routing.curve = modulation::Curve::from_str(s);
                                routing.reset_runtime();
                            }
                        }
                        "curve_amount" => {
                            if let Some(n) = value.as_f64() {
                                routing.curve_amount = (n as f32).clamp(-2.0, 2.0);
                                routing.reset_runtime();
                            }
                        }
                        "attack" => {
                            if let Some(n) = value.as_f64() {
                                routing.attack = (n as f32).clamp(0.0, 10.0);
                                routing.reset_runtime();
                            }
                        }
                        "release" => {
                            if let Some(n) = value.as_f64() {
                                routing.release = (n as f32).clamp(0.0, 10.0);
                                routing.reset_runtime();
                            }
                        }
                        _ => {}
                    }
                }
            }
            WebAction::SetLayerParam {
                index,
                layer_id,
                param,
                value,
            } => {
                if let Some(index) = self.resolve_layer_index(index, &layer_id) {
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
                        "fps" => {
                            if let Some(v) = value.as_f64() {
                                let fps = v as f32;
                                if fps.is_finite() {
                                    layer.fps = fps.clamp(1.0, 240.0);
                                    layer.reset_transport_timing();
                                }
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
            WebAction::SetLayerEffect {
                index,
                layer_id,
                param,
                value,
            } => {
                if let Some(index) = self.resolve_layer_index(index, &layer_id) {
                    let layer = &mut self.layers[index];
                    let mut snapshot = web::state::EffectsSnapshot::from_uniforms(&layer.effects);
                    snapshot.apply_param(&param, &value);
                    snapshot.apply_to_uniforms(&mut layer.effects);
                }
            }
            WebAction::ResetLayerFx { index, layer_id } => {
                if let Some(index) = self.resolve_layer_index(index, &layer_id) {
                    self.layers[index].effects.reset();
                }
            }
            WebAction::SetNtscParam { param, value } => {
                self.ntsc_params.set_param(&param, &value);
            }
            WebAction::StartExport {
                width,
                height,
                fps,
                duration_secs,
                audio_layer,
                audio_layer_id,
            } => {
                if self
                    .export_job
                    .as_ref()
                    .map(render_export::ExportJob::can_replace)
                    .unwrap_or(true)
                {
                    let audio_index = match audio_layer_id.as_deref().filter(|id| !id.is_empty()) {
                        Some(id) => {
                            let Some(id) = id.parse::<u64>().ok() else {
                                log::warn!("Rejected export with malformed audio layer ID");
                                return;
                            };
                            let Some(index) =
                                self.layers.iter().position(|layer| layer.layer_id() == id)
                            else {
                                log::warn!("Rejected export with stale audio layer ID {id}");
                                return;
                            };
                            Some(index)
                        }
                        None => audio_layer.filter(|index| *index < self.layers.len()),
                    };
                    let patch = patch::PatchState::capture(
                        &self.master_effects,
                        &self.layers,
                        &self.ntsc_params,
                        &self.mod_matrix,
                        &self.temporal_params,
                        self.master_paused,
                        &self.morph,
                    );
                    // Millisecond resolution prevents a second completed/failed
                    // export launched in the same wall-clock second from
                    // overwriting the first one (ffmpeg intentionally uses -y).
                    let timestamp = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis();
                    let output_dir = self
                        .library_folder
                        .as_ref()
                        .map(|f| f.parent().unwrap_or(f).join("renders"))
                        .unwrap_or_else(|| std::path::PathBuf::from("renders"));
                    let output_path = format!(
                        "{}/patch_{}_{width}x{height}.mp4",
                        output_dir.display(),
                        timestamp
                    );
                    let lib_folder = self
                        .library_folder
                        .as_ref()
                        .map(|f| f.to_string_lossy().to_string())
                        .unwrap_or_else(|| ".".to_string());
                    let config = render_export::ExportConfig {
                        width,
                        height,
                        fps,
                        duration_secs,
                        output_path,
                        audio_path: audio_index
                            .and_then(|index| self.layers.get(index))
                            .filter(|layer| layer.is_video())
                            .map(|layer| layer.source_path.clone()),
                    };
                    for layer in self.layers.iter().filter(|layer| !layer.is_video()) {
                        log::warn!(
                            "Export: live source '{}' will be represented by deterministic black",
                            layer.filename
                        );
                    }
                    let renderer = self
                        .renderer
                        .as_ref()
                        .expect("renderer exists while handling export action");
                    self.export_job = Some(render_export::ExportJob::start_with_gpu(
                        patch,
                        config,
                        &lib_folder,
                        renderer.device.clone(),
                        renderer.queue.clone(),
                    ));
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
            MidiSnapshot, ModSnapshot, MorphSnapshot, NtscSnapshot, SpoutSnapshot,
            TemporalSnapshot,
        };

        let (export_progress, export_error, export_status) = match self.export_job.as_ref() {
            None => (0.0, String::new(), "idle"),
            Some(job)
                if !job.is_done()
                    && job
                        .progress
                        .cancel
                        .load(std::sync::atomic::Ordering::Acquire) =>
            {
                (job.progress.progress_f32(), String::new(), "cancelling")
            }
            Some(job) if !job.is_done() => (job.progress.progress_f32(), String::new(), "running"),
            Some(job) => {
                let error = job.progress.error.lock().unwrap().clone();
                if job
                    .progress
                    .cancelled
                    .load(std::sync::atomic::Ordering::Relaxed)
                {
                    (job.progress.progress_f32(), error, "cancelled")
                } else if error.is_empty() {
                    (1.0, error, "succeeded")
                } else {
                    (job.progress.progress_f32(), error, "failed")
                }
            }
        };

        let mut ntsc_snapshot = NtscSnapshot::from_params(&self.ntsc_params);
        ntsc_snapshot.error = self.ntsc_worker.error().to_string();

        let snapshot = AppSnapshot {
            msg_type: "state".to_string(),
            effects: EffectsSnapshot::from_uniforms(&self.master_effects),
            ntsc: ntsc_snapshot,
            layers: self
                .layers
                .iter()
                .map(|layer| {
                    let spout_status = layer.spout_status();
                    let video_health = layer.video_health();
                    let (video_active, video_error) = match video_health {
                        Some(video::threaded::DecoderHealth::Healthy) | None => {
                            (true, String::new())
                        }
                        Some(video::threaded::DecoderHealth::Failed(error)) => (false, error),
                    };
                    LayerSnapshot {
                        layer_id: layer.layer_id().to_string(),
                        filename: layer.filename.clone(),
                        visible: layer.visible,
                        paused: layer.paused,
                        opacity: layer.opacity,
                        speed: layer.speed,
                        fps: layer.fps,
                        blend_mode: layer.blend_mode.key().to_string(),
                        progress: layer.progress(),
                        key_mode: layer.effects.key_mode as u32,
                        key_threshold: layer.effects.key_threshold,
                        key_softness: layer.effects.key_softness,
                        effects: EffectsSnapshot::from_uniforms(&layer.effects),
                        source_kind: layer.source_kind().to_string(),
                        source_name: spout_status
                            .as_ref()
                            .map(|status| status.sender_name.clone())
                            .unwrap_or_default(),
                        source_active: spout_status
                            .as_ref()
                            .map(|status| status.active)
                            .unwrap_or(video_active),
                        source_width: spout_status
                            .as_ref()
                            .map(|status| status.width)
                            .unwrap_or(layer.width),
                        source_height: spout_status
                            .as_ref()
                            .map(|status| status.height)
                            .unwrap_or(layer.height),
                        source_sequence: spout_status
                            .as_ref()
                            .map(|status| status.sequence)
                            .unwrap_or_default(),
                        source_error: spout_status
                            .map(|status| status.error)
                            .unwrap_or(video_error),
                        offline_export_policy: if layer.is_video() {
                            String::new()
                        } else {
                            "Live Spout input renders as deterministic black offline".to_string()
                        },
                    }
                })
                .collect(),
            layer_stack_revision: self.layer_stack_revision,
            library: self
                .library_files
                .iter()
                .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
                .collect(),
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
                devices: self.audio.devices.clone(),
                selected: self.mod_matrix.audio_device.clone(),
                active_device: self.audio.active_device().to_string(),
                using_fallback: self.audio.is_using_device_fallback(),
                band_count: self.mod_matrix.audio_band_config.count(),
                band_edges: self.mod_matrix.audio_band_config.crossovers().to_vec(),
                band_ceiling_hz: self.mod_matrix.audio_band_config.ceiling_hz(),
                bands: self.mod_matrix.audio.bands[..self.mod_matrix.audio_band_config.count()]
                    .to_vec(),
                spectrum: self.audio.spectrum().to_vec(),
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
            remote_url: self
                .web_state
                .lan_url
                .read()
                .map(|s| s.clone())
                .unwrap_or_default(),
            output_window: self
                .renderer
                .as_ref()
                .map(|r| r.has_output())
                .unwrap_or(false),
            output_error: self.output_error.clone(),
            blackout: self.blackout,
            morph: MorphSnapshot {
                has_a: self.morph.a.is_some(),
                has_b: self.morph.b.is_some(),
                active: self.morph.active(),
                t: self.morph.position_at_beat(self.mod_matrix.current_beat),
                blend_law: match self.morph.blend_law {
                    morph::MorphBlendLaw::Linear => "linear",
                    morph::MorphBlendLaw::EqualPower => "equal_power",
                }
                .to_string(),
                gliding: self.morph.glide.is_some(),
                glide_target: self
                    .morph
                    .glide
                    .map(|glide| glide.target)
                    .unwrap_or(self.morph.t),
                glide_duration_beats: self
                    .morph
                    .glide
                    .map(|glide| glide.duration_beats)
                    .unwrap_or(0.0),
            },
            export_progress,
            export_error,
            export_status: export_status.to_string(),
            quantized_pending: self.quantized_actions.len(),
        };

        // Non-blocking: try to write + broadcast
        if let Ok(mut app) = self.web_state.app.try_write() {
            *app = snapshot.clone();
        }
        match serde_json::to_string(&snapshot) {
            Ok(message) => {
                let _ = self.web_state.tx.send(message);
            }
            Err(error) => {
                log::error!("Web snapshot serialization failed; state update skipped: {error}");
            }
        }
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
                let handles: Vec<_> = chunk
                    .iter()
                    .map(|path| {
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
                                    "-i",
                                    &path.to_string_lossy(),
                                    "-vframes",
                                    "1",
                                    "-vf",
                                    "scale=180:-1",
                                    "-f",
                                    "image2pipe",
                                    "-vcodec",
                                    "mjpeg",
                                    "-q:v",
                                    "8",
                                    "-loglevel",
                                    "error",
                                    "pipe:1",
                                ])
                                .output();

                            match output {
                                Ok(result)
                                    if result.status.success() && !result.stdout.is_empty() =>
                                {
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
                    })
                    .collect();

                for h in handles {
                    let _ = h.join();
                }
            }

            log::info!(
                "Generated {}/{total} thumbnails",
                count.load(Ordering::Relaxed)
            );

            // Pass 2: Generate preview frames (~8 per video, parallel batches of 4)
            let preview_count = Arc::new(AtomicUsize::new(0));
            for chunk in paths.chunks(4) {
                let handles: Vec<_> = chunk
                    .iter()
                    .map(|path| {
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
                                    "-v",
                                    "error",
                                    "-show_entries",
                                    "format=duration",
                                    "-of",
                                    "csv=p=0",
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
                                        "-ss",
                                        &seek_str,
                                        "-i",
                                        &path.to_string_lossy(),
                                        "-vframes",
                                        "1",
                                        "-vf",
                                        "scale=180:-1",
                                        "-f",
                                        "image2pipe",
                                        "-vcodec",
                                        "mjpeg",
                                        "-q:v",
                                        "10",
                                        "-loglevel",
                                        "error",
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
                    })
                    .collect();

                for h in handles {
                    let _ = h.join();
                }
            }

            log::info!(
                "Generated {}/{total} preview strips",
                preview_count.load(Ordering::Relaxed)
            );
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
            match video::VideoDecoder::open(path) {
                Ok(decoder) => {
                    output_width = decoder.width;
                    output_height = decoder.height;
                }
                Err(error) => {
                    log::error!(
                        "Initial video could not define the output; using 1280x720: {error}"
                    );
                }
            }
        }

        log::info!("Output: {}x{}", output_width, output_height);

        // Master effects operate in output pixels. Keep their resolution in
        // lock-step with the fixed renderer output so pixelate/RGB split have
        // the same physical scale live and in offline exports.
        self.master_effects.resolution = [output_width as f32, output_height as f32];

        let window_w = output_width;
        let window_h = output_height;

        let window_attrs = WindowAttributes::default()
            .with_title("collide-o-scope")
            .with_inner_size(winit::dpi::LogicalSize::new(window_w, window_h));

        let window = Arc::new(event_loop.create_window(window_attrs).unwrap());

        let renderer = match Renderer::new(window.clone(), output_width, output_height) {
            Ok(renderer) => renderer,
            Err(error) => {
                log::error!("Renderer initialization failed: {error}");
                event_loop.exit();
                return;
            }
        };

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
        window_id: WindowId,
        event: WindowEvent,
    ) {
        // Events for the dedicated output window: it only needs to close,
        // resize, and yield to Escape/O — everything else stays with the
        // main window and never reaches egui.
        if let Some(renderer) = self.renderer.as_mut() {
            if renderer.output_window_id() == Some(window_id) {
                match event {
                    WindowEvent::CloseRequested => renderer.close_output(),
                    WindowEvent::Resized(size) => renderer.resize_output(size.width, size.height),
                    WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                physical_key,
                                state: winit::event::ElementState::Pressed,
                                ..
                            },
                        ..
                    } => {
                        use winit::keyboard::{KeyCode, PhysicalKey};
                        if matches!(
                            physical_key,
                            PhysicalKey::Code(KeyCode::Escape) | PhysicalKey::Code(KeyCode::KeyO)
                        ) {
                            renderer.close_output();
                        }
                    }
                    _ => {}
                }
                return;
            }
        }

        if let Some(egui_winit) = &mut self.egui_winit {
            let response = egui_winit.on_window_event(self.window.as_ref().unwrap(), &event);
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
                if state == winit::event::ElementState::Pressed && self.modifiers.control_key() {
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
                                self.master_paused,
                                &self.morph,
                            );
                            return;
                        }
                        PhysicalKey::Code(KeyCode::KeyO) => {
                            if let Some((loaded, path)) = patch::editor::load_patch() {
                                match self.apply_loaded_patch(loaded, &path) {
                                    Ok(()) => log::info!("Loaded patch: {}", path.display()),
                                    Err(e) => log::error!("Failed to apply patch: {e}"),
                                }
                            }
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
                            ControlFlow::ToggleOutputWindow => {
                                self.toggle_output_window(event_loop)
                            }
                            ControlFlow::ToggleBlackout => self.toggle_blackout(),
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
                        ControlFlow::ToggleOutputWindow => self.toggle_output_window(event_loop),
                        ControlFlow::ToggleBlackout => self.toggle_blackout(),
                        _ => {}
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                let win_size = self.window.as_ref().unwrap().inner_size();
                // Windows may report a zero-sized surface while the preview
                // is minimized. The engine and dedicated output must keep
                // running; only presentation to this surface is skipped.
                let preview_surface_available = win_size.width > 0 && win_size.height > 0;

                let now = Instant::now();
                if now - self.last_frame_time >= FRAME_DURATION {
                    let frame_delta = now
                        .saturating_duration_since(self.last_frame_time)
                        .as_secs_f32();
                    self.last_frame_time = now;

                    // Process actions from web UI
                    let pending_actions: Vec<_> = self
                        .web_state
                        .actions
                        .try_lock()
                        .map(|mut a| a.drain(..).collect())
                        .unwrap_or_default();
                    for action in pending_actions {
                        // Output-window toggling needs the event loop for
                        // window creation; everything else is plain state.
                        // Peel any legacy quantized wrapper so it cannot turn
                        // this immediate window command into a silent no-op.
                        if Self::is_output_window_action(&action) {
                            self.toggle_output_window(event_loop);
                        } else {
                            self.handle_web_action(action);
                        }
                    }

                    // Build minimal egui frame (video display only, no UI panels)
                    let window = self.window.as_ref().unwrap();
                    let egui_winit = self.egui_winit.as_mut().unwrap();
                    let raw_input = egui_winit.take_egui_input(window);

                    let video_egui_texture_id = self.video_egui_texture_id;
                    let output_width = self.renderer.as_ref().unwrap().output_width;
                    let output_height = self.renderer.as_ref().unwrap().output_height;

                    let egui_context = self.egui_ctx.clone();
                    let yaml_editor = &mut self.yaml_editor;
                    let layers = &mut self.layers;
                    let master_effects = &mut self.master_effects;
                    let full_output = egui_context.run_ui(raw_input, |ctx| {
                        if yaml_editor.active {
                            egui::SidePanel::left("yaml_editor_panel")
                                .default_width(360.0)
                                .resizable(true)
                                .show(ctx, |ui| {
                                    ui.heading("Patch editor");
                                    ui.weak("Ctrl+E closes · edits apply live");
                                    ui.separator();
                                    patch::editor::build_yaml_editor_content(
                                        ui,
                                        layers,
                                        master_effects,
                                        yaml_editor,
                                    );
                                });
                        }
                        // Video remains visible beside the optional code editor.
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
                    self.morph.settle_glide_at(self.mod_matrix.current_beat);
                    self.push_web_state();

                    let window = self.window.as_ref().unwrap().clone();
                    let egui_winit = self.egui_winit.as_mut().unwrap();
                    egui_winit.handle_platform_output(&window, full_output.platform_output);

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
                    self.mod_matrix.audio_band_config = self
                        .audio
                        .set_band_config(self.mod_matrix.audio_band_config);
                    if self.mod_matrix.audio_enabled
                        && !self.audio.is_running_for(&self.mod_matrix.audio_device)
                    {
                        let device = self.mod_matrix.audio_device.clone();
                        if self.audio.is_running() {
                            self.audio.stop();
                        }
                        self.audio.start(&device);
                        if !self.audio.error.is_empty() {
                            // Device failed — flip the switch back off so the
                            // UI reflects reality instead of retrying forever.
                            self.mod_matrix.audio_enabled = false;
                        }
                    } else if !self.mod_matrix.audio_enabled && self.audio.is_running() {
                        self.audio.stop();
                    }
                    self.mod_matrix.audio = self.audio.analyze(self.mod_matrix.audio_gain);
                    if !self.audio.is_running() && !self.audio.error.is_empty() {
                        // Runtime capture failures arrive asynchronously; keep
                        // the requested toggle honest and avoid reopening a
                        // dead device every frame.
                        self.mod_matrix.audio_enabled = false;
                    }

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
                        self.mod_matrix.midi[i] = self.midi.cc_value(self.mod_matrix.midi_ccs[i]);
                    }

                    // MIDI clock sync: while pulses arrive, the external clock
                    // owns BPM and beat position; when they stop, the internal
                    // clock resumes from the same position.
                    if self.mod_matrix.midi_clock_sync && self.midi.is_running() {
                        match self.midi.clock_state() {
                            Some((bpm, beat)) => {
                                self.mod_matrix.clock.set_bpm(bpm);
                                self.mod_matrix
                                    .clock
                                    .set_external_beat(Some(beat), Instant::now());
                            }
                            None => {
                                self.mod_matrix
                                    .clock
                                    .set_external_beat(None, Instant::now());
                            }
                        }
                    } else if self.mod_matrix.clock.is_external() {
                        self.mod_matrix
                            .clock
                            .set_external_beat(None, Instant::now());
                    }

                    // Sample the modulation matrix and derive modulated copies.
                    // Base values (what the UI edits) stay untouched; LFOs
                    // breathe around them.
                    self.mod_matrix.update(Instant::now());
                    self.release_quantized_actions_on_downbeat();

                    // Patch morph: while A and B are both set, the crossfader
                    // (plus any "morph" routings — an LFO can sweep worlds)
                    // writes the base parameters as their interpolation. The
                    // matrix then breathes on top of the morphed bases.
                    if self.morph.active() {
                        let base_t = self.morph.settle_glide_at(self.mod_matrix.current_beat);
                        let t_eff =
                            (base_t + self.mod_matrix.target_offset("morph")).clamp(0.0, 1.0);
                        self.morph.apply(
                            t_eff,
                            &mut self.master_effects,
                            &mut self.ntsc_params,
                            &mut self.temporal_params,
                            &mut self.layers,
                        );
                    }

                    // Advance live sources only after this frame's audio,
                    // MIDI, slew, beat latch, and morph state are current.
                    // This gives transport the same modulation phase used by
                    // rendering and by the deterministic exporter.
                    let renderer = self.renderer.as_ref().unwrap();
                    let matrix = &self.mod_matrix;
                    for (i, layer) in self.layers.iter_mut().enumerate() {
                        let speed = matrix
                            .modulate_layer_full(i, &layer.effects, layer.opacity, layer.speed)
                            .speed;
                        let playing = !self.master_paused && !layer.paused;

                        if layer.is_video() {
                            // The decoder seeds frame zero during open and
                            // publishes one latest-only result thereafter.
                            // Harvest before pause gating so a recalled paused
                            // layer displays a defined first frame, matching
                            // deterministic export.
                            match layer.take_ready_video_frame() {
                                Ok(Some(frame)) => layer.upload_frame(&renderer.queue, &frame),
                                Ok(None) => {}
                                Err(error) => log::error!(
                                    "Layer '{}' decoder failed: {error}",
                                    layer.filename
                                ),
                            }
                            if !playing {
                                layer.reset_transport_timing();
                                continue;
                            }
                            if let Err(error) = layer.request_due_video_frames(speed) {
                                log::error!("Layer '{}' decoder failed: {error}", layer.filename);
                            }
                            continue;
                        }

                        if !playing {
                            layer.reset_transport_timing();
                            continue;
                        }

                        // A real-time Spout layer owns one overwrite slot;
                        // pause holds the last frame and video pacing does not
                        // apply.
                        if let Some(frame) = layer.try_spout_frame() {
                            if let Err(error) =
                                layer.upload_spout_frame(&renderer.device, &renderer.queue, frame)
                            {
                                log::error!(
                                    "Layer '{}' rejected a Spout frame: {error}",
                                    layer.filename
                                );
                            }
                        }
                    }

                    let (mod_master, mod_ntsc, mod_temporal) = self.mod_matrix.modulate(
                        &self.master_effects,
                        &self.ntsc_params,
                        &self.temporal_params,
                    );

                    let renderer = self.renderer.as_mut().unwrap();
                    let mut encoder =
                        renderer
                            .device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("Frame Encoder"),
                            });
                    // Per-layer modulated copies: opacity crossfades, key
                    // thresholds breathing shapes in and out — bases untouched.
                    let layer_mods: Vec<(effects::EffectUniforms, f32)> = self
                        .layers
                        .iter()
                        .enumerate()
                        .map(|(i, layer)| {
                            let lm = self.mod_matrix.modulate_layer_full(
                                i,
                                &layer.effects,
                                layer.opacity,
                                layer.speed,
                            );
                            (lm.effects, lm.opacity)
                        })
                        .collect();

                    renderer.render_layers(&mut encoder, &self.layers, &layer_mods);
                    renderer.render_master_effects(&mut encoder, &mod_master);
                    // Temporal effects (feedback/slit-scan) + history recording.
                    renderer.render_temporal_with_dt(&mut encoder, &mod_temporal, frame_delta);

                    // Commit the raw GPU render before any CPU-processed
                    // queue write. A later command buffer would otherwise
                    // overwrite the queued NTSC upload.
                    renderer.queue.submit(std::iter::once(encoder.finish()));
                    encoder =
                        renderer
                            .device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("Post-Process Encoder"),
                            });

                    // Sync the Spout output worker with its toggle.
                    if self.spout_enabled && !self.spout.is_running() {
                        self.spout.start();
                    } else if !self.spout_enabled && self.spout.is_running() {
                        self.spout.stop();
                    }
                    if self.blackout
                        && self.spout_enabled
                        && self.spout.is_running()
                        && !self.spout.black_delivered(self.visual_epoch)
                    {
                        self.spout.cut_to_black(
                            renderer.output_width,
                            renderer.output_height,
                            self.visual_epoch,
                        );
                    }

                    // NTSC/VHS post-process — fully asynchronous. The render
                    // thread never waits on the GPU readback or the CPU effect;
                    // the displayed NTSC image trails the live composite by
                    // ~2 frames, which is invisible for a VHS look. Spout
                    // shares the same readback pipeline: it receives either
                    // the raw composite or, when NTSC is on, the processed
                    // frames — always what the audience sees.
                    // Blackout bypasses the stylized post-process: VHS snow
                    // must never turn an emergency cut into visible texture.
                    let ntsc_enabled = mod_ntsc.enabled && !self.blackout;
                    if ntsc_enabled != self.ntsc_pipeline_enabled {
                        // Changing the CPU post-process mode invalidates every
                        // delayed raw/processed frame from the previous mode.
                        self.ntsc_pipeline_enabled = ntsc_enabled;
                        self.visual_epoch = self.visual_epoch.saturating_add(1);
                        self.ntsc_presented = None;
                        // Entering blackout while NTSC was active advances the
                        // visual epoch here after the first cut request. Replace
                        // it immediately with a cut carrying the final epoch.
                        if self.blackout && self.spout_enabled && self.spout.is_running() {
                            self.spout.cut_to_black(
                                renderer.output_width,
                                renderer.output_height,
                                self.visual_epoch,
                            );
                        }
                    }
                    let spout_active = self.spout_enabled && self.spout.is_running();
                    // Harvest a completed raw readback. The generation tag
                    // rejects both pre-blackout content and delayed blackout
                    // frames that finish after the cut is released.
                    if let Some(frame) = renderer.poll_readback() {
                        if frame.epoch != self.visual_epoch {
                            // Stale visual generation; deliberately discarded.
                        } else if self.blackout {
                            if spout_active {
                                self.spout.try_submit(
                                    frame.pixels,
                                    renderer.output_width,
                                    renderer.output_height,
                                    frame.epoch,
                                );
                            }
                        } else if ntsc_enabled {
                            self.ntsc_worker.try_submit(
                                frame.pixels,
                                renderer.output_width,
                                renderer.output_height,
                                mod_ntsc.clone(),
                                frame.epoch,
                            );
                        } else if spout_active {
                            self.spout.try_submit(
                                frame.pixels,
                                renderer.output_width,
                                renderer.output_height,
                                frame.epoch,
                            );
                        }
                        // Otherwise a stale readback from a just-disabled
                        // feature; dropped.
                    }

                    // Retain the newest processed frame between CPU
                    // completions so NTSC does not alternate with raw frames.
                    if let Some(processed) = self.ntsc_worker.try_recv() {
                        if processed.epoch == self.visual_epoch && ntsc_enabled {
                            if spout_active {
                                self.spout.try_submit(
                                    processed.pixels.clone(),
                                    renderer.output_width,
                                    renderer.output_height,
                                    processed.epoch,
                                );
                            }
                            self.ntsc_presented = Some((processed.epoch, processed.pixels));
                        }
                    }

                    // Read back the clean raw composite before overlaying the
                    // delayed NTSC result, preventing recursive reprocessing.
                    let need_raw_readback = !self.blackout && (ntsc_enabled || spout_active);
                    if need_raw_readback {
                        let slot = renderer.begin_readback(&mut encoder, self.visual_epoch);
                        renderer.queue.submit(std::iter::once(encoder.finish()));
                        if let Some(idx) = slot {
                            renderer.map_readback(idx);
                        }
                        encoder = renderer.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("Audience Encoder"),
                            },
                        );
                    }

                    if ntsc_enabled {
                        if let Some((epoch, pixels)) = self.ntsc_presented.as_ref() {
                            if *epoch == self.visual_epoch {
                                // Ordered after submitted raw render/readback
                                // and before all audience presentation passes.
                                renderer.write_composite(pixels);
                            }
                        }
                    }

                    // Absolute final image operation: the emergency cut wins
                    // over every delayed post-process result.
                    if self.blackout {
                        renderer.clear_composite(&mut encoder);
                        renderer.queue.submit(std::iter::once(encoder.finish()));
                        encoder = renderer.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("Blackout Audience Encoder"),
                            },
                        );
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

                    if !preview_surface_available {
                        for id in &full_output.textures_delta.free {
                            egui_renderer.free_texture(id);
                        }
                        let output_present = renderer.render_output(&mut encoder);
                        renderer.queue.submit(std::iter::once(encoder.finish()));
                        if let Some(t) = output_present {
                            t.present();
                        }
                        if let Some(w) = &self.window {
                            w.request_redraw();
                        }
                        return;
                    }

                    let surface_texture = match renderer.surface.get_current_texture() {
                        wgpu::CurrentSurfaceTexture::Success(t)
                        | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
                        wgpu::CurrentSurfaceTexture::Outdated
                        | wgpu::CurrentSurfaceTexture::Lost => {
                            // The engine frame (including temporal history and
                            // dedicated output) is independent of the preview
                            // swapchain. Commit it before repairing the preview
                            // surface so CPU/GPU history indices cannot diverge.
                            for id in &full_output.textures_delta.free {
                                egui_renderer.free_texture(id);
                            }
                            let output_present = renderer.render_output(&mut encoder);
                            renderer.queue.submit(std::iter::once(encoder.finish()));
                            if let Some(texture) = output_present {
                                texture.present();
                            }
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
                            for id in &full_output.textures_delta.free {
                                egui_renderer.free_texture(id);
                            }
                            let output_present = renderer.render_output(&mut encoder);
                            renderer.queue.submit(std::iter::once(encoder.finish()));
                            if let Some(texture) = output_present {
                                texture.present();
                            }
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
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
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
                                })],
                                depth_stencil_attachment: None,
                                ..Default::default()
                            })
                            .forget_lifetime();

                        egui_renderer.render(&mut render_pass, &tris, &screen_desc);
                    }

                    for id in &full_output.textures_delta.free {
                        egui_renderer.free_texture(id);
                    }

                    // Blit the final composite to the fullscreen output
                    // window (if open) in the same submission.
                    let output_present = renderer.render_output(&mut encoder);

                    renderer.queue.submit(std::iter::once(encoder.finish()));
                    surface_texture.present();
                    if let Some(t) = output_present {
                        t.present();
                    }
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            _ => {}
        }
    }
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
                fonts
                    .font_data
                    .insert(name.to_owned(), Arc::new(egui::FontData::from_owned(bytes)));
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
            // Default library: ./videos/ — created if absent, so uploads
            // and drag-drop always have a home even on a bare launch.
            let default_lib = PathBuf::from("videos");
            if !default_lib.is_dir() {
                if let Err(e) = std::fs::create_dir_all(&default_lib) {
                    log::warn!("Could not create default library folder: {e}");
                }
            }
            if default_lib.is_dir() {
                (None, Some(default_lib))
            } else {
                (None, None)
            }
        }
    };

    // Start web control panel server
    let web_state = match WebState::new() {
        Ok(state) => state,
        Err(error) => {
            log::error!("Cannot start securely without an OS-random control token: {error}");
            return;
        }
    };
    let url = web::server::spawn(web_state.clone(), 3030);
    log::info!("Opening control panel: {}", url);
    let _ = open::that(&url);

    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(initial_video, library_folder, web_state);
    event_loop.run_app(&mut app).unwrap();
}

#[cfg(test)]
mod app_state_tests {
    use super::*;

    #[test]
    fn patch_generation_discards_old_latches_and_advances_epochs() {
        let web_state = WebState::new().expect("test token");
        let mut app = App::new(None, None, web_state);
        app.mod_matrix.current_beat = 9.25;
        app.quantized_actions.push(web::state::WebAction::SetParam {
            param: "brightness".to_string(),
            value: serde_json::json!(0.75),
        });
        app.web_state
            .actions
            .blocking_lock()
            .push(web::state::WebAction::RemoveLayer {
                index: 0,
                layer_id: None,
            });
        app.quantized_bar = Some(0);
        app.mod_matrix.midi_learn = Some(2);
        app.ntsc_presented = Some((7, vec![255; 4]));
        let old_stack_revision = app.layer_stack_revision;
        let old_visual_epoch = app.visual_epoch;

        app.reset_patch_generation();

        assert!(app.quantized_actions.is_empty());
        assert!(app.mod_matrix.midi_learn.is_none());
        assert!(app.web_state.actions.blocking_lock().is_empty());
        assert_eq!(app.quantized_bar, Some(2));
        assert!(app.ntsc_presented.is_none());
        assert_ne!(app.layer_stack_revision, old_stack_revision);
        assert_ne!(app.visual_epoch, old_visual_epoch);
    }

    #[test]
    fn explicit_blackout_setter_is_idempotent() {
        let web_state = WebState::new().expect("test token");
        let mut app = App::new(None, None, web_state);

        app.set_blackout(true);
        let epoch = app.visual_epoch;
        app.set_blackout(true);

        assert!(app.blackout);
        assert_eq!(app.visual_epoch, epoch);
    }

    #[test]
    fn removing_an_earlier_layer_preserves_selected_identity() {
        assert_eq!(selected_layer_after_remove(Some(2), 0, 2), Some(1));
        assert_eq!(selected_layer_after_remove(Some(0), 0, 2), Some(0));
        assert_eq!(selected_layer_after_remove(Some(2), 2, 2), Some(1));
        assert_eq!(selected_layer_after_remove(Some(0), 0, 0), None);
    }
}
