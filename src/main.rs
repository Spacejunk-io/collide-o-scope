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
mod procedural;
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
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Fullscreen, Window, WindowAttributes, WindowId};

use input::{apply_action, map_key, ControlFlow};
use layers::{is_still_image_file, is_supported_visual_file, spout_sender_from_source_path, Layer};
use renderer::Renderer;
use web::state::WebState;

const TARGET_FPS: u64 = 30;
const FRAME_DURATION: Duration = Duration::from_millis(1000 / TARGET_FPS);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputWindowCommand {
    Set(bool),
    Toggle,
}

/// A dedicated swapchain is useful only when it can live on a genuinely
/// separate display. Creating a second borderless swapchain on the same HDR
/// monitor is both redundant and fragile on some Windows display drivers.
const fn use_dedicated_output(main_monitor_is_known: bool, has_distinct_monitor: bool) -> bool {
    main_monitor_is_known && has_distinct_monitor
}

const fn resolve_output_window_command(current: bool, command: OutputWindowCommand) -> bool {
    match command {
        OutputWindowCommand::Set(enabled) => enabled,
        OutputWindowCommand::Toggle => !current,
    }
}

fn is_discrete_window_key(key: PhysicalKey) -> bool {
    matches!(
        key,
        PhysicalKey::Code(KeyCode::Escape)
            | PhysicalKey::Code(KeyCode::KeyF)
            | PhysicalKey::Code(KeyCode::KeyO)
    )
}

fn ignore_discrete_window_key_repeat(key: PhysicalKey, repeat: bool) -> bool {
    repeat && is_discrete_window_key(key)
}

const fn show_editor_panel(output_on_main: bool, editor_active: bool) -> bool {
    editor_active && !output_on_main
}

#[derive(Debug)]
struct OutputRenderFailure {
    message: String,
    device_lost: bool,
}

/// Encode the dedicated output and classify failures before callers submit
/// the frame encoder. A surface-only failure closes that output and leaves the
/// main preview alive; device loss is terminal because every GPU handle owned
/// by the application has become invalid.
fn render_output_checked(
    renderer: &mut Renderer,
    encoder: &mut wgpu::CommandEncoder,
) -> Result<Option<wgpu::SurfaceTexture>, OutputRenderFailure> {
    let output = renderer.render_output(encoder).map_err(|message| {
        let device_lost = renderer.device_error().is_some();
        renderer.close_output();
        OutputRenderFailure {
            message,
            device_lost,
        }
    })?;
    // Never turn a successful call into a health error here: if `output` owns
    // a texture, it must escape this helper and be presented. Every caller
    // checks the health latch after consuming all acquired textures.
    Ok(output)
}

/// Stop at the first GPU submission/poll boundary that reports device loss.
/// In particular, do not let the same frame continue into wgpu helpers that
/// allocate mapped-at-creation buffers, because an invalid fallback handle
/// would turn the original device loss into a misleading mapping panic.
fn exit_on_renderer_device_loss(
    renderer: &Renderer,
    output_error: &mut String,
    event_loop: &ActiveEventLoop,
) -> bool {
    let Some(error) = renderer.device_error() else {
        return false;
    };
    *output_error = format!("GPU device lost: {error}. Restart collide-o-scope.");
    log::error!("{output_error}");
    event_loop.exit();
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LiveNtscPath {
    Disabled,
    LegacyGlobal,
    SelectivePerLayer,
}

fn is_selective_path_edge(previous: LiveNtscPath, requested: LiveNtscPath) -> bool {
    previous != requested
        && (previous == LiveNtscPath::SelectivePerLayer
            || requested == LiveNtscPath::SelectivePerLayer)
}

fn selective_generation_boundary(path_changed: bool, topology_changed: bool) -> bool {
    path_changed || topology_changed
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HeldAudienceAction {
    Capture,
    Restore,
    Keep,
}

fn held_audience_action(
    was_holding: bool,
    blackout: bool,
    snapshot_valid: bool,
) -> HeldAudienceAction {
    if !blackout && snapshot_valid {
        HeldAudienceAction::Restore
    } else if !was_holding && !snapshot_valid {
        HeldAudienceAction::Capture
    } else {
        HeldAudienceAction::Keep
    }
}

/// Determine the copy needed at an actual audience blackout edge. Capture is
/// independent of the current VHS path so controls may change while the cut is
/// active. Restore is required only when Pause promises the exact prior image;
/// a running program will render a fresh authoritative frame instead.
fn blackout_audience_edge_action(
    blackout: bool,
    blackout_presented: bool,
    program_transport_paused: bool,
    snapshot_valid: bool,
) -> Option<HeldAudienceAction> {
    if blackout && !blackout_presented {
        Some(if snapshot_valid {
            HeldAudienceAction::Keep
        } else {
            HeldAudienceAction::Capture
        })
    } else if !blackout && blackout_presented && program_transport_paused && snapshot_valid {
        Some(HeldAudienceAction::Restore)
    } else {
        None
    }
}

fn selective_ntsc_topology_signature(
    layers: &[Layer],
    mods: &[(effects::EffectUniforms, f32)],
) -> u64 {
    // Explicit FNV-1a keeps this stable across process/library versions. The
    // signature contains only topology/membership facts, not continuously
    // animated values, so normal modulation does not invalidate every job.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut mix = |value: u64| {
        hash ^= value;
        hash = hash.wrapping_mul(0x100_0000_01b3);
    };
    mix(layers.len() as u64);
    for (index, layer) in layers.iter().enumerate() {
        mix(index as u64);
        mix(layer.layer_id());
        mix(layer.visible as u64);
        mix(layer.bypass_master_fx as u64);
        let contributes = mods
            .get(index)
            .is_some_and(|(_, opacity)| opacity.is_finite() && *opacity > 0.0);
        mix(contributes as u64);
    }
    hash
}

#[cfg(test)]
fn selective_topology_generation_after_signature(
    prior_signature: u64,
    current_signature: u64,
    prior_generation: u64,
) -> u64 {
    if prior_signature == current_signature {
        prior_generation
    } else {
        prior_generation.wrapping_add(1).max(1)
    }
}

fn selective_ntsc_transform_fingerprint(layer_id: u64) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in layer_id.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

fn live_selective_ntsc_plan(
    generation: ntsc::SelectiveNtscGeneration,
    metadata: ntsc::NtscFrameMetadata,
    layers: &[Layer],
    mods: &[(effects::EffectUniforms, f32)],
) -> Option<ntsc::SelectiveNtscPlan> {
    ntsc::plan_selective_ntsc(
        generation,
        metadata,
        layers.iter().zip(mods).map(|(layer, (_effects, opacity))| {
            ntsc::SelectiveNtscLayerDescriptor {
                layer_id: layer.layer_id(),
                visible: layer.visible,
                bypass_master_fx: layer.bypass_master_fx,
                opacity: *opacity,
                blend_mode: layer.blend_mode.as_u32(),
                transform_fingerprint: selective_ntsc_transform_fingerprint(layer.layer_id()),
            }
        }),
    )
}

fn selective_spout_sample_is_eligible(
    frame_epoch: u64,
    sample: Option<ntsc::SelectiveNtscGeneration>,
    current: ntsc::SelectiveNtscGeneration,
    spout_active: bool,
    blackout: bool,
) -> bool {
    spout_active
        && !blackout
        && frame_epoch == current.visual_epoch
        && sample.is_some_and(|sample| ntsc::selective_generation_compatible(sample, current))
}

/// A selective rebuild has no valid replacement sample yet. While transport
/// is playing, clearing prevents a legacy/global frame from flashing under
/// the new routing. While paused, the audience contract is stronger: retain
/// the already materialized final image until Resume produces the first exact
/// selective sample, which then replaces it in one Temporal+opaque command
/// sequence.
fn selective_rebuild_should_clear_audience(program_transport_paused: bool) -> bool {
    !program_transport_paused
}

fn direct_path_may_replace_selective_hold(
    program_transport_paused: bool,
    selective_transition_holding: bool,
) -> bool {
    !program_transport_paused || !selective_transition_holding
}

fn rendered_audience_may_discard_blackout_snapshot(
    program_transport_paused: bool,
    blackout: bool,
) -> bool {
    !program_transport_paused && !blackout
}

fn raw_audience_readback_required(
    blackout: bool,
    selective_transition_holding: bool,
    path: LiveNtscPath,
    spout_active: bool,
) -> bool {
    !blackout
        && !selective_transition_holding
        && (path == LiveNtscPath::LegacyGlobal || (path == LiveNtscPath::Disabled && spout_active))
}

/// Piece-local time advances only while the master transport is playing.
/// Keeping this independent from wall time makes every time-authored visual
/// (shader animation, temporal accumulation, NTSC phase, and imported-audio
/// analysis) hold exactly during a global pause and resume without catch-up.
#[derive(Debug, Default)]
struct ProgramClock {
    elapsed: Duration,
}

impl ProgramClock {
    fn reset(&mut self) {
        self.elapsed = Duration::ZERO;
    }

    fn tick(&mut self, wall_delta: Duration, paused: bool) -> (Duration, f32) {
        let program_delta = if paused { Duration::ZERO } else { wall_delta };
        self.elapsed = self.elapsed.saturating_add(program_delta);
        (self.elapsed, program_delta.as_secs_f32())
    }
}

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

/// Resolve a browser-authored modulation target against immutable layer
/// identities. Positional `layerN_*` names remain the shader/runtime format,
/// but a supplied ID is authoritative and is translated to the layer's
/// current position before the target is committed.
fn resolve_routing_target_for_layer_ids(
    proposed: &str,
    target_layer_id: &Option<String>,
    layer_stack_revision: Option<u64>,
    layer_ids: impl IntoIterator<Item = u64>,
) -> Option<String> {
    let canonical = modulation::canonical_target(proposed);
    let layer_target = canonical
        .strip_prefix("layer")
        .and_then(|rest| rest.split_once('_'))
        .and_then(|(number, suffix)| {
            number
                .parse::<usize>()
                .ok()
                .filter(|layer| (1..=modulation::MAX_MOD_LAYERS).contains(layer))
                .map(|_| suffix)
        });

    match target_layer_id {
        Some(id) => {
            let suffix = layer_target?;
            let stable_id = id.parse::<u64>().ok().filter(|id| *id != 0)?;
            let current_index = layer_ids
                .into_iter()
                .position(|candidate| candidate == stable_id)?;
            if current_index >= modulation::MAX_MOD_LAYERS {
                return None;
            }
            let resolved = format!("layer{}_{suffix}", current_index + 1);
            modulation::is_valid_target(&resolved).then_some(resolved)
        }
        None => {
            // Revision metadata without a stable identity cannot make a
            // positional target safe. Old clients omit both fields and keep
            // their legacy behavior.
            if layer_stack_revision.is_some() || !modulation::is_valid_target(canonical.as_ref()) {
                return None;
            }
            Some(canonical.into_owned())
        }
    }
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
    program_clock: ProgramClock,
    modifiers: ModifiersState,
    // Library
    library_folder: Option<PathBuf>,
    library_files: Vec<PathBuf>,
    audio_library_files: Vec<PathBuf>,
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
    // Last globally processed NTSC frame presented while the next CPU job runs.
    ntsc_presented: Option<(u64, Vec<u8>)>,
    ntsc_pipeline_path: LiveNtscPath,
    // Allocated only when a contributing bypass layer requires per-layer VHS.
    selective_ntsc_worker: Option<ntsc::SelectiveNtscWorker>,
    // Program time accumulated while the asynchronous selective worker is
    // producing the next clean pre-temporal composite.
    selective_temporal_debt: f32,
    selective_topology_signature: u64,
    selective_topology_generation: u64,
    selective_sample_sequence: u64,
    /// Renderer/preflight failures are published through the existing VHS
    /// status surface. The audience retains its last exact frame meanwhile.
    selective_ntsc_runtime_error: String,
    /// A selective path/topology edge committed while paused retains slot2.
    /// Entering selective holds through resumed warm-up until the first exact
    /// accepted sample; leaving selective releases on the first resumed direct
    /// render. This also forbids misclassifying the held pixels as a new raw
    /// legacy-Spout/NTSC sample.
    selective_transition_holding: bool,
    /// True after slot2 has been copied to the dedicated held-audience GPU
    /// texture. It survives blackout until a real resumed path replaces it.
    selective_hold_snapshot_valid: bool,
    /// Spout's non-black generation barrier and the exact held-image readback
    /// are tracked separately so a temporarily unavailable GPU readback slot
    /// can retry without repeatedly flushing a valid same-epoch mailbox item.
    selective_hold_spout_barrier_epoch: Option<u64>,
    selective_hold_spout_readback_epoch: Option<u64>,
    // Modulation matrix (BPM clock, LFOs, routings)
    mod_matrix: modulation::ModMatrix,
    // Patch morphing crossfader (A/B slots + t)
    morph: morph::Morph,
    // Blackout: output cut to black (B key / panel button)
    blackout: bool,
    /// Whether the last audience presentation actually contained the absolute
    /// blackout. This remains distinct from the requested flag so a paused
    /// entry can snapshot slot2 before the first clear and restore it later.
    blackout_presented: bool,
    // Advanced at every blackout edge so delayed GPU/CPU frames are rejected.
    visual_epoch: u64,
    // Audio input analysis (modulation source)
    audio: audio::AudioAnalyzer,
    audio_clip_loader: audio::AudioClipLoader,
    audio_clip: Option<audio::AudioClip>,
    audio_clip_spectrum: [f32; audio::AUDIO_SPECTRUM_BINS],
    audio_clip_error: String,
    /// Hold piece time while an enabled deterministic clip is decoding. This
    /// prevents machine-dependent load latency from changing the first routed
    /// frames or temporal history after patch recall/source selection.
    audio_clip_blocks_program: bool,
    // MIDI input (modulation source)
    midi: midi::MidiEngine,
    // Temporal effects (feedback trails, slit-scan)
    temporal_params: effects::params::TemporalParams,
    // Spout texture-sharing output
    spout: spout_out::SpoutOut,
    spout_enabled: bool,
    /// True when audience Output reuses the main window's existing surface on
    /// a single-monitor system instead of creating a second same-display
    /// swapchain.
    output_on_main: bool,
    output_error: String,
    // Web control panel
    web_state: Arc<WebState>,
    patch_collector: procedural::PatchCollector,
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
        let audio_library_files = library_folder
            .as_ref()
            .map(scan_audio_folder)
            .unwrap_or_default();

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
            program_clock: ProgramClock::default(),
            modifiers: ModifiersState::empty(),
            library_folder,
            library_files,
            audio_library_files,
            yaml_editor: patch::editor::EditorState::default(),
            egui_ctx: egui::Context::default(),
            egui_winit: None,
            egui_renderer: None,
            video_egui_texture_id: None,
            ntsc_params: ntsc::NtscParams::default(),
            ntsc_worker: ntsc::NtscWorker::new(),
            ntsc_presented: None,
            ntsc_pipeline_path: LiveNtscPath::Disabled,
            selective_ntsc_worker: None,
            selective_temporal_debt: 0.0,
            selective_topology_signature: 0,
            selective_topology_generation: 1,
            selective_sample_sequence: 0,
            selective_ntsc_runtime_error: String::new(),
            selective_transition_holding: false,
            selective_hold_snapshot_valid: false,
            selective_hold_spout_barrier_epoch: None,
            selective_hold_spout_readback_epoch: None,
            mod_matrix: modulation::ModMatrix::new(),
            morph: morph::Morph::default(),
            blackout: false,
            blackout_presented: false,
            visual_epoch: 0,
            audio: audio::AudioAnalyzer::new(),
            audio_clip_loader: audio::AudioClipLoader::new(),
            audio_clip: None,
            audio_clip_spectrum: [0.0; audio::AUDIO_SPECTRUM_BINS],
            audio_clip_error: String::new(),
            audio_clip_blocks_program: false,
            midi: midi::MidiEngine::new(),
            temporal_params: effects::params::TemporalParams::default(),
            spout: spout_out::SpoutOut::new(),
            spout_enabled: false,
            output_on_main: false,
            output_error: String::new(),
            web_state,
            patch_collector: procedural::PatchCollector::new(),
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
                // Appending leaves every captured position valid. The new
                // source is deliberately untouched by an existing morph.
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
                // Appending leaves every captured position valid.
            }
            Err(error) => log::error!("Failed to create Spout input layer: {error}"),
        }
    }

    fn bump_layer_stack_revision(&mut self) {
        self.layer_stack_revision = self.layer_stack_revision.wrapping_add(1).max(1);
        // Legacy clients omitted a stack revision. Do not let one of their
        // already-latched captures bind itself to a later topology.
        self.quantized_actions
            .retain(|action| !matches!(action, web::state::WebAction::MorphCapture { .. }));
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

    /// Resolve a browser route selector with the same stale-ID rejection used
    /// for layers. A supplied ID is authoritative; index is legacy fallback.
    fn resolve_routing_index(&self, index: usize, route_id: &Option<String>) -> Option<usize> {
        match route_id.as_deref().filter(|id| !id.is_empty()) {
            Some(id) => {
                let id = id.parse::<u64>().ok()?;
                self.mod_matrix
                    .routings
                    .iter()
                    .position(|routing| routing.route_id() == id)
            }
            None => (index < self.mod_matrix.routings.len()).then_some(index),
        }
    }

    fn reset_patch_generation(&mut self) {
        // Animated effects use a piece-local clock. A recalled patch therefore
        // begins at the same t=0 phase as its deterministic offline export.
        self.program_clock.reset();
        self.quantized_actions.clear();
        // Patch construction can block on multiple decoders. Do not interpret
        // that wall time as modulation spring/slew motion on the first frame
        // after the atomic commit; clock phase and current beat stay intact.
        self.mod_matrix.reset_update_timing();
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
        self.selective_temporal_debt = 0.0;
        self.selective_transition_holding = false;
        self.selective_hold_snapshot_valid = false;
        self.selective_hold_spout_barrier_epoch = None;
        self.selective_hold_spout_readback_epoch = None;
        // A recalled blackout starts a new program generation. Never snapshot
        // or restore pixels belonging to the previously loaded patch.
        self.blackout_presented = self.blackout;
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.reset_visual_generation();
        }
    }

    fn capture_current_patch(&self) -> patch::PatchState {
        patch::PatchState::capture(
            &self.master_effects,
            &self.layers,
            &self.ntsc_params,
            &self.mod_matrix,
            &self.temporal_params,
            self.master_paused,
            &self.morph,
        )
    }

    /// Open the separate-monitor audience window. The caller has already
    /// proved `target_monitor` differs from the main window's monitor.
    fn open_dedicated_output_window(
        &mut self,
        event_loop: &ActiveEventLoop,
        target_monitor: winit::monitor::MonitorHandle,
    ) {
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };

        let attrs = WindowAttributes::default()
            .with_title("collide-o-scope — output")
            .with_fullscreen(Some(Fullscreen::Borderless(Some(target_monitor))));

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

    fn output_window_open(&self) -> bool {
        self.output_on_main || self.renderer.as_ref().is_some_and(Renderer::has_output)
    }

    /// Close either form of audience Output. On one monitor this restores the
    /// ordinary main window; on two or more it drops only the dedicated
    /// projector window and its surface.
    fn close_output_window(&mut self) {
        if self.output_on_main {
            if let Some(window) = self.window.as_ref() {
                window.set_fullscreen(None);
                window.set_cursor_visible(true);
            }
            self.output_on_main = false;
        }
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.close_output();
        }
        self.output_error.clear();
        log::info!("Output window closed");
    }

    fn toggle_preview_fullscreen(&mut self) {
        // In single-monitor Output mode the preview and audience are the same
        // window, so F is also an authoritative Output close.
        if self.output_on_main {
            self.close_output_window();
            return;
        }
        if let Some(window) = self.window.as_ref() {
            let fullscreen = if window.fullscreen().is_some() {
                None
            } else {
                Some(Fullscreen::Borderless(None))
            };
            window.set_fullscreen(fullscreen);
        }
    }

    /// Set audience Output explicitly. A real second monitor receives a
    /// dedicated fullscreen surface so the performer keeps the editor. With
    /// only one monitor, reuse the main window's already-proven surface and
    /// make that window borderless instead of creating a second HDR swapchain
    /// on the same display.
    fn set_output_window(&mut self, event_loop: &ActiveEventLoop, enabled: bool) {
        if self.output_window_open() == enabled {
            return;
        }
        if !enabled {
            self.close_output_window();
            return;
        }

        let main_monitor = self
            .window
            .as_ref()
            .and_then(|window| window.current_monitor());
        // A transiently unknown main monitor is not proof that another one is
        // distinct. Fail closed to the existing surface in that case.
        let distinct_monitor = main_monitor.as_ref().and_then(|main| {
            event_loop
                .available_monitors()
                .find(|monitor| monitor != main)
        });

        if !use_dedicated_output(main_monitor.is_some(), distinct_monitor.is_some()) {
            let Some(window) = self.window.as_ref() else {
                self.output_error = "main output window is not ready".to_string();
                return;
            };
            window.set_fullscreen(Some(Fullscreen::Borderless(main_monitor)));
            window.set_cursor_visible(false);
            self.output_on_main = true;
            self.output_error.clear();
            log::info!("Output opened on the main window (single-monitor mode)");
            return;
        }

        self.open_dedicated_output_window(event_loop, distinct_monitor.unwrap());
    }

    fn apply_output_window_command(
        &mut self,
        event_loop: &ActiveEventLoop,
        command: OutputWindowCommand,
    ) {
        let enabled = resolve_output_window_command(self.output_window_open(), command);
        self.set_output_window(event_loop, enabled);
        self.exit_if_device_lost(event_loop);
    }

    /// Device loss invalidates every resource owned by this Renderer. Stop at
    /// the event-loop boundary before another mapped-at-creation helper can
    /// touch an invalid handle and panic with an unrelated buffer label.
    fn exit_if_device_lost(&mut self, event_loop: &ActiveEventLoop) -> bool {
        let Some(renderer) = self.renderer.as_ref() else {
            return false;
        };
        exit_on_renderer_device_loss(renderer, &mut self.output_error, event_loop)
    }

    fn set_library_folder(&mut self, folder: PathBuf) {
        self.library_files = scan_folder(&folder);
        self.audio_library_files = scan_audio_folder(&folder);
        if let Ok(mut lf) = self.web_state.library_folder.write() {
            *lf = Some(folder.clone());
        }
        self.library_folder = Some(folder);
    }

    fn resolve_audio_clip_path_near(
        &self,
        source: &str,
        patch_dir: Option<&std::path::Path>,
    ) -> Option<PathBuf> {
        let persisted = PathBuf::from(source);
        if persisted.is_file() && audio::is_supported_audio_file(&persisted) {
            return Some(persisted);
        }
        if let Some(patch_dir) = patch_dir {
            let candidate = patch_dir.join(&persisted);
            if candidate.is_file() && audio::is_supported_audio_file(&candidate) {
                return Some(candidate);
            }
        }
        let basename = persisted.file_name()?;
        self.audio_library_files
            .iter()
            .find(|path| path.file_name().is_some_and(|name| name == basename))
            .cloned()
    }

    fn resolve_audio_clip_path(&self, source: &str) -> Option<PathBuf> {
        self.resolve_audio_clip_path_near(source, None)
    }

    fn selected_audio_clip_name(&self) -> String {
        PathBuf::from(&self.mod_matrix.audio_clip_path)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.mod_matrix.audio_clip_path.clone())
    }

    fn request_selected_audio_clip(&mut self) {
        let requested = self.mod_matrix.audio_clip_path.clone();
        let now = Instant::now();
        self.audio_clip = None;
        self.audio_clip_spectrum.fill(0.0);
        self.audio_clip_error.clear();
        if requested.is_empty() {
            self.audio_clip_loader.cancel();
            self.set_audio_clip_blocks_program_at(false, now);
            return;
        }
        match self.resolve_audio_clip_path(&requested) {
            Some(path) => {
                self.mod_matrix.audio_clip_path = path.to_string_lossy().into_owned();
                self.audio_clip_loader
                    .request(path.to_string_lossy().into_owned());
                self.set_audio_clip_blocks_program_at(self.mod_matrix.audio_enabled, now);
            }
            None => {
                self.audio_clip_loader.cancel();
                self.audio_clip_error = format!("audio clip not found: {requested}");
                self.set_audio_clip_blocks_program_at(false, now);
            }
        }
    }

    fn poll_audio_clip_loader(&mut self) -> bool {
        let Some(completion) = self.audio_clip_loader.poll() else {
            return false;
        };
        self.set_audio_clip_blocks_program_at(false, Instant::now());
        match completion.result {
            Ok(clip) => {
                self.audio_clip = Some(clip);
                self.audio_clip_spectrum.fill(0.0);
                self.audio_clip_error.clear();
            }
            Err(error) => {
                self.audio_clip = None;
                self.audio_clip_spectrum.fill(0.0);
                self.audio_clip_error = error;
            }
        }
        true
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
        if self.mod_matrix.audio_source_kind == modulation::AUDIO_SOURCE_FILE
            && !self.mod_matrix.audio_clip_path.is_empty()
        {
            if let Some(path) =
                self.resolve_audio_clip_path_near(&self.mod_matrix.audio_clip_path, Some(patch_dir))
            {
                self.mod_matrix.audio_clip_path = path.to_string_lossy().into_owned();
            }
        }
        let recalled_master_paused = patch.master_paused;
        // Patch modulation replacement may leave the old clock's paused flag
        // behind when loading a legacy patch with no modulation section.
        // Start from a running logical clock, then apply the recalled absolute
        // transport state at the common commit boundary below.
        self.mod_matrix.clock.set_paused(false, Instant::now());
        self.master_paused = false;
        self.audio_clip_blocks_program = false;
        // All decoders were opened sequentially, but playback begins at one
        // atomic commit boundary. Re-anchor every pacer to the same instant so
        // early layers do not begin with artificial catch-up debt.
        let commit_time = Instant::now();
        for layer in &mut rebuilt {
            layer.reset_transport_timing_at(commit_time);
        }
        self.mod_matrix
            .clock
            .set_paused(recalled_master_paused, commit_time);
        self.master_paused = recalled_master_paused;
        self.mod_matrix.reset_update_timing();
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
        if self.mod_matrix.audio_source_kind == modulation::AUDIO_SOURCE_FILE {
            self.request_selected_audio_clip();
        } else {
            self.audio_clip_loader.cancel();
            self.audio_clip = None;
            self.audio_clip_error.clear();
            self.set_audio_clip_blocks_program_at(false, Instant::now());
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
            WebAction::MorphCapture { slot, .. } => Some(format!("morph:capture:{slot}")),
            WebAction::MorphClear => Some("morph:clear".to_string()),
            _ => None,
        }
    }

    /// Output-window creation needs the event loop and therefore cannot be
    /// delegated to the plain state-action handler. Treat it as immediate even
    /// if an old or hand-authored client wrapped it in one or more latches.
    fn output_window_command(action: &web::state::WebAction) -> Option<OutputWindowCommand> {
        match action {
            web::state::WebAction::SetOutputWindow { enabled } => {
                Some(OutputWindowCommand::Set(*enabled))
            }
            web::state::WebAction::ToggleOutputWindow => Some(OutputWindowCommand::Toggle),
            web::state::WebAction::Quantized { inner } => Self::output_window_command(inner),
            _ => None,
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

    fn program_transport_paused(&self) -> bool {
        self.master_paused || self.audio_clip_blocks_program
    }

    /// Commit an effective program-pause edge. Pause causes remain separate so
    /// an asynchronous clip load never changes the user-facing master state.
    fn commit_program_transport_edge_at(&mut self, was_paused: bool, now: Instant) {
        let paused = self.program_transport_paused();
        if paused == was_paused {
            return;
        }
        self.mod_matrix.clock.set_paused(paused, now);
        self.mod_matrix.reset_update_timing();
        for layer in &mut self.layers {
            layer.reset_transport_timing_at(now);
        }
    }

    fn set_audio_clip_blocks_program_at(&mut self, blocked: bool, now: Instant) {
        if self.audio_clip_blocks_program == blocked {
            return;
        }
        let was_paused = self.program_transport_paused();
        self.audio_clip_blocks_program = blocked;
        self.commit_program_transport_edge_at(was_paused, now);
    }

    /// Absolute master transport setter used by every control surface. An
    /// already-applied value is a no-op; real effective edges re-anchor all
    /// decoder pacers and the modulation clock so neither repeated network
    /// messages nor time spent paused can create playback debt.
    fn set_master_paused_at(&mut self, paused: bool, now: Instant) {
        if self.master_paused == paused {
            return;
        }
        let was_paused = self.program_transport_paused();
        self.master_paused = paused;
        self.commit_program_transport_edge_at(was_paused, now);
    }

    fn set_master_paused(&mut self, paused: bool) {
        self.set_master_paused_at(paused, Instant::now());
    }

    /// Commit the exact effective morph world at the current authoritative
    /// beat. This is also the ordering boundary used before slot capture.
    fn materialize_morph_at_current_beat_with_offset(&mut self, morph_offset: f32) {
        if !self.morph.active() {
            return;
        }
        let base_t = self.morph.position_at_beat(self.mod_matrix.current_beat);
        let t = (base_t + morph_offset).clamp(0.0, 1.0);
        self.morph.apply(
            t,
            &mut self.master_effects,
            &mut self.ntsc_params,
            &mut self.temporal_params,
            &mut self.layers,
        );
    }

    fn materialize_morph_at_current_beat(&mut self) {
        let morph_offset = self.mod_matrix.frame().morph_offset();
        self.materialize_morph_at_current_beat_with_offset(morph_offset);
    }

    /// Restore the complete post-stack visual program to neutral while
    /// preserving layers, transport, BPM, and live input/device choices.
    /// Automation that could immediately overwrite the defaults is cleared.
    fn revert_master_visual_state(&mut self) {
        self.master_effects.reset();
        self.ntsc_params = ntsc::NtscParams::default();
        self.temporal_params = effects::params::TemporalParams::default();
        self.mod_matrix.reset();
        self.morph = morph::Morph::default();

        // A previously latched master command must not resurrect state on a
        // later downbeat. Pending layer edits remain valid and are preserved.
        self.quantized_actions.retain(|action| {
            !matches!(
                action,
                web::state::WebAction::SetParam { .. }
                    | web::state::WebAction::SetNtscParam { .. }
                    | web::state::WebAction::SetTemporal { .. }
                    | web::state::WebAction::SetMorph { .. }
                    | web::state::WebAction::SetMorphLaw { .. }
                    | web::state::WebAction::MorphCapture { .. }
                    | web::state::WebAction::MorphClear
                    | web::state::WebAction::MorphGlide { .. }
            )
        });

        // Invalidate temporal history and delayed NTSC/readback frames from
        // the pre-revert visual generation.
        self.visual_epoch = self.visual_epoch.wrapping_add(1);
        self.ntsc_presented = None;
        self.selective_temporal_debt = 0.0;
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.reset_visual_generation();
        }
    }

    fn set_blackout(&mut self, enabled: bool) {
        if self.blackout == enabled {
            return;
        }
        self.blackout = enabled;
        self.visual_epoch = self.visual_epoch.saturating_add(1);
        self.ntsc_presented = None;
        self.selective_temporal_debt = 0.0;

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
        // Capture observes every state transition before it, so it is both an
        // eligible latched command and an ordering barrier, never a value to
        // coalesce. Absolute controls may coalesce only after the latest such
        // barrier and are moved to the tail to preserve cross-control order.
        let is_capture = matches!(action, web::state::WebAction::MorphCapture { .. });
        let barrier = self
            .quantized_actions
            .iter()
            .rposition(|candidate| matches!(candidate, web::state::WebAction::MorphCapture { .. }))
            .map_or(0, |position| position + 1);
        let existing = (!is_capture)
            .then(|| {
                self.quantized_actions[barrier..]
                    .iter()
                    .rposition(|candidate| {
                        Self::quantized_action_key(candidate).as_deref() == Some(key.as_str())
                    })
                    .map(|position| barrier + position)
            })
            .flatten();
        if let Some(existing) = existing {
            self.quantized_actions.remove(existing);
            self.quantized_actions.push(action);
        } else if self.quantized_actions.len() < 256 {
            self.quantized_actions.push(action);
        }
    }

    fn release_quantized_actions_on_downbeat(&mut self) {
        let bar = (self.mod_matrix.current_beat / 4.0).floor() as i64;
        // A downbeat is a forward crossing. Tap/MIDI reanchors may move the
        // clock backwards, but must not flush work queued for the next bar.
        let crossed = self
            .quantized_bar
            .map(|previous| previous < bar)
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
            WebAction::QuickSavePatch => {
                let result = self
                    .patch_collector
                    .try_submit(self.capture_current_patch(), PathBuf::from("patches"));
                if result == procedural::CaptureSubmit::Busy {
                    log::warn!("Patch capture queue is busy; capture was not enqueued");
                }
            }
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
                    self.mod_matrix.remap_layer_targets_after_remove(index);
                    self.remap_quantized_layers_after_remove(index);
                    self.bump_layer_stack_revision();
                    self.morph.remap_layers_after_remove(index);
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
                        self.mod_matrix.remap_layer_targets_after_move(from, to);
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
                        self.morph.remap_layers_after_move(from, to);
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
                    self.layers[index].reset_transport_timing();
                    log::info!("Layer {index} paused → {}", self.layers[index].paused);
                }
            }
            WebAction::ToggleMasterPause => {
                self.set_master_paused(!self.master_paused);
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
                    if self.layers[index].paused != paused {
                        self.layers[index].paused = paused;
                        self.layers[index].reset_transport_timing();
                    }
                }
            }
            WebAction::SetMasterPaused { paused } => {
                self.set_master_paused(paused);
            }
            WebAction::SetBlackout { enabled } => self.set_blackout(enabled),
            WebAction::ResetFx => {
                self.revert_master_visual_state();
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
                    "key" => {
                        self.master_effects.key_mode = defaults.key_mode;
                        self.master_effects.key_color = defaults.key_color;
                        self.master_effects.key_threshold = defaults.key_threshold;
                        self.master_effects.key_softness = defaults.key_softness;
                        self.master_effects.key_tolerance = defaults.key_tolerance;
                    }
                    "motion" => {
                        self.master_effects.breathe_scale = defaults.breathe_scale;
                        self.master_effects.breathe_rotation = defaults.breathe_rotation;
                        self.master_effects.breathe_position = defaults.breathe_position;
                    }
                    "cellular" => {
                        self.master_effects.cellular_amount = defaults.cellular_amount;
                        self.master_effects.cellular_scale = defaults.cellular_scale;
                        self.master_effects.cellular_warp = defaults.cellular_warp;
                        self.master_effects.cellular_speed = defaults.cellular_speed;
                        self.master_effects.cellular_gap_amount = defaults.cellular_gap_amount;
                        self.master_effects.cellular_gap_threshold =
                            defaults.cellular_gap_threshold;
                        self.master_effects.cellular_gap_softness = defaults.cellular_gap_softness;
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
                self.mod_matrix.clock.set_bpm_at(value, Instant::now());
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
            WebAction::RemoveRouting { index, route_id } => {
                if let Some(index) = self.resolve_routing_index(index, &route_id) {
                    self.mod_matrix.remove_routing(index);
                }
            }
            WebAction::SetAudio { param, value } => {
                match param.as_str() {
                    "enabled" => {
                        if let Some(b) = value.as_bool() {
                            self.mod_matrix.audio_enabled = b;
                            let blocks_program = b
                                && self.mod_matrix.audio_source_kind
                                    == modulation::AUDIO_SOURCE_FILE
                                && self.audio_clip.is_none()
                                && self.audio_clip_loader.state()
                                    == audio::AudioClipLoadState::Loading;
                            self.set_audio_clip_blocks_program_at(blocks_program, Instant::now());
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
                    "source_kind" => {
                        if let Some(source) = value.as_str() {
                            let source = modulation::normalize_audio_source_kind(source);
                            self.mod_matrix.audio_source_kind = source.to_string();
                            if source == modulation::AUDIO_SOURCE_FILE {
                                if self.audio.is_running() {
                                    self.audio.stop();
                                }
                                self.request_selected_audio_clip();
                            } else {
                                self.audio_clip_loader.cancel();
                                self.audio_clip = None;
                                self.audio_clip_spectrum.fill(0.0);
                                self.audio_clip_error.clear();
                                self.set_audio_clip_blocks_program_at(false, Instant::now());
                            }
                        }
                    }
                    "clip" => {
                        if let Some(source) = value.as_str() {
                            self.mod_matrix.audio_clip_path = self
                                .resolve_audio_clip_path(source)
                                .map(|path| path.to_string_lossy().into_owned())
                                .unwrap_or_else(|| source.to_string());
                            if self.mod_matrix.audio_source_kind == modulation::AUDIO_SOURCE_FILE {
                                self.request_selected_audio_clip();
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
            // The web server consumes this connection-scoped declaration.
            // Keep a no-op arm for direct/legacy action injection.
            WebAction::GyroStream { .. } => {}
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
            WebAction::SetOutputWindow { .. } | WebAction::ToggleOutputWindow => {
                // Handled in the action drain loop, which has the event
                // loop needed for window creation. Never reaches here.
            }
            WebAction::MorphCapture {
                slot,
                stack_revision,
            } => {
                if stack_revision.is_some_and(|revision| revision != self.layer_stack_revision) {
                    log::warn!(
                        "Rejected stale morph capture at revision {:?}; current revision is {}",
                        stack_revision,
                        self.layer_stack_revision
                    );
                    return;
                }
                self.materialize_morph_at_current_beat();
                let snap = morph::MorphSlot::capture(
                    &self.master_effects,
                    &self.ntsc_params,
                    &self.temporal_params,
                    &self.layers,
                );
                match slot.as_str() {
                    "a" => self.morph.a = Some(snap),
                    "b" => self.morph.b = Some(snap),
                    _ => log::warn!("Ignored invalid morph slot '{slot}'"),
                }
            }
            WebAction::MorphClear => {
                self.morph.clear();
            }
            WebAction::SetMorph { value } => {
                self.morph.set_position(value);
                // Manual performance edits remain responsive while program
                // time is held; only automatic glide/clock motion is frozen.
                self.materialize_morph_at_current_beat();
            }
            WebAction::SetMorphLaw { law } => {
                match law.as_str() {
                    "linear" => self.morph.blend_law = morph::MorphBlendLaw::Linear,
                    "equal_power" => {
                        self.morph.blend_law = morph::MorphBlendLaw::EqualPower;
                    }
                    _ => log::warn!("Ignored invalid morph law '{law}'"),
                }
                self.materialize_morph_at_current_beat();
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
                    self.audio_library_files = scan_audio_folder(&folder);
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
                    "key_mode" => {
                        if let Some(n) = value.as_u64() {
                            p.key_mode = n.min(4) as f32;
                        }
                    }
                    "key_threshold" => {
                        if let Some(n) = value.as_f64() {
                            p.key_threshold = (n as f32).clamp(0.0, 1.0);
                        }
                    }
                    "key_softness" => {
                        if let Some(n) = value.as_f64() {
                            p.key_softness = (n as f32).clamp(0.0, 0.5);
                        }
                    }
                    "key_history" => {
                        if let Some(n) = value.as_f64() {
                            p.key_history = (n as f32).round().clamp(1.0, 23.0);
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
                route_id,
                target_layer_id,
                layer_stack_revision,
                param,
                value,
            } => {
                if let Some(index) = self.resolve_routing_index(index, &route_id) {
                    let resolved_target = (param == "target")
                        .then(|| {
                            value.as_str().and_then(|target| {
                                resolve_routing_target_for_layer_ids(
                                    target,
                                    &target_layer_id,
                                    layer_stack_revision,
                                    self.layers.iter().map(Layer::layer_id),
                                )
                            })
                        })
                        .flatten();
                    if target_layer_id.is_some()
                        && layer_stack_revision
                            .is_some_and(|revision| revision != self.layer_stack_revision)
                    {
                        log::debug!(
                            "Resolved routing target by stable layer ID across stack revision {:?} -> {}",
                            layer_stack_revision,
                            self.layer_stack_revision
                        );
                    }
                    let routing = &mut self.mod_matrix.routings[index];
                    match param.as_str() {
                        "source" => {
                            if let Some(s) = value.as_str() {
                                if let Some(source) = modulation::ModSource::try_from_str(s) {
                                    if routing.source != source {
                                        routing.source = source;
                                        routing.reset_runtime();
                                    }
                                }
                            }
                        }
                        "target" => {
                            if let Some(target) = resolved_target {
                                routing.set_target(target);
                            }
                        }
                        "depth" => {
                            if let Some(n) = value.as_f64() {
                                routing.depth = (n as f32).clamp(-1.0, 1.0);
                            }
                        }
                        "curve" => {
                            if let Some(s) = value.as_str() {
                                let curve = modulation::Curve::from_str(s);
                                if routing.curve != curve {
                                    routing.curve = curve;
                                    routing.reset_runtime();
                                }
                            }
                        }
                        "curve_amount" => {
                            if let Some(n) = value.as_f64() {
                                let amount = (n as f32).clamp(-2.0, 2.0);
                                if routing.curve_amount != amount {
                                    routing.curve_amount = amount;
                                    routing.reset_runtime();
                                }
                            }
                        }
                        "attack" => {
                            if let Some(n) = value.as_f64() {
                                let attack = (n as f32).clamp(0.0, 10.0);
                                if routing.attack != attack {
                                    routing.attack = attack;
                                }
                            }
                        }
                        "release" => {
                            if let Some(n) = value.as_f64() {
                                let release = (n as f32).clamp(0.0, 10.0);
                                if routing.release != release {
                                    routing.release = release;
                                }
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
                        "bypass_master_fx" => {
                            if let Some(enabled) = value.as_bool() {
                                // Bypass is a reversible routing choice. Never
                                // erase the layer's own effect settings when it
                                // is enabled or disabled.
                                layer.bypass_master_fx = enabled;
                            }
                        }
                        "key_mode" => {
                            if let Some(v) = value.as_u64() {
                                layer.effects.key_mode = (v.min(4)) as f32;
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
                        "key_color_r" | "key_color_g" | "key_color_b" => {
                            if let Some(v) = value.as_f64() {
                                let channel = match param.as_str() {
                                    "key_color_r" => 0,
                                    "key_color_g" => 1,
                                    _ => 2,
                                };
                                layer.effects.key_color[channel] = (v as f32).clamp(0.0, 1.0);
                            }
                        }
                        "key_tolerance" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.key_tolerance = (v as f32).clamp(0.0, 1.0);
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
                    let patch = self.capture_current_patch();
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
                    for layer in self.layers.iter().filter(|layer| !layer.is_file_media()) {
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

    /// Neutralize a stopped or silent phone before modulation is sampled.
    fn release_stale_gyro(&mut self) {
        if self.web_state.gyro_status().stale {
            self.mod_matrix.recenter_gyro();
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
        if let Some(error) = self
            .selective_ntsc_worker
            .as_ref()
            .map(ntsc::SelectiveNtscWorker::error)
            .filter(|error| !error.is_empty())
        {
            if !ntsc_snapshot.error.is_empty() {
                ntsc_snapshot.error.push_str("; ");
            }
            ntsc_snapshot.error.push_str(error);
        }
        if !self.selective_ntsc_runtime_error.is_empty() {
            if !ntsc_snapshot.error.is_empty() {
                ntsc_snapshot.error.push_str("; ");
            }
            ntsc_snapshot
                .error
                .push_str(&self.selective_ntsc_runtime_error);
        }
        let mut modulation_snapshot = ModSnapshot::from_matrix(&self.mod_matrix);
        modulation_snapshot.gyro_status = self.web_state.gyro_status();
        let morph_glide = self
            .morph
            .glide
            .filter(|glide| !glide.is_complete_at(self.mod_matrix.current_beat));
        let morph_glide_remaining = morph_glide
            .map(|glide| {
                (glide.start_beat + glide.duration_beats - self.mod_matrix.current_beat).max(0.0)
            })
            .unwrap_or(0.0);

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
                        bypass_master_fx: layer.bypass_master_fx,
                        progress: layer.progress(),
                        key_mode: layer.effects.key_mode as u32,
                        key_threshold: layer.effects.key_threshold,
                        key_softness: layer.effects.key_softness,
                        key_color: layer.effects.key_color,
                        key_tolerance: layer.effects.key_tolerance,
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
                        offline_export_policy: if layer.is_file_media() {
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
            modulation: modulation_snapshot,
            audio: AudioSnapshot {
                enabled: self.mod_matrix.audio_enabled,
                source_kind: self.mod_matrix.audio_source_kind.clone(),
                gain: self.mod_matrix.audio_gain,
                level: self.mod_matrix.audio.level,
                bass: self.mod_matrix.audio.bass,
                mid: self.mod_matrix.audio.mid,
                high: self.mod_matrix.audio.high,
                onset: self.mod_matrix.audio.onset,
                bright: self.mod_matrix.audio.bright,
                noise: self.mod_matrix.audio.noise,
                device: self.audio.device_name.clone(),
                error: if self.mod_matrix.audio_source_kind == modulation::AUDIO_SOURCE_FILE {
                    self.audio_clip_error.clone()
                } else {
                    self.audio.error.clone()
                },
                devices: self.audio.devices.clone(),
                system_playback_devices: self.audio.system_playback_devices.clone(),
                selected: self.mod_matrix.audio_device.clone(),
                active_device: self.audio.active_device().to_string(),
                using_fallback: self.audio.is_using_device_fallback(),
                clip_files: self
                    .audio_library_files
                    .iter()
                    .filter_map(|path| {
                        path.file_name()
                            .map(|name| name.to_string_lossy().into_owned())
                    })
                    .collect(),
                clip_path: self.selected_audio_clip_name(),
                clip_loading: self.audio_clip_loader.state() == audio::AudioClipLoadState::Loading,
                clip_duration_secs: self
                    .audio_clip
                    .as_ref()
                    .map(|clip| clip.info().duration_secs)
                    .unwrap_or(0.0),
                band_count: self.mod_matrix.audio_band_config.count(),
                band_edges: self.mod_matrix.audio_band_config.crossovers().to_vec(),
                band_ceiling_hz: self.mod_matrix.audio_band_config.ceiling_hz(),
                bands: self.mod_matrix.audio.bands[..self.mod_matrix.audio_band_config.count()]
                    .to_vec(),
                spectrum: if self.mod_matrix.audio_source_kind == modulation::AUDIO_SOURCE_FILE {
                    self.audio_clip_spectrum.to_vec()
                } else {
                    self.audio.spectrum().to_vec()
                },
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
                clock_bpm: self.mod_matrix.clock.bpm,
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
            output_window: self.output_window_open(),
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
                gliding: morph_glide.is_some(),
                glide_target: morph_glide
                    .map(|glide| glide.target)
                    .unwrap_or(self.morph.t),
                glide_duration_beats: morph_glide_remaining,
            },
            export_progress,
            export_error,
            export_status: export_status.to_string(),
            patch_save_status: self.patch_collector.status(),
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

/// Scan a directory for supported visual files, returning a sorted list.
fn scan_folder(folder: &PathBuf) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(folder) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| !is_upload_reservation(p) && p.is_file() && is_supported_visual_file(p))
        .collect();
    files.sort();
    files
}

fn scan_audio_folder(folder: &PathBuf) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(folder) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            !is_upload_reservation(path) && path.is_file() && audio::is_supported_audio_file(path)
        })
        .collect();
    files.sort();
    files
}

fn is_upload_reservation(path: &std::path::Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with(".upload-"))
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
            let dimensions = if is_still_image_file(std::path::Path::new(path)) {
                video::decode_still_image(std::path::Path::new(path), None)
                    .map(|image| (image.width, image.height))
            } else {
                video::VideoDecoder::open(path).map(|decoder| (decoder.width, decoder.height))
            };
            match dimensions {
                Ok((width, height)) => {
                    output_width = width;
                    output_height = height;
                }
                Err(error) => log::error!(
                    "Initial visual could not define the output; using 1280x720: {error}"
                ),
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
        let output_window_id = self.renderer.as_ref().and_then(Renderer::output_window_id);
        if output_window_id == Some(window_id) {
            match event {
                WindowEvent::CloseRequested => self.close_output_window(),
                WindowEvent::Resized(size) => {
                    if let Some(renderer) = self.renderer.as_mut() {
                        renderer.resize_output(size.width, size.height);
                    }
                }
                WindowEvent::ScaleFactorChanged { .. } => {
                    let size = self
                        .renderer
                        .as_ref()
                        .and_then(Renderer::output_window_size);
                    if let (Some(renderer), Some(size)) = (self.renderer.as_mut(), size) {
                        renderer.resize_output(size.width, size.height);
                    }
                }
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key,
                            state: winit::event::ElementState::Pressed,
                            repeat: false,
                            ..
                        },
                    ..
                } => {
                    if matches!(
                        physical_key,
                        PhysicalKey::Code(KeyCode::Escape) | PhysicalKey::Code(KeyCode::KeyO)
                    ) {
                        self.close_output_window();
                    }
                }
                _ => {}
            }
            return;
        }

        // Winit can deliver a final queued event after a dedicated output
        // window has been dropped. Never reinterpret that stale WindowId as
        // an event for the main preview (which could otherwise resize or close
        // the entire application).
        if self
            .window
            .as_ref()
            .is_none_or(|window| window.id() != window_id)
        {
            return;
        }

        // Discrete window commands execute once per physical press. In
        // particular, an Escape repeat delivered to the main window after
        // closing single-monitor Output must not become an application quit.
        if let WindowEvent::KeyboardInput {
            event:
                KeyEvent {
                    physical_key,
                    state,
                    repeat,
                    ..
                },
            ..
        } = &event
        {
            if ignore_discrete_window_key_repeat(*physical_key, *repeat) {
                return;
            }
            if self.output_on_main
                && *state == winit::event::ElementState::Pressed
                && is_discrete_window_key(*physical_key)
            {
                self.close_output_window();
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
                } else if is_supported_visual_file(&path) {
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
                    if idx < self.layers.len() {
                        let flow = {
                            let layer = &mut self.layers[idx];
                            apply_action(action, &mut layer.effects)
                        };
                        match flow {
                            ControlFlow::Quit => event_loop.exit(),
                            ControlFlow::TogglePause => {
                                let layer = &mut self.layers[idx];
                                layer.paused = !layer.paused;
                                layer.reset_transport_timing();
                            }
                            ControlFlow::ToggleFullscreen => self.toggle_preview_fullscreen(),
                            ControlFlow::ToggleOutputWindow => self.apply_output_window_command(
                                event_loop,
                                OutputWindowCommand::Toggle,
                            ),
                            ControlFlow::ToggleBlackout => self.toggle_blackout(),
                            ControlFlow::Continue => {}
                        }
                    }
                } else {
                    let mut dummy = effects::EffectUniforms::default();
                    match apply_action(action, &mut dummy) {
                        ControlFlow::Quit => event_loop.exit(),
                        ControlFlow::ToggleFullscreen => self.toggle_preview_fullscreen(),
                        ControlFlow::ToggleOutputWindow => self
                            .apply_output_window_command(event_loop, OutputWindowCommand::Toggle),
                        ControlFlow::ToggleBlackout => self.toggle_blackout(),
                        ControlFlow::TogglePause => self.set_master_paused(!self.master_paused),
                        _ => {}
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                if self.exit_if_device_lost(event_loop) {
                    return;
                }

                let now = Instant::now();
                if now - self.last_frame_time >= FRAME_DURATION {
                    let wall_frame_delta = now.saturating_duration_since(self.last_frame_time);
                    self.last_frame_time = now;

                    // Process actions from web UI
                    let pending_actions: Vec<_> = self
                        .web_state
                        .actions
                        .try_lock()
                        .map(|mut a| a.drain(..).collect())
                        .unwrap_or_default();
                    let mut requested_output = None;
                    for action in pending_actions {
                        // Output-window creation needs the event loop. Fold all
                        // requests in this packet batch into one final state so
                        // retries or multiple clients cannot churn swapchains
                        // several times in a single frame.
                        if let Some(command) = Self::output_window_command(&action) {
                            let current =
                                requested_output.unwrap_or_else(|| self.output_window_open());
                            requested_output =
                                Some(resolve_output_window_command(current, command));
                        } else {
                            self.handle_web_action(action);
                        }
                    }
                    if let Some(enabled) = requested_output {
                        self.set_output_window(event_loop, enabled);
                        if self.exit_if_device_lost(event_loop) {
                            return;
                        }
                    }
                    let audio_load_completed = self.poll_audio_clip_loader();
                    let program_transport_paused = self.program_transport_paused();
                    let program_wall_delta = if audio_load_completed {
                        Duration::ZERO
                    } else {
                        wall_frame_delta
                    };
                    let (elapsed_duration, program_delta) = self
                        .program_clock
                        .tick(program_wall_delta, program_transport_paused);
                    self.release_stale_gyro();

                    // Build minimal egui frame (video display only, no UI panels)
                    let window = self.window.as_ref().unwrap();
                    let egui_winit = self.egui_winit.as_mut().unwrap();
                    let raw_input = egui_winit.take_egui_input(window);

                    let video_egui_texture_id = self.video_egui_texture_id;
                    let output_on_main = self.output_on_main;
                    let output_width = self.renderer.as_ref().unwrap().output_width;
                    let output_height = self.renderer.as_ref().unwrap().output_height;

                    let egui_context = self.egui_ctx.clone();
                    let yaml_editor = &mut self.yaml_editor;
                    let layers = &mut self.layers;
                    let master_effects = &mut self.master_effects;
                    let full_output = egui_context.run_ui(raw_input, |ctx| {
                        if show_editor_panel(output_on_main, yaml_editor.active) {
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

                    let window = self.window.as_ref().unwrap().clone();
                    let egui_winit = self.egui_winit.as_mut().unwrap();
                    egui_winit.handle_platform_output(&window, full_output.platform_output);

                    let tris = self
                        .egui_ctx
                        .tessellate(full_output.shapes, full_output.pixels_per_point);

                    // Set time uniform on all effects (drives animated noise/breathing)
                    let elapsed = elapsed_duration.as_secs_f32();
                    for layer in &mut self.layers {
                        layer.effects.time = elapsed;
                    }
                    self.master_effects.time = elapsed;

                    // Sync audio capture with the requested state, then feed
                    // the latest analysis into the matrix as a source.
                    self.mod_matrix.audio_band_config = self
                        .audio
                        .set_band_config(self.mod_matrix.audio_band_config);
                    if self.mod_matrix.audio_source_kind == modulation::AUDIO_SOURCE_FILE {
                        if self.audio.is_running() {
                            self.audio.stop();
                        }
                        if !program_transport_paused {
                            if self.mod_matrix.audio_enabled {
                                if let Some(clip) = &self.audio_clip {
                                    let analysis = clip.analyze_at_time(
                                        elapsed_duration.as_secs_f64(),
                                        self.mod_matrix.audio_gain,
                                        self.mod_matrix.audio_band_config,
                                    );
                                    self.mod_matrix.audio = analysis.levels;
                                    self.audio_clip_spectrum = analysis.spectrum;
                                } else {
                                    self.mod_matrix.audio = audio::AudioLevels::default();
                                    self.audio_clip_spectrum.fill(0.0);
                                }
                            } else {
                                self.mod_matrix.audio = audio::AudioLevels::default();
                                self.audio_clip_spectrum.fill(0.0);
                            }
                        }
                    } else {
                        if self.mod_matrix.audio_enabled
                            && !self.audio.is_running_for(&self.mod_matrix.audio_device)
                        {
                            let device = self.mod_matrix.audio_device.clone();
                            if self.audio.is_running() {
                                self.audio.stop();
                            }
                            self.audio.start(&device);
                            if !self.audio.error.is_empty() {
                                self.mod_matrix.audio_enabled = false;
                            }
                        } else if !self.mod_matrix.audio_enabled && self.audio.is_running() {
                            self.audio.stop();
                        }
                        if !program_transport_paused {
                            self.mod_matrix.audio = self.audio.analyze(self.mod_matrix.audio_gain);
                            if !self.audio.is_running() && !self.audio.error.is_empty() {
                                self.mod_matrix.audio_enabled = false;
                            }
                        }
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
                                let now = Instant::now();
                                self.mod_matrix.clock.set_bpm_at(bpm, now);
                                self.mod_matrix.clock.set_external_beat(Some(beat), now);
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
                    if program_transport_paused {
                        // Telemetry above remains live, but cached routing
                        // outputs, LFO phase, spring/slew, and beat latches are
                        // part of the frozen visual program.
                        self.mod_matrix.reset_update_timing();
                    } else {
                        self.mod_matrix.update(Instant::now());
                        self.release_quantized_actions_on_downbeat();
                    }
                    let modulation_frame = self.mod_matrix.frame();

                    // Patch morph: while A and B are both set, the crossfader
                    // (plus any "morph" routings — an LFO can sweep worlds)
                    // writes the base parameters as their interpolation. The
                    // matrix then breathes on top of the morphed bases.
                    if !program_transport_paused {
                        // Glide settlement is independent of slot activation:
                        // one-slot automation must still finish cleanly.
                        self.morph.settle_glide_at(self.mod_matrix.current_beat);
                        self.materialize_morph_at_current_beat_with_offset(
                            modulation_frame.morph_offset(),
                        );
                    }

                    // Broadcast after this frame's program-time sampling and
                    // morph application so the panel reflects the exact
                    // authoritative state being rendered, not the prior one.
                    self.push_web_state();

                    // Resolve all per-layer route destinations in one O(routes)
                    // batch and reuse the exact results for transport and GPU
                    // rendering. This also prevents duplicated route work from
                    // giving either consumer a subtly different future path.
                    let layer_modulations = modulation_frame.modulate_layers(
                        self.layers
                            .iter()
                            .map(|layer| (&layer.effects, layer.opacity, layer.speed, layer.fps)),
                    );

                    // Advance live sources only after this frame's audio,
                    // MIDI, slew, beat latch, and morph state are current.
                    // This gives transport the same modulation phase used by
                    // rendering and by the deterministic exporter.
                    let renderer = self.renderer.as_ref().unwrap();
                    for (layer, layer_mod) in self.layers.iter_mut().zip(layer_modulations.iter()) {
                        let playing = !program_transport_paused && !layer.paused;

                        if layer.is_file_media() {
                            // The decoder seeds frame zero during open and
                            // publishes one latest-only result thereafter.
                            // Harvest before pause gating so a recalled paused
                            // layer displays a defined first frame, matching
                            // deterministic export.
                            match layer.take_ready_media_frame() {
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
                            if let Err(error) =
                                layer.request_due_video_frames_at(layer_mod.fps, layer_mod.speed)
                            {
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

                    let (mod_master, mod_ntsc, mod_temporal) = modulation_frame.modulate(
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

                    // Snapshot the audience at the actual blackout edge before
                    // any render or absolute clear can overwrite slot2. This is
                    // path-independent: VHS/bypass controls may change while
                    // black, yet releasing the cut under Pause must restore the
                    // exact pre-cut pixels. A running release renders afresh.
                    if let Some(action) = blackout_audience_edge_action(
                        self.blackout,
                        self.blackout_presented,
                        program_transport_paused,
                        self.selective_hold_snapshot_valid,
                    ) {
                        match action {
                            HeldAudienceAction::Capture => {
                                renderer.capture_held_audience(&mut encoder);
                                self.selective_hold_snapshot_valid = true;
                            }
                            HeldAudienceAction::Restore => {
                                renderer.restore_held_audience(&mut encoder);
                            }
                            HeldAudienceAction::Keep => {}
                        }
                        if program_transport_paused {
                            self.selective_transition_holding = true;
                            self.selective_hold_spout_barrier_epoch = None;
                            self.selective_hold_spout_readback_epoch = None;
                        }
                    }
                    if !self.blackout && self.blackout_presented {
                        self.blackout_presented = false;
                    }

                    // Per-layer modulated copies: opacity crossfades, key
                    // thresholds breathing shapes in and out — bases untouched.
                    let layer_mods: Vec<(effects::EffectUniforms, f32)> = layer_modulations
                        .iter()
                        .map(|lm| (lm.effects, lm.opacity))
                        .collect();

                    let selective_required = mod_ntsc.enabled
                        && !self.blackout
                        && ntsc::selective_ntsc_required(self.layers.iter().zip(&layer_mods).map(
                            |(layer, (_, opacity))| ntsc::SelectiveNtscLayerDescriptor {
                                layer_id: layer.layer_id(),
                                visible: layer.visible,
                                bypass_master_fx: layer.bypass_master_fx,
                                opacity: *opacity,
                                blend_mode: layer.blend_mode.as_u32(),
                                transform_fingerprint: 0,
                            },
                        ));
                    let selective_contributing_layers = self
                        .layers
                        .iter()
                        .zip(&layer_mods)
                        .filter(|(layer, (_, opacity))| {
                            layer.visible && opacity.is_finite() && *opacity > 0.0
                        })
                        .count();
                    let selective_budget_error = selective_required
                        .then(|| {
                            ntsc::validate_selective_ntsc_live_memory(
                                renderer.output_width,
                                renderer.output_height,
                                selective_contributing_layers,
                            )
                            .and_then(|memory| {
                                ntsc::validate_selective_ntsc_gpu_staging_limit(
                                    memory,
                                    renderer.device.limits().max_buffer_size,
                                )
                            })
                        })
                        .and_then(Result::err);
                    let next_runtime_error = selective_budget_error.clone().unwrap_or_default();
                    if next_runtime_error != self.selective_ntsc_runtime_error {
                        if !next_runtime_error.is_empty() {
                            log::error!("{next_runtime_error}");
                        }
                        self.selective_ntsc_runtime_error = next_runtime_error;
                    }
                    let requested_ntsc_path = if !mod_ntsc.enabled || self.blackout {
                        LiveNtscPath::Disabled
                    } else if selective_required {
                        LiveNtscPath::SelectivePerLayer
                    } else {
                        LiveNtscPath::LegacyGlobal
                    };
                    let mut selective_audience_sample = None;
                    let ntsc_path_changed = requested_ntsc_path != self.ntsc_pipeline_path;
                    let selective_edge =
                        is_selective_path_edge(self.ntsc_pipeline_path, requested_ntsc_path);
                    let topology_signature =
                        selective_ntsc_topology_signature(&self.layers, &layer_mods);
                    let selective_topology_changed = selective_required
                        && topology_signature != self.selective_topology_signature;
                    let selective_rebuild = selective_edge || selective_topology_changed;
                    let hold_rebuild = program_transport_paused || selective_budget_error.is_some();

                    // Path and topology can change in the same frame (most
                    // notably on first entry into selective VHS). Treat them
                    // as one committed generation boundary so async work is
                    // invalidated exactly once.
                    if selective_generation_boundary(ntsc_path_changed, selective_topology_changed)
                    {
                        self.visual_epoch = self.visual_epoch.wrapping_add(1);
                        self.ntsc_presented = None;
                    }
                    if ntsc_path_changed {
                        self.ntsc_pipeline_path = requested_ntsc_path;
                    }
                    if selective_topology_changed {
                        self.selective_topology_signature = topology_signature;
                        self.selective_topology_generation =
                            self.selective_topology_generation.wrapping_add(1).max(1);
                    }
                    if selective_rebuild {
                        let was_holding = self.selective_transition_holding;
                        self.selective_temporal_debt = 0.0;
                        renderer.reset_visual_generation();
                        if hold_rebuild {
                            match held_audience_action(
                                was_holding,
                                self.blackout,
                                self.selective_hold_snapshot_valid,
                            ) {
                                HeldAudienceAction::Capture => {
                                    renderer.capture_held_audience(&mut encoder);
                                    self.selective_hold_snapshot_valid = true;
                                }
                                HeldAudienceAction::Restore => {
                                    // Blackout clears slot2, but never the
                                    // saved audience texture. Restore it on a
                                    // paused return to selective/direct output.
                                    renderer.restore_held_audience(&mut encoder);
                                }
                                HeldAudienceAction::Keep => {}
                            }
                            self.selective_transition_holding = true;
                            self.selective_hold_spout_barrier_epoch = None;
                            self.selective_hold_spout_readback_epoch = None;
                        } else {
                            self.selective_transition_holding = false;
                            // A playing blackout may later be paused before it
                            // is released. Retain the pre-cut snapshot for that
                            // entire interval; a real non-black render below or
                            // an accepted selective sample disposes of it.
                            if rendered_audience_may_discard_blackout_snapshot(
                                program_transport_paused,
                                self.blackout,
                            ) {
                                self.selective_hold_snapshot_valid = false;
                            }
                            self.selective_hold_spout_barrier_epoch = None;
                            self.selective_hold_spout_readback_epoch = None;
                            if requested_ntsc_path == LiveNtscPath::SelectivePerLayer
                                && selective_budget_error.is_none()
                                && selective_rebuild_should_clear_audience(false)
                            {
                                // Never flash a globally processed frame while
                                // the first selective generation is prepared.
                                renderer.clear_composite(&mut encoder);
                            }
                        }
                    }

                    // The established global pipeline remains literally the
                    // same command order unless a visible, positive-opacity
                    // bypass layer requires selective VHS.
                    if !selective_required {
                        if direct_path_may_replace_selective_hold(
                            program_transport_paused,
                            self.selective_transition_holding,
                        ) {
                            renderer.render_layers_and_master(
                                &mut encoder,
                                &self.layers,
                                &layer_mods,
                                &mod_master,
                            );
                            renderer.render_temporal_with_dt(
                                &mut encoder,
                                &mod_temporal,
                                program_delta,
                                !program_transport_paused,
                            );
                            renderer.render_opaque_output(&mut encoder);
                            if rendered_audience_may_discard_blackout_snapshot(
                                program_transport_paused,
                                self.blackout,
                            ) {
                                self.selective_transition_holding = false;
                                self.selective_hold_snapshot_valid = false;
                                self.selective_hold_spout_barrier_epoch = None;
                                self.selective_hold_spout_readback_epoch = None;
                            }
                        }
                    } else {
                        if !program_transport_paused && selective_budget_error.is_none() {
                            self.selective_temporal_debt =
                                (self.selective_temporal_debt.clamp(0.0, 1.0) + program_delta)
                                    .min(1.0);
                        } else if selective_budget_error.is_some() {
                            self.selective_temporal_debt = 0.0;
                        }
                        let generation = ntsc::SelectiveNtscGeneration {
                            visual_epoch: self.visual_epoch,
                            topology_generation: self.selective_topology_generation,
                            width: renderer.output_width,
                            height: renderer.output_height,
                            sample_sequence: self.selective_sample_sequence,
                        };
                        let worker = self
                            .selective_ntsc_worker
                            .get_or_insert_with(ntsc::SelectiveNtscWorker::new);
                        let current_plan = live_selective_ntsc_plan(
                            generation,
                            ntsc::NtscFrameMetadata {
                                params: mod_ntsc.clone(),
                                reference_frame: ntsc::reference_frame_for_time(
                                    elapsed_duration.as_secs_f64(),
                                ),
                            },
                            &self.layers,
                            &layer_mods,
                        );

                        // Move the newest complete GPU batch into the bounded
                        // CPU worker before checking for a processed result.
                        // Stale mapped batches are recycled without submission.
                        if selective_budget_error.is_none() && worker.is_idle() {
                            if let Some(batch) = renderer.poll_selective_ntsc_readback() {
                                if current_plan.as_ref().is_some_and(|current| {
                                    ntsc::selective_plan_compatible(&batch.plan, current)
                                }) {
                                    worker.try_submit(batch);
                                }
                            }
                            if exit_on_renderer_device_loss(
                                renderer,
                                &mut self.output_error,
                                event_loop,
                            ) {
                                return;
                            }
                        }

                        // A finished generation becomes the sole input to the
                        // downstream temporal stage. Paused transport holds the
                        // last output exactly; no new batch is submitted.
                        if selective_budget_error.is_none() && !program_transport_paused {
                            if let Some(processed) = worker.try_recv() {
                                if current_plan.as_ref().is_some_and(|current| {
                                    ntsc::selective_plan_compatible(&processed.plan, current)
                                }) {
                                    if let Err(error) =
                                        renderer.write_engine_composite(&processed.pixels)
                                    {
                                        log::error!("Rejected selective NTSC result: {error}");
                                        self.selective_ntsc_runtime_error = error;
                                    } else {
                                        renderer.render_temporal_with_dt(
                                            &mut encoder,
                                            &mod_temporal,
                                            self.selective_temporal_debt,
                                            !program_transport_paused,
                                        );
                                        renderer.render_opaque_output(&mut encoder);
                                        self.selective_temporal_debt = 0.0;
                                        self.selective_transition_holding = false;
                                        self.selective_hold_snapshot_valid = false;
                                        self.selective_hold_spout_barrier_epoch = None;
                                        self.selective_hold_spout_readback_epoch = None;
                                        selective_audience_sample = Some(processed.plan.generation);
                                    }
                                }
                            }
                        }

                        if selective_budget_error.is_none()
                            && !program_transport_paused
                            && worker.is_idle()
                        {
                            if let Some(mut plan) = current_plan {
                                let submitted_sequence =
                                    self.selective_sample_sequence.wrapping_add(1);
                                plan.generation.sample_sequence = submitted_sequence;
                                match renderer.begin_selective_ntsc_readback(
                                    &mut encoder,
                                    &self.layers,
                                    &layer_mods,
                                    &mod_master,
                                    plan,
                                ) {
                                    Ok(true) => {
                                        self.selective_sample_sequence = submitted_sequence;
                                        self.selective_ntsc_runtime_error.clear();
                                    }
                                    Ok(false) => {}
                                    Err(error) => {
                                        log::error!("Selective NTSC snapshot rejected: {error}");
                                        self.selective_ntsc_runtime_error = error;
                                    }
                                }
                            }
                        }
                    }

                    // A paused selective transition advances Spout's colour
                    // intent before copying the held audience. The barrier
                    // drops pending/dequeued old-epoch frames without changing
                    // the receiver's last texture; the tagged readback then
                    // converges it to the exact slot2 image. If Spout starts
                    // during the hold, this same branch runs on the next frame.
                    let held_spout_active = self.selective_transition_holding
                        && !self.blackout
                        && self.spout_enabled
                        && self.spout.is_running();
                    if !held_spout_active {
                        self.selective_hold_spout_barrier_epoch = None;
                        self.selective_hold_spout_readback_epoch = None;
                    }
                    if held_spout_active
                        && self.selective_hold_spout_barrier_epoch != Some(self.visual_epoch)
                        && self.spout.hold_colour_epoch(self.visual_epoch)
                    {
                        self.selective_hold_spout_barrier_epoch = Some(self.visual_epoch);
                        self.selective_hold_spout_readback_epoch = None;
                    }
                    let held_audience_readback = (held_spout_active
                        && self.selective_hold_spout_barrier_epoch == Some(self.visual_epoch)
                        && self.selective_hold_spout_readback_epoch != Some(self.visual_epoch))
                    .then(|| renderer.begin_held_audience_readback(&mut encoder, self.visual_epoch))
                    .flatten();
                    if held_audience_readback.is_some() {
                        self.selective_hold_spout_readback_epoch = Some(self.visual_epoch);
                    }

                    // Commit the raw GPU render before any CPU-processed
                    // queue write. A later command buffer would otherwise
                    // overwrite the queued NTSC upload.
                    // Selective Spout coupling is encoded immediately after
                    // the exact sample's Temporal+opaque passes, in this same
                    // command stream. A held redraw schedules no duplicate
                    // readback and can therefore never masquerade as a newer
                    // sample.
                    let selective_audience_readback =
                        selective_audience_sample.and_then(|sample| {
                            (self.spout_enabled && self.spout.is_running())
                                .then(|| {
                                    renderer.begin_selective_audience_readback(&mut encoder, sample)
                                })
                                .flatten()
                        });

                    renderer.queue.submit(std::iter::once(encoder.finish()));
                    if exit_on_renderer_device_loss(renderer, &mut self.output_error, event_loop) {
                        return;
                    }
                    if selective_required {
                        renderer.map_selective_ntsc_readback();
                    }
                    if let Some(index) = selective_audience_readback {
                        renderer.map_readback(index);
                    }
                    if let Some(index) = held_audience_readback {
                        renderer.map_readback(index);
                    }
                    if exit_on_renderer_device_loss(renderer, &mut self.output_error, event_loop) {
                        return;
                    }
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
                    // thread never waits on GPU readback or CPU processing.
                    // Latency is bounded/latest-only but depends on resolution,
                    // contributing layers, settings, and hardware. Spout is
                    // tagged to exact processed audience samples instead of
                    // inferring identity from a later held redraw.
                    // Blackout bypasses the stylized post-process: VHS snow
                    // must never turn an emergency cut into visible texture.
                    let ntsc_path = requested_ntsc_path;
                    if ntsc_path_changed
                        && self.blackout
                        && self.spout_enabled
                        && self.spout.is_running()
                    {
                        // Entering blackout while NTSC was active advances the
                        // visual epoch here after the first cut request. Replace
                        // it immediately with a cut carrying the final epoch.
                        self.spout.cut_to_black(
                            renderer.output_width,
                            renderer.output_height,
                            self.visual_epoch,
                        );
                    }
                    let spout_active = self.spout_enabled && self.spout.is_running();
                    // Harvest a completed raw readback. The generation tag
                    // rejects both pre-blackout content and delayed blackout
                    // frames that finish after the cut is released.
                    let readback_poll = renderer.poll_readback();
                    if exit_on_renderer_device_loss(renderer, &mut self.output_error, event_loop) {
                        return;
                    }
                    if readback_poll.held_audience_not_harvested
                        && self.selective_transition_holding
                    {
                        // Retry after a GPU map failure or a rarer superseded
                        // map. The Spout epoch barrier remains authoritative;
                        // only the exact held-image copy is rescheduled.
                        self.selective_hold_spout_readback_epoch = None;
                    }
                    if let Some(frame) = readback_poll.frame {
                        if frame.epoch != self.visual_epoch {
                            // Stale visual generation; deliberately discarded.
                            if frame.held_audience && self.selective_transition_holding {
                                self.selective_hold_spout_readback_epoch = None;
                            }
                        } else if self.blackout {
                            if spout_active {
                                self.spout.try_submit(
                                    frame.pixels,
                                    renderer.output_width,
                                    renderer.output_height,
                                    frame.epoch,
                                );
                            }
                        } else if frame.held_audience {
                            if self.selective_transition_holding
                                && spout_active
                                && !self.spout.try_submit(
                                    frame.pixels,
                                    renderer.output_width,
                                    renderer.output_height,
                                    frame.epoch,
                                )
                            {
                                // Retry with another exact held readback;
                                // normal Spout submission is deliberately
                                // nonblocking and may lose a lock race.
                                self.selective_hold_spout_readback_epoch = None;
                            }
                        } else if ntsc_path == LiveNtscPath::LegacyGlobal {
                            if let Some(metadata) = frame.ntsc_metadata {
                                self.ntsc_worker.try_submit(
                                    frame.pixels,
                                    renderer.output_width,
                                    renderer.output_height,
                                    metadata,
                                    frame.epoch,
                                );
                            } else {
                                log::warn!(
                                    "Discarded an NTSC readback without its sampled parameters"
                                );
                            }
                        } else if ntsc_path == LiveNtscPath::SelectivePerLayer {
                            if selective_spout_sample_is_eligible(
                                frame.epoch,
                                frame.selective_sample,
                                ntsc::SelectiveNtscGeneration {
                                    visual_epoch: self.visual_epoch,
                                    topology_generation: self.selective_topology_generation,
                                    width: renderer.output_width,
                                    height: renderer.output_height,
                                    sample_sequence: self.selective_sample_sequence,
                                },
                                spout_active,
                                self.blackout,
                            ) {
                                self.spout.try_submit(
                                    frame.pixels,
                                    renderer.output_width,
                                    renderer.output_height,
                                    frame.epoch,
                                );
                            }
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
                        if processed.epoch == self.visual_epoch
                            && ntsc_path == LiveNtscPath::LegacyGlobal
                        {
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
                    let need_raw_readback = raw_audience_readback_required(
                        self.blackout,
                        self.selective_transition_holding,
                        ntsc_path,
                        spout_active,
                    );
                    if need_raw_readback {
                        let ntsc_metadata = (ntsc_path == LiveNtscPath::LegacyGlobal).then(|| {
                            ntsc::NtscFrameMetadata {
                                params: mod_ntsc.clone(),
                                reference_frame: ntsc::reference_frame_for_time(
                                    elapsed_duration.as_secs_f64(),
                                ),
                            }
                        });
                        let slot =
                            renderer.begin_readback(&mut encoder, self.visual_epoch, ntsc_metadata);
                        renderer.queue.submit(std::iter::once(encoder.finish()));
                        if exit_on_renderer_device_loss(
                            renderer,
                            &mut self.output_error,
                            event_loop,
                        ) {
                            return;
                        }
                        if let Some(idx) = slot {
                            renderer.map_readback(idx);
                        }
                        if exit_on_renderer_device_loss(
                            renderer,
                            &mut self.output_error,
                            event_loop,
                        ) {
                            return;
                        }
                        encoder = renderer.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("Audience Encoder"),
                            },
                        );
                    }

                    if ntsc_path == LiveNtscPath::LegacyGlobal {
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
                        if exit_on_renderer_device_loss(
                            renderer,
                            &mut self.output_error,
                            event_loop,
                        ) {
                            return;
                        }
                        self.blackout_presented = true;
                        encoder = renderer.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("Blackout Audience Encoder"),
                            },
                        );
                    }

                    // Surface configuration is intentionally inside the
                    // accepted 30 Hz frame and after resize events have been
                    // coalesced. This prevents fullscreen/WM_SIZE storms from
                    // repeatedly forcing the DX12 queue idle between frames.
                    // A minimized preview skips only this presentation target;
                    // the engine and dedicated output continue to run.
                    let win_size = window.inner_size();
                    let preview_surface_available = if win_size.width == 0 || win_size.height == 0 {
                        renderer.resize(0, 0);
                        false
                    } else {
                        match renderer.prepare_main_surface() {
                            Ok(ready) => ready,
                            Err(error) => {
                                self.output_error = format!("Main output surface: {error}");
                                log::error!("{}", self.output_error);
                                if exit_on_renderer_device_loss(
                                    renderer,
                                    &mut self.output_error,
                                    event_loop,
                                ) {
                                    return;
                                }
                                false
                            }
                        }
                    };

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
                        let output_present = match render_output_checked(renderer, &mut encoder) {
                            Ok(texture) => texture,
                            Err(failure) => {
                                self.output_error = failure.message;
                                log::error!("Output presentation failed: {}", self.output_error);
                                if failure.device_lost {
                                    event_loop.exit();
                                    return;
                                }
                                None
                            }
                        };
                        renderer.queue.submit(std::iter::once(encoder.finish()));
                        // Once acquired, always consume a surface texture with
                        // present before honoring device loss. Dropping it
                        // would ask wgpu to discard through the invalid device.
                        if let Some(t) = output_present {
                            t.present();
                        }
                        if exit_on_renderer_device_loss(
                            renderer,
                            &mut self.output_error,
                            event_loop,
                        ) {
                            return;
                        }
                        if let Some(w) = &self.window {
                            w.request_redraw();
                        }
                        return;
                    }

                    let surface_texture = match renderer.surface.get_current_texture() {
                        wgpu::CurrentSurfaceTexture::Success(t) => t,
                        wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                            // Present this usable texture, then repair the
                            // swapchain once at the next render boundary.
                            renderer.reconfigure_surface();
                            t
                        }
                        status @ (wgpu::CurrentSurfaceTexture::Outdated
                        | wgpu::CurrentSurfaceTexture::Lost) => {
                            // The engine frame (including temporal history and
                            // dedicated output) is independent of the preview
                            // swapchain. Commit it before repairing the preview
                            // surface so CPU/GPU history indices cannot diverge.
                            for id in &full_output.textures_delta.free {
                                egui_renderer.free_texture(id);
                            }
                            let output_present = match render_output_checked(renderer, &mut encoder)
                            {
                                Ok(texture) => texture,
                                Err(failure) => {
                                    self.output_error = failure.message;
                                    log::error!(
                                        "Output presentation failed: {}",
                                        self.output_error
                                    );
                                    if failure.device_lost {
                                        event_loop.exit();
                                        return;
                                    }
                                    None
                                }
                            };
                            renderer.queue.submit(std::iter::once(encoder.finish()));
                            if let Some(texture) = output_present {
                                texture.present();
                            }
                            if exit_on_renderer_device_loss(
                                renderer,
                                &mut self.output_error,
                                event_loop,
                            ) {
                                return;
                            }
                            let size = window.inner_size();
                            let r = self.renderer.as_mut().unwrap();
                            let repaired = match status {
                                wgpu::CurrentSurfaceTexture::Lost => {
                                    r.recreate_main_surface(window.clone())
                                }
                                wgpu::CurrentSurfaceTexture::Outdated
                                    if size.width > 0 && size.height > 0 =>
                                {
                                    r.resize(size.width, size.height);
                                    // Outdated can be reported without an
                                    // extent change, so resize alone is not a
                                    // sufficient liveness signal.
                                    r.reconfigure_surface();
                                    Ok(())
                                }
                                _ => {
                                    r.resize(0, 0);
                                    Ok(())
                                }
                            };
                            if let Err(error) = repaired {
                                self.output_error = format!("Main output surface: {error}");
                                log::error!("{}", self.output_error);
                                if r.device_error().is_some() {
                                    event_loop.exit();
                                }
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
                            let output_present = match render_output_checked(renderer, &mut encoder)
                            {
                                Ok(texture) => texture,
                                Err(failure) => {
                                    self.output_error = failure.message;
                                    log::error!(
                                        "Output presentation failed: {}",
                                        self.output_error
                                    );
                                    if failure.device_lost {
                                        event_loop.exit();
                                        return;
                                    }
                                    None
                                }
                            };
                            renderer.queue.submit(std::iter::once(encoder.finish()));
                            if let Some(texture) = output_present {
                                texture.present();
                            }
                            if exit_on_renderer_device_loss(
                                renderer,
                                &mut self.output_error,
                                event_loop,
                            ) {
                                return;
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
                    let mut output_device_lost = false;
                    let output_present = match render_output_checked(renderer, &mut encoder) {
                        Ok(texture) => texture,
                        Err(failure) => {
                            self.output_error = failure.message;
                            log::error!("Output presentation failed: {}", self.output_error);
                            if failure.device_lost {
                                event_loop.exit();
                                output_device_lost = true;
                            }
                            None
                        }
                    };

                    if output_device_lost {
                        // The main texture was acquired before Output reported
                        // the terminal error. Consume it without submitting
                        // more work so Drop cannot attempt an invalid discard.
                        surface_texture.present();
                        return;
                    }
                    renderer.queue.submit(std::iter::once(encoder.finish()));
                    // Both API textures must be marked consumed even when the
                    // first present or submission latches device loss.
                    surface_texture.present();
                    if let Some(t) = output_present {
                        t.present();
                    }
                    if exit_on_renderer_device_loss(renderer, &mut self.output_error, event_loop) {
                        return;
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

fn procedural_usage() -> &'static str {
    "Procedural patch generation (patch-only; rendering remains explicit):\n\
     collide-o-scope generate --anchor <patch.yaml> --output <directory>\n\
       [--count 10] [--temperature 0.5] [--seed 0] [--allow-black-sources]"
}

fn run_generation_cli(arguments: &[String]) -> Result<Vec<PathBuf>, String> {
    if arguments.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{}", procedural_usage());
        return Ok(Vec::new());
    }

    let mut anchor_path: Option<PathBuf> = None;
    let mut output_path = PathBuf::from("generated");
    let mut count = 10usize;
    let mut temperature = 0.5f32;
    let mut seed = 0u64;
    let mut allow_black_sources = false;
    let mut index = 0usize;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--anchor" | "--output" | "--count" | "--temperature" | "--seed" => {
                let flag = arguments[index].clone();
                index += 1;
                let value = arguments
                    .get(index)
                    .ok_or_else(|| format!("{flag} requires a value"))?;
                match flag.as_str() {
                    "--anchor" => anchor_path = Some(PathBuf::from(value)),
                    "--output" => output_path = PathBuf::from(value),
                    "--count" => {
                        count = value
                            .parse()
                            .map_err(|_| format!("invalid --count value {value:?}"))?
                    }
                    "--temperature" => {
                        temperature = value
                            .parse()
                            .map_err(|_| format!("invalid --temperature value {value:?}"))?
                    }
                    "--seed" => {
                        seed = value
                            .parse()
                            .map_err(|_| format!("invalid --seed value {value:?}"))?
                    }
                    _ => unreachable!(),
                }
            }
            "--allow-black-sources" => allow_black_sources = true,
            "--patch-only" => {}
            unknown => {
                return Err(format!(
                    "unknown generate option {unknown:?}\n{}",
                    procedural_usage()
                ))
            }
        }
        index += 1;
    }
    let anchor_path =
        anchor_path.ok_or_else(|| format!("--anchor is required\n{}", procedural_usage()))?;
    let yaml = std::fs::read_to_string(&anchor_path)
        .map_err(|e| format!("read anchor {}: {e}", anchor_path.display()))?;
    let anchor: patch::PatchState = serde_yaml::from_str(&yaml)
        .map_err(|e| format!("parse anchor {}: {e}", anchor_path.display()))?;
    let pieces = procedural::generate(
        &anchor,
        &procedural::GenerationConfig {
            seed,
            count,
            temperature,
            allow_black_sources,
        },
    )?;
    procedural::write_patch_only(&pieces, &output_path)
}

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    if args
        .get(1)
        .is_some_and(|arg| arg == "generate" || arg == "--generate")
    {
        match run_generation_cli(&args[2..]) {
            Ok(paths) => {
                for path in paths {
                    println!("{}", path.display());
                }
            }
            Err(error) => {
                eprintln!("Generation failed: {error}");
                std::process::exit(2);
            }
        }
        return;
    }
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
    fn output_policy_reuses_main_surface_without_a_distinct_monitor() {
        assert!(!use_dedicated_output(false, false));
        assert!(
            !use_dedicated_output(false, true),
            "an unknown main monitor is not proof of a distinct display"
        );
        assert!(!use_dedicated_output(true, false));
        assert!(use_dedicated_output(true, true));
    }

    #[test]
    fn absolute_output_commands_are_idempotent_and_legacy_toggle_is_parity() {
        assert!(resolve_output_window_command(
            false,
            OutputWindowCommand::Set(true)
        ));
        assert!(resolve_output_window_command(
            true,
            OutputWindowCommand::Set(true)
        ));
        assert!(!resolve_output_window_command(
            true,
            OutputWindowCommand::Set(false)
        ));
        assert!(resolve_output_window_command(
            false,
            OutputWindowCommand::Toggle
        ));
        assert!(!resolve_output_window_command(
            true,
            OutputWindowCommand::Toggle
        ));
    }

    #[test]
    fn held_discrete_window_keys_do_not_repeat_actions() {
        for key in [
            PhysicalKey::Code(KeyCode::Escape),
            PhysicalKey::Code(KeyCode::KeyF),
            PhysicalKey::Code(KeyCode::KeyO),
        ] {
            assert!(ignore_discrete_window_key_repeat(key, true));
            assert!(!ignore_discrete_window_key_repeat(key, false));
        }
        assert!(!ignore_discrete_window_key_repeat(
            PhysicalKey::Code(KeyCode::KeyP),
            true
        ));
    }

    #[test]
    fn single_monitor_output_suppresses_editor_panel() {
        assert!(show_editor_panel(false, true));
        assert!(!show_editor_panel(true, true));
        assert!(!show_editor_panel(false, false));
    }

    #[test]
    fn selective_rebuild_holds_paused_audience_until_resume() {
        assert!(!selective_rebuild_should_clear_audience(true));
        assert!(selective_rebuild_should_clear_audience(false));
    }

    #[test]
    fn paused_selective_edges_hold_disabled_and_legacy_in_both_directions() {
        for direct in [LiveNtscPath::Disabled, LiveNtscPath::LegacyGlobal] {
            assert!(is_selective_path_edge(
                direct,
                LiveNtscPath::SelectivePerLayer
            ));
            assert!(is_selective_path_edge(
                LiveNtscPath::SelectivePerLayer,
                direct
            ));

            // A paused edge retains slot2 through all subsequent paused
            // redraws, and cannot tag those held pixels as new direct/legacy
            // output for Spout or the global NTSC worker.
            assert!(!direct_path_may_replace_selective_hold(true, true));
            assert!(!raw_audience_readback_required(false, true, direct, true));

            // Resume permits the first proper direct render. Only after that
            // logical hold release may the matching path schedule readback.
            assert!(direct_path_may_replace_selective_hold(false, true));
            assert!(raw_audience_readback_required(false, false, direct, true));
            assert_eq!(
                raw_audience_readback_required(false, false, direct, false),
                direct == LiveNtscPath::LegacyGlobal
            );
        }
        assert!(!is_selective_path_edge(
            LiveNtscPath::Disabled,
            LiveNtscPath::LegacyGlobal
        ));
    }

    #[test]
    fn selective_hold_never_overrides_blackout_or_forges_spout_eligibility() {
        assert!(!raw_audience_readback_required(
            true,
            false,
            LiveNtscPath::Disabled,
            true
        ));
        let current = ntsc::SelectiveNtscGeneration {
            visual_epoch: 2,
            topology_generation: 5,
            width: 1280,
            height: 720,
            sample_sequence: 9,
        };
        assert!(!selective_spout_sample_is_eligible(
            current.visual_epoch,
            None,
            current,
            true,
            false,
        ));
        assert!(selective_spout_sample_is_eligible(
            current.visual_epoch,
            Some(current),
            current,
            true,
            false,
        ));
    }

    #[test]
    fn committed_topology_signature_is_the_single_generation_boundary() {
        assert_eq!(selective_topology_generation_after_signature(9, 9, 41), 41);
        assert_eq!(selective_topology_generation_after_signature(9, 10, 41), 42);
        assert_eq!(
            selective_topology_generation_after_signature(9, 10, u64::MAX),
            1
        );
    }

    #[test]
    fn simultaneous_path_and_topology_edge_is_one_generation_boundary() {
        assert!(selective_generation_boundary(true, true));
        assert!(selective_generation_boundary(true, false));
        assert!(selective_generation_boundary(false, true));
        assert!(!selective_generation_boundary(false, false));
    }

    #[test]
    fn paused_selective_blackout_restores_the_pre_cut_audience() {
        assert_eq!(
            blackout_audience_edge_action(true, false, false, false),
            Some(HeldAudienceAction::Capture),
            "every presented blackout edge captures slot2 before the absolute cut"
        );
        assert_eq!(
            blackout_audience_edge_action(true, true, true, true),
            None,
            "continued blackout must not overwrite the saved image with black"
        );
        assert_eq!(
            blackout_audience_edge_action(false, true, true, true),
            Some(HeldAudienceAction::Restore),
            "releasing blackout while paused restores the exact saved image"
        );
        assert_eq!(
            blackout_audience_edge_action(false, true, false, true),
            None,
            "a running release renders a fresh authoritative program frame"
        );
        assert_eq!(
            held_audience_action(false, false, true),
            HeldAudienceAction::Restore,
            "a selective rebuild after blackout must restore rather than recapture black"
        );
        assert!(!rendered_audience_may_discard_blackout_snapshot(
            false, true
        ));
        assert!(!rendered_audience_may_discard_blackout_snapshot(
            true, false
        ));
        assert!(rendered_audience_may_discard_blackout_snapshot(
            false, false
        ));
    }

    #[test]
    fn selective_spout_accepts_only_the_exact_current_audience_generation() {
        let current = ntsc::SelectiveNtscGeneration {
            visual_epoch: 8,
            topology_generation: 13,
            width: 1920,
            height: 1080,
            sample_sequence: 21,
        };
        let displayed = ntsc::SelectiveNtscGeneration {
            sample_sequence: 20,
            ..current
        };
        assert!(selective_spout_sample_is_eligible(
            current.visual_epoch,
            Some(displayed),
            current,
            true,
            false,
        ));
        assert!(!selective_spout_sample_is_eligible(
            current.visual_epoch,
            None,
            current,
            true,
            false,
        ));
        assert!(!selective_spout_sample_is_eligible(
            current.visual_epoch,
            Some(displayed),
            current,
            true,
            true,
        ));
        assert!(!selective_spout_sample_is_eligible(
            current.visual_epoch - 1,
            Some(displayed),
            current,
            true,
            false,
        ));
        assert!(!selective_spout_sample_is_eligible(
            current.visual_epoch,
            Some(ntsc::SelectiveNtscGeneration {
                topology_generation: current.topology_generation - 1,
                ..displayed
            }),
            current,
            true,
            false,
        ));
    }

    #[test]
    fn routing_target_identity_survives_index_drift_and_rejects_stale_ids() {
        // Action was authored against A/B/C when B occupied layer2. A is
        // removed before the queued action reaches the engine, so B is now
        // layer1. Its stable ID, not the stale positional spelling, wins.
        let resolved = resolve_routing_target_for_layer_ids(
            "layer2_opacity",
            &Some("20".into()),
            Some(7),
            [20, 30],
        );
        assert_eq!(resolved.as_deref(), Some("layer1_opacity"));

        // If B itself disappeared, never fall back to stale layer2 (C).
        assert_eq!(
            resolve_routing_target_for_layer_ids(
                "layer2_opacity",
                &Some("20".into()),
                Some(7),
                [10, 30],
            ),
            None
        );
        assert_eq!(
            resolve_routing_target_for_layer_ids(
                "layer2_opacity",
                &Some("not-an-id".into()),
                Some(7),
                [10, 20, 30],
            ),
            None
        );

        // Identity is forbidden for a master target; identity-free legacy
        // clients retain the old positional/master behavior.
        assert_eq!(
            resolve_routing_target_for_layer_ids(
                "brightness",
                &Some("20".into()),
                Some(7),
                [10, 20, 30],
            ),
            None
        );
        assert_eq!(
            resolve_routing_target_for_layer_ids("brightness", &None, None, [10, 20, 30])
                .as_deref(),
            Some("brightness")
        );
        assert_eq!(
            resolve_routing_target_for_layer_ids("layer2_opacity", &None, None, [10, 20, 30])
                .as_deref(),
            Some("layer2_opacity")
        );
    }

    #[test]
    fn routing_ids_reject_stale_actions_and_target_changes_reset_runtime_once() {
        let web_state = WebState::new().expect("test token");
        let mut app = App::new(None, None, web_state);
        app.mod_matrix.midi[0] = 1.0;
        app.mod_matrix.routings.push(modulation::Routing::new(
            modulation::ModSource::Midi(0),
            "brightness",
            0.25,
        ));
        app.mod_matrix.update_at_beat(0.0, 0.0);
        let id = app.mod_matrix.routings[0].route_id().to_string();
        assert_eq!(app.mod_matrix.routings[0].cached_value(), 1.0);

        app.handle_web_action(web::state::WebAction::SetRouting {
            index: 0,
            route_id: Some("999999999".into()),
            target_layer_id: None,
            layer_stack_revision: None,
            param: "depth".into(),
            value: serde_json::json!(1.0),
        });
        assert_eq!(app.mod_matrix.routings[0].depth, 0.25);

        app.handle_web_action(web::state::WebAction::SetRouting {
            index: 0,
            route_id: Some(id.clone()),
            target_layer_id: None,
            layer_stack_revision: None,
            param: "target".into(),
            value: serde_json::json!("brightness"),
        });
        assert_eq!(app.mod_matrix.routings[0].cached_value(), 1.0);

        for (param, value) in [("attack", 2.0), ("release", 3.0)] {
            app.handle_web_action(web::state::WebAction::SetRouting {
                index: 0,
                route_id: Some(id.clone()),
                target_layer_id: None,
                layer_stack_revision: None,
                param: param.into(),
                value: serde_json::json!(value),
            });
            assert_eq!(
                app.mod_matrix.routings[0].cached_value(),
                1.0,
                "editing {param} must preserve the live signal"
            );
        }

        app.handle_web_action(web::state::WebAction::SetRouting {
            index: 0,
            route_id: Some(id.clone()),
            target_layer_id: None,
            layer_stack_revision: None,
            param: "target".into(),
            value: serde_json::json!("contrast"),
        });
        assert_eq!(app.mod_matrix.routings[0].cached_value(), 0.0);

        app.handle_web_action(web::state::WebAction::RemoveRouting {
            index: 0,
            route_id: Some("999999999".into()),
        });
        assert_eq!(app.mod_matrix.routings.len(), 1);
        app.handle_web_action(web::state::WebAction::RemoveRouting {
            index: 0,
            route_id: Some(id),
        });
        assert!(app.mod_matrix.routings.is_empty());
    }

    #[test]
    fn program_clock_holds_exactly_while_paused_and_never_catches_up() {
        let mut clock = ProgramClock::default();
        assert_eq!(
            clock.tick(Duration::from_millis(40), false),
            (Duration::from_millis(40), 0.04)
        );
        assert_eq!(
            clock.tick(Duration::from_secs(30), true),
            (Duration::from_millis(40), 0.0)
        );
        assert_eq!(
            clock.tick(Duration::from_millis(20), false),
            (Duration::from_millis(60), 0.02)
        );
        clock.reset();
        assert_eq!(clock.elapsed, Duration::ZERO);
    }

    #[test]
    fn master_pause_edges_freeze_beat_and_duplicate_setters_are_noops() {
        let web_state = WebState::new().expect("test token");
        let mut app = App::new(None, None, web_state);
        let downbeat = Instant::now();
        app.mod_matrix.clock.tap(downbeat);
        let pause = downbeat + Duration::from_secs(2);
        app.set_master_paused_at(true, pause);
        let frozen = app.mod_matrix.clock.beat(pause);
        assert!(app.master_paused);
        assert!(app.mod_matrix.clock.is_paused());

        app.set_master_paused_at(true, pause + Duration::from_secs(50));
        assert_eq!(
            app.mod_matrix.clock.beat(pause + Duration::from_secs(50)),
            frozen
        );

        let resume = pause + Duration::from_secs(50);
        app.set_master_paused_at(false, resume);
        assert!(!app.master_paused);
        assert!(!app.mod_matrix.clock.is_paused());
        assert!((app.mod_matrix.clock.beat(resume) - frozen).abs() < 1e-9);
    }

    #[test]
    fn manual_morph_materializes_before_ordered_capture_even_while_paused() {
        let web_state = WebState::new().expect("test token");
        let mut app = App::new(None, None, web_state);
        let a = morph::MorphSlot::capture(
            &effects::EffectUniforms {
                brightness: 0.0,
                ..Default::default()
            },
            &ntsc::NtscParams::default(),
            &effects::params::TemporalParams::default(),
            &[],
        );
        let b = morph::MorphSlot::capture(
            &effects::EffectUniforms {
                brightness: 1.0,
                ..Default::default()
            },
            &ntsc::NtscParams::default(),
            &effects::params::TemporalParams::default(),
            &[],
        );
        app.morph.a = Some(a);
        app.morph.b = Some(b);
        app.master_paused = true;

        app.handle_web_action(web::state::WebAction::SetMorph { value: 0.8 });
        assert!((app.master_effects.brightness - 0.8).abs() < 1.0e-6);
        app.handle_web_action(web::state::WebAction::MorphCapture {
            slot: "a".into(),
            stack_revision: Some(app.layer_stack_revision),
        });
        assert!(
            (app.morph.a.as_ref().unwrap().master.brightness - 0.8).abs() < 1.0e-6,
            "capture must observe the preceding fader action, not the prior frame"
        );
    }

    #[test]
    fn stale_morph_capture_cannot_cross_a_layer_topology_generation() {
        let web_state = WebState::new().expect("test token");
        let mut app = App::new(None, None, web_state);
        app.master_effects.brightness = 0.7;
        let stale = app.layer_stack_revision;
        app.bump_layer_stack_revision();

        app.handle_web_action(web::state::WebAction::MorphCapture {
            slot: "a".into(),
            stack_revision: Some(stale),
        });
        assert!(app.morph.a.is_none());
    }

    #[test]
    fn beat_latched_capture_preserves_fader_order_and_is_never_coalesced() {
        let web_state = WebState::new().expect("test token");
        let mut app = App::new(None, None, web_state);
        app.queue_quantized_action(web::state::WebAction::SetMorph { value: 0.2 });
        app.queue_quantized_action(web::state::WebAction::MorphCapture {
            slot: "a".into(),
            stack_revision: Some(app.layer_stack_revision),
        });
        app.queue_quantized_action(web::state::WebAction::SetMorph { value: 0.8 });
        app.queue_quantized_action(web::state::WebAction::MorphCapture {
            slot: "a".into(),
            stack_revision: Some(app.layer_stack_revision),
        });

        assert_eq!(app.quantized_actions.len(), 4);
        assert!(matches!(
            app.quantized_actions.as_slice(),
            [
                web::state::WebAction::SetMorph { value: first },
                web::state::WebAction::MorphCapture { .. },
                web::state::WebAction::SetMorph { value: second },
                web::state::WebAction::MorphCapture { .. },
            ] if *first == 0.2 && *second == 0.8
        ));

        app.bump_layer_stack_revision();
        assert!(app
            .quantized_actions
            .iter()
            .all(|action| !matches!(action, web::state::WebAction::MorphCapture { .. })));
    }

    #[test]
    fn audio_clip_loading_freezes_the_internal_program_without_changing_master_pause() {
        let web_state = WebState::new().expect("test token");
        let mut app = App::new(None, None, web_state);
        let start = Instant::now();
        app.mod_matrix.clock.set_bpm_at(120.0, start);

        let load_start = start + Duration::from_secs(2);
        let beat_at_load = app.mod_matrix.clock.beat(load_start);
        app.set_audio_clip_blocks_program_at(true, load_start);

        assert!(
            !app.master_paused,
            "the browser Pause state stays authoritative"
        );
        assert!(app.program_transport_paused());
        assert!(app.mod_matrix.clock.is_paused());

        let slow_decode_completion = load_start + Duration::from_secs(45);
        assert_eq!(
            app.mod_matrix.clock.beat(slow_decode_completion),
            beat_at_load,
            "decoder latency must not advance beat/LFO/morph phase"
        );
        assert_eq!(
            app.program_clock
                .tick(Duration::from_secs(45), app.program_transport_paused()),
            (Duration::ZERO, 0.0),
            "decoder latency must not advance shader, audio, temporal, or NTSC time"
        );

        app.set_audio_clip_blocks_program_at(false, slow_decode_completion);
        assert!(!app.program_transport_paused());
        assert!(!app.mod_matrix.clock.is_paused());
        assert!((app.mod_matrix.clock.beat(slow_decode_completion) - beat_at_load).abs() < 1e-9);

        let first_render = slow_decode_completion + Duration::from_millis(16);
        assert!((app.mod_matrix.clock.beat(first_render) - (beat_at_load + 0.032)).abs() < 1e-6);
    }

    #[test]
    fn master_revert_is_complete_but_preserves_transport_inputs_and_layer_latches() {
        let web_state = WebState::new().expect("test token");
        let mut app = App::new(None, None, web_state);
        app.master_effects.brightness = 0.8;
        app.ntsc_params.enabled = true;
        app.temporal_params.feedback = 0.9;
        app.mod_matrix.clock.set_bpm(147.0);
        app.mod_matrix.audio_enabled = true;
        app.mod_matrix.audio_gain = 2.25;
        app.mod_matrix.audio_device = "preserved input".to_string();
        app.mod_matrix.midi_enabled = true;
        app.mod_matrix.midi_ccs = [21, 22, 23, 24];
        app.mod_matrix.routings.push(modulation::Routing::new(
            modulation::ModSource::Lfo(0),
            "brightness",
            1.0,
        ));
        let slot = morph::MorphSlot::capture(
            &app.master_effects,
            &app.ntsc_params,
            &app.temporal_params,
            &[],
        );
        app.morph.a = Some(slot.clone());
        app.morph.b = Some(slot);
        app.morph.start_glide(1.0, 8.0, 0.0);
        app.master_paused = true;
        app.quantized_actions = vec![
            web::state::WebAction::SetParam {
                param: "contrast".to_string(),
                value: serde_json::json!(0.5),
            },
            web::state::WebAction::SetNtscParam {
                param: "snow_intensity".to_string(),
                value: serde_json::json!(0.8),
            },
            web::state::WebAction::SetLayerEffect {
                index: 0,
                layer_id: None,
                param: "vignette".to_string(),
                value: serde_json::json!(0.4),
            },
        ];
        let old_epoch = app.visual_epoch;

        app.revert_master_visual_state();

        assert_eq!(app.master_effects.brightness, 0.0);
        assert_eq!(app.ntsc_params, ntsc::NtscParams::default());
        assert_eq!(app.temporal_params.feedback, 0.0);
        assert!(app.mod_matrix.routings.is_empty());
        assert!(app.morph.a.is_none());
        assert!(app.morph.b.is_none());
        assert!(app.morph.glide.is_none());
        assert_eq!(app.quantized_actions.len(), 1);
        assert!(matches!(
            app.quantized_actions[0],
            web::state::WebAction::SetLayerEffect { .. }
        ));
        assert_eq!(app.mod_matrix.clock.bpm, 147.0);
        assert!(app.mod_matrix.audio_enabled);
        assert_eq!(app.mod_matrix.audio_gain, 2.25);
        assert_eq!(app.mod_matrix.audio_device, "preserved input");
        assert!(app.mod_matrix.midi_enabled);
        assert_eq!(app.mod_matrix.midi_ccs, [21, 22, 23, 24]);
        assert!(app.master_paused);
        assert_ne!(app.visual_epoch, old_epoch);
    }

    #[test]
    fn patch_generation_discards_old_latches_and_advances_epochs() {
        let web_state = WebState::new().expect("test token");
        let mut app = App::new(None, None, web_state);
        let timing_anchor = Instant::now();
        app.mod_matrix.pad_config.spring_enabled = true;
        app.mod_matrix.pad_config.spring_rate = 4.0;
        app.mod_matrix.set_pad(1.0, 0.0, false);
        app.mod_matrix.update(timing_anchor);
        app.mod_matrix.current_beat = 9.25;
        let clock_probe = timing_anchor + Duration::from_secs(1);
        let clock_beat_before = app.mod_matrix.clock.beat(clock_probe);
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
        app.selective_transition_holding = true;
        app.selective_hold_snapshot_valid = true;
        app.selective_hold_spout_barrier_epoch = Some(11);
        app.selective_hold_spout_readback_epoch = Some(11);
        app.blackout = true;
        app.blackout_presented = false;
        let old_stack_revision = app.layer_stack_revision;
        let old_visual_epoch = app.visual_epoch;

        app.reset_patch_generation();

        assert!(app.quantized_actions.is_empty());
        assert!(app.mod_matrix.midi_learn.is_none());
        assert_eq!(app.mod_matrix.current_beat, 9.25);
        assert_eq!(app.mod_matrix.clock.beat(clock_probe), clock_beat_before);
        assert!(app.web_state.actions.blocking_lock().is_empty());
        assert_eq!(app.quantized_bar, Some(2));
        assert!(app.ntsc_presented.is_none());
        assert!(!app.selective_transition_holding);
        assert!(!app.selective_hold_snapshot_valid);
        assert!(app.selective_hold_spout_barrier_epoch.is_none());
        assert!(app.selective_hold_spout_readback_epoch.is_none());
        assert!(app.blackout_presented);
        assert_ne!(app.layer_stack_revision, old_stack_revision);
        assert_ne!(app.visual_epoch, old_visual_epoch);

        // The first post-commit frame observes zero elapsed modulation time,
        // regardless of how long patch reconstruction took.
        app.mod_matrix
            .update(timing_anchor + Duration::from_secs(30));
        assert_eq!(app.mod_matrix.pad, [1.0, 0.0]);
    }

    #[test]
    fn manual_bpm_change_does_not_release_quantized_action_before_downbeat() {
        let web_state = WebState::new().expect("test token");
        let mut app = App::new(None, None, web_state);
        let downbeat = Instant::now();
        let change = downbeat + Duration::from_millis(1_950);
        app.mod_matrix.clock.tap(downbeat);
        app.mod_matrix.update(change);
        assert!((app.mod_matrix.current_beat - 3.9).abs() < 1e-9);

        app.master_effects.brightness = 0.0;
        app.quantized_bar = Some(0);
        app.queue_quantized_action(web::state::WebAction::SetParam {
            param: "brightness".to_string(),
            value: serde_json::json!(0.75),
        });

        app.mod_matrix.clock.set_bpm_at(300.0, change);
        app.mod_matrix.update(change + Duration::from_millis(10));
        app.release_quantized_actions_on_downbeat();
        assert_eq!(app.quantized_actions.len(), 1);
        assert_eq!(app.master_effects.brightness, 0.0);

        app.mod_matrix.update(change + Duration::from_millis(25));
        app.release_quantized_actions_on_downbeat();
        assert!(app.quantized_actions.is_empty());
        assert_eq!(app.master_effects.brightness, 0.75);
        assert_eq!(app.quantized_bar, Some(1));
    }

    #[test]
    fn backwards_clock_reanchor_does_not_release_quantized_actions() {
        let web_state = WebState::new().expect("test token");
        let mut app = App::new(None, None, web_state);
        app.master_effects.brightness = 0.0;
        app.quantized_bar = Some(3);
        app.mod_matrix.current_beat = 0.0;
        app.queue_quantized_action(web::state::WebAction::SetParam {
            param: "brightness".to_string(),
            value: serde_json::json!(0.75),
        });

        app.release_quantized_actions_on_downbeat();

        assert_eq!(app.quantized_actions.len(), 1);
        assert_eq!(app.master_effects.brightness, 0.0);
        assert_eq!(app.quantized_bar, Some(0));
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
    fn stopped_gyro_stream_is_recentered_before_render_modulation() {
        let web_state = WebState::new().expect("test token");
        let mut app = App::new(None, None, web_state.clone());
        web_state.set_gyro_stream(17, true);
        web_state.note_gyro_sample(17);
        app.mod_matrix.set_gyro_degrees(45.0, 30.0, -20.0);
        let live_pose = app.mod_matrix.gyro;
        assert_ne!(live_pose, [0.5; 3]);

        app.release_stale_gyro();
        assert_eq!(app.mod_matrix.gyro, live_pose);

        web_state.set_gyro_stream(17, false);
        app.release_stale_gyro();
        assert_eq!(app.mod_matrix.gyro, [0.5; 3]);
        assert_eq!(app.mod_matrix.gyro_raw, [0.0; 3]);
    }

    #[test]
    fn removing_an_earlier_layer_preserves_selected_identity() {
        assert_eq!(selected_layer_after_remove(Some(2), 0, 2), Some(1));
        assert_eq!(selected_layer_after_remove(Some(0), 0, 2), Some(0));
        assert_eq!(selected_layer_after_remove(Some(2), 2, 2), Some(1));
        assert_eq!(selected_layer_after_remove(Some(0), 0, 0), None);
    }
}
