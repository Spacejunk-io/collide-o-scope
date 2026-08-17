#![allow(deprecated)] // egui 0.34 deprecation warnings for panel API renames

mod audio;
mod effects;
mod input;
mod layers;
mod media_safety;
mod media_source;
mod midi;
mod modulation;
mod morph;
mod ntsc;
mod patch;
mod procedural;
mod randomization;
mod render_export;
mod renderer;
mod spout_in;
mod spout_out;
mod video;
mod web;

use std::collections::VecDeque;
use std::io::Read as _;
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
use layers::{is_still_image_file, is_supported_visual_file, Layer};
use renderer::Renderer;
use web::state::{ControlServerInfo, ControlServerStatus, WebState};

const TARGET_FPS: u64 = 30;
const FRAME_DURATION: Duration = Duration::from_millis(1000 / TARGET_FPS);
const FALLBACK_OUTPUT_WIDTH: u32 = 1280;
const FALLBACK_OUTPUT_HEIGHT: u32 = 720;

const fn renderer_recovery_output(width: u32, height: u32) -> Option<(u32, u32)> {
    if width == FALLBACK_OUTPUT_WIDTH && height == FALLBACK_OUTPUT_HEIGHT {
        None
    } else {
        Some((FALLBACK_OUTPUT_WIDTH, FALLBACK_OUTPUT_HEIGHT))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputWindowCommand {
    Set(bool),
    Toggle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AppliedLookScope {
    mapped_layer_ids: Vec<u64>,
    applied_ntsc: bool,
    applied_temporal: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum WebActionBatchDisposition {
    Continue,
    SnapshotCommitted,
    LookApplied(AppliedLookScope),
}

enum StagedPatchAudio {
    /// Legacy patches without a modulation section leave the live analysis
    /// source and any in-flight clip generation untouched.
    Preserve,
    /// An explicit non-file (or empty file) source owns no decoded clip.
    Clear,
    /// Exact snapshot recall publishes only a clip that has already decoded.
    Loaded {
        resolved_path: PathBuf,
        persisted_source_reference: Option<String>,
        clip: audio::AudioClip,
    },
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
            | PhysicalKey::Code(KeyCode::KeyM)
            | PhysicalKey::Code(KeyCode::KeyO)
    )
}

fn ignore_discrete_window_key_repeat(key: PhysicalKey, repeat: bool) -> bool {
    repeat && is_discrete_window_key(key)
}

const fn show_editor_panel(output_on_main: bool, editor_active: bool) -> bool {
    editor_active && !output_on_main
}

/// Single-monitor audience Output reuses the main swapchain, so every native
/// control must disappear there just like the YAML editor does. A dedicated
/// output has its own clean surface and may leave this strip on the preview.
const fn show_native_recovery_strip(output_on_main: bool) -> bool {
    !output_on_main
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeRecoveryAction {
    SetProgramFrozen(bool),
    SetBlackout(bool),
    RevertVisualProgram,
    RescanLibrary,
    ChooseLibrary,
}

struct NativeRecoveryView {
    control_server: ControlServerInfo,
    browser_connections: usize,
    program_frozen: bool,
    blackout: bool,
    library_folder: Option<PathBuf>,
    visual_files: usize,
    audio_files: usize,
    output_status: String,
    media_status: String,
}

fn build_native_recovery_strip(
    ctx: &egui::Context,
    view: &NativeRecoveryView,
    actions: &mut Vec<NativeRecoveryAction>,
) {
    egui::TopBottomPanel::top("native_recovery_strip")
        .resizable(false)
        .show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.strong("RECOVERY");

                let (server_text, server_color, server_error) = match &view.control_server.status {
                    ControlServerStatus::NotStarted => (
                        "Panel server not started".to_string(),
                        egui::Color32::YELLOW,
                        None,
                    ),
                    ControlServerStatus::Starting => (
                        "Panel server starting".to_string(),
                        egui::Color32::YELLOW,
                        None,
                    ),
                    ControlServerStatus::Listening => (
                        "Panel server ready".to_string(),
                        egui::Color32::LIGHT_GREEN,
                        None,
                    ),
                    ControlServerStatus::Unavailable(error) => (
                        "Panel server unavailable".to_string(),
                        egui::Color32::LIGHT_RED,
                        Some(error.as_str()),
                    ),
                };
                let status = ui.colored_label(server_color, server_text);
                if let Some(error) = server_error {
                    status.on_hover_text(error);
                    ui.colored_label(egui::Color32::LIGHT_RED, format!("({error})"));
                }

                if !view.control_server.local_url.is_empty() {
                    ui.hyperlink_to("Open Panel", &view.control_server.local_url)
                        .on_hover_text("Open the authenticated loopback control panel");
                }
                ui.weak(match view.browser_connections {
                    0 => "No browser connected".to_string(),
                    1 => "1 browser connected".to_string(),
                    count => format!("{count} browsers connected"),
                });

                ui.separator();
                if ui
                    .selectable_label(view.program_frozen, "Freeze Program")
                    .on_hover_text("Hold the complete visual program without catch-up")
                    .clicked()
                {
                    actions.push(NativeRecoveryAction::SetProgramFrozen(!view.program_frozen));
                }
                if ui
                    .selectable_label(view.blackout, "Blackout")
                    .on_hover_text("Cut every audience output to absolute black")
                    .clicked()
                {
                    actions.push(NativeRecoveryAction::SetBlackout(!view.blackout));
                }
                if ui
                    .button("Revert Visuals")
                    .on_hover_text("Revert the complete master visual program")
                    .clicked()
                {
                    actions.push(NativeRecoveryAction::RevertVisualProgram);
                }

                ui.separator();
                let folder_path = view
                    .library_folder
                    .as_ref()
                    .map(|folder| folder.display().to_string())
                    .unwrap_or_else(|| "No library selected".to_string());
                let folder_name = view
                    .library_folder
                    .as_ref()
                    .and_then(|folder| folder.file_name())
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_else(|| folder_path.clone());
                ui.label(format!(
                    "Library: {folder_name} ({} visual, {} audio)",
                    view.visual_files, view.audio_files
                ))
                .on_hover_text(folder_path);
                if ui.button("Choose Library").clicked() {
                    actions.push(NativeRecoveryAction::ChooseLibrary);
                }
                if ui
                    .add_enabled(view.library_folder.is_some(), egui::Button::new("Rescan"))
                    .clicked()
                {
                    actions.push(NativeRecoveryAction::RescanLibrary);
                }
            });
            if !view.output_status.is_empty() || !view.media_status.is_empty() {
                ui.horizontal_wrapped(|ui| {
                    ui.strong("Status");
                    if !view.output_status.is_empty() {
                        ui.colored_label(egui::Color32::LIGHT_RED, &view.output_status);
                    }
                    if !view.media_status.is_empty() {
                        ui.label(&view.media_status);
                    }
                });
            }
        });
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

/// The selective GPU stage only owns an admission attempt once a readback can
/// actually begin. A healthy busy worker and an unavailable worker both close
/// that gate, but they must update different counters.
fn selective_ntsc_gpu_gate_is_open(
    outcome: ntsc::NtscSubmitOutcome,
    metrics: &mut ntsc::NtscPathMetrics,
) -> bool {
    match outcome {
        ntsc::NtscSubmitOutcome::Accepted => true,
        ntsc::NtscSubmitOutcome::Busy | ntsc::NtscSubmitOutcome::Unavailable => {
            metrics.record_admission(outcome);
            false
        }
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TransportGates {
    program_running: bool,
    media_running: bool,
}

struct RerollRequest {
    scope: web::state::RerollScope,
    index: Option<usize>,
    layer_id: Option<String>,
    stack_revision: Option<u64>,
    supplied_seed: Option<u32>,
    mode: web::state::RerollMode,
    amount: f32,
    include_grain_controls: bool,
}

fn transport_gates(
    program_frozen: bool,
    media_frozen: bool,
    audio_clip_blocks_program: bool,
) -> TransportGates {
    let program_running = !(program_frozen || audio_clip_blocks_program);
    TransportGates {
        program_running,
        media_running: program_running && !media_frozen,
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
                .filter(|layer| *layer != 0)
                .map(|_| suffix)
        });

    match target_layer_id {
        Some(id) => {
            let suffix = layer_target?;
            let stable_id = id.parse::<u64>().ok().filter(|id| *id != 0)?;
            let current_index = layer_ids
                .into_iter()
                .position(|candidate| candidate == stable_id)?;
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
    media_frozen: bool,
    last_frame_time: Instant,
    program_clock: ProgramClock,
    modifiers: ModifiersState,
    // Library
    library_folder: Option<PathBuf>,
    library_files: Vec<PathBuf>,
    audio_library_files: Vec<PathBuf>,
    /// Host-session-only source admission policy. Patches deliberately cannot
    /// raise this boundary, and switching back to Safe affects future opens
    /// without destroying sources that already hold an Expert reservation.
    media_safety_policy: media_safety::MediaSafetyPolicy,
    media_safety_status: String,
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
    /// Process-lifetime diagnostics stay path-specific because global and
    /// selective VHS shed work at different bounded pipeline stages.
    ntsc_live_metrics: ntsc::LiveNtscMetrics,
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
    patch_load_status: String,
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
        let media_safety_policy = media_safety::MediaSafetyPolicy::default();

        // The web server needs the folder for clip uploads.
        if let Ok(mut lf) = web_state.library_folder.write() {
            *lf = library_folder.clone();
        }

        let library_generation = web_state.begin_library_generation();
        // Generate thumbnails on background thread
        generate_thumbnails(
            &library_files,
            web_state.clone(),
            library_generation,
            media_safety_policy.clone(),
            media_safety::MediaDeviceLimits::none(),
        );

        Self {
            initial_video,
            window: None,
            renderer: None,
            layers: Vec::new(),
            selected_layer: None,
            layer_stack_revision: 1,
            master_effects: effects::EffectUniforms::default(),
            master_paused: false,
            media_frozen: false,
            last_frame_time: Instant::now(),
            program_clock: ProgramClock::default(),
            modifiers: ModifiersState::empty(),
            library_folder,
            library_files,
            audio_library_files,
            media_safety_policy,
            media_safety_status: String::new(),
            yaml_editor: patch::editor::EditorState::default(),
            egui_ctx: egui::Context::default(),
            egui_winit: None,
            egui_renderer: None,
            video_egui_texture_id: None,
            ntsc_params: ntsc::NtscParams::default(),
            ntsc_worker: ntsc::NtscWorker::new(),
            ntsc_live_metrics: ntsc::LiveNtscMetrics::default(),
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
            patch_load_status: String::new(),
            export_job: None,
            quantized_actions: Vec::new(),
            quantized_bar: None,
        }
    }

    fn add_layer(&mut self, path: &str) {
        let renderer = self.renderer.as_ref().unwrap();
        match Layer::new_with_media_policy(path, &renderer.device, &self.media_safety_policy) {
            Ok(layer) => {
                let filename = layer.filename.clone();
                self.layers.push(layer);
                self.selected_layer = Some(self.layers.len() - 1);
                self.bump_layer_stack_revision();
                self.media_safety_status = format!("Loaded {filename}");
                // Appending leaves every captured position valid. The new
                // source is deliberately untouched by an existing morph.
            }
            Err(e) => {
                let filename = std::path::Path::new(path)
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.to_string());
                self.media_safety_status = format!("Rejected {filename}: {e}");
                eprintln!("Failed to open video: {e}");
            }
        }
    }

    fn add_spout_layer(&mut self, sender_name: &str) {
        let Some(renderer) = self.renderer.as_ref() else {
            log::error!("Cannot add Spout input before the renderer is ready");
            return;
        };
        match Layer::new_spout_with_media_policy(
            sender_name,
            &renderer.device,
            &renderer.queue,
            &self.media_safety_policy,
        ) {
            Ok(layer) => {
                self.layers.push(layer);
                self.selected_layer = Some(self.layers.len() - 1);
                self.bump_layer_stack_revision();
                self.media_safety_status = format!("Listening for Spout sender {sender_name}");
                // Appending leaves every captured position valid.
            }
            Err(error) => {
                self.media_safety_status = format!("Rejected Spout sender {sender_name}: {error}");
                log::error!("Failed to create Spout input layer: {error}");
            }
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
            patch::PatchTransportState {
                master_paused: self.master_paused,
                media_frozen: self.media_frozen,
            },
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
        let library_files = scan_folder(&folder);
        let audio_library_files = scan_audio_folder(&folder);
        let generation = self.web_state.begin_library_generation();

        // Cache keys are portable basenames, so changing folders can otherwise
        // show media from the old folder when both contain `clip.mp4`. Advance
        // the generation first, then clear; old workers double-check the
        // generation while holding the cache lock and cannot repopulate it.
        self.web_state.clear_library_media_caches();

        if let Ok(mut lf) = self.web_state.library_folder.write() {
            *lf = Some(folder.clone());
        }
        self.library_folder = Some(folder);
        self.library_files = library_files;
        self.audio_library_files = audio_library_files;
        let device_limits =
            self.renderer
                .as_ref()
                .map_or_else(media_safety::MediaDeviceLimits::none, |renderer| {
                    let limits = renderer.device.limits();
                    media_safety::MediaDeviceLimits::new(
                        limits.max_texture_dimension_2d,
                        limits.max_buffer_size,
                    )
                });
        generate_thumbnails(
            &self.library_files,
            self.web_state.clone(),
            generation,
            self.media_safety_policy.clone(),
            device_limits,
        );
    }

    fn apply_native_recovery_action(&mut self, action: NativeRecoveryAction) {
        use web::state::WebAction;
        match action {
            NativeRecoveryAction::SetProgramFrozen(frozen) => {
                self.handle_web_action(WebAction::SetProgramFrozen { frozen });
            }
            NativeRecoveryAction::SetBlackout(enabled) => {
                self.handle_web_action(WebAction::SetBlackout { enabled });
            }
            NativeRecoveryAction::RevertVisualProgram => {
                self.handle_web_action(WebAction::ResetVisualProgram);
            }
            NativeRecoveryAction::RescanLibrary => {
                self.handle_web_action(WebAction::RescanLibrary);
            }
            NativeRecoveryAction::ChooseLibrary => self.choose_library_folder(),
        }
    }

    fn choose_library_folder(&mut self) {
        let mut dialog = rfd::FileDialog::new().set_title("Choose media library folder");
        if let Some(folder) = self.library_folder.as_ref() {
            dialog = dialog.set_directory(folder);
        }

        let modal_started = Instant::now();
        let was_program_paused = self.program_transport_paused();
        self.mod_matrix.clock.set_paused(true, modal_started);
        let folder = dialog.pick_folder();
        self.finish_native_modal(was_program_paused, Instant::now());

        if let Some(folder) = folder {
            self.set_library_folder(folder);
        }
    }

    /// Native dialogs are modal on the event thread. Preserve the beat that
    /// was visible when the dialog opened and discard its wall-clock duration
    /// from frame/video pacers, without changing the user-facing Freeze state.
    fn finish_native_modal(&mut self, was_program_paused: bool, now: Instant) {
        self.mod_matrix.clock.set_paused(was_program_paused, now);
        self.mod_matrix.reset_update_timing();
        for layer in &mut self.layers {
            layer.reset_transport_timing_at(now);
        }
        self.last_frame_time = now;
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

    fn stage_patch_analysis_audio(
        patch: &patch::PatchState,
        source_context: &media_source::ResolveContext,
        source_fingerprints: &mut media_source::FingerprintSession,
    ) -> Result<StagedPatchAudio, String> {
        let Some(modulation) = patch.modulation.as_ref() else {
            return Ok(StagedPatchAudio::Preserve);
        };
        if modulation::normalize_audio_source_kind(&modulation.audio_source_kind)
            != modulation::AUDIO_SOURCE_FILE
            || modulation.audio_clip_path.is_empty()
        {
            return Ok(StagedPatchAudio::Clear);
        }

        let requested = &modulation.audio_clip_path;
        let content_identity =
            media_source::parse_content_reference(requested).map_err(|error| error.to_string())?;
        let logical_name = if content_identity.is_some() {
            String::new()
        } else {
            PathBuf::from(requested)
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| requested.clone())
        };
        let resolved = media_source::resolve_file_source(
            requested,
            &logical_name,
            source_context,
            None,
            |path: &std::path::Path| audio::is_supported_audio_file(path),
            source_fingerprints,
        )
        .map_err(|error| format!("analysis audio: {error}"))?;
        let clip = audio::AudioClip::open(&resolved.path)
            .map_err(|error| format!("analysis audio: {error}"))?;
        Ok(StagedPatchAudio::Loaded {
            resolved_path: resolved.path,
            persisted_source_reference: content_identity.map(|_| requested.clone()),
            clip,
        })
    }

    fn selected_audio_clip_name(&self) -> String {
        PathBuf::from(&self.mod_matrix.audio_clip_path)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.mod_matrix.audio_clip_path.clone())
    }

    fn request_selected_audio_clip(&mut self) {
        let requested = self.mod_matrix.audio_clip_path.clone();
        if media_source::parse_content_reference(&requested)
            .ok()
            .flatten()
            .is_some()
        {
            self.mod_matrix.audio_clip_source_reference = Some(requested.clone());
        }
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
        let patch_dir = patch_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));
        let source_context = media_source::ResolveContext::new(
            Some(patch_dir.to_path_buf()),
            self.library_folder.clone(),
        );
        let mut source_fingerprints =
            media_source::FingerprintSession::new(media_source::FingerprintLimits::default())
                .map_err(|error| error.to_string())?;
        let mut rebuilt = Vec::with_capacity(patch.layers.len());

        for config in &patch.layers {
            let persisted_source_reference =
                media_source::parse_content_reference(&config.source_path)
                    .map_err(|error| format!("{}: {error}", config.filename))?
                    .map(|_| config.source_path.clone());
            let renderer = self.renderer.as_ref().ok_or("renderer is not ready")?;
            let resolved = media_source::resolve_visual_source(
                &config.source_path,
                &config.filename,
                &source_context,
                None,
                is_supported_visual_file,
                &mut source_fingerprints,
            )
            .map_err(|error| format!("{}: {error}", config.filename))?;
            if let media_source::ResolvedVisualSource::Spout { sender } = resolved {
                let mut layer = Layer::new_spout_with_media_policy(
                    &sender,
                    &renderer.device,
                    &renderer.queue,
                    &self.media_safety_policy,
                )
                .map_err(|error| format!("{}: {error}", config.filename))?;
                config.apply_to_layer(&mut layer);
                layer.set_persisted_source_reference(persisted_source_reference);
                rebuilt.push(layer);
                continue;
            }
            let media_source::ResolvedVisualSource::File(resolved) = resolved else {
                unreachable!("Spout sources returned above")
            };
            let path = resolved.path;
            let path_text = path.to_string_lossy();
            let mut layer = Layer::new_with_media_policy(
                &path_text,
                &renderer.device,
                &self.media_safety_policy,
            )
            .map_err(|e| format!("{}: {e}", config.filename))?;
            config.apply_to_layer(&mut layer);
            layer.set_persisted_source_reference(persisted_source_reference);
            rebuilt.push(layer);
        }

        // Audio decoding is part of the same staging transaction as visual
        // decoder construction. A missing or corrupt saved clip therefore
        // rejects the snapshot before any live performance state is replaced.
        let staged_analysis_audio =
            Self::stage_patch_analysis_audio(&patch, &source_context, &mut source_fingerprints)?;

        patch.apply(
            &mut self.master_effects,
            &mut rebuilt,
            &mut self.ntsc_params,
            &mut self.mod_matrix,
            &mut self.temporal_params,
        );
        let recalled_master_paused = patch.master_paused;
        let recalled_media_frozen = patch.media_frozen;
        let preserve_audio_load_block =
            matches!(&staged_analysis_audio, StagedPatchAudio::Preserve)
                && self.audio_clip_blocks_program;
        // Patch modulation replacement may leave the old clock's paused flag
        // behind when loading a legacy patch with no modulation section.
        // Start from a running logical clock, then apply the recalled absolute
        // transport state at the common commit boundary below.
        self.mod_matrix.clock.set_paused(false, Instant::now());
        self.master_paused = false;
        self.media_frozen = false;
        // All decoders were opened sequentially, but playback begins at one
        // atomic commit boundary. Re-anchor every pacer to the same instant so
        // early layers do not begin with artificial catch-up debt.
        let commit_time = Instant::now();
        for layer in &mut rebuilt {
            layer.reset_transport_timing_at(commit_time);
        }
        self.mod_matrix.clock.set_paused(
            recalled_master_paused || preserve_audio_load_block,
            commit_time,
        );
        self.master_paused = recalled_master_paused;
        self.media_frozen = recalled_media_frozen;
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
        match staged_analysis_audio {
            StagedPatchAudio::Preserve => {}
            StagedPatchAudio::Clear => {
                self.audio_clip_loader.cancel();
                self.audio_clip = None;
                self.audio_clip_spectrum.fill(0.0);
                self.audio_clip_error.clear();
                self.set_audio_clip_blocks_program_at(false, commit_time);
            }
            StagedPatchAudio::Loaded {
                resolved_path,
                persisted_source_reference,
                clip,
            } => {
                self.audio_clip_loader.cancel();
                self.mod_matrix.audio_clip_path = resolved_path.to_string_lossy().into_owned();
                self.mod_matrix.audio_clip_source_reference = persisted_source_reference;
                self.audio_clip = Some(clip);
                self.audio_clip_spectrum.fill(0.0);
                self.audio_clip_error.clear();
                self.set_audio_clip_blocks_program_at(false, commit_time);
            }
        }
        Ok(())
    }

    fn layer_action_targets_look(
        index: usize,
        layer_id: &Option<String>,
        mapped_layer_ids: &[u64],
    ) -> bool {
        match layer_id
            .as_deref()
            .filter(|id| !id.is_empty())
            .and_then(|id| id.parse::<u64>().ok())
        {
            Some(id) => mapped_layer_ids.contains(&id),
            None => index < mapped_layer_ids.len(),
        }
    }

    fn action_conflicts_with_applied_look(
        action: &web::state::WebAction,
        mapped_layer_ids: &[u64],
        applied_ntsc: bool,
        applied_temporal: bool,
    ) -> bool {
        use web::state::WebAction;
        match action {
            WebAction::Quantized { inner } => Self::action_conflicts_with_applied_look(
                inner,
                mapped_layer_ids,
                applied_ntsc,
                applied_temporal,
            ),
            WebAction::ResetGroup { group } => match group.as_str() {
                "digital" | "analog" | "key" | "motion" | "cellular" | "shift" => true,
                "vhs" => applied_ntsc,
                "temporal" => applied_temporal,
                // Apply Look deliberately preserves modulation. Unknown reset
                // groups are no-ops and likewise cannot conflict.
                "mod" => false,
                _ => false,
            },
            WebAction::SetParam { .. }
            | WebAction::ResetFx
            | WebAction::ResetVisualProgram
            | WebAction::AddLayer { .. }
            | WebAction::AddSpoutLayer { .. }
            | WebAction::RemoveLayer { .. }
            | WebAction::MoveLayer { .. }
            | WebAction::MorphCapture { .. }
            | WebAction::MorphClear
            | WebAction::SetMorph { .. }
            | WebAction::SetMorphLaw { .. }
            | WebAction::MorphGlide { .. }
            | WebAction::OpenPatchSnapshot
            | WebAction::OpenPatchLook { .. } => true,
            WebAction::SetNtscParam { .. } => applied_ntsc,
            WebAction::SetTemporal { .. } => applied_temporal,
            WebAction::SetLayerEffect {
                index, layer_id, ..
            }
            | WebAction::ResetLayerFx { index, layer_id }
            | WebAction::SetLayerVisibility {
                index, layer_id, ..
            } => Self::layer_action_targets_look(*index, layer_id, mapped_layer_ids),
            WebAction::SetLayerParam {
                index,
                layer_id,
                param,
                ..
            } if matches!(
                param.as_str(),
                "opacity"
                    | "blend_mode"
                    | "visible"
                    | "bypass_master_fx"
                    | "key_mode"
                    | "key_threshold"
                    | "key_softness"
                    | "key_color_r"
                    | "key_color_g"
                    | "key_color_b"
                    | "key_tolerance"
            ) =>
            {
                Self::layer_action_targets_look(*index, layer_id, mapped_layer_ids)
            }
            WebAction::ToggleVisibility { index } => *index < mapped_layer_ids.len(),
            WebAction::Reroll {
                scope: web::state::RerollScope::Master | web::state::RerollScope::All,
                ..
            } => true,
            WebAction::Reroll {
                scope: web::state::RerollScope::Layer,
                index,
                layer_id,
                ..
            } => index.is_some_and(|index| {
                Self::layer_action_targets_look(index, layer_id, mapped_layer_ids)
            }),
            _ => false,
        }
    }

    /// Invalidate retained pixels/history after a bulk visual transfer without
    /// resetting piece time, modulation, source identity, or stack revision.
    fn invalidate_visual_generation_after_look(
        &mut self,
        mapped_layer_ids: &[u64],
        applied_ntsc: bool,
        applied_temporal: bool,
    ) {
        self.quantized_actions.retain(|action| {
            !Self::action_conflicts_with_applied_look(
                action,
                mapped_layer_ids,
                applied_ntsc,
                applied_temporal,
            )
        });
        self.web_state.actions.blocking_lock().retain(|action| {
            !Self::action_conflicts_with_applied_look(
                action,
                mapped_layer_ids,
                applied_ntsc,
                applied_temporal,
            )
        });
        self.visual_epoch = self.visual_epoch.wrapping_add(1);
        self.ntsc_presented = None;
        self.selective_temporal_debt = 0.0;
        self.selective_transition_holding = false;
        self.selective_hold_snapshot_valid = false;
        self.selective_hold_spout_barrier_epoch = None;
        self.selective_hold_spout_readback_epoch = None;
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.reset_visual_generation();
        }
    }

    fn apply_patch_look(
        &mut self,
        patch: patch::PatchState,
    ) -> (patch::LookApplySummary, AppliedLookScope) {
        self.release_active_morph_for_manual_edit();
        let applied_ntsc = patch.ntsc.is_some();
        let applied_temporal = patch.temporal.is_some();
        let summary = patch.apply_look(
            &mut self.master_effects,
            &mut self.layers,
            &mut self.ntsc_params,
            &mut self.temporal_params,
        );
        let mapped_layer_ids: Vec<_> = self
            .layers
            .iter()
            .take(summary.mapped_layers)
            .map(Layer::layer_id)
            .collect();
        self.invalidate_visual_generation_after_look(
            &mapped_layer_ids,
            applied_ntsc,
            applied_temporal,
        );
        (
            summary,
            AppliedLookScope {
                mapped_layer_ids,
                applied_ntsc,
                applied_temporal,
            },
        )
    }

    fn choose_snapshot_patch(&mut self) -> bool {
        match patch::editor::choose_patch() {
            Ok(Some((loaded, path))) => match self.apply_loaded_patch(loaded, &path) {
                Ok(()) => {
                    self.patch_load_status = format!("Loaded snapshot {}", path.display());
                    log::info!("{}", self.patch_load_status);
                    true
                }
                Err(error) => {
                    self.patch_load_status = format!("Error: {error}");
                    log::error!("Failed to apply patch snapshot: {error}");
                    false
                }
            },
            Ok(None) => {
                self.patch_load_status = "Snapshot load cancelled".to_string();
                false
            }
            Err(error) => {
                self.patch_load_status = format!("Error: {error}");
                log::error!("{}", self.patch_load_status);
                false
            }
        }
    }

    fn choose_and_apply_patch_look(
        &mut self,
        expected_stack_revision: Option<u64>,
    ) -> Option<AppliedLookScope> {
        if expected_stack_revision.is_some_and(|revision| revision != self.layer_stack_revision) {
            self.patch_load_status = "Error: layer stack changed; choose Apply Look again".into();
            return None;
        }
        let revision_before_dialog = self.layer_stack_revision;
        match patch::editor::choose_patch() {
            Ok(Some((loaded, path))) => {
                if self.layer_stack_revision != revision_before_dialog {
                    self.patch_load_status =
                        "Error: layer stack changed while choosing the look".into();
                    return None;
                }
                let (summary, scope) = self.apply_patch_look(loaded);
                self.patch_load_status = format!(
                    "Applied {} to {} layer{}; {} current unchanged; {} saved unused",
                    path.file_name()
                        .map(|name| name.to_string_lossy())
                        .unwrap_or_else(|| path.as_os_str().to_string_lossy()),
                    summary.mapped_layers,
                    if summary.mapped_layers == 1 { "" } else { "s" },
                    summary.untouched_live_layers,
                    summary.unused_patch_layers,
                );
                log::info!("{}", self.patch_load_status);
                Some(scope)
            }
            Ok(None) => {
                self.patch_load_status = "Apply Look cancelled".to_string();
                None
            }
            Err(error) => {
                self.patch_load_status = format!("Error: {error}");
                log::error!("{}", self.patch_load_status);
                None
            }
        }
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

    fn apply_web_action_batch_disposition(
        pending: &mut VecDeque<web::state::WebAction>,
        disposition: WebActionBatchDisposition,
    ) {
        match disposition {
            WebActionBatchDisposition::Continue => {}
            WebActionBatchDisposition::SnapshotCommitted => pending.clear(),
            WebActionBatchDisposition::LookApplied(scope) => pending.retain(|action| {
                !Self::action_conflicts_with_applied_look(
                    action,
                    &scope.mapped_layer_ids,
                    scope.applied_ntsc,
                    scope.applied_temporal,
                )
            }),
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
        !transport_gates(
            self.master_paused,
            self.media_frozen,
            self.audio_clip_blocks_program,
        )
        .program_running
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

    fn set_media_frozen_at(&mut self, frozen: bool, now: Instant) {
        if self.media_frozen == frozen {
            return;
        }
        self.media_frozen = frozen;
        for layer in &mut self.layers {
            layer.reset_transport_timing_at(now);
        }
    }

    fn set_media_frozen(&mut self, frozen: bool) {
        self.set_media_frozen_at(frozen, Instant::now());
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
        let morph_offset = self.mod_matrix.frame(0).morph_offset();
        self.materialize_morph_at_current_beat_with_offset(morph_offset);
    }

    /// Hand control back to a direct editor without jumping away from the
    /// currently displayed A/B interpolation. The sampled world becomes the
    /// new base, then the caller's edit is applied on top of it. A single
    /// captured slot is deliberately preserved so the performer can continue
    /// authoring its partner.
    fn release_active_morph_for_manual_edit(&mut self) {
        if !self.morph.active() {
            return;
        }
        self.materialize_morph_at_current_beat();
        self.morph.clear();
    }

    fn reroll_effect_target(
        effects: &mut effects::EffectUniforms,
        supplied_seed: Option<u32>,
        stream: u64,
        mode: web::state::RerollMode,
        amount: f32,
        include_grain_controls: bool,
    ) {
        let seed = supplied_seed.map_or_else(
            || {
                let advanced = randomization::next_seed(effects.random_seed);
                if stream == 0 {
                    advanced
                } else {
                    randomization::stream_seed(advanced, stream)
                }
            },
            |base| {
                if stream == 0 {
                    base
                } else {
                    randomization::stream_seed(base, stream)
                }
            },
        );
        effects.random_seed = seed;
        if mode == web::state::RerollMode::Variation {
            randomization::mutate_live_effects(
                effects,
                amount,
                include_grain_controls,
                seed,
                stream,
            );
        }
    }

    fn reroll_master(
        &mut self,
        supplied_seed: Option<u32>,
        mode: web::state::RerollMode,
        amount: f32,
        include_grain_controls: bool,
        stream: u64,
    ) {
        Self::reroll_effect_target(
            &mut self.master_effects,
            supplied_seed,
            stream,
            mode,
            amount,
            include_grain_controls,
        );
        let master_seed = self.master_effects.random_seed;
        for (index, lfo) in self.mod_matrix.lfos.iter_mut().enumerate() {
            let lfo_stream = 0x4c46_4f00_u64 + index as u64;
            lfo.seed = randomization::stream_seed(master_seed, lfo_stream);
        }
    }

    fn apply_reroll(&mut self, request: RerollRequest) {
        let RerollRequest {
            scope,
            index,
            layer_id,
            stack_revision,
            supplied_seed,
            mode,
            amount,
            include_grain_controls,
        } = request;
        use web::state::RerollScope;
        match scope {
            RerollScope::Master => {
                self.reroll_master(supplied_seed, mode, amount, include_grain_controls, 0)
            }
            RerollScope::Layer => {
                let Some(index) =
                    index.and_then(|index| self.resolve_layer_index(index, &layer_id))
                else {
                    return;
                };
                Self::reroll_effect_target(
                    &mut self.layers[index].effects,
                    supplied_seed,
                    0,
                    mode,
                    amount,
                    include_grain_controls,
                );
            }
            RerollScope::All => {
                if stack_revision != Some(self.layer_stack_revision) {
                    return;
                }
                self.reroll_master(supplied_seed, mode, amount, include_grain_controls, 0);
                for (index, layer) in self.layers.iter_mut().enumerate() {
                    Self::reroll_effect_target(
                        &mut layer.effects,
                        supplied_seed,
                        index as u64 + 1,
                        mode,
                        amount,
                        include_grain_controls,
                    );
                }
            }
        }
    }

    fn valid_effect_edit(param: &str, value: &serde_json::Value) -> bool {
        match param {
            "invert" | "color_grain" => value.as_bool().is_some(),
            "grain_algo" | "key_mode" => value.as_u64().is_some(),
            "random_seed" => value
                .as_u64()
                .is_some_and(|number| u32::try_from(number).is_ok()),
            "pixelate"
            | "downsample"
            | "rgb_split"
            | "hue_shift"
            | "saturation"
            | "brightness"
            | "contrast"
            | "posterize"
            | "grain_intensity"
            | "grain_size"
            | "vignette"
            | "color_drift"
            | "breathe_scale"
            | "breathe_rotation"
            | "breathe_position"
            | "key_color_r"
            | "key_color_g"
            | "key_color_b"
            | "key_threshold"
            | "key_softness"
            | "key_tolerance"
            | "cellular_amount"
            | "cellular_scale"
            | "cellular_warp"
            | "cellular_speed"
            | "cellular_gap_amount"
            | "cellular_gap_threshold"
            | "cellular_gap_softness"
            | "shift_amount"
            | "shift_block_size"
            | "shift_density"
            | "shift_speed" => value.as_f64().is_some_and(f64::is_finite),
            _ => false,
        }
    }

    fn finite_f32_edit(value: &serde_json::Value) -> bool {
        value
            .as_f64()
            .map(|number| number as f32)
            .is_some_and(f32::is_finite)
    }

    fn valid_ntsc_edit(param: &str, value: &serde_json::Value) -> bool {
        match param {
            "enabled"
            | "edge_wave_enabled"
            | "head_switching_enabled"
            | "tracking_noise_enabled" => value.as_bool().is_some(),
            "tape_speed" => value.as_u64().is_some(),
            "head_switching_height" | "tracking_noise_height" => value.as_i64().is_some(),
            "chroma_loss"
            | "edge_wave_intensity"
            | "edge_wave_speed"
            | "head_switching_shift"
            | "tracking_noise_wave"
            | "tracking_noise_snow"
            | "snow_intensity"
            | "composite_noise_intensity"
            | "luma_noise_intensity"
            | "chroma_noise_intensity"
            | "luma_smear"
            | "composite_sharpening" => Self::finite_f32_edit(value),
            _ => false,
        }
    }

    fn valid_temporal_edit(param: &str, value: &serde_json::Value) -> bool {
        match param {
            "slit_axis" | "key_mode" => value.as_u64().is_some(),
            "slit_angle" => Self::finite_f32_edit(value),
            "feedback" | "fb_zoom" | "fb_rotate" | "slitscan" | "key_threshold"
            | "key_softness" | "key_history" => value.as_f64().is_some_and(f64::is_finite),
            _ => false,
        }
    }

    fn layer_param_morph_control(
        param: &str,
        value: &serde_json::Value,
    ) -> Option<morph::LayerMorphControl> {
        use morph::LayerMorphControl as Control;
        match param {
            "opacity" if value.as_f64().is_some_and(f64::is_finite) => Some(Control::Opacity),
            "speed" if value.as_f64().is_some_and(f64::is_finite) => Some(Control::Speed),
            "fps" if Self::finite_f32_edit(value) => Some(Control::Fps),
            "blend_mode"
                if matches!(
                    value.as_str(),
                    Some("normal" | "screen" | "multiply" | "difference")
                ) =>
            {
                Some(Control::BlendMode)
            }
            "bypass_master_fx" if value.as_bool().is_some() => Some(Control::BypassMasterFx),
            "key_threshold" if value.as_f64().is_some_and(f64::is_finite) => {
                Some(Control::KeyThreshold)
            }
            "key_mode" if value.as_u64().is_some() => Some(Control::Effects),
            "key_softness" | "key_color_r" | "key_color_g" | "key_color_b" | "key_tolerance"
                if value.as_f64().is_some_and(f64::is_finite) =>
            {
                Some(Control::Effects)
            }
            _ => None,
        }
    }

    /// Quantized wrappers return false here: ownership transfers only when the
    /// inner edit actually executes on its downbeat, never while it is merely
    /// pending. Stable layer IDs are resolved before deciding whether an edit
    /// touches a captured layer; stale/removed targets cannot clear A/B.
    fn manual_action_targets_active_morph(&self, action: &web::state::WebAction) -> bool {
        use web::state::WebAction;
        if !self.morph.active() {
            return false;
        }
        match action {
            WebAction::SetParam { param, value }
                if param == "random_seed" && Self::valid_effect_edit(param, value) =>
            {
                false
            }
            WebAction::SetParam { param, value } => Self::valid_effect_edit(param, value),
            WebAction::SetNtscParam { param, value } => Self::valid_ntsc_edit(param, value),
            WebAction::SetTemporal { param, value } => Self::valid_temporal_edit(param, value),
            WebAction::ResetGroup { group } => matches!(
                group.as_str(),
                "digital" | "analog" | "key" | "motion" | "cellular" | "shift" | "vhs" | "temporal"
            ),
            WebAction::SetLayerParam {
                index,
                layer_id,
                param,
                value,
            } => Self::layer_param_morph_control(param, value).is_some_and(|control| {
                self.resolve_layer_index(*index, layer_id)
                    .is_some_and(|resolved| self.morph.controls_layer_field(resolved, control))
            }),
            WebAction::SetLayerEffect {
                index,
                layer_id,
                param,
                value,
            } if Self::valid_effect_edit(param, value) => {
                if param == "random_seed" {
                    return false;
                }
                let control = if param == "key_threshold" {
                    morph::LayerMorphControl::KeyThreshold
                } else {
                    morph::LayerMorphControl::Effects
                };
                self.resolve_layer_index(*index, layer_id)
                    .is_some_and(|resolved| self.morph.controls_layer_field(resolved, control))
            }
            WebAction::ResetLayerFx { index, layer_id } => self
                .resolve_layer_index(*index, layer_id)
                .is_some_and(|resolved| {
                    self.morph
                        .controls_layer_field(resolved, morph::LayerMorphControl::AnyEffect)
                }),
            WebAction::SetLayerVisibility {
                index, layer_id, ..
            } => self
                .resolve_layer_index(*index, layer_id)
                .is_some_and(|resolved| {
                    self.morph
                        .controls_layer_field(resolved, morph::LayerMorphControl::Visible)
                }),
            WebAction::SetLayerPaused {
                index, layer_id, ..
            } => self
                .resolve_layer_index(*index, layer_id)
                .is_some_and(|resolved| {
                    self.morph
                        .controls_layer_field(resolved, morph::LayerMorphControl::Paused)
                }),
            WebAction::ToggleVisibility { index } | WebAction::ToggleLayerPause { index } => {
                let control = if matches!(action, WebAction::ToggleVisibility { .. }) {
                    morph::LayerMorphControl::Visible
                } else {
                    morph::LayerMorphControl::Paused
                };
                *index < self.layers.len() && self.morph.controls_layer_field(*index, control)
            }
            WebAction::Reroll {
                mode: web::state::RerollMode::Variation,
                scope,
                index,
                layer_id,
                stack_revision,
                ..
            } => match scope {
                web::state::RerollScope::Master => true,
                web::state::RerollScope::Layer => index
                    .and_then(|index| self.resolve_layer_index(index, layer_id))
                    .is_some(),
                web::state::RerollScope::All => *stack_revision == Some(self.layer_stack_revision),
            },
            _ => false,
        }
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
        self.selective_transition_holding = false;
        self.selective_hold_snapshot_valid = false;
        self.selective_hold_spout_barrier_epoch = None;
        self.selective_hold_spout_readback_epoch = None;
        // A broad revert starts a new visual generation. If a blackout is
        // already requested, consider its audience edge consumed so the next
        // frame cannot capture (and later restore) pre-revert pixels.
        self.blackout_presented = self.blackout;
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

    fn queue_quantized_action(
        &mut self,
        action: web::state::WebAction,
    ) -> WebActionBatchDisposition {
        let Some(key) = Self::quantized_action_key(&action) else {
            // Quantization is deliberately limited to non-emergency,
            // continuously expressive controls. Unknown wrappers retain
            // normal immediate behavior.
            return self.handle_web_action(action);
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
        WebActionBatchDisposition::Continue
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
    fn handle_web_action(&mut self, action: web::state::WebAction) -> WebActionBatchDisposition {
        use web::state::WebAction;
        if self.manual_action_targets_active_morph(&action) {
            self.release_active_morph_for_manual_edit();
        }
        let mut disposition = WebActionBatchDisposition::Continue;
        match action {
            WebAction::Quantized { inner } => return self.queue_quantized_action(*inner),
            WebAction::OpenPatchSnapshot => {
                if self.choose_snapshot_patch() {
                    disposition = WebActionBatchDisposition::SnapshotCommitted;
                }
            }
            WebAction::OpenPatchLook { stack_revision } => {
                if let Some(scope) = self.choose_and_apply_patch_look(Some(stack_revision)) {
                    disposition = WebActionBatchDisposition::LookApplied(scope);
                }
            }
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
                } else {
                    self.media_safety_status =
                        format!("Source not found in the active library: {filename}");
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
            WebAction::SetProgramFrozen { frozen } => {
                self.set_master_paused(frozen);
            }
            WebAction::SetMediaFrozen { frozen } => {
                self.set_media_frozen(frozen);
            }
            WebAction::SetMediaSafetyMode { mode } => {
                match self.media_safety_policy.set_mode(mode) {
                    Ok(()) => {
                        self.media_safety_status = match mode {
                            media_safety::MediaSafetyMode::Safe => {
                                "Safe media mode enabled for future source opens".to_string()
                            }
                            media_safety::MediaSafetyMode::Expert => {
                                "Expert media mode enabled for future source opens".to_string()
                            }
                        };
                    }
                    Err(error) => {
                        self.media_safety_status = format!("Expert mode unavailable: {error}");
                        log::warn!("{}", self.media_safety_status);
                    }
                }
            }
            WebAction::Reroll {
                scope,
                index,
                layer_id,
                stack_revision,
                seed,
                mode,
                amount,
                include_grain_controls,
            } => self.apply_reroll(RerollRequest {
                scope,
                index,
                layer_id,
                stack_revision,
                supplied_seed: seed,
                mode,
                amount,
                include_grain_controls,
            }),
            WebAction::SetLayerRerollOnLoop {
                index,
                layer_id,
                enabled,
            } => {
                if let Some(index) = self.resolve_layer_index(index, &layer_id) {
                    if self.layers[index].is_video() {
                        self.layers[index].reroll_on_loop = enabled;
                    }
                }
            }
            WebAction::SetBlackout { enabled } => self.set_blackout(enabled),
            // `reset_fx` predates the enriched visual program. Keep its wire
            // contract exact for older remotes: direct master uniforms only.
            WebAction::ResetFx => self.master_effects.reset(),
            WebAction::ResetVisualProgram => self.revert_master_visual_state(),
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
                    "shift" => {
                        self.master_effects.shift_amount = defaults.shift_amount;
                        self.master_effects.shift_block_size = defaults.shift_block_size;
                        self.master_effects.shift_density = defaults.shift_density;
                        self.master_effects.shift_speed = defaults.shift_speed;
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
                        "seed" => {
                            if let Some(n) = value.as_u64().and_then(|n| u32::try_from(n).ok()) {
                                lfo.seed = n;
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
                            self.mod_matrix.audio_clip_source_reference = None;
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
                    return disposition;
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
                    // A rescan supersedes any scan still in flight. The shared
                    // helper gate makes this a global single-flight handoff:
                    // the replacement waits until stale children are reaped.
                    let generation = self.web_state.begin_library_generation();
                    // Decode only entries whose convenience cache is not yet
                    // complete. Videos may have a thumbnail but still be
                    // waiting for their hover strip when the barrier arrives.
                    let new_files: Vec<PathBuf> = self
                        .library_files
                        .iter()
                        .filter(|p| {
                            let name = p
                                .file_name()
                                .map(|n| n.to_string_lossy().to_string())
                                .unwrap_or_default();
                            let thumbnail_cached = self
                                .web_state
                                .thumbnails
                                .read()
                                .map(|c| c.contains_key(&name))
                                .unwrap_or(false);
                            let preview_cached = self
                                .web_state
                                .preview_frames
                                .read()
                                .map(|c| c.contains_key(&name))
                                .unwrap_or(false);
                            !thumbnail_cached
                                || (!is_still_image_file(p.as_path()) && !preview_cached)
                        })
                        .cloned()
                        .collect();
                    if !new_files.is_empty() {
                        log::info!("Library rescan: {} new clip(s)", new_files.len());
                        generate_thumbnails(
                            &new_files,
                            self.web_state.clone(),
                            generation,
                            self.media_safety_policy.clone(),
                            self.renderer.as_ref().map_or_else(
                                media_safety::MediaDeviceLimits::none,
                                |renderer| {
                                    let limits = renderer.device.limits();
                                    media_safety::MediaDeviceLimits::new(
                                        limits.max_texture_dimension_2d,
                                        limits.max_buffer_size,
                                    )
                                },
                            ),
                        );
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
                                let blend_mode = match s {
                                    "normal" => Some(crate::layers::BlendMode::Normal),
                                    "screen" => Some(crate::layers::BlendMode::Screen),
                                    "multiply" => Some(crate::layers::BlendMode::Multiply),
                                    "difference" => Some(crate::layers::BlendMode::Difference),
                                    _ => None,
                                };
                                if let Some(blend_mode) = blend_mode {
                                    layer.blend_mode = blend_mode;
                                }
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
                ntsc_quality,
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
                                return disposition;
                            };
                            let Some(index) =
                                self.layers.iter().position(|layer| layer.layer_id() == id)
                            else {
                                log::warn!("Rejected export with stale audio layer ID {id}");
                                return disposition;
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
                            .map(|layer| layer.source_reference_for_persistence().to_owned()),
                        audio_path_hint: audio_index
                            .and_then(|index| self.layers.get(index))
                            .filter(|layer| layer.is_video())
                            .map(|layer| layer.source_path.clone()),
                        layer_source_hints: self
                            .layers
                            .iter()
                            .map(|layer| layer.source_path.clone())
                            .collect(),
                        analysis_audio_path_hint: self
                            .mod_matrix
                            .audio_clip_source_reference
                            .as_ref()
                            .map(|_| self.mod_matrix.audio_clip_path.clone()),
                        ntsc_quality,
                        media_safety_policy: self.media_safety_policy.clone(),
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
        disposition
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
        let export_warnings = self
            .export_job
            .as_ref()
            .map(|job| job.progress.warnings())
            .unwrap_or_default();

        let mut ntsc_snapshot = NtscSnapshot::from_params(&self.ntsc_params);
        ntsc_snapshot.live_metrics = web::state::NtscLiveMetricsSnapshot {
            active_path: match self.ntsc_pipeline_path {
                LiveNtscPath::Disabled => "off",
                LiveNtscPath::LegacyGlobal => "global",
                LiveNtscPath::SelectivePerLayer => "selective",
            }
            .to_string(),
            global: self.ntsc_live_metrics.global,
            selective: self.ntsc_live_metrics.selective,
            busy: match self.ntsc_pipeline_path {
                LiveNtscPath::Disabled => false,
                LiveNtscPath::LegacyGlobal => self.ntsc_worker.is_busy(),
                LiveNtscPath::SelectivePerLayer => {
                    self.selective_ntsc_worker
                        .as_ref()
                        .is_some_and(ntsc::SelectiveNtscWorker::is_busy)
                        || self
                            .renderer
                            .as_ref()
                            .is_some_and(Renderer::selective_ntsc_readback_busy)
                }
            },
        };
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
        let media_device_limits =
            self.renderer
                .as_ref()
                .map_or_else(media_safety::MediaDeviceLimits::none, |renderer| {
                    let limits = renderer.device.limits();
                    media_safety::MediaDeviceLimits::new(
                        limits.max_texture_dimension_2d,
                        limits.max_buffer_size,
                    )
                });
        let media_safety_snapshot = web::state::MediaSafetySnapshot::from_policy(
            self.media_safety_policy.snapshot(media_device_limits),
            self.media_safety_status.clone(),
        );
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
            media_safety: media_safety_snapshot,
            layers: self
                .layers
                .iter()
                .map(|layer| {
                    let spout_status = layer.spout_status();
                    let video_health = layer.video_health();
                    let (video_active, video_error) = if layer.source_error().is_empty() {
                        match video_health {
                            Some(video::threaded::DecoderHealth::Healthy) | None => {
                                (true, String::new())
                            }
                            Some(video::threaded::DecoderHealth::Failed(error)) => (false, error),
                        }
                    } else {
                        (false, layer.source_error().to_string())
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
                        reroll_on_loop: layer.reroll_on_loop,
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
            program_frozen: self.master_paused,
            media_frozen: self.media_frozen,
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
            export_warnings,
            patch_save_status: self.patch_collector.status(),
            patch_load_status: self.patch_load_status.clone(),
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

fn publish_thumbnail_if_current(
    web_state: &WebState,
    generation: u64,
    filename: String,
    bytes: Vec<u8>,
) -> bool {
    web_state.publish_thumbnail_with_budget(generation, filename, bytes, LIBRARY_MEDIA_CACHE_LIMIT)
}

fn publish_preview_if_current(
    web_state: &WebState,
    generation: u64,
    filename: String,
    frames: Vec<Vec<u8>>,
) -> bool {
    web_state.publish_preview_with_budget(generation, filename, frames, LIBRARY_MEDIA_CACHE_LIMIT)
}

const MAX_THUMBNAIL_CANDIDATES: usize = 4_096;
const THUMBNAIL_COMMAND_TIMEOUT: Duration = Duration::from_secs(20);
const THUMBNAIL_STDOUT_LIMIT: u64 = 8 * 1024 * 1024;
const THUMBNAIL_STDERR_LIMIT: u64 = 256 * 1024;
const THUMBNAIL_IMAGE_LIMIT: usize = 512 * 1024;
const LIBRARY_MEDIA_CACHE_LIMIT: u64 = 64 * 1024 * 1024;
const THUMBNAIL_SCALE_FILTER: &str = "scale=180:180:force_original_aspect_ratio=decrease";

#[derive(Clone, Debug)]
struct ThumbnailCandidate {
    path: PathBuf,
    source_kind: media_safety::MediaSourceKind,
    width: u32,
    height: u32,
    duration_secs: Option<f64>,
}

#[derive(Debug)]
struct BoundedMediaOutput {
    status: std::process::ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

const fn thumbnail_parallelism(mode: media_safety::MediaSafetyMode) -> (usize, usize) {
    match mode {
        // Four one-frame jobs and two preview jobs preserve useful library
        // throughput without multiplying four UHD decoders eightfold.
        media_safety::MediaSafetyMode::Safe => (4, 2),
        // An Expert source can consume most of the host planning allowance by
        // itself. Run all external decoders sequentially in that session.
        media_safety::MediaSafetyMode::Expert => (1, 1),
    }
}

fn read_bounded_pipe(
    pipe: impl std::io::Read,
    limit: u64,
    stream_name: &'static str,
) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    pipe.take(limit.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read {stream_name}: {error}"))?;
    if bytes.len() as u64 > limit {
        return Err(format!("{stream_name} exceeded the {limit}-byte limit"));
    }
    Ok(bytes)
}

fn reap_media_helper_in_background(
    mut child: std::process::Child,
    stdout_reader: std::thread::JoinHandle<Result<Vec<u8>, String>>,
    stderr_reader: std::thread::JoinHandle<Result<Vec<u8>, String>>,
) {
    if let Err(error) = std::thread::Builder::new()
        .name("media-helper-reap".into())
        .spawn(move || {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
        })
    {
        // Dropping Child is non-blocking. A platform that refuses both the
        // synchronous kill below and this bounded handoff must not wedge the
        // library/render thread while the operating system owns cleanup.
        log::error!("Could not start media-helper reaper: {error}");
    }
}

/// Run one ffmpeg-family child while retaining cancellation ownership. A
/// library switch kills and reaps old-generation work instead of merely
/// ignoring a late cache write; a wedged/corrupt file is bounded by a timeout.
fn run_bounded_media_command(
    mut command: std::process::Command,
    web_state: &WebState,
    generation: u64,
    timeout: Duration,
) -> Result<Option<BoundedMediaOutput>, String> {
    use std::process::Stdio;

    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .map_err(|error| format!("start media helper: {error}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or("media helper did not expose stdout")?;
    let stderr = child
        .stderr
        .take()
        .ok_or("media helper did not expose stderr")?;
    let stdout_reader = std::thread::spawn(move || {
        read_bounded_pipe(stdout, THUMBNAIL_STDOUT_LIMIT, "media-helper stdout")
    });
    let stderr_reader = std::thread::spawn(move || {
        read_bounded_pipe(stderr, THUMBNAIL_STDERR_LIMIT, "media-helper stderr")
    });
    let started = Instant::now();
    let mut cancelled = false;
    let status = loop {
        if started.elapsed() >= timeout {
            let _ = child.kill();
            reap_media_helper_in_background(child, stdout_reader, stderr_reader);
            return Err(format!(
                "media helper exceeded its {}s timeout",
                timeout.as_secs()
            ));
        }
        if !web_state.library_generation_is_current(generation) && !cancelled {
            cancelled = true;
            if let Err(error) = child.kill() {
                log::warn!("Could not immediately terminate stale media helper: {error}");
            }
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(error) => {
                let _ = child.kill();
                reap_media_helper_in_background(child, stdout_reader, stderr_reader);
                return Err(format!("poll media helper: {error}"));
            }
        }
    };
    let stdout = stdout_reader
        .join()
        .map_err(|_| "media-helper stdout reader panicked".to_string())??;
    let stderr = stderr_reader
        .join()
        .map_err(|_| "media-helper stderr reader panicked".to_string())??;
    if cancelled {
        Ok(None)
    } else {
        Ok(Some(BoundedMediaOutput {
            status,
            stdout,
            stderr,
        }))
    }
}

fn parse_ffprobe_visual_info(bytes: &[u8]) -> Result<(u32, u32, Option<f64>), String> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|error| format!("parse ffprobe JSON: {error}"))?;
    let stream = value
        .get("streams")
        .and_then(serde_json::Value::as_array)
        .and_then(|streams| streams.first())
        .ok_or("ffprobe reported no video stream")?;
    let width = stream
        .get("width")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or("ffprobe reported no valid video width")?;
    let height = stream
        .get("height")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or("ffprobe reported no valid video height")?;
    let duration_secs = value
        .get("format")
        .and_then(|format| format.get("duration"))
        .and_then(serde_json::Value::as_str)
        .and_then(|duration| duration.parse::<f64>().ok())
        .filter(|duration| duration.is_finite() && *duration >= 0.0);
    Ok((width, height, duration_secs))
}

fn inspect_thumbnail_candidate(
    path: PathBuf,
    web_state: &WebState,
    generation: u64,
    media_policy: &media_safety::MediaSafetyPolicy,
    device_limits: media_safety::MediaDeviceLimits,
) -> Result<Option<ThumbnailCandidate>, String> {
    if !web_state.library_generation_is_current(generation) {
        return Ok(None);
    }
    let (source_kind, width, height, duration_secs) = if is_still_image_file(&path) {
        let plan = video::probe_still_image_dimensions_with_media_policy(
            &path,
            media_policy,
            device_limits,
        )?;
        (
            media_safety::MediaSourceKind::Still,
            plan.width,
            plan.height,
            None,
        )
    } else {
        let mut command = std::process::Command::new("ffprobe");
        command
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height:format=duration",
                "-of",
                "json",
            ])
            .arg(&path);
        let Some(output) =
            run_bounded_media_command(command, web_state, generation, THUMBNAIL_COMMAND_TIMEOUT)?
        else {
            return Ok(None);
        };
        if !output.status.success() {
            return Err(format!(
                "ffprobe failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        let (width, height, duration) = parse_ffprobe_visual_info(&output.stdout)?;
        media_policy
            .plan(
                media_safety::MediaSourceKind::Video,
                width,
                height,
                device_limits,
            )
            .map_err(|error| error.to_string())?;
        (
            media_safety::MediaSourceKind::Video,
            width,
            height,
            duration,
        )
    };
    Ok(Some(ThumbnailCandidate {
        path,
        source_kind,
        width,
        height,
        duration_secs,
    }))
}

fn generate_one_thumbnail(
    candidate: ThumbnailCandidate,
    web_state: Arc<WebState>,
    generation: u64,
    media_policy: media_safety::MediaSafetyPolicy,
    device_limits: media_safety::MediaDeviceLimits,
) -> bool {
    let filename = match candidate.path.file_name() {
        Some(name) => name.to_string_lossy().into_owned(),
        None => return false,
    };
    let _reservation = match media_policy.reserve_source(
        candidate.source_kind,
        candidate.width,
        candidate.height,
        device_limits,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            log::warn!("Thumbnail skipped for {filename}: {error}");
            return false;
        }
    };
    let mut command = std::process::Command::new("ffmpeg");
    command.arg("-i").arg(&candidate.path).args([
        "-vframes",
        "1",
        "-vf",
        THUMBNAIL_SCALE_FILTER,
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-q:v",
        "8",
        "-loglevel",
        "error",
        "pipe:1",
    ]);
    match run_bounded_media_command(command, &web_state, generation, THUMBNAIL_COMMAND_TIMEOUT) {
        Ok(Some(output))
            if output.status.success()
                && !output.stdout.is_empty()
                && output.stdout.len() <= THUMBNAIL_IMAGE_LIMIT =>
        {
            publish_thumbnail_if_current(&web_state, generation, filename, output.stdout)
        }
        Ok(Some(output)) => {
            log::warn!(
                "Thumbnail rejected for {filename} (status={}, bytes={}): {}",
                output.status,
                output.stdout.len(),
                String::from_utf8_lossy(&output.stderr).trim(),
            );
            false
        }
        Ok(None) => false,
        Err(error) => {
            log::warn!("Thumbnail helper failed for {filename}: {error}");
            false
        }
    }
}

fn generate_one_preview(
    candidate: ThumbnailCandidate,
    web_state: Arc<WebState>,
    generation: u64,
    media_policy: media_safety::MediaSafetyPolicy,
    device_limits: media_safety::MediaDeviceLimits,
) -> bool {
    let Some(duration) = candidate.duration_secs.filter(|duration| *duration >= 0.5) else {
        return false;
    };
    let filename = match candidate.path.file_name() {
        Some(name) => name.to_string_lossy().into_owned(),
        None => return false,
    };
    let _reservation = match media_policy.reserve_source(
        candidate.source_kind,
        candidate.width,
        candidate.height,
        device_limits,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            log::warn!("Preview skipped for {filename}: {error}");
            return false;
        }
    };
    const NUM_FRAMES: usize = 8;
    let mut frames = Vec::with_capacity(NUM_FRAMES);
    for index in 0..NUM_FRAMES {
        if !web_state.library_generation_is_current(generation) {
            return false;
        }
        let seek = duration * index as f64 / NUM_FRAMES as f64;
        let seek_text = format!("{seek:.2}");
        let mut command = std::process::Command::new("ffmpeg");
        command
            .args(["-ss", seek_text.as_str(), "-i"])
            .arg(&candidate.path)
            .args([
                "-vframes",
                "1",
                "-vf",
                THUMBNAIL_SCALE_FILTER,
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "-q:v",
                "10",
                "-loglevel",
                "error",
                "pipe:1",
            ]);
        match run_bounded_media_command(command, &web_state, generation, THUMBNAIL_COMMAND_TIMEOUT)
        {
            Ok(Some(output))
                if output.status.success()
                    && !output.stdout.is_empty()
                    && output.stdout.len() <= THUMBNAIL_IMAGE_LIMIT =>
            {
                frames.push(output.stdout);
            }
            Ok(Some(output)) => {
                log::warn!(
                    "Preview frame rejected for {filename} (status={}, bytes={}): {}",
                    output.status,
                    output.stdout.len(),
                    String::from_utf8_lossy(&output.stderr).trim(),
                )
            }
            Ok(None) => return false,
            Err(error) => log::warn!("Preview helper failed for {filename}: {error}"),
        }
    }
    !frames.is_empty() && publish_preview_if_current(&web_state, generation, filename, frames)
}

/// Generate bounded thumbnails and preview frames for admitted library files.
/// Metadata is probed under the same policy as source opening before ffmpeg is
/// allowed to decode. Every child is timeout-bounded, killed/reaped when its
/// library generation becomes stale, and covered by an Expert reservation.
fn generate_thumbnails(
    files: &[PathBuf],
    web_state: Arc<web::state::WebState>,
    generation: u64,
    media_policy: media_safety::MediaSafetyPolicy,
    device_limits: media_safety::MediaDeviceLimits,
) {
    let paths: Vec<PathBuf> = files
        .iter()
        .take(MAX_THUMBNAIL_CANDIDATES)
        .cloned()
        .collect();
    if files.len() > paths.len() {
        log::warn!(
            "Thumbnail scan limited to the first {} of {} library files",
            paths.len(),
            files.len()
        );
    }
    std::thread::Builder::new()
        .name("thumb-gen".into())
        .spawn(move || {
            use std::sync::atomic::{AtomicUsize, Ordering};

            let _single_flight = web_state.lock_library_media_helpers();
            if !web_state.library_generation_is_current(generation) {
                return;
            }
            let mut candidates = Vec::with_capacity(paths.len());
            for path in paths {
                if !web_state.library_generation_is_current(generation) {
                    return;
                }
                let name = path
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "unnamed media".to_string());
                match inspect_thumbnail_candidate(
                    path,
                    &web_state,
                    generation,
                    &media_policy,
                    device_limits,
                ) {
                    Ok(Some(candidate)) => candidates.push(candidate),
                    Ok(None) => return,
                    Err(error) => log::warn!("Thumbnail preflight rejected {name}: {error}"),
                }
            }

            let total = candidates.len();
            let (thumbnail_parallel, preview_parallel) = thumbnail_parallelism(media_policy.mode());
            let count = Arc::new(AtomicUsize::new(0));
            for chunk in candidates.chunks(thumbnail_parallel) {
                if !web_state.library_generation_is_current(generation) {
                    return;
                }
                let handles: Vec<_> = chunk
                    .iter()
                    .cloned()
                    .map(|candidate| {
                        let web_state = web_state.clone();
                        let media_policy = media_policy.clone();
                        let count = count.clone();
                        std::thread::spawn(move || {
                            if generate_one_thumbnail(
                                candidate,
                                web_state,
                                generation,
                                media_policy,
                                device_limits,
                            ) {
                                count.fetch_add(1, Ordering::Relaxed);
                            }
                        })
                    })
                    .collect();
                for handle in handles {
                    let _ = handle.join();
                }
            }
            log::info!(
                "Generated {}/{total} policy-admitted thumbnails",
                count.load(Ordering::Relaxed)
            );

            let preview_count = Arc::new(AtomicUsize::new(0));
            for chunk in candidates.chunks(preview_parallel) {
                if !web_state.library_generation_is_current(generation) {
                    return;
                }
                let handles: Vec<_> = chunk
                    .iter()
                    .cloned()
                    .map(|candidate| {
                        let web_state = web_state.clone();
                        let media_policy = media_policy.clone();
                        let preview_count = preview_count.clone();
                        std::thread::spawn(move || {
                            if generate_one_preview(
                                candidate,
                                web_state,
                                generation,
                                media_policy,
                                device_limits,
                            ) {
                                preview_count.fetch_add(1, Ordering::Relaxed);
                            }
                        })
                    })
                    .collect();
                for handle in handles {
                    let _ = handle.join();
                }
            }
            log::info!(
                "Generated {}/{total} policy-admitted preview strips",
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

        let mut output_width = FALLBACK_OUTPUT_WIDTH;
        let mut output_height = FALLBACK_OUTPUT_HEIGHT;

        if let Some(ref path) = self.initial_video {
            let safe = media_safety::MediaSafetyPolicy::safe();
            let plan = if is_still_image_file(std::path::Path::new(path)) {
                video::probe_still_image_dimensions_with_media_policy(
                    std::path::Path::new(path),
                    &safe,
                    media_safety::MediaDeviceLimits::none(),
                )
            } else {
                video::VideoDecoder::probe_dimensions_with_media_policy(
                    path,
                    &safe,
                    media_safety::MediaDeviceLimits::none(),
                )
            };
            match plan {
                Ok(plan) => {
                    output_width = plan.width;
                    output_height = plan.height;
                }
                Err(error) => {
                    self.media_safety_status = format!(
                        "Initial visual could not define a safe output; using {FALLBACK_OUTPUT_WIDTH}x{FALLBACK_OUTPUT_HEIGHT}: {error}"
                    );
                    log::error!("{}", self.media_safety_status);
                }
            }
        }

        let create_window = |width, height| {
            let attrs = WindowAttributes::default()
                .with_title("collide-o-scope")
                .with_inner_size(winit::dpi::LogicalSize::new(width, height));
            event_loop.create_window(attrs).map(Arc::new)
        };
        let requested_output = (output_width, output_height);
        let recovery_output = renderer_recovery_output(output_width, output_height);
        let requested_window = match create_window(output_width, output_height) {
            Ok(window) => window,
            Err(error) => {
                log::error!("Preview window creation failed: {error}");
                event_loop.exit();
                return;
            }
        };
        let (window, renderer) = match Renderer::new(
            requested_window.clone(),
            output_width,
            output_height,
        ) {
            Ok(renderer) => (requested_window, renderer),
            Err(primary_error) if recovery_output.is_some() => {
                drop(requested_window);
                (output_width, output_height) = recovery_output.expect("guarded recovery output");
                self.output_error = format!(
                    "Initial {}x{} output was unavailable ({primary_error}); recovered at {output_width}x{output_height}",
                    requested_output.0, requested_output.1
                );
                log::warn!("{}", self.output_error);
                let fallback_window = match create_window(output_width, output_height) {
                    Ok(window) => window,
                    Err(error) => {
                        log::error!("Fallback preview window creation failed: {error}");
                        event_loop.exit();
                        return;
                    }
                };
                match Renderer::new(fallback_window.clone(), output_width, output_height) {
                    Ok(renderer) => (fallback_window, renderer),
                    Err(fallback_error) => {
                        log::error!(
                            "Renderer initialization failed at requested and recovery sizes: requested={primary_error}; fallback={fallback_error}"
                        );
                        event_loop.exit();
                        return;
                    }
                }
            }
            Err(error) => {
                log::error!("Renderer initialization failed: {error}");
                event_loop.exit();
                return;
            }
        };

        log::info!("Output: {}x{}", output_width, output_height);

        // Master effects operate in output pixels. Keep their resolution in
        // lock-step with the final renderer output, including safe recovery.
        self.master_effects.resolution = [output_width as f32, output_height as f32];

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
                                patch::PatchTransportState {
                                    master_paused: self.master_paused,
                                    media_frozen: self.media_frozen,
                                },
                                &self.morph,
                            );
                            return;
                        }
                        PhysicalKey::Code(KeyCode::KeyO) => {
                            if self.modifiers.shift_key() {
                                self.choose_and_apply_patch_look(None);
                            } else {
                                self.choose_snapshot_patch();
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
                            ControlFlow::ToggleMediaFreeze => {
                                self.set_media_frozen(!self.media_frozen)
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
                        ControlFlow::ToggleMediaFreeze => self.set_media_frozen(!self.media_frozen),
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
                    let mut pending_actions: VecDeque<_> = self
                        .web_state
                        .actions
                        .try_lock()
                        .map(|mut a| a.drain(..).collect())
                        .unwrap_or_default();
                    let mut requested_output = None;
                    while let Some(action) = pending_actions.pop_front() {
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
                            let disposition = self.handle_web_action(action);
                            Self::apply_web_action_batch_disposition(
                                &mut pending_actions,
                                disposition,
                            );
                        }
                    }
                    if let Some(enabled) = requested_output {
                        self.set_output_window(event_loop, enabled);
                        if self.exit_if_device_lost(event_loop) {
                            return;
                        }
                    }
                    let audio_load_completed = self.poll_audio_clip_loader();

                    // Build the preview and the native recovery controls. The
                    // recovery actions are collected here, then applied below
                    // before program time is sampled for this same frame.
                    let window = self.window.as_ref().unwrap();
                    let egui_winit = self.egui_winit.as_mut().unwrap();
                    let raw_input = egui_winit.take_egui_input(window);

                    let video_egui_texture_id = self.video_egui_texture_id;
                    let output_on_main = self.output_on_main;
                    let output_width = self.renderer.as_ref().unwrap().output_width;
                    let output_height = self.renderer.as_ref().unwrap().output_height;
                    let native_recovery_view = NativeRecoveryView {
                        control_server: self.web_state.control_server_info(),
                        browser_connections: self.web_state.tx.receiver_count(),
                        program_frozen: self.master_paused,
                        blackout: self.blackout,
                        library_folder: self.library_folder.clone(),
                        visual_files: self.library_files.len(),
                        audio_files: self.audio_library_files.len(),
                        output_status: self.output_error.clone(),
                        media_status: self.media_safety_status.clone(),
                    };
                    let mut native_recovery_actions = Vec::new();

                    let egui_context = self.egui_ctx.clone();
                    let yaml_editor = &mut self.yaml_editor;
                    let layers = &mut self.layers;
                    let master_effects = &mut self.master_effects;
                    let full_output = egui_context.run_ui(raw_input, |ctx| {
                        if show_native_recovery_strip(output_on_main) {
                            build_native_recovery_strip(
                                ctx,
                                &native_recovery_view,
                                &mut native_recovery_actions,
                            );
                        }
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

                    // Browser ingress was drained first; a physical operator's
                    // native action is therefore deterministic local-last and
                    // does not depend on the server queue or any live browser.
                    for action in native_recovery_actions {
                        self.apply_native_recovery_action(action);
                    }

                    let gates = transport_gates(
                        self.master_paused,
                        self.media_frozen,
                        self.audio_clip_blocks_program,
                    );
                    let program_transport_paused = !gates.program_running;
                    let program_wall_delta = if audio_load_completed {
                        Duration::ZERO
                    } else {
                        wall_frame_delta
                    };
                    let (elapsed_duration, program_delta) = self
                        .program_clock
                        .tick(program_wall_delta, program_transport_paused);
                    self.release_stale_gyro();

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

                    // Harvest completed file frames before modulation is
                    // sampled. Authoritative loop generations can therefore
                    // reroll the base seed used by the very first rendered
                    // frame of the new loop, even when the worker mailbox has
                    // overwritten intermediate decoded images.
                    {
                        let renderer = self.renderer.as_ref().unwrap();
                        for layer in &mut self.layers {
                            let playing = gates.media_running && !layer.paused;
                            if !layer.is_file_media()
                                || (!playing && layer.source_frame_initialized())
                            {
                                continue;
                            }
                            match layer.take_ready_media_frame() {
                                Ok(Some(frame)) => {
                                    randomization::apply_live_loop_reroll(
                                        &mut layer.effects,
                                        layer.reroll_on_loop,
                                        frame.loops_advanced,
                                    );
                                    if let Err(error) = layer.upload_frame(
                                        &renderer.device,
                                        &renderer.queue,
                                        &frame.rgba,
                                    ) {
                                        log::error!(
                                            "Layer '{}' GPU upload failed: {error}",
                                            layer.filename
                                        );
                                        layer.restore_ready_media_frame_after_failed_upload(frame);
                                    }
                                }
                                Ok(None) => {}
                                Err(error) => log::error!(
                                    "Layer '{}' decoder failed: {error}",
                                    layer.filename
                                ),
                            }
                        }
                    }

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
                    let modulation_frame = self.mod_matrix.frame(self.layers.len());

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
                        let playing = gates.media_running && !layer.paused;

                        if layer.is_file_media() {
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
                        let worker_admission = worker.admission_outcome();
                        if selective_budget_error.is_none()
                            && !matches!(worker_admission, ntsc::NtscSubmitOutcome::Busy)
                        {
                            if let Some(batch) = renderer.poll_selective_ntsc_readback() {
                                if current_plan.as_ref().is_some_and(|current| {
                                    ntsc::selective_plan_compatible(&batch.plan, current)
                                }) {
                                    let outcome = match worker_admission {
                                        ntsc::NtscSubmitOutcome::Accepted => {
                                            worker.try_submit_outcome(batch)
                                        }
                                        ntsc::NtscSubmitOutcome::Unavailable => {
                                            ntsc::NtscSubmitOutcome::Unavailable
                                        }
                                        ntsc::NtscSubmitOutcome::Busy => unreachable!(
                                            "busy worker is excluded from selective polling"
                                        ),
                                    };
                                    if !outcome.is_accepted() {
                                        // The GPU sample was already admitted. Keep a
                                        // downstream worker rejection distinct from a
                                        // topology/epoch incompatibility.
                                        self.ntsc_live_metrics
                                            .selective
                                            .record_downstream_rejection(outcome);
                                    }
                                } else {
                                    self.ntsc_live_metrics.selective.record_stale();
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
                                } else {
                                    self.ntsc_live_metrics.selective.record_stale();
                                }
                            }
                        }

                        if selective_budget_error.is_none() && !program_transport_paused {
                            if let Some(mut plan) = current_plan {
                                let worker_admission = worker.admission_outcome();
                                if selective_ntsc_gpu_gate_is_open(
                                    worker_admission,
                                    &mut self.ntsc_live_metrics.selective,
                                ) {
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
                                        Ok(accepted) => {
                                            self.ntsc_live_metrics
                                                .selective
                                                .record_attempt(accepted);
                                            if accepted {
                                                self.selective_sample_sequence = submitted_sequence;
                                                self.selective_ntsc_runtime_error.clear();
                                            }
                                        }
                                        Err(error) => {
                                            log::error!(
                                                "Selective NTSC snapshot rejected: {error}"
                                            );
                                            self.selective_ntsc_runtime_error = error;
                                        }
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
                    let held_audience_readback = if held_spout_active
                        && self.selective_hold_spout_barrier_epoch == Some(self.visual_epoch)
                        && self.selective_hold_spout_readback_epoch != Some(self.visual_epoch)
                    {
                        match renderer.begin_held_audience_readback(&mut encoder, self.visual_epoch)
                        {
                            Ok(slot) => slot,
                            Err(error) => {
                                self.output_error =
                                    format!("Spout held-audience readback unavailable: {error}");
                                log::error!("{}", self.output_error);
                                None
                            }
                        }
                    } else {
                        None
                    };
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
                    let selective_audience_readback = if self.spout_enabled
                        && self.spout.is_running()
                    {
                        selective_audience_sample.and_then(|sample| {
                            match renderer.begin_selective_audience_readback(&mut encoder, sample) {
                                Ok(slot) => slot,
                                Err(error) => {
                                    self.output_error = format!(
                                        "Spout selective-audience readback unavailable: {error}"
                                    );
                                    log::error!("{}", self.output_error);
                                    None
                                }
                            }
                        })
                    } else {
                        None
                    };

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
                            if frame.ntsc_metadata.is_some() {
                                self.ntsc_live_metrics.global.record_stale();
                            }
                            if frame.selective_sample.is_some() {
                                self.ntsc_live_metrics.selective.record_stale();
                            }
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
                                let outcome = self.ntsc_worker.try_submit_outcome(
                                    frame.pixels,
                                    renderer.output_width,
                                    renderer.output_height,
                                    metadata,
                                    frame.epoch,
                                );
                                self.ntsc_live_metrics.global.record_admission(outcome);
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
                        } else {
                            self.ntsc_live_metrics.global.record_stale();
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
                        let slot = match renderer.begin_readback(
                            &mut encoder,
                            self.visual_epoch,
                            ntsc_metadata,
                        ) {
                            Ok(slot) => slot,
                            Err(error) => {
                                if ntsc_path == LiveNtscPath::LegacyGlobal {
                                    self.ntsc_live_metrics
                                        .global
                                        .record_admission(ntsc::NtscSubmitOutcome::Unavailable);
                                }
                                self.output_error =
                                    format!("NTSC/Spout audience readback unavailable: {error}");
                                log::error!("{}", self.output_error);
                                None
                            }
                        };
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
    "Procedural patch generation with bounded source preflight (rendering remains explicit):\n\
     collide-o-scope generate --anchor <patch.yaml> --output <directory>\n\
       [--library <directory>] [--count 10] [--temperature 0.5] [--seed 0]\n\
       [--max-fingerprint-bytes 68719476736] [--allow-unverified-sources]\n\
       [--allow-black-sources]"
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
    let mut allow_unverified_sources = false;
    let mut library_path: Option<PathBuf> = None;
    let mut max_fingerprint_bytes = media_source::DEFAULT_MAX_FINGERPRINT_BYTES;
    let mut index = 0usize;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--anchor"
            | "--output"
            | "--library"
            | "--count"
            | "--temperature"
            | "--seed"
            | "--max-fingerprint-bytes" => {
                let flag = arguments[index].clone();
                index += 1;
                let value = arguments
                    .get(index)
                    .ok_or_else(|| format!("{flag} requires a value"))?;
                match flag.as_str() {
                    "--anchor" => anchor_path = Some(PathBuf::from(value)),
                    "--output" => output_path = PathBuf::from(value),
                    "--library" => library_path = Some(PathBuf::from(value)),
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
                    "--max-fingerprint-bytes" => {
                        max_fingerprint_bytes = value.parse().map_err(|_| {
                            format!("invalid --max-fingerprint-bytes value {value:?}")
                        })?
                    }
                    _ => unreachable!(),
                }
            }
            "--allow-black-sources" => allow_black_sources = true,
            "--allow-unverified-sources" => allow_unverified_sources = true,
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
    let inventory = procedural::preflight_sources(
        &anchor,
        &procedural::SourcePreflightConfig {
            anchor_dir: anchor_path.parent().map(std::path::Path::to_path_buf),
            library_dir: library_path,
            max_fingerprint_bytes,
            allow_unverified_sources,
        },
    )?;
    let pieces = procedural::generate_with_inventory(
        &anchor,
        &procedural::GenerationConfig {
            seed,
            count,
            temperature,
            allow_black_sources,
        },
        &inventory,
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
    fn thumbnail_workers_are_policy_bounded_and_expert_is_sequential() {
        assert_eq!(
            thumbnail_parallelism(media_safety::MediaSafetyMode::Safe),
            (4, 2)
        );
        assert_eq!(
            thumbnail_parallelism(media_safety::MediaSafetyMode::Expert),
            (1, 1)
        );
        assert_eq!(
            THUMBNAIL_SCALE_FILTER,
            "scale=180:180:force_original_aspect_ratio=decrease"
        );
        assert!(THUMBNAIL_IMAGE_LIMIT < THUMBNAIL_STDOUT_LIMIT as usize);
        assert_eq!(LIBRARY_MEDIA_CACHE_LIMIT, 64 * 1024 * 1024);
    }

    #[test]
    fn thumbnail_and_preview_caches_share_one_replacement_aware_byte_budget() {
        let state = WebState::new().expect("test web state");
        let generation = state.begin_library_generation();

        assert!(state.publish_thumbnail_with_budget(
            generation,
            "a.jpg".to_string(),
            vec![1; 3],
            5,
        ));
        assert!(state.publish_preview_with_budget(
            generation,
            "b.jpg".to_string(),
            vec![vec![2; 2]],
            5,
        ));
        assert_eq!(state.library_media_cache_bytes(), 5);
        assert!(!state.publish_thumbnail_with_budget(generation, "c.jpg".to_string(), vec![3], 5,));

        // Replacing an entry releases its former charge before the new value
        // is admitted, making room for a different cache without a race.
        assert!(state.publish_thumbnail_with_budget(generation, "a.jpg".to_string(), vec![4], 5,));
        assert_eq!(state.library_media_cache_bytes(), 3);
        assert!(state.publish_preview_with_budget(
            generation,
            "c.jpg".to_string(),
            vec![vec![5; 2]],
            5,
        ));
        assert_eq!(state.library_media_cache_bytes(), 5);

        state.remove_library_media_cache_entry("b.jpg");
        assert_eq!(state.library_media_cache_bytes(), 3);
        assert!(!state
            .preview_frames
            .read()
            .expect("preview cache")
            .contains_key("b.jpg"));

        state.clear_library_media_caches();
        assert_eq!(state.library_media_cache_bytes(), 0);
        assert!(state.thumbnails.read().expect("thumbnail cache").is_empty());
        assert!(state
            .preview_frames
            .read()
            .expect("preview cache")
            .is_empty());
    }

    #[test]
    fn thumbnail_helper_pool_is_process_wide_single_flight() {
        let state = WebState::new().expect("test web state");
        let first_generation = state.begin_library_generation();
        let held = state.lock_library_media_helpers();
        let contender = state.clone();
        let (acquired_tx, acquired_rx) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            let _guard = contender.lock_library_media_helpers();
            acquired_tx.send(()).expect("publish gate acquisition");
        });

        assert!(acquired_rx.recv_timeout(Duration::from_millis(50)).is_err());
        let replacement_generation = state.begin_library_generation();
        assert!(!state.library_generation_is_current(first_generation));
        assert!(state.library_generation_is_current(replacement_generation));

        drop(held);
        acquired_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("replacement helper acquires after prior scan exits");
        thread.join().expect("single-flight contender");
    }

    #[test]
    fn thumbnail_probe_contract_extracts_bounded_visual_metadata() {
        let json = br#"{
            "streams": [{"width": 5120, "height": 2160}],
            "format": {"duration": "12.500000"}
        }"#;
        assert_eq!(
            parse_ffprobe_visual_info(json).unwrap(),
            (5120, 2160, Some(12.5))
        );
        assert!(parse_ffprobe_visual_info(br#"{"streams": []}"#).is_err());

        let safe = media_safety::MediaSafetyPolicy::for_test(
            media_safety::MediaSafetyMode::Safe,
            2 * 1024 * 1024 * 1024,
        );
        assert!(safe
            .plan(
                media_safety::MediaSourceKind::Video,
                5120,
                2160,
                media_safety::MediaDeviceLimits::new(16_384, u64::MAX),
            )
            .is_err());
        let expert = media_safety::MediaSafetyPolicy::for_test(
            media_safety::MediaSafetyMode::Expert,
            2 * 1024 * 1024 * 1024,
        );
        assert!(expert
            .plan(
                media_safety::MediaSourceKind::Video,
                5120,
                2160,
                media_safety::MediaDeviceLimits::new(16_384, u64::MAX),
            )
            .is_ok());
    }

    #[test]
    fn thumbnail_generation_change_cancels_and_reaps_a_running_helper() {
        let state = WebState::new().expect("test web state");
        let generation = state.begin_library_generation();
        let invalidator = state.clone();
        let invalidate = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            invalidator.begin_library_generation();
        });
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "app_state_tests::thumbnail_cancellable_child_fixture",
                "--nocapture",
            ])
            .env("COLLIDE_O_SCOPE_THUMBNAIL_CHILD_FIXTURE", "1");
        let started = Instant::now();
        let output =
            run_bounded_media_command(command, &state, generation, Duration::from_secs(3)).unwrap();
        invalidate.join().unwrap();
        assert!(output.is_none());
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn thumbnail_cancellable_child_fixture() {
        if std::env::var_os("COLLIDE_O_SCOPE_THUMBNAIL_CHILD_FIXTURE").is_some() {
            std::thread::sleep(Duration::from_secs(60));
        }
    }

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
            PhysicalKey::Code(KeyCode::KeyM),
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
        assert!(show_native_recovery_strip(false));
        assert!(!show_native_recovery_strip(true));
    }

    #[test]
    fn selective_rebuild_holds_paused_audience_until_resume() {
        assert!(!selective_rebuild_should_clear_audience(true));
        assert!(selective_rebuild_should_clear_audience(false));
    }

    #[test]
    fn selective_worker_gate_separates_busy_from_unavailable_metrics() {
        let mut metrics = ntsc::NtscPathMetrics::default();
        assert!(selective_ntsc_gpu_gate_is_open(
            ntsc::NtscSubmitOutcome::Accepted,
            &mut metrics,
        ));
        assert_eq!(metrics, ntsc::NtscPathMetrics::default());

        assert!(!selective_ntsc_gpu_gate_is_open(
            ntsc::NtscSubmitOutcome::Busy,
            &mut metrics,
        ));
        assert_eq!(metrics.attempted, 1);
        assert_eq!(metrics.skipped, 1);
        assert_eq!(metrics.unavailable, 0);

        assert!(!selective_ntsc_gpu_gate_is_open(
            ntsc::NtscSubmitOutcome::Unavailable,
            &mut metrics,
        ));
        assert_eq!(metrics.attempted, 2);
        assert_eq!(metrics.skipped, 1);
        assert_eq!(metrics.unavailable, 1);
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

        // Stable identities remain authoritative beyond the former fixed
        // topology boundary, and legacy positional clients may address any
        // positive layer number without creating proportional storage.
        let ids: Vec<u64> = (1..=24).collect();
        assert_eq!(
            resolve_routing_target_for_layer_ids(
                "layer17_opacity",
                &Some("24".into()),
                Some(11),
                ids,
            )
            .as_deref(),
            Some("layer24_opacity")
        );
        assert_eq!(
            resolve_routing_target_for_layer_ids("layer4096_opacity", &None, None, []).as_deref(),
            Some("layer4096_opacity")
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
    fn program_and_media_freezes_have_independent_transport_gates() {
        assert_eq!(
            transport_gates(false, false, false),
            TransportGates {
                program_running: true,
                media_running: true,
            }
        );
        assert_eq!(
            transport_gates(false, true, false),
            TransportGates {
                program_running: true,
                media_running: false,
            }
        );
        for (program_frozen, media_frozen, audio_blocked) in [
            (true, false, false),
            (true, true, false),
            (false, false, true),
        ] {
            assert_eq!(
                transport_gates(program_frozen, media_frozen, audio_blocked),
                TransportGates {
                    program_running: false,
                    media_running: false,
                }
            );
        }

        let web_state = WebState::new().expect("test token");
        let mut app = App::new(None, None, web_state);
        let now = Instant::now();
        app.mod_matrix.clock.tap(now);
        let beat = app.mod_matrix.clock.beat(now);
        app.set_media_frozen_at(true, now + Duration::from_secs(20));
        assert!(app.media_frozen);
        assert!(!app.mod_matrix.clock.is_paused());
        assert_eq!(app.mod_matrix.clock.beat(now), beat);
    }

    #[test]
    fn pattern_reroll_is_exact_replayable_and_does_not_disengage_morph() {
        let mut first = App::new(None, None, WebState::new().expect("test token"));
        let mut second = App::new(None, None, WebState::new().expect("test token"));
        first.morph.a = Some(morph::MorphSlot::default());
        first.morph.b = Some(morph::MorphSlot::default());
        second.morph.a = Some(morph::MorphSlot::default());
        second.morph.b = Some(morph::MorphSlot::default());

        let action = web::state::WebAction::Reroll {
            scope: web::state::RerollScope::Master,
            index: None,
            layer_id: None,
            stack_revision: None,
            seed: Some(0x1234_5678),
            mode: web::state::RerollMode::Pattern,
            amount: 2.0,
            include_grain_controls: true,
        };
        first.handle_web_action(action.clone());
        second.handle_web_action(action);

        assert!(first.morph.active());
        assert!(second.morph.active());
        assert_eq!(first.master_effects.random_seed, 0x1234_5678);
        assert_eq!(
            bytemuck::bytes_of(&first.master_effects),
            bytemuck::bytes_of(&second.master_effects)
        );
        assert_eq!(
            first
                .mod_matrix
                .lfos
                .iter()
                .map(|lfo| lfo.seed)
                .collect::<Vec<_>>(),
            second
                .mod_matrix
                .lfos
                .iter()
                .map(|lfo| lfo.seed)
                .collect::<Vec<_>>()
        );
        assert!(first.mod_matrix.lfos.iter().all(|lfo| lfo.seed != 0));
    }

    #[test]
    fn variation_takes_morph_ownership_but_stale_everything_reroll_is_atomic() {
        let mut app = App::new(None, None, WebState::new().expect("test token"));
        app.morph.a = Some(morph::MorphSlot::default());
        app.morph.b = Some(morph::MorphSlot::default());
        let stale_seed = app.master_effects.random_seed;
        let stale_revision = app.layer_stack_revision.wrapping_add(1);
        app.handle_web_action(web::state::WebAction::Reroll {
            scope: web::state::RerollScope::All,
            index: None,
            layer_id: None,
            stack_revision: Some(stale_revision),
            seed: Some(77),
            mode: web::state::RerollMode::Variation,
            amount: 1.0,
            include_grain_controls: true,
        });
        assert!(app.morph.active());
        assert_eq!(app.master_effects.random_seed, stale_seed);

        app.handle_web_action(web::state::WebAction::Reroll {
            scope: web::state::RerollScope::Master,
            index: None,
            layer_id: None,
            stack_revision: None,
            seed: Some(77),
            mode: web::state::RerollMode::Variation,
            amount: 1.0,
            include_grain_controls: true,
        });
        assert!(!app.morph.active());
        assert_eq!(app.master_effects.random_seed, 77);
        assert!((1.0..=32.0).contains(&app.master_effects.pixelate_size));
        assert!((0.05..=1.0).contains(&app.master_effects.downsample));
        assert!((0.0..=3.0).contains(&app.master_effects.grain_algo));
    }

    #[test]
    fn explicit_zero_reroll_restores_every_legacy_pattern_seed() {
        let mut app = App::new(None, None, WebState::new().expect("test token"));
        app.master_effects.random_seed = 99;
        for (index, lfo) in app.mod_matrix.lfos.iter_mut().enumerate() {
            lfo.seed = index as u32 + 1;
        }
        app.handle_web_action(web::state::WebAction::Reroll {
            scope: web::state::RerollScope::Master,
            index: None,
            layer_id: None,
            stack_revision: None,
            seed: Some(0),
            mode: web::state::RerollMode::Pattern,
            amount: 0.7,
            include_grain_controls: false,
        });
        assert_eq!(app.master_effects.random_seed, 0);
        assert!(app.mod_matrix.lfos.iter().all(|lfo| lfo.seed == 0));
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
    fn direct_manual_edit_commits_current_morph_then_takes_ownership() {
        let web_state = WebState::new().expect("test token");
        let mut app = App::new(None, None, web_state);
        let a = morph::MorphSlot::capture(
            &effects::EffectUniforms {
                brightness: 0.0,
                saturation: 0.0,
                ..Default::default()
            },
            &ntsc::NtscParams::default(),
            &effects::params::TemporalParams::default(),
            &[],
        );
        let b = morph::MorphSlot::capture(
            &effects::EffectUniforms {
                brightness: 1.0,
                saturation: 1.0,
                ..Default::default()
            },
            &ntsc::NtscParams::default(),
            &effects::params::TemporalParams::default(),
            &[],
        );
        app.morph.a = Some(a);
        app.morph.b = Some(b);
        app.morph.t = 0.5;

        app.handle_web_action(web::state::WebAction::SetParam {
            param: "brightness".into(),
            value: serde_json::json!(0.8),
        });

        assert!(!app.morph.active());
        assert!(app.morph.a.is_none() && app.morph.b.is_none());
        assert!((app.master_effects.brightness - 0.8).abs() < 1.0e-6);
        assert!(
            (app.master_effects.saturation - 0.5).abs() < 1.0e-6,
            "an untouched field must prove the visible morph world was committed first"
        );
        app.materialize_morph_at_current_beat();
        assert!((app.master_effects.brightness - 0.8).abs() < 1.0e-6);
    }

    #[test]
    fn invalid_manual_commands_never_disengage_an_active_morph() {
        let web_state = WebState::new().expect("test token");
        let mut app = App::new(None, None, web_state);
        app.morph.a = Some(morph::MorphSlot::default());
        app.morph.b = Some(morph::MorphSlot::default());

        for action in [
            web::state::WebAction::SetParam {
                param: "future_effect".into(),
                value: serde_json::json!(0.5),
            },
            web::state::WebAction::SetParam {
                param: "brightness".into(),
                value: serde_json::json!("not a number"),
            },
            web::state::WebAction::SetNtscParam {
                param: "future_vhs".into(),
                value: serde_json::json!(true),
            },
            web::state::WebAction::SetTemporal {
                param: "feedback".into(),
                value: serde_json::json!(false),
            },
            web::state::WebAction::SetNtscParam {
                param: "edge_wave_intensity".into(),
                value: serde_json::json!(1.0e300),
            },
            web::state::WebAction::SetTemporal {
                param: "slit_angle".into(),
                value: serde_json::json!(1.0e300),
            },
        ] {
            app.handle_web_action(action);
            assert!(app.morph.active());
        }
    }

    #[test]
    fn single_slot_survives_manual_edit_and_quantized_pair_releases_only_on_downbeat() {
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
        app.morph.a = Some(a.clone());
        app.handle_web_action(web::state::WebAction::SetParam {
            param: "brightness".into(),
            value: serde_json::json!(0.2),
        });
        assert!(app.morph.a.is_some() && app.morph.b.is_none());
        assert!((app.master_effects.brightness - 0.2).abs() < 1.0e-6);

        app.morph.a = Some(a);
        app.morph.b = Some(b);
        app.morph.t = 0.5;
        app.materialize_morph_at_current_beat();
        app.quantized_bar = Some(0);
        app.queue_quantized_action(web::state::WebAction::SetParam {
            param: "brightness".into(),
            value: serde_json::json!(0.75),
        });
        assert!(app.morph.active());
        assert!((app.master_effects.brightness - 0.5).abs() < 1.0e-6);

        app.mod_matrix.current_beat = 4.0;
        app.release_quantized_actions_on_downbeat();
        assert!(app.quantized_actions.is_empty());
        assert!(!app.morph.active());
        assert!((app.master_effects.brightness - 0.75).abs() < 1.0e-6);
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
        app.selective_transition_holding = true;
        app.selective_hold_snapshot_valid = true;
        app.selective_hold_spout_barrier_epoch = Some(17);
        app.selective_hold_spout_readback_epoch = Some(17);
        app.blackout = true;
        app.blackout_presented = false;
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
        assert!(!app.selective_transition_holding);
        assert!(!app.selective_hold_snapshot_valid);
        assert!(app.selective_hold_spout_barrier_epoch.is_none());
        assert!(app.selective_hold_spout_readback_epoch.is_none());
        assert!(app.blackout);
        assert!(app.blackout_presented);
        assert_ne!(app.visual_epoch, old_epoch);
    }

    #[test]
    fn reset_actions_dispatch_from_json_with_versioned_scope() {
        let web_state = WebState::new().expect("test token");
        let mut app = App::new(None, None, web_state);
        app.master_effects.resolution = [1920.0, 1080.0];
        app.master_effects.brightness = 0.8;
        app.master_effects.random_seed = 41;
        app.ntsc_params.enabled = true;
        app.ntsc_params.snow_intensity = 0.6;
        app.temporal_params.feedback = 0.9;
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
        app.quantized_actions.push(web::state::WebAction::SetParam {
            param: "contrast".to_string(),
            value: serde_json::json!(0.5),
        });
        let initial_epoch = app.visual_epoch;

        let legacy: web::state::WebAction =
            serde_json::from_str(r#"{"action":"reset_fx"}"#).unwrap();
        app.handle_web_action(legacy);

        assert_eq!(app.master_effects.brightness, 0.0);
        assert_eq!(app.master_effects.random_seed, 0);
        assert_eq!(app.master_effects.resolution, [1920.0, 1080.0]);
        assert!(app.ntsc_params.enabled);
        assert_eq!(app.ntsc_params.snow_intensity, 0.6);
        assert_eq!(app.temporal_params.feedback, 0.9);
        assert_eq!(app.mod_matrix.routings.len(), 1);
        assert!(app.morph.active());
        assert_eq!(app.quantized_actions.len(), 1);
        assert_eq!(app.visual_epoch, initial_epoch);

        let broad: web::state::WebAction =
            serde_json::from_str(r#"{"action":"reset_visual_program"}"#).unwrap();
        app.handle_web_action(broad);

        assert_eq!(app.ntsc_params, ntsc::NtscParams::default());
        assert_eq!(
            app.temporal_params.feedback,
            effects::params::TemporalParams::default().feedback
        );
        assert!(app.mod_matrix.routings.is_empty());
        assert!(!app.morph.active());
        assert!(app.quantized_actions.is_empty());
        assert_ne!(app.visual_epoch, initial_epoch);
    }

    #[test]
    fn native_recovery_actions_are_absolute_and_work_without_a_browser() {
        let web_state = WebState::new().expect("test token");
        assert_eq!(web_state.tx.receiver_count(), 0);
        assert_eq!(
            web_state.control_server_info().status,
            ControlServerStatus::NotStarted
        );
        web_state
            .actions
            .try_lock()
            .expect("test action queue")
            .push(web::state::WebAction::SetParam {
                param: "contrast".to_string(),
                value: serde_json::json!(0.25),
            });

        let mut app = App::new(None, None, web_state.clone());
        app.master_effects.brightness = 0.8;
        app.ntsc_params.enabled = true;
        app.temporal_params.feedback = 0.9;
        app.mod_matrix.routings.push(modulation::Routing::new(
            modulation::ModSource::Lfo(0),
            "brightness",
            1.0,
        ));

        app.apply_native_recovery_action(NativeRecoveryAction::SetProgramFrozen(true));
        app.apply_native_recovery_action(NativeRecoveryAction::SetProgramFrozen(true));
        assert!(app.master_paused);
        assert!(app.mod_matrix.clock.is_paused());

        app.apply_native_recovery_action(NativeRecoveryAction::SetBlackout(true));
        let blackout_epoch = app.visual_epoch;
        app.apply_native_recovery_action(NativeRecoveryAction::SetBlackout(true));
        assert!(app.blackout);
        assert_eq!(app.visual_epoch, blackout_epoch);

        app.apply_native_recovery_action(NativeRecoveryAction::RevertVisualProgram);
        assert_eq!(app.master_effects.brightness, 0.0);
        assert_eq!(app.ntsc_params, ntsc::NtscParams::default());
        assert_eq!(app.temporal_params.feedback, 0.0);
        assert!(app.mod_matrix.routings.is_empty());
        assert!(app.master_paused, "broad Revert preserves transport");
        assert!(app.blackout, "broad Revert preserves the safety cut");
        assert_eq!(
            web_state
                .actions
                .try_lock()
                .expect("test action queue")
                .len(),
            1,
            "native controls do not depend on or mutate browser ingress"
        );
    }

    #[test]
    fn native_modal_reanchors_program_timing_without_changing_freeze_state() {
        let mut app = App::new(None, None, WebState::new().expect("test token"));
        let modal_started = Instant::now();
        let beat_before = app.mod_matrix.clock.beat(modal_started);
        app.mod_matrix.clock.set_paused(true, modal_started);

        let modal_finished = modal_started + Duration::from_secs(45);
        app.finish_native_modal(false, modal_finished);

        assert!(!app.master_paused);
        assert!(!app.mod_matrix.clock.is_paused());
        assert_eq!(app.last_frame_time, modal_finished);
        assert!((app.mod_matrix.clock.beat(modal_finished) - beat_before).abs() < 1.0e-9);
        assert!(
            app.mod_matrix
                .clock
                .beat(modal_finished + Duration::from_millis(10))
                > beat_before
        );
    }

    #[test]
    fn folder_switch_clears_caches_and_native_rescan_discovers_new_media() {
        let unique = format!(
            "collide-o-scope-p3-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .as_nanos()
        );
        let folder = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&folder).expect("create test library");

        let web_state = WebState::new().expect("test token");
        web_state
            .thumbnails
            .write()
            .expect("thumbnail cache")
            .insert("same-name.png".to_string(), vec![1]);
        web_state
            .preview_frames
            .write()
            .expect("preview cache")
            .insert("same-name.png".to_string(), vec![vec![2]]);
        let mut app = App::new(None, None, web_state.clone());

        app.set_library_folder(folder.clone());
        assert!(web_state
            .thumbnails
            .read()
            .expect("thumbnail cache")
            .is_empty());
        assert!(web_state
            .preview_frames
            .read()
            .expect("preview cache")
            .is_empty());
        assert_eq!(
            web_state
                .library_folder
                .read()
                .expect("shared library")
                .as_ref(),
            Some(&folder)
        );

        let added = folder.join("new-visual.png");
        std::fs::write(&added, []).expect("create visual fixture");
        // Mark the basename cached so this focused scan test does not launch
        // ffmpeg for an intentionally empty extension-only fixture.
        web_state
            .thumbnails
            .write()
            .expect("thumbnail cache")
            .insert("new-visual.png".to_string(), Vec::new());
        app.apply_native_recovery_action(NativeRecoveryAction::RescanLibrary);
        assert_eq!(app.library_files, vec![added]);

        std::fs::remove_dir_all(folder).expect("remove test library");
    }

    #[test]
    fn stale_library_workers_cannot_repopulate_current_caches() {
        let web_state = WebState::new().expect("test token");
        let old_generation = web_state.begin_library_generation();
        assert!(publish_thumbnail_if_current(
            &web_state,
            old_generation,
            "clip.png".to_string(),
            vec![1]
        ));
        assert!(publish_preview_if_current(
            &web_state,
            old_generation,
            "clip.png".to_string(),
            vec![vec![1]]
        ));

        let current_generation = web_state.begin_library_generation();
        web_state.clear_library_media_caches();

        assert!(!publish_thumbnail_if_current(
            &web_state,
            old_generation,
            "clip.png".to_string(),
            vec![2]
        ));
        assert!(!publish_preview_if_current(
            &web_state,
            old_generation,
            "clip.png".to_string(),
            vec![vec![2]]
        ));
        assert!(publish_thumbnail_if_current(
            &web_state,
            current_generation,
            "clip.png".to_string(),
            vec![3]
        ));
        assert_eq!(
            web_state
                .thumbnails
                .read()
                .expect("thumbnail cache")
                .get("clip.png"),
            Some(&vec![3])
        );
        assert!(web_state
            .preview_frames
            .read()
            .expect("preview cache")
            .is_empty());
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
    fn snapshot_audio_decode_failure_is_atomic_and_success_commits_a_ready_clip() {
        fn mono_pcm16_wav(sample_count: usize) -> Vec<u8> {
            let data_bytes = u32::try_from(sample_count * 2).unwrap();
            let mut wav = Vec::with_capacity(44 + data_bytes as usize);
            wav.extend_from_slice(b"RIFF");
            wav.extend_from_slice(&(36 + data_bytes).to_le_bytes());
            wav.extend_from_slice(b"WAVEfmt ");
            wav.extend_from_slice(&16_u32.to_le_bytes());
            wav.extend_from_slice(&1_u16.to_le_bytes());
            wav.extend_from_slice(&1_u16.to_le_bytes());
            wav.extend_from_slice(&48_000_u32.to_le_bytes());
            wav.extend_from_slice(&96_000_u32.to_le_bytes());
            wav.extend_from_slice(&2_u16.to_le_bytes());
            wav.extend_from_slice(&16_u16.to_le_bytes());
            wav.extend_from_slice(b"data");
            wav.extend_from_slice(&data_bytes.to_le_bytes());
            wav.resize(44 + data_bytes as usize, 0);
            wav
        }

        let unique = format!(
            "collide-o-scope-snapshot-audio-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .as_nanos()
        );
        let folder = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&folder).expect("create snapshot fixture folder");
        let corrupt_path = folder.join("corrupt.wav");
        let valid_path = folder.join("ready.wav");
        std::fs::write(&corrupt_path, b"not a wave file").expect("write corrupt audio fixture");
        std::fs::write(&valid_path, mono_pcm16_wav(2048)).expect("write valid audio fixture");

        let web_state = WebState::new().expect("test token");
        let mut app = App::new(None, None, web_state);
        app.master_effects.brightness = 0.25;
        app.mod_matrix.audio_clip_path = "pre-existing.wav".to_string();
        let original_visual_epoch = app.visual_epoch;
        let original_stack_revision = app.layer_stack_revision;

        let mut corrupt_patch = app.capture_current_patch();
        corrupt_patch.master.brightness = 0.9;
        let corrupt_modulation = corrupt_patch
            .modulation
            .as_mut()
            .expect("captured modulation");
        corrupt_modulation.audio_enabled = true;
        corrupt_modulation.audio_source_kind = modulation::AUDIO_SOURCE_FILE.to_string();
        corrupt_modulation.audio_clip_path = "corrupt.wav".to_string();
        let patch_path = folder.join("snapshot.yaml");

        let mut missing_patch = corrupt_patch.clone();
        missing_patch
            .modulation
            .as_mut()
            .expect("captured modulation")
            .audio_clip_path = "missing.wav".to_string();
        let error = app
            .apply_loaded_patch(missing_patch, &patch_path)
            .expect_err("missing imported audio must reject the whole snapshot");
        assert!(
            error.contains("analysis audio"),
            "unexpected error: {error}"
        );
        assert_eq!(app.master_effects.brightness, 0.25);
        assert_eq!(app.mod_matrix.audio_clip_path, "pre-existing.wav");
        assert_eq!(app.visual_epoch, original_visual_epoch);
        assert_eq!(app.layer_stack_revision, original_stack_revision);

        let error = app
            .apply_loaded_patch(corrupt_patch.clone(), &patch_path)
            .expect_err("corrupt imported audio must reject the whole snapshot");
        assert!(
            error.contains("analysis audio"),
            "unexpected error: {error}"
        );
        assert_eq!(app.master_effects.brightness, 0.25);
        assert_eq!(app.mod_matrix.audio_clip_path, "pre-existing.wav");
        assert_eq!(app.visual_epoch, original_visual_epoch);
        assert_eq!(app.layer_stack_revision, original_stack_revision);

        let mut valid_patch = corrupt_patch;
        valid_patch
            .modulation
            .as_mut()
            .expect("captured modulation")
            .audio_clip_path = "ready.wav".to_string();
        app.apply_loaded_patch(valid_patch, &patch_path)
            .expect("fully staged snapshot");

        assert_eq!(app.master_effects.brightness, 0.9);
        let expected_path = valid_path.canonicalize().expect("canonical fixture path");
        assert_eq!(
            PathBuf::from(&app.mod_matrix.audio_clip_path)
                .canonicalize()
                .expect("canonical committed path"),
            expected_path
        );
        assert_eq!(
            PathBuf::from(
                &app.audio_clip
                    .as_ref()
                    .expect("decoded clip committed")
                    .info()
                    .path
            )
            .canonicalize()
            .expect("canonical decoded clip path"),
            expected_path
        );
        assert_eq!(
            app.audio_clip_loader.state(),
            audio::AudioClipLoadState::Idle
        );
        assert!(!app.audio_clip_blocks_program);

        let mut fingerprints =
            media_source::FingerprintSession::new(media_source::FingerprintLimits::default())
                .expect("fingerprint session");
        let content_reference = fingerprints
            .fingerprint(&valid_path)
            .expect("fingerprint valid analysis clip")
            .source_reference();
        let mut content_patch = app.capture_current_patch();
        content_patch
            .modulation
            .as_mut()
            .expect("captured modulation")
            .audio_clip_path = content_reference.clone();
        app.apply_loaded_patch(content_patch, &patch_path)
            .expect("load content-addressed analysis clip");

        assert_eq!(
            PathBuf::from(&app.mod_matrix.audio_clip_path)
                .canonicalize()
                .expect("resolved runtime analysis path"),
            expected_path
        );
        assert_eq!(
            app.mod_matrix.audio_clip_source_reference.as_deref(),
            Some(content_reference.as_str())
        );
        assert_eq!(
            app.capture_current_patch()
                .modulation
                .expect("recaptured modulation")
                .audio_clip_path,
            content_reference,
            "exact load -> runtime -> capture must retain the content identity"
        );

        std::fs::remove_dir_all(folder).expect("remove snapshot fixture folder");
    }

    #[test]
    fn successful_patch_dispositions_filter_only_the_remaining_conflicting_scope() {
        use web::state::{RerollMode, RerollScope, WebAction};

        let mut snapshot_remainder = VecDeque::from(vec![
            WebAction::SetProgramFrozen { frozen: true },
            WebAction::SetParam {
                param: "brightness".to_string(),
                value: serde_json::json!(0.75),
            },
        ]);
        App::apply_web_action_batch_disposition(
            &mut snapshot_remainder,
            WebActionBatchDisposition::Continue,
        );
        assert_eq!(snapshot_remainder.len(), 2, "failed/cancelled chooser");
        App::apply_web_action_batch_disposition(
            &mut snapshot_remainder,
            WebActionBatchDisposition::SnapshotCommitted,
        );
        assert!(snapshot_remainder.is_empty());

        let mut look_remainder = VecDeque::from(vec![
            WebAction::AddLayer {
                filename: "queued.mp4".to_string(),
            },
            WebAction::AddSpoutLayer {
                sender: "Queued Sender".to_string(),
            },
            WebAction::RemoveLayer {
                index: 0,
                layer_id: Some("11".to_string()),
            },
            WebAction::MoveLayer {
                from: 0,
                to: 1,
                layer_id: Some("11".to_string()),
                stack_revision: Some(7),
            },
            WebAction::SetParam {
                param: "brightness".to_string(),
                value: serde_json::json!(-0.5),
            },
            WebAction::Quantized {
                inner: Box::new(WebAction::SetNtscParam {
                    param: "chroma_loss".to_string(),
                    value: serde_json::json!(0.8),
                }),
            },
            WebAction::Reroll {
                scope: RerollScope::Master,
                index: None,
                layer_id: None,
                stack_revision: None,
                seed: Some(99),
                mode: RerollMode::Pattern,
                amount: 1.0,
                include_grain_controls: false,
            },
            WebAction::SetLayerEffect {
                index: 0,
                layer_id: Some("11".to_string()),
                param: "brightness".to_string(),
                value: serde_json::json!(0.4),
            },
            WebAction::SetLayerParam {
                index: 0,
                layer_id: Some("11".to_string()),
                param: "key_threshold".to_string(),
                value: serde_json::json!(0.9),
            },
            WebAction::OpenPatchSnapshot,
            WebAction::Quantized {
                inner: Box::new(WebAction::OpenPatchLook { stack_revision: 7 }),
            },
            WebAction::SetLayerParam {
                index: 1,
                layer_id: Some("22".to_string()),
                param: "opacity".to_string(),
                value: serde_json::json!(0.4),
            },
            WebAction::SetLayerParam {
                index: 0,
                layer_id: Some("11".to_string()),
                param: "speed".to_string(),
                value: serde_json::json!(1.5),
            },
            WebAction::SetTemporal {
                param: "feedback".to_string(),
                value: serde_json::json!(0.4),
            },
            WebAction::ResetGroup {
                group: "digital".to_string(),
            },
            WebAction::ResetGroup {
                group: "vhs".to_string(),
            },
            WebAction::ResetGroup {
                group: "temporal".to_string(),
            },
            WebAction::ResetGroup {
                group: "mod".to_string(),
            },
            WebAction::SetProgramFrozen { frozen: true },
            WebAction::SetMediaSafetyMode {
                mode: media_safety::MediaSafetyMode::Safe,
            },
            WebAction::SetBlackout { enabled: true },
        ]);
        App::apply_web_action_batch_disposition(
            &mut look_remainder,
            WebActionBatchDisposition::LookApplied(AppliedLookScope {
                mapped_layer_ids: vec![11],
                applied_ntsc: true,
                applied_temporal: false,
            }),
        );

        assert_eq!(look_remainder.len(), 8);
        assert!(matches!(
            &look_remainder[0],
            WebAction::SetLayerParam { layer_id, .. } if layer_id.as_deref() == Some("22")
        ));
        assert!(matches!(
            &look_remainder[1],
            WebAction::SetLayerParam { layer_id, param, .. }
                if layer_id.as_deref() == Some("11") && param == "speed"
        ));
        assert!(matches!(look_remainder[2], WebAction::SetTemporal { .. }));
        assert!(matches!(
            &look_remainder[3],
            WebAction::ResetGroup { group } if group == "temporal"
        ));
        assert!(matches!(
            &look_remainder[4],
            WebAction::ResetGroup { group } if group == "mod"
        ));
        assert!(matches!(
            look_remainder[5],
            WebAction::SetProgramFrozen { frozen: true }
        ));
        assert!(matches!(
            look_remainder[6],
            WebAction::SetMediaSafetyMode { .. }
        ));
        assert!(matches!(
            look_remainder[7],
            WebAction::SetBlackout { enabled: true }
        ));
    }

    #[test]
    fn applied_look_filters_keying_and_nested_patch_actions_from_every_queue() {
        use web::state::WebAction;

        let web_state = WebState::new().expect("test token");
        let mut app = App::new(None, None, web_state);
        app.quantized_actions = vec![
            WebAction::SetLayerParam {
                index: 0,
                layer_id: Some("11".to_string()),
                param: "key_color_r".to_string(),
                value: serde_json::json!(0.8),
            },
            WebAction::Quantized {
                inner: Box::new(WebAction::OpenPatchSnapshot),
            },
            WebAction::SetLayerParam {
                index: 0,
                layer_id: Some("11".to_string()),
                param: "speed".to_string(),
                value: serde_json::json!(1.5),
            },
            WebAction::SetLayerParam {
                index: 1,
                layer_id: Some("22".to_string()),
                param: "key_threshold".to_string(),
                value: serde_json::json!(0.7),
            },
            WebAction::Quantized {
                inner: Box::new(WebAction::ResetGroup {
                    group: "mod".to_string(),
                }),
            },
            WebAction::Quantized {
                inner: Box::new(WebAction::ResetGroup {
                    group: "temporal".to_string(),
                }),
            },
            WebAction::Quantized {
                inner: Box::new(WebAction::ResetGroup {
                    group: "digital".to_string(),
                }),
            },
        ];
        app.web_state.actions.blocking_lock().extend([
            WebAction::OpenPatchLook { stack_revision: 7 },
            WebAction::SetLayerParam {
                index: 0,
                layer_id: Some("11".to_string()),
                param: "key_softness".to_string(),
                value: serde_json::json!(0.2),
            },
            WebAction::SetBlackout { enabled: true },
            WebAction::SetLayerParam {
                index: 0,
                layer_id: Some("11".to_string()),
                param: "fps".to_string(),
                value: serde_json::json!(60.0),
            },
            WebAction::ResetGroup {
                group: "vhs".to_string(),
            },
            WebAction::ResetGroup {
                group: "key".to_string(),
            },
        ]);

        app.invalidate_visual_generation_after_look(&[11], false, false);

        assert_eq!(app.quantized_actions.len(), 4);
        assert!(matches!(
            &app.quantized_actions[0],
            WebAction::SetLayerParam { layer_id, param, .. }
                if layer_id.as_deref() == Some("11") && param == "speed"
        ));
        assert!(matches!(
            &app.quantized_actions[1],
            WebAction::SetLayerParam { layer_id, param, .. }
                if layer_id.as_deref() == Some("22") && param == "key_threshold"
        ));
        assert!(matches!(
            &app.quantized_actions[2],
            WebAction::Quantized { inner }
                if matches!(inner.as_ref(), WebAction::ResetGroup { group } if group == "mod")
        ));
        assert!(matches!(
            &app.quantized_actions[3],
            WebAction::Quantized { inner }
                if matches!(inner.as_ref(), WebAction::ResetGroup { group } if group == "temporal")
        ));

        let shared = app.web_state.actions.blocking_lock();
        assert_eq!(shared.len(), 3);
        assert!(matches!(
            shared[0],
            WebAction::SetBlackout { enabled: true }
        ));
        assert!(matches!(
            &shared[1],
            WebAction::SetLayerParam { layer_id, param, .. }
                if layer_id.as_deref() == Some("11") && param == "fps"
        ));
        assert!(matches!(
            &shared[2],
            WebAction::ResetGroup { group } if group == "vhs"
        ));
        assert!(App::action_conflicts_with_applied_look(
            &WebAction::ResetGroup {
                group: "temporal".to_string(),
            },
            &[],
            false,
            true,
        ));
        assert!(!App::action_conflicts_with_applied_look(
            &WebAction::ResetGroup {
                group: "mod".to_string(),
            },
            &[],
            true,
            true,
        ));
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

    #[test]
    fn media_safety_mode_is_immediate_idempotent_and_never_patch_persistent() {
        let web_state = WebState::new().expect("test token");
        let mut app = App::new(None, None, web_state);
        app.media_safety_policy = media_safety::MediaSafetyPolicy::for_test(
            media_safety::MediaSafetyMode::Safe,
            1024 * 1024 * 1024,
        );

        app.handle_web_action(web::state::WebAction::Quantized {
            inner: Box::new(web::state::WebAction::SetMediaSafetyMode {
                mode: media_safety::MediaSafetyMode::Expert,
            }),
        });
        assert_eq!(
            app.media_safety_policy.mode(),
            media_safety::MediaSafetyMode::Expert,
            "a quantized wrapper must not delay a host resource policy change"
        );
        assert!(app.quantized_actions.is_empty());

        for _ in 0..2 {
            app.handle_web_action(web::state::WebAction::SetMediaSafetyMode {
                mode: media_safety::MediaSafetyMode::Expert,
            });
            assert_eq!(
                app.media_safety_policy.mode(),
                media_safety::MediaSafetyMode::Expert
            );
        }
        app.handle_web_action(web::state::WebAction::SetMediaSafetyMode {
            mode: media_safety::MediaSafetyMode::Safe,
        });
        assert_eq!(
            app.media_safety_policy.mode(),
            media_safety::MediaSafetyMode::Safe
        );

        let yaml = serde_yaml::to_string(&app.capture_current_patch()).unwrap();
        assert!(!yaml.contains("media_safety"));
        assert!(!yaml.contains("expert"));
    }
}
