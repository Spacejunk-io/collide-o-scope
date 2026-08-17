use crate::composition::RuntimeComposition;
use crate::effects::EffectUniforms;
use crate::layers::Layer;
use crate::motion::MotionParams;
use crate::ntsc::NtscParams;
use crate::visual_rack::RuntimeVisualRack;

use super::{param_meta, EffectsConfig, LayerConfig, PatchMasterVisual, PatchState};

const MAX_EDITOR_HISTORY_BOUNDARIES: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditorHistoryBoundary {
    Begin {
        gesture_id: u64,
        label: String,
        category: String,
    },
    End {
        gesture_id: u64,
    },
}

pub struct EditorState {
    pub active: bool,
    pub tab: usize,                   // 0 = Master, 1+ = layer index + 1
    pub active_field: Option<String>, // which param key is being edited
    pub field_buffer: String,         // text in the active pill input
    pub request_focus: bool,          // request focus on next frame
    history_boundaries: Vec<EditorHistoryBoundary>,
    active_history_gesture: Option<u64>,
    next_history_gesture: u64,
}

impl Default for EditorState {
    fn default() -> Self {
        Self {
            active: true,
            tab: 0,
            active_field: None,
            field_buffer: String::new(),
            request_focus: false,
            history_boundaries: Vec::new(),
            active_history_gesture: None,
            next_history_gesture: 1,
        }
    }
}

impl EditorState {
    fn push_history_boundary(&mut self, boundary: EditorHistoryBoundary) -> bool {
        if self.history_boundaries.len() == MAX_EDITOR_HISTORY_BOUNDARIES {
            // Main drains these every UI frame. Fail closed if an integration
            // bug stops draining: forget the entirely unobserved batch and do
            // not publish an unmatched Begin that could strand Main open.
            self.history_boundaries.clear();
            self.active_history_gesture = None;
            return false;
        }
        self.history_boundaries.push(boundary);
        true
    }

    fn begin_history_gesture(&mut self, label: String) -> bool {
        self.end_history_gesture();
        let gesture_id = self.next_history_gesture.max(1);
        self.next_history_gesture = gesture_id.checked_add(1).unwrap_or(1);
        let accepted = self.push_history_boundary(EditorHistoryBoundary::Begin {
            gesture_id,
            label,
            category: "native_editor".to_string(),
        });
        if accepted {
            self.active_history_gesture = Some(gesture_id);
        }
        accepted
    }

    fn end_history_gesture(&mut self) {
        let Some(gesture_id) = self.active_history_gesture.take() else {
            return;
        };
        let _ = self.push_history_boundary(EditorHistoryBoundary::End { gesture_id });
    }

    /// Drain ordered native edit boundaries after the current egui frame.
    /// Main captures the exact authored world on Begin and commits it on End.
    pub fn take_history_boundaries(&mut self) -> Vec<EditorHistoryBoundary> {
        std::mem::take(&mut self.history_boundaries)
    }

    /// Close a focused text gesture before hiding the editor or replacing its
    /// state through another transaction.
    pub fn finish_active_edit(&mut self) {
        self.active_field = None;
        self.end_history_gesture();
    }
}

// Colors for syntax-highlighted code look
const KEY_COLOR: egui::Color32 = egui::Color32::from_rgb(130, 170, 255);
const VALUE_COLOR: egui::Color32 = egui::Color32::from_rgb(200, 230, 200);
const BOOL_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 180, 100);
const STRING_COLOR: egui::Color32 = egui::Color32::from_rgb(200, 160, 255);
const PILL_BG: egui::Color32 = egui::Color32::from_rgb(50, 55, 65);
const GROUP_COLOR: egui::Color32 = egui::Color32::from_rgb(90, 130, 160);

// Key column width in pixels (enough for "breathe_rotation:" in monospace 13)
const KEY_COL_WIDTH: f32 = 140.0;

/// Render a single param row: key: [value pill]
/// `key_width` aligns the value column (in pixels). Pass 0.0 for no alignment.
/// Returns the new value if changed this frame.
fn param_row(
    ui: &mut egui::Ui,
    key: &str,
    value: &str,
    editor: &mut EditorState,
    key_width: f32,
) -> Option<String> {
    let mut new_value: Option<String> = None;
    let meta = param_meta(key);

    // Fixed pill width for consistent alignment
    const PILL_WIDTH: f32 = 72.0;

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;

        // Key label (padded to key_width for alignment)
        let key_text = format!("{}:", key);
        if key_width > 0.0 {
            ui.allocate_ui_with_layout(
                egui::vec2(key_width, ui.spacing().interact_size.y),
                egui::Layout::left_to_right(egui::Align::Center),
                |ui| {
                    ui.label(
                        egui::RichText::new(&key_text)
                            .monospace()
                            .size(13.0)
                            .color(KEY_COLOR),
                    );
                },
            );
        } else {
            ui.label(
                egui::RichText::new(&key_text)
                    .monospace()
                    .size(13.0)
                    .color(KEY_COLOR),
            );
        }

        let is_active = editor.active_field.as_deref() == Some(key);

        let pill_response = if is_active {
            // Active pill: singleline text input (fixed width)
            let id = ui.make_persistent_id(format!("pill_{}", key));
            let response = ui.add(
                egui::TextEdit::singleline(&mut editor.field_buffer)
                    .id(id)
                    .desired_width(PILL_WIDTH)
                    .font(egui::FontId::monospace(13.0))
                    .background_color(PILL_BG),
            );

            // Request focus on activation frame
            if editor.request_focus {
                response.request_focus();
                // Select all text
                if let Some(mut state) = egui::TextEdit::load_state(ui.ctx(), id) {
                    state
                        .cursor
                        .set_char_range(Some(egui::text::CCursorRange::two(
                            egui::text::CCursor::new(0),
                            egui::text::CCursor::new(editor.field_buffer.len()),
                        )));
                    state.store(ui.ctx(), id);
                }
                editor.request_focus = false;
            }

            // Up/Down stepping
            let up = ui.input(|i| i.key_pressed(egui::Key::ArrowUp));
            let down = ui.input(|i| i.key_pressed(egui::Key::ArrowDown));
            if let Some(m) = meta.as_ref().filter(|_| up || down) {
                if let Ok(current) = editor.field_buffer.parse::<f32>() {
                    let delta = if up { m.step } else { -m.step };
                    let stepped = (current + delta).clamp(m.min, m.max);
                    editor.field_buffer = format_value(stepped, m.step);
                    new_value = Some(editor.field_buffer.clone());
                }
            }

            // Apply on every text change
            if response.changed() {
                new_value = Some(editor.field_buffer.clone());
            }

            // Confirm on Enter or lost focus
            let enter = ui.input(|i| i.key_pressed(egui::Key::Enter));
            if enter || response.lost_focus() {
                new_value = Some(editor.field_buffer.clone());
                editor.active_field = None;
                editor.end_history_gesture();
            }

            // Escape cancels
            if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                editor.active_field = None;
                // Values apply live on every valid text change. Escape closes
                // that one transaction; fingerprint equality removes a truly
                // unchanged edit from history.
                editor.end_history_gesture();
            }

            response
        } else {
            // Inactive: clickable value pill (fixed width via allocate_ui)
            let color = if value == "true" || value == "false" {
                BOOL_COLOR
            } else if value.parse::<f64>().is_ok() {
                VALUE_COLOR
            } else {
                STRING_COLOR
            };

            let inner = ui.allocate_ui_with_layout(
                egui::vec2(PILL_WIDTH, ui.spacing().interact_size.y),
                egui::Layout::left_to_right(egui::Align::Center),
                |ui| {
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(value)
                                .monospace()
                                .size(13.0)
                                .color(color)
                                .background_color(PILL_BG),
                        )
                        .sense(egui::Sense::click()),
                    )
                },
            );

            let response = inner.inner;
            if response.clicked() && editor.begin_history_gesture(format!("Edit {key}")) {
                editor.active_field = Some(key.to_string());
                editor.field_buffer = value.to_string();
                editor.request_focus = true;
            }

            response
        };

        // Show comment as hover tooltip instead of inline
        if let Some(m) = &meta {
            pill_response.on_hover_text(format!("{} [{} to {}]", m.desc, m.min, m.max));
        }
    });

    new_value
}

fn format_value(v: f32, step: f32) -> String {
    if step >= 1.0 {
        format!("{:.1}", v)
    } else if step >= 0.01 {
        format!("{:.2}", v)
    } else {
        format!("{:.3}", v)
    }
}

/// Build the YAML editor content (rendered inside the shared left panel).
pub fn build_yaml_editor_content(
    ui: &mut egui::Ui,
    layers: &mut [Layer],
    master_effects: &mut EffectUniforms,
    editor: &mut EditorState,
) {
    // Tab bar: Master + per-layer tabs
    ui.horizontal(|ui| {
        if ui.selectable_label(editor.tab == 0, "Master").clicked() && editor.tab != 0 {
            editor.end_history_gesture();
            editor.tab = 0;
            editor.active_field = None;
        }

        for i in 0..layers.len() {
            let tab_id = i + 1;
            let label = format!("Layer {}", i + 1);
            if ui.selectable_label(editor.tab == tab_id, &label).clicked() && editor.tab != tab_id {
                editor.end_history_gesture();
                editor.tab = tab_id;
                editor.active_field = None;
            }
        }
    });

    ui.separator();

    // Render fields based on active tab
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            match editor.tab {
                0 => {
                    // Master effects — grouped
                    let config = EffectsConfig::from_uniforms(master_effects);
                    let groups = config.grouped_fields();
                    let mut updated_config = config.clone();
                    let mut changed = false;

                    for (group_name, fields) in &groups {
                        ui.add_space(6.0);
                        ui.label(
                            egui::RichText::new(format!("# {}", group_name))
                                .monospace()
                                .size(11.0)
                                .color(GROUP_COLOR),
                        );
                        for (key, value) in fields {
                            if let Some(new_val) = param_row(ui, key, value, editor, KEY_COL_WIDTH)
                            {
                                if updated_config.set_field(key, &new_val) {
                                    changed = true;
                                }
                            }
                        }
                    }

                    if changed {
                        updated_config.apply_to_uniforms(master_effects);
                    }
                }
                n => {
                    let idx = n - 1;
                    if idx < layers.len() {
                        let config = LayerConfig::from_layer(&layers[idx]);
                        let top_fields = config.top_fields();
                        let effect_groups = config.effects.grouped_fields();

                        let mut updated_config = config.clone();
                        let mut changed = false;

                        // Layer top-level fields
                        for (key, value) in &top_fields {
                            if *key == "filename" {
                                // Read-only filename
                                ui.horizontal(|ui| {
                                    ui.spacing_mut().item_spacing.x = 2.0;
                                    ui.allocate_ui_with_layout(
                                        egui::vec2(KEY_COL_WIDTH, ui.spacing().interact_size.y),
                                        egui::Layout::left_to_right(egui::Align::Center),
                                        |ui| {
                                            ui.label(
                                                egui::RichText::new("filename:")
                                                    .monospace()
                                                    .size(13.0)
                                                    .color(KEY_COLOR),
                                            );
                                        },
                                    );
                                    ui.label(
                                        egui::RichText::new(value)
                                            .monospace()
                                            .size(13.0)
                                            .color(STRING_COLOR),
                                    );
                                });
                                continue;
                            }
                            if let Some(new_val) = param_row(ui, key, value, editor, KEY_COL_WIDTH)
                            {
                                if updated_config.set_field(key, &new_val) {
                                    changed = true;
                                }
                            }
                        }

                        ui.add_space(8.0);
                        ui.label(
                            egui::RichText::new("effects:")
                                .monospace()
                                .size(13.0)
                                .color(KEY_COLOR),
                        );

                        // Effect fields — grouped
                        for (group_name, fields) in &effect_groups {
                            ui.add_space(6.0);
                            ui.label(
                                egui::RichText::new(format!("  # {}", group_name))
                                    .monospace()
                                    .size(11.0)
                                    .color(GROUP_COLOR),
                            );
                            for (key, value) in fields {
                                if let Some(new_val) =
                                    param_row(ui, key, value, editor, KEY_COL_WIDTH)
                                {
                                    if updated_config.effects.set_field(key, &new_val) {
                                        changed = true;
                                    }
                                }
                            }
                        }

                        if changed {
                            updated_config.apply_to_layer(&mut layers[idx]);
                        }
                    }
                }
            }
        });

    // Status line
    ui.add_space(4.0);
    ui.separator();
    ui.weak("click value to edit · ↑↓ step · enter to confirm");
}

/// Save full patch state to a YAML file via native dialog.
// This UI boundary deliberately mirrors the complete PatchState capture
// inputs so call sites cannot accidentally omit performance state.
#[allow(
    dead_code,
    reason = "native dialog save entrypoint remains a compatibility UI API"
)]
#[allow(
    clippy::too_many_arguments,
    reason = "the compatibility UI boundary mirrors complete patch capture inputs"
)]
pub fn save_patch(
    master: PatchMasterVisual<'_>,
    layers: &[Layer],
    ntsc_params: &NtscParams,
    mod_matrix: &crate::modulation::ModMatrix,
    temporal: &crate::effects::params::TemporalParams,
    transport: super::PatchTransportState,
    morph: &crate::morph::Morph,
    scenes: &crate::performance::Scenes,
) {
    let mut patch = PatchState::capture(
        master,
        layers,
        ntsc_params,
        mod_matrix,
        temporal,
        transport,
        morph,
    );
    patch.scenes = scenes.clone();
    let yaml = serde_yaml::to_string(&patch).unwrap_or_default();

    if let Some(path) = rfd::FileDialog::new()
        .set_file_name("patch.yaml")
        .add_filter("YAML", &["yaml", "yml"])
        .save_file()
    {
        if let Err(e) = std::fs::write(&path, &yaml) {
            eprintln!("Failed to save patch: {e}");
        }
    }
}

/// Save the complete M2 creative graph. Capture resolves all live stable IDs
/// to saved positional identities before the file dialog is shown, so a
/// failed mapping cannot leave a partially authored patch on disk.
#[allow(
    dead_code,
    reason = "pre-M4 compatibility wrapper preserves the historical native API"
)]
#[allow(clippy::too_many_arguments)]
pub fn save_patch_with_composition(
    master: PatchMasterVisual<'_>,
    layers: &[Layer],
    master_rack: &RuntimeVisualRack,
    layer_racks: &[RuntimeVisualRack],
    composition: &RuntimeComposition,
    ntsc_params: &NtscParams,
    mod_matrix: &crate::modulation::ModMatrix,
    temporal: &crate::effects::params::TemporalParams,
    transport: super::PatchTransportState,
    morph: &crate::morph::Morph,
    scenes: &crate::performance::Scenes,
) -> Result<(), String> {
    save_patch_with_composition_and_motion(
        master,
        &MotionParams::default(),
        layers,
        master_rack,
        layer_racks,
        composition,
        ntsc_params,
        mod_matrix,
        temporal,
        transport,
        morph,
        scenes,
    )
}

/// Save the complete creative graph plus authored M4 master/layer Motion.
/// Runtime vector fields, carrier pixels, codec products, and telemetry are
/// absent from `PatchState`, so this remains an authored-only transaction.
#[allow(clippy::too_many_arguments)]
pub fn save_patch_with_composition_and_motion(
    master: PatchMasterVisual<'_>,
    master_motion: &MotionParams,
    layers: &[Layer],
    master_rack: &RuntimeVisualRack,
    layer_racks: &[RuntimeVisualRack],
    composition: &RuntimeComposition,
    ntsc_params: &NtscParams,
    mod_matrix: &crate::modulation::ModMatrix,
    temporal: &crate::effects::params::TemporalParams,
    transport: super::PatchTransportState,
    morph: &crate::morph::Morph,
    scenes: &crate::performance::Scenes,
) -> Result<(), String> {
    let mut patch = PatchState::capture_with_composition_and_motion(
        master,
        master_motion,
        layers,
        master_rack,
        layer_racks,
        composition,
        ntsc_params,
        mod_matrix,
        temporal,
        transport,
        morph,
    )?;
    patch.scenes = scenes.clone();
    let yaml =
        serde_yaml::to_string(&patch).map_err(|error| format!("serialize patch: {error}"))?;

    let Some(path) = rfd::FileDialog::new()
        .set_file_name("patch.yaml")
        .add_filter("YAML", &["yaml", "yml"])
        .save_file()
    else {
        return Ok(());
    };
    std::fs::write(&path, yaml).map_err(|error| format!("write patch {}: {error}", path.display()))
}

/// Pick and parse a patch via the native dialog. Cancellation is distinct from
/// parse/read failure so browser and keyboard callers can report honestly.
pub fn choose_patch() -> Result<Option<(PatchState, std::path::PathBuf)>, String> {
    let Some(path) = rfd::FileDialog::new()
        .add_filter("YAML", &["yaml", "yml"])
        .pick_file()
    else {
        return Ok(None);
    };
    load_patch_path(&path).map(|patch| Some((patch, path)))
}

/// Parse a specific patch path through the same bounded semantic entry point
/// used by the native picker.
pub fn load_patch_path(path: &std::path::Path) -> Result<PatchState, String> {
    let yaml = std::fs::read_to_string(path)
        .map_err(|error| format!("read patch {}: {error}", path.display()))?;
    serde_yaml::from_str::<PatchState>(&yaml)
        .map_err(|error| format!("parse patch {}: {error}", path.display()))
}

/// Compatibility wrapper retained for existing native callers.
#[allow(dead_code)]
pub fn load_patch() -> Option<(PatchState, std::path::PathBuf)> {
    match choose_patch() {
        Ok(result) => result,
        Err(error) => {
            eprintln!("Failed to load patch: {error}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP_PATCH: AtomicU64 = AtomicU64::new(0);

    struct TempPatch(std::path::PathBuf);

    impl TempPatch {
        fn new() -> Self {
            let sequence = NEXT_TEMP_PATCH.fetch_add(1, Ordering::Relaxed);
            Self(std::env::temp_dir().join(format!(
                "collide-o-scope-patch-test-{}-{sequence}.yaml",
                std::process::id()
            )))
        }
    }

    impl Drop for TempPatch {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn load_patch_path_accepts_legacy_media_freeze_shape() {
        let path = TempPatch::new();
        std::fs::write(&path.0, "master: {}\nlayers: []\nmaster_paused: true\n").unwrap();

        let legacy = load_patch_path(&path.0).unwrap();
        assert!(legacy.master_paused);
        assert!(!legacy.media_frozen);

        std::fs::write(
            &path.0,
            "master: {}\nlayers: []\nmaster_paused: false\nmedia_frozen: true\n",
        )
        .unwrap();
        let current = load_patch_path(&path.0).unwrap();
        assert!(!current.master_paused);
        assert!(current.media_frozen);
    }

    #[test]
    fn native_editor_boundaries_are_ordered_and_one_focus_session_is_one_gesture() {
        let mut editor = EditorState::default();
        editor.begin_history_gesture("Edit brightness".to_string());
        // Live character/step changes do not manufacture more checkpoints.
        assert_eq!(editor.take_history_boundaries().len(), 1);
        editor.end_history_gesture();
        editor.end_history_gesture();
        let boundaries = editor.take_history_boundaries();
        assert_eq!(boundaries.len(), 1);
        let EditorHistoryBoundary::End { gesture_id } = boundaries[0] else {
            panic!("expected end boundary")
        };
        assert_eq!(gesture_id, 1);

        editor.begin_history_gesture("Edit contrast".to_string());
        editor.begin_history_gesture("Edit saturation".to_string());
        assert_eq!(
            editor.take_history_boundaries(),
            vec![
                EditorHistoryBoundary::Begin {
                    gesture_id: 2,
                    label: "Edit contrast".to_string(),
                    category: "native_editor".to_string(),
                },
                EditorHistoryBoundary::End { gesture_id: 2 },
                EditorHistoryBoundary::Begin {
                    gesture_id: 3,
                    label: "Edit saturation".to_string(),
                    category: "native_editor".to_string(),
                },
            ]
        );
    }

    #[test]
    fn undrained_native_boundary_overflow_never_emits_an_unmatched_begin() {
        let mut editor = EditorState::default();
        for index in 0..(MAX_EDITOR_HISTORY_BOUNDARIES / 2) {
            assert!(editor.begin_history_gesture(format!("Edit {index}")));
            editor.end_history_gesture();
        }
        assert_eq!(
            editor.history_boundaries.len(),
            MAX_EDITOR_HISTORY_BOUNDARIES
        );
        assert!(!editor.begin_history_gesture("Overflow".to_string()));
        assert!(editor.take_history_boundaries().is_empty());
        assert!(editor.active_history_gesture.is_none());
    }
}
