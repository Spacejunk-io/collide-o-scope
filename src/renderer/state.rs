use std::cell::RefCell;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use winit::window::Window;

use crate::effects::params::TemporalParams;
use crate::evaluated_frame::{EvaluatedFramePlan, ResolvedImageInput, SourceTap};
use crate::layers::Layer;
use crate::ntsc::{
    checked_rgba_frame_len, validate_selective_ntsc_gpu_staging_limit,
    validate_selective_ntsc_live_memory, NtscFrameMetadata, SelectiveNtscBatch,
    SelectiveNtscGeneration, SelectiveNtscPlan,
};
use crate::spatial::EffectPassUniforms;
#[cfg(test)]
use crate::temporal::temporal_key_reference_layer;
pub(crate) use crate::temporal::TemporalState;
use crate::temporal::{
    TemporalFrameAction, TemporalFrameInput, TemporalGpuUniforms, TemporalOriginalsGpuUniforms,
    TemporalResetCause, TemporalStateMetrics, TEMPORAL_HISTORY_LEN,
};

use super::blend::composite_shader_source;
use super::compositor::{
    encode_matte_composite, encode_program_history_copy, validate_selective_matte_topology,
    ImageRoutingGpuResources, ImageTapTexture, MatteCompositePipeline, MatteCompositeUniforms,
    MatteResourceLimits, MatteResourcePlan,
};
use super::readback::{
    PreparedRgbaReadback, RecorderReadbackAdmission, RecorderReadbackAllocationSnapshot,
    RecorderReadbackError, RecorderReadbackPoll, RecorderReadbackReadiness,
    RecorderReadbackRequest, RecorderReadbackReservation, RecorderReadbackTag,
};

/// Frame-local ownership of the only live-layer state the GPU encoder needs:
/// immutable texture views and their plan identity. Creating fresh views pins
/// the exact source textures selected for this frame even if transport/UI code
/// replaces a `Layer` texture after evaluation.
pub struct LiveFrameResources {
    layers: Vec<LiveLayerResource>,
}

struct LiveLayerResource {
    source: SourceTap,
    texture_view: wgpu::TextureView,
}

impl LiveFrameResources {
    pub fn capture(layers: &[Layer]) -> Self {
        Self {
            layers: layers
                .iter()
                .enumerate()
                .map(|(slot, layer)| LiveLayerResource {
                    source: SourceTap::new(layer.layer_id(), slot, layer.width, layer.height),
                    texture_view: layer
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default()),
                })
                .collect(),
        }
    }

    pub fn sources(&self) -> impl ExactSizeIterator<Item = SourceTap> + '_ {
        self.layers.iter().map(|layer| layer.source)
    }

    pub(crate) fn texture_view(&self, source: SourceTap) -> Result<&wgpu::TextureView, String> {
        let resource = self.layers.get(source.slot).ok_or_else(|| {
            format!(
                "evaluated layer {} refers to missing live resource slot {}",
                source.stable_id, source.slot
            )
        })?;
        if resource.source != source {
            return Err(format!(
                "evaluated layer resource mismatch at slot {}: planned id {} at {}x{}, captured id {} at {}x{}",
                source.slot,
                source.stable_id,
                source.size[0],
                source.size[1],
                resource.source.stable_id,
                resource.source.size[0],
                resource.source.size[1]
            ));
        }
        Ok(&resource.texture_view)
    }
}

/// Frames of output history kept for temporal effects (0.8s at 30fps).
pub const HISTORY_LEN: u32 = TEMPORAL_HISTORY_LEN;

/// Full-frame RGBA textures owned unconditionally by a live renderer: the
/// history ring, one feedback texture, three composite targets, and one held
/// audience frame. This is an exact payload floor, not a VRAM estimate: driver
/// padding, surfaces, buffers, and media/layer textures are intentionally not
/// included.
const RENDERER_OWNED_FULL_FRAME_RGBA_TEXTURES: u64 = HISTORY_LEN as u64 + 5;

fn renderer_owned_full_frame_texture_floor_bytes(width: u32, height: u32) -> Option<u64> {
    u64::from(width)
        .checked_mul(u64::from(height))?
        .checked_mul(4)?
        .checked_mul(RENDERER_OWNED_FULL_FRAME_RGBA_TEXTURES)
}

#[derive(Debug, Default, PartialEq, Eq)]
struct ScopedGpuErrors {
    out_of_memory: Option<String>,
    internal: Option<String>,
    validation: Option<String>,
}

/// Turn wgpu's three handle-returning error channels into one stable,
/// user-facing failure. Allocation pressure wins over backend and validation
/// noise because it gives the operator the most actionable recovery signal.
fn scoped_gpu_error_message(context: &str, errors: ScopedGpuErrors) -> Option<String> {
    [
        ("out of memory", errors.out_of_memory),
        ("internal/backend", errors.internal),
        ("validation", errors.validation),
    ]
    .into_iter()
    .find_map(|(kind, error)| error.map(|error| format!("{context} failed ({kind}): {error}")))
}

/// Construct handle-returning wgpu resources under all recoverable error
/// scopes and inspect those scopes before any returned handle is used.
fn create_gpu_resources_checked<T>(
    device: &wgpu::Device,
    context: &str,
    create: impl FnOnce() -> T,
) -> Result<T, String> {
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let resources = create();
    let errors = ScopedGpuErrors {
        out_of_memory: pollster::block_on(out_of_memory.pop()).map(|error| error.to_string()),
        internal: pollster::block_on(internal.pop()).map(|error| error.to_string()),
        validation: pollster::block_on(validation.pop()).map(|error| error.to_string()),
    };
    match scoped_gpu_error_message(context, errors) {
        Some(error) => Err(error),
        None => Ok(resources),
    }
}

fn validate_readback_buffer_size(
    padded_row_bytes: u32,
    height: u32,
    max_buffer_size: u64,
) -> Result<u64, String> {
    let size = u64::from(padded_row_bytes)
        .checked_mul(u64::from(height))
        .ok_or_else(|| "audience readback buffer size overflowed".to_string())?;
    if size == 0 {
        return Err("audience readback buffer cannot be empty".to_string());
    }
    if size > max_buffer_size {
        return Err(format!(
            "audience readback requires {size} bytes, exceeding the GPU buffer limit of \
             {max_buffer_size} bytes"
        ));
    }
    Ok(size)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct SelectiveNtscAllocationSnapshot {
    pub scratch_bytes: u64,
    pub staging_bytes: u64,
}

impl SelectiveNtscAllocationSnapshot {
    pub const fn total_bytes(self) -> u64 {
        self.scratch_bytes.saturating_add(self.staging_bytes)
    }
}

/// Exact payload accounting for the two RGBA8 selective-NTSC scratch images
/// and its current mapped staging capacity. Renderer dimensions have already
/// passed the device edge limits; saturating arithmetic keeps operator
/// telemetry truthful even for hostile synthetic dimensions.
const fn selective_ntsc_allocation_snapshot_for(
    width: u32,
    height: u32,
    staging_capacity: u64,
) -> SelectiveNtscAllocationSnapshot {
    let scratch_bytes = (width as u64)
        .saturating_mul(height as u64)
        .saturating_mul(4)
        .saturating_mul(2);
    SelectiveNtscAllocationSnapshot {
        scratch_bytes,
        staging_bytes: staging_capacity,
    }
}

/// Internal render targets carry sRGB-encoded bytes but are sampled through
/// sRGB views so all shader math happens in linear light.
const COMPOSITE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// egui-wgpu performs its own gamma-to-linear conversion. Giving it an sRGB
/// view would make the hardware decode first and egui decode a second time.
const EGUI_OUTPUT_VIEW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const EGUI_OUTPUT_VIEW_FORMATS: &[wgpu::TextureFormat] = &[EGUI_OUTPUT_VIEW_FORMAT];

const fn transparent_accumulation_clear() -> wgpu::Color {
    wgpu::Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    }
}

/// Upload one small uniform without `DeviceExt::create_buffer_init`.
///
/// That convenience helper creates a mapped-at-creation buffer and immediately
/// calls `get_mapped_range_mut`. If the device was lost moments earlier, wgpu
/// represents the allocation as an invalid labeled resource and the mapping
/// accessor is deliberately fatal. Queue upload keeps validation recoverable
/// through the renderer's health latch instead of turning device loss into an
/// unrelated mapped-range panic.
fn create_uploaded_uniform<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &'static str,
    value: &T,
) -> wgpu::Buffer {
    let bytes = bytemuck::bytes_of(value);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytes);
    buffer
}

#[cfg(test)]
mod temporal_state_tests {
    use super::*;
    use crate::effects::params::TEMPORAL_REFERENCE_FPS;
    use crate::temporal::{
        CollisionAtlasParams, RefreshGardenGate, RefreshGardenParams, TemporalFrameEvents,
        TemporalFreezeState, TemporalInterpolation, TemporalLoomParams, TemporalOriginalsParams,
        TemporalResetCause, TemporalTopology,
    };
    use sha2::{Digest, Sha256};

    #[test]
    fn gpu_health_latches_the_first_terminal_error() {
        let health = GpuHealth::default();
        assert_eq!(health.error(), None);
        health.record("device lost during output configure".to_string());
        health.record("later validation noise".to_string());
        assert_eq!(
            health.error().as_deref(),
            Some("device lost during output configure")
        );
    }

    #[test]
    fn renderer_owned_full_frame_texture_floor_is_exact_and_checked() {
        assert_eq!(RENDERER_OWNED_FULL_FRAME_RGBA_TEXTURES, 29);
        assert_eq!(
            renderer_owned_full_frame_texture_floor_bytes(1280, 720),
            Some(106_905_600)
        );
        assert_eq!(
            renderer_owned_full_frame_texture_floor_bytes(3840, 2160),
            Some(962_150_400)
        );
        assert_eq!(
            renderer_owned_full_frame_texture_floor_bytes(u32::MAX, u32::MAX),
            None
        );
    }

    #[test]
    fn scoped_gpu_errors_are_descriptive_and_prioritize_allocation_pressure() {
        let message = scoped_gpu_error_message(
            "selective NTSC GPU resources",
            ScopedGpuErrors {
                out_of_memory: Some("allocation rejected".to_string()),
                internal: Some("backend also reported an error".to_string()),
                validation: Some("descriptor also rejected".to_string()),
            },
        );
        assert_eq!(
            message.as_deref(),
            Some("selective NTSC GPU resources failed (out of memory): allocation rejected")
        );
        assert_eq!(
            scoped_gpu_error_message(
                "audience readback buffer allocation",
                ScopedGpuErrors {
                    validation: Some("size is invalid".to_string()),
                    ..ScopedGpuErrors::default()
                },
            )
            .as_deref(),
            Some("audience readback buffer allocation failed (validation): size is invalid")
        );
        assert_eq!(
            scoped_gpu_error_message("unused", ScopedGpuErrors::default()),
            None
        );
    }

    #[test]
    fn audience_readback_buffer_size_is_checked_against_the_device_limit() {
        assert_eq!(
            validate_readback_buffer_size(5_120, 720, 8_000_000),
            Ok(3_686_400)
        );
        assert!(validate_readback_buffer_size(5_120, 720, 3_686_399)
            .unwrap_err()
            .contains("exceeding the GPU buffer limit"));
        assert_eq!(
            validate_readback_buffer_size(u32::MAX, u32::MAX, u64::MAX),
            Ok(u64::from(u32::MAX) * u64::from(u32::MAX))
        );
        assert!(validate_readback_buffer_size(0, 720, u64::MAX)
            .unwrap_err()
            .contains("cannot be empty"));
    }

    fn advance(state: &mut TemporalState, delta_seconds: f32) -> u32 {
        let plan = state.stage_frame(
            &TemporalParams::default(),
            TemporalFrameInput::legacy(delta_seconds, true),
            [1_920, 1_080],
        );
        let ticks = plan.observation_ticks;
        state.commit_staged();
        ticks
    }

    fn plan_and_commit(
        state: &mut TemporalState,
        delta_seconds: f32,
        advance_program: bool,
    ) -> TemporalFrameAction {
        let action = state
            .stage_frame(
                &TemporalParams::default(),
                TemporalFrameInput::legacy(delta_seconds, advance_program),
                [1_920, 1_080],
            )
            .action;
        state.commit_staged();
        action
    }

    #[test]
    fn history_is_primed_then_advances_at_thirty_hz() {
        let mut state = TemporalState::default();

        assert_eq!(advance(&mut state, 1.0 / 60.0), 1);
        assert_eq!(state.history_valid, 1);
        assert_eq!(advance(&mut state, 1.0 / 60.0), 0);
        assert_eq!(advance(&mut state, 1.0 / 60.0), 1);
        assert_eq!(state.history_valid, 2);
        assert_eq!(state.history_write, 1);
    }

    #[test]
    fn history_cadence_is_bounded_by_both_program_clock_and_real_observations() {
        for (render_fps, expected_observations) in [(24, 24), (30, 30), (60, 30), (240, 30)] {
            let mut state = TemporalState::default();
            for _ in 0..render_fps {
                advance(&mut state, 1.0 / render_fps as f32);
            }
            assert_eq!(
                state.total_history_frames, expected_observations,
                "unexpected one-second history cadence at {render_fps} fps"
            );
        }
    }

    #[test]
    fn paused_startup_holds_clean_output_without_manufacturing_history() {
        let mut state = TemporalState::default();
        assert_eq!(
            plan_and_commit(&mut state, 0.0, false),
            TemporalFrameAction::PrimeFrozenOutput
        );
        for _ in 1..120 {
            assert_eq!(
                plan_and_commit(&mut state, 0.0, false),
                TemporalFrameAction::HoldFrozenOutput
            );
        }
        assert!(!state.initialized);
        assert_eq!(state.history_valid, 0);
        assert_eq!(temporal_key_reference_layer(0, 0, 1.0), None);

        assert_eq!(
            plan_and_commit(&mut state, 0.0, false),
            TemporalFrameAction::HoldFrozenOutput
        );
        assert_eq!(state.history_valid, 0);

        let resume = state.stage_frame(
            &TemporalParams::default(),
            TemporalFrameInput::legacy(0.0, true),
            [1_920, 1_080],
        );
        assert_eq!(
            resume.action,
            TemporalFrameAction::Advance {
                record_history: true
            },
            "unpaused frame zero must prime clean history for live/export parity"
        );
        assert!(state.initialized);
        assert_eq!(state.history_valid, 1, "the complete next state is staged");
        state.discard_staged();
        assert_eq!(state.history_valid, 0, "discard restores committed state");
    }

    #[test]
    fn pause_and_resume_never_accrue_history_catch_up_debt() {
        let mut state = TemporalState::default();
        assert_eq!(advance(&mut state, 1.0 / 30.0), 1);
        state.feedback_valid = true;
        let frozen_write = state.history_write;
        let frozen_valid = state.history_valid;
        let frozen_total = state.total_history_frames;
        let frozen_accumulator = state.history_accumulator;

        for _ in 0..240 {
            assert_eq!(
                plan_and_commit(&mut state, 1.0, false),
                TemporalFrameAction::HoldFrozenOutput
            );
        }
        assert_eq!(state.history_write, frozen_write);
        assert_eq!(state.history_valid, frozen_valid);
        assert_eq!(state.total_history_frames, frozen_total);
        assert_eq!(state.history_accumulator, frozen_accumulator);

        assert_eq!(
            plan_and_commit(&mut state, 1.0 / 60.0, true),
            TemporalFrameAction::Advance {
                record_history: false
            },
            "resume uses only post-resume program time"
        );
        assert_eq!(state.history_write, frozen_write);
        assert_eq!(state.history_valid, frozen_valid);
    }

    #[test]
    fn valid_history_never_exceeds_the_ring() {
        let mut state = TemporalState::default();
        advance(&mut state, 1.0 / TEMPORAL_REFERENCE_FPS);
        for _ in 0..(HISTORY_LEN * 2) {
            advance(&mut state, 1.0 / TEMPORAL_REFERENCE_FPS);
        }

        assert_eq!(state.history_valid, HISTORY_LEN);
        assert!(state.history_write < HISTORY_LEN as usize);
    }

    #[test]
    fn temporal_uniform_layout_matches_four_shader_vec4s() {
        assert_eq!(std::mem::size_of::<TemporalGpuUniforms>(), 64);
    }

    #[test]
    fn temporal_key_reference_tracks_history_ticks_without_sampling_unwritten_layers() {
        // The first frame has only the current image in history, so a prior
        // reference does not exist and the key must pass through.
        assert_eq!(temporal_key_reference_layer(0, 0, 1.0), None);
        // At high display rates there can be no history write this frame; the
        // latest ring slot is then the correct one-frame reference.
        assert_eq!(temporal_key_reference_layer(0, 1, 1.0), Some(0));
        // Before a new snapshot lands, depth one still means the newest
        // genuinely earlier image, independent of the elapsed tick count.
        assert_eq!(temporal_key_reference_layer(0, 1, 1.0), Some(0));
        // Deeper requests remain invalid until that many clean snapshots exist.
        assert_eq!(temporal_key_reference_layer(0, 1, 2.0), None);
        // Ring wrap is explicit and deterministic.
        assert_eq!(temporal_key_reference_layer(0, HISTORY_LEN, 3.0), Some(22));
    }

    #[test]
    fn catch_up_ticks_record_only_one_real_observation() {
        let mut state = TemporalState::default();
        assert_eq!(advance(&mut state, 1.0 / TEMPORAL_REFERENCE_FPS), 1);
        let prior_write = state.history_write;
        let prior_valid = state.history_valid;

        assert_eq!(advance(&mut state, 2.0 / TEMPORAL_REFERENCE_FPS), 2);
        assert_eq!(
            state.history_write,
            (prior_write + 1) % HISTORY_LEN as usize
        );
        assert_eq!(state.history_valid, prior_valid + 1);
        assert_eq!(
            temporal_key_reference_layer(prior_write, prior_valid, 1.0),
            Some(prior_write),
            "two elapsed ticks must still reference the prior real image"
        );
    }

    #[test]
    fn a_large_stall_cannot_replace_the_whole_ring_with_one_image() {
        let mut state = TemporalState::default();
        for _ in 0..HISTORY_LEN {
            advance(&mut state, 1.0 / TEMPORAL_REFERENCE_FPS);
        }
        let prior_write = state.history_write;

        assert_eq!(advance(&mut state, 30.0), HISTORY_LEN);
        assert_eq!(state.history_valid, HISTORY_LEN);
        assert_eq!(
            state.history_write,
            (prior_write + 1) % HISTORY_LEN as usize
        );
        assert_eq!(
            temporal_key_reference_layer(prior_write, HISTORY_LEN, 1.0),
            Some(prior_write)
        );
        assert_eq!(
            temporal_key_reference_layer(prior_write, HISTORY_LEN, 23.0),
            Some((prior_write + HISTORY_LEN as usize - 22) % HISTORY_LEN as usize)
        );
    }

    #[test]
    fn reset_revokes_all_temporal_validity() {
        let mut state = TemporalState::default();
        advance(&mut state, 1.0 / TEMPORAL_REFERENCE_FPS);
        state.feedback_valid = true;

        state.reset();

        assert_eq!(state.history_valid, 0);
        assert!(!state.feedback_valid);
        assert!(!state.initialized);
        assert_eq!(state.total_history_frames, 0);
    }

    #[test]
    fn temporal_shader_multiplies_straight_alpha_and_never_premultiplies_output_rgb() {
        let shader = include_str!("../shaders/temporal.wgsl");
        assert!(shader.contains("color.a *= mask"));
        assert!(shader.contains("let current_covered = current.rgb * current.a"));
        assert!(!shader.contains("color.rgb *= mask"));
        assert!(shader.contains("u.key_valid > 0.5"));
        let guard = shader
            .find("if requested_depth >= 1.0 && max_depth >= 1.0")
            .expect("slit-scan materialized-history guard");
        let sample = shader
            .find("let hist = textureSample(history_tex")
            .expect("slit-scan history sample");
        assert!(
            guard < sample,
            "startup must not even sample an uninitialized history layer"
        );
    }

    #[test]
    fn slitscan_virtual_current_exposes_only_materialized_past_layers() {
        fn virtual_ring(previous_write: usize, previous_valid: u32) -> (usize, u32) {
            let write = if previous_valid == 0 {
                0
            } else {
                (previous_write + 1) % HISTORY_LEN as usize
            };
            let valid = if previous_valid == 0 {
                0
            } else {
                (previous_valid + 1).min(HISTORY_LEN)
            };
            (write, valid)
        }

        assert_eq!(virtual_ring(0, 0), (0, 0));
        assert_eq!(virtual_ring(0, 1), (1, 2));
        assert_eq!(virtual_ring(23, HISTORY_LEN), (0, HISTORY_LEN));
    }

    #[test]
    fn effects_shader_exposes_all_static_key_modes_in_straight_alpha() {
        let shader = include_str!("../shaders/effects.wgsl");
        assert!(shader.contains("Mode 3 removes the selected"));
        assert!(shader.contains("mode 4 retains it"));
        assert!(shader.contains("alpha *= outside"));
        assert!(!shader.contains("rgb *= outside"));
        assert!(shader.contains("dot(rgb, vec3f(0.2126, 0.7152, 0.0722))"));
        assert!(!shader.contains("dot(rgb, vec3f(0.299, 0.587, 0.114))"));
        assert!(shader.contains("MAX_DISPLAY_CHROMA_DISTANCE: f32 = 1.1913178"));
        assert!(shader.contains("/ MAX_DISPLAY_CHROMA_DISTANCE"));
    }

    #[test]
    fn display_chroma_normalization_spans_the_rgb_cube_and_ignores_neutral_luma() {
        fn chroma([r, g, b]: [f32; 3]) -> [f32; 2] {
            let y = r * 0.2126 + g * 0.7152 + b * 0.0722;
            [(b - y) / 1.8556, (r - y) / 1.5748]
        }
        fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
        }

        assert!(distance(chroma([0.0; 3]), chroma([1.0; 3])) < 1.0e-6);
        let cube_span = distance(chroma([0.0, 1.0, 0.0]), chroma([1.0, 0.0, 1.0]));
        assert!((cube_span - 1.1913178).abs() < 1.0e-6);
        assert!((cube_span / 1.1913178 - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn output_capability_selection_handles_empty_and_fallback_lists() {
        assert_eq!(preferred_surface_format(&[]), None);
        assert_eq!(preferred_present_mode(&[]), None);
        assert_eq!(
            preferred_surface_format(&[wgpu::TextureFormat::Rgba8Unorm]),
            Some(wgpu::TextureFormat::Rgba8Unorm)
        );
        assert_eq!(
            preferred_present_mode(&[wgpu::PresentMode::Immediate]),
            Some(wgpu::PresentMode::Immediate)
        );
        assert_eq!(
            preferred_present_mode(&[wgpu::PresentMode::Immediate, wgpu::PresentMode::Fifo]),
            Some(wgpu::PresentMode::Fifo)
        );
    }

    #[test]
    fn live_layer_accumulation_starts_transparent() {
        let clear = transparent_accumulation_clear();
        assert_eq!([clear.r, clear.g, clear.b, clear.a], [0.0; 4]);
    }

    #[test]
    fn master_bypass_path_only_changes_for_a_visible_bypass() {
        assert_eq!(
            master_fx_composition_path([(true, false, 1.0), (true, false, 1.0)]),
            MasterFxCompositionPath::LegacyPostComposite
        );
        assert_eq!(
            master_fx_composition_path([(true, false, 1.0), (false, true, 1.0)]),
            MasterFxCompositionPath::LegacyPostComposite,
            "a hidden bypass must not perturb the legacy frame"
        );
        assert_eq!(
            master_fx_composition_path([(true, false, 1.0), (true, true, 1.0)]),
            MasterFxCompositionPath::ConditionalPerLayer
        );
        for opacity in [0.0, -0.0, -0.5] {
            assert_eq!(
                master_fx_composition_path([(true, true, opacity)]),
                MasterFxCompositionPath::LegacyPostComposite,
                "finite non-contributing opacity must leave the legacy path unchanged"
            );
        }
        assert_eq!(
            master_fx_composition_path([(true, true, f32::NAN)]),
            MasterFxCompositionPath::ConditionalPerLayer
        );
        assert_eq!(
            master_fx_composition_path([(true, true, f32::INFINITY)]),
            MasterFxCompositionPath::ConditionalPerLayer
        );
    }

    #[test]
    fn conditional_master_path_ping_pongs_without_an_extra_texture() {
        assert_eq!(
            conditional_layer_slots(false),
            ConditionalLayerSlots {
                master_output: Some(2),
                composite_output: 1,
            }
        );
        assert_eq!(
            conditional_layer_slots(true),
            ConditionalLayerSlots {
                master_output: None,
                composite_output: 2,
            }
        );
    }

    #[test]
    fn conditional_path_preserves_bottom_to_top_visible_stack_order() {
        assert_eq!(visible_stack_indices([true, false, true, true]), [3, 2, 0]);
    }

    #[test]
    fn composite_shader_uses_straight_alpha_source_over() {
        let shader = composite_shader_source();
        assert!(
            shader.contains("let output_alpha = source_alpha + base_alpha * (1.0 - source_alpha);")
        );
        assert!(shader.contains("output_premultiplied / max(output_alpha, BLEND_EPSILON)"));

        // Transparent base + half-opacity source must retain straight RGB,
        // while alpha carries the half coverage. This is the fringe-sensitive
        // case that the old mix/max equation got wrong.
        let base_alpha = 0.0_f32;
        let source_alpha = 0.5_f32;
        let overlay = [0.8_f32, 0.2, 0.4];
        let output_alpha = source_alpha + base_alpha * (1.0 - source_alpha);
        let output_premultiplied = overlay.map(|channel| channel * source_alpha);
        let output = output_premultiplied.map(|channel| channel / output_alpha);
        assert_eq!(output, overlay);
        assert_eq!(output_alpha, 0.5);
    }

    #[test]
    fn effects_shader_preserves_legacy_sampling_and_exposes_spatial_modes() {
        let shader = include_str!("../shaders/effects.wgsl");
        assert!(shader.contains("@group(0) @binding(2) var nearest_samp: sampler;"));
        assert!(shader.contains("if uniforms.spatial_modes.w == 0u"));
        assert!(shader.contains("return textureSample(tex, samp, output_uv);"));
        assert!(shader.contains("if uniforms.spatial_modes.z == 0u"));
        assert!(shader.contains("sampled = textureSample(tex, nearest_samp, source_uv);"));
        assert!(shader.contains("else if uniforms.spatial_modes.w == 1u"));
        assert!(shader.contains("sampled = textureSample(tex, samp, source_uv);"));
        assert!(shader.contains("sampled = sample_source_premultiplied_linear(source_uv);"));
        assert!(shader.contains("return sampled * coverage;"));
    }

    #[test]
    fn opaque_output_flattens_once_and_blit_does_not_repeat_it() {
        let flatten = include_str!("../shaders/opaque_output.wgsl");
        assert!(flatten.contains("straight.rgb * coverage"));
        assert!(flatten.contains("vec4f(straight.rgb * coverage, 1.0)"));

        let blit = include_str!("../shaders/blit.wgsl");
        assert!(blit.contains("return textureSample(tex, samp, uv);"));
        assert!(!blit.contains("rgb *"));
    }

    #[test]
    fn egui_view_is_the_raw_twin_of_the_srgb_audience_view() {
        assert!(COMPOSITE_FORMAT.is_srgb());
        assert!(!EGUI_OUTPUT_VIEW_FORMAT.is_srgb());
        assert_eq!(
            COMPOSITE_FORMAT.remove_srgb_suffix(),
            EGUI_OUTPUT_VIEW_FORMAT
        );
        assert_eq!(EGUI_OUTPUT_VIEW_FORMATS, &[EGUI_OUTPUT_VIEW_FORMAT]);
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_flattens_straight_alpha_in_linear_light_and_supports_raw_egui_view() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("GPU adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Opaque output regression device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .expect("GPU device");

        let size = wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        };
        let source = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Straight-alpha fixture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COMPOSITE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let source_pixel = [200_u8, 100, 50, 128];
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &source,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &source_pixel,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            size,
        );
        let source_view = source.create_view(&wgpu::TextureViewDescriptor::default());

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Opaque output fixture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COMPOSITE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: EGUI_OUTPUT_VIEW_FORMATS,
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let _egui_raw_view = target.create_view(&wgpu::TextureViewDescriptor {
            format: Some(EGUI_OUTPUT_VIEW_FORMAT),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
        let (pipeline, layout) = build_opaque_output_pipeline(&device);
        let bind_group = build_opaque_output_bind_group(&device, &layout, &source_view, &sampler);
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Opaque output fixture readback"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Opaque output regression encoder"),
        });
        encode_opaque_output(&mut encoder, &pipeline, &bind_group, &target_view);
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(1),
                },
            },
            size,
        );
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (send, receive) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = send.send(result);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU wait");
        receive.recv().expect("map callback").expect("map result");
        let data = slice.get_mapped_range();
        let actual = [data[0], data[1], data[2], data[3]];

        fn srgb_to_linear(encoded: f32) -> f32 {
            if encoded <= 0.04045 {
                encoded / 12.92
            } else {
                ((encoded + 0.055) / 1.055).powf(2.4)
            }
        }
        fn linear_to_srgb(linear: f32) -> f32 {
            if linear <= 0.003_130_8 {
                linear * 12.92
            } else {
                1.055 * linear.powf(1.0 / 2.4) - 0.055
            }
        }
        let alpha = source_pixel[3] as f32 / 255.0;
        let expected_rgb = [source_pixel[0], source_pixel[1], source_pixel[2]].map(|channel| {
            (linear_to_srgb(srgb_to_linear(channel as f32 / 255.0) * alpha) * 255.0).round() as u8
        });
        for (actual, expected) in actual[..3].iter().zip(expected_rgb) {
            assert!(
                actual.abs_diff(expected) <= 1,
                "actual={actual}, expected={expected}"
            );
        }
        assert_eq!(actual[3], 255);
        drop(data);
        staging.unmap();
    }

    struct TemporalOriginalsGpuFixture {
        device: wgpu::Device,
        queue: wgpu::Queue,
        legacy_pipeline: wgpu::RenderPipeline,
        prepared: PreparedTemporalGpuResources,
        composite_textures: [wgpu::Texture; 3],
        composite_views: [wgpu::TextureView; 3],
        history_texture: wgpu::Texture,
        feedback_texture: wgpu::Texture,
        width: u32,
        height: u32,
    }

    impl TemporalOriginalsGpuFixture {
        fn new(width: u32, height: u32) -> Self {
            let instance = wgpu::Instance::default();
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }))
                .expect("GPU adapter");
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("Temporal Originals golden device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                }))
                .expect("GPU device");
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
            let (legacy_pipeline, texture_layout, legacy_uniform_layout) =
                build_temporal_pipeline(&device);
            let (history_texture, history_view) = build_history_texture(&device, width, height);
            let (feedback_texture, feedback_view) = build_feedback_texture(&device, width, height);
            let usage = wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST;
            let composite_textures = std::array::from_fn(|index| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(if index == 0 {
                        "Temporal Originals golden current"
                    } else {
                        "Temporal Originals golden scratch"
                    }),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: COMPOSITE_FORMAT,
                    usage,
                    view_formats: &[],
                })
            });
            let composite_views = std::array::from_fn(|index| {
                composite_textures[index].create_view(&wgpu::TextureViewDescriptor::default())
            });
            let prepared = build_prepared_temporal_gpu_resources(
                &device,
                &texture_layout,
                &legacy_uniform_layout,
                &composite_views[0],
                &history_view,
                &sampler,
                &feedback_view,
            );
            for layer in 0..HISTORY_LEN {
                let pixel = [
                    (layer.wrapping_mul(47).wrapping_add(13) & 0xff) as u8,
                    (layer.wrapping_mul(89).wrapping_add(29) & 0xff) as u8,
                    (layer.wrapping_mul(131).wrapping_add(53) & 0xff) as u8,
                    255,
                ];
                let pixels = pixel.repeat((width * height) as usize);
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &history_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: layer,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &pixels,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(width * 4),
                        rows_per_image: Some(height),
                    },
                    wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                );
            }
            Self {
                device,
                queue,
                legacy_pipeline,
                prepared,
                composite_textures,
                composite_views,
                history_texture,
                feedback_texture,
                width,
                height,
            }
        }

        fn render(
            &self,
            params: &TemporalParams,
            state: &mut TemporalState,
            current_pixel: [u8; 4],
        ) -> Vec<u8> {
            self.render_with_input(
                params,
                state,
                current_pixel,
                TemporalFrameInput::legacy(0.0, true),
            )
        }

        fn render_with_input(
            &self,
            params: &TemporalParams,
            state: &mut TemporalState,
            current_pixel: [u8; 4],
            input: TemporalFrameInput,
        ) -> Vec<u8> {
            let pixels = current_pixel.repeat((self.width * self.height) as usize);
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.composite_textures[0],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &pixels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.width * 4),
                    rows_per_image: Some(self.height),
                },
                wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
            );
            let padded_row = (self.width * 4).div_ceil(256) * 256;
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Temporal Originals golden readback"),
                size: u64::from(padded_row) * u64::from(self.height),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Temporal Originals golden encoder"),
                });
            encode_temporal_prepared_frame(
                &self.queue,
                &mut encoder,
                params,
                &self.legacy_pipeline,
                &self.prepared,
                &self.composite_textures,
                &self.composite_views,
                &self.history_texture,
                &self.feedback_texture,
                state,
                input,
                self.width,
                self.height,
            );
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.composite_textures[0],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_row),
                        rows_per_image: Some(self.height),
                    },
                },
                wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
            );
            self.queue.submit(std::iter::once(encoder.finish()));
            state.commit_staged();
            let slice = staging.slice(..);
            let (send, receive) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = send.send(result);
            });
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("GPU wait");
            receive.recv().expect("map callback").expect("map result");
            let mapped = slice.get_mapped_range();
            let mut output = Vec::with_capacity((self.width * self.height * 4) as usize);
            for row in mapped.chunks_exact(padded_row as usize) {
                output.extend_from_slice(&row[..(self.width * 4) as usize]);
            }
            drop(mapped);
            staging.unmap();
            output
        }
    }

    fn full_temporal_history_state() -> TemporalState {
        let mut state = TemporalState::default();
        state.history_write = (HISTORY_LEN - 1) as usize;
        state.history_valid = HISTORY_LEN;
        state.initialized = true;
        state.total_history_frames = HISTORY_LEN as usize;
        state.total_reference_ticks = u64::from(HISTORY_LEN);
        state
    }

    fn pixel_hash(pixels: &[u8]) -> String {
        format!("{:x}", Sha256::digest(pixels))
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_temporal_originals_topology_interpolation_atlas_and_startup_goldens() {
        let fixture = TemporalOriginalsGpuFixture::new(13, 9);
        let current = [7, 19, 41, 255];
        let mut hashes = Vec::new();
        for topology in [
            TemporalTopology::Linear,
            TemporalTopology::Radial,
            TemporalTopology::Spiral,
            TemporalTopology::Contour,
            TemporalTopology::Folded,
            TemporalTopology::Kaleidoscopic,
        ] {
            for interpolation in [TemporalInterpolation::Floor, TemporalInterpolation::Linear] {
                let params = TemporalParams {
                    originals: TemporalOriginalsParams {
                        loom: TemporalLoomParams {
                            amount: 1.0,
                            topology,
                            interpolation,
                            depth: 0.91,
                            phase: 0.137,
                            scale: 1.23,
                            angle: 17.0,
                            folds: 7,
                            quantization: 8,
                        },
                        ..TemporalOriginalsParams::default()
                    },
                    ..TemporalParams::default()
                };
                let pixels = fixture.render(&params, &mut full_temporal_history_state(), current);
                hashes.push(pixel_hash(&pixels));
            }
        }
        assert_eq!(
            hashes,
            [
                "b424307b25095a25ad496358fc8eb05a146f029f7bed478d6bcc5221ab66ff2d",
                "e274ac119511d9f8a171e5e97148ff6307d50d0482b50e3e43a91937cb915726",
                "71c169cc53b19e1f6b8e07632100083d8a987316ab56cb11c84a2b5e2909a3a7",
                "9e80bb41d5d4e52a3b9250186386a4da50c17ba62def5c433f37fc47109632c3",
                "ec839399a4ea13851d75ddaa03192d382edfb46b720d2ada36c410d10e844368",
                "893ca533aae6b28564d76a3ea56274f5af831c5dcf62261a75348f760d9b9eba",
                "746a6c684e97ccf1ffcd52756346f2477cb2e4ed52445544da4b5a84bff73582",
                "74058cd035c596674e05d885e064455411abf1e4310130dc98eb077690e68289",
                "47251753e3cf1eb92eb6329bf929385c16c0b0aba850e3c3f273b23c501fcdd3",
                "21204ec4184f72778fd772cb70fd5ddad312f7194530e8a19cd10dcc5dcfce18",
                "a8b3d3261300014b536f2ae3a389665a801120f0aff833e75820d984a2ff2b50",
                "f805cb8dd874a6e4af571a90c515588f8067975521b20fb8a996d929bc20b65c",
            ]
            .map(str::to_string),
            "topology/interpolation ring golden changed"
        );

        let atlas_params = |seed| TemporalParams {
            originals: TemporalOriginalsParams {
                atlas: CollisionAtlasParams {
                    amount: 1.0,
                    seed,
                    territories: 19,
                    collision: 0.8,
                },
                ..TemporalOriginalsParams::default()
            },
            ..TemporalParams::default()
        };
        let seed_zero_a = pixel_hash(&fixture.render(
            &atlas_params(0),
            &mut full_temporal_history_state(),
            current,
        ));
        let seed_zero_b = pixel_hash(&fixture.render(
            &atlas_params(0),
            &mut full_temporal_history_state(),
            current,
        ));
        let seed_one = pixel_hash(&fixture.render(
            &atlas_params(1),
            &mut full_temporal_history_state(),
            current,
        ));
        assert_eq!(seed_zero_a, seed_zero_b, "seed zero is deterministic");
        assert_ne!(seed_zero_a, seed_one, "seed is artistically effective");
        assert_eq!(
            seed_zero_a,
            "0c6f9bb2ecc078c4ace5b55886adbc9ec53c1a71d03228706d212dd3f044c5f4"
        );
        assert_eq!(
            seed_one,
            "1def1434bdd829bca31b6e7e9a151182a63e3208f4856ba530dde4eb75858404"
        );

        let startup = fixture.render(&atlas_params(0), &mut TemporalState::default(), current);
        assert!(
            startup.chunks_exact(4).all(|pixel| pixel == current),
            "unwritten magenta/color-coded ring layers must be unreachable at startup"
        );
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_time_displace_keeps_the_ramp_floor_path_exact_and_every_map_reaches_the_pixels() {
        use crate::effects::params::TimeDisplaceMap;

        let fixture = TemporalOriginalsGpuFixture::new(13, 9);
        let current = [7, 19, 41, 255];
        // Loom shares the frame at 0.3 so the originals shader is selected on
        // the pre-B12 build too and the slit output survives the loom mix.
        let params_for = |map, interp| TemporalParams {
            slitscan: 0.8,
            slit_map: map,
            slit_interp: interp,
            originals: TemporalOriginalsParams {
                loom: TemporalLoomParams {
                    amount: 0.3,
                    ..TemporalLoomParams::default()
                },
                ..TemporalOriginalsParams::default()
            },
            ..TemporalParams::default()
        };

        // Pixel exactness of the rewritten slit block: this golden was
        // measured on the pre-B12 build (533b434) with the identical
        // loom + slitscan params, where the block was the inline legacy
        // expression. Ramp with the floor law must reproduce it byte for
        // byte through the new `history_age_sample` routing.
        let ramp = pixel_hash(&fixture.render(
            &params_for(TimeDisplaceMap::Ramp, false),
            &mut full_temporal_history_state(),
            current,
        ));
        assert_eq!(
            ramp, "3abc4fef404f4d37128df8bc53460c423b7c50413a0f7b667655c847c48c59a3",
            "Ramp/floor must be pixel-exact with the pre-B12 slit block"
        );

        // Every non-default map is deterministic and demonstrably reaches
        // the pixels: each hash differs from Ramp's and from every other's.
        let mut seen = vec![ramp];
        for map in [
            TimeDisplaceMap::Brightness,
            TimeDisplaceMap::Radial,
            TimeDisplaceMap::TbcRamp,
            TimeDisplaceMap::Sweep,
        ] {
            let hash = pixel_hash(&fixture.render(
                &params_for(map, false),
                &mut full_temporal_history_state(),
                current,
            ));
            let again = pixel_hash(&fixture.render(
                &params_for(map, false),
                &mut full_temporal_history_state(),
                current,
            ));
            assert_eq!(hash, again, "{map:?} must render deterministically");
            assert!(!seen.contains(&hash), "{map:?} must change the pixels");
            seen.push(hash);
        }
        let interpolated = pixel_hash(&fixture.render(
            &params_for(TimeDisplaceMap::Ramp, true),
            &mut full_temporal_history_state(),
            current,
        ));
        assert!(
            !seen.contains(&interpolated),
            "the interpolation toggle must change the banded pixels"
        );

        // The unwritten-history guard on the GPU: a fresh ring under a
        // non-default map with interpolation must pass the current image
        // through untouched rather than sample unwritten layers.
        let startup = fixture.render(
            &params_for(TimeDisplaceMap::Brightness, true),
            &mut TemporalState::default(),
            current,
        );
        assert!(
            startup.chunks_exact(4).all(|pixel| pixel == current),
            "unwritten ring layers must be unreachable at startup"
        );
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_refresh_garden_gates_recurrence_max_hold_freeze_blackout_reset_and_rate_goldens() {
        let fixture = TemporalOriginalsGpuFixture::new(13, 9);
        let prime = [15, 25, 35, 255];
        let current = [220, 90, 180, 128];
        let garden_params = |gate| TemporalParams {
            originals: TemporalOriginalsParams {
                garden: RefreshGardenParams {
                    amount: 0.65,
                    gate,
                    threshold: 0.4,
                    softness: 0.07,
                    decay: 0.98,
                    max_hold_ticks: 0,
                    ..RefreshGardenParams::default()
                },
                ..TemporalOriginalsParams::default()
            },
            ..TemporalParams::default()
        };
        let mut gate_hashes = Vec::new();
        for gate in [
            RefreshGardenGate::TemporalDelta,
            RefreshGardenGate::Luma,
            RefreshGardenGate::Chroma,
            RefreshGardenGate::CellularRidge,
            RefreshGardenGate::AudioEnergy,
            RefreshGardenGate::AudioOnset,
            RefreshGardenGate::Matte,
            RefreshGardenGate::Motion,
        ] {
            let params = garden_params(gate);
            let mut state = TemporalState::default();
            fixture.render_with_input(
                &params,
                &mut state,
                prime,
                TemporalFrameInput::legacy(0.0, true),
            );
            let input = TemporalFrameInput::new(
                1.0 / 30.0,
                TemporalFreezeState::Running,
                false,
                TemporalFrameEvents {
                    audio_onset_events: 1,
                    ..TemporalFrameEvents::default()
                },
            )
            .with_audio_energy(0.9);
            gate_hashes.push(pixel_hash(
                &fixture.render_with_input(&params, &mut state, current, input),
            ));
        }
        assert_eq!(
            gate_hashes,
            [
                "16bf9582362dcb852f5e3e13cc715b9b0db0660d85b1a48ebf1711ad6a0b8613",
                "16bf9582362dcb852f5e3e13cc715b9b0db0660d85b1a48ebf1711ad6a0b8613",
                "16bf9582362dcb852f5e3e13cc715b9b0db0660d85b1a48ebf1711ad6a0b8613",
                "4409bae76a57060b3252cd9b8608853bcc40a90b3d222c66008922a3706bfd27",
                "1ec23f37246c24d4ec875fd975df0804f676028fd9e22674f5b554702ab914c8",
                "1ec23f37246c24d4ec875fd975df0804f676028fd9e22674f5b554702ab914c8",
                // LegacyExact retains its frozen current-alpha Matte gate;
                // Advanced stable-route Matte is covered by the routed pass.
                "1ec23f37246c24d4ec875fd975df0804f676028fd9e22674f5b554702ab914c8",
                "16bf9582362dcb852f5e3e13cc715b9b0db0660d85b1a48ebf1711ad6a0b8613",
            ]
            .map(str::to_string),
            "{gate_hashes:?}"
        );

        let hold_params = TemporalParams {
            originals: TemporalOriginalsParams {
                garden: RefreshGardenParams {
                    amount: 1.0,
                    gate: RefreshGardenGate::Luma,
                    threshold: 1.0,
                    softness: 0.0,
                    decay: 1.0,
                    max_hold_ticks: 3,
                    ..RefreshGardenParams::default()
                },
                ..TemporalOriginalsParams::default()
            },
            ..TemporalParams::default()
        };
        let mut held_state = TemporalState::default();
        let held_prime = fixture.render_with_input(
            &hold_params,
            &mut held_state,
            prime,
            TemporalFrameInput::legacy(0.0, true),
        );
        let before_force = fixture.render_with_input(
            &hold_params,
            &mut held_state,
            current,
            TemporalFrameInput::legacy(1.0 / 30.0, true),
        );
        assert_eq!(before_force, held_prime, "closed gate holds the carrier");
        let forced = fixture.render_with_input(
            &hold_params,
            &mut held_state,
            current,
            TemporalFrameInput::legacy(1.0 / 30.0, true),
        );
        assert_ne!(
            forced, held_prime,
            "max-hold forces deterministic admission"
        );

        let frozen = fixture.render_with_input(
            &hold_params,
            &mut held_state,
            [0, 255, 0, 255],
            TemporalFrameInput::new(
                10.0,
                TemporalFreezeState::ProgramFrozen,
                false,
                TemporalFrameEvents::default(),
            ),
        );
        assert_eq!(
            frozen, forced,
            "Program Freeze holds the exact audience image"
        );

        let blackout_sequence = |blackout| {
            let mut state = TemporalState::default();
            fixture.render_with_input(
                &garden_params(RefreshGardenGate::TemporalDelta),
                &mut state,
                prime,
                TemporalFrameInput::legacy(0.0, true),
            );
            fixture.render_with_input(
                &garden_params(RefreshGardenGate::TemporalDelta),
                &mut state,
                current,
                TemporalFrameInput::new(
                    1.0 / 30.0,
                    TemporalFreezeState::Running,
                    blackout,
                    TemporalFrameEvents::default(),
                ),
            )
        };
        assert_eq!(blackout_sequence(false), blackout_sequence(true));

        held_state.reset_for(TemporalResetCause::ManualClear);
        let after_clear = fixture.render_with_input(
            &hold_params,
            &mut held_state,
            [3, 9, 27, 255],
            TemporalFrameInput::legacy(1.0 / 30.0, true),
        );
        assert!(after_clear
            .chunks_exact(4)
            .all(|pixel| pixel == [3, 9, 27, 255]));

        let mut rate_hashes = Vec::new();
        for fps in [24_u32, 30, 60] {
            let params = garden_params(RefreshGardenGate::Luma);
            let mut state = TemporalState::default();
            fixture.render_with_input(
                &params,
                &mut state,
                prime,
                TemporalFrameInput::legacy(0.0, true),
            );
            let mut pixels = Vec::new();
            for _ in 0..fps {
                pixels = fixture.render_with_input(
                    &params,
                    &mut state,
                    current,
                    TemporalFrameInput::legacy(1.0 / fps as f32, true),
                );
            }
            rate_hashes.push(pixel_hash(&pixels));
        }
        assert_eq!(
            rate_hashes,
            [
                "250b46816f5648c5eb790bde9c732f6ae5b77ba6e329c4c675a506da89b95cd4",
                "cda11931771d68593d3df94536e297e72cc9426bf46f65fe9b7f72dcd66ea277",
                "cda11931771d68593d3df94536e297e72cc9426bf46f65fe9b7f72dcd66ea277",
            ]
            .map(str::to_string),
            "{rate_hashes:?}"
        );
    }

    #[test]
    fn selective_ntsc_snapshot_counts_both_scratch_textures_and_staging() {
        assert_eq!(
            selective_ntsc_allocation_snapshot_for(1_920, 1_080, 8_294_400),
            SelectiveNtscAllocationSnapshot {
                scratch_bytes: 16_588_800,
                staging_bytes: 8_294_400,
            }
        );
        assert_eq!(
            selective_ntsc_allocation_snapshot_for(u32::MAX, u32::MAX, u64::MAX).total_bytes(),
            u64::MAX
        );
    }
}

// Readback slot lifecycle for the async NTSC pipeline.
const SLOT_IDLE: u8 = 0;
const SLOT_MAP_PENDING: u8 = 1;
const SLOT_MAPPED: u8 = 2;
const SLOT_MAP_FAILED: u8 = 3;
const SLOT_MAP_REQUESTED: u8 = 4;
const MAX_READBACK_SLOTS: usize = 3;

/// A staging buffer that can have a GPU→CPU copy in flight without
/// blocking the render thread.
struct ReadbackSlot {
    buffer: wgpu::Buffer,
    status: Arc<AtomicU8>,
    /// Monotonic order in which the GPU copy was submitted.
    sequence: u64,
    /// Application-owned visual generation associated with this copy.
    /// Consumers use it to reject frames captured before a blackout edge.
    epoch: u64,
    /// Parameters and deterministic phase sampled with this raw frame.
    /// `None` is used when the readback exists only for Spout output.
    ntsc_metadata: Option<NtscFrameMetadata>,
    /// Selective audience sample copied only after that exact sample's
    /// Temporal and opaque passes were encoded. `None` is the legacy/global
    /// readback path.
    selective_sample: Option<SelectiveNtscGeneration>,
    /// True when this copy rebases Spout to the exact audience image retained
    /// across a paused selective-path transition. It is deliberately distinct
    /// from both raw/global NTSC input and a newly processed selective sample.
    held_audience: bool,
}

/// A completed asynchronous composite readback.
///
/// `epoch` is opaque to the renderer. The live application advances it at
/// blackout transitions so delayed GPU/CPU work can never reveal an older
/// visual generation.
pub struct ReadbackFrame {
    pub pixels: Vec<u8>,
    pub epoch: u64,
    pub ntsc_metadata: Option<NtscFrameMetadata>,
    pub selective_sample: Option<SelectiveNtscGeneration>,
    pub held_audience: bool,
}

/// Nonblocking readback poll result. A held-audience copy that could not be
/// harvested is reported explicitly so the application can schedule another
/// exact copy instead of mistaking a failed GPU map for successful delivery.
pub struct ReadbackPoll {
    pub frame: Option<ReadbackFrame>,
    pub held_audience_not_harvested: bool,
}

#[derive(Debug, Clone, Copy)]
struct ArmedLegacyScopeReadback {
    reservation: RecorderReadbackReservation,
    captured: bool,
}

/// The Exact renderer shares one fixed staging pool between final Program and
/// one stable post-local layer target. Groups are an Advanced-only boundary.
struct PreparedRendererRecorderReadback {
    target: crate::program_recorder::CaptureTarget,
    staging: PreparedRgbaReadback,
    armed_scope: Option<ArmedLegacyScopeReadback>,
}

/// One lazy, bounded GPU batch for selective per-layer VHS. A single slot is
/// intentional: a second full stack at 4K can consume hundreds of MiB and
/// would only increase CPU post-process latency. New snapshots are skipped
/// until this slot is harvested, making the path latest-only rather than a
/// backlog.
struct SelectiveNtscReadbackSlot {
    buffer: wgpu::Buffer,
    capacity: u64,
    status: Arc<AtomicU8>,
    used_size: u64,
    padded_row_bytes: u32,
    slice_stride: u64,
    plan: Option<SelectiveNtscPlan>,
}

struct SelectiveNtscGpuState {
    scratch_textures: [wgpu::Texture; 2],
    scratch_views: [wgpu::TextureView; 2],
    slot: SelectiveNtscReadbackSlot,
}

/// One frame's selective bindings. Keeping the uniform buffers beside their
/// bind groups makes their lifetime explicit until command encoding finishes.
struct SelectiveNtscLayerGpuBindings {
    _uniform_buffer: wgpu::Buffer,
    texture_bind_group: wgpu::BindGroup,
    uniform_bind_group: wgpu::BindGroup,
}

struct SelectiveNtscBatchGpuBindings {
    _master_uniform_buffer: wgpu::Buffer,
    master_texture_bind_group: wgpu::BindGroup,
    master_uniform_bind_group: wgpu::BindGroup,
    layers: Vec<SelectiveNtscLayerGpuBindings>,
}

/// Build the frozen Compat8 effects pipeline and layouts used by live Exact.
/// Keeping construction in one function lets physical-GPU acceptance fixtures
/// execute the identical production shader, target format, and binding law.
pub(crate) fn build_effects_pipeline(
    device: &wgpu::Device,
) -> (
    wgpu::RenderPipeline,
    wgpu::BindGroupLayout,
    wgpu::BindGroupLayout,
    wgpu::ShaderModule,
) {
    let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Effects Texture BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Effects Uniform BGL"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let vertex = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Vertex Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fullscreen.wgsl").into()),
    });
    let fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Effects Fragment"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/effects.wgsl").into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Effects Pipeline Layout"),
        bind_group_layouts: &[Some(&texture_layout), Some(&uniform_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Effects Pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &vertex,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &fragment,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    (pipeline, texture_layout, uniform_layout, vertex)
}

/// Build the temporal pipeline and its layouts. Shared by the live
/// renderer and the offline exporter so both apply identical passes.
pub(crate) fn build_temporal_pipeline(
    device: &wgpu::Device,
) -> (
    wgpu::RenderPipeline,
    wgpu::BindGroupLayout,
    wgpu::BindGroupLayout,
) {
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Temporal BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // Last frame's post-temporal output — feedback compounds on this.
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });

    let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Temporal Uniform BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // B3 feedback rig, a third fixed block beside the frozen 64-byte
            // legacy uniform. Identity keeps the historical expression.
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let vertex = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Temporal Vertex"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fullscreen.wgsl").into()),
    });
    let fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Temporal Fragment"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/temporal.wgsl").into()),
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Temporal Pipeline Layout"),
        bind_group_layouts: &[Some(&bind_group_layout), Some(&uniform_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Temporal Pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &vertex,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &fragment,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    (pipeline, bind_group_layout, uniform_layout)
}

/// Allocation-free prepared bindings for the live Exact temporal path. The
/// legacy and originals pipelines deliberately own distinct group-1 layouts:
/// zero originals therefore execute the frozen shader/pipeline contract,
/// while the additive path binds the separate 128-byte originals block.
pub(crate) struct PreparedTemporalGpuResources {
    originals_pipeline: wgpu::RenderPipeline,
    texture_bind_group: wgpu::BindGroup,
    legacy_uniform_buffer: wgpu::Buffer,
    legacy_uniform_group: wgpu::BindGroup,
    originals_uniform_buffer: wgpu::Buffer,
    originals_uniform_group: wgpu::BindGroup,
    rig_uniform_buffer: wgpu::Buffer,
}

fn build_temporal_originals_pipeline(
    device: &wgpu::Device,
    texture_layout: &wgpu::BindGroupLayout,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Temporal Originals Uniform BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // B3 feedback rig block, mirrored from the legacy layout.
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let vertex = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Temporal Originals Vertex"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fullscreen.wgsl").into()),
    });
    let fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Temporal Originals Fragment"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/temporal_originals.wgsl").into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Temporal Originals Pipeline Layout"),
        bind_group_layouts: &[Some(texture_layout), Some(&uniform_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Temporal Originals Pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &vertex,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &fragment,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    (pipeline, uniform_layout)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_prepared_temporal_gpu_resources(
    device: &wgpu::Device,
    texture_layout: &wgpu::BindGroupLayout,
    legacy_uniform_layout: &wgpu::BindGroupLayout,
    current_view: &wgpu::TextureView,
    history_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    feedback_view: &wgpu::TextureView,
) -> PreparedTemporalGpuResources {
    let (originals_pipeline, originals_uniform_layout) =
        build_temporal_originals_pipeline(device, texture_layout);
    let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Prepared Temporal Textures BG"),
        layout: texture_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(current_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(history_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(feedback_view),
            },
        ],
    });
    let legacy_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Prepared Temporal Legacy Uniforms"),
        size: std::mem::size_of::<TemporalGpuUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let originals_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Prepared Temporal Originals Uniforms"),
        size: std::mem::size_of::<TemporalOriginalsGpuUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let rig_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Prepared Temporal Rig Uniforms"),
        size: std::mem::size_of::<crate::temporal::TemporalRigGpuUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let legacy_uniform_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Prepared Temporal Legacy Uniform BG"),
        layout: legacy_uniform_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: legacy_uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: rig_uniform_buffer.as_entire_binding(),
            },
        ],
    });
    let originals_uniform_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Prepared Temporal Originals Uniform BG"),
        layout: &originals_uniform_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: legacy_uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: originals_uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: rig_uniform_buffer.as_entire_binding(),
            },
        ],
    });
    PreparedTemporalGpuResources {
        originals_pipeline,
        texture_bind_group,
        legacy_uniform_buffer,
        legacy_uniform_group,
        originals_uniform_buffer,
        originals_uniform_group,
        rig_uniform_buffer,
    }
}

fn encode_prepared_temporal_pass(
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    legacy_pipeline: &wgpu::RenderPipeline,
    resources: &PreparedTemporalGpuResources,
    plan: &crate::temporal::TemporalFramePlan,
    target: &wgpu::TextureView,
) {
    queue.write_buffer(
        &resources.legacy_uniform_buffer,
        0,
        bytemuck::bytes_of(&plan.uniforms),
    );
    queue.write_buffer(
        &resources.rig_uniform_buffer,
        0,
        bytemuck::bytes_of(&plan.rig_uniforms),
    );
    if plan.originals_shader_active {
        queue.write_buffer(
            &resources.originals_uniform_buffer,
            0,
            bytemuck::bytes_of(&plan.originals_uniforms),
        );
    }
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(if plan.originals_shader_active {
            "Temporal Originals Pass"
        } else {
            "Temporal Legacy Prepared Pass"
        }),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        ..Default::default()
    });
    pass.set_pipeline(if plan.originals_shader_active {
        &resources.originals_pipeline
    } else {
        legacy_pipeline
    });
    pass.set_bind_group(0, &resources.texture_bind_group, &[]);
    pass.set_bind_group(
        1,
        if plan.originals_shader_active {
            &resources.originals_uniform_group
        } else {
            &resources.legacy_uniform_group
        },
        &[],
    );
    pass.draw(0..3, 0..1);
}

/// Build the one final pass that turns the engine's straight-alpha image into
/// the opaque program image shared by every audience-facing consumer.
pub(crate) fn build_opaque_output_pipeline(
    device: &wgpu::Device,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Opaque Output BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let vertex = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Opaque Output Vertex"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fullscreen.wgsl").into()),
    });
    let fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Opaque Output Fragment"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/opaque_output.wgsl").into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Opaque Output Pipeline Layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Opaque Output Pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &vertex,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &fragment,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: COMPOSITE_FORMAT,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    (pipeline, bind_group_layout)
}

pub(crate) fn build_opaque_output_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    source: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Opaque Output BG"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(source),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

pub(crate) fn encode_opaque_output(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    target: &wgpu::TextureView,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Opaque Output Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        ..Default::default()
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}

/// Create the frame-history texture array and its D2Array view.
pub(crate) fn build_history_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Frame History"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: HISTORY_LEN,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    (texture, view)
}

/// Single texture holding last frame's post-temporal output, for
/// compounding feedback trails.
pub(crate) fn build_feedback_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Feedback Frame"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Encode the temporal pass and record at most one clean current image in the
/// history ring. Shared by the live renderer and the offline exporter.
#[allow(clippy::too_many_arguments)]
#[allow(
    dead_code,
    reason = "frozen legacy/export adapter retained for compatibility goldens"
)]
pub(crate) fn encode_temporal_with_dt(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    params: &TemporalParams,
    pipeline: &wgpu::RenderPipeline,
    bind_group_layout: &wgpu::BindGroupLayout,
    uniform_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    composite_textures: &[wgpu::Texture; 3],
    composite_views: &[wgpu::TextureView; 3],
    history_texture: &wgpu::Texture,
    history_view: &wgpu::TextureView,
    feedback_texture: &wgpu::Texture,
    feedback_view: &wgpu::TextureView,
    state: &mut TemporalState,
    delta_seconds: f32,
    advance_program: bool,
    width: u32,
    height: u32,
) {
    let plan = state.stage_frame(
        params,
        TemporalFrameInput::legacy(delta_seconds, advance_program),
        [width, height],
    );
    match plan.action {
        TemporalFrameAction::PrimeFrozenOutput => {
            // There is no completed temporal output to hold yet. Preserve the
            // clean program already in slot 0 and seed the hold memory once.
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &composite_textures[0],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: feedback_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            return;
        }
        TemporalFrameAction::HoldFrozenOutput => {
            // Render-loop state may continue to redraw while paused; always
            // restore the exact completed program image from the pause edge.
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: feedback_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &composite_textures[0],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            return;
        }
        TemporalFrameAction::Advance { .. } => {}
    }

    if plan.legacy_shader_active {
        let uniform_buffer =
            create_uploaded_uniform(device, queue, "Temporal Uniforms", &plan.uniforms);

        let tex_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Temporal Textures BG"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&composite_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(history_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(feedback_view),
                },
            ],
        });
        let rig_uniform_buffer =
            create_uploaded_uniform(device, queue, "Temporal Rig Uniforms", &plan.rig_uniforms);
        let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Temporal Uniform BG"),
            layout: uniform_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: rig_uniform_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Temporal Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &composite_views[2],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &tex_bg, &[]);
            pass.set_bind_group(1, &uniform_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // Preserve the clean pre-temporal image only after the shader has
        // finished reading the old ring, and before slot 0 is replaced with
        // the temporal result below.
        if let Some(history_write_target) = plan.history_write_target {
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &composite_textures[0],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: history_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: history_write_target as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }

        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &composite_textures[2],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &composite_textures[0],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    } else if let Some(history_write_target) = plan.history_write_target {
        // Even with temporal processing disabled, maintain clean history so
        // enabling a key/slit-scan later starts from real prior observations.
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &composite_textures[0],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: history_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: history_write_target as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    // Record the finished (post-temporal) frame for next frame's trails.
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &composite_textures[0],
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: feedback_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

/// Live Exact encoder using only resources allocated at renderer preparation.
/// Warmed frames perform queue writes, pass encoding, and texture copies only.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_temporal_prepared_frame(
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    params: &TemporalParams,
    legacy_pipeline: &wgpu::RenderPipeline,
    resources: &PreparedTemporalGpuResources,
    composite_textures: &[wgpu::Texture; 3],
    composite_views: &[wgpu::TextureView; 3],
    history_texture: &wgpu::Texture,
    feedback_texture: &wgpu::Texture,
    state: &mut TemporalState,
    input: TemporalFrameInput,
    width: u32,
    height: u32,
) {
    let plan = state.stage_frame(params, input, [width, height]);
    match plan.action {
        TemporalFrameAction::PrimeFrozenOutput => {
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &composite_textures[0],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: feedback_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            return;
        }
        TemporalFrameAction::HoldFrozenOutput => {
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: feedback_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &composite_textures[0],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            return;
        }
        TemporalFrameAction::Advance { .. } => {}
    }

    let shader_active = plan.legacy_shader_active || plan.originals_shader_active;
    if shader_active {
        encode_prepared_temporal_pass(
            queue,
            encoder,
            legacy_pipeline,
            resources,
            &plan,
            &composite_views[2],
        );
        if let Some(history_write_target) = plan.history_write_target {
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &composite_textures[0],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: history_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: history_write_target as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &composite_textures[2],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &composite_textures[0],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    } else if let Some(history_write_target) = plan.history_write_target {
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &composite_textures[0],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: history_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: history_write_target as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &composite_textures[0],
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: feedback_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

/// Compatibility adapter for callers which have not yet materialized the
/// full event/freeze input contract.
#[allow(clippy::too_many_arguments)]
#[allow(
    dead_code,
    reason = "dt/bool compatibility adapter retained beside the full event input"
)]
pub(crate) fn encode_temporal_prepared_with_dt(
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    params: &TemporalParams,
    legacy_pipeline: &wgpu::RenderPipeline,
    resources: &PreparedTemporalGpuResources,
    composite_textures: &[wgpu::Texture; 3],
    composite_views: &[wgpu::TextureView; 3],
    history_texture: &wgpu::Texture,
    feedback_texture: &wgpu::Texture,
    state: &mut TemporalState,
    delta_seconds: f32,
    advance_program: bool,
    width: u32,
    height: u32,
) {
    encode_temporal_prepared_frame(
        queue,
        encoder,
        params,
        legacy_pipeline,
        resources,
        composite_textures,
        composite_views,
        history_texture,
        feedback_texture,
        state,
        TemporalFrameInput::legacy(delta_seconds, advance_program),
        width,
        height,
    );
}

/// Uniforms for the composite shader.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CompositeUniforms {
    opacity: f32,
    blend_mode: u32,
    _pad: [f32; 2],
}

/// How the direct master shader is scheduled for one rendered frame.
///
/// The legacy post-composite path is deliberately retained whenever every
/// visible layer inherits the master. Besides avoiding extra work, that keeps
/// existing patches byte/order compatible with the renderer they were built
/// against. A single visible bypass requires conditional per-layer master
/// passes so the ordinary stack can still occlude and blend in its original
/// order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MasterFxCompositionPath {
    LegacyPostComposite,
    ConditionalPerLayer,
}

pub(crate) fn master_fx_composition_path<I>(layers: I) -> MasterFxCompositionPath
where
    I: IntoIterator<Item = (bool, bool, f32)>,
{
    let has_contributing_bypass =
        layers
            .into_iter()
            .any(|(visible, bypass_master_fx, effective_opacity)| {
                // The compositor clamps finite opacity to [0, 1]. A bypassed
                // layer that contributes no pixels must not perturb the legacy
                // master law for every other layer. Invalid values take the
                // conservative conditional path.
                let opacity_contributes = effective_opacity > 0.0 || !effective_opacity.is_finite();
                visible && bypass_master_fx && opacity_contributes
            });
    if has_contributing_bypass {
        MasterFxCompositionPath::ConditionalPerLayer
    } else {
        MasterFxCompositionPath::LegacyPostComposite
    }
}

/// Return visible source indices in compositor execution order: bottom to
/// top. Source/UI index zero is the top layer, so execution reverses the
/// visible subset without moving bypassed layers into a separate overlay.
pub(crate) fn visible_stack_indices<I>(visibility: I) -> Vec<usize>
where
    I: IntoIterator<Item = bool>,
{
    let mut indices: Vec<usize> = visibility
        .into_iter()
        .enumerate()
        .filter_map(|(index, visible)| visible.then_some(index))
        .collect();
    indices.reverse();
    indices
}

/// Scratch-slot plan for one layer in the conditional path. Local layer FX
/// always write slot 1. An inherited master writes 1 -> 2 and therefore
/// composites 0 + 2 -> 1; a bypass composites 0 + 1 -> 2 directly. This is
/// what preserves order with the renderer's original three full-frame slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ConditionalLayerSlots {
    pub master_output: Option<usize>,
    pub composite_output: usize,
}

pub(crate) const fn conditional_layer_slots(bypass_master_fx: bool) -> ConditionalLayerSlots {
    if bypass_master_fx {
        ConditionalLayerSlots {
            master_output: None,
            composite_output: 2,
        }
    } else {
        ConditionalLayerSlots {
            master_output: Some(2),
            composite_output: 1,
        }
    }
}

/// Terminal health latch for the wgpu device. Wgpu reports device loss only
/// through a callback; without this latch a failed `Surface::configure` can
/// return to the caller and the next `create_buffer_init` fatally maps the
/// invalid placeholder buffer produced for the lost device.
#[derive(Clone, Default)]
struct GpuHealth {
    error: Arc<Mutex<Option<String>>>,
}

impl GpuHealth {
    fn record(&self, error: String) {
        if let Ok(mut stored) = self.error.lock() {
            if stored.is_none() {
                *stored = Some(error);
            }
        }
    }

    fn error(&self) -> Option<String> {
        match self.error.lock() {
            Ok(stored) => stored.clone(),
            Err(_) => Some("GPU health state is unavailable".to_string()),
        }
    }
}

/// Configure a surface under explicit scopes because wgpu's public configure
/// API returns `()`. Invalid/outdated surface errors remain local and
/// recoverable; actual device loss is reported independently by `GpuHealth`.
fn configure_surface_checked(
    device: &wgpu::Device,
    surface: &wgpu::Surface<'_>,
    config: &wgpu::SurfaceConfiguration,
    health: &GpuHealth,
) -> Result<(), String> {
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    surface.configure(device, config);

    let errors = [
        pollster::block_on(out_of_memory.pop()),
        pollster::block_on(internal.pop()),
        pollster::block_on(validation.pop()),
    ];
    if let Some(error) = errors.into_iter().flatten().next() {
        return Err(format!("surface configuration failed: {error}"));
    }
    health.error().map_or(Ok(()), Err)
}

pub struct Renderer {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    main_surface_suspended: bool,
    main_needs_reconfigure: bool,

    // Device loss is terminal for this renderer. Surface validation failures
    // are scoped at the configure call and remain recoverable; other
    // uncaptured GPU errors are logged without being mislabeled as device loss.
    // The app checks this latch before issuing further GPU calls and exits
    // cleanly instead of allowing wgpu's invalid-resource fallback to panic.
    gpu_health: GpuHealth,

    // Effects pipeline (per-layer: applies pixelate/rgb_split/color to a single layer)
    effects_pipeline: wgpu::RenderPipeline,
    effects_bind_group_layout: wgpu::BindGroupLayout,
    effects_uniform_layout: wgpu::BindGroupLayout,

    // Composite pipeline (blends overlay onto base)
    composite_pipeline: wgpu::RenderPipeline,
    composite_bind_group_layout: wgpu::BindGroupLayout,
    composite_uniform_layout: wgpu::BindGroupLayout,
    matte_composite: MatteCompositePipeline,
    image_routing_gpu: Mutex<Option<ImageRoutingGpuResources>>,

    // Final straight-alpha -> opaque-black conversion. It writes into
    // composite slot 2 only after all engine effects have finished.
    opaque_output_pipeline: wgpu::RenderPipeline,
    opaque_output_bind_group: wgpu::BindGroup,

    // Shared linear sampler plus the authored nearest-neighbour alternative.
    sampler: wgpu::Sampler,
    nearest_sampler: wgpu::Sampler,

    // Three textures for compositing:
    // [0] = accumulated result (base)
    // [1] = current layer after effects (overlay)
    // [2] = scratch during engine passes, then final opaque audience output
    pub composite_textures: [wgpu::Texture; 3],
    pub composite_views: [wgpu::TextureView; 3],

    // Baseline exact opaque audience snapshot retained by the global
    // Pause/blackout contract and reused across selective/VHS path changes.
    // This texture is never sampled as an effect input; it is only copied
    // to/from final audience slot 2.
    held_audience_texture: wgpu::Texture,

    // Raw (non-sRGB-decoding) view of slot 2 for egui-wgpu, whose shader
    // performs the gamma-to-linear conversion itself.
    pub output_view: wgpu::TextureView,

    pub output_width: u32,
    pub output_height: u32,

    // Staging buffers for async NTSC readback (created lazily, reused)
    readback_slots: Vec<ReadbackSlot>,
    next_readback_sequence: u64,
    last_harvested_readback_sequence: u64,
    selective_ntsc_gpu: Option<SelectiveNtscGpuState>,

    // Recorder/capture readback is a separate fixed two-slot pipeline. It is
    // cold-prepared explicitly, never allocated by a warmed frame, and reads
    // only the exact final audience slot after global output treatment.
    #[allow(
        dead_code,
        reason = "native Main consumes recorder capture; alternate targets retain the prepared pool without a caller"
    )]
    program_recorder_readback: RefCell<Option<PreparedRendererRecorderReadback>>,

    // Temporal effects: ring buffer of past output frames (texture array)
    temporal_pipeline: wgpu::RenderPipeline,
    temporal_prepared: PreparedTemporalGpuResources,
    history_texture: wgpu::Texture,
    feedback_texture: wgpu::Texture,
    /// Fixed-rate history clock and validity for temporal GPU memories.
    temporal_state: TemporalState,
    /// The B4 display-physics stage (fields, phosphor, display model) on the
    /// slot-0 seam between temporal and the opaque resolve. Its surfaces are
    /// lazy: a default session charges nothing.
    display_physics: crate::renderer::display_physics::DisplayPhysicsGpu,

    // Instance + adapter kept for creating additional surfaces (output
    // window). Capabilities must be queried against the SAME adapter the
    // device came from — a freshly requested adapter handle invalidates
    // the device when its capabilities disagree.
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    // Dedicated fullscreen output window (projector), when open
    output: Option<OutputTarget>,
}

/// A second window showing the final composite, letterboxed — the face the
/// audience sees while the main window and web panel stay with the performer.
struct OutputTarget {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    /// Resize/fullscreen events can arrive in bursts. Coalesce them and
    /// configure once at the render boundary, after the prior surface texture
    /// has been presented or dropped.
    needs_reconfigure: bool,
    suspended: bool,
}

fn preferred_surface_format(formats: &[wgpu::TextureFormat]) -> Option<wgpu::TextureFormat> {
    formats
        .iter()
        .find(|format| format.is_srgb())
        .copied()
        .or_else(|| formats.first().copied())
}

fn preferred_present_mode(modes: &[wgpu::PresentMode]) -> Option<wgpu::PresentMode> {
    if modes.contains(&wgpu::PresentMode::Fifo) {
        Some(wgpu::PresentMode::Fifo)
    } else {
        modes.first().copied()
    }
}

impl Renderer {
    pub fn new(window: Arc<Window>, output_width: u32, output_height: u32) -> Result<Self, String> {
        // Reject corrupt/hostile initial dimensions before constructing any
        // full-frame GPU resource. Adapter limits remain an additional check,
        // never the sole memory-safety boundary.
        crate::video::decoder::validate_media_dimensions(output_width, output_height, None)
            .map_err(|error| format!("invalid renderer output dimensions: {error}"))?;
        let full_frame_texture_floor = renderer_owned_full_frame_texture_floor_bytes(
            output_width,
            output_height,
        )
        .ok_or_else(|| {
            format!("renderer output texture footprint overflows at {output_width}x{output_height}")
        })?;
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window)
            .map_err(|error| format!("failed to create renderer surface: {error}"))?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .map_err(|error| format!("no suitable GPU adapter found: {error}"))?;

        crate::video::decoder::validate_media_dimensions(
            output_width,
            output_height,
            Some(adapter.limits().max_texture_dimension_2d),
        )
        .map_err(|error| format!("unsupported renderer output dimensions: {error}"))?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .map_err(|error| format!("failed to create renderer device: {error}"))?;

        let gpu_health = GpuHealth::default();
        let device_loss_health = gpu_health.clone();
        device.set_device_lost_callback(move |reason, message| {
            let detail = if message.trim().is_empty() {
                format!("GPU device lost ({reason:?})")
            } else {
                format!("GPU device lost ({reason:?}): {message}")
            };
            device_loss_health.record(detail);
        });
        device.on_uncaptured_error(Arc::new(move |error| {
            // Avoid wgpu's default panic while preserving diagnostics. Device
            // loss has a separate callback/latch and is the only condition
            // that makes every resource in this Renderer invalid.
            log::error!("Uncaptured GPU error: {error}");
        }));

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        configure_surface_checked(&device, &surface, &config, &gpu_health)?;

        // Resource constructors return handles rather than Results. Keep the
        // complete startup allocation/pipeline phase under explicit scopes so
        // validation, backend, and allocation failures can activate the
        // caller's lower-resolution recovery path instead of surfacing later
        // as an uncaptured error.
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // --- Effects pipeline (single texture + uniforms → render target) ---
        let (effects_pipeline, effects_bind_group_layout, effects_uniform_layout, vertex_shader) =
            build_effects_pipeline(&device);

        // --- Composite pipeline (two textures + uniforms → render target) ---
        let composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Composite BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let composite_uniform_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Composite Uniform BGL"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let composite_fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Composite Fragment"),
            source: wgpu::ShaderSource::Wgsl(composite_shader_source()),
        });

        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Composite Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&composite_bind_group_layout),
                    Some(&composite_uniform_layout),
                ],
                immediate_size: 0,
            });

        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Composite Pipeline"),
            layout: Some(&composite_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_fragment,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let matte_composite = MatteCompositePipeline::build(&device, &vertex_shader);

        // --- Temporal pipeline + frame-history ring (shared with export) ---
        let (temporal_pipeline, temporal_bind_group_layout, temporal_uniform_layout) =
            build_temporal_pipeline(&device);
        let (history_texture, history_view) =
            build_history_texture(&device, output_width, output_height);
        let (feedback_texture, feedback_view) =
            build_feedback_texture(&device, output_width, output_height);
        let display_physics = crate::renderer::display_physics::DisplayPhysicsGpu::new(
            &device,
            COMPOSITE_FORMAT,
            [output_width, output_height],
        );

        // --- Three composite textures ---
        let tex_usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST;

        let composite_textures: [wgpu::Texture; 3] = std::array::from_fn(|i| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Composite {i}")),
                size: wgpu::Extent3d {
                    width: output_width,
                    height: output_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: COMPOSITE_FORMAT,
                usage: tex_usage,
                // Slot 2 is the opaque audience image. egui-wgpu expects a
                // raw/gamma texture and performs its own decode, so permit a
                // non-sRGB twin view of exactly this texture.
                view_formats: if i == 2 {
                    EGUI_OUTPUT_VIEW_FORMATS
                } else {
                    &[]
                },
            })
        });

        let composite_views: [wgpu::TextureView; 3] = std::array::from_fn(|i| {
            composite_textures[i].create_view(&wgpu::TextureViewDescriptor::default())
        });

        let output_view = composite_textures[2].create_view(&wgpu::TextureViewDescriptor {
            label: Some("egui opaque output (raw gamma view)"),
            format: Some(EGUI_OUTPUT_VIEW_FORMAT),
            ..Default::default()
        });
        let held_audience_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Held opaque audience"),
            size: wgpu::Extent3d {
                width: output_width,
                height: output_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COMPOSITE_FORMAT,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let (opaque_output_pipeline, opaque_output_bind_group_layout) =
            build_opaque_output_pipeline(&device);
        let opaque_output_bind_group = build_opaque_output_bind_group(
            &device,
            &opaque_output_bind_group_layout,
            &composite_views[0],
            &sampler,
        );
        let temporal_prepared = build_prepared_temporal_gpu_resources(
            &device,
            &temporal_bind_group_layout,
            &temporal_uniform_layout,
            &composite_views[0],
            &history_view,
            &sampler,
            &feedback_view,
        );

        let allocation_errors = [
            pollster::block_on(out_of_memory.pop()),
            pollster::block_on(internal.pop()),
            pollster::block_on(validation.pop()),
        ];
        if let Some(error) = allocation_errors.into_iter().flatten().next() {
            return Err(format!(
                "renderer resource allocation failed at {output_width}x{output_height} \
                 ({full_frame_texture_floor} bytes of mandatory owned full-frame RGBA texture \
                 payload): {error}"
            ));
        }
        if let Some(error) = gpu_health.error() {
            return Err(format!(
                "renderer resource allocation failed at {output_width}x{output_height}: {error}"
            ));
        }

        Ok(Self {
            surface,
            device,
            queue,
            config,
            main_surface_suspended: false,
            main_needs_reconfigure: false,
            gpu_health,
            effects_pipeline,
            effects_bind_group_layout,
            effects_uniform_layout,
            composite_pipeline,
            composite_bind_group_layout,
            composite_uniform_layout,
            matte_composite,
            image_routing_gpu: Mutex::new(None),
            opaque_output_pipeline,
            opaque_output_bind_group,
            sampler,
            nearest_sampler,
            composite_textures,
            composite_views,
            held_audience_texture,
            output_view,
            output_width,
            output_height,
            readback_slots: Vec::new(),
            next_readback_sequence: 1,
            last_harvested_readback_sequence: 0,
            selective_ntsc_gpu: None,
            program_recorder_readback: RefCell::new(None),
            temporal_pipeline,
            temporal_prepared,
            history_texture,
            feedback_texture,
            temporal_state: TemporalState::default(),
            display_physics,
            instance,
            adapter,
            output: None,
        })
    }

    /// Open the fullscreen output window's rendering surface.
    pub fn create_output(&mut self, window: Arc<Window>) -> Result<(), String> {
        self.ensure_device_healthy()?;
        let size = window.inner_size();
        let surface = self
            .instance
            .create_surface(window.clone())
            .map_err(|error| format!("failed to create output-window surface: {error}"))?;

        let caps = surface.get_capabilities(&self.adapter);
        if !caps.usages.contains(wgpu::TextureUsages::RENDER_ATTACHMENT) {
            return Err(
                "output-window surface does not support render attachments on the active GPU"
                    .to_string(),
            );
        }
        let format = preferred_surface_format(&caps.formats).ok_or_else(|| {
            "output-window surface is incompatible with the active GPU (no texture formats)"
                .to_string()
        })?;
        let alpha_mode = caps.alpha_modes.first().copied().ok_or_else(|| {
            "output-window surface is incompatible with the active GPU (no alpha modes)".to_string()
        })?;
        let present_mode = preferred_present_mode(&caps.present_modes).ok_or_else(|| {
            "output-window surface is incompatible with the active GPU (no present modes)"
                .to_string()
        })?;

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        // A just-created Windows fullscreen window may transiently report a
        // zero client extent. Do not configure a synthetic 1x1 swapchain:
        // keep the target suspended and let its first nonzero resize schedule
        // one real configuration at the render boundary.
        let suspended = size.width == 0 || size.height == 0;
        if !suspended {
            configure_surface_checked(&self.device, &surface, &config, &self.gpu_health)?;
        }

        // Blit pipeline targeting this surface's format.
        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Blit BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let vertex = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Blit Vertex"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fullscreen.wgsl").into()),
            });
        let fragment = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Blit Fragment"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/blit.wgsl").into()),
            });

        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Blit PL"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });

        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Blit Pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &vertex,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fragment,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        // An sRGB surface needs the sRGB view (hardware decode + encode). A
        // rare non-sRGB fallback surface needs the raw gamma view instead.
        let source_view = if format.is_srgb() {
            &self.composite_views[2]
        } else {
            &self.output_view
        };
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blit BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        self.output = Some(OutputTarget {
            window,
            surface,
            config,
            pipeline,
            bind_group,
            needs_reconfigure: suspended,
            suspended,
        });
        Ok(())
    }

    /// Return the first terminal GPU/device error, if one has occurred.
    pub fn device_error(&self) -> Option<String> {
        self.gpu_health.error()
    }

    /// Fail closed before issuing more GPU work after device loss. Recovering
    /// in place would require rebuilding the renderer, egui renderer, and all
    /// layer textures; using any old resource is invalid by definition.
    pub fn ensure_device_healthy(&self) -> Result<(), String> {
        self.device_error().map_or(Ok(()), Err)
    }

    /// Stable operator-facing adapter facts. Memory budgets are intentionally
    /// reported separately because wgpu does not expose a portable residency
    /// budget on every backend.
    pub fn gpu_identity(&self) -> (String, String, String) {
        let info = self.adapter.get_info();
        (
            info.name,
            format!("{:?}", info.backend),
            if info.driver_info.is_empty() {
                info.driver
            } else {
                format!("{} ({})", info.driver, info.driver_info)
            },
        )
    }

    /// Create an additional presentation surface that is guaranteed to belong
    /// to the same instance/adapter pair as this renderer's device. Venue
    /// outputs keep ownership of the returned surface and window; this method
    /// does not register it as the legacy single audience output.
    pub(crate) fn create_compatible_surface(
        &self,
        window: Arc<Window>,
    ) -> Result<wgpu::Surface<'static>, String> {
        self.instance
            .create_surface(window)
            .map_err(|error| format!("failed to create compatible output surface: {error}"))
    }

    /// Query capabilities against the renderer's actual adapter. Asking a new
    /// adapter for these capabilities can produce a format that this device
    /// cannot legally use.
    pub(crate) fn compatible_surface_capabilities(
        &self,
        surface: &wgpu::Surface<'_>,
    ) -> wgpu::SurfaceCapabilities {
        surface.get_capabilities(&self.adapter)
    }

    /// Configure a compatible venue surface under the renderer's scoped GPU
    /// error handling, without mutating the legacy audience output target.
    pub(crate) fn configure_compatible_surface(
        &self,
        surface: &wgpu::Surface<'_>,
        config: &wgpu::SurfaceConfiguration,
    ) -> Result<(), String> {
        configure_surface_checked(&self.device, surface, config, &self.gpu_health)
    }

    /// Start a new patch/visual generation without sampling any temporal or
    /// asynchronous readback data produced by the prior generation.
    ///
    /// The application must also advance its external visual epoch and clear
    /// any CPU-side NTSC-presented frame. Pending map callbacks may complete,
    /// but their sequence numbers are at or below this invalidation watermark
    /// and `poll_readback` will recycle them without returning their pixels.
    pub fn reset_visual_generation(&mut self) {
        self.reset_visual_generation_for(TemporalResetCause::PatchGeneration);
    }

    pub(crate) fn reset_visual_generation_for(&mut self, cause: TemporalResetCause) {
        self.temporal_state.reset_for(cause);
        self.last_harvested_readback_sequence = self
            .last_harvested_readback_sequence
            .max(self.next_readback_sequence.saturating_sub(1));
        // A pending map cannot be cancelled portably. Revoking its plan makes
        // the callback harmless; poll will unmap/recycle it without exposing
        // pixels to the CPU worker.
        if let Some(state) = self.selective_ntsc_gpu.as_mut() {
            state.slot.plan = None;
        }
    }

    /// Snapshot the exact opaque audience image without involving temporal,
    /// effects, CPU processing, or readback. Used when Pause must survive a
    /// selective/VHS path transition (including a temporary blackout).
    pub fn capture_held_audience(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.composite_textures[2],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.held_audience_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.output_width,
                height: self.output_height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Restore a previously captured audience snapshot to final slot 2.
    pub fn restore_held_audience(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.held_audience_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.composite_textures[2],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.output_width,
                height: self.output_height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Blackout: clear the final composite to black. Everything downstream
    /// — panel preview, output window, Spout, NTSC — goes dark together.
    /// The display-physics memories go dark with it: a blacked-out audience
    /// must not retain a glowing phosphor wake or a held field.
    pub fn clear_composite(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.display_physics.clear_for_blackout(encoder);
        self.clear_composite_slots(encoder);
    }

    fn clear_composite_slots(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Blackout"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.composite_views[0],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(transparent_accumulation_clear()),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.composite_views[2],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
            ],
            depth_stencil_attachment: None,
            ..Default::default()
        });
    }

    pub fn close_output(&mut self) {
        self.output = None;
    }

    pub fn has_output(&self) -> bool {
        self.output.is_some()
    }

    pub fn output_window_id(&self) -> Option<winit::window::WindowId> {
        self.output.as_ref().map(|o| o.window.id())
    }

    pub fn output_window_size(&self) -> Option<winit::dpi::PhysicalSize<u32>> {
        self.output
            .as_ref()
            .map(|output| output.window.inner_size())
    }

    pub fn resize_output(&mut self, width: u32, height: u32) {
        if let Some(out) = self.output.as_mut() {
            if width == 0 || height == 0 {
                out.suspended = true;
                return;
            }
            let resumed = std::mem::replace(&mut out.suspended, false);
            if width > 0 && height > 0 && (out.config.width != width || out.config.height != height)
            {
                out.config.width = width;
                out.config.height = height;
                out.needs_reconfigure = true;
            } else if resumed {
                out.needs_reconfigure = true;
            }
        }
    }

    /// Encode the letterboxed blit onto the output window's surface.
    /// Returns the surface texture to present after the encoder is
    /// submitted; None if the surface wasn't ready this frame.
    pub fn render_output(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<Option<wgpu::SurfaceTexture>, String> {
        self.ensure_device_healthy()?;
        let Some(out) = self.output.as_mut() else {
            return Ok(None);
        };
        if out.suspended {
            return Ok(None);
        }

        if out.needs_reconfigure {
            configure_surface_checked(&self.device, &out.surface, &out.config, &self.gpu_health)?;
            out.needs_reconfigure = false;
        }

        let surface_texture = match out.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => {
                out.needs_reconfigure = true;
                texture
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                configure_surface_checked(
                    &self.device,
                    &out.surface,
                    &out.config,
                    &self.gpu_health,
                )?;
                return Ok(None);
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                return Err("output surface was lost; close and reopen Output".to_string());
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(None);
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err("output surface validation failed".to_string());
            }
        };
        let view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Letterbox: fit the composite's aspect into the window.
        let (sw, sh) = (out.config.width as f32, out.config.height as f32);
        let aspect = self.output_width as f32 / self.output_height as f32;
        let (vw, vh) = if sw / sh > aspect {
            (sh * aspect, sh)
        } else {
            (sw, sw / aspect)
        };
        let (vx, vy) = ((sw - vw) * 0.5, (sh - vh) * 0.5);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Output Blit"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(&out.pipeline);
        pass.set_bind_group(0, &out.bind_group, &[]);
        pass.set_viewport(vx, vy, vw.max(1.0), vh.max(1.0), 0.0, 1.0);
        pass.draw(0..3, 0..1);
        drop(pass);

        Ok(Some(surface_texture))
    }

    /// Temporal pass driven by a real elapsed time. Live rendering should pass
    /// its measured frame delta; deterministic export should pass `1.0 / fps`.
    #[allow(
        dead_code,
        reason = "public dt/bool compatibility adapter retained for embedders"
    )]
    pub fn render_temporal_with_dt(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        params: &TemporalParams,
        delta_seconds: f32,
        advance_program: bool,
    ) {
        encode_temporal_prepared_with_dt(
            &self.queue,
            encoder,
            params,
            &self.temporal_pipeline,
            &self.temporal_prepared,
            &self.composite_textures,
            &self.composite_views,
            &self.history_texture,
            &self.feedback_texture,
            &mut self.temporal_state,
            delta_seconds,
            advance_program,
            self.output_width,
            self.output_height,
        );
    }

    /// Full T3 temporal adapter. Events, independent Program/Media freeze, and
    /// blackout policy enter the shared transactional state unchanged.
    #[allow(
        dead_code,
        reason = "native Main uses the full event adapter; compatibility targets retain the same entry point"
    )]
    pub(crate) fn render_temporal_frame(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        params: &TemporalParams,
        input: TemporalFrameInput,
    ) {
        encode_temporal_prepared_frame(
            &self.queue,
            encoder,
            params,
            &self.temporal_pipeline,
            &self.temporal_prepared,
            &self.composite_textures,
            &self.composite_views,
            &self.history_texture,
            &self.feedback_texture,
            &mut self.temporal_state,
            input,
            self.output_width,
            self.output_height,
        );
    }

    /// Publish the CPU temporal state staged by the most recently encoded
    /// Exact frame. Call only after its command buffer has been accepted.
    pub fn commit_temporal_frame(&mut self) {
        self.temporal_state.commit_staged();
    }

    /// Roll back an Exact temporal frame whose encoder was abandoned or whose
    /// outer frame was rejected before acceptance.
    pub fn discard_temporal_frame(&mut self) {
        self.temporal_state.discard_staged();
    }

    /// Revoke clean-history/carrier/Score validity without touching authored
    /// parameters or the frozen audience pixels. Validity gates make the old
    /// GPU contents unreachable, so no full-frame clear pass is required.
    pub fn clear_temporal_memory(&mut self) {
        self.temporal_state
            .reset_for(TemporalResetCause::ManualClear);
    }

    #[allow(
        dead_code,
        reason = "native telemetry selects this adapter only while LegacyExact is active"
    )]
    pub(crate) fn temporal_state_metrics(&self) -> TemporalStateMetrics {
        self.temporal_state.metrics()
    }

    /// `None` means the lazy selective-NTSC resources have never been
    /// allocated. `Some(0)` is therefore never used to disguise an unknown
    /// state as a measured zero.
    pub(crate) fn selective_ntsc_allocation_bytes(&self) -> Option<u64> {
        self.selective_ntsc_allocation_snapshot()
            .map(SelectiveNtscAllocationSnapshot::total_bytes)
    }

    pub(crate) fn selective_ntsc_allocation_snapshot(
        &self,
    ) -> Option<SelectiveNtscAllocationSnapshot> {
        self.selective_ntsc_gpu.as_ref().map(|state| {
            selective_ntsc_allocation_snapshot_for(
                self.output_width,
                self.output_height,
                state.slot.capacity,
            )
        })
    }

    /// The B4 display-physics stage — fields, phosphor, display model — on
    /// the slot-0 seam between the temporal pass and the opaque resolve, the
    /// one adjacency every audience path shares. A dormant stage (all three
    /// sub-blocks off) encodes nothing and slot 0 reaches the resolve
    /// untouched. `delta_seconds` is the program-advancing delta (zero while
    /// frozen), so Pause holds the trail and the field clock exactly as it
    /// holds everything.
    pub(crate) fn render_display_physics(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        display: &crate::display_physics::DisplayPhysicsParams,
        delta_seconds: f32,
    ) -> bool {
        self.display_physics.encode(
            &self.device,
            &self.queue,
            encoder,
            &self.composite_textures,
            &self.composite_views,
            display,
            delta_seconds,
        )
    }

    /// Resolve the straight-alpha engine image into one opaque audience image.
    /// This runs exactly once after layer, master, and temporal rendering and
    /// before preview, projector, readback, NTSC, Spout, or export consumes it.
    pub fn render_opaque_output(&self, encoder: &mut wgpu::CommandEncoder) {
        encode_opaque_output(
            encoder,
            &self.opaque_output_pipeline,
            &self.opaque_output_bind_group,
            &self.composite_views[2],
        );
    }

    /// Cold preparation for final-Program capture. This owns exactly two
    /// staging buffers and is intentionally separate from the lazy NTSC/Spout
    /// readback path. Calling it again at unchanged output dimensions is a
    /// no-op, so an armed recorder never causes a warmed allocation.
    #[allow(
        dead_code,
        reason = "native Main consumes recorder capture; alternate targets retain the prepared pool without a caller"
    )]
    pub(crate) fn prepare_program_recorder_readback(
        &mut self,
    ) -> Result<RecorderReadbackAllocationSnapshot, RecorderReadbackError> {
        self.prepare_renderer_recorder_readback(crate::program_recorder::CaptureTarget::Program)
    }

    /// Cold-select one Exact stable layer. The same prepared buffers used by
    /// Program are reused; a group must be captured by the Advanced executor.
    #[allow(
        dead_code,
        reason = "native Main consumes recorder capture; alternate targets retain the prepared pool without a caller"
    )]
    pub(crate) fn prepare_legacy_scope_recorder_readback(
        &mut self,
        target: crate::program_recorder::CaptureTarget,
    ) -> Result<RecorderReadbackAllocationSnapshot, RecorderReadbackError> {
        if !matches!(target, crate::program_recorder::CaptureTarget::Layer(_)) {
            return Err(RecorderReadbackError::UnsupportedTarget(target));
        }
        self.prepare_renderer_recorder_readback(target)
    }

    fn prepare_renderer_recorder_readback(
        &mut self,
        target: crate::program_recorder::CaptureTarget,
    ) -> Result<RecorderReadbackAllocationSnapshot, RecorderReadbackError> {
        self.ensure_device_healthy()
            .map_err(RecorderReadbackError::DeviceUnavailable)?;
        let prepared = self.program_recorder_readback.get_mut();
        if let Some(prepared) = prepared.as_mut() {
            if prepared.armed_scope.is_some() {
                return Err(RecorderReadbackError::InvalidReservation);
            }
            prepared.target = target;
            return Ok(prepared.staging.allocation_snapshot(0));
        }
        let staging = PreparedRgbaReadback::prepare(
            &self.device,
            [self.output_width, self.output_height],
            0,
        )?;
        let snapshot = staging.allocation_snapshot(0);
        *prepared = Some(PreparedRendererRecorderReadback {
            target,
            staging,
            armed_scope: None,
        });
        Ok(snapshot)
    }

    /// Encode a copy of the exact final audience image. The caller must invoke
    /// this only after global NTSC and absolute blackout have materialized in
    /// slot 2, then submit the encoder before calling `map`.
    #[allow(
        dead_code,
        reason = "native Main consumes recorder capture; alternate targets retain the prepared pool without a caller"
    )]
    pub(crate) fn begin_final_program_recorder_readback(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        tag: RecorderReadbackTag,
    ) -> Result<RecorderReadbackAdmission, RecorderReadbackError> {
        self.ensure_device_healthy()
            .map_err(RecorderReadbackError::DeviceUnavailable)?;
        let mut prepared = self.program_recorder_readback.borrow_mut();
        let Some(prepared) = prepared.as_mut() else {
            return Ok(RecorderReadbackAdmission::Unprepared);
        };
        if prepared.target != crate::program_recorder::CaptureTarget::Program {
            return Ok(RecorderReadbackAdmission::SourceUnavailable);
        }
        prepared.staging.schedule_texture(
            encoder,
            &self.composite_textures[2],
            RecorderReadbackRequest::new(crate::program_recorder::CaptureTarget::Program, tag),
        )
    }

    /// Arm the selected Exact layer before `render_evaluated_frame`. A hidden
    /// or deleted layer is diagnosed by `finish`; it is never substituted by
    /// the accumulated Program image.
    #[allow(
        dead_code,
        reason = "native Main consumes recorder capture; alternate targets retain the prepared pool without a caller"
    )]
    pub(crate) fn begin_legacy_scope_recorder_readback(
        &self,
        tag: RecorderReadbackTag,
    ) -> RecorderReadbackAdmission {
        let mut prepared = self.program_recorder_readback.borrow_mut();
        let Some(prepared) = prepared.as_mut() else {
            return RecorderReadbackAdmission::Unprepared;
        };
        if prepared.armed_scope.is_some()
            || !matches!(
                prepared.target,
                crate::program_recorder::CaptureTarget::Layer(_)
            )
        {
            return RecorderReadbackAdmission::Busy;
        }
        let admission = prepared
            .staging
            .reserve(RecorderReadbackRequest::new(prepared.target, tag));
        if let RecorderReadbackAdmission::Scheduled(reservation) = admission {
            prepared.armed_scope = Some(ArmedLegacyScopeReadback {
                reservation,
                captured: false,
            });
        }
        admission
    }

    /// Confirm that the stable layer boundary was encountered by this Exact
    /// frame. SourceUnavailable consumes the unsubmitted reservation only.
    #[allow(
        dead_code,
        reason = "native Main consumes recorder capture; alternate targets retain the prepared pool without a caller"
    )]
    pub(crate) fn finish_legacy_scope_recorder_readback(
        &self,
        reservation: RecorderReadbackReservation,
    ) -> Result<super::readback::RecorderReadbackCaptureStatus, RecorderReadbackError> {
        let mut prepared = self.program_recorder_readback.borrow_mut();
        let prepared = prepared
            .as_mut()
            .ok_or(RecorderReadbackError::InvalidReservation)?;
        let Some(armed) = prepared.armed_scope else {
            return Err(RecorderReadbackError::InvalidReservation);
        };
        if armed.reservation != reservation {
            return Err(RecorderReadbackError::InvalidReservation);
        }
        prepared.armed_scope = None;
        if armed.captured {
            Ok(super::readback::RecorderReadbackCaptureStatus::Captured)
        } else {
            prepared.staging.discard_unsubmitted(reservation)?;
            Ok(super::readback::RecorderReadbackCaptureStatus::SourceUnavailable)
        }
    }

    /// Request asynchronous mapping after submission. This never polls or
    /// waits for completion.
    #[allow(
        dead_code,
        reason = "native Main consumes recorder capture; alternate targets retain the prepared pool without a caller"
    )]
    pub(crate) fn map_recorder_readback(
        &self,
        reservation: RecorderReadbackReservation,
    ) -> Result<(), RecorderReadbackError> {
        self.ensure_device_healthy()
            .map_err(RecorderReadbackError::DeviceUnavailable)?;
        self.program_recorder_readback
            .borrow()
            .as_ref()
            .ok_or(RecorderReadbackError::InvalidReservation)?
            .staging
            .map(reservation)
    }

    /// Drive map callbacks without waiting and harvest the oldest capture into
    /// exact-size caller-owned RGBA storage.
    #[allow(
        dead_code,
        reason = "native Main consumes recorder capture; alternate targets retain the prepared pool without a caller"
    )]
    pub(crate) fn poll_recorder_readback_into(
        &self,
        destination: &mut [u8],
    ) -> Result<RecorderReadbackPoll, RecorderReadbackError> {
        self.ensure_device_healthy()
            .map_err(RecorderReadbackError::DeviceUnavailable)?;
        let _ = self.device.poll(wgpu::PollType::Poll);
        self.ensure_device_healthy()
            .map_err(RecorderReadbackError::DeviceUnavailable)?;
        let mut prepared = self.program_recorder_readback.borrow_mut();
        match prepared.as_mut() {
            Some(prepared) => prepared.staging.poll_into(destination),
            None => Ok(RecorderReadbackPoll::Idle),
        }
    }

    /// Drive callbacks without waiting, then inspect the oldest slot without
    /// consuming it. Main uses this before acquiring a bounded CPU frame lease.
    #[allow(
        dead_code,
        reason = "native Main consumes recorder capture; alternate targets retain the prepared pool without a caller"
    )]
    pub(crate) fn recorder_readback_readiness(
        &self,
    ) -> Result<RecorderReadbackReadiness, RecorderReadbackError> {
        self.ensure_device_healthy()
            .map_err(RecorderReadbackError::DeviceUnavailable)?;
        let _ = self.device.poll(wgpu::PollType::Poll);
        self.ensure_device_healthy()
            .map_err(RecorderReadbackError::DeviceUnavailable)?;
        Ok(self
            .program_recorder_readback
            .borrow()
            .as_ref()
            .map_or(RecorderReadbackReadiness::Idle, |prepared| {
                prepared.staging.oldest_readiness()
            }))
    }

    /// Recycle an oldest completion whose exposed tag belongs to a stale
    /// capture generation, without acquiring or touching a CPU frame lease.
    #[allow(
        dead_code,
        reason = "native Main consumes recorder capture; alternate targets retain the prepared pool without a caller"
    )]
    pub(crate) fn recycle_ready_recorder_readback_without_copy(
        &self,
    ) -> Result<RecorderReadbackPoll, RecorderReadbackError> {
        self.ensure_device_healthy()
            .map_err(RecorderReadbackError::DeviceUnavailable)?;
        let _ = self.device.poll(wgpu::PollType::Poll);
        self.ensure_device_healthy()
            .map_err(RecorderReadbackError::DeviceUnavailable)?;
        match self.program_recorder_readback.borrow_mut().as_mut() {
            Some(prepared) => prepared.staging.recycle_oldest_ready_without_copy(),
            None => Ok(RecorderReadbackPoll::Idle),
        }
    }

    /// Recycle a capture only when its command encoder will be abandoned and
    /// never submitted.
    #[allow(
        dead_code,
        reason = "native Main consumes recorder capture; alternate targets retain the prepared pool without a caller"
    )]
    pub(crate) fn discard_unsubmitted_recorder_readback(
        &self,
        reservation: RecorderReadbackReservation,
    ) -> Result<(), RecorderReadbackError> {
        let mut prepared = self.program_recorder_readback.borrow_mut();
        let prepared = prepared
            .as_mut()
            .ok_or(RecorderReadbackError::InvalidReservation)?;
        if prepared
            .armed_scope
            .is_some_and(|armed| armed.reservation == reservation)
        {
            prepared.armed_scope = None;
        }
        prepared.staging.discard_unsubmitted(reservation)
    }

    pub fn resize(&mut self, new_width: u32, new_height: u32) {
        if new_width == 0 || new_height == 0 {
            self.main_surface_suspended = true;
            return;
        }
        let resumed = std::mem::replace(&mut self.main_surface_suspended, false);
        if self.config.width != new_width || self.config.height != new_height {
            self.config.width = new_width;
            self.config.height = new_height;
            self.main_needs_reconfigure = true;
        } else if resumed {
            self.main_needs_reconfigure = true;
        }
    }

    /// Apply the newest coalesced resize once at the render boundary, after
    /// every surface texture from the previous frame has been released.
    pub fn prepare_main_surface(&mut self) -> Result<bool, String> {
        self.ensure_device_healthy()?;
        if self.main_surface_suspended {
            return Ok(false);
        }
        if self.main_needs_reconfigure {
            configure_surface_checked(&self.device, &self.surface, &self.config, &self.gpu_health)?;
            self.main_needs_reconfigure = false;
        }
        Ok(true)
    }

    /// Recreate the main presentation surface after wgpu reports `Lost`.
    /// Reconfiguring the old surface is explicitly insufficient in wgpu 29.
    pub fn recreate_main_surface(&mut self, window: Arc<Window>) -> Result<(), String> {
        self.ensure_device_healthy()?;
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Err("main window surface is suspended at zero size".to_string());
        }
        let surface = self
            .instance
            .create_surface(window)
            .map_err(|error| format!("failed to recreate main surface: {error}"))?;
        let caps = surface.get_capabilities(&self.adapter);
        if !caps.usages.contains(wgpu::TextureUsages::RENDER_ATTACHMENT) {
            return Err("recreated main surface does not support render attachments".to_string());
        }
        if !caps.formats.contains(&self.config.format) {
            return Err(format!(
                "recreated main surface no longer supports {:?}",
                self.config.format
            ));
        }
        if !caps.present_modes.contains(&self.config.present_mode) {
            self.config.present_mode = preferred_present_mode(&caps.present_modes)
                .ok_or_else(|| "recreated main surface has no present modes".to_string())?;
        }
        if !caps.alpha_modes.contains(&self.config.alpha_mode) {
            self.config.alpha_mode = caps
                .alpha_modes
                .first()
                .copied()
                .ok_or_else(|| "recreated main surface has no alpha modes".to_string())?;
        }
        self.config.width = size.width;
        self.config.height = size.height;
        configure_surface_checked(&self.device, &surface, &self.config, &self.gpu_health)?;
        self.surface = surface;
        self.main_surface_suspended = false;
        self.main_needs_reconfigure = false;
        Ok(())
    }

    /// Reconfigure an existing but outdated main surface. A `Lost` surface
    /// must instead go through `recreate_main_surface`.
    pub fn reconfigure_surface(&mut self) {
        self.main_needs_reconfigure = true;
    }

    /// Whether a routed frame can resolve ProgramHistory to the exact previous
    /// clean program. Call this immediately before frame evaluation.
    pub fn program_history_initialized(&self) -> bool {
        self.image_routing_gpu
            .lock()
            .map(|resources| {
                resources
                    .as_ref()
                    .is_some_and(|resources| resources.history_valid)
            })
            .unwrap_or(false)
    }

    fn ensure_image_routing_resources(
        &self,
        evaluated: &EvaluatedFramePlan,
    ) -> Result<std::sync::MutexGuard<'_, Option<ImageRoutingGpuResources>>, String> {
        let request = evaluated.image_routing().resource_plan().ok_or_else(|| {
            "image-routing GPU resources requested for an inactive plan".to_string()
        })?;
        let adapter_plan = MatteResourcePlan::validate(
            request.output_size,
            evaluated.image_routing().taps().len(),
            MatteResourceLimits::from_wgpu(&self.device.limits()),
        )?;
        if request != adapter_plan {
            return Err(format!(
                "evaluated image-routing resource plan {request:?} differs from adapter plan {adapter_plan:?}"
            ));
        }

        let mut locked = self
            .image_routing_gpu
            .lock()
            .map_err(|_| "image-routing GPU resource lock is poisoned".to_string())?;
        match locked.as_mut() {
            None => {
                let resources = create_gpu_resources_checked(
                    &self.device,
                    "image-routing full-frame allocation",
                    || ImageRoutingGpuResources::build(&self.device, adapter_plan),
                )?;
                *locked = Some(resources);
            }
            Some(resources) => {
                if resources.output_size != adapter_plan.output_size {
                    return Err(format!(
                        "image-routing resources are {}x{}, requested {}x{}",
                        resources.output_size[0],
                        resources.output_size[1],
                        adapter_plan.output_size[0],
                        adapter_plan.output_size[1]
                    ));
                }
                if resources.tap_layers != adapter_plan.tap_layers {
                    let taps = create_gpu_resources_checked(
                        &self.device,
                        "image tap array reallocation",
                        || {
                            ImageTapTexture::build(
                                &self.device,
                                adapter_plan.output_size,
                                adapter_plan.tap_layers,
                            )
                        },
                    )?;
                    resources.taps = taps;
                    resources.tap_layers = adapter_plan.tap_layers;
                }
            }
        }
        Ok(locked)
    }

    fn materialize_image_taps(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        resources: &LiveFrameResources,
        evaluated: &EvaluatedFramePlan,
        gpu: &ImageRoutingGpuResources,
    ) -> Result<(), String> {
        let Some(tap_texture) = gpu.taps.as_ref() else {
            if evaluated.image_routing().taps().is_empty() {
                return Ok(());
            }
            return Err("image-routing plan has taps but no GPU tap array".into());
        };
        for tap in evaluated.image_routing().taps() {
            let layer = evaluated
                .layers()
                .get(tap.donor_layer_index)
                .ok_or_else(|| {
                    format!(
                        "image tap donor index {} is outside the evaluated layer stack",
                        tap.donor_layer_index
                    )
                })?;
            let output_view = tap_texture
                .views
                .get(tap.array_layer as usize)
                .ok_or_else(|| format!("image tap array layer {} is missing", tap.array_layer))?;
            let pass_uniforms = match tap.stage {
                crate::image_routing::LayerImageStage::PreLocalEffects => {
                    evaluated.layer_pre_passes().get(tap.donor_layer_index)
                }
                crate::image_routing::LayerImageStage::PostLocalEffects => {
                    evaluated.layer_passes().get(tap.donor_layer_index)
                }
            }
            .ok_or_else(|| "image tap pass/layer alignment mismatch".to_string())?;
            let input_view = resources.texture_view(layer.source)?;
            let fx_buffer = create_uploaded_uniform(
                &self.device,
                &self.queue,
                "Image Tap FX Uniforms",
                pass_uniforms,
            );
            let texture_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Image Tap FX Input"),
                layout: &self.effects_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.nearest_sampler),
                    },
                ],
            });
            let uniform_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Image Tap FX Uniforms BG"),
                layout: &self.effects_uniform_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: fx_buffer.as_entire_binding(),
                }],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Materialize Composition-Aligned Image Tap"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(transparent_accumulation_clear()),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&self.effects_pipeline);
            pass.set_bind_group(0, &texture_group, &[]);
            pass.set_bind_group(1, &uniform_group, &[]);
            pass.draw(0..3, 0..1);
        }
        Ok(())
    }

    fn encode_matte_composite(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        base: &wgpu::TextureView,
        overlay: &wgpu::TextureView,
        donor: &wgpu::TextureView,
        output: &wgpu::TextureView,
        uniforms: MatteCompositeUniforms,
    ) {
        encode_matte_composite(
            &self.device,
            &self.queue,
            encoder,
            &self.matte_composite,
            &self.sampler,
            base,
            overlay,
            donor,
            output,
            uniforms,
        );
    }

    /// Copy slot 1 only at the requested stable layer's post-local boundary.
    /// Every non-target layer is a no-op and no warmed resource is created.
    fn capture_legacy_layer_output(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        stable_id: u64,
    ) -> Result<(), String> {
        let mut prepared = self.program_recorder_readback.borrow_mut();
        let Some(prepared) = prepared.as_mut() else {
            return Ok(());
        };
        let crate::program_recorder::CaptureTarget::Layer(target) = prepared.target else {
            return Ok(());
        };
        if target.get() != stable_id {
            return Ok(());
        }
        let Some(armed) = prepared.armed_scope else {
            return Ok(());
        };
        if armed.captured {
            return Ok(());
        }
        prepared
            .staging
            .encode_reserved(encoder, armed.reservation, &self.composite_textures[1])
            .map_err(|error| error.to_string())?;
        prepared.armed_scope = Some(ArmedLegacyScopeReadback {
            captured: true,
            ..armed
        });
        Ok(())
    }

    /// Render all plan-selected layers. Final result ends up in
    /// `composite_views[0]`; authored `Layer` state is never consulted.
    fn render_evaluated_layers(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        resources: &LiveFrameResources,
        evaluated: &EvaluatedFramePlan,
    ) -> Result<(), String> {
        // Render in reverse order: last layer in the vec is the bottom,
        // first layer (index 0, "Layer 1" in UI) ends up on top.
        let visible_layers =
            visible_stack_indices(evaluated.layers().iter().map(|layer| layer.visible));

        // Visibility is a transport control, not just a compositing hint. If
        // every layer is hidden, clear the prior accumulation instead of
        // leaving the last visible frame latched on every output.
        if visible_layers.is_empty() {
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.composite_views[0],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(transparent_accumulation_clear()),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            return Ok(());
        }

        for (i, &layer_index) in visible_layers.iter().enumerate() {
            let layer = &evaluated.layers()[layer_index];
            let pass_uniforms = &evaluated.layer_passes()[layer_index];
            let texture_view = resources.texture_view(layer.source)?;

            // Each pass needs its own buffer because queue.write_buffer writes
            // all execute before the encoder's render passes run on the GPU.
            let fx_buffer = create_uploaded_uniform(
                &self.device,
                &self.queue,
                "Layer FX Uniforms",
                pass_uniforms,
            );

            let tex_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.effects_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.nearest_sampler),
                    },
                ],
            });

            let uniform_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.effects_uniform_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: fx_buffer.as_entire_binding(),
                }],
            });

            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Layer FX"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.composite_views[1],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                pass.set_pipeline(&self.effects_pipeline);
                pass.set_bind_group(0, &tex_bg, &[]);
                pass.set_bind_group(1, &uniform_bg, &[]);
                pass.draw(0..3, 0..1);
            }
            self.capture_legacy_layer_output(encoder, layer.source.stable_id)?;

            // Step 2: Composite layer onto accumulated result.
            // The bottom layer composites onto a cleared black base with its
            // (modulated) opacity — it was previously copied raw, which
            // silently ignored opacity whenever only one layer was visible.
            // Its blend mode is forced to Normal: screen against black is
            // identity and multiply is a black frame — only surprises.
            if i == 0 {
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Clear Base"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.composite_views[0],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(transparent_accumulation_clear()),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
            }
            {
                // Composite base[0] + overlay[1] → temp[2]
                let composite_uniforms = CompositeUniforms {
                    opacity: layer.opacity,
                    blend_mode: layer.blend_mode.as_u32(),
                    _pad: [0.0; 2],
                };
                let comp_buffer = create_uploaded_uniform(
                    &self.device,
                    &self.queue,
                    "Composite Uniforms",
                    &composite_uniforms,
                );

                let composite_tex_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Composite Textures BG"),
                    layout: &self.composite_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&self.composite_views[0]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&self.composite_views[1]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                });

                let composite_uniform_bg =
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Composite Uniform BG"),
                        layout: &self.composite_uniform_layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: comp_buffer.as_entire_binding(),
                        }],
                    });

                // Render composite to [2]
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Composite Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.composite_views[2],
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        ..Default::default()
                    });
                    pass.set_pipeline(&self.composite_pipeline);
                    pass.set_bind_group(0, &composite_tex_bg, &[]);
                    pass.set_bind_group(1, &composite_uniform_bg, &[]);
                    pass.draw(0..3, 0..1);
                }

                // Copy [2] → [0] so it becomes the new accumulated base
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.composite_textures[2],
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.composite_textures[0],
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: self.output_width,
                        height: self.output_height,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }
        Ok(())
    }

    fn render_routed_layers(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        resources: &LiveFrameResources,
        evaluated: &EvaluatedFramePlan,
        gpu: &ImageRoutingGpuResources,
        path: MasterFxCompositionPath,
    ) -> Result<(), String> {
        let visible_layers =
            visible_stack_indices(evaluated.layers().iter().map(|layer| layer.visible));
        if visible_layers.is_empty() {
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Routed Clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.composite_views[0],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(transparent_accumulation_clear()),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            if path == MasterFxCompositionPath::LegacyPostComposite {
                self.render_master_pass(encoder, evaluated.master_pass());
            }
            return Ok(());
        }

        let conditional_master_groups = (path == MasterFxCompositionPath::ConditionalPerLayer)
            .then(|| {
                let buffer = create_uploaded_uniform(
                    &self.device,
                    &self.queue,
                    "Routed Conditional Master FX Uniforms",
                    evaluated.master_pass(),
                );
                let texture_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Routed Conditional Master FX Input"),
                    layout: &self.effects_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&self.composite_views[1]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.nearest_sampler),
                        },
                    ],
                });
                let uniform_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Routed Conditional Master FX Uniforms BG"),
                    layout: &self.effects_uniform_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    }],
                });
                (texture_group, uniform_group)
            });

        for (stack_index, &layer_index) in visible_layers.iter().enumerate() {
            let layer = &evaluated.layers()[layer_index];
            let pass_uniforms = &evaluated.layer_passes()[layer_index];
            let texture_view = resources.texture_view(layer.source)?;
            let fx_buffer = create_uploaded_uniform(
                &self.device,
                &self.queue,
                "Routed Layer FX Uniforms",
                pass_uniforms,
            );
            let layer_texture_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Routed Layer FX Input"),
                layout: &self.effects_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.nearest_sampler),
                    },
                ],
            });
            let layer_uniform_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Routed Layer FX Uniforms BG"),
                layout: &self.effects_uniform_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: fx_buffer.as_entire_binding(),
                }],
            });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Routed Layer FX Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.composite_views[1],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                pass.set_pipeline(&self.effects_pipeline);
                pass.set_bind_group(0, &layer_texture_group, &[]);
                pass.set_bind_group(1, &layer_uniform_group, &[]);
                pass.draw(0..3, 0..1);
            }
            self.capture_legacy_layer_output(encoder, layer.source.stable_id)?;

            let mut overlay_slot = 1;
            if path == MasterFxCompositionPath::ConditionalPerLayer && !layer.bypass_master_fx {
                let (master_texture_group, master_uniform_group) = conditional_master_groups
                    .as_ref()
                    .ok_or_else(|| "conditional master groups are missing".to_string())?;
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Routed Conditional Master FX Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.composite_views[2],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                pass.set_pipeline(&self.effects_pipeline);
                pass.set_bind_group(0, master_texture_group, &[]);
                pass.set_bind_group(1, master_uniform_group, &[]);
                pass.draw(0..3, 0..1);
                overlay_slot = 2;
            }

            if stack_index == 0 {
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Routed Clear Base"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.composite_views[0],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(transparent_accumulation_clear()),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
            }

            let matte = evaluated
                .image_routing()
                .mattes()
                .get(layer_index)
                .ok_or_else(|| "image matte/layer alignment mismatch".to_string())?;
            let mut params = matte.params;
            let donor_view = match matte.resolved_input {
                ResolvedImageInput::Disabled => {
                    params.amount = 0.0;
                    params.donor_valid = false;
                    &gpu.history.view
                }
                ResolvedImageInput::MaterializedTap { tap_index } => {
                    let tap = evaluated
                        .image_routing()
                        .taps()
                        .get(tap_index)
                        .ok_or_else(|| {
                            format!("resolved matte refers to missing tap {tap_index}")
                        })?;
                    gpu.taps
                        .as_ref()
                        .and_then(|texture| texture.views.get(tap.array_layer as usize))
                        .ok_or_else(|| {
                            format!(
                                "resolved matte tap array layer {} is missing",
                                tap.array_layer
                            )
                        })?
                }
                ResolvedImageInput::AllBelow => &self.composite_views[0],
                ResolvedImageInput::ProgramHistory => {
                    params.donor_valid &= gpu.history_valid;
                    &gpu.history.view
                }
                ResolvedImageInput::Transparent => {
                    params.donor_valid = false;
                    &gpu.history.view
                }
            };
            let output_slot = if overlay_slot == 1 { 2 } else { 1 };
            let uniforms =
                MatteCompositeUniforms::new(layer.opacity, layer.blend_mode.as_u32(), params);
            self.encode_matte_composite(
                encoder,
                &self.composite_views[0],
                &self.composite_views[overlay_slot],
                donor_view,
                &self.composite_views[output_slot],
                uniforms,
            );
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.composite_textures[output_slot],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &self.composite_textures[0],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: self.output_width,
                    height: self.output_height,
                    depth_or_array_layers: 1,
                },
            );
        }

        if path == MasterFxCompositionPath::LegacyPostComposite {
            self.render_master_pass(encoder, evaluated.master_pass());
        }
        Ok(())
    }

    fn encode_program_history(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gpu: &mut ImageRoutingGpuResources,
    ) {
        encode_program_history_copy(encoder, &self.composite_textures[0], gpu);
    }

    /// Render the complete direct-effects portion of the program.
    ///
    /// With no visible bypass this calls the two established passes verbatim:
    /// all layers composite first and the master shader runs exactly once on
    /// the result. If any visible layer bypasses master FX, every layer keeps
    /// its original stack position, local FX, opacity, and blend mode, while
    /// inherited layers receive the master shader immediately before their
    /// composite pass. Temporal and NTSC/VHS remain downstream global stages.
    pub fn render_evaluated_frame(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        resources: &LiveFrameResources,
        evaluated: &EvaluatedFramePlan,
    ) -> Result<(), String> {
        if evaluated.context().output_size != [self.output_width, self.output_height] {
            return Err(format!(
                "evaluated frame is {}x{}, renderer is {}x{}",
                evaluated.context().output_size[0],
                evaluated.context().output_size[1],
                self.output_width,
                self.output_height
            ));
        }
        if evaluated.layers().len() != evaluated.layer_passes().len() {
            return Err("evaluated layer metadata/pass alignment mismatch".into());
        }
        if evaluated.layers().len() != evaluated.layer_pre_passes().len() {
            return Err("evaluated layer/pre-pass alignment mismatch".into());
        }
        let path = master_fx_composition_path(
            evaluated
                .layers()
                .iter()
                .map(|layer| (layer.visible, layer.bypass_master_fx, layer.opacity)),
        );

        if evaluated.image_routing().is_active() {
            if evaluated.image_routing().mattes().len() != evaluated.layers().len() {
                return Err("evaluated image matte/layer alignment mismatch".into());
            }
            if evaluated.image_routing().stable_layers().len() != evaluated.layers().len() {
                return Err("evaluated stable-ID map/layer alignment mismatch".into());
            }
            let mut locked = self.ensure_image_routing_resources(evaluated)?;
            let gpu = locked
                .as_mut()
                .ok_or_else(|| "image-routing GPU resources were not retained".to_string())?;
            self.materialize_image_taps(encoder, resources, evaluated, gpu)?;
            self.render_routed_layers(encoder, resources, evaluated, gpu, path)?;
            // Capture after local/composite/master and before the caller's
            // temporal or NTSC stage: this is the exact clean program at N.
            self.encode_program_history(encoder, gpu);
            return Ok(());
        }

        match path {
            MasterFxCompositionPath::LegacyPostComposite => {
                // Keep this sequence exactly equivalent to the pre-bypass
                // renderer for old patches and all-inherited performances.
                self.render_evaluated_layers(encoder, resources, evaluated)?;
                self.render_master_pass(encoder, evaluated.master_pass());
            }
            MasterFxCompositionPath::ConditionalPerLayer => {
                self.render_evaluated_layers_with_conditional_master(
                    encoder, resources, evaluated,
                )?;
            }
        }
        // Once routing has ever been used, keep exactly one clean N-1 image
        // current even through disabled frames. This copy does not alter the
        // untouched legacy render/composite command sequence or arithmetic.
        let mut locked = self
            .image_routing_gpu
            .lock()
            .map_err(|_| "image-routing GPU resource lock is poisoned".to_string())?;
        if let Some(gpu) = locked.as_mut() {
            self.encode_program_history(encoder, gpu);
        }
        Ok(())
    }

    fn render_evaluated_layers_with_conditional_master(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        resources: &LiveFrameResources,
        evaluated: &EvaluatedFramePlan,
    ) -> Result<(), String> {
        // Preserve the established stack law: the last vector element is the
        // bottom and UI Layer 1 (index 0) is composited last, on top.
        let visible_layers =
            visible_stack_indices(evaluated.layers().iter().map(|layer| layer.visible));

        debug_assert!(visible_layers
            .iter()
            .any(|&index| evaluated.layers()[index].bypass_master_fx));

        // Every inherited layer reads the local-FX image from slot 1. Reuse
        // one immutable master uniform buffer and bind groups for all of them.
        let master_buffer = create_uploaded_uniform(
            &self.device,
            &self.queue,
            "Conditional Master FX Uniforms",
            evaluated.master_pass(),
        );
        let master_tex_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Conditional Master FX Input"),
            layout: &self.effects_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.composite_views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.nearest_sampler),
                },
            ],
        });
        let master_uniform_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Conditional Master FX Uniforms BG"),
            layout: &self.effects_uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: master_buffer.as_entire_binding(),
            }],
        });

        for (stack_index, &layer_index) in visible_layers.iter().enumerate() {
            let layer = &evaluated.layers()[layer_index];
            let pass_uniforms = &evaluated.layer_passes()[layer_index];
            let texture_view = resources.texture_view(layer.source)?;
            let fx_buffer = create_uploaded_uniform(
                &self.device,
                &self.queue,
                "Conditional Layer FX Uniforms",
                pass_uniforms,
            );
            let layer_tex_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Conditional Layer FX Input"),
                layout: &self.effects_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.nearest_sampler),
                    },
                ],
            });
            let layer_uniform_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Conditional Layer FX Uniforms BG"),
                layout: &self.effects_uniform_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: fx_buffer.as_entire_binding(),
                }],
            });

            // Source -> local layer FX -> slot 1.
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Conditional Layer FX Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.composite_views[1],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                pass.set_pipeline(&self.effects_pipeline);
                pass.set_bind_group(0, &layer_tex_bg, &[]);
                pass.set_bind_group(1, &layer_uniform_bg, &[]);
                pass.draw(0..3, 0..1);
            }
            self.capture_legacy_layer_output(encoder, layer.source.stable_id)?;

            let slots = conditional_layer_slots(layer.bypass_master_fx);
            if let Some(master_output) = slots.master_output {
                debug_assert_eq!(master_output, 2);
                // Local output slot 1 -> modulated master FX -> slot 2.
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Conditional Master FX Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.composite_views[master_output],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                pass.set_pipeline(&self.effects_pipeline);
                pass.set_bind_group(0, &master_tex_bg, &[]);
                pass.set_bind_group(1, &master_uniform_bg, &[]);
                pass.draw(0..3, 0..1);
            }

            if stack_index == 0 {
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Conditional Clear Base"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.composite_views[0],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(transparent_accumulation_clear()),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
            }

            let overlay_slot = slots.master_output.unwrap_or(1);
            let composite_uniforms = CompositeUniforms {
                opacity: layer.opacity,
                blend_mode: layer.blend_mode.as_u32(),
                _pad: [0.0; 2],
            };
            let comp_buffer = create_uploaded_uniform(
                &self.device,
                &self.queue,
                "Conditional Composite Uniforms",
                &composite_uniforms,
            );
            let composite_tex_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Conditional Composite Textures BG"),
                layout: &self.composite_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.composite_views[0]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &self.composite_views[overlay_slot],
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            let composite_uniform_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Conditional Composite Uniform BG"),
                layout: &self.composite_uniform_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: comp_buffer.as_entire_binding(),
                }],
            });

            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Conditional Composite Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.composite_views[slots.composite_output],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                pass.set_pipeline(&self.composite_pipeline);
                pass.set_bind_group(0, &composite_tex_bg, &[]);
                pass.set_bind_group(1, &composite_uniform_bg, &[]);
                pass.draw(0..3, 0..1);
            }

            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.composite_textures[slots.composite_output],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &self.composite_textures[0],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: self.output_width,
                    height: self.output_height,
                    depth_or_array_layers: 1,
                },
            );
        }
        Ok(())
    }

    /// Apply master effects to the final composite (already in [0]).
    /// Reads [0], applies effects → [2], copies back to [0].
    fn render_master_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pass_uniforms: &EffectPassUniforms,
    ) {
        let fx_buffer = create_uploaded_uniform(
            &self.device,
            &self.queue,
            "Master FX Uniforms",
            pass_uniforms,
        );

        let tex_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Master FX Input"),
            layout: &self.effects_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.composite_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.nearest_sampler),
                },
            ],
        });

        let uniform_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Master FX Uniforms BG"),
            layout: &self.effects_uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: fx_buffer.as_entire_binding(),
            }],
        });

        // Render effects from [0] → [2]
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Master FX Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.composite_views[2],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&self.effects_pipeline);
            pass.set_bind_group(0, &tex_bg, &[]);
            pass.set_bind_group(1, &uniform_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // Copy [2] → [0]
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.composite_textures[2],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.composite_textures[0],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.output_width,
                height: self.output_height,
                depth_or_array_layers: 1,
            },
        );
    }

    fn selective_batch_layout(&self, layer_count: usize) -> Result<(u32, u64, u64), String> {
        let memory = validate_selective_ntsc_live_memory(
            self.output_width,
            self.output_height,
            layer_count,
        )?;
        let memory = validate_selective_ntsc_gpu_staging_limit(
            memory,
            self.device.limits().max_buffer_size,
        )?;
        Ok((
            memory.padded_row_bytes,
            memory.slice_stride,
            memory.gpu_staging_bytes,
        ))
    }

    fn ensure_selective_ntsc_gpu(&mut self, required_capacity: u64) -> Result<(), String> {
        self.ensure_device_healthy()
            .map_err(|error| format!("selective NTSC GPU allocation unavailable: {error}"))?;
        let needs_state = self.selective_ntsc_gpu.is_none();
        if needs_state {
            let state = create_gpu_resources_checked(
                &self.device,
                &format!(
                    "selective NTSC full-output resources at {}x{} with a \
                     {required_capacity}-byte staging buffer",
                    self.output_width, self.output_height
                ),
                || {
                    let usage = wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC;
                    let scratch_textures: [wgpu::Texture; 2] = std::array::from_fn(|index| {
                        self.device.create_texture(&wgpu::TextureDescriptor {
                            label: Some(&format!("Selective NTSC Scratch {index}")),
                            size: wgpu::Extent3d {
                                width: self.output_width,
                                height: self.output_height,
                                depth_or_array_layers: 1,
                            },
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: COMPOSITE_FORMAT,
                            usage,
                            view_formats: &[],
                        })
                    });
                    let scratch_views = std::array::from_fn(|index| {
                        scratch_textures[index].create_view(&wgpu::TextureViewDescriptor::default())
                    });
                    let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("Selective NTSC Readback Batch"),
                        size: required_capacity,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    });
                    SelectiveNtscGpuState {
                        scratch_textures,
                        scratch_views,
                        slot: SelectiveNtscReadbackSlot {
                            buffer,
                            capacity: required_capacity,
                            status: Arc::new(AtomicU8::new(SLOT_IDLE)),
                            used_size: 0,
                            padded_row_bytes: 0,
                            slice_stride: 0,
                            plan: None,
                        },
                    }
                },
            )?;
            self.ensure_device_healthy().map_err(|error| {
                format!("selective NTSC full-output resource allocation failed: {error}")
            })?;
            self.selective_ntsc_gpu = Some(state);
            return Ok(());
        }

        let current_capacity = self
            .selective_ntsc_gpu
            .as_ref()
            .map(|state| state.slot.capacity)
            .unwrap();
        if current_capacity < required_capacity {
            debug_assert_eq!(
                self.selective_ntsc_gpu
                    .as_ref()
                    .unwrap()
                    .slot
                    .status
                    .load(Ordering::Acquire),
                SLOT_IDLE
            );
            let buffer = create_gpu_resources_checked(
                &self.device,
                &format!(
                    "selective NTSC staging-buffer resize from {} to {required_capacity} bytes",
                    current_capacity
                ),
                || {
                    self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("Selective NTSC Readback Batch"),
                        size: required_capacity,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    })
                },
            )?;
            self.ensure_device_healthy()
                .map_err(|error| format!("selective NTSC staging-buffer resize failed: {error}"))?;
            let state = self.selective_ntsc_gpu.as_mut().unwrap();
            state.slot.buffer = buffer;
            state.slot.capacity = required_capacity;
        }
        Ok(())
    }

    /// Encode one generation-coherent selective-VHS batch. Every contributing
    /// layer is rendered through local FX and, unless bypassed, direct master
    /// FX into dedicated scratch textures. All slices are then copied into one
    /// aligned staging allocation in this same command stream. Milestone 1
    /// rejects active image mattes before touching the one-slot queue, allowing
    /// the caller to retain its last accepted audience frame with a visible
    /// status instead of emitting unmasked slices.
    pub fn begin_selective_ntsc_readback(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        resources: &LiveFrameResources,
        evaluated: &EvaluatedFramePlan,
        plan: SelectiveNtscPlan,
    ) -> Result<bool, String> {
        self.ensure_device_healthy()?;
        validate_selective_matte_topology(evaluated.image_routing().is_active())?;
        if plan.generation.width != self.output_width
            || plan.generation.height != self.output_height
        {
            return Err("selective NTSC plan dimensions do not match the renderer".into());
        }
        if evaluated.context().output_size != [self.output_width, self.output_height] {
            return Err("selective NTSC evaluated-frame dimensions do not match renderer".into());
        }
        if evaluated.layers().len() != evaluated.layer_passes().len() {
            return Err("selective NTSC evaluated layer/pass alignment mismatch".into());
        }
        if self
            .selective_ntsc_gpu
            .as_ref()
            .is_some_and(|state| state.slot.status.load(Ordering::Acquire) != SLOT_IDLE)
        {
            return Ok(false);
        }

        let mut source_views = Vec::with_capacity(plan.layers.len());
        let mut layer_uniforms = Vec::with_capacity(plan.layers.len());
        for planned_layer in &plan.layers {
            let source_index = evaluated
                .layers()
                .iter()
                .position(|layer| layer.source.stable_id == planned_layer.layer_id)
                .ok_or_else(|| {
                    format!(
                        "selective NTSC layer {} disappeared before encoding",
                        planned_layer.layer_id
                    )
                })?;
            let layer = &evaluated.layers()[source_index];
            if !layer.visible
                || layer.bypass_master_fx != planned_layer.bypass_master_fx
                || layer.opacity != planned_layer.opacity
                || layer.blend_mode.as_u32() != planned_layer.blend_mode
            {
                return Err(format!(
                    "selective NTSC layer {} changed before encoding",
                    planned_layer.layer_id
                ));
            }
            source_views.push(resources.texture_view(layer.source)?);
            layer_uniforms.push(evaluated.layer_passes()[source_index]);
        }

        let (padded_row_bytes, slice_stride, used_size) =
            self.selective_batch_layout(plan.layers.len())?;
        self.ensure_selective_ntsc_gpu(used_size)?;

        let state = self.selective_ntsc_gpu.as_ref().unwrap();
        let bindings = create_gpu_resources_checked(
            &self.device,
            "selective NTSC per-batch uniform buffers and bind groups",
            || {
                let master_uniform_buffer = create_uploaded_uniform(
                    &self.device,
                    &self.queue,
                    "Selective NTSC Master FX Uniforms",
                    evaluated.master_pass(),
                );
                let master_texture_bind_group =
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Selective NTSC Master FX Input"),
                        layout: &self.effects_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &state.scratch_views[0],
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&self.sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Sampler(&self.nearest_sampler),
                            },
                        ],
                    });
                let master_uniform_bind_group =
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Selective NTSC Master FX Uniforms BG"),
                        layout: &self.effects_uniform_layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: master_uniform_buffer.as_entire_binding(),
                        }],
                    });
                let layers = source_views
                    .iter()
                    .zip(&layer_uniforms)
                    .map(|(&source_view, uniforms)| {
                        let uniform_buffer = create_uploaded_uniform(
                            &self.device,
                            &self.queue,
                            "Selective NTSC Layer FX Uniforms",
                            uniforms,
                        );
                        let texture_bind_group =
                            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("Selective NTSC Layer FX Input"),
                                layout: &self.effects_bind_group_layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: wgpu::BindingResource::TextureView(source_view),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: wgpu::BindingResource::Sampler(
                                            &self.nearest_sampler,
                                        ),
                                    },
                                ],
                            });
                        let uniform_bind_group =
                            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("Selective NTSC Layer FX Uniforms BG"),
                                layout: &self.effects_uniform_layout,
                                entries: &[wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: uniform_buffer.as_entire_binding(),
                                }],
                            });
                        SelectiveNtscLayerGpuBindings {
                            _uniform_buffer: uniform_buffer,
                            texture_bind_group,
                            uniform_bind_group,
                        }
                    })
                    .collect();
                SelectiveNtscBatchGpuBindings {
                    _master_uniform_buffer: master_uniform_buffer,
                    master_texture_bind_group,
                    master_uniform_bind_group,
                    layers,
                }
            },
        )?;
        self.ensure_device_healthy().map_err(|error| {
            format!("selective NTSC per-batch resource allocation failed: {error}")
        })?;

        for (slice_index, planned_layer) in plan.layers.iter().enumerate() {
            let layer_bindings = &bindings.layers[slice_index];

            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Selective NTSC Layer FX Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &state.scratch_views[0],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                pass.set_pipeline(&self.effects_pipeline);
                pass.set_bind_group(0, &layer_bindings.texture_bind_group, &[]);
                pass.set_bind_group(1, &layer_bindings.uniform_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }

            let output_index = if planned_layer.bypass_master_fx {
                0
            } else {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Selective NTSC Direct Master FX Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &state.scratch_views[1],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                pass.set_pipeline(&self.effects_pipeline);
                pass.set_bind_group(0, &bindings.master_texture_bind_group, &[]);
                pass.set_bind_group(1, &bindings.master_uniform_bind_group, &[]);
                pass.draw(0..3, 0..1);
                1
            };

            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &state.scratch_textures[output_index],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &state.slot.buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: slice_stride * slice_index as u64,
                        bytes_per_row: Some(padded_row_bytes),
                        rows_per_image: Some(self.output_height),
                    },
                },
                wgpu::Extent3d {
                    width: self.output_width,
                    height: self.output_height,
                    depth_or_array_layers: 1,
                },
            );
        }

        let slot = &mut self.selective_ntsc_gpu.as_mut().unwrap().slot;
        slot.used_size = used_size;
        slot.padded_row_bytes = padded_row_bytes;
        slot.slice_stride = slice_stride;
        slot.plan = Some(plan);
        slot.status.store(SLOT_MAP_PENDING, Ordering::Release);
        Ok(true)
    }

    /// Request the asynchronous map after the encoder containing the batch
    /// copy has been submitted.
    pub fn map_selective_ntsc_readback(&self) {
        if self.ensure_device_healthy().is_err() {
            return;
        }
        let Some(state) = self.selective_ntsc_gpu.as_ref() else {
            return;
        };
        if state.slot.status.load(Ordering::Acquire) != SLOT_MAP_PENDING {
            return;
        }
        // Mark the request before handing control to wgpu. The render loop
        // may call this method again before the callback fires; a distinct
        // state prevents a second map_async on the same buffer range.
        state
            .slot
            .status
            .store(SLOT_MAP_REQUESTED, Ordering::Release);
        let status = state.slot.status.clone();
        state.slot.buffer.slice(0..state.slot.used_size).map_async(
            wgpu::MapMode::Read,
            move |result| {
                status.store(
                    if result.is_ok() {
                        SLOT_MAPPED
                    } else {
                        SLOT_MAP_FAILED
                    },
                    Ordering::Release,
                );
            },
        );
    }

    /// Whether the one-slot selective staging pipeline currently owns a GPU
    /// batch. This is separate from CPU-worker occupancy and is surfaced only
    /// as live diagnostics; callers must still use `begin_*` for admission.
    pub fn selective_ntsc_readback_busy(&self) -> bool {
        self.selective_ntsc_gpu
            .as_ref()
            .is_some_and(|state| state.slot.status.load(Ordering::Acquire) != SLOT_IDLE)
    }

    /// Harvest the single newest mapped selective batch without blocking.
    pub fn poll_selective_ntsc_readback(&mut self) -> Option<SelectiveNtscBatch> {
        if self.ensure_device_healthy().is_err() {
            return None;
        }
        let _ = self.device.poll(wgpu::PollType::Poll);
        if self.gpu_health.error().is_some() {
            return None;
        }
        let state = self.selective_ntsc_gpu.as_mut()?;
        match state.slot.status.load(Ordering::Acquire) {
            SLOT_MAPPED => {
                let plan = state.slot.plan.take();
                let data = state
                    .slot
                    .buffer
                    .slice(0..state.slot.used_size)
                    .get_mapped_range();
                let batch = plan.and_then(|plan| {
                    let row_bytes = (plan.generation.width as usize).checked_mul(4)?;
                    let height = plan.generation.height as usize;
                    let mut slices = Vec::with_capacity(plan.layers.len());
                    for index in 0..plan.layers.len() {
                        let slice_start = state.slot.slice_stride as usize * index;
                        let mut pixels = Vec::with_capacity(row_bytes.checked_mul(height)?);
                        for row in 0..height {
                            let start = slice_start + row * state.slot.padded_row_bytes as usize;
                            pixels.extend_from_slice(&data[start..start + row_bytes]);
                        }
                        slices.push(pixels);
                    }
                    Some(SelectiveNtscBatch { plan, slices })
                });
                drop(data);
                state.slot.buffer.unmap();
                state.slot.status.store(SLOT_IDLE, Ordering::Release);
                batch
            }
            SLOT_MAP_FAILED => {
                state.slot.plan = None;
                state.slot.status.store(SLOT_IDLE, Ordering::Release);
                None
            }
            _ => None,
        }
    }

    /// Upload a finished selective straight-alpha composite as the input to
    /// the shared temporal stage.
    pub fn write_engine_composite(&self, pixels: &[u8]) -> Result<(), String> {
        let expected = checked_rgba_frame_len(self.output_width, self.output_height)
            .ok_or_else(|| "engine composite dimensions overflow".to_string())?;
        if pixels.len() != expected {
            return Err(format!(
                "engine composite has {} bytes; expected {expected}",
                pixels.len()
            ));
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.composite_textures[0],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.output_width * 4),
                rows_per_image: Some(self.output_height),
            },
            wgpu::Extent3d {
                width: self.output_width,
                height: self.output_height,
                depth_or_array_layers: 1,
            },
        );
        Ok(())
    }

    /// Row stride aligned to wgpu's 256-byte copy requirement.
    fn readback_bytes_per_row(&self) -> u32 {
        (self.output_width * 4 + 255) & !255
    }

    /// Phase 1 of async readback: encode a copy of the final opaque audience
    /// image in composite_textures[2]
    /// into a free staging buffer. Returns the slot index to pass to
    /// `map_readback` after the encoder is submitted, or None if every
    /// slot still has a copy in flight. Lazy allocation/validation failures
    /// are returned before an invalid buffer handle can enter a command.
    pub fn begin_readback(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        epoch: u64,
        ntsc_metadata: Option<NtscFrameMetadata>,
    ) -> Result<Option<usize>, String> {
        self.begin_readback_tagged(encoder, epoch, ntsc_metadata, None, false)
    }

    /// Encode a readback of one exact selectively processed audience sample.
    /// The tag is retained with its pixels across asynchronous map completion,
    /// so Spout can reject a pre-transition result rather than inferring its
    /// identity from whichever frame happens to be displayed later.
    pub fn begin_selective_audience_readback(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        sample: SelectiveNtscGeneration,
    ) -> Result<Option<usize>, String> {
        self.begin_readback_tagged(encoder, sample.visual_epoch, None, Some(sample), false)
    }

    /// Read back the already materialized audience image retained by Pause.
    /// The explicit tag prevents it from being mistaken for raw global-NTSC
    /// input or for a newly processed selective generation.
    pub fn begin_held_audience_readback(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        epoch: u64,
    ) -> Result<Option<usize>, String> {
        self.begin_readback_tagged(encoder, epoch, None, None, true)
    }

    fn begin_readback_tagged(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        epoch: u64,
        ntsc_metadata: Option<NtscFrameMetadata>,
        selective_sample: Option<SelectiveNtscGeneration>,
        held_audience: bool,
    ) -> Result<Option<usize>, String> {
        self.ensure_device_healthy()
            .map_err(|error| format!("audience readback unavailable: {error}"))?;
        let padded_row_bytes = self.readback_bytes_per_row();
        let buffer_size = validate_readback_buffer_size(
            padded_row_bytes,
            self.output_height,
            self.device.limits().max_buffer_size,
        )?;

        let idx = match self
            .readback_slots
            .iter()
            .position(|s| s.status.load(Ordering::Acquire) == SLOT_IDLE)
        {
            Some(idx) => idx,
            None if self.readback_slots.len() < MAX_READBACK_SLOTS => {
                let buffer = create_gpu_resources_checked(
                    &self.device,
                    &format!(
                        "audience NTSC/Spout readback buffer {} of {MAX_READBACK_SLOTS} \
                         ({buffer_size} bytes at {}x{})",
                        self.readback_slots.len() + 1,
                        self.output_width,
                        self.output_height
                    ),
                    || {
                        self.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("NTSC/Spout Audience Readback Slot"),
                            size: buffer_size,
                            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                            mapped_at_creation: false,
                        })
                    },
                )?;
                self.ensure_device_healthy().map_err(|error| {
                    format!("audience NTSC/Spout readback buffer allocation failed: {error}")
                })?;
                self.readback_slots.push(ReadbackSlot {
                    buffer,
                    status: Arc::new(AtomicU8::new(SLOT_IDLE)),
                    sequence: 0,
                    epoch,
                    ntsc_metadata: None,
                    selective_sample: None,
                    held_audience: false,
                });
                self.readback_slots.len() - 1
            }
            None => return Ok(None),
        };

        let sequence = self.next_readback_sequence;
        self.next_readback_sequence = self.next_readback_sequence.saturating_add(1);
        self.readback_slots[idx].sequence = sequence;
        self.readback_slots[idx].epoch = epoch;
        self.readback_slots[idx].ntsc_metadata = ntsc_metadata;
        self.readback_slots[idx].selective_sample = selective_sample;
        self.readback_slots[idx].held_audience = held_audience;
        let slot = &self.readback_slots[idx];
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.composite_textures[2],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &slot.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row_bytes),
                    rows_per_image: Some(self.output_height),
                },
            },
            wgpu::Extent3d {
                width: self.output_width,
                height: self.output_height,
                depth_or_array_layers: 1,
            },
        );
        slot.status.store(SLOT_MAP_PENDING, Ordering::Release);
        Ok(Some(idx))
    }

    /// Phase 2: request the async map. Must be called after the encoder
    /// from `begin_readback` has been submitted. Never blocks.
    pub fn map_readback(&self, idx: usize) {
        if self.ensure_device_healthy().is_err() {
            return;
        }
        let slot = &self.readback_slots[idx];
        if slot.status.load(Ordering::Acquire) != SLOT_MAP_PENDING {
            return;
        }
        slot.status.store(SLOT_MAP_REQUESTED, Ordering::Release);
        let status = slot.status.clone();
        slot.buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let new = if result.is_ok() {
                    SLOT_MAPPED
                } else {
                    SLOT_MAP_FAILED
                };
                status.store(new, Ordering::Release);
            });
    }

    /// Phase 3: harvest a completed readback if any. Non-blocking; returns
    /// the freshest completed frame. Completed or subsequently completing
    /// older slots are discarded, so callback reordering can never make an
    /// old NTSC/Spout frame overwrite a newer composite.
    pub fn poll_readback(&mut self) -> ReadbackPoll {
        if self.readback_slots.is_empty() {
            return ReadbackPoll {
                frame: None,
                held_audience_not_harvested: false,
            };
        }
        if self.ensure_device_healthy().is_err() {
            return ReadbackPoll {
                frame: None,
                held_audience_not_harvested: false,
            };
        }
        // Drive map callbacks without waiting.
        let _ = self.device.poll(wgpu::PollType::Poll);
        if self.gpu_health.error().is_some() {
            return ReadbackPoll {
                frame: None,
                held_audience_not_harvested: false,
            };
        }

        let newest_idx = self
            .readback_slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| {
                slot.status.load(Ordering::Acquire) == SLOT_MAPPED
                    && slot.sequence > self.last_harvested_readback_sequence
            })
            .max_by_key(|(_, slot)| slot.sequence)
            .map(|(idx, _)| idx);
        let newest_sequence = newest_idx.map(|idx| self.readback_slots[idx].sequence);

        let row_bytes = (self.output_width * 4) as usize;
        let padded_row = self.readback_bytes_per_row() as usize;
        let h = self.output_height as usize;
        let mut harvested: Option<ReadbackFrame> = None;
        let mut held_audience_not_harvested = false;
        for (idx, slot) in self.readback_slots.iter_mut().enumerate() {
            match slot.status.load(Ordering::Acquire) {
                SLOT_MAPPED if Some(idx) == newest_idx => {
                    let data = slot.buffer.slice(..).get_mapped_range();
                    let mut pixels = Vec::with_capacity(row_bytes * h);
                    for row in 0..h {
                        let start = row * padded_row;
                        pixels.extend_from_slice(&data[start..start + row_bytes]);
                    }
                    drop(data);
                    slot.buffer.unmap();
                    slot.status.store(SLOT_IDLE, Ordering::Release);
                    harvested = Some(ReadbackFrame {
                        pixels,
                        epoch: slot.epoch,
                        ntsc_metadata: slot.ntsc_metadata.take(),
                        selective_sample: slot.selective_sample.take(),
                        held_audience: std::mem::take(&mut slot.held_audience),
                    });
                }
                SLOT_MAPPED => {
                    // This completed map is older than the freshest mapped
                    // frame (or than one already returned on an earlier poll).
                    slot.buffer.unmap();
                    slot.ntsc_metadata = None;
                    slot.selective_sample = None;
                    held_audience_not_harvested |= std::mem::take(&mut slot.held_audience);
                    slot.status.store(SLOT_IDLE, Ordering::Release);
                }
                SLOT_MAP_FAILED => {
                    // Device hiccup (e.g. surface loss); recycle the slot.
                    slot.ntsc_metadata = None;
                    slot.selective_sample = None;
                    held_audience_not_harvested |= std::mem::take(&mut slot.held_audience);
                    slot.status.store(SLOT_IDLE, Ordering::Release);
                }
                _ => {}
            }
        }
        if let Some(sequence) = newest_sequence {
            self.last_harvested_readback_sequence = sequence;
        }
        ReadbackPoll {
            frame: harvested,
            held_audience_not_harvested,
        }
    }

    /// Write processed opaque RGBA pixels back to the audience image. The
    /// straight-alpha engine accumulation in slot 0 remains untouched.
    pub fn write_composite(&self, pixels: &[u8]) {
        let w = self.output_width;
        let h = self.output_height;
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.composite_textures[2],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
    }
}
