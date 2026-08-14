use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use winit::window::Window;

use crate::effects::params::{normalized_slit_direction, TemporalParams, TEMPORAL_REFERENCE_FPS};
use crate::effects::EffectUniforms;
use crate::layers::Layer;
use crate::ntsc::{
    checked_rgba_frame_len, validate_selective_ntsc_gpu_staging_limit,
    validate_selective_ntsc_live_memory, NtscFrameMetadata, SelectiveNtscBatch,
    SelectiveNtscGeneration, SelectiveNtscPlan,
};

/// Frames of output history kept for temporal effects (0.8s at 30fps).
pub const HISTORY_LEN: u32 = 24;

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

/// Uniforms for the temporal (feedback/slit-scan) pass.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TemporalUniforms {
    feedback: f32,
    fb_zoom: f32,
    fb_rotate: f32,
    slitscan: f32,
    history_len: f32,
    write_index: f32,
    valid_history: f32,
    feedback_valid: f32,
    slit_direction: [f32; 2],
    key_reference_layer: f32,
    key_valid: f32,
    key_mode: f32,
    key_threshold: f32,
    key_softness: f32,
    _pad: f32,
}

/// Resolve a temporal key reference from the ring as it existed before the
/// current clean image is recorded. One render can observe several elapsed
/// 30-Hz clock ticks after a stall, but those ticks are not distinct images and
/// must never turn duplicate copies of the current image into fake history.
fn temporal_key_reference_layer(
    history_write: usize,
    history_valid: u32,
    requested_depth: f32,
) -> Option<usize> {
    let depth = if requested_depth.is_finite() {
        requested_depth.round().clamp(1.0, (HISTORY_LEN - 1) as f32) as usize
    } else {
        1
    };
    let offset = depth.saturating_sub(1);
    if history_valid == 0 || offset >= history_valid as usize {
        return None;
    }
    Some((history_write + HISTORY_LEN as usize - offset) % HISTORY_LEN as usize)
}

/// CPU-side lifetime for temporal GPU memories.
///
/// History snapshots are gated by a fixed 30 Hz clock and record at most one
/// distinct observation per render, while `history_valid` and `feedback_valid`
/// prevent the shader from ever touching an unwritten texture. The texture
/// contents themselves do not need an eager clear because invalid layers
/// remain unreachable.
#[derive(Debug, Clone)]
pub(crate) struct TemporalState {
    history_write: usize,
    history_valid: u32,
    history_accumulator: f64,
    feedback_valid: bool,
    initialized: bool,
    total_history_frames: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TemporalFramePlan {
    /// No completed program image exists yet; retain this clean image and seed
    /// the frozen-output memory without advancing clean history.
    PrimeFrozenOutput,
    /// Re-present the exact completed program image held in feedback memory.
    HoldFrozenOutput,
    /// Render a live program step and optionally record its clean observation.
    Advance { record_history: bool },
}

impl Default for TemporalState {
    fn default() -> Self {
        Self {
            history_write: 0,
            history_valid: 0,
            history_accumulator: 0.0,
            feedback_valid: false,
            initialized: false,
            total_history_frames: 0,
        }
    }
}

impl TemporalState {
    /// Reset validity immediately. Stale GPU pixels can remain allocated: the
    /// shader cannot sample them until fresh frames make them valid again.
    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }

    /// Number of 30-Hz clock ticks observed by this render step. The first
    /// rendered clean frame primes the clock immediately. The encoder records
    /// the current image at most once even when several ticks elapsed: a render
    /// supplies one observation, not one new image per missed wall-clock tick.
    fn history_ticks_for_delta(&mut self, delta_seconds: f32) -> u32 {
        if !self.initialized {
            self.initialized = true;
            return 1;
        }

        let reference_delta = 1.0 / TEMPORAL_REFERENCE_FPS as f64;
        let delta = if delta_seconds.is_finite() {
            delta_seconds.max(0.0) as f64
        } else {
            reference_delta
        };
        self.history_accumulator += delta;

        let elapsed_steps = (self.history_accumulator / reference_delta).floor() as u64;
        if elapsed_steps == 0 {
            return 0;
        }
        self.history_accumulator -= elapsed_steps as f64 * reference_delta;
        elapsed_steps.min(HISTORY_LEN as u64) as u32
    }

    fn plan_frame(&mut self, delta_seconds: f32, advance_program: bool) -> TemporalFramePlan {
        if !advance_program {
            return if self.feedback_valid {
                TemporalFramePlan::HoldFrozenOutput
            } else {
                TemporalFramePlan::PrimeFrozenOutput
            };
        }
        TemporalFramePlan::Advance {
            record_history: self.history_ticks_for_delta(delta_seconds) > 0,
        }
    }

    fn record_history_frame(&mut self) {
        if self.history_valid > 0 {
            self.history_write = (self.history_write + 1) % HISTORY_LEN as usize;
        }
        self.history_valid = (self.history_valid + 1).min(HISTORY_LEN);
        self.total_history_frames = self.total_history_frames.saturating_add(1);
    }
}

#[cfg(test)]
mod temporal_state_tests {
    use super::*;

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

    fn advance(state: &mut TemporalState, delta_seconds: f32) -> u32 {
        let ticks = state.history_ticks_for_delta(delta_seconds);
        if ticks > 0 {
            state.record_history_frame();
        }
        ticks
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
        for _ in 0..120 {
            assert_eq!(
                state.plan_frame(0.0, false),
                TemporalFramePlan::PrimeFrozenOutput
            );
        }
        assert!(!state.initialized);
        assert_eq!(state.history_valid, 0);
        assert_eq!(temporal_key_reference_layer(0, 0, 1.0), None);

        // The encoder seeds feedback after the first paused presentation.
        state.feedback_valid = true;
        assert_eq!(
            state.plan_frame(0.0, false),
            TemporalFramePlan::HoldFrozenOutput
        );
        assert_eq!(state.history_valid, 0);

        assert_eq!(
            state.plan_frame(0.0, true),
            TemporalFramePlan::Advance {
                record_history: true
            },
            "unpaused frame zero must prime clean history for live/export parity"
        );
        assert!(state.initialized);
        assert_eq!(state.history_valid, 0, "planning precedes the GPU copy");
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
                state.plan_frame(1.0, false),
                TemporalFramePlan::HoldFrozenOutput
            );
        }
        assert_eq!(state.history_write, frozen_write);
        assert_eq!(state.history_valid, frozen_valid);
        assert_eq!(state.total_history_frames, frozen_total);
        assert_eq!(state.history_accumulator, frozen_accumulator);

        assert_eq!(
            state.plan_frame(1.0 / 60.0, true),
            TemporalFramePlan::Advance {
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
        assert_eq!(std::mem::size_of::<TemporalUniforms>(), 64);
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
        let shader = include_str!("../shaders/composite.wgsl");
        assert!(shader.contains("let output_alpha = source_alpha + base.a * (1.0 - source_alpha);"));
        assert!(shader.contains("output_premultiplied / max(output_alpha, 0.000001)"));

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
    let record_current = match state.plan_frame(delta_seconds, advance_program) {
        TemporalFramePlan::PrimeFrozenOutput => {
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
            state.feedback_valid = true;
            return;
        }
        TemporalFramePlan::HoldFrozenOutput => {
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
        TemporalFramePlan::Advance { record_history } => record_history,
    };

    // Resolve every history read against the ring *before* recording this
    // clean image. A slow render can span several 30-Hz ticks, but it still
    // contributes only one distinct observation. Recording it repeatedly
    // would erase real history and make a motion key compare the image with
    // duplicate copies of itself.
    let previous_write = state.history_write;
    let previous_valid = state.history_valid;

    // The shader treats `write_index` as the current clean frame and subtracts
    // one for the newest historical sample. The current frame intentionally
    // remains in composite slot 0 during the pass, so expose its virtual ring
    // position one slot beyond the newest materialized history image.
    let virtual_write = if previous_valid == 0 {
        0
    } else {
        (previous_write + 1) % HISTORY_LEN as usize
    };
    let virtual_valid = if previous_valid == 0 {
        0
    } else {
        (previous_valid + 1).min(HISTORY_LEN)
    };

    let frame_params = params.for_frame_delta(delta_seconds);
    if frame_params.is_active() {
        let key_reference =
            temporal_key_reference_layer(previous_write, previous_valid, frame_params.key_history);
        let uniforms = TemporalUniforms {
            feedback: frame_params.feedback,
            fb_zoom: frame_params.fb_zoom,
            fb_rotate: frame_params.fb_rotate,
            slitscan: frame_params.slitscan,
            history_len: HISTORY_LEN as f32,
            write_index: virtual_write as f32,
            valid_history: virtual_valid as f32,
            feedback_valid: if state.feedback_valid { 1.0 } else { 0.0 },
            slit_direction: normalized_slit_direction(frame_params.slit_angle, width, height),
            key_reference_layer: key_reference.unwrap_or(0) as f32,
            key_valid: if key_reference.is_some() { 1.0 } else { 0.0 },
            key_mode: frame_params.key_mode,
            key_threshold: frame_params.key_threshold,
            key_softness: frame_params.key_softness,
            _pad: 0.0,
        };
        let uniform_buffer = create_uploaded_uniform(device, queue, "Temporal Uniforms", &uniforms);

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
        let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Temporal Uniform BG"),
            layout: uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
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
        if record_current {
            state.record_history_frame();
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
                        z: state.history_write as u32,
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
    } else if record_current {
        // Even with temporal processing disabled, maintain clean history so
        // enabling a key/slit-scan later starts from real prior observations.
        state.record_history_frame();
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
                    z: state.history_write as u32,
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
    state.feedback_valid = true;
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

    // Final straight-alpha -> opaque-black conversion. It writes into
    // composite slot 2 only after all engine effects have finished.
    opaque_output_pipeline: wgpu::RenderPipeline,
    opaque_output_bind_group: wgpu::BindGroup,

    // Shared sampler
    sampler: wgpu::Sampler,

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

    // Temporal effects: ring buffer of past output frames (texture array)
    temporal_pipeline: wgpu::RenderPipeline,
    temporal_bind_group_layout: wgpu::BindGroupLayout,
    temporal_uniform_layout: wgpu::BindGroupLayout,
    history_texture: wgpu::Texture,
    history_view: wgpu::TextureView,
    feedback_texture: wgpu::Texture,
    feedback_view: wgpu::TextureView,
    /// Fixed-rate history clock and validity for temporal GPU memories.
    temporal_state: TemporalState,

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

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // --- Effects pipeline (single texture + uniforms → render target) ---
        let effects_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                ],
            });

        let effects_uniform_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let vertex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vertex Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fullscreen.wgsl").into()),
        });

        let effects_fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Effects Fragment"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/effects.wgsl").into()),
        });

        let effects_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Effects Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&effects_bind_group_layout),
                    Some(&effects_uniform_layout),
                ],
                immediate_size: 0,
            });

        let effects_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Effects Pipeline"),
            layout: Some(&effects_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &effects_fragment,
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/composite.wgsl").into()),
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

        // --- Temporal pipeline + frame-history ring (shared with export) ---
        let (temporal_pipeline, temporal_bind_group_layout, temporal_uniform_layout) =
            build_temporal_pipeline(&device);
        let (history_texture, history_view) =
            build_history_texture(&device, output_width, output_height);
        let (feedback_texture, feedback_view) =
            build_feedback_texture(&device, output_width, output_height);

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
            opaque_output_pipeline,
            opaque_output_bind_group,
            sampler,
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
            temporal_pipeline,
            temporal_bind_group_layout,
            temporal_uniform_layout,
            history_texture,
            history_view,
            feedback_texture,
            feedback_view,
            temporal_state: TemporalState::default(),
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

    /// Start a new patch/visual generation without sampling any temporal or
    /// asynchronous readback data produced by the prior generation.
    ///
    /// The application must also advance its external visual epoch and clear
    /// any CPU-side NTSC-presented frame. Pending map callbacks may complete,
    /// but their sequence numbers are at or below this invalidation watermark
    /// and `poll_readback` will recycle them without returning their pixels.
    pub fn reset_visual_generation(&mut self) {
        self.temporal_state.reset();
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
    pub fn clear_composite(&self, encoder: &mut wgpu::CommandEncoder) {
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
    pub fn render_temporal_with_dt(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        params: &TemporalParams,
        delta_seconds: f32,
        advance_program: bool,
    ) {
        encode_temporal_with_dt(
            &self.device,
            &self.queue,
            encoder,
            params,
            &self.temporal_pipeline,
            &self.temporal_bind_group_layout,
            &self.temporal_uniform_layout,
            &self.sampler,
            &self.composite_textures,
            &self.composite_views,
            &self.history_texture,
            &self.history_view,
            &self.feedback_texture,
            &self.feedback_view,
            &mut self.temporal_state,
            delta_seconds,
            advance_program,
            self.output_width,
            self.output_height,
        );
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

    /// Render all layers composited together. Final result ends up in composite_views[0].
    /// `mods` is aligned with `layers` and carries each layer's modulated
    /// effect uniforms and opacity (bases stay untouched on the Layer).
    pub fn render_layers(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layers: &[Layer],
        mods: &[(EffectUniforms, f32)],
    ) {
        // Render in reverse order: last layer in the vec is the bottom,
        // first layer (index 0, "Layer 1" in UI) ends up on top.
        let visible_layers: Vec<(&Layer, &(EffectUniforms, f32))> = layers
            .iter()
            .zip(mods.iter())
            .filter(|(l, _)| l.visible)
            .rev()
            .collect();

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
            return;
        }

        for (i, (layer, (uniforms, opacity))) in visible_layers.iter().enumerate() {
            // The effects pass writes an output-sized composite even when its
            // source layer has different dimensions. Spatial effects therefore
            // need the composite resolution, not the source texture resolution.
            let uniforms = uniforms.for_render_target(self.output_width, self.output_height);

            // Each pass needs its own buffer because queue.write_buffer writes
            // all execute before the encoder's render passes run on the GPU.
            let fx_buffer =
                create_uploaded_uniform(&self.device, &self.queue, "Layer FX Uniforms", &uniforms);

            let tex_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.effects_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&layer.texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
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
                    opacity: *opacity,
                    blend_mode: if i == 0 { 0 } else { layer.blend_mode.as_u32() },
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
    }

    /// Render the complete direct-effects portion of the program.
    ///
    /// With no visible bypass this calls the two established passes verbatim:
    /// all layers composite first and the master shader runs exactly once on
    /// the result. If any visible layer bypasses master FX, every layer keeps
    /// its original stack position, local FX, opacity, and blend mode, while
    /// inherited layers receive the master shader immediately before their
    /// composite pass. Temporal and NTSC/VHS remain downstream global stages.
    pub fn render_layers_and_master(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layers: &[Layer],
        mods: &[(EffectUniforms, f32)],
        master_uniforms: &EffectUniforms,
    ) {
        let path = master_fx_composition_path(
            layers
                .iter()
                .zip(mods.iter())
                .map(|(layer, (_, opacity))| (layer.visible, layer.bypass_master_fx, *opacity)),
        );
        match path {
            MasterFxCompositionPath::LegacyPostComposite => {
                // Keep this sequence exactly equivalent to the pre-bypass
                // renderer for old patches and all-inherited performances.
                self.render_layers(encoder, layers, mods);
                self.render_master_effects(encoder, master_uniforms);
            }
            MasterFxCompositionPath::ConditionalPerLayer => {
                self.render_layers_with_conditional_master(encoder, layers, mods, master_uniforms);
            }
        }
    }

    fn render_layers_with_conditional_master(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layers: &[Layer],
        mods: &[(EffectUniforms, f32)],
        master_uniforms: &EffectUniforms,
    ) {
        // Preserve the established stack law: the last vector element is the
        // bottom and UI Layer 1 (index 0) is composited last, on top.
        let visible_layers: Vec<(&Layer, &(EffectUniforms, f32))> = visible_stack_indices(
            layers
                .iter()
                .zip(mods.iter())
                .map(|(layer, _)| layer.visible),
        )
        .into_iter()
        .map(|index| (&layers[index], &mods[index]))
        .collect();

        debug_assert!(visible_layers
            .iter()
            .any(|(layer, _)| layer.bypass_master_fx));

        // Every inherited layer reads the local-FX image from slot 1. Reuse
        // one immutable master uniform buffer and bind groups for all of them.
        let master_buffer = create_uploaded_uniform(
            &self.device,
            &self.queue,
            "Conditional Master FX Uniforms",
            master_uniforms,
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

        for (stack_index, (layer, (uniforms, opacity))) in visible_layers.iter().enumerate() {
            let uniforms = uniforms.for_render_target(self.output_width, self.output_height);
            let fx_buffer = create_uploaded_uniform(
                &self.device,
                &self.queue,
                "Conditional Layer FX Uniforms",
                &uniforms,
            );
            let layer_tex_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Conditional Layer FX Input"),
                layout: &self.effects_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&layer.texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
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
                opacity: *opacity,
                blend_mode: if stack_index == 0 {
                    0
                } else {
                    layer.blend_mode.as_u32()
                },
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
    }

    /// Apply master effects to the final composite (already in [0]).
    /// Reads [0], applies effects → [2], copies back to [0].
    pub fn render_master_effects(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        master_uniforms: &EffectUniforms,
    ) {
        let fx_buffer = create_uploaded_uniform(
            &self.device,
            &self.queue,
            "Master FX Uniforms",
            master_uniforms,
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

    fn ensure_selective_ntsc_gpu(&mut self, required_capacity: u64) {
        let needs_state = self.selective_ntsc_gpu.is_none();
        if needs_state {
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
            self.selective_ntsc_gpu = Some(SelectiveNtscGpuState {
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
            });
            return;
        }

        let state = self.selective_ntsc_gpu.as_mut().unwrap();
        if state.slot.capacity < required_capacity {
            debug_assert_eq!(state.slot.status.load(Ordering::Acquire), SLOT_IDLE);
            state.slot.buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Selective NTSC Readback Batch"),
                size: required_capacity,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            state.slot.capacity = required_capacity;
        }
    }

    /// Encode one generation-coherent selective-VHS batch. Every contributing
    /// layer is rendered through local FX and, unless bypassed, direct master
    /// FX into dedicated scratch textures. All slices are then copied into one
    /// aligned staging allocation in this same command stream.
    pub fn begin_selective_ntsc_readback(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        layers: &[Layer],
        mods: &[(EffectUniforms, f32)],
        master_uniforms: &EffectUniforms,
        plan: SelectiveNtscPlan,
    ) -> Result<bool, String> {
        self.ensure_device_healthy()?;
        if plan.generation.width != self.output_width
            || plan.generation.height != self.output_height
        {
            return Err("selective NTSC plan dimensions do not match the renderer".into());
        }
        if layers.len() != mods.len() {
            return Err("selective NTSC layer/modulation alignment mismatch".into());
        }
        if self
            .selective_ntsc_gpu
            .as_ref()
            .is_some_and(|state| state.slot.status.load(Ordering::Acquire) != SLOT_IDLE)
        {
            return Ok(false);
        }

        let (padded_row_bytes, slice_stride, used_size) =
            self.selective_batch_layout(plan.layers.len())?;
        self.ensure_selective_ntsc_gpu(used_size);

        let state = self.selective_ntsc_gpu.as_ref().unwrap();
        let master_buffer = create_uploaded_uniform(
            &self.device,
            &self.queue,
            "Selective NTSC Master FX Uniforms",
            master_uniforms,
        );
        let master_tex_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Selective NTSC Master FX Input"),
            layout: &self.effects_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&state.scratch_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        let master_uniform_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Selective NTSC Master FX Uniforms BG"),
            layout: &self.effects_uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: master_buffer.as_entire_binding(),
            }],
        });

        for (slice_index, planned_layer) in plan.layers.iter().enumerate() {
            let source_index = layers
                .iter()
                .position(|layer| layer.layer_id() == planned_layer.layer_id)
                .ok_or_else(|| {
                    format!(
                        "selective NTSC layer {} disappeared before encoding",
                        planned_layer.layer_id
                    )
                })?;
            let layer = &layers[source_index];
            if !layer.visible || layer.bypass_master_fx != planned_layer.bypass_master_fx {
                return Err(format!(
                    "selective NTSC layer {} changed before encoding",
                    planned_layer.layer_id
                ));
            }
            let uniforms = mods[source_index]
                .0
                .for_render_target(self.output_width, self.output_height);
            let fx_buffer = create_uploaded_uniform(
                &self.device,
                &self.queue,
                "Selective NTSC Layer FX Uniforms",
                &uniforms,
            );
            let layer_tex_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Selective NTSC Layer FX Input"),
                layout: &self.effects_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&layer.texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            let layer_uniform_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Selective NTSC Layer FX Uniforms BG"),
                layout: &self.effects_uniform_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: fx_buffer.as_entire_binding(),
                }],
            });

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
                pass.set_bind_group(0, &layer_tex_bg, &[]);
                pass.set_bind_group(1, &layer_uniform_bg, &[]);
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
                pass.set_bind_group(0, &master_tex_bg, &[]);
                pass.set_bind_group(1, &master_uniform_bg, &[]);
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
    /// slot still has a copy in flight.
    pub fn begin_readback(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        epoch: u64,
        ntsc_metadata: Option<NtscFrameMetadata>,
    ) -> Option<usize> {
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
    ) -> Option<usize> {
        self.begin_readback_tagged(encoder, sample.visual_epoch, None, Some(sample), false)
    }

    /// Read back the already materialized audience image retained by Pause.
    /// The explicit tag prevents it from being mistaken for raw global-NTSC
    /// input or for a newly processed selective generation.
    pub fn begin_held_audience_readback(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        epoch: u64,
    ) -> Option<usize> {
        self.begin_readback_tagged(encoder, epoch, None, None, true)
    }

    fn begin_readback_tagged(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        epoch: u64,
        ntsc_metadata: Option<NtscFrameMetadata>,
        selective_sample: Option<SelectiveNtscGeneration>,
        held_audience: bool,
    ) -> Option<usize> {
        if self.ensure_device_healthy().is_err() {
            return None;
        }
        let buffer_size = (self.readback_bytes_per_row() * self.output_height) as u64;

        let idx = match self
            .readback_slots
            .iter()
            .position(|s| s.status.load(Ordering::Acquire) == SLOT_IDLE)
        {
            Some(idx) => idx,
            None if self.readback_slots.len() < MAX_READBACK_SLOTS => {
                self.readback_slots.push(ReadbackSlot {
                    buffer: self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("NTSC Readback Slot"),
                        size: buffer_size,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    }),
                    status: Arc::new(AtomicU8::new(SLOT_IDLE)),
                    sequence: 0,
                    epoch,
                    ntsc_metadata: None,
                    selective_sample: None,
                    held_audience: false,
                });
                self.readback_slots.len() - 1
            }
            None => return None,
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
                    bytes_per_row: Some(self.readback_bytes_per_row()),
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
        Some(idx)
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
