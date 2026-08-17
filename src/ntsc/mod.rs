//! ntsc-rs VHS effect wrapper.
//!
//! Applies analog VHS effects (head switching, tracking noise, snow, etc.)
//! as a CPU-based post-process on the final composite RGBA buffer.
//!
//! For live rendering, processing runs on a dedicated worker thread
//! (`NtscWorker`) fed by async GPU readbacks, so the render loop never
//! blocks on the CPU-bound effect. The displayed NTSC output is a coherent,
//! bounded delayed sample; exact latency depends on resolution, layer count,
//! effect settings, and CPU/GPU throughput.

use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, TrySendError};

use ntsc_rs::settings::standard::*;
use ntsc_rs::yiq_fielding::Rgbx;
use ntsc_rs::{Context, NtscEffect};

/// User-facing VHS parameters (mirrored in the web UI).
#[derive(Debug, Clone, PartialEq)]
pub struct NtscParams {
    pub enabled: bool,

    // VHS tape settings
    pub tape_speed: u32, // 0=SP, 1=LP, 2=EP
    pub chroma_loss: f32,

    // Edge wave
    pub edge_wave_enabled: bool,
    pub edge_wave_intensity: f32,
    pub edge_wave_speed: f32,

    // Head switching
    pub head_switching_enabled: bool,
    pub head_switching_height: i32,
    pub head_switching_shift: f32,

    // Tracking noise
    pub tracking_noise_enabled: bool,
    pub tracking_noise_height: i32,
    pub tracking_noise_wave: f32,
    pub tracking_noise_snow: f32,

    // Snow
    pub snow_intensity: f32,

    // Noise
    pub composite_noise_intensity: f32,
    pub luma_noise_intensity: f32,
    pub chroma_noise_intensity: f32,

    // Post-process
    pub luma_smear: f32,
    pub composite_sharpening: f32,
}

/// NTSC animation advances against a fixed reference clock, independent of
/// live readback latency or export frame rate.
pub const NTSC_REFERENCE_FPS: f64 = 30.0;

/// Spatial resolution used by the CPU NTSC processor.
///
/// Live rendering intentionally keeps the historical half-resolution path for
/// bounded latency. Offline export may opt into native resolution when visual
/// fidelity is more important than render time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NtscExportQuality {
    #[default]
    LiveParity,
    Native,
}

/// Effect state sampled alongside one raw composite readback.
///
/// Keeping this metadata attached to the pixels prevents a delayed GPU map
/// from being processed with parameters sampled from a newer render frame.
#[derive(Debug, Clone, PartialEq)]
pub struct NtscFrameMetadata {
    pub params: NtscParams,
    pub reference_frame: usize,
}

/// Identity of one selective-VHS render generation.
///
/// `visual_epoch` changes across patch recall, blackout, and post-process mode
/// edges. `topology_generation` changes whenever the live layer stack or an
/// authored selective-control semantic is changed. Dimensions are included
/// because CPU results must never be uploaded into a differently sized render
/// target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectiveNtscGeneration {
    pub visual_epoch: u64,
    pub topology_generation: u64,
    pub width: u32,
    pub height: u32,
    /// Monotonic visual sample sequence. Selective processing intentionally
    /// trails live input; this distinguishes successive valid frames without
    /// treating them as topology changes.
    pub sample_sequence: u64,
}

/// Immutable facts needed to plan one source layer. Descriptors arrive in UI
/// order (top to bottom); the planner returns contributing layers in actual
/// compositor order (bottom to top).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelectiveNtscLayerDescriptor {
    pub layer_id: u64,
    pub visible: bool,
    pub bypass_master_fx: bool,
    pub opacity: f32,
    pub blend_mode: u32,
    pub transform_fingerprint: u64,
}

/// One contributing layer in a selective-VHS batch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelectiveNtscLayerPlan {
    pub layer_id: u64,
    pub bypass_master_fx: bool,
    pub opacity: f32,
    pub blend_mode: u32,
    /// Caller-owned semantic fingerprint. Pixel processing does not interpret
    /// it. Live uses a stable source-identity projection (never frame-varying
    /// time or modulation); synchronous export may retain an exact transform
    /// digest.
    pub transform_fingerprint: u64,
}

/// Pure, serializable-in-memory plan shared by live rendering and export.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectiveNtscPlan {
    pub generation: SelectiveNtscGeneration,
    pub metadata: NtscFrameMetadata,
    pub layers: Vec<SelectiveNtscLayerPlan>,
}

/// Tight RGBA slices aligned with [`SelectiveNtscPlan::layers`]. The renderer
/// removes GPU row padding before constructing this value.
#[derive(Debug)]
pub struct SelectiveNtscBatch {
    pub plan: SelectiveNtscPlan,
    pub slices: Vec<Vec<u8>>,
}

/// Finished straight-alpha, sRGB-encoded composite. Temporal processing and
/// the final opaque resolve deliberately remain downstream on the GPU.
#[derive(Debug)]
pub struct SelectiveNtscProcessedFrame {
    pub pixels: Vec<u8>,
    pub plan: SelectiveNtscPlan,
}

/// Upper bound for the incremental live selective-VHS payload working set.
///
/// The renderer already owns its normal composite/history textures and one
/// baseline audience-hold texture used by Pause/blackout even when selective
/// VHS is disabled. This cap covers every *incremental selective-VHS* payload
/// that can be retained concurrently: the aligned GPU staging batch, two GPU
/// scratch images, the tight host batch owned by the worker, its output
/// composite, and the largest per-slice NTSC conversion workspace (saved
/// alpha, half-resolution RGBA, and half-resolution planar YIQ). Keeping the
/// cap here makes preflight, renderer allocation, and tests share one law
/// without charging global transport safety to this mode.
pub const MAX_SELECTIVE_NTSC_LIVE_BYTES: u64 = 320 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectiveNtscLiveMemory {
    pub gpu_staging_bytes: u64,
    pub gpu_scratch_bytes: u64,
    pub host_batch_bytes: u64,
    pub worker_output_bytes: u64,
    pub ntsc_alpha_bytes: u64,
    pub ntsc_half_rgba_bytes: u64,
    pub ntsc_yiq_bytes: u64,
    pub total_bytes: u64,
    pub padded_row_bytes: u32,
    pub slice_stride: u64,
}

/// Checked incremental live memory footprint for one selective generation.
pub fn estimate_selective_ntsc_live_memory(
    width: u32,
    height: u32,
    layer_count: usize,
) -> Result<SelectiveNtscLiveMemory, String> {
    if width == 0 || height == 0 || layer_count == 0 {
        return Err("selective VHS requires non-zero dimensions and at least one layer".into());
    }
    let row_bytes = width
        .checked_mul(4)
        .ok_or_else(|| "selective VHS row size overflow".to_string())?;
    let padded_row_bytes = row_bytes
        .checked_add(255)
        .map(|bytes| bytes & !255)
        .ok_or_else(|| "selective VHS aligned row size overflow".to_string())?;
    let slice_stride = (padded_row_bytes as u64)
        .checked_mul(height as u64)
        .ok_or_else(|| "selective VHS slice size overflow".to_string())?;
    let layer_count =
        u64::try_from(layer_count).map_err(|_| "selective VHS layer count overflow".to_string())?;
    let tight_frame_bytes = (row_bytes as u64)
        .checked_mul(height as u64)
        .ok_or_else(|| "selective VHS tight frame size overflow".to_string())?;
    let gpu_staging_bytes = slice_stride
        .checked_mul(layer_count)
        .ok_or_else(|| "selective VHS GPU staging size overflow".to_string())?;
    let gpu_scratch_bytes = tight_frame_bytes
        .checked_mul(2)
        .ok_or_else(|| "selective VHS GPU scratch size overflow".to_string())?;
    let host_batch_bytes = tight_frame_bytes
        .checked_mul(layer_count)
        .ok_or_else(|| "selective VHS host batch size overflow".to_string())?;
    let worker_output_bytes = tight_frame_bytes;
    // NtscState processes inherited slices sequentially, so only one slice's
    // conversion workspace is live at once. Alpha is preserved at full
    // resolution. The RGBX buffer is ceil-half-sized, while ntsc-rs's default
    // interleaved field stores four f32 planes for every half-resolution pixel.
    let ntsc_alpha_bytes = (width as u64)
        .checked_mul(height as u64)
        .ok_or_else(|| "selective VHS alpha workspace size overflow".to_string())?;
    let half_width = u64::from(width.div_ceil(2));
    let half_height = u64::from(height.div_ceil(2));
    let half_pixels = half_width
        .checked_mul(half_height)
        .ok_or_else(|| "selective VHS half-resolution size overflow".to_string())?;
    let ntsc_half_rgba_bytes = half_pixels
        .checked_mul(4)
        .ok_or_else(|| "selective VHS half-RGBA workspace size overflow".to_string())?;
    let ntsc_yiq_bytes = half_pixels
        .checked_mul(4)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>() as u64))
        .ok_or_else(|| "selective VHS YIQ workspace size overflow".to_string())?;
    let total_bytes = gpu_staging_bytes
        .checked_add(gpu_scratch_bytes)
        .and_then(|bytes| bytes.checked_add(host_batch_bytes))
        .and_then(|bytes| bytes.checked_add(worker_output_bytes))
        .and_then(|bytes| bytes.checked_add(ntsc_alpha_bytes))
        .and_then(|bytes| bytes.checked_add(ntsc_half_rgba_bytes))
        .and_then(|bytes| bytes.checked_add(ntsc_yiq_bytes))
        .ok_or_else(|| "selective VHS aggregate working-set size overflow".to_string())?;
    Ok(SelectiveNtscLiveMemory {
        gpu_staging_bytes,
        gpu_scratch_bytes,
        host_batch_bytes,
        worker_output_bytes,
        ntsc_alpha_bytes,
        ntsc_half_rgba_bytes,
        ntsc_yiq_bytes,
        total_bytes,
        padded_row_bytes,
        slice_stride,
    })
}

/// Enforce the live safety budget before allocating or clearing the audience.
pub fn validate_selective_ntsc_live_memory(
    width: u32,
    height: u32,
    layer_count: usize,
) -> Result<SelectiveNtscLiveMemory, String> {
    let memory = estimate_selective_ntsc_live_memory(width, height, layer_count)?;
    if memory.total_bytes > MAX_SELECTIVE_NTSC_LIVE_BYTES {
        let mib_ceil = |bytes: u64| bytes.saturating_add((1 << 20) - 1) >> 20;
        return Err(format!(
            "Selective VHS needs about {} MiB for {layer_count} contributing layers at {width}x{height}; the live safety budget is {} MiB. Hide zero-priority layers, reduce their opacity to zero, or use a lower output resolution. The prior audience frame is being held.",
            mib_ceil(memory.total_bytes),
            mib_ceil(MAX_SELECTIVE_NTSC_LIVE_BYTES),
        ));
    }
    Ok(memory)
}

/// Check the one contiguous GPU staging allocation against the actual device
/// limit. The aggregate host/GPU budget and the adapter limit are independent
/// constraints, so neither is allowed to stand in for the other.
pub fn validate_selective_ntsc_gpu_staging_limit(
    memory: SelectiveNtscLiveMemory,
    max_buffer_size: u64,
) -> Result<SelectiveNtscLiveMemory, String> {
    if memory.gpu_staging_bytes > max_buffer_size {
        let mib_ceil = |bytes: u64| bytes.saturating_add((1 << 20) - 1) >> 20;
        return Err(format!(
            "Selective VHS needs a {} MiB GPU staging buffer, but this graphics device permits at most {} MiB. Hide zero-priority layers or use a lower output resolution. The prior audience frame is being held.",
            mib_ceil(memory.gpu_staging_bytes),
            mib_ceil(max_buffer_size),
        ));
    }
    Ok(memory)
}

/// Central stale-result gate used by live presentation and available to
/// offline orchestration. Every component must match; accepting on epoch alone
/// could put a pre-reorder slice back under the wrong layer semantics.
#[cfg(test)]
pub fn selective_generation_is_current(
    completed: SelectiveNtscGeneration,
    current: SelectiveNtscGeneration,
) -> bool {
    completed == current
}

/// Stale-topology/control gate for a delayed live frame. The sample sequence
/// is allowed to trail; all structural axes must still match.
pub fn selective_generation_compatible(
    completed: SelectiveNtscGeneration,
    current: SelectiveNtscGeneration,
) -> bool {
    completed.visual_epoch == current.visual_epoch
        && completed.topology_generation == current.topology_generation
        && completed.width == current.width
        && completed.height == current.height
        && completed.sample_sequence <= current.sample_sequence
}

/// Accept a delayed live plan only within the same structural generation.
/// Continuously sampled values (program time; sliders/morph; LFO, audio, MIDI,
/// gyro; frame-local opacity/effect/NTSC/blend values) remain attached to their
/// coherent pixel batch and may trail. Requiring those values to equal the
/// next render frame would starve an asynchronous pipeline. This comparison
/// instead verifies the immutable ordered layer/routing projection.
pub fn selective_plan_compatible(
    completed: &SelectiveNtscPlan,
    current: &SelectiveNtscPlan,
) -> bool {
    selective_generation_compatible(completed.generation, current.generation)
        && completed.layers.len() == current.layers.len()
        && completed.layers.iter().zip(&current.layers).all(|(a, b)| {
            a.layer_id == b.layer_id
                && a.bypass_master_fx == b.bypass_master_fx
                && a.transform_fingerprint == b.transform_fingerprint
        })
}

/// True only when a visible bypass layer can contribute pixels. This is the
/// exact switch that protects the established global post-composite VHS path
/// for every pre-existing patch and for zero-opacity/hidden bypass layers.
pub fn selective_ntsc_required<I>(layers: I) -> bool
where
    I: IntoIterator<Item = SelectiveNtscLayerDescriptor>,
{
    layers.into_iter().any(|layer| {
        layer.visible && layer.bypass_master_fx && layer.opacity.is_finite() && layer.opacity > 0.0
    })
}

/// Build the exact bottom-to-top compositor plan for selective VHS.
///
/// Invisible and finite non-positive-opacity layers cannot affect the current
/// straight-alpha compositor and are omitted, keeping readback memory bounded.
/// The bottom contributing layer uses Normal just like the GPU path; its blend
/// mode is immaterial over transparent black, but making the law explicit
/// prevents live/export drift.
pub fn plan_selective_ntsc<I>(
    generation: SelectiveNtscGeneration,
    metadata: NtscFrameMetadata,
    layers: I,
) -> Option<SelectiveNtscPlan>
where
    I: IntoIterator<Item = SelectiveNtscLayerDescriptor>,
{
    if generation.width == 0 || generation.height == 0 || !metadata.params.enabled {
        return None;
    }
    let descriptors: Vec<_> = layers.into_iter().collect();
    if !selective_ntsc_required(descriptors.iter().copied()) {
        return None;
    }

    let mut planned: Vec<_> = descriptors
        .into_iter()
        .filter(|layer| layer.visible && layer.opacity.is_finite() && layer.opacity > 0.0)
        .rev()
        .map(|layer| SelectiveNtscLayerPlan {
            layer_id: layer.layer_id,
            bypass_master_fx: layer.bypass_master_fx,
            opacity: layer.opacity.clamp(0.0, 1.0),
            blend_mode: layer.blend_mode.min(3),
            transform_fingerprint: layer.transform_fingerprint,
        })
        .collect();
    if let Some(bottom) = planned.first_mut() {
        bottom.blend_mode = 0;
    }
    Some(SelectiveNtscPlan {
        generation,
        metadata,
        layers: planned,
    })
}

pub(crate) fn checked_rgba_frame_len(width: u32, height: u32) -> Option<usize> {
    (width as usize)
        .checked_mul(height as usize)?
        .checked_mul(4)
}

/// Map piece-local seconds to the shared 30 Hz NTSC phase clock.
pub fn reference_frame_for_time(seconds: f64) -> usize {
    if !seconds.is_finite() || seconds <= 0.0 {
        return 0;
    }
    (seconds * NTSC_REFERENCE_FPS)
        .floor()
        .min(usize::MAX as f64) as usize
}

/// Exact integer mapping for an offline output frame. This avoids floating
/// point boundary drift at rates such as 30 and 60 fps.
pub fn reference_frame_for_output(frame_index: u64, output_fps: u32) -> usize {
    let frame = frame_index as u128 * NTSC_REFERENCE_FPS as u128 / output_fps.max(1) as u128;
    frame.min(usize::MAX as u128) as usize
}

impl Default for NtscParams {
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

/// Holds ntsc-rs processing state.
pub struct NtscState {
    ctx: Context,
    effect: NtscEffect,
    pub params: NtscParams,
}

impl NtscState {
    pub fn new() -> Self {
        Self {
            ctx: Context::new(),
            effect: NtscEffect::default(),
            params: NtscParams::default(),
        }
    }

    /// Apply one explicitly phased frame at half resolution. Live and offline paths use this
    /// entry point so dropped live jobs and differing export FPS values cannot
    /// change the wall-time speed of NTSC noise and scan motion.
    pub fn apply_at_reference_frame(
        &mut self,
        pixels: &mut [u8],
        width: u32,
        height: u32,
        reference_frame: usize,
    ) -> bool {
        self.apply_at_reference_frame_with_resolution(
            pixels,
            width,
            height,
            reference_frame,
            NtscExportQuality::LiveParity,
        )
    }

    /// Apply one explicitly phased frame at the requested export quality.
    /// Native processing is intentionally opt-in because its CPU and memory
    /// cost scale with the full output dimensions.
    pub fn apply_at_reference_frame_with_resolution(
        &mut self,
        pixels: &mut [u8],
        width: u32,
        height: u32,
        reference_frame: usize,
        quality: NtscExportQuality,
    ) -> bool {
        if !self.params.enabled {
            return false;
        }

        let w = width as usize;
        let h = height as usize;
        let Some(expected_len) = w.checked_mul(h).and_then(|n| n.checked_mul(4)) else {
            log::warn!("NTSC frame dimensions overflow: {width}x{height}");
            return false;
        };
        if w == 0 || h == 0 || pixels.len() < expected_len {
            log::warn!(
                "NTSC frame buffer is invalid: {width}x{height}, {} bytes",
                pixels.len()
            );
            return false;
        }

        self.sync_effect_from_params();

        // ntsc-rs consumes RGBX and is free to rewrite the fourth channel.
        // Alpha is application coverage (luma/cellular keying), not video
        // color, so retain it bit-for-bit across this RGB-only post-process.
        let source_alpha: Vec<u8> = pixels[..expected_len]
            .chunks_exact(4)
            .map(|pixel| pixel[3])
            .collect();

        match quality {
            NtscExportQuality::LiveParity => {
                // Ceil division keeps a real sample for an odd final row/column.
                let half_w = w.div_ceil(2);
                let half_h = h.div_ceil(2);
                let mut small = downscale_rgba_2x(pixels, w, h);

                self.effect.apply_effect_to_buffer::<Rgbx, u8>(
                    &self.ctx,
                    (half_w, half_h),
                    &mut small,
                    reference_frame,
                    [1.0, 1.0],
                );

                // Preserve the established live look and cost profile.
                upscale_rgba_2x(&small, half_w, pixels, w, h);
            }
            NtscExportQuality::Native => {
                self.effect.apply_effect_to_buffer::<Rgbx, u8>(
                    &self.ctx,
                    (w, h),
                    &mut pixels[..expected_len],
                    reference_frame,
                    [1.0, 1.0],
                );
            }
        }
        restore_alpha(pixels, &source_alpha);

        true
    }

    /// Sync the ntsc-rs NtscEffect struct from our user-facing params.
    fn sync_effect_from_params(&mut self) {
        let p = &self.params;

        // VHS settings
        self.effect.vhs_settings.enabled = true;
        self.effect.vhs_settings.settings.tape_speed = match p.tape_speed {
            1 => VHSTapeSpeed::LP,
            2 => VHSTapeSpeed::EP,
            _ => VHSTapeSpeed::SP,
        };
        self.effect.vhs_settings.settings.chroma_loss = p.chroma_loss;

        // Edge wave
        self.effect.vhs_settings.settings.edge_wave.enabled = p.edge_wave_enabled;
        self.effect
            .vhs_settings
            .settings
            .edge_wave
            .settings
            .intensity = p.edge_wave_intensity;
        self.effect.vhs_settings.settings.edge_wave.settings.speed = p.edge_wave_speed;

        // Head switching
        self.effect.head_switching.enabled = p.head_switching_enabled;
        self.effect.head_switching.settings.height = p.head_switching_height;
        self.effect.head_switching.settings.horiz_shift = p.head_switching_shift;

        // Tracking noise
        self.effect.tracking_noise.enabled = p.tracking_noise_enabled;
        self.effect.tracking_noise.settings.height = p.tracking_noise_height;
        self.effect.tracking_noise.settings.wave_intensity = p.tracking_noise_wave;
        self.effect.tracking_noise.settings.snow_intensity = p.tracking_noise_snow;

        // Snow
        self.effect.snow_intensity = p.snow_intensity;

        // Noise
        self.effect.composite_noise.enabled = p.composite_noise_intensity > 0.0;
        self.effect.composite_noise.settings.intensity = p.composite_noise_intensity;
        self.effect.luma_noise.enabled = p.luma_noise_intensity > 0.0;
        self.effect.luma_noise.settings.intensity = p.luma_noise_intensity;
        self.effect.chroma_noise.enabled = p.chroma_noise_intensity > 0.0;
        self.effect.chroma_noise.settings.intensity = p.chroma_noise_intensity;

        // Post-process
        self.effect.luma_smear = p.luma_smear;
        self.effect.composite_sharpening = p.composite_sharpening;
    }
}

fn restore_alpha(pixels: &mut [u8], alpha: &[u8]) {
    for (pixel, source_alpha) in pixels.chunks_exact_mut(4).zip(alpha.iter().copied()) {
        pixel[3] = source_alpha;
    }
}

/// Box-filter a frame to ceil(width / 2) × ceil(height / 2). At odd edges,
/// clamp the missing neighbour to the last real pixel rather than indexing
/// beyond the source buffer.
fn downscale_rgba_2x(pixels: &[u8], width: usize, height: usize) -> Vec<u8> {
    let out_w = width.div_ceil(2);
    let out_h = height.div_ceil(2);
    let mut out = vec![0u8; out_w * out_h * 4];

    for oy in 0..out_h {
        let y0 = oy * 2;
        let y1 = (y0 + 1).min(height - 1);
        for ox in 0..out_w {
            let x0 = ox * 2;
            let x1 = (x0 + 1).min(width - 1);
            let dst = (oy * out_w + ox) * 4;
            let src = [
                (y0 * width + x0) * 4,
                (y0 * width + x1) * 4,
                (y1 * width + x0) * 4,
                (y1 * width + x1) * 4,
            ];
            for channel in 0..4 {
                let sum: u32 = src
                    .iter()
                    .map(|&index| pixels[index + channel] as u32)
                    .sum();
                out[dst + channel] = (sum / 4) as u8;
            }
        }
    }
    out
}

fn upscale_rgba_2x(
    small: &[u8],
    small_width: usize,
    pixels: &mut [u8],
    width: usize,
    height: usize,
) {
    for y in 0..height {
        for x in 0..width {
            let src = ((y / 2) * small_width + x / 2) * 4;
            let dst = (y * width + x) * 4;
            pixels[dst..dst + 4].copy_from_slice(&small[src..src + 4]);
        }
    }
}

impl NtscParams {
    /// Apply a named parameter from a JSON value.
    pub fn set_param(&mut self, param: &str, value: &serde_json::Value) {
        let finite = |fallback: f32| {
            value
                .as_f64()
                .map(|number| number as f32)
                .filter(|number| number.is_finite())
                .unwrap_or(fallback)
        };
        match param {
            "enabled" => {
                if let Some(b) = value.as_bool() {
                    self.enabled = b;
                }
            }
            "tape_speed" => {
                if let Some(n) = value.as_u64() {
                    self.tape_speed = n.min(2) as u32;
                }
            }
            "chroma_loss" => {
                self.chroma_loss = finite(self.chroma_loss).clamp(0.0, 0.01);
            }
            "edge_wave_enabled" => {
                if let Some(b) = value.as_bool() {
                    self.edge_wave_enabled = b;
                }
            }
            "edge_wave_intensity" => {
                self.edge_wave_intensity = finite(self.edge_wave_intensity).clamp(0.0, 20.0);
            }
            "edge_wave_speed" => {
                self.edge_wave_speed = finite(self.edge_wave_speed).clamp(0.0, 10.0);
            }
            "head_switching_enabled" => {
                if let Some(b) = value.as_bool() {
                    self.head_switching_enabled = b;
                }
            }
            "head_switching_height" => {
                if let Some(n) = value.as_i64() {
                    self.head_switching_height = n.clamp(0, 24) as i32;
                }
            }
            "head_switching_shift" => {
                self.head_switching_shift = finite(self.head_switching_shift).clamp(-100.0, 100.0);
            }
            "tracking_noise_enabled" => {
                if let Some(b) = value.as_bool() {
                    self.tracking_noise_enabled = b;
                }
            }
            "tracking_noise_height" => {
                if let Some(n) = value.as_i64() {
                    self.tracking_noise_height = n.clamp(0, 120) as i32;
                }
            }
            "tracking_noise_wave" => {
                self.tracking_noise_wave = finite(self.tracking_noise_wave).clamp(0.0, 50.0);
            }
            "tracking_noise_snow" => {
                self.tracking_noise_snow = finite(self.tracking_noise_snow).clamp(0.0, 1.0);
            }
            "snow_intensity" => {
                self.snow_intensity = finite(self.snow_intensity).clamp(0.0, 1.0);
            }
            "composite_noise_intensity" => {
                self.composite_noise_intensity =
                    finite(self.composite_noise_intensity).clamp(0.0, 0.5);
            }
            "luma_noise_intensity" => {
                self.luma_noise_intensity = finite(self.luma_noise_intensity).clamp(0.0, 0.2);
            }
            "chroma_noise_intensity" => {
                self.chroma_noise_intensity = finite(self.chroma_noise_intensity).clamp(0.0, 0.5);
            }
            "luma_smear" => {
                self.luma_smear = finite(self.luma_smear).clamp(0.0, 1.0);
            }
            "composite_sharpening" => {
                self.composite_sharpening = finite(self.composite_sharpening).clamp(-1.0, 2.0);
            }
            _ => {}
        }
    }
}

fn srgb_byte_to_linear(value: u8) -> f32 {
    let encoded = value as f32 / 255.0;
    if encoded <= 0.04045 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb_byte(value: f32) -> u8 {
    let linear = value.clamp(0.0, 1.0);
    let encoded = if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0).round().clamp(0.0, 255.0) as u8
}

fn linear_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round().clamp(0.0, 255.0) as u8
}

/// CPU mirror of `shaders/composite.wgsl` for already output-sized slices.
///
/// Each layer pass decodes sRGB to linear light, executes the shader's exact
/// straight-alpha blend law, and quantizes back to RGBA8 sRGB before the next
/// layer. Quantizing between layers is important: the GPU path writes an
/// Rgba8UnormSrgb target after every pass rather than accumulating in float.
pub(crate) fn composite_selective_ntsc_layers(
    plan: &SelectiveNtscPlan,
    slices: &[Vec<u8>],
) -> Result<Vec<u8>, String> {
    let expected = checked_rgba_frame_len(plan.generation.width, plan.generation.height)
        .ok_or_else(|| "selective NTSC dimensions overflow".to_string())?;
    if slices.len() != plan.layers.len() {
        return Err(format!(
            "selective NTSC slice count mismatch: expected {}, got {}",
            plan.layers.len(),
            slices.len()
        ));
    }
    if let Some((index, slice)) = slices
        .iter()
        .enumerate()
        .find(|(_, slice)| slice.len() != expected)
    {
        return Err(format!(
            "selective NTSC layer {index} has {} bytes; expected {expected}",
            slice.len()
        ));
    }

    if plan.layers.len() == 1
        && plan.layers[0].bypass_master_fx
        && plan.layers[0].opacity >= 1.0
        && plan.layers[0].blend_mode == 0
    {
        // One fully opaque bypass layer over transparent black is an exact
        // identity operation in the GPU compositor, including RGB stored
        // under zero alpha. Avoid all float/transfer-function round trips.
        return Ok(slices[0].clone());
    }

    let mut base = vec![0u8; expected];
    for (layer, overlay) in plan.layers.iter().zip(slices) {
        let opacity = layer.opacity.clamp(0.0, 1.0);
        for (base_pixel, overlay_pixel) in base.chunks_exact_mut(4).zip(overlay.chunks_exact(4)) {
            let source_alpha = opacity * (overlay_pixel[3] as f32 / 255.0);
            if base_pixel[3] == 0 && source_alpha >= 1.0 {
                // Exact GPU Normal-over-transparent result, including the
                // otherwise unobservable RGB carried by alpha-zero texels.
                // Preserving these bytes is important for a fully bypassed
                // slice and avoids a needless transfer-function round trip.
                base_pixel.copy_from_slice(overlay_pixel);
                continue;
            }
            let base_rgb = [
                srgb_byte_to_linear(base_pixel[0]),
                srgb_byte_to_linear(base_pixel[1]),
                srgb_byte_to_linear(base_pixel[2]),
            ];
            let overlay_rgb = [
                srgb_byte_to_linear(overlay_pixel[0]),
                srgb_byte_to_linear(overlay_pixel[1]),
                srgb_byte_to_linear(overlay_pixel[2]),
            ];
            let base_alpha = base_pixel[3] as f32 / 255.0;
            let output_alpha = source_alpha + base_alpha * (1.0 - source_alpha);

            let mut output_rgb = [0.0f32; 3];
            for channel in 0..3 {
                let blended = match layer.blend_mode {
                    1 => 1.0 - (1.0 - base_rgb[channel]) * (1.0 - overlay_rgb[channel]),
                    2 => base_rgb[channel] * overlay_rgb[channel],
                    3 => (base_rgb[channel] - overlay_rgb[channel]).abs(),
                    _ => overlay_rgb[channel],
                };
                let premultiplied = base_rgb[channel] * base_alpha * (1.0 - source_alpha)
                    + overlay_rgb[channel] * source_alpha * (1.0 - base_alpha)
                    + blended * base_alpha * source_alpha;
                output_rgb[channel] = if output_alpha > 0.000_001 {
                    premultiplied / output_alpha.max(0.000_001)
                } else {
                    0.0
                };
            }

            base_pixel[0] = linear_to_srgb_byte(output_rgb[0]);
            base_pixel[1] = linear_to_srgb_byte(output_rgb[1]);
            base_pixel[2] = linear_to_srgb_byte(output_rgb[2]);
            base_pixel[3] = linear_byte(output_alpha);
        }
    }
    Ok(base)
}

/// Apply VHS only to inherited slices, then execute the shared exact CPU
/// compositor. Bypassed slice allocations are borrowed and never mutated.
/// Export may call this synchronously with its own persistent [`NtscState`].
pub(crate) fn process_selective_ntsc_batch_with_state(
    state: &mut NtscState,
    batch: SelectiveNtscBatch,
) -> Result<SelectiveNtscProcessedFrame, String> {
    process_selective_ntsc_batch_with_state_and_resolution(
        state,
        batch,
        NtscExportQuality::LiveParity,
    )
}

/// Resolution-selectable form used by offline export. The live worker calls
/// the compatibility wrapper above and therefore always remains half-size.
pub(crate) fn process_selective_ntsc_batch_with_state_and_resolution(
    state: &mut NtscState,
    batch: SelectiveNtscBatch,
    quality: NtscExportQuality,
) -> Result<SelectiveNtscProcessedFrame, String> {
    let expected =
        checked_rgba_frame_len(batch.plan.generation.width, batch.plan.generation.height)
            .ok_or_else(|| "selective NTSC dimensions overflow".to_string())?;
    if batch.slices.len() != batch.plan.layers.len() {
        return Err(format!(
            "selective NTSC slice count mismatch: expected {}, got {}",
            batch.plan.layers.len(),
            batch.slices.len()
        ));
    }

    state.params = batch.plan.metadata.params.clone();
    let mut processed = Vec::with_capacity(batch.slices.len());
    for (index, (layer, mut slice)) in batch.plan.layers.iter().zip(batch.slices).enumerate() {
        if slice.len() != expected {
            return Err(format!(
                "selective NTSC layer {index} has {} bytes; expected {expected}",
                slice.len()
            ));
        }
        if !layer.bypass_master_fx
            && !state.apply_at_reference_frame_with_resolution(
                &mut slice,
                batch.plan.generation.width,
                batch.plan.generation.height,
                batch.plan.metadata.reference_frame,
                quality,
            )
        {
            return Err(format!(
                "NTSC rejected selective layer {index} at {}x{}",
                batch.plan.generation.width, batch.plan.generation.height
            ));
        }
        processed.push(slice);
    }

    let pixels = composite_selective_ntsc_layers(&batch.plan, &processed)?;
    Ok(SelectiveNtscProcessedFrame {
        pixels,
        plan: batch.plan,
    })
}

/// Convenience entry point for deterministic callers that do not retain a
/// processor. Live rendering uses [`SelectiveNtscWorker`] to stay non-blocking.
#[cfg(test)]
pub(crate) fn process_selective_ntsc_batch(
    batch: SelectiveNtscBatch,
) -> Result<SelectiveNtscProcessedFrame, String> {
    process_selective_ntsc_batch_with_state(&mut NtscState::new(), batch)
}

/// A frame of work for the NTSC worker thread.
struct NtscJob {
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    metadata: NtscFrameMetadata,
    epoch: u64,
}

/// A processed frame tagged with the visual generation from which it came.
/// The caller must discard it when that generation is no longer current.
pub struct NtscProcessedFrame {
    pub pixels: Vec<u8>,
    pub epoch: u64,
    #[cfg(test)]
    pub metadata: NtscFrameMetadata,
}

/// Result of a non-blocking admission attempt into either live NTSC worker.
///
/// `Busy` is bounded backpressure: the worker is healthy but already owns a
/// sample. `Unavailable` is terminal for the current worker instance and is
/// kept separate so diagnostics never misreport a failed/disconnected worker
/// as ordinary performance shedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use = "NTSC admission outcomes distinguish performance backpressure from worker failure"]
pub enum NtscSubmitOutcome {
    Accepted,
    Busy,
    Unavailable,
}

impl NtscSubmitOutcome {
    pub const fn is_accepted(self) -> bool {
        matches!(self, Self::Accepted)
    }
}

/// Saturating counters for one live NTSC path.
///
/// `attempted`, `accepted`, `skipped`, and `unavailable` describe admission
/// decisions. `skipped` is reserved for healthy bounded backpressure;
/// disconnected/failed workers are counted separately so a fault is never
/// presented as ordinary load shedding.
/// `stale` is deliberately orthogonal: already-admitted asynchronous work may
/// later be rejected at a visual-generation or topology boundary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct NtscPathMetrics {
    pub attempted: u64,
    pub accepted: u64,
    pub skipped: u64,
    pub unavailable: u64,
    pub stale: u64,
}

impl NtscPathMetrics {
    /// Record one admission decision at the path's bounded-work boundary.
    pub fn record_admission(&mut self, outcome: NtscSubmitOutcome) {
        self.attempted = self.attempted.saturating_add(1);
        match outcome {
            NtscSubmitOutcome::Accepted => {
                self.accepted = self.accepted.saturating_add(1);
            }
            NtscSubmitOutcome::Busy => {
                self.skipped = self.skipped.saturating_add(1);
            }
            NtscSubmitOutcome::Unavailable => {
                self.unavailable = self.unavailable.saturating_add(1);
            }
        }
    }

    /// Record an admission decision made by a non-worker stage, such as the
    /// selective GPU staging slot.
    pub fn record_attempt(&mut self, accepted: bool) {
        self.record_admission(if accepted {
            NtscSubmitOutcome::Accepted
        } else {
            NtscSubmitOutcome::Busy
        });
    }

    /// Record one asynchronously completed sample rejected by the caller's
    /// current visual-generation/path compatibility gate.
    pub fn record_stale(&mut self) {
        self.stale = self.stale.saturating_add(1);
    }

    /// Record a rejection at a downstream stage after this path's primary
    /// admission was already counted. This is orthogonal to `attempted`:
    /// selective VHS first admits a GPU readback, then offers its mapped batch
    /// to the CPU worker.
    pub fn record_downstream_rejection(&mut self, outcome: NtscSubmitOutcome) {
        match outcome {
            NtscSubmitOutcome::Accepted => {}
            NtscSubmitOutcome::Busy => {
                self.skipped = self.skipped.saturating_add(1);
            }
            NtscSubmitOutcome::Unavailable => {
                self.unavailable = self.unavailable.saturating_add(1);
            }
        }
    }

    /// Add another interval or collector into this one without wrapping.
    #[cfg(test)]
    pub fn merge(&mut self, other: Self) {
        self.attempted = self.attempted.saturating_add(other.attempted);
        self.accepted = self.accepted.saturating_add(other.accepted);
        self.skipped = self.skipped.saturating_add(other.skipped);
        self.unavailable = self.unavailable.saturating_add(other.unavailable);
        self.stale = self.stale.saturating_add(other.stale);
    }

    #[cfg(test)]
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Process-lifetime live metrics remain split because the global and
/// selective paths apply bounded backpressure at different pipeline stages.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct LiveNtscMetrics {
    pub global: NtscPathMetrics,
    pub selective: NtscPathMetrics,
}

impl LiveNtscMetrics {
    #[cfg(test)]
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Runs ntsc-rs on a dedicated thread. The render loop submits composite
/// readbacks with `try_submit` and collects processed frames with
/// `try_recv` — both non-blocking. At most one job is in flight, keeping
/// output latency bounded; if the worker is still busy, the frame is
/// simply skipped (the VHS look tolerates a dropped frame far better
/// than the render loop tolerates a stall).
pub struct NtscWorker {
    job_tx: SyncSender<NtscJob>,
    result_rx: Receiver<Result<NtscProcessedFrame, String>>,
    in_flight: usize,
    failed: bool,
    last_error: String,
}

impl NtscWorker {
    pub fn new() -> Self {
        let (job_tx, job_rx) = std::sync::mpsc::sync_channel::<NtscJob>(1);
        let (result_tx, result_rx) =
            std::sync::mpsc::sync_channel::<Result<NtscProcessedFrame, String>>(1);

        let spawn_result = std::thread::Builder::new()
            .name("ntsc-worker".into())
            .spawn(move || {
                let mut state = NtscState::new();
                while let Ok(job) = job_rx.recv() {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let mut pixels = job.pixels;
                        state.params = job.metadata.params.clone();
                        if state.apply_at_reference_frame(
                            &mut pixels,
                            job.width,
                            job.height,
                            job.metadata.reference_frame,
                        ) {
                            Ok(NtscProcessedFrame {
                                pixels,
                                epoch: job.epoch,
                                #[cfg(test)]
                                metadata: job.metadata,
                            })
                        } else {
                            Err(format!(
                                "NTSC rejected frame {}x{} (invalid buffer or disabled effect)",
                                job.width, job.height
                            ))
                        }
                    }))
                    .unwrap_or_else(|_| {
                        Err("NTSC worker panicked while processing a frame".into())
                    });
                    if result_tx.send(result).is_err() {
                        return;
                    }
                }
            });

        let (failed, last_error) = match spawn_result {
            Ok(_) => (false, String::new()),
            Err(error) => {
                let message = format!("Failed to spawn NTSC worker: {error}");
                log::error!("{message}");
                (true, message)
            }
        };

        Self {
            job_tx,
            result_rx,
            in_flight: 0,
            failed,
            last_error,
        }
    }

    /// Submit a frame for processing and distinguish bounded backpressure from
    /// a worker that can no longer accept work.
    pub fn try_submit_outcome(
        &mut self,
        pixels: Vec<u8>,
        width: u32,
        height: u32,
        metadata: NtscFrameMetadata,
        epoch: u64,
    ) -> NtscSubmitOutcome {
        if self.failed {
            return NtscSubmitOutcome::Unavailable;
        }
        if self.in_flight > 0 {
            return NtscSubmitOutcome::Busy;
        }
        let job = NtscJob {
            pixels,
            width,
            height,
            metadata,
            epoch,
        };
        match self.job_tx.try_send(job) {
            Ok(()) => {
                self.in_flight += 1;
                NtscSubmitOutcome::Accepted
            }
            Err(TrySendError::Full(_)) => NtscSubmitOutcome::Busy,
            Err(TrySendError::Disconnected(_)) => {
                self.mark_failed("NTSC worker input disconnected");
                NtscSubmitOutcome::Unavailable
            }
        }
    }

    /// Compatibility wrapper for callers interested only in admission.
    #[allow(dead_code)] // retained for older internal callers and focused tests
    pub fn try_submit(
        &mut self,
        pixels: Vec<u8>,
        width: u32,
        height: u32,
        metadata: NtscFrameMetadata,
        epoch: u64,
    ) -> bool {
        self.try_submit_outcome(pixels, width, height, metadata, epoch)
            .is_accepted()
    }

    /// Collect a processed frame if one is ready.
    pub fn try_recv(&mut self) -> Option<NtscProcessedFrame> {
        match self.result_rx.try_recv() {
            Ok(Ok(pixels)) => {
                self.in_flight = self.in_flight.saturating_sub(1);
                Some(pixels)
            }
            Ok(Err(error)) => {
                self.in_flight = self.in_flight.saturating_sub(1);
                self.last_error = error;
                log::error!("{}", self.last_error);
                None
            }
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                self.mark_failed("NTSC worker output disconnected");
                None
            }
        }
    }

    pub fn error(&self) -> &str {
        &self.last_error
    }

    pub fn is_busy(&self) -> bool {
        !self.failed && self.in_flight > 0
    }

    fn mark_failed(&mut self, message: &str) {
        self.failed = true;
        self.in_flight = 0;
        if self.last_error.is_empty() {
            self.last_error = message.to_string();
            log::error!("{message}");
        }
    }
}

/// Dedicated bounded worker for selective per-layer VHS. It intentionally has
/// no pending queue: while one generation is processing, newer mapped batches
/// are dropped, guaranteeing bounded latency and memory instead of presenting
/// an obsolete backlog. Renderer-side harvesting likewise keeps only the
/// newest completed GPU batch.
pub struct SelectiveNtscWorker {
    job_tx: SyncSender<SelectiveNtscBatch>,
    result_rx: Receiver<Result<SelectiveNtscProcessedFrame, String>>,
    in_flight: bool,
    failed: bool,
    last_error: String,
}

impl SelectiveNtscWorker {
    pub fn new() -> Self {
        let (job_tx, job_rx) = std::sync::mpsc::sync_channel::<SelectiveNtscBatch>(1);
        let (result_tx, result_rx) =
            std::sync::mpsc::sync_channel::<Result<SelectiveNtscProcessedFrame, String>>(1);
        let spawn_result = std::thread::Builder::new()
            .name("selective-ntsc-worker".into())
            .spawn(move || {
                let mut state = NtscState::new();
                while let Ok(batch) = job_rx.recv() {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        process_selective_ntsc_batch_with_state(&mut state, batch)
                    }))
                    .unwrap_or_else(|_| Err("selective NTSC worker panicked".into()));
                    if result_tx.send(result).is_err() {
                        return;
                    }
                }
            });
        let (failed, last_error) = match spawn_result {
            Ok(_) => (false, String::new()),
            Err(error) => {
                let message = format!("Failed to spawn selective NTSC worker: {error}");
                log::error!("{message}");
                (true, message)
            }
        };
        Self {
            job_tx,
            result_rx,
            in_flight: false,
            failed,
            last_error,
        }
    }

    #[allow(dead_code)] // compatibility helper; admission_outcome preserves failure semantics
    pub fn is_idle(&self) -> bool {
        matches!(self.admission_outcome(), NtscSubmitOutcome::Accepted)
    }

    pub fn is_busy(&self) -> bool {
        matches!(self.admission_outcome(), NtscSubmitOutcome::Busy)
    }

    /// Current classification at the CPU-worker admission boundary. Callers
    /// must not infer failure from `!is_idle()`: healthy in-flight work and a
    /// disconnected worker have different telemetry and recovery meaning.
    pub fn admission_outcome(&self) -> NtscSubmitOutcome {
        if self.failed {
            NtscSubmitOutcome::Unavailable
        } else if self.in_flight {
            NtscSubmitOutcome::Busy
        } else {
            NtscSubmitOutcome::Accepted
        }
    }

    pub fn try_submit_outcome(&mut self, batch: SelectiveNtscBatch) -> NtscSubmitOutcome {
        if self.failed {
            return NtscSubmitOutcome::Unavailable;
        }
        if self.in_flight {
            return NtscSubmitOutcome::Busy;
        }
        match self.job_tx.try_send(batch) {
            Ok(()) => {
                self.in_flight = true;
                NtscSubmitOutcome::Accepted
            }
            Err(TrySendError::Full(_)) => NtscSubmitOutcome::Busy,
            Err(TrySendError::Disconnected(_)) => {
                self.mark_failed("selective NTSC worker input disconnected");
                NtscSubmitOutcome::Unavailable
            }
        }
    }

    /// Compatibility wrapper for callers interested only in admission.
    #[allow(dead_code)] // retained for older internal callers and focused tests
    pub fn try_submit(&mut self, batch: SelectiveNtscBatch) -> bool {
        self.try_submit_outcome(batch).is_accepted()
    }

    pub fn try_recv(&mut self) -> Option<SelectiveNtscProcessedFrame> {
        match self.result_rx.try_recv() {
            Ok(Ok(frame)) => {
                self.in_flight = false;
                Some(frame)
            }
            Ok(Err(error)) => {
                self.in_flight = false;
                self.last_error = error;
                log::error!("{}", self.last_error);
                None
            }
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                self.mark_failed("selective NTSC worker output disconnected");
                None
            }
        }
    }

    pub fn error(&self) -> &str {
        &self.last_error
    }

    fn mark_failed(&mut self, message: &str) {
        self.failed = true;
        self.in_flight = false;
        if self.last_error.is_empty() {
            self.last_error = message.to_string();
            log::error!("{message}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        composite_selective_ntsc_layers, downscale_rgba_2x, plan_selective_ntsc,
        process_selective_ntsc_batch, process_selective_ntsc_batch_with_state_and_resolution,
        reference_frame_for_output, reference_frame_for_time, restore_alpha,
        selective_generation_compatible, selective_generation_is_current, selective_ntsc_required,
        selective_plan_compatible, srgb_byte_to_linear, upscale_rgba_2x,
        validate_selective_ntsc_gpu_staging_limit, validate_selective_ntsc_live_memory,
        LiveNtscMetrics, NtscExportQuality, NtscFrameMetadata, NtscParams, NtscPathMetrics,
        NtscState, NtscSubmitOutcome, NtscWorker, SelectiveNtscBatch, SelectiveNtscGeneration,
        SelectiveNtscLayerDescriptor, SelectiveNtscWorker, MAX_SELECTIVE_NTSC_LIVE_BYTES,
    };
    use std::time::{Duration, Instant};

    #[test]
    fn half_resolution_round_trip_handles_odd_dimensions() {
        let width = 3usize;
        let height = 3usize;
        let mut source = Vec::new();
        for pixel in 0..width * height {
            source.extend_from_slice(&[pixel as u8, 0, 0, 255]);
        }

        let small = downscale_rgba_2x(&source, width, height);
        assert_eq!(small.len(), 2 * 2 * 4);
        // The bottom-right output sample clamps to source pixel 8.
        assert_eq!(&small[12..16], &[8, 0, 0, 255]);

        let mut restored = vec![0u8; source.len()];
        upscale_rgba_2x(&small, 2, &mut restored, width, height);
        assert_eq!(&restored[(8 * 4)..(9 * 4)], &[8, 0, 0, 255]);
    }

    #[test]
    fn rgb_post_process_restores_keyed_alpha_exactly() {
        let mut pixels = vec![10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255];
        restore_alpha(&mut pixels, &[0, 127, 231]);
        assert_eq!(pixels, [10, 20, 30, 0, 40, 50, 60, 127, 70, 80, 90, 231]);
    }

    #[test]
    fn export_quality_defaults_to_live_parity_and_native_preserves_alpha() {
        assert_eq!(NtscExportQuality::default(), NtscExportQuality::LiveParity);
        assert_eq!(
            serde_json::from_str::<NtscExportQuality>("\"live_parity\"").unwrap(),
            NtscExportQuality::LiveParity
        );
        assert_eq!(
            serde_json::from_str::<NtscExportQuality>("\"native\"").unwrap(),
            NtscExportQuality::Native
        );
        assert!(serde_json::from_str::<NtscExportQuality>("\"unknown\"").is_err());

        let width = 8u32;
        let height = 8u32;
        let mut source = Vec::with_capacity((width * height * 4) as usize);
        for index in 0..(width * height) {
            let value = if (index + index / width).is_multiple_of(2) {
                0
            } else {
                255
            };
            source.extend_from_slice(&[value, 255 - value, value, (index * 3) as u8]);
        }
        let expected_alpha: Vec<_> = source.chunks_exact(4).map(|pixel| pixel[3]).collect();

        let params = NtscParams {
            enabled: true,
            snow_intensity: 0.35,
            ..Default::default()
        };
        let mut half = source.clone();
        let mut explicit_half = source.clone();
        let mut native = source;
        let mut half_state = NtscState::new();
        half_state.params = params.clone();
        assert!(half_state.apply_at_reference_frame(&mut half, width, height, 17));
        let mut explicit_half_state = NtscState::new();
        explicit_half_state.params = params.clone();
        assert!(
            explicit_half_state.apply_at_reference_frame_with_resolution(
                &mut explicit_half,
                width,
                height,
                17,
                NtscExportQuality::LiveParity,
            )
        );
        let mut native_state = NtscState::new();
        native_state.params = params;
        assert!(native_state.apply_at_reference_frame_with_resolution(
            &mut native,
            width,
            height,
            17,
            NtscExportQuality::Native,
        ));

        assert_eq!(half, explicit_half);
        assert_ne!(half, native);
        assert_eq!(
            native
                .chunks_exact(4)
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>(),
            expected_alpha
        );
    }

    #[test]
    fn reference_phase_matches_at_thirty_and_sixty_fps() {
        for frame_30 in 0..120u64 {
            let at_30 = reference_frame_for_output(frame_30, 30);
            let at_60 = reference_frame_for_output(frame_30 * 2, 60);
            assert_eq!(at_30, frame_30 as usize);
            assert_eq!(at_60, frame_30 as usize);
            assert_eq!(
                reference_frame_for_output(frame_30 * 2 + 1, 60),
                frame_30 as usize
            );
        }
        assert_eq!(reference_frame_for_time(f64::NAN), 0);
        assert_eq!(reference_frame_for_time(-1.0), 0);
    }

    #[test]
    fn live_metrics_are_path_specific_saturating_and_backward_compatible() {
        let mut metrics = LiveNtscMetrics::default();
        metrics.global.record_admission(NtscSubmitOutcome::Accepted);
        metrics.global.record_admission(NtscSubmitOutcome::Busy);
        metrics
            .global
            .record_admission(NtscSubmitOutcome::Unavailable);
        metrics.global.record_stale();
        metrics.selective.record_attempt(true);
        metrics.selective.record_attempt(false);
        metrics
            .selective
            .record_downstream_rejection(NtscSubmitOutcome::Unavailable);

        assert_eq!(
            metrics.global,
            NtscPathMetrics {
                attempted: 3,
                accepted: 1,
                skipped: 1,
                unavailable: 1,
                stale: 1,
            }
        );
        assert_eq!(
            metrics.selective,
            NtscPathMetrics {
                attempted: 2,
                accepted: 1,
                skipped: 1,
                unavailable: 1,
                stale: 0,
            }
        );

        let legacy: LiveNtscMetrics = serde_json::from_str("{}").unwrap();
        assert_eq!(legacy, LiveNtscMetrics::default());

        let mut saturated = NtscPathMetrics {
            attempted: u64::MAX,
            accepted: u64::MAX,
            skipped: u64::MAX,
            unavailable: u64::MAX,
            stale: u64::MAX,
        };
        saturated.record_admission(NtscSubmitOutcome::Accepted);
        saturated.record_admission(NtscSubmitOutcome::Busy);
        saturated.record_stale();
        saturated.merge(NtscPathMetrics {
            attempted: 1,
            accepted: 1,
            skipped: 1,
            unavailable: 1,
            stale: 1,
        });
        assert_eq!(
            saturated,
            NtscPathMetrics {
                attempted: u64::MAX,
                accepted: u64::MAX,
                skipped: u64::MAX,
                unavailable: u64::MAX,
                stale: u64::MAX,
            }
        );

        metrics.reset();
        assert_eq!(metrics, LiveNtscMetrics::default());
        saturated.reset();
        assert_eq!(saturated, NtscPathMetrics::default());
    }

    #[test]
    fn worker_preserves_metadata_when_a_newer_job_is_dropped() {
        let mut worker = NtscWorker::new();
        let first = NtscFrameMetadata {
            params: NtscParams {
                enabled: true,
                snow_intensity: 0.125,
                ..NtscParams::default()
            },
            reference_frame: 17,
        };
        let newer = NtscFrameMetadata {
            params: NtscParams {
                enabled: true,
                snow_intensity: 0.75,
                ..NtscParams::default()
            },
            reference_frame: 41,
        };
        assert_eq!(
            worker.try_submit_outcome(vec![128; 64 * 64 * 4], 64, 64, first.clone(), 9),
            NtscSubmitOutcome::Accepted
        );
        assert!(worker.is_busy());
        assert_eq!(
            worker.try_submit_outcome(vec![64; 64 * 64 * 4], 64, 64, newer, 9),
            NtscSubmitOutcome::Busy
        );

        let deadline = Instant::now() + Duration::from_secs(2);
        let processed = loop {
            if let Some(frame) = worker.try_recv() {
                break frame;
            }
            assert!(Instant::now() < deadline, "NTSC worker did not respond");
            std::thread::yield_now();
        };
        assert_eq!(processed.epoch, 9);
        assert_eq!(processed.metadata, first);
        assert!(!worker.is_busy());

        assert!(worker.try_submit(vec![128; 4 * 4 * 4], 4, 4, enabled_metadata(), 10,));
    }

    fn descriptor(
        id: u64,
        visible: bool,
        bypass: bool,
        opacity: f32,
        blend_mode: u32,
    ) -> SelectiveNtscLayerDescriptor {
        SelectiveNtscLayerDescriptor {
            layer_id: id,
            visible,
            bypass_master_fx: bypass,
            opacity,
            blend_mode,
            transform_fingerprint: id.wrapping_mul(37),
        }
    }

    fn generation(width: u32, height: u32) -> SelectiveNtscGeneration {
        SelectiveNtscGeneration {
            visual_epoch: 7,
            topology_generation: 11,
            width,
            height,
            sample_sequence: 1,
        }
    }

    fn enabled_metadata() -> NtscFrameMetadata {
        NtscFrameMetadata {
            params: NtscParams {
                enabled: true,
                ..NtscParams::default()
            },
            reference_frame: 3,
        }
    }

    #[test]
    fn selective_planner_truth_table_preserves_legacy_switch() {
        assert!(!selective_ntsc_required([]));
        assert!(!selective_ntsc_required([descriptor(
            1, false, true, 1.0, 0
        )]));
        assert!(!selective_ntsc_required([descriptor(
            1, true, true, 0.0, 0
        )]));
        assert!(!selective_ntsc_required([descriptor(
            1,
            true,
            true,
            f32::NAN,
            0,
        )]));
        assert!(!selective_ntsc_required([descriptor(
            1, true, false, 1.0, 0
        )]));
        assert!(selective_ntsc_required([descriptor(
            1, true, true, 0.001, 0
        )]));

        let disabled = NtscFrameMetadata {
            params: NtscParams::default(),
            reference_frame: 0,
        };
        assert!(plan_selective_ntsc(
            generation(1, 1),
            disabled,
            [descriptor(1, true, true, 1.0, 0)]
        )
        .is_none());
    }

    #[test]
    fn selective_plan_is_bottom_to_top_and_forces_only_bottom_normal() {
        let plan = plan_selective_ntsc(
            generation(2, 1),
            enabled_metadata(),
            [
                descriptor(10, true, true, 0.5, 3),
                descriptor(20, false, false, 1.0, 2),
                descriptor(30, true, false, 1.0, 2),
            ],
        )
        .unwrap();
        assert_eq!(
            plan.layers
                .iter()
                .map(|layer| layer.layer_id)
                .collect::<Vec<_>>(),
            [30, 10]
        );
        assert_eq!(plan.layers[0].blend_mode, 0);
        assert_eq!(plan.layers[1].blend_mode, 3);
        assert!(!plan.layers[0].bypass_master_fx);
        assert!(plan.layers[1].bypass_master_fx);
    }

    #[test]
    fn live_memory_budget_accepts_1080p_max_stack_and_bounds_4k() {
        let full_hd = validate_selective_ntsc_live_memory(1920, 1080, 16).unwrap();
        assert!(full_hd.total_bytes <= MAX_SELECTIVE_NTSC_LIVE_BYTES);
        assert_eq!(full_hd.gpu_staging_bytes, 1920 * 4 * 1080 * 16);

        let four_k_two = validate_selective_ntsc_live_memory(3840, 2160, 2).unwrap();
        assert_eq!(four_k_two.ntsc_alpha_bytes, 3840 * 2160);
        assert_eq!(four_k_two.ntsc_half_rgba_bytes, 1920 * 1080 * 4);
        assert_eq!(four_k_two.ntsc_yiq_bytes, 1920 * 1080 * 4 * 4);
        let three_layer_error = validate_selective_ntsc_live_memory(3840, 2160, 3).unwrap_err();
        assert!(three_layer_error.contains("prior audience frame is being held"));
        assert!(validate_selective_ntsc_live_memory(3840, 2160, 16).is_err());
    }

    #[test]
    fn gpu_staging_respects_the_exact_device_buffer_boundary() {
        let memory = validate_selective_ntsc_live_memory(1920, 1080, 16).unwrap();
        assert!(
            validate_selective_ntsc_gpu_staging_limit(memory, memory.gpu_staging_bytes).is_ok()
        );
        let error = validate_selective_ntsc_gpu_staging_limit(memory, memory.gpu_staging_bytes - 1)
            .unwrap_err();
        assert!(error.contains("graphics device permits at most"));
        assert!(error.contains("prior audience frame is being held"));
    }

    #[test]
    fn plan_acceptance_allows_sampled_controls_but_rejects_structural_identity() {
        let completed = plan_selective_ntsc(
            generation(2, 1),
            enabled_metadata(),
            [descriptor(44, true, true, 0.4, 3)],
        )
        .unwrap();
        let mut current = completed.clone();
        current.generation.sample_sequence += 1;
        current.metadata.reference_frame += 1;
        current.metadata.params.snow_intensity = 0.7;
        current.layers[0].opacity = 0.9;
        current.layers[0].blend_mode = 1;
        assert!(
            selective_plan_compatible(&completed, &current),
            "frame-local modulation remains a coherent delayed sample"
        );

        current.layers[0].transform_fingerprint ^= 1;
        assert!(!selective_plan_compatible(&completed, &current));
        current.layers[0].transform_fingerprint ^= 1;
        current.generation.topology_generation += 1;
        assert!(!selective_plan_compatible(&completed, &current));
    }

    #[test]
    fn all_bypass_processor_is_byte_exact_at_full_opacity() {
        let plan = plan_selective_ntsc(
            generation(2, 1),
            enabled_metadata(),
            [descriptor(44, true, true, 1.0, 0)],
        )
        .unwrap();
        let dry = vec![3, 17, 249, 0, 255, 128, 7, 231];
        for quality in [NtscExportQuality::LiveParity, NtscExportQuality::Native] {
            let processed = process_selective_ntsc_batch_with_state_and_resolution(
                &mut NtscState::new(),
                SelectiveNtscBatch {
                    plan: plan.clone(),
                    slices: vec![dry.clone()],
                },
                quality,
            )
            .unwrap();
            assert_eq!(processed.pixels, dry, "quality {quality:?}");
            assert_eq!(processed.plan.generation, generation(2, 1));
        }
    }

    #[test]
    fn selective_export_quality_changes_only_the_inherited_processing_route() {
        let width = 8;
        let height = 8;
        let plan = plan_selective_ntsc(
            generation(width, height),
            NtscFrameMetadata {
                params: NtscParams {
                    enabled: true,
                    snow_intensity: 0.35,
                    ..Default::default()
                },
                reference_frame: 19,
            },
            [
                descriptor(10, true, true, 0.35, 0),
                descriptor(20, true, false, 1.0, 0),
            ],
        )
        .unwrap();
        assert!(!plan.layers[0].bypass_master_fx);
        assert!(plan.layers[1].bypass_master_fx);

        let mut inherited = Vec::with_capacity((width * height * 4) as usize);
        for index in 0..(width * height) {
            let value = if (index + index / width) % 2 == 0 {
                0
            } else {
                255
            };
            inherited.extend_from_slice(&[value, 255 - value, value, 255]);
        }
        let bypass = vec![24; (width * height * 4) as usize];

        let run = |quality| {
            process_selective_ntsc_batch_with_state_and_resolution(
                &mut NtscState::new(),
                SelectiveNtscBatch {
                    plan: plan.clone(),
                    slices: vec![inherited.clone(), bypass.clone()],
                },
                quality,
            )
            .unwrap()
            .pixels
        };
        assert_ne!(
            run(NtscExportQuality::LiveParity),
            run(NtscExportQuality::Native)
        );
    }

    #[test]
    fn cpu_compositor_matches_straight_alpha_order_and_blend_laws() {
        let base = [128, 64, 32, 128];
        let overlay = [32, 192, 224, 128];
        for mode in 0..=3 {
            let plan = plan_selective_ntsc(
                generation(1, 1),
                enabled_metadata(),
                [
                    descriptor(1, true, true, 0.75, mode),
                    descriptor(2, true, true, 1.0, 0),
                ],
            )
            .unwrap();
            // UI order is top then bottom, so slices are bottom then top.
            let result =
                composite_selective_ntsc_layers(&plan, &[base.to_vec(), overlay.to_vec()]).unwrap();
            // Bottom alpha 128/255, top effective alpha .75*128/255.
            let base_alpha = 128.0_f32 / 255.0;
            let source_alpha = 0.75 * 128.0 / 255.0;
            let expected_alpha =
                ((source_alpha + base_alpha * (1.0 - source_alpha)) * 255.0).round() as u8;
            assert_eq!(result[3], expected_alpha, "mode {mode}");
            assert!(result[..3].iter().any(|channel| *channel != 0));
        }

        let normal_plan = plan_selective_ntsc(
            generation(1, 1),
            enabled_metadata(),
            [
                descriptor(1, true, true, 1.0, 0),
                descriptor(2, true, true, 1.0, 0),
            ],
        )
        .unwrap();
        let reversed =
            composite_selective_ntsc_layers(&normal_plan, &[overlay.to_vec(), base.to_vec()])
                .unwrap();
        assert_ne!(reversed, overlay, "layer order must be observable");
    }

    #[test]
    fn selective_processor_rejects_generation_shape_mismatch() {
        let plan = plan_selective_ntsc(
            generation(2, 2),
            enabled_metadata(),
            [descriptor(1, true, true, 1.0, 0)],
        )
        .unwrap();
        let error = process_selective_ntsc_batch(SelectiveNtscBatch {
            plan,
            slices: vec![vec![0; 4]],
        })
        .unwrap_err();
        assert!(error.contains("expected 16"));
    }

    #[test]
    fn selective_generation_gate_rejects_every_stale_axis() {
        let current = generation(1920, 1080);
        assert!(selective_generation_is_current(current, current));
        for stale in [
            SelectiveNtscGeneration {
                visual_epoch: current.visual_epoch + 1,
                ..current
            },
            SelectiveNtscGeneration {
                topology_generation: current.topology_generation + 1,
                ..current
            },
            SelectiveNtscGeneration {
                width: current.width + 1,
                ..current
            },
            SelectiveNtscGeneration {
                height: current.height + 1,
                ..current
            },
            SelectiveNtscGeneration {
                sample_sequence: current.sample_sequence + 1,
                ..current
            },
        ] {
            assert!(!selective_generation_is_current(stale, current));
        }

        let trailing = SelectiveNtscGeneration {
            sample_sequence: current.sample_sequence.saturating_sub(1),
            ..current
        };
        assert!(selective_generation_compatible(trailing, current));
        assert!(!selective_generation_compatible(
            SelectiveNtscGeneration {
                sample_sequence: current.sample_sequence + 1,
                ..current
            },
            current
        ));
        assert!(!selective_generation_compatible(
            SelectiveNtscGeneration {
                topology_generation: current.topology_generation + 1,
                ..trailing
            },
            current
        ));
    }

    #[test]
    fn srgb_transfer_runs_compositing_in_linear_light() {
        assert_eq!(srgb_byte_to_linear(0), 0.0);
        assert!((srgb_byte_to_linear(188) - 0.502_886_6).abs() < 0.000_01);
        assert_eq!(srgb_byte_to_linear(255), 1.0);
    }

    #[test]
    fn selective_worker_is_bounded_to_one_generation() {
        let plan = plan_selective_ntsc(
            generation(1, 1),
            enabled_metadata(),
            [descriptor(1, true, true, 1.0, 0)],
        )
        .unwrap();
        let mut worker = SelectiveNtscWorker::new();
        assert_eq!(
            worker.try_submit_outcome(SelectiveNtscBatch {
                plan: plan.clone(),
                slices: vec![vec![4, 3, 2, 1]],
            }),
            NtscSubmitOutcome::Accepted
        );
        assert!(worker.is_busy());
        assert_eq!(
            worker.try_submit_outcome(SelectiveNtscBatch {
                plan,
                slices: vec![vec![9, 8, 7, 6]],
            }),
            NtscSubmitOutcome::Busy
        );

        let deadline = Instant::now() + Duration::from_secs(2);
        let frame = loop {
            if let Some(frame) = worker.try_recv() {
                break frame;
            }
            assert!(
                Instant::now() < deadline,
                "selective worker did not respond"
            );
            std::thread::yield_now();
        };
        assert_eq!(frame.pixels, [4, 3, 2, 1]);
        assert!(!worker.is_busy());
        assert_eq!(worker.admission_outcome(), NtscSubmitOutcome::Accepted);
    }

    #[test]
    fn failed_selective_worker_is_unavailable_never_healthy_backpressure() {
        let mut worker = SelectiveNtscWorker::new();
        worker.mark_failed("selective worker failure fixture");

        assert_eq!(worker.admission_outcome(), NtscSubmitOutcome::Unavailable);
        assert!(!worker.is_idle());
        assert!(!worker.is_busy());

        let mut metrics = NtscPathMetrics::default();
        metrics.record_admission(worker.admission_outcome());
        assert_eq!(metrics.attempted, 1);
        assert_eq!(metrics.accepted, 0);
        assert_eq!(metrics.skipped, 0);
        assert_eq!(metrics.unavailable, 1);
    }
}
