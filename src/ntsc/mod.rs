//! ntsc-rs VHS effect wrapper.
//!
//! Applies analog VHS effects (head switching, tracking noise, snow, etc.)
//! as a CPU-based post-process on the final composite RGBA buffer.
//!
//! For live rendering, processing runs on a dedicated worker thread
//! (`NtscWorker`) fed by async GPU readbacks, so the render loop never
//! blocks on the CPU-bound effect. The displayed NTSC output trails the
//! live composite by ~2 frames, which is imperceptible for a VHS look.

use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, TrySendError};

use ntsc_rs::settings::standard::*;
use ntsc_rs::yiq_fielding::Rgbx;
use ntsc_rs::{Context, NtscEffect};

/// User-facing VHS parameters (mirrored in the web UI).
#[derive(Debug, Clone)]
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
    frame_num: usize,
}

impl NtscState {
    pub fn new() -> Self {
        Self {
            ctx: Context::new(),
            effect: NtscEffect::default(),
            params: NtscParams::default(),
            frame_num: 0,
        }
    }

    /// Apply VHS effects to an RGBA buffer in-place.
    /// Processes at half resolution for performance, then upscales back.
    /// Returns true if effects were applied, false if disabled/skipped.
    pub fn apply(&mut self, pixels: &mut [u8], width: u32, height: u32) -> bool {
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

        // Ceil division keeps a real sample for an odd final row/column.
        let half_w = w.div_ceil(2);
        let half_h = h.div_ceil(2);
        let mut small = downscale_rgba_2x(pixels, w, h);

        // Apply ntsc-rs at half resolution
        self.effect.apply_effect_to_buffer::<Rgbx, u8>(
            &self.ctx,
            (half_w, half_h),
            &mut small,
            self.frame_num,
            [1.0, 1.0],
        );

        // Upscale back with nearest-neighbor (VHS doesn't need bilinear).
        upscale_rgba_2x(&small, half_w, pixels, w, h);

        self.frame_num = self.frame_num.wrapping_add(1);
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

/// A frame of work for the NTSC worker thread.
struct NtscJob {
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    params: NtscParams,
    epoch: u64,
}

/// A processed frame tagged with the visual generation from which it came.
/// The caller must discard it when that generation is no longer current.
pub struct NtscProcessedFrame {
    pub pixels: Vec<u8>,
    pub epoch: u64,
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
                        state.params = job.params;
                        if state.apply(&mut pixels, job.width, job.height) {
                            Ok(NtscProcessedFrame {
                                pixels,
                                epoch: job.epoch,
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

    /// Submit a frame for processing. Returns false (dropping the frame)
    /// if the worker is still busy with the previous one.
    pub fn try_submit(
        &mut self,
        pixels: Vec<u8>,
        width: u32,
        height: u32,
        params: NtscParams,
        epoch: u64,
    ) -> bool {
        if self.failed || self.in_flight > 0 {
            return false;
        }
        let job = NtscJob {
            pixels,
            width,
            height,
            params,
            epoch,
        };
        match self.job_tx.try_send(job) {
            Ok(()) => {
                self.in_flight += 1;
                true
            }
            Err(TrySendError::Full(_)) => false,
            Err(TrySendError::Disconnected(_)) => {
                self.mark_failed("NTSC worker input disconnected");
                false
            }
        }
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

    fn mark_failed(&mut self, message: &str) {
        self.failed = true;
        self.in_flight = 0;
        if self.last_error.is_empty() {
            self.last_error = message.to_string();
            log::error!("{message}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{downscale_rgba_2x, upscale_rgba_2x};

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
}
