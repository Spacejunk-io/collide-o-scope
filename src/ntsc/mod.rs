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
use ntsc_rs::{Context, NtscEffect};
use ntsc_rs::yiq_fielding::Rgbx;

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

        self.sync_effect_from_params();

        let w = width as usize;
        let h = height as usize;
        let half_w = w / 2;
        let half_h = h / 2;

        // Downscale 2x with box filter (average 2x2 blocks)
        let mut small = vec![0u8; half_w * half_h * 4];
        for sy in 0..half_h {
            for sx in 0..half_w {
                let dst = (sy * half_w + sx) * 4;
                let s00 = ((sy * 2) * w + sx * 2) * 4;
                let s10 = ((sy * 2) * w + sx * 2 + 1) * 4;
                let s01 = ((sy * 2 + 1) * w + sx * 2) * 4;
                let s11 = ((sy * 2 + 1) * w + sx * 2 + 1) * 4;
                for c in 0..4 {
                    small[dst + c] = ((pixels[s00 + c] as u16
                        + pixels[s10 + c] as u16
                        + pixels[s01 + c] as u16
                        + pixels[s11 + c] as u16) / 4) as u8;
                }
            }
        }

        // Apply ntsc-rs at half resolution
        self.effect.apply_effect_to_buffer::<Rgbx, u8>(
            &self.ctx,
            (half_w, half_h),
            &mut small,
            self.frame_num,
            [1.0, 1.0],
        );

        // Upscale back with nearest-neighbor (VHS doesn't need bilinear)
        for y in 0..h {
            for x in 0..w {
                let sx = x / 2;
                let sy = y / 2;
                let src = (sy * half_w + sx) * 4;
                let dst = (y * w + x) * 4;
                pixels[dst..dst + 4].copy_from_slice(&small[src..src + 4]);
            }
        }

        self.frame_num = self.frame_num.wrapping_add(1);
        true
    }

    /// Apply VHS effects at full (native) resolution. For offline rendering.
    pub fn apply_full_res(&mut self, pixels: &mut [u8], width: u32, height: u32) -> bool {
        if !self.params.enabled {
            return false;
        }

        self.sync_effect_from_params();

        let w = width as usize;
        let h = height as usize;

        self.effect.apply_effect_to_buffer::<Rgbx, u8>(
            &self.ctx,
            (w, h),
            pixels,
            self.frame_num,
            [1.0, 1.0],
        );

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
        self.effect.vhs_settings.settings.edge_wave.settings.intensity = p.edge_wave_intensity;
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

impl NtscParams {
    /// Apply a named parameter from a JSON value.
    pub fn set_param(&mut self, param: &str, value: &serde_json::Value) {
        match param {
            "enabled" => {
                if let Some(b) = value.as_bool() { self.enabled = b; }
            }
            "tape_speed" => {
                if let Some(n) = value.as_u64() { self.tape_speed = n as u32; }
            }
            "chroma_loss" => {
                if let Some(n) = value.as_f64() { self.chroma_loss = n as f32; }
            }
            "edge_wave_enabled" => {
                if let Some(b) = value.as_bool() { self.edge_wave_enabled = b; }
            }
            "edge_wave_intensity" => {
                if let Some(n) = value.as_f64() { self.edge_wave_intensity = n as f32; }
            }
            "edge_wave_speed" => {
                if let Some(n) = value.as_f64() { self.edge_wave_speed = n as f32; }
            }
            "head_switching_enabled" => {
                if let Some(b) = value.as_bool() { self.head_switching_enabled = b; }
            }
            "head_switching_height" => {
                if let Some(n) = value.as_i64() { self.head_switching_height = n as i32; }
            }
            "head_switching_shift" => {
                if let Some(n) = value.as_f64() { self.head_switching_shift = n as f32; }
            }
            "tracking_noise_enabled" => {
                if let Some(b) = value.as_bool() { self.tracking_noise_enabled = b; }
            }
            "tracking_noise_height" => {
                if let Some(n) = value.as_i64() { self.tracking_noise_height = n as i32; }
            }
            "tracking_noise_wave" => {
                if let Some(n) = value.as_f64() { self.tracking_noise_wave = n as f32; }
            }
            "tracking_noise_snow" => {
                if let Some(n) = value.as_f64() { self.tracking_noise_snow = n as f32; }
            }
            "snow_intensity" => {
                if let Some(n) = value.as_f64() { self.snow_intensity = n as f32; }
            }
            "composite_noise_intensity" => {
                if let Some(n) = value.as_f64() { self.composite_noise_intensity = n as f32; }
            }
            "luma_noise_intensity" => {
                if let Some(n) = value.as_f64() { self.luma_noise_intensity = n as f32; }
            }
            "chroma_noise_intensity" => {
                if let Some(n) = value.as_f64() { self.chroma_noise_intensity = n as f32; }
            }
            "luma_smear" => {
                if let Some(n) = value.as_f64() { self.luma_smear = n as f32; }
            }
            "composite_sharpening" => {
                if let Some(n) = value.as_f64() { self.composite_sharpening = n as f32; }
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
}

/// Runs ntsc-rs on a dedicated thread. The render loop submits composite
/// readbacks with `try_submit` and collects processed frames with
/// `try_recv` — both non-blocking. At most one job is in flight, keeping
/// output latency bounded; if the worker is still busy, the frame is
/// simply skipped (the VHS look tolerates a dropped frame far better
/// than the render loop tolerates a stall).
pub struct NtscWorker {
    job_tx: SyncSender<NtscJob>,
    result_rx: Receiver<Vec<u8>>,
    in_flight: usize,
}

impl NtscWorker {
    pub fn new() -> Self {
        let (job_tx, job_rx) = std::sync::mpsc::sync_channel::<NtscJob>(1);
        let (result_tx, result_rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(1);

        std::thread::Builder::new()
            .name("ntsc-worker".into())
            .spawn(move || {
                let mut state = NtscState::new();
                while let Ok(job) = job_rx.recv() {
                    let mut pixels = job.pixels;
                    state.params = job.params;
                    state.apply(&mut pixels, job.width, job.height);
                    if result_tx.send(pixels).is_err() {
                        return;
                    }
                }
            })
            .expect("Failed to spawn ntsc-worker thread");

        Self {
            job_tx,
            result_rx,
            in_flight: 0,
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
    ) -> bool {
        if self.in_flight > 0 {
            return false;
        }
        let job = NtscJob { pixels, width, height, params };
        match self.job_tx.try_send(job) {
            Ok(()) => {
                self.in_flight += 1;
                true
            }
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => false,
        }
    }

    /// Collect a processed frame if one is ready.
    pub fn try_recv(&mut self) -> Option<Vec<u8>> {
        match self.result_rx.try_recv() {
            Ok(pixels) => {
                self.in_flight = self.in_flight.saturating_sub(1);
                Some(pixels)
            }
            Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => None,
        }
    }
}
