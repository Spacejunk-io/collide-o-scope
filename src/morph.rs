//! Patch morphing: deterministic interpolation between two captured states.
//!
//! The serializable snapshot types in this module deliberately contain only
//! performance parameters. Render-loop values such as output resolution and
//! elapsed shader time are not captured. [`Morph::sample`] is pure: live and
//! offline renderers can consume the same detached [`MorphSample`] without
//! constructing or mutating a [`Layer`]. [`Morph::apply`] remains as the
//! compatibility adapter used by the live renderer.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::effects::params::TemporalParams;
use crate::effects::EffectUniforms;
use crate::layers::{BlendMode, Layer};
use crate::ntsc::NtscParams;

/// Parameter interpolation law for the A/B crossfader.
///
/// `EqualPower` uses complementary squared sine/cosine weights. They sum to
/// one, so equal snapshots remain unchanged, while their underlying amplitude
/// gains obey the conventional equal-power `cos^2 + sin^2 = 1` relationship.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MorphBlendLaw {
    #[default]
    Linear,
    EqualPower,
}

impl MorphBlendLaw {
    /// Return the contribution of snapshots A and B for a crossfader value.
    /// Non-finite values are treated as A and finite values are clamped.
    pub fn weights(self, t: f32) -> [f32; 2] {
        let t = normalized_position(t);
        // Endpoints are state recalls, not merely limiting cases. Return
        // exact weights so a zero channel/key value cannot inherit the tiny
        // cosine residue produced by finite-precision PI/2 evaluation.
        if t <= 0.0 {
            return [1.0, 0.0];
        }
        if t >= 1.0 {
            return [0.0, 1.0];
        }
        match self {
            Self::Linear => [1.0 - t, t],
            Self::EqualPower => {
                let angle = t * std::f32::consts::FRAC_PI_2;
                let a = angle.cos();
                let b = angle.sin();
                [a * a, b * b]
            }
        }
    }
}

/// A deterministic crossfader movement described entirely in beat space.
///
/// Callers own the clock. Passing the same beat always produces the same
/// position, whether the caller is the live loop, an exporter, or a test.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MorphGlide {
    pub start: f32,
    pub target: f32,
    pub start_beat: f64,
    pub duration_beats: f64,
}

impl MorphGlide {
    /// Construct an operator-requested glide. Positive durations follow the
    /// control panel's quarter-beat minimum; zero is the explicit snap.
    pub fn new(start: f32, target: f32, start_beat: f64, duration_beats: f64) -> Self {
        let duration_beats = finite_f64_or(duration_beats, 0.0);
        let duration_beats = if duration_beats <= 0.0 {
            0.0
        } else {
            duration_beats.clamp(0.25, 64.0)
        };
        Self::with_remaining(start, target, start_beat, duration_beats)
    }

    /// Construct internal/persisted remaining movement. A glide saved less
    /// than a quarter beat before its endpoint must not be stretched back to
    /// the UI's minimum duration on recall.
    fn with_remaining(start: f32, target: f32, start_beat: f64, duration_beats: f64) -> Self {
        let duration_beats = finite_f64_or(duration_beats, 0.0);
        Self {
            start: normalized_position(start),
            target: normalized_position(target),
            start_beat: finite_f64_or(start_beat, 0.0),
            duration_beats: duration_beats.clamp(0.0, 64.0),
        }
    }

    /// Evaluate the glide at an explicit beat. A zero-duration glide snaps to
    /// its target; beats before the start remain at the starting position.
    pub fn position_at(self, beat: f64) -> f32 {
        let clean = self.sanitized();
        if clean.duration_beats <= 0.0 {
            return clean.target;
        }
        let beat = finite_f64_or(beat, clean.start_beat);
        let progress = ((beat - clean.start_beat) / clean.duration_beats).clamp(0.0, 1.0) as f32;
        lerp(clean.start, clean.target, [1.0 - progress, progress])
    }

    pub fn is_complete_at(self, beat: f64) -> bool {
        let clean = self.sanitized();
        clean.duration_beats <= 0.0
            || finite_f64_or(beat, clean.start_beat) >= clean.start_beat + clean.duration_beats
    }

    fn sanitized(self) -> Self {
        Self::with_remaining(
            self.start,
            self.target,
            self.start_beat,
            self.duration_beats,
        )
    }
}

/// Serializable master-effect values captured by a morph slot.
/// Runtime-owned `resolution`, `time`, and `random_seed` uniforms are
/// intentionally absent. Pattern identity must not drift while Morph moves.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MorphMasterSnapshot {
    pub pixelate_size: f32,
    pub rgb_split: f32,
    pub hue_shift: f32,
    pub saturation: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub posterize: f32,
    pub invert: f32,
    pub downsample: f32,
    pub grain_intensity: f32,
    pub grain_size: f32,
    pub grain_algo: f32,
    pub color_grain: f32,
    pub breathe_scale: f32,
    pub breathe_rotation: f32,
    pub breathe_position: f32,
    pub vignette: f32,
    pub color_drift: f32,
    pub key_mode: f32,
    pub key_threshold: f32,
    pub key_softness: f32,
    pub key_color: [f32; 3],
    pub key_tolerance: f32,
    pub cellular_amount: f32,
    pub cellular_scale: f32,
    pub cellular_warp: f32,
    pub cellular_speed: f32,
    pub cellular_gap_amount: f32,
    pub cellular_gap_threshold: f32,
    pub cellular_gap_softness: f32,
    pub shift_amount: f32,
    pub shift_block_size: f32,
    pub shift_density: f32,
    pub shift_speed: f32,
}

impl Default for MorphMasterSnapshot {
    fn default() -> Self {
        Self::capture(&EffectUniforms::default())
    }
}

impl MorphMasterSnapshot {
    pub fn capture(value: &EffectUniforms) -> Self {
        Self {
            pixelate_size: value.pixelate_size,
            rgb_split: value.rgb_split,
            hue_shift: value.hue_shift,
            saturation: value.saturation,
            brightness: value.brightness,
            contrast: value.contrast,
            posterize: value.posterize,
            invert: value.invert,
            downsample: value.downsample,
            grain_intensity: value.grain_intensity,
            grain_size: value.grain_size,
            grain_algo: value.grain_algo,
            color_grain: value.color_grain,
            breathe_scale: value.breathe_scale,
            breathe_rotation: value.breathe_rotation,
            breathe_position: value.breathe_position,
            vignette: value.vignette,
            color_drift: value.color_drift,
            key_mode: value.key_mode,
            key_threshold: value.key_threshold,
            key_softness: value.key_softness,
            key_color: value.key_color,
            key_tolerance: value.key_tolerance,
            cellular_amount: value.cellular_amount,
            cellular_scale: value.cellular_scale,
            cellular_warp: value.cellular_warp,
            cellular_speed: value.cellular_speed,
            cellular_gap_amount: value.cellular_gap_amount,
            cellular_gap_threshold: value.cellular_gap_threshold,
            cellular_gap_softness: value.cellular_gap_softness,
            shift_amount: value.shift_amount,
            shift_block_size: value.shift_block_size,
            shift_density: value.shift_density,
            shift_speed: value.shift_speed,
        }
        .sanitized()
    }

    /// Normalize untrusted persisted values to the same ranges accepted by
    /// patch loading and the control panel. GPU uniforms must never receive
    /// NaN or infinity, even when a hand-edited patch contains them.
    pub fn sanitized(&self) -> Self {
        Self {
            pixelate_size: finite_clamp(self.pixelate_size, 1.0, 1.0, 32.0),
            rgb_split: finite_clamp(self.rgb_split, 0.0, 0.0, 30.0),
            hue_shift: finite_clamp(self.hue_shift, 0.0, -180.0, 180.0),
            saturation: finite_clamp(self.saturation, 0.0, -1.0, 1.0),
            brightness: finite_clamp(self.brightness, 0.0, -1.0, 1.0),
            contrast: finite_clamp(self.contrast, 0.0, -1.0, 1.0),
            posterize: finite_clamp(self.posterize, 0.0, 0.0, 16.0),
            invert: discrete_f32(self.invert, 0.0, 1.0),
            downsample: finite_clamp(self.downsample, 1.0, 0.05, 1.0),
            grain_intensity: finite_clamp(self.grain_intensity, 0.0, 0.0, 0.3),
            grain_size: finite_clamp(self.grain_size, 1.0, 1.0, 4.0),
            grain_algo: discrete_f32(self.grain_algo, 0.0, 3.0),
            color_grain: discrete_f32(self.color_grain, 0.0, 1.0),
            breathe_scale: finite_clamp(self.breathe_scale, 0.0, 0.0, 0.05),
            breathe_rotation: finite_clamp(self.breathe_rotation, 0.0, 0.0, 2.0),
            breathe_position: finite_clamp(self.breathe_position, 0.0, 0.0, 0.02),
            vignette: finite_clamp(self.vignette, 0.0, 0.0, 1.5),
            color_drift: finite_clamp(self.color_drift, 0.0, 0.0, 0.02),
            key_mode: discrete_f32(self.key_mode, 0.0, 4.0),
            key_threshold: finite_clamp(self.key_threshold, 0.5, 0.0, 1.0),
            key_softness: finite_clamp(self.key_softness, 0.1, 0.0, 0.5),
            key_color: [
                finite_clamp(self.key_color[0], 0.0, 0.0, 1.0),
                finite_clamp(self.key_color[1], 1.0, 0.0, 1.0),
                finite_clamp(self.key_color[2], 0.0, 0.0, 1.0),
            ],
            key_tolerance: finite_clamp(self.key_tolerance, 0.15, 0.0, 1.0),
            cellular_amount: finite_clamp(self.cellular_amount, 0.0, 0.0, 1.0),
            cellular_scale: finite_clamp(self.cellular_scale, 10.0, 2.0, 32.0),
            cellular_warp: finite_clamp(self.cellular_warp, 0.35, 0.0, 1.0),
            cellular_speed: finite_clamp(self.cellular_speed, 0.25, 0.0, 2.0),
            cellular_gap_amount: finite_clamp(self.cellular_gap_amount, 0.0, 0.0, 1.0),
            cellular_gap_threshold: finite_clamp(self.cellular_gap_threshold, 0.65, 0.0, 1.0),
            cellular_gap_softness: finite_clamp(self.cellular_gap_softness, 0.08, 0.0, 0.5),
            shift_amount: finite_clamp(self.shift_amount, 0.0, 0.0, 1.0),
            shift_block_size: finite_clamp(self.shift_block_size, 8.0, 2.0, 256.0),
            shift_density: finite_clamp(self.shift_density, 0.5, 0.0, 1.0),
            shift_speed: finite_clamp(self.shift_speed, 3.0, 0.0, 20.0),
        }
    }

    /// Apply the captured performance values while preserving render-loop
    /// ownership of resolution, time, and deterministic pattern identity.
    pub fn apply_to(&self, value: &mut EffectUniforms) {
        let clean = self.sanitized();
        value.pixelate_size = clean.pixelate_size;
        value.rgb_split = clean.rgb_split;
        value.hue_shift = clean.hue_shift;
        value.saturation = clean.saturation;
        value.brightness = clean.brightness;
        value.contrast = clean.contrast;
        value.posterize = clean.posterize;
        value.invert = clean.invert;
        value.downsample = clean.downsample;
        value.grain_intensity = clean.grain_intensity;
        value.grain_size = clean.grain_size;
        value.grain_algo = clean.grain_algo;
        value.color_grain = clean.color_grain;
        value.breathe_scale = clean.breathe_scale;
        value.breathe_rotation = clean.breathe_rotation;
        value.breathe_position = clean.breathe_position;
        value.vignette = clean.vignette;
        value.color_drift = clean.color_drift;
        value.key_mode = clean.key_mode;
        value.key_threshold = clean.key_threshold;
        value.key_softness = clean.key_softness;
        value.key_color = clean.key_color;
        value.key_tolerance = clean.key_tolerance;
        value.cellular_amount = clean.cellular_amount;
        value.cellular_scale = clean.cellular_scale;
        value.cellular_warp = clean.cellular_warp;
        value.cellular_speed = clean.cellular_speed;
        value.cellular_gap_amount = clean.cellular_gap_amount;
        value.cellular_gap_threshold = clean.cellular_gap_threshold;
        value.cellular_gap_softness = clean.cellular_gap_softness;
        value.shift_amount = clean.shift_amount;
        value.shift_block_size = clean.shift_block_size;
        value.shift_density = clean.shift_density;
        value.shift_speed = clean.shift_speed;
    }

    fn interpolate(a: &Self, b: &Self, weights: [f32; 2], choose_b: bool) -> Self {
        let a = a.sanitized();
        let b = b.sanitized();
        Self {
            pixelate_size: blend_finite(a.pixelate_size, b.pixelate_size, weights),
            rgb_split: blend_finite(a.rgb_split, b.rgb_split, weights),
            hue_shift: blend_wrapped_degrees(a.hue_shift, b.hue_shift, weights),
            saturation: blend_finite(a.saturation, b.saturation, weights),
            brightness: blend_finite(a.brightness, b.brightness, weights),
            contrast: blend_finite(a.contrast, b.contrast, weights),
            posterize: blend_finite(a.posterize, b.posterize, weights),
            invert: pick_finite(a.invert, b.invert, choose_b),
            downsample: blend_finite(a.downsample, b.downsample, weights),
            grain_intensity: blend_finite(a.grain_intensity, b.grain_intensity, weights),
            grain_size: blend_finite(a.grain_size, b.grain_size, weights),
            grain_algo: pick_finite(a.grain_algo, b.grain_algo, choose_b),
            color_grain: pick_finite(a.color_grain, b.color_grain, choose_b),
            breathe_scale: blend_finite(a.breathe_scale, b.breathe_scale, weights),
            breathe_rotation: blend_finite(a.breathe_rotation, b.breathe_rotation, weights),
            breathe_position: blend_finite(a.breathe_position, b.breathe_position, weights),
            vignette: blend_finite(a.vignette, b.vignette, weights),
            color_drift: blend_finite(a.color_drift, b.color_drift, weights),
            key_mode: pick_finite(a.key_mode, b.key_mode, choose_b),
            key_threshold: blend_finite(a.key_threshold, b.key_threshold, weights),
            key_softness: blend_finite(a.key_softness, b.key_softness, weights),
            key_color: [
                blend_finite(a.key_color[0], b.key_color[0], weights),
                blend_finite(a.key_color[1], b.key_color[1], weights),
                blend_finite(a.key_color[2], b.key_color[2], weights),
            ],
            key_tolerance: blend_finite(a.key_tolerance, b.key_tolerance, weights),
            cellular_amount: blend_finite(a.cellular_amount, b.cellular_amount, weights),
            cellular_scale: blend_finite(a.cellular_scale, b.cellular_scale, weights),
            cellular_warp: blend_finite(a.cellular_warp, b.cellular_warp, weights),
            cellular_speed: blend_finite(a.cellular_speed, b.cellular_speed, weights),
            cellular_gap_amount: blend_finite(
                a.cellular_gap_amount,
                b.cellular_gap_amount,
                weights,
            ),
            cellular_gap_threshold: blend_finite(
                a.cellular_gap_threshold,
                b.cellular_gap_threshold,
                weights,
            ),
            cellular_gap_softness: blend_finite(
                a.cellular_gap_softness,
                b.cellular_gap_softness,
                weights,
            ),
            shift_amount: blend_finite(a.shift_amount, b.shift_amount, weights),
            shift_block_size: blend_finite(a.shift_block_size, b.shift_block_size, weights),
            shift_density: blend_finite(a.shift_density, b.shift_density, weights),
            shift_speed: blend_finite(a.shift_speed, b.shift_speed, weights),
        }
        .sanitized()
    }
}

/// Serializable NTSC values captured by a morph slot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MorphNtscSnapshot {
    pub enabled: bool,
    pub tape_speed: u32,
    pub chroma_loss: f32,
    pub edge_wave_enabled: bool,
    pub edge_wave_intensity: f32,
    pub edge_wave_speed: f32,
    pub head_switching_enabled: bool,
    pub head_switching_height: i32,
    pub head_switching_shift: f32,
    pub tracking_noise_enabled: bool,
    pub tracking_noise_height: i32,
    pub tracking_noise_wave: f32,
    pub tracking_noise_snow: f32,
    pub snow_intensity: f32,
    pub composite_noise_intensity: f32,
    pub luma_noise_intensity: f32,
    pub chroma_noise_intensity: f32,
    pub luma_smear: f32,
    pub composite_sharpening: f32,
}

impl Default for MorphNtscSnapshot {
    fn default() -> Self {
        Self::capture(&NtscParams::default())
    }
}

impl MorphNtscSnapshot {
    pub fn capture(value: &NtscParams) -> Self {
        Self {
            enabled: value.enabled,
            tape_speed: value.tape_speed,
            chroma_loss: value.chroma_loss,
            edge_wave_enabled: value.edge_wave_enabled,
            edge_wave_intensity: value.edge_wave_intensity,
            edge_wave_speed: value.edge_wave_speed,
            head_switching_enabled: value.head_switching_enabled,
            head_switching_height: value.head_switching_height,
            head_switching_shift: value.head_switching_shift,
            tracking_noise_enabled: value.tracking_noise_enabled,
            tracking_noise_height: value.tracking_noise_height,
            tracking_noise_wave: value.tracking_noise_wave,
            tracking_noise_snow: value.tracking_noise_snow,
            snow_intensity: value.snow_intensity,
            composite_noise_intensity: value.composite_noise_intensity,
            luma_noise_intensity: value.luma_noise_intensity,
            chroma_noise_intensity: value.chroma_noise_intensity,
            luma_smear: value.luma_smear,
            composite_sharpening: value.composite_sharpening,
        }
        .sanitized()
    }

    pub fn sanitized(&self) -> Self {
        Self {
            enabled: self.enabled,
            tape_speed: self.tape_speed.min(2),
            chroma_loss: finite_clamp(self.chroma_loss, 0.0, 0.0, 0.01),
            edge_wave_enabled: self.edge_wave_enabled,
            edge_wave_intensity: finite_clamp(self.edge_wave_intensity, 0.0, 0.0, 20.0),
            edge_wave_speed: finite_clamp(self.edge_wave_speed, 0.5, 0.0, 10.0),
            head_switching_enabled: self.head_switching_enabled,
            head_switching_height: self.head_switching_height.clamp(0, 24),
            head_switching_shift: finite_clamp(self.head_switching_shift, 0.0, -100.0, 100.0),
            tracking_noise_enabled: self.tracking_noise_enabled,
            tracking_noise_height: self.tracking_noise_height.clamp(0, 120),
            tracking_noise_wave: finite_clamp(self.tracking_noise_wave, 0.0, 0.0, 50.0),
            tracking_noise_snow: finite_clamp(self.tracking_noise_snow, 0.0, 0.0, 1.0),
            snow_intensity: finite_clamp(self.snow_intensity, 0.0, 0.0, 1.0),
            composite_noise_intensity: finite_clamp(self.composite_noise_intensity, 0.0, 0.0, 0.5),
            luma_noise_intensity: finite_clamp(self.luma_noise_intensity, 0.0, 0.0, 0.2),
            chroma_noise_intensity: finite_clamp(self.chroma_noise_intensity, 0.0, 0.0, 0.5),
            luma_smear: finite_clamp(self.luma_smear, 0.0, 0.0, 1.0),
            composite_sharpening: finite_clamp(self.composite_sharpening, 0.0, -1.0, 2.0),
        }
    }

    pub fn to_params(&self) -> NtscParams {
        let clean = self.sanitized();
        NtscParams {
            enabled: clean.enabled,
            tape_speed: clean.tape_speed,
            chroma_loss: clean.chroma_loss,
            edge_wave_enabled: clean.edge_wave_enabled,
            edge_wave_intensity: clean.edge_wave_intensity,
            edge_wave_speed: clean.edge_wave_speed,
            head_switching_enabled: clean.head_switching_enabled,
            head_switching_height: clean.head_switching_height,
            head_switching_shift: clean.head_switching_shift,
            tracking_noise_enabled: clean.tracking_noise_enabled,
            tracking_noise_height: clean.tracking_noise_height,
            tracking_noise_wave: clean.tracking_noise_wave,
            tracking_noise_snow: clean.tracking_noise_snow,
            snow_intensity: clean.snow_intensity,
            composite_noise_intensity: clean.composite_noise_intensity,
            luma_noise_intensity: clean.luma_noise_intensity,
            chroma_noise_intensity: clean.chroma_noise_intensity,
            luma_smear: clean.luma_smear,
            composite_sharpening: clean.composite_sharpening,
        }
    }

    fn interpolate(a: &Self, b: &Self, weights: [f32; 2], choose_b: bool) -> Self {
        let a = a.sanitized();
        let b = b.sanitized();
        Self {
            enabled: pick(a.enabled, b.enabled, choose_b),
            tape_speed: pick(a.tape_speed, b.tape_speed, choose_b),
            chroma_loss: blend_finite(a.chroma_loss, b.chroma_loss, weights),
            edge_wave_enabled: pick(a.edge_wave_enabled, b.edge_wave_enabled, choose_b),
            edge_wave_intensity: blend_finite(
                a.edge_wave_intensity,
                b.edge_wave_intensity,
                weights,
            ),
            edge_wave_speed: blend_finite(a.edge_wave_speed, b.edge_wave_speed, weights),
            head_switching_enabled: pick(
                a.head_switching_enabled,
                b.head_switching_enabled,
                choose_b,
            ),
            head_switching_height: blend_i32(
                a.head_switching_height,
                b.head_switching_height,
                weights,
            ),
            head_switching_shift: blend_finite(
                a.head_switching_shift,
                b.head_switching_shift,
                weights,
            ),
            tracking_noise_enabled: pick(
                a.tracking_noise_enabled,
                b.tracking_noise_enabled,
                choose_b,
            ),
            tracking_noise_height: blend_i32(
                a.tracking_noise_height,
                b.tracking_noise_height,
                weights,
            ),
            tracking_noise_wave: blend_finite(
                a.tracking_noise_wave,
                b.tracking_noise_wave,
                weights,
            ),
            tracking_noise_snow: blend_finite(
                a.tracking_noise_snow,
                b.tracking_noise_snow,
                weights,
            ),
            snow_intensity: blend_finite(a.snow_intensity, b.snow_intensity, weights),
            composite_noise_intensity: blend_finite(
                a.composite_noise_intensity,
                b.composite_noise_intensity,
                weights,
            ),
            luma_noise_intensity: blend_finite(
                a.luma_noise_intensity,
                b.luma_noise_intensity,
                weights,
            ),
            chroma_noise_intensity: blend_finite(
                a.chroma_noise_intensity,
                b.chroma_noise_intensity,
                weights,
            ),
            luma_smear: blend_finite(a.luma_smear, b.luma_smear, weights),
            composite_sharpening: blend_finite(
                a.composite_sharpening,
                b.composite_sharpening,
                weights,
            ),
        }
        .sanitized()
    }
}

/// Serializable temporal values captured by a morph slot.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MorphTemporalSnapshot {
    pub feedback: f32,
    pub fb_zoom: f32,
    pub fb_rotate: f32,
    pub slitscan: f32,
    pub slit_axis: f32,
    pub slit_angle: f32,
    pub key_mode: f32,
    pub key_threshold: f32,
    pub key_softness: f32,
    pub key_history: f32,
}

impl Default for MorphTemporalSnapshot {
    fn default() -> Self {
        Self::capture(&TemporalParams::default())
    }
}

impl MorphTemporalSnapshot {
    pub fn capture(value: &TemporalParams) -> Self {
        Self {
            feedback: value.feedback,
            fb_zoom: value.fb_zoom,
            fb_rotate: value.fb_rotate,
            slitscan: value.slitscan,
            slit_axis: value.slit_axis,
            slit_angle: value.slit_angle,
            key_mode: value.key_mode,
            key_threshold: value.key_threshold,
            key_softness: value.key_softness,
            key_history: value.key_history,
        }
        .sanitized()
    }

    pub fn sanitized(&self) -> Self {
        Self {
            feedback: finite_clamp(self.feedback, 0.0, 0.0, 0.95),
            fb_zoom: finite_clamp(self.fb_zoom, 1.0, 0.9, 1.1),
            fb_rotate: finite_clamp(self.fb_rotate, 0.0, -5.0, 5.0),
            slitscan: finite_clamp(self.slitscan, 0.0, 0.0, 1.0),
            slit_axis: finite_clamp(self.slit_axis, 0.0, 0.0, 1.0),
            slit_angle: finite_clamp(self.slit_angle, 0.0, -180.0, 180.0),
            key_mode: discrete_f32(self.key_mode, 0.0, 4.0),
            key_threshold: finite_clamp(self.key_threshold, 0.1, 0.0, 1.0),
            key_softness: finite_clamp(self.key_softness, 0.03, 0.0, 0.5),
            key_history: finite_clamp(self.key_history, 1.0, 1.0, 23.0).round(),
        }
    }

    pub fn to_params(self) -> TemporalParams {
        let clean = self.sanitized();
        TemporalParams {
            feedback: clean.feedback,
            fb_zoom: clean.fb_zoom,
            fb_rotate: clean.fb_rotate,
            slitscan: clean.slitscan,
            slit_axis: clean.slit_axis,
            slit_angle: clean.slit_angle,
            key_mode: clean.key_mode,
            key_threshold: clean.key_threshold,
            key_softness: clean.key_softness,
            key_history: clean.key_history,
        }
    }

    fn interpolate(a: &Self, b: &Self, weights: [f32; 2], choose_b: bool) -> Self {
        let a = a.sanitized();
        let b = b.sanitized();
        Self {
            feedback: blend_finite(a.feedback, b.feedback, weights),
            fb_zoom: blend_finite(a.fb_zoom, b.fb_zoom, weights),
            fb_rotate: blend_finite(a.fb_rotate, b.fb_rotate, weights),
            slitscan: blend_finite(a.slitscan, b.slitscan, weights),
            slit_axis: pick_finite(a.slit_axis, b.slit_axis, choose_b),
            slit_angle: blend_wrapped_degrees(a.slit_angle, b.slit_angle, weights),
            key_mode: pick_finite(a.key_mode, b.key_mode, choose_b),
            key_threshold: blend_finite(a.key_threshold, b.key_threshold, weights),
            key_softness: blend_finite(a.key_softness, b.key_softness, weights),
            key_history: blend_finite(a.key_history, b.key_history, weights),
        }
        .sanitized()
    }
}

/// Stable serialized representation of the renderer's layer blend modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MorphLayerBlendMode {
    #[default]
    Normal,
    Screen,
    Multiply,
    Difference,
}

impl MorphLayerBlendMode {
    fn capture(value: BlendMode) -> Self {
        match value {
            BlendMode::Normal => Self::Normal,
            BlendMode::Screen => Self::Screen,
            BlendMode::Multiply => Self::Multiply,
            BlendMode::Difference => Self::Difference,
        }
    }

    pub(crate) fn to_blend_mode(self) -> BlendMode {
        match self {
            Self::Normal => BlendMode::Normal,
            Self::Screen => BlendMode::Screen,
            Self::Multiply => BlendMode::Multiply,
            Self::Difference => BlendMode::Difference,
        }
    }
}

/// Per-layer values are explicitly keyed by stack position for now. This
/// makes the positional identity assumption visible to future persistence and
/// exporter code instead of hiding it in a vector index.
///
/// Fields introduced after the original opacity/speed/key format are optional
/// on disk. A legacy slot therefore continues to control exactly the values it
/// captured rather than resetting every newly supported parameter to defaults.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LayerMorphSnapshot {
    pub position: usize,
    pub opacity: f32,
    pub speed: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effects: Option<MorphMasterSnapshot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blend_mode: Option<MorphLayerBlendMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visible: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paused: Option<bool>,
    /// Discrete per-layer master-shader bypass. Optional so morph slots
    /// captured by older builds leave the live value untouched.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bypass_master_fx: Option<bool>,
    /// Compatibility field for pre-full-state snapshots. Newly captured
    /// snapshots store this value inside `effects` and omit this duplicate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_threshold: Option<f32>,
}

/// One independently optional field family in a persisted layer capture.
/// Opacity and speed existed in the original schema; newer fields must only
/// claim editor ownership when both engaged slots actually contain them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayerMorphControl {
    Opacity,
    Speed,
    Fps,
    Effects,
    AnyEffect,
    KeyThreshold,
    BlendMode,
    Visible,
    Paused,
    BypassMasterFx,
}

impl Default for LayerMorphSnapshot {
    fn default() -> Self {
        Self {
            position: 0,
            opacity: 1.0,
            speed: 1.0,
            fps: None,
            effects: None,
            blend_mode: None,
            visible: None,
            paused: None,
            bypass_master_fx: None,
            key_threshold: None,
        }
    }
}

impl LayerMorphSnapshot {
    fn capture(position: usize, layer: &Layer) -> Self {
        Self {
            position,
            opacity: layer.opacity,
            speed: layer.speed,
            fps: Some(layer.fps),
            effects: Some(MorphMasterSnapshot::capture(&layer.effects)),
            blend_mode: Some(MorphLayerBlendMode::capture(layer.blend_mode)),
            visible: Some(layer.visible),
            paused: Some(layer.paused),
            bypass_master_fx: Some(layer.bypass_master_fx),
            key_threshold: None,
        }
        .sanitized()
    }

    fn sanitized(&self) -> Self {
        let legacy_key = self
            .key_threshold
            .map(|value| finite_clamp(value, 0.5, 0.0, 1.0));
        let mut effects = self.effects.map(|value| value.sanitized());
        if let (Some(effects), Some(key_threshold)) = (&mut effects, legacy_key) {
            effects.key_threshold = key_threshold;
        }
        let has_effects = effects.is_some();
        Self {
            position: self.position,
            opacity: finite_clamp(self.opacity, 1.0, 0.0, 1.0),
            speed: finite_clamp(self.speed, 1.0, 0.25, 4.0),
            fps: self.fps.map(|value| finite_clamp(value, 30.0, 1.0, 240.0)),
            effects,
            blend_mode: self.blend_mode,
            visible: self.visible,
            paused: self.paused,
            bypass_master_fx: self.bypass_master_fx,
            key_threshold: if has_effects { None } else { legacy_key },
        }
    }

    fn effective_key_threshold(&self) -> Option<f32> {
        self.key_threshold
            .or_else(|| self.effects.as_ref().map(|effects| effects.key_threshold))
    }

    fn interpolate(a: &Self, b: &Self, weights: [f32; 2], choose_b: bool) -> Self {
        let a = a.sanitized();
        let b = b.sanitized();
        let effects = match (a.effects, b.effects) {
            (Some(a), Some(b)) => Some(MorphMasterSnapshot::interpolate(&a, &b, weights, choose_b)),
            _ => None,
        };
        let key_threshold = if effects.is_some() {
            None
        } else {
            match (a.effective_key_threshold(), b.effective_key_threshold()) {
                (Some(a), Some(b)) => {
                    Some(finite_clamp(blend_finite(a, b, weights), 0.5, 0.0, 1.0))
                }
                _ => None,
            }
        };
        Self {
            position: a.position,
            opacity: finite_clamp(blend_finite(a.opacity, b.opacity, weights), 1.0, 0.0, 1.0),
            speed: finite_clamp(blend_finite(a.speed, b.speed, weights), 1.0, 0.25, 4.0),
            fps: match (a.fps, b.fps) {
                (Some(a), Some(b)) => {
                    Some(finite_clamp(blend_finite(a, b, weights), 30.0, 1.0, 240.0))
                }
                _ => None,
            },
            effects,
            blend_mode: match (a.blend_mode, b.blend_mode) {
                (Some(a), Some(b)) => Some(pick(a, b, choose_b)),
                _ => None,
            },
            visible: match (a.visible, b.visible) {
                (Some(a), Some(b)) => Some(pick(a, b, choose_b)),
                _ => None,
            },
            paused: match (a.paused, b.paused) {
                (Some(a), Some(b)) => Some(pick(a, b, choose_b)),
                _ => None,
            },
            bypass_master_fx: match (a.bypass_master_fx, b.bypass_master_fx) {
                (Some(a), Some(b)) => Some(pick(a, b, choose_b)),
                _ => None,
            },
            key_threshold,
        }
    }

    fn apply_to(&self, layer: &mut Layer) {
        let clean = self.sanitized();
        layer.opacity = clean.opacity;
        layer.speed = clean.speed;
        if let Some(fps) = clean.fps {
            layer.fps = fps;
        }
        if let Some(effects) = clean.effects {
            effects.apply_to(&mut layer.effects);
        }
        if let Some(key_threshold) = clean.key_threshold {
            layer.effects.key_threshold = key_threshold;
        }
        if let Some(blend_mode) = clean.blend_mode {
            layer.blend_mode = blend_mode.to_blend_mode();
        }
        if let Some(visible) = clean.visible {
            layer.visible = visible;
        }
        if let Some(paused) = clean.paused {
            layer.paused = paused;
        }
        if let Some(bypass_master_fx) = clean.bypass_master_fx {
            layer.bypass_master_fx = bypass_master_fx;
        }
    }
}

/// One serializable A or B morph slot. This is the neutral persistence type;
/// it has no dependency on `patch` and no runtime layer handles.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct MorphSlot {
    pub master: MorphMasterSnapshot,
    pub ntsc: MorphNtscSnapshot,
    pub temporal: MorphTemporalSnapshot,
    pub layers: Vec<LayerMorphSnapshot>,
}

impl MorphSlot {
    pub fn capture(
        master: &EffectUniforms,
        ntsc: &NtscParams,
        temporal: &TemporalParams,
        layers: &[Layer],
    ) -> Self {
        Self {
            master: MorphMasterSnapshot::capture(master),
            ntsc: MorphNtscSnapshot::capture(ntsc),
            temporal: MorphTemporalSnapshot::capture(temporal),
            layers: layers
                .iter()
                .enumerate()
                .map(|(position, layer)| LayerMorphSnapshot::capture(position, layer))
                .collect(),
        }
        .sanitized()
    }

    pub fn sanitized(&self) -> Self {
        let mut positions = HashSet::with_capacity(self.layers.len());
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            if !positions.insert(layer.position) {
                continue;
            }
            layers.push(layer.sanitized());
        }
        Self {
            master: self.master.sanitized(),
            ntsc: self.ntsc.sanitized(),
            temporal: self.temporal.sanitized(),
            layers,
        }
    }

    /// Keep positional snapshots aligned after removing one live layer while
    /// preserving the independent master/NTSC/temporal worlds.
    fn remap_layers_after_remove(&mut self, removed: usize) {
        self.layers.retain_mut(|layer| {
            if layer.position == removed {
                return false;
            }
            if layer.position > removed {
                layer.position -= 1;
            }
            true
        });
    }

    /// Apply the same stable stack permutation as the live layer move.
    fn remap_layers_after_move(&mut self, from: usize, to: usize) {
        if from == to {
            return;
        }
        for layer in &mut self.layers {
            layer.position = if layer.position == from {
                to
            } else if from < to && layer.position > from && layer.position <= to {
                layer.position - 1
            } else if to < from && layer.position >= to && layer.position < from {
                layer.position + 1
            } else {
                layer.position
            };
        }
    }
}

/// A pure, detached morph result suitable for either live or offline use.
#[derive(Debug, Clone, PartialEq)]
pub struct MorphSample {
    pub master: MorphMasterSnapshot,
    pub ntsc: MorphNtscSnapshot,
    pub temporal: MorphTemporalSnapshot,
    pub layers: Vec<LayerMorphSnapshot>,
}

impl MorphSample {
    /// Compatibility adapter for the live renderer. Runtime master fields and
    /// layers missing from either morph slot are left untouched.
    pub fn apply_to(
        &self,
        master: &mut EffectUniforms,
        ntsc: &mut NtscParams,
        temporal: &mut TemporalParams,
        layers: &mut [Layer],
    ) {
        self.master.apply_to(master);
        *ntsc = self.ntsc.to_params();
        *temporal = self.temporal.to_params();
        for sampled in &self.layers {
            let Some(layer) = layers.get_mut(sampled.position) else {
                continue;
            };
            sampled.apply_to(layer);
        }
    }
}

/// Serializable runtime state for patch persistence. Patch code can embed
/// this type directly or convert it into a versioned wrapper of its own.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct MorphStateSnapshot {
    pub a: Option<MorphSlot>,
    pub b: Option<MorphSlot>,
    pub t: f32,
    pub blend_law: MorphBlendLaw,
    pub glide: Option<MorphGlide>,
}

impl MorphStateSnapshot {
    /// Capture at an explicit beat and store only the remaining movement.
    /// The serialized glide starts at beat zero so it is independent of the
    /// live clock's old anchor.
    pub fn from_morph_at_beat(morph: &Morph, beat: f64) -> Self {
        let beat = finite_f64_or(beat, 0.0);
        let t = morph.position_at_beat(beat);
        let glide = morph.glide.map(MorphGlide::sanitized).and_then(|glide| {
            if glide.is_complete_at(beat) || positions_equal(t, glide.target) {
                None
            } else {
                let end_beat = glide.start_beat + glide.duration_beats;
                let sampled_beat = beat.clamp(glide.start_beat, end_beat);
                let remaining = (end_beat - sampled_beat).max(0.0);
                (remaining > 0.0)
                    .then(|| MorphGlide::with_remaining(t, glide.target, 0.0, remaining))
            }
        });
        Self {
            a: morph.a.as_ref().map(MorphSlot::sanitized),
            b: morph.b.as_ref().map(MorphSlot::sanitized),
            t,
            blend_law: morph.blend_law,
            glide,
        }
    }

    pub fn into_morph(self) -> Morph {
        self.into_morph_at_beat(0.0)
    }

    /// Restore a relative snapshot against the caller's current clock. Old
    /// snapshots that contain absolute glide beats are also normalized from
    /// their persisted `t`, so they no longer rewind to the old glide start.
    pub fn into_morph_at_beat(self, beat: f64) -> Morph {
        let t = normalized_position(self.t);
        let beat = finite_f64_or(beat, 0.0);
        Morph {
            a: self.a.map(|slot| slot.sanitized()),
            b: self.b.map(|slot| slot.sanitized()),
            t,
            blend_law: self.blend_law,
            glide: self
                .glide
                .map(MorphGlide::sanitized)
                .and_then(|glide| rebase_glide_from_position(glide, t, beat)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Morph {
    pub a: Option<MorphSlot>,
    pub b: Option<MorphSlot>,
    /// Manual crossfader position, 0 = A, 1 = B.
    pub t: f32,
    pub blend_law: MorphBlendLaw,
    pub glide: Option<MorphGlide>,
}

impl Default for Morph {
    fn default() -> Self {
        Self {
            a: None,
            b: None,
            t: 0.0,
            blend_law: MorphBlendLaw::Linear,
            glide: None,
        }
    }
}

impl Morph {
    pub fn active(&self) -> bool {
        self.a.is_some() && self.b.is_some()
    }

    /// Whether the engaged A/B pair owns a particular layer field at this
    /// stack position. Newly appended layers remain directly editable, and a
    /// legacy slot cannot claim fields that did not yet exist when captured.
    pub(crate) fn controls_layer_field(&self, position: usize, control: LayerMorphControl) -> bool {
        let (Some(a), Some(b)) = (&self.a, &self.b) else {
            return false;
        };
        let Some(a) = a.layers.iter().find(|layer| layer.position == position) else {
            return false;
        };
        let Some(b) = b.layers.iter().find(|layer| layer.position == position) else {
            return false;
        };
        match control {
            LayerMorphControl::Opacity | LayerMorphControl::Speed => true,
            LayerMorphControl::Fps => a.fps.is_some() && b.fps.is_some(),
            LayerMorphControl::Effects => a.effects.is_some() && b.effects.is_some(),
            LayerMorphControl::AnyEffect => {
                (a.effects.is_some() && b.effects.is_some())
                    || (a.effective_key_threshold().is_some()
                        && b.effective_key_threshold().is_some())
            }
            LayerMorphControl::KeyThreshold => {
                a.effective_key_threshold().is_some() && b.effective_key_threshold().is_some()
            }
            LayerMorphControl::BlendMode => a.blend_mode.is_some() && b.blend_mode.is_some(),
            LayerMorphControl::Visible => a.visible.is_some() && b.visible.is_some(),
            LayerMorphControl::Paused => a.paused.is_some() && b.paused.is_some(),
            LayerMorphControl::BypassMasterFx => {
                a.bypass_master_fx.is_some() && b.bypass_master_fx.is_some()
            }
        }
    }

    pub fn clear(&mut self) {
        self.a = None;
        self.b = None;
        self.glide = None;
    }

    /// Adding at the end leaves every captured position valid; no slot change
    /// is required. The new layer is deliberately untouched by the morph.
    pub fn remap_layers_after_remove(&mut self, removed: usize) {
        if let Some(slot) = &mut self.a {
            slot.remap_layers_after_remove(removed);
        }
        if let Some(slot) = &mut self.b {
            slot.remap_layers_after_remove(removed);
        }
    }

    pub fn remap_layers_after_move(&mut self, from: usize, to: usize) {
        if let Some(slot) = &mut self.a {
            slot.remap_layers_after_move(from, to);
        }
        if let Some(slot) = &mut self.b {
            slot.remap_layers_after_move(from, to);
        }
    }

    /// Set a manual position and cancel any automatic movement.
    pub fn set_position(&mut self, t: f32) {
        self.t = normalized_position(t);
        self.glide = None;
    }

    /// Begin a deterministic glide from the position evaluated at
    /// `start_beat`. Starting a new glide during an old one is continuous.
    pub fn start_glide(&mut self, target: f32, duration_beats: f64, start_beat: f64) {
        let start = self.position_at_beat(start_beat);
        let glide = MorphGlide::new(start, target, start_beat, duration_beats);
        self.t = start;
        if glide.duration_beats <= 0.0 {
            self.t = glide.target;
            self.glide = None;
        } else {
            self.glide = Some(glide);
        }
    }

    /// Return the manual or gliding crossfader position at an explicit beat.
    pub fn position_at_beat(&self, beat: f64) -> f32 {
        self.glide
            .map(|glide| glide.position_at(beat))
            .unwrap_or_else(|| normalized_position(self.t))
    }

    /// Commit the current glide position into `t`. A completed glide is
    /// removed; an in-flight glide remains deterministic and active.
    pub fn settle_glide_at(&mut self, beat: f64) -> f32 {
        let position = self.position_at_beat(beat);
        self.t = position;
        if self.glide.is_some_and(|glide| glide.is_complete_at(beat)) {
            self.glide = None;
        }
        position
    }

    /// Purely sample the A/B snapshots at an explicit crossfader position.
    pub fn sample(&self, t: f32) -> Option<MorphSample> {
        let (Some(a), Some(b)) = (&self.a, &self.b) else {
            return None;
        };
        let a = a.sanitized();
        let b = b.sanitized();
        let t = normalized_position(t);
        let weights = self.blend_law.weights(t);
        let choose_b = t >= 0.5;

        let layers = a
            .layers
            .iter()
            .filter_map(|layer_a| {
                let layer_b = b
                    .layers
                    .iter()
                    .find(|candidate| candidate.position == layer_a.position)?;
                Some(LayerMorphSnapshot::interpolate(
                    layer_a, layer_b, weights, choose_b,
                ))
            })
            .collect();

        Some(MorphSample {
            master: MorphMasterSnapshot::interpolate(&a.master, &b.master, weights, choose_b),
            ntsc: MorphNtscSnapshot::interpolate(&a.ntsc, &b.ntsc, weights, choose_b),
            temporal: MorphTemporalSnapshot::interpolate(
                &a.temporal,
                &b.temporal,
                weights,
                choose_b,
            ),
            layers,
        })
    }

    /// Write a sampled state into live base parameters. This preserves the
    /// pre-existing API while sharing the exact interpolation path with the
    /// exporter-facing pure sampler.
    pub fn apply(
        &self,
        t: f32,
        master: &mut EffectUniforms,
        ntsc: &mut NtscParams,
        temporal: &mut TemporalParams,
        layers: &mut [Layer],
    ) {
        if let Some(sample) = self.sample(t) {
            sample.apply_to(master, ntsc, temporal, layers);
        }
    }

    /// Persist this state at an authoritative beat while rebasing any
    /// in-flight glide to a zero-based clock and retaining only its remaining
    /// duration.
    pub fn snapshot_at_beat(&self, beat: f64) -> MorphStateSnapshot {
        MorphStateSnapshot::from_morph_at_beat(self, beat)
    }

    pub fn from_snapshot(snapshot: MorphStateSnapshot) -> Self {
        snapshot.into_morph()
    }

    /// Restore a relative persisted glide at the caller's current beat.
    pub fn from_snapshot_at_beat(snapshot: MorphStateSnapshot, beat: f64) -> Self {
        snapshot.into_morph_at_beat(beat)
    }
}

fn normalized_position(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn finite_f64_or(value: f64, fallback: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

fn positions_equal(a: f32, b: f32) -> bool {
    (normalized_position(a) - normalized_position(b)).abs() <= f32::EPSILON
}

/// Convert either a legacy absolute glide or an already-relative glide into
/// a movement beginning at `new_start_beat`. The persisted `position` is the
/// source of truth because live code settles it every frame before capture.
fn rebase_glide_from_position(
    glide: MorphGlide,
    position: f32,
    new_start_beat: f64,
) -> Option<MorphGlide> {
    let glide = glide.sanitized();
    let position = normalized_position(position);
    if glide.duration_beats <= 0.0 || positions_equal(position, glide.target) {
        return None;
    }

    let distance = (glide.target - glide.start).abs() as f64;
    let remaining_distance = (glide.target - position).abs() as f64;
    let remaining = if distance > f32::EPSILON as f64 {
        glide.duration_beats * (remaining_distance / distance).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (remaining > 0.0)
        .then(|| MorphGlide::with_remaining(position, glide.target, new_start_beat, remaining))
}

fn lerp(a: f32, b: f32, weights: [f32; 2]) -> f32 {
    a * weights[0] + b * weights[1]
}

fn blend_finite(a: f32, b: f32, weights: [f32; 2]) -> f32 {
    match (a.is_finite(), b.is_finite()) {
        (true, true) => lerp(a, b, weights),
        (true, false) => a,
        (false, true) => b,
        (false, false) => 0.0,
    }
}

/// Interpolate an orientation in degrees over the shortest signed arc. Exact
/// endpoint selection preserves the captured representation at A and B.
fn blend_wrapped_degrees(a: f32, b: f32, weights: [f32; 2]) -> f32 {
    if weights[1] <= 0.0 {
        return a;
    }
    if weights[0] <= 0.0 {
        return b;
    }
    if !a.is_finite() || !b.is_finite() {
        return blend_finite(a, b, weights);
    }
    let direct_delta = b - a;
    // At the exactly-opposite 180-degree tie, retain the direction encoded by
    // the captured endpoints. This is deterministic and avoids surprising a
    // legacy 0 -> 180 morph by sending it through -90 at its midpoint.
    let delta = if direct_delta.abs() <= 180.0 {
        direct_delta
    } else {
        (direct_delta + 180.0).rem_euclid(360.0) - 180.0
    };
    // `weights` are complementary but EqualPower is nonlinear in t. Recover
    // B's normalized share explicitly so the wrap path honors either law.
    let sum = weights[0] + weights[1];
    let progress = if sum.is_finite() && sum > f32::EPSILON {
        weights[1] / sum
    } else {
        0.0
    };
    (a + delta * progress + 180.0).rem_euclid(360.0) - 180.0
}

fn finite_clamp(value: f32, fallback: f32, min: f32, max: f32) -> f32 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        fallback.clamp(min, max)
    }
}

fn discrete_f32(value: f32, fallback: f32, max: f32) -> f32 {
    finite_clamp(value, fallback, 0.0, max).round()
}

fn blend_i32(a: i32, b: i32, weights: [f32; 2]) -> i32 {
    let value = a as f64 * weights[0] as f64 + b as f64 * weights[1] as f64;
    value.round().clamp(i32::MIN as f64, i32::MAX as f64) as i32
}

fn pick<T: Copy>(a: T, b: T, choose_b: bool) -> T {
    if choose_b {
        b
    } else {
        a
    }
}

fn pick_finite(a: f32, b: f32, choose_b: bool) -> f32 {
    let (selected, fallback) = if choose_b { (b, a) } else { (a, b) };
    if selected.is_finite() {
        selected
    } else if fallback.is_finite() {
        fallback
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) {
        assert!((a - b).abs() < 1.0e-5, "{a} != {b}");
    }

    fn slot(pixelate: f32, invert: f32, opacity: f32) -> MorphSlot {
        let mut slot = MorphSlot::default();
        slot.master.pixelate_size = pixelate;
        slot.master.invert = invert;
        slot.layers = vec![LayerMorphSnapshot {
            position: 3,
            opacity,
            speed: opacity + 1.0,
            key_threshold: Some(opacity / 2.0),
            ..Default::default()
        }];
        slot
    }

    #[test]
    fn layer_ownership_requires_the_position_in_both_engaged_slots() {
        let a = slot(1.0, 0.0, 0.25);
        let mut b = slot(2.0, 1.0, 0.75);
        let mut morph = Morph {
            a: Some(a),
            b: Some(b.clone()),
            ..Default::default()
        };
        assert!(morph.controls_layer_field(3, LayerMorphControl::Opacity));
        assert!(morph.controls_layer_field(3, LayerMorphControl::Speed));
        assert!(morph.controls_layer_field(3, LayerMorphControl::KeyThreshold));
        assert!(morph.controls_layer_field(3, LayerMorphControl::AnyEffect));
        assert!(!morph.controls_layer_field(3, LayerMorphControl::Fps));
        assert!(!morph.controls_layer_field(3, LayerMorphControl::Effects));
        assert!(!morph.controls_layer_field(3, LayerMorphControl::Visible));
        assert!(!morph.controls_layer_field(3, LayerMorphControl::Paused));
        assert!(!morph.controls_layer_field(3, LayerMorphControl::BypassMasterFx));
        assert!(!morph.controls_layer_field(4, LayerMorphControl::Opacity));

        b.layers[0].position = 4;
        morph.b = Some(b);
        assert!(!morph.controls_layer_field(3, LayerMorphControl::Opacity));
        assert!(!morph.controls_layer_field(4, LayerMorphControl::Opacity));

        morph.b = None;
        assert!(!morph.controls_layer_field(3, LayerMorphControl::Opacity));
    }

    fn full_layer(position: usize, high: bool) -> LayerMorphSnapshot {
        let effects = if high {
            MorphMasterSnapshot {
                pixelate_size: 32.0,
                rgb_split: 30.0,
                hue_shift: 180.0,
                saturation: 1.0,
                brightness: 1.0,
                contrast: 1.0,
                posterize: 16.0,
                invert: 1.0,
                downsample: 0.05,
                grain_intensity: 0.3,
                grain_size: 4.0,
                grain_algo: 3.0,
                color_grain: 1.0,
                breathe_scale: 0.05,
                breathe_rotation: 2.0,
                breathe_position: 0.02,
                vignette: 1.5,
                color_drift: 0.02,
                key_mode: 2.0,
                key_threshold: 1.0,
                key_softness: 0.5,
                key_color: [1.0, 0.0, 1.0],
                key_tolerance: 1.0,
                cellular_amount: 1.0,
                cellular_scale: 32.0,
                cellular_warp: 1.0,
                cellular_speed: 2.0,
                cellular_gap_amount: 1.0,
                cellular_gap_threshold: 1.0,
                cellular_gap_softness: 0.5,
                shift_amount: 1.0,
                shift_block_size: 256.0,
                shift_density: 1.0,
                shift_speed: 20.0,
            }
        } else {
            MorphMasterSnapshot::default()
        };
        LayerMorphSnapshot {
            position,
            opacity: if high { 1.0 } else { 0.0 },
            speed: if high { 4.0 } else { 0.25 },
            fps: Some(if high { 240.0 } else { 1.0 }),
            effects: Some(effects),
            blend_mode: Some(if high {
                MorphLayerBlendMode::Difference
            } else {
                MorphLayerBlendMode::Multiply
            }),
            visible: Some(high),
            paused: Some(high),
            bypass_master_fx: Some(high),
            key_threshold: None,
        }
    }

    #[test]
    fn linear_sampling_lerps_continuous_and_switches_discrete() {
        let mut a = slot(1.0, 0.0, 0.0);
        a.ntsc.enabled = false;
        let mut b = slot(31.0, 1.0, 1.0);
        b.ntsc.enabled = true;
        b.ntsc.snow_intensity = 1.0;
        let morph = Morph {
            a: Some(a),
            b: Some(b),
            ..Default::default()
        };

        let midpoint = morph.sample(0.5).unwrap();
        close(midpoint.master.pixelate_size, 16.0);
        assert_eq!(midpoint.master.invert, 1.0);
        assert!(midpoint.ntsc.enabled);
        close(midpoint.ntsc.snow_intensity, 0.5);
        assert_eq!(midpoint.layers[0].position, 3);
        close(midpoint.layers[0].opacity, 0.5);

        let quarter = morph.sample(0.25).unwrap();
        close(quarter.master.pixelate_size, 8.5);
        assert_eq!(quarter.master.invert, 0.0);
        assert!(!quarter.ntsc.enabled);
    }

    #[test]
    fn equal_power_weights_are_complementary_and_preserve_equal_values() {
        for t in [0.0, 0.1, 0.5, 0.9, 1.0] {
            let [a, b] = MorphBlendLaw::EqualPower.weights(t);
            close(a + b, 1.0);
        }

        let morph = Morph {
            a: Some(slot(7.0, 0.0, 0.4)),
            b: Some(slot(7.0, 0.0, 0.4)),
            blend_law: MorphBlendLaw::EqualPower,
            ..Default::default()
        };
        for t in [0.0, 0.2, 0.5, 0.8, 1.0] {
            let sample = morph.sample(t).unwrap();
            close(sample.master.pixelate_size, 7.0);
            close(sample.layers[0].opacity, 0.4);
        }
    }

    #[test]
    fn layers_match_by_explicit_position_and_unmatched_layers_are_omitted() {
        let a = MorphSlot {
            layers: vec![
                LayerMorphSnapshot {
                    position: 7,
                    opacity: 0.0,
                    ..Default::default()
                },
                LayerMorphSnapshot {
                    position: 2,
                    opacity: 0.2,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let b = MorphSlot {
            layers: vec![
                LayerMorphSnapshot {
                    position: 2,
                    opacity: 0.8,
                    ..Default::default()
                },
                LayerMorphSnapshot {
                    position: 9,
                    opacity: 1.0,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let morph = Morph {
            a: Some(a),
            b: Some(b),
            ..Default::default()
        };

        let sample = morph.sample(0.5).unwrap();
        assert_eq!(sample.layers.len(), 1);
        assert_eq!(sample.layers[0].position, 2);
        close(sample.layers[0].opacity, 0.5);
    }

    #[test]
    fn full_layer_state_interpolates_and_switches_discrete_values_at_midpoint() {
        let a = MorphSlot {
            layers: vec![full_layer(2, false)],
            ..Default::default()
        };
        let b = MorphSlot {
            layers: vec![full_layer(2, true)],
            ..Default::default()
        };
        let morph = Morph {
            a: Some(a),
            b: Some(b),
            ..Default::default()
        };

        let before = morph.sample(0.499).unwrap().layers.remove(0);
        assert_eq!(before.blend_mode, Some(MorphLayerBlendMode::Multiply));
        assert_eq!(before.visible, Some(false));
        assert_eq!(before.paused, Some(false));
        assert_eq!(before.bypass_master_fx, Some(false));

        let midpoint = morph.sample(0.5).unwrap().layers.remove(0);
        close(midpoint.opacity, 0.5);
        close(midpoint.speed, 2.125);
        close(midpoint.fps.unwrap(), 120.5);
        assert_eq!(midpoint.blend_mode, Some(MorphLayerBlendMode::Difference));
        assert_eq!(midpoint.visible, Some(true));
        assert_eq!(midpoint.paused, Some(true));
        assert_eq!(midpoint.bypass_master_fx, Some(true));
        let effects = midpoint.effects.unwrap();
        close(effects.pixelate_size, 16.5);
        close(effects.rgb_split, 15.0);
        close(effects.hue_shift, 90.0);
        close(effects.saturation, 0.5);
        close(effects.brightness, 0.5);
        close(effects.contrast, 0.5);
        close(effects.posterize, 8.0);
        assert_eq!(effects.invert, 1.0);
        close(effects.downsample, 0.525);
        close(effects.grain_intensity, 0.15);
        close(effects.grain_size, 2.5);
        assert_eq!(effects.grain_algo, 3.0);
        assert_eq!(effects.color_grain, 1.0);
        close(effects.breathe_scale, 0.025);
        close(effects.breathe_rotation, 1.0);
        close(effects.breathe_position, 0.01);
        close(effects.vignette, 0.75);
        close(effects.color_drift, 0.01);
        assert_eq!(effects.key_mode, 2.0);
        close(effects.key_threshold, 0.75);
        close(effects.key_softness, 0.3);
        close(effects.cellular_amount, 0.5);
        close(effects.cellular_scale, 21.0);
        close(effects.cellular_warp, 0.675);
        close(effects.cellular_speed, 1.125);
        close(effects.cellular_gap_amount, 0.5);
        close(effects.cellular_gap_threshold, 0.825);
        close(effects.cellular_gap_softness, 0.29);
        close(effects.shift_amount, 0.5);
        close(effects.shift_block_size, 132.0);
        close(effects.shift_density, 0.75);
        close(effects.shift_speed, 11.5);

        assert_eq!(morph.sample(0.0).unwrap().layers[0], full_layer(2, false));
        assert_eq!(morph.sample(1.0).unwrap().layers[0], full_layer(2, true));
    }

    #[test]
    fn legacy_layer_yaml_keeps_new_fields_absent_and_morphs_old_key() {
        let yaml = r#"
position: 4
opacity: 0.25
speed: 1.5
key_threshold: 0.2
"#;
        let legacy: LayerMorphSnapshot = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(legacy.fps, None);
        assert_eq!(legacy.effects, None);
        assert_eq!(legacy.blend_mode, None);
        assert_eq!(legacy.visible, None);
        assert_eq!(legacy.paused, None);
        assert_eq!(legacy.bypass_master_fx, None);
        assert_eq!(legacy.key_threshold, Some(0.2));

        let mut other = legacy;
        other.opacity = 0.75;
        other.key_threshold = Some(0.8);
        let morph = Morph {
            a: Some(MorphSlot {
                layers: vec![legacy],
                ..Default::default()
            }),
            b: Some(MorphSlot {
                layers: vec![other],
                ..Default::default()
            }),
            ..Default::default()
        };
        let sampled = morph.sample(0.5).unwrap().layers.remove(0);
        close(sampled.opacity, 0.5);
        close(sampled.key_threshold.unwrap(), 0.5);
        assert!(sampled.effects.is_none());
        assert!(sampled.blend_mode.is_none());
    }

    #[test]
    fn explicit_beat_glide_is_deterministic_and_clamped() {
        let glide = MorphGlide::new(0.2, 0.8, 10.0, 4.0);
        close(glide.position_at(8.0), 0.2);
        close(glide.position_at(10.0), 0.2);
        close(glide.position_at(12.0), 0.5);
        close(glide.position_at(14.0), 0.8);
        close(glide.position_at(200.0), 0.8);
        close(glide.position_at(f64::NAN), 0.2);
        assert!(!glide.is_complete_at(13.999));
        assert!(glide.is_complete_at(14.0));
    }

    #[test]
    fn replacing_an_active_glide_has_no_position_jump() {
        let mut morph = Morph {
            t: 0.0,
            ..Default::default()
        };
        morph.start_glide(1.0, 4.0, 20.0);
        close(morph.position_at_beat(22.0), 0.5);

        morph.start_glide(0.0, 2.0, 22.0);
        close(morph.position_at_beat(22.0), 0.5);
        close(morph.position_at_beat(23.0), 0.25);
        close(morph.position_at_beat(24.0), 0.0);
        close(morph.settle_glide_at(24.0), 0.0);
        assert!(morph.glide.is_none());
    }

    #[test]
    fn persisted_sub_quarter_beat_remainder_is_not_stretched() {
        let mut morph = Morph::default();
        morph.start_glide(1.0, 4.0, 10.0);

        let snapshot = morph.snapshot_at_beat(13.9);
        let glide = snapshot.glide.expect("a tenth-beat remainder");
        assert!((glide.duration_beats - 0.1).abs() < 1.0e-9);

        let restored = Morph::from_snapshot_at_beat(snapshot, 20.0);
        close(restored.position_at_beat(20.0), 0.975);
        close(restored.position_at_beat(20.05), 0.9875);
        close(restored.position_at_beat(20.1), 1.0);
    }

    #[test]
    fn hue_and_slit_orientations_take_the_shortest_wrapped_arc() {
        let mut a = MorphSlot::default();
        a.master.hue_shift = 179.0;
        a.temporal.slit_angle = 179.0;
        let mut b = MorphSlot::default();
        b.master.hue_shift = -179.0;
        b.temporal.slit_angle = -179.0;
        let morph = Morph {
            a: Some(a),
            b: Some(b),
            ..Default::default()
        };

        let midpoint = morph.sample(0.5).unwrap();
        close(midpoint.master.hue_shift.abs(), 180.0);
        close(midpoint.temporal.slit_angle.abs(), 180.0);
        close(morph.sample(0.0).unwrap().master.hue_shift, 179.0);
        close(morph.sample(1.0).unwrap().master.hue_shift, -179.0);

        let equal_power = Morph {
            blend_law: MorphBlendLaw::EqualPower,
            ..morph
        };
        let quarter = equal_power.sample(0.25).unwrap();
        assert!(quarter.master.hue_shift.abs() > 179.0);
        assert!(quarter.temporal.slit_angle.abs() > 179.0);
    }

    #[test]
    fn layer_stack_remaps_preserve_master_world_and_surviving_identity() {
        let make_layers = |high| {
            (0..4)
                .map(|position| {
                    let mut layer = full_layer(position, high);
                    layer.opacity = position as f32 / 4.0;
                    layer
                })
                .collect::<Vec<_>>()
        };
        let mut a = MorphSlot::default();
        a.master.brightness = 0.25;
        a.layers = make_layers(false);
        let mut b = MorphSlot::default();
        b.master.brightness = 0.75;
        b.layers = make_layers(true);
        let mut morph = Morph {
            a: Some(a),
            b: Some(b),
            ..Default::default()
        };

        morph.remap_layers_after_remove(1);
        assert_eq!(
            morph
                .a
                .as_ref()
                .unwrap()
                .layers
                .iter()
                .map(|layer| layer.position)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        close(morph.a.as_ref().unwrap().master.brightness, 0.25);
        close(morph.b.as_ref().unwrap().master.brightness, 0.75);

        morph.remap_layers_after_move(2, 0);
        let sampled = morph.sample(0.5).unwrap();
        let mut positions = sampled
            .layers
            .iter()
            .map(|layer| layer.position)
            .collect::<Vec<_>>();
        positions.sort_unstable();
        assert_eq!(positions, vec![0, 1, 2]);
        // Original layer 3 survived removal of 1 (becoming 2), then moved to
        // 0. Its distinctive value follows that identity/permutation.
        close(
            sampled
                .layers
                .iter()
                .find(|layer| layer.position == 0)
                .unwrap()
                .opacity,
            0.75,
        );
        close(sampled.master.brightness, 0.5);
    }

    #[test]
    fn runtime_fields_survive_sample_application() {
        let morph = Morph {
            a: Some(slot(2.0, 0.0, 0.0)),
            b: Some(slot(10.0, 1.0, 1.0)),
            ..Default::default()
        };
        let mut master = EffectUniforms {
            resolution: [3840.0, 2160.0],
            time: 99.0,
            ..Default::default()
        };
        let sample = morph.sample(0.5).unwrap();
        sample.master.apply_to(&mut master);

        assert_eq!(master.resolution, [3840.0, 2160.0]);
        assert_eq!(master.time, 99.0);
        close(master.pixelate_size, 6.0);
    }

    #[test]
    fn cellular_morph_capture_sanitize_apply_and_interpolate_are_complete() {
        let a_uniforms = EffectUniforms {
            cellular_amount: 0.2,
            cellular_scale: 4.0,
            cellular_warp: 0.1,
            cellular_speed: 0.5,
            cellular_gap_amount: 0.2,
            cellular_gap_threshold: 0.3,
            cellular_gap_softness: 0.04,
            ..Default::default()
        };
        let b_uniforms = EffectUniforms {
            cellular_amount: 1.0,
            cellular_scale: 28.0,
            cellular_warp: 0.9,
            cellular_speed: 1.5,
            cellular_gap_amount: 1.0,
            cellular_gap_threshold: 0.9,
            cellular_gap_softness: 0.2,
            ..Default::default()
        };
        let a = MorphMasterSnapshot::capture(&a_uniforms);
        let b = MorphMasterSnapshot::capture(&b_uniforms);
        close(a.cellular_amount, 0.2);
        close(a.cellular_scale, 4.0);
        close(a.cellular_warp, 0.1);
        close(a.cellular_speed, 0.5);
        close(a.cellular_gap_amount, 0.2);
        close(a.cellular_gap_threshold, 0.3);
        close(a.cellular_gap_softness, 0.04);

        let midpoint = MorphMasterSnapshot::interpolate(&a, &b, [0.5, 0.5], true);
        close(midpoint.cellular_amount, 0.6);
        close(midpoint.cellular_scale, 16.0);
        close(midpoint.cellular_warp, 0.5);
        close(midpoint.cellular_speed, 1.0);
        close(midpoint.cellular_gap_amount, 0.6);
        close(midpoint.cellular_gap_threshold, 0.6);
        close(midpoint.cellular_gap_softness, 0.12);

        let mut applied = EffectUniforms {
            resolution: [1920.0, 1080.0],
            time: 42.0,
            ..Default::default()
        };
        midpoint.apply_to(&mut applied);
        close(applied.cellular_amount, 0.6);
        close(applied.cellular_scale, 16.0);
        close(applied.cellular_warp, 0.5);
        close(applied.cellular_speed, 1.0);
        close(applied.cellular_gap_amount, 0.6);
        close(applied.cellular_gap_threshold, 0.6);
        close(applied.cellular_gap_softness, 0.12);
        assert_eq!(applied.resolution, [1920.0, 1080.0]);
        close(applied.time, 42.0);

        let invalid = MorphMasterSnapshot {
            cellular_amount: f32::NAN,
            cellular_scale: 99.0,
            cellular_warp: -4.0,
            cellular_speed: f32::INFINITY,
            cellular_gap_amount: 4.0,
            cellular_gap_threshold: f32::NAN,
            cellular_gap_softness: 3.0,
            ..Default::default()
        }
        .sanitized();
        close(invalid.cellular_amount, 0.0);
        close(invalid.cellular_scale, 32.0);
        close(invalid.cellular_warp, 0.0);
        close(invalid.cellular_speed, 0.25);
        close(invalid.cellular_gap_amount, 1.0);
        close(invalid.cellular_gap_threshold, 0.65);
        close(invalid.cellular_gap_softness, 0.5);

        let legacy: MorphMasterSnapshot = serde_yaml::from_str("pixelate_size: 2\n").unwrap();
        close(legacy.cellular_amount, 0.0);
        close(legacy.cellular_scale, 10.0);
        close(legacy.cellular_warp, 0.35);
        close(legacy.cellular_speed, 0.25);
        close(legacy.cellular_gap_amount, 0.0);
        close(legacy.cellular_gap_threshold, 0.65);
        close(legacy.cellular_gap_softness, 0.08);
    }

    #[test]
    fn shift_morph_interpolates_controls_but_preserves_pattern_seed() {
        let a = MorphMasterSnapshot::capture(&EffectUniforms {
            shift_amount: 0.2,
            shift_block_size: 12.0,
            shift_density: 0.3,
            shift_speed: 2.0,
            random_seed: 11,
            ..Default::default()
        });
        let b = MorphMasterSnapshot::capture(&EffectUniforms {
            shift_amount: 1.0,
            shift_block_size: 52.0,
            shift_density: 0.9,
            shift_speed: 10.0,
            random_seed: 99,
            ..Default::default()
        });
        let midpoint = MorphMasterSnapshot::interpolate(&a, &b, [0.5, 0.5], true);
        close(midpoint.shift_amount, 0.6);
        close(midpoint.shift_block_size, 32.0);
        close(midpoint.shift_density, 0.6);
        close(midpoint.shift_speed, 6.0);

        let mut applied = EffectUniforms {
            resolution: [1280.0, 720.0],
            time: 17.0,
            random_seed: 0x1234_5678,
            ..Default::default()
        };
        midpoint.apply_to(&mut applied);
        close(applied.shift_amount, 0.6);
        close(applied.shift_block_size, 32.0);
        close(applied.shift_density, 0.6);
        close(applied.shift_speed, 6.0);
        assert_eq!(applied.resolution, [1280.0, 720.0]);
        close(applied.time, 17.0);
        assert_eq!(applied.random_seed, 0x1234_5678);

        let invalid = MorphMasterSnapshot {
            shift_amount: f32::NAN,
            shift_block_size: -5.0,
            shift_density: 8.0,
            shift_speed: f32::INFINITY,
            ..Default::default()
        }
        .sanitized();
        close(invalid.shift_amount, 0.0);
        close(invalid.shift_block_size, 2.0);
        close(invalid.shift_density, 1.0);
        close(invalid.shift_speed, 3.0);

        let legacy: MorphMasterSnapshot = serde_yaml::from_str("pixelate_size: 2\n").unwrap();
        close(legacy.shift_amount, 0.0);
        close(legacy.shift_block_size, 8.0);
        close(legacy.shift_density, 0.5);
        close(legacy.shift_speed, 3.0);
    }

    #[test]
    fn chroma_and_temporal_key_morphing_is_bounded_and_discrete_where_required() {
        let a = MorphMasterSnapshot::capture(&EffectUniforms {
            key_mode: 3.0,
            key_color: [0.0, 1.0, 0.0],
            key_tolerance: 0.1,
            key_softness: 0.02,
            ..Default::default()
        });
        let b = MorphMasterSnapshot::capture(&EffectUniforms {
            key_mode: 4.0,
            key_color: [1.0, 0.0, 0.5],
            key_tolerance: 0.5,
            key_softness: 0.1,
            ..Default::default()
        });
        let midpoint = MorphMasterSnapshot::interpolate(&a, &b, [0.5, 0.5], true);
        close(midpoint.key_mode, 4.0);
        assert_eq!(midpoint.key_color, [0.5, 0.5, 0.25]);
        close(midpoint.key_tolerance, 0.3);
        close(midpoint.key_softness, 0.06);

        let temporal_a = MorphTemporalSnapshot::capture(&TemporalParams {
            key_mode: 1.0,
            key_threshold: 0.08,
            key_softness: 0.02,
            key_history: 1.0,
            ..Default::default()
        });
        let temporal_b = MorphTemporalSnapshot::capture(&TemporalParams {
            key_mode: 4.0,
            key_threshold: 0.48,
            key_softness: 0.1,
            key_history: 9.0,
            ..Default::default()
        });
        let temporal =
            MorphTemporalSnapshot::interpolate(&temporal_a, &temporal_b, [0.5, 0.5], true);
        close(temporal.key_mode, 4.0);
        close(temporal.key_threshold, 0.28);
        close(temporal.key_softness, 0.06);
        close(temporal.key_history, 5.0);

        let legacy_master: MorphMasterSnapshot =
            serde_yaml::from_str("pixelate_size: 2\n").unwrap();
        assert_eq!(legacy_master.key_mode, 0.0);
        assert_eq!(legacy_master.key_color, [0.0, 1.0, 0.0]);
        close(legacy_master.key_tolerance, 0.15);
        let legacy_temporal: MorphTemporalSnapshot =
            serde_yaml::from_str("feedback: 0.2\n").unwrap();
        assert_eq!(legacy_temporal.key_mode, 0.0);
        close(legacy_temporal.key_threshold, 0.1);
        close(legacy_temporal.key_softness, 0.03);
        close(legacy_temporal.key_history, 1.0);
    }

    #[test]
    fn state_snapshot_round_trips_through_serde() {
        let mut morph = Morph {
            a: Some(slot(1.0, 0.0, 0.2)),
            b: Some(slot(9.0, 1.0, 0.8)),
            t: 0.3,
            blend_law: MorphBlendLaw::EqualPower,
            glide: None,
        };
        morph.start_glide(1.0, 8.0, 32.0);
        let before = morph.snapshot_at_beat(36.0);
        let yaml = serde_yaml::to_string(&before).unwrap();
        let decoded: MorphStateSnapshot = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(decoded, before);

        let restored = Morph::from_snapshot(decoded);
        close(restored.position_at_beat(0.0), morph.position_at_beat(36.0));
        close(restored.position_at_beat(2.0), morph.position_at_beat(38.0));
        close(restored.position_at_beat(4.0), morph.position_at_beat(40.0));
        assert_eq!(restored.blend_law, MorphBlendLaw::EqualPower);
    }

    #[test]
    fn full_layer_state_round_trips_through_serde() {
        let before = MorphStateSnapshot {
            a: Some(MorphSlot {
                layers: vec![full_layer(0, false)],
                ..Default::default()
            }),
            b: Some(MorphSlot {
                layers: vec![full_layer(0, true)],
                ..Default::default()
            }),
            t: 0.625,
            blend_law: MorphBlendLaw::EqualPower,
            glide: Some(MorphGlide::new(0.625, 1.0, 0.0, 2.0)),
        };
        let yaml = serde_yaml::to_string(&before).unwrap();
        let decoded: MorphStateSnapshot = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(decoded, before);
        let sampled = Morph::from_snapshot(decoded).sample(1.0).unwrap();
        assert_eq!(sampled.layers[0], full_layer(0, true));
    }

    #[test]
    fn deserialized_values_are_sanitized_before_runtime_use() {
        let yaml = r#"
a:
  master:
    pixelate_size: .nan
    rgb_split: 999
    hue_shift: -999
    invert: -5
    downsample: .inf
    grain_algo: 9.7
    cellular_amount: .nan
    cellular_scale: 999
    cellular_warp: -3
    cellular_speed: .inf
  ntsc:
    tape_speed: 99
    chroma_loss: 1
    edge_wave_speed: .nan
    head_switching_height: -9
    tracking_noise_height: 999
  temporal:
    feedback: .nan
    fb_zoom: 99
    fb_rotate: -99
    slit_angle: .inf
  layers:
    - position: 0
      opacity: .nan
      speed: 99
      fps: .nan
      effects:
        brightness: .nan
        key_threshold: 9
      blend_mode: difference
      visible: false
      paused: true
    - position: 999
      opacity: 0
t: .inf
glide:
  start: .nan
  target: 4
  start_beat: .inf
  duration_beats: -.inf
"#;
        let decoded: MorphStateSnapshot = serde_yaml::from_str(yaml).unwrap();
        let restored = Morph::from_snapshot(decoded);
        close(restored.t, 0.0);
        assert!(restored.glide.is_none());

        let slot = restored.a.unwrap();
        close(slot.master.pixelate_size, 1.0);
        close(slot.master.rgb_split, 30.0);
        close(slot.master.hue_shift, -180.0);
        close(slot.master.invert, 0.0);
        close(slot.master.downsample, 1.0);
        close(slot.master.grain_algo, 3.0);
        close(slot.master.cellular_amount, 0.0);
        close(slot.master.cellular_scale, 32.0);
        close(slot.master.cellular_warp, 0.0);
        close(slot.master.cellular_speed, 0.25);
        assert_eq!(slot.ntsc.tape_speed, 2);
        close(slot.ntsc.chroma_loss, 0.01);
        close(slot.ntsc.edge_wave_speed, 0.5);
        assert_eq!(slot.ntsc.head_switching_height, 0);
        assert_eq!(slot.ntsc.tracking_noise_height, 120);
        close(slot.temporal.feedback, 0.0);
        close(slot.temporal.fb_zoom, 1.1);
        close(slot.temporal.fb_rotate, -5.0);
        close(slot.temporal.slit_angle, 0.0);
        assert_eq!(slot.layers.len(), 2);
        let layer = *slot
            .layers
            .iter()
            .find(|layer| layer.position == 0)
            .unwrap();
        close(layer.opacity, 1.0);
        close(layer.speed, 4.0);
        close(layer.fps.unwrap(), 30.0);
        assert_eq!(layer.blend_mode, Some(MorphLayerBlendMode::Difference));
        assert_eq!(layer.visible, Some(false));
        assert_eq!(layer.paused, Some(true));
        let effects = layer.effects.unwrap();
        close(effects.brightness, 0.0);
        close(effects.key_threshold, 1.0);

        let far_layer = *slot
            .layers
            .iter()
            .find(|layer| layer.position == 999)
            .unwrap();
        close(far_layer.opacity, 0.0);
    }

    #[test]
    fn morph_sanitizer_retains_unbounded_positions_and_deduplicates_first_wins() {
        let first = LayerMorphSnapshot {
            position: usize::MAX,
            opacity: 0.25,
            ..Default::default()
        };
        let duplicate = LayerMorphSnapshot {
            position: usize::MAX,
            opacity: 0.75,
            ..Default::default()
        };
        let above_legacy_cap = LayerMorphSnapshot {
            position: 16,
            opacity: 0.5,
            ..Default::default()
        };

        let slot = MorphSlot {
            layers: vec![first, duplicate, above_legacy_cap],
            ..Default::default()
        }
        .sanitized();

        assert_eq!(slot.layers.len(), 2);
        assert_eq!(slot.layers[0].position, usize::MAX);
        close(slot.layers[0].opacity, 0.25);
        assert_eq!(slot.layers[1].position, 16);
        close(slot.layers[1].opacity, 0.5);
    }

    #[test]
    fn in_flight_glide_restores_without_rewinding_and_can_reanchor() {
        let mut morph = Morph::default();
        morph.start_glide(1.0, 8.0, 100.0);

        let snapshot = morph.snapshot_at_beat(103.0);
        close(snapshot.t, 0.375);
        let persisted = snapshot.glide.unwrap();
        close(persisted.start, 0.375);
        assert_eq!(persisted.start_beat, 0.0);
        assert_eq!(persisted.duration_beats, 5.0);

        let restored_at_zero = Morph::from_snapshot(snapshot.clone());
        close(restored_at_zero.position_at_beat(0.0), 0.375);
        close(restored_at_zero.position_at_beat(2.5), 0.6875);
        close(restored_at_zero.position_at_beat(5.0), 1.0);

        let restored_live = Morph::from_snapshot_at_beat(snapshot, 40.0);
        close(restored_live.position_at_beat(40.0), 0.375);
        close(restored_live.position_at_beat(45.0), 1.0);
    }

    #[test]
    fn legacy_absolute_glide_is_normalized_from_its_settled_position() {
        let snapshot = MorphStateSnapshot {
            t: 0.5,
            glide: Some(MorphGlide::new(0.0, 1.0, 200.0, 8.0)),
            ..Default::default()
        };

        let restored = Morph::from_snapshot(snapshot);
        close(restored.position_at_beat(0.0), 0.5);
        close(restored.position_at_beat(2.0), 0.75);
        close(restored.position_at_beat(4.0), 1.0);
    }

    #[test]
    fn invalid_positions_and_endpoints_fall_back_to_control_defaults() {
        let mut a = slot(f32::NAN, f32::NAN, f32::NAN);
        let b = slot(12.0, 1.0, 0.75);
        a.temporal.feedback = f32::INFINITY;
        let morph = Morph {
            a: Some(a),
            b: Some(b),
            ..Default::default()
        };

        let sample = morph.sample(f32::NAN).unwrap();
        close(sample.master.pixelate_size, 1.0);
        close(sample.master.invert, 0.0);
        close(sample.layers[0].opacity, 1.0);
        assert!(sample.temporal.feedback.is_finite());
    }
}
