//! Patch morphing: deterministic interpolation between two captured states.
//!
//! The serializable snapshot types in this module deliberately contain only
//! performance parameters. Render-loop values such as output resolution and
//! elapsed shader time are not captured. [`Morph::sample`] is pure: live and
//! offline renderers can consume the same detached [`MorphSample`] without
//! constructing or mutating a [`Layer`]. [`Morph::apply`] remains as the
//! compatibility adapter used by the live renderer.

use std::collections::HashSet;

use serde::de::{self, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

use crate::composition::{CompositionTree, RuntimeComposition};
use crate::effects::params::TemporalParams;
use crate::effects::EffectUniforms;
use crate::layers::{BlendMode, Layer};
use crate::motion::MotionParams;
use crate::ntsc::NtscParams;
use crate::patch::{
    CollisionAtlasConfig, CollisionScoreLoopDriverConfig, CurvedShutterConfig, FaradayConfig,
    FieldColliderConfig, FlowShapingConfig, GestureCanvasConfig, LongExposureConfig, MotionConfig,
    MotionDonorConfig, ProceduralFieldConfig, RefreshGardenConfig, RefreshGardenMatteRouteConfig,
    RefreshGardenMotionRouteConfig, TemporalLoomConfig, TemporalOriginalsConfig,
    TemporalResetPolicyConfig, TemporalRigConfig, TimeDisplaceMapConfig,
};
use crate::scan_processor::ScanProcessorParams;
use crate::spatial::SpatialTransform;
use crate::symmetry::{
    SavedMotionDonor, SymmetryParams, SYMMETRY_IMAGE_SLOTS, SYMMETRY_MOTION_SLOTS,
};
use crate::temporal::CollisionScoreLoopDriver;
use crate::visual_rack::{
    CellularParams, DigitalColorParams, DisplaceParams, EllipseMask, GrainParams, ImageMatte,
    KeyParams, MaskParams, RectangleMask, ResidualParams, RuntimeImageMatte, RuntimeMaskParams,
    RuntimeVisualNodeKind, RuntimeVisualRack, SavedImageSource, SavedImageTap, ShiftParams,
    VisualNodeKind, VisualRack,
};

// Morph's positional layer world follows the existing saved stack bound, not
// the smaller advanced-composition planner limit. This preserves flat legacy
// and dynamic stacks beyond 256 while still bounding hostile raw sequences.
const MAX_MORPH_LAYER_RACKS: usize = crate::performance::MAX_SAVED_LAYER_POSITION as usize + 1;

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
    // B13 small effects. `negative_mode` is a discrete law and recalls an
    // endpoint at the midpoint; the four angle/hue controls blend on their
    // shortest wrapped arcs.
    pub contour: f32,
    pub contour_bands: f32,
    pub contour_width: f32,
    pub contour_hue: f32,
    pub contour_fill: f32,
    pub flatten: f32,
    pub flatten_levels: f32,
    pub contour_dither: f32,
    pub solarize: f32,
    pub negative: f32,
    pub negative_mode: f32,
    pub colourpass: f32,
    pub colourpass_hue: f32,
    pub colourpass_width: f32,
    pub edge_amount: f32,
    pub edge_hue: f32,
    pub emboss: f32,
    pub emboss_angle: f32,
    pub halftone: f32,
    pub halftone_pitch: f32,
    pub halftone_angle: f32,
    pub moire: f32,
    pub moire_freq: f32,
    pub row_smear: f32,
    pub bitcrush: f32,
    pub bitcrush_levels: f32,
    pub bitcrush_dither: f32,
    // B8 key dressing. `key_border_color` is a discrete closed table and
    // recalls an endpoint at the midpoint.
    #[serde(default)]
    pub key_border: f32,
    #[serde(default)]
    pub key_border_color: f32,
    #[serde(default)]
    pub key_shadow: f32,
    pub multi_grid_x: f32,
    pub multi_grid_y: f32,
    pub barrel: f32,
    pub chroma_aberration: f32,
    pub anamorphic_streak: f32,
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
            contour: value.contour,
            contour_bands: value.contour_bands,
            contour_width: value.contour_width,
            contour_hue: value.contour_hue,
            contour_fill: value.contour_fill,
            flatten: value.flatten,
            flatten_levels: value.flatten_levels,
            contour_dither: value.contour_dither,
            solarize: value.solarize,
            negative: value.negative,
            negative_mode: value.negative_mode,
            colourpass: value.colourpass,
            colourpass_hue: value.colourpass_hue,
            colourpass_width: value.colourpass_width,
            edge_amount: value.edge_amount,
            edge_hue: value.edge_hue,
            emboss: value.emboss,
            emboss_angle: value.emboss_angle,
            halftone: value.halftone,
            halftone_pitch: value.halftone_pitch,
            halftone_angle: value.halftone_angle,
            moire: value.moire,
            moire_freq: value.moire_freq,
            row_smear: value.row_smear,
            bitcrush: value.bitcrush,
            bitcrush_levels: value.bitcrush_levels,
            bitcrush_dither: value.bitcrush_dither,
            key_border: value.key_border,
            key_border_color: value.key_border_color,
            key_shadow: value.key_shadow,
            multi_grid_x: value.multi_grid_x,
            multi_grid_y: value.multi_grid_y,
            barrel: value.barrel,
            chroma_aberration: value.chroma_aberration,
            anamorphic_streak: value.anamorphic_streak,
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
            contour: finite_clamp(self.contour, 0.0, 0.0, 1.0),
            contour_bands: finite_clamp(self.contour_bands, 10.0, 2.0, 40.0),
            contour_width: finite_clamp(self.contour_width, 1.2, 0.2, 6.0),
            contour_hue: finite_clamp(self.contour_hue, 0.0, 0.0, 1.0),
            contour_fill: finite_clamp(self.contour_fill, 0.25, 0.0, 1.0),
            flatten: finite_clamp(self.flatten, 0.0, 0.0, 1.0),
            flatten_levels: finite_clamp(self.flatten_levels, 5.0, 2.0, 16.0),
            contour_dither: finite_clamp(self.contour_dither, 0.0, 0.0, 1.0),
            solarize: finite_clamp(self.solarize, 0.0, 0.0, 1.0),
            negative: finite_clamp(self.negative, 0.0, 0.0, 1.0),
            negative_mode: discrete_f32(self.negative_mode, 0.0, 2.0),
            colourpass: finite_clamp(self.colourpass, 0.0, 0.0, 1.0),
            colourpass_hue: finite_clamp(self.colourpass_hue, 0.0, -180.0, 180.0),
            colourpass_width: finite_clamp(self.colourpass_width, 0.25, 0.0, 1.0),
            edge_amount: finite_clamp(self.edge_amount, 0.0, 0.0, 1.0),
            edge_hue: finite_clamp(self.edge_hue, 0.0, -180.0, 180.0),
            emboss: finite_clamp(self.emboss, 0.0, 0.0, 1.0),
            emboss_angle: finite_clamp(self.emboss_angle, 45.0, -180.0, 180.0),
            halftone: finite_clamp(self.halftone, 0.0, 0.0, 1.0),
            halftone_pitch: finite_clamp(self.halftone_pitch, 0.4, 0.0, 1.0),
            halftone_angle: finite_clamp(self.halftone_angle, 0.0, -180.0, 180.0),
            moire: finite_clamp(self.moire, 0.0, 0.0, 1.0),
            moire_freq: finite_clamp(self.moire_freq, 0.4, 0.0, 1.0),
            row_smear: finite_clamp(self.row_smear, 0.0, 0.0, 1.0),
            bitcrush: finite_clamp(self.bitcrush, 0.0, 0.0, 1.0),
            bitcrush_levels: finite_clamp(self.bitcrush_levels, 2.0, 2.0, 16.0),
            bitcrush_dither: finite_clamp(self.bitcrush_dither, 1.0, 0.0, 1.0),
            key_border: finite_clamp(self.key_border, 0.0, 0.0, 1.0),
            key_border_color: discrete_f32(self.key_border_color, 0.0, 7.0),
            key_shadow: finite_clamp(self.key_shadow, 0.0, 0.0, 1.0),
            multi_grid_x: finite_clamp(self.multi_grid_x, 1.0, 1.0, 8.0),
            multi_grid_y: finite_clamp(self.multi_grid_y, 1.0, 1.0, 8.0),
            barrel: finite_clamp(self.barrel, 0.0, -1.0, 1.0),
            chroma_aberration: finite_clamp(self.chroma_aberration, 0.0, 0.0, 1.0),
            anamorphic_streak: finite_clamp(self.anamorphic_streak, 0.0, 0.0, 1.0),
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
        value.contour = clean.contour;
        value.contour_bands = clean.contour_bands;
        value.contour_width = clean.contour_width;
        value.contour_hue = clean.contour_hue;
        value.contour_fill = clean.contour_fill;
        value.flatten = clean.flatten;
        value.flatten_levels = clean.flatten_levels;
        value.contour_dither = clean.contour_dither;
        value.solarize = clean.solarize;
        value.negative = clean.negative;
        value.negative_mode = clean.negative_mode;
        value.colourpass = clean.colourpass;
        value.colourpass_hue = clean.colourpass_hue;
        value.colourpass_width = clean.colourpass_width;
        value.edge_amount = clean.edge_amount;
        value.edge_hue = clean.edge_hue;
        value.emboss = clean.emboss;
        value.emboss_angle = clean.emboss_angle;
        value.halftone = clean.halftone;
        value.halftone_pitch = clean.halftone_pitch;
        value.halftone_angle = clean.halftone_angle;
        value.moire = clean.moire;
        value.moire_freq = clean.moire_freq;
        value.row_smear = clean.row_smear;
        value.bitcrush = clean.bitcrush;
        value.bitcrush_levels = clean.bitcrush_levels;
        value.bitcrush_dither = clean.bitcrush_dither;
        value.key_border = clean.key_border;
        value.key_border_color = clean.key_border_color;
        value.key_shadow = clean.key_shadow;
        value.multi_grid_x = clean.multi_grid_x;
        value.multi_grid_y = clean.multi_grid_y;
        value.barrel = clean.barrel;
        value.chroma_aberration = clean.chroma_aberration;
        value.anamorphic_streak = clean.anamorphic_streak;
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
            contour: blend_finite(a.contour, b.contour, weights),
            contour_bands: blend_finite(a.contour_bands, b.contour_bands, weights),
            contour_width: blend_finite(a.contour_width, b.contour_width, weights),
            contour_hue: blend_finite(a.contour_hue, b.contour_hue, weights),
            contour_fill: blend_finite(a.contour_fill, b.contour_fill, weights),
            flatten: blend_finite(a.flatten, b.flatten, weights),
            flatten_levels: blend_finite(a.flatten_levels, b.flatten_levels, weights),
            contour_dither: blend_finite(a.contour_dither, b.contour_dither, weights),
            solarize: blend_finite(a.solarize, b.solarize, weights),
            negative: blend_finite(a.negative, b.negative, weights),
            negative_mode: pick_finite(a.negative_mode, b.negative_mode, choose_b),
            colourpass: blend_finite(a.colourpass, b.colourpass, weights),
            colourpass_hue: blend_wrapped_degrees(a.colourpass_hue, b.colourpass_hue, weights),
            colourpass_width: blend_finite(a.colourpass_width, b.colourpass_width, weights),
            edge_amount: blend_finite(a.edge_amount, b.edge_amount, weights),
            edge_hue: blend_wrapped_degrees(a.edge_hue, b.edge_hue, weights),
            emboss: blend_finite(a.emboss, b.emboss, weights),
            emboss_angle: blend_wrapped_degrees(a.emboss_angle, b.emboss_angle, weights),
            halftone: blend_finite(a.halftone, b.halftone, weights),
            halftone_pitch: blend_finite(a.halftone_pitch, b.halftone_pitch, weights),
            halftone_angle: blend_wrapped_degrees(a.halftone_angle, b.halftone_angle, weights),
            moire: blend_finite(a.moire, b.moire, weights),
            moire_freq: blend_finite(a.moire_freq, b.moire_freq, weights),
            row_smear: blend_finite(a.row_smear, b.row_smear, weights),
            bitcrush: blend_finite(a.bitcrush, b.bitcrush, weights),
            bitcrush_levels: blend_finite(a.bitcrush_levels, b.bitcrush_levels, weights),
            bitcrush_dither: blend_finite(a.bitcrush_dither, b.bitcrush_dither, weights),
            key_border: blend_finite(a.key_border, b.key_border, weights),
            key_border_color: pick_finite(a.key_border_color, b.key_border_color, choose_b),
            key_shadow: blend_finite(a.key_shadow, b.key_shadow, weights),
            multi_grid_x: blend_finite(a.multi_grid_x, b.multi_grid_x, weights),
            multi_grid_y: blend_finite(a.multi_grid_y, b.multi_grid_y, weights),
            barrel: blend_finite(a.barrel, b.barrel, weights),
            chroma_aberration: blend_finite(a.chroma_aberration, b.chroma_aberration, weights),
            anamorphic_streak: blend_finite(a.anamorphic_streak, b.anamorphic_streak, weights),
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
    /// B12 discrete laws: the time-displace map and interpolation toggle
    /// recall an endpoint at the morph midpoint, never a synthesized third.
    pub slit_map: TimeDisplaceMapConfig,
    pub slit_interp: bool,
    pub key_mode: f32,
    pub key_threshold: f32,
    pub key_softness: f32,
    pub key_history: f32,
    /// M3 originals capture authored configuration only. Renderer history,
    /// Garden carrier pixels, and the Collision Score ordinal are runtime
    /// memory and deliberately never enter a morph slot.
    pub originals: TemporalOriginalsConfig,
    /// B3 feedback rig, captured as its serializable config block.
    pub rig: TemporalRigConfig,
    /// B4 display physics, captured whole: the params struct is its own
    /// sanitizing serde block. Continuous values blend; the interlace mode,
    /// the order fault, and the display model recall an endpoint at the
    /// midpoint.
    pub display: crate::display_physics::DisplayPhysicsParams,
    /// B8 melting edge, captured whole: the params struct is its own
    /// sanitizing serde block. All six controls are continuous and blend.
    #[serde(default)]
    pub melt: crate::mixing_boundary::MeltParams,
    /// B5 codec mosh, captured whole: the params struct is its own
    /// sanitizing serde block. The eight continuous controls blend; the
    /// discrete recycle law recalls an endpoint at the midpoint.
    #[serde(default)]
    pub mosh: crate::codec_mosh::CodecMoshParams,
    /// B14 sync latch, captured whole: the params struct is its own
    /// sanitizing serde block. The four continuous controls blend; the
    /// discrete latch switch recalls an endpoint at the midpoint, exactly as
    /// `servo_defeated` does.
    #[serde(default)]
    pub sync: crate::sync_latch::SyncLatchParams,
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
            slit_map: TimeDisplaceMapConfig::from_runtime(value.slit_map),
            slit_interp: value.slit_interp,
            key_mode: value.key_mode,
            key_threshold: value.key_threshold,
            key_softness: value.key_softness,
            key_history: value.key_history,
            originals: TemporalOriginalsConfig::from_params(value.originals),
            rig: TemporalRigConfig::from_params(value.rig),
            display: value.display,
            melt: value.melt,
            mosh: value.mosh,
            sync: value.sync,
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
            slit_map: self.slit_map,
            slit_interp: self.slit_interp,
            key_mode: discrete_f32(self.key_mode, 0.0, 4.0),
            key_threshold: finite_clamp(self.key_threshold, 0.1, 0.0, 1.0),
            key_softness: finite_clamp(self.key_softness, 0.03, 0.0, 0.5),
            key_history: finite_clamp(self.key_history, 1.0, 1.0, 23.0).round(),
            originals: self.originals.sanitized(),
            rig: self.rig.sanitized(),
            display: self.display.sanitized(),
            melt: self.melt.sanitized(),
            mosh: self.mosh.sanitized(),
            sync: self.sync.sanitized(),
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
            slit_map: clean.slit_map.to_runtime(),
            slit_interp: clean.slit_interp,
            key_mode: clean.key_mode,
            key_threshold: clean.key_threshold,
            key_softness: clean.key_softness,
            key_history: clean.key_history,
            originals: clean.originals.to_params(),
            display: clean.display,
            melt: clean.melt,
            mosh: clean.mosh,
            sync: clean.sync,
            rig: clean.rig.to_params(),
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
            slit_map: if choose_b { b.slit_map } else { a.slit_map },
            slit_interp: if choose_b {
                b.slit_interp
            } else {
                a.slit_interp
            },
            key_mode: pick_finite(a.key_mode, b.key_mode, choose_b),
            key_threshold: blend_finite(a.key_threshold, b.key_threshold, weights),
            key_softness: blend_finite(a.key_softness, b.key_softness, weights),
            key_history: blend_finite(a.key_history, b.key_history, weights),
            originals: interpolate_temporal_originals(a.originals, b.originals, weights, choose_b),
            rig: interpolate_temporal_rig(a.rig, b.rig, weights, choose_b),
            display: interpolate_display_physics(a.display, b.display, weights, choose_b),
            melt: interpolate_master_melt(a.melt, b.melt, weights),
            mosh: interpolate_codec_mosh(a.mosh, b.mosh, weights, choose_b),
            sync: interpolate_sync_latch(a.sync, b.sync, weights, choose_b),
        }
        .sanitized()
    }
}

/// B14 sync-latch morphing: the four continuous controls blend, and the
/// latch switch recalls an endpoint at the midpoint. A morph position can
/// therefore cross into (or out of) a latched program, but it can never
/// synthesize a third switch state neither slot captured.
fn interpolate_sync_latch(
    a: crate::sync_latch::SyncLatchParams,
    b: crate::sync_latch::SyncLatchParams,
    weights: [f32; 2],
    choose_b: bool,
) -> crate::sync_latch::SyncLatchParams {
    let a = a.sanitized();
    let b = b.sanitized();
    crate::sync_latch::SyncLatchParams {
        amount: blend_finite(a.amount, b.amount, weights),
        rate: blend_finite(a.rate, b.rate, weights),
        spread: blend_finite(a.spread, b.spread, weights),
        bias: blend_finite(a.bias, b.bias, weights),
        latched: if choose_b { b.latched } else { a.latched },
    }
    .sanitized()
}

/// B5 codec-mosh morphing: the eight continuous controls blend, and the
/// discrete recycle law recalls an endpoint at the midpoint.
fn interpolate_codec_mosh(
    a: crate::codec_mosh::CodecMoshParams,
    b: crate::codec_mosh::CodecMoshParams,
    weights: [f32; 2],
    choose_b: bool,
) -> crate::codec_mosh::CodecMoshParams {
    let a = a.sanitized();
    let b = b.sanitized();
    crate::codec_mosh::CodecMoshParams {
        amount: blend_finite(a.amount, b.amount, weights),
        key_removal: blend_finite(a.key_removal, b.key_removal, weights),
        hold: blend_finite(a.hold, b.hold, weights),
        drop: blend_finite(a.drop, b.drop, weights),
        shuffle: blend_finite(a.shuffle, b.shuffle, weights),
        rate: blend_finite(a.rate, b.rate, weights),
        bitrate_starve: blend_finite(a.bitrate_starve, b.bitrate_starve, weights),
        resync: blend_finite(a.resync, b.resync, weights),
        recycle: if choose_b { b.recycle } else { a.recycle },
    }
    .sanitized()
}

/// B8 master-melt morphing: all six controls are continuous and blend.
fn interpolate_master_melt(
    a: crate::mixing_boundary::MeltParams,
    b: crate::mixing_boundary::MeltParams,
    weights: [f32; 2],
) -> crate::mixing_boundary::MeltParams {
    let a = a.sanitized();
    let b = b.sanitized();
    crate::mixing_boundary::MeltParams {
        melt: blend_finite(a.melt, b.melt, weights),
        width: blend_finite(a.width, b.width, weights),
        hold: blend_finite(a.hold, b.hold, weights),
        swirl: blend_finite(a.swirl, b.swirl, weights),
        chroma: blend_finite(a.chroma, b.chroma, weights),
        creep: blend_finite(a.creep, b.creep, weights),
    }
    .sanitized()
}

/// B4 display morphing: the seventeen continuous controls blend, and the
/// three discrete laws — the interlace mode, the field-order fault, and the
/// display model — recall an endpoint at the midpoint.
fn interpolate_display_physics(
    a: crate::display_physics::DisplayPhysicsParams,
    b: crate::display_physics::DisplayPhysicsParams,
    weights: [f32; 2],
    choose_b: bool,
) -> crate::display_physics::DisplayPhysicsParams {
    let a = a.sanitized();
    let b = b.sanitized();
    crate::display_physics::DisplayPhysicsParams {
        il_amount: blend_finite(a.il_amount, b.il_amount, weights),
        il_mode: pick(a.il_mode, b.il_mode, choose_b),
        il_order: pick(a.il_order, b.il_order, choose_b),
        il_twitter: blend_finite(a.il_twitter, b.il_twitter, weights),
        il_judder: blend_finite(a.il_judder, b.il_judder, weights),
        phosphor: blend_finite(a.phosphor, b.phosphor, weights),
        phos_r: blend_finite(a.phos_r, b.phos_r, weights),
        phos_g: blend_finite(a.phos_g, b.phos_g, weights),
        phos_b: blend_finite(a.phos_b, b.phos_b, weights),
        model: pick(a.model, b.model, choose_b),
        scanlines: blend_finite(a.scanlines, b.scanlines, weights),
        beam_width: blend_finite(a.beam_width, b.beam_width, weights),
        beam_shape: blend_finite(a.beam_shape, b.beam_shape, weights),
        mask_strength: blend_finite(a.mask_strength, b.mask_strength, weights),
        mask_dark: blend_finite(a.mask_dark, b.mask_dark, weights),
        bloom: blend_finite(a.bloom, b.bloom, weights),
        bloom_radius: blend_finite(a.bloom_radius, b.bloom_radius, weights),
        halation: blend_finite(a.halation, b.halation, weights),
        defocus: blend_finite(a.defocus, b.defocus, weights),
        sag: blend_finite(a.sag, b.sag, weights),
    }
}

/// B3 rig morphing: continuous values blend, the in-loop hue rotation blends
/// on its wrapped arc, and the discrete laws — reflections, shape, edge, and
/// the two servo switches — recall an endpoint at the midpoint.
fn interpolate_temporal_rig(
    a: TemporalRigConfig,
    b: TemporalRigConfig,
    weights: [f32; 2],
    choose_b: bool,
) -> TemporalRigConfig {
    let a = a.sanitized();
    let b = b.sanitized();
    TemporalRigConfig {
        offset_x: blend_finite(a.offset_x, b.offset_x, weights),
        offset_y: blend_finite(a.offset_y, b.offset_y, weights),
        reflect_x: pick(a.reflect_x, b.reflect_x, choose_b),
        reflect_y: pick(a.reflect_y, b.reflect_y, choose_b),
        hue_rotate: blend_wrapped_degrees(a.hue_rotate, b.hue_rotate, weights),
        saturation: blend_finite(a.saturation, b.saturation, weights),
        gain_r: blend_finite(a.gain_r, b.gain_r, weights),
        gain_g: blend_finite(a.gain_g, b.gain_g, weights),
        gain_b: blend_finite(a.gain_b, b.gain_b, weights),
        chroma_displace: blend_finite(a.chroma_displace, b.chroma_displace, weights),
        blur: blend_finite(a.blur, b.blur, weights),
        sharpen: blend_finite(a.sharpen, b.sharpen, weights),
        shape: pick(a.shape, b.shape, choose_b),
        drive: blend_finite(a.drive, b.drive, weights),
        pivot: blend_finite(a.pivot, b.pivot, weights),
        threshold: blend_finite(a.threshold, b.threshold, weights),
        noise: blend_finite(a.noise, b.noise, weights),
        edge: pick(a.edge, b.edge, choose_b),
        servo: pick(a.servo, b.servo, choose_b),
        servo_defeated: pick(a.servo_defeated, b.servo_defeated, choose_b),
    }
    .sanitized()
}

fn blend_u8(a: u8, b: u8, weights: [f32; 2], min: u8, max: u8) -> u8 {
    blend_finite(f32::from(a), f32::from(b), weights)
        .round()
        .clamp(f32::from(min), f32::from(max)) as u8
}

fn blend_u32(a: u32, b: u32, weights: [f32; 2]) -> u32 {
    let value = f64::from(a) * f64::from(weights[0]) + f64::from(b) * f64::from(weights[1]);
    if value.is_finite() {
        value.round().clamp(0.0, f64::from(u32::MAX)) as u32
    } else {
        0
    }
}

fn interpolate_temporal_originals(
    a: TemporalOriginalsConfig,
    b: TemporalOriginalsConfig,
    weights: [f32; 2],
    choose_b: bool,
) -> TemporalOriginalsConfig {
    TemporalOriginalsConfig {
        loom: TemporalLoomConfig {
            amount: blend_finite(a.loom.amount, b.loom.amount, weights),
            topology: pick(a.loom.topology, b.loom.topology, choose_b),
            interpolation: pick(a.loom.interpolation, b.loom.interpolation, choose_b),
            depth: blend_finite(a.loom.depth, b.loom.depth, weights),
            phase: blend_finite(a.loom.phase, b.loom.phase, weights),
            scale: blend_finite(a.loom.scale, b.loom.scale, weights),
            angle: blend_wrapped_degrees(a.loom.angle, b.loom.angle, weights),
            folds: blend_u8(a.loom.folds, b.loom.folds, weights, 1, 16),
            quantization: blend_u8(a.loom.quantization, b.loom.quantization, weights, 0, 24),
        },
        atlas: CollisionAtlasConfig {
            amount: blend_finite(a.atlas.amount, b.atlas.amount, weights),
            // Seed identity is an endpoint recall, never an interpolated RNG.
            seed: pick(a.atlas.seed, b.atlas.seed, choose_b),
            territories: blend_u8(a.atlas.territories, b.atlas.territories, weights, 1, 64),
            collision: blend_finite(a.atlas.collision, b.atlas.collision, weights),
        },
        garden: RefreshGardenConfig {
            amount: blend_finite(a.garden.amount, b.garden.amount, weights),
            gate: pick(a.garden.gate, b.garden.gate, choose_b),
            threshold: blend_finite(a.garden.threshold, b.garden.threshold, weights),
            softness: blend_finite(a.garden.softness, b.garden.softness, weights),
            decay: blend_finite(a.garden.decay, b.garden.decay, weights),
            max_hold_ticks: blend_u32(a.garden.max_hold_ticks, b.garden.max_hold_ticks, weights),
            matte_route: pick(a.garden.matte_route, b.garden.matte_route, choose_b),
            motion_route: pick(a.garden.motion_route, b.garden.motion_route, choose_b),
        },
        long_exposure: LongExposureConfig {
            amount: blend_finite(a.long_exposure.amount, b.long_exposure.amount, weights),
            shutter_frames: blend_u8(
                a.long_exposure.shutter_frames,
                b.long_exposure.shutter_frames,
                weights,
                2,
                24,
            ),
        },
        // Score configuration is entirely discrete: an A/B move may recall
        // either conductor, but may not synthesize a third sequence.
        score: pick(a.score, b.score, choose_b),
        reset: TemporalResetPolicyConfig {
            loop_boundary: pick(a.reset.loop_boundary, b.reset.loop_boundary, choose_b),
            downbeat: pick(a.reset.downbeat, b.reset.downbeat, choose_b),
        },
    }
}

fn resolve_score_loop_driver(
    driver: CollisionScoreLoopDriverConfig,
    resolve_position: impl FnOnce(
        crate::performance::SavedLayerPosition,
    ) -> Option<crate::image_routing::StableLayerId>,
) -> CollisionScoreLoopDriver {
    match driver {
        CollisionScoreLoopDriverConfig::None => CollisionScoreLoopDriver::None,
        CollisionScoreLoopDriverConfig::SelectedLayer { saved_position } => {
            resolve_position(saved_position).map_or(
                CollisionScoreLoopDriver::MissingSelectedLayer { saved_position },
                |layer_id| CollisionScoreLoopDriver::SelectedLayer {
                    layer_id,
                    saved_position,
                },
            )
        }
        CollisionScoreLoopDriverConfig::MissingSelectedLayer { saved_position } => {
            CollisionScoreLoopDriver::MissingSelectedLayer { saved_position }
        }
    }
}

fn interpolate_motion_config(
    a: MotionConfig,
    b: MotionConfig,
    weights: [f32; 2],
    choose_b: bool,
) -> MotionConfig {
    let a = a.sanitized();
    let b = b.sanitized();
    MotionConfig {
        // Provenance, algorithms, source laws, identities, and fixed quality
        // tiers are endpoint recalls. Morph never invents a third algorithm,
        // RNG identity, route, or sample count.
        algorithm_version: pick(a.algorithm_version, b.algorithm_version, choose_b),
        field_source: pick(a.field_source, b.field_source, choose_b),
        lattice_quality: pick(a.lattice_quality, b.lattice_quality, choose_b),
        // The procedural kind rides `field_source`'s midpoint switch above;
        // its two scalars are ordinary continuous state and interpolate.
        procedural: ProceduralFieldConfig {
            scale: blend_finite(a.procedural.scale, b.procedural.scale, weights),
            rate: blend_finite(a.procedural.rate, b.procedural.rate, weights),
        },
        // All four shaping controls are continuous values.
        shaping: FlowShapingConfig {
            stretch: blend_finite(a.shaping.stretch, b.shaping.stretch, weights),
            edge_repel: blend_finite(a.shaping.edge_repel, b.shaping.edge_repel, weights),
            vector_trash: blend_finite(a.shaping.vector_trash, b.shaping.vector_trash, weights),
            trash_block_size: blend_finite(
                a.shaping.trash_block_size,
                b.shaping.trash_block_size,
                weights,
            ),
        },
        transplant: FaradayConfig {
            amount: blend_finite(a.transplant.amount, b.transplant.amount, weights),
            donor: pick(a.transplant.donor, b.transplant.donor, choose_b),
            carrier: pick(a.transplant.carrier, b.transplant.carrier, choose_b),
            confidence_threshold: blend_finite(
                a.transplant.confidence_threshold,
                b.transplant.confidence_threshold,
                weights,
            ),
            confidence_softness: blend_finite(
                a.transplant.confidence_softness,
                b.transplant.confidence_softness,
                weights,
            ),
            refresh: blend_finite(a.transplant.refresh, b.transplant.refresh, weights),
            decay: blend_finite(a.transplant.decay, b.transplant.decay, weights),
            occlusion: blend_finite(a.transplant.occlusion, b.transplant.occlusion, weights),
        },
        shutter: CurvedShutterConfig {
            angle_degrees: blend_finite(a.shutter.angle_degrees, b.shutter.angle_degrees, weights),
            phase: blend_finite(a.shutter.phase, b.shutter.phase, weights),
            curvature: blend_finite(a.shutter.curvature, b.shutter.curvature, weights),
            chromatic_lag: blend_finite(a.shutter.chromatic_lag, b.shutter.chromatic_lag, weights),
            quality: pick(a.shutter.quality, b.shutter.quality, choose_b),
        },
        // Field Collider v1 has no continuous control, so the *entire* block is
        // one endpoint recall taken at the midpoint. Picking the block whole —
        // rather than each field independently — is what makes the choice
        // endpoint-exact: a per-field pick would be identical here only by
        // accident, and would start synthesizing third configurations the
        // instant v2 adds a field whose meaning depends on the mode or on which
        // pair of donors is armed.
        collider: pick(a.collider, b.collider, choose_b),
    }
    .sanitized()
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
    Add,
    Subtract,
    Darken,
    Lighten,
    Overlay,
    SoftLight,
    HardLight,
    Exclusion,
    Dodge,
    Burn,
    AlphaCut,
    VividLight,
    PinLight,
    Divide,
    WrapAdd,
    Xor,
    And,
    Hue,
    Saturation,
    Color,
    Luminosity,
}

impl MorphLayerBlendMode {
    fn capture(value: BlendMode) -> Self {
        match value {
            BlendMode::Normal => Self::Normal,
            BlendMode::Screen => Self::Screen,
            BlendMode::Multiply => Self::Multiply,
            BlendMode::Difference => Self::Difference,
            BlendMode::Add => Self::Add,
            BlendMode::Subtract => Self::Subtract,
            BlendMode::Darken => Self::Darken,
            BlendMode::Lighten => Self::Lighten,
            BlendMode::Overlay => Self::Overlay,
            BlendMode::SoftLight => Self::SoftLight,
            BlendMode::HardLight => Self::HardLight,
            BlendMode::Exclusion => Self::Exclusion,
            BlendMode::Dodge => Self::Dodge,
            BlendMode::Burn => Self::Burn,
            BlendMode::AlphaCut => Self::AlphaCut,
            BlendMode::VividLight => Self::VividLight,
            BlendMode::PinLight => Self::PinLight,
            BlendMode::Divide => Self::Divide,
            BlendMode::WrapAdd => Self::WrapAdd,
            BlendMode::Xor => Self::Xor,
            BlendMode::And => Self::And,
            BlendMode::Hue => Self::Hue,
            BlendMode::Saturation => Self::Saturation,
            BlendMode::Color => Self::Color,
            BlendMode::Luminosity => Self::Luminosity,
        }
    }

    pub(crate) fn to_blend_mode(self) -> BlendMode {
        match self {
            Self::Normal => BlendMode::Normal,
            Self::Screen => BlendMode::Screen,
            Self::Multiply => BlendMode::Multiply,
            Self::Difference => BlendMode::Difference,
            Self::Add => BlendMode::Add,
            Self::Subtract => BlendMode::Subtract,
            Self::Darken => BlendMode::Darken,
            Self::Lighten => BlendMode::Lighten,
            Self::Overlay => BlendMode::Overlay,
            Self::SoftLight => BlendMode::SoftLight,
            Self::HardLight => BlendMode::HardLight,
            Self::Exclusion => BlendMode::Exclusion,
            Self::Dodge => BlendMode::Dodge,
            Self::Burn => BlendMode::Burn,
            Self::AlphaCut => BlendMode::AlphaCut,
            Self::VividLight => BlendMode::VividLight,
            Self::PinLight => BlendMode::PinLight,
            Self::Divide => BlendMode::Divide,
            Self::WrapAdd => BlendMode::WrapAdd,
            Self::Xor => BlendMode::Xor,
            Self::And => BlendMode::And,
            Self::Hue => BlendMode::Hue,
            Self::Saturation => BlendMode::Saturation,
            Self::Color => BlendMode::Color,
            Self::Luminosity => BlendMode::Luminosity,
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
    /// Optional for schema evolution: a slot written before spatial controls
    /// existed must not claim or reset a live layer's transform.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transform: Option<SpatialTransform>,
    /// Optional for schema evolution. Runtime field pixels and stable IDs are
    /// absent; Selected donor intent stores only its saved layer position.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub motion: Option<MotionConfig>,
    /// Compatibility field for pre-full-state snapshots. Newly captured
    /// snapshots store this value inside `effects` and omit this duplicate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_threshold: Option<f32>,
    /// B7 pattern-synth capture, `Some` only when the captured layer was a
    /// pattern source. Interpolation requires both slots to carry it, and
    /// application lands only on a live pattern layer, so no morph position
    /// can put pattern values on a different source kind.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pattern: Option<crate::patch::PatternSynthConfig>,
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
    Transform,
    Motion,
    Pattern,
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
            transform: None,
            motion: None,
            key_threshold: None,
            pattern: None,
        }
    }
}

impl LayerMorphSnapshot {
    fn capture(
        position: usize,
        layer: &Layer,
        layer_ids: &[crate::image_routing::StableLayerId],
    ) -> Self {
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
            transform: Some(layer.transform.sanitized()),
            motion: Some(MotionConfig::from_params_for_capture(
                layer.motion,
                layer_ids,
            )),
            key_threshold: None,
            pattern: layer
                .pattern_params()
                .map(crate::patch::PatternSynthConfig::from_params),
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
            transform: self.transform.map(SpatialTransform::sanitized),
            motion: self.motion.map(MotionConfig::sanitized),
            key_threshold: if has_effects { None } else { legacy_key },
            pattern: self
                .pattern
                .map(crate::patch::PatternSynthConfig::sanitized),
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
            transform: match (a.transform, b.transform) {
                (Some(a), Some(b)) => Some(SpatialTransform::interpolate(a, b, weights, choose_b)),
                _ => None,
            },
            motion: match (a.motion, b.motion) {
                (Some(a), Some(b)) => Some(interpolate_motion_config(a, b, weights, choose_b)),
                _ => None,
            },
            key_threshold,
            pattern: match (a.pattern, b.pattern) {
                (Some(a), Some(b)) => Some(interpolate_pattern_config(a, b, weights, choose_b)),
                _ => None,
            },
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
        if let Some(transform) = clean.transform {
            layer.transform = transform;
        }
        if let Some(motion) = clean.motion {
            layer.motion = motion.to_params().sanitized();
        }
        // Kind-gated at application: pattern values land only on a layer
        // that actually is a pattern source.
        if let (Some(config), Some(params)) = (clean.pattern, layer.pattern_params_mut()) {
            *params = config.to_params();
        }
    }
}

/// Blend the B7 pattern-synth capture: the twenty-two continuous values
/// interpolate — hue along its shortest wrapped unit arc — and the three
/// discrete vocabularies recall an endpoint at the midpoint.
fn interpolate_pattern_config(
    a: crate::patch::PatternSynthConfig,
    b: crate::patch::PatternSynthConfig,
    weights: [f32; 2],
    choose_b: bool,
) -> crate::patch::PatternSynthConfig {
    let a = a.sanitized();
    let b = b.sanitized();
    let blend = |x: f32, y: f32| blend_finite(x, y, weights);
    let hue =
        blend_wrapped_degrees(a.hue * 360.0, b.hue * 360.0, weights).rem_euclid(360.0) / 360.0;
    crate::patch::PatternSynthConfig {
        shape: pick(a.shape, b.shape, choose_b),
        wave: pick(a.wave, b.wave, choose_b),
        color_mode: pick(a.color_mode, b.color_mode, choose_b),
        freq_x: blend(a.freq_x, b.freq_x),
        freq_y: blend(a.freq_y, b.freq_y),
        phase: blend(a.phase, b.phase),
        rate: blend(a.rate, b.rate),
        cross_mod: blend(a.cross_mod, b.cross_mod),
        wavefold: blend(a.wavefold, b.wavefold),
        pulse_width: blend(a.pulse_width, b.pulse_width),
        comparator: blend(a.comparator, b.comparator),
        comp_threshold: blend(a.comp_threshold, b.comp_threshold),
        comp_soft: blend(a.comp_soft, b.comp_soft),
        symmetry: blend(a.symmetry, b.symmetry),
        zoom: blend(a.zoom, b.zoom),
        rotate: blend(a.rotate, b.rotate),
        skew: blend(a.skew, b.skew),
        center_x: blend(a.center_x, b.center_x),
        center_y: blend(a.center_y, b.center_y),
        warp: blend(a.warp, b.warp),
        hue,
        hue_spread: blend(a.hue_spread, b.hue_spread),
        saturation: blend(a.saturation, b.saturation),
        brightness: blend(a.brightness, b.brightness),
        color_bands: blend(a.color_bands, b.color_bands),
    }
    .sanitized()
}

/// How many whole-rig slots the B15 snapshot bank holds. Fixed rather than
/// dynamic: a bank is a row of buttons an operator learns by position, and a
/// growable one would make slot 5 mean something different tomorrow.
pub const SNAPSHOT_BANK_SLOTS: usize = 8;

/// The bank's glide bounds, in beats. Zero is the explicit snap; the upper
/// bound matches the Morph glide the bank hands its recall to.
pub const SNAPSHOT_BANK_MAX_GLIDE_BEATS: f64 = 64.0;

/// B15's snapshot bank: eight whole-rig slots and one glide time.
///
/// A slot holds exactly what a Morph slot holds, because recall does not
/// invent a second way to interpolate a rig — it loads the slot into the
/// existing Morph A/B and glides. Ownership transfer, midpoint discretes,
/// wrapped hues, and stale-topology purges therefore come free, and there is
/// only ever one law deciding what "between two rigs" means.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SnapshotBank {
    /// Exactly [`SNAPSHOT_BANK_SLOTS`] entries after sanitize. `None` is an
    /// empty slot, which recall refuses rather than treating as a default rig.
    pub slots: Vec<Option<MorphSlot>>,
    /// How long a recall takes to travel, in beats.
    pub glide_beats: f64,
}

impl Default for SnapshotBank {
    fn default() -> Self {
        Self {
            slots: vec![None; SNAPSHOT_BANK_SLOTS],
            glide_beats: 4.0,
        }
    }
}

impl SnapshotBank {
    /// Normalize a bank that arrived from a patch: the slot vector is padded
    /// or truncated to the fixed width, and the glide is clamped into range
    /// with a non-finite value taking the neutral default rather than an
    /// extreme.
    pub fn sanitized(&self) -> Self {
        let mut slots = self.slots.clone();
        slots.resize(SNAPSHOT_BANK_SLOTS, None);
        let default = Self::default();
        let glide_beats = if self.glide_beats.is_finite() {
            self.glide_beats.clamp(0.0, SNAPSHOT_BANK_MAX_GLIDE_BEATS)
        } else {
            default.glide_beats
        };
        Self { slots, glide_beats }
    }

    /// Whether any slot holds a rig. An empty bank is skipped in a patch, so
    /// every pre-B15 patch keeps its bytes and its canonical hash.
    pub fn is_empty(&self) -> bool {
        self.slots.iter().all(Option::is_none)
    }

    /// Whether this bank is the exact pre-B15 state: nothing stored, and the
    /// glide untouched.
    pub fn is_default(&self) -> bool {
        let clean = self.sanitized();
        clean.is_empty() && (clean.glide_beats - Self::default().glide_beats).abs() < f64::EPSILON
    }

    pub fn slot(&self, index: usize) -> Option<&MorphSlot> {
        self.slots.get(index).and_then(Option::as_ref)
    }

    /// Store a rig. Out-of-range indices are refused rather than clamped: a
    /// bank button that silently wrote to a different slot would be worse than
    /// one that did nothing.
    pub fn store(&mut self, index: usize, slot: MorphSlot) -> bool {
        if index >= SNAPSHOT_BANK_SLOTS {
            return false;
        }
        if self.slots.len() != SNAPSHOT_BANK_SLOTS {
            self.slots.resize(SNAPSHOT_BANK_SLOTS, None);
        }
        self.slots[index] = Some(slot);
        true
    }

    /// Empty one slot. Returns whether anything was there.
    pub fn clear_slot(&mut self, index: usize) -> bool {
        if index >= SNAPSHOT_BANK_SLOTS || self.slots.len() <= index {
            return false;
        }
        self.slots[index].take().is_some()
    }

    /// Which slots currently hold a rig, for the panel's lamps.
    pub fn filled(&self) -> Vec<bool> {
        let clean = self.sanitized();
        clean.slots.iter().map(Option::is_some).collect()
    }
}

/// One serializable A or B morph slot. This is the neutral persistence type;
/// it has no dependency on `patch` and no runtime layer handles.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct MorphSlot {
    pub master: MorphMasterSnapshot,
    /// Optional so legacy A/B slots leave the newly introduced master
    /// transform directly editable instead of silently claiming identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub master_transform: Option<SpatialTransform>,
    /// Optional so legacy slots leave M4 controls directly editable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub master_motion: Option<MotionConfig>,
    pub ntsc: MorphNtscSnapshot,
    pub temporal: MorphTemporalSnapshot,
    pub layers: Vec<LayerMorphSnapshot>,
    /// Authored master Collision Rack. Absence means this slot predates racks
    /// and therefore cannot claim any rack value while the crossfader moves.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub master_rack: Option<VisualRack>,
    /// One rack per saved layer position. The outer option distinguishes a
    /// legacy omitted section from an explicitly captured vector containing
    /// empty racks. Raw deserialization is bounded before allocation grows.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_layer_racks"
    )]
    pub layer_racks: Option<Vec<VisualRack>>,
    /// One-level group values and group racks. Its strict deserializer rejects
    /// duplicate/zero IDs, invalid cursors, and malformed one-level topology.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub composition: Option<CompositionTree>,
    /// Authored S3b gesture state. Absence means this slot predates gesture
    /// etching and therefore cannot claim any canvas value while the crossfader
    /// moves.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gesture: Option<GestureMorphSnapshot>,
}

/// The gesture world one Morph slot owns.
///
/// The split here is the whole S3b Morph law. `canvas` holds the three authored
/// continuous controls and interpolates like any other value. `track_checksum`
/// is the canonical digest of the recording the slot was captured against — a
/// *recorded track is topology, not a value*, so Morph holds only its identity,
/// carries that identity from A after equality has been proven, and can never
/// blend, rewrite, truncate, or reorder a single recorded event. Two slots
/// captured against different recordings are two different pieces rather than
/// two ends of a blend, exactly as two different Displace donors are.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GestureMorphSnapshot {
    pub canvas: GestureCanvasConfig,
    /// Empty means the slot was captured with nothing recorded.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub track_checksum: String,
}

impl GestureMorphSnapshot {
    pub fn sanitized(&self) -> Self {
        Self {
            canvas: self.canvas.sanitized(),
            track_checksum: self.track_checksum.clone(),
        }
    }
}

/// A recorded track is topology: two slots may only blend their canvas values
/// when they name the exact same recording. This is the S3b analogue of
/// `displace_route_matches`.
fn gesture_track_matches(a: &GestureMorphSnapshot, b: &GestureMorphSnapshot) -> bool {
    a.track_checksum == b.track_checksum
}

/// Blend only the authored canvas controls. The recorded track's identity is
/// carried from A — A/B compatibility already proved it equal — and is never
/// interpolated, so no morph position can synthesize a third recording that
/// neither slot captured.
fn interpolate_gesture(
    a: &GestureMorphSnapshot,
    b: &GestureMorphSnapshot,
    weights: [f32; 2],
) -> Option<GestureMorphSnapshot> {
    if !gesture_track_matches(a, b) {
        return None;
    }
    Some(GestureMorphSnapshot {
        canvas: GestureCanvasConfig {
            radius: blend_finite(a.canvas.radius, b.canvas.radius, weights),
            strength: blend_finite(a.canvas.strength, b.canvas.strength, weights),
            retention: blend_finite(a.canvas.retention, b.canvas.retention, weights),
        }
        .sanitized(),
        track_checksum: a.track_checksum.clone(),
    })
}

fn remapped_saved_position_after_move(
    position: crate::performance::SavedLayerPosition,
    from: usize,
    to: usize,
) -> crate::performance::SavedLayerPosition {
    let position_index = position.get() as usize;
    let remapped = if position_index == from {
        to
    } else if from < to && position_index > from && position_index <= to {
        position_index - 1
    } else if to < from && position_index >= to && position_index < from {
        position_index + 1
    } else {
        position_index
    };
    u32::try_from(remapped)
        .ok()
        .and_then(crate::performance::SavedLayerPosition::new)
        .unwrap_or(position)
}

/// Every saved image route one node owns, addressed by slot.
///
/// Slot index is route identity, so a node owning two routes hands back two
/// independent borrows and a positional layer edit can never slide one slot's
/// donor into the other. The match is exhaustive on purpose: a future kind that
/// carries a route must be listed here, not silently skipped.
fn saved_node_image_taps_mut(
    kind: &mut VisualNodeKind,
) -> [Option<&mut SavedImageTap>; SYMMETRY_IMAGE_SLOTS] {
    match kind {
        VisualNodeKind::Mask(MaskParams::Image(matte)) => [Some(&mut matte.tap), None],
        VisualNodeKind::Displace(params) => [Some(&mut params.tap), None],
        // Residual names both of its slots. Rewriting one route of the pair
        // and leaving its partner stale would desynchronize the two Morph
        // endpoints and make the node's own route-equality gate refuse to
        // interpolate.
        VisualNodeKind::Residual(params) => [Some(&mut params.structure), Some(&mut params.detail)],
        VisualNodeKind::Symmetry(params) => {
            let [first, second] = &mut params.donors;
            [Some(first), Some(second)]
        }
        VisualNodeKind::LegacyCanonical
        | VisualNodeKind::LegacyTemporal
        | VisualNodeKind::Transform(_)
        | VisualNodeKind::DigitalColor(_)
        | VisualNodeKind::Key(_)
        | VisualNodeKind::Cellular(_)
        | VisualNodeKind::Shift(_)
        | VisualNodeKind::Grain(_)
        | VisualNodeKind::Study(_)
        | VisualNodeKind::ScanProcessor(_)
        | VisualNodeKind::BlockDct(_)
        | VisualNodeKind::PixelSort(_)
        | VisualNodeKind::Avalanche(_)
        | VisualNodeKind::Mask(MaskParams::Rectangle(_) | MaskParams::Ellipse(_)) => [None, None],
    }
}

/// Every saved motion route one node owns, addressed by slot. A motion route
/// never enters the image dependency graph, but it names the same saved layer
/// positions and must survive a reorder identically.
fn saved_node_motion_donors_mut(
    kind: &mut VisualNodeKind,
) -> [Option<&mut SavedMotionDonor>; SYMMETRY_MOTION_SLOTS] {
    match kind {
        VisualNodeKind::Symmetry(params) => {
            let [first, second] = &mut params.motion;
            [Some(first), Some(second)]
        }
        VisualNodeKind::LegacyCanonical
        | VisualNodeKind::LegacyTemporal
        | VisualNodeKind::Transform(_)
        | VisualNodeKind::DigitalColor(_)
        | VisualNodeKind::Key(_)
        | VisualNodeKind::Cellular(_)
        | VisualNodeKind::Shift(_)
        | VisualNodeKind::Grain(_)
        | VisualNodeKind::Mask(_)
        | VisualNodeKind::Residual(_)
        | VisualNodeKind::Study(_)
        | VisualNodeKind::ScanProcessor(_)
        | VisualNodeKind::BlockDct(_)
        | VisualNodeKind::PixelSort(_)
        | VisualNodeKind::Avalanche(_)
        | VisualNodeKind::Displace(_) => [None, None],
    }
}

fn remap_saved_tap_after_move(tap: &mut SavedImageTap, from: usize, to: usize) {
    let SavedImageSource::SelectedLayer {
        layer_position,
        stage,
    } = tap.source
    else {
        return;
    };
    tap.source = SavedImageSource::SelectedLayer {
        layer_position: remapped_saved_position_after_move(layer_position, from, to),
        stage,
    };
}

fn remap_saved_tap_after_remove(tap: &mut SavedImageTap, removed: usize) {
    let SavedImageSource::SelectedLayer {
        layer_position,
        stage,
    } = tap.source
    else {
        return;
    };
    let position_index = layer_position.get() as usize;
    tap.source = if position_index == removed {
        // The vacated position becomes a tombstone that never rebinds to a
        // later occupant of the same index.
        SavedImageSource::MissingSelectedLayer {
            saved_position: layer_position,
            stage,
        }
    } else if position_index > removed {
        SavedImageSource::SelectedLayer {
            layer_position: crate::performance::SavedLayerPosition::new(
                layer_position
                    .get()
                    .checked_sub(1)
                    .expect("a position greater than the removed index is nonzero"),
            )
            .expect("decrementing a valid saved position remains valid"),
            stage,
        }
    } else {
        tap.source
    };
}

fn remap_saved_motion_donor_after_move(donor: &mut SavedMotionDonor, from: usize, to: usize) {
    let SavedMotionDonor::Selected { saved_position } = *donor else {
        return;
    };
    *donor = SavedMotionDonor::Selected {
        saved_position: remapped_saved_position_after_move(saved_position, from, to),
    };
}

fn remap_saved_motion_donor_after_remove(donor: &mut SavedMotionDonor, removed: usize) {
    let SavedMotionDonor::Selected { saved_position } = *donor else {
        return;
    };
    let position_index = saved_position.get() as usize;
    *donor = if position_index == removed {
        SavedMotionDonor::Missing { saved_position }
    } else if position_index > removed {
        SavedMotionDonor::Selected {
            saved_position: crate::performance::SavedLayerPosition::new(
                saved_position
                    .get()
                    .checked_sub(1)
                    .expect("a position greater than the removed index is nonzero"),
            )
            .expect("decrementing a valid saved position remains valid"),
        }
    } else {
        *donor
    };
}

fn remap_saved_rack_routes_after_move(rack: &mut VisualRack, from: usize, to: usize) {
    let node_ids = rack.iter().map(|node| node.stable_id).collect::<Vec<_>>();
    for node_id in node_ids {
        let Some(node) = rack.get_mut(node_id) else {
            continue;
        };
        for tap in saved_node_image_taps_mut(&mut node.kind)
            .into_iter()
            .flatten()
        {
            remap_saved_tap_after_move(tap, from, to);
        }
        for donor in saved_node_motion_donors_mut(&mut node.kind)
            .into_iter()
            .flatten()
        {
            remap_saved_motion_donor_after_move(donor, from, to);
        }
    }
}

fn remap_saved_rack_routes_after_remove(rack: &mut VisualRack, removed: usize) {
    let node_ids = rack.iter().map(|node| node.stable_id).collect::<Vec<_>>();
    for node_id in node_ids {
        let Some(node) = rack.get_mut(node_id) else {
            continue;
        };
        for tap in saved_node_image_taps_mut(&mut node.kind)
            .into_iter()
            .flatten()
        {
            remap_saved_tap_after_remove(tap, removed);
        }
        for donor in saved_node_motion_donors_mut(&mut node.kind)
            .into_iter()
            .flatten()
        {
            remap_saved_motion_donor_after_remove(donor, removed);
        }
    }
}

fn remap_score_driver_after_move(
    driver: &mut CollisionScoreLoopDriverConfig,
    from: usize,
    to: usize,
) {
    let CollisionScoreLoopDriverConfig::SelectedLayer { saved_position } = driver else {
        return;
    };
    *saved_position = remapped_saved_position_after_move(*saved_position, from, to);
}

fn remap_score_driver_after_remove(driver: &mut CollisionScoreLoopDriverConfig, removed: usize) {
    let CollisionScoreLoopDriverConfig::SelectedLayer { saved_position } = *driver else {
        return;
    };
    let position = saved_position.get() as usize;
    *driver = if position == removed {
        CollisionScoreLoopDriverConfig::MissingSelectedLayer { saved_position }
    } else if position > removed {
        CollisionScoreLoopDriverConfig::SelectedLayer {
            saved_position: crate::performance::SavedLayerPosition::new(
                saved_position
                    .get()
                    .checked_sub(1)
                    .expect("a position greater than the removed index is nonzero"),
            )
            .expect("decrementing a valid saved position remains valid"),
        }
    } else {
        *driver
    };
}

fn remap_garden_matte_route_after_move(
    route: &mut RefreshGardenMatteRouteConfig,
    from: usize,
    to: usize,
) {
    let RefreshGardenMatteRouteConfig::SelectedLayer { saved_position, .. } = route else {
        return;
    };
    *saved_position = remapped_saved_position_after_move(*saved_position, from, to);
}

fn remap_garden_matte_route_after_remove(
    route: &mut RefreshGardenMatteRouteConfig,
    removed: usize,
) {
    let RefreshGardenMatteRouteConfig::SelectedLayer {
        saved_position,
        stage,
    } = *route
    else {
        return;
    };
    let position = saved_position.get() as usize;
    *route = if position == removed {
        RefreshGardenMatteRouteConfig::MissingSelectedLayer {
            saved_position,
            stage,
        }
    } else if position > removed {
        RefreshGardenMatteRouteConfig::SelectedLayer {
            saved_position: crate::performance::SavedLayerPosition::new(
                saved_position
                    .get()
                    .checked_sub(1)
                    .expect("a position greater than the removed index is nonzero"),
            )
            .expect("decrementing a valid saved position remains valid"),
            stage,
        }
    } else {
        *route
    };
}

fn remap_garden_motion_route_after_move(
    route: &mut RefreshGardenMotionRouteConfig,
    from: usize,
    to: usize,
) {
    let RefreshGardenMotionRouteConfig::SelectedLayer { saved_position } = route else {
        return;
    };
    *saved_position = remapped_saved_position_after_move(*saved_position, from, to);
}

fn remap_garden_motion_route_after_remove(
    route: &mut RefreshGardenMotionRouteConfig,
    removed: usize,
) {
    let RefreshGardenMotionRouteConfig::SelectedLayer { saved_position } = *route else {
        return;
    };
    let position = saved_position.get() as usize;
    *route = if position == removed {
        RefreshGardenMotionRouteConfig::MissingSelectedLayer { saved_position }
    } else if position > removed {
        RefreshGardenMotionRouteConfig::SelectedLayer {
            saved_position: crate::performance::SavedLayerPosition::new(
                saved_position
                    .get()
                    .checked_sub(1)
                    .expect("a position greater than the removed index is nonzero"),
            )
            .expect("decrementing a valid saved position remains valid"),
        }
    } else {
        *route
    };
}

fn remap_motion_donor_after_move(donor: &mut MotionDonorConfig, from: usize, to: usize) {
    let MotionDonorConfig::Selected { saved_position } = donor else {
        return;
    };
    *saved_position = remapped_saved_position_after_move(*saved_position, from, to);
}

/// Remap both Field Collider inputs after a layer move.
///
/// The two slots are remapped independently through the same single-donor law
/// the transplant uses. This is deliberately a two-slot *variant* rather than a
/// loop over a collection: slot identity is route identity, so A and B are
/// named fields, and remapping them one at a time is what keeps input B's saved
/// position from sliding when input A is cleared.
///
/// A Field Collider donor lives on the Motion subsystem path, not the visual
/// rack. It is deliberately absent from `remap_saved_rack_routes_after_move`
/// and its siblings, which match on `VisualNodeKind` and would never see it.
fn remap_motion_collider_inputs_after_move(
    collider: &mut FieldColliderConfig,
    from: usize,
    to: usize,
) {
    remap_motion_donor_after_move(&mut collider.input_a, from, to);
    remap_motion_donor_after_move(&mut collider.input_b, from, to);
}

/// Remap both Field Collider inputs after a layer removal. A slot whose layer
/// was the one removed becomes a `Missing` tombstone and never rebinds.
fn remap_motion_collider_inputs_after_remove(collider: &mut FieldColliderConfig, removed: usize) {
    remap_motion_donor_after_remove(&mut collider.input_a, removed);
    remap_motion_donor_after_remove(&mut collider.input_b, removed);
}

fn remap_motion_donor_after_remove(donor: &mut MotionDonorConfig, removed: usize) {
    let MotionDonorConfig::Selected { saved_position } = *donor else {
        return;
    };
    let position = saved_position.get() as usize;
    *donor = if position == removed {
        MotionDonorConfig::Missing { saved_position }
    } else if position > removed {
        MotionDonorConfig::Selected {
            saved_position: crate::performance::SavedLayerPosition::new(
                saved_position
                    .get()
                    .checked_sub(1)
                    .expect("a position greater than the removed index is nonzero"),
            )
            .expect("decrementing a valid saved position remains valid"),
        }
    } else {
        *donor
    };
}

impl MorphSlot {
    pub fn capture(
        master: &EffectUniforms,
        master_transform: &SpatialTransform,
        ntsc: &NtscParams,
        temporal: &TemporalParams,
        layers: &[Layer],
    ) -> Self {
        let layer_ids: Vec<_> = layers.iter().map(Layer::stable_layer_id).collect();
        let mut temporal_snapshot = MorphTemporalSnapshot::capture(temporal);
        temporal_snapshot.originals.garden.matte_route =
            RefreshGardenMatteRouteConfig::from_runtime_for_capture(
                temporal.originals.garden.matte_route,
                &layer_ids,
            );
        temporal_snapshot.originals.garden.motion_route =
            RefreshGardenMotionRouteConfig::from_runtime_for_capture(
                temporal.originals.garden.motion_route,
                &layer_ids,
            );
        temporal_snapshot.originals.score.loop_driver =
            CollisionScoreLoopDriverConfig::from_runtime_for_capture(
                temporal.originals.score.loop_driver,
                &layer_ids,
            );
        Self {
            master: MorphMasterSnapshot::capture(master),
            master_transform: Some(master_transform.sanitized()),
            master_motion: None,
            ntsc: MorphNtscSnapshot::capture(ntsc),
            temporal: temporal_snapshot,
            layers: layers
                .iter()
                .enumerate()
                .map(|(position, layer)| LayerMorphSnapshot::capture(position, layer, &layer_ids))
                .collect(),
            master_rack: None,
            layer_racks: None,
            composition: None,
            gesture: None,
        }
        .sanitized()
    }

    /// Attach the authored gesture world to an already-captured slot.
    ///
    /// `track_checksum` is the recording's canonical digest, never its events:
    /// a Morph slot names a recording, it does not own one.
    pub fn with_gesture(
        mut self,
        canvas: crate::gesture_canvas::GestureCanvasParams,
        track_checksum: String,
    ) -> Self {
        self.gesture = Some(GestureMorphSnapshot {
            canvas: GestureCanvasConfig::from_params(canvas).sanitized(),
            track_checksum,
        });
        self
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub fn capture_with_motion(
        master: &EffectUniforms,
        master_transform: &SpatialTransform,
        master_motion: &MotionParams,
        ntsc: &NtscParams,
        temporal: &TemporalParams,
        layers: &[Layer],
    ) -> Self {
        let mut slot = Self::capture(master, master_transform, ntsc, temporal, layers);
        slot.master_motion = Some(MotionConfig::from_params(*master_motion).sanitized());
        slot
    }

    /// Capture the same legacy parameter world plus the complete authored
    /// creative-value world. Topology is retained only as an ownership
    /// signature: sampling and application never insert, remove, or reorder a
    /// node, layer, group, or image edge.
    #[allow(clippy::too_many_arguments)]
    pub fn capture_with_composition(
        master: &EffectUniforms,
        master_transform: &SpatialTransform,
        ntsc: &NtscParams,
        temporal: &TemporalParams,
        layers: &[Layer],
        master_rack: &RuntimeVisualRack,
        layer_racks: &[RuntimeVisualRack],
        composition: &RuntimeComposition,
    ) -> Result<Self, String> {
        if layer_racks.len() != layers.len() {
            return Err(format!(
                "morph capture has {} layer racks for {} layers",
                layer_racks.len(),
                layers.len()
            ));
        }
        if layer_racks.len() > MAX_MORPH_LAYER_RACKS {
            return Err(format!(
                "morph capture has {} layer racks; limit is {MAX_MORPH_LAYER_RACKS}",
                layer_racks.len()
            ));
        }
        master_rack
            .validate_for_scope(crate::visual_rack::LegacyRackScope::Master)
            .map_err(|error| format!("invalid master rack: {error}"))?;
        for (position, rack) in layer_racks.iter().enumerate() {
            rack.validate_for_scope(crate::visual_rack::LegacyRackScope::Layer)
                .map_err(|error| format!("invalid layer rack {position}: {error}"))?;
        }
        let position_of_layer = |wanted| {
            layers
                .iter()
                .position(|layer| layer.stable_layer_id() == wanted)
                .and_then(|position| u32::try_from(position).ok())
                .and_then(crate::performance::SavedLayerPosition::new)
        };
        let saved_master_rack = master_rack
            .capture_routes(position_of_layer)
            .map_err(|error| format!("capture morph master rack: {error}"))?;
        let saved_layer_racks = layer_racks
            .iter()
            .enumerate()
            .map(|(position, rack)| {
                rack.capture_routes(position_of_layer)
                    .map_err(|error| format!("capture morph layer rack {position}: {error}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let saved_composition = composition
            .capture(|wanted| {
                layers
                    .iter()
                    .position(|layer| layer.stable_layer_id() == wanted)
                    .and_then(|position| u32::try_from(position).ok())
                    .and_then(crate::performance::SavedLayerPosition::new)
            })
            .map_err(|error| format!("capture morph composition: {error}"))?;

        let mut slot = Self::capture(master, master_transform, ntsc, temporal, layers);
        slot.master_rack = Some(saved_master_rack);
        slot.layer_racks = Some(saved_layer_racks);
        slot.composition = Some(saved_composition);
        Ok(slot)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn capture_with_composition_and_motion(
        master: &EffectUniforms,
        master_transform: &SpatialTransform,
        master_motion: &MotionParams,
        ntsc: &NtscParams,
        temporal: &TemporalParams,
        layers: &[Layer],
        master_rack: &RuntimeVisualRack,
        layer_racks: &[RuntimeVisualRack],
        composition: &RuntimeComposition,
    ) -> Result<Self, String> {
        let mut slot = Self::capture_with_composition(
            master,
            master_transform,
            ntsc,
            temporal,
            layers,
            master_rack,
            layer_racks,
            composition,
        )?;
        slot.master_motion = Some(MotionConfig::from_params(*master_motion).sanitized());
        Ok(slot)
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
            master_transform: self.master_transform.map(SpatialTransform::sanitized),
            master_motion: self.master_motion.map(MotionConfig::sanitized),
            ntsc: self.ntsc.sanitized(),
            temporal: self.temporal.sanitized(),
            layers,
            master_rack: self.master_rack.clone(),
            layer_racks: self.layer_racks.clone(),
            composition: self.composition.clone(),
            gesture: self.gesture.as_ref().map(GestureMorphSnapshot::sanitized),
        }
    }

    /// Keep positional snapshots aligned after removing one live layer while
    /// preserving the independent master/NTSC/temporal worlds.
    fn remap_layers_after_remove(&mut self, removed: usize) {
        remap_garden_matte_route_after_remove(
            &mut self.temporal.originals.garden.matte_route,
            removed,
        );
        remap_garden_motion_route_after_remove(
            &mut self.temporal.originals.garden.motion_route,
            removed,
        );
        remap_score_driver_after_remove(&mut self.temporal.originals.score.loop_driver, removed);
        if let Some(motion) = &mut self.master_motion {
            remap_motion_donor_after_remove(&mut motion.transplant.donor, removed);
            remap_motion_collider_inputs_after_remove(&mut motion.collider, removed);
        }
        for layer in &mut self.layers {
            if let Some(motion) = &mut layer.motion {
                remap_motion_donor_after_remove(&mut motion.transplant.donor, removed);
                remap_motion_collider_inputs_after_remove(&mut motion.collider, removed);
            }
        }
        if let Some(master_rack) = &mut self.master_rack {
            remap_saved_rack_routes_after_remove(master_rack, removed);
        }
        if let Some(racks) = &mut self.layer_racks {
            for rack in racks.iter_mut() {
                remap_saved_rack_routes_after_remove(rack, removed);
            }
        }
        self.layers.retain_mut(|layer| {
            if layer.position == removed {
                return false;
            }
            if layer.position > removed {
                layer.position -= 1;
            }
            true
        });
        if let Some(racks) = &mut self.layer_racks {
            if removed < racks.len() {
                racks.remove(removed);
            } else {
                self.layer_racks = None;
            }
        }
        // CompositionTree intentionally exposes only validated structural
        // transactions. A legacy positional layer edit cannot partially
        // rewrite a captured group graph, so group-value ownership is dropped.
        self.composition = None;
    }

    /// Apply the same stable stack permutation as the live layer move.
    fn remap_layers_after_move(&mut self, from: usize, to: usize) {
        if from == to {
            return;
        }
        remap_garden_matte_route_after_move(
            &mut self.temporal.originals.garden.matte_route,
            from,
            to,
        );
        remap_garden_motion_route_after_move(
            &mut self.temporal.originals.garden.motion_route,
            from,
            to,
        );
        remap_score_driver_after_move(&mut self.temporal.originals.score.loop_driver, from, to);
        if let Some(motion) = &mut self.master_motion {
            remap_motion_donor_after_move(&mut motion.transplant.donor, from, to);
            remap_motion_collider_inputs_after_move(&mut motion.collider, from, to);
        }
        for layer in &mut self.layers {
            if let Some(motion) = &mut layer.motion {
                remap_motion_donor_after_move(&mut motion.transplant.donor, from, to);
                remap_motion_collider_inputs_after_move(&mut motion.collider, from, to);
            }
        }
        if let Some(master_rack) = &mut self.master_rack {
            remap_saved_rack_routes_after_move(master_rack, from, to);
        }
        if let Some(racks) = &mut self.layer_racks {
            for rack in racks.iter_mut() {
                remap_saved_rack_routes_after_move(rack, from, to);
            }
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
        if let Some(racks) = &mut self.layer_racks {
            if from < racks.len() && to < racks.len() {
                let rack = racks.remove(from);
                racks.insert(to, rack);
            } else {
                self.layer_racks = None;
            }
        }
        self.composition = None;
    }
}

struct BoundedMorphLayerRacks(Vec<VisualRack>);

impl<'de> Deserialize<'de> for BoundedMorphLayerRacks {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct RacksVisitor;

        impl<'de> Visitor<'de> for RacksVisitor {
            type Value = BoundedMorphLayerRacks;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "at most {MAX_MORPH_LAYER_RACKS} layer racks")
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut racks = Vec::with_capacity(
                    sequence.size_hint().unwrap_or(0).min(MAX_MORPH_LAYER_RACKS),
                );
                while let Some(rack) = sequence.next_element::<VisualRack>()? {
                    if racks.len() == MAX_MORPH_LAYER_RACKS {
                        return Err(de::Error::custom(format_args!(
                            "morph slot may contain at most {MAX_MORPH_LAYER_RACKS} layer racks"
                        )));
                    }
                    racks.push(rack);
                }
                Ok(BoundedMorphLayerRacks(racks))
            }
        }

        deserializer.deserialize_seq(RacksVisitor)
    }
}

fn deserialize_optional_layer_racks<'de, D>(
    deserializer: D,
) -> Result<Option<Vec<VisualRack>>, D::Error>
where
    D: Deserializer<'de>,
{
    Option::<BoundedMorphLayerRacks>::deserialize(deserializer)
        .map(|value| value.map(|value| value.0))
}

/// A pure, detached morph result suitable for either live or offline use.
#[derive(Debug, Clone, PartialEq)]
pub struct MorphSample {
    pub master: MorphMasterSnapshot,
    pub master_transform: Option<SpatialTransform>,
    pub master_motion: Option<MotionConfig>,
    pub ntsc: MorphNtscSnapshot,
    pub temporal: MorphTemporalSnapshot,
    pub layers: Vec<LayerMorphSnapshot>,
    /// Present only when both slots contain the same ordered master topology.
    pub master_rack: Option<VisualRack>,
    /// Present only when both slots own every positional rack and each ordered
    /// node identity/kind signature matches.
    pub layer_racks: Option<Vec<VisualRack>>,
    /// Present only when group membership/root topology and every group rack
    /// signature match. The sampled tree contains values, never new topology.
    pub composition: Option<CompositionTree>,
    /// Present only when both slots captured a gesture world against the same
    /// recording. It carries authored canvas values and the track's identity —
    /// never any recorded event.
    pub gesture: Option<GestureMorphSnapshot>,
}

impl MorphSample {
    /// Compatibility adapter for the live renderer. Runtime master fields and
    /// layers missing from either morph slot are left untouched.
    pub fn apply_to(
        &self,
        master: &mut EffectUniforms,
        master_transform: &mut SpatialTransform,
        ntsc: &mut NtscParams,
        temporal: &mut TemporalParams,
        layers: &mut [Layer],
    ) {
        self.master.apply_to(master);
        if let Some(transform) = self.master_transform {
            *master_transform = transform.sanitized();
        }
        *ntsc = self.ntsc.to_params();
        let mut sampled_temporal = self.temporal.to_params();
        let layer_ids: Vec<_> = layers.iter().map(Layer::stable_layer_id).collect();
        sampled_temporal.originals.garden.matte_route = self
            .temporal
            .originals
            .garden
            .matte_route
            .resolve_runtime(&layer_ids);
        sampled_temporal.originals.garden.motion_route = self
            .temporal
            .originals
            .garden
            .motion_route
            .resolve_runtime(&layer_ids);
        sampled_temporal.originals.score.loop_driver = resolve_score_loop_driver(
            self.temporal.originals.score.loop_driver,
            |saved_position| saved_position.resolve(layers).map(Layer::stable_layer_id),
        );
        *temporal = sampled_temporal;
        for sampled in &self.layers {
            let Some(layer) = layers.get_mut(sampled.position) else {
                continue;
            };
            sampled.apply_to(layer);
            if let Some(motion) = sampled.motion {
                layer.motion = motion.resolve_runtime(&layer_ids);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub fn apply_to_with_motion(
        &self,
        master: &mut EffectUniforms,
        master_transform: &mut SpatialTransform,
        master_motion: &mut MotionParams,
        ntsc: &mut NtscParams,
        temporal: &mut TemporalParams,
        layers: &mut [Layer],
    ) {
        self.apply_to(master, master_transform, ntsc, temporal, layers);
        if let Some(motion) = self.master_motion {
            let layer_ids: Vec<_> = layers.iter().map(Layer::stable_layer_id).collect();
            *master_motion = motion.resolve_runtime(&layer_ids);
        }
    }

    /// Apply advanced creative values without granting a morph snapshot
    /// authority over topology or ID cursors. Any live mismatch is inert.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_to_with_composition(
        &self,
        master: &mut EffectUniforms,
        master_transform: &mut SpatialTransform,
        ntsc: &mut NtscParams,
        temporal: &mut TemporalParams,
        layers: &mut [Layer],
        master_rack: &mut RuntimeVisualRack,
        layer_racks: &mut [RuntimeVisualRack],
        composition: &mut RuntimeComposition,
    ) {
        self.apply_to(master, master_transform, ntsc, temporal, layers);
        let layer_ids: Vec<_> = layers.iter().map(Layer::stable_layer_id).collect();
        let group_exists = |group_id| composition.contains_group(group_id);
        if let Some(sampled) = &self.master_rack {
            let sampled = sampled.resolve_routes(
                |position| position.resolve(&layer_ids).copied(),
                group_exists,
            );
            let _ = apply_runtime_rack_values_strict(&sampled, master_rack);
        }
        if let Some(sampled) = &self.layer_racks {
            if sampled.len() == layer_racks.len() {
                for (sampled, live) in sampled.iter().zip(layer_racks) {
                    let sampled = sampled.resolve_routes(
                        |position| position.resolve(&layer_ids).copied(),
                        group_exists,
                    );
                    let _ = apply_runtime_rack_values_strict(&sampled, live);
                }
            }
        }
        if let Some(sampled) = &self.composition {
            if let Ok(sampled) = sampled.resolve(|saved_position| {
                saved_position.resolve(layers).map(Layer::stable_layer_id)
            }) {
                let _ = apply_runtime_composition_values_strict(&sampled, composition);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn apply_to_with_composition_and_motion(
        &self,
        master: &mut EffectUniforms,
        master_transform: &mut SpatialTransform,
        master_motion: &mut MotionParams,
        ntsc: &mut NtscParams,
        temporal: &mut TemporalParams,
        layers: &mut [Layer],
        master_rack: &mut RuntimeVisualRack,
        layer_racks: &mut [RuntimeVisualRack],
        composition: &mut RuntimeComposition,
        gesture_canvas: &mut crate::gesture_canvas::GestureCanvasParams,
    ) {
        self.apply_to_with_composition(
            master,
            master_transform,
            ntsc,
            temporal,
            layers,
            master_rack,
            layer_racks,
            composition,
        );
        if let Some(motion) = self.master_motion {
            let layer_ids: Vec<_> = layers.iter().map(Layer::stable_layer_id).collect();
            *master_motion = motion.resolve_runtime(&layer_ids);
        }
        self.apply_gesture_to(gesture_canvas);
    }

    /// Write only the sampled authored canvas controls.
    ///
    /// There is deliberately no recorded-track destination here. A sample
    /// carries a track's identity so ownership can be decided; it never
    /// carries events, so no morph position can rewrite, truncate, or reorder
    /// what the operator actually recorded.
    pub fn apply_gesture_to(&self, canvas: &mut crate::gesture_canvas::GestureCanvasParams) {
        if let Some(gesture) = &self.gesture {
            *canvas = gesture.canvas.to_params();
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
            LayerMorphControl::Transform => a.transform.is_some() && b.transform.is_some(),
            LayerMorphControl::Motion => a.motion.is_some() && b.motion.is_some(),
            // Both slots must have captured a pattern source at this
            // position; two slots captured against different source kinds
            // own nothing and stay directly editable.
            LayerMorphControl::Pattern => a.pattern.is_some() && b.pattern.is_some(),
        }
    }

    /// Legacy slots did not contain this field and therefore cannot own it.
    pub(crate) fn controls_master_transform(&self) -> bool {
        matches!(
            (&self.a, &self.b),
            (Some(a), Some(b)) if a.master_transform.is_some() && b.master_transform.is_some()
        )
    }

    pub(crate) fn controls_master_motion(&self) -> bool {
        matches!(
            (&self.a, &self.b),
            (Some(a), Some(b)) if a.master_motion.is_some() && b.master_motion.is_some()
        )
    }

    /// Ownership of the authored canvas controls additionally requires that
    /// both slots name the same recording. Two slots captured against
    /// different tracks own nothing and stay directly editable.
    pub(crate) fn controls_master_gesture(&self) -> bool {
        matches!(
            (&self.a, &self.b),
            (Some(a), Some(b))
                if match (&a.gesture, &b.gesture) {
                    (Some(a), Some(b)) => gesture_track_matches(a, b),
                    _ => false,
                }
        )
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

        let master_rack = match (&a.master_rack, &b.master_rack) {
            (Some(a), Some(b)) => interpolate_rack(a, b, weights, choose_b),
            _ => None,
        };
        let layer_racks = match (&a.layer_racks, &b.layer_racks) {
            (Some(a), Some(b)) if a.len() == b.len() => a
                .iter()
                .zip(b)
                .map(|(a, b)| interpolate_rack(a, b, weights, choose_b))
                .collect::<Option<Vec<_>>>(),
            _ => None,
        };
        let composition = match (&a.composition, &b.composition) {
            (Some(a), Some(b)) => interpolate_composition(a, b, weights, choose_b),
            _ => None,
        };

        Some(MorphSample {
            master: MorphMasterSnapshot::interpolate(&a.master, &b.master, weights, choose_b),
            master_transform: match (a.master_transform, b.master_transform) {
                (Some(a), Some(b)) => Some(SpatialTransform::interpolate(a, b, weights, choose_b)),
                _ => None,
            },
            master_motion: match (a.master_motion, b.master_motion) {
                (Some(a), Some(b)) => Some(interpolate_motion_config(a, b, weights, choose_b)),
                _ => None,
            },
            ntsc: MorphNtscSnapshot::interpolate(&a.ntsc, &b.ntsc, weights, choose_b),
            temporal: MorphTemporalSnapshot::interpolate(
                &a.temporal,
                &b.temporal,
                weights,
                choose_b,
            ),
            layers,
            master_rack,
            layer_racks,
            composition,
            gesture: match (&a.gesture, &b.gesture) {
                (Some(a), Some(b)) => interpolate_gesture(a, b, weights),
                _ => None,
            },
        })
    }

    /// Write a sampled state into live base parameters. This preserves the
    /// pre-existing API while sharing the exact interpolation path with the
    /// exporter-facing pure sampler.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "legacy flat apply API is retained for compatibility and parity tests"
        )
    )]
    pub fn apply(
        &self,
        t: f32,
        master: &mut EffectUniforms,
        master_transform: &mut SpatialTransform,
        ntsc: &mut NtscParams,
        temporal: &mut TemporalParams,
        layers: &mut [Layer],
    ) {
        if let Some(sample) = self.sample(t) {
            sample.apply_to(master, master_transform, ntsc, temporal, layers);
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub fn apply_with_motion(
        &self,
        t: f32,
        master: &mut EffectUniforms,
        master_transform: &mut SpatialTransform,
        master_motion: &mut MotionParams,
        ntsc: &mut NtscParams,
        temporal: &mut TemporalParams,
        layers: &mut [Layer],
    ) {
        if let Some(sample) = self.sample(t) {
            sample.apply_to_with_motion(
                master,
                master_transform,
                master_motion,
                ntsc,
                temporal,
                layers,
            );
        }
    }

    /// Advanced sibling of [`Self::apply`]. The sampled value world is
    /// written only into live racks/groups with an identical topology
    /// signature; mismatches remain directly editable and untouched.
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub fn apply_with_composition(
        &self,
        t: f32,
        master: &mut EffectUniforms,
        master_transform: &mut SpatialTransform,
        ntsc: &mut NtscParams,
        temporal: &mut TemporalParams,
        layers: &mut [Layer],
        master_rack: &mut RuntimeVisualRack,
        layer_racks: &mut [RuntimeVisualRack],
        composition: &mut RuntimeComposition,
    ) {
        if let Some(sample) = self.sample(t) {
            sample.apply_to_with_composition(
                master,
                master_transform,
                ntsc,
                temporal,
                layers,
                master_rack,
                layer_racks,
                composition,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn apply_with_composition_and_motion(
        &self,
        t: f32,
        master: &mut EffectUniforms,
        master_transform: &mut SpatialTransform,
        master_motion: &mut MotionParams,
        ntsc: &mut NtscParams,
        temporal: &mut TemporalParams,
        layers: &mut [Layer],
        master_rack: &mut RuntimeVisualRack,
        layer_racks: &mut [RuntimeVisualRack],
        composition: &mut RuntimeComposition,
        gesture_canvas: &mut crate::gesture_canvas::GestureCanvasParams,
    ) {
        if let Some(sample) = self.sample(t) {
            sample.apply_to_with_composition_and_motion(
                master,
                master_transform,
                master_motion,
                ntsc,
                temporal,
                layers,
                master_rack,
                layer_racks,
                composition,
                gesture_canvas,
            );
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

fn racks_share_topology(a: &VisualRack, b: &VisualRack) -> bool {
    a.topology_signature() == b.topology_signature()
        && a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(a, b)| a.stable_id == b.stable_id && saved_node_topology_matches(a.kind, b.kind))
}

fn saved_node_topology_matches(a: VisualNodeKind, b: VisualNodeKind) -> bool {
    if a.tag() != b.tag() {
        return false;
    }
    match (a, b) {
        (
            VisualNodeKind::Mask(MaskParams::Rectangle(_)),
            VisualNodeKind::Mask(MaskParams::Rectangle(_)),
        )
        | (
            VisualNodeKind::Mask(MaskParams::Ellipse(_)),
            VisualNodeKind::Mask(MaskParams::Ellipse(_)),
        ) => true,
        (
            VisualNodeKind::Mask(MaskParams::Image(a)),
            VisualNodeKind::Mask(MaskParams::Image(b)),
        ) => image_matte_route_matches(a, b),
        (VisualNodeKind::Mask(_), VisualNodeKind::Mask(_)) => false,
        (VisualNodeKind::Displace(a), VisualNodeKind::Displace(b)) => displace_route_matches(a, b),
        (VisualNodeKind::Symmetry(a), VisualNodeKind::Symmetry(b)) => symmetry_route_matches(a, b),
        (VisualNodeKind::Residual(a), VisualNodeKind::Residual(b)) => residual_route_matches(a, b),
        _ => true,
    }
}

fn image_matte_route_matches(a: ImageMatte, b: ImageMatte) -> bool {
    a.tap == b.tap && a.channel == b.channel && a.invert == b.invert
}

/// Displace amounts interpolate only when both slots name the exact same donor.
/// Two different routes are different topology, not two ends of a blend.
fn displace_route_matches(a: DisplaceParams, b: DisplaceParams) -> bool {
    a.tap == b.tap
}

/// A Symmetry Field's geometry interpolates only when both slots describe the
/// exact same routing graph, AND-ed across all four fixed slots.
///
/// The two masks join the routes because they decide admission: a slot whose
/// source or motion bit is clear claims no dependency edge and reserves no
/// binding, so two slots with different masks describe two different graphs
/// rather than two ends of a blend. Slot index is route identity, so the
/// comparison is positional and a swapped pair is a mismatch, not a match.
fn symmetry_route_matches(a: SymmetryParams, b: SymmetryParams) -> bool {
    a.donors == b.donors
        && a.motion == b.motion
        && a.source_mask.sanitized() == b.source_mask.sanitized()
        && a.motion_mask == b.motion_mask
}

/// Residual owns two authored routes and both are compared, slot by slot. A
/// pair that agrees on structure but not on detail is still different topology:
/// blending across a mismatched donor would recombine one snapshot's large
/// scale with an image the other snapshot never named.
fn residual_route_matches(a: ResidualParams, b: ResidualParams) -> bool {
    a.routes() == b.routes()
}

fn interpolate_rack(
    a: &VisualRack,
    b: &VisualRack,
    weights: [f32; 2],
    choose_b: bool,
) -> Option<VisualRack> {
    if !racks_share_topology(a, b) {
        return None;
    }
    let mut sampled = a.clone();
    for (node_a, node_b) in a.iter().zip(b.iter()) {
        let node = sampled
            .get_mut(node_a.stable_id)
            .expect("topology-compatible rack contains every sampled node");
        node.enabled = pick(node_a.enabled, node_b.enabled, choose_b);
        node.wet = finite_clamp(blend_finite(node_a.wet, node_b.wet, weights), 1.0, 0.0, 1.0);
        node.blend = pick(node_a.blend, node_b.blend, choose_b);
        node.kind = interpolate_node_kind(node_a.kind, node_b.kind, weights, choose_b)?;
    }
    Some(sampled)
}

fn interpolate_node_kind(
    a: VisualNodeKind,
    b: VisualNodeKind,
    weights: [f32; 2],
    choose_b: bool,
) -> Option<VisualNodeKind> {
    Some(match (a, b) {
        (VisualNodeKind::LegacyCanonical, VisualNodeKind::LegacyCanonical) => {
            VisualNodeKind::LegacyCanonical
        }
        (VisualNodeKind::LegacyTemporal, VisualNodeKind::LegacyTemporal) => {
            VisualNodeKind::LegacyTemporal
        }
        (VisualNodeKind::Transform(a), VisualNodeKind::Transform(b)) => {
            VisualNodeKind::Transform(SpatialTransform::interpolate(a, b, weights, choose_b))
        }
        (VisualNodeKind::DigitalColor(a), VisualNodeKind::DigitalColor(b)) => {
            VisualNodeKind::DigitalColor(interpolate_digital(a, b, weights))
        }
        (VisualNodeKind::Key(a), VisualNodeKind::Key(b)) => {
            VisualNodeKind::Key(interpolate_key(a, b, weights, choose_b))
        }
        (VisualNodeKind::Cellular(a), VisualNodeKind::Cellular(b)) => {
            VisualNodeKind::Cellular(interpolate_cellular(a, b, weights, choose_b))
        }
        (VisualNodeKind::Shift(a), VisualNodeKind::Shift(b)) => {
            VisualNodeKind::Shift(interpolate_shift(a, b, weights, choose_b))
        }
        (VisualNodeKind::Grain(a), VisualNodeKind::Grain(b)) => {
            VisualNodeKind::Grain(interpolate_grain(a, b, weights, choose_b))
        }
        (VisualNodeKind::Mask(a), VisualNodeKind::Mask(b)) => {
            VisualNodeKind::Mask(interpolate_mask(a, b, weights, choose_b)?)
        }
        (VisualNodeKind::Displace(a), VisualNodeKind::Displace(b)) => {
            VisualNodeKind::Displace(interpolate_displace(a, b, weights, choose_b)?)
        }
        (VisualNodeKind::Symmetry(a), VisualNodeKind::Symmetry(b)) => {
            VisualNodeKind::Symmetry(interpolate_symmetry(a, b, weights, choose_b)?)
        }
        (VisualNodeKind::Residual(a), VisualNodeKind::Residual(b)) => {
            VisualNodeKind::Residual(interpolate_residual(a, b, weights, choose_b)?)
        }
        // A Study has no continuous value: the digest names a whole validated
        // document, so the pair is one discrete endpoint choice at the
        // midpoint — the Field Collider whole-block law. No position can
        // synthesize a third document neither slot captured.
        (VisualNodeKind::Study(a), VisualNodeKind::Study(b)) => {
            VisualNodeKind::Study(pick(a, b, choose_b))
        }
        (VisualNodeKind::ScanProcessor(a), VisualNodeKind::ScanProcessor(b)) => {
            VisualNodeKind::ScanProcessor(interpolate_scan_processor(a, b, weights, choose_b))
        }
        // The B6 corruption trio owns no routes, so any same-kind pair
        // interpolates: continuous values blend and the avalanche's
        // predictor axis recalls an endpoint at the midpoint.
        (VisualNodeKind::BlockDct(a), VisualNodeKind::BlockDct(b)) => {
            VisualNodeKind::BlockDct(crate::block_dct::BlockDctParams {
                amount: blend_finite(a.amount, b.amount, weights),
                quantize: blend_finite(a.quantize, b.quantize, weights),
                hf_penalty: blend_finite(a.hf_penalty, b.hf_penalty, weights),
                chroma_crush: blend_finite(a.chroma_crush, b.chroma_crush, weights),
                block: blend_finite(a.block, b.block, weights),
            })
        }
        (VisualNodeKind::PixelSort(a), VisualNodeKind::PixelSort(b)) => {
            VisualNodeKind::PixelSort(crate::pixel_sort::PixelSortParams {
                amount: blend_finite(a.amount, b.amount, weights),
                threshold: blend_finite(a.threshold, b.threshold, weights),
            })
        }
        (VisualNodeKind::Avalanche(a), VisualNodeKind::Avalanche(b)) => {
            VisualNodeKind::Avalanche(crate::filter_avalanche::AvalancheParams {
                amount: blend_finite(a.amount, b.amount, weights),
                run: blend_finite(a.run, b.run, weights),
                axis: pick(a.axis, b.axis, choose_b),
            })
        }
        _ => return None,
    })
}

/// The Scan Processor's fifteen continuous controls blend; the two geometry
/// counts are plan-time draw sizes and the two reversals are discrete laws, so
/// all four recall an endpoint at the midpoint like every other authored
/// discrete choice. No route exists, so any pair of scan nodes interpolates.
fn interpolate_scan_processor(
    a: ScanProcessorParams,
    b: ScanProcessorParams,
    weights: [f32; 2],
    choose_b: bool,
) -> ScanProcessorParams {
    ScanProcessorParams {
        lines: pick(a.lines, b.lines, choose_b),
        samples_per_line: pick(a.samples_per_line, b.samples_per_line, choose_b),
        amount: blend_finite(a.amount, b.amount, weights),
        ribbon_width: blend_finite(a.ribbon_width, b.ribbon_width, weights),
        velocity_mix: blend_finite(a.velocity_mix, b.velocity_mix, weights),
        tilt_x: blend_finite(a.tilt_x, b.tilt_x, weights),
        tilt_y: blend_finite(a.tilt_y, b.tilt_y, weights),
        perspective: blend_finite(a.perspective, b.perspective, weights),
        s_curve: blend_finite(a.s_curve, b.s_curve, weights),
        skew: blend_finite(a.skew, b.skew, weights),
        collapse: blend_finite(a.collapse, b.collapse, weights),
        reverse_h: pick(a.reverse_h, b.reverse_h, choose_b),
        reverse_v: pick(a.reverse_v, b.reverse_v, choose_b),
        osc_amount: blend_finite(a.osc_amount, b.osc_amount, weights),
        osc_freq: blend_finite(a.osc_freq, b.osc_freq, weights),
        osc_lock: blend_finite(a.osc_lock, b.osc_lock, weights),
        lissajous: blend_finite(a.lissajous, b.lissajous, weights),
        mono: blend_finite(a.mono, b.mono, weights),
        hue: blend_finite(a.hue, b.hue, weights),
    }
}

/// Amounts blend continuously; the boundary law is discrete and switches at the
/// midpoint like every other authored enum. The donor route is topology: A/B
/// compatibility already proved it equal, so it is carried, never interpolated.
fn interpolate_displace(
    a: DisplaceParams,
    b: DisplaceParams,
    weights: [f32; 2],
    choose_b: bool,
) -> Option<DisplaceParams> {
    if !displace_route_matches(a, b) {
        return None;
    }
    Some(DisplaceParams {
        tap: a.tap,
        amount_x: blend_finite(a.amount_x, b.amount_x, weights),
        amount_y: blend_finite(a.amount_y, b.amount_y, weights),
        boundary: pick(a.boundary, b.boundary, choose_b),
    })
}

/// The Symmetry Field's declared continuous controls blend; every angular one
/// blends along its shortest wrapped arc. Mode and boundary are discrete
/// authored laws and switch at the midpoint like every other enum. The four
/// routes and the two masks are topology: A/B compatibility already proved them
/// equal, so they are carried, never interpolated.
fn interpolate_symmetry(
    a: SymmetryParams,
    b: SymmetryParams,
    weights: [f32; 2],
    choose_b: bool,
) -> Option<SymmetryParams> {
    if !symmetry_route_matches(a, b) {
        return None;
    }
    Some(SymmetryParams {
        mode: pick(a.mode, b.mode, choose_b),
        base_folds: blend_finite(a.base_folds, b.base_folds, weights),
        fold_offset: blend_finite(a.fold_offset, b.fold_offset, weights),
        radial_phase_deg: blend_wrapped_degrees(a.radial_phase_deg, b.radial_phase_deg, weights),
        orbit_phase: blend_finite(a.orbit_phase, b.orbit_phase, weights),
        planar_axis_deg: blend_wrapped_degrees(a.planar_axis_deg, b.planar_axis_deg, weights),
        planar_phase: blend_finite(a.planar_phase, b.planar_phase, weights),
        cell_skew: blend_finite(a.cell_skew, b.cell_skew, weights),
        spiral_scale: blend_finite(a.spiral_scale, b.spiral_scale, weights),
        orbit_radius: blend_finite(a.orbit_radius, b.orbit_radius, weights),
        orbit_spin_deg: blend_wrapped_degrees(a.orbit_spin_deg, b.orbit_spin_deg, weights),
        center: [
            blend_finite(a.center[0], b.center[0], weights),
            blend_finite(a.center[1], b.center[1], weights),
        ],
        boundary: pick(a.boundary, b.boundary, choose_b),
        motion_gain: blend_finite(a.motion_gain, b.motion_gain, weights),
        hue_span: blend_finite(a.hue_span, b.hue_span, weights),
        // Seed identity is an endpoint recall, never an interpolated RNG.
        // Blending it would synthesize a third 32-record sector table that
        // neither slot ever captured.
        seed: pick(a.seed, b.seed, choose_b),
        source_mask: a.source_mask,
        motion_mask: a.motion_mask,
        donors: a.donors,
        motion: a.motion,
    })
}

/// `mix` and `detail_gain` blend continuously; the block vocabulary and the
/// quantization law are discrete and switch at the midpoint like every other
/// authored enum. Both routes are topology: A/B compatibility already proved
/// the pair equal, so they are carried, never interpolated.
///
/// Each endpoint is normalized before the blend because `detail_gain` is
/// neutral at one, not at zero: `blend_finite`'s hostile fallback would
/// otherwise silence a gain that neither snapshot authored as silent.
fn interpolate_residual(
    a: ResidualParams,
    b: ResidualParams,
    weights: [f32; 2],
    choose_b: bool,
) -> Option<ResidualParams> {
    if !residual_route_matches(a, b) {
        return None;
    }
    Some(ResidualParams {
        algorithm_version: pick(a.algorithm_version, b.algorithm_version, choose_b),
        structure: a.structure,
        detail: a.detail,
        block: pick(a.block, b.block, choose_b),
        quantization: pick(a.quantization, b.quantization, choose_b),
        mix: blend_finite(
            finite_clamp(a.mix, 0.0, 0.0, 1.0),
            finite_clamp(b.mix, 0.0, 0.0, 1.0),
            weights,
        ),
        detail_gain: blend_finite(
            finite_clamp(a.detail_gain, 1.0, 0.0, 4.0),
            finite_clamp(b.detail_gain, 1.0, 0.0, 4.0),
            weights,
        ),
        // Seed identity is an endpoint recall, never an interpolated RNG.
        seed: pick(a.seed, b.seed, choose_b),
    })
}

fn interpolate_digital(
    a: DigitalColorParams,
    b: DigitalColorParams,
    weights: [f32; 2],
) -> DigitalColorParams {
    DigitalColorParams {
        pixelate_size: blend_finite(a.pixelate_size, b.pixelate_size, weights),
        rgb_split: blend_finite(a.rgb_split, b.rgb_split, weights),
        downsample: blend_finite(a.downsample, b.downsample, weights),
        hue_shift: blend_wrapped_degrees(a.hue_shift, b.hue_shift, weights),
        saturation: blend_finite(a.saturation, b.saturation, weights),
        brightness: blend_finite(a.brightness, b.brightness, weights),
        contrast: blend_finite(a.contrast, b.contrast, weights),
        posterize: blend_finite(a.posterize, b.posterize, weights),
        invert: blend_finite(a.invert, b.invert, weights),
        vignette: blend_finite(a.vignette, b.vignette, weights),
        color_drift: blend_finite(a.color_drift, b.color_drift, weights),
    }
}

fn interpolate_key(a: KeyParams, b: KeyParams, weights: [f32; 2], choose_b: bool) -> KeyParams {
    KeyParams {
        mode: pick(a.mode, b.mode, choose_b),
        threshold: blend_finite(a.threshold, b.threshold, weights),
        softness: blend_finite(a.softness, b.softness, weights),
        color: std::array::from_fn(|index| blend_finite(a.color[index], b.color[index], weights)),
        tolerance: blend_finite(a.tolerance, b.tolerance, weights),
        invert: pick(a.invert, b.invert, choose_b),
    }
}

fn interpolate_cellular(
    a: CellularParams,
    b: CellularParams,
    weights: [f32; 2],
    choose_b: bool,
) -> CellularParams {
    CellularParams {
        amount: blend_finite(a.amount, b.amount, weights),
        scale: blend_finite(a.scale, b.scale, weights),
        warp: blend_finite(a.warp, b.warp, weights),
        speed: blend_finite(a.speed, b.speed, weights),
        gap_amount: blend_finite(a.gap_amount, b.gap_amount, weights),
        gap_threshold: blend_finite(a.gap_threshold, b.gap_threshold, weights),
        gap_softness: blend_finite(a.gap_softness, b.gap_softness, weights),
        seed: pick(a.seed, b.seed, choose_b),
    }
}

fn interpolate_shift(
    a: ShiftParams,
    b: ShiftParams,
    weights: [f32; 2],
    choose_b: bool,
) -> ShiftParams {
    ShiftParams {
        amount: blend_finite(a.amount, b.amount, weights),
        block_size: blend_finite(a.block_size, b.block_size, weights),
        density: blend_finite(a.density, b.density, weights),
        speed: blend_finite(a.speed, b.speed, weights),
        seed: pick(a.seed, b.seed, choose_b),
    }
}

fn interpolate_grain(
    a: GrainParams,
    b: GrainParams,
    weights: [f32; 2],
    choose_b: bool,
) -> GrainParams {
    GrainParams {
        intensity: blend_finite(a.intensity, b.intensity, weights),
        size: blend_finite(a.size, b.size, weights),
        algorithm: pick(a.algorithm, b.algorithm, choose_b),
        color: pick(a.color, b.color, choose_b),
        seed: pick(a.seed, b.seed, choose_b),
    }
}

fn interpolate_rectangle(
    a: RectangleMask,
    b: RectangleMask,
    weights: [f32; 2],
    choose_b: bool,
) -> RectangleMask {
    RectangleMask {
        center: std::array::from_fn(|index| {
            blend_finite(a.center[index], b.center[index], weights)
        }),
        size: std::array::from_fn(|index| blend_finite(a.size[index], b.size[index], weights)),
        rotation_deg: blend_wrapped_degrees(a.rotation_deg, b.rotation_deg, weights),
        feather: blend_finite(a.feather, b.feather, weights),
        invert: pick(a.invert, b.invert, choose_b),
    }
}

fn interpolate_ellipse(
    a: EllipseMask,
    b: EllipseMask,
    weights: [f32; 2],
    choose_b: bool,
) -> EllipseMask {
    EllipseMask {
        center: std::array::from_fn(|index| {
            blend_finite(a.center[index], b.center[index], weights)
        }),
        radii: std::array::from_fn(|index| blend_finite(a.radii[index], b.radii[index], weights)),
        rotation_deg: blend_wrapped_degrees(a.rotation_deg, b.rotation_deg, weights),
        feather: blend_finite(a.feather, b.feather, weights),
        invert: pick(a.invert, b.invert, choose_b),
    }
}

fn interpolate_image_matte(a: ImageMatte, b: ImageMatte, weights: [f32; 2]) -> ImageMatte {
    ImageMatte {
        // Donor, timing, channel and inversion are routing topology. A/B
        // compatibility already proved equality, so the snapshot carries the
        // route only as a signature and never morphs it.
        tap: a.tap,
        channel: a.channel,
        invert: a.invert,
        amount: blend_finite(a.amount, b.amount, weights),
        threshold: blend_finite(a.threshold, b.threshold, weights),
        softness: blend_finite(a.softness, b.softness, weights),
    }
}

fn interpolate_mask(
    a: MaskParams,
    b: MaskParams,
    weights: [f32; 2],
    choose_b: bool,
) -> Option<MaskParams> {
    match (a, b) {
        (MaskParams::Rectangle(a), MaskParams::Rectangle(b)) => Some(MaskParams::Rectangle(
            interpolate_rectangle(a, b, weights, choose_b),
        )),
        (MaskParams::Ellipse(a), MaskParams::Ellipse(b)) => Some(MaskParams::Ellipse(
            interpolate_ellipse(a, b, weights, choose_b),
        )),
        (MaskParams::Image(a), MaskParams::Image(b)) => {
            Some(MaskParams::Image(interpolate_image_matte(a, b, weights)))
        }
        _ => None,
    }
}

fn composition_topology_matches(a: &CompositionTree, b: &CompositionTree) -> bool {
    a.root() == b.root()
        && a.groups().len() == b.groups().len()
        && a.groups().zip(b.groups()).all(|(a, b)| {
            a.id == b.id
                && a.members == b.members
                && match (a.matte, b.matte) {
                    (Some(a), Some(b)) => image_matte_route_matches(a, b),
                    (None, None) => true,
                    _ => false,
                }
                && racks_share_topology(&a.rack, &b.rack)
        })
}

/// Blend the B8 bus-mixer bundle: continuous values follow the engaged
/// blend law, discrete laws (pattern, invert, rep, border colour, blend
/// mode) recall an endpoint at the midpoint like every closed vocabulary.
fn interpolate_bus_mixer(
    a: crate::mixing_boundary::BusMixerState,
    b: crate::mixing_boundary::BusMixerState,
    weights: [f32; 2],
    choose_b: bool,
) -> crate::mixing_boundary::BusMixerState {
    use crate::mixing_boundary::{BusMixParams, BusMixerState, DirtParams, MeltParams};
    fn pick<T>(a: T, b: T, choose_b: bool) -> T {
        if choose_b {
            b
        } else {
            a
        }
    }
    let value = |a_value: f32, b_value: f32, neutral: f32, min: f32, max: f32| {
        finite_clamp(blend_finite(a_value, b_value, weights), neutral, min, max)
    };
    BusMixerState {
        mix: BusMixParams {
            pattern: pick(a.mix.pattern, b.mix.pattern, choose_b),
            soft: value(a.mix.soft, b.mix.soft, 0.03, 0.0, 1.0),
            origin_x: value(a.mix.origin_x, b.mix.origin_x, 0.0, -1.0, 1.0),
            origin_y: value(a.mix.origin_y, b.mix.origin_y, 0.0, -1.0, 1.0),
            detail: value(a.mix.detail, b.mix.detail, 0.3, 0.0, 1.0),
            invert: pick(a.mix.invert, b.mix.invert, choose_b),
            rep: pick(a.mix.rep, b.mix.rep, choose_b),
            border: value(a.mix.border, b.mix.border, 0.0, 0.0, 1.0),
            border_color: pick(a.mix.border_color, b.mix.border_color, choose_b),
            blend: pick(a.mix.blend, b.mix.blend, choose_b),
        },
        dirt: DirtParams {
            dirt: value(a.dirt.dirt, b.dirt.dirt, 0.0, 0.0, 1.0),
            rate: value(a.dirt.rate, b.dirt.rate, 0.3, 0.0, 1.0),
            drop: value(a.dirt.drop, b.dirt.drop, 0.5, 0.0, 1.0),
            cut: value(a.dirt.cut, b.dirt.cut, 0.4, 0.0, 1.0),
            knock: value(a.dirt.knock, b.dirt.knock, 0.5, 0.0, 1.0),
            noise: value(a.dirt.noise, b.dirt.noise, 0.35, 0.0, 1.0),
        },
        melt: MeltParams {
            melt: value(a.melt.melt, b.melt.melt, 0.0, 0.0, 2.0),
            width: value(a.melt.width, b.melt.width, 0.3, 0.0, 2.0),
            hold: value(a.melt.hold, b.melt.hold, 0.6, 0.0, 1.5),
            swirl: value(a.melt.swirl, b.melt.swirl, 0.0, -1.0, 1.0),
            chroma: value(a.melt.chroma, b.melt.chroma, 0.5, 0.0, 1.0),
            creep: value(a.melt.creep, b.melt.creep, 0.35, 0.0, 1.0),
        },
    }
    .sanitized()
}

fn interpolate_composition(
    a: &CompositionTree,
    b: &CompositionTree,
    weights: [f32; 2],
    choose_b: bool,
) -> Option<CompositionTree> {
    if !composition_topology_matches(a, b) {
        return None;
    }
    let mut sampled = a.clone();
    sampled.set_bus_crossfade(finite_clamp(
        blend_finite(a.bus_crossfade(), b.bus_crossfade(), weights),
        0.5,
        0.0,
        1.0,
    ));
    sampled.set_mixer(interpolate_bus_mixer(
        a.mixer(),
        b.mixer(),
        weights,
        choose_b,
    ));
    for group_a in a.groups() {
        let group_b = b.group(group_a.id)?;
        let sampled_group = sampled.group_mut(group_a.id)?;
        sampled_group.opacity = finite_clamp(
            blend_finite(group_a.opacity, group_b.opacity, weights),
            1.0,
            0.0,
            1.0,
        );
        sampled_group.transform =
            SpatialTransform::interpolate(group_a.transform, group_b.transform, weights, choose_b);
        sampled_group.rack = interpolate_rack(&group_a.rack, &group_b.rack, weights, choose_b)?;
        sampled_group.matte = match (group_a.matte, group_b.matte) {
            (Some(a), Some(b)) => Some(interpolate_image_matte(a, b, weights)),
            (None, None) => None,
            _ => return None,
        };
        sampled_group.solo = pick(group_a.solo, group_b.solo, choose_b);
        sampled_group.bypass = pick(group_a.bypass, group_b.bypass, choose_b);
        sampled_group.bus = pick(group_a.bus, group_b.bus, choose_b);
    }
    Some(sampled)
}

/// Copy values between topology-compatible racks while retaining the live
/// rack's monotonic node cursor and ordered storage.
#[allow(
    dead_code,
    reason = "saved-rack value application supports compatibility/parity paths"
)]
pub(crate) fn apply_rack_values(sampled: &VisualRack, live: &mut VisualRack) -> bool {
    if !racks_share_topology(sampled, live) {
        return false;
    }
    for sampled_node in sampled.iter() {
        let live_node = live
            .get_mut(sampled_node.stable_id)
            .expect("topology-compatible live rack contains every node");
        live_node.enabled = sampled_node.enabled;
        live_node.wet = sampled_node.wet;
        live_node.blend = sampled_node.blend;
        if !apply_saved_node_kind_values(sampled_node.kind, &mut live_node.kind) {
            return false;
        }
    }
    true
}

#[allow(
    dead_code,
    reason = "implementation detail of the retained saved-rack apply API"
)]
fn apply_saved_node_kind_values(sampled: VisualNodeKind, live: &mut VisualNodeKind) -> bool {
    match (sampled, live) {
        (VisualNodeKind::LegacyCanonical, VisualNodeKind::LegacyCanonical)
        | (VisualNodeKind::LegacyTemporal, VisualNodeKind::LegacyTemporal) => true,
        (VisualNodeKind::Transform(value), VisualNodeKind::Transform(live)) => {
            *live = value;
            true
        }
        (VisualNodeKind::DigitalColor(value), VisualNodeKind::DigitalColor(live)) => {
            *live = value;
            true
        }
        (VisualNodeKind::Key(value), VisualNodeKind::Key(live)) => {
            *live = value;
            true
        }
        (VisualNodeKind::Cellular(value), VisualNodeKind::Cellular(live)) => {
            *live = value;
            true
        }
        (VisualNodeKind::Shift(value), VisualNodeKind::Shift(live)) => {
            *live = value;
            true
        }
        (VisualNodeKind::Grain(value), VisualNodeKind::Grain(live)) => {
            *live = value;
            true
        }
        (
            VisualNodeKind::Mask(MaskParams::Rectangle(value)),
            VisualNodeKind::Mask(MaskParams::Rectangle(live)),
        ) => {
            *live = value;
            true
        }
        (
            VisualNodeKind::Mask(MaskParams::Ellipse(value)),
            VisualNodeKind::Mask(MaskParams::Ellipse(live)),
        ) => {
            *live = value;
            true
        }
        (
            VisualNodeKind::Mask(MaskParams::Image(value)),
            VisualNodeKind::Mask(MaskParams::Image(live)),
        ) => {
            apply_saved_image_matte_values(value, live);
            true
        }
        (VisualNodeKind::Displace(value), VisualNodeKind::Displace(live)) => {
            apply_saved_displace_values(value, live);
            true
        }
        (VisualNodeKind::Symmetry(value), VisualNodeKind::Symmetry(live)) => {
            apply_saved_symmetry_values(value, live);
            true
        }
        (VisualNodeKind::Residual(value), VisualNodeKind::Residual(live)) => {
            apply_saved_residual_values(value, live);
            true
        }
        // Every Scan Processor field is a value — no route exists — so the
        // whole params bundle transfers, geometry counts and reversals
        // included, exactly like Cellular or Shift.
        (VisualNodeKind::ScanProcessor(value), VisualNodeKind::ScanProcessor(live)) => {
            *live = value;
            true
        }
        // The corruption trio is route-free too: the whole params bundle
        // transfers, the avalanche axis included, exactly like Cellular.
        (VisualNodeKind::BlockDct(value), VisualNodeKind::BlockDct(live)) => {
            *live = value;
            true
        }
        (VisualNodeKind::PixelSort(value), VisualNodeKind::PixelSort(live)) => {
            *live = value;
            true
        }
        (VisualNodeKind::Avalanche(value), VisualNodeKind::Avalanche(live)) => {
            *live = value;
            true
        }
        _ => false,
    }
}

/// Values only. The live donor route is preserved so applying a Look or preset
/// can never silently retarget a Displace at another image.
#[allow(
    dead_code,
    reason = "implementation detail of retained saved value application"
)]
fn apply_saved_displace_values(sampled: DisplaceParams, live: &mut DisplaceParams) {
    live.amount_x = sampled.amount_x;
    live.amount_y = sampled.amount_y;
    live.boundary = sampled.boundary;
}

/// Values only. The four live routes and the two live masks are preserved, so
/// applying a Look or a preset can never silently retarget a Symmetry Field at
/// another image or arm a source the operator never armed. The seed travels
/// with the values because it is a value: it selects a table, it does not route
/// one.
#[allow(
    dead_code,
    reason = "implementation detail of retained saved value application"
)]
fn apply_saved_symmetry_values(sampled: SymmetryParams, live: &mut SymmetryParams) {
    live.mode = sampled.mode;
    live.base_folds = sampled.base_folds;
    live.fold_offset = sampled.fold_offset;
    live.radial_phase_deg = sampled.radial_phase_deg;
    live.orbit_phase = sampled.orbit_phase;
    live.planar_axis_deg = sampled.planar_axis_deg;
    live.planar_phase = sampled.planar_phase;
    live.cell_skew = sampled.cell_skew;
    live.spiral_scale = sampled.spiral_scale;
    live.orbit_radius = sampled.orbit_radius;
    live.orbit_spin_deg = sampled.orbit_spin_deg;
    live.center = sampled.center;
    live.boundary = sampled.boundary;
    live.motion_gain = sampled.motion_gain;
    live.hue_span = sampled.hue_span;
    live.seed = sampled.seed;
}

/// Values only. Both live donor routes are preserved so applying a Look or a
/// preset can never silently retarget either half of a Residual recombination.
/// The quantization seed is a captured value here, not routing topology, so it
/// transfers exactly like the block and quantization laws.
#[allow(
    dead_code,
    reason = "implementation detail of retained saved value application"
)]
fn apply_saved_residual_values(sampled: ResidualParams, live: &mut ResidualParams) {
    live.algorithm_version = sampled.algorithm_version;
    live.block = sampled.block;
    live.quantization = sampled.quantization;
    live.mix = sampled.mix;
    live.detail_gain = sampled.detail_gain;
    live.seed = sampled.seed;
}

#[allow(
    dead_code,
    reason = "implementation detail of retained saved value application"
)]
fn apply_saved_image_matte_values(sampled: ImageMatte, live: &mut ImageMatte) {
    live.amount = sampled.amount;
    live.threshold = sampled.threshold;
    live.softness = sampled.softness;
}

/// Copy only group/rack values. Membership, root order, names, identities,
/// and monotonic cursors remain owned by the live composition.
#[allow(
    dead_code,
    reason = "saved-composition apply supports compatibility/parity paths"
)]
pub(crate) fn apply_composition_values(
    sampled: &CompositionTree,
    live: &mut CompositionTree,
) -> bool {
    if !composition_topology_matches(sampled, live) {
        return false;
    }
    live.set_bus_crossfade(sampled.bus_crossfade());
    live.set_mixer(sampled.mixer());
    for sampled_group in sampled.groups() {
        let live_group = live
            .group_mut(sampled_group.id)
            .expect("topology-compatible composition contains every group");
        live_group.opacity = sampled_group.opacity;
        live_group.transform = sampled_group.transform;
        let _ = apply_rack_values(&sampled_group.rack, &mut live_group.rack);
        if let (Some(sampled), Some(live)) = (sampled_group.matte, &mut live_group.matte) {
            apply_saved_image_matte_values(sampled, live);
        }
        live_group.solo = sampled_group.solo;
        live_group.bypass = sampled_group.bypass;
        live_group.bus = sampled_group.bus;
    }
    true
}

fn runtime_racks_share_value_topology(a: &RuntimeVisualRack, b: &RuntimeVisualRack) -> bool {
    a.topology_signature() == b.topology_signature()
        && a.len() == b.len()
        && a.iter().zip(b.iter()).all(|(a, b)| {
            a.stable_id == b.stable_id && runtime_node_value_topology_matches(a.kind, b.kind)
        })
}

fn runtime_node_value_topology_matches(a: RuntimeVisualNodeKind, b: RuntimeVisualNodeKind) -> bool {
    if a.tag() != b.tag() {
        return false;
    }
    match (a, b) {
        (
            RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Rectangle(_)),
            RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Rectangle(_)),
        )
        | (
            RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Ellipse(_)),
            RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Ellipse(_)),
        )
        | (
            RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(_)),
            RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(_)),
        ) => true,
        (RuntimeVisualNodeKind::Mask(_), RuntimeVisualNodeKind::Mask(_)) => false,
        _ => true,
    }
}

#[allow(
    dead_code,
    reason = "topology guard for retained saved-to-runtime apply API"
)]
fn saved_and_runtime_racks_share_value_topology(
    saved: &VisualRack,
    live: &RuntimeVisualRack,
) -> bool {
    saved.topology_signature() == live.topology_signature()
        && saved.len() == live.len()
        && saved.iter().zip(live.iter()).all(|(saved, live)| {
            saved.stable_id == live.stable_id
                && match (saved.kind, live.kind) {
                    (
                        VisualNodeKind::Mask(MaskParams::Rectangle(_)),
                        RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Rectangle(_)),
                    )
                    | (
                        VisualNodeKind::Mask(MaskParams::Ellipse(_)),
                        RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Ellipse(_)),
                    )
                    | (
                        VisualNodeKind::Mask(MaskParams::Image(_)),
                        RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(_)),
                    ) => true,
                    (VisualNodeKind::Mask(_), RuntimeVisualNodeKind::Mask(_)) => false,
                    (saved, live) => saved.tag() == live.tag(),
                }
        })
}

#[allow(
    dead_code,
    reason = "saved-to-runtime apply supports compatibility/parity paths"
)]
pub(crate) fn apply_saved_rack_values_to_runtime(
    sampled: &VisualRack,
    live: &mut RuntimeVisualRack,
) -> bool {
    if !saved_and_runtime_racks_share_value_topology(sampled, live) {
        return false;
    }
    for sampled_node in sampled.iter() {
        let live_node = live
            .get_mut(sampled_node.stable_id)
            .expect("topology-compatible runtime rack contains every node");
        live_node.enabled = sampled_node.enabled;
        live_node.wet = sampled_node.wet;
        live_node.blend = sampled_node.blend;
        if !apply_saved_node_kind_values_to_runtime(sampled_node.kind, &mut live_node.kind) {
            return false;
        }
    }
    true
}

#[allow(
    dead_code,
    reason = "implementation detail of retained saved-to-runtime apply API"
)]
fn apply_saved_node_kind_values_to_runtime(
    sampled: VisualNodeKind,
    live: &mut RuntimeVisualNodeKind,
) -> bool {
    match (sampled, live) {
        (VisualNodeKind::LegacyCanonical, RuntimeVisualNodeKind::LegacyCanonical)
        | (VisualNodeKind::LegacyTemporal, RuntimeVisualNodeKind::LegacyTemporal) => true,
        (VisualNodeKind::Transform(value), RuntimeVisualNodeKind::Transform(live)) => {
            *live = value;
            true
        }
        (VisualNodeKind::DigitalColor(value), RuntimeVisualNodeKind::DigitalColor(live)) => {
            *live = value;
            true
        }
        (VisualNodeKind::Key(value), RuntimeVisualNodeKind::Key(live)) => {
            *live = value;
            true
        }
        (VisualNodeKind::Cellular(value), RuntimeVisualNodeKind::Cellular(live)) => {
            *live = value;
            true
        }
        (VisualNodeKind::Shift(value), RuntimeVisualNodeKind::Shift(live)) => {
            *live = value;
            true
        }
        (VisualNodeKind::Grain(value), RuntimeVisualNodeKind::Grain(live)) => {
            *live = value;
            true
        }
        (
            VisualNodeKind::Mask(MaskParams::Rectangle(value)),
            RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Rectangle(live)),
        ) => {
            *live = value;
            true
        }
        (
            VisualNodeKind::Mask(MaskParams::Ellipse(value)),
            RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Ellipse(live)),
        ) => {
            *live = value;
            true
        }
        (
            VisualNodeKind::Mask(MaskParams::Image(value)),
            RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(live)),
        ) => {
            live.amount = value.amount;
            live.threshold = value.threshold;
            live.softness = value.softness;
            true
        }
        (VisualNodeKind::Displace(value), RuntimeVisualNodeKind::Displace(live)) => {
            live.amount_x = value.amount_x;
            live.amount_y = value.amount_y;
            live.boundary = value.boundary;
            true
        }
        (VisualNodeKind::Symmetry(value), RuntimeVisualNodeKind::Symmetry(live)) => {
            // Values only; the four live routes and the two live masks stay
            // under topology control so a preset never rewires the field.
            live.mode = value.mode;
            live.base_folds = value.base_folds;
            live.fold_offset = value.fold_offset;
            live.radial_phase_deg = value.radial_phase_deg;
            live.orbit_phase = value.orbit_phase;
            live.planar_axis_deg = value.planar_axis_deg;
            live.planar_phase = value.planar_phase;
            live.cell_skew = value.cell_skew;
            live.spiral_scale = value.spiral_scale;
            live.orbit_radius = value.orbit_radius;
            live.orbit_spin_deg = value.orbit_spin_deg;
            live.center = value.center;
            live.boundary = value.boundary;
            live.motion_gain = value.motion_gain;
            live.hue_span = value.hue_span;
            live.seed = value.seed;
            true
        }
        (VisualNodeKind::Residual(value), RuntimeVisualNodeKind::Residual(live)) => {
            // Values only; both live donor routes stay under topology control.
            live.algorithm_version = value.algorithm_version;
            live.block = value.block;
            live.quantization = value.quantization;
            live.mix = value.mix;
            live.detail_gain = value.detail_gain;
            live.seed = value.seed;
            true
        }
        // Every Scan Processor field is a value — no route exists — so the
        // whole params bundle transfers.
        (VisualNodeKind::ScanProcessor(value), RuntimeVisualNodeKind::ScanProcessor(live)) => {
            *live = value;
            true
        }
        // The corruption trio is route-free: the whole params bundle
        // transfers.
        (VisualNodeKind::BlockDct(value), RuntimeVisualNodeKind::BlockDct(live)) => {
            *live = value;
            true
        }
        (VisualNodeKind::PixelSort(value), RuntimeVisualNodeKind::PixelSort(live)) => {
            *live = value;
            true
        }
        (VisualNodeKind::Avalanche(value), RuntimeVisualNodeKind::Avalanche(live)) => {
            *live = value;
            true
        }
        _ => false,
    }
}

fn apply_runtime_rack_values(sampled: &RuntimeVisualRack, live: &mut RuntimeVisualRack) -> bool {
    if !runtime_racks_share_value_topology(sampled, live) {
        return false;
    }
    for sampled_node in sampled.iter() {
        let live_node = live
            .get_mut(sampled_node.stable_id)
            .expect("topology-compatible runtime rack contains every node");
        live_node.enabled = sampled_node.enabled;
        live_node.wet = sampled_node.wet;
        live_node.blend = sampled_node.blend;
        if !apply_runtime_node_kind_values(sampled_node.kind, &mut live_node.kind) {
            return false;
        }
    }
    true
}

fn runtime_racks_share_strict_topology(a: &RuntimeVisualRack, b: &RuntimeVisualRack) -> bool {
    runtime_racks_share_value_topology(a, b)
        && a.iter().zip(b.iter()).all(|(a, b)| match (a.kind, b.kind) {
            (
                RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(a)),
                RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(b)),
            ) => a.tap == b.tap && a.channel == b.channel && a.invert == b.invert,
            (RuntimeVisualNodeKind::Displace(a), RuntimeVisualNodeKind::Displace(b)) => {
                a.tap == b.tap
            }
            // All four slots plus both masks, exactly as the saved gate. A Look
            // that lands on a differently routed or differently armed Symmetry
            // Field must refuse the whole rack rather than rewire it.
            (RuntimeVisualNodeKind::Symmetry(a), RuntimeVisualNodeKind::Symmetry(b)) => {
                a.donors == b.donors
                    && a.motion == b.motion
                    && a.source_mask.sanitized() == b.source_mask.sanitized()
                    && a.motion_mask == b.motion_mask
            }
            // Both authored routes are compared slot by slot: a Look whose
            // structure route agrees but whose detail route does not is a
            // different recombination, not a values-only difference.
            (RuntimeVisualNodeKind::Residual(a), RuntimeVisualNodeKind::Residual(b)) => {
                a.routes() == b.routes()
            }
            _ => true,
        })
}

pub(crate) fn apply_runtime_rack_values_strict(
    sampled: &RuntimeVisualRack,
    live: &mut RuntimeVisualRack,
) -> bool {
    runtime_racks_share_strict_topology(sampled, live) && apply_runtime_rack_values(sampled, live)
}

fn apply_runtime_node_kind_values(
    sampled: RuntimeVisualNodeKind,
    live: &mut RuntimeVisualNodeKind,
) -> bool {
    match (sampled, live) {
        (RuntimeVisualNodeKind::LegacyCanonical, RuntimeVisualNodeKind::LegacyCanonical)
        | (RuntimeVisualNodeKind::LegacyTemporal, RuntimeVisualNodeKind::LegacyTemporal) => true,
        (RuntimeVisualNodeKind::Transform(value), RuntimeVisualNodeKind::Transform(live)) => {
            *live = value;
            true
        }
        (RuntimeVisualNodeKind::DigitalColor(value), RuntimeVisualNodeKind::DigitalColor(live)) => {
            *live = value;
            true
        }
        (RuntimeVisualNodeKind::Key(value), RuntimeVisualNodeKind::Key(live)) => {
            *live = value;
            true
        }
        (RuntimeVisualNodeKind::Cellular(value), RuntimeVisualNodeKind::Cellular(live)) => {
            *live = value;
            true
        }
        (RuntimeVisualNodeKind::Shift(value), RuntimeVisualNodeKind::Shift(live)) => {
            *live = value;
            true
        }
        (RuntimeVisualNodeKind::Grain(value), RuntimeVisualNodeKind::Grain(live)) => {
            *live = value;
            true
        }
        (
            RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Rectangle(value)),
            RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Rectangle(live)),
        ) => {
            *live = value;
            true
        }
        (
            RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Ellipse(value)),
            RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Ellipse(live)),
        ) => {
            *live = value;
            true
        }
        (
            RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(value)),
            RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(live)),
        ) => {
            apply_runtime_image_matte_values(value, live);
            true
        }
        (RuntimeVisualNodeKind::Displace(value), RuntimeVisualNodeKind::Displace(live)) => {
            // Values only; the live donor route stays under topology control.
            live.amount_x = value.amount_x;
            live.amount_y = value.amount_y;
            live.boundary = value.boundary;
            true
        }
        (RuntimeVisualNodeKind::Symmetry(value), RuntimeVisualNodeKind::Symmetry(live)) => {
            // Values only; the four live routes and both live masks stay under
            // topology control.
            live.mode = value.mode;
            live.base_folds = value.base_folds;
            live.fold_offset = value.fold_offset;
            live.radial_phase_deg = value.radial_phase_deg;
            live.orbit_phase = value.orbit_phase;
            live.planar_axis_deg = value.planar_axis_deg;
            live.planar_phase = value.planar_phase;
            live.cell_skew = value.cell_skew;
            live.spiral_scale = value.spiral_scale;
            live.orbit_radius = value.orbit_radius;
            live.orbit_spin_deg = value.orbit_spin_deg;
            live.center = value.center;
            live.boundary = value.boundary;
            live.motion_gain = value.motion_gain;
            live.hue_span = value.hue_span;
            live.seed = value.seed;
            true
        }
        (RuntimeVisualNodeKind::Residual(value), RuntimeVisualNodeKind::Residual(live)) => {
            // Values only; both live donor routes stay under topology control.
            live.algorithm_version = value.algorithm_version;
            live.block = value.block;
            live.quantization = value.quantization;
            live.mix = value.mix;
            live.detail_gain = value.detail_gain;
            live.seed = value.seed;
            true
        }
        // Every Scan Processor field is a value — no route exists — so the
        // whole params bundle transfers.
        (
            RuntimeVisualNodeKind::ScanProcessor(value),
            RuntimeVisualNodeKind::ScanProcessor(live),
        ) => {
            *live = value;
            true
        }
        // The corruption trio is route-free: the whole params bundle
        // transfers.
        (RuntimeVisualNodeKind::BlockDct(value), RuntimeVisualNodeKind::BlockDct(live)) => {
            *live = value;
            true
        }
        (RuntimeVisualNodeKind::PixelSort(value), RuntimeVisualNodeKind::PixelSort(live)) => {
            *live = value;
            true
        }
        (RuntimeVisualNodeKind::Avalanche(value), RuntimeVisualNodeKind::Avalanche(live)) => {
            *live = value;
            true
        }
        _ => false,
    }
}

fn apply_runtime_image_matte_values(sampled: RuntimeImageMatte, live: &mut RuntimeImageMatte) {
    live.amount = sampled.amount;
    live.threshold = sampled.threshold;
    live.softness = sampled.softness;
}

fn runtime_composition_topology_matches(a: &RuntimeComposition, b: &RuntimeComposition) -> bool {
    a.root() == b.root()
        && a.groups().len() == b.groups().len()
        && a.groups().zip(b.groups()).all(|(a, b)| {
            a.id == b.id
                && a.members == b.members
                && a.matte.is_some() == b.matte.is_some()
                && runtime_racks_share_value_topology(&a.rack, &b.rack)
        })
}

fn runtime_composition_strict_topology_matches(
    a: &RuntimeComposition,
    b: &RuntimeComposition,
) -> bool {
    runtime_composition_topology_matches(a, b)
        && a.groups().zip(b.groups()).all(|(a, b)| {
            (match (a.matte, b.matte) {
                (Some(a), Some(b)) => {
                    a.tap == b.tap && a.channel == b.channel && a.invert == b.invert
                }
                (None, None) => true,
                _ => false,
            }) && runtime_racks_share_strict_topology(&a.rack, &b.rack)
        })
}

fn apply_runtime_composition_values(
    sampled: &RuntimeComposition,
    live: &mut RuntimeComposition,
) -> bool {
    if !runtime_composition_topology_matches(sampled, live) {
        return false;
    }
    live.set_bus_crossfade(sampled.bus_crossfade());
    live.set_mixer(sampled.mixer());
    for sampled_group in sampled.groups() {
        let live_group = live
            .group_mut(sampled_group.id)
            .expect("topology-compatible runtime composition contains every group");
        live_group.opacity = sampled_group.opacity;
        live_group.transform = sampled_group.transform;
        let _ = apply_runtime_rack_values(&sampled_group.rack, &mut live_group.rack);
        if let (Some(sampled), Some(live)) = (sampled_group.matte, &mut live_group.matte) {
            apply_runtime_image_matte_values(sampled, live);
        }
        live_group.solo = sampled_group.solo;
        live_group.bypass = sampled_group.bypass;
        live_group.bus = sampled_group.bus;
    }
    true
}

pub(crate) fn apply_runtime_composition_values_strict(
    sampled: &RuntimeComposition,
    live: &mut RuntimeComposition,
) -> bool {
    runtime_composition_strict_topology_matches(sampled, live)
        && apply_runtime_composition_values(sampled, live)
}

fn normalized_position(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Deterministic detached-preflight order shared by live and offline Morph
/// materialization: requested sample, nearest captured endpoint (tie B), then
/// the other endpoint. The vector is hard-bounded to three unique entries.
pub(crate) fn preflight_sample_positions(value: f32) -> Vec<f32> {
    let requested = normalized_position(value);
    let nearest = if requested < 0.5 { 0.0 } else { 1.0 };
    let other = 1.0 - nearest;
    let mut attempts = Vec::with_capacity(3);
    attempts.push(requested);
    for endpoint in [nearest, other] {
        if !attempts
            .iter()
            .any(|candidate| (candidate - endpoint).abs() <= f32::EPSILON)
        {
            attempts.push(endpoint);
        }
    }
    attempts
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
    #[test]
    fn a_snapshot_bank_is_fixed_width_bounded_and_empty_by_default() {
        let bank = SnapshotBank::default();
        assert_eq!(bank.slots.len(), SNAPSHOT_BANK_SLOTS);
        assert!(bank.is_empty(), "a fresh bank stores nothing");
        assert!(bank.is_default(), "a fresh bank is the exact pre-B15 state");
        assert!(bank.filled().iter().all(|filled| !filled));

        // A patch may arrive with a short, long, or hostile bank.
        let ragged = SnapshotBank {
            slots: vec![Some(MorphSlot::default())],
            glide_beats: f64::NAN,
        }
        .sanitized();
        assert_eq!(ragged.slots.len(), SNAPSHOT_BANK_SLOTS, "short banks pad");
        assert!(ragged.slot(0).is_some());
        assert_eq!(
            ragged.glide_beats,
            SnapshotBank::default().glide_beats,
            "a non-finite glide takes the neutral default, not an extreme"
        );

        let overlong = SnapshotBank {
            slots: vec![None; SNAPSHOT_BANK_SLOTS + 5],
            glide_beats: 1_000.0,
        }
        .sanitized();
        assert_eq!(
            overlong.slots.len(),
            SNAPSHOT_BANK_SLOTS,
            "long banks truncate"
        );
        assert_eq!(overlong.glide_beats, SNAPSHOT_BANK_MAX_GLIDE_BEATS);
    }

    #[test]
    fn a_bank_slot_is_addressed_exactly_and_never_clamped_onto_a_neighbour() {
        let mut bank = SnapshotBank::default();
        assert!(bank.store(3, MorphSlot::default()), "slot 3 exists");
        assert!(bank.slot(3).is_some());
        assert!(bank.filled()[3]);
        assert!(!bank.filled()[4], "storing must not smear");
        assert!(!bank.is_empty());
        assert!(!bank.is_default());

        // Out of range is refused, not clamped: a bank button that silently
        // wrote to a different slot would be worse than one that did nothing.
        assert!(!bank.store(SNAPSHOT_BANK_SLOTS, MorphSlot::default()));
        assert!(!bank.clear_slot(SNAPSHOT_BANK_SLOTS));
        assert_eq!(
            bank.filled().iter().filter(|filled| **filled).count(),
            1,
            "a refused write must store nothing at all"
        );

        assert!(bank.clear_slot(3));
        assert!(
            !bank.clear_slot(3),
            "clearing an empty slot reports nothing"
        );
        assert!(bank.is_empty());
    }

    #[test]
    fn a_bank_round_trips_through_serde_and_an_absent_section_is_the_prior_path() {
        let mut bank = SnapshotBank::default();
        bank.store(0, MorphSlot::default());
        bank.glide_beats = 2.5;
        let yaml = serde_yaml::to_string(&bank).unwrap();
        let restored: SnapshotBank = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(restored.sanitized(), bank.sanitized());

        // Unknown fields are a rejection rather than a silent default.
        assert!(
            serde_yaml::from_str::<SnapshotBank>("slots: []\nglide_beats: 1.0\nextra: 3\n")
                .is_err(),
            "an unknown bank field must be refused"
        );
    }

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
    fn a_study_pair_is_one_discrete_endpoint_choice_at_the_midpoint() {
        use crate::visual_rack::{StudyRackParams, VisualNodeKind};
        let a = VisualNodeKind::Study(StudyRackParams {
            document_digest: Some([0x11; 32]),
        });
        let b = VisualNodeKind::Study(StudyRackParams {
            document_digest: Some([0x22; 32]),
        });
        // Before the midpoint the A document holds; after it, B — no
        // position can synthesize a third document neither slot captured.
        let early = interpolate_node_kind(a, b, [0.75, 0.25], false).unwrap();
        assert_eq!(early, a);
        let late = interpolate_node_kind(a, b, [0.25, 0.75], true).unwrap();
        assert_eq!(late, b);
        // Equal documents are trivially carried.
        let same = interpolate_node_kind(a, a, [0.5, 0.5], true).unwrap();
        assert_eq!(same, a);
    }

    /// A Scan Processor pair blends its fifteen continuous controls while
    /// the two geometry counts and the two reversals — plan-time geometry
    /// and discrete laws — recall an endpoint at the midpoint. No route
    /// exists, so any pair interpolates.
    #[test]
    fn a_scan_processor_pair_blends_values_and_recalls_discrete_laws_at_the_midpoint() {
        use crate::scan_processor::ScanProcessorParams;
        use crate::visual_rack::VisualNodeKind;
        let a = VisualNodeKind::ScanProcessor(ScanProcessorParams {
            amount: 0.2,
            tilt_x: -0.5,
            lines: 100,
            samples_per_line: 64,
            reverse_h: false,
            ..ScanProcessorParams::default()
        });
        let b = VisualNodeKind::ScanProcessor(ScanProcessorParams {
            amount: 0.8,
            tilt_x: 0.5,
            lines: 400,
            samples_per_line: 128,
            reverse_h: true,
            ..ScanProcessorParams::default()
        });
        let VisualNodeKind::ScanProcessor(early) =
            interpolate_node_kind(a, b, [0.75, 0.25], false).unwrap()
        else {
            panic!("scan processor kind")
        };
        assert!((early.amount - 0.35).abs() < 1e-6);
        assert!((early.tilt_x - (-0.25)).abs() < 1e-6);
        assert_eq!(early.lines, 100);
        assert_eq!(early.samples_per_line, 64);
        assert!(!early.reverse_h);
        let VisualNodeKind::ScanProcessor(late) =
            interpolate_node_kind(a, b, [0.25, 0.75], true).unwrap()
        else {
            panic!("scan processor kind")
        };
        assert!((late.amount - 0.65).abs() < 1e-6);
        assert_eq!(late.lines, 400);
        assert_eq!(late.samples_per_line, 128);
        assert!(late.reverse_h);
    }

    #[test]
    fn all_layer_blend_modes_round_trip_exactly_through_morph_storage() {
        for mode in BlendMode::ALL {
            let stored = MorphLayerBlendMode::capture(mode);
            assert_eq!(stored.to_blend_mode(), mode);
            let yaml = serde_yaml::to_string(&stored).unwrap();
            let restored: MorphLayerBlendMode = serde_yaml::from_str(&yaml).unwrap();
            assert_eq!(restored, stored);
        }
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
                ..MorphMasterSnapshot::default()
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
            transform: Some(SpatialTransform {
                position: if high { [0.75, -0.5] } else { [-0.25, 0.5] },
                scale: if high { [4.0, 4.0] } else { [1.0, 1.0] },
                rotation_deg: if high { -170.0 } else { 170.0 },
                ..SpatialTransform::default()
            }),
            motion: None,
            key_threshold: None,
            pattern: None,
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
    fn motion_morph_is_continuous_for_numbers_and_endpoint_exact_for_laws() {
        use crate::patch::{
            CurvedShutterConfig, CurvedShutterQualityConfig, FaradayConfig, MotionCarrierConfig,
            MotionDonorConfig, MotionFieldSourceConfig, MotionLatticeQualityConfig,
        };
        use crate::performance::SavedLayerPosition;

        let low = MotionConfig {
            field_source: MotionFieldSourceConfig::CodecVectors,
            lattice_quality: MotionLatticeQualityConfig::Draft,
            transplant: FaradayConfig {
                donor: MotionDonorConfig::Selected {
                    saved_position: SavedLayerPosition::new(0).unwrap(),
                },
                carrier: MotionCarrierConfig::Transparent,
                ..FaradayConfig::default()
            },
            shutter: CurvedShutterConfig {
                quality: CurvedShutterQualityConfig::Sharp,
                ..CurvedShutterConfig::default()
            },
            ..MotionConfig::default()
        };

        let high = MotionConfig {
            field_source: MotionFieldSourceConfig::Lattice,
            lattice_quality: MotionLatticeQualityConfig::High,
            transplant: FaradayConfig {
                amount: 1.0,
                donor: MotionDonorConfig::Missing {
                    saved_position: SavedLayerPosition::new(1).unwrap(),
                },
                carrier: MotionCarrierConfig::FirstSourceFrame,
                confidence_softness: 0.5,
                ..FaradayConfig::default()
            },
            shutter: CurvedShutterConfig {
                angle_degrees: 360.0,
                phase: 1.0,
                curvature: 2.0,
                chromatic_lag: 1.0,
                quality: CurvedShutterQualityConfig::High,
            },
            ..MotionConfig::default()
        };

        let mut a = MorphSlot {
            master_motion: Some(low),
            ..MorphSlot::default()
        };
        a.layers.push(LayerMorphSnapshot {
            position: 0,
            motion: Some(low),
            ..LayerMorphSnapshot::default()
        });
        let mut b = MorphSlot {
            master_motion: Some(high),
            ..MorphSlot::default()
        };
        b.layers.push(LayerMorphSnapshot {
            position: 0,
            motion: Some(high),
            ..LayerMorphSnapshot::default()
        });
        let morph = Morph {
            a: Some(a),
            b: Some(b),
            ..Morph::default()
        };

        let quarter = morph.sample(0.25).unwrap();
        let motion = quarter.master_motion.unwrap();
        close(motion.transplant.amount, 0.25);
        close(motion.shutter.angle_degrees, 90.0);
        assert_eq!(motion.field_source, MotionFieldSourceConfig::CodecVectors);
        assert_eq!(motion.lattice_quality, MotionLatticeQualityConfig::Draft);
        assert_eq!(motion.transplant.donor, low.transplant.donor);
        assert_eq!(motion.transplant.carrier, MotionCarrierConfig::Transparent);
        assert_eq!(motion.shutter.quality, CurvedShutterQualityConfig::Sharp);

        let midpoint = morph.sample(0.5).unwrap();
        let motion = midpoint.master_motion.unwrap();
        close(motion.transplant.amount, 0.5);
        close(motion.transplant.confidence_softness, 0.275);
        close(motion.shutter.angle_degrees, 180.0);
        assert_eq!(
            motion.algorithm_version,
            crate::motion::MOTION_ALGORITHM_VERSION
        );
        assert_eq!(motion.field_source, MotionFieldSourceConfig::Lattice);
        assert_eq!(motion.lattice_quality, MotionLatticeQualityConfig::High);
        assert_eq!(motion.transplant.donor, high.transplant.donor);
        assert_eq!(
            motion.transplant.carrier,
            MotionCarrierConfig::FirstSourceFrame
        );
        assert_eq!(motion.shutter.quality, CurvedShutterQualityConfig::High);
        assert_eq!(midpoint.layers[0].motion, Some(motion));

        let legacy: MorphSlot = serde_yaml::from_str("master: {}\n").unwrap();
        assert_eq!(legacy.master_motion, None);
        assert!(legacy.layers.is_empty());
    }

    #[test]
    fn morph_chooses_the_whole_collider_block_endpoint_exact() {
        use crate::patch::{
            FieldColliderConfig, FieldColliderModeConfig, MotionBoundaryModeConfig, MotionConfig,
            MotionDonorConfig,
        };
        use crate::performance::SavedLayerPosition;

        let block = |enabled, mode, boundary, a: u32, b: u32| FieldColliderConfig {
            enabled,
            mode,
            boundary,
            input_a: MotionDonorConfig::Selected {
                saved_position: SavedLayerPosition::new(a).unwrap(),
            },
            input_b: MotionDonorConfig::Selected {
                saved_position: SavedLayerPosition::new(b).unwrap(),
            },
            ..FieldColliderConfig::default()
        };
        let a_block = block(
            true,
            FieldColliderModeConfig::Curl,
            MotionBoundaryModeConfig::Wrap,
            0,
            1,
        );
        let b_block = block(
            false,
            FieldColliderModeConfig::Projection,
            MotionBoundaryModeConfig::Hold,
            2,
            3,
        );
        let a = MotionConfig {
            collider: a_block,
            ..MotionConfig::default()
        };
        let b = MotionConfig {
            collider: b_block,
            ..MotionConfig::default()
        };

        // Before the midpoint the whole block is A's, after it the whole block
        // is B's. No position synthesizes a third configuration — a mode from
        // one end with a boundary or an input from the other.
        for (weights, choose_b, expected) in [
            ([1.0_f32, 0.0_f32], false, a_block),
            ([0.75, 0.25], false, a_block),
            ([0.5, 0.5], false, a_block),
            ([0.25, 0.75], true, b_block),
            ([0.0, 1.0], true, b_block),
        ] {
            let blended = interpolate_motion_config(a, b, weights, choose_b);
            assert_eq!(blended.collider, expected, "weights {weights:?}");
        }

        // Endpoint exactness: each end recalls its own authored block bit for
        // bit, including both saved positions.
        assert_eq!(
            interpolate_motion_config(a, b, [1.0, 0.0], false)
                .collider
                .input_b,
            a_block.input_b
        );
        assert_eq!(
            interpolate_motion_config(a, b, [0.0, 1.0], true)
                .collider
                .input_a,
            b_block.input_a
        );
    }

    #[test]
    fn temporal_rig_morphs_continuous_values_and_recalls_discrete_laws() {
        let a = MorphTemporalSnapshot {
            rig: TemporalRigConfig {
                offset_x: -0.2,
                hue_rotate: 170.0,
                gain_r: 0.5,
                shape: crate::patch::FeedbackShapeConfig::Soft,
                reflect_x: true,
                servo: true,
                ..TemporalRigConfig::default()
            },
            ..MorphTemporalSnapshot::default()
        };
        let b = MorphTemporalSnapshot {
            rig: TemporalRigConfig {
                offset_x: 0.4,
                hue_rotate: -170.0,
                gain_r: 1.5,
                shape: crate::patch::FeedbackShapeConfig::Wrap,
                servo_defeated: true,
                ..TemporalRigConfig::default()
            },
            ..MorphTemporalSnapshot::default()
        };

        let quarter = MorphTemporalSnapshot::interpolate(&a, &b, [0.75, 0.25], false);
        assert!((quarter.rig.offset_x - -0.05).abs() < 1.0e-6);
        assert!((quarter.rig.gain_r - 0.75).abs() < 1.0e-6);
        // The in-loop hue takes the short wrapped arc through 180 (170 plus
        // a quarter of the 20-degree arc), not the long way through zero,
        // which would have landed at 85.
        assert!(
            (quarter.rig.hue_rotate - 175.0).abs() < 1.0e-3,
            "wrapped hue arc, got {}",
            quarter.rig.hue_rotate
        );
        // Discrete laws recall an endpoint at the midpoint, never a blend.
        assert_eq!(quarter.rig.shape, crate::patch::FeedbackShapeConfig::Soft);
        assert!(quarter.rig.reflect_x);
        assert!(quarter.rig.servo);
        assert!(!quarter.rig.servo_defeated);
        let past = MorphTemporalSnapshot::interpolate(&a, &b, [0.25, 0.75], true);
        assert_eq!(past.rig.shape, crate::patch::FeedbackShapeConfig::Wrap);
        assert!(!past.rig.reflect_x);
        assert!(past.rig.servo_defeated);
        // Endpoints recall the authored blocks exactly.
        assert_eq!(
            MorphTemporalSnapshot::interpolate(&a, &b, [1.0, 0.0], false).rig,
            a.rig.sanitized()
        );
        assert_eq!(
            MorphTemporalSnapshot::interpolate(&a, &b, [0.0, 1.0], true).rig,
            b.rig.sanitized()
        );
    }

    #[test]
    fn display_physics_morphs_values_continuously_and_recalls_discrete_laws() {
        use crate::display_physics::{DisplayModel, DisplayPhysicsParams, InterlaceMode};
        let a = MorphTemporalSnapshot {
            display: DisplayPhysicsParams {
                il_amount: 0.2,
                phosphor: 0.2,
                scanlines: 0.4,
                il_mode: InterlaceMode::Weave,
                il_order: false,
                model: DisplayModel::Flat,
                ..DisplayPhysicsParams::default()
            },
            ..MorphTemporalSnapshot::default()
        };
        let b = MorphTemporalSnapshot {
            display: DisplayPhysicsParams {
                il_amount: 0.8,
                phosphor: 0.6,
                scanlines: 0.8,
                il_mode: InterlaceMode::Bob,
                il_order: true,
                model: DisplayModel::Mono,
                ..DisplayPhysicsParams::default()
            },
            ..MorphTemporalSnapshot::default()
        };

        let quarter = MorphTemporalSnapshot::interpolate(&a, &b, [0.75, 0.25], false);
        assert!((quarter.display.il_amount - 0.35).abs() < 1.0e-6);
        assert!((quarter.display.phosphor - 0.3).abs() < 1.0e-6);
        assert!((quarter.display.scanlines - 0.5).abs() < 1.0e-6);
        // Discrete laws recall an endpoint at the midpoint, never a blend.
        assert_eq!(quarter.display.il_mode, InterlaceMode::Weave);
        assert!(!quarter.display.il_order);
        assert_eq!(quarter.display.model, DisplayModel::Flat);
        let past = MorphTemporalSnapshot::interpolate(&a, &b, [0.25, 0.75], true);
        assert_eq!(past.display.il_mode, InterlaceMode::Bob);
        assert!(past.display.il_order);
        assert_eq!(past.display.model, DisplayModel::Mono);
        // Endpoints recall the authored blocks exactly.
        assert_eq!(
            MorphTemporalSnapshot::interpolate(&a, &b, [1.0, 0.0], false).display,
            a.display.sanitized()
        );
        assert_eq!(
            MorphTemporalSnapshot::interpolate(&a, &b, [0.0, 1.0], true).display,
            b.display.sanitized()
        );
    }

    #[test]
    fn sync_latch_morphs_values_continuously_and_recalls_the_switch() {
        use crate::sync_latch::SyncLatchParams;
        let a = MorphTemporalSnapshot {
            sync: SyncLatchParams {
                amount: 0.2,
                rate: 0.4,
                spread: 0.2,
                bias: -0.4,
                latched: false,
            },
            ..MorphTemporalSnapshot::default()
        };
        let b = MorphTemporalSnapshot {
            sync: SyncLatchParams {
                amount: 0.8,
                rate: 0.8,
                spread: 0.6,
                bias: 0.4,
                latched: true,
            },
            ..MorphTemporalSnapshot::default()
        };

        let quarter = MorphTemporalSnapshot::interpolate(&a, &b, [0.75, 0.25], false);
        assert!((quarter.sync.amount - 0.35).abs() < 1.0e-6);
        assert!((quarter.sync.rate - 0.5).abs() < 1.0e-6);
        assert!((quarter.sync.spread - 0.3).abs() < 1.0e-6);
        assert!((quarter.sync.bias - -0.2).abs() < 1.0e-6);
        // The switch recalls an endpoint, never a blend: a failure switch is
        // thrown, and no morph position may invent a half-latched program.
        assert!(!quarter.sync.latched);
        let past = MorphTemporalSnapshot::interpolate(&a, &b, [0.25, 0.75], true);
        assert!(past.sync.latched);
        // Endpoints recall the authored blocks exactly.
        assert_eq!(
            MorphTemporalSnapshot::interpolate(&a, &b, [1.0, 0.0], false).sync,
            a.sync.sanitized()
        );
        assert_eq!(
            MorphTemporalSnapshot::interpolate(&a, &b, [0.0, 1.0], true).sync,
            b.sync.sanitized()
        );
    }

    #[test]
    fn codec_mosh_morphs_values_continuously_and_recalls_the_recycle_law() {
        use crate::codec_mosh::CodecMoshParams;
        let a = MorphTemporalSnapshot {
            mosh: CodecMoshParams {
                amount: 0.2,
                hold: 0.4,
                bitrate_starve: 0.2,
                recycle: false,
                ..CodecMoshParams::default()
            },
            ..MorphTemporalSnapshot::default()
        };
        let b = MorphTemporalSnapshot {
            mosh: CodecMoshParams {
                amount: 0.8,
                hold: 0.8,
                bitrate_starve: 0.6,
                recycle: true,
                ..CodecMoshParams::default()
            },
            ..MorphTemporalSnapshot::default()
        };

        let quarter = MorphTemporalSnapshot::interpolate(&a, &b, [0.75, 0.25], false);
        assert!((quarter.mosh.amount - 0.35).abs() < 1.0e-6);
        assert!((quarter.mosh.hold - 0.5).abs() < 1.0e-6);
        assert!((quarter.mosh.bitrate_starve - 0.3).abs() < 1.0e-6);
        // The discrete recycle law recalls an endpoint, never a blend.
        assert!(!quarter.mosh.recycle);
        let past = MorphTemporalSnapshot::interpolate(&a, &b, [0.25, 0.75], true);
        assert!(past.mosh.recycle);
        // Endpoints recall the authored blocks exactly.
        assert_eq!(
            MorphTemporalSnapshot::interpolate(&a, &b, [1.0, 0.0], false).mosh,
            a.mosh.sanitized()
        );
        assert_eq!(
            MorphTemporalSnapshot::interpolate(&a, &b, [0.0, 1.0], true).mosh,
            b.mosh.sanitized()
        );
    }

    #[test]
    fn small_effects_morph_blends_values_and_recalls_discrete_laws() {
        let a = MorphMasterSnapshot {
            contour: 0.2,
            colourpass_hue: 170.0,
            negative_mode: 0.0,
            bitcrush_levels: 2.0,
            barrel: -0.4,
            ..MorphMasterSnapshot::default()
        };
        let b = MorphMasterSnapshot {
            contour: 1.0,
            colourpass_hue: -170.0,
            negative_mode: 2.0,
            bitcrush_levels: 10.0,
            barrel: 0.4,
            ..MorphMasterSnapshot::default()
        };

        let quarter = MorphMasterSnapshot::interpolate(&a, &b, [0.75, 0.25], false);
        close(quarter.contour, 0.4);
        close(quarter.bitcrush_levels, 4.0);
        close(quarter.barrel, -0.2);
        // The colourpass hue takes the short wrapped arc through 180, not
        // the long way through zero (which would have landed at 85).
        close(quarter.colourpass_hue, 175.0);
        // The discrete negative mode recalls an endpoint at the midpoint.
        assert_eq!(quarter.negative_mode, 0.0);
        let past = MorphMasterSnapshot::interpolate(&a, &b, [0.25, 0.75], true);
        assert_eq!(past.negative_mode, 2.0);

        // Capture/apply carries every B13 field through the uniforms.
        let mut uniforms = crate::effects::EffectUniforms::default();
        b.apply_to(&mut uniforms);
        assert_eq!(uniforms.contour, 1.0);
        assert_eq!(uniforms.negative_mode, 2.0);
        assert_eq!(uniforms.bitcrush_levels, 10.0);
        assert_eq!(uniforms.barrel, 0.4);
        let recaptured = MorphMasterSnapshot::capture(&uniforms);
        assert_eq!(recaptured, b.sanitized());
    }

    #[test]
    fn time_displace_map_and_interp_recall_an_endpoint_at_the_midpoint() {
        let a = MorphTemporalSnapshot {
            slitscan: 0.2,
            slit_map: TimeDisplaceMapConfig::Radial,
            slit_interp: false,
            ..MorphTemporalSnapshot::default()
        };
        let b = MorphTemporalSnapshot {
            slitscan: 0.8,
            slit_map: TimeDisplaceMapConfig::Sweep,
            slit_interp: true,
            ..MorphTemporalSnapshot::default()
        };

        // The continuous depth blends; the two B12 discrete laws recall an
        // endpoint at the midpoint, never a synthesized third configuration.
        let quarter = MorphTemporalSnapshot::interpolate(&a, &b, [0.75, 0.25], false);
        assert!((quarter.slitscan - 0.35).abs() < 1.0e-6);
        assert_eq!(quarter.slit_map, TimeDisplaceMapConfig::Radial);
        assert!(!quarter.slit_interp);
        let past = MorphTemporalSnapshot::interpolate(&a, &b, [0.25, 0.75], true);
        assert_eq!(past.slit_map, TimeDisplaceMapConfig::Sweep);
        assert!(past.slit_interp);

        // The capture/to_params pair carries both laws exactly.
        let params = crate::effects::params::TemporalParams {
            slitscan: 0.5,
            slit_map: crate::effects::params::TimeDisplaceMap::TbcRamp,
            slit_interp: true,
            ..crate::effects::params::TemporalParams::default()
        };
        let restored = MorphTemporalSnapshot::capture(&params).to_params();
        assert_eq!(
            restored.slit_map,
            crate::effects::params::TimeDisplaceMap::TbcRamp
        );
        assert!(restored.slit_interp);
    }

    #[test]
    fn morph_interpolates_procedural_scalars_and_switches_the_kind_at_midpoint() {
        use crate::patch::{MotionConfig, MotionFieldSourceConfig, ProceduralFieldConfig};

        let a = MotionConfig {
            field_source: MotionFieldSourceConfig::ProceduralCurl,
            procedural: ProceduralFieldConfig {
                scale: 0.2,
                rate: -1.0,
            },
            ..MotionConfig::default()
        };
        let b = MotionConfig {
            field_source: MotionFieldSourceConfig::ProceduralRadial,
            procedural: ProceduralFieldConfig {
                scale: 0.8,
                rate: 1.0,
            },
            ..MotionConfig::default()
        };
        let quarter = interpolate_motion_config(a, b, [0.75, 0.25], false);
        assert_eq!(
            quarter.field_source,
            MotionFieldSourceConfig::ProceduralCurl
        );
        assert!((quarter.procedural.scale - 0.35).abs() < 1.0e-6);
        assert!((quarter.procedural.rate - -0.5).abs() < 1.0e-6);
        let past_midpoint = interpolate_motion_config(a, b, [0.25, 0.75], true);
        assert_eq!(
            past_midpoint.field_source,
            MotionFieldSourceConfig::ProceduralRadial
        );
        assert!((past_midpoint.procedural.scale - 0.65).abs() < 1.0e-6);
        // Endpoints recall their authored values exactly.
        assert_eq!(
            interpolate_motion_config(a, b, [1.0, 0.0], false).procedural,
            a.procedural
        );
        assert_eq!(
            interpolate_motion_config(a, b, [0.0, 1.0], true).procedural,
            b.procedural
        );
    }

    #[test]
    fn pattern_capture_blends_values_wraps_hue_and_recalls_discrete_laws_at_midpoint() {
        use crate::patch::{
            PatternColorModeConfig, PatternShapeConfig, PatternSynthConfig, PatternWaveConfig,
        };

        let a = PatternSynthConfig {
            shape: PatternShapeConfig::Scan,
            wave: PatternWaveConfig::Sine,
            color_mode: PatternColorModeConfig::Mono,
            wavefold: 0.2,
            hue: 0.95,
            ..PatternSynthConfig::default()
        };
        let b = PatternSynthConfig {
            shape: PatternShapeConfig::Polygon,
            wave: PatternWaveConfig::SampleHold,
            color_mode: PatternColorModeConfig::Bands,
            wavefold: 0.8,
            hue: 0.05,
            ..PatternSynthConfig::default()
        };
        // Before the midpoint: A's discrete laws, blended values, and the
        // hue takes the short arc across the wrap rather than sweeping the
        // whole circle.
        let quarter = interpolate_pattern_config(a, b, [0.75, 0.25], false);
        assert_eq!(quarter.shape, PatternShapeConfig::Scan);
        assert_eq!(quarter.wave, PatternWaveConfig::Sine);
        assert_eq!(quarter.color_mode, PatternColorModeConfig::Mono);
        assert!((quarter.wavefold - 0.35).abs() < 1.0e-6);
        let wrapped = quarter.hue;
        assert!(
            !(0.2..=0.8).contains(&wrapped),
            "hue must cross the wrap, got {wrapped}"
        );
        // Past the midpoint: B's discrete laws.
        let past = interpolate_pattern_config(a, b, [0.25, 0.75], true);
        assert_eq!(past.shape, PatternShapeConfig::Polygon);
        assert_eq!(past.wave, PatternWaveConfig::SampleHold);
        assert_eq!(past.color_mode, PatternColorModeConfig::Bands);
        // Endpoints are exact.
        assert_eq!(interpolate_pattern_config(a, b, [1.0, 0.0], false), a);
        assert_eq!(interpolate_pattern_config(a, b, [0.0, 1.0], true), b);
    }

    #[test]
    fn pattern_ownership_requires_both_slots_and_interpolation_requires_both_captures() {
        use crate::patch::PatternSynthConfig;

        // A snapshot pair where only one side captured a pattern source
        // interpolates to no pattern claim at all.
        let with_pattern = LayerMorphSnapshot {
            pattern: Some(PatternSynthConfig::default()),
            ..LayerMorphSnapshot::default()
        };
        let without = LayerMorphSnapshot::default();
        let mixed = LayerMorphSnapshot::interpolate(&with_pattern, &without, [0.5, 0.5], false);
        assert_eq!(mixed.pattern, None);
        let paired =
            LayerMorphSnapshot::interpolate(&with_pattern, &with_pattern, [0.5, 0.5], false);
        assert!(paired.pattern.is_some());

        // Ownership answers from the same both-slots gate.
        let mut morph = Morph::default();
        let make_slot = |pattern: Option<PatternSynthConfig>| MorphSlot {
            layers: vec![LayerMorphSnapshot {
                position: 0,
                pattern,
                ..LayerMorphSnapshot::default()
            }],
            ..MorphSlot::default()
        };
        morph.a = Some(make_slot(Some(PatternSynthConfig::default())));
        morph.b = Some(make_slot(None));
        assert!(!morph.controls_layer_field(0, LayerMorphControl::Pattern));
        morph.b = Some(make_slot(Some(PatternSynthConfig::default())));
        assert!(morph.controls_layer_field(0, LayerMorphControl::Pattern));
        // An appended layer outside the capture stays directly editable.
        assert!(!morph.controls_layer_field(1, LayerMorphControl::Pattern));
    }

    #[test]
    fn morph_remaps_both_collider_inputs_independently_and_a_tombstone_never_rebinds() {
        use crate::patch::{FieldColliderConfig, MotionConfig, MotionDonorConfig};
        use crate::performance::SavedLayerPosition;

        let selected = |position: u32| MotionDonorConfig::Selected {
            saved_position: SavedLayerPosition::new(position).unwrap(),
        };
        let armed = |a: u32, b: u32| MotionConfig {
            collider: FieldColliderConfig {
                enabled: true,
                input_a: selected(a),
                input_b: selected(b),
                ..FieldColliderConfig::default()
            },
            ..MotionConfig::default()
        };

        let mut slot = MorphSlot {
            master_motion: Some(armed(0, 2)),
            layers: vec![LayerMorphSnapshot {
                position: 0,
                motion: Some(armed(1, 2)),
                ..LayerMorphSnapshot::default()
            }],
            ..MorphSlot::default()
        };

        // Moving layer 0 to position 2 shifts 0 -> 2, 1 -> 0, 2 -> 1. Each slot
        // follows its OWN saved position; B does not inherit A's.
        slot.remap_layers_after_move(0, 2);
        let master = slot.master_motion.unwrap().collider;
        assert_eq!(master.input_a, selected(2));
        assert_eq!(master.input_b, selected(1));
        let layer = slot.layers[0].motion.unwrap().collider;
        assert_eq!(layer.input_a, selected(0));
        assert_eq!(layer.input_b, selected(1));

        // Removing the layer at position 1 tombstones exactly the slot that
        // named it and decrements only the slots above it.
        let mut removal = MorphSlot {
            master_motion: Some(armed(1, 2)),
            layers: vec![LayerMorphSnapshot {
                position: 0,
                motion: Some(armed(0, 1)),
                ..LayerMorphSnapshot::default()
            }],
            ..MorphSlot::default()
        };
        removal.remap_layers_after_remove(1);
        let master = removal.master_motion.unwrap().collider;
        assert_eq!(
            master.input_a,
            MotionDonorConfig::Missing {
                saved_position: SavedLayerPosition::new(1).unwrap()
            },
            "the slot that named the removed layer must tombstone"
        );
        assert_eq!(master.input_b, selected(1), "position 2 shifts down to 1");
        let layer = removal.layers[0].motion.unwrap().collider;
        assert_eq!(
            layer.input_a,
            selected(0),
            "position 0 is below the removal"
        );
        assert_eq!(
            layer.input_b,
            MotionDonorConfig::Missing {
                saved_position: SavedLayerPosition::new(1).unwrap()
            }
        );

        // A tombstone never rebinds under a later permutation.
        removal.remap_layers_after_move(0, 1);
        assert!(matches!(
            removal.master_motion.unwrap().collider.input_a,
            MotionDonorConfig::Missing { saved_position } if saved_position.get() == 1
        ));
    }

    #[test]
    fn motion_donors_follow_stack_permutations_and_missing_never_rebinds() {
        use crate::patch::{MotionConfig, MotionDonorConfig};
        use crate::performance::SavedLayerPosition;

        let selected = |position| {
            let mut config = MotionConfig::default();
            config.transplant.donor = MotionDonorConfig::Selected {
                saved_position: SavedLayerPosition::new(position).unwrap(),
            };
            config
        };
        let mut missing = MotionConfig::default();
        missing.transplant.donor = MotionDonorConfig::Missing {
            saved_position: SavedLayerPosition::new(0).unwrap(),
        };
        let mut slot = MorphSlot {
            master_motion: Some(selected(0)),
            layers: vec![
                LayerMorphSnapshot {
                    position: 0,
                    motion: Some(selected(2)),
                    ..LayerMorphSnapshot::default()
                },
                LayerMorphSnapshot {
                    position: 1,
                    motion: Some(missing),
                    ..LayerMorphSnapshot::default()
                },
                LayerMorphSnapshot {
                    position: 2,
                    motion: None,
                    ..LayerMorphSnapshot::default()
                },
            ],
            ..MorphSlot::default()
        };

        slot.remap_layers_after_move(0, 2);
        assert!(matches!(
            slot.master_motion.unwrap().transplant.donor,
            MotionDonorConfig::Selected { saved_position } if saved_position.get() == 2
        ));
        let moved_selected = slot
            .layers
            .iter()
            .find(|layer| {
                layer.motion.is_some_and(|motion| {
                    matches!(motion.transplant.donor, MotionDonorConfig::Selected { .. })
                })
            })
            .unwrap();
        assert!(matches!(
            moved_selected.motion.unwrap().transplant.donor,
            MotionDonorConfig::Selected { saved_position } if saved_position.get() == 1
        ));
        let authored_missing = slot
            .layers
            .iter()
            .find_map(|layer| {
                layer.motion.filter(|motion| {
                    matches!(motion.transplant.donor, MotionDonorConfig::Missing { .. })
                })
            })
            .unwrap();
        assert!(matches!(
            authored_missing.transplant.donor,
            MotionDonorConfig::Missing { saved_position } if saved_position.get() == 0
        ));

        slot.remap_layers_after_remove(1);
        assert!(matches!(
            slot.master_motion.unwrap().transplant.donor,
            MotionDonorConfig::Selected { saved_position } if saved_position.get() == 1
        ));
        assert!(slot.layers.iter().all(|layer| {
            !matches!(
                layer.motion.map(|motion| motion.transplant.donor),
                Some(MotionDonorConfig::Selected { saved_position }) if saved_position.get() == 1
            )
        }));
    }

    #[test]
    fn temporal_originals_morph_numeric_values_and_recall_discrete_laws() {
        use crate::image_routing::LayerImageStage;
        use crate::patch::{
            CollisionScoreConfig, CollisionScoreTriggerConfig, RefreshGardenGateConfig,
            RefreshGardenMatteRouteConfig, RefreshGardenMotionRouteConfig,
            TemporalEventResetModeConfig, TemporalInterpolationConfig, TemporalTopologyConfig,
        };
        use crate::performance::SavedLayerPosition;

        let mut a = MorphSlot::default();
        a.temporal.originals.loom.amount = 0.0;
        a.temporal.originals.loom.depth = 0.0;
        a.temporal.originals.loom.phase = -100.0;
        a.temporal.originals.loom.scale = 1.0;
        a.temporal.originals.loom.angle = 170.0;
        a.temporal.originals.loom.folds = 1;
        a.temporal.originals.loom.quantization = 0;
        a.temporal.originals.loom.topology = TemporalTopologyConfig::Linear;
        a.temporal.originals.loom.interpolation = TemporalInterpolationConfig::Floor;
        a.temporal.originals.atlas.amount = 0.0;
        a.temporal.originals.atlas.seed = 11;
        a.temporal.originals.atlas.territories = 2;
        a.temporal.originals.atlas.collision = 0.2;
        a.temporal.originals.garden.amount = 0.0;
        a.temporal.originals.garden.gate = RefreshGardenGateConfig::Luma;
        a.temporal.originals.garden.threshold = 0.2;
        a.temporal.originals.garden.softness = 0.1;
        a.temporal.originals.garden.decay = 1.0;
        a.temporal.originals.garden.max_hold_ticks = 10;
        a.temporal.originals.garden.matte_route = RefreshGardenMatteRouteConfig::SelectedLayer {
            saved_position: SavedLayerPosition::new(1).unwrap(),
            stage: LayerImageStage::PreLocalEffects,
        };
        a.temporal.originals.score = CollisionScoreConfig {
            enabled: false,
            seed: 31,
            state_count: 4,
            trigger: CollisionScoreTriggerConfig::Boundary,
            loop_driver: CollisionScoreLoopDriverConfig::SelectedLayer {
                saved_position: SavedLayerPosition::new(1).unwrap(),
            },
        };
        a.temporal.originals.reset.loop_boundary = TemporalEventResetModeConfig::Memory;

        let mut b = MorphSlot::default();
        b.temporal.originals.loom.amount = 1.0;
        b.temporal.originals.loom.depth = 1.0;
        b.temporal.originals.loom.phase = 100.0;
        b.temporal.originals.loom.scale = 3.0;
        b.temporal.originals.loom.angle = -170.0;
        b.temporal.originals.loom.folds = 16;
        b.temporal.originals.loom.quantization = 24;
        b.temporal.originals.loom.topology = TemporalTopologyConfig::Kaleidoscopic;
        b.temporal.originals.loom.interpolation = TemporalInterpolationConfig::Linear;
        b.temporal.originals.atlas.amount = 1.0;
        b.temporal.originals.atlas.seed = 22;
        b.temporal.originals.atlas.territories = 10;
        b.temporal.originals.atlas.collision = 0.8;
        b.temporal.originals.garden.amount = 1.0;
        b.temporal.originals.garden.gate = RefreshGardenGateConfig::Matte;
        b.temporal.originals.garden.threshold = 0.8;
        b.temporal.originals.garden.softness = 0.3;
        b.temporal.originals.garden.decay = 0.0;
        b.temporal.originals.garden.max_hold_ticks = 20;
        b.temporal.originals.garden.matte_route =
            RefreshGardenMatteRouteConfig::MissingSelectedLayer {
                saved_position: SavedLayerPosition::new(3).unwrap(),
                stage: LayerImageStage::PostLocalEffects,
            };
        b.temporal.originals.garden.motion_route = RefreshGardenMotionRouteConfig::SelectedLayer {
            saved_position: SavedLayerPosition::new(2).unwrap(),
        };
        b.temporal.originals.score = CollisionScoreConfig {
            enabled: true,
            seed: 47,
            state_count: 9,
            trigger: CollisionScoreTriggerConfig::Manual,
            loop_driver: CollisionScoreLoopDriverConfig::SelectedLayer {
                saved_position: SavedLayerPosition::new(3).unwrap(),
            },
        };
        b.temporal.originals.reset.loop_boundary = TemporalEventResetModeConfig::All;

        let morph = Morph {
            a: Some(a.clone()),
            b: Some(b.clone()),
            ..Default::default()
        };
        assert_eq!(
            morph.sample(0.0).unwrap().temporal.originals,
            a.temporal.originals
        );
        assert_eq!(
            morph.sample(1.0).unwrap().temporal.originals,
            b.temporal.originals
        );

        let midpoint = morph.sample(0.5).unwrap().temporal.originals;
        close(midpoint.loom.amount, 0.5);
        close(midpoint.loom.depth, 0.5);
        close(midpoint.loom.phase, 0.0);
        close(midpoint.loom.scale, 2.0);
        close(midpoint.loom.angle.abs(), 180.0);
        assert_eq!(midpoint.loom.folds, 9);
        assert_eq!(midpoint.loom.quantization, 12);
        assert_eq!(
            midpoint.loom.topology,
            TemporalTopologyConfig::Kaleidoscopic
        );
        assert_eq!(
            midpoint.loom.interpolation,
            TemporalInterpolationConfig::Linear
        );
        close(midpoint.atlas.amount, 0.5);
        assert_eq!(midpoint.atlas.seed, 22);
        assert_eq!(midpoint.atlas.territories, 6);
        close(midpoint.atlas.collision, 0.5);
        close(midpoint.garden.amount, 0.5);
        close(midpoint.garden.threshold, 0.5);
        close(midpoint.garden.softness, 0.2);
        close(midpoint.garden.decay, 0.5);
        assert_eq!(midpoint.garden.max_hold_ticks, 15);
        assert_eq!(midpoint.garden.gate, RefreshGardenGateConfig::Matte);
        assert_eq!(
            midpoint.garden.matte_route,
            b.temporal.originals.garden.matte_route
        );
        assert_eq!(
            midpoint.garden.motion_route,
            b.temporal.originals.garden.motion_route
        );
        assert_eq!(midpoint.score, b.temporal.originals.score);
        assert_eq!(
            midpoint.reset.loop_boundary,
            TemporalEventResetModeConfig::All
        );

        let before_endpoint = morph.sample(0.499).unwrap().temporal.originals;
        assert_eq!(
            before_endpoint.loom.topology,
            TemporalTopologyConfig::Linear
        );
        assert_eq!(before_endpoint.atlas.seed, 11);
        assert_eq!(before_endpoint.score, a.temporal.originals.score);
        assert_eq!(
            before_endpoint.garden.matte_route,
            a.temporal.originals.garden.matte_route
        );
        assert_eq!(
            before_endpoint.garden.motion_route,
            a.temporal.originals.garden.motion_route
        );
        assert!(matches!(
            before_endpoint.score.loop_driver,
            CollisionScoreLoopDriverConfig::SelectedLayer { saved_position }
                if saved_position.get() == 1
        ));
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
        assert_eq!(legacy.transform, None);
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
    fn spatial_morph_is_exact_at_endpoints_and_legacy_slots_do_not_claim_it() {
        let layer_a = SpatialTransform {
            position: [-0.25, 0.5],
            scale: [1.0, -1.0],
            rotation_deg: 170.0,
            fit: crate::spatial::FitMode::Fit,
            ..SpatialTransform::default()
        };
        let layer_b = SpatialTransform {
            position: [0.75, -0.5],
            scale: [4.0, 1.0],
            rotation_deg: -170.0,
            fit: crate::spatial::FitMode::Fill,
            ..SpatialTransform::default()
        };
        let master_a = SpatialTransform {
            skew_axis_deg: 170.0,
            ..SpatialTransform::default()
        };
        let master_b = SpatialTransform {
            skew_axis_deg: -170.0,
            ..SpatialTransform::default()
        };
        let morph = Morph {
            a: Some(MorphSlot {
                master_transform: Some(master_a),
                layers: vec![LayerMorphSnapshot {
                    position: 0,
                    transform: Some(layer_a),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            b: Some(MorphSlot {
                master_transform: Some(master_b),
                layers: vec![LayerMorphSnapshot {
                    position: 0,
                    transform: Some(layer_b),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };

        assert!(morph.controls_master_transform());
        assert!(morph.controls_layer_field(0, LayerMorphControl::Transform));
        assert_eq!(morph.sample(0.0).unwrap().master_transform, Some(master_a));
        assert_eq!(morph.sample(1.0).unwrap().master_transform, Some(master_b));
        assert_eq!(
            morph.sample(0.0).unwrap().layers[0].transform,
            Some(layer_a)
        );
        assert_eq!(
            morph.sample(1.0).unwrap().layers[0].transform,
            Some(layer_b)
        );
        let middle = morph.sample(0.5).unwrap();
        let layer = middle.layers[0].transform.unwrap();
        close(layer.scale[0], 2.0);
        close(layer.scale[1], 0.0);
        close(layer.rotation_deg.abs(), 180.0);
        assert_eq!(layer.fit, crate::spatial::FitMode::Fill);
        close(middle.master_transform.unwrap().skew_axis_deg.abs(), 180.0);

        let legacy = Morph {
            a: Some(MorphSlot {
                layers: vec![LayerMorphSnapshot {
                    position: 0,
                    ..Default::default()
                }],
                ..Default::default()
            }),
            b: Some(MorphSlot {
                layers: vec![LayerMorphSnapshot {
                    position: 0,
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(!legacy.controls_master_transform());
        assert!(!legacy.controls_layer_field(0, LayerMorphControl::Transform));
        let sample = legacy.sample(0.5).unwrap();
        assert_eq!(sample.master_transform, None);
        assert_eq!(sample.layers[0].transform, None);
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
    fn layer_stack_remaps_preserve_every_saved_rack_donor_identity() {
        use crate::image_routing::LayerImageStage;
        use crate::performance::SavedLayerPosition;
        use crate::visual_rack::{EdgeTiming, GroupId, NodeId, SavedImageTap, VisualNode};

        let selected_rack = |position: u32| {
            VisualRack::try_from_parts(
                vec![VisualNode::authored(
                    NodeId::new(3).unwrap(),
                    VisualNodeKind::Mask(MaskParams::Image(ImageMatte {
                        tap: SavedImageTap {
                            source: SavedImageSource::SelectedLayer {
                                layer_position: SavedLayerPosition::new(position).unwrap(),
                                stage: LayerImageStage::PostLocalEffects,
                            },
                            timing: EdgeTiming::CurrentFrame,
                        },
                        ..ImageMatte::default()
                    })),
                )],
                Some(4),
            )
            .unwrap()
        };
        let route = |rack: &VisualRack| {
            let node = rack.get(NodeId::new(3).unwrap()).unwrap();
            let VisualNodeKind::Mask(MaskParams::Image(matte)) = node.kind else {
                panic!("fixture node must remain an image mask");
            };
            matte.tap.source
        };

        let mut master_rack = selected_rack(0);
        let group_node = master_rack
            .push(VisualNodeKind::Mask(MaskParams::Image(ImageMatte {
                tap: SavedImageTap {
                    source: SavedImageSource::GroupOutput {
                        group_id: GroupId::new(91).unwrap(),
                    },
                    timing: EdgeTiming::PreviousFrame,
                },
                ..ImageMatte::default()
            })))
            .unwrap();
        let mut slot = MorphSlot {
            master_rack: Some(master_rack),
            layer_racks: Some(vec![selected_rack(2), selected_rack(1), selected_rack(0)]),
            ..MorphSlot::default()
        };
        slot.temporal.originals.score.loop_driver = CollisionScoreLoopDriverConfig::SelectedLayer {
            saved_position: SavedLayerPosition::new(0).unwrap(),
        };
        slot.temporal.originals.garden.matte_route = RefreshGardenMatteRouteConfig::SelectedLayer {
            saved_position: SavedLayerPosition::new(0).unwrap(),
            stage: LayerImageStage::PostLocalEffects,
        };
        slot.temporal.originals.garden.motion_route =
            RefreshGardenMotionRouteConfig::SelectedLayer {
                saved_position: SavedLayerPosition::new(1).unwrap(),
            };

        slot.remap_layers_after_move(0, 2);
        assert!(matches!(
            slot.temporal.originals.score.loop_driver,
            CollisionScoreLoopDriverConfig::SelectedLayer { saved_position }
                if saved_position.get() == 2
        ));
        assert!(matches!(
            slot.temporal.originals.garden.matte_route,
            RefreshGardenMatteRouteConfig::SelectedLayer { saved_position, .. }
                if saved_position.get() == 2
        ));
        assert!(matches!(
            slot.temporal.originals.garden.motion_route,
            RefreshGardenMotionRouteConfig::SelectedLayer { saved_position }
                if saved_position.get() == 0
        ));
        let master = slot.master_rack.as_ref().unwrap();
        assert!(matches!(
            route(master),
            SavedImageSource::SelectedLayer { layer_position, .. }
                if layer_position == SavedLayerPosition::new(2).unwrap()
        ));
        let VisualNodeKind::Mask(MaskParams::Image(group_matte)) =
            master.get(group_node).unwrap().kind
        else {
            panic!("group fixture node must remain an image mask");
        };
        assert_eq!(
            group_matte.tap.source,
            SavedImageSource::GroupOutput {
                group_id: GroupId::new(91).unwrap()
            },
            "layer permutations must never rewrite stable group identities"
        );
        let racks = slot.layer_racks.as_ref().unwrap();
        assert!(
            matches!(route(&racks[0]), SavedImageSource::SelectedLayer { layer_position, .. } if layer_position.get() == 0)
        );
        assert!(
            matches!(route(&racks[1]), SavedImageSource::SelectedLayer { layer_position, .. } if layer_position.get() == 2)
        );
        assert!(
            matches!(route(&racks[2]), SavedImageSource::SelectedLayer { layer_position, .. } if layer_position.get() == 1)
        );

        slot.remap_layers_after_remove(1);
        assert!(matches!(
            slot.temporal.originals.score.loop_driver,
            CollisionScoreLoopDriverConfig::SelectedLayer { saved_position }
                if saved_position.get() == 1
        ));
        assert!(matches!(
            slot.temporal.originals.garden.matte_route,
            RefreshGardenMatteRouteConfig::SelectedLayer { saved_position, .. }
                if saved_position.get() == 1
        ));
        assert!(matches!(
            slot.temporal.originals.garden.motion_route,
            RefreshGardenMotionRouteConfig::SelectedLayer { saved_position }
                if saved_position.get() == 0
        ));
        assert!(matches!(
            route(slot.master_rack.as_ref().unwrap()),
            SavedImageSource::SelectedLayer { layer_position, .. } if layer_position.get() == 1
        ));
        let racks = slot.layer_racks.as_ref().unwrap();
        assert_eq!(racks.len(), 2);
        assert!(
            matches!(route(&racks[0]), SavedImageSource::SelectedLayer { layer_position, .. } if layer_position.get() == 0)
        );
        assert!(matches!(
            route(&racks[1]),
            SavedImageSource::MissingSelectedLayer { saved_position, stage }
                if saved_position.get() == 1
                    && stage == LayerImageStage::PostLocalEffects
        ));

        slot.temporal.originals.score.loop_driver = CollisionScoreLoopDriverConfig::SelectedLayer {
            saved_position: SavedLayerPosition::new(1).unwrap(),
        };
        slot.temporal.originals.garden.matte_route = RefreshGardenMatteRouteConfig::SelectedLayer {
            saved_position: SavedLayerPosition::new(1).unwrap(),
            stage: LayerImageStage::PreLocalEffects,
        };
        slot.temporal.originals.garden.motion_route =
            RefreshGardenMotionRouteConfig::SelectedLayer {
                saved_position: SavedLayerPosition::new(1).unwrap(),
            };
        slot.remap_layers_after_remove(1);
        assert!(matches!(
            slot.temporal.originals.score.loop_driver,
            CollisionScoreLoopDriverConfig::MissingSelectedLayer { saved_position }
                if saved_position.get() == 1
        ));
        assert!(matches!(
            slot.temporal.originals.garden.matte_route,
            RefreshGardenMatteRouteConfig::MissingSelectedLayer { saved_position, stage }
                if saved_position.get() == 1 && stage == LayerImageStage::PreLocalEffects
        ));
        assert!(matches!(
            slot.temporal.originals.garden.motion_route,
            RefreshGardenMotionRouteConfig::MissingSelectedLayer { saved_position }
                if saved_position.get() == 1
        ));
    }

    #[test]
    fn sampled_score_driver_resolves_only_selected_configuration() {
        use crate::image_routing::StableLayerId;
        use crate::performance::SavedLayerPosition;

        let saved_position = SavedLayerPosition::new(3).unwrap();
        let layer_id = StableLayerId::new(91).unwrap();
        assert_eq!(
            resolve_score_loop_driver(
                CollisionScoreLoopDriverConfig::SelectedLayer { saved_position },
                |position| (position == saved_position).then_some(layer_id),
            ),
            CollisionScoreLoopDriver::SelectedLayer {
                layer_id,
                saved_position,
            }
        );
        assert_eq!(
            resolve_score_loop_driver(
                CollisionScoreLoopDriverConfig::SelectedLayer { saved_position },
                |_| None,
            ),
            CollisionScoreLoopDriver::MissingSelectedLayer { saved_position }
        );
        assert_eq!(
            resolve_score_loop_driver(
                CollisionScoreLoopDriverConfig::MissingSelectedLayer { saved_position },
                |_| Some(layer_id),
            ),
            CollisionScoreLoopDriver::MissingSelectedLayer { saved_position },
            "an authored tombstone must never bind to a new occupant"
        );
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
            originals: crate::temporal::TemporalOriginalsParams {
                long_exposure: crate::temporal::LongExposureParams {
                    amount: 0.2,
                    shutter_frames: 4,
                },
                ..Default::default()
            },
            ..Default::default()
        });
        let temporal_b = MorphTemporalSnapshot::capture(&TemporalParams {
            key_mode: 4.0,
            key_threshold: 0.48,
            key_softness: 0.1,
            key_history: 9.0,
            originals: crate::temporal::TemporalOriginalsParams {
                long_exposure: crate::temporal::LongExposureParams {
                    amount: 0.8,
                    shutter_frames: 20,
                },
                ..Default::default()
            },
            ..Default::default()
        });
        let temporal =
            MorphTemporalSnapshot::interpolate(&temporal_a, &temporal_b, [0.5, 0.5], true);
        close(temporal.key_mode, 4.0);
        close(temporal.key_threshold, 0.28);
        close(temporal.key_softness, 0.06);
        close(temporal.key_history, 5.0);
        close(temporal.originals.long_exposure.amount, 0.5);
        assert_eq!(temporal.originals.long_exposure.shutter_frames, 12);

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
        assert_eq!(
            legacy_temporal.originals.long_exposure,
            LongExposureConfig::default()
        );
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

    #[test]
    fn image_route_topology_is_never_morph_owned_or_reverted() {
        use crate::image_routing::{LayerImageStage, StableLayerId};
        use crate::performance::SavedLayerPosition;
        use crate::visual_rack::{
            EdgeTiming, MatteChannel, NodeId, ResolvedImageSource, RuntimeVisualNodeKind,
            SavedImageSource, SavedImageTap, VisualNode,
        };

        let saved_rack = |position: u32, amount: f32| {
            let matte = ImageMatte {
                tap: SavedImageTap {
                    source: SavedImageSource::SelectedLayer {
                        layer_position: SavedLayerPosition::new(position).unwrap(),
                        stage: LayerImageStage::PostLocalEffects,
                    },
                    timing: EdgeTiming::CurrentFrame,
                },
                channel: MatteChannel::Luma,
                invert: true,
                amount,
                threshold: amount,
                softness: amount * 0.25,
            };
            VisualRack::try_from_parts(
                vec![VisualNode::authored(
                    NodeId::new(3).unwrap(),
                    VisualNodeKind::Mask(MaskParams::Image(matte)),
                )],
                Some(4),
            )
            .unwrap()
        };
        let a = saved_rack(0, 0.2);
        let b = saved_rack(0, 0.8);
        let mismatched_b = saved_rack(1, 0.8);
        assert!(interpolate_rack(&a, &mismatched_b, [0.5, 0.5], true).is_none());

        let sampled_saved = interpolate_rack(&a, &b, [0.5, 0.5], true).unwrap();
        let donor_x = StableLayerId::new(10).unwrap();
        let donor_y = StableLayerId::new(20).unwrap();
        let sampled = sampled_saved.resolve_routes(
            |position| (position.get() == 0).then_some(donor_x),
            |_| false,
        );
        let mut live = sampled.clone();
        let node = live.get_mut(NodeId::new(3).unwrap()).unwrap();
        let RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(live_matte)) = &mut node.kind
        else {
            panic!("expected image mask");
        };
        live_matte.tap.source = ResolvedImageSource::SelectedLayer {
            layer_id: donor_y,
            saved_position: SavedLayerPosition::new(1).unwrap(),
            stage: LayerImageStage::PostLocalEffects,
        };
        live_matte.channel = MatteChannel::Blue;
        live_matte.invert = false;
        live_matte.amount = 0.11;
        let before = live.clone();

        assert!(!apply_runtime_rack_values_strict(&sampled, &mut live));
        assert_eq!(
            live, before,
            "live reroute and values must remain untouched"
        );
    }

    #[test]
    fn morph_layer_racks_round_trip_the_257th_saved_position() {
        let mut slot = MorphSlot::default();
        let mut racks = vec![VisualRack::empty(); 257];
        racks[256] = VisualRack::synthetic_legacy(crate::visual_rack::LegacyRackScope::Layer);
        slot.layer_racks = Some(racks);

        let yaml = serde_yaml::to_string(&slot).unwrap();
        let restored: MorphSlot = serde_yaml::from_str(&yaml).unwrap();
        let restored = restored.layer_racks.unwrap();
        assert_eq!(restored.len(), 257);
        assert_eq!(
            restored[256].topology_signature(),
            crate::visual_rack::LEGACY_LAYER_RACK_SIGNATURE
        );
    }

    fn displace_rack(params: crate::visual_rack::DisplaceParams) -> VisualRack {
        let mut rack = VisualRack::empty();
        rack.push(VisualNodeKind::Displace(params)).unwrap();
        rack
    }

    fn displace_tap(position: u32) -> crate::visual_rack::SavedImageTap {
        crate::visual_rack::SavedImageTap {
            source: SavedImageSource::SelectedLayer {
                layer_position: crate::performance::SavedLayerPosition::new(position).unwrap(),
                stage: crate::image_routing::LayerImageStage::PostLocalEffects,
            },
            timing: crate::visual_rack::EdgeTiming::CurrentFrame,
        }
    }

    #[test]
    fn morph_interpolates_displace_amounts_only_on_an_exact_route_match() {
        use crate::visual_rack::{DisplaceBoundary, DisplaceParams};

        let tap = displace_tap(3);
        let a = displace_rack(DisplaceParams {
            tap,
            amount_x: 0.0,
            amount_y: -1.0,
            boundary: DisplaceBoundary::Wrap,
        });
        let b = displace_rack(DisplaceParams {
            tap,
            amount_x: 1.0,
            amount_y: 1.0,
            boundary: DisplaceBoundary::Mirror,
        });

        // Midpoint: amounts blend, the discrete boundary switches at the
        // midpoint, and the shared route is carried rather than interpolated.
        let sampled = interpolate_rack(&a, &b, [0.5, 0.5], true).expect("routes match");
        let VisualNodeKind::Displace(params) = sampled.iter().next().unwrap().kind else {
            panic!("displace node")
        };
        assert!((params.amount_x - 0.5).abs() <= 1.0e-6);
        assert!(params.amount_y.abs() <= 1.0e-6);
        assert_eq!(params.boundary, DisplaceBoundary::Mirror);
        assert_eq!(params.tap, tap);

        // Endpoints are exact on both the continuous and the discrete fields.
        for (weights, choose_b, expected_x, expected_boundary) in [
            ([1.0_f32, 0.0_f32], false, 0.0_f32, DisplaceBoundary::Wrap),
            ([0.0, 1.0], true, 1.0, DisplaceBoundary::Mirror),
        ] {
            let sampled = interpolate_rack(&a, &b, weights, choose_b).unwrap();
            let VisualNodeKind::Displace(params) = sampled.iter().next().unwrap().kind else {
                panic!("displace node")
            };
            assert_eq!(params.amount_x, expected_x);
            assert_eq!(params.boundary, expected_boundary);
        }

        // A different donor is different topology, not two ends of a blend.
        let rerouted = displace_rack(DisplaceParams {
            tap: displace_tap(4),
            amount_x: 1.0,
            amount_y: 1.0,
            boundary: DisplaceBoundary::Mirror,
        });
        assert!(interpolate_rack(&a, &rerouted, [0.5, 0.5], false).is_none());

        // So is a differently timed edge on the same layer.
        let retimed = displace_rack(DisplaceParams {
            tap: crate::visual_rack::SavedImageTap {
                timing: crate::visual_rack::EdgeTiming::PreviousFrame,
                ..tap
            },
            ..DisplaceParams::default()
        });
        assert!(interpolate_rack(&a, &retimed, [0.5, 0.5], false).is_none());
    }

    #[test]
    fn applying_displace_values_never_retargets_the_live_donor() {
        use crate::visual_rack::{DisplaceBoundary, DisplaceParams, RuntimeDisplaceParams};

        // Saved-to-runtime apply (Look, preset) copies values only.
        let sampled = displace_rack(DisplaceParams {
            tap: displace_tap(3),
            amount_x: 0.75,
            amount_y: -0.25,
            boundary: DisplaceBoundary::Hold,
        });
        let live_tap = crate::visual_rack::ResolvedImageTap {
            source: crate::visual_rack::ResolvedImageSource::OneBelow,
            timing: crate::visual_rack::EdgeTiming::CurrentFrame,
        };
        let mut live = sampled.resolve_routes(|_| None, |_| false);
        let node_id = live.iter().next().unwrap().stable_id;
        let RuntimeVisualNodeKind::Displace(params) = &mut live.get_mut(node_id).unwrap().kind
        else {
            panic!("displace node")
        };
        *params = RuntimeDisplaceParams {
            tap: live_tap,
            amount_x: 0.0,
            amount_y: 0.0,
            boundary: DisplaceBoundary::Transparent,
        };

        assert!(apply_saved_rack_values_to_runtime(&sampled, &mut live));
        let RuntimeVisualNodeKind::Displace(params) = live.get(node_id).unwrap().kind else {
            panic!("displace node")
        };
        assert_eq!(params.amount_x, 0.75);
        assert_eq!(params.amount_y, -0.25);
        assert_eq!(params.boundary, DisplaceBoundary::Hold);
        assert_eq!(
            params.tap, live_tap,
            "value transfer must never rewrite the live donor route"
        );

        // The strict runtime-to-runtime path instead refuses a route mismatch.
        let mut other = live.clone();
        let RuntimeVisualNodeKind::Displace(params) = &mut other.get_mut(node_id).unwrap().kind
        else {
            panic!("displace node")
        };
        params.tap = crate::visual_rack::ResolvedImageTap {
            source: crate::visual_rack::ResolvedImageSource::CleanProgram,
            timing: crate::visual_rack::EdgeTiming::PreviousFrame,
        };
        let mut target = live.clone();
        assert!(
            !apply_runtime_rack_values_strict(&other, &mut target),
            "strict apply must reject a Displace whose donor differs"
        );
        assert!(apply_runtime_rack_values_strict(&live.clone(), &mut target));
    }

    fn symmetry_rack(params: SymmetryParams) -> VisualRack {
        let mut rack = VisualRack::empty();
        rack.push(VisualNodeKind::Symmetry(params)).unwrap();
        rack
    }

    fn symmetry_donor(position: u32) -> SavedImageTap {
        SavedImageTap {
            source: SavedImageSource::SelectedLayer {
                layer_position: crate::performance::SavedLayerPosition::new(position).unwrap(),
                stage: crate::image_routing::LayerImageStage::PostLocalEffects,
            },
            timing: crate::visual_rack::EdgeTiming::CurrentFrame,
        }
    }

    /// A fully routed, fully armed field so every one of the four slots and both
    /// masks is load bearing in the topology predicate.
    fn routed_symmetry() -> SymmetryParams {
        use crate::symmetry::{SymmetryMotionMask, SymmetrySourceMask};

        SymmetryParams {
            source_mask: SymmetrySourceMask {
                carrier: true,
                donor0: true,
                donor1: true,
                clean_history: true,
            },
            motion_mask: SymmetryMotionMask {
                slot0: true,
                slot1: true,
            },
            donors: [symmetry_donor(1), symmetry_donor(2)],
            motion: [
                SavedMotionDonor::Selected {
                    saved_position: crate::performance::SavedLayerPosition::new(3).unwrap(),
                },
                SavedMotionDonor::Selected {
                    saved_position: crate::performance::SavedLayerPosition::new(4).unwrap(),
                },
            ],
            ..SymmetryParams::default()
        }
    }

    /// Closes `saved_node_topology_matches` and `interpolate_node_kind`. Without
    /// the topology arm two slots naming different donors would be declared
    /// compatible and their geometry blended across mismatched routes; without
    /// the interpolate arm even a matching pair would fall into `_ => None` and
    /// silently stop the whole rack from morphing.
    #[test]
    fn morph_interpolates_symmetry_geometry_only_on_an_exact_four_slot_route_match() {
        use crate::symmetry::{SymmetryBoundary, SymmetryMode, SymmetryMotionMask};

        let base = routed_symmetry();
        let a = symmetry_rack(SymmetryParams {
            mode: SymmetryMode::Cyclic,
            boundary: SymmetryBoundary::Wrap,
            base_folds: 2.0,
            radial_phase_deg: 170.0,
            hue_span: 0.0,
            seed: 11,
            ..base
        });
        let b = symmetry_rack(SymmetryParams {
            mode: SymmetryMode::PlanarPmm,
            boundary: SymmetryBoundary::CellularReentry,
            base_folds: 8.0,
            radial_phase_deg: -170.0,
            hue_span: 1.0,
            seed: 22,
            ..base
        });

        // Midpoint: continuous controls blend, angles take the shortest wrapped
        // arc, discrete laws switch, and every route plus both masks is carried.
        let sampled = interpolate_rack(&a, &b, [0.5, 0.5], true).expect("routes match");
        let VisualNodeKind::Symmetry(params) = sampled.iter().next().unwrap().kind else {
            panic!("symmetry node")
        };
        assert!((params.base_folds - 5.0).abs() <= 1.0e-6);
        assert!((params.hue_span - 0.5).abs() <= 1.0e-6);
        assert!(
            params.radial_phase_deg.abs() >= 179.0,
            "170 deg to -170 deg is a 20 deg arc through 180, not a 340 deg sweep: {}",
            params.radial_phase_deg
        );
        assert_eq!(params.mode, SymmetryMode::PlanarPmm);
        assert_eq!(params.boundary, SymmetryBoundary::CellularReentry);
        assert_eq!(params.donors, base.donors);
        assert_eq!(params.motion, base.motion);
        assert_eq!(params.source_mask, base.source_mask);
        assert_eq!(params.motion_mask, base.motion_mask);
        assert!(
            params.seed == 11 || params.seed == 22,
            "the seed is an endpoint recall, never a mixture: {}",
            params.seed
        );

        // Endpoints are exact on the continuous, discrete, and seed fields.
        for (weights, choose_b, folds, mode, seed) in [
            (
                [1.0_f32, 0.0_f32],
                false,
                2.0_f32,
                SymmetryMode::Cyclic,
                11_u32,
            ),
            ([0.0, 1.0], true, 8.0, SymmetryMode::PlanarPmm, 22),
        ] {
            let sampled = interpolate_rack(&a, &b, weights, choose_b).unwrap();
            let VisualNodeKind::Symmetry(params) = sampled.iter().next().unwrap().kind else {
                panic!("symmetry node")
            };
            assert_eq!(params.base_folds, folds);
            assert_eq!(params.mode, mode);
            assert_eq!(params.seed, seed);
        }

        // Each of the four slots independently makes the pair incompatible, and
        // slot index is route identity, so swapping the two donors is a
        // mismatch rather than a match.
        let mismatches = [
            SymmetryParams {
                donors: [symmetry_donor(9), base.donors[1]],
                ..base
            },
            SymmetryParams {
                donors: [base.donors[0], symmetry_donor(9)],
                ..base
            },
            SymmetryParams {
                donors: [base.donors[1], base.donors[0]],
                ..base
            },
            SymmetryParams {
                motion: [
                    SavedMotionDonor::None,
                    SavedMotionDonor::Selected {
                        saved_position: crate::performance::SavedLayerPosition::new(4).unwrap(),
                    },
                ],
                ..base
            },
            SymmetryParams {
                motion: [base.motion[0], SavedMotionDonor::None],
                ..base
            },
            // A retimed edge on the same layer is different topology too.
            SymmetryParams {
                donors: [
                    SavedImageTap {
                        timing: crate::visual_rack::EdgeTiming::PreviousFrame,
                        ..base.donors[0]
                    },
                    base.donors[1],
                ],
                ..base
            },
            // A mask decides admission, so a differently armed field describes a
            // different graph.
            SymmetryParams {
                source_mask: crate::symmetry::SymmetrySourceMask {
                    donor1: false,
                    ..base.source_mask
                },
                ..base
            },
            SymmetryParams {
                motion_mask: SymmetryMotionMask {
                    slot0: true,
                    slot1: false,
                },
                ..base
            },
        ];
        for (index, mismatch) in mismatches.into_iter().enumerate() {
            let rack = symmetry_rack(mismatch);
            assert!(
                interpolate_rack(&a, &rack, [0.5, 0.5], false).is_none(),
                "mismatch {index} must stop the rack from morphing"
            );
        }
    }

    /// Closes `apply_saved_node_kind_values`, its saved-to-runtime twin, and
    /// `apply_runtime_node_kind_values`. A missing arm returns `false` there,
    /// which silently refuses the entire Look or preset for the whole rack.
    #[test]
    fn applying_symmetry_values_never_retargets_a_slot_or_rearms_a_mask() {
        use crate::symmetry::{
            RuntimeSymmetryParams, SymmetryBoundary, SymmetryMode, SymmetryMotionMask,
            SymmetrySourceMask,
        };

        let sampled = symmetry_rack(SymmetryParams {
            mode: SymmetryMode::Dihedral,
            boundary: SymmetryBoundary::Hold,
            base_folds: 7.0,
            fold_offset: -2.0,
            radial_phase_deg: 45.0,
            orbit_phase: 0.25,
            planar_axis_deg: -30.0,
            planar_phase: 1.5,
            cell_skew: 0.5,
            spiral_scale: -0.25,
            orbit_radius: 0.75,
            orbit_spin_deg: 90.0,
            center: [0.25, 0.75],
            motion_gain: -0.5,
            hue_span: 0.5,
            seed: 4_242,
            ..routed_symmetry()
        });

        // Saved-to-runtime apply (Look, preset) copies values only.
        let mut live = sampled.resolve_routes(|_| None, |_| false);
        let node_id = live.iter().next().unwrap().stable_id;
        let live_donor = crate::visual_rack::ResolvedImageTap {
            source: crate::visual_rack::ResolvedImageSource::OneBelow,
            timing: crate::visual_rack::EdgeTiming::CurrentFrame,
        };
        let live_mask = SymmetrySourceMask {
            carrier: true,
            donor0: false,
            donor1: false,
            clean_history: false,
        };
        let RuntimeVisualNodeKind::Symmetry(params) = &mut live.get_mut(node_id).unwrap().kind
        else {
            panic!("symmetry node")
        };
        *params = RuntimeSymmetryParams {
            donors: [live_donor, live_donor],
            source_mask: live_mask,
            motion_mask: SymmetryMotionMask {
                slot0: false,
                slot1: false,
            },
            ..RuntimeSymmetryParams::default()
        };

        assert!(apply_saved_rack_values_to_runtime(&sampled, &mut live));
        let RuntimeVisualNodeKind::Symmetry(params) = live.get(node_id).unwrap().kind else {
            panic!("symmetry node")
        };
        assert_eq!(params.mode, SymmetryMode::Dihedral);
        assert_eq!(params.boundary, SymmetryBoundary::Hold);
        assert_eq!(params.base_folds, 7.0);
        assert_eq!(params.fold_offset, -2.0);
        assert_eq!(params.radial_phase_deg, 45.0);
        assert_eq!(params.orbit_phase, 0.25);
        assert_eq!(params.planar_axis_deg, -30.0);
        assert_eq!(params.planar_phase, 1.5);
        assert_eq!(params.cell_skew, 0.5);
        assert_eq!(params.spiral_scale, -0.25);
        assert_eq!(params.orbit_radius, 0.75);
        assert_eq!(params.orbit_spin_deg, 90.0);
        assert_eq!(params.center, [0.25, 0.75]);
        assert_eq!(params.motion_gain, -0.5);
        assert_eq!(params.hue_span, 0.5);
        assert_eq!(params.seed, 4_242);
        assert_eq!(
            params.donors,
            [live_donor, live_donor],
            "value transfer must never rewrite a live route slot"
        );
        assert_eq!(
            params.motion,
            [crate::motion::MotionDonor::None; SYMMETRY_MOTION_SLOTS]
        );
        assert_eq!(
            params.source_mask, live_mask,
            "value transfer must never arm a source the operator never armed"
        );

        // The strict runtime path refuses a route or mask mismatch outright.
        let mut rerouted = live.clone();
        let RuntimeVisualNodeKind::Symmetry(params) = &mut rerouted.get_mut(node_id).unwrap().kind
        else {
            panic!("symmetry node")
        };
        params.donors[1] = crate::visual_rack::ResolvedImageTap {
            source: crate::visual_rack::ResolvedImageSource::CleanProgram,
            timing: crate::visual_rack::EdgeTiming::PreviousFrame,
        };
        let mut target = live.clone();
        assert!(
            !apply_runtime_rack_values_strict(&rerouted, &mut target),
            "strict apply must reject a Symmetry Field whose second slot differs"
        );

        let mut rearmed = live.clone();
        let RuntimeVisualNodeKind::Symmetry(params) = &mut rearmed.get_mut(node_id).unwrap().kind
        else {
            panic!("symmetry node")
        };
        params.source_mask.donor1 = true;
        assert!(
            !apply_runtime_rack_values_strict(&rearmed, &mut target),
            "strict apply must reject a differently armed Symmetry Field"
        );
        assert!(apply_runtime_rack_values_strict(&live.clone(), &mut target));

        // The saved-to-saved path is gated by the strict saved topology, so a
        // route-equal target transfers values while the routes stay put, and a
        // route-differing target is refused before any field is written.
        let mut saved_live = symmetry_rack(routed_symmetry());
        assert!(apply_rack_values(&sampled, &mut saved_live));
        let VisualNodeKind::Symmetry(params) = saved_live.iter().next().unwrap().kind else {
            panic!("symmetry node")
        };
        assert_eq!(params.base_folds, 7.0);
        assert_eq!(params.seed, 4_242);
        assert_eq!(params.donors, routed_symmetry().donors);
        assert_eq!(params.motion, routed_symmetry().motion);

        let mut saved_rerouted = symmetry_rack(SymmetryParams {
            donors: [symmetry_donor(7), symmetry_donor(8)],
            ..routed_symmetry()
        });
        assert!(
            !apply_rack_values(&sampled, &mut saved_rerouted),
            "the saved values-only path refuses a differently routed field"
        );
    }

    /// Closes `remap_saved_rack_routes_after_move`/`_after_remove` for every
    /// slot a Symmetry Field owns. Without them a stored slot diverges from its
    /// A/B twin after a stack edit, `symmetry_route_matches` then reports false,
    /// and the whole rack silently stops morphing.
    #[test]
    fn layer_stack_remaps_carry_every_symmetry_slot_and_tombstone_the_removed_one() {
        use crate::performance::SavedLayerPosition;

        let position = |value: u32| SavedLayerPosition::new(value).unwrap();
        let params = SymmetryParams {
            donors: [symmetry_donor(0), symmetry_donor(2)],
            motion: [
                SavedMotionDonor::Selected {
                    saved_position: position(1),
                },
                SavedMotionDonor::Selected {
                    saved_position: position(2),
                },
            ],
            ..routed_symmetry()
        };
        let mut slot = MorphSlot {
            master_rack: Some(symmetry_rack(params)),
            ..MorphSlot::default()
        };
        let read = |slot: &MorphSlot| {
            let VisualNodeKind::Symmetry(params) = slot
                .master_rack
                .as_ref()
                .unwrap()
                .iter()
                .next()
                .unwrap()
                .kind
            else {
                panic!("symmetry node")
            };
            params
        };

        slot.remap_layers_after_move(0, 2);
        let moved = read(&slot);
        assert_eq!(moved.donors[0], symmetry_donor(2));
        assert_eq!(moved.donors[1], symmetry_donor(1));
        assert_eq!(
            moved.motion,
            [
                SavedMotionDonor::Selected {
                    saved_position: position(0)
                },
                SavedMotionDonor::Selected {
                    saved_position: position(1)
                },
            ]
        );

        slot.remap_layers_after_remove(1);
        let removed = read(&slot);
        assert_eq!(
            removed.donors[0],
            symmetry_donor(1),
            "a position above the removed index decrements"
        );
        assert_eq!(
            removed.donors[1].source,
            SavedImageSource::MissingSelectedLayer {
                saved_position: position(1),
                stage: crate::image_routing::LayerImageStage::PostLocalEffects,
            },
            "the vacated position becomes a tombstone that never rebinds"
        );
        assert_eq!(
            removed.motion,
            [
                SavedMotionDonor::Selected {
                    saved_position: position(0)
                },
                SavedMotionDonor::Missing {
                    saved_position: position(1)
                },
            ]
        );

        // A tombstone stays a tombstone through every later stack edit.
        slot.remap_layers_after_move(0, 1);
        let later = read(&slot);
        assert_eq!(later.donors[1].source, removed.donors[1].source);
        assert_eq!(later.motion[1], removed.motion[1]);

        // Group identities are stable and never rewritten by a layer edit.
        let group_tap = SavedImageTap {
            source: SavedImageSource::GroupOutput {
                group_id: crate::visual_rack::GroupId::new(91).unwrap(),
            },
            timing: crate::visual_rack::EdgeTiming::PreviousFrame,
        };
        let mut group_slot = MorphSlot {
            master_rack: Some(symmetry_rack(SymmetryParams {
                donors: [group_tap, group_tap],
                ..routed_symmetry()
            })),
            ..MorphSlot::default()
        };
        group_slot.remap_layers_after_move(0, 2);
        group_slot.remap_layers_after_remove(0);
        assert_eq!(read(&group_slot).donors, [group_tap, group_tap]);
    }

    fn residual_rack(params: ResidualParams) -> VisualRack {
        let mut rack = VisualRack::empty();
        rack.push(VisualNodeKind::Residual(params)).unwrap();
        rack
    }

    /// The single saved Residual node of a fixture rack.
    fn residual_of(rack: &VisualRack) -> ResidualParams {
        let VisualNodeKind::Residual(params) = rack.iter().next().unwrap().kind else {
            panic!("residual node")
        };
        params
    }

    #[test]
    fn morph_interpolates_residual_values_only_when_both_route_slots_match() {
        use crate::visual_rack::{ResidualBlock, ResidualQuantization};

        // `displace_tap` is a plain saved layer route and carries no kind of
        // its own; both Residual slots are built from it at distinct positions.
        let structure = displace_tap(3);
        let detail = displace_tap(5);
        let a = residual_rack(ResidualParams {
            structure,
            detail,
            block: ResidualBlock::Four,
            quantization: ResidualQuantization::Off,
            mix: 0.0,
            detail_gain: 0.0,
            seed: 11,
            ..ResidualParams::default()
        });
        let b = residual_rack(ResidualParams {
            structure,
            detail,
            block: ResidualBlock::SixtyFour,
            quantization: ResidualQuantization::Fine,
            mix: 1.0,
            detail_gain: 4.0,
            seed: 97,
            ..ResidualParams::default()
        });

        // Midpoint: both continuous values blend, both discrete laws switch at
        // the midpoint, both routes are carried, and the seed is recalled from
        // an endpoint rather than mixed into a third unauthored lattice.
        let sampled = interpolate_rack(&a, &b, [0.5, 0.5], true).expect("both route slots match");
        let params = residual_of(&sampled);
        assert!((params.mix - 0.5).abs() <= 1.0e-6);
        assert!((params.detail_gain - 2.0).abs() <= 1.0e-6);
        assert_eq!(params.block, ResidualBlock::SixtyFour);
        assert_eq!(params.quantization, ResidualQuantization::Fine);
        assert_eq!(params.structure, structure);
        assert_eq!(params.detail, detail);
        assert_eq!(
            params.seed, 97,
            "a seed is an endpoint recall, never an interpolated RNG"
        );

        // Endpoints are exact on the continuous values, the two discrete laws,
        // and the seed.
        for (weights, choose_b, expected_mix, expected_gain, expected_block, expected_seed) in [
            (
                [1.0_f32, 0.0_f32],
                false,
                0.0_f32,
                0.0_f32,
                ResidualBlock::Four,
                11_u32,
            ),
            ([0.0, 1.0], true, 1.0, 4.0, ResidualBlock::SixtyFour, 97),
        ] {
            let sampled = interpolate_rack(&a, &b, weights, choose_b).unwrap();
            let params = residual_of(&sampled);
            assert_eq!(params.mix, expected_mix);
            assert_eq!(params.detail_gain, expected_gain);
            assert_eq!(params.block, expected_block);
            assert_eq!(params.seed, expected_seed);
        }

        // A different structure route alone is different topology.
        let restructured = residual_rack(ResidualParams {
            structure: displace_tap(4),
            ..residual_of(&b)
        });
        assert!(interpolate_rack(&a, &restructured, [0.5, 0.5], false).is_none());

        // And so is a different detail route alone. A one-slot route predicate
        // would agree on `structure` and blend two unrelated recombinations.
        let redetailed = residual_rack(ResidualParams {
            detail: displace_tap(6),
            ..residual_of(&b)
        });
        assert!(
            interpolate_rack(&a, &redetailed, [0.5, 0.5], false).is_none(),
            "the detail slot must join the route-equality gate"
        );

        // A retimed edge on the same layer is a different route on either slot.
        for retimed in [
            residual_rack(ResidualParams {
                structure: crate::visual_rack::SavedImageTap {
                    timing: crate::visual_rack::EdgeTiming::PreviousFrame,
                    ..structure
                },
                ..residual_of(&b)
            }),
            residual_rack(ResidualParams {
                detail: crate::visual_rack::SavedImageTap {
                    timing: crate::visual_rack::EdgeTiming::PreviousFrame,
                    ..detail
                },
                ..residual_of(&b)
            }),
        ] {
            assert!(interpolate_rack(&a, &retimed, [0.5, 0.5], false).is_none());
        }

        // The same saved-topology gate also guards the saved values-only apply,
        // where no interpolator re-check exists to catch a mismatch later.
        let mut mismatched = redetailed.clone();
        assert!(
            !apply_rack_values(&a, &mut mismatched),
            "a mismatched detail route must be refused, not silently accepted"
        );
        let mut matching = b.clone();
        assert!(apply_rack_values(&a, &mut matching));
        let applied = residual_of(&matching);
        assert_eq!(applied.mix, 0.0);
        assert_eq!(applied.detail_gain, 0.0);
        assert_eq!(applied.block, ResidualBlock::Four);
        assert_eq!(applied.quantization, ResidualQuantization::Off);
        assert_eq!(applied.seed, 11);
        assert_eq!(applied.structure, structure);
        assert_eq!(applied.detail, detail);
    }

    #[test]
    fn applying_residual_values_never_retargets_either_live_donor_route() {
        use crate::visual_rack::{
            EdgeTiming, ResidualBlock, ResidualQuantization, ResolvedImageSource, ResolvedImageTap,
            RuntimeResidualParams,
        };

        let sampled = residual_rack(ResidualParams {
            structure: displace_tap(3),
            detail: displace_tap(5),
            block: ResidualBlock::Sixteen,
            quantization: ResidualQuantization::Medium,
            mix: 0.75,
            detail_gain: 2.5,
            seed: 4242,
            ..ResidualParams::default()
        });

        // Saved-to-runtime apply (Look, preset) copies values only. Both live
        // routes are deliberately unlike the captured ones.
        let live_structure = ResolvedImageTap {
            source: ResolvedImageSource::OneBelow,
            timing: EdgeTiming::CurrentFrame,
        };
        let live_detail = ResolvedImageTap {
            source: ResolvedImageSource::CleanProgram,
            timing: EdgeTiming::PreviousFrame,
        };
        let mut live = sampled.resolve_routes(|_| None, |_| false);
        let node_id = live.iter().next().unwrap().stable_id;
        let RuntimeVisualNodeKind::Residual(params) = &mut live.get_mut(node_id).unwrap().kind
        else {
            panic!("residual node")
        };
        *params = RuntimeResidualParams {
            structure: live_structure,
            detail: live_detail,
            ..RuntimeResidualParams::default()
        };

        assert!(apply_saved_rack_values_to_runtime(&sampled, &mut live));
        let RuntimeVisualNodeKind::Residual(params) = live.get(node_id).unwrap().kind else {
            panic!("residual node")
        };
        assert_eq!(params.mix, 0.75);
        assert_eq!(params.detail_gain, 2.5);
        assert_eq!(params.block, ResidualBlock::Sixteen);
        assert_eq!(params.quantization, ResidualQuantization::Medium);
        assert_eq!(params.seed, 4242);
        assert_eq!(
            [params.structure, params.detail],
            [live_structure, live_detail],
            "value transfer must never rewrite either live donor route"
        );

        // The strict runtime-to-runtime path instead refuses a mismatch, and it
        // must do so for either slot independently.
        for slot in [
            crate::visual_rack::RESIDUAL_STRUCTURE_SLOT,
            crate::visual_rack::RESIDUAL_DETAIL_SLOT,
        ] {
            let mut other = live.clone();
            let RuntimeVisualNodeKind::Residual(params) = &mut other.get_mut(node_id).unwrap().kind
            else {
                panic!("residual node")
            };
            *params.route_mut(slot).expect("both slots name a route") = ResolvedImageTap {
                source: ResolvedImageSource::AllBelow,
                timing: EdgeTiming::PreviousFrame,
            };
            let mut target = live.clone();
            assert!(
                !apply_runtime_rack_values_strict(&other, &mut target),
                "strict apply must reject a Residual whose slot {slot} donor differs"
            );
        }
        let mut target = live.clone();
        assert!(apply_runtime_rack_values_strict(&live.clone(), &mut target));
        let RuntimeVisualNodeKind::Residual(params) = target.get(node_id).unwrap().kind else {
            panic!("residual node")
        };
        assert_eq!(
            [params.structure, params.detail],
            [live_structure, live_detail]
        );
    }

    #[test]
    fn layer_stack_remaps_keep_both_residual_route_slots_aligned() {
        use crate::image_routing::LayerImageStage;
        use crate::performance::SavedLayerPosition;
        use crate::visual_rack::{EdgeTiming, GroupId, SavedImageTap};

        let layer_tap = |position: u32| SavedImageTap {
            source: SavedImageSource::SelectedLayer {
                layer_position: SavedLayerPosition::new(position).unwrap(),
                stage: LayerImageStage::PostLocalEffects,
            },
            timing: EdgeTiming::CurrentFrame,
        };
        let group_tap = SavedImageTap {
            source: SavedImageSource::GroupOutput {
                group_id: GroupId::new(91).unwrap(),
            },
            timing: EdgeTiming::PreviousFrame,
        };

        let residual_at = |rack: &VisualRack, index: usize| {
            let VisualNodeKind::Residual(params) = rack.iter().nth(index).unwrap().kind else {
                panic!("residual node")
            };
            params
        };
        let slot_with = |mix: f32| {
            let mut rack = VisualRack::empty();
            rack.push(VisualNodeKind::Residual(ResidualParams {
                structure: layer_tap(0),
                detail: layer_tap(2),
                mix,
                ..ResidualParams::default()
            }))
            .unwrap();
            rack.push(VisualNodeKind::Residual(ResidualParams {
                structure: layer_tap(1),
                detail: group_tap,
                mix,
                ..ResidualParams::default()
            }))
            .unwrap();
            MorphSlot {
                master_rack: Some(rack),
                ..MorphSlot::default()
            }
        };
        let mut morph = Morph {
            a: Some(slot_with(0.25)),
            b: Some(slot_with(0.75)),
            ..Default::default()
        };

        // Moving position 0 to 2 pushes the old 2 down to 1. Each slot follows
        // its own saved position; a one-slot remap would leave `detail` stale.
        morph.remap_layers_after_move(0, 2);
        for slot in [morph.a.as_ref().unwrap(), morph.b.as_ref().unwrap()] {
            let rack = slot.master_rack.as_ref().unwrap();
            assert_eq!(residual_at(rack, 0).structure, layer_tap(2));
            assert_eq!(residual_at(rack, 0).detail, layer_tap(1));
            assert_eq!(residual_at(rack, 1).structure, layer_tap(0));
            assert_eq!(
                residual_at(rack, 1).detail,
                group_tap,
                "layer permutations must never rewrite stable group identities"
            );
        }

        // Removing position 1 decrements the structure slot and tombstones the
        // detail slot, independently and without rebinding either.
        morph.remap_layers_after_remove(1);
        for slot in [morph.a.as_ref().unwrap(), morph.b.as_ref().unwrap()] {
            let rack = slot.master_rack.as_ref().unwrap();
            assert_eq!(residual_at(rack, 0).structure, layer_tap(1));
            assert_eq!(
                residual_at(rack, 0).detail.source,
                SavedImageSource::MissingSelectedLayer {
                    saved_position: SavedLayerPosition::new(1).unwrap(),
                    stage: LayerImageStage::PostLocalEffects,
                }
            );
            assert_eq!(residual_at(rack, 1).structure, layer_tap(0));
            assert_eq!(residual_at(rack, 1).detail, group_tap);
        }

        // Both endpoints moved identically, so the pair is still route-equal and
        // Morph keeps sampling the rack instead of silently dropping it.
        let sampled = morph.sample(0.5).unwrap();
        let rack = sampled.master_rack.as_ref().expect("rack still morphs");
        assert!((residual_at(rack, 0).mix - 0.5).abs() <= 1.0e-6);
    }

    fn gesture_slot(radius: f32, strength: f32, retention: f32, track: &str) -> MorphSlot {
        let mut slot = slot(1.0, 0.0, 0.0);
        slot.gesture = Some(GestureMorphSnapshot {
            canvas: GestureCanvasConfig {
                radius,
                strength,
                retention,
            },
            track_checksum: track.to_string(),
        });
        slot
    }

    /// A recorded track is topology, not a value. The canvas controls blend;
    /// the recording is carried by identity from A and two slots naming
    /// different recordings do not blend at all.
    #[test]
    fn morph_blends_gesture_canvas_values_only_on_an_exact_recorded_track_match() {
        let digest = "a".repeat(64);
        let morph = Morph {
            a: Some(gesture_slot(0.2, 0.4, 0.8, &digest)),
            b: Some(gesture_slot(0.6, 0.8, 0.4, &digest)),
            ..Default::default()
        };
        assert!(morph.controls_master_gesture());

        let midpoint = morph.sample(0.5).unwrap();
        let gesture = midpoint.gesture.as_ref().expect("both slots own gesture");
        close(gesture.canvas.radius, 0.4);
        close(gesture.canvas.strength, 0.6);
        close(gesture.canvas.retention, 0.6);
        assert_eq!(
            gesture.track_checksum, digest,
            "the recording is carried, never blended"
        );

        // Endpoints are exact, and a sampled world writes only the authored
        // canvas values into the live state.
        for (position, expected) in [(0.0_f32, [0.2_f32, 0.4, 0.8]), (1.0, [0.6, 0.8, 0.4])] {
            let sample = morph.sample(position).unwrap();
            let mut canvas = crate::gesture_canvas::GestureCanvasParams::default();
            sample.apply_gesture_to(&mut canvas);
            close(canvas.radius, expected[0]);
            close(canvas.strength, expected[1]);
            close(canvas.retention, expected[2]);
        }

        // Two different recordings are two pieces, not two ends of a blend.
        let rerouted = Morph {
            a: Some(gesture_slot(0.2, 0.4, 0.8, &digest)),
            b: Some(gesture_slot(0.6, 0.8, 0.4, &"b".repeat(64))),
            ..Default::default()
        };
        assert!(!rerouted.controls_master_gesture());
        assert!(rerouted.sample(0.5).unwrap().gesture.is_none());
        let mut untouched = crate::gesture_canvas::GestureCanvasParams::default();
        let before = untouched;
        rerouted
            .sample(0.5)
            .unwrap()
            .apply_gesture_to(&mut untouched);
        assert_eq!(untouched, before);

        // A legacy slot that predates gesture etching claims nothing.
        let legacy = Morph {
            a: Some(gesture_slot(0.2, 0.4, 0.8, &digest)),
            b: Some(slot(1.0, 0.0, 0.0)),
            ..Default::default()
        };
        assert!(!legacy.controls_master_gesture());
        assert!(legacy.sample(0.5).unwrap().gesture.is_none());
    }

    /// The slot names a recording; it never owns one. Nothing in the Morph
    /// snapshot can add, remove, retime, or reorder a recorded event, and the
    /// section is additive so a legacy slot round-trips unchanged.
    #[test]
    fn a_gesture_morph_slot_carries_only_a_recording_identity_and_never_its_events() {
        let digest = "c".repeat(64);
        let slot = gesture_slot(0.3, 0.5, 0.7, &digest);
        let value = serde_json::to_value(&slot).unwrap();
        let gesture = value["gesture"].as_object().expect("gesture section");
        let mut keys: Vec<_> = gesture.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            ["canvas", "track_checksum"],
            "a Morph slot must carry a recording's identity and nothing else"
        );
        let canvas = gesture["canvas"].as_object().expect("canvas values");
        let mut canvas_keys: Vec<_> = canvas.keys().map(String::as_str).collect();
        canvas_keys.sort_unstable();
        assert_eq!(canvas_keys, ["radius", "retention", "strength"]);
        assert!(
            !value.to_string().contains("events"),
            "no recorded event may reach a Morph slot"
        );

        let restored: MorphSlot = serde_json::from_value(value).unwrap();
        assert_eq!(restored.gesture, slot.gesture);

        // Absent is exactly the pre-gesture path.
        let legacy = MorphSlot::default();
        assert_eq!(legacy.gesture, None);
        let legacy_json = serde_json::to_value(&legacy).unwrap();
        assert!(legacy_json.get("gesture").is_none());
        assert_eq!(
            serde_json::from_value::<MorphSlot>(legacy_json)
                .unwrap()
                .gesture,
            None
        );

        // Sanitization runs on the section without touching the identity.
        let hostile = MorphSlot {
            gesture: Some(GestureMorphSnapshot {
                canvas: GestureCanvasConfig {
                    radius: f32::NAN,
                    strength: 9.0,
                    retention: -3.0,
                },
                track_checksum: digest.clone(),
            }),
            ..MorphSlot::default()
        }
        .sanitized();
        let sanitized = hostile.gesture.expect("section retained");
        close(
            sanitized.canvas.radius,
            GestureCanvasConfig::default().radius,
        );
        assert_eq!(sanitized.canvas.strength, 1.0);
        assert_eq!(sanitized.canvas.retention, 0.0);
        assert_eq!(sanitized.track_checksum, digest);
    }
}
