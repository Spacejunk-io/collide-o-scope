/// The rate at which the temporal controls were authored. Feedback retention,
/// zoom, rotation, and the history ring are all expressed against this
/// reference so a render at 24, 30, or 60 fps describes the same motion.
pub const TEMPORAL_REFERENCE_FPS: f32 = 30.0;

#[allow(
    unused_imports,
    reason = "T0 exposes the frozen authoring vocabulary before T1 consumers land"
)]
pub use crate::temporal::{
    CollisionAtlasParams, CollisionScoreParams, CollisionScoreTrigger, RefreshGardenGate,
    RefreshGardenParams, TemporalInterpolation, TemporalLoomParams, TemporalOriginalsParams,
    TemporalTopology,
};

/// Temporal (frame-history) effect parameters: feedback trails and
/// slit-scan, driven by the renderer's ring buffer of past output frames.
#[derive(Debug, Clone, Copy)]
pub struct TemporalParams {
    pub feedback: f32,  // retention per 1/30 second; 0 = off
    pub fb_zoom: f32,   // zoom per 1/30 second (1.0 = none)
    pub fb_rotate: f32, // rotation per 1/30 second, degrees
    pub slitscan: f32,  // 0 = off, 1 = deepest time-warp across the frame
    /// Arbitrary scan direction in degrees: 0 scans along Y, 90 along X.
    pub slit_angle: f32,
    /// Legacy 0/1 axis retained for old patches and protocol clients.
    pub slit_axis: f32,
    /// 0=off, 1=keep motion, 2=keep stillness, 3=keep brightening,
    /// 4=keep darkening. The mask compares the clean current program with a
    /// frame from the fixed-rate history ring.
    pub key_mode: f32,
    /// Normalized color/luminance delta at which the temporal mask opens.
    pub key_threshold: f32,
    /// Feather around [`Self::key_threshold`].
    pub key_softness: f32,
    /// Age of the reference image in 30 Hz history frames (1..23).
    pub key_history: f32,
    /// Inert/zero by default. T0 freezes the authoring contract while the
    /// legacy 64-byte GPU path remains the only materialized implementation.
    pub originals: TemporalOriginalsParams,
}

/// Return an aspect-correct, normalized scan direction for shader use.
/// `dot(uv - 0.5, direction) + 0.5` then spans exactly 0..1 over the
/// rectangular frame at every angle, including both diagonal corners.
pub(crate) fn normalized_slit_direction(angle_degrees: f32, width: u32, height: u32) -> [f32; 2] {
    let angle = finite_or(angle_degrees, 0.0).to_radians();
    let aspect = width.max(1) as f32 / height.max(1) as f32;
    let physical = [angle.sin(), angle.cos()];
    let uv_direction = [physical[0] * aspect, physical[1]];
    let projection_span = uv_direction[0].abs() + uv_direction[1].abs();
    if projection_span <= f32::EPSILON || !projection_span.is_finite() {
        [0.0, 1.0]
    } else {
        [
            uv_direction[0] / projection_span,
            uv_direction[1] / projection_span,
        ]
    }
}

impl Default for TemporalParams {
    fn default() -> Self {
        Self {
            feedback: 0.0,
            fb_zoom: 1.0,
            fb_rotate: 0.0,
            slitscan: 0.0,
            slit_angle: 0.0,
            slit_axis: 0.0,
            key_mode: 0.0,
            key_threshold: 0.1,
            key_softness: 0.03,
            key_history: 1.0,
            originals: TemporalOriginalsParams::default(),
        }
    }
}

impl TemporalParams {
    pub fn is_active(&self) -> bool {
        self.feedback > 0.0 || self.slitscan > 0.0 || self.key_mode > 0.0
    }

    /// Reset only temporal-domain keying while preserving trails/slit-scan.
    #[allow(dead_code)]
    pub fn reset_key(&mut self) {
        let defaults = Self::default();
        self.key_mode = defaults.key_mode;
        self.key_threshold = defaults.key_threshold;
        self.key_softness = defaults.key_softness;
        self.key_history = defaults.key_history;
    }

    /// Convert the 30-fps-authored controls into values for one render step.
    ///
    /// Retention and zoom compose multiplicatively, while rotation composes
    /// additively. Applying the returned values twice at 60 fps therefore
    /// produces the same transform as applying the source values once at
    /// 30 fps. A non-finite delta falls back to one reference frame; negative
    /// deltas are treated as zero elapsed time.
    pub(crate) fn for_frame_delta(&self, delta_seconds: f32) -> Self {
        let reference_delta = 1.0 / TEMPORAL_REFERENCE_FPS;
        let delta = if delta_seconds.is_finite() {
            delta_seconds.max(0.0)
        } else {
            reference_delta
        };
        let frame_scale = delta * TEMPORAL_REFERENCE_FPS;

        let feedback = finite_or(self.feedback, 0.0).clamp(0.0, 0.98);
        let zoom = finite_or(self.fb_zoom, 1.0).clamp(0.5, 2.0);
        let rotate = finite_or(self.fb_rotate, 0.0);

        Self {
            // Keep an explicitly disabled effect disabled even at dt=0,
            // where IEEE pow(0, 0) would otherwise yield 1.
            feedback: if feedback == 0.0 {
                0.0
            } else {
                feedback.powf(frame_scale)
            },
            fb_zoom: finite_or(zoom.powf(frame_scale), 1.0).clamp(0.01, 100.0),
            fb_rotate: finite_or(rotate * frame_scale, 0.0),
            slitscan: finite_or(self.slitscan, 0.0).clamp(0.0, 1.0),
            slit_angle: finite_or(self.slit_angle, 0.0).clamp(-180.0, 180.0),
            slit_axis: finite_or(self.slit_axis, 0.0).clamp(0.0, 1.0),
            key_mode: finite_or(self.key_mode, 0.0).round().clamp(0.0, 4.0),
            key_threshold: finite_or(self.key_threshold, 0.1).clamp(0.0, 1.0),
            key_softness: finite_or(self.key_softness, 0.03).clamp(0.0, 0.5),
            key_history: finite_or(self.key_history, 1.0).round().clamp(1.0, 23.0),
            originals: self.originals.sanitized(),
        }
    }
}

fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

#[cfg(test)]
mod temporal_tests {
    use super::*;

    fn close(a: f32, b: f32) {
        assert!((a - b).abs() < 1.0e-5, "{a} != {b}");
    }

    #[test]
    fn two_sixty_fps_steps_equal_one_reference_step() {
        let params = TemporalParams {
            feedback: 0.81,
            fb_zoom: 1.04,
            fb_rotate: 3.0,
            slitscan: 0.7,
            slit_angle: 42.0,
            slit_axis: 1.0,
            key_mode: 3.0,
            key_threshold: 0.2,
            key_softness: 0.04,
            key_history: 7.0,
            originals: TemporalOriginalsParams::default(),
        };
        let half = params.for_frame_delta(1.0 / 60.0);

        close(half.feedback * half.feedback, params.feedback);
        close(half.fb_zoom * half.fb_zoom, params.fb_zoom);
        close(half.fb_rotate + half.fb_rotate, params.fb_rotate);
        close(half.slitscan, params.slitscan);
        close(half.key_mode, params.key_mode);
        close(half.key_threshold, params.key_threshold);
        close(half.key_softness, params.key_softness);
        close(half.key_history, params.key_history);
    }

    #[test]
    fn invalid_values_are_sanitized() {
        let params = TemporalParams {
            feedback: f32::NAN,
            fb_zoom: f32::INFINITY,
            fb_rotate: f32::NEG_INFINITY,
            slitscan: f32::NAN,
            slit_angle: f32::INFINITY,
            slit_axis: 8.0,
            key_mode: f32::INFINITY,
            key_threshold: f32::NAN,
            key_softness: f32::NEG_INFINITY,
            key_history: 999.0,
            originals: TemporalOriginalsParams::default(),
        };
        let normalized = params.for_frame_delta(f32::NAN);

        assert_eq!(normalized.feedback, 0.0);
        assert_eq!(normalized.fb_zoom, 1.0);
        assert_eq!(normalized.fb_rotate, 0.0);
        assert_eq!(normalized.slitscan, 0.0);
        assert_eq!(normalized.slit_angle, 0.0);
        assert_eq!(normalized.slit_axis, 1.0);
        assert_eq!(normalized.key_mode, 0.0);
        assert_eq!(normalized.key_threshold, 0.1);
        assert_eq!(normalized.key_softness, 0.03);
        assert_eq!(normalized.key_history, 23.0);
    }

    #[test]
    fn key_reset_preserves_other_temporal_effects() {
        let mut params = TemporalParams {
            feedback: 0.7,
            slitscan: 0.4,
            key_mode: 4.0,
            key_threshold: 0.8,
            key_softness: 0.2,
            key_history: 12.0,
            ..Default::default()
        };
        params.reset_key();
        assert_eq!(params.feedback, 0.7);
        assert_eq!(params.slitscan, 0.4);
        assert_eq!(params.key_mode, 0.0);
        assert_eq!(params.key_threshold, 0.1);
        assert_eq!(params.key_softness, 0.03);
        assert_eq!(params.key_history, 1.0);
    }

    #[test]
    fn slit_direction_is_aspect_correct_and_spans_every_frame() {
        let horizontal = normalized_slit_direction(90.0, 1920, 1080);
        close(horizontal[0], 1.0);
        close(horizontal[1], 0.0);

        let vertical = normalized_slit_direction(0.0, 1920, 1080);
        close(vertical[0], 0.0);
        close(vertical[1], 1.0);

        let diagonal = normalized_slit_direction(45.0, 1920, 1080);
        close(diagonal[0].abs() + diagonal[1].abs(), 1.0);
        // Physical 45 degrees on a 16:9 frame needs a wider UV-space X
        // component; the shader therefore compensates the aspect ratio.
        assert!(diagonal[0] > diagonal[1]);
        close((-0.5 * diagonal[0] - 0.5 * diagonal[1]) + 0.5, 0.0);
        close((0.5 * diagonal[0] + 0.5 * diagonal[1]) + 0.5, 1.0);
    }
}

/// GPU-side effect parameters, uploaded as a uniform buffer each frame.
/// Must be 16-byte aligned (160 bytes total = 10 x vec4).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EffectUniforms {
    // vec4 #1
    pub pixelate_size: f32, // 1.0 = off, 2..32 = block size in pixels
    pub rgb_split: f32,     // 0.0 = off, 1..30 = horizontal pixel offset
    pub resolution: [f32; 2],
    // vec4 #2
    pub hue_shift: f32,  // -180..180 degrees
    pub saturation: f32, // -1..1 (0 = no change)
    pub brightness: f32, // -1..1 (0 = no change)
    pub contrast: f32,   // -1..1 (0 = no change)
    // vec4 #3
    pub posterize: f32,  // 0 = off, 2..16 = color levels
    pub invert: f32,     // 0.0 = off, 1.0 = full invert
    pub downsample: f32, // 1.0 = full res, 0.05..1.0 = fraction (lower = blurrier)
    pub time: f32,       // elapsed seconds (for animated noise)
    // vec4 #4 — Analog: grain
    pub grain_intensity: f32, // 0.0 = off, 0.01..0.3
    pub grain_size: f32,      // 1.0 = fine, 2..4 = coarse
    pub grain_algo: f32,      // 0=gaussian, 1=perlin, 2=salt_pepper, 3=blue
    pub color_grain: f32,     // 0=mono, 1=chromatic
    // vec4 #5 — Analog: breathing + vignette
    pub breathe_scale: f32,    // 0.0 = off, 0.0..0.05 (±zoom)
    pub breathe_rotation: f32, // 0.0 = off, 0.0..2.0 (degrees)
    pub breathe_position: f32, // 0.0 = off, 0.0..0.02 (drift)
    pub vignette: f32,         // 0.0 = off, 0.0..1.5
    // vec4 #6 — color drift + luma key
    pub color_drift: f32,   // 0.0 = off, 0.0..0.02 (per-frame random aberration)
    pub key_mode: f32,      // 0=off, 1=keep bright, 2=keep dark, 3=remove color, 4=keep color
    pub key_threshold: f32, // 0..1 luminance cut point
    pub key_softness: f32,  // 0..0.5 smoothstep half-width around threshold
    // vec4 #7 — animated cellular / Worley domain warp
    pub cellular_amount: f32, // 0.0 = off, 0.0..1.0 blend and ridge strength
    pub cellular_scale: f32,  // 2.0..32.0 cells across the frame height
    pub cellular_warp: f32,   // 0.0..1.0 bounded displacement within a cell
    pub cellular_speed: f32,  // 0.0..2.0 deterministic target epochs per second
    // vec4 #8 — cellular ridge-to-alpha key
    pub cellular_gap_amount: f32, // 0.0 = opaque/back-compatible, 1.0 = fully key ridges
    pub cellular_gap_threshold: f32, // 0.0..1.0 ridge strength where the key opens
    pub cellular_gap_softness: f32, // 0.0..0.5 feather around the threshold
    /// Public deterministic pattern seed. Zero preserves the legacy shader
    /// sequence byte-for-byte; this replaces an unused pad at the same offset.
    pub random_seed: u32,
    // vec4 #9 — chroma key target in display/sRGB coordinates
    pub key_color: [f32; 3], // 0.0..1.0, default green-screen green
    pub key_tolerance: f32,  // 0.0..1.0 chroma-plane distance
    // vec4 #10 — deterministic horizontal block displacement
    pub shift_amount: f32, // 0.0 = exact legacy path, 1.0 = full displacement
    pub shift_block_size: f32, // 2.0..256.0 screen pixels per horizontal band
    pub shift_density: f32, // 0.0..1.0 fraction of bands displaced per epoch
    pub shift_speed: f32,  // 0.0..20.0 deterministic epochs per second
}

impl Default for EffectUniforms {
    fn default() -> Self {
        Self {
            pixelate_size: 1.0,
            rgb_split: 0.0,
            resolution: [1280.0, 720.0],
            hue_shift: 0.0,
            saturation: 0.0,
            brightness: 0.0,
            contrast: 0.0,
            posterize: 0.0,
            invert: 0.0,
            downsample: 1.0,
            time: 0.0,
            grain_intensity: 0.0,
            grain_size: 1.0,
            grain_algo: 0.0,
            color_grain: 0.0,
            breathe_scale: 0.0,
            breathe_rotation: 0.0,
            breathe_position: 0.0,
            vignette: 0.0,
            color_drift: 0.0,
            key_mode: 0.0,
            key_threshold: 0.5,
            key_softness: 0.1,
            cellular_amount: 0.0,
            cellular_scale: 10.0,
            cellular_warp: 0.35,
            cellular_speed: 0.25,
            cellular_gap_amount: 0.0,
            cellular_gap_threshold: 0.65,
            cellular_gap_softness: 0.08,
            random_seed: 0,
            key_color: [0.0, 1.0, 0.0],
            key_tolerance: 0.15,
            shift_amount: 0.0,
            shift_block_size: 8.0,
            shift_density: 0.5,
            shift_speed: 3.0,
        }
    }
}

impl EffectUniforms {
    /// Return a frame-local copy whose pixel and aspect calculations match the
    /// render target that the effects shader writes into.
    ///
    /// Layer textures can have a different size and aspect ratio from the
    /// composite. The shader operates in output UV space, so using the source
    /// texture dimensions here would stretch spatial effects such as the
    /// cellular field. Keeping this as a copy also leaves patch/modulation
    /// state independent from runtime output dimensions.
    pub(crate) fn for_render_target(mut self, width: u32, height: u32) -> Self {
        self.resolution = [width.max(1) as f32, height.max(1) as f32];
        self
    }

    pub fn increase_pixelate(&mut self) {
        let doubled = self.pixelate_size * 2.0;
        self.pixelate_size = if doubled.is_nan() {
            32.0
        } else {
            doubled.clamp(2.0, 32.0)
        };
    }

    pub fn decrease_pixelate(&mut self) {
        self.pixelate_size = (self.pixelate_size / 2.0).max(1.0);
    }

    pub fn increase_rgb_split(&mut self) {
        self.rgb_split = (self.rgb_split + 5.0).min(30.0);
    }

    pub fn decrease_rgb_split(&mut self) {
        self.rgb_split = (self.rgb_split - 5.0).max(0.0);
    }

    pub fn reset(&mut self) {
        let res = self.resolution;
        *self = Self::default();
        self.resolution = res;
    }

    /// Reset every alpha-key control without disturbing color/spatial effects.
    #[allow(dead_code)]
    pub fn reset_key(&mut self) {
        let defaults = Self::default();
        self.key_mode = defaults.key_mode;
        self.key_threshold = defaults.key_threshold;
        self.key_softness = defaults.key_softness;
        self.key_color = defaults.key_color;
        self.key_tolerance = defaults.key_tolerance;
        self.cellular_gap_amount = defaults.cellular_gap_amount;
        self.cellular_gap_threshold = defaults.cellular_gap_threshold;
        self.cellular_gap_softness = defaults.cellular_gap_softness;
    }
}

#[cfg(test)]
mod uniform_tests {
    use super::*;

    #[test]
    fn effect_uniform_layout_is_ten_vec4s() {
        assert_eq!(std::mem::size_of::<EffectUniforms>(), 160);
        assert!(std::mem::size_of::<EffectUniforms>().is_multiple_of(16));
        assert_eq!(std::mem::offset_of!(EffectUniforms, cellular_amount), 96);
        assert_eq!(std::mem::offset_of!(EffectUniforms, cellular_scale), 100);
        assert_eq!(std::mem::offset_of!(EffectUniforms, cellular_warp), 104);
        assert_eq!(std::mem::offset_of!(EffectUniforms, cellular_speed), 108);
        assert_eq!(
            std::mem::offset_of!(EffectUniforms, cellular_gap_amount),
            112
        );
        assert_eq!(
            std::mem::offset_of!(EffectUniforms, cellular_gap_threshold),
            116
        );
        assert_eq!(
            std::mem::offset_of!(EffectUniforms, cellular_gap_softness),
            120
        );
        assert_eq!(std::mem::offset_of!(EffectUniforms, random_seed), 124);
        assert_eq!(std::mem::offset_of!(EffectUniforms, key_color), 128);
        assert_eq!(std::mem::offset_of!(EffectUniforms, key_tolerance), 140);
        assert_eq!(std::mem::offset_of!(EffectUniforms, shift_amount), 144);
        assert_eq!(std::mem::offset_of!(EffectUniforms, shift_block_size), 148);
        assert_eq!(std::mem::offset_of!(EffectUniforms, shift_density), 152);
        assert_eq!(std::mem::offset_of!(EffectUniforms, shift_speed), 156);
        assert_eq!(EffectUniforms::default().random_seed, 0);

        let shader = include_str!("../shaders/effects.wgsl");
        let amount = shader.find("cellular_gap_amount: f32").unwrap();
        let threshold = shader.find("cellular_gap_threshold: f32").unwrap();
        let softness = shader.find("cellular_gap_softness: f32").unwrap();
        let seed = shader.find("random_seed: u32").unwrap();
        let color = shader.find("key_color: vec3f").unwrap();
        let tolerance = shader.find("key_tolerance: f32").unwrap();
        let shift_amount = shader.find("shift_amount: f32").unwrap();
        let shift_block_size = shader.find("shift_block_size: f32").unwrap();
        let shift_density = shader.find("shift_density: f32").unwrap();
        let shift_speed = shader.find("shift_speed: f32").unwrap();
        assert!(
            amount < threshold
                && threshold < softness
                && softness < seed
                && seed < color
                && color < tolerance
                && tolerance < shift_amount
                && shift_amount < shift_block_size
                && shift_block_size < shift_density
                && shift_density < shift_speed
        );
        assert!(shader.contains("if uniforms.shift_amount > 0.0001"));
        assert!(shader.contains("pattern_seed_offset()"));
        assert!(shader.contains("if uniforms.random_seed == 0u"));
        assert!(shader.contains("if uniforms.random_seed != 0u"));
    }

    #[test]
    fn cellular_defaults_are_bounded_and_disabled() {
        let defaults = EffectUniforms::default();
        assert_eq!(defaults.cellular_amount, 0.0);
        assert_eq!(defaults.cellular_scale, 10.0);
        assert_eq!(defaults.cellular_warp, 0.35);
        assert_eq!(defaults.cellular_speed, 0.25);
        assert_eq!(defaults.cellular_gap_amount, 0.0);
        assert_eq!(defaults.cellular_gap_threshold, 0.65);
        assert_eq!(defaults.cellular_gap_softness, 0.08);
        assert_eq!(defaults.key_color, [0.0, 1.0, 0.0]);
        assert_eq!(defaults.key_tolerance, 0.15);
        assert_eq!(defaults.shift_amount, 0.0);
        assert_eq!(defaults.shift_block_size, 8.0);
        assert_eq!(defaults.shift_density, 0.5);
        assert_eq!(defaults.shift_speed, 3.0);

        let mut changed = defaults;
        changed.cellular_amount = 1.0;
        changed.cellular_scale = 32.0;
        changed.cellular_warp = 1.0;
        changed.cellular_speed = 2.0;
        changed.cellular_gap_amount = 1.0;
        changed.cellular_gap_threshold = 0.0;
        changed.cellular_gap_softness = 0.5;
        changed.key_color = [1.0, 0.0, 1.0];
        changed.key_tolerance = 0.9;
        changed.shift_amount = 1.0;
        changed.shift_block_size = 256.0;
        changed.shift_density = 1.0;
        changed.shift_speed = 20.0;
        changed.reset();
        assert_eq!(changed.cellular_amount, 0.0);
        assert_eq!(changed.cellular_scale, 10.0);
        assert_eq!(changed.cellular_warp, 0.35);
        assert_eq!(changed.cellular_speed, 0.25);
        assert_eq!(changed.cellular_gap_amount, 0.0);
        assert_eq!(changed.cellular_gap_threshold, 0.65);
        assert_eq!(changed.cellular_gap_softness, 0.08);
        assert_eq!(changed.key_color, [0.0, 1.0, 0.0]);
        assert_eq!(changed.key_tolerance, 0.15);
        assert_eq!(changed.shift_amount, 0.0);
        assert_eq!(changed.shift_block_size, 8.0);
        assert_eq!(changed.shift_density, 0.5);
        assert_eq!(changed.shift_speed, 3.0);
    }

    #[test]
    fn key_reset_preserves_non_key_effects_and_resets_all_alpha_masks() {
        let mut effects = EffectUniforms {
            brightness: 0.4,
            key_mode: 4.0,
            key_threshold: 0.8,
            key_softness: 0.3,
            key_color: [1.0, 0.2, 0.5],
            key_tolerance: 0.6,
            cellular_gap_amount: 1.0,
            cellular_gap_threshold: 0.2,
            cellular_gap_softness: 0.4,
            ..Default::default()
        };
        effects.reset_key();
        assert_eq!(effects.brightness, 0.4);
        assert_eq!(effects.key_mode, 0.0);
        assert_eq!(effects.key_threshold, 0.5);
        assert_eq!(effects.key_softness, 0.1);
        assert_eq!(effects.key_color, [0.0, 1.0, 0.0]);
        assert_eq!(effects.key_tolerance, 0.15);
        assert_eq!(effects.cellular_gap_amount, 0.0);
        assert_eq!(effects.cellular_gap_threshold, 0.65);
        assert_eq!(effects.cellular_gap_softness, 0.08);
    }

    #[test]
    fn render_target_resolution_controls_spatial_effect_aspect_without_mutating_base() {
        let base = EffectUniforms {
            resolution: [640.0, 480.0],
            cellular_amount: 0.8,
            cellular_scale: 18.0,
            ..Default::default()
        };

        let frame = base.for_render_target(1920, 1080);

        assert_eq!(frame.resolution, [1920.0, 1080.0]);
        assert_eq!(frame.cellular_amount, base.cellular_amount);
        assert_eq!(frame.cellular_scale, base.cellular_scale);
        assert_eq!(base.resolution, [640.0, 480.0]);

        // Defensive normalization keeps malformed/transitionary target sizes
        // from introducing division by zero into the shader.
        assert_eq!(base.for_render_target(0, 0).resolution, [1.0, 1.0]);
    }
}
