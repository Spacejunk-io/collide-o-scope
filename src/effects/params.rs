/// The rate at which the temporal controls were authored. Feedback retention,
/// zoom, rotation, and the history ring are all expressed against this
/// reference so a render at 24, 30, or 60 fps describes the same motion.
pub const TEMPORAL_REFERENCE_FPS: f32 = 30.0;

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
        }
    }
}

impl TemporalParams {
    pub fn is_active(&self) -> bool {
        self.feedback > 0.0 || self.slitscan > 0.0
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
        };
        let half = params.for_frame_delta(1.0 / 60.0);

        close(half.feedback * half.feedback, params.feedback);
        close(half.fb_zoom * half.fb_zoom, params.fb_zoom);
        close(half.fb_rotate + half.fb_rotate, params.fb_rotate);
        close(half.slitscan, params.slitscan);
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
        };
        let normalized = params.for_frame_delta(f32::NAN);

        assert_eq!(normalized.feedback, 0.0);
        assert_eq!(normalized.fb_zoom, 1.0);
        assert_eq!(normalized.fb_rotate, 0.0);
        assert_eq!(normalized.slitscan, 0.0);
        assert_eq!(normalized.slit_angle, 0.0);
        assert_eq!(normalized.slit_axis, 1.0);
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
/// Must be 16-byte aligned (96 bytes total = 6 × vec4).
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
    pub key_mode: f32,      // 0=off, 1=keep bright, 2=keep dark
    pub key_threshold: f32, // 0..1 luminance cut point
    pub key_softness: f32,  // 0..0.5 smoothstep half-width around threshold
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
        }
    }
}

impl EffectUniforms {
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
}
