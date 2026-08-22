/// The rate at which the temporal controls were authored. Feedback retention,
/// zoom, rotation, and the history ring are all expressed against this
/// reference so a render at 24, 30, or 60 fps describes the same motion.
pub const TEMPORAL_REFERENCE_FPS: f32 = 30.0;

#[allow(
    unused_imports,
    reason = "T0 exposes the frozen authoring vocabulary before T1 consumers land"
)]
pub use crate::temporal::{
    CollisionAtlasParams, CollisionScoreParams, CollisionScoreTrigger, LongExposureParams,
    RefreshGardenGate, RefreshGardenParams, TemporalInterpolation, TemporalLoomParams,
    TemporalOriginalsParams, TemporalTopology, TimeDisplaceMap,
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
    /// B12: how slit-scan turns image position into a history age. `Ramp` is
    /// the exact existing angle path; any other map (or `slit_interp`)
    /// selects the bounded additive originals pipeline.
    pub slit_map: TimeDisplaceMap,
    /// B12: linear interpolation between adjacent ring layers. Off is the
    /// exact banded prior path (floor law); on costs at most one extra
    /// history load per pixel.
    pub slit_interp: bool,
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
    /// B4 display physics: the field domain, phosphor persistence, and the
    /// display model, seated after the temporal pass and before the opaque
    /// resolve. Exact-off by default; deliberately absent from every
    /// temporal activity/shader-selection predicate, because the stage owns
    /// its own pass and its own shader.
    pub display: crate::display_physics::DisplayPhysicsParams,
    /// B8 melting edge: the master melt over the program's own coverage,
    /// seated on the same slot-0 seam immediately before the display stage.
    /// Exact-off by default; like `display` it is deliberately absent from
    /// every temporal activity/shader-selection predicate, because the
    /// stage owns its own pass and its own shader.
    pub melt: crate::mixing_boundary::MeltParams,
    /// B5 codec mosh: the real encode→break→decode round trip over the
    /// finished audience image, a CPU stage downstream of every GPU pass.
    /// Exact bypass by default; like `display` and `melt` it is deliberately
    /// absent from every temporal activity/shader-selection predicate,
    /// because the stage owns no shader at all.
    pub mosh: crate::codec_mosh::CodecMoshParams,
    /// B14 sync latch: the tape/NTSC horizontal shear on the same slot-0
    /// seam, between the melting edge and the display stage. Exact-off by
    /// default; like `display`, `melt`, and `mosh` it is deliberately
    /// absent from every temporal activity/shader-selection predicate,
    /// because the stage owns its own pass and its own shader.
    pub sync: crate::sync_latch::SyncLatchParams,
    /// B3 feedback rig. Identity by default, which is the exact prior
    /// feedback path: the shader takes the historical expression untouched.
    pub rig: FeedbackRigParams,
}

/// The B3 waveshaper vocabulary applied to the fed-back sample.
/// Permanent append-only codes; `Clamp` is the default and, at drive 1 and
/// pivot 0.5, the exact identity on in-range values.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FeedbackShape {
    #[default]
    Clamp,
    Soft,
    Wrap,
    Fold,
}

impl FeedbackShape {
    /// Permanent append-only shader code. Never renumber an existing entry.
    pub const fn code(self) -> u32 {
        match self {
            Self::Clamp => 0,
            Self::Soft => 1,
            Self::Wrap => 2,
            Self::Fold => 3,
        }
    }

    #[allow(
        dead_code,
        reason = "the closed-vocabulary fixtures iterate the frozen code table"
    )]
    pub const ALL: [Self; 4] = [Self::Clamp, Self::Soft, Self::Wrap, Self::Fold];
}

/// The complete B3 feedback rig: everything the loop does to the fed-back
/// sample beyond the frozen zoom/rotate/retention trio. Identity is the exact
/// prior path — the shader's rig-active flag is false and the historical
/// feedback expression runs untouched, byte for byte.
///
/// Rate law: rate-like controls (offset, hue rotate, chroma displace, blur,
/// sharpen, noise) scale linearly per 1/30-second reference tick;
/// multiplicative controls (saturation, per-channel gain) exponentiate, the
/// `feedback`/`fb_zoom` law; the nonlinear stage (shape/drive/threshold) and
/// the servo mix toward identity by the clamped tick fraction, exact at the
/// 30 Hz reference and rate-independent to first order elsewhere.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeedbackRigParams {
    /// Fed-back image offset per reference tick, output UV.
    pub offset_x: f32,
    pub offset_y: f32,
    /// Discrete reflections about the frame centre — the regime no rotation
    /// can reach.
    pub reflect_x: bool,
    pub reflect_y: bool,
    /// In-loop hue rotation, degrees per reference tick.
    pub hue_rotate: f32,
    /// In-loop saturation pull per reference tick; 1 is identity.
    pub saturation: f32,
    /// Per-channel loop gain per reference tick; 1 is identity. Above 1 the
    /// loop can exceed unity and run away — that is what the servo is for.
    pub gain_r: f32,
    pub gain_g: f32,
    pub gain_b: f32,
    /// Chromatic displacement of the fed-back lookup, UV per reference tick.
    pub chroma_displace: f32,
    /// Cross-blur of the fed-back sample; with `sharpen` this is the
    /// activator-inhibitor pair.
    pub blur: f32,
    /// Unsharp gain over the same cross taps.
    pub sharpen: f32,
    pub shape: FeedbackShape,
    /// Waveshaper drive about the pivot; 1 is identity.
    pub drive: f32,
    pub pivot: f32,
    /// Fed-back luma below this level decays out of the loop; 0 is exact off.
    pub threshold: f32,
    /// Deterministic loop noise per reference tick.
    pub noise: f32,
    /// Boundary law for the fed-back lookup. `Transparent` is the exact
    /// historical inside test; the numbering is the frozen program-wide
    /// boundary table.
    pub edge: crate::motion::MotionBoundaryMode,
    /// Engage the per-pixel compressive auto-level. Deliberately deterministic
    /// (no measured-mean loop): a readback-driven servo would give live and
    /// export different dynamics, which the export contract forbids.
    pub servo: bool,
    /// B14's philosophy: defeated, the loop may run to white or black and
    /// stay there. Defeat wins over engage.
    pub servo_defeated: bool,
}

impl Default for FeedbackRigParams {
    fn default() -> Self {
        Self {
            offset_x: 0.0,
            offset_y: 0.0,
            reflect_x: false,
            reflect_y: false,
            hue_rotate: 0.0,
            saturation: 1.0,
            gain_r: 1.0,
            gain_g: 1.0,
            gain_b: 1.0,
            chroma_displace: 0.0,
            blur: 0.0,
            sharpen: 0.0,
            shape: FeedbackShape::Clamp,
            drive: 1.0,
            pivot: 0.5,
            threshold: 0.0,
            noise: 0.0,
            edge: crate::motion::MotionBoundaryMode::Transparent,
            servo: false,
            servo_defeated: false,
        }
    }
}

impl FeedbackRigParams {
    /// Clamp continuous values and replace non-finite input with the field's
    /// neutral value, never a clamped extreme.
    pub fn sanitized(self) -> Self {
        let unit = |value: f32, neutral: f32| finite_or(value, neutral).clamp(0.0, 1.0);
        Self {
            offset_x: finite_or(self.offset_x, 0.0).clamp(-0.5, 0.5),
            offset_y: finite_or(self.offset_y, 0.0).clamp(-0.5, 0.5),
            reflect_x: self.reflect_x,
            reflect_y: self.reflect_y,
            hue_rotate: finite_or(self.hue_rotate, 0.0).clamp(-180.0, 180.0),
            saturation: finite_or(self.saturation, 1.0).clamp(0.0, 2.0),
            gain_r: finite_or(self.gain_r, 1.0).clamp(0.0, 2.0),
            gain_g: finite_or(self.gain_g, 1.0).clamp(0.0, 2.0),
            gain_b: finite_or(self.gain_b, 1.0).clamp(0.0, 2.0),
            chroma_displace: finite_or(self.chroma_displace, 0.0).clamp(0.0, 0.05),
            blur: unit(self.blur, 0.0),
            sharpen: finite_or(self.sharpen, 0.0).clamp(0.0, 2.0),
            shape: self.shape,
            drive: finite_or(self.drive, 1.0).clamp(0.25, 4.0),
            pivot: unit(self.pivot, 0.5),
            threshold: unit(self.threshold, 0.0),
            noise: unit(self.noise, 0.0),
            edge: self.edge,
            servo: self.servo,
            servo_defeated: self.servo_defeated,
        }
    }

    /// True when the rig is the exact prior feedback path.
    pub fn is_identity(self) -> bool {
        self.sanitized() == Self::default()
    }

    /// The rate law: convert 30 Hz-authored values into one render step.
    /// Rate-like terms scale linearly; multiplicative terms exponentiate. The
    /// nonlinear stage keeps its authored values — the shader mixes it toward
    /// identity by the clamped tick fraction carried beside these params.
    pub(crate) fn for_frame_scale(self, frame_scale: f32) -> Self {
        let sanitized = self.sanitized();
        let power = |value: f32| finite_or(value.powf(frame_scale), 1.0).clamp(0.0, 4.0);
        Self {
            offset_x: sanitized.offset_x * frame_scale,
            offset_y: sanitized.offset_y * frame_scale,
            hue_rotate: sanitized.hue_rotate * frame_scale,
            saturation: power(sanitized.saturation),
            gain_r: power(sanitized.gain_r),
            gain_g: power(sanitized.gain_g),
            gain_b: power(sanitized.gain_b),
            chroma_displace: sanitized.chroma_displace * frame_scale,
            blur: sanitized.blur * frame_scale,
            sharpen: sanitized.sharpen * frame_scale,
            noise: sanitized.noise * frame_scale,
            ..sanitized
        }
    }
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
            slit_map: TimeDisplaceMap::Ramp,
            slit_interp: false,
            key_mode: 0.0,
            key_threshold: 0.1,
            key_softness: 0.03,
            key_history: 1.0,
            originals: TemporalOriginalsParams::default(),
            display: crate::display_physics::DisplayPhysicsParams::default(),
            melt: crate::mixing_boundary::MeltParams::default(),
            mosh: crate::codec_mosh::CodecMoshParams::default(),
            sync: crate::sync_latch::SyncLatchParams::default(),
            rig: FeedbackRigParams::default(),
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
            slit_map: self.slit_map,
            slit_interp: self.slit_interp,
            key_mode: finite_or(self.key_mode, 0.0).round().clamp(0.0, 4.0),
            key_threshold: finite_or(self.key_threshold, 0.1).clamp(0.0, 1.0),
            key_softness: finite_or(self.key_softness, 0.03).clamp(0.0, 0.5),
            key_history: finite_or(self.key_history, 1.0).round().clamp(1.0, 23.0),
            originals: self.originals.sanitized(),
            // The display stage owns its own pass and applies its own rate
            // law there (fractional-tick decay in the store); its authored
            // values pass through this per-frame conversion sanitized only.
            display: self.display.sanitized(),
            // The melting edge likewise owns its own pass; its store clock
            // applies the reference-tick law in the stage itself.
            melt: self.melt.sanitized(),
            // The sync latch likewise owns its own pass; its fault clock
            // applies the reference-tick law in the stage itself.
            sync: self.sync.sanitized(),
            // The mosh is a CPU stage with its own reference-frame fault
            // clock; its authored values pass through sanitized only.
            mosh: self.mosh.sanitized(),
            rig: self.rig.for_frame_scale(frame_scale),
        }
    }

    /// True when the B12 time-displace state departs from the exact legacy
    /// slit-scan path: an authored non-`Ramp` map or the interpolation
    /// toggle, under an active slit-scan. This selects the bounded additive
    /// originals pipeline; the frozen legacy shader keeps the Ramp/floor
    /// path untouched.
    pub(crate) fn time_displace_active(&self) -> bool {
        self.slitscan > 0.0 && (self.slit_map != TimeDisplaceMap::Ramp || self.slit_interp)
    }

    /// The clamped tick fraction the shader uses to mix the rig's nonlinear
    /// stage toward identity. Exactly one at the 30 Hz reference; capped at
    /// one below it so a slow display cannot extrapolate the shaper.
    pub(crate) fn rig_tick_mix(delta_seconds: f32) -> f32 {
        let reference_delta = 1.0 / TEMPORAL_REFERENCE_FPS;
        let delta = if delta_seconds.is_finite() {
            delta_seconds.max(0.0)
        } else {
            reference_delta
        };
        (delta * TEMPORAL_REFERENCE_FPS).clamp(0.0, 1.0)
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
            rig: FeedbackRigParams {
                offset_x: 0.2,
                offset_y: -0.1,
                hue_rotate: 24.0,
                saturation: 1.44,
                gain_r: 1.21,
                gain_g: 0.81,
                gain_b: 1.69,
                chroma_displace: 0.02,
                blur: 0.4,
                sharpen: 0.8,
                drive: 2.0,
                pivot: 0.3,
                threshold: 0.25,
                noise: 0.5,
                ..FeedbackRigParams::default()
            },
            slitscan: 0.7,
            slit_angle: 42.0,
            slit_axis: 1.0,
            key_mode: 3.0,
            key_threshold: 0.2,
            key_softness: 0.04,
            key_history: 7.0,
            originals: TemporalOriginalsParams::default(),
            ..TemporalParams::default()
        };
        let half = params.for_frame_delta(1.0 / 60.0);

        close(half.feedback * half.feedback, params.feedback);
        close(half.fb_zoom * half.fb_zoom, params.fb_zoom);
        close(half.fb_rotate + half.fb_rotate, params.fb_rotate);
        // The rig obeys the same two laws: rate-like terms halve, and
        // multiplicative terms take the square root.
        close(half.rig.offset_x + half.rig.offset_x, params.rig.offset_x);
        close(
            half.rig.hue_rotate + half.rig.hue_rotate,
            params.rig.hue_rotate,
        );
        close(
            half.rig.chroma_displace + half.rig.chroma_displace,
            params.rig.chroma_displace,
        );
        close(half.rig.blur + half.rig.blur, params.rig.blur);
        close(half.rig.sharpen + half.rig.sharpen, params.rig.sharpen);
        close(half.rig.noise + half.rig.noise, params.rig.noise);
        close(
            half.rig.saturation * half.rig.saturation,
            params.rig.saturation,
        );
        close(half.rig.gain_r * half.rig.gain_r, params.rig.gain_r);
        close(half.rig.gain_g * half.rig.gain_g, params.rig.gain_g);
        close(half.rig.gain_b * half.rig.gain_b, params.rig.gain_b);
        // The nonlinear stage keeps its authored values; the tick fraction the
        // shader mixes by is computed beside them.
        close(half.rig.drive, params.rig.drive);
        close(half.rig.pivot, params.rig.pivot);
        close(half.rig.threshold, params.rig.threshold);
        close(TemporalParams::rig_tick_mix(1.0 / 60.0), 0.5);
        close(TemporalParams::rig_tick_mix(1.0 / 30.0), 1.0);
        // Below the reference rate the shaper saturates at one tick.
        close(TemporalParams::rig_tick_mix(1.0 / 24.0), 1.0);
        close(half.slitscan, params.slitscan);
        close(half.key_mode, params.key_mode);
        close(half.key_threshold, params.key_threshold);
        close(half.key_softness, params.key_softness);
        close(half.key_history, params.key_history);
    }

    #[test]
    fn time_displace_activity_requires_slitscan_and_a_non_default_path() {
        let mut params = TemporalParams::default();
        assert!(!params.time_displace_active());
        params.slit_map = TimeDisplaceMap::Brightness;
        assert!(
            !params.time_displace_active(),
            "no slit-scan means no displacement to run"
        );
        params.slitscan = 0.4;
        assert!(params.time_displace_active());
        params.slit_map = TimeDisplaceMap::Ramp;
        assert!(
            !params.time_displace_active(),
            "Ramp with the floor law is the exact legacy path"
        );
        params.slit_interp = true;
        assert!(params.time_displace_active());
        // The frame-delta law carries both discrete choices through unchanged.
        let frame = params.for_frame_delta(1.0 / 60.0);
        assert_eq!(frame.slit_map, TimeDisplaceMap::Ramp);
        assert!(frame.slit_interp);
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
            rig: FeedbackRigParams::default(),
            ..TemporalParams::default()
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
    fn feedback_rig_identity_is_the_exact_default_and_sanitize_is_neutral() {
        assert!(FeedbackRigParams::default().is_identity());
        let hostile = FeedbackRigParams {
            offset_x: f32::NAN,
            saturation: f32::INFINITY,
            gain_g: f32::NEG_INFINITY,
            chroma_displace: 9.0,
            drive: f32::NAN,
            pivot: -3.0,
            ..FeedbackRigParams::default()
        }
        .sanitized();
        // Non-finite input takes the field's neutral value, never a clamped
        // extreme; finite input clamps.
        assert_eq!(hostile.offset_x, 0.0);
        assert_eq!(hostile.saturation, 1.0);
        assert_eq!(hostile.gain_g, 1.0);
        assert_eq!(hostile.chroma_displace, 0.05);
        assert_eq!(hostile.drive, 1.0);
        assert_eq!(hostile.pivot, 0.0);
        // A reflection alone is emphatically not identity: it is the regime
        // no rotation can reach.
        assert!(!FeedbackRigParams {
            reflect_x: true,
            ..FeedbackRigParams::default()
        }
        .is_identity());
        assert!(!FeedbackRigParams {
            servo: true,
            ..FeedbackRigParams::default()
        }
        .is_identity());
        // Shape codes are permanent and append-only.
        let codes: Vec<u32> = FeedbackShape::ALL
            .iter()
            .map(|shape| shape.code())
            .collect();
        assert_eq!(codes, [0, 1, 2, 3]);
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
/// Must be 16-byte aligned (288 bytes total = 18 x vec4: the ten legacy
/// slots plus the eight B13 small-effects slots).
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
    // B13 small-effects tranche. Every amount's zero default takes an exact
    // no-op shader branch, preserving legacy pixels byte for byte. The laws
    // are derived from BENDR (MIT, © 2026 Steve Blythe); every one is a
    // rewrite in linear light.
    // vec4 #11 — isolines between brightness bands
    pub contour: f32,       // 0.0 = off, 0.0..1.0 isoline strength
    pub contour_bands: f32, // 2.0..40.0 luma bands across the tonal range
    pub contour_width: f32, // 0.2..6.0 isoline width in pixels
    pub contour_hue: f32,   // 0.0..1.0 palette phase; near zero = white lines
    // vec4 #12 — keep-fill plus luma flattening
    pub contour_fill: f32,   // 0.0..1.0 surviving fill between isolines
    pub flatten: f32,        // 0.0 = off, 0.0..1.0 quantize luma to solid fields
    pub flatten_levels: f32, // 2.0..16.0 luma steps
    pub contour_dither: f32, // 0.0..1.0 ordered Bayer dither on the flatten
    // vec4 #13 — solarize plus the negative family
    pub solarize: f32,      // 0.0 = off, 0.0..1.0 fold-back exposure
    pub negative: f32,      // 0.0 = off, 0.0..1.0 inversion amount
    pub negative_mode: f32, // 0=rgb, 1=luma-only, 2=hue-flip (permanent codes)
    pub colourpass: f32,    // 0.0 = off, 0.0..1.0 hue-window isolation
    // vec4 #14 — colourpass window plus find-edge
    pub colourpass_hue: f32,   // -180..180 degrees, surviving hue centre
    pub colourpass_width: f32, // 0.0..1.0 window width
    pub edge_amount: f32,      // 0.0 = off, 0.0..1.0 Sobel outline
    pub edge_hue: f32,         // -180..180 degrees, outline hue
    // vec4 #15 — emboss plus halftone
    pub emboss: f32,         // 0.0 = off, 0.0..1.0 directional relief
    pub emboss_angle: f32,   // -180..180 degrees light direction
    pub halftone: f32,       // 0.0 = off, 0.0..1.0 dot-screen dropout
    pub halftone_pitch: f32, // 0.0..1.0 dot pitch (0 coarse, 1 fine)
    // vec4 #16 — halftone screen angle, moiré, row smear
    pub halftone_angle: f32, // -180..180 degrees screen angle
    pub moire: f32,          // 0.0 = off, 0.0..1.0 interference strength
    pub moire_freq: f32,     // 0.0..1.0 virtual grid frequency
    pub row_smear: f32,      // 0.0 = off, 0.0..1.0 wrong-predictor row shear
    // vec4 #17 — ordered-dither crush plus the dumb tile
    pub bitcrush: f32,        // 0.0 = off, 0.0..1.0 mono quantize amount
    pub bitcrush_levels: f32, // 2.0..16.0 luma levels (2 = 1-bit)
    pub bitcrush_dither: f32, // 0.0..1.0 ordered Bayer dither amount
    pub multi_grid_x: f32,    // 1.0..8.0 tiles across; 1 = exact off
    // vec4 #18 — the tile's Y count plus the master-only optics
    pub multi_grid_y: f32,      // 1.0..8.0 tiles down; 1 = exact off
    pub barrel: f32,            // -1.0..1.0 radial distortion (master only)
    pub chroma_aberration: f32, // 0.0..1.0 radial fringe (master only)
    pub anamorphic_streak: f32, // 0.0..1.0 horizontal flare (master only)
    // vec4 #19 — B8 key dressing: border and shadow join the key signal
    pub key_border: f32,         // 0.0 = off, 0.0..1.0 matte-grow outline amount
    pub key_border_color: f32,   // 0..7 closed back-colour table (permanent codes)
    pub key_shadow: f32,         // 0.0 = off, 0.0..1.0 offset darkened copy
    pub key_dress_reserved: f32, // alignment slot, exact zero
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
            contour: 0.0,
            contour_bands: 10.0,
            contour_width: 1.2,
            contour_hue: 0.0,
            contour_fill: 0.25,
            flatten: 0.0,
            flatten_levels: 5.0,
            contour_dither: 0.0,
            solarize: 0.0,
            negative: 0.0,
            negative_mode: 0.0,
            colourpass: 0.0,
            colourpass_hue: 0.0,
            colourpass_width: 0.25,
            edge_amount: 0.0,
            edge_hue: 0.0,
            emboss: 0.0,
            emboss_angle: 45.0,
            halftone: 0.0,
            halftone_pitch: 0.4,
            halftone_angle: 0.0,
            moire: 0.0,
            moire_freq: 0.4,
            row_smear: 0.0,
            bitcrush: 0.0,
            bitcrush_levels: 2.0,
            bitcrush_dither: 1.0,
            multi_grid_x: 1.0,
            multi_grid_y: 1.0,
            barrel: 0.0,
            chroma_aberration: 0.0,
            anamorphic_streak: 0.0,
            key_border: 0.0,
            key_border_color: 0.0,
            key_shadow: 0.0,
            key_dress_reserved: 0.0,
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

    /// The three B13 optics are master-scope only. Every seam that authors
    /// layer effect state calls this after applying, so a hostile patch or a
    /// legacy wire client cannot install an optic on a layer copy.
    pub fn clear_master_only_effects(&mut self) {
        let defaults = Self::default();
        self.barrel = defaults.barrel;
        self.chroma_aberration = defaults.chroma_aberration;
        self.anamorphic_streak = defaults.anamorphic_streak;
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
        self.key_border = defaults.key_border;
        self.key_border_color = defaults.key_border_color;
        self.key_shadow = defaults.key_shadow;
        self.cellular_gap_amount = defaults.cellular_gap_amount;
        self.cellular_gap_threshold = defaults.cellular_gap_threshold;
        self.cellular_gap_softness = defaults.cellular_gap_softness;
    }
}

#[cfg(test)]
mod uniform_tests {
    use super::*;

    #[test]
    fn effect_uniform_layout_is_nineteen_vec4s() {
        assert_eq!(std::mem::size_of::<EffectUniforms>(), 304);
        assert_eq!(std::mem::offset_of!(EffectUniforms, key_border), 288);
        assert_eq!(std::mem::offset_of!(EffectUniforms, key_shadow), 296);
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
        // B13 small-effects lanes: eight appended vec4s at offsets 160..288.
        assert_eq!(std::mem::offset_of!(EffectUniforms, contour), 160);
        assert_eq!(std::mem::offset_of!(EffectUniforms, contour_fill), 176);
        assert_eq!(std::mem::offset_of!(EffectUniforms, solarize), 192);
        assert_eq!(std::mem::offset_of!(EffectUniforms, colourpass_hue), 208);
        assert_eq!(std::mem::offset_of!(EffectUniforms, emboss), 224);
        assert_eq!(std::mem::offset_of!(EffectUniforms, halftone_angle), 240);
        assert_eq!(std::mem::offset_of!(EffectUniforms, bitcrush), 256);
        assert_eq!(std::mem::offset_of!(EffectUniforms, multi_grid_y), 272);
        assert_eq!(std::mem::offset_of!(EffectUniforms, anamorphic_streak), 284);
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

        // B13: the eight appended vec4s sit between the ten legacy slots and
        // the four spatial slots, in the exact CPU field order.
        let b13_first = shader.find("contour: f32").unwrap();
        let b13_last = shader.find("anamorphic_streak: f32").unwrap();
        let spatial_first = shader.find("spatial_inverse_row_0: vec4f").unwrap();
        assert!(shift_speed < b13_first && b13_first < b13_last && b13_last < spatial_first);
    }

    /// The B13 golden-branch law: every small effect is gated on its own
    /// authored amount, so a default patch never enters a new shader branch
    /// and the prior image is byte-exact. The three optics are additionally
    /// pinned as master-only through the clearing helper.
    #[test]
    fn small_effects_are_amount_gated_and_optics_are_master_only() {
        let shader = include_str!("../shaders/effects.wgsl");
        for gate in [
            "if uniforms.contour > 0.0001",
            "if uniforms.flatten > 0.0001",
            "if uniforms.solarize > 0.0001",
            "if uniforms.negative > 0.0001",
            "if uniforms.colourpass > 0.0001",
            "if uniforms.edge_amount > 0.0001",
            "if uniforms.emboss > 0.0001",
            "if uniforms.halftone > 0.0001",
            "if uniforms.moire > 0.0001",
            "if uniforms.row_smear > 0.0001",
            "if uniforms.bitcrush > 0.0001",
            "if uniforms.multi_grid_x >= 1.5 || uniforms.multi_grid_y >= 1.5",
            "if abs(uniforms.barrel) > 0.0001",
            "if uniforms.chroma_aberration > 0.0001",
            "if uniforms.anamorphic_streak > 0.0001",
        ] {
            assert!(shader.contains(gate), "missing amount gate: {gate}");
        }
        // Secondary controls (dither) stay inside their owner's gate:
        // contour_dither is consulted only under flatten or bitcrush.
        assert!(!shader.contains("if uniforms.contour_dither > 0.0001\n    {"));

        // Defaults are exact off, and reset restores every B13 value.
        let defaults = EffectUniforms::default();
        for (name, value, expected) in [
            ("contour", defaults.contour, 0.0),
            ("flatten", defaults.flatten, 0.0),
            ("solarize", defaults.solarize, 0.0),
            ("negative", defaults.negative, 0.0),
            ("negative_mode", defaults.negative_mode, 0.0),
            ("colourpass", defaults.colourpass, 0.0),
            ("edge_amount", defaults.edge_amount, 0.0),
            ("emboss", defaults.emboss, 0.0),
            ("halftone", defaults.halftone, 0.0),
            ("moire", defaults.moire, 0.0),
            ("row_smear", defaults.row_smear, 0.0),
            ("bitcrush", defaults.bitcrush, 0.0),
            ("multi_grid_x", defaults.multi_grid_x, 1.0),
            ("multi_grid_y", defaults.multi_grid_y, 1.0),
            ("barrel", defaults.barrel, 0.0),
            ("chroma_aberration", defaults.chroma_aberration, 0.0),
            ("anamorphic_streak", defaults.anamorphic_streak, 0.0),
        ] {
            assert_eq!(value, expected, "{name} default must be exact off");
        }
        let mut changed = defaults;
        changed.contour = 1.0;
        changed.bitcrush = 0.7;
        changed.multi_grid_x = 4.0;
        changed.barrel = -0.5;
        changed.reset();
        assert_eq!(changed.contour, 0.0);
        assert_eq!(changed.bitcrush, 0.0);
        assert_eq!(changed.multi_grid_x, 1.0);
        assert_eq!(changed.barrel, 0.0);

        // The master-only clear removes exactly the three optics and leaves
        // every shared control untouched.
        let mut layered = EffectUniforms {
            contour: 0.8,
            halftone: 0.6,
            barrel: 0.9,
            chroma_aberration: 0.5,
            anamorphic_streak: 0.4,
            ..EffectUniforms::default()
        };
        layered.clear_master_only_effects();
        assert_eq!(layered.contour, 0.8);
        assert_eq!(layered.halftone, 0.6);
        assert_eq!(layered.barrel, 0.0);
        assert_eq!(layered.chroma_aberration, 0.0);
        assert_eq!(layered.anamorphic_streak, 0.0);
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
