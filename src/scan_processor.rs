//! The B1 Scan Processor law — a Rutt/Etra-style drawn raster.
//!
//! A scan processor intercepts a monitor's deflection signals before the yoke
//! and patches video luminance into the vertical position control, so bright
//! parts of the picture physically pull the scan line up the tube; a camera
//! re-shoots the tube. The apparent depth is an artefact of photographing a 2D
//! deflection, never a 3D scene. The part every displacement-map imitation
//! misses is that the result is *drawn* rather than sampled: a stack of
//! continuous glowing lines whose bunching makes bright caustic ridges and
//! whose splaying makes dark gaps. A fragment shader has no notion of line
//! density, so the pass is real instanced geometry, accumulated additively,
//! with the beam getting brighter where it sweeps slower — a slower beam
//! deposits more energy per unit length.
//!
//! The `beam_position` composition order and the beam-energy law
//! (`gain = 2 / speed`) are derived from BENDR (MIT, © 2026 Steve Blythe) and
//! transcribed faithfully with attribution; the surrounding machinery is a
//! rewrite (Rust / wgpu 29 / WGSL, linear light, Rec.709 luma). This module is
//! the independent CPU reference the GPU pass is checked against, in the
//! `gesture.rs` tradition: no wgpu, clock, filesystem, or UI dependency.

use serde::{Deserialize, Serialize};

/// Authored scanline count bounds. The line count is plan-time geometry — it
/// sizes the instanced draw and therefore the vertex ledger — so like the
/// Residual block grid it is an authored integer with no modulatable address.
pub const SCAN_MIN_LINES: u32 = 16;
pub const SCAN_MAX_LINES: u32 = 1_080;
pub const SCAN_DEFAULT_LINES: u32 = 320;

/// Authored samples-per-line bounds, the same plan-time-geometry law.
pub const SCAN_MIN_SAMPLES: u32 = 64;
pub const SCAN_MAX_SAMPLES: u32 = 512;
pub const SCAN_DEFAULT_SAMPLES: u32 = 256;

/// The named vertex budget for the one pass in the tree that owns geometry.
/// Two vertices per sample make the ribbon; one instance per scanline. The
/// cap is structural — the authored maxima admit exactly this many — and the
/// planner still refuses a plan one vertex over rather than trusting the
/// clamps alone.
pub const MAX_SCAN_PROCESSOR_VERTICES: u32 = SCAN_MAX_LINES * SCAN_MAX_SAMPLES * 2;
const _: () = assert!(MAX_SCAN_PROCESSOR_VERTICES == 1_105_920);

/// The beam-energy clamp: gain one at nominal sweep speed, at most eight
/// where the displacement slows the beam, at least 0.05 where it is thrown
/// across the screen. A flat, undisplaced line sweeps two clip units per unit
/// of `sx`, so that is the reference speed.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub const SCAN_GAIN_MIN: f32 = 0.05;
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub const SCAN_GAIN_MAX: f32 = 8.0;
/// Beam speed floor: the tangent of a fully collapsed raster can approach
/// zero length, and the energy law divides by speed.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub const SCAN_SPEED_FLOOR: f32 = 0.02;
/// The luma pivot: deflection is centred on this luminance, so a mid-grey
/// raster sits near its undeflected position rather than riding high.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub const SCAN_LUMA_PIVOT: f32 = 0.35;
/// Full-amount luminance deflection span in clip units.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub const SCAN_DEFLECT_SPAN: f32 = 1.6;

/// The authored Scan Processor node state. Continuous fields are modulatable;
/// `reverse_h`/`reverse_v` are discrete laws; `lines`/`samples_per_line` are
/// plan-time geometry (see above).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ScanProcessorParams {
    /// Scanline count: one drawn ribbon per line.
    pub lines: u32,
    /// Beam samples along each line: two vertices per sample.
    pub samples_per_line: u32,
    /// Luminance-into-vertical-deflection amount. The control the machine is
    /// actually for.
    pub amount: f32,
    /// Drawn ribbon width.
    pub ribbon_width: f32,
    /// Velocity-brightness mix: zero renders flat energy, one renders the
    /// full beam-energy law.
    pub velocity_mix: f32,
    /// Tilt about the horizontal axis, turning the deflection into an
    /// apparent surface. Signed, in radian units as authored.
    pub tilt_x: f32,
    /// Tilt about the vertical axis.
    pub tilt_y: f32,
    /// Perspective strength on the tilted deflection.
    pub perspective: f32,
    /// The continuous-wind yoke bending the whole raster.
    pub s_curve: f32,
    /// Parallelogram skew, the bench-monitor control.
    pub skew: f32,
    /// Raster collapse: remove the vertical deflection current and the frame
    /// smears down onto a single line.
    pub collapse: f32,
    /// Reverse the horizontal sweep. Discrete law.
    pub reverse_h: bool,
    /// Reverse the field. Discrete law.
    pub reverse_v: bool,
    /// Deflection-oscillator amount.
    pub osc_amount: f32,
    /// Deflection-oscillator frequency control. Locked, it quantizes to a
    /// whole multiple of the field rate and the pattern stands still.
    pub osc_freq: f32,
    /// Oscillator lock: one is locked (standing pattern), zero fully detuned
    /// (the crawl that is the instrument's gesture).
    pub osc_lock: f32,
    /// Second-axis oscillator amount, making the wobble a Lissajous figure.
    pub lissajous: f32,
    /// Monochrome mix on the drawn beam.
    pub mono: f32,
    /// Colourise: repaint the beam from a luma-indexed hue sweep.
    pub hue: f32,
}

impl Default for ScanProcessorParams {
    fn default() -> Self {
        Self {
            lines: SCAN_DEFAULT_LINES,
            samples_per_line: SCAN_DEFAULT_SAMPLES,
            amount: 0.0,
            ribbon_width: 0.12,
            velocity_mix: 0.8,
            tilt_x: 0.0,
            tilt_y: 0.0,
            perspective: 0.3,
            s_curve: 0.0,
            skew: 0.0,
            collapse: 0.0,
            reverse_h: false,
            reverse_v: false,
            osc_amount: 0.0,
            osc_freq: 0.25,
            osc_lock: 1.0,
            lissajous: 0.0,
            mono: 0.0,
            hue: 0.0,
        }
    }
}

impl ScanProcessorParams {
    /// Clamp every authored value into its declared range. Hostile non-finite
    /// input takes the neutral default rather than a clamped extreme.
    pub fn sanitized(self) -> Self {
        let defaults = Self::default();
        Self {
            lines: self.lines.clamp(SCAN_MIN_LINES, SCAN_MAX_LINES),
            samples_per_line: self
                .samples_per_line
                .clamp(SCAN_MIN_SAMPLES, SCAN_MAX_SAMPLES),
            amount: finite_clamp(self.amount, defaults.amount, 0.0, 1.0),
            ribbon_width: finite_clamp(self.ribbon_width, defaults.ribbon_width, 0.0, 1.0),
            velocity_mix: finite_clamp(self.velocity_mix, defaults.velocity_mix, 0.0, 1.0),
            tilt_x: finite_clamp(self.tilt_x, defaults.tilt_x, -1.0, 1.0),
            tilt_y: finite_clamp(self.tilt_y, defaults.tilt_y, -1.0, 1.0),
            perspective: finite_clamp(self.perspective, defaults.perspective, 0.0, 1.0),
            s_curve: finite_clamp(self.s_curve, defaults.s_curve, -1.0, 1.0),
            skew: finite_clamp(self.skew, defaults.skew, -1.0, 1.0),
            collapse: finite_clamp(self.collapse, defaults.collapse, 0.0, 1.0),
            reverse_h: self.reverse_h,
            reverse_v: self.reverse_v,
            osc_amount: finite_clamp(self.osc_amount, defaults.osc_amount, 0.0, 1.0),
            osc_freq: finite_clamp(self.osc_freq, defaults.osc_freq, 0.0, 1.0),
            osc_lock: finite_clamp(self.osc_lock, defaults.osc_lock, 0.0, 1.0),
            lissajous: finite_clamp(self.lissajous, defaults.lissajous, 0.0, 1.0),
            mono: finite_clamp(self.mono, defaults.mono, 0.0, 1.0),
            hue: finite_clamp(self.hue, defaults.hue, 0.0, 1.0),
        }
    }

    /// True when no deflection is authored, so the planner collects nothing
    /// and the executor encodes no pass: the carrier passes through untouched.
    ///
    /// The wake set is the *deflection* set — amount, collapse, oscillator
    /// amount, S-curve, skew, both tilts, and the two reversals. Dressing
    /// controls (ribbon width, velocity mix, perspective, oscillator
    /// frequency/lock, Lissajous, mono, hue) shape a raster that exists only
    /// once a deflection is authored, and perspective without tilt or
    /// deflection is arithmetic identity because the depth term is zero.
    /// BENDR's own stage gate is the precedent; ours widens it to include
    /// skew and the tilts, which genuinely author deflection on their own —
    /// a control that changes nothing when authored would be a control that
    /// lies.
    pub fn is_exact_bypass(self) -> bool {
        let clean = self.sanitized();
        clean.amount == 0.0
            && clean.collapse == 0.0
            && clean.osc_amount == 0.0
            && clean.s_curve == 0.0
            && clean.skew == 0.0
            && clean.tilt_x == 0.0
            && clean.tilt_y == 0.0
            && !clean.reverse_h
            && !clean.reverse_v
    }

    /// The instanced draw this node's authored geometry requests: two
    /// vertices per sample per line.
    pub fn vertex_count(self) -> u32 {
        let clean = self.sanitized();
        clean.lines * clean.samples_per_line * 2
    }
}

/// Alpha-covered Rec.709 luma of a premultiplied linear sample. Hostile RGB
/// behind zero coverage steers nothing by arithmetic, exactly as the B2
/// Contour/Chroma fields and the B12 Brightness map read their images.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn scan_luma(premultiplied_rgb: [f32; 3]) -> f32 {
    0.2126 * premultiplied_rgb[0] + 0.7152 * premultiplied_rgb[1] + 0.0722 * premultiplied_rgb[2]
}

/// The source coordinate the beam reads at `sx` along scanline `line_v`
/// (both 0..1): the two discrete reversals apply to the *read*, while the
/// drawn position keeps its geometric address, so reversing the sweep mirrors
/// the picture rather than the raster.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn beam_source_uv(params: &ScanProcessorParams, sx: f32, line_v: f32) -> [f32; 2] {
    let u = if params.reverse_h { 1.0 - sx } else { sx };
    let v = if params.reverse_v {
        1.0 - line_v
    } else {
        line_v
    };
    [u, v]
}

/// Where the beam is when it is this far along this line, in wgpu clip
/// coordinates (+y up, scanline 0 at the top), before the ribbon is built
/// around it. Everything that bends the raster happens here, so it all
/// composes with everything else. The composition order is BENDR's, kept
/// whole: sweep/field reversal (in [`beam_source_uv`]), S-curve, skew,
/// deflection oscillator, raster collapse, luminance into vertical
/// deflection, then tilt/perspective as a photographed 2D deflection.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn beam_position(
    params: &ScanProcessorParams,
    sx: f32,
    line_v: f32,
    luma: f32,
    time_seconds: f32,
) -> [f32; 2] {
    use std::f32::consts::PI;
    let mut px = sx * 2.0 - 1.0;
    let mut py = 1.0 - line_v * 2.0;
    // S-curve: the continuous-wind yoke, bending the whole raster.
    px += (py * PI).sin() * params.s_curve * 0.4;
    // Skew, which is the same control a bench monitor calls parallelogram.
    px += py * params.skew * 0.5;
    // The deflection oscillators. Locked to a multiple of the field rate the
    // pattern stands still; detuned it crawls, and that crawl is the whole
    // gesture of the instrument.
    if params.osc_amount > 0.000_5 {
        let f = (params.osc_freq * 12.0 + 0.5).floor()
            + (1.0 - params.osc_lock) * (params.osc_freq * 12.0).fract();
        let ph = py * f * PI + time_seconds * (1.0 - params.osc_lock) * 2.0;
        px += ph.sin() * params.osc_amount * 0.5;
        if params.lissajous > 0.000_5 {
            py += (px * f * 1.618_03 * PI + time_seconds * (1.0 - params.osc_lock) * 1.7).sin()
                * params.osc_amount
                * params.lissajous
                * 0.5;
        }
    }
    // Raster collapse: remove the current from one deflection system and the
    // whole frame smears down onto a single line.
    py *= 1.0 - params.collapse.clamp(0.0, 1.0);
    // And the thing the machine is actually for: luminance into the vertical
    // position control.
    py += (luma - SCAN_LUMA_PIVOT) * params.amount * SCAN_DEFLECT_SPAN;
    // Tilt is what turns a deflection into an apparent surface. The depth
    // term is the luminance deflection itself, photographed rather than
    // modelled.
    let (cx, sx2) = (params.tilt_x.cos(), params.tilt_x.sin());
    let (cy, sy) = (params.tilt_y.cos(), params.tilt_y.sin());
    let dz = (luma - SCAN_LUMA_PIVOT) * params.amount * SCAN_DEFLECT_SPAN;
    let q = [px, py, dz];
    let q = [q[0] * cy + q[2] * sy, q[1], -q[0] * sy + q[2] * cy];
    let q = [q[0], q[1] * cx - q[2] * sx2, q[1] * sx2 + q[2] * cx];
    let w = (1.0 + q[2] * params.perspective * 0.6).max(0.15);
    [q[0] / w, q[1] / w]
}

/// Beam speed from the central-difference tangent: the tangent gives which
/// way to lay the ribbon and how fast the beam is travelling at once.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn beam_speed(ahead: [f32; 2], back: [f32; 2], half_step: f32) -> f32 {
    let tx = ahead[0] - back[0];
    let ty = ahead[1] - back[1];
    ((tx * tx + ty * ty).sqrt() / (2.0 * half_step)).max(SCAN_SPEED_FLOOR)
}

/// A slower beam deposits more energy per unit length. Without this term the
/// pass is a displacement map with extra steps. Gain one at the nominal
/// two-clip-units-per-sweep speed, brighter where the displacement slows the
/// beam, dimmer where it is thrown across.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn beam_gain(speed: f32, velocity_mix: f32) -> f32 {
    let energetic = (2.0 / speed).clamp(SCAN_GAIN_MIN, SCAN_GAIN_MAX);
    1.0 + (energetic - 1.0) * velocity_mix
}

/// The ribbon normal from the central-difference tangent, with the x
/// component aspect-corrected so a horizontal ribbon keeps its authored
/// thickness on a non-square output. A degenerate tangent (a fully collapsed
/// raster can produce one) takes the vertical normal rather than dividing by
/// zero — the one deviation from the transcription, and it is the house
/// non-finite law, not a behavior change on any finite path.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn ribbon_normal(ahead: [f32; 2], back: [f32; 2], output_aspect: f32) -> [f32; 2] {
    let tx = ahead[0] - back[0];
    let ty = ahead[1] - back[1];
    let nx = -ty;
    let ny = tx * output_aspect;
    let len = (nx * nx + ny * ny).sqrt();
    if len <= 1.0e-6 || !len.is_finite() {
        return [0.0, 1.0];
    }
    [nx / len, ny / len]
}

/// Half the drawn ribbon's thickness in clip units, from the authored width
/// and the output height in pixels: 0.7 px at zero width, 7.7 px at full.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn ribbon_half_width_clip(ribbon_width: f32, output_height: f32) -> f32 {
    (0.7 + ribbon_width * 7.0) / output_height.max(1.0) * 2.0
}

/// The beam colour law applied before accumulation, in premultiplied linear
/// light: mono folds to Rec.709 luma, and colourise repaints the beam from a
/// luma-indexed HSV sweep scaled by the luma itself so black stays black.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn scan_colorize(premultiplied_rgb: [f32; 3], mono: f32, hue: f32) -> [f32; 3] {
    let y = scan_luma(premultiplied_rgb);
    let mut c = [
        premultiplied_rgb[0] + (y - premultiplied_rgb[0]) * mono,
        premultiplied_rgb[1] + (y - premultiplied_rgb[1]) * mono,
        premultiplied_rgb[2] + (y - premultiplied_rgb[2]) * mono,
    ];
    if hue > 0.002 {
        let swept = hsv_to_rgb((hue + y * 0.35).fract(), 0.85, 1.0);
        for (channel, target) in c.iter_mut().zip(swept) {
            *channel += (target * y - *channel) * hue;
        }
    }
    c
}

/// The classic branch-free HSV→RGB, matching the WGSL expression.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let p = |offset: f32| ((h + offset).fract() * 6.0 - 3.0).abs();
    let channel = |offset: f32| v * (1.0 + s * ((p(offset) - 1.0).clamp(0.0, 1.0) - 1.0));
    [channel(0.0), channel(2.0 / 3.0), channel(1.0 / 3.0)]
}

fn finite_clamp(value: f32, fallback: f32, min: f32, max: f32) -> f32 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deflected() -> ScanProcessorParams {
        // Perspective divides the deflection when the depth term is nonzero,
        // so the analytic span fixture authors it off.
        ScanProcessorParams {
            amount: 0.5,
            perspective: 0.0,
            ..ScanProcessorParams::default()
        }
    }

    #[test]
    fn the_default_is_an_exact_bypass_and_any_deflection_wakes_it() {
        assert!(ScanProcessorParams::default().is_exact_bypass());
        for wake in [
            ScanProcessorParams {
                amount: 0.1,
                ..ScanProcessorParams::default()
            },
            ScanProcessorParams {
                collapse: 0.1,
                ..ScanProcessorParams::default()
            },
            ScanProcessorParams {
                osc_amount: 0.1,
                ..ScanProcessorParams::default()
            },
            ScanProcessorParams {
                s_curve: -0.1,
                ..ScanProcessorParams::default()
            },
            ScanProcessorParams {
                skew: 0.1,
                ..ScanProcessorParams::default()
            },
            ScanProcessorParams {
                tilt_x: 0.1,
                ..ScanProcessorParams::default()
            },
            ScanProcessorParams {
                tilt_y: -0.1,
                ..ScanProcessorParams::default()
            },
            ScanProcessorParams {
                reverse_h: true,
                ..ScanProcessorParams::default()
            },
            ScanProcessorParams {
                reverse_v: true,
                ..ScanProcessorParams::default()
            },
        ] {
            assert!(!wake.is_exact_bypass());
        }
    }

    #[test]
    fn dressing_controls_alone_stay_bypassed() {
        // Perspective without a deflection is arithmetic identity (the depth
        // term is zero), and the remaining dressing controls shape a raster
        // that is never drawn while bypassed.
        let dressed = ScanProcessorParams {
            ribbon_width: 1.0,
            velocity_mix: 0.0,
            perspective: 1.0,
            osc_freq: 0.9,
            osc_lock: 0.0,
            lissajous: 1.0,
            mono: 1.0,
            hue: 1.0,
            ..ScanProcessorParams::default()
        };
        assert!(dressed.is_exact_bypass());
    }

    #[test]
    fn hostile_scalars_sanitize_to_the_neutral_default_not_a_clamped_extreme() {
        let hostile = ScanProcessorParams {
            amount: f32::NAN,
            ribbon_width: f32::INFINITY,
            velocity_mix: f32::NEG_INFINITY,
            perspective: f32::NAN,
            osc_lock: f32::NAN,
            lines: 0,
            samples_per_line: u32::MAX,
            ..ScanProcessorParams::default()
        };
        let clean = hostile.sanitized();
        let defaults = ScanProcessorParams::default();
        assert_eq!(clean.amount, defaults.amount);
        assert_eq!(clean.ribbon_width, defaults.ribbon_width);
        assert_eq!(clean.velocity_mix, defaults.velocity_mix);
        assert_eq!(clean.perspective, defaults.perspective);
        assert_eq!(clean.osc_lock, defaults.osc_lock);
        assert_eq!(clean.lines, SCAN_MIN_LINES);
        assert_eq!(clean.samples_per_line, SCAN_MAX_SAMPLES);
        assert!(hostile.sanitized().is_exact_bypass());
    }

    #[test]
    fn the_vertex_ledger_is_two_per_sample_per_line_and_the_cap_is_the_maxima() {
        let clean = ScanProcessorParams::default();
        assert_eq!(clean.vertex_count(), 320 * 256 * 2);
        let maxed = ScanProcessorParams {
            lines: SCAN_MAX_LINES,
            samples_per_line: SCAN_MAX_SAMPLES,
            ..ScanProcessorParams::default()
        };
        assert_eq!(maxed.vertex_count(), MAX_SCAN_PROCESSOR_VERTICES);
    }

    #[test]
    fn an_undeflected_beam_traces_the_flat_raster_exactly() {
        // With no deflection authored, the beam at (sx, line) is the plain
        // clip-space address of that raster position, whatever the luma.
        let p = ScanProcessorParams::default();
        for &(sx, line) in &[(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (0.25, 0.75)] {
            let pos = beam_position(&p, sx, line, 0.9, 123.0);
            assert!((pos[0] - (sx * 2.0 - 1.0)).abs() < 1.0e-6);
            assert!((pos[1] - (1.0 - line * 2.0)).abs() < 1.0e-6);
        }
    }

    #[test]
    fn luminance_pulls_the_line_up_the_tube_about_the_pivot() {
        let p = deflected();
        let bright = beam_position(&p, 0.5, 0.5, 1.0, 0.0);
        let pivot = beam_position(&p, 0.5, 0.5, SCAN_LUMA_PIVOT, 0.0);
        let dark = beam_position(&p, 0.5, 0.5, 0.0, 0.0);
        assert!(bright[1] > pivot[1]);
        assert!(dark[1] < pivot[1]);
        // The pivot itself is undeflected.
        assert!((pivot[1] - 0.0).abs() < 1.0e-6);
        // And the span is the analytic law.
        let expected = (1.0 - SCAN_LUMA_PIVOT) * 0.5 * SCAN_DEFLECT_SPAN;
        assert!((bright[1] - expected).abs() < 1.0e-6);
    }

    #[test]
    fn reversal_mirrors_the_read_not_the_drawn_position() {
        let p = ScanProcessorParams {
            reverse_h: true,
            reverse_v: true,
            ..ScanProcessorParams::default()
        };
        assert_eq!(beam_source_uv(&p, 0.25, 0.1), [0.75, 0.9]);
        // The drawn address is untouched by the reversal.
        let pos = beam_position(&p, 0.25, 0.1, SCAN_LUMA_PIVOT, 0.0);
        assert!((pos[0] - (-0.5)).abs() < 1.0e-6);
        assert!((pos[1] - 0.8).abs() < 1.0e-6);
    }

    #[test]
    fn collapse_smears_the_raster_onto_a_single_line() {
        let p = ScanProcessorParams {
            collapse: 1.0,
            ..ScanProcessorParams::default()
        };
        for &line in &[0.0_f32, 0.25, 0.5, 1.0] {
            let pos = beam_position(&p, 0.3, line, SCAN_LUMA_PIVOT, 0.0);
            assert!(pos[1].abs() < 1.0e-6);
        }
    }

    #[test]
    fn a_locked_oscillator_stands_still_and_a_detuned_one_crawls() {
        let locked = ScanProcessorParams {
            osc_amount: 0.5,
            osc_freq: 0.5,
            osc_lock: 1.0,
            ..ScanProcessorParams::default()
        };
        let a = beam_position(&locked, 0.4, 0.3, SCAN_LUMA_PIVOT, 0.0);
        let b = beam_position(&locked, 0.4, 0.3, SCAN_LUMA_PIVOT, 97.3);
        assert_eq!(a, b);
        let detuned = ScanProcessorParams {
            osc_lock: 0.0,
            osc_freq: 0.51,
            ..locked
        };
        let c = beam_position(&detuned, 0.4, 0.3, SCAN_LUMA_PIVOT, 0.0);
        let d = beam_position(&detuned, 0.4, 0.3, SCAN_LUMA_PIVOT, 0.25);
        assert_ne!(c, d);
    }

    #[test]
    fn the_locked_frequency_quantizes_to_whole_multiples() {
        // At lock 1 the effective frequency is floor(freq*12 + 0.5): two
        // authored frequencies rounding to the same multiple draw the same
        // standing pattern.
        let base = ScanProcessorParams {
            osc_amount: 0.5,
            osc_lock: 1.0,
            ..ScanProcessorParams::default()
        };
        let low = ScanProcessorParams {
            osc_freq: 0.48,
            ..base
        };
        let high = ScanProcessorParams {
            osc_freq: 0.52,
            ..base
        };
        let a = beam_position(&low, 0.4, 0.3, SCAN_LUMA_PIVOT, 5.0);
        let b = beam_position(&high, 0.4, 0.3, SCAN_LUMA_PIVOT, 5.0);
        assert!((a[0] - b[0]).abs() < 1.0e-6);
        assert!((a[1] - b[1]).abs() < 1.0e-6);
    }

    #[test]
    fn the_beam_energy_law_is_two_over_speed_clamped() {
        assert!((beam_gain(2.0, 1.0) - 1.0).abs() < 1.0e-6);
        assert!((beam_gain(1.0, 1.0) - 2.0).abs() < 1.0e-6);
        assert!((beam_gain(0.05, 1.0) - SCAN_GAIN_MAX).abs() < 1.0e-6);
        assert!((beam_gain(1_000.0, 1.0) - SCAN_GAIN_MIN).abs() < 1.0e-6);
        // The mix disengages the law entirely at zero.
        assert!((beam_gain(0.05, 0.0) - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn beam_speed_is_the_central_difference_with_the_floor() {
        let speed = beam_speed([0.6, 0.0], [0.4, 0.0], 0.05);
        assert!((speed - 2.0).abs() < 1.0e-6);
        assert_eq!(beam_speed([0.5, 0.0], [0.5, 0.0], 0.05), SCAN_SPEED_FLOOR);
    }

    #[test]
    fn the_ribbon_normal_is_perpendicular_and_fails_safe() {
        let n = ribbon_normal([0.6, 0.0], [0.4, 0.0], 16.0 / 9.0);
        assert!(n[0].abs() < 1.0e-6);
        assert!((n[1] - 1.0).abs() < 1.0e-6);
        assert_eq!(ribbon_normal([0.5, 0.5], [0.5, 0.5], 1.0), [0.0, 1.0]);
    }

    #[test]
    fn ribbon_width_is_the_pixel_law() {
        let w = ribbon_half_width_clip(0.0, 1_080.0);
        assert!((w - 0.7 / 1_080.0 * 2.0).abs() < 1.0e-7);
        let wide = ribbon_half_width_clip(1.0, 1_080.0);
        assert!((wide - 7.7 / 1_080.0 * 2.0).abs() < 1.0e-7);
        // A hostile zero height takes the guard, not a division by zero.
        assert!(ribbon_half_width_clip(0.5, 0.0).is_finite());
    }

    #[test]
    fn colorize_folds_to_luma_and_black_stays_black() {
        let c = scan_colorize([0.5, 0.25, 0.125], 1.0, 0.0);
        let y = scan_luma([0.5, 0.25, 0.125]);
        for channel in c {
            assert!((channel - y).abs() < 1.0e-6);
        }
        // Full colourise of black is black: the sweep is scaled by luma.
        let black = scan_colorize([0.0, 0.0, 0.0], 0.0, 1.0);
        for channel in black {
            assert!(channel.abs() < 1.0e-6);
        }
        // Zero mono and hue is the identity.
        assert_eq!(scan_colorize([0.3, 0.6, 0.9], 0.0, 0.0), [0.3, 0.6, 0.9]);
    }

    #[test]
    fn tilt_alone_authors_geometry_and_perspective_alone_does_not() {
        let tilted = ScanProcessorParams {
            tilt_y: 0.5,
            ..ScanProcessorParams::default()
        };
        let flat = beam_position(&ScanProcessorParams::default(), 0.9, 0.5, 0.5, 0.0);
        let bent = beam_position(&tilted, 0.9, 0.5, 0.5, 0.0);
        assert!((flat[0] - bent[0]).abs() > 1.0e-3);
        // Perspective with no depth term moves nothing.
        let persp = ScanProcessorParams {
            perspective: 1.0,
            ..ScanProcessorParams::default()
        };
        let unmoved = beam_position(&persp, 0.9, 0.5, 0.5, 0.0);
        assert!((flat[0] - unmoved[0]).abs() < 1.0e-6);
        assert!((flat[1] - unmoved[1]).abs() < 1.0e-6);
    }

    #[test]
    fn serde_rejects_unknown_fields_and_fills_absent_ones() {
        let yaml = "amount: 0.5\n";
        let parsed: ScanProcessorParams = serde_yaml::from_str(yaml).expect("partial params");
        assert_eq!(parsed.amount, 0.5);
        assert_eq!(parsed.lines, SCAN_DEFAULT_LINES);
        let hostile = "amount: 0.5\nunknown_field: 1\n";
        assert!(serde_yaml::from_str::<ScanProcessorParams>(hostile).is_err());
    }
}
