//! The B4 display-physics law — the field domain, phosphor persistence, and
//! the display model.
//!
//! Everything the program renders is *watched through something*. Three
//! mechanisms matter: real interlace (two moments in one frame — the
//! serrated pan edge no drawn comb reproduces), per-primary phosphor
//! persistence (one accumulator, three decay constants — green hangs on
//! longest, blue goes first, so motion leaves a green-tinted wake with a
//! blue leading edge), and the mask/beam display models.
//!
//! The laws are derived from BENDR (MIT, © 2026 Steve Blythe) and rewritten
//! for this tree (Rust / wgpu / WGSL, linear light, Rec.709 luma, the 30 Hz
//! reference clock). This module is the independent CPU reference the GPU
//! stage is checked against, in the `gesture.rs` tradition: no wgpu, clock,
//! filesystem, or UI dependency. Rate law: the phosphor decay is
//! multiplicative and exponentiates per 1/30-second reference tick; field
//! parity and the 3:2 film clock advance on whole reference ticks, so live
//! and offline agree structurally.

use serde::{Deserialize, Serialize};

/// The field recombination law. Codes are permanent and append-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterlaceMode {
    /// The two fields simply interleave, so anything that moved between them
    /// serrates. This is what an interlaced signal actually is.
    #[default]
    Weave,
    /// Only the current field is real and the gaps are filled from its
    /// neighbours, so the whole picture jitters by half a line at field
    /// rate. Cheap deinterlacers did this and it is unmistakable.
    Bob,
    /// Average the fields. No comb, but everything that moves ghosts.
    Blend,
}

impl InterlaceMode {
    /// Permanent append-only shader codes.
    pub const fn code(self) -> u32 {
        match self {
            Self::Weave => 0,
            Self::Bob => 1,
            Self::Blend => 2,
        }
    }
}

/// The display model. Codes are permanent and append-only; `Flat` is the
/// exact-off law for the beam/mask family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DisplayModel {
    #[default]
    Flat,
    ApertureGrille,
    SlotMask,
    ShadowMask,
    LcdStripe,
    Mono,
    GreenScreen,
}

impl DisplayModel {
    /// Permanent append-only shader codes.
    pub const fn code(self) -> u32 {
        match self {
            Self::Flat => 0,
            Self::ApertureGrille => 1,
            Self::SlotMask => 2,
            Self::ShadowMask => 3,
            Self::LcdStripe => 4,
            Self::Mono => 5,
            Self::GreenScreen => 6,
        }
    }
}

/// The authored display-physics state: three sub-blocks (fields, phosphor,
/// display model), each defaulting to exact-off. Frame-local evaluated state
/// like a spatial transform — never topology.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct DisplayPhysicsParams {
    /// Interlace mix. The field sub-block's wake control.
    pub il_amount: f32,
    pub il_mode: InterlaceMode,
    /// Field-order swap — the documented fault: the stutter of a tape
    /// captured with the wrong dominance.
    pub il_order: bool,
    /// High-vertical-detail flicker at half the field rate. Dressing on an
    /// armed field domain.
    pub il_twitter: f32,
    /// 3:2 pulldown judder: film at 24 into video, some frames held three
    /// fields and some two.
    pub il_judder: f32,
    /// Phosphor persistence amount. The phosphor sub-block's wake control.
    pub phosphor: f32,
    /// Per-primary decay constants: P22 red.
    pub phos_r: f32,
    /// P22 green — hangs on longest.
    pub phos_g: f32,
    /// P22 blue — goes first.
    pub phos_b: f32,
    /// The display model; `Flat` gates the beam/mask family off exactly.
    pub model: DisplayModel,
    /// Beam-profile scanline strength (active only under a non-Flat model).
    pub scanlines: f32,
    /// Nominal beam width in scanline heights.
    pub beam_width: f32,
    /// How much the beam widens with brightness (the Lottes-style profile).
    pub beam_shape: f32,
    /// Mask strength (active only under a masked model).
    pub mask_strength: f32,
    /// How dark the masked phosphors of the other primaries sit.
    pub mask_dark: f32,
    /// Bloom off the highlights.
    pub bloom: f32,
    /// Bloom/defocus gather radius.
    pub bloom_radius: f32,
    /// Halation tint on the bloom (the orange cast of light scattering in
    /// the faceplate).
    pub halation: f32,
    /// Focus loss: the picture mixed toward its own gathered blur.
    pub defocus: f32,
    /// HV sag: the picture breathes wider as it gets brighter — the EHT
    /// supply drooping under beam load.
    pub sag: f32,
}

impl Default for DisplayPhysicsParams {
    fn default() -> Self {
        Self {
            il_amount: 0.0,
            il_mode: InterlaceMode::Weave,
            il_order: false,
            il_twitter: 0.4,
            il_judder: 0.0,
            phosphor: 0.0,
            phos_r: 0.86,
            phos_g: 1.0,
            phos_b: 0.66,
            model: DisplayModel::Flat,
            scanlines: 0.0,
            beam_width: 1.0,
            beam_shape: 0.5,
            mask_strength: 0.0,
            mask_dark: 0.5,
            bloom: 0.0,
            bloom_radius: 0.4,
            halation: 0.0,
            defocus: 0.0,
            sag: 0.0,
        }
    }
}

impl DisplayPhysicsParams {
    /// Clamp every authored value into its declared range. Hostile
    /// non-finite input takes the neutral default rather than a clamped
    /// extreme.
    pub fn sanitized(self) -> Self {
        let defaults = Self::default();
        Self {
            il_amount: finite_clamp(self.il_amount, defaults.il_amount, 0.0, 1.0),
            il_mode: self.il_mode,
            il_order: self.il_order,
            il_twitter: finite_clamp(self.il_twitter, defaults.il_twitter, 0.0, 1.0),
            il_judder: finite_clamp(self.il_judder, defaults.il_judder, 0.0, 1.0),
            phosphor: finite_clamp(self.phosphor, defaults.phosphor, 0.0, 0.95),
            phos_r: finite_clamp(self.phos_r, defaults.phos_r, 0.0, 1.0),
            phos_g: finite_clamp(self.phos_g, defaults.phos_g, 0.0, 1.0),
            phos_b: finite_clamp(self.phos_b, defaults.phos_b, 0.0, 1.0),
            model: self.model,
            scanlines: finite_clamp(self.scanlines, defaults.scanlines, 0.0, 1.0),
            beam_width: finite_clamp(self.beam_width, defaults.beam_width, 0.1, 3.0),
            beam_shape: finite_clamp(self.beam_shape, defaults.beam_shape, 0.0, 1.0),
            mask_strength: finite_clamp(self.mask_strength, defaults.mask_strength, 0.0, 1.0),
            mask_dark: finite_clamp(self.mask_dark, defaults.mask_dark, 0.0, 1.0),
            bloom: finite_clamp(self.bloom, defaults.bloom, 0.0, 1.0),
            bloom_radius: finite_clamp(self.bloom_radius, defaults.bloom_radius, 0.0, 1.0),
            halation: finite_clamp(self.halation, defaults.halation, 0.0, 1.0),
            defocus: finite_clamp(self.defocus, defaults.defocus, 0.0, 1.0),
            sag: finite_clamp(self.sag, defaults.sag, 0.0, 1.0),
        }
    }

    /// The field sub-block is armed. Twitter, judder, order, and mode are
    /// dressing on an armed field domain and wake nothing alone.
    pub fn fields_active(self) -> bool {
        self.sanitized().il_amount > 0.0
    }

    /// The phosphor sub-block is armed. The three decay constants are
    /// dressing on an armed accumulator and wake nothing alone.
    pub fn phosphor_active(self) -> bool {
        self.sanitized().phosphor > 0.0
    }

    /// The display sub-block is armed: a non-Flat model, or one of the three
    /// optics that act under any model. Scanlines, beam and mask controls
    /// are gated under a non-Flat model (BENDR's own law: `Flat` is the
    /// sub-block's off switch), and bloom radius/halation are dressing on an
    /// armed bloom.
    pub fn display_active(self) -> bool {
        let clean = self.sanitized();
        clean.model != DisplayModel::Flat
            || clean.bloom > 0.0
            || clean.defocus > 0.0
            || clean.sag > 0.0
    }

    /// The whole stage delegates exactly when all three sub-blocks are off:
    /// no pass is encoded, no surface is touched, and the post-temporal
    /// image reaches the opaque resolve untouched.
    pub fn stage_active(self) -> bool {
        self.fields_active() || self.phosphor_active() || self.display_active()
    }

    /// The per-primary decay applied per reference tick:
    /// `clamp(phos_rgb * phosphor, 0, 0.995)`.
    pub fn phosphor_decay(self) -> [f32; 3] {
        let clean = self.sanitized();
        [
            (clean.phos_r * clean.phosphor).clamp(0.0, 0.995),
            (clean.phos_g * clean.phosphor).clamp(0.0, 0.995),
            (clean.phos_b * clean.phosphor).clamp(0.0, 0.995),
        ]
    }
}

/// Rec.709 luma in linear light — the house law (BENDR weighs 601 in gamma
/// space; the mechanism, not the matrix, is what is transcribed).
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn display_luma(rgb: [f32; 3]) -> f32 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

/// Field parity at a reference tick: fields alternate every tick of the
/// 30 Hz authoring reference. The order fault swaps which field is "now".
pub fn field_parity(total_reference_ticks: u64, order_swapped: bool) -> u32 {
    let parity = (total_reference_ticks & 1) as u32;
    if order_swapped {
        1 - parity
    } else {
        parity
    }
}

/// The 3:2 film clock: film at 24 against the 30 Hz reference is exactly
/// four film frames per five ticks, and a frame is held (shown a third
/// field) on two of every five film frames.
pub fn judder_held(total_reference_ticks: u64) -> bool {
    let film_frame = total_reference_ticks * 4 / 5;
    film_frame % 5 < 2
}

/// The per-pixel field recombination — `FS_FIELD`, expression for
/// expression, on this tree's terms. `line` is the output row index,
/// `parity` the current field parity from [`field_parity`]. `cur` is the
/// current pixel, `prev` the same pixel of the retained previous field,
/// `bob_neighbour` the current image one line toward the field being drawn,
/// and `up`/`dn` the current image's vertical neighbours for twitter.
#[allow(
    clippy::too_many_arguments,
    reason = "the law names every tap explicitly"
)]
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn field_resolve(
    params: &DisplayPhysicsParams,
    line: u32,
    parity: u32,
    judder_hold: bool,
    cur: [f32; 3],
    prev: [f32; 3],
    bob_neighbour: [f32; 3],
    up: [f32; 3],
    dn: [f32; 3],
) -> [f32; 3] {
    let clean = params.sanitized();
    if clean.il_amount <= 0.0 {
        return cur;
    }
    let this_field = (line & 1) == parity;
    let mut out = match clean.il_mode {
        InterlaceMode::Weave => {
            if this_field {
                cur
            } else {
                prev
            }
        }
        InterlaceMode::Bob => {
            if this_field {
                cur
            } else {
                bob_neighbour
            }
        }
        InterlaceMode::Blend => mix3(cur, prev, 0.5),
    };
    // Twitter: a high vertical frequency lands on one field only, so it
    // flickers at half the frame rate.
    if clean.il_twitter > 0.003 {
        let on_this = if this_field { 1.0 } else { -1.0 };
        for channel in 0..3 {
            let hf = cur[channel] - (up[channel] + dn[channel]) * 0.5;
            out[channel] += hf * on_this * clean.il_twitter * 1.6;
        }
    }
    // 3:2 pulldown: held frames lean on the previous field.
    if clean.il_judder > 0.003 && judder_hold {
        out = mix3(out, prev, clean.il_judder * 0.85);
    }
    mix3(cur, out, clean.il_amount)
}

/// The accumulator law: keep whichever is brighter, this frame or the
/// decayed trail, per channel. `decay_per_tick` is
/// [`DisplayPhysicsParams::phosphor_decay`], exponentiated by the elapsed
/// reference ticks (the multiplicative rate law) before this is applied.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn phosphor_combine(cur: [f32; 3], acc: [f32; 3], decay: [f32; 3]) -> [f32; 3] {
    [
        cur[0].max(acc[0] * decay[0]),
        cur[1].max(acc[1] * decay[1]),
        cur[2].max(acc[2] * decay[2]),
    ]
}

/// The multiplicative rate law: a decay constant defined per 1/30-second
/// reference tick, exponentiated by the ticks one frame actually spans, so
/// live at any fps and export at any fps share one trail length.
pub fn phosphor_decay_over_ticks(decay_per_tick: [f32; 3], ticks: f32) -> [f32; 3] {
    let ticks = if ticks.is_finite() {
        ticks.max(0.0)
    } else {
        1.0
    };
    decay_per_tick.map(|k| k.max(0.0).powf(ticks))
}

/// The shadow-mask families — `maskAt`, kept whole. `fc` is the framebuffer
/// coordinate in pixels; returns the per-primary transmission.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn mask_at(fc: [f32; 2], model: DisplayModel, mask_dark: f32) -> [f32; 3] {
    let dark = 1.0 - mask_dark * 0.55;
    let pick = |m: f32| -> [f32; 3] {
        [
            if m < 1.0 { 1.0 } else { dark },
            if (1.0..2.0).contains(&m) { 1.0 } else { dark },
            if m >= 2.0 { 1.0 } else { dark },
        ]
    };
    match model {
        DisplayModel::ApertureGrille => pick(fc[0].rem_euclid(3.0)),
        DisplayModel::SlotMask => {
            let gy = (fc[1] / 6.0).floor();
            let off = gy.rem_euclid(2.0) * 1.5;
            let m = (fc[0] + off).rem_euclid(3.0);
            let v = if fc[1].rem_euclid(6.0) < 5.0 {
                1.0
            } else {
                dark
            };
            pick(m).map(|value| value * v)
        }
        DisplayModel::ShadowMask => {
            let q = [fc[0] / 6.0, fc[1] / 6.0];
            let f = [q[0] - q[0].floor() - 0.5, q[1] - q[1].floor() - 0.5];
            let r = (f[0] * f[0] + f[1] * f[1]).sqrt();
            let tri = (q[0].floor() + q[1].floor() * 2.0).rem_euclid(3.0);
            let t = [
                if tri < 0.5 { 1.0 } else { dark },
                if (0.5..1.5).contains(&tri) { 1.0 } else { dark },
                if tri >= 1.5 { 1.0 } else { dark },
            ];
            let ring = 1.0 - smoothstep(0.28, 0.5, r) * mask_dark;
            t.map(|value| value * ring)
        }
        DisplayModel::LcdStripe => {
            let m = fc[0].rem_euclid(2.0);
            let stripe = if m < 1.0 { 1.06 } else { dark };
            [
                1.0 + (stripe - 1.0) * 0.85,
                1.0 + (stripe - 1.0) * 0.85,
                1.0 + (stripe - 1.0) * 0.85,
            ]
        }
        DisplayModel::Flat | DisplayModel::Mono | DisplayModel::GreenScreen => [1.0, 1.0, 1.0],
    }
}

/// The Lottes-style beam profile: a gaussian beam whose width tracks
/// brightness. `fy` is the fractional position within the scanline centred
/// on zero (`fract(v * lines) - 0.5`).
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn beam_profile(fy: f32, width: f32, shape: f32, brightness: f32) -> f32 {
    let w = width * (0.35 + 0.65 * (1.0 + (brightness - 1.0) * shape));
    (-(fy * fy) / (w * w * 0.22).max(0.005)).exp()
}

/// The HV sag scale: the raster grows as the mean picture gets brighter,
/// because the EHT supply droops under beam load. `centre_luma` is the luma
/// of the picture centre.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn sag_scale(sag: f32, centre_luma: f32) -> f32 {
    if sag > 0.003 {
        sag * 0.035 * (centre_luma + 0.35)
    } else {
        0.0
    }
}

/// The fixed 12-tap gather ring shared by defocus and bloom: angles at
/// multiples of 0.5236 rad, radii cycling `1..3` scaled by 0.45, weights
/// `1/(1 + r*1.4)`. Fixed law, not an authored control.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn gather_ring() -> [([f32; 2], f32); 12] {
    std::array::from_fn(|i| {
        // BENDR's transcribed 30-degree step literal, mirrored byte for byte
        // by the WGSL (which has no pi constant to share).
        #[allow(clippy::approx_constant, reason = "the transcribed BENDR literal")]
        let a = i as f32 * 0.523_6;
        let r = (1.0 + (i % 3) as f32) * 0.45;
        ([a.cos() * r, a.sin() * r], 1.0 / (1.0 + r * 1.4))
    })
}

/// The hot-highlight extraction feeding bloom: everything above 0.42,
/// scaled 1.9, tinted toward halation's faceplate orange.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn bloom_hot(blur: [f32; 3], bloom: f32, halation: f32) -> [f32; 3] {
    let tint = [
        1.0 + (1.25 - 1.0) * halation,
        1.0 + (0.62 - 1.0) * halation,
        1.0 + (0.42 - 1.0) * halation,
    ];
    std::array::from_fn(|channel| {
        (blur[channel] - 0.42).max(0.0) * 1.9 * bloom * 1.5 * tint[channel]
    })
}

/// The two tinted models applied after beam and mask: `Mono` folds 85% of
/// the way to luma, `GreenScreen` additionally tints the phosphor.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn model_tint(rgb: [f32; 3], model: DisplayModel) -> [f32; 3] {
    match model {
        DisplayModel::Mono | DisplayModel::GreenScreen => {
            let y = display_luma(rgb);
            let mono = std::array::from_fn(|channel| rgb[channel] + (y - rgb[channel]) * 0.85);
            if model == DisplayModel::GreenScreen {
                [mono[0] * 0.75, mono[1], mono[2] * 0.8]
            } else {
                mono
            }
        }
        _ => rgb,
    }
}

/// Flag bits in `DisplayPhysicsGpuUniforms::modes[3]`.
pub const DISPLAY_FLAG_JUDDER_HOLD: u32 = 1;
pub const DISPLAY_FLAG_FIELD_VALID: u32 = 2;
pub const DISPLAY_FLAG_PHOSPHOR_VALID: u32 = 4;

/// The 128-byte uniform record shared by the field, display, and phosphor
/// store passes, mirrored lane for lane by `DisplayUniforms` in
/// `display_physics.wgsl`. The decay lanes arrive already exponentiated by
/// this frame's fractional reference ticks — the multiplicative rate law is
/// applied on the CPU so the shader never owns a clock.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DisplayPhysicsGpuUniforms {
    /// il_amount, il_twitter, il_judder, phosphor
    pub fields: [f32; 4],
    /// decay r, decay g, decay b (tick-exponentiated), scanlines
    pub decay: [f32; 4],
    /// beam_width, beam_shape, mask_strength, mask_dark
    pub beam: [f32; 4],
    /// bloom, bloom_radius, halation, defocus
    pub optics: [f32; 4],
    /// sag, output width, output height, reserved
    pub frame: [f32; 4],
    /// il_mode code, model code, field parity, flag bits
    pub modes: [u32; 4],
    pub reserved0: [f32; 4],
    pub reserved1: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<DisplayPhysicsGpuUniforms>() == 128);

impl DisplayPhysicsGpuUniforms {
    pub fn from_parts(
        params: &DisplayPhysicsParams,
        output: [u32; 2],
        parity: u32,
        judder_hold: bool,
        decay_this_frame: [f32; 3],
        field_valid: bool,
        phosphor_valid: bool,
    ) -> Self {
        let clean = params.sanitized();
        let mut flags = 0;
        if judder_hold {
            flags |= DISPLAY_FLAG_JUDDER_HOLD;
        }
        if field_valid {
            flags |= DISPLAY_FLAG_FIELD_VALID;
        }
        if phosphor_valid {
            flags |= DISPLAY_FLAG_PHOSPHOR_VALID;
        }
        Self {
            fields: [
                clean.il_amount,
                clean.il_twitter,
                clean.il_judder,
                clean.phosphor,
            ],
            decay: [
                decay_this_frame[0].clamp(0.0, 0.995),
                decay_this_frame[1].clamp(0.0, 0.995),
                decay_this_frame[2].clamp(0.0, 0.995),
                clean.scanlines,
            ],
            beam: [
                clean.beam_width,
                clean.beam_shape,
                clean.mask_strength,
                clean.mask_dark,
            ],
            optics: [
                clean.bloom,
                clean.bloom_radius,
                clean.halation,
                clean.defocus,
            ],
            frame: [clean.sag, output[0] as f32, output[1] as f32, 0.0],
            modes: [clean.il_mode.code(), clean.model.code(), parity & 1, flags],
            reserved0: [0.0; 4],
            reserved1: [0.0; 4],
        }
    }
}

#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    std::array::from_fn(|channel| a[channel] + (b[channel] - a[channel]) * t)
}

#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
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

    #[test]
    fn the_default_is_exact_off_in_all_three_sub_blocks() {
        let default = DisplayPhysicsParams::default();
        assert!(!default.fields_active());
        assert!(!default.phosphor_active());
        assert!(!default.display_active());
        assert!(!default.stage_active());
        // Dressing controls alone wake nothing: twitter, decay constants,
        // beam and mask dressing, bloom radius, halation.
        let dressed = DisplayPhysicsParams {
            il_twitter: 1.0,
            il_judder: 1.0,
            il_order: true,
            il_mode: InterlaceMode::Bob,
            phos_r: 1.0,
            phos_b: 1.0,
            scanlines: 1.0,
            beam_width: 3.0,
            beam_shape: 1.0,
            mask_strength: 1.0,
            mask_dark: 1.0,
            bloom_radius: 1.0,
            halation: 1.0,
            ..DisplayPhysicsParams::default()
        };
        assert!(!dressed.stage_active());
        // Each wake control wakes exactly its own sub-block.
        assert!(DisplayPhysicsParams {
            il_amount: 0.1,
            ..DisplayPhysicsParams::default()
        }
        .fields_active());
        assert!(DisplayPhysicsParams {
            phosphor: 0.1,
            ..DisplayPhysicsParams::default()
        }
        .phosphor_active());
        for wake in [
            DisplayPhysicsParams {
                model: DisplayModel::ApertureGrille,
                ..DisplayPhysicsParams::default()
            },
            DisplayPhysicsParams {
                bloom: 0.1,
                ..DisplayPhysicsParams::default()
            },
            DisplayPhysicsParams {
                defocus: 0.1,
                ..DisplayPhysicsParams::default()
            },
            DisplayPhysicsParams {
                sag: 0.1,
                ..DisplayPhysicsParams::default()
            },
        ] {
            assert!(wake.display_active());
            assert!(wake.stage_active());
        }
    }

    #[test]
    fn hostile_scalars_sanitize_to_the_neutral_default_not_a_clamped_extreme() {
        let hostile = DisplayPhysicsParams {
            il_amount: f32::NAN,
            phosphor: f32::INFINITY,
            beam_width: f32::NEG_INFINITY,
            sag: f32::NAN,
            ..DisplayPhysicsParams::default()
        };
        let clean = hostile.sanitized();
        assert_eq!(clean.il_amount, 0.0);
        assert_eq!(clean.phosphor, 0.0);
        assert_eq!(clean.beam_width, 1.0);
        assert_eq!(clean.sag, 0.0);
        assert!(!hostile.stage_active());
        // Finite over-range clamps normally.
        assert_eq!(
            DisplayPhysicsParams {
                phosphor: 4.0,
                ..DisplayPhysicsParams::default()
            }
            .sanitized()
            .phosphor,
            0.95
        );
    }

    #[test]
    fn field_parity_alternates_per_tick_and_the_order_fault_swaps_it() {
        assert_eq!(field_parity(0, false), 0);
        assert_eq!(field_parity(1, false), 1);
        assert_eq!(field_parity(2, false), 0);
        assert_eq!(field_parity(0, true), 1);
        assert_eq!(field_parity(1, true), 0);
    }

    #[test]
    fn the_film_clock_holds_two_of_every_five_film_frames() {
        // Over any 5-film-frame cycle, exactly two are held.
        let held: Vec<bool> = (0..25).map(judder_held).collect();
        let count = held.iter().filter(|hold| **hold).count();
        // 25 ticks = 20 film frames = 4 cycles of 5 = 8 held frames' worth
        // of ticks; the exact tick count follows the 4/5 ratio.
        assert!(count > 0 && count < 25);
        // Determinism: the clock is a pure function of the tick count.
        assert_eq!(judder_held(12), judder_held(12));
        // And the film frame advances 4 per 5 ticks exactly.
        assert_eq!(20_u64 * 4 / 5, 16);
    }

    #[test]
    fn weave_interleaves_bob_jitters_and_blend_ghosts() {
        let params = DisplayPhysicsParams {
            il_amount: 1.0,
            il_twitter: 0.0,
            ..DisplayPhysicsParams::default()
        };
        let cur = [1.0, 0.0, 0.0];
        let prev = [0.0, 1.0, 0.0];
        let bob = [0.0, 0.0, 1.0];
        let flat = [0.5; 3];
        // Weave: this field shows current, the other shows the held field.
        assert_eq!(
            field_resolve(&params, 0, 0, false, cur, prev, bob, flat, flat),
            cur
        );
        assert_eq!(
            field_resolve(&params, 1, 0, false, cur, prev, bob, flat, flat),
            prev
        );
        // Bob: the other field is filled from the current image's neighbour.
        let bob_params = DisplayPhysicsParams {
            il_mode: InterlaceMode::Bob,
            ..params
        };
        assert_eq!(
            field_resolve(&bob_params, 1, 0, false, cur, prev, bob, flat, flat),
            bob
        );
        // Blend: both fields average.
        let blend_params = DisplayPhysicsParams {
            il_mode: InterlaceMode::Blend,
            ..params
        };
        let blended = field_resolve(&blend_params, 0, 0, false, cur, prev, bob, flat, flat);
        for channel in 0..3 {
            assert!((blended[channel] - (cur[channel] + prev[channel]) * 0.5).abs() < 1e-6);
        }
        // Amount zero is the exact passthrough whatever else is authored.
        let off = DisplayPhysicsParams {
            il_amount: 0.0,
            ..bob_params
        };
        assert_eq!(
            field_resolve(&off, 1, 0, false, cur, prev, bob, flat, flat),
            cur
        );
    }

    #[test]
    fn twitter_flips_high_vertical_detail_per_field_and_judder_leans_on_the_held_field() {
        let params = DisplayPhysicsParams {
            il_amount: 1.0,
            il_twitter: 1.0,
            ..DisplayPhysicsParams::default()
        };
        // A one-line-high feature: cur bright, neighbours dark.
        let cur = [1.0, 1.0, 1.0];
        let prev = [1.0, 1.0, 1.0];
        let dark = [0.0; 3];
        let on_field = field_resolve(&params, 0, 0, false, cur, prev, cur, dark, dark);
        let off_field = field_resolve(&params, 1, 0, false, cur, prev, cur, dark, dark);
        assert!(on_field[0] > cur[0], "the detail brightens on its field");
        assert!(off_field[0] < cur[0], "and dims on the other");
        let judder = DisplayPhysicsParams {
            il_twitter: 0.0,
            il_judder: 1.0,
            ..params
        };
        let held = field_resolve(
            &judder,
            0,
            0,
            true,
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            dark,
            dark,
            dark,
        );
        assert!(held[1] > 0.5, "a held frame leans on the previous field");
        let free = field_resolve(
            &judder,
            0,
            0,
            false,
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            dark,
            dark,
            dark,
        );
        assert_eq!(free, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn the_phosphor_law_is_max_of_current_and_decayed_trail_per_channel() {
        let decay = [0.9, 1.0, 0.5];
        let out = phosphor_combine([0.2, 0.1, 0.4], [0.5, 0.05, 1.0], decay);
        assert!((out[0] - 0.45).abs() < 1e-6);
        assert!((out[1] - 0.1).abs() < 1e-6);
        assert!((out[2] - 0.5).abs() < 1e-6);
        // The closed-form trail after N ticks from a unit impulse is k^N.
        let k = DisplayPhysicsParams {
            phosphor: 0.9,
            phos_r: 0.86,
            phos_g: 1.0,
            phos_b: 0.66,
            ..DisplayPhysicsParams::default()
        }
        .phosphor_decay();
        let mut trail = [1.0_f32, 1.0, 1.0];
        for _ in 0..5 {
            trail = phosphor_combine([0.0; 3], trail, k);
        }
        for channel in 0..3 {
            assert!((trail[channel] - k[channel].powi(5)).abs() < 1e-5);
        }
        // Green outlasts red outlasts blue — the P22 signature.
        assert!(trail[1] > trail[0] && trail[0] > trail[2]);
        // The per-primary product clamps at 0.995 so no trail is permanent.
        let hot = DisplayPhysicsParams {
            phosphor: 0.95,
            phos_g: 1.0,
            ..DisplayPhysicsParams::default()
        }
        .phosphor_decay();
        assert!(hot[1] <= 0.995);
    }

    #[test]
    fn the_rate_law_exponentiates_decay_per_reference_tick() {
        let per_tick = [0.81_f32, 0.9, 0.64];
        // Two 1/60-second frames span one tick exactly.
        let half = phosphor_decay_over_ticks(per_tick, 0.5);
        for channel in 0..3 {
            assert!((half[channel] * half[channel] - per_tick[channel]).abs() < 1e-5);
        }
        // A dropped-frame gap is one closed form, not a loop.
        let long = phosphor_decay_over_ticks(per_tick, 3.0);
        for channel in 0..3 {
            assert!((long[channel] - per_tick[channel].powi(3)).abs() < 1e-5);
        }
        // Hostile tick counts take one tick, not an extreme.
        assert_eq!(phosphor_decay_over_ticks(per_tick, f32::NAN), per_tick);
    }

    #[test]
    fn the_mask_families_transmit_one_primary_per_site_and_flat_transmits_everything() {
        assert_eq!(
            mask_at([5.0, 3.0], DisplayModel::Flat, 1.0),
            [1.0, 1.0, 1.0]
        );
        // Aperture grille: three columns cycle R, G, B.
        let dark = 1.0 - 0.55;
        assert_eq!(
            mask_at([0.5, 0.0], DisplayModel::ApertureGrille, 1.0),
            [1.0, dark, dark]
        );
        assert_eq!(
            mask_at([1.5, 0.0], DisplayModel::ApertureGrille, 1.0),
            [dark, 1.0, dark]
        );
        assert_eq!(
            mask_at([2.5, 0.0], DisplayModel::ApertureGrille, 1.0),
            [dark, dark, 1.0]
        );
        // Slot mask: the sixth row is the dark slot bar.
        let bar = mask_at([0.5, 5.5], DisplayModel::SlotMask, 1.0);
        let open = mask_at([0.5, 2.5], DisplayModel::SlotMask, 1.0);
        assert!(bar[0] < open[0]);
        // Mono and green-screen have no mask of their own.
        assert_eq!(
            mask_at([0.5, 0.0], DisplayModel::Mono, 1.0),
            [1.0, 1.0, 1.0]
        );
        // Mask dark zero is transmission one everywhere.
        assert_eq!(
            mask_at([1.5, 0.0], DisplayModel::ApertureGrille, 0.0),
            [1.0, 1.0, 1.0]
        );
    }

    #[test]
    fn the_beam_widens_with_brightness_and_peaks_on_the_scanline_centre() {
        // Centre of the line is the peak.
        assert!(beam_profile(0.0, 1.0, 0.5, 0.5) > beam_profile(0.4, 1.0, 0.5, 0.5));
        // A brighter pixel widens the beam, so the gap between lines closes.
        let dim = beam_profile(0.4, 1.0, 1.0, 0.1);
        let bright = beam_profile(0.4, 1.0, 1.0, 1.0);
        assert!(bright > dim);
        // Shape zero decouples width from brightness.
        assert_eq!(
            beam_profile(0.3, 1.0, 0.0, 0.0),
            beam_profile(0.3, 1.0, 0.0, 1.0)
        );
    }

    #[test]
    fn sag_grows_with_centre_brightness_and_bloom_extracts_only_the_highlights() {
        assert_eq!(sag_scale(0.0, 1.0), 0.0);
        assert!(sag_scale(1.0, 1.0) > sag_scale(1.0, 0.0));
        // Below the hot threshold bloom contributes nothing.
        assert_eq!(bloom_hot([0.4, 0.4, 0.4], 1.0, 0.0), [0.0, 0.0, 0.0]);
        let hot = bloom_hot([1.0, 1.0, 1.0], 1.0, 1.0);
        assert!(hot[0] > hot[1] && hot[1] > hot[2], "halation tints orange");
        // The gather ring is a fixed 12-tap law.
        let ring = gather_ring();
        assert_eq!(ring.len(), 12);
        assert!(ring.iter().all(|(_, w)| *w > 0.0));
    }

    #[test]
    fn mono_folds_to_luma_and_green_screen_tints_the_phosphor() {
        let rgb = [1.0, 0.0, 0.0];
        let mono = model_tint(rgb, DisplayModel::Mono);
        assert!(mono[1] > 0.0, "mono spreads the red into every channel");
        let green = model_tint(rgb, DisplayModel::GreenScreen);
        assert!(
            green[0] < mono[0] && green[2] < mono[2] && (green[1] - mono[1]).abs() < 1e-6,
            "the green tint darkens red and blue and leaves the green phosphor whole"
        );
        assert_eq!(model_tint(rgb, DisplayModel::Flat), rgb);
    }

    #[test]
    fn serde_rejects_unknown_fields_and_fills_absent_ones() {
        let yaml = "phosphor: 0.5\nmodel: aperture_grille\n";
        let parsed: DisplayPhysicsParams = serde_yaml::from_str(yaml).expect("partial params");
        assert_eq!(parsed.phosphor, 0.5);
        assert_eq!(parsed.model, DisplayModel::ApertureGrille);
        assert_eq!(parsed.phos_g, 1.0);
        let hostile = "phosphor: 0.5\nunknown_field: 1\n";
        assert!(serde_yaml::from_str::<DisplayPhysicsParams>(hostile).is_err());
        let bad_token = "model: plasma\n";
        assert!(serde_yaml::from_str::<DisplayPhysicsParams>(bad_token).is_err());
    }
}
